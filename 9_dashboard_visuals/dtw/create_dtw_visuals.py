#!/usr/bin/env python3
"""
Create and publish DTW visuals for the dashboard (Step 3 of DTW workflow).

DTW alignment IS computed (via create_dtw_features.py using dtaidistance), but features are not added 
to model data due to concern about target leakage. Used for dashboard visualization only.

Data flow to visualizations:
- Input: dtw_features_{cohort}_{age_band}.csv from create_dtw_features.py (columns: mi_person_key, target,
  seq_pattern_str, admin_icd_event_count, dtw_min_distance, trajectory_length, ...).
- Validated/coerced: mi_person_key (str), target (0/1 int), seq_pattern_str (str, no NaN), admin_icd_event_count (int).
- Cluster plots: create_dtw_plots.create_trajectory_cluster_plots(dtw_df) uses seq_pattern_str -> code counts
  -> top_codes (excluding nan/none/null) -> Plotly 1D/3D scatter; writes dtw_trajectory_cluster_*.png/html.
  We also copy that PNG to dtw_trajectory_analysis_*.png and dtw_sample_trajectories_*.png so API URLs work.
- chart_data.json: _build_dtw_chart_data(dtw_df) builds charts including:
  1. routine_comparison: outcome rate by routine vs no routine (admin ICD). Core production analysis: highlights how routine screenings (admin codes) may reduce extreme outcomes; always built when admin_icd_event_count is present.
  2. routine_comparison_counts: mean medical events (ICD/CPT) and mean prescription events (drugs) per patient by routine vs no routine; shows routine screenings associate with lower medical and prescription event counts.
  3. high_risk_trajectories: outcome rate by trajectory archetype (quartiles)
  4. target_pathway_patterns: common codes in target=1 trajectories
  Frontend (index.html) expects chart_data JSON with these chart objects (x, y, type, name, x_label, y_label; routine_comparison_counts uses series: [{ name, y }]).
- Outputs: outputs/{cohort}/{age_band_fname}/plots/*.png/html, chart_data.json, sequence_heatmap.json written
  locally (so check_dashboard_artifact_paths.py can validate) and uploaded to S3. DTW CSV files are NOT uploaded.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shutil

# Same pattern as BupaR/FP-Growth: use setup_pipeline_logger (repo root from py_helpers → project-level logs)
REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DTW_VIZ_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from py_helpers.fe_monitor import function_block  # noqa: E402
from py_helpers.pipeline_logger import setup_pipeline_logger  # noqa: E402


def _dtw_output_root(project_root: Path) -> Path:
    """Dashboard visualization outputs (step 10); creation code lives in 9_dashboard_visuals/dtw."""
    return project_root / "10_risk_dashboard" / "visualizations" / "dtw"


def _read_dtw_features_parquet_or_csv(path_parquet: Path, path_csv: Path) -> Optional[pd.DataFrame]:
    """Load DataFrame from parquet if present, else CSV. Prefer Parquet + DuckDB for transformations."""
    if path_parquet.exists():
        try:
            import duckdb
            con = duckdb.connect(":memory:")
            try:
                return con.execute("SELECT * FROM read_parquet(?)", [str(path_parquet)]).df()
            finally:
                con.close()
        except ImportError:
            return pd.read_parquet(path_parquet)
    if path_csv.exists():
        return pd.read_csv(path_csv)
    return None


_N3_METRIC_COLUMNS = ("mean_days_between_events", "days_first_event_to_target")


def _agg_n3_via_duckdb(df: pd.DataFrame, metric_col: str, bucket_label_0: str, bucket_label_else: str) -> Optional[pd.DataFrame]:
    """Run N3-style bucket aggregation in DuckDB when available. Returns agg with columns bucket, mean_days, n."""
    if metric_col not in _N3_METRIC_COLUMNS:
        return None
    try:
        import duckdb
    except ImportError:
        return None
    if df.empty or metric_col not in df.columns or "admin_icd_event_count" not in df.columns:
        return None
    con = duckdb.connect(":memory:")
    try:
        con.register("t", df[["admin_icd_event_count", metric_col]].dropna(subset=[metric_col]))
        if con.execute("SELECT COUNT(*) FROM t").fetchone()[0] < 4:
            return None
        # metric_col is from allowlist _N3_METRIC_COLUMNS
        agg = con.execute(
            "SELECT CASE WHEN admin_icd_event_count = 0 THEN ? ELSE ? END AS bucket, "
            "AVG(" + metric_col + ") AS mean_days, COUNT(*) AS n FROM t GROUP BY 1 ORDER BY 1",
            [bucket_label_0, bucket_label_else],
        ).df()
        return agg
    finally:
        con.close()


if str(DTW_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(DTW_VIZ_DIR))  # noqa: E402 — so create_dtw_plots can be imported

from py_helpers.checkpoint_utils import check_step_checkpoint_exists, save_step_checkpoint  # noqa: E402


def create_dtw_visuals(
    project_root: Path,
    cohort_name: str,
    age_band: str,
    force: bool = False,
    logger: Optional[logging.Logger] = None,
    log_path: Optional[Path] = None,
) -> None:
    """
    Create and publish DTW visuals for the dashboard. Does not add DTW features to model data.
    Loads the DTW features CSV from create_dtw_features.py, generates plots and chart_data,
    and uploads to the dashboard bucket. DTW CSV files are not uploaded (dashboard only uses plots/chart_data).
    If force is False and plots already exist, skips (idempotent).
    """
    def _log(level: str, msg: str, *args: Any) -> None:
        if logger is not None:
            getattr(logger, level)(msg, *args)
        else:
            prefix = "[%s] " % level.upper()
            print(prefix + (msg % args if args else msg))

    age_band_fname = age_band.replace("-", "_")
    dtw_out = _dtw_output_root(project_root)
    out_dir = dtw_out / cohort_name / age_band_fname
    _log("info", "DTW outputs (EC2): project_root=%s ; dtw_out=%s ; out_dir=%s", project_root, dtw_out, out_dir)

    # Idempotency: skip only when all dashboard artifacts exist (plots + chart_data + sequence_heatmap)
    plots_dir = out_dir / "plots"
    chart_path = out_dir / "chart_data.json"
    heatmap_path = out_dir / "sequence_heatmap.json"
    all_exist = (
        plots_dir.exists() and list(plots_dir.glob("*.png"))
        and chart_path.exists()
        and heatmap_path.exists()
    )
    if not force and all_exist:
        _log("info", "DTW artifacts exist at %s (plots + chart_data + sequence_heatmap); skipping (use --force to re-run)", out_dir)
        return
    if not force and check_step_checkpoint_exists("9_dashboard_visuals", cohort_name, age_band, logger=logger) and all_exist:
        _log("info", "Pipeline checkpoint exists for 9_dashboard_visuals %s/%s and artifacts present; skipping (use --force to re-run)", cohort_name, age_band)
        return

    # Load DTW features: prefer sub-cohort (per-density) when present; else single. Prefer parquet then CSV.
    fe_dir = _dtw_output_root(project_root) / "feature_engineering"
    base_name = f"dtw_features_{cohort_name}_{age_band_fname}"
    single_parquet = fe_dir / f"{base_name}.parquet"
    single_csv = fe_dir / f"{base_name}.csv"
    density_parquet = list(fe_dir.glob(f"{base_name}_density_*.parquet"))
    density_csv = list(fe_dir.glob(f"{base_name}_density_*.csv"))
    bin_stems = {p.stem.replace(f"{base_name}_density_", "") for p in density_parquet + density_csv}
    density_paths = [(fe_dir / f"{base_name}_density_{b}.parquet", fe_dir / f"{base_name}_density_{b}.csv") for b in sorted(bin_stems)]
    _log("info", "DTW input: fe_dir=%s ; single (parquet=%s, csv=%s) ; density bins=%d", fe_dir, single_parquet.exists(), single_csv.exists(), len(bin_stems))

    dtw_df = None
    if density_paths:
        # Sub-cohort outputs: load each bin (parquet or csv) and concatenate
        parts = []
        for path_pq, path_csv in density_paths:
            part = _read_dtw_features_parquet_or_csv(path_pq, path_csv)
            if part is None:
                continue
            bin_name = path_pq.stem.replace(f"{base_name}_density_", "") if path_pq.exists() else path_csv.stem.replace(f"{base_name}_density_", "")
            if "event_density_bin" not in part.columns:
                part["event_density_bin"] = bin_name
            parts.append(part)
        if parts:
            dtw_df = pd.concat(parts, ignore_index=True)
            _log("info", "Loaded DTW features from %d density sub-cohorts", len(parts))
    if dtw_df is None:
        dtw_df = _read_dtw_features_parquet_or_csv(single_parquet, single_csv)
        if dtw_df is not None:
            _log("info", "Reading DTW features from %s or %s", single_parquet, single_csv)

    if dtw_df is None:
        # Only skip if final dashboard visuals are already present; otherwise fail so the step is not silently skipped.
        out_dir = _dtw_output_root(project_root) / cohort_name / age_band_fname
        chart_path = out_dir / "chart_data.json"
        heatmap_path = out_dir / "sequence_heatmap.json"
        plots_dir = out_dir / "plots"
        has_chart = chart_path.exists()
        has_heatmap = heatmap_path.exists()
        has_plots = plots_dir.is_dir() and any(plots_dir.iterdir())
        if has_chart and has_heatmap and has_plots:
            _log("info", "DTW features not found: %s or %s; skipping (final visuals already present: chart_data, sequence_heatmap, plots)", single_parquet, single_csv)
            return
        _log("warning", "DTW features not found: %s / %s (and no density sub-cohorts); final visuals not present (chart_data=%s, sequence_heatmap=%s, plots=%s).", single_parquet, single_csv, has_chart, has_heatmap, has_plots)
        try:
            from py_helpers.model_data_paths import get_path_check_listings
            path_listings = get_path_check_listings([str(single_parquet), str(single_csv)])
            path_listings_str = " ; ".join(path_listings) if path_listings else ""
        except Exception:  # noqa: BLE001
            path_listings_str = ""
        _log("error", "step=5_dtw cohort_name=%s age_band=%s error=DTW features not found expected_path=%s or %s (no EC2 artifacts written, no S3 upload)", cohort_name, age_band, single_parquet, single_csv)
        if path_listings_str:
            _log("error", "step=5_dtw path_listings: %s", path_listings_str)
            raise FileNotFoundError(
            f"DTW features not found: {single_parquet} or {single_csv}. Final visuals not present (chart_data={has_chart}, sequence_heatmap={has_heatmap}, plots={has_plots}). "
            "Run create_dtw_features/create_dtw_trajectories first, or ensure visuals exist for this cohort/age_band."
        )

    keys_expected_dtw = ["mi_person_key", "target", "seq_pattern_str", "admin_icd_event_count", "dtw_min_distance", "trajectory_length"]
    keys_expected_n3 = ["mean_days_between_events", "days_first_event_to_target", "admin_icd_event_count"]
    keys_received_dtw = list(dtw_df.columns)
    available = set(dtw_df.columns)
    missing_core = [k for k in keys_expected_dtw if k not in available]
    missing_n3 = [k for k in keys_expected_n3 if k not in available]
    _log("info", "DTW columns for cohort %s/%s: available=%s", cohort_name, age_band, keys_received_dtw)
    _log("info", "DTW columns expected (core): %s; missing=%s", keys_expected_dtw, missing_core if missing_core else "none")
    _log("info", "DTW columns expected (N3 time-between): %s; missing=%s", keys_expected_n3, missing_n3 if missing_n3 else "none")

    # --- Validate and coerce data structure for visualizations ---
    if "mi_person_key" not in dtw_df.columns:
        _log("error", "step=5_dtw keys_expected=%s keys_received=%s", keys_expected_dtw, keys_received_dtw)
        raise ValueError("DTW features CSV must contain 'mi_person_key' column")
    dtw_df["mi_person_key"] = dtw_df["mi_person_key"].astype(str)

    if "target" not in dtw_df.columns:
        _log("warning", "DTW features have no 'target' column; keys_received=%s. Chart_data will be skipped.", keys_received_dtw)
    else:
        # Coerce target to numeric (0/1) for chart_data
        dtw_df["target"] = pd.to_numeric(dtw_df["target"], errors="coerce").fillna(0).astype(int)

    if "seq_pattern_str" in dtw_df.columns:
        dtw_df["seq_pattern_str"] = dtw_df["seq_pattern_str"].fillna("").astype(str)
    else:
        _log("warning", "DTW features have no 'seq_pattern_str'; keys_received=%s. Trajectory cluster plots will be skipped.", keys_received_dtw)

    if "admin_icd_event_count" in dtw_df.columns:
        dtw_df["admin_icd_event_count"] = pd.to_numeric(dtw_df["admin_icd_event_count"], errors="coerce").fillna(0).astype(int)

    _log("info", "Loaded %d patients with %d DTW features", len(dtw_df), len(dtw_df.columns) - 1)

    # Create 3D/1D trajectory cluster plots (Plotly) then upload plots to dashboard bucket
    plot_written: List[str] = []
    try:
        from create_dtw_plots import create_trajectory_cluster_plots
        written_paths = create_trajectory_cluster_plots(
            project_root=project_root,
            cohort_name=cohort_name,
            age_band=age_band,
            dtw_df=dtw_df,
            force=force,
        )
        if written_paths:
            plot_written = [p.name for p in written_paths]
        # API/frontend expect these filenames (lambda_function.py, index.html)
        plots_dir = _dtw_output_root(project_root) / cohort_name / age_band_fname / "plots"
        overview_name = f"dtw_trajectory_analysis_{cohort_name}_{age_band_fname}.png"
        sample_name = f"dtw_sample_trajectories_{cohort_name}_{age_band_fname}.png"
        if plots_dir.exists():
            cluster_pngs = list(plots_dir.glob("dtw_trajectory_cluster_*.png"))
            if cluster_pngs:
                src = cluster_pngs[0]
                for name in (overview_name, sample_name):
                    dest = plots_dir / name
                    if dest != src:
                        shutil.copy2(src, dest)
                        plot_written.append(name)
                        _log("info", "Wrote %s for API overview/sample URLs", name)
    except Exception as e:
        exc_type = type(e).__name__
        _log("warning", "DTW trajectory cluster plots failed (%s): %s", exc_type, e)
        if exc_type == "MemoryError":
            _log("info", "Tip: cluster plot subsamples to 25,000 rows for large cohorts; if OOM persists, reduce MAX_PLOT_ROWS in create_dtw_plots.py or increase process memory.")
    _upload_dtw_plots_to_dashboard_s3(project_root, cohort_name, age_band, logger=logger)

    # Prebuild chart data (routine vs no routine, high-risk trajectories); write locally and upload to S3
    # No empty artifacts: when nothing is produced, write JSON with message + metrics (why) so dashboard can show reason.
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_data = _build_dtw_chart_data(dtw_df, logger=logger)
    if chart_data is None:
        _log("warning", "DTW chart_data not produced for %s/%s: empty dataframe (writing empty-state JSON with message and metrics)", cohort_name, age_band)
        chart_data = {
            "message": f"No DTW chart data for {cohort_name}/{age_band}.",
            "empty": True,
            "cohort": cohort_name,
            "age_band": age_band,
            "summary": _build_chart_data_summary(dtw_df) if dtw_df is not None else _build_chart_data_summary(pd.DataFrame()),
            "metrics": {
                "reason": "empty_dataframe",
                "dtw_rows": 0,
                "charts_built": [],
                "charts_not_built": {},
                "success": False,
            },
        }
    elif not chart_data:
        _log("warning", "DTW chart_data empty for %s/%s: no charts built (writing empty-state JSON with metrics)", cohort_name, age_band)
        chart_data = {
            "message": f"No DTW chart data for {cohort_name}/{age_band} (no charts built).",
            "empty": True,
            "cohort": cohort_name,
            "age_band": age_band,
            "summary": _build_chart_data_summary(dtw_df) if dtw_df is not None else _build_chart_data_summary(pd.DataFrame()),
            "metrics": {
                "reason": "no_charts_built",
                "dtw_rows": len(dtw_df) if dtw_df is not None else 0,
                "charts_built": [],
                "charts_not_built": {},
                "success": False,
            },
        }
    with open(chart_path, "w", encoding="utf-8") as f:
        json.dump(chart_data, f, indent=0)
    _log("info", "Wrote %s", chart_path)
    _upload_dtw_chart_data_to_dashboard_s3(project_root, cohort_name, age_band, chart_data, logger=logger)

    # Sequence heatmap (code × position counts); no empty artifacts: when no data, write JSON with message + metrics (why).
    heatmap_data = _build_sequence_heatmap_data(dtw_df)
    if heatmap_data is None:
        _log("warning", "DTW sequence_heatmap not produced for %s/%s (writing empty-state JSON with message and metrics)", cohort_name, age_band)
        heatmap_data = {
            "message": f"No sequence heatmap for {cohort_name}/{age_band}.",
            "empty": True,
            "cohort": cohort_name,
            "age_band": age_band,
            "metrics": {"reason": "empty_dataframe_or_no_seq_pattern_str", "dtw_rows": len(dtw_df) if dtw_df is not None and not dtw_df.empty else 0},
        }
    else:
        # If all slices are empty (no codes), still provide envelope so dashboard can show why
        has_any = any(
            (h.get("codes") or h.get("counts"))
            for h in (heatmap_data.get("drug"), heatmap_data.get("icd"), heatmap_data.get("cpt"))
            if isinstance(h, dict)
        )
        if not has_any and len(dtw_df) > 0:
            heatmap_data = {
                "message": f"No sequence heatmap data for {cohort_name}/{age_band} (no code counts in sequences).",
                "empty": True,
                "cohort": cohort_name,
                "age_band": age_band,
                "metrics": {"reason": "no_code_counts", "dtw_rows": len(dtw_df)},
            }
    with open(heatmap_path, "w", encoding="utf-8") as f:
        json.dump(heatmap_data, f, indent=0)
    _log("info", "Wrote %s", heatmap_path)
    _upload_sequence_heatmap_to_s3(project_root, cohort_name, age_band, heatmap_data, logger=logger)

    # Save pipeline checkpoint (dashboard artifacts complete: plots + chart_data)
    s3_output_paths = [
        f"s3://{os.environ.get('S3_DASHBOARD_BUCKET', 'jerome-dixon.io')}/{os.environ.get('S3_DASHBOARD_PREFIX', 'vcu/pgx-risk-calculator')}/visualizations/dtw/{cohort_name}/{age_band}/plots/"
    ]
    try:
        save_step_checkpoint(
            "9_dashboard_visuals",
            cohort_name,
            age_band,
            metadata={"dtw_plots": "uploaded"},
            output_paths=s3_output_paths,
            logger=logger,
        )
    except Exception as exc:  # pragma: no cover
        _log("warning", "Could not save pipeline checkpoint: %s", exc)

    successful = ["chart_data.json", "sequence_heatmap.json"]
    if plot_written:
        successful.append("plots: " + ", ".join(plot_written))
    _log("info", "DTW visuals successful: %s", successful)
    _log("info", "DTW artifacts (EC2): chart_data=%s ; sequence_heatmap=%s ; plots_dir=%s", chart_path, heatmap_path, plots_dir)
    _log("info", "Done.")
    _log("info", "DTW visuals complete. Plots and chart_data uploaded to dashboard S3: trajectory cluster plots (3D/1D), chart_data.json, sequence_heatmap.json. CSV files not uploaded; dashboard uses plots only.")



def _upload_dtw_plots_to_dashboard_s3(
    project_root: Path,
    cohort_name: str,
    age_band: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Upload DTW plot PNGs and Plotly HTML to the dashboard bucket under visualizations/dtw/{cohort}/{age_band}/plots/ (same pattern as FP-Growth/BupaR).
    When SKIP_DASHBOARD_S3_UPLOAD=1, no upload (notebook 5 Step 6 syncs from local)."""
    if (os.environ.get("SKIP_DASHBOARD_S3_UPLOAD", "") or "").strip().lower() in ("1", "true", "yes"):
        if logger:
            logger.debug("SKIP_DASHBOARD_S3_UPLOAD set; DTW plots S3 upload skipped.")
        return
    age_band_fname = age_band.replace("-", "_")
    plots_dir = _dtw_output_root(project_root) / cohort_name / age_band_fname / "plots"
    if not plots_dir.exists():
        if logger:
            logger.info("DTW plots upload skipped: plots_dir does not exist: %s", plots_dir)
        return
    plot_files = list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.html")) + list(plots_dir.glob("*.json"))
    if not plot_files:
        if logger:
            logger.info("DTW plots upload skipped: no .png/.html/.json in %s", plots_dir)
        return

    s3_bucket = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
    dashboard_prefix = os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator")
    use_builds = (os.environ.get("S3_VISUALIZATIONS_BUILDS", "") or "").strip().lower() in ("1", "true", "yes")
    builds_suffix = "/builds" if use_builds else ""
    s3_prefix = f"{dashboard_prefix.rstrip('/')}/visualizations/dtw{builds_suffix}/{cohort_name}/{age_band}/plots"
    if logger:
        logger.info("DTW plots upload: %d file(s) from %s -> s3://%s/%s/", len(plot_files), plots_dir, s3_bucket, s3_prefix)

    try:
        from py_helpers.checkpoint_utils import upload_file_to_s3
    except ImportError as e:
        if logger:
            logger.warning("DTW plots upload skipped: could not import upload_file_to_s3: %s", e)
        return

    uploaded = 0
    for p in plot_files:
        key = f"{s3_prefix}/{p.name}"
        s3_path = f"s3://{s3_bucket}/{key}"
        if upload_file_to_s3(p, s3_path, logger=logger, check_exists=True):
            uploaded += 1
    if uploaded and logger:
        logger.info("Uploaded %d DTW plot(s) to dashboard S3 s3://%s/%s/", uploaded, s3_bucket, s3_prefix)


def _count_drug_events_in_sequence(seq_str: Any) -> int:
    """Count prescription (DRUG:) events in seq_pattern_str. Used to show routine vs drug/medical counts."""
    if seq_str is None or (isinstance(seq_str, float) and pd.isna(seq_str)):
        return 0
    s = str(seq_str).strip()
    if not s:
        return 0
    return sum(1 for t in s.split("_") if t.strip().upper().startswith("DRUG:"))


def _agg_routine_comparison_via_duckdb(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Outcome rate by routine vs no routine using DuckDB when available. Returns chart dict or None."""
    try:
        import duckdb
    except ImportError:
        return None
    if df.empty or "target" not in df.columns or "admin_icd_event_count" not in df.columns:
        return None
    con = duckdb.connect(":memory:")
    try:
        bucket_0 = "No routine appointments (0 admin ICD events)"
        bucket_1 = "Routine appointments (1+ admin ICD events)"
        con.register("t", df[["admin_icd_event_count", "target"]])
        agg = con.execute(
            "SELECT CASE WHEN admin_icd_event_count = 0 THEN ? ELSE ? END AS bucket, "
            "AVG(target) AS target_rate, COUNT(*) AS n FROM t GROUP BY 1 ORDER BY 1",
            [bucket_0, bucket_1],
        ).df()
        if agg.empty or len(agg) < 2 or agg["n"].sum() < 10:
            return None
        agg = agg.set_index("bucket").reindex([bucket_0, bucket_1]).reset_index()
        agg = agg.dropna(subset=["target_rate"])
        if agg.empty or agg["n"].sum() == 0:
            return None
        return {
            "x": agg["bucket"].astype(str).tolist(),
            "y": [float(round(v, 4)) for v in agg["target_rate"]],
            "n": [int(v) for v in agg["n"]],
            "type": "bar",
            "name": "Outcome rate",
            "x_label": "Routine vs no routine (admin ICD filter)",
            "y_label": "Target outcome rate",
        }
    finally:
        con.close()


def _compute_dtw_routine_comparison(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Outcome rate by routine vs no routine (admin ICD filter) or by trajectory intensity. Uses DuckDB when available."""
    if df.empty or "target" not in df.columns:
        return None
    if "admin_icd_event_count" in df.columns:
        res = _agg_routine_comparison_via_duckdb(df)
        if res is not None:
            return res
        use_df = df[["admin_icd_event_count", "target"]].copy()
        use_df["bucket"] = use_df["admin_icd_event_count"].apply(
            lambda x: "No routine appointments (0 admin ICD events)" if x == 0 else "Routine appointments (1+ admin ICD events)"
        )
        x_label = "Routine vs no routine (admin ICD filter)"
    elif "trajectory_length" in df.columns:
        col = "trajectory_length"
        use_df = df[[col, "target"]].dropna()
        if len(use_df) < 10:
            return None
        q1, q2 = use_df[col].quantile(0.33), use_df[col].quantile(0.67)
        use_df = use_df.copy()
        use_df["bucket"] = use_df[col].apply(
            lambda x: "Low (fewer events)" if x <= q1 else ("Medium" if x <= q2 else "High (more events)")
        )
        x_label = "Trajectory intensity (event count)"
    else:
        return None
    use_df = use_df.dropna(subset=["bucket"])
    if len(use_df) < 10:
        return None
    agg = use_df.groupby("bucket", as_index=False, observed=True).agg(target_rate=("target", "mean"), n=("target", "count"))
    order = (
        ["No routine appointments (0 admin ICD events)", "Routine appointments (1+ admin ICD events)"]
        if "admin_icd_event_count" in df.columns
        else ["Low (fewer events)", "Medium", "High (more events)"]
    )
    agg = agg.set_index("bucket").reindex([b for b in order if b in agg.index]).reset_index()
    agg = agg.dropna(subset=["target_rate"])
    if agg.empty or agg["n"].sum() == 0:
        return None
    # Frontend expects: x, y, type, x_label, y_label; n for robustness (multiple visuals)
    return {
        "x": agg["bucket"].astype(str).tolist(),
        "y": [float(round(v, 4)) for v in agg["target_rate"]],
        "n": [int(v) for v in agg["n"]],
        "type": "bar",
        "name": "Outcome rate",
        "x_label": x_label,
        "y_label": "Target outcome rate",
    }


def _agg_routine_comparison_counts_via_duckdb(use_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Aggregate routine_comparison_counts in DuckDB when available. use_df must have bucket, medical_event_count, drug_event_count."""
    try:
        import duckdb
    except ImportError:
        return None
    if use_df.empty or "bucket" not in use_df.columns or len(use_df) < 10:
        return None
    con = duckdb.connect(":memory:")
    try:
        con.register("t", use_df[["bucket", "medical_event_count", "drug_event_count"]])
        agg = con.execute(
            "SELECT bucket, AVG(medical_event_count) AS mean_medical, AVG(drug_event_count) AS mean_drug, COUNT(*) AS n FROM t GROUP BY bucket ORDER BY bucket"
        ).df()
        bucket_0 = "No routine appointments (0 admin ICD events)"
        bucket_1 = "Routine appointments (1+ admin ICD events)"
        order = [bucket_0, bucket_1]
        agg = agg.set_index("bucket").reindex([b for b in order if b in agg.index]).reset_index()
        agg = agg.dropna(subset=["mean_medical", "mean_drug"])
        if agg.empty or agg["n"].sum() == 0:
            return None
        return {
            "x": agg["bucket"].astype(str).tolist(),
            "series": [
                {"name": "Mean medical events (ICD/CPT) per patient", "y": [float(round(v, 2)) for v in agg["mean_medical"]]},
                {"name": "Mean prescription events (drugs) per patient", "y": [float(round(v, 2)) for v in agg["mean_drug"]]},
            ],
            "n": [int(v) for v in agg["n"]],
            "type": "bar",
            "x_label": "Routine vs no routine (admin ICD filter)",
            "y_label": "Mean events per patient",
        }
    finally:
        con.close()


def _compute_dtw_routine_comparison_counts(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Mean medical and mean prescription (drug) events per patient by routine vs no routine. Uses DuckDB for aggregation when available."""
    if df.empty or "admin_icd_event_count" not in df.columns:
        return None
    need = ["admin_icd_event_count", "trajectory_length"]
    if "seq_pattern_str" not in df.columns or not all(c in df.columns for c in need):
        return None
    use_df = df[["admin_icd_event_count", "trajectory_length", "seq_pattern_str"]].copy()
    use_df["bucket"] = use_df["admin_icd_event_count"].apply(
        lambda x: "No routine appointments (0 admin ICD events)" if x == 0 else "Routine appointments (1+ admin ICD events)"
    )
    use_df["drug_event_count"] = use_df["seq_pattern_str"].apply(_count_drug_events_in_sequence)
    use_df["medical_event_count"] = (use_df["trajectory_length"] - use_df["drug_event_count"]).clip(lower=0)
    use_df = use_df.dropna(subset=["bucket", "trajectory_length"])
    if len(use_df) < 10:
        return None
    res = _agg_routine_comparison_counts_via_duckdb(use_df)
    if res is not None:
        return res
    agg = use_df.groupby("bucket", as_index=False, observed=True).agg(
        mean_medical=("medical_event_count", "mean"),
        mean_drug=("drug_event_count", "mean"),
        n=("trajectory_length", "count"),
    )
    order = ["No routine appointments (0 admin ICD events)", "Routine appointments (1+ admin ICD events)"]
    agg = agg.set_index("bucket").reindex([b for b in order if b in agg.index]).reset_index()
    agg = agg.dropna(subset=["mean_medical", "mean_drug"])
    if agg.empty or agg["n"].sum() == 0:
        return None
    return {
        "x": agg["bucket"].astype(str).tolist(),
        "series": [
            {"name": "Mean medical events (ICD/CPT) per patient", "y": [float(round(v, 2)) for v in agg["mean_medical"]]},
            {"name": "Mean prescription events (drugs) per patient", "y": [float(round(v, 2)) for v in agg["mean_drug"]]},
        ],
        "n": [int(v) for v in agg["n"]],
        "type": "bar",
        "x_label": "Routine vs no routine (admin ICD filter)",
        "y_label": "Mean events per patient",
    }


def _compute_dtw_high_risk_trajectories(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Target outcome rate by trajectory archetype (quartiles). Prebuilt on EC2."""
    if df.empty or "target" not in df.columns:
        return None
    col = "dtw_min_distance" if "dtw_min_distance" in df.columns else "trajectory_length"
    if col not in df.columns:
        return None
    use_df = df[["target", col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use_df) < 10 or use_df[col].nunique() < 2:
        return None
    try:
        use_df = use_df.copy()
        use_df["q"] = pd.qcut(
            use_df[col], q=4, labels=["Q1 (closest)", "Q2", "Q3", "Q4 (furthest)"], duplicates="drop"
        )
    except (ValueError, TypeError):
        return None
    agg = use_df.groupby("q", as_index=False, observed=True).agg(target_rate=("target", "mean"), n=("target", "count"))
    if agg.empty or agg["n"].sum() == 0:
        return None
    # Frontend: x, y, n for robustness (multiple visuals)
    return {
        "x": [str(v) for v in agg["q"]],
        "y": [float(round(v, 4)) for v in agg["target_rate"]],
        "n": [int(v) for v in agg["n"]],
        "type": "bar",
        "name": "Outcome rate by archetype",
        "x_label": "Trajectory archetype (by DTW distance)" if col == "dtw_min_distance" else "Trajectory archetype (by length)",
        "y_label": "Target outcome rate",
    }


def _compute_times_between_sequences(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """N3: Mean days between consecutive drug events by routine vs no routine (times between sequences).
    Uses Parquet+DuckDB for aggregation when available; else pandas. mean_days_between_events is defined only for sequences with >=2 events."""
    if df.empty or "mean_days_between_events" not in df.columns or "admin_icd_event_count" not in df.columns:
        return None
    bucket_0 = "No routine (0 admin ICD events)"
    bucket_1 = "Routine (1+ admin ICD events)"
    agg = _agg_n3_via_duckdb(df, "mean_days_between_events", bucket_0, bucket_1)
    if agg is None:
        use_df = df[["admin_icd_event_count", "mean_days_between_events"]].copy()
        use_df["bucket"] = use_df["admin_icd_event_count"].apply(
            lambda x: bucket_0 if x == 0 else bucket_1
        )
        use_df = use_df.dropna(subset=["mean_days_between_events"])
        if len(use_df) < 4:
            return None
        agg = use_df.groupby("bucket", as_index=False, observed=True).agg(
            mean_days=("mean_days_between_events", "mean"),
            n=("mean_days_between_events", "count"),
        )
    order = [bucket_0, bucket_1]
    agg = agg.set_index("bucket").reindex([b for b in order if b in agg.index]).reset_index()
    agg = agg.dropna(subset=["mean_days"])
    if agg.empty:
        return None
    return {
        "x": agg["bucket"].astype(str).tolist(),
        "y": [float(round(v, 1)) for v in agg["mean_days"]],
        "n": [int(v) for v in agg["n"]],
        "type": "bar",
        "name": "Mean days between consecutive events",
        "x_label": "Routine vs no routine (admin ICD filter)",
        "y_label": "Mean days between consecutive events",
    }


def _compute_time_to_target_sequences(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """N3: Mean days from first drug event to target (target=1 only) by routine vs no routine. Uses DuckDB when available."""
    if df.empty or "days_first_event_to_target" not in df.columns or "target" not in df.columns:
        return None
    target_df = df[df["target"] == 1].copy()
    if len(target_df) < 4 or "admin_icd_event_count" not in target_df.columns:
        return None
    bucket_0 = "No routine (0 admin ICD events)"
    bucket_1 = "Routine (1+ admin ICD events)"
    agg = _agg_n3_via_duckdb(target_df, "days_first_event_to_target", bucket_0, bucket_1)
    if agg is None:
        use_df = target_df[["admin_icd_event_count", "days_first_event_to_target"]].dropna(
            subset=["days_first_event_to_target"]
        )
        if len(use_df) < 4:
            return None
        use_df["bucket"] = use_df["admin_icd_event_count"].apply(
            lambda x: bucket_0 if x == 0 else bucket_1
        )
        agg = use_df.groupby("bucket", as_index=False, observed=True).agg(
            mean_days=("days_first_event_to_target", "mean"),
            n=("days_first_event_to_target", "count"),
        )
    order = [bucket_0, bucket_1]
    agg = agg.set_index("bucket").reindex([b for b in order if b in agg.index]).reset_index()
    agg = agg.dropna(subset=["mean_days"])
    if agg.empty:
        return None
    return {
        "x": agg["bucket"].astype(str).tolist(),
        "y": [float(round(v, 1)) for v in agg["mean_days"]],
        "n": [int(v) for v in agg["n"]],
        "type": "bar",
        "name": "Mean days from first event to target",
        "x_label": "Routine vs no routine (admin ICD filter)",
        "y_label": "Mean days from first event to target (target=1 only)",
    }


def _compute_target_pathway_patterns(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Analyze target=1 patients to identify common trajectory patterns leading to adverse events. Prebuilt on EC2."""
    if df.empty or "target" not in df.columns or "seq_pattern_str" not in df.columns:
        return None
    
    # Filter to target=1 patients only
    target_df = df[df["target"] == 1].copy()
    if len(target_df) < 10:
        return None
    
    # Extract top codes from sequences in target=1 population
    from collections import Counter
    all_codes = []
    for seq in target_df["seq_pattern_str"]:
        if pd.isna(seq) or not isinstance(seq, str):
            continue
        tokens = [s.strip() for s in seq.split("_") if s.strip()]
        all_codes.extend([t for t in tokens if t.lower() not in {"nan", "none", "null", ""}])
    
    if not all_codes:
        return None
    
    # Count frequency of each code in target=1 trajectories
    code_counts = Counter(all_codes)
    top_codes = code_counts.most_common(8)  # Top 8 codes in target=1 trajectories
    
    if not top_codes:
        return None
    
    # Calculate what % of target=1 patients have each top code
    code_prevalence = []
    for code, _ in top_codes:
        n_patients_with_code = sum(1 for seq in target_df["seq_pattern_str"] 
                                   if isinstance(seq, str) and code in seq)
        pct = (n_patients_with_code / len(target_df)) * 100
        code_prevalence.append({"code": code, "prevalence_pct": pct, "n_patients": n_patients_with_code})
    
    # Sort by prevalence
    code_prevalence.sort(key=lambda x: x["prevalence_pct"], reverse=True)
    
    # Frontend expects: x, y, type, x_label, y_label, and optional name
    return {
        "x": [item["code"] for item in code_prevalence],
        "y": [float(round(item["prevalence_pct"], 1)) for item in code_prevalence],
        "type": "bar",
        "name": "Common codes in adverse event trajectories",
        "x_label": "Activity Code (SHAP/FFA Important Features)",
        "y_label": "% of Target=1 Patients with Code",
        "metadata": {
            "total_target_patients": int(len(target_df)),
            "total_control_patients": int(len(df[df["target"] == 0])),
        }
    }


def _build_sequence_heatmap_data(dtw_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Build heatmap data for drug, ICD, and CPT activity types (opioid_ed and all cohorts).
    Returns dict with keys 'drug', 'icd', 'cpt'; each value has codes, positions, counts (code × position).
    Dashboard can show Drug / ICD / CPT via activity-type selector.
    """
    if dtw_df.empty or "seq_pattern_str" not in dtw_df.columns:
        return None
    skip = {"nan", "none", "null", ""}
    # type -> (code -> (position -> count))
    pos_counts: Dict[str, Dict[str, Dict[int, int]]] = {
        t: defaultdict(lambda: defaultdict(int)) for t in ("drug", "icd", "cpt")
    }
    max_pos = 0
    for seq in dtw_df["seq_pattern_str"]:
        if pd.isna(seq) or not isinstance(seq, str):
            continue
        tokens = [t.strip() for t in seq.split("_") if t.strip() and t.strip().lower() not in skip]
        for pos, token in enumerate(tokens):
            if ":" in token:
                prefix, code = token.split(":", 1)
                key = prefix.strip().upper()
                if key == "DRUG":
                    typ = "drug"
                elif key == "ICD":
                    typ = "icd"
                elif key == "CPT":
                    typ = "cpt"
                else:
                    continue
                code_val = code.strip() if code else token
                if code_val:
                    pos_counts[typ][code_val][pos] += 1
            max_pos = max(max_pos, pos)
        max_pos = max(max_pos, len(tokens) - 1) if tokens else max_pos
    n_cols = max_pos + 1
    positions = list(range(n_cols))
    out: Dict[str, Any] = {}
    for typ in ("drug", "icd", "cpt"):
        counts_map = pos_counts[typ]
        if not counts_map:
            out[typ] = {"codes": [], "positions": positions, "counts": []}
        else:
            codes = sorted(counts_map.keys())
            counts = [[counts_map[code].get(p, 0) for p in positions] for code in codes]
            out[typ] = {"codes": codes, "positions": positions, "counts": counts}
    return out


DENSITY_BINS = ("low", "medium", "high", "extreme")


def _build_chart_data_summary(dtw_df: pd.DataFrame) -> Dict[str, Any]:
    """Build a reusable summary block for chart_data.json so multiple visuals can use the same counts and stats.
    All trajectories are drug-only; one row = one patient trajectory."""
    if dtw_df is None or dtw_df.empty:
        return {
            "total_trajectories": 0,
            "trajectories_with_time_between": 0,
            "trajectories_target1_with_time_to_target": 0,
            "trajectory_length": {"min": None, "max": None, "mean": None, "median": None},
            "has_dtw_distances": False,
            "target_counts": {"target_1": 0, "target_0": 0},
        }
    total = int(len(dtw_df))
    # Trajectories with >=2 drug events (so mean_days_between_events is defined)
    if "mean_days_between_events" in dtw_df.columns:
        n_time_between = int(dtw_df["mean_days_between_events"].notna().sum())
    else:
        n_time_between = 0
    # Target=1 with valid days to target
    if "target" in dtw_df.columns and "days_first_event_to_target" in dtw_df.columns:
        target1 = dtw_df[dtw_df["target"] == 1]
        n_time_to_target = int(target1["days_first_event_to_target"].notna().sum())
    else:
        n_time_to_target = 0
    # Trajectory length stats (drug events per trajectory)
    if "trajectory_length" in dtw_df.columns:
        tl = dtw_df["trajectory_length"].dropna()
        if len(tl) > 0:
            tlstats = {
                "min": int(tl.min()),
                "max": int(tl.max()),
                "mean": float(round(tl.mean(), 2)),
                "median": float(round(tl.median(), 2)),
            }
        else:
            tlstats = {"min": None, "max": None, "mean": None, "median": None}
    else:
        tlstats = {"min": None, "max": None, "mean": None, "median": None}
    has_dtw = "dtw_min_distance" in dtw_df.columns and dtw_df["dtw_min_distance"].notna().any()
    if "target" in dtw_df.columns:
        t1 = int((dtw_df["target"] == 1).sum())
        t0 = int((dtw_df["target"] == 0).sum())
    else:
        t1 = t0 = 0
    return {
        "total_trajectories": total,
        "trajectories_with_time_between": n_time_between,
        "trajectories_target1_with_time_to_target": n_time_to_target,
        "trajectory_length": tlstats,
        "has_dtw_distances": bool(has_dtw),
        "target_counts": {"target_1": t1, "target_0": t0},
    }


def _reason_routine_comparison(df: pd.DataFrame) -> str:
    """Reason string when routine_comparison is not built. Used for charts_not_built and error logging."""
    if df.empty:
        return "empty dataframe"
    if "target" not in df.columns:
        return "missing target column"
    if "admin_icd_event_count" not in df.columns and "trajectory_length" not in df.columns:
        return "missing admin_icd_event_count and trajectory_length"
    if "admin_icd_event_count" in df.columns:
        if len(df) < 10:
            return "fewer than 10 rows (have %d)" % len(df)
        n_no_routine = int((df["admin_icd_event_count"] == 0).sum())
        n_routine = int((df["admin_icd_event_count"] > 0).sum())
        if n_no_routine == 0 or n_routine == 0:
            return "only one bucket (routine vs no routine): n_no_routine=%d, n_routine=%d" % (n_no_routine, n_routine)
        return "insufficient data or only one bucket (routine vs no routine)"
    if "trajectory_length" in df.columns and len(df) < 10:
        return "fewer than 10 rows (have %d)" % len(df)
    return "insufficient data or only one bucket (routine vs no routine)"


def _reason_routine_comparison_counts(df: pd.DataFrame) -> str:
    """Reason string when routine_comparison_counts is not built. Used for charts_not_built and error logging."""
    if df.empty:
        return "empty dataframe"
    if "admin_icd_event_count" not in df.columns:
        return "missing admin_icd_event_count"
    if "trajectory_length" not in df.columns:
        return "missing trajectory_length"
    if "seq_pattern_str" not in df.columns:
        return "missing seq_pattern_str"
    use_df = df[["admin_icd_event_count", "trajectory_length", "seq_pattern_str"]].dropna(subset=["trajectory_length"])
    if len(use_df) < 10:
        return "fewer than 10 rows after dropna (have %d)" % len(use_df)
    n_no_routine = int((use_df["admin_icd_event_count"] == 0).sum())
    n_routine = int((use_df["admin_icd_event_count"] > 0).sum())
    if n_no_routine == 0 or n_routine == 0:
        return "only one bucket: n_no_routine=%d, n_routine=%d" % (n_no_routine, n_routine)
    return "insufficient rows or aggregation failed"


def _build_dtw_chart_data(dtw_df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> Optional[Dict[str, Any]]:
    """Build chart_data.json for dashboard and other visuals. Structure is robust for multiple consumers:
    - summary: total_trajectories, trajectories_with_time_between, trajectories_target1_with_time_to_target,
      trajectory_length (min/max/mean/median), has_dtw_distances, target_counts (target_1, target_0).
    - Each chart object includes n (sample sizes per category) where applicable for reliability/display.
    - routine_comparison, routine_comparison_counts, high_risk_trajectories, target_pathway_patterns,
      times_between_sequences, time_to_target_sequences; when event_density_bin present, *_by_density too.
    - metrics: dtw_rows, charts_built, charts_not_built, success."""
    def _log_n3(level: str, msg: str, *args: Any) -> None:
        if logger is not None:
            getattr(logger, level)(msg, *args)

    if dtw_df.empty:
        return None
    out: Dict[str, Any] = {}
    # Reusable summary for multiple visuals (counts, trajectory stats, target split)
    out["summary"] = _build_chart_data_summary(dtw_df)
    charts_built: List[str] = []
    charts_not_built: Dict[str, str] = {}

    routine = _compute_dtw_routine_comparison(dtw_df)
    if routine:
        out["routine_comparison"] = routine
        charts_built.append("routine_comparison")
    else:
        reason = _reason_routine_comparison(dtw_df)
        charts_not_built["routine_comparison"] = reason
        _log_n3("info", "routine_comparison (Routine vs No Routine outcomes): not built — %s", reason)

    routine_counts = _compute_dtw_routine_comparison_counts(dtw_df)
    if routine_counts:
        out["routine_comparison_counts"] = routine_counts
        charts_built.append("routine_comparison_counts")
    else:
        reason = _reason_routine_comparison_counts(dtw_df)
        charts_not_built["routine_comparison_counts"] = reason
        _log_n3("info", "routine_comparison_counts (medical/prescription events by routine): not built — %s", reason)

    high_risk = _compute_dtw_high_risk_trajectories(dtw_df)
    if high_risk:
        out["high_risk_trajectories"] = high_risk
        charts_built.append("high_risk_trajectories")
    else:
        charts_not_built["high_risk_trajectories"] = "missing dtw_min_distance/trajectory_length or insufficient rows" if not dtw_df.empty else "empty dataframe"

    target_pathways = _compute_target_pathway_patterns(dtw_df)
    if target_pathways:
        out["target_pathway_patterns"] = target_pathways
        charts_built.append("target_pathway_patterns")
    else:
        charts_not_built["target_pathway_patterns"] = "missing seq_pattern_str or fewer than 10 target=1 rows" if "target" in dtw_df.columns else "missing target"

    # N3: times between sequences (requires mean_days_between_events from timestamped event column)
    times_between = _compute_times_between_sequences(dtw_df)
    if times_between:
        out["times_between_sequences"] = times_between
        charts_built.append("times_between_sequences")
        _log_n3("info", "N3 times_between_sequences: built with %d categories (mean days between consecutive events by routine vs no routine)", len(times_between.get("x", [])))
    else:
        if "mean_days_between_events" not in dtw_df.columns:
            reason = "missing mean_days_between_events (run create_dtw_trajectories with timestamp column)"
            charts_not_built["times_between_sequences"] = reason
            _log_n3("info", "N3 times_between_sequences: not built — %s", reason)
        elif "admin_icd_event_count" not in dtw_df.columns:
            reason = "missing admin_icd_event_count"
            charts_not_built["times_between_sequences"] = reason
            _log_n3("info", "N3 times_between_sequences: not built — %s", reason)
        else:
            reason = "insufficient rows or no valid mean_days_between_events"
            charts_not_built["times_between_sequences"] = reason
            _log_n3("info", "N3 times_between_sequences: not built — %s", reason)

    time_to_target = _compute_time_to_target_sequences(dtw_df)
    if time_to_target:
        out["time_to_target_sequences"] = time_to_target
        charts_built.append("time_to_target_sequences")
        _log_n3("info", "N3 time_to_target_sequences: built with %d categories (mean days from first event to target by routine vs no routine)", len(time_to_target.get("x", [])))
    else:
        if "days_first_event_to_target" not in dtw_df.columns:
            reason = "missing days_first_event_to_target (run create_dtw_trajectories with timestamp column)"
            charts_not_built["time_to_target_sequences"] = reason
            _log_n3("info", "N3 time_to_target_sequences: not built — %s", reason)
        else:
            reason = "insufficient target=1 rows or no valid days_first_event_to_target"
            charts_not_built["time_to_target_sequences"] = reason
            _log_n3("info", "N3 time_to_target_sequences: not built — %s", reason)

    # Stratify by event_density_bin for dashboard filter (same bins as create_dtw_trajectories)
    if "event_density_bin" in dtw_df.columns:
        out["event_density_bins"] = list(DENSITY_BINS)
        out["routine_comparison_by_density"] = {}
        out["routine_comparison_counts_by_density"] = {}
        out["high_risk_trajectories_by_density"] = {}
        for bin_name in DENSITY_BINS:
            sub = dtw_df[dtw_df["event_density_bin"] == bin_name]
            if len(sub) < 10:
                continue
            r = _compute_dtw_routine_comparison(sub)
            if r:
                out["routine_comparison_by_density"][bin_name] = r
            rc = _compute_dtw_routine_comparison_counts(sub)
            if rc:
                out["routine_comparison_counts_by_density"][bin_name] = rc
            hr = _compute_dtw_high_risk_trajectories(sub)
            if hr:
                out["high_risk_trajectories_by_density"][bin_name] = hr

    # Always attach metrics so output is either successful JSON or JSON with explicit build status
    out["metrics"] = {
        "dtw_rows": int(len(dtw_df)),
        "charts_built": charts_built,
        "charts_not_built": charts_not_built,
        "success": len(charts_not_built) == 0,
    }
    if charts_not_built:
        out.setdefault("message", "Some charts were not built; see metrics.charts_not_built for reasons.")
    out["empty"] = len(charts_built) == 0  # true only when no charts built; frontend can show metrics
    return out if out else None


def _upload_dtw_chart_data_to_dashboard_s3(
    project_root: Path,
    cohort_name: str,
    age_band: str,
    chart_data: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> None:
    """Upload prebuilt DTW chart_data.json to dashboard bucket for direct dashboard integration.
    When SKIP_DASHBOARD_S3_UPLOAD=1, no upload (notebook 5 Step 6 syncs from local)."""
    if (os.environ.get("SKIP_DASHBOARD_S3_UPLOAD", "") or "").strip().lower() in ("1", "true", "yes"):
        if logger:
            logger.debug("SKIP_DASHBOARD_S3_UPLOAD set; DTW chart_data S3 upload skipped.")
        return
    s3_bucket = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
    dashboard_prefix = os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator")
    use_builds = (os.environ.get("S3_VISUALIZATIONS_BUILDS", "") or "").strip().lower() in ("1", "true", "yes")
    builds_suffix = "/builds" if use_builds else ""
    base_key = f"{dashboard_prefix.rstrip('/')}/visualizations/dtw{builds_suffix}/{cohort_name}/{age_band}"
    key = f"{base_key}/chart_data.json"
    s3_path = f"s3://{s3_bucket}/{key}"
    if logger:
        logger.info("DTW chart_data upload -> %s", s3_path)
    try:
        from py_helpers.checkpoint_utils import upload_file_to_s3
    except ImportError as e:
        if logger:
            logger.warning("DTW chart_data upload skipped: could not import upload_file_to_s3: %s", e)
        return
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(chart_data, f, indent=0)
        path = Path(f.name)
    try:
        if upload_file_to_s3(path, s3_path, logger=logger, check_exists=False) and logger:
            logger.info("Uploaded DTW chart_data.json to dashboard S3 %s", s3_path)
    finally:
        path.unlink(missing_ok=True)


def _upload_sequence_heatmap_to_s3(
    project_root: Path,
    cohort_name: str,
    age_band: str,
    heatmap_data: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> None:
    """Upload sequence_heatmap.json (drug, icd, cpt slices) for dashboard common-sequences heatmap.
    When SKIP_DASHBOARD_S3_UPLOAD=1, no upload (notebook 5 Step 6 syncs from local)."""
    if (os.environ.get("SKIP_DASHBOARD_S3_UPLOAD", "") or "").strip().lower() in ("1", "true", "yes"):
        if logger:
            logger.debug("SKIP_DASHBOARD_S3_UPLOAD set; DTW sequence_heatmap S3 upload skipped.")
        return
    s3_bucket = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
    dashboard_prefix = os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator")
    use_builds = (os.environ.get("S3_VISUALIZATIONS_BUILDS", "") or "").strip().lower() in ("1", "true", "yes")
    builds_suffix = "/builds" if use_builds else ""
    base_key = f"{dashboard_prefix.rstrip('/')}/visualizations/dtw{builds_suffix}/{cohort_name}/{age_band}"
    key = f"{base_key}/sequence_heatmap.json"
    s3_path = f"s3://{s3_bucket}/{key}"
    if logger:
        logger.info("DTW sequence_heatmap upload -> %s", s3_path)
    try:
        from py_helpers.checkpoint_utils import upload_file_to_s3
    except ImportError as e:
        if logger:
            logger.warning("DTW sequence_heatmap upload skipped: could not import upload_file_to_s3: %s", e)
        return
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(heatmap_data, f, indent=0)
        path = Path(f.name)
    try:
        if upload_file_to_s3(path, s3_path, logger=logger, check_exists=False) and logger:
            logger.info("Uploaded DTW sequence_heatmap.json to dashboard S3 %s", s3_path)
    finally:
        path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create and publish DTW visuals for the dashboard (copy CSV, upload plots and chart_data). "
            "Does not add DTW features to model data. Run after create_dtw_features.py."
        )
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root path (default: current directory)",
    )
    parser.add_argument(
        "--cohort-name",
        type=str,
        required=True,
        help="Cohort name (e.g., opioid_ed)",
    )
    parser.add_argument(
        "--age-band",
        type=str,
        required=True,
        help="Age band (e.g., 13-24)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if output already exists (default: skip when idempotent)",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    # If 4_model_data is not under project_root (e.g. cwd was visualizations), use repo root
    if not (project_root / "4_model_data").exists():
        project_root = REPO_ROOT
    pl = setup_pipeline_logger(
        step_name="5_dtw",
        cohort=args.cohort_name,
        age_band=args.age_band,
        script_name="create_dtw_visuals",
    )
    with function_block("5_dtw", "create_dtw_visuals", logger=pl.logger):
        pl.info("Starting DTW visuals for %s / %s", args.cohort_name, args.age_band)
        create_dtw_visuals(
            project_root=project_root,
            cohort_name=args.cohort_name,
            age_band=args.age_band,
            force=args.force,
            logger=pl.logger,
            log_path=pl.log_file_path,
        )
    pl.log_summary()


if __name__ == "__main__":
    main()
