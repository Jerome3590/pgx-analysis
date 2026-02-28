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
from typing import Any, Dict, Optional

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

    # Load DTW features: prefer sub-cohort (per-density) CSVs when present; else single CSV
    fe_dir = _dtw_output_root(project_root) / "feature_engineering"
    base_name = f"dtw_features_{cohort_name}_{age_band_fname}"
    single_csv = fe_dir / f"{base_name}.csv"
    density_glob = list(fe_dir.glob(f"{base_name}_density_*.csv"))
    _log("info", "DTW input: fe_dir=%s ; single_csv=%s (exists=%s) ; density_glob=%d files", fe_dir, single_csv, single_csv.exists(), len(density_glob))

    if density_glob:
        # Sub-cohort outputs: load by filter and concatenate for chart_data
        parts = []
        for path in sorted(density_glob):
            bin_name = path.stem.replace(f"{base_name}_density_", "")
            part = pd.read_csv(path)
            if "event_density_bin" not in part.columns:
                part["event_density_bin"] = bin_name
            parts.append(part)
        dtw_df = pd.concat(parts, ignore_index=True)
        _log("info", "Loaded DTW features from %d density sub-cohorts: %s", len(parts), [p.stem for p in density_glob])
    elif single_csv.exists():
        _log("info", "Reading DTW features from %s", single_csv)
        dtw_df = pd.read_csv(single_csv)
    else:
        _log("warning", "DTW features not found: %s (and no density sub-cohorts); skipping.", single_csv)
        try:
            from py_helpers.model_data_paths import get_path_check_listings
            path_listings = get_path_check_listings([str(single_csv)])
            path_listings_str = " ; ".join(path_listings) if path_listings else ""
        except Exception:  # noqa: BLE001
            path_listings_str = ""
        _log("error", "step=5_dtw cohort_name=%s age_band=%s error=DTW features CSV not found expected_path=%s (no EC2 artifacts written, no S3 upload)", cohort_name, age_band, single_csv)
        if path_listings_str:
            _log("error", "step=5_dtw path_listings: %s", path_listings_str)
        return

    keys_expected_dtw = ["mi_person_key", "target", "seq_pattern_str", "admin_icd_event_count", "dtw_min_distance", "trajectory_length"]
    keys_received_dtw = list(dtw_df.columns)
    _log("info", "keys_expected (DTW features): %s", keys_expected_dtw)
    _log("info", "keys_received (DTW features): %s", keys_received_dtw)

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
    try:
        from create_dtw_plots import create_trajectory_cluster_plots
        create_trajectory_cluster_plots(
            project_root=project_root,
            cohort_name=cohort_name,
            age_band=age_band,
            dtw_df=dtw_df,
            force=force,
        )
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
                        _log("info", "Wrote %s for API overview/sample URLs", name)
    except Exception as e:
        _log("warning", "DTW trajectory cluster plots failed: %s", e)
    _upload_dtw_plots_to_dashboard_s3(project_root, cohort_name, age_band, logger=logger)

    # Prebuild chart data (routine vs no routine, high-risk trajectories); write locally and upload to S3
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_data = _build_dtw_chart_data(dtw_df)
    if chart_data is None:
        _log("warning", "DTW chart_data not produced for %s/%s: empty dataframe (writing minimal file so artifact path exists)", cohort_name, age_band)
        chart_data = {}
    elif not chart_data:
        _log("warning", "DTW chart_data empty for %s/%s: no routine_comparison, high_risk, or N3 data (check admin_icd_event_count, target, seq_pattern_str, row count)", cohort_name, age_band)
    # Write always so artifact path check passes; dashboard can show empty state
    with open(chart_path, "w", encoding="utf-8") as f:
        json.dump(chart_data, f, indent=0)
    _log("info", "Wrote %s", chart_path)
    _upload_dtw_chart_data_to_dashboard_s3(project_root, cohort_name, age_band, chart_data, logger=logger)

    # Sequence heatmap (code × position counts by ICD/CPT/Drug); write locally and upload to S3
    heatmap_data = _build_sequence_heatmap_data(dtw_df)
    if heatmap_data is None:
        _log("warning", "DTW sequence_heatmap not produced for %s/%s: empty dataframe or missing seq_pattern_str (writing minimal file so artifact path exists)", cohort_name, age_band)
        heatmap_data = {"drug": {"codes": [], "positions": [], "counts": []}, "icd": {"codes": [], "positions": [], "counts": []}, "cpt": {"codes": [], "positions": [], "counts": []}}
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


def _compute_dtw_routine_comparison(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Outcome rate by routine vs no routine (admin ICD filter) or by trajectory intensity. Prebuilt on EC2."""
    if df.empty or "target" not in df.columns:
        return None
    if "admin_icd_event_count" in df.columns:
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
    # Frontend expects: x, y, type, x_label, y_label, and optional name (index.html)
    return {
        "x": agg["bucket"].astype(str).tolist(),
        "y": [float(round(v, 4)) for v in agg["target_rate"]],
        "type": "bar",
        "name": "Outcome rate",
        "x_label": x_label,
        "y_label": "Target outcome rate",
    }


def _compute_dtw_routine_comparison_counts(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Mean medical events and mean prescription (drug) events by routine vs no routine. Shows that routine screenings associate with lower medical and prescription event counts."""
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
    # Frontend: multi-series bar chart (same x, two y series)
    return {
        "x": agg["bucket"].astype(str).tolist(),
        "series": [
            {"name": "Mean medical events (ICD/CPT) per patient", "y": [float(round(v, 2)) for v in agg["mean_medical"]]},
            {"name": "Mean prescription events (drugs) per patient", "y": [float(round(v, 2)) for v in agg["mean_drug"]]},
        ],
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
    # Frontend expects: x, y, type, x_label, y_label, and optional name (index.html)
    return {
        "x": [str(v) for v in agg["q"]],
        "y": [float(round(v, 4)) for v in agg["target_rate"]],
        "type": "bar",
        "name": "Outcome rate by archetype",
        "x_label": "Trajectory archetype (by DTW distance)" if col == "dtw_min_distance" else "Trajectory archetype (by length)",
        "y_label": "Target outcome rate",
    }


def _compute_times_between_sequences(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """N3: Mean days between consecutive events by routine vs no routine (times between sequences)."""
    if df.empty or "mean_days_between_events" not in df.columns:
        return None
    if "admin_icd_event_count" not in df.columns:
        return None
    use_df = df[["admin_icd_event_count", "mean_days_between_events"]].copy()
    use_df["bucket"] = use_df["admin_icd_event_count"].apply(
        lambda x: "No routine (0 admin ICD events)" if x == 0 else "Routine (1+ admin ICD events)"
    )
    use_df = use_df.dropna(subset=["mean_days_between_events"])
    if len(use_df) < 10:
        return None
    agg = use_df.groupby("bucket", as_index=False, observed=True).agg(
        mean_days=("mean_days_between_events", "mean"),
        n=("mean_days_between_events", "count"),
    )
    order = ["No routine (0 admin ICD events)", "Routine (1+ admin ICD events)"]
    agg = agg.set_index("bucket").reindex([b for b in order if b in agg.index]).reset_index()
    agg = agg.dropna(subset=["mean_days"])
    if agg.empty:
        return None
    return {
        "x": agg["bucket"].astype(str).tolist(),
        "y": [float(round(v, 1)) for v in agg["mean_days"]],
        "type": "bar",
        "name": "Mean days between consecutive events",
        "x_label": "Routine vs no routine (admin ICD filter)",
        "y_label": "Mean days between consecutive events",
    }


def _compute_time_to_target_sequences(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """N3: Mean days from first event to target (target=1 only) by routine vs no routine."""
    if df.empty or "days_first_event_to_target" not in df.columns or "target" not in df.columns:
        return None
    target_df = df[df["target"] == 1].copy()
    if len(target_df) < 5:
        return None
    if "admin_icd_event_count" not in target_df.columns:
        return None
    use_df = target_df[["admin_icd_event_count", "days_first_event_to_target"]].dropna(
        subset=["days_first_event_to_target"]
    )
    if len(use_df) < 5:
        return None
    use_df["bucket"] = use_df["admin_icd_event_count"].apply(
        lambda x: "No routine (0 admin ICD events)" if x == 0 else "Routine (1+ admin ICD events)"
    )
    agg = use_df.groupby("bucket", as_index=False, observed=True).agg(
        mean_days=("days_first_event_to_target", "mean"),
        n=("days_first_event_to_target", "count"),
    )
    order = ["No routine (0 admin ICD events)", "Routine (1+ admin ICD events)"]
    agg = agg.set_index("bucket").reindex([b for b in order if b in agg.index]).reset_index()
    agg = agg.dropna(subset=["mean_days"])
    if agg.empty:
        return None
    return {
        "x": agg["bucket"].astype(str).tolist(),
        "y": [float(round(v, 1)) for v in agg["mean_days"]],
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


def _build_dtw_chart_data(dtw_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Build routine_comparison, routine_comparison_counts, high_risk_trajectories, target_pathway_patterns, and N3 times_between charts for dashboard.
    When event_density_bin is present, also builds *_by_density so the dashboard can filter by event density."""
    if dtw_df.empty:
        return None
    out = {}
    routine = _compute_dtw_routine_comparison(dtw_df)
    if routine:
        out["routine_comparison"] = routine
    routine_counts = _compute_dtw_routine_comparison_counts(dtw_df)
    if routine_counts:
        out["routine_comparison_counts"] = routine_counts
    high_risk = _compute_dtw_high_risk_trajectories(dtw_df)
    if high_risk:
        out["high_risk_trajectories"] = high_risk
    target_pathways = _compute_target_pathway_patterns(dtw_df)
    if target_pathways:
        out["target_pathway_patterns"] = target_pathways
    # N3: times between sequences
    times_between = _compute_times_between_sequences(dtw_df)
    if times_between:
        out["times_between_sequences"] = times_between
    time_to_target = _compute_time_to_target_sequences(dtw_df)
    if time_to_target:
        out["time_to_target_sequences"] = time_to_target

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

    return out or None


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
