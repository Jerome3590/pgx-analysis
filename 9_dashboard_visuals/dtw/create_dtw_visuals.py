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
- chart_data.json: _build_dtw_chart_data(dtw_df) builds three charts:
  1. routine_comparison: outcome rate by routine vs no routine appointments (admin ICD filter)
  2. high_risk_trajectories: outcome rate by trajectory archetype (quartiles)
  3. target_pathway_patterns: common codes in target=1 trajectories (shows what leads to adverse events)
  Frontend (index.html) expects chart_data_url -> JSON with these three chart objects (x, y, type, name, x_label, y_label).
- Outputs: outputs/{cohort}/{age_band}/plots/*.png/html + chart_data.json uploaded to S3 dashboard bucket.
  DTW CSV files (dtw_features, dtw_added_features) are NOT uploaded; dashboard only uses plots and chart_data.
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

# Repo root; outputs go to 10_risk_dashboard/visualizations/dtw (same pattern as final_model, shap, ffa)
REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DTW_VIZ_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from py_helpers.fe_monitor import function_block, mirror_log_to_s3  # noqa: E402


def _get_logger(cohort_name: str, age_band: str) -> tuple[logging.Logger, Path]:
    """Create a logger with both console and file handlers (same pattern as BupaR/FP-Growth). Logs under repo root: pgx-analysis/9_dashboard_visuals/logs/5_dtw."""
    logs_dir = REPO_ROOT / "9_dashboard_visuals" / "logs" / "5_dtw"
    logs_dir.mkdir(parents=True, exist_ok=True)
    age_band_fname = age_band.replace("-", "_")
    log_path = logs_dir / f"dtw_{cohort_name}_{age_band_fname}.log"
    logger = logging.getLogger(f"dtw.{cohort_name}.{age_band_fname}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    logger.propagate = False
    return logger, log_path


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

    # Idempotency: check if plots directory exists (dashboard deliverables)
    plots_dir = dtw_out / "outputs" / cohort_name / age_band_fname / "plots"
    if not force and plots_dir.exists() and list(plots_dir.glob("*.png")):
        _log("info", "DTW plots exist at %s; skipping (use --force to re-run)", plots_dir)
        return
    if not force and check_step_checkpoint_exists("9_dashboard_visuals", cohort_name, age_band, logger=logger):
        _log("info", "Pipeline checkpoint exists for 9_dashboard_visuals %s/%s; skipping (use --force to re-run)", cohort_name, age_band)
        return

    # Load DTW features (created by create_dtw_features.py)
    dtw_features_csv = (
        _dtw_output_root(project_root)
        / "outputs"
        / "feature_engineering"
        / f"dtw_features_{cohort_name}_{age_band_fname}.csv"
    )

    if not dtw_features_csv.exists():
        _log("warning", "DTW features not found: %s; skipping (create_dtw_features.py did not produce output).", dtw_features_csv)
        try:
            from py_helpers.model_data_paths import get_path_check_listings
            path_listings = get_path_check_listings([str(dtw_features_csv)])
            path_listings_str = " ; ".join(path_listings) if path_listings else ""
        except Exception:  # noqa: BLE001
            path_listings_str = ""
        _log("error", "step=5_dtw cohort_name=%s age_band=%s error=DTW features CSV not found expected_path=%s", cohort_name, age_band, dtw_features_csv)
        if path_listings_str:
            _log("error", "step=5_dtw path_listings: %s", path_listings_str)
        return

    _log("info", "Reading DTW features from %s", dtw_features_csv)
    dtw_df = pd.read_csv(dtw_features_csv)

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
        plots_dir = _dtw_output_root(project_root) / "outputs" / cohort_name / age_band_fname / "plots"
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

    # Prebuild chart data (routine vs no routine, high-risk trajectories) and upload to dashboard S3 for direct dashboard integration
    chart_data = _build_dtw_chart_data(dtw_df)
    if chart_data:
        _upload_dtw_chart_data_to_dashboard_s3(project_root, cohort_name, age_band, chart_data, logger=logger)

    # Sequence heatmap (code × position counts by ICD/CPT/Drug) for dashboard heatmap with dynamic code-type filter
    heatmap_data = _build_sequence_heatmap_data(dtw_df)
    if heatmap_data:
        _upload_sequence_heatmap_to_s3(project_root, cohort_name, age_band, heatmap_data, logger=logger)

    # Save pipeline checkpoint (dashboard artifacts complete: plots + chart_data)
    s3_output_paths = [
        f"s3://{os.environ.get('S3_DASHBOARD_BUCKET', 'jerome-dixon.io')}/{os.environ.get('S3_DASHBOARD_PREFIX', 'vcu/pgx-risk-calculator')}/dtw/{cohort_name}/{age_band}/plots/"
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

    _log("info", "Done.")
    _log("info", "DTW visuals complete. Plots and chart_data uploaded to dashboard S3: trajectory cluster plots (3D/1D), chart_data.json, sequence_heatmap.json. CSV files not uploaded; dashboard uses plots only.")
    if log_path and logger:
        mirror_log_to_s3("5_dtw", cohort_name, age_band, log_path, logger)



def _upload_dtw_plots_to_dashboard_s3(
    project_root: Path,
    cohort_name: str,
    age_band: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Upload DTW plot PNGs and Plotly HTML to the dashboard bucket under dtw/{cohort}/{age_band}/plots/ (same pattern as FP-Growth/BupaR)."""
    age_band_fname = age_band.replace("-", "_")
    plots_dir = _dtw_output_root(project_root) / "outputs" / cohort_name / age_band_fname / "plots"
    if not plots_dir.exists():
        return
    plot_files = list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.html"))
    if not plot_files:
        return

    s3_bucket = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
    dashboard_prefix = os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator")
    s3_prefix = f"{dashboard_prefix.rstrip('/')}/dtw/{cohort_name}/{age_band}/plots"

    try:
        from py_helpers.checkpoint_utils import upload_file_to_s3
    except ImportError:
        return

    uploaded = 0
    for p in plot_files:
        key = f"{s3_prefix}/{p.name}"
        s3_path = f"s3://{s3_bucket}/{key}"
        if upload_file_to_s3(p, s3_path, logger=logger, check_exists=True):
            uploaded += 1
    if uploaded and logger:
        logger.info("Uploaded %d DTW plot(s) to dashboard S3 s3://%s/%s/", uploaded, s3_bucket, s3_prefix)


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
    Build heatmap data by code type: for each type (ICD, CPT, Drug), matrix of
    (code × position) counts. Darker = more trajectories have that code at that position.
    Returns dict with keys 'icd', 'cpt', 'drug'; each value has codes, positions, counts (2D).
    """
    if dtw_df.empty or "seq_pattern_str" not in dtw_df.columns:
        return None
    skip = {"nan", "none", "null", ""}
    # (code_type -> (code -> (position -> count)))
    by_type: Dict[str, Dict[str, Dict[int, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    max_pos = 0
    for seq in dtw_df["seq_pattern_str"]:
        if pd.isna(seq) or not isinstance(seq, str):
            continue
        tokens = [t.strip() for t in seq.split("_") if t.strip() and t.strip().lower() not in skip]
        for pos, token in enumerate(tokens):
            if ":" in token:
                prefix, code = token.split(":", 1)
                prefix = prefix.strip().upper()
                code_val = code.strip() if code else token
                if not code_val:
                    continue
                if prefix == "ICD":
                    by_type["icd"][code_val][pos] += 1
                elif prefix == "CPT":
                    by_type["cpt"][code_val][pos] += 1
                elif prefix == "DRUG":
                    by_type["drug"][code_val][pos] += 1
            max_pos = max(max_pos, pos)
        max_pos = max(max_pos, len(tokens) - 1) if tokens else max_pos
    n_cols = max_pos + 1  # 0-indexed to column count
    out = {}
    for code_type in ("icd", "cpt", "drug"):
        code_to_pos_counts = by_type.get(code_type)
        if not code_to_pos_counts:
            out[code_type] = {"codes": [], "positions": list(range(n_cols)), "counts": []}
            continue
        codes = sorted(code_to_pos_counts.keys())
        positions = list(range(n_cols))
        counts = []
        for code in codes:
            row = [code_to_pos_counts[code].get(p, 0) for p in positions]
            counts.append(row)
        out[code_type] = {"codes": codes, "positions": positions, "counts": counts}
    return out


def _build_dtw_chart_data(dtw_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Build routine_comparison, high_risk_trajectories, target_pathway_patterns, and N3 times_between charts for dashboard."""
    if dtw_df.empty:
        return None
    out = {}
    routine = _compute_dtw_routine_comparison(dtw_df)
    if routine:
        out["routine_comparison"] = routine
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
    return out or None


def _upload_dtw_chart_data_to_dashboard_s3(
    project_root: Path,
    cohort_name: str,
    age_band: str,
    chart_data: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> None:
    """Upload prebuilt DTW chart_data.json to dashboard bucket for direct dashboard integration."""
    s3_bucket = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
    dashboard_prefix = os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator")
    base_key = f"{dashboard_prefix.rstrip('/')}/dtw/{cohort_name}/{age_band}"
    key = f"{base_key}/chart_data.json"
    try:
        from py_helpers.checkpoint_utils import upload_file_to_s3
    except ImportError:
        return
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(chart_data, f, indent=0)
        path = Path(f.name)
    try:
        s3_path = f"s3://{s3_bucket}/{key}"
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
    """Upload sequence_heatmap.json (icd/cpt/drug by position) for dashboard heatmap."""
    s3_bucket = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
    dashboard_prefix = os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator")
    base_key = f"{dashboard_prefix.rstrip('/')}/dtw/{cohort_name}/{age_band}"
    key = f"{base_key}/sequence_heatmap.json"
    try:
        from py_helpers.checkpoint_utils import upload_file_to_s3
    except ImportError:
        return
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(heatmap_data, f, indent=0)
        path = Path(f.name)
    try:
        s3_path = f"s3://{s3_bucket}/{key}"
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
    logger, log_path = _get_logger(args.cohort_name, args.age_band)
    with function_block("5_dtw", "create_dtw_visuals", logger=logger):
        logger.info("Starting DTW visuals for %s / %s", args.cohort_name, args.age_band)
        create_dtw_visuals(
            project_root=project_root,
            cohort_name=args.cohort_name,
            age_band=args.age_band,
            force=args.force,
            logger=logger,
            log_path=log_path,
        )


if __name__ == "__main__":
    main()
