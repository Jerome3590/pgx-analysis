#!/usr/bin/env python3
"""
Merge DTW features into a final tabular dataset.

This script combines DTW features (created by create_dtw_features.py) 
into a final feature file ready for model training.

Output:
- Saves final merged features to: outputs/feature_engineering/dtw_added_features_{cohort}_{age_band}.csv
- This is the final file ready for joining with model_data in the final model step.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import subprocess
import shutil

# Repo root (pgx-analysis) so py_helpers and 4_model_data are found when run from any cwd
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # noqa: E402

from py_helpers.fe_monitor import mirror_checkpoint_to_s3  # noqa: E402


def add_dtw_features(
    project_root: Path,
    cohort_name: str,
    age_band: str,
) -> None:
    """
    Merge DTW features into a final tabular dataset.
    
    This script loads DTW features (created by create_dtw_features.py)
    and saves them as the final feature file ready for model training.
    
    Output:
    - Saves final merged features to: outputs/feature_engineering/dtw_added_features_{cohort}_{age_band}.csv
    - This is the final file ready for joining with model_data in the final model step.
    """
    
    age_band_fname = age_band.replace("-", "_")
    
    # Load DTW features (created by create_dtw_features.py)
    dtw_features_csv = (
        project_root
        / "10d_dtw_dashboard_visual"
        / "outputs"
        / "feature_engineering"
        / f"dtw_features_{cohort_name}_{age_band_fname}.csv"
    )
    
    if not dtw_features_csv.exists():
        raise FileNotFoundError(
            f"DTW features not found: {dtw_features_csv}\n"
            f"Run create_dtw_features.py first to generate features."
        )
    
    print(f"[INFO] Reading DTW features from {dtw_features_csv}")
    dtw_df = pd.read_csv(dtw_features_csv)
    
    # Ensure mi_person_key column exists
    if 'mi_person_key' not in dtw_df.columns:
        raise ValueError("DTW features CSV must contain 'mi_person_key' column")
    
    # Ensure mi_person_key is string type for consistent merging
    dtw_df['mi_person_key'] = dtw_df['mi_person_key'].astype(str)
    
    print(f"[INFO] Loaded {len(dtw_df)} patients with {len(dtw_df.columns) - 1} DTW features")
    
    # Output to feature_engineering directory
    out_dir = project_root / "10d_dtw_dashboard_visual" / "outputs" / "feature_engineering"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / f"dtw_added_features_{cohort_name}_{age_band_fname}.csv"
    print(f"[INFO] Writing final DTW features to {out_path} ({len(dtw_df)} rows)")
    dtw_df.to_csv(out_path, index=False)

    # Mirror DTW features and added-features to central 5_feature_engineering/feature_engineering_outputs directory
    try:
        fe_root = project_root / "5_feature_engineering" / "feature_engineering_outputs" / "6_dtw" / cohort_name / age_band
        fe_root.mkdir(parents=True, exist_ok=True)

        # Copy raw DTW features
        dtw_mirror = fe_root / dtw_features_csv.name
        print(f"[INFO] Copying DTW features to {dtw_mirror}")
        shutil.copy2(dtw_features_csv, dtw_mirror)

        # Copy final added-features
        added_mirror = fe_root / out_path.name
        print(f"[INFO] Copying final DTW features to {added_mirror}")
        shutil.copy2(out_path, added_mirror)
    except Exception as e:  # pragma: no cover - best-effort mirror
        print(f"[WARNING] Could not mirror DTW features to feature_engineering_outputs: {e}")
    
    # Upload to S3 gold location (legacy feature_engineering path)
    s3_path = f"s3://pgxdatalake/gold/feature_engineering/6_dtw/{cohort_name}/{age_band}/dtw_added_features_{cohort_name}_{age_band_fname}.csv"
    
    aws_cli = shutil.which("aws")
    if aws_cli:
        try:
            print(f"[INFO] Uploading to S3: {s3_path}")
            subprocess.run(
                [aws_cli, "s3", "cp", str(out_path), s3_path],
                check=True,
                capture_output=True
            )
            print("[INFO] S3 upload successful")
        except subprocess.CalledProcessError as e:
            print(f"[WARNING] S3 upload failed: {e.stderr.decode() if e.stderr else 'Unknown error'}")
    else:
        print("[INFO] AWS CLI not found, skipping S3 upload")
    
    # Mirror checkpoint CSV to pgx-repository/6_dtw_checkpoint (best-effort)
    try:
        mirror_checkpoint_to_s3(
            feature_step="6_dtw",
            cohort=cohort_name,
            age_band=age_band,
            local_path=out_path,
            logger=None,
        )
    except Exception as exc:  # pragma: no cover - best-effort
        print(f"[WARNING] Could not mirror DTW checkpoint to S3: {exc}")

    # Upload DTW plot PNGs to dashboard bucket (same pattern as FP-Growth/BupaR)
    _upload_dtw_plots_to_dashboard_s3(project_root, cohort_name, age_band)

    # Prebuild chart data (routine vs no routine, high-risk trajectories) and upload to dashboard S3 for direct dashboard integration
    chart_data = _build_dtw_chart_data(dtw_df)
    if chart_data:
        _upload_dtw_chart_data_to_dashboard_s3(project_root, cohort_name, age_band, chart_data)

    print("[INFO] Done.")
    print(f"\nFinal output: {out_path}")
    print("Ready for joining with model_data using mi_person_key")


def _upload_dtw_plots_to_dashboard_s3(
    project_root: Path,
    cohort_name: str,
    age_band: str,
) -> None:
    """Upload DTW plot PNGs to the dashboard bucket under dtw/{cohort}/{age_band}/plots/ (same pattern as FP-Growth/BupaR)."""
    age_band_fname = age_band.replace("-", "_")
    plots_dir_10d = project_root / "10d_dtw_dashboard_visual" / "outputs" / cohort_name / age_band_fname / "plots"
    fe_plots_dir = project_root / "5_feature_engineering" / "feature_engineering_outputs" / "6_dtw" / cohort_name / age_band / "plots"
    plots_dir = plots_dir_10d if plots_dir_10d.exists() else (fe_plots_dir if fe_plots_dir.exists() else None)
    if plots_dir is None or not list(plots_dir.glob("*.png")):
        return

    s3_bucket = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
    dashboard_prefix = os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator")
    s3_prefix = f"{dashboard_prefix.rstrip('/')}/dtw/{cohort_name}/{age_band}/plots"

    try:
        from py_helpers.checkpoint_utils import upload_file_to_s3
    except ImportError:
        return

    uploaded = 0
    for p in plots_dir.glob("*.png"):
        key = f"{s3_prefix}/{p.name}"
        s3_path = f"s3://{s3_bucket}/{key}"
        if upload_file_to_s3(p, s3_path, logger=None, check_exists=True):
            uploaded += 1
    if uploaded:
        print(f"[INFO] Uploaded {uploaded} DTW plot(s) to dashboard S3 s3://{s3_bucket}/{s3_prefix}/")


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
    agg = use_df.groupby("bucket", as_index=False).agg(target_rate=("target", "mean"), n=("target", "count"))
    order = (
        ["No routine appointments (0 admin ICD events)", "Routine appointments (1+ admin ICD events)"]
        if "admin_icd_event_count" in df.columns
        else ["Low (fewer events)", "Medium", "High (more events)"]
    )
    agg = agg.set_index("bucket").reindex([b for b in order if b in agg.index]).reset_index()
    agg = agg.dropna(subset=["target_rate"])
    if agg.empty or agg["n"].sum() == 0:
        return None
    return {
        "x": agg["bucket"].astype(str).tolist(),
        "y": [float(round(v, 4)) for v in agg["target_rate"]],
        "type": "bar",
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
    use_df = df[["target", col]].dropna()
    if len(use_df) < 10:
        return None
    try:
        use_df = use_df.copy()
        use_df["q"] = pd.qcut(
            use_df[col], q=4, labels=["Q1 (closest)", "Q2", "Q3", "Q4 (furthest)"], duplicates="drop"
        )
    except (ValueError, TypeError):
        return None
    agg = use_df.groupby("q", as_index=False).agg(target_rate=("target", "mean"), n=("target", "count"))
    if agg.empty or agg["n"].sum() == 0:
        return None
    return {
        "x": [str(v) for v in agg["q"]],
        "y": [float(round(v, 4)) for v in agg["target_rate"]],
        "type": "bar",
        "x_label": "Trajectory archetype (by DTW distance)" if col == "dtw_min_distance" else "Trajectory archetype (by length)",
        "y_label": "Target outcome rate",
    }


def _build_dtw_chart_data(dtw_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Build routine_comparison and high_risk_trajectories chart payloads for dashboard. Prebuilt on EC2."""
    if dtw_df.empty:
        return None
    out = {}
    routine = _compute_dtw_routine_comparison(dtw_df)
    if routine:
        out["routine_comparison"] = routine
    high_risk = _compute_dtw_high_risk_trajectories(dtw_df)
    if high_risk:
        out["high_risk_trajectories"] = high_risk
    return out if out else None


def _upload_dtw_chart_data_to_dashboard_s3(
    project_root: Path,
    cohort_name: str,
    age_band: str,
    chart_data: Dict[str, Any],
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
        if upload_file_to_s3(path, s3_path, logger=None, check_exists=False):
            print(f"[INFO] Uploaded DTW chart_data.json to dashboard S3 {s3_path}")
    finally:
        path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge DTW features into a final tabular dataset ready for model training. "
            "This is the final aggregation step after create_dtw_features.py."
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
        help="Age band (e.g., 0-12)",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    # If 4_model_data is not under project_root (e.g. cwd was visualizations), use repo root
    if not (project_root / "4_model_data").exists():
        project_root = REPO_ROOT
    add_dtw_features(
        project_root=project_root,
        cohort_name=args.cohort_name,
        age_band=args.age_band,
    )


if __name__ == "__main__":
    main()

