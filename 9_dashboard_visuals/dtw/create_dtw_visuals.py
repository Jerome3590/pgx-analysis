#!/usr/bin/env python3
"""
Create and publish DTW visuals for the dashboard (we do not add DTW features to model data).

Takes the DTW features CSV produced by create_dtw_features.py and:
- Copies to outputs/feature_engineering/dtw_added_features_{cohort}_{age_band}.csv
- Mirrors to 5_feature_engineering/feature_engineering_outputs/6_dtw/
- Uploads to S3 gold/feature_engineering and dashboard bucket (plots, chart_data.json)
DTW features remain a standalone artifact for dashboard visuals; they are not merged into model_events or model data.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import subprocess
import shutil

# Repo root; outputs go to 10_risk_dashboard/visualizations/dtw (same pattern as final_model, shap, ffa)
REPO_ROOT = Path(__file__).resolve().parents[2]
DTW_VIZ_DIR = Path(__file__).resolve().parent

def _dtw_output_root(project_root: Path) -> Path:
    """Dashboard visualization outputs (step 10); creation code lives in 9_dashboard_visuals/dtw."""
    return project_root / "10_risk_dashboard" / "visualizations" / "dtw"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # noqa: E402
if str(DTW_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(DTW_VIZ_DIR))  # noqa: E402 — so create_dtw_plots can be imported

from py_helpers.fe_monitor import mirror_checkpoint_to_s3  # noqa: E402
from py_helpers.checkpoint_utils import check_step_checkpoint_exists, save_step_checkpoint  # noqa: E402


def create_dtw_visuals(
    project_root: Path,
    cohort_name: str,
    age_band: str,
    force: bool = False,
) -> None:
    """
    Create and publish DTW visuals for the dashboard. Does not add DTW features to model data.
    Loads the DTW features CSV from create_dtw_features.py, writes a copy to
    outputs/feature_engineering/dtw_added_features_{cohort}_{age_band}.csv,
    mirrors to feature_engineering_outputs, uploads to S3, and uploads plots + chart_data to the dashboard bucket.
    If force is False and the output CSV already exists, skips (idempotent).
    """
    age_band_fname = age_band.replace("-", "_")
    dtw_out = _dtw_output_root(project_root)
    out_dir = dtw_out / "outputs" / "feature_engineering"
    out_path = out_dir / f"dtw_added_features_{cohort_name}_{age_band_fname}.csv"
    # Idempotency: skip if local output exists or pipeline checkpoint exists (aligns with pipeline_checkpoints)
    if not force and out_path.exists():
        print(f"[INFO] Output exists at {out_path}; skipping (use --force to re-run)")
        return
    if not force and check_step_checkpoint_exists("9_dashboard_visuals", cohort_name, age_band, logger=None):
        print(f"[INFO] Pipeline checkpoint exists for 9_dashboard_visuals {cohort_name}/{age_band}; skipping (use --force to re-run)")
        return

    # Load DTW features (created by create_dtw_features.py)
    dtw_features_csv = (
        _dtw_output_root(project_root)
        / "outputs"
        / "feature_engineering"
        / f"dtw_features_{cohort_name}_{age_band_fname}.csv"
    )

    if not dtw_features_csv.exists():
        print(
            f"[WARN] DTW features not found: {dtw_features_csv}\n"
            f"  Skipping (create_dtw_features.py did not produce output—often because 4_model_data for this cohort/age_band is missing)."
        )
        return

    print(f"[INFO] Reading DTW features from {dtw_features_csv}")
    dtw_df = pd.read_csv(dtw_features_csv)

    # Ensure mi_person_key column exists
    if "mi_person_key" not in dtw_df.columns:
        raise ValueError("DTW features CSV must contain 'mi_person_key' column")

    # Ensure mi_person_key is string type for consistent merging
    dtw_df["mi_person_key"] = dtw_df["mi_person_key"].astype(str)

    print(f"[INFO] Loaded {len(dtw_df)} patients with {len(dtw_df.columns) - 1} DTW features")

    # Output to feature_engineering directory
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing final DTW features to {out_path} ({len(dtw_df)} rows)")
    dtw_df.to_csv(out_path, index=False)

    # Mirror DTW features and added-features to central 5_feature_engineering/feature_engineering_outputs directory
    try:
        fe_root = (
            project_root
            / "5_feature_engineering"
            / "feature_engineering_outputs"
            / "6_dtw"
            / cohort_name
            / age_band
        )
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
                capture_output=True,
            )
            print("[INFO] S3 upload successful")
        except subprocess.CalledProcessError as e:
            print(f"[WARNING] S3 upload failed: {e.stderr.decode() if e.stderr else 'Unknown error'}")
    else:
        print("[INFO] AWS CLI not found, skipping S3 upload")

    # Pipeline checkpoint (pipeline_checkpoints/9_dashboard_visuals/) — used for idempotency and status
    s3_output_paths = [
        f"s3://pgxdatalake/gold/feature_engineering/6_dtw/{cohort_name}/{age_band}/dtw_added_features_{cohort_name}_{age_band_fname}.csv",
    ]
    try:
        save_step_checkpoint(
            "9_dashboard_visuals",
            cohort_name,
            age_band,
            metadata={"output_csv": out_path.name},
            output_paths=s3_output_paths,
            logger=None,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[WARNING] Could not save pipeline checkpoint: {exc}")

    # Optional: mirror CSV to 6_dtw_checkpoint (legacy/observability; see README_DTW_S3_CHECKPOINTS.md)
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
    except Exception as e:
        print(f"[WARNING] DTW trajectory cluster plots failed: {e}")
    _upload_dtw_plots_to_dashboard_s3(project_root, cohort_name, age_band)

    # Prebuild chart data (routine vs no routine, high-risk trajectories) and upload to dashboard S3 for direct dashboard integration
    chart_data = _build_dtw_chart_data(dtw_df)
    if chart_data:
        _upload_dtw_chart_data_to_dashboard_s3(project_root, cohort_name, age_band, chart_data)

    print("[INFO] Done.")
    print(f"\nFinal output: {out_path} (standalone DTW features for dashboard; not added to model data)")


def _upload_dtw_plots_to_dashboard_s3(
    project_root: Path,
    cohort_name: str,
    age_band: str,
) -> None:
    """Upload DTW plot PNGs and Plotly HTML to the dashboard bucket under dtw/{cohort}/{age_band}/plots/ (same pattern as FP-Growth/BupaR)."""
    age_band_fname = age_band.replace("-", "_")
    plots_dir_10d = _dtw_output_root(project_root) / "outputs" / cohort_name / age_band_fname / "plots"
    fe_plots_dir = (
        project_root / "5_feature_engineering" / "feature_engineering_outputs" / "6_dtw" / cohort_name / age_band / "plots"
    )
    plots_dir = plots_dir_10d if plots_dir_10d.exists() else (fe_plots_dir if fe_plots_dir.exists() else None)
    if plots_dir is None:
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
    return out or None


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
    create_dtw_visuals(
        project_root=project_root,
        cohort_name=args.cohort_name,
        age_band=args.age_band,
        force=args.force,
    )


if __name__ == "__main__":
    main()
