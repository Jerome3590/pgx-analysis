#!/usr/bin/env python3
"""
Merge FP-Growth features into a standalone tabular file for dashboard visualization only.

We do NOT add FP-Growth (or DTW) features to model data. This script combines FP-Growth
features (created by create_fpgrowth_features.py) into a single CSV used only for
dashboard visuals (itemsets, rules, network plots). The final model step does not use
these features.

Output:
- Saves to: outputs/feature_engineering/fpgrowth_added_features_{cohort}_{age_band}.csv
- Used by dashboard/visualizations only; not merged into 4_model_data or 6_final_model.
"""

import argparse
import sys
import subprocess
import shutil
from pathlib import Path

import pandas as pd

# Add project root to path
# Script lives in 9_dashboard_visuals/fpgrowth; outputs go to 10_risk_dashboard/visualizations/fpgrowth
REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FPGROWTH_OUT = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "fpgrowth"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPO_ROOT))


def add_fpgrowth_features(
    project_root: Path,
    cohort_name: str,
    age_band: str,
) -> None:
    """
    Merge FP-Growth features into a standalone CSV for dashboard visualization only.
    We do not add FP-Growth features to model data; they are not used in 6_final_model.
    """
    
    age_band_fname = age_band.replace("-", "_")
    
    # Load FP-Growth features (created by create_fpgrowth_features.py)
    fpgrowth_features_csv = (
        FPGROWTH_OUT
        / "outputs"
        / "feature_engineering"
        / f"fpgrowth_features_{cohort_name}_{age_band_fname}.csv"
    )
    
    if not fpgrowth_features_csv.exists():
        raise FileNotFoundError(
            f"FP-Growth features not found: {fpgrowth_features_csv}\n"
            f"Run create_fpgrowth_features.py first to generate features."
        )
    
    print(f"[INFO] Reading FP-Growth features from {fpgrowth_features_csv}")
    try:
        # Primary read – should succeed for well-formed CSVs.
        fpgrowth_df = pd.read_csv(fpgrowth_features_csv)
    except pd.errors.ParserError as e:
        # Fallback: use python engine and skip malformed lines, logging a warning.
        print(
            "[WARNING] ParserError while reading FP-Growth features "
            f"(will retry with python engine, skipping bad lines): {e}"
        )
        fpgrowth_df = pd.read_csv(
            fpgrowth_features_csv,
            engine="python",
            on_bad_lines="skip",
        )
    
    # Ensure mi_person_key column exists
    if 'mi_person_key' not in fpgrowth_df.columns:
        raise ValueError("FP-Growth features CSV must contain 'mi_person_key' column")
    
    print(f"[INFO] Loaded {len(fpgrowth_df)} patients with {len(fpgrowth_df.columns) - 1} FP-Growth features")
    
    # Output to feature_engineering directory (step 10: risk dashboard visualization outputs)
    out_dir = FPGROWTH_OUT / "outputs" / "feature_engineering"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / f"fpgrowth_added_features_{cohort_name}_{age_band_fname}.csv"
    print(f"[INFO] Writing final FP-Growth features to {out_path} ({len(fpgrowth_df)} rows)")
    fpgrowth_df.to_csv(out_path, index=False)

    # Upload to S3 gold location
    s3_path = f"s3://pgxdatalake/gold/feature_engineering/4_fpgrowth/{cohort_name}/{age_band}/fpgrowth_added_features_{cohort_name}_{age_band_fname}.csv"
    
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
    
    print("[INFO] Done.")
    print(f"\nFinal output: {out_path} (dashboard visualization only; not added to model data)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge FP-Growth features into a standalone CSV for dashboard visualization only. "
            "FP-Growth features are not added to model data. Run after create_fpgrowth_features.py."
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
    add_fpgrowth_features(
        project_root=project_root,
        cohort_name=args.cohort_name,
        age_band=args.age_band,
    )


if __name__ == "__main__":
    main()

