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
import sys
from pathlib import Path
import pandas as pd
import subprocess
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E402


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
        / "3b_feature_importance_eda"
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
    out_dir = (
        project_root
        / "3b_feature_importance_eda"
        / "outputs"
        / "feature_engineering"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / f"dtw_added_features_{cohort_name}_{age_band_fname}.csv"
    print(f"[INFO] Writing final DTW features to {out_path} ({len(dtw_df)} rows)")
    dtw_df.to_csv(out_path, index=False)
    
    # Upload to S3 gold location
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
    
    print("[INFO] Done.")
    print(f"\nFinal output: {out_path}")
    print("Ready for joining with model_data using mi_person_key")


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
    add_dtw_features(
        project_root=project_root,
        cohort_name=args.cohort_name,
        age_band=args.age_band,
    )


if __name__ == "__main__":
    main()

