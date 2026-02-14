#!/usr/bin/env python3
"""
Merge PGx features into a final tabular dataset.

This script combines PGx features (created by create_pgx_features_patient_level.py) 
into a final feature file ready for model training.

Output:
- Saves final merged features to: outputs/feature_engineering/pgx_added_features_{cohort}_{age_band}.csv
- This is the final file ready for joining with model_data in the final model step.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import subprocess
import shutil

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E402

from py_helpers.fe_monitor import mirror_checkpoint_to_s3  # noqa: E402


def add_pgx_features(
    project_root: Path,
    cohort_name: str,
    age_band: str,
) -> None:
    """
    Merge PGx features into a final tabular dataset.
    
    This script loads PGx features (created by create_pgx_features_patient_level.py)
    and saves them as the final feature file ready for model training.
    
    Output:
    - Saves final merged features to: outputs/feature_engineering/pgx_added_features_{cohort}_{age_band}.csv
    - This is the final file ready for joining with model_data in the final model step.
    """
    
    age_band_fname = age_band.replace("-", "_")
    
    # Load PGx features (created by create_pgx_features_patient_level.py)
    pgx_features_csv = (
        project_root
        / "5_pgx_analysis"
        / "outputs"
        / "feature_engineering"
        / f"pgx_features_{cohort_name}_{age_band_fname}.csv"
    )
    
    if not pgx_features_csv.exists():
        raise FileNotFoundError(
            f"PGx features not found: {pgx_features_csv}\n"
            f"Run create_pgx_features_patient_level.py first to generate features."
        )
    
    print(f"[INFO] Reading PGx features from {pgx_features_csv}")
    pgx_df = pd.read_csv(pgx_features_csv)
    
    # Validate: Check for duplicate columns before processing
    duplicate_cols = pgx_df.columns[pgx_df.columns.duplicated()].tolist()
    if duplicate_cols:
        raise ValueError(
            f"Duplicate columns detected in PGx features file for {cohort_name}/{age_band}: {duplicate_cols}. "
            f"File: {pgx_features_csv}. "
            f"This will cause issues in downstream processing. Please regenerate the feature file."
        )
    
    # Validate: Ensure feature column names are unique (excluding mi_person_key)
    feature_cols = [c for c in pgx_df.columns if c != "mi_person_key"]
    if len(feature_cols) != len(set(feature_cols)):
        duplicates = [col for col in feature_cols if feature_cols.count(col) > 1]
        unique_duplicates = list(set(duplicates))
        raise ValueError(
            f"Duplicate feature names detected in PGx features file for {cohort_name}/{age_band}: {unique_duplicates}. "
            f"File: {pgx_features_csv}. "
            f"Total features: {len(feature_cols)}, Unique features: {len(set(feature_cols))}. "
            f"This will cause issues in downstream processing. Please regenerate the feature file."
        )
    
    # Ensure mi_person_key column exists
    if 'mi_person_key' not in pgx_df.columns:
        raise ValueError("PGx features CSV must contain 'mi_person_key' column")
    
    # Ensure mi_person_key is string type for consistent merging
    pgx_df['mi_person_key'] = pgx_df['mi_person_key'].astype(str)
    
    print(f"[INFO] Loaded {len(pgx_df)} patients with {len(pgx_df.columns) - 1} PGx features")
    
    # Output to feature_engineering directory
    out_dir = (
        project_root
        / "5_pgx_analysis"
        / "outputs"
        / "feature_engineering"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / f"pgx_added_features_{cohort_name}_{age_band_fname}.csv"
    print(f"[INFO] Writing final PGx features to {out_path} ({len(pgx_df)} rows)")
    pgx_df.to_csv(out_path, index=False)
    
    # Upload to S3 gold location (primary: gold/pgx_features/, also mirror to legacy location)
    s3_path_primary = f"s3://pgxdatalake/gold/pgx_features/{cohort_name}/{age_band}/pgx_added_features_{cohort_name}_{age_band_fname}.csv"
    s3_path_legacy = f"s3://pgxdatalake/gold/feature_engineering/7_pgx/{cohort_name}/{age_band}/pgx_added_features_{cohort_name}_{age_band_fname}.csv"
    
    aws_cli = shutil.which("aws")
    if aws_cli:
        # Upload to primary location (gold/pgx_features/)
        try:
            print(f"[INFO] Uploading to S3 (primary): {s3_path_primary}")
            subprocess.run(
                [aws_cli, "s3", "cp", str(out_path), s3_path_primary],
                check=True,
                capture_output=True
            )
            print("[INFO] Primary S3 upload successful")
        except subprocess.CalledProcessError as e:
            print(f"[WARNING] Primary S3 upload failed: {e.stderr.decode() if e.stderr else 'Unknown error'}")
        
        # Also upload to legacy location for backward compatibility
        try:
            print(f"[INFO] Uploading to S3 (legacy): {s3_path_legacy}")
            subprocess.run(
                [aws_cli, "s3", "cp", str(out_path), s3_path_legacy],
                check=True,
                capture_output=True
            )
            print("[INFO] Legacy S3 upload successful")
        except subprocess.CalledProcessError as e:
            print(f"[WARNING] Legacy S3 upload failed: {e.stderr.decode() if e.stderr else 'Unknown error'}")
    else:
        print("[INFO] AWS CLI not found, skipping S3 upload")
    
    # Mirror PGx checkpoint to pgx-repository/5_pgx_analysis_checkpoint (best-effort)
    try:
        mirror_checkpoint_to_s3(
            feature_step="5_pgx_analysis",
            cohort=cohort_name,
            age_band=age_band,
            local_path=out_path,
            logger=None,
        )
    except Exception as exc:  # pragma: no cover - best-effort
        print(f"[WARNING] Could not mirror PGx checkpoint to S3: {exc}")

    print("[INFO] Done.")
    print(f"\nFinal output: {out_path}")
    print("Ready for joining with model_data using mi_person_key")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge PGx features into a final tabular dataset ready for model training. "
            "This is the final aggregation step after create_pgx_features_patient_level.py."
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
    add_pgx_features(
        project_root=project_root,
        cohort_name=args.cohort_name,
        age_band=args.age_band,
    )


if __name__ == "__main__":
    main()

