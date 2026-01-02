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

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E402

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
        / "5d_dtw_analysis"
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
    out_dir = project_root / "5d_dtw_analysis" / "outputs" / "feature_engineering"
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

