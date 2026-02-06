#!/usr/bin/env python3
"""
Prepare and upload train/test datasets to S3 for final model training.

Saving to S3 is **required** (not optional): SHAP (Step 7) and FFA (Step 8) analysis
depend on model training input data in S3 when run in separate environments or
when syncing from S3. This script fails if upload fails.

This script:
1. Loads the final feature table
2. Splits into train (2016-2018) and test (2019) using temporal validation
3. Saves train and test datasets to S3 gold location (required)
4. Organizes by cohort and age_band

S3 Structure:
- s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/inputs/model_train/final_features.parquet
- s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/inputs/model_test/final_features.parquet

Usage:
    python prepare_train_test_s3.py --cohort-name opioid_ed --age-band 0-12
"""

import argparse
import sys
import subprocess
import shutil
from pathlib import Path
import pandas as pd
import duckdb
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def prepare_train_test_s3(
    project_root: Path,
    cohort_name: str,
    age_band: str,
) -> None:
    """Prepare train/test datasets and upload to S3."""
    
    age_band_fname = age_band.replace("-", "_")
    
    # Load final feature table (prefer no_leakage version)
    # Try 6_final_model first (current structure), then 8_final_model (legacy)
    feature_table_path = (
        project_root
        / "6_final_model"
        / "outputs"
        / cohort_name
        / age_band_fname
        / f"{cohort_name}_{age_band_fname}_train_final_features_no_leakage.csv"
    )
    
    # Fallback to regular version if no_leakage doesn't exist
    if not feature_table_path.exists():
        feature_table_path = (
            project_root
            / "6_final_model"
            / "outputs"
            / cohort_name
            / age_band_fname
            / f"{cohort_name}_{age_band_fname}_train_final_features.csv"
        )
    
    # Legacy fallback: try 8_final_model
    if not feature_table_path.exists():
        feature_table_path = (
            project_root
            / "8_final_model"
            / "outputs"
            / cohort_name
            / age_band_fname
            / f"{cohort_name}_{age_band_fname}_train_final_features_no_leakage.csv"
        )
    
    if not feature_table_path.exists():
        feature_table_path = (
            project_root
            / "8_final_model"
            / "outputs"
            / cohort_name
            / age_band_fname
            / f"{cohort_name}_{age_band_fname}_train_final_features.csv"
        )
    
    if not feature_table_path.exists():
        raise FileNotFoundError(f"Feature table not found: {feature_table_path}")
    
    print(f"[INFO] Loading feature table from {feature_table_path}")
    df = pd.read_csv(feature_table_path)
    
    # Ensure mi_person_key is string type
    df['mi_person_key'] = df['mi_person_key'].astype(str)
    
    print(f"[INFO] Loaded {len(df)} patients with {len(df.columns)} columns")
    
    # Load model_data to get event_year information for temporal split (single canonical location)
    from py_helpers.env_utils import get_model_data_root

    model_data_root = get_model_data_root()
    canonical_path = (
        model_data_root
        / f"cohort_name={cohort_name}"
        / f"age_band={age_band}"
        / "model_events.parquet"
    )
    
    model_data_path = canonical_path if canonical_path.exists() else None
    
    # If not found locally, try downloading from S3
    if model_data_path is None:
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            s3_client = boto3.client("s3")
            S3_BUCKET = "pgxdatalake"
            s3_key = f"gold/cohorts_model_data/cohort_name={cohort_name}/age_band={age_band}/model_events.parquet"
            
            # Download to canonical location
            local_download_path = canonical_path
            local_download_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"[INFO] model_events.parquet not found locally. Checking S3: s3://{S3_BUCKET}/{s3_key}")
            try:
                s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
                print(f"[INFO] Downloading from S3 to {local_download_path}")
                s3_client.download_file(S3_BUCKET, s3_key, str(local_download_path))
                model_data_path = local_download_path
                print(f"[OK] Downloaded model_events.parquet from S3")
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    raise FileNotFoundError(
                        f"model_data not found locally or in S3. Checked:\n"
                        f"  - {canonical_path}\n"
                        f"  - s3://{S3_BUCKET}/{s3_key}\n"
                        f"Please run Step 4 first to create model_events.parquet"
                    )
                else:
                    raise
        except ImportError:
            raise FileNotFoundError(
                f"model_data not found locally. Checked:\n"
                f"  - {canonical_path}\n"
                f"boto3 not available for S3 download. Please run Step 4 first to create model_events.parquet"
            )
    
    if model_data_path is None or not model_data_path.exists():
        raise FileNotFoundError(f"model_data not found: {model_data_path}")
    
    print(f"[INFO] Loading event year information from {model_data_path}")
    con = duckdb.connect()
    
    # Get max event year per patient (to determine train vs test)
    # Include both target and control patients
    patient_years_df = con.execute(
        f"""
        SELECT 
            CAST(mi_person_key AS VARCHAR) AS mi_person_key,
            MAX(event_year) AS max_event_year,
            MIN(event_year) AS min_event_year,
            MAX(target) AS target  -- Preserve target label
        FROM read_parquet('{model_data_path}')
        WHERE target IN (0, 1)
        GROUP BY mi_person_key
        """
    ).df()
    con.close()
    
    patient_years_df['mi_person_key'] = patient_years_df['mi_person_key'].astype(str)
    
    print(f"[INFO] Loaded event year info for {len(patient_years_df)} patients")
    print(f"[INFO] Target distribution: {patient_years_df['target'].value_counts().to_dict()}")
    print(f"[INFO] Year distribution:")
    print(patient_years_df['max_event_year'].value_counts().sort_index())
    
    # Merge year info with feature table
    # Use left merge to keep all patients from feature table (including controls that might not be in model_data)
    # Only merge year columns, preserve target column from feature table
    df_with_years = df.merge(
        patient_years_df[['mi_person_key', 'max_event_year', 'min_event_year']], 
        on='mi_person_key', 
        how='left'
    )
    
    # If target column is missing, try to get it from patient_years_df (shouldn't happen, but handle gracefully)
    if 'target' not in df_with_years.columns and 'target' in patient_years_df.columns:
        df_with_years = df_with_years.merge(
            patient_years_df[['mi_person_key', 'target']],
            on='mi_person_key',
            how='left'
        )
    
    # For patients without year info (shouldn't happen, but handle gracefully), assign to train
    df_with_years['max_event_year'] = df_with_years['max_event_year'].fillna(2018)
    
    # Temporal split: train = 2016-2018, test = 2019
    train_mask = df_with_years['max_event_year'] <= 2018
    test_mask = df_with_years['max_event_year'] == 2019
    
    train_df = df_with_years[train_mask].copy()
    test_df = df_with_years[test_mask].copy()
    
    print(f"\n[INFO] Temporal Split:")
    print(f"  Train (2016-2018): {len(train_df)} patients")
    if 'target' in train_df.columns:
        print(f"    Target: {train_df['target'].sum()}, Control: {(train_df['target'] == 0).sum()}")
    print(f"  Test (2019): {len(test_df)} patients")
    if 'target' in test_df.columns:
        print(f"    Target: {test_df['target'].sum()}, Control: {(test_df['target'] == 0).sum()}")
    
    if len(train_df) == 0:
        raise ValueError("No patients in training set (2016-2018)")
    if len(test_df) == 0:
        print("[WARNING] No patients in test set (2019) - only training data will be saved")
    
    # Drop year columns from feature tables (keep only features)
    year_cols = ['max_event_year', 'min_event_year']
    train_features = train_df.drop(columns=year_cols, errors='ignore')
    test_features = test_df.drop(columns=year_cols, errors='ignore')
    
    # Ensure target column exists (should already be in feature table from build_final_cohort_model_features.py)
    if 'target' not in train_features.columns:
        print("[WARNING] Target column missing in train features, adding default target=1")
        train_features['target'] = 1
    if 'target' not in test_features.columns:
        print("[WARNING] Target column missing in test features, adding default target=1")
        test_features['target'] = 1
    
    # Create local input directories
    # Location: 6_final_model/outputs (for Step 8 FFA analysis)
    input_dir = (
        project_root
        / "6_final_model"
        / "outputs"
        / cohort_name
        / age_band_fname
        / "inputs"
    )
    input_dir.mkdir(parents=True, exist_ok=True)
    
    train_dir = input_dir / "model_train"
    test_dir = input_dir / "model_test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    # Save locally as Parquet (more efficient than CSV)
    train_path = train_dir / "final_features.parquet"
    test_path = test_dir / "final_features.parquet"
    
    print(f"\n[INFO] Saving train dataset to: {train_path}")
    train_features.to_parquet(train_path, index=False, engine='pyarrow')
    print(f"[INFO] Train dataset: {len(train_features)} rows, {len(train_features.columns)} columns")
    
    if len(test_features) > 0:
        print(f"[INFO] Saving test dataset to: {test_path}")
        test_features.to_parquet(test_path, index=False, engine='pyarrow')
        print(f"[INFO] Test dataset: {len(test_features)} rows, {len(test_features.columns)} columns")
    
    # train_path and test_path are already set above
    
    # Upload to S3 (CRITICAL: Training data must be in S3 for FFA analysis)
    # S3 structure: inputs folder (replicating local structure)
    s3_base = f"s3://pgxdatalake/gold/final_model/{cohort_name}/{age_band}"
    s3_train_path = f"{s3_base}/inputs/model_train/final_features.parquet"
    s3_test_path = f"{s3_base}/inputs/model_test/final_features.parquet"
    
    # Also maintain backward compatibility with old location
    s3_train_path_legacy = f"{s3_base}/model_train/final_features.parquet"
    s3_test_path_legacy = f"{s3_base}/model_test/final_features.parquet"
    
    upload_success = False
    
    # Try AWS CLI first (faster for large files)
    aws_cli = shutil.which("aws")
    if aws_cli:
        print(f"\n[INFO] Uploading train dataset to S3 using AWS CLI: {s3_train_path}")
        try:
            subprocess.run(
                [aws_cli, "s3", "cp", str(train_path), s3_train_path],
                check=True,
                capture_output=True
            )
            print(f"[INFO] Train dataset uploaded successfully to S3")
            upload_success = True
        except subprocess.CalledProcessError as e:
            print(f"[WARNING] AWS CLI upload failed: {e.stderr.decode() if e.stderr else 'Unknown error'}")
            print(f"[INFO] Trying boto3 fallback...")
    
    # Fallback to boto3 if AWS CLI fails or is not available
    if not upload_success:
        try:
            import boto3
            s3_client = boto3.client('s3')
            bucket = 'pgxdatalake'
            s3_key_train = f"gold/final_model/{cohort_name}/{age_band}/inputs/model_train/final_features.parquet"
            
            print(f"\n[INFO] Uploading train dataset to S3 using boto3: s3://{bucket}/{s3_key_train}")
            s3_client.upload_file(str(train_path), bucket, s3_key_train)
            print(f"[INFO] Train dataset uploaded successfully to S3")
            upload_success = True
        except ImportError:
            print(f"[ERROR] boto3 not available. Cannot upload to S3.")
            print(f"[ERROR] Training data must be uploaded to S3 for FFA analysis to work!")
            raise RuntimeError("S3 upload failed and boto3 is not available. Install boto3: pip install boto3")
        except Exception as e:
            print(f"[ERROR] boto3 upload failed: {e}")
            raise RuntimeError(f"Failed to upload training data to S3: {e}")
    
    # Test dataset upload is required for SHAP/FFA (same as train).
    if len(test_features) > 0:
        upload_test_success = False
        if aws_cli:
            print(f"[INFO] Uploading test dataset to S3 using AWS CLI: {s3_test_path}")
            try:
                subprocess.run(
                    [aws_cli, "s3", "cp", str(test_path), s3_test_path],
                    check=True,
                    capture_output=True
                )
                print(f"[INFO] Test dataset uploaded successfully to S3")
                upload_test_success = True
            except subprocess.CalledProcessError as e:
                print(f"[WARNING] AWS CLI upload failed: {e.stderr.decode() if e.stderr else 'Unknown error'}")
        
        if not upload_test_success:
            try:
                import boto3
                s3_client = boto3.client('s3')
                bucket = 'pgxdatalake'
                s3_key_test = f"gold/final_model/{cohort_name}/{age_band}/inputs/model_test/final_features.parquet"
                
                print(f"[INFO] Uploading test dataset to S3 using boto3: s3://{bucket}/{s3_key_test}")
                s3_client.upload_file(str(test_path), bucket, s3_key_test)
                print(f"[INFO] Test dataset uploaded successfully to S3")
                upload_test_success = True
            except ImportError:
                print(f"[ERROR] boto3 not available. Test data must be in S3 for SHAP/FFA.")
                raise RuntimeError("S3 upload failed and boto3 is not available. Install boto3: pip install boto3")
            except Exception as e:
                print(f"[ERROR] Test dataset upload failed: {e}")
                raise RuntimeError(f"Failed to upload test data to S3 (required for SHAP/FFA): {e}")
    
    # Also save metadata files (in both locations)
    metadata_train = {
        'cohort': cohort_name,
        'age_band': age_band,
        'split_type': 'train',
        'years': '2016-2018',
        'n_patients': len(train_features),
        'n_features': len(train_features.columns) - 1,  # Exclude mi_person_key
        'target_distribution': train_features['target'].value_counts().to_dict() if 'target' in train_features.columns else {},
    }
    
    metadata_test = {
        'cohort': cohort_name,
        'age_band': age_band,
        'split_type': 'test',
        'years': '2019',
        'n_patients': len(test_features) if len(test_features) > 0 else 0,
        'n_features': len(test_features.columns) - 1 if len(test_features) > 0 else 0,
        'target_distribution': test_features['target'].value_counts().to_dict() if len(test_features) > 0 and 'target' in test_features.columns else {},
    }
    
    import json
    # Save metadata in primary location
    metadata_train_path = primary_train_dir / "metadata.json"
    metadata_test_path = primary_test_dir / "metadata.json"
    
    with open(metadata_train_path, 'w') as f:
        json.dump(metadata_train, f, indent=2)
    print(f"[INFO] Saved train metadata to {metadata_train_path}")
    
    if len(test_features) > 0:
        with open(metadata_test_path, 'w') as f:
            json.dump(metadata_test, f, indent=2)
        print(f"[INFO] Saved test metadata to {metadata_test_path}")
    
    print("\n[INFO] Train/test dataset preparation complete!")
    print(f"\nS3 Locations:")
    print(f"  Train: {s3_train_path}")
    if len(test_features) > 0:
        print(f"  Test: {s3_test_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare train/test datasets and upload to S3"
    )
    parser.add_argument(
        "--cohort-name",
        type=str,
        default="opioid_ed",
        help="Cohort name (e.g., opioid_ed)",
    )
    parser.add_argument(
        "--age-band",
        type=str,
        default="0-12",
        help="Age band (e.g., 0-12)",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root path (default: current directory)",
    )
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    prepare_train_test_s3(
        project_root=project_root,
        cohort_name=args.cohort_name,
        age_band=args.age_band,
    )


if __name__ == "__main__":
    main()

