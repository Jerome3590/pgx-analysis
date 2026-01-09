#!/usr/bin/env python3
"""
Clear Step 8 (FFA Analysis) outputs to allow workflow restart.

Removes:
- Local output files
- S3 checkpoint
- S3 output files
"""

import sys
import io
from pathlib import Path
import argparse

# Set UTF-8 encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    print("[WARNING] boto3 not available. S3 operations will be skipped.")

try:
    from py_helpers.checkpoint_utils import get_s3_client
    CHECKPOINT_UTILS_AVAILABLE = True
except ImportError:
    CHECKPOINT_UTILS_AVAILABLE = False


def delete_local_files(cohort: str, age_band: str, model_type: str = "xgboost", dry_run: bool = False):
    """Delete local Step 8 output files."""
    age_band_fname = age_band.replace("-", "_")
    output_dir = PROJECT_ROOT / "8_ffa_analysis" / "outputs" / cohort / age_band_fname / model_type
    
    files_to_delete = [
        "axp_explanations.parquet",
        "feature_importance_axp.parquet",
        "causal_importance.parquet",
        "interaction_analysis.parquet",
        "analysis_summary.json",
    ]
    
    deleted = 0
    skipped = 0
    
    print("LOCAL FILES")
    print("-" * 80)
    print(f"Directory: {output_dir}")
    print()
    
    if not output_dir.exists():
        print(f"[SKIP] Directory does not exist: {output_dir}")
        return deleted, skipped
    
    for file_name in files_to_delete:
        file_path = output_dir / file_name
        if file_path.exists():
            if dry_run:
                print(f"[DRY RUN] Would delete: {file_path}")
            else:
                try:
                    file_path.unlink()
                    print(f"✓ Deleted: {file_name}")
                    deleted += 1
                except Exception as e:
                    print(f"✗ Failed to delete {file_name}: {e}")
        else:
            print(f"[SKIP] Not found: {file_name}")
            skipped += 1
    
    return deleted, skipped


def delete_s3_checkpoint(cohort: str, age_band: str, dry_run: bool = False):
    """Delete S3 checkpoint."""
    if not BOTO3_AVAILABLE or not CHECKPOINT_UTILS_AVAILABLE:
        print("[SKIP] boto3 or checkpoint_utils not available")
        return 0
    
    age_band_fname = age_band.replace("-", "_")
    checkpoint_key = f"pipeline_checkpoints/8_ffa_analysis/{cohort}/{age_band_fname}/checkpoint.json"
    bucket = "pgx-repository"
    
    print("S3 CHECKPOINT")
    print("-" * 80)
    print(f"s3://{bucket}/{checkpoint_key}")
    print()
    
    try:
        s3_client = get_s3_client()
        
        # Check if exists
        try:
            s3_client.head_object(Bucket=bucket, Key=checkpoint_key)
            exists = True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                exists = False
            else:
                raise
        
        if exists:
            if dry_run:
                print(f"[DRY RUN] Would delete: s3://{bucket}/{checkpoint_key}")
                return 0
            else:
                s3_client.delete_object(Bucket=bucket, Key=checkpoint_key)
                print(f"✓ Deleted: s3://{bucket}/{checkpoint_key}")
                return 1
        else:
            print(f"[SKIP] Checkpoint does not exist")
            return 0
    except Exception as e:
        print(f"✗ Error deleting checkpoint: {e}")
        return 0


def delete_s3_outputs(cohort: str, age_band: str, model_type: str = "xgboost", dry_run: bool = False):
    """Delete S3 output files."""
    if not BOTO3_AVAILABLE or not CHECKPOINT_UTILS_AVAILABLE:
        print("[SKIP] boto3 or checkpoint_utils not available")
        return 0, 0
    
    bucket = "pgxdatalake"
    s3_base = f"gold/ffa_analysis/{cohort}/{age_band}/{model_type}"
    
    files_to_delete = [
        "axp_explanations.parquet",
        "feature_importance_axp.parquet",
        "causal_importance.parquet",
        "interaction_analysis.parquet",
    ]
    
    deleted = 0
    skipped = 0
    
    print("S3 OUTPUT FILES")
    print("-" * 80)
    print(f"Base path: s3://{bucket}/{s3_base}")
    print()
    
    try:
        s3_client = get_s3_client()
        
        for file_name in files_to_delete:
            s3_key = f"{s3_base}/{file_name}"
            s3_path = f"s3://{bucket}/{s3_key}"
            
            try:
                # Check if exists
                s3_client.head_object(Bucket=bucket, Key=s3_key)
                exists = True
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    exists = False
                else:
                    raise
            
            if exists:
                if dry_run:
                    print(f"[DRY RUN] Would delete: {s3_path}")
                else:
                    s3_client.delete_object(Bucket=bucket, Key=s3_key)
                    print(f"✓ Deleted: {file_name}")
                    deleted += 1
            else:
                print(f"[SKIP] Not found: {file_name}")
                skipped += 1
    except Exception as e:
        print(f"✗ Error deleting S3 outputs: {e}")
    
    return deleted, skipped


def main():
    parser = argparse.ArgumentParser(description="Clear Step 8 (FFA Analysis) outputs")
    parser.add_argument("--cohort-name", type=str, default="opioid_ed", help="Cohort name")
    parser.add_argument("--age-band", type=str, default="13-24", help="Age band")
    parser.add_argument("--model-type", type=str, default="xgboost", help="Model type")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't delete")
    parser.add_argument("--all-cohorts", action="store_true", help="Clear all cohorts")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Clear Step 8 (FFA Analysis) Outputs")
    print("=" * 80)
    print(f"Cohort: {args.cohort_name}")
    print(f"Age Band: {args.age_band}")
    print(f"Model Type: {args.model_type}")
    print(f"Mode: {'DRY RUN (preview only)' if args.dry_run else 'DELETE'}")
    print("=" * 80)
    print()
    
    if args.all_cohorts:
        cohorts = [
            ("opioid_ed", "13-24"),
            ("opioid_ed", "25-44"),
            ("opioid_ed", "45-54"),
            ("opioid_ed", "55-64"),
            ("non_opioid_ed", "65-74"),
            ("non_opioid_ed", "75-84"),
            ("non_opioid_ed", "85-94"),
        ]
        
        total_deleted = 0
        total_skipped = 0
        
        for cohort, age_band in cohorts:
            print(f"\n{'=' * 80}")
            print(f"Processing: {cohort}/{age_band}")
            print(f"{'=' * 80}\n")
            
            local_del, local_skip = delete_local_files(cohort, age_band, args.model_type, args.dry_run)
            print()
            
            s3_checkpoint_del = delete_s3_checkpoint(cohort, age_band, args.dry_run)
            print()
            
            s3_outputs_del, s3_outputs_skip = delete_s3_outputs(cohort, age_band, args.model_type, args.dry_run)
            print()
            
            total_deleted += local_del + s3_checkpoint_del + s3_outputs_del
            total_skipped += local_skip + s3_outputs_skip
        
        print("=" * 80)
        print("Summary (All Cohorts)")
        print("=" * 80)
        if args.dry_run:
            print("[DRY RUN] No files were actually deleted")
        else:
            print(f"Deleted: {total_deleted} files")
            print(f"Skipped (not found): {total_skipped} files")
    else:
        local_del, local_skip = delete_local_files(args.cohort_name, args.age_band, args.model_type, args.dry_run)
        print()
        
        s3_checkpoint_del = delete_s3_checkpoint(args.cohort_name, args.age_band, args.dry_run)
        print()
        
        s3_outputs_del, s3_outputs_skip = delete_s3_outputs(args.cohort_name, args.age_band, args.model_type, args.dry_run)
        print()
        
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        if args.dry_run:
            print("[DRY RUN] No files were actually deleted")
        else:
            print(f"Deleted: {local_del + s3_checkpoint_del + s3_outputs_del} files")
            print(f"Skipped (not found): {local_skip + s3_outputs_skip} files")
            print()
            print("Step 8 outputs cleared. Workflow will restart at Step 8.")
        print("=" * 80)


if __name__ == "__main__":
    main()
