#!/usr/bin/env python3
"""
Sync local gold_cohorts data to S3 gold/model_training_data folder.

This script syncs the local gold_cohorts directory structure to:
s3://pgxdatalake/gold/model_training_data/
"""
import subprocess
import sys
from pathlib import Path
import shutil

# S3 destination
S3_BUCKET = 'pgxdatalake'
S3_DEST = f's3://{S3_BUCKET}/gold/model_training_data/'

def check_aws_cli():
    """Check if AWS CLI is available."""
    aws_cli = shutil.which("aws")
    if not aws_cli:
        print("[ERROR] AWS CLI not found. Please install AWS CLI first.")
        print("  Download from: https://aws.amazon.com/cli/")
        return None
    return aws_cli

def check_local_path(source_path):
    """Check if local source path exists."""
    if not source_path.exists():
        print(f"[ERROR] Local source path does not exist: {source_path}")
        return False
    
    # Count parquet files
    parquet_files = list(source_path.rglob('cohort.parquet'))
    if not parquet_files:
        print(f"[ERROR] No cohort.parquet files found in {source_path}")
        return False
    
    print(f"[INFO] Found {len(parquet_files)} cohort.parquet files in {source_path}")
    return True

def sync_to_s3(aws_cli, source_path, dry_run=False):
    """Sync local gold_cohorts to S3."""
    print("\n" + "=" * 80)
    print("Syncing Gold Cohorts to S3")
    print("=" * 80)
    print(f"Source: {source_path}")
    print(f"Destination: {S3_DEST}")
    
    if dry_run:
        print("\n[DRY RUN] Would sync the following:")
        cmd = [aws_cli, "s3", "sync", str(source_path), S3_DEST, "--dryrun"]
    else:
        print("\n[INFO] Starting sync...")
        cmd = [aws_cli, "s3", "sync", str(source_path), S3_DEST, "--no-progress"]
    
    print(f"\nCommand: {' '.join(cmd)}")
    print("-" * 80)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        
        if result.returncode == 0:
            if dry_run:
                print(result.stdout)
                print("\n[DRY RUN] Complete. Run without --dry-run to perform actual sync.")
            else:
                print("[SUCCESS] Sync completed successfully!")
                if result.stdout:
                    print(result.stdout)
        else:
            print(f"[ERROR] Sync failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Exception during sync: {e}")
        return False
    
    return True

def main():
    """Main function."""
    import argparse
    
    # Default source path
    default_source = Path(r'C:\Projects\pgx-datasets\data\gold_cohorts')
    
    parser = argparse.ArgumentParser(
        description='Sync local gold_cohorts data to S3 gold/model_training_data folder'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be synced without actually syncing'
    )
    parser.add_argument(
        '--source',
        type=str,
        default=str(default_source),
        help=f'Local source path (default: {default_source})'
    )
    args = parser.parse_args()
    
    # Get source path
    source_path = Path(args.source)
    
    print("=" * 80)
    print("Gold Cohorts S3 Sync Tool")
    print("=" * 80)
    
    # Check prerequisites
    aws_cli = check_aws_cli()
    if not aws_cli:
        sys.exit(1)
    
    if not check_local_path(source_path):
        sys.exit(1)
    
    # Perform sync
    success = sync_to_s3(aws_cli, source_path, dry_run=args.dry_run)
    
    if success:
        print("\n" + "=" * 80)
        print("Sync Summary")
        print("=" * 80)
        print(f"Source: {source_path}")
        print(f"Destination: {S3_DEST}")
        print("\n[SUCCESS] All files synced successfully!")
        sys.exit(0)
    else:
        print("\n[ERROR] Sync failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == '__main__':
    main()
