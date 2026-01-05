#!/usr/bin/env python3
"""
Clear Step 5 (PGx analysis) outputs from S3 and local checkpoints to force regeneration.

This script:
1. Deletes S3 outputs for PGx features
2. Deletes S3 checkpoints
3. Optionally clears local files and time tracking
"""

import argparse
import sys
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

PROJECT_ROOT = Path(__file__).parent.parent
S3_BUCKET = "pgxdatalake"
s3_client = boto3.client("s3")


def delete_s3_object(bucket: str, key: str) -> bool:
    """Delete an S3 object."""
    try:
        s3_client.delete_object(Bucket=bucket, Key=key)
        print(f"✓ Deleted: s3://{bucket}/{key}")
        return True
    except ClientError as e:
        print(f"✗ Failed to delete s3://{bucket}/{key}: {e}")
        return False


def list_and_delete_s3_prefix(bucket: str, prefix: str, dry_run: bool = False) -> int:
    """List and delete all objects with a given prefix."""
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        count = 0
        
        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    if dry_run:
                        print(f"[DRY RUN] Would delete: s3://{bucket}/{key}")
                    else:
                        if delete_s3_object(bucket, key):
                            count += 1
        return count
    except Exception as e:
        print(f"Error listing/deleting {prefix}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Clear Step 5 (PGx analysis) outputs to force regeneration"
    )
    parser.add_argument("--cohort", required=True, help="Cohort name (e.g., opioid_ed)")
    parser.add_argument("--age-band", required=True, help="Age band (e.g., 13-24)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--clear-local",
        action="store_true",
        help="Also clear local files and time tracking",
    )
    parser.add_argument(
        "--clear-prerequisites",
        action="store_true",
        help="Also clear drug-gene mappings and allele frequencies (forces regeneration)",
    )

    args = parser.parse_args()
    cohort_name = args.cohort
    age_band = args.age_band
    age_band_fname = age_band.replace("-", "_")

    print("=" * 70)
    print(f"Clearing Step 5 outputs for {cohort_name} / {age_band}")
    print("=" * 70)
    if args.dry_run:
        print("[DRY RUN MODE - No files will be deleted]")
    print()

    deleted_count = 0

    # 1. Clear S3 PGx feature outputs (primary location)
    print("1. Clearing S3 PGx feature outputs (gold/pgx_features/)...")
    s3_prefixes = [
        f"gold/pgx_features/{cohort_name}/{age_band}/pgx_added_features_{cohort_name}_{age_band_fname}.csv",
        f"gold/pgx_features/{cohort_name}/{age_band}/pgx_features_{cohort_name}_{age_band_fname}.csv",
        f"gold/pgx_features/{cohort_name}/{age_band}/{cohort_name}_drug_gene_mappings.csv",
        f"gold/pgx_features/{cohort_name}/{age_band}/{cohort_name}_allele_frequencies.csv",
    ]
    
    for prefix in s3_prefixes:
        if args.dry_run:
            try:
                s3_client.head_object(Bucket=S3_BUCKET, Key=prefix)
                print(f"[DRY RUN] Would delete: s3://{S3_BUCKET}/{prefix}")
            except ClientError:
                print(f"[DRY RUN] Not found: s3://{S3_BUCKET}/{prefix}")
        else:
            if delete_s3_object(S3_BUCKET, prefix):
                deleted_count += 1

    # 2. Clear S3 legacy location
    print("\n2. Clearing S3 legacy PGx outputs (gold/feature_engineering/7_pgx/)...")
    legacy_prefix = f"gold/feature_engineering/7_pgx/{cohort_name}/{age_band}/"
    count = list_and_delete_s3_prefix(S3_BUCKET, legacy_prefix, args.dry_run)
    deleted_count += count

    # 3. Clear S3 checkpoints
    print("\n3. Clearing S3 checkpoints...")
    checkpoint_prefix = f"pipeline_checkpoints/5_pgx_analysis/{cohort_name}/{age_band}/"
    count = list_and_delete_s3_prefix(S3_BUCKET, checkpoint_prefix, args.dry_run)
    deleted_count += count

    # 4. Clear prerequisites if requested
    if args.clear_prerequisites:
        print("\n4. Clearing prerequisite files (drug-gene mappings and allele frequencies)...")
        
        # Clear global cache
        global_prefixes = [
            "gold/pgx_features/global/pgx_drug_gene_mappings_global.csv",
            "gold/pgx_features/global/pgx_allele_frequencies_global.csv",
        ]
        
        # Clear cohort-level
        cohort_prefixes = [
            f"gold/pgx_features/{cohort_name}/{cohort_name}_drug_gene_mappings.csv",
            f"gold/pgx_features/{cohort_name}/{cohort_name}_allele_frequencies.csv",
        ]
        
        for prefix in global_prefixes + cohort_prefixes:
            if args.dry_run:
                try:
                    s3_client.head_object(Bucket=S3_BUCKET, Key=prefix)
                    print(f"[DRY RUN] Would delete: s3://{S3_BUCKET}/{prefix}")
                except ClientError:
                    print(f"[DRY RUN] Not found: s3://{S3_BUCKET}/{prefix}")
            else:
                if delete_s3_object(S3_BUCKET, prefix):
                    deleted_count += 1

    # 5. Clear local files if requested
    if args.clear_local:
        print("\n5. Clearing local files...")
        
        # Local PGx outputs
        local_paths = [
            PROJECT_ROOT / "5_feature_engineering" / "feature_engineering_outputs" / "7_pgx" / cohort_name / age_band,
            PROJECT_ROOT / "5_pgx_analysis" / "outputs" / cohort_name / age_band_fname,
        ]
        
        if args.clear_prerequisites:
            local_paths.extend([
                PROJECT_ROOT / "5_pgx_analysis" / "outputs" / cohort_name / f"{cohort_name}_drug_gene_mappings.csv",
                PROJECT_ROOT / "5_pgx_analysis" / "outputs" / cohort_name / f"{cohort_name}_allele_frequencies.csv",
                PROJECT_ROOT / "5_pgx_analysis" / "outputs" / "global" / "pgx_drug_gene_mappings_global.csv",
                PROJECT_ROOT / "5_pgx_analysis" / "outputs" / "global" / "pgx_allele_frequencies_global.csv",
            ])
        
        for path in local_paths:
            if path.exists():
                if args.dry_run:
                    print(f"[DRY RUN] Would delete: {path}")
                else:
                    if path.is_file():
                        path.unlink()
                        print(f"✓ Deleted local file: {path}")
                    elif path.is_dir():
                        import shutil
                        shutil.rmtree(path)
                        print(f"✓ Deleted local directory: {path}")
                    deleted_count += 1
            else:
                if args.dry_run:
                    print(f"[DRY RUN] Not found: {path}")

        # Clear time tracking for Step 5
        print("\n6. Clearing time tracking for Step 5...")
        time_log_file = PROJECT_ROOT / "logs" / "time_tracking" / f"{cohort_name}_{age_band_fname}.json"
        if time_log_file.exists():
            if args.dry_run:
                print(f"[DRY RUN] Would clear Step 5 completion flag in: {time_log_file}")
            else:
                import json
                try:
                    with open(time_log_file, "r") as f:
                        data = json.load(f)
                    if "step_times" in data and "5" in data["step_times"]:
                        data["step_times"]["5"]["completed"] = False
                        with open(time_log_file, "w") as f:
                            json.dump(data, f, indent=2)
                        print(f"✓ Cleared Step 5 completion flag in: {time_log_file}")
                except Exception as e:
                    print(f"✗ Failed to clear time tracking: {e}")

    print("\n" + "=" * 70)
    if args.dry_run:
        print("[DRY RUN] No files were actually deleted")
    else:
        print(f"✓ Deleted {deleted_count} S3 objects/files")
        print("\nStep 5 will now regenerate on next run.")
    print("=" * 70)


if __name__ == "__main__":
    main()

