#!/usr/bin/env python3
"""
Migrate S3 paths to match updated folder structure.

This script:
1. Moves checkpoints from old step names to new step names
2. Ensures aggregated feature importances are accessible
3. Updates any S3 paths that reference old folder names

Old -> New mappings:
- pipeline_checkpoints/5c_pgx_analysis/ -> pipeline_checkpoints/5_pgx_analysis/
- pipeline_checkpoints/6b_final_model_selection/ -> pipeline_checkpoints/6_final_model/
- gold/pgx_features/ (no change needed, but verify structure)
"""

import sys
import boto3
from pathlib import Path
from botocore.exceptions import ClientError
from typing import List, Dict
import json

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# S3 clients
s3_client = boto3.client('s3')
CHECKPOINT_BUCKET = "pgx-repository"
OUTPUT_BUCKET = "pgxdatalake"

# Cohorts and age bands
COHORTS = {
    "opioid_ed": ["13-24", "25-44", "45-54", "55-64"],
    "non_opioid_ed": ["65-74", "75-84", "85-94"],
}

# Migration mappings: old_step_name -> new_step_name
CHECKPOINT_MIGRATIONS = {
    "5c_pgx_analysis": "5_pgx_analysis",
    "6b_final_model_selection": "6_final_model",
}

# S3 path migrations (old_prefix -> new_prefix)
OUTPUT_MIGRATIONS = {
    # PGx features: standardize on gold/pgx_features/ (some code uses gold/feature_engineering/7_pgx/)
    # We'll keep both for backward compatibility but prefer gold/pgx_features/
}


def list_s3_objects(bucket: str, prefix: str) -> List[Dict]:
    """List all objects with given prefix."""
    objects = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            objects.extend(page['Contents'])
    return objects


def copy_s3_object(source_bucket: str, source_key: str, dest_bucket: str, dest_key: str) -> bool:
    """Copy an S3 object from source to destination."""
    try:
        copy_source = {'Bucket': source_bucket, 'Key': source_key}
        s3_client.copy_object(
            CopySource=copy_source,
            Bucket=dest_bucket,
            Key=dest_key
        )
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to copy {source_key} -> {dest_key}: {e}")
        return False


def migrate_checkpoints(old_step: str, new_step: str, dry_run: bool = True) -> Dict[str, int]:
    """Migrate checkpoints from old step name to new step name."""
    stats = {"copied": 0, "skipped": 0, "errors": 0}
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Migrating checkpoints: {old_step} -> {new_step}")
    print("-" * 80)
    
    for cohort, age_bands in COHORTS.items():
        for age_band in age_bands:
            age_band_fname = age_band.replace("-", "_")
            
            old_key = f"pipeline_checkpoints/{old_step}/{cohort}/{age_band_fname}/checkpoint.json"
            new_key = f"pipeline_checkpoints/{new_step}/{cohort}/{age_band_fname}/checkpoint.json"
            
            # Check if old checkpoint exists
            try:
                s3_client.head_object(Bucket=CHECKPOINT_BUCKET, Key=old_key)
                
                # Check if new checkpoint already exists
                try:
                    s3_client.head_object(Bucket=CHECKPOINT_BUCKET, Key=new_key)
                    print(f"  ⊙ {cohort}/{age_band}: New checkpoint already exists, skipping")
                    stats["skipped"] += 1
                    continue
                except ClientError:
                    pass  # New checkpoint doesn't exist, proceed with copy
                
                if dry_run:
                    print(f"  → Would copy: {old_key} -> {new_key}")
                    stats["copied"] += 1
                else:
                    if copy_s3_object(CHECKPOINT_BUCKET, old_key, CHECKPOINT_BUCKET, new_key):
                        print(f"  ✓ Copied: {cohort}/{age_band}")
                        stats["copied"] += 1
                    else:
                        stats["errors"] += 1
                        
            except ClientError:
                # Old checkpoint doesn't exist, skip
                pass
    
    return stats


def verify_aggregated_feature_importance_access() -> Dict[str, List[str]]:
    """Verify aggregated feature importance files are accessible in S3."""
    print("\nVerifying aggregated feature importance access...")
    print("-" * 80)
    
    results = {"found": [], "missing": []}
    
    for cohort, age_bands in COHORTS.items():
        for age_band in age_bands:
            age_band_fname = age_band.replace("-", "_")
            s3_key = (
                f"gold/feature_importance/{cohort}/{age_band}/"
                f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
            )
            
            try:
                s3_client.head_object(Bucket=OUTPUT_BUCKET, Key=s3_key)
                print(f"  ✓ {cohort}/{age_band}: Found")
                results["found"].append(f"{cohort}/{age_band}")
            except ClientError:
                print(f"  ✗ {cohort}/{age_band}: Missing")
                results["missing"].append(f"{cohort}/{age_band}")
    
    return results


def standardize_pgx_s3_paths(dry_run: bool = True) -> Dict[str, int]:
    """Ensure PGx features use consistent S3 paths."""
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Standardizing PGx S3 paths...")
    print("-" * 80)
    
    stats = {"copied": 0, "skipped": 0, "errors": 0}
    
    # Check for files in old location (gold/feature_engineering/7_pgx/)
    # and ensure they're also in new location (gold/pgx_features/)
    for cohort, age_bands in COHORTS.items():
        for age_band in age_bands:
            age_band_fname = age_band.replace("-", "_")
            
            # Check old location
            old_prefix = f"gold/feature_engineering/7_pgx/{cohort}/{age_band}/"
            old_objects = list_s3_objects(OUTPUT_BUCKET, old_prefix)
            
            # Check new location
            new_prefix = f"gold/pgx_features/{cohort}/{age_band}/"
            
            for obj in old_objects:
                old_key = obj['Key']
                filename = old_key.split('/')[-1]
                new_key = f"{new_prefix}{filename}"
                
                # Check if already exists in new location
                try:
                    s3_client.head_object(Bucket=OUTPUT_BUCKET, Key=new_key)
                    if dry_run:
                        print(f"  ⊙ {filename}: Already exists in new location")
                    stats["skipped"] += 1
                except ClientError:
                    # Copy to new location
                    if dry_run:
                        print(f"  → Would copy: {old_key} -> {new_key}")
                        stats["copied"] += 1
                    else:
                        if copy_s3_object(OUTPUT_BUCKET, old_key, OUTPUT_BUCKET, new_key):
                            print(f"  ✓ Copied: {filename}")
                            stats["copied"] += 1
                        else:
                            stats["errors"] += 1
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate S3 paths to match updated folder structure")
    parser.add_argument("--execute", action="store_true", help="Actually perform migrations (default is dry-run)")
    parser.add_argument("--checkpoints-only", action="store_true", help="Only migrate checkpoints")
    parser.add_argument("--verify-fi", action="store_true", help="Only verify aggregated feature importance access")
    
    args = parser.parse_args()
    dry_run = not args.execute
    
    print("=" * 80)
    print("S3 Path Migration Script")
    print("=" * 80)
    print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    print()
    
    if args.verify_fi:
        # Only verify feature importance access
        results = verify_aggregated_feature_importance_access()
        print(f"\nSummary: {len(results['found'])} found, {len(results['missing'])} missing")
        return
    
    # Migrate checkpoints
    if not args.checkpoints_only:
        print("\n" + "=" * 80)
        print("STEP 1: Migrating Checkpoints")
        print("=" * 80)
        
        total_stats = {"copied": 0, "skipped": 0, "errors": 0}
        for old_step, new_step in CHECKPOINT_MIGRATIONS.items():
            stats = migrate_checkpoints(old_step, new_step, dry_run)
            total_stats["copied"] += stats["copied"]
            total_stats["skipped"] += stats["skipped"]
            total_stats["errors"] += stats["errors"]
        
        print(f"\nCheckpoint Migration Summary:")
        print(f"  Copied: {total_stats['copied']}")
        print(f"  Skipped: {total_stats['skipped']}")
        print(f"  Errors: {total_stats['errors']}")
    
    # Verify aggregated feature importance
    print("\n" + "=" * 80)
    print("STEP 2: Verifying Aggregated Feature Importance Access")
    print("=" * 80)
    results = verify_aggregated_feature_importance_access()
    print(f"\nSummary: {len(results['found'])} found, {len(results['missing'])} missing")
    
    if results["missing"]:
        print("\n⚠ Missing aggregated feature importance files:")
        for combo in results["missing"]:
            print(f"  - {combo}")
        print("\nThese need to be generated by running Step 3 (Feature Importance)")
    
    # Standardize PGx paths
    if not args.checkpoints_only:
        print("\n" + "=" * 80)
        print("STEP 3: Standardizing PGx S3 Paths")
        print("=" * 80)
        pgx_stats = standardize_pgx_s3_paths(dry_run)
        print(f"\nPGx Path Standardization Summary:")
        print(f"  Copied: {pgx_stats['copied']}")
        print(f"  Skipped: {pgx_stats['skipped']}")
        print(f"  Errors: {pgx_stats['errors']}")
    
    print("\n" + "=" * 80)
    if dry_run:
        print("DRY RUN COMPLETE")
        print("Run with --execute to perform actual migrations")
    else:
        print("MIGRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

