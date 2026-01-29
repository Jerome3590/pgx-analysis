#!/usr/bin/env python3
"""
Clear S3 checkpoints for pipeline steps.
Works on both Windows and Linux.

This script deletes checkpoint metadata from:
s3://pgx-repository/pipeline_checkpoints/{step_name}/{cohort}/{age_band}/checkpoint.json
"""

import sys
import json
from pathlib import Path
from typing import List, Optional

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("ERROR: boto3 not installed. Install with: pip install boto3")
    sys.exit(1)

# S3 client
s3_client = boto3.client('s3')
CHECKPOINT_BUCKET = "pgx-repository"
CHECKPOINT_PREFIX = "pipeline_checkpoints/"

# Define all cohorts and age bands
COHORTS = {
    "opioid_ed": ["13-24", "25-44", "45-54", "55-64"],
    "non_opioid_ed": ["65-74", "75-84", "85-94"],
}

# Common step names
COMMON_STEPS = [
    "3_feature_importance",
    "3b_feature_importance_eda",
    "4a_model_data",
    "4b_event_filter",
    "5_pgx_analysis",
    "6_final_model_selection",
    "7_shap_analysis",
    "8_ffa_analysis",
    "9_risk_dashboard",
]


def list_all_checkpoints(step_name: Optional[str] = None, cohort: Optional[str] = None, age_band: Optional[str] = None) -> List[str]:
    """List all checkpoint keys in S3."""
    checkpoint_keys = []
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        
        # Build prefix
        prefix = CHECKPOINT_PREFIX
        if step_name:
            prefix = f"{CHECKPOINT_PREFIX}{step_name}/"
        if cohort:
            prefix = f"{prefix}{cohort}/"
        if age_band:
            age_band_fname = age_band.replace("-", "_")
            prefix = f"{prefix}{age_band_fname}/"
        
        # List all objects with this prefix
        for page in paginator.paginate(Bucket=CHECKPOINT_BUCKET, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith('checkpoint.json'):
                        checkpoint_keys.append(obj['Key'])
    
    except ClientError as e:
        print(f"Error listing checkpoints: {e}")
        return []
    
    return checkpoint_keys


def delete_checkpoint(checkpoint_key: str) -> bool:
    """Delete a single checkpoint from S3."""
    try:
        s3_client.delete_object(Bucket=CHECKPOINT_BUCKET, Key=checkpoint_key)
        return True
    except ClientError as e:
        print(f"Error deleting {checkpoint_key}: {e}")
        return False


def clear_all_checkpoints(step_name: Optional[str] = None, cohort: Optional[str] = None, age_band: Optional[str] = None, dry_run: bool = False) -> int:
    """Clear all checkpoints matching the criteria."""
    print("=" * 80)
    print("Clear S3 Checkpoints")
    print("=" * 80)
    print(f"S3 Bucket: {CHECKPOINT_BUCKET}")
    print(f"Prefix: {CHECKPOINT_PREFIX}")
    
    if step_name:
        print(f"Step filter: {step_name}")
    if cohort:
        print(f"Cohort filter: {cohort}")
    if age_band:
        print(f"Age band filter: {age_band}")
    
    if dry_run:
        print("\n[DRY RUN] No checkpoints will be deleted")
    
    print()
    
    # List all matching checkpoints
    checkpoint_keys = list_all_checkpoints(step_name, cohort, age_band)
    
    if not checkpoint_keys:
        print("No checkpoints found matching the criteria.")
        return 0
    
    print(f"Found {len(checkpoint_keys)} checkpoint(s):")
    for key in checkpoint_keys:
        print(f"  - s3://{CHECKPOINT_BUCKET}/{key}")
    
    if dry_run:
        print(f"\n[DRY RUN] Would delete {len(checkpoint_keys)} checkpoint(s)")
        return len(checkpoint_keys)
    
    # Confirm deletion
    print(f"\n⚠️  About to delete {len(checkpoint_keys)} checkpoint(s) from S3")
    response = input("Continue? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("Cancelled.")
        return 0
    
    # Delete checkpoints
    print("\nDeleting checkpoints...")
    deleted_count = 0
    failed_count = 0
    
    for key in checkpoint_keys:
        if delete_checkpoint(key):
            print(f"  [OK] Deleted: {key}")
            deleted_count += 1
        else:
            print(f"  [FAILED] Could not delete: {key}")
            failed_count += 1
    
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Deleted: {deleted_count}")
    if failed_count > 0:
        print(f"Failed: {failed_count}")
    
    return deleted_count


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clear S3 checkpoints for pipeline steps"
    )
    parser.add_argument(
        "--step",
        type=str,
        help="Clear checkpoints for specific step (e.g., '3b_feature_importance_eda')"
    )
    parser.add_argument(
        "--cohort",
        type=str,
        choices=list(COHORTS.keys()),
        help="Clear checkpoints for specific cohort"
    )
    parser.add_argument(
        "--age-band",
        type=str,
        help="Clear checkpoints for specific age band (requires --cohort)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Clear ALL checkpoints (use with caution!)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List checkpoints that would be deleted without actually deleting them"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.age_band and not args.cohort:
        print("ERROR: --age-band requires --cohort")
        sys.exit(1)
    
    if not args.all and not args.step and not args.cohort:
        print("ERROR: Must specify --step, --cohort, or --all")
        print("\nExamples:")
        print("  # Clear all checkpoints for Step 3b")
        print("  python clear_s3_checkpoints.py --step 3b_feature_importance_eda")
        print("\n  # Clear all checkpoints for a cohort")
        print("  python clear_s3_checkpoints.py --cohort opioid_ed")
        print("\n  # Clear checkpoints for specific cohort/age_band")
        print("  python clear_s3_checkpoints.py --cohort opioid_ed --age-band 13-24")
        print("\n  # Dry run to see what would be deleted")
        print("  python clear_s3_checkpoints.py --all --dry-run")
        sys.exit(1)
    
    # Clear checkpoints
    deleted = clear_all_checkpoints(
        step_name=args.step,
        cohort=args.cohort,
        age_band=args.age_band,
        dry_run=args.dry_run
    )
    
    if deleted > 0:
        print(f"\n✓ Cleared {deleted} checkpoint(s). Steps will rerun on next workflow execution.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
