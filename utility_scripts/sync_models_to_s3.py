#!/usr/bin/env python3
"""
Sync best models from EC2 local storage to S3.

This script uploads model files from 6_final_model/outputs/ to S3,
matching the expected S3 paths used by FFA analysis and other workflows.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Cohorts and age bands
COHORTS: Dict[str, List[str]] = {
    'opioid_ed': ['13-24', '25-44', '45-54', '55-64'],
    'non_opioid_ed': ['65-74', '75-84', '85-94']
}

# S3 configuration
S3_BUCKET = "pgxdatalake"
S3_PROFILE = "mushin"  # Default AWS profile


def check_s3_file_exists(s3_path: str, profile: str = S3_PROFILE) -> bool:
    """Check if a file exists in S3."""
    try:
        result = subprocess.run(
            ['aws', 's3', 'ls', s3_path, '--profile', profile],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def upload_file_to_s3(local_path: Path, s3_path: str, profile: str = S3_PROFILE) -> bool:
    """Upload a file to S3."""
    try:
        result = subprocess.run(
            ['aws', 's3', 'cp', str(local_path), s3_path, '--profile', profile],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"  [ERROR] Failed to upload: {e}")
        return False


def sync_models_to_s3(
    project_root: Path,
    model_base_dir: Path,
    dry_run: bool = False,
    profile: str = S3_PROFILE
) -> Tuple[int, int, int, int]:
    """
    Sync model files from local EC2 storage to S3.
    
    Returns:
        Tuple of (total_files, uploaded, skipped, errors)
    """
    total_files = 0
    uploaded = 0
    skipped = 0
    errors = 0
    
    print("=" * 80)
    print("Sync Models from EC2 to S3")
    print("=" * 80)
    print()
    print(f"Project Root: {project_root}")
    print(f"Model Base Dir: {model_base_dir}")
    print(f"S3 Bucket: {S3_BUCKET}")
    print(f"AWS Profile: {profile}")
    if dry_run:
        print(f"[DRY RUN] No files will be uploaded")
    print()
    
    # Check if model base directory exists
    if not model_base_dir.exists():
        print(f"[ERROR] Model base directory not found: {model_base_dir}")
        return (0, 0, 0, 1)
    
    # Process each cohort
    for cohort, age_bands in COHORTS.items():
        print()
        print("=" * 80)
        print(f"Processing Cohort: {cohort}")
        print("=" * 80)
        
        for age_band in age_bands:
            age_band_fname = age_band.replace('-', '_')
            
            print()
            print(f"Age Band: {age_band} ({age_band_fname})")
            
            # Local directory
            local_dir = model_base_dir / cohort / age_band_fname / "final_model_json"
            
            if not local_dir.exists():
                print(f"  [SKIP] Directory not found: {local_dir}")
                continue
            
            # Files to sync
            files_to_sync = [
                (
                    local_dir / f"{cohort}_{age_band_fname}_best_xgboost_model.json",
                    f"s3://{S3_BUCKET}/gold/final_model/{cohort}/{age_band}/{cohort}_{age_band_fname}_best_xgboost_model.json",
                    "Best XGBoost JSON"
                ),
                (
                    local_dir / f"{cohort}_{age_band_fname}_best_catboost_model.json",
                    f"s3://{S3_BUCKET}/gold/final_model/{cohort}/{age_band}/{cohort}_{age_band_fname}_best_catboost_model.json",
                    "Best CatBoost JSON"
                ),
                (
                    local_dir / f"{cohort}_{age_band_fname}_best_catboost_model.cbm",
                    f"s3://{S3_BUCKET}/gold/final_model/{cohort}/{age_band}/{cohort}_{age_band_fname}_best_catboost_model.cbm",
                    "Best CatBoost Binary"
                ),
            ]
            
            # Also sync model selection metadata
            metadata_local = model_base_dir / cohort / age_band_fname / f"{cohort}_{age_band_fname}_model_selection_metadata.json"
            if metadata_local.exists():
                files_to_sync.append((
                    metadata_local,
                    f"s3://{S3_BUCKET}/gold/final_model/{cohort}/{age_band}/{cohort}_{age_band_fname}_model_selection_metadata.json",
                    "Model Selection Metadata"
                ))
            
            # Sync each file
            for local_file, s3_path, file_type in files_to_sync:
                total_files += 1
                
                if not local_file.exists():
                    print(f"  [SKIP] File not found: {local_file.name}")
                    skipped += 1
                    continue
                
                # Check if already exists in S3
                if check_s3_file_exists(s3_path, profile):
                    print(f"  [SKIP] Already exists in S3: {s3_path}")
                    skipped += 1
                    continue
                
                # Upload file
                print(f"  [UPLOAD] {file_type}")
                print(f"    Local:  {local_file}")
                print(f"    S3:     {s3_path}")
                
                if dry_run:
                    print(f"    [DRY RUN] Would upload")
                    uploaded += 1
                else:
                    if upload_file_to_s3(local_file, s3_path, profile):
                        print(f"    [OK] Uploaded successfully")
                        uploaded += 1
                    else:
                        print(f"    [ERROR] Failed to upload")
                        errors += 1
    
    return (total_files, uploaded, skipped, errors)


def main():
    parser = argparse.ArgumentParser(
        description="Sync best models from EC2 local storage to S3"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory (default: auto-detect)"
    )
    parser.add_argument(
        "--model-base-dir",
        type=str,
        default=None,
        help="Model base directory (default: {project_root}/6_final_model/outputs)"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=S3_PROFILE,
        help=f"AWS profile to use (default: {S3_PROFILE})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - show what would be uploaded without actually uploading"
    )
    
    args = parser.parse_args()
    
    # Determine project root
    if args.project_root:
        project_root = Path(args.project_root).resolve()
    else:
        # Try to auto-detect
        project_root = PROJECT_ROOT
        if not project_root.exists():
            # Try common EC2 locations
            for path_str in ["/home/pgx3874/pgx-analysis", "/mnt/nvme/pgx-analysis", Path.home() / "pgx-analysis"]:
                test_path = Path(path_str)
                if test_path.exists():
                    project_root = test_path
                    break
            else:
                print("[ERROR] Could not find project root. Please specify --project-root")
                sys.exit(1)
    
    # Determine model base directory
    if args.model_base_dir:
        model_base_dir = Path(args.model_base_dir).resolve()
    else:
        model_base_dir = project_root / "6_final_model" / "outputs"
    
    # Run sync
    total_files, uploaded, skipped, errors = sync_models_to_s3(
        project_root=project_root,
        model_base_dir=model_base_dir,
        dry_run=args.dry_run,
        profile=args.profile
    )
    
    # Summary
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total files processed: {total_files}")
    print(f"Uploaded: {uploaded}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Errors: {errors}")
    print()
    
    if errors == 0:
        print("✓ All files synced successfully!")
        sys.exit(0)
    else:
        print("✗ Some errors occurred. Please check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
