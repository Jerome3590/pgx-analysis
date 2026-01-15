#!/usr/bin/env python3
"""
Sync missing model files from EC2 local storage to S3.

This script:
1. Checks which model files are expected in S3 (based on evaluation script requirements)
2. Checks which files exist locally on EC2
3. Identifies missing files
4. Syncs only the missing files to S3

Model files needed:
- XGBoost: xgboost.joblib, xgboost_model.ubj, or best_xgboost_model.json
- CatBoost: catboost_model.cbm, catboost.joblib, or best_catboost_model.json
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess
import os

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
# Auto-detect EC2 vs local
if os.path.exists('/sys/hypervisor/uuid') or os.path.exists('/sys/class/dmi/id'):
    S3_PROFILE = None  # EC2 - use IAM role
else:
    S3_PROFILE = os.environ.get('AWS_PROFILE', 'mushin')  # Local - use profile


def check_s3_file_exists(s3_path: str, profile: Optional[str] = None) -> bool:
    """Check if a file exists in S3."""
    try:
        cmd = ['aws', 's3', 'ls', s3_path]
        if profile:
            cmd.extend(['--profile', profile])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def upload_file_to_s3(local_path: Path, s3_path: str, profile: Optional[str] = None) -> bool:
    """Upload a file to S3."""
    try:
        cmd = ['aws', 's3', 'cp', str(local_path), s3_path]
        if profile:
            cmd.extend(['--profile', profile])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return True
        else:
            print(f"    [ERROR] AWS CLI error: {result.stderr}")
            return False
    except Exception as e:
        print(f"    [ERROR] Failed to upload: {e}")
        return False


def find_local_model_files(
    project_root: Path,
    cohort: str,
    age_band: str
) -> Dict[str, List[Path]]:
    """
    Find all local model files for a cohort/age_band.
    Checks both outputs/ and model_outputs/ directories (matching run_final_model.py).
    
    Returns:
        Dict with keys: 'xgboost_joblib', 'xgboost_ubj', 'xgboost_json',
                        'catboost_cbm', 'catboost_joblib', 'catboost_json',
                        'metadata'
    """
    age_band_fname = age_band.replace('-', '_')
    found_files = {
        'xgboost_joblib': [],
        'xgboost_ubj': [],
        'xgboost_json': [],
        'catboost_cbm': [],
        'catboost_joblib': [],
        'catboost_json': [],
        'metadata': []
    }
    
    # Check both outputs/ and model_outputs/ directories
    possible_bases = [
        project_root / "6_final_model" / "outputs",
        project_root / "6_final_model" / "model_outputs",
    ]
    
    for base_dir in possible_bases:
        if not base_dir.exists():
            continue
        
        # Check final_model_json directory (in outputs/)
        if base_dir.name == "outputs":
            final_model_json_dir = base_dir / cohort / age_band_fname / "final_model_json"
            if final_model_json_dir.exists():
                # XGBoost JSON
                xgb_json = final_model_json_dir / f"{cohort}_{age_band_fname}_best_xgboost_model.json"
                if xgb_json.exists() and xgb_json not in found_files['xgboost_json']:
                    found_files['xgboost_json'].append(xgb_json)
                
                # CatBoost JSON
                cb_json = final_model_json_dir / f"{cohort}_{age_band_fname}_best_catboost_model.json"
                if cb_json.exists() and cb_json not in found_files['catboost_json']:
                    found_files['catboost_json'].append(cb_json)
                
                # CatBoost CBM
                cb_cbm = final_model_json_dir / f"{cohort}_{age_band_fname}_best_catboost_model.cbm"
                if cb_cbm.exists() and cb_cbm not in found_files['catboost_cbm']:
                    found_files['catboost_cbm'].append(cb_cbm)
            
            # Check models directory (in outputs/)
            models_dir = base_dir / cohort / age_band_fname / "models"
            if models_dir.exists():
                # XGBoost joblib
                xgb_joblib = models_dir / "xgboost.joblib"
                if xgb_joblib.exists() and xgb_joblib not in found_files['xgboost_joblib']:
                    found_files['xgboost_joblib'].append(xgb_joblib)
                
                # XGBoost UBJ
                xgb_ubj = models_dir / "xgboost_model.ubj"
                if xgb_ubj.exists() and xgb_ubj not in found_files['xgboost_ubj']:
                    found_files['xgboost_ubj'].append(xgb_ubj)
                
                # CatBoost joblib
                cb_joblib = models_dir / "catboost.joblib"
                if cb_joblib.exists() and cb_joblib not in found_files['catboost_joblib']:
                    found_files['catboost_joblib'].append(cb_joblib)
                
                # CatBoost CBM
                cb_cbm2 = models_dir / "catboost_model.cbm"
                if cb_cbm2.exists() and cb_cbm2 not in found_files['catboost_cbm']:
                    found_files['catboost_cbm'].append(cb_cbm2)
            
            # Check for metadata (in outputs/)
            metadata_file = base_dir / cohort / age_band_fname / f"{cohort}_{age_band_fname}_model_selection_metadata.json"
            if metadata_file.exists() and metadata_file not in found_files['metadata']:
                found_files['metadata'].append(metadata_file)
        
        # Check model_outputs/ directory (mirror location)
        elif base_dir.name == "model_outputs":
            model_outputs_dir = base_dir / cohort / age_band_fname
            if model_outputs_dir.exists():
                # XGBoost JSON (mirror)
                xgb_json = model_outputs_dir / f"{cohort}_{age_band_fname}_best_xgboost_model.json"
                if xgb_json.exists() and xgb_json not in found_files['xgboost_json']:
                    found_files['xgboost_json'].append(xgb_json)
                
                # CatBoost JSON (mirror)
                cb_json = model_outputs_dir / f"{cohort}_{age_band_fname}_best_catboost_model.json"
                if cb_json.exists() and cb_json not in found_files['catboost_json']:
                    found_files['catboost_json'].append(cb_json)
                
                # CatBoost CBM (mirror)
                cb_cbm = model_outputs_dir / f"{cohort}_{age_band_fname}_best_catboost_model.cbm"
                if cb_cbm.exists() and cb_cbm not in found_files['catboost_cbm']:
                    found_files['catboost_cbm'].append(cb_cbm)
    
    return found_files


def get_expected_s3_files(cohort: str, age_band: str) -> Dict[str, str]:
    """
    Get expected S3 paths for model files.
    
    Returns:
        Dict mapping file type to S3 path
    """
    age_band_fname = age_band.replace('-', '_')
    base_s3 = f"s3://{S3_BUCKET}/gold/final_model/{cohort}/{age_band}"
    
    return {
        'xgboost_joblib': f"{base_s3}/xgboost.joblib",
        'xgboost_ubj': f"{base_s3}/xgboost_model.ubj",
        'xgboost_json': f"{base_s3}/{cohort}_{age_band_fname}_best_xgboost_model.json",
        'catboost_cbm': f"{base_s3}/catboost_model.cbm",
        'catboost_joblib': f"{base_s3}/catboost.joblib",
        'catboost_json': f"{base_s3}/{cohort}_{age_band_fname}_best_catboost_model.json",
        'metadata': f"{base_s3}/{cohort}_{age_band_fname}_model_selection_metadata.json",
    }


def sync_missing_models(
    project_root: Path,
    model_base_dir: Path,
    dry_run: bool = False,
    profile: Optional[str] = None
) -> Tuple[int, int, int, int]:
    """
    Sync missing model files from local EC2 storage to S3.
    
    Returns:
        Tuple of (total_checked, uploaded, skipped, errors)
    """
    total_checked = 0
    uploaded = 0
    skipped = 0
    errors = 0
    
    print("=" * 80)
    print("Sync Missing Model Files from EC2 to S3")
    print("=" * 80)
    print()
    print(f"Project Root: {project_root}")
    print(f"Model Base Dir: {model_base_dir}")
    print(f"S3 Bucket: {S3_BUCKET}")
    if profile:
        print(f"AWS Profile: {profile}")
    else:
        print(f"AWS Profile: (using IAM role - EC2)")
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
            print()
            print(f"Age Band: {age_band}")
            
            # Find local files (checks both outputs/ and model_outputs/)
            local_files = find_local_model_files(project_root, cohort, age_band)
            
            # Get expected S3 paths
            s3_paths = get_expected_s3_files(cohort, age_band)
            
            # Check each file type
            for file_type, s3_path in s3_paths.items():
                total_checked += 1
                
                # Check if already exists in S3
                if check_s3_file_exists(s3_path, profile):
                    print(f"  [OK] Already in S3: {Path(s3_path).name}")
                    skipped += 1
                    continue
                
                # Check if we have a local file
                local_file_list = local_files.get(file_type, [])
                if not local_file_list:
                    print(f"  [SKIP] Not found locally: {Path(s3_path).name}")
                    skipped += 1
                    continue
                
                # Use first available local file
                local_file = local_file_list[0]
                
                # Upload file
                print(f"  [UPLOAD] {Path(s3_path).name}")
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
    
    return (total_checked, uploaded, skipped, errors)


def main():
    parser = argparse.ArgumentParser(
        description="Sync missing model files from EC2 local storage to S3"
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
        help=f"AWS profile to use (default: auto-detect, None on EC2)"
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
            for path_str in [
                "/home/pgx3874/pgx-analysis",
                "/mnt/nvme/pgx-analysis",
                Path.home() / "pgx-analysis"
            ]:
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
        # Try multiple possible locations
        possible_dirs = [
            project_root / "6_final_model" / "outputs",
            project_root / "6_final_model" / "model_outputs",
            project_root / "6_final_model_selection" / "outputs",
            Path("/mnt/nvme") / "6_final_model" / "outputs",
            Path("/mnt/nvme") / "6_final_model" / "model_outputs",
        ]
        
        model_base_dir = None
        for test_dir in possible_dirs:
            if test_dir.exists():
                model_base_dir = test_dir
                print(f"[INFO] Using model directory: {model_base_dir}")
                break
        
        if model_base_dir is None:
            print("[ERROR] Could not find model base directory. Please specify --model-base-dir")
            print(f"Tried: {[str(d) for d in possible_dirs]}")
            sys.exit(1)
    
    # Run sync
    total_checked, uploaded, skipped, errors = sync_missing_models(
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
    print(f"Total files checked: {total_checked}")
    print(f"Uploaded: {uploaded}")
    print(f"Skipped (already exists or not found locally): {skipped}")
    print(f"Errors: {errors}")
    print()
    
    if errors == 0:
        print("[OK] All missing files synced successfully!")
        sys.exit(0)
    else:
        print("[ERROR] Some errors occurred. Please check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
