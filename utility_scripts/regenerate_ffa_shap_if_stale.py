#!/usr/bin/env python3
"""
Regenerate Step 7 (FFA) and Step 8 (SHAP) outputs if Step 6 model outputs are newer.

This script checks if Step 6 model outputs are more than 5 minutes newer than
Step 7 or Step 8 outputs. If so, it clears the stale outputs and regenerates them.

Usage:
    python utility_scripts/regenerate_ffa_shap_if_stale.py --cohort opioid_ed --age-band 13-24
    python utility_scripts/regenerate_ffa_shap_if_stale.py --cohort opioid_ed --age-band 13-24 --force
    python utility_scripts/regenerate_ffa_shap_if_stale.py --all
"""

import argparse
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent


def get_step6_output_timestamp(cohort: str, age_band: str) -> Optional[float]:
    """Get the most recent modification time of Step 6 outputs."""
    age_band_fname = age_band.replace("-", "_")
    
    step6_dir = PROJECT_ROOT / "6_final_model" / "outputs" / cohort / age_band_fname
    
    if not step6_dir.exists():
        return None
    
    # Check key Step 6 output files
    key_files = [
        step6_dir / f"{cohort}_{age_band_fname}_model_selection_metadata.json",
        step6_dir / f"{cohort}_{age_band_fname}_train_final_features_no_leakage.csv",
        step6_dir / "final_model_json" / f"{cohort}_{age_band_fname}_best_xgboost_model.json",
        step6_dir / "final_model_json" / f"{cohort}_{age_band_fname}_best_catboost_model.cbm",
    ]
    
    max_time = 0.0
    for file_path in key_files:
        if file_path.exists():
            mtime = file_path.stat().st_mtime
            max_time = max(max_time, mtime)
    
    # If no key files found, check directory modification time
    if max_time == 0.0:
        if step6_dir.exists():
            max_time = step6_dir.stat().st_mtime
    
    return max_time if max_time > 0 else None


def get_step7_output_timestamp(cohort: str, age_band: str) -> Optional[float]:
    """Get the most recent modification time of Step 7 (FFA) outputs."""
    age_band_fname = age_band.replace("-", "_")
    
    step7_dir = PROJECT_ROOT / "7_ffa_analysis" / "outputs" / cohort / age_band_fname
    
    if not step7_dir.exists():
        return None
    
    # Check key Step 7 output files
    key_files = [
        step7_dir / "xgboost" / "axp_explanations.csv",
        step7_dir / "xgboost" / "feature_importance_axp.csv",
    ]
    
    max_time = 0.0
    for file_path in key_files:
        if file_path.exists():
            mtime = file_path.stat().st_mtime
            max_time = max(max_time, mtime)
    
    # If no key files found, check directory modification time
    if max_time == 0.0:
        if step7_dir.exists():
            max_time = step7_dir.stat().st_mtime
    
    return max_time if max_time > 0 else None


def get_step8_output_timestamp(cohort: str, age_band: str) -> Optional[float]:
    """Get the most recent modification time of Step 8 (SHAP) outputs."""
    age_band_fname = age_band.replace("-", "_")
    
    step8_dir = PROJECT_ROOT / "8_shap_analysis" / "outputs" / cohort / age_band_fname
    
    if not step8_dir.exists():
        return None
    
    # Check key Step 8 output files
    key_files = [
        step8_dir / f"{cohort}_{age_band_fname}_shap_global_importance_xgboost.csv",
        step8_dir / f"{cohort}_{age_band_fname}_shap_global_importance_catboost.csv",
        step8_dir / f"{cohort}_{age_band_fname}_shap_sample_values_xgboost.parquet",
    ]
    
    max_time = 0.0
    for file_path in key_files:
        if file_path.exists():
            mtime = file_path.stat().st_mtime
            max_time = max(max_time, mtime)
    
    # If no key files found, check directory modification time
    if max_time == 0.0:
        if step8_dir.exists():
            max_time = step8_dir.stat().st_mtime
    
    return max_time if max_time > 0 else None


def clear_step7_outputs(cohort: str, age_band: str, clear_s3: bool = True) -> None:
    """Clear Step 7 (FFA) outputs for a specific cohort/age band."""
    age_band_fname = age_band.replace("-", "_")
    
    # Clear local outputs
    step7_dir = PROJECT_ROOT / "7_ffa_analysis" / "outputs" / cohort / age_band_fname
    if step7_dir.exists():
        import shutil
        print(f"  Removing local Step 7 outputs: {step7_dir}")
        shutil.rmtree(step7_dir)
        print(f"  ✅ Local Step 7 outputs cleared")
    
    # Clear S3 outputs and checkpoints
    if clear_s3:
        try:
            import boto3
            s3_client = boto3.client("s3")
            bucket = "pgxdatalake"
            
            # Clear S3 outputs
            s3_prefix = f"gold/ffa_analysis/{cohort}/{age_band}/"
            paginator = s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=s3_prefix)
            
            objects_to_delete = []
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        objects_to_delete.append({"Key": obj["Key"]})
            
            if objects_to_delete:
                s3_client.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": objects_to_delete}
                )
                print(f"  ✅ S3 Step 7 outputs cleared")
            
            # Clear checkpoints (in pgx-repository bucket)
            checkpoint_prefix = f"pipeline_checkpoints/7_ffa_analysis/{cohort}/{age_band_fname}/"
            try:
                checkpoint_bucket = "pgx-repository"
                checkpoint_paginator = s3_client.get_paginator("list_objects_v2")
                checkpoint_pages = checkpoint_paginator.paginate(Bucket=checkpoint_bucket, Prefix=checkpoint_prefix)
                
                checkpoint_objects = []
                for page in checkpoint_pages:
                    if "Contents" in page:
                        for obj in page["Contents"]:
                            checkpoint_objects.append({"Key": obj["Key"]})
                
                if checkpoint_objects:
                    s3_client.delete_objects(
                        Bucket=checkpoint_bucket,
                        Delete={"Objects": checkpoint_objects}
                    )
                    print(f"  ✅ Step 7 checkpoints cleared")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not clear Step 7 checkpoints: {e}")
                
        except Exception as e:
            print(f"  ⚠️  Warning: Could not clear S3 Step 7 outputs: {e}")


def clear_step8_outputs(cohort: str, age_band: str, clear_s3: bool = True) -> None:
    """Clear Step 8 (SHAP) outputs for a specific cohort/age band."""
    age_band_fname = age_band.replace("-", "_")
    
    # Clear local outputs
    step8_dir = PROJECT_ROOT / "8_shap_analysis" / "outputs" / cohort / age_band_fname
    if step8_dir.exists():
        import shutil
        print(f"  Removing local Step 8 outputs: {step8_dir}")
        shutil.rmtree(step8_dir)
        print(f"  ✅ Local Step 8 outputs cleared")
    
    # Clear S3 outputs and checkpoints
    if clear_s3:
        try:
            import boto3
            s3_client = boto3.client("s3")
            bucket = "pgxdatalake"
            
            # Clear S3 outputs
            s3_prefix = f"gold/shap_analysis/{cohort}/{age_band}/"
            paginator = s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=s3_prefix)
            
            objects_to_delete = []
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        objects_to_delete.append({"Key": obj["Key"]})
            
            if objects_to_delete:
                s3_client.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": objects_to_delete}
                )
                print(f"  ✅ S3 Step 8 outputs cleared")
            
            # Clear checkpoints (in pgx-repository bucket)
            checkpoint_prefix = f"pipeline_checkpoints/8_shap_analysis/{cohort}/{age_band_fname}/"
            try:
                checkpoint_bucket = "pgx-repository"
                checkpoint_paginator = s3_client.get_paginator("list_objects_v2")
                checkpoint_pages = checkpoint_paginator.paginate(Bucket=checkpoint_bucket, Prefix=checkpoint_prefix)
                
                checkpoint_objects = []
                for page in checkpoint_pages:
                    if "Contents" in page:
                        for obj in page["Contents"]:
                            checkpoint_objects.append({"Key": obj["Key"]})
                
                if checkpoint_objects:
                    s3_client.delete_objects(
                        Bucket=checkpoint_bucket,
                        Delete={"Objects": checkpoint_objects}
                    )
                    print(f"  ✅ Step 8 checkpoints cleared")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not clear Step 8 checkpoints: {e}")
                
        except Exception as e:
            print(f"  ⚠️  Warning: Could not clear S3 Step 8 outputs: {e}")


def run_step7(cohort: str, age_band: str) -> bool:
    """Run Step 7 (FFA Analysis)."""
    script_path = PROJECT_ROOT / "7_ffa_analysis" / "run_full_ffa_analysis.py"
    
    if not script_path.exists():
        print(f"  ❌ Step 7 script not found: {script_path}")
        return False
    
    print(f"  Running Step 7 (FFA Analysis)...")
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--cohort-name",
                cohort,
                "--age-band",
                age_band,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"  ✅ Step 7 completed successfully")
        if result.stdout:
            print(f"  Step 7 stdout:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Step 7 failed (returncode={e.returncode})")
        if e.stderr:
            print(f"  Step 7 stderr:\n{e.stderr}")
        return False


def run_step8(cohort: str, age_band: str) -> bool:
    """Run Step 8 (SHAP Analysis)."""
    script_path = PROJECT_ROOT / "8_shap_analysis" / "run_shap_analysis.py"
    
    if not script_path.exists():
        print(f"  ❌ Step 8 script not found: {script_path}")
        return False
    
    print(f"  Running Step 8 (SHAP Analysis)...")
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--cohort",
                cohort,
                "--age_band",
                age_band,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"  ✅ Step 8 completed successfully")
        if result.stdout:
            print(f"  Step 8 stdout:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Step 8 failed (returncode={e.returncode})")
        if e.stderr:
            print(f"  Step 8 stderr:\n{e.stderr}")
        return False


def check_and_regenerate(
    cohort: str,
    age_band: str,
    stale_threshold_minutes: int = 5,
    force: bool = False,
    clear_s3: bool = True,
    regenerate: bool = True,
) -> Tuple[bool, bool]:
    """
    Check if Step 6 outputs are newer than Step 7/8 and regenerate if needed.
    
    Returns:
        Tuple of (step7_regenerated, step8_regenerated) booleans
    """
    age_band_fname = age_band.replace("-", "_")
    
    print(f"\n{'='*60}")
    print(f"Checking: cohort={cohort}, age_band={age_band}")
    print(f"{'='*60}")
    
    # Get timestamps
    step6_time = get_step6_output_timestamp(cohort, age_band)
    step7_time = get_step7_output_timestamp(cohort, age_band)
    step8_time = get_step8_output_timestamp(cohort, age_band)
    
    if step6_time is None:
        print(f"  ⚠️  Step 6 outputs not found for {cohort}/{age_band}")
        print(f"     Skipping regeneration (Step 6 must exist first)")
        return False, False
    
    step6_dt = datetime.fromtimestamp(step6_time)
    print(f"  Step 6 (models) timestamp: {step6_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    
    stale_threshold = timedelta(minutes=stale_threshold_minutes)
    step7_regenerated = False
    step8_regenerated = False
    
    # Check Step 7
    if step7_time is None:
        print(f"  Step 7 (FFA) outputs: Not found")
        if regenerate:
            print(f"  → Step 7 outputs missing, will regenerate")
            clear_step7_outputs(cohort, age_band, clear_s3=clear_s3)
            if regenerate:
                step7_regenerated = run_step7(cohort, age_band)
    else:
        step7_dt = datetime.fromtimestamp(step7_time)
        age_diff = step6_dt - step7_dt
        print(f"  Step 7 (FFA) timestamp: {step7_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Age difference: {age_diff}")
        
        if force or age_diff > stale_threshold:
            print(f"  → Step 7 outputs are stale (Step 6 is {age_diff} newer)")
            clear_step7_outputs(cohort, age_band, clear_s3=clear_s3)
            if regenerate:
                step7_regenerated = run_step7(cohort, age_band)
        else:
            print(f"  ✓ Step 7 outputs are up-to-date")
    
    # Check Step 8
    if step8_time is None:
        print(f"  Step 8 (SHAP) outputs: Not found")
        if regenerate:
            print(f"  → Step 8 outputs missing, will regenerate")
            clear_step8_outputs(cohort, age_band, clear_s3=clear_s3)
            if regenerate:
                step8_regenerated = run_step8(cohort, age_band)
    else:
        step8_dt = datetime.fromtimestamp(step8_time)
        age_diff = step6_dt - step8_dt
        print(f"  Step 8 (SHAP) timestamp: {step8_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Age difference: {age_diff}")
        
        if force or age_diff > stale_threshold:
            print(f"  → Step 8 outputs are stale (Step 6 is {age_diff} newer)")
            clear_step8_outputs(cohort, age_band, clear_s3=clear_s3)
            if regenerate:
                step8_regenerated = run_step8(cohort, age_band)
        else:
            print(f"  ✓ Step 8 outputs are up-to-date")
    
    return step7_regenerated, step8_regenerated


def find_all_cohorts_age_bands() -> list[Tuple[str, str]]:
    """Find all cohort/age_band combinations from Step 6 outputs."""
    step6_base = PROJECT_ROOT / "6_final_model" / "outputs"
    
    if not step6_base.exists():
        return []
    
    combinations = []
    for cohort_dir in step6_base.iterdir():
        if not cohort_dir.is_dir():
            continue
        
        cohort = cohort_dir.name
        for age_band_dir in cohort_dir.iterdir():
            if not age_band_dir.is_dir():
                continue
            
            age_band_fname = age_band_dir.name
            age_band = age_band_fname.replace("_", "-")
            
            # Check if Step 6 outputs actually exist
            if get_step6_output_timestamp(cohort, age_band) is not None:
                combinations.append((cohort, age_band))
    
    return combinations


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate Step 7 (FFA) and Step 8 (SHAP) outputs if Step 6 model outputs "
            "are newer than 5 minutes. This ensures downstream analyses stay in sync with models."
        )
    )
    parser.add_argument(
        "--cohort",
        type=str,
        help="Cohort name (e.g., opioid_ed). Required unless --all is used.",
    )
    parser.add_argument(
        "--age-band",
        type=str,
        help="Age band (e.g., 13-24). Required unless --all is used.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all cohort/age_band combinations found in Step 6 outputs",
    )
    parser.add_argument(
        "--stale-threshold-minutes",
        type=int,
        default=5,
        help="Minimum age difference (minutes) to consider outputs stale (default: 5)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if outputs are not stale",
    )
    parser.add_argument(
        "--no-s3",
        action="store_true",
        help="Skip clearing S3 outputs (only clear local)",
    )
    parser.add_argument(
        "--no-regenerate",
        action="store_true",
        help="Only clear stale outputs, do not regenerate",
    )
    
    args = parser.parse_args()
    
    if args.all:
        combinations = find_all_cohorts_age_bands()
        if not combinations:
            print("No cohort/age_band combinations found in Step 6 outputs")
            return
        
        print(f"Found {len(combinations)} cohort/age_band combinations")
        for cohort, age_band in combinations:
            check_and_regenerate(
                cohort=cohort,
                age_band=age_band,
                stale_threshold_minutes=args.stale_threshold_minutes,
                force=args.force,
                clear_s3=not args.no_s3,
                regenerate=not args.no_regenerate,
            )
    else:
        if not args.cohort or not args.age_band:
            parser.error("--cohort and --age-band are required unless --all is used")
        
        check_and_regenerate(
            cohort=args.cohort,
            age_band=args.age_band,
            stale_threshold_minutes=args.stale_threshold_minutes,
            force=args.force,
            clear_s3=not args.no_s3,
            regenerate=not args.no_regenerate,
        )


if __name__ == "__main__":
    main()

