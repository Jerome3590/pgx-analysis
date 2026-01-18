#!/usr/bin/env python3
"""
Create BupaR visualizations for Step 3b.

This script generates BupaR process mining visualizations including:
- Activity frequency plots
- Gantt charts (overall, by code type)
- Activity sequence plots
- Pre/post-target visualizations
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import age_band_to_fname

try:
    from py_helpers.common_imports import s3_client, S3_BUCKET
except ImportError:
    import boto3
    s3_client = boto3.client("s3")
    S3_BUCKET = "pgxdatalake"

try:
    from py_helpers.checkpoint_utils import upload_file_to_s3
except ImportError:
    def upload_file_to_s3(local_path: Path, s3_path: str, logger=None, check_exists: bool = True) -> bool:
        """Upload file to S3 using boto3."""
        if not local_path.exists():
            print(f"[ERROR] Local file does not exist: {local_path}")
            return False
        
        try:
            if s3_path.startswith("s3://"):
                s3_path = s3_path[5:]
            parts = s3_path.split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            
            if check_exists:
                try:
                    s3_client.head_object(Bucket=bucket, Key=key)
                    print(f"[OK] File already exists in S3: s3://{bucket}/{key} (skipping upload)")
                    return True
                except s3_client.exceptions.ClientError as e:
                    if e.response["Error"]["Code"] not in ["404", "NoSuchKey"]:
                        raise
            
            s3_client.upload_file(str(local_path), bucket, key)
            print(f"[OK] Uploaded to S3: s3://{bucket}/{key}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to upload {local_path} to s3://{bucket}/{key}: {e}")
            return False


def find_r_script(cohort: str) -> Optional[Path]:
    """Find the appropriate R script for the cohort."""
    r_scripts_dir = PROJECT_ROOT / "9_risk_dashboard" / "bupaR_dashboard_visual"
    
    if cohort == "opioid_ed":
        r_script = r_scripts_dir / "create_bupar_outputs_opioid_ed.R"
    elif cohort == "non_opioid_ed":
        r_script = r_scripts_dir / "create_bupar_outputs_non_opioid_ed.R"
    else:
        # Try opioid_ed script as fallback
        r_script = r_scripts_dir / "create_bupar_outputs_opioid_ed.R"
    
    if r_script.exists():
        return r_script
    return None


def check_prerequisites(cohort: str, age_band: str) -> tuple[bool, str]:
    """Check if prerequisites exist for BupaR visualization."""
    age_band_fname = age_band_to_fname(age_band)
    
    # Check model_events.parquet exists
    model_events_paths = [
        PROJECT_ROOT / "4a_model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
        PROJECT_ROOT / "4a_model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events_no_protocols.parquet"
    ]
    
    model_events_path = None
    for path in model_events_paths:
        if path.exists():
            model_events_path = path
            break
    
    if not model_events_path:
        return False, f"Model events file not found for {cohort}/{age_band}"
    
    # Check R script exists
    r_script = find_r_script(cohort)
    if not r_script:
        return False, f"R script not found for cohort {cohort}"
    
    # R script uses model_events.parquet directly
    return True, ""


def create_bupar_visualizations(cohort: str, age_band: str, r_script_path: Path) -> bool:
    """Create BupaR visualizations by calling R script."""
    print(f"\n{'='*80}")
    print(f"Creating BupaR Visualizations: {cohort} / {age_band}")
    print(f"{'='*80}")
    
    # Change to project root directory for R script
    original_cwd = Path.cwd()
    os.chdir(PROJECT_ROOT)
    
    try:
        # Call R script with age_band as argument
        cmd = [
            "Rscript",
            str(r_script_path),
            age_band
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        
        if result.returncode != 0:
            print(f"[ERROR] R script failed with return code {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False
        
        print(f"[OK] R script completed successfully")
        if result.stdout:
            print(f"R Output:\n{result.stdout}")
        
        return True
        
    except FileNotFoundError:
        print(f"[ERROR] Rscript not found. Please ensure R is installed and in PATH")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to run R script: {e}")
        return False
    finally:
        os.chdir(original_cwd)


def copy_visualizations_to_step3b(cohort: str, age_band: str) -> list[Path]:
    """Copy visualizations from BupaR output directory to Step 3b outputs."""
    age_band_fname = age_band_to_fname(age_band)
    
    # Try multiple possible source locations (R script saves to different places)
    possible_sources = [
        # Primary location: 10c_bupaR_dashboard_visual (or 9_risk_dashboard)
        PROJECT_ROOT / "10c_bupaR_dashboard_visual" / "outputs" / cohort / age_band_fname / "plots",
        PROJECT_ROOT / "9_risk_dashboard" / "bupaR_dashboard_visual" / "outputs" / cohort / age_band_fname / "plots",
        # Secondary location: 5_feature_engineering (where R script copies plots)
        PROJECT_ROOT / "5_feature_engineering" / "feature_engineering_outputs" / "5_bupar" / cohort / age_band / "plots",
    ]
    
    # Destination: Step 3b outputs
    dest_plots_dir = (
        PROJECT_ROOT / "3b_feature_importance_eda" / "outputs" / 
        cohort / age_band_fname / "plots"
    )
    
    copied_files = []
    source_plots_dir = None
    
    # Find the first existing source directory
    for source_dir in possible_sources:
        if source_dir.exists():
            source_plots_dir = source_dir
            print(f"Found plots in: {source_dir}")
            break
    
    if not source_plots_dir:
        print(f"[WARN] Source plots directory not found in any expected location")
        print(f"Checked: {[str(s) for s in possible_sources]}")
        return copied_files
    
    # Create destination directory
    dest_plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all PNG files (skip PDF files)
    import shutil
    for png_file in source_plots_dir.glob("*.png"):
        dest_file = dest_plots_dir / png_file.name
        shutil.copy2(png_file, dest_file)
        copied_files.append(dest_file)
        print(f"Copied: {png_file.name}")
    
    return copied_files


def upload_visualizations_to_s3(cohort: str, age_band: str, plot_files: list[Path]) -> None:
    """Upload visualization files to S3."""
    age_band_fname = age_band_to_fname(age_band)
    
    for plot_file in plot_files:
        s3_path = (
            f"s3://{S3_BUCKET}/gold/feature_importance/{cohort}/{age_band}/"
            f"plots/{plot_file.name}"
        )
        upload_file_to_s3(plot_file, s3_path, check_exists=True)


def main():
    import os
    
    parser = argparse.ArgumentParser(
        description="Create BupaR visualizations for Step 3b"
    )
    parser.add_argument("--cohort", required=True, help="Cohort name")
    parser.add_argument("--age-band", required=True, help="Age band")
    parser.add_argument(
        "--skip-r",
        action="store_true",
        help="Skip R script execution (only copy existing plots)"
    )
    
    args = parser.parse_args()
    
    # Check prerequisites
    prereq_ok, prereq_msg = check_prerequisites(args.cohort, args.age_band)
    if not prereq_ok:
        print(f"[ERROR] Prerequisites not met: {prereq_msg}")
        sys.exit(1)
    
    # Find R script
    r_script = find_r_script(args.cohort)
    if not r_script:
        print(f"[ERROR] Could not find R script for cohort {args.cohort}")
        sys.exit(1)
    
    # Run R script to create visualizations
    if not args.skip_r:
        success = create_bupar_visualizations(args.cohort, args.age_band, r_script)
        if not success:
            print(f"[WARN] R script execution had issues, but continuing...")
    
    # Copy visualizations to Step 3b outputs
    plot_files = copy_visualizations_to_step3b(args.cohort, args.age_band)
    
    if not plot_files:
        print(f"[WARN] No visualization files found to copy")
        sys.exit(1)
    
    print(f"\n[OK] Copied {len(plot_files)} visualization files to Step 3b outputs")
    
    # Upload to S3
    print(f"\nUploading visualizations to S3...")
    upload_visualizations_to_s3(args.cohort, args.age_band, plot_files)
    
    print(f"\n[OK] BupaR visualizations created for {args.cohort} / {args.age_band}")


if __name__ == "__main__":
    main()
