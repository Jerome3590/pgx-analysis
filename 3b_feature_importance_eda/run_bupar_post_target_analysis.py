#!/usr/bin/env python3
"""
BupaR Post-Target Event Analysis

Calls the working BupaR R scripts to perform comprehensive BupaR analysis:
1. Builds BupaR event logs from model_events.parquet
2. Runs pre- and post-F1120 sequence analyses
3. Generates comprehensive BupaR features
4. Merges features into final output ready for model training

This script orchestrates the R-based BupaR pipeline.
"""

import argparse
import sys
import subprocess
import os
import platform
from pathlib import Path
from typing import Optional

# Detect operating system and set project root
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'

if IS_WINDOWS:
    # Windows: Use current workspace directory
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
elif IS_LINUX:
    # Linux/EC2: Use EC2 path
    PROJECT_ROOT = Path('/home/pgx3874/pgx-analysis')
else:
    # Fallback: Use current file's parent directory
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import age_band_to_fname


def find_rscript() -> Optional[str]:
    """Find Rscript executable."""
    import shutil
    rscript = shutil.which("Rscript")
    if rscript:
        return rscript
    
    # Try common locations
    common_paths = [
        "C:/Program Files/R/R-4.5.0/bin/Rscript.exe",
        "C:/Program Files/R/R-4.4.0/bin/Rscript.exe",
        "/usr/bin/Rscript",
        "/usr/local/bin/Rscript"
    ]
    
    for path in common_paths:
        if Path(path).exists():
            return path
    
    return None


def run_bupar_analysis(
    cohort: str,
    age_band: str,
    project_root: Path
) -> bool:
    """
    Run comprehensive BupaR analysis using working R scripts.
    
    Args:
        cohort: Cohort name (e.g., 'opioid_ed')
        age_band: Age band (e.g., '13-24')
        project_root: Project root directory
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"BupaR Analysis: {cohort} / {age_band}")
    print(f"{'='*80}")
    
    # Find Rscript
    rscript = find_rscript()
    if not rscript:
        print("[ERROR] Rscript not found. Please ensure R is installed and in PATH.")
        return False
    
    print(f"[INFO] Using Rscript: {rscript}")
    
    # Determine which R script to use based on cohort
    bupar_dir = project_root / "3b_feature_importance_eda" / "1_bupaR"
    
    if cohort == "opioid_ed":
        r_script = bupar_dir / "create_bupar_outputs_opioid_ed.R"
    elif cohort == "non_opioid_ed":
        r_script = bupar_dir / "create_bupar_outputs_non_opioid_ed.R"
    else:
        print(f"[ERROR] Unknown cohort: {cohort}")
        return False
    
    if not r_script.exists():
        print(f"[ERROR] R script not found: {r_script}")
        return False
    
    # Change to project root directory for R script
    original_cwd = os.getcwd()
    os.chdir(project_root)
    
    try:
        # Call R script with age_band as argument
        cmd = [rscript, str(r_script), age_band]
        
        print(f"[INFO] Running: {' '.join(cmd)}")
        print(f"[INFO] Working directory: {project_root}")
        
        # Set environment to use UTF-8 encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        if IS_WINDOWS:
            # On Windows, ensure UTF-8 is used
            env['PYTHONUTF8'] = '1'
            # Also set R's encoding
            env['R_ENCODING'] = 'UTF-8'
        
        # Use Popen with explicit encoding to avoid threading issues
        import io
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(project_root),
            env=env
        )
        
        # Read output with explicit UTF-8 encoding and error handling
        stdout_text = ""
        stderr_text = ""
        
        try:
            stdout_bytes, stderr_bytes = process.communicate(timeout=3600)  # 1 hour timeout
            stdout_text = stdout_bytes.decode('utf-8', errors='replace')
            stderr_text = stderr_bytes.decode('utf-8', errors='replace')
        except subprocess.TimeoutExpired:
            process.kill()
            stdout_bytes, stderr_bytes = process.communicate()
            stdout_text = stdout_bytes.decode('utf-8', errors='replace')
            stderr_text = stderr_bytes.decode('utf-8', errors='replace')
            print("[ERROR] R script timed out after 1 hour")
            return False
        
        # Create result-like object
        class Result:
            def __init__(self, returncode, stdout, stderr):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr
        
        result = Result(process.returncode, stdout_text, stderr_text)
        
        # Filter out encoding errors from stderr (they're harmless)
        if result.stderr:
            # Remove UnicodeDecodeError messages from stderr
            stderr_lines = result.stderr.split('\n')
            filtered_stderr = [line for line in stderr_lines 
                             if 'UnicodeDecodeError' not in line 
                             and 'charmap' not in line 
                             and 'codec' not in line]
            if filtered_stderr:
                result.stderr = '\n'.join(filtered_stderr)
            else:
                result.stderr = ''
        
        if result.returncode != 0:
            print(f"[ERROR] R script failed with return code {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False
        
        print(f"[OK] R script completed successfully")
        if result.stdout:
            print(f"R Output:\n{result.stdout}")
        
        # Verify outputs were created
        age_band_fname = age_band_to_fname(age_band)
        output_dir = project_root / "3b_feature_importance_eda" / "outputs" / cohort / age_band_fname
        
        # Check for key output files
        expected_files = [
            output_dir / "features" / f"{cohort}_{age_band_fname}_train_target_pre_f1120_patient_features_bupar.csv",
            output_dir / "features" / f"{cohort}_{age_band_fname}_train_target_post_f1120_patient_features_bupar.csv",
            output_dir / "features" / f"{cohort}_{age_band_fname}_train_target_time_to_f1120_features_bupar.csv"
        ]
        
        missing_files = [f for f in expected_files if not f.exists()]
        if missing_files:
            print(f"[WARN] Some expected output files are missing:")
            for f in missing_files:
                print(f"  - {f}")
        else:
            print(f"[OK] All expected BupaR output files created")
        
        # Create post-target analysis CSV
        print(f"\n[INFO] Creating post-target analysis CSV...")
        try:
            create_analysis_script = project_root / "3b_feature_importance_eda" / "create_bupar_post_target_analysis.py"
            if create_analysis_script.exists():
                # Note: subprocess already imported at top of file
                cmd = [
                    sys.executable,
                    str(create_analysis_script),
                    "--cohort", cohort,
                    "--age-band", age_band
                ]
                result_analysis = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(project_root)
                )
                if result_analysis.returncode == 0:
                    print(f"[OK] Post-target analysis CSV created successfully")
                    if result_analysis.stdout:
                        print(result_analysis.stdout)
                else:
                    print(f"[WARN] Failed to create post-target analysis CSV:")
                    print(result_analysis.stderr)
            else:
                print(f"[WARN] Post-target analysis script not found: {create_analysis_script}")
        except Exception as e:
            print(f"[WARN] Error creating post-target analysis CSV: {e}")
        
        return True
        
    except FileNotFoundError:
        print(f"[ERROR] Rscript not found. Please ensure R is installed and in PATH")
        return False
    except Exception as e:
        print(f"[ERROR] Error running BupaR analysis: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(original_cwd)


def main():
    parser = argparse.ArgumentParser(
        description="BupaR post-target event analysis using working R scripts"
    )
    parser.add_argument("--cohort", required=True, help="Cohort name")
    parser.add_argument("--age-band", required=True, help="Age band")
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Determine project root
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        project_root = PROJECT_ROOT
    
    # Run analysis
    success = run_bupar_analysis(
        cohort=args.cohort,
        age_band=args.age_band,
        project_root=project_root
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
