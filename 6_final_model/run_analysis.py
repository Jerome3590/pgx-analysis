#!/usr/bin/env python3
"""
Orchestration script for Final Model Training.

This script runs the complete final model workflow:
1. Build final feature table (merge all features)
2. Remove target leakage
3. Prepare train/test splits
4. Train final model
5. Extract feature importance
6. Create visualizations

Usage:
    python 8_final_model/run_analysis.py --cohort-name opioid_ed --age-band 0-12
"""

import argparse
import sys
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import age_band_to_fname


def run_final_model_analysis(
    cohort_name: str,
    age_band: str,
    skip_feature_build: bool = False,
    skip_leakage_removal: bool = False,
    skip_train_test: bool = False,
    skip_training: bool = False,
    skip_visualizations: bool = False,
) -> bool:
    """
    Run complete final model workflow.
    
    Parameters
    ----------
    cohort_name : str
        Cohort name (e.g., "opioid_ed")
    age_band : str
        Age band (e.g., "0-12")
    skip_feature_build : bool
        Skip final feature table build (default: False)
    skip_leakage_removal : bool
        Skip target leakage removal (default: False)
    skip_train_test : bool
        Skip train/test split (default: False)
    skip_training : bool
        Skip model training (default: False)
    skip_visualizations : bool
        Skip visualization creation (default: False)
    
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    age_band_fname = age_band_to_fname(age_band)
    
    print("=" * 80)
    print(f"Final Model Analysis: {cohort_name} / {age_band}")
    print("=" * 80)
    
    # Step 1: Build final feature table
    if not skip_feature_build:
        print("\n[1/6] Building final feature table...")
        try:
            script_path = PROJECT_ROOT / "8_final_model" / "build_final_cohort_model_features.py"
            result = subprocess.run(
                [sys.executable, str(script_path), "--cohort-name", cohort_name, "--age-band", age_band],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            print("[OK] Final feature table built")
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Feature table build failed: {e.stderr}")
            return False
        except Exception as e:
            print(f"[ERROR] Feature table build failed with exception: {e}")
            return False
    else:
        print("\n[1/6] Skipping feature table build (using existing table)")
    
    # Step 2: Remove target leakage
    if not skip_leakage_removal:
        print("\n[2/6] Removing target leakage...")
        try:
            script_path = PROJECT_ROOT / "8_final_model" / "remove_target_leakage.py"
            result = subprocess.run(
                [sys.executable, str(script_path), "--cohort-name", cohort_name, "--age-band", age_band],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            print("[OK] Target leakage removed")
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Leakage removal failed: {e.stderr}")
            return False
        except Exception as e:
            print(f"[ERROR] Leakage removal failed with exception: {e}")
            return False
    else:
        print("\n[2/6] Skipping leakage removal (using existing cleaned table)")
    
    # Step 3: Prepare train/test splits
    if not skip_train_test:
        print("\n[3/6] Preparing train/test splits...")
        try:
            script_path = PROJECT_ROOT / "8_final_model" / "prepare_train_test_s3.py"
            result = subprocess.run(
                [sys.executable, str(script_path), "--cohort-name", cohort_name, "--age-band", age_band],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            print("[OK] Train/test splits prepared")
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Train/test split failed: {e.stderr}")
            return False
        except Exception as e:
            print(f"[ERROR] Train/test split failed with exception: {e}")
            return False
    else:
        print("\n[3/6] Skipping train/test split (using existing splits)")
    
    # Step 4: Train final model
    if not skip_training:
        print("\n[4/6] Training final model...")
        try:
            script_path = PROJECT_ROOT / "8_final_model" / "train_final_model.py"
            result = subprocess.run(
                [sys.executable, str(script_path), "--cohort-name", cohort_name, "--age-band", age_band],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            print("[OK] Model training completed")
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Model training failed: {e.stderr}")
            return False
        except Exception as e:
            print(f"[ERROR] Model training failed with exception: {e}")
            return False
        
        # Step 5: Extract feature importance
        print("\n[5/6] Extracting feature importance...")
        try:
            script_path = PROJECT_ROOT / "8_final_model" / "extract_final_feature_importance.py"
            result = subprocess.run(
                [sys.executable, str(script_path), "--cohort-name", cohort_name, "--age-band", age_band],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            print("[OK] Feature importance extracted")
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Feature importance extraction failed: {e.stderr}")
            return False
        except Exception as e:
            print(f"[ERROR] Feature importance extraction failed with exception: {e}")
            return False
    else:
        print("\n[4/5] Skipping model training (using existing model)")
    
    # Step 6: Create visualizations
    if not skip_visualizations:
        print("\n[6/6] Creating visualizations...")
        try:
            script_path = PROJECT_ROOT / "8_final_model" / "create_model_plots.py"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--cohort-name", cohort_name,
                    "--age-band", age_band,
                    "--event-year", "2019",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            print("[OK] Visualizations created")
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Visualization creation failed: {e.stderr}")
            return False
        except Exception as e:
            print(f"[ERROR] Visualization creation failed with exception: {e}")
            return False
    
    print("\n" + "=" * 80)
    print("Final Model Analysis Complete!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run complete final model workflow"
    )
    parser.add_argument(
        "--cohort-name",
        type=str,
        required=True,
        help="Cohort name (e.g., opioid_ed)"
    )
    parser.add_argument(
        "--age-band",
        type=str,
        required=True,
        help="Age band (e.g., 0-12)"
    )
    parser.add_argument(
        "--skip-feature-build",
        action="store_true",
        help="Skip final feature table build"
    )
    parser.add_argument(
        "--skip-leakage-removal",
        action="store_true",
        help="Skip target leakage removal"
    )
    parser.add_argument(
        "--skip-train-test",
        action="store_true",
        help="Skip train/test split"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training"
    )
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip visualization creation"
    )
    
    args = parser.parse_args()
    
    success = run_final_model_analysis(
        cohort_name=args.cohort_name,
        age_band=args.age_band,
        skip_feature_build=args.skip_feature_build,
        skip_leakage_removal=args.skip_leakage_removal,
        skip_train_test=args.skip_train_test,
        skip_training=args.skip_training,
        skip_visualizations=args.skip_visualizations,
    )
    
    sys.exit(0 if success else 1)


