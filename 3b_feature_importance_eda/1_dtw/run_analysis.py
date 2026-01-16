#!/usr/bin/env python3
"""
Orchestration script for DTW Trajectory Analysis.

This script runs the complete DTW workflow:
1. Create predictive time features
2. Create DTW features
3. Add DTW features to model data

Usage:
    python 6_dtw_analysis/run_analysis.py --cohort-name opioid_ed --age-band 0-12
"""

import argparse
import sys
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E402


def run_dtw_analysis(
    cohort_name: str,
    age_band: str,
    skip_feature_engineering: bool = False,
) -> bool:
    """
    Run complete DTW analysis workflow.
    
    Parameters
    ----------
    cohort_name : str
        Cohort name (e.g., "opioid_ed")
    age_band : str
        Age band (e.g., "0-12")
    skip_feature_engineering : bool
        Skip feature engineering steps (default: False)
    
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    print("=" * 80)
    print(f"DTW Analysis: {cohort_name} / {age_band}")
    print("=" * 80)
    
    if not skip_feature_engineering:
        # Step 1: Create predictive time features
        print("\n[1/4] Creating predictive time features...")
        try:
            script_path = PROJECT_ROOT / "6_dtw_analysis" / "create_predictive_time_features.py"
            result = subprocess.run(
                [sys.executable, str(script_path), "--cohort-name", cohort_name, "--age-band", age_band],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            print("[OK] Predictive time features created")
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Predictive time feature creation failed: {e.stderr}")
            return False
        except Exception as e:
            print(f"[ERROR] Predictive time feature creation failed with exception: {e}")
            return False
        
        # Step 2: Create DTW features
        print("\n[2/4] Creating DTW features...")
        try:
            script_path = PROJECT_ROOT / "6_dtw_analysis" / "create_dtw_features.py"
            result = subprocess.run(
                [sys.executable, str(script_path), "--cohort", cohort_name, "--age_band", age_band],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            print("[OK] DTW features created")
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] DTW feature creation failed: {e.stderr}")
            return False
        except Exception as e:
            print(f"[ERROR] DTW feature creation failed with exception: {e}")
            return False
        
        # Step 3: Add DTW features to model data
        print("\n[3/4] Adding DTW features to model data...")
        try:
            script_path = PROJECT_ROOT / "6_dtw_analysis" / "add_dtw_features_to_model_data.py"
            result = subprocess.run(
                [sys.executable, str(script_path), "--cohort-name", cohort_name, "--age-band", age_band],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            print("[OK] Features added to model data")
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Feature merge failed: {e.stderr}")
            return False
        except Exception as e:
            print(f"[ERROR] Feature merge failed with exception: {e}")
            return False
    else:
        print("\n[1/1] Skipping feature engineering (using existing features)")
    
    print("\n" + "=" * 80)
    print("DTW Analysis Complete!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run complete DTW analysis workflow"
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
        "--skip-feature-engineering",
        action="store_true",
        help="Skip feature engineering steps"
    )
    
    args = parser.parse_args()
    
    success = run_dtw_analysis(
        cohort_name=args.cohort_name,
        age_band=args.age_band,
        skip_feature_engineering=args.skip_feature_engineering,
    )
    
    sys.exit(0 if success else 1)


