#!/usr/bin/env python3
"""
DTW Trajectory Analysis for Non-Value-Added Codes

Calls the working DTW Python scripts to perform comprehensive DTW analysis:
1. Creates DTW trajectory features from patient sequences
2. Computes DTW distances to prototype trajectories
3. Creates patient-level features for model training
4. Merges features into final output ready for model training

This script orchestrates the Python-based DTW pipeline.
"""

import argparse
import sys
import subprocess
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import age_band_to_fname


def run_dtw_analysis(
    cohort: str,
    age_band: str,
    project_root: Path
) -> bool:
    """
    Run comprehensive DTW analysis using working Python scripts.
    
    Args:
        cohort: Cohort name (e.g., 'opioid_ed')
        age_band: Age band (e.g., '13-24')
        project_root: Project root directory
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"DTW Trajectory Analysis: {cohort} / {age_band}")
    print(f"{'='*80}")
    
    dtw_dir = project_root / "3b_feature_importance_eda" / "1_dtw"
    
    # Step 1: Create DTW features
    create_features_script = dtw_dir / "create_dtw_features.py"
    if not create_features_script.exists():
        print(f"[ERROR] DTW script not found: {create_features_script}")
        return False
    
    print(f"[INFO] Step 1: Creating DTW features...")
    cmd1 = [
        sys.executable,
        str(create_features_script),
        "--cohort", cohort,
        "--age_band", age_band
    ]
    
    try:
        result1 = subprocess.run(
            cmd1,
            capture_output=True,
            text=True,
            cwd=str(project_root)
        )
        
        if result1.returncode != 0:
            print(f"[ERROR] DTW feature creation failed with return code {result1.returncode}")
            print(f"STDOUT:\n{result1.stdout}")
            print(f"STDERR:\n{result1.stderr}")
            return False
        
        print(f"[OK] DTW features created successfully")
        if result1.stdout:
            print(f"Output:\n{result1.stdout}")
    
    except Exception as e:
        print(f"[ERROR] Error running DTW feature creation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Merge DTW features
    merge_features_script = dtw_dir / "add_dtw_features_to_model_data.py"
    if not merge_features_script.exists():
        print(f"[WARN] DTW merge script not found: {merge_features_script}")
        print(f"[INFO] Skipping merge step (features may already be merged)")
        return True
    
    print(f"[INFO] Step 2: Merging DTW features...")
    cmd2 = [
        sys.executable,
        str(merge_features_script),
        "--cohort-name", cohort,
        "--age-band", age_band,
        "--project-root", str(project_root)
    ]
    
    try:
        result2 = subprocess.run(
            cmd2,
            capture_output=True,
            text=True,
            cwd=str(project_root)
        )
        
        if result2.returncode != 0:
            print(f"[WARN] DTW feature merge failed (non-critical): {result2.returncode}")
            print(f"STDOUT:\n{result2.stdout}")
            print(f"STDERR:\n{result2.stderr}")
            # Don't fail the whole process if merge fails
        else:
            print(f"[OK] DTW features merged successfully")
            if result2.stdout:
                print(f"Output:\n{result2.stdout}")
    
    except Exception as e:
        print(f"[WARN] Error running DTW feature merge (non-critical): {e}")
    
    # Verify outputs were created
    age_band_fname = age_band_to_fname(age_band)
    output_dir = project_root / "3b_feature_importance_eda" / "outputs" / "feature_engineering"
    
    # Check for DTW feature files
    expected_files = [
        output_dir / f"dtw_features_{cohort}_{age_band_fname}.csv",
        output_dir / f"dtw_added_features_{cohort}_{age_band_fname}.csv"
    ]
    
    existing_files = [f for f in expected_files if f.exists()]
    if existing_files:
        print(f"[OK] DTW output files created:")
        for f in existing_files:
            print(f"  - {f}")
    else:
        print(f"[WARN] Expected DTW output files not found (may be in different location)")
    
    # Check for visualization files
    plots_dir = project_root / "3b_feature_importance_eda" / "outputs" / cohort / age_band_fname / "plots"
    if plots_dir.exists():
        viz_files = list(plots_dir.glob("dtw_*.png"))
        if viz_files:
            print(f"[OK] DTW visualization files created ({len(viz_files)} files):")
            for f in viz_files:
                print(f"  - {f.name}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="DTW trajectory analysis using working Python scripts"
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
    success = run_dtw_analysis(
        cohort=args.cohort,
        age_band=args.age_band,
        project_root=project_root
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
