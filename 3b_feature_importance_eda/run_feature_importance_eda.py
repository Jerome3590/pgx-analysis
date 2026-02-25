#!/usr/bin/env python3
"""
Feature Importance EDA and Refinement - Orchestration Script

Runs all Feature Importance EDA analyses in order:
1. Administrative/Non-informative code filtering (remove non-informative ICD/CPT codes)
2. BupaR post-target analysis (identify pre/post target events; F1120 for opioid_ed, HCG for polypharmacy)
3. Create safe feature filter JSON (idempotent: skip if already exists; exclude leakage, keep pre-target features)
3.5. Ensure aggregated feature importance (Step 3a): if missing or empty, rerun Step 3a for this cohort/age_band
4. Filter and refine feature importances (uses safe_feature_filter.json when present)
5. Create BupaR visualizations

Outputs refined cohort_feature_importance files for Step 4.
"""

import argparse
import gc
import sys
import subprocess
import os
import platform
from pathlib import Path
from typing import List, Tuple

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

from py_helpers.constants import age_band_to_fname, REQUIRED_COHORTS
from py_helpers.env_utils import get_workflow_python_bin

try:
    from py_helpers.feature_importance_eda_utils import (
        resolve_aggregated_fi_path,
        load_aggregated_feature_importance,
    )
except ImportError:
    resolve_aggregated_fi_path = None
    load_aggregated_feature_importance = None

try:
    from py_helpers.checkpoint_utils import save_step_checkpoint
except ImportError:
    def save_step_checkpoint(step_name: str, cohort: str, age_band: str, metadata=None, output_paths=None, logger=None) -> bool:
        """Dummy checkpoint function if checkpoint_utils not available."""
        return True

# Both cohorts use full set of age bands (from py_helpers.constants)
COHORTS = REQUIRED_COHORTS


def run_bupar_analysis(cohort: str, age_band: str, script_dir: Path) -> bool:
    """Run BupaR post-target analysis."""
    print(f"\n{'='*80}")
    print(f"Running BupaR Post-Target Analysis: {cohort} / {age_band}")
    print(f"{'='*80}")
    
    script_path = script_dir / "1_bupaR" / "run_bupar_post_target_analysis.py"
    cmd = [
        str(get_workflow_python_bin()),
        str(script_path),
        "--cohort", cohort,
        "--age-band", age_band
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] BupaR analysis failed: {e}")
        return False




def run_create_safe_feature_filter(cohort: str, age_band: str, script_dir: Path) -> bool:
    """Create safe feature filter JSON (idempotent: skip if JSON already exists)."""
    age_band_fname = age_band_to_fname(age_band)
    output_dir = script_dir / "outputs" / cohort / age_band_fname
    json_path = output_dir / f"{cohort}_{age_band_fname}_safe_feature_filter.json"
    if json_path.exists():
        print(f"\n[INFO] Safe feature filter already exists (idempotent skip): {json_path}")
        return True
    print(f"\n{'='*80}")
    print(f"Creating Safe Feature Filter: {cohort} / {age_band}")
    print(f"{'='*80}")
    script_path = script_dir / "2_filtering" / "create_safe_feature_filter_json.py"
    cmd = [
        str(get_workflow_python_bin()),
        str(script_path),
        "--cohort", cohort,
        "--age-band", age_band,
    ]
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Create safe feature filter failed: {e}")
        return False


def ensure_aggregated_fi(cohort: str, age_band: str, script_dir: Path) -> bool:
    """
    Ensure aggregated feature importance (Step 3a) exists and is non-empty for this cohort/age_band.
    If missing or empty, runs Step 3a (run_mc_feature_importance) then returns.
    Returns True if data is available (or 3a ran successfully), False if 3a failed.
    """
    if resolve_aggregated_fi_path is None or load_aggregated_feature_importance is None:
        return True  # Skip check if utils not available
    try:
        load_aggregated_feature_importance(cohort, age_band, PROJECT_ROOT)
        return True
    except FileNotFoundError:
        print(f"[INFO] Aggregated feature importance missing for {cohort}/{age_band}; running Step 3a...")
    except ValueError as e:
        if "empty" in str(e).lower() or "0 rows" in str(e):
            print(f"[INFO] Aggregated feature importance empty for {cohort}/{age_band}; running Step 3a...")
        else:
            print(f"[WARN] {e}")
            return False
    script_3a = PROJECT_ROOT / "3a_feature_importance" / "run_mc_feature_importance.py"
    if not script_3a.exists():
        print(f"[ERROR] Step 3a script not found: {script_3a}")
        return False
    cmd = [str(get_workflow_python_bin()), str(script_3a), "--cohort", cohort, "--age_band", age_band]
    try:
        result = subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            return False
        print(f"[OK] Step 3a completed for {cohort}/{age_band}; continuing with filter and refine.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Step 3a failed for {cohort}/{age_band}: {e}")
        return False


def run_filter_and_refine(cohort: str, age_band: str, script_dir: Path) -> bool:
    """Run filter and refine step."""
    print(f"\n{'='*80}")
    print(f"Filtering and Refining Features: {cohort} / {age_band}")
    print(f"{'='*80}")
    
    script_path = script_dir / "2_filtering" / "filter_and_refine_features.py"
    cmd = [
        str(get_workflow_python_bin()),
        str(script_path),
        "--cohort", cohort,
        "--age-band", age_band
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Filter and refine failed: {e}")
        return False


def create_bupar_visualizations(cohort: str, age_band: str, script_dir: Path) -> bool:
    """Create BupaR visualizations."""
    print(f"\n{'='*80}")
    print(f"Creating BupaR Visualizations: {cohort} / {age_band}")
    print(f"{'='*80}")
    
    script_path = script_dir / "1_bupaR" / "create_bupar_visualizations.py"
    cmd = [
        str(get_workflow_python_bin()),
        str(script_path),
        "--cohort", cohort,
        "--age-band", age_band
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] BupaR visualization creation failed: {e}")
        return False


def run_feature_importance_eda_for_cohort(cohort: str, age_band: str, script_dir: Path) -> bool:
    """Run all Feature Importance EDA analyses for a single cohort/age_band."""
    print(f"\n{'='*80}")
    print(f"Feature Importance EDA and Refinement")
    print(f"Cohort: {cohort} / Age Band: {age_band}")
    print(f"{'='*80}")
    
    # Step 1: Administrative/Non-informative code filtering (runs first)
    # This loads administrative codes from lookup table and filters them
    # Note: This is handled in filter_and_refine_features.py, but we document it here
    print(f"\n[INFO] Step 1: Administrative/Non-informative code filtering")
    print(f"       (Handled in filter_and_refine step using administrative_codes_lookup.json)")
    
    # Step 2: BupaR post-target analysis (identify pre/post target events)
    if not run_bupar_analysis(cohort, age_band, script_dir):
        print(f"[WARN] BupaR analysis failed, continuing...")
    
    # Step 3: Create safe feature filter JSON (idempotent: skip if already exists)
    if not run_create_safe_feature_filter(cohort, age_band, script_dir):
        print("[WARN] Safe feature filter not created; filter_and_refine will use BupaR CSV fallback")

    # Step 3.5: Ensure aggregated feature importance (Step 3a) exists and is non-empty; rerun 3a if missing
    if not ensure_aggregated_fi(cohort, age_band, script_dir):
        print("[ERROR] Could not obtain aggregated feature importance (Step 3a run failed or skipped)")
        return False

    # Step 4: Filter and refine (combines all filtering results; uses safe_feature_filter.json when present)
    if not run_filter_and_refine(cohort, age_band, script_dir):
        print(f"[ERROR] Filter and refine failed")
        return False
    
    # Step 5: Create BupaR visualizations
    if not create_bupar_visualizations(cohort, age_band, script_dir):
        print(f"[WARN] BupaR visualization creation failed, continuing...")
    
    # Save checkpoint to S3
    age_band_fname = age_band_to_fname(age_band)
    output_paths = [
        f"s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/{cohort}_{age_band_fname}_cohort_feature_importance.csv",
        f"s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/{cohort}_{age_band_fname}_feature_filtering_summary.json"
    ]
    
    metadata = {
        "step": "3b_feature_importance_eda",
        "cohort": cohort,
        "age_band": age_band,
        "status": "completed"
    }
    
    if save_step_checkpoint(
        step_name="3b_feature_importance_eda",
        cohort=cohort,
        age_band=age_band,
        metadata=metadata,
        output_paths=output_paths
    ):
        print(f"[OK] Checkpoint saved to S3")
    
    print(f"\n[OK] Feature Importance EDA completed for {cohort} / {age_band}")
    # Encourage GC before next age band when running multiple in sequence
    gc.collect()
    return True


def clear_age_band_memory():
    """
    Run garbage collection to free memory after processing an age band.
    Call this in notebooks/workflows after each age band when running multiple
    in the same session (e.g. after loading aggregated_fi, bupar_results,
    model_events, or refined_fi) to avoid memory growth across bands.
    """
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Feature Importance EDA and Refinement"
    )
    parser.add_argument(
        "--cohort",
        type=str,
        choices=list(COHORTS.keys()),
        help="Cohort name (required unless --all-cohorts)"
    )
    parser.add_argument(
        "--age-band",
        type=str,
        help="Age band (required unless --all-cohorts)"
    )
    parser.add_argument(
        "--all-cohorts",
        action="store_true",
        help="Run for all cohorts and age bands"
    )
    parser.add_argument(
        "--skip-bupar",
        action="store_true",
        help="Skip BupaR analysis"
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    
    # Determine cohorts to process
    if args.all_cohorts:
        cohorts_to_process = []
        for cohort, age_bands in COHORTS.items():
            for age_band in age_bands:
                cohorts_to_process.append((cohort, age_band))
    elif args.cohort and args.age_band:
        cohorts_to_process = [(args.cohort, args.age_band)]
    elif args.cohort:
        cohorts_to_process = [
            (args.cohort, age_band) 
            for age_band in COHORTS[args.cohort]
        ]
    else:
        print("[ERROR] Must specify --cohort and --age-band, or --all-cohorts")
        sys.exit(1)
    
    print("=" * 80)
    print("Feature Importance EDA and Refinement")
    print("=" * 80)
    print(f"Processing {len(cohorts_to_process)} cohort/age_band combinations")
    print()
    
    # Process each cohort
    success_count = 0
    fail_count = 0
    
    for cohort, age_band in cohorts_to_process:
        if run_feature_importance_eda_for_cohort(cohort, age_band, script_dir):
            success_count += 1
        else:
            fail_count += 1
        # Clear memory after each age band when running multiple in sequence
        clear_age_band_memory()
    
    # Summary
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total: {len(cohorts_to_process)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print()
    
    if fail_count == 0:
        print("[OK] All Feature Importance EDA analyses completed successfully!")
        sys.exit(0)
    else:
        print("[ERROR] Some analyses failed. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
