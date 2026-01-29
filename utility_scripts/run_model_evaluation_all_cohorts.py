#!/usr/bin/env python3
"""
Run model evaluation for each cohort separately.

This script runs evaluation for each cohort/age_band combination individually
to manage memory and handle different models.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Cohorts and age bands
COHORTS = {
    'opioid_ed': ['13-24', '25-44', '45-54', '55-64'],
    'non_opioid_ed': ['65-74', '75-84', '85-94']
}

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = PROJECT_ROOT / "utility_scripts" / "evaluate_models_test_data.py"


def run_evaluation(
    cohort: str,
    age_band: str,
    model_type: str = 'both',
    n_shap_samples: int = 1000,
    profile: str = None
) -> bool:
    """Run evaluation for a single cohort/age_band."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {cohort} / {age_band}")
    print(f"{'='*80}\n")
    
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        '--cohort', cohort,
        '--age-band', age_band,
        '--model-type', model_type,
        '--n-shap-samples', str(n_shap_samples)
    ]
    
    if profile:
        cmd.extend(['--profile', profile])
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n[OK] Completed evaluation for {cohort}/{age_band}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed evaluation for {cohort}/{age_band}: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error for {cohort}/{age_band}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run model evaluation for each cohort separately"
    )
    parser.add_argument(
        "--cohort",
        type=str,
        choices=list(COHORTS.keys()),
        help="Process only this cohort (default: all)"
    )
    parser.add_argument(
        "--age-band",
        type=str,
        help="Process only this age band (requires --cohort)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=['xgboost', 'catboost', 'both'],
        default='both',
        help="Model type to evaluate (default: both)"
    )
    parser.add_argument(
        "--n-shap-samples",
        type=int,
        default=1000,
        help="Number of samples for SHAP analysis (default: 1000)"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="AWS profile (default: auto-detect)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between evaluations in seconds (default: 2.0)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Model Evaluation - Running Each Cohort Separately")
    print("="*80)
    print()
    print(f"Script: {SCRIPT_PATH}")
    print(f"SHAP Samples: {args.n_shap_samples}")
    print(f"Model Type: {args.model_type}")
    print()
    
    # Determine cohorts to process
    if args.cohort:
        if args.age_band:
            cohorts_to_process = [(args.cohort, args.age_band)]
        else:
            cohorts_to_process = [
                (args.cohort, age_band) 
                for age_band in COHORTS[args.cohort]
            ]
    else:
        # Process all cohorts
        cohorts_to_process = []
        for cohort, age_bands in COHORTS.items():
            for age_band in age_bands:
                cohorts_to_process.append((cohort, age_band))
    
    # Track results
    total = len(cohorts_to_process)
    success = 0
    failed = 0
    
    # Process each cohort
    for cohort, age_band in cohorts_to_process:
        if run_evaluation(
            cohort=cohort,
            age_band=age_band,
            model_type=args.model_type,
            n_shap_samples=args.n_shap_samples,
            profile=args.profile
        ):
            success += 1
        else:
            failed += 1
        
        # Delay between runs
        if args.delay > 0:
            import time
            time.sleep(args.delay)
    
    # Summary
    print()
    print("="*80)
    print("Summary")
    print("="*80)
    print(f"Total cohorts processed: {total}")
    print(f"Successful: {success}")
    print(f"Failed: {failed}")
    print()
    
    if failed == 0:
        print("✓ All evaluations completed successfully!")
        sys.exit(0)
    else:
        print("✗ Some evaluations failed. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
