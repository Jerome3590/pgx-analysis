#!/usr/bin/env python3
"""
Check if SHAP analysis files exist in S3 for a given cohort and age band.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.checkpoint_utils import check_s3_output_exists


def check_shap_files(cohort: str, age_band: str) -> None:
    """Check if SHAP files exist in S3."""
    age_band_fname = age_band.replace("-", "_")
    
    # XGBoost SHAP files
    xgb_global = f"s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/{cohort}_{age_band_fname}_shap_global_importance_xgboost.csv"
    xgb_sample = f"s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/{cohort}_{age_band_fname}_shap_sample_values_xgboost.parquet"
    
    # CatBoost SHAP files
    cb_global = f"s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/{cohort}_{age_band_fname}_shap_global_importance_catboost.csv"
    cb_sample = f"s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/{cohort}_{age_band_fname}_shap_sample_values_catboost.parquet"
    
    print(f"\n{'='*80}")
    print(f"Checking SHAP files in S3 for: {cohort} / {age_band}")
    print(f"{'='*80}\n")
    
    files_to_check = [
        ("XGBoost Global Importance", xgb_global),
        ("XGBoost Sample Values", xgb_sample),
        ("CatBoost Global Importance", cb_global),
        ("CatBoost Sample Values", cb_sample),
    ]
    
    results = {}
    for name, s3_path in files_to_check:
        exists = check_s3_output_exists(s3_path)
        results[name] = (exists, s3_path)
        status = "[EXISTS]" if exists else "[NOT FOUND]"
        print(f"{status:15} {name:30} {s3_path}")
    
    print(f"\n{'='*80}")
    print("Summary:")
    print(f"{'='*80}")
    existing = [name for name, (exists, _) in results.items() if exists]
    missing = [name for name, (exists, _) in results.items() if not exists]
    
    if existing:
        print(f"\n[FOUND] {len(existing)} file(s):")
        for name in existing:
            print(f"  - {name}")
    
    if missing:
        print(f"\n[MISSING] {len(missing)} file(s):")
        for name in missing:
            print(f"  - {name}")
    
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check if SHAP files exist in S3")
    parser.add_argument("--cohort", required=True, help="Cohort name (e.g., opioid_ed)")
    parser.add_argument("--age-band", required=True, help="Age band (e.g., 13-24)")
    
    args = parser.parse_args()
    check_shap_files(args.cohort, args.age_band)

