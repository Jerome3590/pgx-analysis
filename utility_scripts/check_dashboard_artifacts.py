#!/usr/bin/env python3
"""
Check if dashboard artifacts exist in S3 for a given cohort and age band.

This script checks for all artifacts needed for the dashboard:
- SHAP analysis outputs (Step 7)
- FFA analysis outputs (Step 8): explanations, feature importance, causal results, interactions
- Model artifacts (Step 6)
"""

import sys
from pathlib import Path
from typing import Dict
from collections import defaultdict

# Add project root to path before importing project modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import boto3
except ImportError:
    boto3 = None  # type: ignore

from py_helpers.checkpoint_utils import check_s3_output_exists

# Define all cohorts and age bands
COHORTS = {
    "opioid_ed": ["13-24", "25-44", "45-54", "55-64"],
    "non_opioid_ed": ["65-74", "75-84", "85-94"],
}


def check_dashboard_artifacts(cohort: str, age_band: str, verbose: bool = False) -> Dict:
    """Check if all dashboard artifacts exist in S3."""
    age_band_fname = age_band.replace("-", "_")
    
    artifacts = {
        "SHAP Analysis (Step 7)": {
            "XGBoost Global Importance": f"s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/{cohort}_{age_band_fname}_shap_global_importance_xgboost.csv",
            "XGBoost Sample Values": f"s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/{cohort}_{age_band_fname}_shap_sample_values_xgboost.parquet",
            "CatBoost Global Importance": f"s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/{cohort}_{age_band_fname}_shap_global_importance_catboost.csv",
            "CatBoost Sample Values": f"s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/{cohort}_{age_band_fname}_shap_sample_values_catboost.parquet",
        },
        "FFA Analysis (Step 8)": {
            "AXP Explanations": f"s3://pgxdatalake/gold/ffa_analysis/{cohort}/{age_band}/xgboost/axp_explanations.parquet",
            "Feature Importance (AXP)": f"s3://pgxdatalake/gold/ffa_analysis/{cohort}/{age_band}/xgboost/feature_importance_axp.parquet",
            "Causal Importance": f"s3://pgxdatalake/gold/ffa_analysis/{cohort}/{age_band}/xgboost/causal_importance.parquet",
            "Interaction Analysis": f"s3://pgxdatalake/gold/ffa_analysis/{cohort}/{age_band}/xgboost/interaction_analysis.parquet",
        },
        "Model Artifacts (Step 6)": {
            "XGBoost Model JSON": f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band_fname}/*_best_xgboost_model.json",
            "CatBoost Model CBM": f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band_fname}/*_best_catboost_model.cbm",
            "Model Selection Metadata": f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band_fname}/*_model_selection_metadata.json",
        },
    }
    
    results = {}
    summary = defaultdict(lambda: {"found": 0, "missing": 0, "total": 0})
    
    print(f"\n{'='*80}")
    print("Checking Dashboard Artifacts in S3")
    print(f"Cohort: {cohort} | Age Band: {age_band}")
    print(f"{'='*80}\n")
    
    for category, files in artifacts.items():
        print(f"\n{category}:")
        print("-" * 80)
        category_results = {}
        
        for name, s3_path in files.items():
            # Handle wildcard patterns for model artifacts
            if "*" in s3_path:
                # For wildcards, we'll check if any file exists in that prefix
                # Extract prefix before wildcard
                prefix = s3_path.split("*")[0].replace("s3://pgxdatalake/", "")
                exists = check_s3_prefix_has_files("pgxdatalake", prefix)
            else:
                exists = check_s3_output_exists(s3_path)
            
            category_results[name] = {
                "exists": exists,
                "path": s3_path
            }
            
            status = "[EXISTS]" if exists else "[MISSING]"
            print(f"  {status:15} {name:35} {s3_path}")
            
            summary[category]["total"] += 1
            if exists:
                summary[category]["found"] += 1
            else:
                summary[category]["missing"] += 1
        
        results[category] = category_results
    
    # Print summary
    print(f"\n{'='*80}")
    print("Summary:")
    print(f"{'='*80}")
    
    total_found = 0
    total_missing = 0
    total_files = 0
    
    for category, stats in summary.items():
        total_found += stats["found"]
        total_missing += stats["missing"]
        total_files += stats["total"]
        
        completion = (stats["found"] / stats["total"] * 100) if stats["total"] > 0 else 0
        status_icon = "✅" if stats["missing"] == 0 else "⚠️" if stats["found"] > 0 else "❌"
        
        print(f"\n{status_icon} {category}:")
        print(f"    Found: {stats['found']}/{stats['total']} ({completion:.1f}%)")
        if stats["missing"] > 0:
            print(f"    Missing: {stats['missing']} file(s)")
    
    overall_completion = (total_found / total_files * 100) if total_files > 0 else 0
    print(f"\n{'='*80}")
    print(f"Overall: {total_found}/{total_files} files found ({overall_completion:.1f}%)")
    
    if total_missing == 0:
        print("✅ All dashboard artifacts are available!")
    elif total_found == 0:
        print("❌ No dashboard artifacts found. Pipeline may not have run yet.")
    else:
        print(f"⚠️  {total_missing} artifact(s) missing. Dashboard may have limited functionality.")
    
    print(f"{'='*80}\n")
    
    return results


def check_s3_prefix_has_files(bucket: str, prefix: str) -> bool:
    """Check if any files exist in an S3 prefix."""
    if boto3 is None:
        return False
    try:
        s3_client = boto3.client('s3')
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=1
        )
        return 'Contents' in response and len(response['Contents']) > 0
    except Exception:
        return False


def check_all_cohorts() -> None:
    """Check dashboard artifacts for all cohorts and age bands."""
    print(f"\n{'='*80}")
    print("Dashboard Artifacts Status - All Cohorts")
    print(f"{'='*80}\n")
    
    all_results = {}
    
    for cohort, age_bands in COHORTS.items():
        print(f"\n{'='*80}")
        print(f"Cohort: {cohort.upper()}")
        print(f"{'='*80}")
        
        cohort_results = {}
        for age_band in age_bands:
            print(f"\nAge Band: {age_band}")
            results = check_dashboard_artifacts(cohort, age_band, verbose=False)
            cohort_results[age_band] = results
        
        all_results[cohort] = cohort_results
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("Overall Summary Across All Cohorts")
    print(f"{'='*80}\n")
    
    total_cohorts = 0
    complete_cohorts = 0
    
    for cohort, cohort_results in all_results.items():
        for age_band, results in cohort_results.items():
            total_cohorts += 1
            # Count missing artifacts
            total_missing = 0
            for category_results in results.values():
                for file_result in category_results.values():
                    if not file_result["exists"]:
                        total_missing += 1
            
            if total_missing == 0:
                complete_cohorts += 1
                status = "✅"
            elif total_missing < 4:  # Some missing but not critical
                status = "⚠️"
            else:
                status = "❌"
            
            print(f"{status} {cohort:20} {age_band:10} - {total_missing} missing artifact(s)")
    
    print(f"\nComplete: {complete_cohorts}/{total_cohorts} cohort/age-band combinations")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check if dashboard artifacts exist in S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check specific cohort/age band
  python utility_scripts/check_dashboard_artifacts.py --cohort opioid_ed --age-band 13-24
  
  # Check all cohorts
  python utility_scripts/check_dashboard_artifacts.py --all-cohorts
  
  # Check with verbose output
  python utility_scripts/check_dashboard_artifacts.py --cohort opioid_ed --age-band 13-24 --verbose
        """
    )
    parser.add_argument("--cohort", help="Cohort name (e.g., opioid_ed)")
    parser.add_argument("--age-band", help="Age band (e.g., 13-24)")
    parser.add_argument("--all-cohorts", action="store_true", help="Check all cohorts and age bands")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.all_cohorts:
        check_all_cohorts()
    elif args.cohort and args.age_band:
        check_dashboard_artifacts(args.cohort, args.age_band, verbose=args.verbose)
    else:
        parser.print_help()
        sys.exit(1)

