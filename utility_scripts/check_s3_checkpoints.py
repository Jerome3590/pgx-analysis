#!/usr/bin/env python3
"""
Check S3 checkpoints and outputs status for all pipeline steps.

This script checks:
1. Checkpoint metadata in s3://pgx-repository/pipeline_checkpoints/
2. Output files in s3://pgxdatalake/gold/
3. Provides a summary status for each cohort/age_band combination
"""

import sys
import json
from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("ERROR: boto3 not installed. Install with: pip install boto3")
    sys.exit(1)

# S3 clients
s3_client = boto3.client('s3')
CHECKPOINT_BUCKET = "pgx-repository"
OUTPUT_BUCKET = "pgxdatalake"

# Define all cohorts and age bands
COHORTS = {
    "opioid_ed": ["13-24", "25-44", "45-54", "55-64"],
    "non_opioid_ed": ["65-74", "75-84", "85-94"],
}

# Step definitions with their S3 output paths
STEPS = {
    "4a_model_data": {
        "checkpoint_prefix": "pipeline_checkpoints/4a_model_data",
        "output_prefix": "gold/cohorts_model_data",
        "output_file": "model_events.parquet",
    },
    "4b_dtw_filter": {
        "checkpoint_prefix": "pipeline_checkpoints/4b_dtw_filter",
        "output_prefix": "gold/dtw_filter",
        "output_files": [
            "model_events_no_protocols.parquet",
            "protocol_summary_*.csv",
            "event_intervals_*.parquet",
        ],
    },
    "5_pgx_analysis": {
        "checkpoint_prefix": "pipeline_checkpoints/5_pgx_analysis",
        "output_prefix": "gold/pgx_features",  # Primary location (also checks legacy: gold/feature_engineering/7_pgx)
        "output_files": [
            "pgx_added_features_*.csv",
            "*_drug_gene_mappings.csv",
            "*_allele_frequencies.csv",
        ],
    },
    "6_final_model": {
        "checkpoint_prefix": "pipeline_checkpoints/6_final_model",
        "output_prefix": "gold/final_model",
        "output_files": [
            "*_best_xgboost_model.json",
            "*_best_catboost_model.cbm",
            "*_model_selection_metadata.json",
        ],
    },
    "7_ffa_analysis": {
        "checkpoint_prefix": "pipeline_checkpoints/7_ffa_analysis",
        "output_prefix": "gold/ffa_analysis",
        "output_files": [
            "xgboost/axp_explanations.csv",
            "xgboost/feature_importance_axp.csv",
        ],
    },
    "8_shap_analysis": {
        "checkpoint_prefix": "pipeline_checkpoints/8_shap_analysis",
        "output_prefix": "gold/shap_analysis",
        "output_files": [
            "*_shap_global_importance_xgboost.csv",
            "*_shap_sample_values_xgboost.parquet",
        ],
    },
    "9_combined_shap_ffa": {
        "checkpoint_prefix": "pipeline_checkpoints/9_combined_shap_ffa",
        "output_prefix": "gold/combined_analysis",
        "output_files": [
            "*_combined_shap_ffa_importance.csv",
            "*_consensus_features.json",
        ],
    },
}


def check_s3_object_exists(bucket: str, key: str) -> bool:
    """Check if an S3 object exists."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] in ['404', 'NoSuchKey']:
            return False
        raise


def get_checkpoint_info(cohort: str, age_band: str, step_name: str) -> Optional[Dict]:
    """Get checkpoint information for a specific step."""
    age_band_fname = age_band.replace("-", "_")
    checkpoint_key = (
        f"pipeline_checkpoints/{step_name}/{cohort}/{age_band_fname}/checkpoint.json"
    )
    
    try:
        obj = s3_client.get_object(Bucket=CHECKPOINT_BUCKET, Key=checkpoint_key)
        checkpoint_data = json.loads(obj['Body'].read().decode('utf-8'))
        return {
            "exists": True,
            "completed_at": checkpoint_data.get("completed_at"),
            "status": checkpoint_data.get("status"),
            "metadata": checkpoint_data.get("metadata", {}),
            "output_paths": checkpoint_data.get("output_paths", []),
        }
    except ClientError as e:
        if e.response['Error']['Code'] in ['404', 'NoSuchKey']:
            return {"exists": False}
        raise


def check_outputs_exist(cohort: str, age_band: str, step_config: Dict) -> Dict:
    """Check if step outputs exist in S3."""
    age_band_fname = age_band.replace("-", "_")
    output_prefix = step_config['output_prefix']
    
    results = {}
    
    if "output_file" in step_config:
        # Single file
        if output_prefix == "gold/cohorts_model_data":
            key = f"{output_prefix}/cohort_name={cohort}/age_band={age_band}/{step_config['output_file']}"
        else:
            key = f"{output_prefix}/{step_config['output_file']}"
        results[step_config['output_file']] = check_s3_object_exists(OUTPUT_BUCKET, key)
    elif "output_files" in step_config:
        # Multiple files (check at least one exists)
        found_any = False
        
        # Build the search prefix based on step
        if output_prefix == "gold/pgx_features":
            # Primary location for PGx features (also check legacy location)
            search_prefix = f"{output_prefix}/{cohort}/{age_band}"
            search_prefix_legacy = f"gold/feature_engineering/7_pgx/{cohort}/{age_band}"
        elif output_prefix == "gold/ffa_analysis":
            search_prefix = f"{output_prefix}/{cohort}/{age_band}"
        elif output_prefix == "gold/shap_analysis":
            search_prefix = f"{output_prefix}/{cohort}/{age_band}"
        elif output_prefix == "gold/combined_analysis":
            search_prefix = f"{output_prefix}/{cohort}/{age_band}"
        elif output_prefix == "gold/final_model":
            search_prefix = f"{output_prefix}/{cohort}/{age_band_fname}"
        elif output_prefix == "gold/dtw_filter":
            search_prefix = f"{output_prefix}/{cohort}/{age_band_fname}"
        else:
            search_prefix = f"{output_prefix}/{cohort}/{age_band}"
        
        for pattern in step_config["output_files"]:
            # For patterns with wildcards, list objects
            if "*" in pattern:
                # List objects in the prefix (try primary, then legacy for PGx)
                prefixes_to_try = [search_prefix]
                if output_prefix == "gold/pgx_features" and "search_prefix_legacy" in locals():
                    prefixes_to_try.append(search_prefix_legacy)
                
                for prefix_to_check in prefixes_to_try:
                    try:
                        response = s3_client.list_objects_v2(
                            Bucket=OUTPUT_BUCKET,
                            Prefix=prefix_to_check,
                            MaxKeys=100
                        )
                        if 'Contents' in response:
                            # Check if any file matches the pattern
                            import fnmatch
                            for obj in response['Contents']:
                                obj_key = obj['Key']
                                obj_name = obj_key.split('/')[-1]
                                if fnmatch.fnmatch(obj_name, pattern):
                                    found_any = True
                                    results[pattern] = True
                                    break
                            if found_any:
                                break
                    except Exception:
                        pass
            else:
                # Try primary location first, then legacy for PGx
                prefixes_to_try = [search_prefix]
                if output_prefix == "gold/pgx_features" and "search_prefix_legacy" in locals():
                    prefixes_to_try.append(search_prefix_legacy)
                
                for prefix_to_check in prefixes_to_try:
                    key = f"{prefix_to_check}/{pattern}"
                    if check_s3_object_exists(OUTPUT_BUCKET, key):
                        found_any = True
                        results[pattern] = True
                        break
                if not found_any:
                    results[pattern] = False
        results["_any_exists"] = found_any
    
    return results


def main():
    print("=" * 80)
    print("S3 Checkpoint and Output Status")
    print("=" * 80)
    print()
    
    # Collect status for all cohort/age_band combinations
    status_summary = defaultdict(lambda: defaultdict(dict))
    
    for cohort, age_bands in COHORTS.items():
        for age_band in age_bands:
            print(f"Checking {cohort}/{age_band}...")
            
            for step_name, step_config in STEPS.items():
                # Check checkpoint
                checkpoint_info = get_checkpoint_info(cohort, age_band, step_name)
                
                # Check outputs
                output_status = check_outputs_exist(cohort, age_band, step_config)
                
                status_summary[cohort][age_band][step_name] = {
                    "checkpoint": checkpoint_info,
                    "outputs": output_status,
                }
    
    # Print summary table
    print("\n" + "=" * 80)
    print("STATUS SUMMARY")
    print("=" * 80)
    print()
    
    # Header
    step_names = list(STEPS.keys())
    header = f"{'Cohort/Age Band':<25} " + " ".join([f"{s:<20}" for s in step_names])
    print(header)
    print("-" * len(header))
    
    # Status for each cohort/age_band
    for cohort, age_bands in COHORTS.items():
        for age_band in age_bands:
            combo = f"{cohort}/{age_band}"
            row = f"{combo:<25} "
            
            for step_name in step_names:
                status = status_summary[cohort][age_band][step_name]
                checkpoint_exists = status["checkpoint"].get("exists", False)
                outputs_exist = status["outputs"].get("_any_exists", False) or any(status["outputs"].values())
                
                if checkpoint_exists and outputs_exist:
                    symbol = "[OK]"
                elif checkpoint_exists or outputs_exist:
                    symbol = "[PARTIAL]"
                else:
                    symbol = "[MISSING]"
                
                row += f"{symbol:<20} "
            
            print(row)
    
    # Detailed status
    print("\n" + "=" * 80)
    print("DETAILED STATUS")
    print("=" * 80)
    print()
    
    for cohort, age_bands in COHORTS.items():
        for age_band in age_bands:
            print(f"\n{cohort}/{age_band}:")
            print("-" * 60)  # noqa: F541
            
            for step_name in step_names:
                status = status_summary[cohort][age_band][step_name]
                checkpoint = status["checkpoint"]
                outputs = status["outputs"]
                
                print(f"  {step_name}:")
                
                # Checkpoint status
                if checkpoint.get("exists"):
                    completed = checkpoint.get("completed_at", "unknown")
                    print(f"    Checkpoint: [OK] ({completed})")
                else:
                    print("    Checkpoint: [MISSING]")
                
                # Output status
                if "output_file" in STEPS[step_name]:
                    file_exists = outputs.get(STEPS[step_name]["output_file"], False)
                    print(f"    Output: {'[OK]' if file_exists else '[MISSING]'} ({STEPS[step_name]['output_file']})")
                else:
                    any_exists = outputs.get("_any_exists", False) or any(
                        v for k, v in outputs.items() if k != "_any_exists"
                    )
                    print(f"    Outputs: {'[OK]' if any_exists else '[MISSING]'}")
                    if any_exists:
                        for file, exists in outputs.items():
                            if file != "_any_exists":
                                print(f"      - {file}: {'[OK]' if exists else '[MISSING]'}")
    
    # Summary counts
    print("\n" + "=" * 80)
    print("SUMMARY COUNTS")
    print("=" * 80)
    print()
    
    total_combos = sum(len(age_bands) for age_bands in COHORTS.values())
    
    for step_name in step_names:
        checkpoint_count = 0
        output_count = 0
        both_count = 0
        
        for cohort, age_bands in COHORTS.items():
            for age_band in age_bands:
                status = status_summary[cohort][age_band][step_name]
                checkpoint_exists = status["checkpoint"].get("exists", False)
                outputs_exist = status["outputs"].get("_any_exists", False) or any(status["outputs"].values())
                
                if checkpoint_exists:
                    checkpoint_count += 1
                if outputs_exist:
                    output_count += 1
                if checkpoint_exists and outputs_exist:
                    both_count += 1
        
        print(f"{step_name}:")
        print(f"  Checkpoints: {checkpoint_count}/{total_combos}")
        print(f"  Outputs: {output_count}/{total_combos}")
        print(f"  Complete (both): {both_count}/{total_combos}")
        print()


if __name__ == "__main__":
    main()

