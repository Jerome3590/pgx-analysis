#!/usr/bin/env python3
"""
Check S3 checkpoints and outputs for a cohort/age_band, then generate a command to resume from the first missing step.

Usage:
    python utility_scripts/check_and_resume_workflow.py <cohort> <age_band>
    
Example:
    python utility_scripts/check_and_resume_workflow.py opioid_ed 25-44
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, List, Tuple

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

# Step definitions with their S3 output paths and step numbers
# Ordered by execution sequence
# Note: Step 3 uses local time logs, not S3 checkpoints, so it's not included here
STEPS = [
    {
        "step_num": "4a",
        "step_name": "4a_model_data",
        "checkpoint_prefix": "pipeline_checkpoints/4a_model_data",
        "output_prefix": "gold/cohorts_model_data",
        "output_file": "model_events.parquet",
    },
    {
        "step_num": "4b",
        "step_name": "4b_dtw_filter",
        "checkpoint_prefix": "pipeline_checkpoints/4b_dtw_filter",
        "output_prefix": "gold/dtw_filter",
        "output_files": [
            "model_events_no_protocols.parquet",
            "protocol_summary_*.csv",
            "event_intervals_*.parquet",
        ],
    },
    {
        "step_num": "5",
        "step_name": "5_pgx_analysis",
        "checkpoint_prefix": "pipeline_checkpoints/5_pgx_analysis",
        "output_prefix": "gold/pgx_features",
        "output_files": [
            "pgx_added_features_*.csv",
            "*_drug_gene_mappings.csv",
            "*_allele_frequencies.csv",
        ],
    },
    {
        "step_num": "6",
        "step_name": "6_final_model",
        "checkpoint_prefix": "pipeline_checkpoints/6_final_model",
        "output_prefix": "gold/final_model",
        "output_files": [
            "*_best_xgboost_model.json",
            "*_best_catboost_model.cbm",
            "*_model_selection_metadata.json",
        ],
    },
    {
        "step_num": "7",
        "step_name": "7_shap_analysis",
        "checkpoint_prefix": "pipeline_checkpoints/7_shap_analysis",
        "output_prefix": "gold/shap_analysis",
        "output_files": [
            "*_shap_global_importance_xgboost.csv",
            "*_shap_sample_values_xgboost.parquet",
        ],
    },
    {
        "step_num": "8",
        "step_name": "8_ffa_analysis",
        "checkpoint_prefix": "pipeline_checkpoints/8_ffa_analysis",
        "output_prefix": "gold/ffa_analysis",
        "output_files": [
            "xgboost/axp_explanations.parquet",
            "xgboost/feature_importance_axp.parquet",
        ],
    },
]


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
        }
    except ClientError as e:
        if e.response['Error']['Code'] in ['404', 'NoSuchKey']:
            return {"exists": False}
        raise


def check_outputs_exist(cohort: str, age_band: str, step_config: Dict) -> bool:
    """Check if step outputs exist in S3."""
    age_band_fname = age_band.replace("-", "_")
    output_prefix = step_config['output_prefix']
    
    if "output_file" in step_config:
        # Single file
        if output_prefix == "gold/cohorts_model_data":
            key = f"{output_prefix}/cohort_name={cohort}/age_band={age_band}/{step_config['output_file']}"
        else:
            key = f"{output_prefix}/{step_config['output_file']}"
        return check_s3_object_exists(OUTPUT_BUCKET, key)
    elif "output_files" in step_config:
        # Multiple files (check at least one exists)
        # Build the search prefix based on step
        if output_prefix == "gold/pgx_features":
            search_prefix = f"{output_prefix}/{cohort}/{age_band}"
            search_prefix_legacy = f"gold/feature_engineering/7_pgx/{cohort}/{age_band}"
        elif output_prefix == "gold/ffa_analysis":
            search_prefix = f"{output_prefix}/{cohort}/{age_band}"
        elif output_prefix == "gold/shap_analysis":
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
                # List objects in the prefix
                prefixes_to_try = [search_prefix]
                if output_prefix == "gold/pgx_features":
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
                                    return True
                    except Exception:
                        pass
            else:
                # Try primary location first, then legacy for PGx
                prefixes_to_try = [search_prefix]
                if output_prefix == "gold/pgx_features":
                    prefixes_to_try.append(search_prefix_legacy)
                
                for prefix_to_check in prefixes_to_try:
                    key = f"{prefix_to_check}/{pattern}"
                    if check_s3_object_exists(OUTPUT_BUCKET, key):
                        return True
    
    return False


def check_step_status(cohort: str, age_band: str, step_config: Dict) -> Tuple[bool, bool, bool]:
    """
    Check step status.
    Returns: (checkpoint_exists, outputs_exist, is_complete)
    """
    checkpoint_info = get_checkpoint_info(cohort, age_band, step_config["step_name"])
    checkpoint_exists = checkpoint_info.get("exists", False) if checkpoint_info else False
    
    outputs_exist = check_outputs_exist(cohort, age_band, step_config)
    
    # Step is complete if both checkpoint and outputs exist
    is_complete = checkpoint_exists and outputs_exist
    
    return checkpoint_exists, outputs_exist, is_complete


def main():
    parser = argparse.ArgumentParser(
        description="Check S3 checkpoints and outputs, then generate resume command"
    )
    parser.add_argument("cohort", choices=["opioid_ed", "non_opioid_ed"], help="Cohort name")
    parser.add_argument("age_band", help="Age band (e.g., 13-24, 25-44)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed status")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"S3 Status Check for {args.cohort}/{args.age_band}")
    print("=" * 80)
    print()
    print("Note: Step 3 (Feature Importance) uses local time logs, not S3 checkpoints.")
    print("      It will be checked by the workflow script itself.")
    print()
    
    # Check each step
    completed_steps = []
    missing_steps = []
    partial_steps = []
    
    for step_config in STEPS:
        step_num = step_config["step_num"]
        step_name = step_config["step_name"]
        
        checkpoint_exists, outputs_exist, is_complete = check_step_status(
            args.cohort, args.age_band, step_config
        )
        
        if is_complete:
            completed_steps.append(step_num)
            status = "✓ COMPLETE"
        elif checkpoint_exists or outputs_exist:
            partial_steps.append((step_num, checkpoint_exists, outputs_exist))
            status = "⚠ PARTIAL"
        else:
            missing_steps.append(step_num)
            status = "✗ MISSING"
        
        if args.verbose:
            checkpoint_str = "✓" if checkpoint_exists else "✗"
            output_str = "✓" if outputs_exist else "✗"
            print(f"Step {step_num} ({step_name}): {status}")
            print(f"  Checkpoint: {checkpoint_str}  Outputs: {output_str}")
        else:
            print(f"Step {step_num}: {status}")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    if completed_steps:
        print(f"✓ Completed steps: {', '.join(completed_steps)}")
    
    if partial_steps:
        print(f"\n⚠ Partial steps (checkpoint or outputs missing):")
        for step_num, has_checkpoint, has_outputs in partial_steps:
            checkpoint_str = "checkpoint" if has_checkpoint else "no checkpoint"
            output_str = "outputs" if has_outputs else "no outputs"
            print(f"  Step {step_num}: {checkpoint_str}, {output_str}")
    
    if missing_steps:
        print(f"\n✗ Missing steps: {', '.join(missing_steps)}")
    
    print()
    print("=" * 80)
    print("RESUME COMMAND")
    print("=" * 80)
    print()
    
    # Determine first missing step
    if missing_steps:
        first_missing = missing_steps[0]
        # Skip all completed steps before the first missing step
        # Note: We don't skip Step 3 here as it uses local time logs, not S3 checkpoints
        # The workflow script will handle Step 3 automatically via time logs
        steps_to_skip = [s["step_num"] for s in STEPS if s["step_num"] < first_missing and s["step_num"] in completed_steps]
        
        if steps_to_skip:
            skip_arg = f"--skip-steps {','.join(steps_to_skip)}"
        else:
            skip_arg = ""
        
        print(f"First missing step: {first_missing}")
        print()
        print(f"Run this command to resume from Step {first_missing}:")
        print()
        if skip_arg:
            print(f"  bash utility_scripts/run_cohort_workflow.sh {args.cohort} {args.age_band} {skip_arg}")
        else:
            print(f"  bash utility_scripts/run_cohort_workflow.sh {args.cohort} {args.age_band}")
        print()
        print("Note: Step 3 will be automatically checked via local time logs.")
        print()
    elif partial_steps:
        # If there are partial steps, recommend checking them
        first_partial = partial_steps[0][0]
        steps_to_skip = [s["step_num"] for s in STEPS if s["step_num"] < first_partial]
        
        if steps_to_skip:
            skip_arg = f"--skip-steps {','.join(steps_to_skip)}"
        else:
            skip_arg = ""
        
        print("⚠ Some steps are partial (checkpoint or outputs missing).")
        print(f"First partial step: {first_partial}")
        print()
        print(f"Run this command to resume from Step {first_partial}:")
        print()
        if skip_arg:
            print(f"  bash utility_scripts/run_cohort_workflow.sh {args.cohort} {args.age_band} {skip_arg}")
        else:
            print(f"  bash utility_scripts/run_cohort_workflow.sh {args.cohort} {args.age_band}")
        print()
        print("Note: The workflow will re-run partial steps to ensure completeness.")
        print("      Step 3 will be automatically checked via local time logs.")
        print()
    else:
        print("✓ All steps are complete!")
        print()
        print("No resume needed. All checkpoints and outputs exist in S3.")


if __name__ == "__main__":
    main()
