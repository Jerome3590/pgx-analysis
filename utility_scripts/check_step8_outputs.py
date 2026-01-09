#!/usr/bin/env python3
"""
Check Step 8 (FFA Analysis) outputs: local files, S3 checkpoints, and S3 outputs.

This script provides a comprehensive view of all Step 8 artifacts.
"""

import sys
import io
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd

# Set UTF-8 encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    print("[WARNING] boto3 not available. S3 checks will be skipped.")

# Import checkpoint utilities
try:
    from py_helpers.checkpoint_utils import check_step_checkpoint_exists, get_s3_client
    CHECKPOINT_UTILS_AVAILABLE = True
except ImportError:
    CHECKPOINT_UTILS_AVAILABLE = False
    print("[WARNING] checkpoint_utils not available. Checkpoint checks will be skipped.")


def check_local_files(cohort: str, age_band: str, model_type: str = "xgboost") -> Dict[str, bool]:
    """Check for local Step 8 output files."""
    age_band_fname = age_band.replace("-", "_")
    output_dir = PROJECT_ROOT / "8_ffa_analysis" / "outputs" / cohort / age_band_fname / model_type
    
    files_to_check = {
        "axp_explanations.parquet": output_dir / "axp_explanations.parquet",
        "feature_importance_axp.parquet": output_dir / "feature_importance_axp.parquet",
        "causal_importance.parquet": output_dir / "causal_importance.parquet",
        "interaction_analysis.parquet": output_dir / "interaction_analysis.parquet",
        "analysis_summary.json": output_dir / "analysis_summary.json",
    }
    
    results = {}
    for file_name, file_path in files_to_check.items():
        exists = file_path.exists()
        results[file_name] = exists
        if exists:
            try:
                size_mb = file_path.stat().st_size / 1024 / 1024
                results[f"{file_name}_size_mb"] = size_mb
            except Exception:
                results[f"{file_name}_size_mb"] = None
    
    return results


def check_s3_checkpoint(cohort: str, age_band: str) -> Tuple[bool, Dict]:
    """Check for S3 checkpoint."""
    if not CHECKPOINT_UTILS_AVAILABLE:
        return False, {}
    
    try:
        exists = check_step_checkpoint_exists("8_ffa_analysis", cohort, age_band, None)
        checkpoint_info = {}
        
        if exists:
            # Try to get checkpoint details
            try:
                s3_client = get_s3_client()
                checkpoint_key = f"pipeline_checkpoints/8_ffa_analysis/{cohort}/{age_band.replace('-', '_')}/checkpoint.json"
                bucket = "pgx-repository"
                
                response = s3_client.get_object(Bucket=bucket, Key=checkpoint_key)
                import json
                checkpoint_data = json.loads(response['Body'].read())
                checkpoint_info = checkpoint_data
            except Exception as e:
                checkpoint_info = {"error": str(e)}
        
        return exists, checkpoint_info
    except Exception as e:
        return False, {"error": str(e)}


def check_s3_outputs(cohort: str, age_band: str, model_type: str = "xgboost") -> Dict[str, bool]:
    """Check for S3 output files."""
    if not BOTO3_AVAILABLE:
        return {}
    
    s3_base = f"s3://pgxdatalake/gold/ffa_analysis/{cohort}/{age_band}/{model_type}"
    
    files_to_check = {
        "axp_explanations.parquet": f"{s3_base}/axp_explanations.parquet",
        "feature_importance_axp.parquet": f"{s3_base}/feature_importance_axp.parquet",
        "causal_importance.parquet": f"{s3_base}/causal_importance.parquet",
        "interaction_analysis.parquet": f"{s3_base}/interaction_analysis.parquet",
    }
    
    results = {}
    
    try:
        s3_client = get_s3_client()
        bucket = "pgxdatalake"
        
        for file_name, s3_path in files_to_check.items():
            # Parse S3 path
            s3_key = s3_path.replace(f"s3://{bucket}/", "")
            
            try:
                response = s3_client.head_object(Bucket=bucket, Key=s3_key)
                exists = True
                size_mb = response.get('ContentLength', 0) / 1024 / 1024
                results[file_name] = True
                results[f"{file_name}_size_mb"] = size_mb
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    results[file_name] = False
                else:
                    results[file_name] = None
                    results[f"{file_name}_error"] = str(e)
    except Exception as e:
        results["error"] = str(e)
    
    return results


def print_summary(cohort: str, age_band: str, model_type: str = "xgboost"):
    """Print comprehensive summary of Step 8 outputs."""
    age_band_fname = age_band.replace("-", "_")
    
    print("=" * 80)
    print(f"Step 8 (FFA Analysis) Output Status")
    print("=" * 80)
    print(f"Cohort: {cohort}")
    print(f"Age Band: {age_band}")
    print(f"Model Type: {model_type}")
    print("=" * 80)
    print()
    
    # Local files
    print("LOCAL FILES")
    print("-" * 80)
    local_files = check_local_files(cohort, age_band, model_type)
    output_dir = PROJECT_ROOT / "8_ffa_analysis" / "outputs" / cohort / age_band_fname / model_type
    print(f"Directory: {output_dir}")
    print()
    
    for file_name in ["axp_explanations.parquet", "feature_importance_axp.parquet", 
                      "causal_importance.parquet", "interaction_analysis.parquet", 
                      "analysis_summary.json"]:
        exists = local_files.get(file_name, False)
        size_mb = local_files.get(f"{file_name}_size_mb")
        status = "✓ EXISTS" if exists else "✗ MISSING"
        size_str = f" ({size_mb:.2f} MB)" if size_mb else ""
        print(f"  {status:12} {file_name}{size_str}")
    
    print()
    
    # S3 Checkpoint
    print("S3 CHECKPOINT")
    print("-" * 80)
    checkpoint_exists, checkpoint_info = check_s3_checkpoint(cohort, age_band)
    checkpoint_key = f"s3://pgx-repository/pipeline_checkpoints/8_ffa_analysis/{cohort}/{age_band_fname}/checkpoint.json"
    print(f"Checkpoint: {checkpoint_key}")
    print(f"Status: {'✓ EXISTS' if checkpoint_exists else '✗ MISSING'}")
    
    if checkpoint_exists and checkpoint_info:
        if "error" not in checkpoint_info:
            print(f"  Features analyzed: {checkpoint_info.get('features_analyzed', 'N/A')}")
            print(f"  Timestamp: {checkpoint_info.get('timestamp', 'N/A')}")
        else:
            print(f"  Error reading checkpoint: {checkpoint_info['error']}")
    print()
    
    # S3 Outputs
    print("S3 OUTPUT FILES")
    print("-" * 80)
    s3_outputs = check_s3_outputs(cohort, age_band, model_type)
    s3_base = f"s3://pgxdatalake/gold/ffa_analysis/{cohort}/{age_band}/{model_type}"
    print(f"Base path: {s3_base}")
    print()
    
    for file_name in ["axp_explanations.parquet", "feature_importance_axp.parquet", 
                      "causal_importance.parquet", "interaction_analysis.parquet"]:
        exists = s3_outputs.get(file_name, False)
        size_mb = s3_outputs.get(f"{file_name}_size_mb")
        status = "✓ EXISTS" if exists else "✗ MISSING"
        size_str = f" ({size_mb:.2f} MB)" if size_mb else ""
        print(f"  {status:12} {file_name}{size_str}")
    
    if "error" in s3_outputs:
        print(f"\n  Error: {s3_outputs['error']}")
    
    print()
    
    # Summary statistics
    if local_files.get("causal_importance.parquet"):
        try:
            causal_path = PROJECT_ROOT / "8_ffa_analysis" / "outputs" / cohort / age_band_fname / model_type / "causal_importance.parquet"
            df = pd.read_parquet(causal_path)
            
            print("CAUSAL IMPORTANCE SUMMARY")
            print("-" * 80)
            print(f"Total features analyzed: {len(df)}")
            print(f"Binary features: {df['is_binary'].sum() if 'is_binary' in df.columns else 'N/A'}")
            print(f"Features with causal_importance > 0: {(df['causal_importance'] > 0).sum()}")
            print(f"Binary features with causal_importance > 0: {((df['is_binary'] == True) & (df['causal_importance'] > 0)).sum() if 'is_binary' in df.columns else 'N/A'}")
            print()
            print("Top 10 causal importance features:")
            top_10 = df.nlargest(10, 'causal_importance')[['feature', 'causal_importance', 'is_binary']]
            for idx, row in top_10.iterrows():
                binary_str = " (binary)" if row.get('is_binary', False) else ""
                print(f"  {row['feature']:<50} {row['causal_importance']:>10.6f}{binary_str}")
        except Exception as e:
            print(f"Could not load causal importance summary: {e}")
    
    print("=" * 80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check Step 8 (FFA Analysis) outputs")
    parser.add_argument("--cohort-name", type=str, default="opioid_ed", help="Cohort name")
    parser.add_argument("--age-band", type=str, default="13-24", help="Age band")
    parser.add_argument("--model-type", type=str, default="xgboost", help="Model type")
    parser.add_argument("--all-cohorts", action="store_true", help="Check all cohorts")
    
    args = parser.parse_args()
    
    if args.all_cohorts:
        cohorts = [
            ("opioid_ed", "13-24"),
            ("opioid_ed", "25-44"),
            ("opioid_ed", "45-54"),
            ("opioid_ed", "55-64"),
            ("non_opioid_ed", "65-74"),
            ("non_opioid_ed", "75-84"),
            ("non_opioid_ed", "85-94"),
        ]
        
        for cohort, age_band in cohorts:
            print_summary(cohort, age_band, args.model_type)
            print("\n" + "=" * 80 + "\n")
    else:
        print_summary(args.cohort_name, args.age_band, args.model_type)


if __name__ == "__main__":
    main()
