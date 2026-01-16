#!/usr/bin/env python3
"""
Filter and Refine Feature Importances

Combines BupaR post-target analysis and DTW trajectory analysis to filter
and refine aggregated feature importances from Step 3.

Outputs refined cohort_feature_importance files for Step 4a.
"""

import argparse
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import age_band_to_fname

try:
    from py_helpers.common_imports import s3_client, S3_BUCKET
except ImportError:
    import boto3
    s3_client = boto3.client("s3")
    S3_BUCKET = "pgxdatalake"

try:
    from py_helpers.checkpoint_utils import upload_file_to_s3
except ImportError:
    # Fallback upload function if checkpoint_utils not available
    def upload_file_to_s3(local_path: Path, s3_path: str, logger=None, check_exists: bool = True) -> bool:
        """Upload file to S3 using boto3."""
        if not local_path.exists():
            print(f"[ERROR] Local file does not exist: {local_path}")
            return False
        
        try:
            # Parse s3://bucket/key format
            if s3_path.startswith("s3://"):
                s3_path = s3_path[5:]
            parts = s3_path.split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            
            # Check if file already exists (idempotent)
            if check_exists:
                try:
                    s3_client.head_object(Bucket=bucket, Key=key)
                    print(f"[OK] File already exists in S3: s3://{bucket}/{key} (skipping upload)")
                    return True
                except s3_client.exceptions.ClientError as e:
                    if e.response["Error"]["Code"] not in ["404", "NoSuchKey"]:
                        raise
            
            # Upload file
            s3_client.upload_file(str(local_path), bucket, key)
            print(f"[OK] Uploaded to S3: s3://{bucket}/{key}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to upload {local_path} to s3://{bucket}/{key}: {e}")
            return False


def load_aggregated_feature_importance(cohort: str, age_band: str) -> pd.DataFrame:
    """Load aggregated feature importance from Step 3."""
    age_band_fname = age_band_to_fname(age_band)
    
    # Try multiple locations
    possible_paths = [
        PROJECT_ROOT / "3_feature_importance" / "outputs" / cohort / age_band / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv",
        PROJECT_ROOT / "3_feature_importance" / "from_s3" / "by_cohort" / cohort / age_band / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv",
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"Loading aggregated feature importance from: {path}")
            return pd.read_csv(path)
    
    raise FileNotFoundError(f"Could not find aggregated feature importance file for {cohort}/{age_band}")


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace spaces and special characters in column names with underscores.
    
    Args:
        df: DataFrame with potentially problematic column names
    
    Returns:
        DataFrame with sanitized column names
    """
    df = df.copy()
    # Replace spaces and special characters with underscores
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in df.columns]
    # Replace multiple consecutive underscores with single underscore
    df.columns = [re.sub(r'_+', '_', col) for col in df.columns]
    # Remove leading/trailing underscores
    df.columns = [col.strip('_') for col in df.columns]
    return df


def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace spaces and special characters in feature names with underscores.
    
    Args:
        df: DataFrame with 'feature' column containing feature names
    
    Returns:
        DataFrame with sanitized feature names
    """
    df = df.copy()
    if 'feature' in df.columns:
        # Replace spaces and special characters with underscores
        df['feature'] = df['feature'].astype(str).apply(
            lambda x: re.sub(r'[^a-zA-Z0-9_]', '_', x)
        )
        # Replace multiple consecutive underscores with single underscore
        df['feature'] = df['feature'].apply(lambda x: re.sub(r'_+', '_', x))
        # Remove leading/trailing underscores
        df['feature'] = df['feature'].str.strip('_')
    return df


def filter_and_refine_features(
    aggregated_fi: pd.DataFrame,
    bupar_results: pd.DataFrame,
    dtw_results: pd.DataFrame,
    filter_post_target: bool = True,
    filter_non_value_added: bool = True,
    min_importance_threshold: float = 0.0
) -> pd.DataFrame:
    """
    Filter and refine feature importances based on EDA results.
    
    Args:
        aggregated_fi: Aggregated feature importance DataFrame from Step 3
        bupar_results: BupaR post-target analysis results
        dtw_results: DTW trajectory analysis results
        filter_post_target: Whether to filter post-target leakage features
        filter_non_value_added: Whether to filter non-value-added features
        min_importance_threshold: Minimum importance threshold to keep
    
    Returns:
        Refined feature importance DataFrame
    """
    # Sanitize column names and feature names
    aggregated_fi = sanitize_column_names(aggregated_fi)
    aggregated_fi = sanitize_feature_names(aggregated_fi)
    
    refined_fi = aggregated_fi.copy()
    
    # Track filtering decisions
    filtering_summary = {
        'original_count': len(refined_fi),
        'filtered_by_post_target': 0,
        'filtered_by_non_value_added': 0,
        'filtered_by_threshold': 0,
        'final_count': 0
    }
    
    # Filter post-target leakage features
    if filter_post_target and not bupar_results.empty:
        post_target_features = set(
            bupar_results[bupar_results['is_post_target_leakage'] == 1]['feature'].tolist()
        )
        
        before_count = len(refined_fi)
        refined_fi = refined_fi[~refined_fi['feature'].isin(post_target_features)]
        filtering_summary['filtered_by_post_target'] = before_count - len(refined_fi)
        
        print(f"Filtered {filtering_summary['filtered_by_post_target']} post-target leakage features")
    
    # Filter non-value-added features
    if filter_non_value_added and not dtw_results.empty:
        non_value_added_features = set(
            dtw_results[dtw_results['is_non_value_added'] == 1]['feature'].tolist()
        )
        
        before_count = len(refined_fi)
        refined_fi = refined_fi[~refined_fi['feature'].isin(non_value_added_features)]
        filtering_summary['filtered_by_non_value_added'] = before_count - len(refined_fi)
        
        print(f"Filtered {filtering_summary['filtered_by_non_value_added']} non-value-added features")
    
    # Filter by minimum importance threshold
    if 'importance_scaled_by_model_sum' in refined_fi.columns:
        importance_col = 'importance_scaled_by_model_sum'
    elif 'importance_mean' in refined_fi.columns:
        importance_col = 'importance_mean'
    else:
        importance_col = refined_fi.columns[1]  # Use second column as fallback
    
    before_count = len(refined_fi)
    refined_fi = refined_fi[refined_fi[importance_col] >= min_importance_threshold]
    filtering_summary['filtered_by_threshold'] = before_count - len(refined_fi)
    
    filtering_summary['final_count'] = len(refined_fi)
    
    # Sort by importance
    refined_fi = refined_fi.sort_values(importance_col, ascending=False).reset_index(drop=True)
    
    return refined_fi, filtering_summary


def main():
    parser = argparse.ArgumentParser(
        description="Filter and refine feature importances based on EDA"
    )
    parser.add_argument("--cohort", required=True, help="Cohort name")
    parser.add_argument("--age-band", required=True, help="Age band")
    parser.add_argument(
        "--bupar-results",
        type=str,
        default=None,
        help="Path to BupaR results CSV (default: auto-detect)"
    )
    parser.add_argument(
        "--dtw-results",
        type=str,
        default=None,
        help="Path to DTW results CSV (default: auto-detect)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: 3b_feature_importance_eda/outputs)"
    )
    parser.add_argument(
        "--min-importance",
        type=float,
        default=0.0,
        help="Minimum importance threshold (default: 0.0)"
    )
    parser.add_argument(
        "--no-filter-post-target",
        action="store_true",
        help="Don't filter post-target leakage features"
    )
    parser.add_argument(
        "--no-filter-non-value-added",
        action="store_true",
        help="Don't filter non-value-added features"
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "3b_feature_importance_eda" / "outputs" / args.cohort / age_band_to_fname(args.age_band)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    age_band_fname = age_band_to_fname(args.age_band)
    
    # Load aggregated feature importance
    print("=" * 80)
    print(f"Filtering and Refining Features: {args.cohort} / {args.age_band}")
    print("=" * 80)
    
    aggregated_fi = load_aggregated_feature_importance(args.cohort, args.age_band)
    print(f"Loaded {len(aggregated_fi)} features from aggregated importance")
    
    # Load BupaR results
    if args.bupar_results:
        bupar_path = Path(args.bupar_results)
    else:
        bupar_path = output_dir / f"{args.cohort}_{age_band_fname}_bupar_post_target_analysis.csv"
    
    bupar_results = pd.DataFrame()
    if bupar_path.exists():
        print(f"Loading BupaR results from: {bupar_path}")
        bupar_results = pd.read_csv(bupar_path)
    else:
        print(f"[WARN] BupaR results not found: {bupar_path}")
    
    # Load DTW results
    if args.dtw_results:
        dtw_path = Path(args.dtw_results)
    else:
        dtw_path = output_dir / f"{args.cohort}_{age_band_fname}_dtw_trajectory_analysis.csv"
    
    dtw_results = pd.DataFrame()
    if dtw_path.exists():
        print(f"Loading DTW results from: {dtw_path}")
        dtw_results = pd.read_csv(dtw_path)
    else:
        print(f"[WARN] DTW results not found: {dtw_path}")
    
    # Filter and refine
    refined_fi, filtering_summary = filter_and_refine_features(
        aggregated_fi=aggregated_fi,
        bupar_results=bupar_results,
        dtw_results=dtw_results,
        filter_post_target=not args.no_filter_post_target,
        filter_non_value_added=not args.no_filter_non_value_added,
        min_importance_threshold=args.min_importance
    )
    
    # Save refined feature importance
    output_path = output_dir / f"{args.cohort}_{age_band_fname}_cohort_feature_importance.csv"
    refined_fi.to_csv(output_path, index=False)
    print(f"\nSaved refined feature importance to: {output_path}")
    print(f"Features: {len(refined_fi)} (down from {filtering_summary['original_count']})")
    
    # Upload to S3
    s3_path = f"s3://{S3_BUCKET}/gold/feature_importance/{args.cohort}/{args.age_band}/{args.cohort}_{age_band_fname}_cohort_feature_importance.csv"
    if upload_file_to_s3(output_path, s3_path, check_exists=True):
        print(f"[OK] Uploaded cohort_feature_importance to S3")
    else:
        print(f"[ERROR] Failed to upload cohort_feature_importance to S3")
        sys.exit(1)
    
    # Save filtering summary
    summary_path = output_dir / f"{args.cohort}_{age_band_fname}_feature_filtering_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(filtering_summary, f, indent=2)
    print(f"Saved filtering summary to: {summary_path}")
    
    # Upload summary to S3
    summary_s3_path = f"s3://{S3_BUCKET}/gold/feature_importance/{args.cohort}/{args.age_band}/{args.cohort}_{age_band_fname}_feature_filtering_summary.json"
    if upload_file_to_s3(summary_path, summary_s3_path, check_exists=True):
        print(f"[OK] Uploaded filtering summary to S3")
    else:
        print(f"[WARN] Failed to upload filtering summary to S3 (non-critical)")
    
    # Print summary
    print("\nFiltering Summary:")
    print(f"  Original features: {filtering_summary['original_count']}")
    print(f"  Filtered by post-target: {filtering_summary['filtered_by_post_target']}")
    print(f"  Filtered by non-value-added: {filtering_summary['filtered_by_non_value_added']}")
    print(f"  Filtered by threshold: {filtering_summary['filtered_by_threshold']}")
    print(f"  Final features: {filtering_summary['final_count']}")
    
    print("\nTop 10 refined features:")
    print(refined_fi.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
