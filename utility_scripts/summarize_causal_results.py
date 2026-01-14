#!/usr/bin/env python3
"""
Summarize causal analysis results for each completed cohort.

Reads causal_importance.parquet files from S3 and provides summary statistics
including top features, distribution of causal importance scores, and key insights.
"""

import sys
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("ERROR: boto3 not installed. Install with: pip install boto3")
    sys.exit(1)

# S3 client
s3_client = boto3.client('s3')
OUTPUT_BUCKET = "pgxdatalake"

# Define all cohorts and age bands
COHORTS = {
    "opioid_ed": ["13-24", "25-44", "45-54", "55-64"],
    "non_opioid_ed": ["65-74", "75-84", "85-94"],
}


def download_causal_results(cohort: str, age_band: str) -> Optional[pd.DataFrame]:
    """Download causal importance results from S3."""
    age_band_fname = age_band.replace("-", "_")
    s3_key = f"gold/ffa_analysis/{cohort}/{age_band}/xgboost/causal_importance.parquet"
    
    try:
        # Download to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp_file:
            tmp_path = tmp_file.name
        
        s3_client.download_file(OUTPUT_BUCKET, s3_key, tmp_path)
        df = pd.read_parquet(tmp_path)
        
        # Clean up
        Path(tmp_path).unlink()
        
        return df
    except ClientError as e:
        if e.response['Error']['Code'] in ['404', 'NoSuchKey']:
            return None
        raise
    except Exception as e:
        print(f"  Error downloading {cohort}/{age_band}: {e}")
        return None


def summarize_causal_results(df: pd.DataFrame, cohort: str, age_band: str) -> Dict:
    """Summarize causal analysis results."""
    if df.empty:
        return {
            "cohort": cohort,
            "age_band": age_band,
            "status": "empty",
            "total_features": 0,
        }
    
    # Basic statistics
    total_features = len(df)
    features_with_effect = len(df[df['causal_importance'] > 0])
    features_with_strong_effect = len(df[df['causal_importance'] > 0.1])
    features_with_very_strong_effect = len(df[df['causal_importance'] > 0.5])
    
    # Top features
    top_features = df.nlargest(10, 'causal_importance')[
        ['feature', 'causal_importance', 'is_binary', 'intervention']
    ].to_dict('records')
    
    # Statistics
    mean_importance = df['causal_importance'].mean()
    median_importance = df['causal_importance'].median()
    max_importance = df['causal_importance'].max()
    std_importance = df['causal_importance'].std()
    
    # Binary vs continuous
    binary_features = len(df[df['is_binary'] == True]) if 'is_binary' in df.columns else 0
    continuous_features = total_features - binary_features
    
    # Intervention types
    intervention_counts = {}
    if 'intervention' in df.columns:
        intervention_counts = df['intervention'].value_counts().to_dict()
    
    return {
        "cohort": cohort,
        "age_band": age_band,
        "status": "complete",
        "total_features": total_features,
        "features_with_effect": features_with_effect,
        "features_with_strong_effect": features_with_strong_effect,
        "features_with_very_strong_effect": features_with_very_strong_effect,
        "mean_importance": mean_importance,
        "median_importance": median_importance,
        "max_importance": max_importance,
        "std_importance": std_importance,
        "binary_features": binary_features,
        "continuous_features": continuous_features,
        "top_features": top_features,
        "intervention_counts": intervention_counts,
    }


def print_summary(summary: Dict):
    """Print formatted summary."""
    print(f"\n{'='*80}")
    print(f"Causal Analysis Summary: {summary['cohort'].upper()} / {summary['age_band']}")
    print(f"{'='*80}")
    
    if summary['status'] == 'empty':
        print("[WARNING] No causal results found or empty dataset")
        return
    
    # Overview
    print(f"\nOverview:")
    print(f"  Total Features Analyzed: {summary['total_features']}")
    print(f"  Features with Causal Effect (>0): {summary['features_with_effect']} ({summary['features_with_effect']/summary['total_features']*100:.1f}%)")
    print(f"  Features with Strong Effect (>0.1): {summary['features_with_strong_effect']} ({summary['features_with_strong_effect']/summary['total_features']*100:.1f}%)")
    print(f"  Features with Very Strong Effect (>0.5): {summary['features_with_very_strong_effect']} ({summary['features_with_very_strong_effect']/summary['total_features']*100:.1f}%)")
    
    # Statistics
    print(f"\nCausal Importance Statistics:")
    print(f"  Mean: {summary['mean_importance']:.4f}")
    print(f"  Median: {summary['median_importance']:.4f}")
    print(f"  Max: {summary['max_importance']:.4f}")
    print(f"  Std Dev: {summary['std_importance']:.4f}")
    
    # Feature types
    print(f"\nFeature Types:")
    print(f"  Binary Features: {summary['binary_features']} ({summary['binary_features']/summary['total_features']*100:.1f}%)")
    print(f"  Continuous Features: {summary['continuous_features']} ({summary['continuous_features']/summary['total_features']*100:.1f}%)")
    
    # Top features
    print(f"\nTop 10 Features by Causal Importance:")
    print(f"  {'Rank':<6} {'Feature':<40} {'Importance':<12} {'Type':<10} {'Intervention':<20}")
    print(f"  {'-'*6} {'-'*40} {'-'*12} {'-'*10} {'-'*20}")
    for idx, feat in enumerate(summary['top_features'], 1):
        feat_name = feat['feature'][:38] + ".." if len(feat['feature']) > 40 else feat['feature']
        feat_type = "Binary" if feat.get('is_binary', False) else "Continuous"
        intervention = str(feat.get('intervention', 'N/A'))[:18] + ".." if len(str(feat.get('intervention', 'N/A'))) > 20 else str(feat.get('intervention', 'N/A'))
        print(f"  {idx:<6} {feat_name:<40} {feat['causal_importance']:<12.4f} {feat_type:<10} {intervention:<20}")
    
    # Intervention types
    if summary['intervention_counts']:
        print(f"\nIntervention Types:")
        for intervention, count in summary['intervention_counts'].items():
            print(f"  {intervention}: {count}")


def summarize_all_cohorts():
    """Summarize causal results for all cohorts."""
    print(f"\n{'='*80}")
    print("Causal Analysis Results Summary - All Cohorts")
    print(f"{'='*80}\n")
    
    all_summaries = []
    completed_cohorts = []
    
    for cohort, age_bands in COHORTS.items():
        for age_band in age_bands:
            print(f"\nChecking {cohort}/{age_band}...", end=" ")
            
            df = download_causal_results(cohort, age_band)
            if df is not None and not df.empty:
                print("[FOUND]")
                summary = summarize_causal_results(df, cohort, age_band)
                all_summaries.append(summary)
                completed_cohorts.append(f"{cohort}/{age_band}")
                print_summary(summary)
            else:
                print("[NOT FOUND]")
    
    # Overall summary
    if all_summaries:
        print(f"\n{'='*80}")
        print("Overall Summary")
        print(f"{'='*80}\n")
        
        print(f"Completed Cohorts: {len(completed_cohorts)}")
        for cohort_age in completed_cohorts:
            print(f"  - {cohort_age}")
        
        # Aggregate statistics
        total_features_all = sum(s['total_features'] for s in all_summaries)
        total_with_effect = sum(s['features_with_effect'] for s in all_summaries)
        total_strong_effect = sum(s['features_with_strong_effect'] for s in all_summaries)
        
        print(f"\nAggregate Statistics:")
        print(f"  Total Features Analyzed (across all cohorts): {total_features_all}")
        print(f"  Features with Causal Effect: {total_with_effect} ({total_with_effect/total_features_all*100:.1f}%)")
        print(f"  Features with Strong Effect (>0.1): {total_strong_effect} ({total_strong_effect/total_features_all*100:.1f}%)")
        
        # Average statistics
        avg_mean = sum(s['mean_importance'] for s in all_summaries) / len(all_summaries)
        avg_max = sum(s['max_importance'] for s in all_summaries) / len(all_summaries)
        
        print(f"\nAverage Statistics:")
        print(f"  Average Mean Causal Importance: {avg_mean:.4f}")
        print(f"  Average Max Causal Importance: {avg_max:.4f}")
        
        # Find most important features across all cohorts
        print(f"\nMost Important Features Across All Cohorts:")
        all_top_features = []
        for summary in all_summaries:
            for feat in summary['top_features'][:5]:  # Top 5 from each
                all_top_features.append({
                    'cohort': f"{summary['cohort']}/{summary['age_band']}",
                    'feature': feat['feature'],
                    'importance': feat['causal_importance']
                })
        
        # Sort and show top 10 overall
        all_top_features.sort(key=lambda x: x['importance'], reverse=True)
        print(f"  {'Rank':<6} {'Cohort':<25} {'Feature':<40} {'Importance':<12}")
        print(f"  {'-'*6} {'-'*25} {'-'*40} {'-'*12}")
        for idx, feat in enumerate(all_top_features[:10], 1):
            feat_name = feat['feature'][:38] + ".." if len(feat['feature']) > 40 else feat['feature']
            print(f"  {idx:<6} {feat['cohort']:<25} {feat_name:<40} {feat['importance']:<12.4f}")
    else:
        print("\n[WARNING] No completed causal analysis results found in S3.")


def summarize_single_cohort(cohort: str, age_band: str):
    """Summarize causal results for a single cohort."""
    print(f"\n{'='*80}")
    print(f"Causal Analysis Summary: {cohort.upper()} / {age_band}")
    print(f"{'='*80}\n")
    
    df = download_causal_results(cohort, age_band)
    if df is None or df.empty:
        print(f"[ERROR] No causal results found for {cohort}/{age_band}")
        print(f"Expected location: s3://{OUTPUT_BUCKET}/gold/ffa_analysis/{cohort}/{age_band}/xgboost/causal_importance.parquet")
        return
    
    summary = summarize_causal_results(df, cohort, age_band)
    print_summary(summary)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Summarize causal analysis results from S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Summarize all completed cohorts
  python utility_scripts/summarize_causal_results.py --all-cohorts
  
  # Summarize specific cohort
  python utility_scripts/summarize_causal_results.py --cohort opioid_ed --age-band 13-24
        """
    )
    parser.add_argument("--cohort", help="Cohort name (e.g., opioid_ed)")
    parser.add_argument("--age-band", help="Age band (e.g., 13-24)")
    parser.add_argument("--all-cohorts", action="store_true", help="Summarize all completed cohorts")
    
    args = parser.parse_args()
    
    if args.all_cohorts:
        summarize_all_cohorts()
    elif args.cohort and args.age_band:
        summarize_single_cohort(args.cohort, args.age_band)
    else:
        parser.print_help()
        sys.exit(1)

