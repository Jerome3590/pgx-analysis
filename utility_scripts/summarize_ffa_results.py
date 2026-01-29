#!/usr/bin/env python3
"""
Summarize FFA Analysis Results

Downloads and summarizes causal importance and interaction analysis results
from S3 for all completed cohorts.
"""

import sys
import tempfile
from pathlib import Path
import pandas as pd
import duckdb

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from py_helpers.common_imports import s3_client, S3_BUCKET
except ImportError:
    import boto3
    s3_client = boto3.client("s3")
    S3_BUCKET = "pgxdatalake"


def download_and_read_parquet(s3_key: str) -> pd.DataFrame:
    """Download parquet file from S3 and read it."""
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        temp_path = tmp.name
    
    try:
        s3_client.download_file(S3_BUCKET, s3_key, temp_path)
        con = duckdb.connect()
        df = con.execute(f"SELECT * FROM read_parquet('{temp_path}')").df()
        con.close()
        return df
    finally:
        Path(temp_path).unlink(missing_ok=True)


def summarize_causal_factors(cohort: str, age_band: str, top_n: int = 20):
    """Summarize causal factors for a cohort/age_band."""
    s3_key = f"gold/ffa_analysis/{cohort}/{age_band}/xgboost/causal_importance.parquet"
    
    try:
        df = download_and_read_parquet(s3_key)
        top_features = df.nlargest(top_n, 'causal_importance')
        
        print(f"\n{'='*80}")
        print(f"{cohort.upper()} / {age_band} - TOP {top_n} CAUSAL FACTORS")
        print(f"{'='*80}")
        print(f"\n{'Causal Importance':<20} {'Feature':<60} {'Support':<12} {'Confidence':<12}")
        print("-" * 104)
        
        for idx, row in top_features.iterrows():
            causal_imp = f"{row['causal_importance']:.6f}"
            feature = str(row['feature'])[:58]
            support = str(row.get('support', 'N/A'))[:10]
            confidence = f"{row.get('confidence', 0):.4f}" if 'confidence' in row else 'N/A'
            print(f"{causal_imp:<20} {feature:<60} {support:<12} {confidence:<12}")
        
        return top_features
    except Exception as e:
        print(f"\n[ERROR] Could not load causal factors for {cohort}/{age_band}: {e}")
        return pd.DataFrame()


def summarize_interactions(cohort: str, age_band: str, top_n: int = 20):
    """Summarize interactions for a cohort/age_band."""
    s3_key = f"gold/ffa_analysis/{cohort}/{age_band}/xgboost/interaction_analysis.parquet"
    
    try:
        df = download_and_read_parquet(s3_key)
        
        if df.empty:
            print(f"\n[INFO] No interactions found for {cohort}/{age_band}")
            return pd.DataFrame()
        
        # Determine which column contains the feature combination
        feature_col = None
        for col in ['features', 'feature_combination', 'feature_pair']:
            if col in df.columns:
                feature_col = col
                break
        
        if feature_col is None:
            print(f"\n[WARN] Could not find feature combination column. Columns: {df.columns.tolist()}")
            print(df.head())
            return df
        
        # Sort by combined_causal_importance
        if 'combined_causal_importance' in df.columns:
            top_interactions = df.nlargest(top_n, 'combined_causal_importance')
        else:
            print(f"\n[WARN] No combined_causal_importance column. Columns: {df.columns.tolist()}")
            return df
        
        print(f"\n{'='*80}")
        print(f"{cohort.upper()} / {age_band} - TOP {top_n} INTERACTIONS")
        print(f"{'='*80}")
        print(f"\n{'Combined Causal':<20} {'Interaction Effect':<20} {'Features':<60}")
        print("-" * 100)
        
        for idx, row in top_interactions.iterrows():
            combined = f"{row['combined_causal_importance']:.6f}"
            interaction_effect = f"{row.get('interaction_effect', 0):.6f}" if 'interaction_effect' in row else 'N/A'
            features = str(row[feature_col])[:58]
            print(f"{combined:<20} {interaction_effect:<20} {features:<60}")
        
        return top_interactions
    except Exception as e:
        print(f"\n[ERROR] Could not load interactions for {cohort}/{age_band}: {e}")
        return pd.DataFrame()


def main():
    """Main function to summarize all FFA results."""
    print("=" * 80)
    print("FFA ANALYSIS RESULTS SUMMARY")
    print("=" * 80)
    
    cohorts = {
        'opioid_ed': ['13-24', '25-44', '45-54', '55-64'],
        'non_opioid_ed': ['65-74', '75-84', '85-94']
    }
    
    for cohort, age_bands in cohorts.items():
        print(f"\n\n{'#'*80}")
        print(f"# {cohort.upper()} COHORT")
        print(f"{'#'*80}")
        
        for age_band in age_bands:
            # Check if causal_importance exists
            s3_key_causal = f"gold/ffa_analysis/{cohort}/{age_band}/xgboost/causal_importance.parquet"
            try:
                s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key_causal)
                summarize_causal_factors(cohort, age_band, top_n=20)
                
                # Check if interactions exist
                s3_key_interaction = f"gold/ffa_analysis/{cohort}/{age_band}/xgboost/interaction_analysis.parquet"
                try:
                    s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key_interaction)
                    summarize_interactions(cohort, age_band, top_n=20)
                except:
                    print(f"\n[INFO] No interaction analysis available for {cohort}/{age_band}")
            except:
                print(f"\n[INFO] No FFA results available for {cohort}/{age_band}")


if __name__ == '__main__':
    main()
