#!/usr/bin/env python3
"""Read and summarize FFA results from downloaded parquet files for all cohorts."""

import sys
from pathlib import Path

# Try to use DuckDB first (works without pyarrow)
try:
    import duckdb
    USE_DUCKDB = True
except ImportError:
    USE_DUCKDB = False
    try:
        import pandas as pd
        USE_PANDAS = True
    except ImportError:
        USE_PANDAS = False
        print("ERROR: Need either duckdb or pandas to read parquet files")
        sys.exit(1)

sys.stdout.reconfigure(encoding='utf-8')

def read_parquet_file(file_path):
    """Read parquet file using available library."""
    file_path_str = str(file_path).replace('\\', '/')
    if USE_DUCKDB:
        con = duckdb.connect()
        df = con.execute(f"SELECT * FROM read_parquet('{file_path_str}')").df()
        con.close()
        return df
    elif USE_PANDAS:
        return pd.read_parquet(file_path)
    else:
        raise ImportError("No parquet reader available")

def summarize_causal(causal_path, cohort, age_band):
    """Summarize causal importance results."""
    print(f"\n{'='*80}")
    print(f"{cohort.upper()} / {age_band} - TOP 20 CAUSAL FACTORS")
    print(f"{'='*80}")
    
    try:
        df = read_parquet_file(causal_path)
        top_features = df.nlargest(20, 'causal_importance')
        
        print(f"\n| Rank | Causal Importance | Feature |")
        print(f"|------|------------------|---------|")
        
        for rank, (idx, row) in enumerate(top_features.iterrows(), 1):
            causal_imp = f"{row['causal_importance']:.6f}"
            feature = str(row['feature'])
            print(f"| {rank} | {causal_imp} | {feature} |")
        
        return top_features
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def summarize_interactions(interaction_path, cohort, age_band):
    """Summarize interaction analysis results."""
    print(f"\n{'='*80}")
    print(f"{cohort.upper()} / {age_band} - TOP 20 INTERACTIONS")
    print(f"{'='*80}")
    
    try:
        df = read_parquet_file(interaction_path)
        
        if df.empty:
            print("\n[INFO] No interactions found")
            return None
        
        # Find feature combination column
        feature_col = None
        for col in ['feature_combination', 'features', 'feature_pair']:
            if col in df.columns:
                feature_col = col
                break
        
        if feature_col is None:
            print("\n[WARN] Could not find feature combination column")
            print(f"Available columns: {df.columns.tolist()}")
            print(df.head(10).to_string())
            return df
        
        # Sort by combined_causal_importance if available
        if 'combined_causal_importance' in df.columns:
            top_interactions = df.nlargest(20, 'combined_causal_importance')
            
            print(f"\n| Rank | Combined Causal | Interaction Effect | Features |")
            print(f"|------|----------------|-------------------|----------|")
            
            for rank, (idx, row) in enumerate(top_interactions.iterrows(), 1):
                combined = f"{row['combined_causal_importance']:.6f}"
                interaction_effect = f"{row.get('interaction_effect', 0):.6f}" if 'interaction_effect' in row else 'N/A'
                features = str(row[feature_col])
                print(f"| {rank} | {combined} | {interaction_effect} | {features} |")
        else:
            print("\n[WARN] No combined_causal_importance column")
            print(f"Available columns: {df.columns.tolist()}")
            print(df.head(10).to_string())
        
        return df
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    base_dir = Path("8_ffa_analysis/results")
    
    cohorts_data = [
        ('opioid_ed', '13-24'),
        ('opioid_ed', '25-44'),
        ('opioid_ed', '45-54'),
        ('opioid_ed', '55-64'),
        ('non_opioid_ed', '65-74'),
        ('non_opioid_ed', '75-84'),
        ('non_opioid_ed', '85-94'),
    ]
    
    print("=" * 80)
    print("FFA ANALYSIS RESULTS SUMMARY - ALL COHORTS")
    print("=" * 80)
    
    results = {}
    
    for cohort, age_band in cohorts_data:
        age_band_fname = age_band.replace('-', '_')
        causal_path = base_dir / f"causal_importance_{cohort}_{age_band_fname}.parquet"
        interaction_path = base_dir / f"interaction_analysis_{cohort}_{age_band_fname}.parquet"
        
        results[cohort, age_band] = {}
        
        if causal_path.exists():
            causal_df = summarize_causal(causal_path, cohort, age_band)
            results[cohort, age_band]['causal'] = causal_df
        else:
            print(f"\n[WARN] Causal file not found: {causal_path}")
        
        if interaction_path.exists():
            interaction_df = summarize_interactions(interaction_path, cohort, age_band)
            results[cohort, age_band]['interaction'] = interaction_df
        else:
            print(f"\n[WARN] Interaction file not found: {interaction_path}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
