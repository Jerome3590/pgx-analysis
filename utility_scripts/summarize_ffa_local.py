#!/usr/bin/env python3
"""Read and summarize FFA results from downloaded parquet files."""

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
        
        print(f"\n{'Causal Importance':<20} {'Feature':<60}")
        print("-" * 80)
        
        for idx, row in top_features.iterrows():
            causal_imp = f"{row['causal_importance']:.6f}"
            feature = str(row['feature'])[:58]
            print(f"{causal_imp:<20} {feature:<60}")
        
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
        
        print(f"\nColumns: {df.columns.tolist()}")
        
        # Find feature combination column
        feature_col = None
        for col in ['features', 'feature_combination', 'feature_pair']:
            if col in df.columns:
                feature_col = col
                break
        
        if feature_col is None:
            print("\n[WARN] Could not find feature combination column")
            print(df.head(10).to_string())
            return df
        
        # Sort by combined_causal_importance if available
        if 'combined_causal_importance' in df.columns:
            top_interactions = df.nlargest(20, 'combined_causal_importance')
            
            print(f"\n{'Combined Causal':<20} {'Interaction Effect':<20} {'Features':<60}")
            print("-" * 100)
            
            for idx, row in top_interactions.iterrows():
                combined = f"{row['combined_causal_importance']:.6f}"
                interaction_effect = f"{row.get('interaction_effect', 0):.6f}" if 'interaction_effect' in row else 'N/A'
                features = str(row[feature_col])[:58]
                print(f"{combined:<20} {interaction_effect:<20} {features:<60}")
        else:
            print("\n[WARN] No combined_causal_importance column")
            print(df.head(10).to_string())
        
        return df
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    base_dir = Path(r"C:\tmp\ffa_summary")
    
    cohorts_data = [
        ('opioid_ed', '13-24'),
        ('opioid_ed', '25-44'),
        ('opioid_ed', '45-54'),
    ]
    
    print("=" * 80)
    print("FFA ANALYSIS RESULTS SUMMARY")
    print("=" * 80)
    
    for cohort, age_band in cohorts_data:
        age_band_fname = age_band.replace('-', '_')
        causal_path = base_dir / f"causal_importance_{cohort}_{age_band_fname}.parquet"
        interaction_path = base_dir / f"interaction_analysis_{cohort}_{age_band_fname}.parquet"
        
        if causal_path.exists():
            summarize_causal(causal_path, cohort, age_band)
        else:
            print(f"\n[WARN] Causal file not found: {causal_path}")
        
        if interaction_path.exists():
            summarize_interactions(interaction_path, cohort, age_band)
        else:
            print(f"\n[WARN] Interaction file not found: {interaction_path}")

if __name__ == '__main__':
    main()
