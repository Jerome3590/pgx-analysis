#!/usr/bin/env python3
"""
Analyze remaining FFA cohorts and generate summary.
Run this on EC2 instance where duckdb/pyarrow are available.
"""

import sys
from pathlib import Path
import json

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

def analyze_cohort(base_dir, cohort, age_band):
    """Analyze a single cohort and return summary."""
    age_band_fname = age_band.replace('-', '_')
    causal_path = base_dir / f"causal_importance_{cohort}_{age_band_fname}.parquet"
    interaction_path = base_dir / f"interaction_analysis_{cohort}_{age_band_fname}.parquet"
    
    result = {
        'cohort': cohort,
        'age_band': age_band,
        'causal_factors': [],
        'interactions': []
    }
    
    # Analyze causal factors
    if causal_path.exists():
        df = read_parquet_file(causal_path)
        top_features = df.nlargest(20, 'causal_importance')
        result['causal_factors'] = [
            {
                'rank': rank,
                'feature': str(row['feature']),
                'causal_importance': float(row['causal_importance'])
            }
            for rank, (idx, row) in enumerate(top_features.iterrows(), 1)
        ]
    
    # Analyze interactions
    if interaction_path.exists():
        df = read_parquet_file(interaction_path)
        if not df.empty:
            feature_col = None
            for col in ['feature_combination', 'features', 'feature_pair']:
                if col in df.columns:
                    feature_col = col
                    break
            
            if feature_col and 'combined_causal_importance' in df.columns:
                top_interactions = df.nlargest(20, 'combined_causal_importance')
                result['interactions'] = [
                    {
                        'rank': rank,
                        'features': str(row[feature_col]),
                        'combined_causal': float(row['combined_causal_importance']),
                        'interaction_effect': float(row.get('interaction_effect', 0))
                    }
                    for rank, (idx, row) in enumerate(top_interactions.iterrows(), 1)
                ]
    
    return result

def generate_markdown_summary(results):
    """Generate markdown summary from results."""
    markdown = []
    
    for result in results:
        cohort = result['cohort']
        age_band = result['age_band']
        
        markdown.append(f"\n## {cohort.upper()} / {age_band} (Ages {age_band})")
        markdown.append("")
        
        # Causal factors
        if result['causal_factors']:
            markdown.append("### Top 20 Causal Factors")
            markdown.append("")
            markdown.append("| Rank | Causal Importance | Feature |")
            markdown.append("|------|------------------|---------|")
            for item in result['causal_factors']:
                markdown.append(f"| {item['rank']} | {item['causal_importance']:.6f} | {item['feature']} |")
            markdown.append("")
        
        # Interactions
        if result['interactions']:
            markdown.append("### Top 20 Interactions")
            markdown.append("")
            markdown.append("| Rank | Combined Causal | Interaction Effect | Features |")
            markdown.append("|------|----------------|-------------------|----------|")
            for item in result['interactions']:
                markdown.append(f"| {item['rank']} | {item['combined_causal']:.6f} | {item['interaction_effect']:.6f} | {item['features']} |")
            markdown.append("")
        
        markdown.append("---")
    
    return "\n".join(markdown)

def main():
    base_dir = Path("8_ffa_analysis/results")
    
    cohorts_data = [
        ('opioid_ed', '55-64'),
        ('non_opioid_ed', '65-74'),
        ('non_opioid_ed', '75-84'),
        ('non_opioid_ed', '85-94'),
    ]
    
    print("Analyzing remaining cohorts...")
    results = []
    
    for cohort, age_band in cohorts_data:
        print(f"Processing {cohort}/{age_band}...")
        result = analyze_cohort(base_dir, cohort, age_band)
        results.append(result)
    
    # Generate markdown
    markdown = generate_markdown_summary(results)
    
    # Save to file
    output_file = base_dir / "REMAINING_COHORTS_ANALYSIS.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# FFA Analysis - Remaining Cohorts\n\n")
        f.write("Analysis of cohorts: opioid_ed/55-64, non_opioid_ed/65-74, non_opioid_ed/75-84, non_opioid_ed/85-94\n\n")
        f.write(markdown)
    
    print(f"\nAnalysis complete! Results saved to: {output_file}")
    
    # Also print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(markdown)

if __name__ == '__main__':
    main()
