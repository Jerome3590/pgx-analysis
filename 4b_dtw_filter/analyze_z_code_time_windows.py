#!/usr/bin/env python3
"""
Analyze Z code time windows to test hypothesis:
"Extreme cohorts have larger time windows than standard cohorts"

This script:
1. Loads all Z code analysis summary statistics
2. Compares time window distributions between standard and extreme cohorts
3. Tests the hypothesis that extreme cohorts have larger absolute time windows
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import COHORT_NAMES, AGE_BANDS

OUTPUT_DIR = PROJECT_ROOT / "4b_dtw_filter" / "outputs" / "z_code_analysis"
RESULTS_DIR = OUTPUT_DIR


def load_summary_statistics() -> pd.DataFrame:
    """Load all summary statistics JSON files."""
    results = []
    
    for cohort_name in COHORT_NAMES:
        for age_band in AGE_BANDS:
            age_fname = age_band.replace("-", "_")
            cohort_fname = cohort_name.replace("_", "-")
            
            # Try both naming patterns
            json_file = RESULTS_DIR / f"z_code_summary_{cohort_fname}_{age_fname}.json"
            if not json_file.exists():
                json_file = RESULTS_DIR / f"z_code_summary_{cohort_name}_{age_fname}.json"
            
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        stats = json.load(f)
                    
                    # Extract relevant statistics
                    record = {
                        'cohort': cohort_name,
                        'age_band': age_band,
                        'standard_events': stats.get('standard_total_events', 0),
                        'extreme_events': stats.get('extreme_total_events', 0),
                        'standard_days_mean': stats.get('standard_days_mean'),
                        'standard_days_median': stats.get('standard_days_median'),
                        'standard_days_std': stats.get('standard_days_std'),
                        'standard_days_q25': stats.get('standard_days_q25'),
                        'standard_days_q75': stats.get('standard_days_q75'),
                        'extreme_days_mean': stats.get('extreme_days_mean'),
                        'extreme_days_median': stats.get('extreme_days_median'),
                        'extreme_days_std': stats.get('extreme_days_std'),
                        'extreme_days_q25': stats.get('extreme_days_q25'),
                        'extreme_days_q75': stats.get('extreme_days_q75'),
                    }
                    
                    # Calculate absolute time windows (distance from target, regardless of direction)
                    if record['standard_days_mean'] is not None:
                        record['standard_abs_mean'] = abs(record['standard_days_mean'])
                    if record['standard_days_median'] is not None:
                        record['standard_abs_median'] = abs(record['standard_days_median'])
                    if record['extreme_days_mean'] is not None:
                        record['extreme_abs_mean'] = abs(record['extreme_days_mean'])
                    if record['extreme_days_median'] is not None:
                        record['extreme_abs_median'] = abs(record['extreme_days_median'])
                    
                    results.append(record)
                except Exception as e:
                    print(f"Warning: Could not load {json_file}: {e}")
    
    return pd.DataFrame(results)


def load_detailed_data() -> pd.DataFrame:
    """Load detailed CSV files to calculate absolute time windows."""
    all_data = []
    
    for cohort_name in COHORT_NAMES:
        for age_band in AGE_BANDS:
            age_fname = age_band.replace("-", "_")
            cohort_fname = cohort_name.replace("_", "-")
            
            csv_file = RESULTS_DIR / f"z_code_analysis_{cohort_fname}_{age_fname}.csv"
            if not csv_file.exists():
                csv_file = RESULTS_DIR / f"z_code_analysis_{cohort_name}_{age_fname}.csv"
            
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    if len(df) > 0:
                        df['cohort'] = cohort_name
                        df['age_band'] = age_band
                        df['abs_days_from_target'] = df['days_from_target'].abs()
                        all_data.append(df)
                except Exception as e:
                    print(f"Warning: Could not load {csv_file}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)


def analyze_time_windows():
    """Main analysis function."""
    print("=" * 80)
    print("Z Code Time Window Analysis: Extreme vs Standard Cohorts")
    print("=" * 80)
    print()
    print("Hypothesis: Extreme cohorts have larger time windows than standard cohorts")
    print()
    
    # Load summary statistics
    print("Loading summary statistics...")
    summary_df = load_summary_statistics()
    
    if len(summary_df) == 0:
        print("No summary statistics found!")
        return
    
    print(f"Loaded {len(summary_df)} cohort/age_band combinations")
    print()
    
    # Filter to only cohorts with both standard and extreme data
    has_both = summary_df[
        (summary_df['standard_events'] > 0) & 
        (summary_df['extreme_events'] > 0) &
        (summary_df['standard_days_mean'].notna()) &
        (summary_df['extreme_days_mean'].notna())
    ].copy()
    
    print(f"Cohorts with both standard and extreme data: {len(has_both)}")
    print()
    
    if len(has_both) == 0:
        print("No cohorts with both standard and extreme data found!")
        print("Loading detailed data for alternative analysis...")
        
        # Try detailed data analysis
        detailed_df = load_detailed_data()
        if len(detailed_df) > 0:
            analyze_detailed_data(detailed_df)
        return
    
    # Calculate absolute time windows
    has_both['standard_abs_mean'] = has_both['standard_days_mean'].abs()
    has_both['extreme_abs_mean'] = has_both['extreme_days_mean'].abs()
    has_both['standard_abs_median'] = has_both['standard_days_median'].abs()
    has_both['extreme_abs_median'] = has_both['extreme_days_median'].abs()
    
    # Calculate differences
    has_both['abs_mean_diff'] = has_both['extreme_abs_mean'] - has_both['standard_abs_mean']
    has_both['abs_median_diff'] = has_both['extreme_abs_median'] - has_both['standard_abs_median']
    
    # Print results
    print("=" * 80)
    print("Time Window Comparison: Extreme vs Standard")
    print("=" * 80)
    print()
    
    for idx, row in has_both.iterrows():
        print(f"{row['cohort']} / {row['age_band']}:")
        print(f"  Standard: Mean={row['standard_abs_mean']:.1f} days, Median={row['standard_abs_median']:.1f} days")
        print(f"  Extreme:  Mean={row['extreme_abs_mean']:.1f} days, Median={row['extreme_abs_median']:.1f} days")
        print(f"  Difference: Mean={row['abs_mean_diff']:.1f} days, Median={row['abs_median_diff']:.1f} days")
        if row['abs_mean_diff'] > 0:
            print(f"  [SUPPORTS HYPOTHESIS] Extreme has larger time window")
        else:
            print(f"  [CONTRADICTS HYPOTHESIS] Standard has larger time window")
        print()
    
    # Overall statistics
    print("=" * 80)
    print("Overall Statistics")
    print("=" * 80)
    print()
    
    mean_diff_mean = has_both['abs_mean_diff'].mean()
    mean_diff_median = has_both['abs_median_diff'].median()
    
    print(f"Average difference in absolute mean time window: {mean_diff_mean:.1f} days")
    print(f"Median difference in absolute median time window: {mean_diff_median:.1f} days")
    print()
    
    supports = (has_both['abs_mean_diff'] > 0).sum()
    contradicts = (has_both['abs_mean_diff'] <= 0).sum()
    
    print(f"Cohorts supporting hypothesis (extreme > standard): {supports}/{len(has_both)} ({100*supports/len(has_both):.1f}%)")
    print(f"Cohorts contradicting hypothesis (standard >= extreme): {contradicts}/{len(has_both)} ({100*contradicts/len(has_both):.1f}%)")
    print()
    
    # Load detailed data for more robust analysis
    print("=" * 80)
    print("Detailed Data Analysis")
    print("=" * 80)
    print()
    
    detailed_df = load_detailed_data()
    if len(detailed_df) > 0:
        analyze_detailed_data(detailed_df)
    
    # Save results
    output_csv = OUTPUT_DIR / "time_window_comparison.csv"
    has_both.to_csv(output_csv, index=False)
    print(f"\n[OK] Saved comparison results: {output_csv}")


def analyze_detailed_data(df: pd.DataFrame):
    """Analyze detailed data for more robust statistics."""
    # Filter to cohorts with both standard and extreme
    has_both = df[
        (df['is_extreme'].notna()) & 
        (df['days_from_target'].notna())
    ].copy()
    
    if len(has_both) == 0:
        print("No detailed data with both standard and extreme cohorts found")
        return
    
    # Calculate absolute time windows
    has_both['abs_days'] = has_both['days_from_target'].abs()
    
    # Group by cohort, age_band, and is_extreme
    grouped = has_both.groupby(['cohort', 'age_band', 'is_extreme'])['abs_days'].agg([
        'mean', 'median', 'std', 'count'
    ]).reset_index()
    
    # Pivot to compare standard vs extreme
    comparison = grouped.pivot_table(
        index=['cohort', 'age_band'],
        columns='is_extreme',
        values=['mean', 'median', 'std', 'count'],
        aggfunc='first'
    ).reset_index()
    
    # Flatten column names
    comparison.columns = ['_'.join(str(col).strip() for col in col).strip('_') 
                          for col in comparison.columns.values]
    
    # Calculate differences
    if 'mean_False' in comparison.columns and 'mean_True' in comparison.columns:
        comparison['mean_diff'] = comparison['mean_True'] - comparison['mean_False']
        comparison['median_diff'] = comparison['median_True'] - comparison['median_False']
        
        print("Detailed Time Window Comparison:")
        print("-" * 80)
        for idx, row in comparison.iterrows():
            if pd.notna(row.get('mean_False')) and pd.notna(row.get('mean_True')):
                print(f"{row['cohort']} / {row['age_band']}:")
                print(f"  Standard: Mean={row['mean_False']:.1f}, Median={row.get('median_False', 'N/A')}")
                print(f"  Extreme:  Mean={row['mean_True']:.1f}, Median={row.get('median_True', 'N/A')}")
                if pd.notna(row.get('mean_diff')):
                    print(f"  Difference: {row['mean_diff']:.1f} days")
                    if row['mean_diff'] > 0:
                        print(f"  [SUPPORTS HYPOTHESIS]")
                    else:
                        print(f"  [CONTRADICTS HYPOTHESIS]")
                print()
        
        # Overall statistics
        valid_diffs = comparison['mean_diff'].dropna()
        if len(valid_diffs) > 0:
            print("=" * 80)
            print("Overall Statistics (Detailed Data)")
            print("=" * 80)
            print()
            print(f"Average difference: {valid_diffs.mean():.1f} days")
            print(f"Median difference: {valid_diffs.median():.1f} days")
            print(f"Standard deviation: {valid_diffs.std():.1f} days")
            print()
            print(f"Supporting hypothesis: {(valid_diffs > 0).sum()}/{len(valid_diffs)} ({100*(valid_diffs > 0).sum()/len(valid_diffs):.1f}%)")
            print(f"Contradicting hypothesis: {(valid_diffs <= 0).sum()}/{len(valid_diffs)} ({100*(valid_diffs <= 0).sum()/len(valid_diffs):.1f}%)")
        
        # Save detailed comparison
        output_csv = OUTPUT_DIR / "time_window_comparison_detailed.csv"
        comparison.to_csv(output_csv, index=False)
        print(f"\n[OK] Saved detailed comparison: {output_csv}")


if __name__ == "__main__":
    analyze_time_windows()
