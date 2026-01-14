#!/usr/bin/env python3
"""
Analyze trajectory and event count features for triggering/thresholding in predictive model.

This script examines:
1. Trajectory features (combined_trajectory_length, combined_trajectory_diversity)
2. Pre-event count features (pre_n_events, pre_n_unique_activities, etc.)
3. Their distributions and predictive power
4. Suitability for triggering/thresholding
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def analyze_trigger_features(cohort_name: str, age_band: str):
    """Analyze features for triggering/thresholding."""
    
    age_band_fname = age_band.replace("-", "_")
    
    # Load cleaned feature table
    feature_path = (
        PROJECT_ROOT
        / "8_final_model"
        / "outputs"
        / cohort_name
        / age_band_fname
        / f"{cohort_name}_{age_band_fname}_train_final_features_no_leakage.csv"
    )
    
    if not feature_path.exists():
        print(f"[ERROR] Feature file not found: {feature_path}")
        return
    
    print(f"[INFO] Loading features from: {feature_path}")
    df = pd.read_csv(feature_path)
    
    print("\n" + "="*80)
    print("FEATURE ANALYSIS FOR TRIGGERING/THRESHOLDING")
    print("="*80)
    
    # Check if target column exists
    has_target = 'target' in df.columns
    
    # ========================================================================
    # 1. TRAJECTORY FEATURES (from DTW analysis)
    # ========================================================================
    print("\n1. TRAJECTORY FEATURES (Filtered by FP-Growth Itemsets)")
    print("-" * 80)
    
    traj_cols = [c for c in df.columns if 'trajectory' in c.lower()]
    
    if traj_cols:
        for col in traj_cols:
            print(f"\n  {col}:")
            print(f"    Description: {'Length' if 'length' in col else 'Diversity'} of patient trajectory")
            print(f"    Source: DTW analysis (filtered by FP-Growth important itemsets)")
            print(f"    Min: {df[col].min():.2f}")
            print(f"    Max: {df[col].max():.2f}")
            print(f"    Mean: {df[col].mean():.2f}")
            print(f"    Median: {df[col].median():.2f}")
            print(f"    25th percentile: {df[col].quantile(0.25):.2f}")
            print(f"    75th percentile: {df[col].quantile(0.75):.2f}")
            print(f"    90th percentile: {df[col].quantile(0.90):.2f}")
            print(f"    95th percentile: {df[col].quantile(0.95):.2f}")
            
            if has_target:
                target_mean = df[df['target']==1][col].mean()
                control_mean = df[df['target']==0][col].mean()
                print(f"    Target mean: {target_mean:.2f}")
                print(f"    Control mean: {control_mean:.2f}")
                print(f"    Difference: {target_mean - control_mean:.2f} ({((target_mean - control_mean) / control_mean * 100) if control_mean > 0 else 0:.1f}%)")
            
            # Suggest thresholds
            q75 = df[col].quantile(0.75)
            q90 = df[col].quantile(0.90)
            q95 = df[col].quantile(0.95)
            print(f"\n    Suggested Trigger Thresholds:")
            print(f"      Medium risk (>75th percentile): > {q75:.0f}")
            print(f"      High risk (>90th percentile): > {q90:.0f}")
            print(f"      Very high risk (>95th percentile): > {q95:.0f}")
    else:
        print("  No trajectory features found")
    
    # ========================================================================
    # 2. PRE-EVENT COUNT FEATURES (from BupaR analysis)
    # ========================================================================
    print("\n\n2. PRE-EVENT COUNT FEATURES (All Events Before F1120)")
    print("-" * 80)
    
    pre_cols = [c for c in df.columns if c.startswith('pre_n_')]
    
    if pre_cols:
        for col in pre_cols[:6]:  # Show first 6
            print(f"\n  {col}:")
            print(f"    Description: Count of {'all events' if 'events' in col and 'unique' not in col else 'unique activities' if 'unique' in col else col.split('_')[-1] + ' events'}")
            print(f"    Source: BupaR analysis (all pre-F1120 events)")
            print(f"    Min: {df[col].min():.2f}")
            print(f"    Max: {df[col].max():.2f}")
            print(f"    Mean: {df[col].mean():.2f}")
            print(f"    Median: {df[col].median():.2f}")
            print(f"    25th percentile: {df[col].quantile(0.25):.2f}")
            print(f"    75th percentile: {df[col].quantile(0.75):.2f}")
            print(f"    90th percentile: {df[col].quantile(0.90):.2f}")
            print(f"    95th percentile: {df[col].quantile(0.95):.2f}")
            
            if has_target:
                target_mean = df[df['target']==1][col].mean()
                control_mean = df[df['target']==0][col].mean()
                print(f"    Target mean: {target_mean:.2f}")
                print(f"    Control mean: {control_mean:.2f}")
                print(f"    Difference: {target_mean - control_mean:.2f} ({((target_mean - control_mean) / control_mean * 100) if control_mean > 0 else 0:.1f}%)")
            
            # Suggest thresholds
            q75 = df[col].quantile(0.75)
            q90 = df[col].quantile(0.90)
            q95 = df[col].quantile(0.95)
            print(f"\n    Suggested Trigger Thresholds:")
            print(f"      Medium risk (>75th percentile): > {q75:.0f}")
            print(f"      High risk (>90th percentile): > {q90:.0f}")
            print(f"      Very high risk (>95th percentile): > {q95:.0f}")
    else:
        print("  No pre-event count features found")
    
    # ========================================================================
    # 3. COMPARISON: Trajectory vs Pre-Event Features
    # ========================================================================
    print("\n\n3. COMPARISON: Trajectory Features vs Pre-Event Features")
    print("-" * 80)
    
    if 'combined_trajectory_length' in df.columns and 'pre_n_events' in df.columns:
        print("\n  Trajectory Length vs Pre-Event Count:")
        print(f"    Trajectory length (filtered): mean={df['combined_trajectory_length'].mean():.2f}, median={df['combined_trajectory_length'].median():.2f}")
        print(f"    Pre-event count (all events): mean={df['pre_n_events'].mean():.2f}, median={df['pre_n_events'].median():.2f}")
        print(f"    Difference: Trajectory features are FILTERED by FP-Growth itemsets (only important codes)")
        print(f"                Pre-event features include ALL events before F1120")
        
        # Correlation
        corr = df['combined_trajectory_length'].corr(df['pre_n_events'])
        print(f"    Correlation: {corr:.3f}")
    
    if 'combined_trajectory_diversity' in df.columns and 'pre_n_unique_activities' in df.columns:
        print("\n  Trajectory Diversity vs Pre-Event Unique Activities:")
        print(f"    Trajectory diversity (filtered): mean={df['combined_trajectory_diversity'].mean():.2f}, median={df['combined_trajectory_diversity'].median():.2f}")
        print(f"    Pre-event unique (all events): mean={df['pre_n_unique_activities'].mean():.2f}, median={df['pre_n_unique_activities'].median():.2f}")
        print(f"    Difference: Trajectory diversity counts unique items in FILTERED trajectory")
        print(f"                Pre-event unique counts unique items in ALL pre-F1120 events")
        
        # Correlation
        corr = df['combined_trajectory_diversity'].corr(df['pre_n_unique_activities'])
        print(f"    Correlation: {corr:.3f}")
    
    # ========================================================================
    # 4. RECOMMENDATIONS FOR TRIGGERING
    # ========================================================================
    print("\n\n4. RECOMMENDATIONS FOR TRIGGERING/THRESHOLDING")
    print("-" * 80)
    
    print("\n  [YES] These features CAN be used for triggering:")
    print("\n    1. Trajectory Features (combined_trajectory_length, combined_trajectory_diversity):")
    print("       - Use when: You want to trigger on patients with complex trajectories")
    print("       - Advantage: Focused on important codes (filtered by FP-Growth)")
    print("       - Best for: Identifying patients with many important events/codes")
    print("       - Trigger example: 'IF combined_trajectory_length > 75th percentile THEN flag'")
    
    print("\n    2. Pre-Event Count Features (pre_n_events, pre_n_unique_activities, etc.):")
    print("       - Use when: You want to trigger on total event volume")
    print("       - Advantage: Captures all events, not just filtered ones")
    print("       - Best for: Identifying patients with high overall healthcare utilization")
    print("       - Trigger example: 'IF pre_n_events > 90th percentile THEN flag'")
    
    print("\n  Feature Importance Ranking (from model):")
    if Path(PROJECT_ROOT / "8_final_model" / "outputs" / cohort_name / age_band_fname / f"{cohort_name}_{age_band_fname}_final_feature_importance_top_50.csv").exists():
        fi_df = pd.read_csv(PROJECT_ROOT / "8_final_model" / "outputs" / cohort_name / age_band_fname / f"{cohort_name}_{age_band_fname}_final_feature_importance_top_50.csv")
        trigger_features = fi_df[fi_df['feature'].str.contains('trajectory|pre_n_', case=False, na=False)]
        if len(trigger_features) > 0:
            print("\n    Top trigger-able features:")
            for idx, row in trigger_features.head(10).iterrows():
                print(f"      {row['rank']:2d}. {row['feature']:40s} (importance: {row['importance_scaled']:.2f})")
    
    print("\n  Best Practices for Triggering:")
    print("     1. Use PERCENTILE-BASED thresholds (75th, 90th, 95th) rather than absolute values")
    print("     2. Combine multiple features: 'IF (trajectory_length > 90th) AND (pre_n_events > 75th) THEN flag'")
    print("     3. Use trajectory features for QUALITY (important codes), pre_n features for QUANTITY (all events)")
    print("     4. For larger cohorts: Trajectory features will scale better (filtered = less noise)")
    print("     5. Consider age-band-specific thresholds (different distributions per age band)")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze features for triggering/thresholding")
    parser.add_argument("--cohort-name", type=str, default="opioid_ed", help="Cohort name")
    parser.add_argument("--age-band", type=str, default="0-12", help="Age band")
    
    args = parser.parse_args()
    
    analyze_trigger_features(args.cohort_name, args.age_band)

