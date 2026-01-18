#!/usr/bin/env python3
"""
Create hybrid feature filter JSON that combines timing analysis with feature importance.

This script:
1. Loads bupar_post_target_analysis.csv for timing information
2. Loads aggregated feature importance for importance scores
3. Creates a hybrid filter that:
   - Excludes pure post-target leakage (>=80% post-F1120)
   - Keeps features with ANY pre-F1120 presence if they have high importance
   - Provides recommendations based on both timing and importance
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import age_band_to_fname


def load_feature_importance(cohort: str, age_band: str, project_root: Path) -> Optional[pd.DataFrame]:
    """Load aggregated feature importance if available."""
    age_band_fname = age_band_to_fname(age_band)
    
    possible_paths = [
        project_root / "3b_feature_importance_eda" / "outputs" / cohort / age_band_fname / f"{cohort}_{age_band_fname}_cohort_feature_importance.csv",
        project_root / "3_feature_importance" / "outputs" / cohort / age_band / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv",
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"[INFO] Loading feature importance from: {path}")
            df = pd.read_csv(path)
            return df
    
    print(f"[WARN] Feature importance file not found. Will use timing-only approach.")
    return None


def create_hybrid_feature_filter_json(
    cohort: str,
    age_band: str,
    post_f1120_threshold: float = 0.8,
    min_pre_ratio: float = 0.0,  # Minimum pre-F1120 ratio to keep (0.0 = keep if any pre presence)
    min_importance: float = 0.0,  # Minimum importance to keep mixed-timing features
):
    """Create hybrid feature filter JSON."""
    age_band_fname = age_band_to_fname(age_band)
    project_root = Path(__file__).parent.parent
    
    # Load BupaR analysis results
    analysis_path = project_root / "3b_feature_importance_eda" / "outputs" / cohort / age_band_fname / f"{cohort}_{age_band_fname}_bupar_post_target_analysis.csv"
    
    if not analysis_path.exists():
        print(f"[ERROR] Analysis file not found: {analysis_path}")
        return None
    
    print(f"\n{'='*80}")
    print(f"Creating Hybrid Feature Filter JSON")
    print(f"Cohort: {cohort} / Age Band: {age_band}")
    print(f"Post-F1120 leakage threshold: >= {post_f1120_threshold:.0%}")
    print(f"Minimum pre-F1120 ratio to keep: >= {min_pre_ratio:.0%}")
    print(f"{'='*80}\n")
    
    timing_df = pd.read_csv(analysis_path)
    
    # Load feature importance if available
    importance_df = load_feature_importance(cohort, age_band, project_root)
    
    # Merge timing and importance data
    if importance_df is not None:
        # Find importance column
        importance_col = None
        for col in ['importance_scaled', 'importance_normalized', 'importance_mean', 'importance']:
            if col in importance_df.columns:
                importance_col = col
                break
        
        if importance_col:
            # Merge on feature name
            merged_df = timing_df.merge(
                importance_df[['feature', importance_col]].rename(columns={importance_col: 'feature_importance'}),
                on='feature',
                how='left'
            )
            merged_df['feature_importance'] = merged_df['feature_importance'].fillna(0.0)
            print(f"[INFO] Merged timing data with feature importance ({importance_col})")
        else:
            merged_df = timing_df.copy()
            merged_df['feature_importance'] = 0.0
            print(f"[WARN] No importance column found in feature importance file")
    else:
        merged_df = timing_df.copy()
        merged_df['feature_importance'] = 0.0
    
    # Categorize features
    # 1. Pure leakage: >=80% post-F1120 (exclude)
    pure_leakage = merged_df[merged_df['post_f1120_ratio'] >= post_f1120_threshold].copy()
    
    # 2. Pure predictive: >=80% pre-F1120 (definitely keep)
    pure_predictive = merged_df[merged_df['pre_f1120_ratio'] >= 0.8].copy()
    
    # 3. Mixed timing: 20-80% pre-F1120 (evaluate based on importance)
    mixed_timing = merged_df[
        (merged_df['pre_f1120_ratio'] >= min_pre_ratio) &
        (merged_df['pre_f1120_ratio'] < 0.8) &
        (merged_df['post_f1120_ratio'] < post_f1120_threshold)
    ].copy()
    
    # 4. Low pre-ratio but not pure leakage: <20% pre but <80% post
    low_pre = merged_df[
        (merged_df['pre_f1120_ratio'] < min_pre_ratio) &
        (merged_df['post_f1120_ratio'] < post_f1120_threshold)
    ].copy()
    
    print(f"Feature categorization:")
    print(f"  Pure leakage (>=80% post): {len(pure_leakage)} features - EXCLUDE")
    print(f"  Pure predictive (>=80% pre): {len(pure_predictive)} features - KEEP")
    print(f"  Mixed timing ({min_pre_ratio:.0%}-80% pre): {len(mixed_timing)} features - EVALUATE")
    print(f"  Low pre-ratio (<{min_pre_ratio:.0%} pre, <80% post): {len(low_pre)} features - EVALUATE")
    
    # For mixed-timing features, keep if they have high importance
    if min_importance > 0:
        mixed_keep = mixed_timing[mixed_timing['feature_importance'] >= min_importance].copy()
        mixed_exclude = mixed_timing[mixed_timing['feature_importance'] < min_importance].copy()
        print(f"\n  Mixed-timing with importance >= {min_importance}: {len(mixed_keep)} - KEEP")
        print(f"  Mixed-timing with importance < {min_importance}: {len(mixed_exclude)} - EXCLUDE")
    else:
        # If no importance threshold, keep all mixed-timing features
        mixed_keep = mixed_timing.copy()
        mixed_exclude = pd.DataFrame()
        print(f"\n  Mixed-timing (all): {len(mixed_keep)} - KEEP (no importance threshold)")
    
    # Always include F1120 for target creation
    f1120_feature = 'item_icd_F1120'
    if f1120_feature not in pure_predictive['feature'].values:
        f1120_row = merged_df[merged_df['feature'] == f1120_feature]
        if len(f1120_row) > 0:
            pure_predictive = pd.concat([pure_predictive, f1120_row], ignore_index=True)
            print(f"[INFO] Added {f1120_feature} to keep list (needed for target creation)")
    
    # Combine features to keep
    features_to_keep = pd.concat([
        pure_predictive,
        mixed_keep
    ], ignore_index=True)
    
    # Features to exclude
    features_to_exclude = pd.concat([
        pure_leakage,
        mixed_exclude
    ], ignore_index=True)
    
    print(f"\nFinal counts:")
    print(f"  Features to KEEP: {len(features_to_keep)}")
    print(f"  Features to EXCLUDE: {len(features_to_exclude)}")
    print(f"  Total features analyzed: {len(merged_df)}")
    
    # Create JSON structure
    filter_json = {
        "description": f"Hybrid feature filter combining timing analysis and feature importance. Excludes pure post-target leakage (>= {post_f1120_threshold:.0%} post-F1120) and keeps features with pre-F1120 presence, prioritizing those with high importance.",
        "version": "1.0",
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        "cohort": cohort,
        "age_band": age_band,
        "post_f1120_threshold": post_f1120_threshold,
        "min_pre_ratio": min_pre_ratio,
        "min_importance": min_importance,
        "approach": "hybrid_timing_and_importance",
        "total_features_to_keep": len(features_to_keep),
        "total_features_to_exclude": len(features_to_exclude),
        "total_features_analyzed": len(merged_df),
        "categorization": {
            "pure_leakage": len(pure_leakage),
            "pure_predictive": len(pure_predictive),
            "mixed_timing_kept": len(mixed_keep),
            "mixed_timing_excluded": len(mixed_exclude),
            "low_pre_ratio": len(low_pre)
        },
        "usage": {
            "cases": "Use features from 'all_features_to_keep' list",
            "controls": "Use the SAME features from 'all_features_to_keep' list",
            "rationale": "Hybrid approach balances preventing leakage with preserving potentially important features that have some pre-F1120 presence"
        },
        "all_features_to_keep": sorted(features_to_keep['feature'].tolist()),
        "all_features_to_exclude": sorted(features_to_exclude['feature'].tolist()),
        "features_by_category": {
            "pure_predictive": sorted(pure_predictive['feature'].tolist()),
            "mixed_timing_kept": sorted(mixed_keep['feature'].tolist()) if len(mixed_keep) > 0 else [],
            "pure_leakage": sorted(pure_leakage['feature'].tolist())
        }
    }
    
    # Add detailed breakdown
    if len(mixed_keep) > 0:
        filter_json["mixed_timing_features_details"] = mixed_keep[[
            'feature', 'pre_f1120_ratio', 'post_f1120_ratio', 
            'pre_count', 'post_count', 'total_count', 'feature_importance'
        ]].to_dict('records')
    
    # Save JSON file
    output_dir = project_root / "3b_feature_importance_eda" / "outputs" / cohort / age_band_fname
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{cohort}_{age_band_fname}_hybrid_feature_filter.json"
    
    with open(output_path, 'w') as f:
        json.dump(filter_json, f, indent=2)
    
    print(f"\n[OK] Saved hybrid feature filter to: {output_path}")
    
    # Show top mixed-timing features we're keeping
    if len(mixed_keep) > 0:
        print(f"\nTop 10 mixed-timing features being kept (by importance or count):")
        top_mixed = mixed_keep.nlargest(10, 'feature_importance' if min_importance > 0 else 'total_count')[
            ['feature', 'pre_f1120_ratio', 'post_f1120_ratio', 'total_count', 'feature_importance']
        ]
        for _, row in top_mixed.iterrows():
            print(f"  {row['feature']}: {row['pre_f1120_ratio']*100:.1f}% pre, "
                  f"{int(row['total_count'])} events, importance={row['feature_importance']:.4f}")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create hybrid feature filter JSON")
    parser.add_argument("--cohort", default="opioid_ed", help="Cohort name")
    parser.add_argument("--age-band", default="13-24", help="Age band")
    parser.add_argument(
        "--post-f1120-threshold",
        type=float,
        default=0.8,
        help="Threshold for post-F1120 ratio to flag as leakage (default: 0.8 = 80%%)"
    )
    parser.add_argument(
        "--min-pre-ratio",
        type=float,
        default=0.0,
        help="Minimum pre-F1120 ratio to keep mixed-timing features (default: 0.0 = keep if any pre presence)"
    )
    parser.add_argument(
        "--min-importance",
        type=float,
        default=0.0,
        help="Minimum feature importance to keep mixed-timing features (default: 0.0 = keep all with pre presence)"
    )
    
    args = parser.parse_args()
    create_hybrid_feature_filter_json(
        args.cohort,
        args.age_band,
        args.post_f1120_threshold,
        args.min_pre_ratio,
        args.min_importance
    )
