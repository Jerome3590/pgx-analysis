#!/usr/bin/env python3
"""
Create safe feature filter JSON: Exclude post-target leakage, keep all pre-target features.

This script:
1. Loads bupar_post_target_analysis.csv
2. Excludes features with >=80% post-F1120 ratio (pure leakage)
3. Keeps ALL features with ANY pre-F1120 presence (maximize information)
4. Creates a JSON file with features to KEEP for both cases and controls
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import age_band_to_fname


def categorize_feature(feature: str) -> tuple:
    """Categorize a feature by type and extract the code."""
    if feature.startswith('item_icd_'):
        code = feature.replace('item_icd_', '')
        return ('ICD', code)
    elif feature.startswith('item_cpt_'):
        code = feature.replace('item_cpt_', '')
        return ('CPT', code)
    elif feature.startswith('item_drug_'):
        drug = feature.replace('item_drug_', '')
        return ('Drug', drug)
    else:
        return ('Unknown', feature)


def create_safe_feature_filter_json(
    cohort: str,
    age_band: str,
    post_f1120_threshold: float = 0.8,
    min_events: int = 1  # Keep features with at least 1 event
):
    """Create safe feature filter: exclude post-target leakage, keep all pre-target features."""
    age_band_fname = age_band_to_fname(age_band)
    project_root = Path(__file__).parent.parent
    
    # Load BupaR analysis results
    analysis_path = project_root / "3b_feature_importance_eda" / "outputs" / cohort / age_band_fname / f"{cohort}_{age_band_fname}_bupar_post_target_analysis.csv"
    
    if not analysis_path.exists():
        print(f"[ERROR] Analysis file not found: {analysis_path}")
        print(f"       Run create_bupar_post_target_analysis.py first")
        return None
    
    print(f"\n{'='*80}")
    print(f"Creating Safe Feature Filter JSON")
    print(f"Cohort: {cohort} / Age Band: {age_band}")
    print(f"Strategy: Exclude post-target leakage (>= {post_f1120_threshold:.0%} post-F1120)")
    print(f"          Keep ALL features with ANY pre-F1120 presence")
    print(f"{'='*80}\n")
    
    df = pd.read_csv(analysis_path)
    
    # Filter: Exclude pure post-target leakage (>=80% post-F1120)
    # Keep everything else (including mixed-timing features with any pre-F1120 presence)
    post_leakage = df[df['post_f1120_ratio'] >= post_f1120_threshold].copy()
    
    # Features to keep: everything that's NOT pure post-target leakage
    # This includes:
    # - Pure pre-target features (>=80% pre-F1120)
    # - Mixed-timing features (any pre-F1120 presence, <80% post-F1120)
    features_to_keep = df[df['post_f1120_ratio'] < post_f1120_threshold].copy()
    
    # Ensure minimum event count
    features_to_keep = features_to_keep[features_to_keep['total_count'] >= min_events].copy()
    
    # IMPORTANT: Always include F1120 for target creation
    f1120_feature = 'item_icd_F1120'
    if f1120_feature not in features_to_keep['feature'].values:
        # Check if F1120 is in the leakage list (it shouldn't be, but just in case)
        f1120_row = df[df['feature'] == f1120_feature]
        if len(f1120_row) > 0:
            features_to_keep = pd.concat([features_to_keep, f1120_row], ignore_index=True)
            print(f"[INFO] Added {f1120_feature} to keep list (needed for target creation)")
    
    print(f"Feature breakdown:")
    print(f"  Total features analyzed: {len(df)}")
    print(f"  Post-target leakage (EXCLUDE): {len(post_leakage)} features")
    print(f"  Features to KEEP: {len(features_to_keep)} features")
    
    # Categorize kept features by timing
    features_to_keep['feature_type'] = features_to_keep['feature'].apply(lambda x: categorize_feature(x)[0])
    features_to_keep['timing_category'] = features_to_keep.apply(
        lambda row: 'pure_predictive' if row['pre_f1120_ratio'] >= 0.8 
                   else 'mixed_timing' if row['pre_f1120_ratio'] > 0
                   else 'low_pre_but_not_leakage',
        axis=1
    )
    
    timing_counts = features_to_keep['timing_category'].value_counts()
    print(f"\n  Timing breakdown of kept features:")
    for category, count in timing_counts.items():
        print(f"    {category}: {count} features")
    
    # Create JSON structure
    filter_json = {
        "description": f"Safe feature filter: Excludes post-target leakage (>= {post_f1120_threshold:.0%} post-F1120) and keeps ALL features with ANY pre-F1120 presence. This maximizes information available to the algorithm while preventing target leakage. Same feature set applied to both cases and controls.",
        "version": "1.0",
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        "cohort": cohort,
        "age_band": age_band,
        "post_f1120_threshold": post_f1120_threshold,
        "min_events": min_events,
        "approach": "exclude_post_target_keep_all_pre",
        "total_features_to_keep": len(features_to_keep),
        "total_features_to_exclude": len(post_leakage),
        "total_features_analyzed": len(df),
        "strategy": {
            "exclude": "Features with >= 80% post-F1120 ratio (pure post-target leakage)",
            "keep": "All features with < 80% post-F1120 ratio (includes pure pre-target, mixed-timing, and low-pre features)",
            "rationale": "Maximize information for training while preventing target leakage. Keeping mixed-timing features ensures algorithm has access to all potentially predictive signals."
        },
        "usage": {
            "cases": "Use ONLY features from 'all_features_to_keep' list",
            "controls": "Use the SAME features from 'all_features_to_keep' list",
            "rationale": "Same feature set ensures fair comparison and prevents bias"
        },
        "all_features_to_keep": sorted(features_to_keep['feature'].tolist()),
        "all_features_to_exclude": sorted(post_leakage['feature'].tolist()),
        "features_by_type": {
            "ICD": sorted(features_to_keep[features_to_keep['feature_type'] == 'ICD']['feature'].tolist()),
            "CPT": sorted(features_to_keep[features_to_keep['feature_type'] == 'CPT']['feature'].tolist()),
            "Drug": sorted(features_to_keep[features_to_keep['feature_type'] == 'Drug']['feature'].tolist())
        },
        "features_by_timing": {
            "pure_predictive": sorted(features_to_keep[features_to_keep['timing_category'] == 'pure_predictive']['feature'].tolist()),
            "mixed_timing": sorted(features_to_keep[features_to_keep['timing_category'] == 'mixed_timing']['feature'].tolist()),
            "low_pre_but_not_leakage": sorted(features_to_keep[features_to_keep['timing_category'] == 'low_pre_but_not_leakage']['feature'].tolist())
        },
        "summary": {
            "pure_predictive_count": len(features_to_keep[features_to_keep['timing_category'] == 'pure_predictive']),
            "mixed_timing_count": len(features_to_keep[features_to_keep['timing_category'] == 'mixed_timing']),
            "low_pre_count": len(features_to_keep[features_to_keep['timing_category'] == 'low_pre_but_not_leakage']),
            "post_leakage_count": len(post_leakage)
        }
    }
    
    # Save JSON file
    output_dir = project_root / "3b_feature_importance_eda" / "outputs" / cohort / age_band_fname
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{cohort}_{age_band_fname}_safe_feature_filter.json"
    
    with open(output_path, 'w') as f:
        json.dump(filter_json, f, indent=2)
    
    print(f"\n[OK] Saved safe feature filter to: {output_path}")
    print(f"\nSummary:")
    print(f"  Features to KEEP: {len(features_to_keep)}")
    print(f"    - Pure predictive (>=80% pre): {filter_json['summary']['pure_predictive_count']}")
    print(f"    - Mixed timing (any pre, <80% post): {filter_json['summary']['mixed_timing_count']}")
    print(f"    - Low pre but not leakage: {filter_json['summary']['low_pre_count']}")
    print(f"  Features to EXCLUDE: {len(post_leakage)} (post-target leakage)")
    print(f"\n  By type:")
    for ftype in ['ICD', 'CPT', 'Drug']:
        count = len(filter_json["features_by_type"][ftype])
        if count > 0:
            print(f"    {ftype}: {count}")
    
    print(f"\n[INFO] This approach:")
    print(f"  - Excludes {len(post_leakage)} post-target leakage features")
    print(f"  - Keeps {len(features_to_keep)} features with pre-F1120 presence")
    print(f"  - Maximizes information available to the algorithm")
    print(f"  - Same feature set for cases and controls")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create safe feature filter: exclude post-target leakage, keep all pre-target features")
    parser.add_argument("--cohort", default="opioid_ed", help="Cohort name")
    parser.add_argument("--age-band", default="13-24", help="Age band")
    parser.add_argument(
        "--post-f1120-threshold",
        type=float,
        default=0.8,
        help="Threshold for post-F1120 ratio to flag as leakage (default: 0.8 = 80%%)"
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=1,
        help="Minimum number of events required (default: 1)"
    )
    
    args = parser.parse_args()
    create_safe_feature_filter_json(
        args.cohort,
        args.age_band,
        args.post_f1120_threshold,
        args.min_events
    )
