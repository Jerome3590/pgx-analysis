#!/usr/bin/env python3
"""
Create safe feature filter JSON: Exclude post-target leakage, keep all pre-target features.

Works for any target: F1120 (opioid use disorder) or HCG windowed (ED visit) target.

This script:
1. Loads bupar_post_target_analysis.csv
2. Excludes features with >=threshold post-target ratio (pure leakage)
3. Keeps ALL features with ANY pre-target presence (maximize information)
4. Creates a JSON file with features to KEEP for both cases and controls
"""

import argparse
import sys
import json
import os
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set
import pandas as pd

# Detect operating system and set project root
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'

if IS_WINDOWS:
    # Windows: Use current workspace directory (go up 2 levels: 2_filtering -> 3b_feature_importance_eda -> project root)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
elif IS_LINUX:
    # Linux/EC2: Use EC2 path
    PROJECT_ROOT = Path('/home/pgx3874/pgx-analysis')
else:
    # Fallback: Use current file's parent directory (go up 2 levels)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import age_band_to_fname, get_target_name_by_cohort, cohort_uses_f1120_target
from py_helpers.feature_utils import categorize_feature


# categorize_feature moved to py_helpers.feature_utils


def create_safe_feature_filter_json(
    cohort: str,
    age_band: str,
    post_target_threshold: float = 0.8,
    min_events: int = 1  # Keep features with at least 1 event
):
    """Create safe feature filter: exclude post-target leakage, keep all pre-target features.
    Works for F1120 target (age < 65) or HCG windowed target (age >= 65)."""
    age_band_fname = age_band_to_fname(age_band)
    project_root = PROJECT_ROOT
    # Target by cohort: opioid_ed=F1120, non_opioid_ed=HCG (polypharmacy / 14-day window)
    target_name = get_target_name_by_cohort(cohort)

    # Load BupaR analysis results
    analysis_path = project_root / "3b_feature_importance_eda" / "outputs" / cohort / age_band_fname / f"{cohort}_{age_band_fname}_bupar_post_target_analysis.csv"
    
    # Check if file exists in wrong location (features/ subdirectory) and move it
    wrong_location = project_root / "3b_feature_importance_eda" / "outputs" / cohort / age_band_fname / "features" / f"{cohort}_{age_band_fname}_bupar_post_target_analysis.csv"
    if wrong_location.exists() and not analysis_path.exists():
        print(f"[INFO] Found file in wrong location: {wrong_location}")
        print(f"       Moving to correct location: {analysis_path}")
        wrong_location.rename(analysis_path)
    elif wrong_location.exists() and analysis_path.exists():
        # Both exist - remove the one in wrong location
        print(f"[INFO] File exists in both locations. Removing wrong location: {wrong_location}")
        wrong_location.unlink()
    
    if not analysis_path.exists():
        print(f"[ERROR] Analysis file not found: {analysis_path}")
        print(f"       Run create_bupar_post_target_analysis.py first")
        return None
    
    print(f"\n{'='*80}")
    print(f"Creating Safe Feature Filter JSON")
    print(f"Cohort: {cohort} / Age Band: {age_band} / Target: {target_name}")
    print(f"Strategy: Exclude post-target leakage (>= {post_target_threshold:.0%} post-{target_name})")
    print(f"          Keep ALL features with ANY pre-{target_name} presence")
    print(f"{'='*80}\n")
    
    # Following cursor dev rules: Use DuckDB to read CSV/Parquet files instead of pandas
    import duckdb
    con = duckdb.connect()
    path_str = str(analysis_path).replace("'", "''")
    # Check for Parquet first (preferred), then CSV
    parquet_path = analysis_path.with_suffix(".parquet")
    if parquet_path.exists():
        parquet_path_str = str(parquet_path).replace("'", "''")
        df = con.execute(f"SELECT * FROM read_parquet('{parquet_path_str}')").df()
    else:
        df = con.execute(f"SELECT * FROM read_csv_auto('{path_str}')").df()
    con.close()

    # Support both generic (post_target_ratio) and legacy (post_f1120_ratio) column names
    post_ratio_col = "post_target_ratio" if "post_target_ratio" in df.columns else "post_f1120_ratio"
    pre_ratio_col = "pre_target_ratio" if "pre_target_ratio" in df.columns else "pre_f1120_ratio"
    if post_ratio_col not in df.columns:
        print(f"[ERROR] Analysis CSV must have 'post_target_ratio' or 'post_f1120_ratio'. Found: {list(df.columns)}")
        return None

    # Filter: Exclude pure post-target leakage (>=threshold post-target)
    # Keep everything else (including mixed-timing features with any pre-target presence)
    post_leakage = df[df[post_ratio_col] >= post_target_threshold].copy()

    # Features to keep: everything that's NOT pure post-target leakage
    features_to_keep = df[df[post_ratio_col] < post_target_threshold].copy()

    # Ensure minimum event count
    features_to_keep = features_to_keep[features_to_keep['total_count'] >= min_events].copy()

    # Target feature to always include (only opioid_ed has single ICD target F1120; non_opioid_ed uses HCG window)
    target_feature = "item_icd_F1120" if cohort_uses_f1120_target(cohort) else None
    if target_feature and target_feature not in features_to_keep['feature'].values:
        target_row = df[df['feature'] == target_feature]
        if len(target_row) > 0:
            features_to_keep = pd.concat([features_to_keep, target_row], ignore_index=True)
            print(f"[INFO] Added {target_feature} to keep list (needed for target creation)")
    
    print(f"Feature breakdown:")
    print(f"  Total features analyzed: {len(df)}")
    print(f"  Post-target leakage (EXCLUDE): {len(post_leakage)} features")
    print(f"  Features to KEEP: {len(features_to_keep)} features")
    
    # Categorize kept features by timing (use pre ratio column - may be pre_target_ratio or pre_f1120_ratio)
    features_to_keep['feature_type'] = features_to_keep['feature'].apply(lambda x: categorize_feature(x)[0])
    pre_col = pre_ratio_col if pre_ratio_col in features_to_keep.columns else None
    features_to_keep['timing_category'] = features_to_keep.apply(
        lambda row: 'pure_predictive' if (pre_col and row[pre_col] >= 0.8)
                   else 'mixed_timing' if (pre_col and row[pre_col] > 0)
                   else 'low_pre_but_not_leakage',
        axis=1
    )
    
    timing_counts = features_to_keep['timing_category'].value_counts()
    print(f"\n  Timing breakdown of kept features:")
    for category, count in timing_counts.items():
        print(f"    {category}: {count} features")
    
    # Create JSON structure
    filter_json = {
        "description": f"Safe feature filter: Excludes post-target leakage (>= {post_target_threshold:.0%} post-{target_name}) and keeps ALL features with ANY pre-{target_name} presence. This maximizes information available to the algorithm while preventing target leakage. Same feature set applied to both cases and controls.",
        "version": "1.0",
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        "cohort": cohort,
        "age_band": age_band,
        "post_target_threshold": post_target_threshold,
        "target_name": target_name,
        "min_events": min_events,
        "approach": "exclude_post_target_keep_all_pre",
        "total_features_to_keep": len(features_to_keep),
        "total_features_to_exclude": len(post_leakage),
        "total_features_analyzed": len(df),
        "strategy": {
            "exclude": f"Features with >= {post_target_threshold:.0%} post-{target_name} ratio (pure post-target leakage)",
            "keep": f"All features with < {post_target_threshold:.0%} post-{target_name} ratio (includes pure pre-target, mixed-timing, and low-pre features)",
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
    print(f"  - Keeps {len(features_to_keep)} features with pre-{target_name} presence")
    print(f"  - Maximizes information available to the algorithm")
    print(f"  - Same feature set for cases and controls")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create safe feature filter: exclude post-target leakage, keep all pre-target features")
    parser.add_argument("--cohort", default="opioid_ed", help="Cohort name")
    parser.add_argument("--age-band", default="13-24", help="Age band")
    parser.add_argument(
        "--post-target-threshold",
        type=float,
        default=0.8,
        dest="post_target_threshold",
        help="Threshold for post-target ratio to flag as leakage (default: 0.8 = 80%%)"
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
        args.post_target_threshold,
        args.min_events
    )
