#!/usr/bin/env python3
"""
Filter and Refine Feature Importances

Combines BupaR post-target analysis to filter
and refine aggregated feature importances from Step 3.

Outputs refined cohort_feature_importance files for Step 4a.
"""

import argparse
import sys
import re
import os
import platform
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd
import json

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

from py_helpers.constants import age_band_to_fname, DRUG_NAMES_EXCLUDED_MODEL_TRAINING
from py_helpers.feature_utils import (
    normalize_feature_name,
    normalize_feature_set,
    sanitize_feature_names,
    sanitize_column_names,
    feature_to_code,
    feature_to_code_type,
    code_to_canonical_feature_name,
    is_substance_use_disorder_code,
)
from py_helpers.feature_importance_eda_utils import (
    load_aggregated_feature_importance,
    load_safe_feature_filter
)

try:
    from py_helpers.common_imports import s3_client, S3_BUCKET
except ImportError:
    import boto3
    s3_client = boto3.client("s3")
    S3_BUCKET = "pgxdatalake"

from py_helpers.checkpoint_utils import upload_file_to_s3


# load_aggregated_feature_importance moved to py_helpers.feature_importance_eda_utils


# sanitize_column_names moved to py_helpers.feature_utils


# sanitize_feature_names, normalize_feature_name, and normalize_feature_set moved to py_helpers.feature_utils


# load_safe_feature_filter moved to py_helpers.feature_importance_eda_utils


def _leakage_feature_set_from_bupar(bupar_results: pd.DataFrame) -> Set[str]:
    """
    Get set of post-target leakage feature names from BupaR CSV.
    Accepts is_post_target_leakage in (1, True, "1", "True"). If column missing,
    uses post_target_ratio or post_f1120_ratio >= 0.8.
    """
    if bupar_results.empty or "feature" not in bupar_results.columns:
        return set()
    # Normalize column names for lookup
    cols_lower = {c.lower(): c for c in bupar_results.columns}
    feature_col = cols_lower.get("feature", "feature")
    df = bupar_results.copy()

    # Leakage flag: accept 1, True, "1", "True"
    leak_col = None
    for c in df.columns:
        if c.lower() == "is_post_target_leakage":
            leak_col = c
            break

    if leak_col is not None:
        leak_vals = df[leak_col]
        mask = leak_vals.astype(str).str.strip().str.lower().isin(("1", "true"))
        try:
            mask = mask | (pd.to_numeric(leak_vals, errors="coerce") == 1).fillna(False)
        except Exception:
            pass
        raw = set(df.loc[mask, feature_col].dropna().astype(str).tolist())
        if raw:
            return raw

    # Fallback: use ratio column >= 0.8
    ratio_col = None
    for c in df.columns:
        if c in ("post_target_ratio", "post_f1120_ratio"):
            ratio_col = c
            break
    if ratio_col is None:
        for c in df.columns:
            if c.lower() in ("post_target_ratio", "post_f1120_ratio"):
                ratio_col = c
                break
    if ratio_col is not None:
        raw = set(df.loc[df[ratio_col].astype(float) >= 0.8, feature_col].dropna().astype(str).tolist())
        return raw

    return set()


def filter_and_refine_features(
    aggregated_fi: pd.DataFrame,
    bupar_results: pd.DataFrame,
    filter_post_target: bool = True,
    min_importance_threshold: float = 0.0,
    safe_feature_filter: Optional[tuple[Set[str], Optional[Set[str]]]] = None,
    cohort: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter and refine feature importances based on EDA results.
    
    Args:
        aggregated_fi: Aggregated feature importance DataFrame from Step 3
        bupar_results: BupaR post-target analysis results
        filter_post_target: Whether to filter post-target leakage features
        min_importance_threshold: Minimum importance threshold to keep
        safe_feature_filter: Tuple of (features_to_keep_for_cases, features_to_exclude_for_controls)
        cohort: Cohort name (e.g. opioid_ed). When opioid_ed, excludes F11.x target-family codes from features.
    
    Returns:
        Refined feature importance DataFrame
    """
    # Sanitize column names and feature names
    aggregated_fi = sanitize_column_names(aggregated_fi)
    aggregated_fi = sanitize_feature_names(aggregated_fi)
    
    refined_fi = aggregated_fi.copy()
    # Resolve feature column (aggregated CSV may have "feature" or first column as feature names)
    feature_col = "feature" if "feature" in refined_fi.columns else refined_fi.columns[0]
    if feature_col != "feature":
        refined_fi = refined_fi.rename(columns={feature_col: "feature"})
        feature_col = "feature"
    
    # Track filtering decisions
    filtering_summary = {
        'original_count': len(refined_fi),
        'filtered_by_post_target': 0,
        'filtered_by_target_family': 0,
        'filtered_by_threshold': 0,
        'filtered_by_safe_filter': 0,
        'filtered_by_drug_name_exclusion': 0,
        'final_count': 0
    }

    # Exclude drug-name values that are not drugs or not used as features (see DRUG_NAMES_EXCLUDED_MODEL_TRAINING)
    code_col = refined_fi["feature"].astype(str).str.replace("^item_", "", regex=True)
    drug_excluded_mask = code_col.isin(DRUG_NAMES_EXCLUDED_MODEL_TRAINING)
    if drug_excluded_mask.any():
        n_drug_excluded = int(drug_excluded_mask.sum())
        filtering_summary["filtered_by_drug_name_exclusion"] = n_drug_excluded
        refined_fi = refined_fi[~drug_excluded_mask].copy()
        print(f"Excluded {n_drug_excluded} drug-name feature(s) (not drugs or excluded from model training): {sorted(code_col[drug_excluded_mask].unique().tolist())}")
    
    # Use safe feature filter if available
    # safe_feature_filter is a tuple: (features_to_keep_for_cases, features_to_exclude_for_controls)
    if safe_feature_filter is not None and filter_post_target:
        features_to_keep, features_to_exclude = safe_feature_filter
        
        if features_to_keep is not None:
            before_count = len(refined_fi)
            
            # Normalize feature names for comparison
            refined_fi['feature_normalized'] = refined_fi['feature'].apply(normalize_feature_name)
            
            # Apply whitelist for cases: keep only features in the whitelist
            # Controls will use blacklist (exclude only leakage) - handled separately in Step 4a
            refined_fi = refined_fi[refined_fi['feature_normalized'].isin(features_to_keep)].copy()
            
            # Drop the temporary normalized column
            if 'feature_normalized' in refined_fi.columns:
                refined_fi = refined_fi.drop(columns=['feature_normalized'])
            
            filtering_summary['filtered_by_safe_filter'] = before_count - len(refined_fi)
            filtering_summary['filtered_by_post_target'] = filtering_summary['filtered_by_safe_filter']
            
            print(f"Applied safe feature filter (whitelist for cases): kept {len(refined_fi)} features")
            print(f"  Excluded {filtering_summary['filtered_by_safe_filter']} features (post-target leakage + not in whitelist)")
            if features_to_exclude:
                print(f"  NOTE: Controls will use blacklist approach (exclude only {len(features_to_exclude)} leakage features, keep all other features)")
            if filtering_summary['filtered_by_safe_filter'] == 0:
                agg_sample = refined_fi['feature'].dropna().head(15).tolist()
                keep_sample = sorted(features_to_keep)[:15] if features_to_keep else []
                print(f"  [DIAG] Filtered=0: aggregated 'feature' sample (raw): {agg_sample}")
                print(f"  [DIAG] Filtered=0: safe filter 'features_to_keep' sample: {keep_sample}")
    
    # Fallback: exclude leakage from BupaR CSV (robust to column names and flag values)
    elif filter_post_target and not bupar_results.empty:
        # No safe filter found; log column name samples for both sources
        bupar_feature_col = next((c for c in bupar_results.columns if c.lower() == "feature"), bupar_results.columns[0] if len(bupar_results.columns) else None)
        agg_raw_sample = refined_fi['feature'].dropna().head(15).tolist()
        bupar_raw_sample = bupar_results[bupar_feature_col].dropna().head(15).tolist() if bupar_feature_col else []
        print(f"  [DIAG] No safe filter; using BupaR fallback. Aggregated 'feature' sample (raw): {agg_raw_sample}")
        print(f"  [DIAG] No safe filter; BupaR CSV '{bupar_feature_col}' sample (raw): {bupar_raw_sample}")

        post_target_features_raw = _leakage_feature_set_from_bupar(bupar_results)
        post_target_features = normalize_feature_set(post_target_features_raw)

        before_count = len(refined_fi)
        refined_fi['feature_normalized'] = refined_fi['feature'].apply(normalize_feature_name)
        agg_norm_all = set(refined_fi['feature_normalized'].dropna().tolist())
        overlap_count = len(agg_norm_all & post_target_features)
        refined_fi = refined_fi[~refined_fi['feature_normalized'].isin(post_target_features)].copy()
        if 'feature_normalized' in refined_fi.columns:
            refined_fi = refined_fi.drop(columns=['feature_normalized'])

        filtering_summary['filtered_by_post_target'] = before_count - len(refined_fi)
        if post_target_features:
            print(f"Filtered {filtering_summary['filtered_by_post_target']} post-target leakage features (fallback: {len(post_target_features)} leakage features from BupaR CSV)")
            print(f"  [DIAG] Aggregated FI normalized set size: {len(agg_norm_all)}; overlap with leakage set: {overlap_count}")
            print(f"  [DIAG] Aggregated normalized sample: {sorted(agg_norm_all)[:15]}")
            print(f"  [DIAG] Leakage normalized sample: {sorted(post_target_features)[:15]}")
            if filtering_summary['filtered_by_post_target'] == 0:
                print(f"  [DIAG] Filtered=0: aggregated 'feature' sample (raw): {agg_raw_sample}")
                print(f"  [DIAG] Filtered=0: BupaR leakage feature sample (raw): {sorted(post_target_features_raw)[:15]}")
        else:
            has_leakage_cols = any(
                c.lower() == "is_post_target_leakage"
                or c.lower() in ("post_target_ratio", "post_f1120_ratio")
                for c in bupar_results.columns
            )
            if has_leakage_cols:
                print(f"[INFO] No post-target leakage features in BupaR results (0 flagged).")
                if cohort and str(cohort).strip().lower() == "non_opioid_ed":
                    print(f"[INFO] Expected for polypharmacy: model_events are built with drug events only up to the windowed HCG/ED visit, so no post-target leakage by construction.")
            else:
                print(f"[WARN] BupaR CSV has no leakage columns (is_post_target_leakage or post_*_ratio). Check create_bupar_post_target_analysis was run for this cohort/age_band.")
    
    # Exclude target-family codes (F10/F11/F19 substance use disorder) for opioid_ed — they are outcome, not predictors
    if cohort and (str(cohort).strip().lower() == "opioid_ed"):
        before_count = len(refined_fi)
        refined_fi["_code"] = refined_fi["feature"].apply(feature_to_code)
        refined_fi = refined_fi[~refined_fi["_code"].apply(is_substance_use_disorder_code)].copy()
        refined_fi = refined_fi.drop(columns=["_code"], errors="ignore")
        n_target_family = before_count - len(refined_fi)
        filtering_summary["filtered_by_target_family"] = n_target_family
        if n_target_family > 0:
            print(f"Excluded {n_target_family} target-family (F10/F11/F19 substance use disorder) features — outcome codes, not predictors")
    
    # Filter by minimum importance threshold
    if 'importance_scaled_by_model_sum' in refined_fi.columns:
        importance_col = 'importance_scaled_by_model_sum'
    elif 'importance_mean' in refined_fi.columns:
        importance_col = 'importance_mean'
    else:
        importance_col = refined_fi.columns[1]  # Use second column as fallback
    
    before_count = len(refined_fi)
    refined_fi = refined_fi[refined_fi[importance_col] >= min_importance_threshold]
    filtering_summary['filtered_by_threshold'] = before_count - len(refined_fi)
    
    filtering_summary['final_count'] = len(refined_fi)
    
    # Sort by importance
    refined_fi = refined_fi.sort_values(importance_col, ascending=False).reset_index(drop=True)
    
    return refined_fi, filtering_summary


def main():
    parser = argparse.ArgumentParser(
        description="Filter and refine feature importances based on EDA"
    )
    parser.add_argument("--cohort", required=True, help="Cohort name")
    parser.add_argument("--age-band", required=True, help="Age band")
    parser.add_argument(
        "--bupar-results",
        type=str,
        default=None,
        help="Path to BupaR results CSV (default: auto-detect)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: 3b_feature_importance_eda/outputs)"
    )
    parser.add_argument(
        "--min-importance",
        type=float,
        default=0.0,
        help="Minimum importance threshold (default: 0.0)"
    )
    parser.add_argument(
        "--no-filter-post-target",
        action="store_true",
        help="Don't filter post-target leakage features"
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "3b_feature_importance_eda" / "outputs" / args.cohort / age_band_to_fname(args.age_band)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    age_band_fname = age_band_to_fname(args.age_band)
    
    # Load aggregated feature importance
    print("=" * 80)
    print(f"Filtering and Refining Features: {args.cohort} / {args.age_band}")
    print("=" * 80)
    
    aggregated_fi = load_aggregated_feature_importance(args.cohort, args.age_band, PROJECT_ROOT)
    print(f"Loaded {len(aggregated_fi)} features from aggregated importance")
    
    # Load BupaR results
    if args.bupar_results:
        bupar_path = Path(args.bupar_results)
    else:
        bupar_path = output_dir / f"{args.cohort}_{age_band_fname}_bupar_post_target_analysis.csv"
    
    bupar_results = pd.DataFrame()
    if bupar_path.exists():
        print(f"Loading BupaR results from: {bupar_path}")
        bupar_results = pd.read_csv(bupar_path)
    else:
        print(f"[WARN] BupaR results not found: {bupar_path}")
    
    # Load safe feature filter (preferred approach)
    # Returns tuple: (features_to_keep_for_cases, features_to_exclude_for_controls)
    safe_feature_filter = None
    if not args.no_filter_post_target:
        features_to_keep, features_to_exclude = load_safe_feature_filter(args.cohort, args.age_band, output_dir)
        if features_to_keep is not None:
            safe_feature_filter = (features_to_keep, features_to_exclude)
            
            # Save control exclusions file for Step 4a
            if features_to_exclude is not None and len(features_to_exclude) > 0:
                control_exclusions_path = output_dir / f"{args.cohort}_{age_band_fname}_control_feature_exclusions.json"
                control_exclusions = {
                    "description": "Features to exclude for controls (blacklist approach). Controls keep all features except these post-target leakage features.",
                    "cohort": args.cohort,
                    "age_band": args.age_band,
                    "approach": "blacklist",
                    "features_to_exclude": sorted(list(features_to_exclude)),
                    "count": len(features_to_exclude)
                }
                with open(control_exclusions_path, 'w') as f:
                    json.dump(control_exclusions, f, indent=2)
                print(f"[OK] Saved control feature exclusions to: {control_exclusions_path}")
    
    # Filter and refine
    refined_fi, filtering_summary = filter_and_refine_features(
        aggregated_fi=aggregated_fi,
        bupar_results=bupar_results,
        filter_post_target=not args.no_filter_post_target,
        min_importance_threshold=args.min_importance,
        safe_feature_filter=safe_feature_filter,
        cohort=args.cohort,
    )
    
    # Save refined feature importance: add code_type and raw_code, and normalize "feature" to canonical
    # form (item_icd_X, item_cpt_X, item_drug_X) so Step 4 and Step 6 read ICD/CPT/drug correctly.
    refined_fi = refined_fi.copy()
    refined_fi["code_type"] = refined_fi["feature"].astype(str).map(lambda f: feature_to_code_type(f))
    refined_fi["raw_code"] = refined_fi["feature"].astype(str).map(lambda f: feature_to_code(f))
    # Canonical feature name so downstream never see item_80307 (ambiguous); they get item_cpt_80307
    def _canonical(row):
        ctype, raw = row["code_type"], row["raw_code"]
        canonical = code_to_canonical_feature_name(ctype, raw)
        return canonical if canonical else row["feature"]
    refined_fi["feature"] = refined_fi.apply(_canonical, axis=1)
    output_path = output_dir / f"{args.cohort}_{age_band_fname}_cohort_feature_importance.csv"
    refined_fi.to_csv(output_path, index=False)
    print(f"\nSaved refined feature importance to: {output_path}")
    print(f"Features: {len(refined_fi)} (down from {filtering_summary['original_count']})")
    
    # Upload to S3
    s3_path = f"s3://{S3_BUCKET}/gold/feature_importance/{args.cohort}/{args.age_band}/{args.cohort}_{age_band_fname}_cohort_feature_importance.csv"
    if upload_file_to_s3(output_path, s3_path, check_exists=True):
        print(f"[OK] Uploaded cohort_feature_importance to S3")
    else:
        print(f"[ERROR] Failed to upload cohort_feature_importance to S3")
        sys.exit(1)
    
    # Save filtering summary
    summary_path = output_dir / f"{args.cohort}_{age_band_fname}_feature_filtering_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(filtering_summary, f, indent=2)
    print(f"Saved filtering summary to: {summary_path}")
    
    # Upload summary to S3
    summary_s3_path = f"s3://{S3_BUCKET}/gold/feature_importance/{args.cohort}/{args.age_band}/{args.cohort}_{age_band_fname}_feature_filtering_summary.json"
    if upload_file_to_s3(summary_path, summary_s3_path, check_exists=True):
        print(f"[OK] Uploaded filtering summary to S3")
    else:
        print(f"[WARN] Failed to upload filtering summary to S3 (non-critical)")
    
    # Print summary
    print("\nFiltering Summary:")
    print(f"  Original features: {filtering_summary['original_count']}")
    if filtering_summary.get('filtered_by_safe_filter', 0) > 0:
        print(f"  Filtered by safe feature filter (whitelist): {filtering_summary['filtered_by_safe_filter']}")
    else:
        print(f"  Filtered by post-target: {filtering_summary['filtered_by_post_target']}")
    if filtering_summary.get('filtered_by_target_family', 0) > 0:
        print(f"  Filtered by target-family (F10/F11/F19): {filtering_summary['filtered_by_target_family']}")
    print(f"  Filtered by threshold: {filtering_summary['filtered_by_threshold']}")
    print(f"  Final features: {filtering_summary['final_count']}")
    
    print("\nTop 10 refined features:")
    print(refined_fi.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
