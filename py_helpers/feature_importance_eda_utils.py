"""
Feature Importance EDA Utilities

Shared utility functions for Step 3b Feature Importance EDA.
Functions for loading feature importance files, admin codes, filters, and related data.

This module now uses the FileResolver pattern for consistent file resolution across
local paths, data root, and S3. See py_helpers.file_resolver for details.
"""

import io
import json
import os
from pathlib import Path
from typing import Optional, Set, Tuple, Dict, List

import pandas as pd
import duckdb

from py_helpers.constants import age_band_to_fname
from py_helpers.file_resolver import (
    FileResolver,
    resolve_cohort_fi_path as _resolve_cohort_fi_path,
    resolve_aggregated_fi_path as _resolve_aggregated_fi_path,
    load_cohort_feature_importance as _load_cohort_feature_importance,
    load_aggregated_feature_importance as _load_aggregated_feature_importance,
    get_administrative_lookup_path as _get_administrative_lookup_path,
    load_administrative_codes as _load_administrative_codes,
)


def resolve_aggregated_fi_path(
    cohort: str,
    age_band: str,
    project_root: Path,
) -> Optional[Path]:
    """
    Resolve path to 3a aggregated feature importance CSV.
    Uses FileResolver for consistent resolution across local, data root, and S3.
    Returns Path if found, None otherwise.
    """
    return _resolve_aggregated_fi_path(cohort, age_band, project_root)


def load_aggregated_fi(
    cohort: str,
    age_band: str,
    project_root: Path,
) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    """
    Load 3a aggregated feature importance. Uses FileResolver for resolution.
    Returns (dataframe, path) or (None, None) if not found.
    """
    path = _resolve_aggregated_fi_path(cohort, age_band, project_root)
    if path is None:
        return None, None
    df = pd.read_csv(path)
    return df, path


# Administrative codes lookup

def get_administrative_lookup_path(project_root: Path) -> Optional[Path]:
    """Return path to administrative_codes_lookup.json using FileResolver."""
    return _get_administrative_lookup_path(project_root)


def load_administrative_codes(project_root: Path) -> Set[str]:
    """Load administrative codes (ICD/CPT/HCPCS) as a set for filtering."""
    codes_dict = _load_administrative_codes(project_root)
    out = set()
    for key in ("icd", "cpt", "hcpcs"):
        out.update(codes_dict.get(key, []))
    return out


def load_aggregated_feature_importance(
    cohort: str,
    age_band: str,
    project_root: Path,
) -> pd.DataFrame:
    """
    Load aggregated feature importance from Step 3a.
    Uses FileResolver (local + S3), then falls back to legacy paths.
    Raises FileNotFoundError if not found; raises ValueError if file is empty.
    """
    return _load_aggregated_feature_importance(cohort, age_band, project_root)


def resolve_cohort_fi_path(
    cohort: str,
    age_band: str,
    project_root: Path,
) -> Optional[Path]:
    """
    Resolve path to Step 3b refined cohort_feature_importance CSV (leakage-filtered).
    Uses FileResolver for consistent resolution.
    Used by Step 4 (model_events filter) and Step 6 (final model features); must match.
    """
    return _resolve_cohort_fi_path(cohort, age_band, project_root)


def load_cohort_feature_importance(
    cohort: str,
    age_band: str,
    project_root: Path,
) -> pd.DataFrame:
    """
    Load Step 3b refined cohort_feature_importance (leakage-filtered).
    Uses FileResolver for consistent resolution.
    Required for Step 4 (filter model_events) and Step 6 (final model features); same source everywhere.
    Raises FileNotFoundError if not found; ValueError if empty.
    """
    return _load_cohort_feature_importance(cohort, age_band, project_root)


def load_safe_feature_filter(
    cohort: str,
    age_band: str,
    output_dir: Path
) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
    """
    Load safe feature filter JSON file.
    
    Returns tuple: (features_to_keep_for_cases, features_to_exclude_for_controls)
    - features_to_keep: Whitelist of features to keep for cases (pre-target predictive features)
    - features_to_exclude: Blacklist of features to exclude for controls (post-target leakage features)
    
    Normalizes feature names to match aggregated importance format:
    - item_cpt_80307 -> item_80307
    - item_drug_SUBOXONE -> item_SUBOXONE
    - item_icd_F1120 -> item_F1120
    
    Args:
        cohort: Cohort name
        age_band: Age band
        output_dir: Output directory where filter JSON should be located
    
    Returns:
        Tuple of (features_to_keep, features_to_exclude), both as normalized Sets or None if file not found
    """
    from py_helpers.feature_utils import normalize_feature_set
    
    age_band_fname = age_band_to_fname(age_band)
    filter_json_path = output_dir / f"{cohort}_{age_band_fname}_safe_feature_filter.json"
    
    if not filter_json_path.exists():
        print(f"[WARN] Safe feature filter not found: {filter_json_path}")
        print(f"       Will fall back to BupaR CSV-based filtering")
        return None, None
    
    try:
        print(f"Loading safe feature filter from: {filter_json_path}")
        with open(filter_json_path, 'r') as f:
            filter_data = json.load(f)
        
        # Extract and normalize feature sets
        features_to_keep_raw = filter_data.get('all_features_to_keep', [])
        features_to_exclude_raw = filter_data.get('all_features_to_exclude', [])
        
        # Normalize feature names to match aggregated importance format
        features_to_keep = normalize_feature_set(set(features_to_keep_raw))
        features_to_exclude = normalize_feature_set(set(features_to_exclude_raw))
        
        print(f"  Found {len(features_to_keep_raw)} features to keep (for cases - whitelist)")
        print(f"  Found {len(features_to_exclude_raw)} features to exclude (for controls - blacklist)")
        print(f"  Normalized: {len(features_to_keep)} keep, {len(features_to_exclude)} exclude")
        
        return features_to_keep, features_to_exclude
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[WARN] Error reading filter JSON: {e}")
        return None, None
