"""
Feature Importance EDA Utilities

Shared utility functions for Step 3b Feature Importance EDA.
Functions for loading feature importance files, filters, and related data.
"""

import duckdb
from pathlib import Path
from typing import Optional, Set, Tuple
import pandas as pd
import json

from py_helpers.constants import age_band_to_fname


def load_aggregated_feature_importance(
    cohort: str,
    age_band: str,
    project_root: Path
) -> pd.DataFrame:
    """
    Load aggregated feature importance from Step 3.
    
    Following cursor dev rules: Use DuckDB to read CSV/Parquet files instead of pandas.
    
    Args:
        cohort: Cohort name
        age_band: Age band
        project_root: Project root directory
    
    Returns:
        DataFrame with aggregated feature importance
    
    Raises:
        FileNotFoundError: If file not found in any expected location
    """
    age_band_fname = age_band_to_fname(age_band)
    
    # Try multiple locations (check for Parquet first, then CSV)
    # Step 3a outputs: 3a_feature_importance/outputs
    possible_paths = []
    # Check for Parquet files first (preferred format)
    possible_paths.extend([
        project_root / "3a_feature_importance" / "outputs" / cohort / age_band / f"{cohort}_{age_band_fname}_aggregated_feature_importance.parquet",
        project_root / "3a_feature_importance" / "from_s3" / "by_cohort" / cohort / age_band / f"{cohort}_{age_band_fname}_aggregated_feature_importance.parquet",
    ])
    # Fallback to CSV files
    possible_paths.extend([
        project_root / "3a_feature_importance" / "outputs" / cohort / age_band / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv",
        project_root / "3a_feature_importance" / "from_s3" / "by_cohort" / cohort / age_band / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv",
    ])
    
    for path in possible_paths:
        if path.exists():
            print(f"Loading aggregated feature importance from: {path}")
            con = duckdb.connect()
            path_str = str(path).replace("'", "''")
            if path.suffix.lower() == '.parquet':
                result = con.execute(f"SELECT * FROM read_parquet('{path_str}')").df()
            else:
                result = con.execute(f"SELECT * FROM read_csv_auto('{path_str}')").df()
            con.close()
            return result
    
    raise FileNotFoundError(f"Could not find aggregated feature importance file for {cohort}/{age_band}")


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
