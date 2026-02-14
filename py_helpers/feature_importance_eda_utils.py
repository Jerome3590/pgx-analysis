"""
Feature Importance EDA Utilities

Shared utility functions for Step 3b Feature Importance EDA.
Functions for loading feature importance files, admin codes, filters, and related data.

Canonical locations for aggregated feature importance (Step 3a output):
  Local (checked in order):
    1. PGX_FEATURE_IMPORTANCE_OUTPUTS / {cohort} / {cohort}_{age_band_fname}_aggregated_feature_importance.csv
    2. 3a_feature_importance/outputs/{cohort}/{filename}  (3a default write path, no age_band subdir)
    3. 3a_feature_importance/outputs/{cohort}/{age_band}/{filename}
    4. DATA_ROOT/gold/feature_importance/{cohort}/{age_band}/{filename}  (S3 sync layout; age_band with hyphen)
    5. 3a_feature_importance/from_s3/by_cohort/{cohort}/{age_band}/{filename}
  S3 (pgxdatalake):
    gold/feature_importance/{cohort}/{age_band}/{filename}  (age_band with hyphen, e.g. 65-74)
"""

import io
import json
import os
from pathlib import Path
from typing import Optional, Set, Tuple

import pandas as pd
import duckdb

from py_helpers.constants import age_band_to_fname


def resolve_aggregated_fi_path(
    cohort: str,
    age_band: str,
    project_root: Path,
) -> Optional[Path]:
    """
    Resolve path to 3a aggregated feature importance CSV.
    Tries local paths (see module docstring), then S3; downloads and saves if from S3.
    Returns Path if found, None otherwise.
    """
    age_band_fname = age_band_to_fname(age_band)
    filename = f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
    base_3a = project_root / "3a_feature_importance" / "outputs"
    possible = [
        base_3a / cohort / filename,
        base_3a / cohort / age_band / filename,
        project_root / "3a_feature_importance" / "from_s3" / "by_cohort" / cohort / age_band / filename,
    ]
    env_3a = os.environ.get("PGX_FEATURE_IMPORTANCE_OUTPUTS")
    if env_3a:
        possible.insert(0, Path(env_3a) / cohort / filename)
    # DATA_ROOT/gold/feature_importance (S3 sync layout; age_band with hyphen)
    try:
        from py_helpers.env_utils import get_data_root
        data_root = get_data_root()
        possible.append(data_root / "gold" / "feature_importance" / cohort / age_band / filename)
    except ImportError:
        pass
    for p in possible:
        if p.exists():
            return p
    try:
        try:
            from py_helpers.common_imports import s3_client, S3_BUCKET
        except ImportError:
            import boto3
            s3_client = boto3.client("s3")
            S3_BUCKET = "pgxdatalake"
        key = f"gold/feature_importance/{cohort}/{age_band}/{filename}"
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        save_path = base_3a / cohort / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        return save_path
    except Exception:
        return None


def load_aggregated_fi(
    cohort: str,
    age_band: str,
    project_root: Path,
) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    """
    Load 3a aggregated feature importance. Uses resolve_aggregated_fi_path then reads CSV.
    Returns (dataframe, path) or (None, None) if not found.
    """
    path = resolve_aggregated_fi_path(cohort, age_band, project_root)
    if path is None:
        return None, None
    df = pd.read_csv(path)
    return df, path


# Administrative codes lookup: same candidate order as workflow and create_bupar_input.
ADMIN_LOOKUP_RELATIVE = [
    "4b_event_filter/administrative_codes_lookup.json",
    "1b_apcd_event_filter/administrative_codes_lookup.json",
    "3b_feature_importance_eda/0_icd_cpt_check/administrative_codes_lookup.json",
]


def get_administrative_lookup_path(project_root: Path) -> Optional[Path]:
    """Return first existing administrative_codes_lookup.json path, or None."""
    for rel in ADMIN_LOOKUP_RELATIVE:
        p = project_root / rel
        if p.exists():
            return p
    return None


def load_administrative_codes(project_root: Path) -> Set[str]:
    """Load administrative codes (ICD/CPT/HCPCS) as a set for filtering."""
    path = get_administrative_lookup_path(project_root)
    if not path:
        return set()
    try:
        with open(path) as f:
            data = json.load(f)
        codes = data.get("administrative_codes", {})
        out = set()
        for key in ("icd", "cpt", "hcpcs"):
            out.update(codes.get(key, []))
        return out
    except Exception:
        return set()


def load_aggregated_feature_importance(
    cohort: str,
    age_band: str,
    project_root: Path,
) -> pd.DataFrame:
    """
    Load aggregated feature importance from Step 3a.
    Uses resolve_aggregated_fi_path (local + S3), then falls back to legacy paths.
    Raises FileNotFoundError if not found; raises ValueError if file is empty (0 rows or no feature column).
    """
    path = resolve_aggregated_fi_path(cohort, age_band, project_root)
    if path:
        df = pd.read_csv(path)
        if df.empty or len(df) == 0:
            raise ValueError(
                f"Aggregated feature importance file is empty (0 rows): {path}\n"
                f"  Run Step 3a for {cohort}/{age_band} to produce a non-empty file, or fix the source file."
            )
        if "feature" not in df.columns and len(df.columns) < 2:
            raise ValueError(
                f"Aggregated feature importance file has no 'feature' column or importance column: {path}\n"
                f"  Expected CSV with at least columns: feature, and one of importance_mean / importance_scaled_by_model_sum."
            )
        return df
    age_band_fname = age_band_to_fname(age_band)
    legacy = [
        project_root / "3a_feature_importance" / "outputs" / cohort / age_band / f"{cohort}_{age_band_fname}_aggregated_feature_importance.parquet",
        project_root / "3a_feature_importance" / "outputs" / cohort / age_band / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv",
    ]
    for p in legacy:
        if p.exists():
            if p.suffix.lower() == ".csv":
                df = pd.read_csv(p)
            else:
                con = duckdb.connect()
                path_esc = str(p).replace("'", "''")
                df = con.execute(f"SELECT * FROM read_parquet('{path_esc}')").df()
                con.close()
            if df.empty or len(df) == 0:
                raise ValueError(
                    f"Aggregated feature importance file is empty (0 rows): {p}\n"
                    f"  Run Step 3a for {cohort}/{age_band} to produce a non-empty file."
                )
            return df
    raise FileNotFoundError(
        f"Could not find aggregated feature importance file for {cohort}/{age_band}. "
        "Checked: PGX_FEATURE_IMPORTANCE_OUTPUTS, 3a_feature_importance/outputs, DATA_ROOT/gold/feature_importance, from_s3, S3 gold/feature_importance."
    )


def resolve_cohort_fi_path(
    cohort: str,
    age_band: str,
    project_root: Path,
) -> Optional[Path]:
    """
    Resolve path to Step 3b refined cohort_feature_importance CSV (leakage-filtered).
    Used by Step 4 (model_events filter) and Step 6 (final model features); must match.
    Tries local 3b outputs, DATA_ROOT/gold, then S3.
    """
    age_band_fname = age_band_to_fname(age_band)
    filename = f"{cohort}_{age_band_fname}_cohort_feature_importance.csv"
    base_3b = project_root / "3b_feature_importance_eda" / "outputs"
    possible = [
        base_3b / cohort / age_band_fname / filename,
        base_3b / cohort / age_band / filename,
    ]
    try:
        from py_helpers.env_utils import get_data_root
        data_root = get_data_root()
        possible.append(data_root / "gold" / "feature_importance" / cohort / age_band / filename)
        possible.append(data_root / "gold" / "feature_importance" / cohort / age_band_fname / filename)
    except ImportError:
        pass
    for p in possible:
        if p.exists():
            return p
    try:
        try:
            from py_helpers.common_imports import s3_client, S3_BUCKET
        except ImportError:
            import boto3
            s3_client = boto3.client("s3")
            S3_BUCKET = "pgxdatalake"
        key = f"gold/feature_importance/{cohort}/{age_band}/{filename}"
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        save_path = base_3b / cohort / age_band_fname / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        return save_path
    except Exception:
        return None


def load_cohort_feature_importance(
    cohort: str,
    age_band: str,
    project_root: Path,
) -> pd.DataFrame:
    """
    Load Step 3b refined cohort_feature_importance (leakage-filtered).
    Required for Step 4 (filter model_events) and Step 6 (final model features); same source everywhere.
    Raises FileNotFoundError if not found; ValueError if empty.
    """
    path = resolve_cohort_fi_path(cohort, age_band, project_root)
    if path is None:
        raise FileNotFoundError(
            f"Step 3b refined cohort_feature_importance not found for {cohort}/{age_band}. "
            "Run: python 3b_feature_importance_eda/run_feature_importance_eda.py --cohort <cohort> --age-band <age_band>"
        )
    df = pd.read_csv(path)
    if df.empty or len(df) == 0:
        raise ValueError(f"Step 3b cohort_feature_importance is empty: {path}")
    if "feature" not in df.columns:
        raise ValueError(f"Step 3b cohort_feature_importance has no 'feature' column: {path}")
    return df


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
