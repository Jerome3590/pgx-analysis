#!/usr/bin/env python3
"""
Filter model_data events: first by aggregated feature importance, then by administrative codes.

Filter order:
1. Aggregated feature importance (first pass): Keep only events whose codes (ICD/CPT/drug) appear
   in the aggregated feature-importance CSV from Step 3/3a. This reduces the number of features
   for final cohorts and makes Step 3a feature importance more accurate on a second pass.
2. Administrative codes and post-event leakage (second pass): Remove administrative codes
   (from administrative_codes_lookup / research) and events occurring on or after target date.

The filtering is based on:
- Aggregated feature importance CSV (Step 3 or 3a: {cohort}_{age_band}_aggregated_feature_importance.csv)
- Administrative codes lookup and research (Step 3b 0_icd_cpt_check, protocol_vs_clinical analysis)
- Post-event leakage (Step 3b 1_bupaR)

Output: model_events_no_protocols.parquet - Event data with low-importance and administrative events removed.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.env_utils import get_data_root, is_linux  # noqa: E402

try:
    from py_helpers.common_imports import s3_client, S3_BUCKET  # noqa: E402
except ImportError:
    import boto3  # noqa: E402
    s3_client = boto3.client("s3")
    S3_BUCKET = "pgxdatalake"

OUTPUT_ROOT = PROJECT_ROOT / "1b_apcd_event_filter" / "outputs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _validate_s3_file_has_controls(s3_path: str) -> dict:
    """
    Validate that an S3 parquet file contains both cases (target=1) and controls (target=0).
    Uses DuckDB's S3 support to query without downloading the entire file.
    
    Returns:
        dict with keys: has_controls (bool), n_cases (int), n_controls (int), error (str or None)
    """
    con = duckdb.connect()
    try:
        result = con.execute(
            f"""
            SELECT 
                COUNT(*) FILTER (WHERE target = 1) AS n_cases,
                COUNT(*) FILTER (WHERE target = 0) AS n_controls
            FROM read_parquet('{s3_path}')
            """
        ).fetchone()
        
        n_cases = result[0] if result else 0
        n_controls = result[1] if result else 0
        has_controls = n_controls > 0
        
        return {
            "has_controls": has_controls,
            "n_cases": n_cases,
            "n_controls": n_controls,
            "error": None,
        }
    except Exception as e:
        return {
            "has_controls": False,
            "n_cases": 0,
            "n_controls": 0,
            "error": str(e),
        }
    finally:
        con.close()


def get_allowed_codes_from_aggregated_fi(agg_csv_path: Path) -> set:
    """
    Load allowed codes from aggregated feature importance CSV (for first-pass filter).

    The CSV has a 'feature' column; values may have an 'item_' prefix (e.g. item_99284,
    item_Z34.03). We strip that and build a set of allowed codes. For ICD-like codes
    (letter + digits) we add both dot and no-dot variants so events match regardless of format.

    Returns
    -------
    set
        Allowed code strings (including ICD dot/no-dot variants) for use in SQL IN (...).
    """
    df = pd.read_csv(agg_csv_path)
    if "feature" not in df.columns:
        return set()
    allowed = set()
    for raw in df["feature"].astype(str).unique():
        code = raw.strip()
        if code.startswith("item_"):
            code = code[5:]
        if not code or code == "nan":
            continue
        allowed.add(code)
        # ICD-style codes: add dot and no-dot variants for matching
        if code[0].isalpha() and any(c.isdigit() for c in code):
            allowed.add(code.replace(".", ""))
            if "." not in code and len(code) >= 4:
                allowed.add(f"{code[:3]}.{code[3:]}")
    return allowed


def _validate_and_filter_aggregated_feature_importance(
    cohort: str, age_band: str
) -> dict:
    """
    Validate and filter the aggregated feature importance CSV:
    - Filter out features with importance <= 0 (or <= 1e-10 for floating point)
    - Remove duplicate features (keep first/highest importance)
    - Save the cleaned CSV back to disk
    
    Returns:
        dict with keys: is_valid (bool), n_features_initial (int), n_features_final (int),
        n_zero_importance (int), n_duplicates (int), cleaned_path (Path), error (str or None)
    """
    from py_helpers.constants import age_band_to_fname
    
    age_band_fname = age_band_to_fname(age_band)
    filename = f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"

    # Prefer original (first-pass) aggregated FI in _baseline; then current (second-pass) location
    agg_csv_path = None
    for step_dir in ("3_feature_importance", "3a_feature_importance"):
        # 1) _baseline (original aggregated FI from first pass)
        candidate_baseline = (
            PROJECT_ROOT / step_dir / "outputs" / cohort / "_baseline" / filename
        )
        if candidate_baseline.exists():
            agg_csv_path = candidate_baseline
            logger.info("Using aggregated feature importance from _baseline: %s", agg_csv_path)
            break
    if agg_csv_path is None:
        for step_dir in ("3_feature_importance", "3a_feature_importance"):
            # 2) Current location (second-pass or legacy)
            candidate = (
                PROJECT_ROOT / step_dir / "outputs" / cohort / filename
            )
            if candidate.exists():
                agg_csv_path = candidate
                break
    # Fallback: try S3 download location (_baseline first, then current)
    if agg_csv_path is None:
        for step_dir in ("3_feature_importance", "3a_feature_importance"):
            candidate_baseline = (
                PROJECT_ROOT / step_dir / "from_s3" / "by_cohort" / cohort / "_baseline" / filename
            )
            if candidate_baseline.exists():
                agg_csv_path = candidate_baseline
                break
    if agg_csv_path is None:
        for step_dir in ("3_feature_importance", "3a_feature_importance"):
            candidate = (
                PROJECT_ROOT / step_dir / "from_s3" / "by_cohort" / cohort / filename
            )
            if candidate.exists():
                agg_csv_path = candidate
                break
    # Fallback: try downloading from S3 (_baseline first, then current)
    if agg_csv_path is None:
        for s3_suffix, subdir in [("_baseline/", "_baseline"), ("", "")]:
            s3_key = (
                f"gold/feature_importance/{cohort}/{age_band}/{s3_suffix}{filename}"
            )
            try:
                s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
                dest_dir = PROJECT_ROOT / "3a_feature_importance" / "outputs" / cohort
                if subdir:
                    dest_dir = dest_dir / subdir
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / filename
                s3_client.download_file(S3_BUCKET, s3_key, str(dest_path))
                agg_csv_path = dest_path
                logger.info(
                    "Downloaded aggregated feature importance from S3: %s -> %s",
                    s3_key,
                    agg_csv_path,
                )
                break
            except Exception:
                continue
    if agg_csv_path is None:
        agg_csv_path = (
            PROJECT_ROOT
            / "3_feature_importance"
            / "outputs"
            / cohort
            / "_baseline"
            / filename
        )
    
    if not agg_csv_path.exists():
        return {
            "is_valid": False,
            "n_features_initial": 0,
            "n_features_final": 0,
            "n_zero_importance": 0,
            "n_duplicates": 0,
            "cleaned_path": None,
            "error": (
                f"Aggregated feature importance CSV not found for {cohort}/{age_band}. "
                f"Expected under outputs/.../_baseline/ or outputs/... (or S3 gold/feature_importance/...). "
                f"Run Step 3a with --baseline first to create baseline FI."
            ),
        }
    
    try:
        df = pd.read_csv(agg_csv_path)
        
        if "feature" not in df.columns:
            return {
                "is_valid": False,
                "n_features_initial": len(df),
                "n_features_final": 0,
                "n_zero_importance": 0,
                "n_duplicates": 0,
                "cleaned_path": None,
                "error": f"'feature' column not found in {agg_csv_path}",
            }
        
        initial_count = len(df)
        
        # Filter zero-importance features
        n_zero_importance = 0
        importance_col = None
        
        if "scaled_importance_mean" in df.columns:
            importance_col = "scaled_importance_mean"
            n_zero_importance = len(df[df["scaled_importance_mean"] <= 1e-10])
            df = df[df["scaled_importance_mean"] > 1e-10].copy()
        elif "importance_mean" in df.columns:
            importance_col = "importance_mean"
            n_zero_importance = len(df[df["importance_mean"] <= 1e-10])
            df = df[df["importance_mean"] > 1e-10].copy()
        elif "importance_scaled" in df.columns:
            importance_col = "importance_scaled"
            n_zero_importance = len(df[df["importance_scaled"] <= 1e-10])
            df = df[df["importance_scaled"] > 1e-10].copy()
        elif "importance_normalized" in df.columns:
            importance_col = "importance_normalized"
            n_zero_importance = len(df[df["importance_normalized"] <= 1e-10])
            df = df[df["importance_normalized"] > 1e-10].copy()
        
        # Remove duplicates (keep first occurrence, which should be highest importance after sorting)
        n_duplicates = len(df) - len(df.drop_duplicates(subset=["feature"], keep="first"))
        df = df.drop_duplicates(subset=["feature"], keep="first")
        
        # Ensure sorted by importance (descending)
        if importance_col:
            df = df.sort_values(importance_col, ascending=False)
        
        final_count = len(df)
        
        # Save cleaned CSV back to the same location
        if n_zero_importance > 0 or n_duplicates > 0:
            df.to_csv(agg_csv_path, index=False)
            logger.info(f"Saved cleaned aggregated feature importance CSV to {agg_csv_path}")
        
        return {
            "is_valid": True,
            "n_features_initial": initial_count,
            "n_features_final": final_count,
            "n_zero_importance": n_zero_importance,
            "n_duplicates": n_duplicates,
            "importance_col": importance_col,
            "cleaned_path": agg_csv_path,
            "error": None,
        }
    except Exception as e:
        return {
            "is_valid": False,
            "n_features_initial": 0,
            "n_features_final": 0,
            "n_zero_importance": 0,
            "n_duplicates": 0,
            "cleaned_path": None,
            "error": f"Error reading/cleaning aggregated feature importance CSV: {str(e)}",
        }


def _validate_model_events_has_controls(parquet_path: Path) -> dict:
    """
    Validate that model_events.parquet contains both cases (target=1) and controls (target=0).
    
    Returns:
        dict with keys: has_controls (bool), n_cases (int), n_controls (int)
    """
    con = duckdb.connect()
    try:
        result = con.execute(
            f"""
            SELECT 
                COUNT(*) FILTER (WHERE target = 1) AS n_cases,
                COUNT(*) FILTER (WHERE target = 0) AS n_controls
            FROM read_parquet('{parquet_path}')
            """
        ).fetchone()
        
        n_cases = result[0] if result else 0
        n_controls = result[1] if result else 0
        has_controls = n_controls > 0
        
        return {
            "has_controls": has_controls,
            "n_cases": n_cases,
            "n_controls": n_controls,
        }
    finally:
        con.close()


def _resolve_model_events_path(cohort: str, age_band: str) -> Path:
    """
    Resolve the path to model_events.parquet, checking multiple locations.
    
    Priority on Linux/EC2:
    1. get_data_root()/4_model_data/... (/mnt/nvme/4_model_data/...)
    2. PROJECT_ROOT/4_model_data/... (fallback)
    3. Try downloading from S3 to get_data_root() if not found locally
    
    Priority on Windows:
    1. PROJECT_ROOT/4_model_data/... (Windows/local dev)
    2. get_data_root()/4_model_data/... (fallback)
    3. Try downloading from S3 to PROJECT_ROOT if not found locally
    
    Returns:
        Path to model_events.parquet file
    """
    data_root = get_data_root()
    is_linux_system = is_linux()
    
    # Build candidate paths - prioritize data root on Linux, project root on Windows
    if is_linux_system:
        # On Linux/EC2: prioritize /mnt/nvme
        candidates = [
            data_root / "4_model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
            PROJECT_ROOT / "4_model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
        ]
        # Download destination: prefer data root on Linux
        download_dest = candidates[0]
    else:
        # On Windows: prioritize project root
        candidates = [
            PROJECT_ROOT / "4_model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
            data_root / "4_model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
        ]
        # Download destination: prefer project root on Windows
        download_dest = candidates[0]
    
    # Check each candidate
    for path in candidates:
        if path.exists():
            logger.info(f"Found model_events.parquet at: {path}")
            # Validate controls for local files too
            validation_result = _validate_model_events_has_controls(path)
            if not validation_result["has_controls"]:
                logger.warning(
                    f"Local file {path} is missing controls! "
                    f"Cases: {validation_result['n_cases']}, Controls: {validation_result['n_controls']}"
                )
                logger.warning(
                    f"This file should be regenerated with controls. "
                    f"Please run: python 4_model_data/create_model_data.py"
                )
            else:
                logger.debug(
                    f"Validation passed: {validation_result['n_cases']} cases, "
                    f"{validation_result['n_controls']} controls"
                )
            return path
    
    # Log which paths we checked
    logger.info(f"Model data not found locally. Checked paths:")
    for path in candidates:
        logger.info(f"  - {path} (exists: {path.exists()})")
    
    # If not found locally, try downloading from S3
    s3_key_candidates = [
        f"gold/cohorts_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet",
        f"gold/model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet",
        f"gold/model_data/{cohort}/{age_band}/model_events.parquet",
    ]
    
    download_dest.parent.mkdir(parents=True, exist_ok=True)
    
    for s3_key in s3_key_candidates:
        try:
            # Check if file exists in S3
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
            s3_path = f"s3://{S3_BUCKET}/{s3_key}"
            
            # Validate controls BEFORE downloading (using DuckDB S3 support)
            logger.info(f"Checking S3 file for controls: {s3_path}")
            validation_result = _validate_s3_file_has_controls(s3_path)
            
            if validation_result.get("error"):
                logger.warning(f"Could not validate S3 file {s3_path}: {validation_result['error']}")
                logger.info("Proceeding with download and will validate after...")
            elif not validation_result.get("has_controls", False):
                logger.error(
                    f"S3 file {s3_path} is missing controls! "
                    f"Cases: {validation_result.get('n_cases', 0)}, Controls: {validation_result.get('n_controls', 0)}"
                )
                logger.error(
                    f"This file should be regenerated with controls. "
                    f"Please run: python 4_model_data/create_model_data.py --cohort {cohort} --age-band {age_band}"
                )
                logger.error("Skipping this S3 file and trying next candidate...")
                continue  # Skip this S3 file, try next candidate
            
            # Download the file
            logger.info(f"Downloading model_events.parquet from S3: {s3_path}")
            logger.info(f"Downloading to: {download_dest}")
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            with open(download_dest, 'wb') as f:
                f.write(obj['Body'].read())
            logger.info(f"Saved to: {download_dest}")
            
            # Validate again after download (double-check)
            validation_result = _validate_model_events_has_controls(download_dest)
            if not validation_result["has_controls"]:
                logger.error(
                    f"Downloaded file is missing controls! "
                    f"Cases: {validation_result['n_cases']}, Controls: {validation_result['n_controls']}"
                )
                logger.error(
                    f"This file should be regenerated with controls. "
                    f"Please run: python 4_model_data/create_model_data.py --cohort {cohort} --age-band {age_band}"
                )
                # Delete the invalid file
                download_dest.unlink()
                logger.error("Deleted invalid file. Trying next S3 candidate...")
                continue
            else:
                logger.info(
                    f"Validation passed: {validation_result['n_cases']} cases, "
                    f"{validation_result['n_controls']} controls"
                )
            
            return download_dest
        except Exception as e:
            logger.debug(f"S3 key not found or error: {s3_key} - {e}")
            continue
    
    # If all checks failed, raise error with helpful message
    error_msg = (
        f"Model data not found for cohort={cohort}, age_band={age_band}.\n"
        "Checked locations:\n"
    )
    for path in candidates:
        error_msg += f"  - {path} (exists: {path.exists()})\n"
    error_msg += "\nS3 locations checked:\n"
    for s3_key in s3_key_candidates:
        error_msg += f"  - s3://{S3_BUCKET}/{s3_key}\n"
    raise FileNotFoundError(error_msg)


def classify_event_as_administrative(
    event_row: pd.Series,
    administrative_codes: Optional[dict] = None,
    cohort_name: str = "",
) -> bool:
    """
    Classify an event as administrative vs. medical/pharmacy.
    
    Administrative events include:
    - Billing codes (specific CPT codes for billing/documentation)
    - Scheduling codes (appointment scheduling, administrative procedures)
    - Post-event documentation (events after target event date - leakage)
    - Codes identified through research as administrative
    
    Parameters
    ----------
    event_row : pd.Series
        Single event row from model_data
    administrative_codes : Optional[dict]
        Dictionary with keys 'icd', 'cpt', 'drug' containing sets of administrative codes
        If None, uses default patterns and research-based classification
    cohort_name : str
        Cohort name for determining target event date field
        
    Returns
    -------
    bool
        True if event is administrative (should be filtered), False if clinical (keep)
    """
    if administrative_codes is None:
        administrative_codes = {
            'icd': set(),  # Will be populated from research
            'cpt': set(),  # Will be populated from research
            'drug': set(),  # Will be populated from research
        }
    
    # Check for post-event leakage (events after target event date)
    if cohort_name:
        if "opioid" in cohort_name.lower():
            target_date_field = "first_opioid_ed_date"
        else:
            target_date_field = "first_ed_non_opioid_date"
        
        if target_date_field in event_row.index:
            target_date = event_row.get(target_date_field)
            event_date = event_row.get("event_date")
            
            if pd.notna(target_date) and pd.notna(event_date):
                # If event occurs on or after target date, it's leakage (administrative)
                if pd.to_datetime(event_date) >= pd.to_datetime(target_date):
                    return True
    
    # Check ICD codes
    for icd_col in ['primary_icd_diagnosis_code', 'two_icd_diagnosis_code', 
                     'three_icd_diagnosis_code', 'four_icd_diagnosis_code', 
                     'five_icd_diagnosis_code']:
        if icd_col in event_row.index:
            icd_code = event_row.get(icd_col)
            if pd.notna(icd_code) and str(icd_code).strip():
                icd_str = str(icd_code).strip()
                # Check both with and without dots (Z34.03 vs Z3403)
                admin_icd_set = administrative_codes.get('icd', set())
                if icd_str in admin_icd_set:
                    return True
                # Also check normalized version (remove dots for comparison)
                icd_normalized = icd_str.replace('.', '')
                if icd_normalized in {code.replace('.', '') for code in admin_icd_set}:
                    return True
    
    # Check CPT codes
    if 'procedure_code' in event_row.index:
        cpt_code = event_row.get('procedure_code')
        if pd.notna(cpt_code) and str(cpt_code).strip():
            if str(cpt_code) in administrative_codes.get('cpt', set()):
                return True
    
    # Check drug codes
    if 'drug_name' in event_row.index:
        drug_name = event_row.get('drug_name')
        if pd.notna(drug_name) and str(drug_name).strip():
            if str(drug_name) in administrative_codes.get('drug', set()):
                return True
    
    # Default: keep all medical/pharmacy events (not administrative)
    return False


def load_administrative_codes_from_research(
    cohort_name: str,
    age_band: str,
    protocol_threshold_pct: float = 80.0,
) -> dict:
    """
    Load administrative codes from research outputs.
    
    Codes that appear in > protocol_threshold_pct of protocol-like sequences
    (events < min_interval_days apart, default: 1 day) are considered administrative.
    
    Parameters
    ----------
    cohort_name : str
        Cohort name
    age_band : str
        Age band
    protocol_threshold_pct : float
        Threshold for considering a code administrative (default: 80%)
        
    Returns
    -------
    dict
        Dictionary with keys 'icd', 'cpt', 'drug' containing sets of administrative codes
    """
    age_band_fname = age_band.replace("-", "_")
    code_analysis_path = (
        OUTPUT_ROOT / "for_review" / cohort_name / age_band_fname /
        f"code_analysis_protocol_vs_clinical_{cohort_name}_{age_band_fname}.csv"
    )
    
    administrative_codes = {
        'icd': set(),
        'cpt': set(),
        'drug': set(),
    }
    
    if not code_analysis_path.exists():
        logger.warning(
            f"Research outputs not found at {code_analysis_path}. "
            "Using default classification (no codes filtered)."
        )
        return administrative_codes
    
    try:
        code_analysis_df = pd.read_csv(code_analysis_path)
        
        # Codes with high protocol_pct are likely administrative
        admin_codes = code_analysis_df[
            code_analysis_df['protocol_pct'] >= protocol_threshold_pct
        ]
        
        for _, row in admin_codes.iterrows():
            code_type = row.get('code_type', '').upper()
            code = str(row.get('code', '')).strip()
            
            if code_type == 'ICD' and code:
                administrative_codes['icd'].add(code)
            elif code_type == 'CPT' and code:
                administrative_codes['cpt'].add(code)
            elif code_type == 'DRUG' and code:
                administrative_codes['drug'].add(code)
        
        logger.info(
            f"Loaded {len(administrative_codes['icd'])} ICD, "
            f"{len(administrative_codes['cpt'])} CPT, "
            f"{len(administrative_codes['drug'])} drug administrative codes from research"
        )
        
    except Exception as e:
        logger.warning(f"Error loading administrative codes: {e}. Using default classification.")
    
    return administrative_codes


def calculate_event_intervals(
    model_data_path: Path,
    min_interval_days: int = 1,
    max_interval_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Calculate time intervals between consecutive events per patient.

    Parameters
    ----------
    model_data_path : Path
        Path to model_events.parquet
    min_interval_days : int
        Minimum interval (days) to consider non-protocol. Events closer than this
        are considered protocol-like.
    max_interval_days : Optional[int]
        Maximum interval (days) to consider. Events further apart may be outliers.

    Returns
    -------
    pd.DataFrame
        DataFrame with event intervals and protocol flags
    """
    logger.info("Calculating event intervals from {0}".format(model_data_path))

    con = duckdb.connect()

    query = f"""
    WITH base AS (
        SELECT
            mi_person_key,
            event_date AS current_event_date,
            target,
            drug_name,
            primary_icd_diagnosis_code,
            procedure_code,
            ROW_NUMBER() OVER (
                PARTITION BY mi_person_key
                ORDER BY event_date
            ) AS event_seq,
            LAG(event_date) OVER (
                PARTITION BY mi_person_key
                ORDER BY event_date
            ) AS previous_event_date
        FROM read_parquet('{model_data_path}')
        WHERE event_date IS NOT NULL
    )
    SELECT
        mi_person_key,
        current_event_date AS event_date,
        target,
        drug_name,
        primary_icd_diagnosis_code,
        procedure_code,
        event_seq,
        previous_event_date,
        DATEDIFF('day', previous_event_date, current_event_date) AS days_since_previous,
        CASE
            WHEN previous_event_date IS NULL THEN 1  -- First event
            WHEN DATEDIFF('day', previous_event_date, current_event_date) < {min_interval_days}
                THEN 1  -- Protocol-like
            ELSE 0  -- Non-protocol
        END AS is_protocol_event
    FROM base
    """

    intervals_df = con.execute(query).df()
    con.close()

    logger.info("Calculated intervals for {0} events".format(len(intervals_df)))
    logger.info(
        "Protocol events (< {0} days apart): {1}".format(
            min_interval_days, (intervals_df["is_protocol_event"] == 1).sum()
        )
    )
    logger.info(
        "Non-protocol events: {0}".format(
            (intervals_df["is_protocol_event"] == 0).sum()
        )
    )

    return intervals_df


def filter_administrative_events(
    model_data_path: Path,
    output_path: Path,
    cohort_name: str,
    age_band: str,
    administrative_codes: Optional[dict] = None,
    keep_first_event: bool = True,
    admin_code_threshold_pct: float = 80.0,
    allowed_codes_from_fi: Optional[set] = None,
) -> dict:
    """
    Filter out administrative events from model_data based on code classification.

    Filter order:
    1. If allowed_codes_from_fi is provided: keep only events where at least one
       of (drug_name, ICD columns, procedure_code) is in the aggregated feature-importance
       allowed set. This reduces features before the final cohort and makes step 3a
       feature importance more accurate on a second pass.
    2. Then: remove administrative codes (from research + hardcoded) and post-event leakage.

    This version keeps processing inside DuckDB (no pandas row-wise apply),
    and writes the filtered dataset directly to Parquet via COPY.

    Returns a small dict of counts (original/filtered/removed, fi_removed if applicable).
    """
    logger.info("Filtering administrative events from %s", model_data_path)

    # Load administrative codes from research if not provided
    if administrative_codes is None:
        administrative_codes = load_administrative_codes_from_research(
            cohort_name=cohort_name,
            age_band=age_band,
            protocol_threshold_pct=admin_code_threshold_pct,
        )
        
        # Add hardcoded administrative codes (preventive/administrative codes that should always be filtered)
        # These codes were identified from aggregated feature importance files as administrative/preventive
        # Only including codes that were actually found in the aggregated feature importance analysis
        hardcoded_admin_icd = set()
        
        # Z00: General health examinations (preventive/routine) - codes found in aggregated FI
        z00_codes = ['Z00.00', 'Z00.01', 'Z00.110', 'Z00.111', 'Z00.12', 'Z00.121', 'Z00.129', 'Z00.3', 'Z00.70', 'Z00.8']
        # Z01: Special examinations (preventive/routine) - codes found in aggregated FI
        z01_codes = ['Z01.00', 'Z01.01', 'Z01.10', 'Z01.30', 'Z01.31', 'Z01.41', 'Z01.411', 'Z01.419', 'Z01.42', 'Z01.70',
                     'Z01.810', 'Z01.811', 'Z01.812', 'Z01.818', 'Z01.82', 'Z01.83', 'Z01.84', 'Z01.89']
        # Z02: Administrative examinations (clearly administrative) - codes found in aggregated FI
        z02_codes = ['Z02.0', 'Z02.1', 'Z02.2', 'Z02.5', 'Z02.83', 'Z02.89', 'Z02.9']
        # Z03: Medical observation for suspected conditions (ruled out - administrative) - codes found in aggregated FI
        z03_codes = ['Z03.6', 'Z03.71', 'Z03.72', 'Z03.73', 'Z03.74', 'Z03.75', 'Z03.79', 'Z03.89']
        # Z04: Examination for legal/administrative purposes (clearly administrative) - codes found in aggregated FI
        z04_codes = ['Z04.1', 'Z04.3', 'Z04.41', 'Z04.42', 'Z04.8', 'Z04.89', 'Z04.9']
        # Z08: Follow-up examination after treatment for malignant neoplasm (administrative follow-up)
        z08_codes = ['Z08']
        # Z09: Follow-up examination after treatment for other conditions (administrative follow-up)
        z09_codes = ['Z09']
        # Z34: Supervision of normal pregnancy (preventive/routine) - codes found in aggregated FI
        z34_codes = ['Z34.00', 'Z34.01', 'Z34.02', 'Z34.03', 'Z34.80', 'Z34.81', 'Z34.82', 'Z34.83', 'Z34.90', 'Z34.91', 'Z34.92', 'Z34.93']
        # Z39: Encounter for maternal postpartum care and examination (administrative follow-up) - codes found in aggregated FI
        z39_codes = ['Z39.0', 'Z39.1', 'Z39.2']
        # Z51: Encounters for other aftercare and medical care (administrative aftercare) - codes found in aggregated FI
        z51_codes = ['Z51.0', 'Z51.11', 'Z51.12', 'Z51.5', 'Z51.6', 'Z51.81', 'Z51.89']
        # V72: Other medical examination (preventive/administrative) - codes found in aggregated FI
        v72_codes = ['V72.31', 'V72.40', 'V72.41', 'V72.42', 'V72.5', 'V72.61', 'V72.7', 'V72.81', 'V72.83', 'V72.85']
        
        # Add all codes (with dots) and also normalized versions (without dots) to handle both formats
        all_icd_codes = z00_codes + z01_codes + z02_codes + z03_codes + z04_codes + z08_codes + z09_codes + z34_codes + z39_codes + z51_codes + v72_codes
        for code in all_icd_codes:
            hardcoded_admin_icd.add(code)  # With dots (standard format)
            hardcoded_admin_icd.add(code.replace('.', ''))  # Without dots (aggregated FI format)
        administrative_codes['icd'].update(hardcoded_admin_icd)
        
        # Add hardcoded administrative CPT codes (preventive/administrative procedures that should always be filtered)
        # These codes were identified from aggregated feature importance files as administrative/preventive
        hardcoded_admin_cpt = set()
        
        # CPT 99000-99099: Administrative services (clearly administrative)
        admin_cpt_99000 = [99000, 99001, 99024, 99050, 99051, 99053, 99058, 99070, 99078]
        # CPT 99400-99499: Preventive medicine (administrative/preventive)
        admin_cpt_99400 = [99401, 99402, 99403, 99404, 99406, 99407, 99408, 99409, 99420, 99429, 99441, 99442, 99443, 99444, 99460, 99462, 99464, 99471, 99472, 99480, 99484, 99487, 99490, 99495, 99496, 99497, 99499]
        # CPT 99381-99397: Preventive visits (administrative/preventive)
        admin_cpt_99381 = [99381, 99382, 99383, 99384, 99385, 99386, 99387, 99391, 99392, 99393, 99394, 99395, 99396, 99397]
        # CPT 99211: Level 1 office visit (minimal complexity, often routine/preventive)
        admin_cpt_level1_office = [99211]

        for code in admin_cpt_99000 + admin_cpt_99400 + admin_cpt_99381 + admin_cpt_level1_office:
            hardcoded_admin_cpt.add(str(code))

        # S codes: Administrative billing codes (non-clinical)
        admin_s_codes = ['S0201', 'S0109', 'S9083', 'S0990XA', 'S0028']
        for code in admin_s_codes:
            hardcoded_admin_cpt.add(code)  # S codes are stored as CPT codes

        administrative_codes['cpt'].update(hardcoded_admin_cpt)

        # If research outputs don't exist, start with hardcoded sets (will filter known admin codes + post-event leakage)
        if (not administrative_codes.get("icd")) and (not administrative_codes.get("cpt")) and (not administrative_codes.get("drug")):
            logger.info(
                "No administrative codes found in research outputs. "
                "Will only filter post-event leakage (events on/after target date)."
            )
        else:
            logger.info(
                f"Using {len(administrative_codes['icd'])} ICD administrative codes "
                f"(including {len(hardcoded_admin_icd)} hardcoded preventive codes) and "
                f"{len(administrative_codes.get('cpt', []))} CPT administrative codes "
                f"(including {len(hardcoded_admin_cpt)} hardcoded administrative/preventive CPT codes)"
            )

    con = duckdb.connect()
    try:
        # Respect parallel runs: allow per-process tuning via env vars
        threads = int(os.getenv("DUCKDB_THREADS", "4"))
        con.execute(f"PRAGMA threads={threads}")

        mem_limit = os.getenv("DUCKDB_MEMORY_LIMIT")
        if mem_limit:
            con.execute(f"PRAGMA memory_limit='{mem_limit}'")

        tmp_dir = os.getenv("DUCKDB_TMP_DIR")
        if tmp_dir:
            Path(tmp_dir).mkdir(parents=True, exist_ok=True)
            con.execute(f"PRAGMA temp_directory='{tmp_dir}'")

        # Discover available columns in the parquet so we can build safe SQL
        desc_rows = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{model_data_path}')"
        ).fetchall()
        available_cols = {r[0] for r in desc_rows}

        icd_cols = [
            "primary_icd_diagnosis_code",
            "two_icd_diagnosis_code",
            "three_icd_diagnosis_code",
            "four_icd_diagnosis_code",
            "five_icd_diagnosis_code",
        ]
        present_icd_cols = [c for c in icd_cols if c in available_cols]

        # Register administrative code lists as small in-memory tables
        # (DuckDB handles joining/IN efficiently; avoids Python per-row checks)
        # Add both dot and no-dot versions to handle different code formats in data
        icd_codes_set = set()
        for code in administrative_codes.get("icd", set()):
            code_str = str(code).strip()
            if code_str:
                icd_codes_set.add(code_str)  # Original format
                icd_codes_set.add(code_str.replace('.', ''))  # No-dot format
                # If no dots, try to add dot version (e.g., Z3403 -> Z34.03)
                if '.' not in code_str and len(code_str) >= 5:
                    if code_str.startswith('Z') or code_str.startswith('V'):
                        # Z/V codes: Z/V + 2 digits + rest (e.g., Z3403 -> Z34.03, V7231 -> V72.31)
                        icd_codes_set.add(f"{code_str[:3]}.{code_str[3:]}")
        
        icd_codes = sorted(icd_codes_set)
        cpt_codes = sorted({str(x) for x in administrative_codes.get("cpt", set()) if str(x).strip()})
        drug_codes = sorted({str(x) for x in administrative_codes.get("drug", set()) if str(x).strip()})

        con.register("admin_icd", pd.DataFrame({"code": icd_codes}) if icd_codes else pd.DataFrame({"code": []}))
        con.register("admin_cpt", pd.DataFrame({"code": cpt_codes}) if cpt_codes else pd.DataFrame({"code": []}))
        con.register("admin_drug", pd.DataFrame({"code": drug_codes}) if drug_codes else pd.DataFrame({"code": []}))

        # Optional first pass: keep only events whose codes appear in aggregated feature importance
        use_fi_filter = allowed_codes_from_fi is not None and len(allowed_codes_from_fi) > 0
        if use_fi_filter:
            allowed_fi_list = sorted(allowed_codes_from_fi)
            con.register("allowed_fi_codes", pd.DataFrame({"code": allowed_fi_list}))
            fi_parts = []
            if "drug_name" in available_cols:
                fi_parts.append("(drug_name IN (SELECT code FROM allowed_fi_codes))")
            for c in present_icd_cols:
                fi_parts.append(f"({c} IN (SELECT code FROM allowed_fi_codes))")
            if "procedure_code" in available_cols:
                fi_parts.append("(procedure_code IN (SELECT code FROM allowed_fi_codes))")
            fi_predicate = "(" + " OR ".join(fi_parts) + ")" if fi_parts else "TRUE"
            logger.info(
                "Applying aggregated feature-importance filter first: keep only events matching %s allowed codes",
                len(allowed_codes_from_fi),
            )

        # Build administrative predicates (only referencing columns that exist)
        predicates = []

        # Leakage predicate: events on/after target date are removed
        target_date_field = None
        if cohort_name:
            if "opioid" in cohort_name.lower():
                target_date_field = "first_opioid_ed_date"
            else:
                target_date_field = "first_ed_non_opioid_date"

        if target_date_field and (target_date_field in available_cols) and ("event_date" in available_cols):
            predicates.append(
                f"(event_date IS NOT NULL AND {target_date_field} IS NOT NULL "
                f"AND CAST(event_date AS TIMESTAMP) >= CAST({target_date_field} AS TIMESTAMP))"
            )

        # ICD predicates across the 5 ICD columns (present_icd_cols already defined above)
        if present_icd_cols and icd_codes:
            icd_or = " OR ".join([f"{c} IN (SELECT code FROM admin_icd)" for c in present_icd_cols])
            predicates.append(f"({icd_or})")

        # CPT predicate
        if ("procedure_code" in available_cols) and cpt_codes:
            predicates.append("(procedure_code IN (SELECT code FROM admin_cpt))")

        # Drug predicate
        if ("drug_name" in available_cols) and drug_codes:
            predicates.append("(drug_name IN (SELECT code FROM admin_drug))")

        if predicates:
            is_admin_expr = " OR ".join(predicates)
        else:
            # No code tables and no leakage field -> nothing to filter
            is_admin_expr = "FALSE"

        # Keep logic
        keep_expr = f"(event_seq = 1) OR (NOT is_administrative)" if keep_first_event else "(NOT is_administrative)"

        # Write filtered dataset directly
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Materialize a temp table so we can get counts without re-running the whole pipeline twice
        # Optional: first restrict to events whose codes appear in aggregated feature importance
        con.execute("DROP TABLE IF EXISTS _flagged_events")
        base_cte = f"""
            WITH base AS (
              SELECT
                *,
                ROW_NUMBER() OVER (
                  PARTITION BY mi_person_key
                  ORDER BY event_date
                ) AS event_seq
              FROM read_parquet('{model_data_path}')
            )"""
        if use_fi_filter:
            source_from = f"""
            , fi_filtered AS (
              SELECT * FROM base WHERE {fi_predicate}
            )
            SELECT
              *,
              ({is_admin_expr}) AS is_administrative
            FROM fi_filtered"""
        else:
            source_from = f"""
            SELECT
              *,
              ({is_admin_expr}) AS is_administrative
            FROM base"""
        con.execute(
            f"""
            CREATE TEMP TABLE _flagged_events AS
            {base_cte}
            {source_from}
            """
        )

        # Original event count (before any filter) for FI-removed reporting
        events_before_fi = int(
            con.execute(f"SELECT COUNT(*)::UBIGINT FROM read_parquet('{model_data_path}')").fetchone()[0]
            or 0
        )
        # Counts from _flagged_events (after FI filter if applied, before admin keep_expr)
        counts = con.execute(
            f"""
            SELECT
              COUNT(*)::UBIGINT AS events_after_fi,
              SUM(CASE WHEN is_administrative THEN 1 ELSE 0 END)::UBIGINT AS administrative_events,
              SUM(CASE WHEN {keep_expr} THEN 1 ELSE 0 END)::UBIGINT AS kept_events
            FROM _flagged_events
            """
        ).fetchone()

        events_after_fi_n = int(counts[0]) if counts and counts[0] is not None else 0
        admin_n = int(counts[1]) if counts and counts[1] is not None else 0
        kept_n = int(counts[2]) if counts and counts[2] is not None else 0
        fi_removed_n = (events_before_fi - events_after_fi_n) if use_fi_filter else 0
        original_n = events_before_fi

        con.execute(
            f"""
            COPY (
              SELECT
                * EXCLUDE (event_seq, is_administrative)
              FROM _flagged_events
              WHERE {keep_expr}
            )
            TO '{output_path}'
            (FORMAT PARQUET)
            """
        )

        if use_fi_filter and fi_removed_n > 0:
            logger.info(
                "Aggregated FI filter: %s events -> %s events (removed %s low-importance)",
                f"{events_before_fi:,}",
                f"{events_after_fi_n:,}",
                f"{fi_removed_n:,}",
            )
        logger.info("Filtered %s events -> %s events", f"{original_n:,}", f"{kept_n:,}")
        logger.info(
            "Removed %s administrative events (%0.1f%%)",
            f"{admin_n:,}",
            (100.0 * admin_n / max(events_after_fi_n, 1)),
        )
        logger.info("Saved filtered model_data to %s", output_path)
        
        # Validate that controls are preserved after filtering
        validation_result = _validate_model_events_has_controls(output_path)
        if not validation_result["has_controls"]:
            logger.error(
                f"Filtered file {output_path} is missing controls! "
                f"Cases: {validation_result['n_cases']}, Controls: {validation_result['n_controls']}"
            )
            logger.error(
                "This indicates controls were incorrectly filtered out. "
                "Administrative event filtering should preserve both cases and controls."
            )
            raise ValueError(
                f"Controls lost during filtering: {validation_result['n_cases']} cases, "
                f"{validation_result['n_controls']} controls"
            )
        
        # Log control preservation
        logger.info(
            f"Controls preserved after filtering: {validation_result['n_cases']} cases, "
            f"{validation_result['n_controls']} controls"
        )

        result = {
            "original_events": original_n,
            "filtered_events": kept_n,
            "removed_events": original_n - kept_n,
            "administrative_events": admin_n,
            "n_cases": validation_result["n_cases"],
            "n_controls": validation_result["n_controls"],
        }
        if use_fi_filter:
            result["fi_removed_events"] = fi_removed_n
        return result
    finally:
        con.close()


def create_research_outputs_for_review(
    intervals_df: pd.DataFrame,
    model_data_path: Path,
    cohort_name: str,
    age_band: str,
    min_interval_days: int = 1,
) -> None:
    """
    Create comprehensive research outputs for review in outputs/for_review folder.
    
    Outputs include:
    - All trajectories with time windows
    - Common sequence patterns
    - Protocol-like sequences
    - Time window statistics
    - Code-level analysis (clinical vs administrative/post-event)
    """
    age_band_fname = age_band.replace("-", "_")
    review_dir = OUTPUT_ROOT / "for_review" / cohort_name / age_band_fname
    review_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating research outputs for review in: {review_dir}")
    
    # NOTE: This function is designed to run during highly-parallel workflows.
    # Avoid materializing large Parquet datasets in pandas. All heavy work is done in DuckDB,
    # and only small aggregates are fetched/written.

    duckdb_threads = int(os.getenv("DUCKDB_THREADS", "4"))
    duckdb_memory_limit = os.getenv("DUCKDB_MEMORY_LIMIT", "")
    # Prefer a per-worker temp directory on fast local storage to avoid contention
    duckdb_temp_dir = os.getenv("DUCKDB_TMP_DIR", "")
    if not duckdb_temp_dir:
        try:
            duckdb_temp_dir = tempfile.mkdtemp(prefix="duckdb_tmp_")
        except Exception:
            duckdb_temp_dir = ""

    con = duckdb.connect()
    try:
        con.execute(f"PRAGMA threads={duckdb_threads}")
        if duckdb_memory_limit:
            con.execute(f"PRAGMA memory_limit='{duckdb_memory_limit}'")
        if duckdb_temp_dir:
            Path(duckdb_temp_dir).mkdir(parents=True, exist_ok=True)
            con.execute(f"PRAGMA temp_directory='{duckdb_temp_dir}'")

        # ------------------------------------------------------------------
        # 1) Trajectories with time windows (DuckDB-only, no pandas merge)
        # ------------------------------------------------------------------
        trajectories_path = review_dir / f"trajectories_with_time_windows_{cohort_name}_{age_band_fname}.parquet"

        con.execute(
            f"""
            COPY (
                WITH base AS (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (PARTITION BY mi_person_key ORDER BY event_date) AS event_seq,
                        LAG(event_date) OVER (PARTITION BY mi_person_key ORDER BY event_date) AS previous_event_date,
                        DATEDIFF(
                            'day',
                            LAG(event_date) OVER (PARTITION BY mi_person_key ORDER BY event_date),
                            event_date
                        ) AS days_since_previous
                    FROM read_parquet('{model_data_path}')
                    WHERE event_date IS NOT NULL
                )
                SELECT
                    *,
                    CASE
                        WHEN previous_event_date IS NULL THEN 1
                        WHEN days_since_previous < {min_interval_days} THEN 1
                        ELSE 0
                    END AS is_protocol_event
                FROM base
            ) TO '{trajectories_path.as_posix()}' (FORMAT PARQUET);
            """
        )
        logger.info(f"Saved trajectories with time windows: {trajectories_path}")

        # Reuse the computed trajectories as a temp view for downstream outputs
        con.execute(f"CREATE OR REPLACE TEMP VIEW traj AS SELECT * FROM read_parquet('{trajectories_path.as_posix()}')")

        # ------------------------------------------------------------------
        # 2) Time window statistics (DuckDB aggregate)
        # ------------------------------------------------------------------
        stats_path = review_dir / f"time_window_statistics_{cohort_name}_{age_band_fname}.csv"
        con.execute(
            f"""
            COPY (
                SELECT
                    mi_person_key,
                    AVG(days_since_previous) AS mean_interval_days,
                    MEDIAN(days_since_previous) AS median_interval_days,
                    STDDEV_SAMP(days_since_previous) AS std_interval_days,
                    MIN(days_since_previous) AS min_interval_days,
                    MAX(days_since_previous) AS max_interval_days,
                    COUNT(*) AS total_events,
                    SUM(is_protocol_event) AS protocol_event_count,
                    (SUM(is_protocol_event) * 100.0) / NULLIF(COUNT(*), 0) AS protocol_event_pct
                FROM traj
                GROUP BY mi_person_key
            ) TO '{stats_path.as_posix()}' (HEADER, DELIMITER ',');
            """
        )
        logger.info(f"Saved time window statistics: {stats_path}")

        # ------------------------------------------------------------------
        # 3) ALL sequence patterns (2- and 3-grams) using integer vocab IDs
        #    This keeps full coverage while reducing string/groupby cost.
        # ------------------------------------------------------------------
        logger.info("Extracting sequence patterns (ID-based, DuckDB)...")

        # Build per-event activity columns (no global ORDER BY; window ORDER BY is sufficient)
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW events_with_activities AS
            SELECT
                mi_person_key,
                event_date,
                ROW_NUMBER() OVER (PARTITION BY mi_person_key ORDER BY event_date) AS event_seq,
                CASE
                    WHEN drug_name IS NOT NULL AND TRIM(drug_name) != '' THEN 'DRUG:' || drug_name
                    ELSE NULL
                END AS drug_activity,
                CASE
                    WHEN primary_icd_diagnosis_code IS NOT NULL AND TRIM(primary_icd_diagnosis_code) != '' THEN 'ICD:' || primary_icd_diagnosis_code
                    ELSE NULL
                END AS icd_activity,
                CASE
                    WHEN procedure_code IS NOT NULL AND TRIM(procedure_code) != '' THEN 'CPT:' || procedure_code
                    ELSE NULL
                END AS cpt_activity
            FROM read_parquet('{model_data_path}')
            WHERE event_date IS NOT NULL;
            """
        )

        # Flatten activities into one ordered stream per patient
        con.execute(
            """
            CREATE OR REPLACE TEMP VIEW all_activities AS
            SELECT
                mi_person_key,
                activity,
                ROW_NUMBER() OVER (PARTITION BY mi_person_key ORDER BY event_seq, activity_type) - 1 AS activity_idx
            FROM (
                SELECT mi_person_key, event_seq, drug_activity AS activity, 1 AS activity_type
                FROM events_with_activities WHERE drug_activity IS NOT NULL
                UNION ALL
                SELECT mi_person_key, event_seq, icd_activity AS activity, 2 AS activity_type
                FROM events_with_activities WHERE icd_activity IS NOT NULL
                UNION ALL
                SELECT mi_person_key, event_seq, cpt_activity AS activity, 3 AS activity_type
                FROM events_with_activities WHERE cpt_activity IS NOT NULL
            );
            """
        )

        # Vocabulary: activity -> integer ID (dense rank is deterministic within a cohort)
        vocab_path = review_dir / f"activity_vocab_{cohort_name}_{age_band_fname}.parquet"
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW vocab AS
            SELECT
                activity,
                DENSE_RANK() OVER (ORDER BY activity) AS activity_id
            FROM (SELECT DISTINCT activity FROM all_activities);

            COPY (
                SELECT activity_id, activity
                FROM vocab
            ) TO '{vocab_path.as_posix()}' (FORMAT PARQUET);
            """
        )

        # Activities with IDs
        con.execute(
            """
            CREATE OR REPLACE TEMP VIEW activities_id AS
            SELECT
                a.mi_person_key,
                a.activity_idx,
                v.activity_id
            FROM all_activities a
            JOIN vocab v USING (activity);
            """
        )

        bigrams_path = review_dir / f"bigrams_{cohort_name}_{age_band_fname}.parquet"
        trigrams_path = review_dir / f"trigrams_{cohort_name}_{age_band_fname}.parquet"

        con.execute(
            f"""
            COPY (
                SELECT
                    a1.activity_id AS id1,
                    a2.activity_id AS id2,
                    COUNT(*)::UBIGINT AS frequency
                FROM activities_id a1
                JOIN activities_id a2
                  ON a1.mi_person_key = a2.mi_person_key
                 AND a2.activity_idx = a1.activity_idx + 1
                GROUP BY 1, 2
            ) TO '{bigrams_path.as_posix()}' (FORMAT PARQUET);
            """
        )

        con.execute(
            f"""
            COPY (
                SELECT
                    a1.activity_id AS id1,
                    a2.activity_id AS id2,
                    a3.activity_id AS id3,
                    COUNT(*)::UBIGINT AS frequency
                FROM activities_id a1
                JOIN activities_id a2
                  ON a1.mi_person_key = a2.mi_person_key
                 AND a2.activity_idx = a1.activity_idx + 1
                JOIN activities_id a3
                  ON a1.mi_person_key = a3.mi_person_key
                 AND a3.activity_idx = a1.activity_idx + 2
                GROUP BY 1, 2, 3
            ) TO '{trigrams_path.as_posix()}' (FORMAT PARQUET);
            """
        )

        # Human-readable CSVs for review (built from compact ID counts)
        sequences_path = review_dir / f"common_sequence_patterns_{cohort_name}_{age_band_fname}.csv"
        top_sequences_path = review_dir / f"top_100_sequences_{cohort_name}_{age_band_fname}.csv"

        con.execute(
            f"""
            COPY (
                WITH v AS (SELECT * FROM read_parquet('{vocab_path.as_posix()}')),
                seq2 AS (
                    SELECT
                        v1.activity || ' -> ' || v2.activity AS sequence,
                        2 AS sequence_length,
                        b.frequency
                    FROM read_parquet('{bigrams_path.as_posix()}') b
                    JOIN v v1 ON v1.activity_id = b.id1
                    JOIN v v2 ON v2.activity_id = b.id2
                ),
                seq3 AS (
                    SELECT
                        v1.activity || ' -> ' || v2.activity || ' -> ' || v3.activity AS sequence,
                        3 AS sequence_length,
                        t.frequency
                    FROM read_parquet('{trigrams_path.as_posix()}') t
                    JOIN v v1 ON v1.activity_id = t.id1
                    JOIN v v2 ON v2.activity_id = t.id2
                    JOIN v v3 ON v3.activity_id = t.id3
                )
                SELECT * FROM (
                    SELECT * FROM seq2
                    UNION ALL
                    SELECT * FROM seq3
                )
                ORDER BY frequency DESC
            ) TO '{sequences_path.as_posix()}' (HEADER, DELIMITER ',');
            """
        )
        logger.info(f"Saved common sequence patterns: {sequences_path}")

        con.execute(
            f"""
            COPY (
                WITH v AS (SELECT * FROM read_parquet('{vocab_path.as_posix()}')),
                seq2 AS (
                    SELECT
                        v1.activity || ' -> ' || v2.activity AS sequence,
                        2 AS sequence_length,
                        b.frequency
                    FROM read_parquet('{bigrams_path.as_posix()}') b
                    JOIN v v1 ON v1.activity_id = b.id1
                    JOIN v v2 ON v2.activity_id = b.id2
                ),
                seq3 AS (
                    SELECT
                        v1.activity || ' -> ' || v2.activity || ' -> ' || v3.activity AS sequence,
                        3 AS sequence_length,
                        t.frequency
                    FROM read_parquet('{trigrams_path.as_posix()}') t
                    JOIN v v1 ON v1.activity_id = t.id1
                    JOIN v v2 ON v2.activity_id = t.id2
                    JOIN v v3 ON v3.activity_id = t.id3
                )
                SELECT * FROM (
                    SELECT * FROM seq2
                    UNION ALL
                    SELECT * FROM seq3
                )
                ORDER BY frequency DESC
                LIMIT 100
            ) TO '{top_sequences_path.as_posix()}' (HEADER, DELIMITER ',');
            """
        )
        logger.info(f"Saved top 100 sequences: {top_sequences_path}")

        # ------------------------------------------------------------------
        # 4) Protocol-like sequences and protocol events with codes
        # ------------------------------------------------------------------
        protocol_path = review_dir / f"protocol_like_sequences_{cohort_name}_{age_band_fname}.parquet"
        con.execute(
            f"""
            COPY (
                SELECT
                    mi_person_key,
                    event_seq,
                    event_date AS current_event_date,
                    previous_event_date,
                    days_since_previous,
                    is_protocol_event
                FROM traj
                WHERE days_since_previous IS NOT NULL
                  AND days_since_previous < {min_interval_days}
            ) TO '{protocol_path.as_posix()}' (FORMAT PARQUET);
            """
        )
        logger.info(f"Saved protocol-like sequences: {protocol_path}")

        protocol_codes_path = review_dir / f"protocol_events_with_codes_{cohort_name}_{age_band_fname}.parquet"
        con.execute(
            f"""
            COPY (
                SELECT *
                FROM traj
                WHERE is_protocol_event = 1
            ) TO '{protocol_codes_path.as_posix()}' (FORMAT PARQUET);
            """
        )
        logger.info(f"Saved protocol events with codes: {protocol_codes_path}")

        # ------------------------------------------------------------------
        # 5) Code-level analysis: protocol vs total frequency (DuckDB)
        # ------------------------------------------------------------------
        code_analysis_path = review_dir / f"code_analysis_protocol_vs_clinical_{cohort_name}_{age_band_fname}.csv"

        con.execute(
            f"""
            COPY (
                WITH
                prot AS (
                    SELECT * FROM traj WHERE is_protocol_event = 1
                ),
                prot_drug AS (
                    SELECT 'DRUG' AS code_type, drug_name AS code, COUNT(*) AS protocol_count
                    FROM prot
                    WHERE drug_name IS NOT NULL AND TRIM(drug_name) != ''
                    GROUP BY 1, 2
                ),
                all_drug AS (
                    SELECT drug_name AS code, COUNT(*) AS total_count
                    FROM traj
                    WHERE drug_name IS NOT NULL AND TRIM(drug_name) != ''
                    GROUP BY 1
                ),
                prot_cpt AS (
                    SELECT 'CPT' AS code_type, procedure_code AS code, COUNT(*) AS protocol_count
                    FROM prot
                    WHERE procedure_code IS NOT NULL AND TRIM(procedure_code) != ''
                    GROUP BY 1, 2
                ),
                all_cpt AS (
                    SELECT procedure_code AS code, COUNT(*) AS total_count
                    FROM traj
                    WHERE procedure_code IS NOT NULL AND TRIM(procedure_code) != ''
                    GROUP BY 1
                ),
                -- ICD columns: union all into one stream for protocol and total
                prot_icd AS (
                    SELECT 'ICD' AS code_type, code, COUNT(*) AS protocol_count
                    FROM (
                        SELECT primary_icd_diagnosis_code AS code FROM prot
                        UNION ALL SELECT two_icd_diagnosis_code AS code FROM prot
                        UNION ALL SELECT three_icd_diagnosis_code AS code FROM prot
                        UNION ALL SELECT four_icd_diagnosis_code AS code FROM prot
                        UNION ALL SELECT five_icd_diagnosis_code AS code FROM prot
                    )
                    WHERE code IS NOT NULL AND TRIM(code) != ''
                    GROUP BY 1, 2
                ),
                all_icd AS (
                    SELECT code, COUNT(*) AS total_count
                    FROM (
                        SELECT primary_icd_diagnosis_code AS code FROM traj
                        UNION ALL SELECT two_icd_diagnosis_code AS code FROM traj
                        UNION ALL SELECT three_icd_diagnosis_code AS code FROM traj
                        UNION ALL SELECT four_icd_diagnosis_code AS code FROM traj
                        UNION ALL SELECT five_icd_diagnosis_code AS code FROM traj
                    )
                    WHERE code IS NOT NULL AND TRIM(code) != ''
                    GROUP BY 1
                ),
                merged AS (
                    SELECT d.code_type, d.code, d.protocol_count, a.total_count
                    FROM prot_drug d
                    JOIN all_drug a USING (code)
                    UNION ALL
                    SELECT c.code_type, c.code, c.protocol_count, a.total_count
                    FROM prot_cpt c
                    JOIN all_cpt a USING (code)
                    UNION ALL
                    SELECT i.code_type, i.code, i.protocol_count, a.total_count
                    FROM prot_icd i
                    JOIN all_icd a USING (code)
                )
                SELECT
                    code_type,
                    code,
                    protocol_count,
                    total_count,
                    (protocol_count * 100.0) / NULLIF(total_count, 0) AS protocol_pct
                FROM merged
                ORDER BY protocol_pct DESC, protocol_count DESC
            ) TO '{code_analysis_path.as_posix()}' (HEADER, DELIMITER ',');
            """
        )
        logger.info(f"Saved code analysis (protocol vs clinical): {code_analysis_path}")

        # ------------------------------------------------------------------
        # Summary report (DuckDB aggregates only)
        # ------------------------------------------------------------------
        summary_report = con.execute(
            f"""
            SELECT
                COUNT(*) AS total_events,
                SUM(is_protocol_event) AS protocol_events,
                AVG(is_protocol_event) * 100.0 AS protocol_event_pct,
                AVG(days_since_previous) AS mean_interval_days,
                MEDIAN(days_since_previous) AS median_interval_days,
                COUNT(DISTINCT mi_person_key) AS unique_patients,
                COUNT(DISTINCT drug_name) FILTER (WHERE drug_name IS NOT NULL AND TRIM(drug_name) != '') AS unique_drugs,
                COUNT(DISTINCT primary_icd_diagnosis_code) FILTER (WHERE primary_icd_diagnosis_code IS NOT NULL AND TRIM(primary_icd_diagnosis_code) != '') AS unique_icd_codes,
                COUNT(DISTINCT procedure_code) FILTER (WHERE procedure_code IS NOT NULL AND TRIM(procedure_code) != '') AS unique_cpt_codes
            FROM traj
            """
        ).fetchone()

        import json
        summary_path = review_dir / f"research_summary_{cohort_name}_{age_band_fname}.json"
        summary_obj = {
            'cohort_name': cohort_name,
            'age_band': age_band,
            'min_interval_days': min_interval_days,
            'total_events': int(summary_report[0] or 0),
            'protocol_events': int(summary_report[1] or 0),
            'protocol_event_pct': float(summary_report[2] or 0.0),
            'mean_interval_days': float(summary_report[3] or 0.0) if summary_report[3] is not None else None,
            'median_interval_days': float(summary_report[4] or 0.0) if summary_report[4] is not None else None,
            'unique_patients': int(summary_report[5] or 0),
            'unique_drugs': int(summary_report[6] or 0),
            'unique_icd_codes': int(summary_report[7] or 0),
            'unique_cpt_codes': int(summary_report[8] or 0),
        }
        with open(summary_path, 'w') as f:
            json.dump(summary_obj, f, indent=2)
        logger.info(f"Saved research summary: {summary_path}")

        logger.info(f"\nResearch outputs saved to: {review_dir}")
        logger.info("Files created:")
        logger.info(f"  - trajectories_with_time_windows_{cohort_name}_{age_band_fname}.parquet")
        logger.info(f"  - time_window_statistics_{cohort_name}_{age_band_fname}.csv")
        logger.info(f"  - activity_vocab_{cohort_name}_{age_band_fname}.parquet")
        logger.info(f"  - bigrams_{cohort_name}_{age_band_fname}.parquet")
        logger.info(f"  - trigrams_{cohort_name}_{age_band_fname}.parquet")
        logger.info(f"  - common_sequence_patterns_{cohort_name}_{age_band_fname}.csv")
        logger.info(f"  - top_100_sequences_{cohort_name}_{age_band_fname}.csv")
        logger.info(f"  - protocol_like_sequences_{cohort_name}_{age_band_fname}.parquet")
        logger.info(f"  - protocol_events_with_codes_{cohort_name}_{age_band_fname}.parquet")
        logger.info(f"  - code_analysis_protocol_vs_clinical_{cohort_name}_{age_band_fname}.csv")
        logger.info(f"  - research_summary_{cohort_name}_{age_band_fname}.json")
    finally:
        con.close()
    
    # (Summary report and logging handled above in DuckDB-only section)


def create_protocol_summary(
    intervals_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Create summary statistics about protocol events.

    Returns
    -------
    pd.DataFrame
        Summary statistics per patient and overall
    """
    patient_summary = intervals_df.groupby("mi_person_key").agg(
        {
            "is_protocol_event": ["sum", "mean", "count"],
            "days_since_previous": ["mean", "median", "min", "max"],
        }
    )

    patient_summary = patient_summary.reset_index()
    patient_summary.columns = [
        "mi_person_key",
        "protocol_event_count",
        "protocol_event_pct",
        "total_events",
        "mean_interval_days",
        "median_interval_days",
        "min_interval_days",
        "max_interval_days",
    ]

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        patient_summary.to_csv(output_path, index=False)
        logger.info("Saved protocol summary to {0}".format(output_path))

    return patient_summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter administrative events using code classification and post-event leakage filtering"
    )
    parser.add_argument(
        "--cohort-name",
        type=str,
        required=True,
        help="Cohort name (e.g., opioid_ed)",
    )
    parser.add_argument(
        "--age-band",
        type=str,
        required=True,
        help="Age band (e.g., 0-12)",
    )
    parser.add_argument(
        "--min-interval-days",
        type=int,
        default=1,
        help="Minimum interval (days) for time window analysis in research outputs (default: 1, matches BupaR). Note: Filtering is based on code classification, not time intervals.",
    )
    parser.add_argument(
        "--keep-first-event",
        action="store_true",
        default=True,
        help="Always keep first event per patient (even if administrative)",
    )
    parser.add_argument(
        "--admin-code-threshold-pct",
        type=float,
        default=80.0,
        help=(
            "Threshold for considering a code administrative from research outputs "
            "(codes with > this % in protocol-like sequences are considered administrative, default: 80.0)"
        ),
    )

    args = parser.parse_args()

    age_band_fname = args.age_band.replace("-", "_")

    # Validate and filter aggregated feature importance CSV before proceeding
    logger.info("Validating and cleaning aggregated feature importance CSV...")
    fi_validation = _validate_and_filter_aggregated_feature_importance(args.cohort_name, args.age_band)
    
    if not fi_validation["is_valid"]:
        logger.error(f"❌ Aggregated feature importance validation failed: {fi_validation.get('error', 'Unknown error')}")
        logger.error(
            "Please generate the baseline aggregated feature importance (first pass) by running Step 3a with --baseline:\n"
            "  python 3a_feature_importance/run_mc_feature_importance.py --cohort %s --age_band %s --baseline\n"
            "Then re-run this event filter. For second-pass FI (after event filter), run without --baseline.",
            args.cohort_name,
            args.age_band,
        )
        sys.exit(1)
    else:
        if fi_validation["n_zero_importance"] > 0 or fi_validation["n_duplicates"] > 0:
            logger.info(
                f"✓ Cleaned aggregated feature importance CSV: "
                f"removed {fi_validation['n_zero_importance']} zero-importance features, "
                f"{fi_validation['n_duplicates']} duplicates. "
                f"Final: {fi_validation['n_features_final']} features (from {fi_validation['n_features_initial']} initial)"
            )
        else:
            logger.info(
                f"✓ Aggregated feature importance CSV is clean: "
                f"{fi_validation['n_features_final']} features, all with importance > 0, no duplicates"
            )
        print(f"\n[INFO] Final aggregated feature importance count: {fi_validation['n_features_final']} features")

    # Use OS-aware path resolution for model_events.parquet (needed for local check)
    model_data_path = _resolve_model_events_path(args.cohort_name, args.age_band)
    
    # Output path: use same base directory as input
    output_path = (
        model_data_path.parent
        / "model_events_no_protocols.parquet"
    )

    # Output paths for audit artifacts (needed for local check)
    audit_dir = OUTPUT_ROOT / args.cohort_name / age_band_fname
    summary_path = audit_dir / f"protocol_summary_{args.cohort_name}_{age_band_fname}.csv"
    intervals_path = audit_dir / f"event_intervals_{args.cohort_name}_{age_band_fname}.parquet"

    # Check for existing local outputs (idempotency - check local first)
    if output_path.exists():
        logger.info(f"Filtered dataset already exists locally: {output_path}")
        logger.info(f"Checking if all outputs are present...")
        
        # Check if all expected outputs exist
        all_outputs_exist = (
            output_path.exists() and
            summary_path.exists() and
            intervals_path.exists()
        )
        
        if all_outputs_exist:
            logger.info(f"Step 4b outputs already exist locally for {args.cohort_name}/{args.age_band}; skipping.")
            logger.info(f"  Main output: {output_path}")
            logger.info(f"  Summary: {summary_path}")
            logger.info(f"  Intervals: {intervals_path}")
            
            # Still try to upload to S3 if not already there (idempotent upload)
            try:
                from py_helpers.checkpoint_utils import upload_file_to_s3, save_step_checkpoint
                
                s3_outputs = []
                s3_output_path = f"s3://pgxdatalake/gold/event_filter/{args.cohort_name}/{args.age_band}/model_events_no_protocols.parquet"
                if upload_file_to_s3(output_path, s3_output_path, logger):
                    s3_outputs.append(s3_output_path)
                
                s3_summary_path = f"s3://pgxdatalake/gold/event_filter/{args.cohort_name}/{args.age_band}/protocol_summary_{args.cohort_name}_{age_band_fname}.csv"
                if upload_file_to_s3(summary_path, s3_summary_path, logger):
                    s3_outputs.append(s3_summary_path)
                
                s3_intervals_path = f"s3://pgxdatalake/gold/event_filter/{args.cohort_name}/{args.age_band}/event_intervals_{args.cohort_name}_{age_band_fname}.parquet"
                if upload_file_to_s3(intervals_path, s3_intervals_path, logger):
                    s3_outputs.append(s3_intervals_path)
                
                # Save checkpoint if outputs uploaded
                if s3_outputs:
                    save_step_checkpoint(
                        step_name="1b_apcd_event_filter",
                        cohort=args.cohort_name,
                        age_band=args.age_band,
                        metadata={
                            "original_events": "unknown",  # Would need to read from file
                            "filtered_events": "unknown",
                        },
                        output_paths=s3_outputs,
                    )
            except ImportError:
                pass  # S3 upload is optional
            
            sys.exit(0)
        else:
            logger.warning(f"Some Step 4b outputs are missing. Will regenerate all outputs.")
            logger.warning(f"  Main output exists: {output_path.exists()}")
            logger.warning(f"  Summary exists: {summary_path.exists()}")
            logger.warning(f"  Intervals exists: {intervals_path.exists()}")

    # Check S3 for existing outputs (idempotency - fallback if local doesn't exist)
    try:
        from py_helpers.checkpoint_utils import check_step_outputs_exist, check_step_checkpoint_exists

        # Define expected S3 output paths
        s3_output_paths = [
            f"s3://pgxdatalake/gold/event_filter/{args.cohort_name}/{args.age_band}/model_events_no_protocols.parquet",
            f"s3://pgxdatalake/gold/event_filter/{args.cohort_name}/{args.age_band}/protocol_summary_{args.cohort_name}_{age_band_fname}.csv",
            f"s3://pgxdatalake/gold/event_filter/{args.cohort_name}/{args.age_band}/event_intervals_{args.cohort_name}_{age_band_fname}.parquet",
        ]

        if check_step_outputs_exist(s3_output_paths, logger) or check_step_checkpoint_exists("1b_apcd_event_filter", args.cohort_name, args.age_band, logger):
            logger.info(f"Step 4b outputs already exist in S3 for {args.cohort_name}/{args.age_band}; downloading to local.")
            
            # Download from S3 to local
            try:
                import boto3
                s3_client = boto3.client("s3")
                S3_BUCKET = "pgxdatalake"
                
                # Download main output
                s3_key = f"gold/event_filter/{args.cohort_name}/{args.age_band}/model_events_no_protocols.parquet"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                s3_client.download_file(S3_BUCKET, s3_key, str(output_path))
                logger.info(f"Downloaded {output_path} from S3")
                
                # Download summary
                s3_key = f"gold/event_filter/{args.cohort_name}/{args.age_band}/protocol_summary_{args.cohort_name}_{age_band_fname}.csv"
                audit_dir.mkdir(parents=True, exist_ok=True)
                s3_client.download_file(S3_BUCKET, s3_key, str(summary_path))
                logger.info(f"Downloaded {summary_path} from S3")
                
                # Download intervals
                s3_key = f"gold/event_filter/{args.cohort_name}/{args.age_band}/event_intervals_{args.cohort_name}_{age_band_fname}.parquet"
                s3_client.download_file(S3_BUCKET, s3_key, str(intervals_path))
                logger.info(f"Downloaded {intervals_path} from S3")
                
                logger.info(f"Step 4b outputs downloaded from S3; skipping regeneration.")
                sys.exit(0)
            except Exception as e:
                logger.warning(f"Could not download from S3: {e}. Will regenerate outputs.")
    except ImportError:
        pass  # Fallback to local-only if checkpoint_utils not available

    # audit_dir, summary_path, and intervals_path are already defined above
    # Just ensure the directory exists
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Load original data count (avoid materializing full dataset in pandas)
    con = duckdb.connect()
    original_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{model_data_path}')"
    ).fetchone()[0]
    con.close()

    # Step 1: Calculate time intervals (for research purposes)
    intervals_df = calculate_event_intervals(model_data_path, args.min_interval_days)

    # Persist full event-level intervals with protocol flags for audit/exploration
    intervals_df.to_parquet(intervals_path, index=False)
    logger.info("Saved event-level intervals to {0}".format(intervals_path))

    # Per-patient summary
    create_protocol_summary(intervals_df, summary_path)
    
    # Step 2: Create comprehensive research outputs for review (used to identify administrative codes)
    create_research_outputs_for_review(
        intervals_df=intervals_df,
        model_data_path=model_data_path,
        cohort_name=args.cohort_name,
        age_band=args.age_band,
        min_interval_days=args.min_interval_days,
    )

    # Step 3: Filter based on aggregated feature importance first, then administrative codes
    allowed_codes_from_fi = get_allowed_codes_from_aggregated_fi(fi_validation["cleaned_path"])
    logger.info(
        "Using %s allowed codes from aggregated feature importance for first-pass filter",
        len(allowed_codes_from_fi),
    )
    filtered_stats = filter_administrative_events(
        model_data_path=model_data_path,
        output_path=output_path,
        cohort_name=args.cohort_name,
        age_band=args.age_band,
        administrative_codes=None,  # Will load from research outputs
        keep_first_event=args.keep_first_event,
        admin_code_threshold_pct=args.admin_code_threshold_pct,
        allowed_codes_from_fi=allowed_codes_from_fi,
    )

    print("\n[INFO] Event filtering complete (aggregated FI first, then administrative)!")
    print(f"  Original events: {original_count:,}")
    if filtered_stats.get("fi_removed_events", 0) > 0:
        print(f"  After FI filter: {filtered_stats['original_events'] - filtered_stats['fi_removed_events']:,} (removed {filtered_stats['fi_removed_events']:,} low-importance)")
    print(f"  Filtered events: {filtered_stats['filtered_events']:,}")
    print(f"  Removed (total): {filtered_stats['removed_events']:,} ({100.0 * filtered_stats['removed_events'] / max(original_count, 1):.1f}%)")
    print("\n[INFO] Research outputs saved to:")
    print(f"  {OUTPUT_ROOT / 'for_review' / args.cohort_name / age_band_fname}")
    print("\n[INFO] Next steps:")
    print("  1. Review code_analysis_protocol_vs_clinical_*.csv to identify administrative codes")
    print("  2. Re-run filter to apply code-based filtering (will use research outputs)")

    print("\n[INFO] Protocol filtering complete!")
    print(f"  Original events: {original_count:,}")
    print(f"  Filtered events: {filtered_stats['filtered_events']:,}")
    print(f"  Removed: {filtered_stats['removed_events']:,} ({100.0 * filtered_stats['removed_events'] / max(original_count, 1):.1f}%)")

    # Upload outputs to S3 and save checkpoint
    try:
        from py_helpers.checkpoint_utils import upload_file_to_s3, save_step_checkpoint

        # Upload main outputs
        s3_outputs = []
        if output_path.exists():
            s3_output_path = f"s3://pgxdatalake/gold/event_filter/{args.cohort_name}/{args.age_band}/model_events_no_protocols.parquet"
            if upload_file_to_s3(output_path, s3_output_path, logger):
                s3_outputs.append(s3_output_path)

        if summary_path.exists():
            s3_summary_path = f"s3://pgxdatalake/gold/event_filter/{args.cohort_name}/{args.age_band}/protocol_summary_{args.cohort_name}_{age_band_fname}.csv"
            if upload_file_to_s3(summary_path, s3_summary_path, logger):
                s3_outputs.append(s3_summary_path)

        if intervals_path.exists():
            s3_intervals_path = f"s3://pgxdatalake/gold/event_filter/{args.cohort_name}/{args.age_band}/event_intervals_{args.cohort_name}_{age_band_fname}.parquet"
            if upload_file_to_s3(intervals_path, s3_intervals_path, logger):
                s3_outputs.append(s3_intervals_path)

        # Save checkpoint
        save_step_checkpoint(
            step_name="1b_apcd_event_filter",
            cohort=args.cohort_name,
            age_band=args.age_band,
            metadata={
                "total_events": int(original_count),
                "filtered_events": int(filtered_stats.get("filtered_events", 0)),
                "protocol_events": int(intervals_df['is_protocol_event'].sum()) if 'is_protocol_event' in intervals_df.columns else 0,
            },
            output_paths=s3_outputs,
            logger=logger,
        )
    except ImportError:
        pass  # Checkpoint saving is optional
