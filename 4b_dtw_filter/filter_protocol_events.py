#!/usr/bin/env python3
"""
Filter out administrative events from model_data before feature engineering.

This script classifies events as administrative vs. medical/pharmacy related and filters
out administrative events (billing, scheduling, post-event documentation) regardless
of time intervals. Medical and pharmacy events are kept even if they occur close together.

The classification is based on:
1. Code patterns (ICD, CPT, drug codes) that indicate administrative vs. clinical events
2. Research outputs that identify which codes are administrative
3. Post-event events (events occurring after target event date - these are leakage)

Time window analysis is still performed for research purposes, but filtering is based
on code classification, not time intervals.
"""

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

OUTPUT_ROOT = PROJECT_ROOT / "4b_dtw_filter" / "outputs"

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
    
    # Try local path first
    agg_csv_path = (
        PROJECT_ROOT
        / "3_feature_importance"
        / "outputs"
        / cohort
        / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
    )
    
    # Fallback: try S3 download location
    if not agg_csv_path.exists():
        agg_csv_path = (
            PROJECT_ROOT
            / "3_feature_importance"
            / "from_s3"
            / "by_cohort"
            / cohort
            / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
        )
    
    if not agg_csv_path.exists():
        return {
            "is_valid": False,
            "n_features_initial": 0,
            "n_features_final": 0,
            "n_zero_importance": 0,
            "n_duplicates": 0,
            "cleaned_path": None,
            "error": f"Aggregated feature importance CSV not found for {cohort}/{age_band}. Expected at: {agg_csv_path}",
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
    1. get_data_root()/4a_model_data/... (/mnt/nvme/4a_model_data/...)
    2. PROJECT_ROOT/4a_model_data/... (fallback)
    3. Try downloading from S3 to get_data_root() if not found locally
    
    Priority on Windows:
    1. PROJECT_ROOT/4a_model_data/... (Windows/local dev)
    2. get_data_root()/4a_model_data/... (fallback)
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
            data_root / "4a_model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
            PROJECT_ROOT / "4a_model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
        ]
        # Download destination: prefer data root on Linux
        download_dest = candidates[0]
    else:
        # On Windows: prioritize project root
        candidates = [
            PROJECT_ROOT / "4a_model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
            data_root / "4a_model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
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
                    f"Please run: python 4a_model_data/create_model_data.py"
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
                    f"Please run: python 4a_model_data/create_model_data.py --cohort {cohort} --age-band {age_band}"
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
                    f"Please run: python 4a_model_data/create_model_data.py --cohort {cohort} --age-band {age_band}"
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
                if str(icd_code) in administrative_codes.get('icd', set()):
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
    WITH patient_events AS (
        SELECT
            mi_person_key,
            event_date,
            target,
            drug_name,
            primary_icd_diagnosis_code,
            procedure_code,
            ROW_NUMBER() OVER (
                PARTITION BY mi_person_key
                ORDER BY event_date
            ) AS event_seq
        FROM read_parquet('{model_data_path}')
        WHERE event_date IS NOT NULL
    ),
    event_intervals AS (
        SELECT
            pe1.mi_person_key,
            pe1.event_date AS current_event_date,
            pe1.target,
            pe1.drug_name,
            pe1.primary_icd_diagnosis_code,
            pe1.procedure_code,
            pe1.event_seq,
            pe2.event_date AS previous_event_date,
            DATEDIFF('day', pe2.event_date, pe1.event_date) AS days_since_previous,
            CASE
                WHEN pe2.event_date IS NULL THEN 1  -- First event
                WHEN DATEDIFF('day', pe2.event_date, pe1.event_date) < {min_interval_days}
                    THEN 1  -- Protocol-like
                ELSE 0  -- Non-protocol
            END AS is_protocol_event
        FROM patient_events pe1
        LEFT JOIN patient_events pe2
            ON pe1.mi_person_key = pe2.mi_person_key
           AND pe1.event_seq = pe2.event_seq + 1
    )
    SELECT * FROM event_intervals
    ORDER BY mi_person_key, event_seq
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
) -> pd.DataFrame:
    """
    Filter out administrative events from model_data based on code classification.

    Parameters
    ----------
    model_data_path : Path
        Input model_events.parquet path
    output_path : Path
        Output filtered model_events.parquet path
    cohort_name : str
        Cohort name for determining target event date field
    age_band : str
        Age band for loading research outputs
    administrative_codes : Optional[dict]
        Dictionary with keys 'icd', 'cpt', 'drug' containing sets of administrative codes.
        If None, loads from research outputs.
    keep_first_event : bool
        If True, always keep the first event per patient (even if administrative)
    admin_code_threshold_pct : float
        Threshold for considering a code administrative from research (default: 80%)

    Returns
    -------
    pd.DataFrame
        Filtered model_data with administrative events removed
    """
    logger.info("Filtering administrative events from {0}".format(model_data_path))

    con = duckdb.connect()
    original_df = con.execute(
        f"SELECT * FROM read_parquet('{model_data_path}')"
    ).df()
    con.close()

    # Load administrative codes from research if not provided
    if administrative_codes is None:
        administrative_codes = load_administrative_codes_from_research(
            cohort_name=cohort_name,
            age_band=age_band,
            protocol_threshold_pct=admin_code_threshold_pct,
        )
        # If research outputs don't exist, start with empty sets (will only filter post-event leakage)
        if not administrative_codes['icd'] and not administrative_codes['cpt'] and not administrative_codes['drug']:
            logger.info(
                "No administrative codes found in research outputs. "
                "Will only filter post-event leakage (events after target date)."
            )

    # Rank events per patient
    event_seq = (
        original_df.groupby("mi_person_key")["event_date"]
        .rank(method="first", ascending=True)
    )
    original_df["event_seq"] = event_seq.fillna(0).astype(int)

    # Classify each event as administrative
    original_df["is_administrative"] = original_df.apply(
        lambda row: classify_event_as_administrative(
            row, administrative_codes, cohort_name
        ),
        axis=1,
    )

    # Filter logic: keep non-administrative events
    if keep_first_event:
        original_df["keep_event"] = (
            (original_df["event_seq"] == 1)  # Always keep first event
            | (~original_df["is_administrative"])  # Keep non-administrative events
        )
    else:
        original_df["keep_event"] = ~original_df["is_administrative"]

    filtered_df = original_df[original_df["keep_event"]].copy()

    # Drop helper columns
    filtered_df = filtered_df.drop(
        columns=[
            "event_seq",
            "is_administrative",
            "keep_event",
        ],
        errors="ignore",
    )

    admin_count = (original_df["is_administrative"] == True).sum()
    logger.info("Filtered {0} events -> {1} events".format(len(original_df), len(filtered_df)))
    logger.info(
        "Removed {0} administrative events ({1:.1f}%)".format(
            admin_count,
            100.0 * admin_count / max(len(original_df), 1),
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_parquet(output_path, index=False)
    logger.info("Saved filtered model_data to {0}".format(output_path))

    return filtered_df


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
    
    # 1. Save detailed event intervals with all event information
    con = duckdb.connect()
    
    # Load full event data
    full_events_df = con.execute(f"SELECT * FROM read_parquet('{model_data_path}')").df()
    
    # Merge with intervals_df
    # Add event_seq to full_events_df for merging
    full_events_df = full_events_df.sort_values(['mi_person_key', 'event_date'])
    full_events_df['event_seq'] = (
        full_events_df.groupby('mi_person_key').cumcount() + 1
    )
    
    # Merge intervals with full events
    full_events_df = full_events_df.merge(
        intervals_df[[
            'mi_person_key', 'event_seq', 'days_since_previous', 
            'is_protocol_event', 'previous_event_date'
        ]],
        on=['mi_person_key', 'event_seq'],
        how='left'
    )
    
    con.close()
    
    # Save full trajectories with time windows
    trajectories_path = review_dir / f"trajectories_with_time_windows_{cohort_name}_{age_band_fname}.parquet"
    full_events_df.to_parquet(trajectories_path, index=False)
    logger.info(f"Saved trajectories with time windows: {trajectories_path}")
    
    # 2. Time window statistics
    time_window_stats = intervals_df.groupby('mi_person_key').agg({
        'days_since_previous': ['mean', 'median', 'std', 'min', 'max', 'count'],
        'is_protocol_event': 'sum'
    }).reset_index()
    time_window_stats.columns = [
        'mi_person_key', 'mean_interval_days', 'median_interval_days', 
        'std_interval_days', 'min_interval_days', 'max_interval_days',
        'total_events', 'protocol_event_count'
    ]
    time_window_stats['protocol_event_pct'] = (
        time_window_stats['protocol_event_count'] / time_window_stats['total_events'] * 100
    )
    
    stats_path = review_dir / f"time_window_statistics_{cohort_name}_{age_band_fname}.csv"
    time_window_stats.to_csv(stats_path, index=False)
    logger.info(f"Saved time window statistics: {stats_path}")
    
    # 3. Common sequence patterns (2-3 event sequences)
    # Group events by patient and create sequences
    patient_sequences = []
    for patient_id in full_events_df['mi_person_key'].unique():
        patient_events = full_events_df[
            full_events_df['mi_person_key'] == patient_id
        ].sort_values('event_date')
        
        # Create activity codes (drug, ICD, CPT)
        activities = []
        for _, row in patient_events.iterrows():
            if pd.notna(row.get('drug_name')) and str(row.get('drug_name')).strip():
                activities.append(f"DRUG:{row['drug_name']}")
            if pd.notna(row.get('primary_icd_diagnosis_code')) and str(row.get('primary_icd_diagnosis_code')).strip():
                activities.append(f"ICD:{row['primary_icd_diagnosis_code']}")
            if pd.notna(row.get('procedure_code')) and str(row.get('procedure_code')).strip():
                activities.append(f"CPT:{row['procedure_code']}")
        
        # Extract 2-event sequences
        for i in range(len(activities) - 1):
            seq_2 = f"{activities[i]} -> {activities[i+1]}"
            patient_sequences.append({
                'mi_person_key': patient_id,
                'sequence': seq_2,
                'sequence_length': 2,
                'position': i
            })
        
        # Extract 3-event sequences
        for i in range(len(activities) - 2):
            seq_3 = f"{activities[i]} -> {activities[i+1]} -> {activities[i+2]}"
            patient_sequences.append({
                'mi_person_key': patient_id,
                'sequence': seq_3,
                'sequence_length': 3,
                'position': i
            })
    
    if patient_sequences:
        sequences_df = pd.DataFrame(patient_sequences)
        sequence_freq = sequences_df.groupby(['sequence', 'sequence_length']).size().reset_index(name='frequency')
        sequence_freq = sequence_freq.sort_values('frequency', ascending=False)
        
        sequences_path = review_dir / f"common_sequence_patterns_{cohort_name}_{age_band_fname}.csv"
        sequence_freq.to_csv(sequences_path, index=False)
        logger.info(f"Saved common sequence patterns: {sequences_path}")
        
        # Top 100 sequences
        top_sequences_path = review_dir / f"top_100_sequences_{cohort_name}_{age_band_fname}.csv"
        sequence_freq.head(100).to_csv(top_sequences_path, index=False)
        logger.info(f"Saved top 100 sequences: {top_sequences_path}")
    
    # 4. Protocol-like sequences (< min_interval_days apart)
    protocol_events = intervals_df[
        (intervals_df['days_since_previous'].notna()) &
        (intervals_df['days_since_previous'] < min_interval_days)
    ].copy()
    
    if not protocol_events.empty:
        protocol_path = review_dir / f"protocol_like_sequences_{cohort_name}_{age_band_fname}.parquet"
        protocol_events.to_parquet(protocol_path, index=False)
        logger.info(f"Saved protocol-like sequences: {protocol_path}")
        
        # Protocol sequence patterns with codes
        protocol_with_codes = full_events_df[
            full_events_df['is_protocol_event'] == 1
        ].copy()
        
        if not protocol_with_codes.empty:
            protocol_codes_path = review_dir / f"protocol_events_with_codes_{cohort_name}_{age_band_fname}.parquet"
            protocol_with_codes.to_parquet(protocol_codes_path, index=False)
            logger.info(f"Saved protocol events with codes: {protocol_codes_path}")
            
            # Code-level analysis: which codes appear in protocol events
            code_analysis = []
            
            # Drug codes in protocol events
            drug_codes = protocol_with_codes[
                protocol_with_codes['drug_name'].notna() &
                (protocol_with_codes['drug_name'] != '')
            ]['drug_name'].value_counts()
            for drug, count in drug_codes.items():
                code_analysis.append({
                    'code_type': 'DRUG',
                    'code': drug,
                    'protocol_count': count,
                    'total_count': len(full_events_df[full_events_df['drug_name'] == drug]),
                    'protocol_pct': count / len(full_events_df[full_events_df['drug_name'] == drug]) * 100 if len(full_events_df[full_events_df['drug_name'] == drug]) > 0 else 0
                })
            
            # ICD codes in protocol events
            for icd_col in ['primary_icd_diagnosis_code', 'two_icd_diagnosis_code', 
                           'three_icd_diagnosis_code', 'four_icd_diagnosis_code', 
                           'five_icd_diagnosis_code']:
                if icd_col in protocol_with_codes.columns:
                    icd_codes = protocol_with_codes[
                        protocol_with_codes[icd_col].notna() &
                        (protocol_with_codes[icd_col] != '')
                    ][icd_col].value_counts()
                    for icd, count in icd_codes.items():
                        code_analysis.append({
                            'code_type': 'ICD',
                            'code': icd,
                            'protocol_count': count,
                            'total_count': len(full_events_df[full_events_df[icd_col] == icd]),
                            'protocol_pct': count / len(full_events_df[full_events_df[icd_col] == icd]) * 100 if len(full_events_df[full_events_df[icd_col] == icd]) > 0 else 0
                        })
            
            # CPT codes in protocol events
            cpt_codes = protocol_with_codes[
                protocol_with_codes['procedure_code'].notna() &
                (protocol_with_codes['procedure_code'] != '')
            ]['procedure_code'].value_counts()
            for cpt, count in cpt_codes.items():
                code_analysis.append({
                    'code_type': 'CPT',
                    'code': cpt,
                    'protocol_count': count,
                    'total_count': len(full_events_df[full_events_df['procedure_code'] == cpt]),
                    'protocol_pct': count / len(full_events_df[full_events_df['procedure_code'] == cpt]) * 100 if len(full_events_df[full_events_df['procedure_code'] == cpt]) > 0 else 0
                })
            
            if code_analysis:
                code_analysis_df = pd.DataFrame(code_analysis)
                code_analysis_df = code_analysis_df.sort_values('protocol_pct', ascending=False)
                
                code_analysis_path = review_dir / f"code_analysis_protocol_vs_clinical_{cohort_name}_{age_band_fname}.csv"
                code_analysis_df.to_csv(code_analysis_path, index=False)
                logger.info(f"Saved code analysis (protocol vs clinical): {code_analysis_path}")
    
    # 5. Summary report
    summary_report = {
        'cohort_name': cohort_name,
        'age_band': age_band,
        'min_interval_days': min_interval_days,
        'total_events': len(full_events_df),
        'protocol_events': int(intervals_df['is_protocol_event'].sum()),
        'protocol_event_pct': float(intervals_df['is_protocol_event'].mean() * 100),
        'mean_interval_days': float(intervals_df['days_since_previous'].mean()),
        'median_interval_days': float(intervals_df['days_since_previous'].median()),
        'unique_patients': int(full_events_df['mi_person_key'].nunique()),
        'unique_drugs': int(full_events_df['drug_name'].nunique()) if 'drug_name' in full_events_df.columns else 0,
        'unique_icd_codes': int(full_events_df['primary_icd_diagnosis_code'].nunique()) if 'primary_icd_diagnosis_code' in full_events_df.columns else 0,
        'unique_cpt_codes': int(full_events_df['procedure_code'].nunique()) if 'procedure_code' in full_events_df.columns else 0,
    }
    
    import json
    summary_path = review_dir / f"research_summary_{cohort_name}_{age_band_fname}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=2)
    logger.info(f"Saved research summary: {summary_path}")
    
    logger.info(f"\nResearch outputs saved to: {review_dir}")
    logger.info("Files created:")
    logger.info(f"  - trajectories_with_time_windows_{cohort_name}_{age_band_fname}.parquet")
    logger.info(f"  - time_window_statistics_{cohort_name}_{age_band_fname}.csv")
    logger.info(f"  - common_sequence_patterns_{cohort_name}_{age_band_fname}.csv")
    logger.info(f"  - top_100_sequences_{cohort_name}_{age_band_fname}.csv")
    logger.info(f"  - protocol_like_sequences_{cohort_name}_{age_band_fname}.parquet")
    logger.info(f"  - protocol_events_with_codes_{cohort_name}_{age_band_fname}.parquet")
    logger.info(f"  - code_analysis_protocol_vs_clinical_{cohort_name}_{age_band_fname}.csv")
    logger.info(f"  - research_summary_{cohort_name}_{age_band_fname}.json")


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
        description="Filter protocol-like events using DTW time windows"
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
            f"Please regenerate the aggregated feature importance CSV by running Step 3:\n"
            f"  python 3_feature_importance/run_mc_feature_importance.py --cohort {args.cohort_name} --age_band {args.age_band} --force"
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
                s3_output_path = f"s3://pgxdatalake/gold/dtw_filter/{args.cohort_name}/{args.age_band}/model_events_no_protocols.parquet"
                if upload_file_to_s3(output_path, s3_output_path, logger):
                    s3_outputs.append(s3_output_path)
                
                s3_summary_path = f"s3://pgxdatalake/gold/dtw_filter/{args.cohort_name}/{args.age_band}/protocol_summary_{args.cohort_name}_{age_band_fname}.csv"
                if upload_file_to_s3(summary_path, s3_summary_path, logger):
                    s3_outputs.append(s3_summary_path)
                
                s3_intervals_path = f"s3://pgxdatalake/gold/dtw_filter/{args.cohort_name}/{args.age_band}/event_intervals_{args.cohort_name}_{age_band_fname}.parquet"
                if upload_file_to_s3(intervals_path, s3_intervals_path, logger):
                    s3_outputs.append(s3_intervals_path)
                
                # Save checkpoint if outputs uploaded
                if s3_outputs:
                    save_step_checkpoint(
                        step_name="4b_dtw_filter",
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
            f"s3://pgxdatalake/gold/dtw_filter/{args.cohort_name}/{args.age_band}/model_events_no_protocols.parquet",
            f"s3://pgxdatalake/gold/dtw_filter/{args.cohort_name}/{args.age_band}/protocol_summary_{args.cohort_name}_{age_band_fname}.csv",
            f"s3://pgxdatalake/gold/dtw_filter/{args.cohort_name}/{args.age_band}/event_intervals_{args.cohort_name}_{age_band_fname}.parquet",
        ]

        if check_step_outputs_exist(s3_output_paths, logger) or check_step_checkpoint_exists("4b_dtw_filter", args.cohort_name, args.age_band, logger):
            logger.info(f"Step 4b outputs already exist in S3 for {args.cohort_name}/{args.age_band}; downloading to local.")
            
            # Download from S3 to local
            try:
                import boto3
                s3_client = boto3.client("s3")
                S3_BUCKET = "pgxdatalake"
                
                # Download main output
                s3_key = f"gold/dtw_filter/{args.cohort_name}/{args.age_band}/model_events_no_protocols.parquet"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                s3_client.download_file(S3_BUCKET, s3_key, str(output_path))
                logger.info(f"Downloaded {output_path} from S3")
                
                # Download summary
                s3_key = f"gold/dtw_filter/{args.cohort_name}/{args.age_band}/protocol_summary_{args.cohort_name}_{age_band_fname}.csv"
                audit_dir.mkdir(parents=True, exist_ok=True)
                s3_client.download_file(S3_BUCKET, s3_key, str(summary_path))
                logger.info(f"Downloaded {summary_path} from S3")
                
                # Download intervals
                s3_key = f"gold/dtw_filter/{args.cohort_name}/{args.age_band}/event_intervals_{args.cohort_name}_{age_band_fname}.parquet"
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

    # Load original data to get count
    con = duckdb.connect()
    original_df = con.execute(
        f"SELECT * FROM read_parquet('{model_data_path}')"
    ).df()
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

    # Step 3: Filter based on code classification (administrative vs. medical/pharmacy)
    filtered_df = filter_administrative_events(
        model_data_path=model_data_path,
        output_path=output_path,
        cohort_name=args.cohort_name,
        age_band=args.age_band,
        administrative_codes=None,  # Will load from research outputs
        keep_first_event=args.keep_first_event,
        admin_code_threshold_pct=args.admin_code_threshold_pct,
    )

    print("\n[INFO] Administrative event filtering complete!")
    print(f"  Original events: {len(original_df)}")
    print(f"  Filtered events: {len(filtered_df)}")
    print(
        "  Removed: {0} ({1:.1f}%)".format(
            len(original_df) - len(filtered_df),
            100.0
            * (len(original_df) - len(filtered_df))
            / max(len(original_df), 1),
        )
    )
    print("\n[INFO] Research outputs saved to:")
    print(f"  {OUTPUT_ROOT / 'for_review' / args.cohort_name / age_band_fname}")
    print("\n[INFO] Next steps:")
    print("  1. Review code_analysis_protocol_vs_clinical_*.csv to identify administrative codes")
    print("  2. Re-run filter to apply code-based filtering (will use research outputs)")

    print("\n[INFO] Protocol filtering complete!")
    print(f"  Original events: {len(intervals_df)}")
    print(f"  Filtered events: {len(filtered_df)}")
    print(
        "  Removed: {0} ({1:.1f}%)".format(
            len(intervals_df) - len(filtered_df),
            100.0
            * (len(intervals_df) - len(filtered_df))
            / max(len(intervals_df), 1),
        )
    )

    # Upload outputs to S3 and save checkpoint
    try:
        from py_helpers.checkpoint_utils import upload_file_to_s3, save_step_checkpoint

        # Upload main outputs
        s3_outputs = []
        if output_path.exists():
            s3_output_path = f"s3://pgxdatalake/gold/dtw_filter/{args.cohort_name}/{args.age_band}/model_events_no_protocols.parquet"
            if upload_file_to_s3(output_path, s3_output_path, logger):
                s3_outputs.append(s3_output_path)

        if summary_path.exists():
            s3_summary_path = f"s3://pgxdatalake/gold/dtw_filter/{args.cohort_name}/{args.age_band}/protocol_summary_{args.cohort_name}_{age_band_fname}.csv"
            if upload_file_to_s3(summary_path, s3_summary_path, logger):
                s3_outputs.append(s3_summary_path)

        if intervals_path.exists():
            s3_intervals_path = f"s3://pgxdatalake/gold/dtw_filter/{args.cohort_name}/{args.age_band}/event_intervals_{args.cohort_name}_{age_band_fname}.parquet"
            if upload_file_to_s3(intervals_path, s3_intervals_path, logger):
                s3_outputs.append(s3_intervals_path)

        # Save checkpoint
        save_step_checkpoint(
            step_name="4b_dtw_filter",
            cohort=args.cohort_name,
            age_band=args.age_band,
            metadata={
                "total_events": len(original_df),
                "filtered_events": len(filtered_df),
                "protocol_events": int(intervals_df['is_protocol_event'].sum()) if 'is_protocol_event' in intervals_df.columns else 0,
            },
            output_paths=s3_outputs,
            logger=logger,
        )
    except ImportError:
        pass  # Checkpoint saving is optional

