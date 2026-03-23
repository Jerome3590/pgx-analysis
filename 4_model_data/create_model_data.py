#!/usr/bin/env python3
"""
Create model-ready event-level data filtered to important features, with
**within-cohort controls** constructed from gold medical/pharmacy tables.

This script is intentionally DuckDB + Parquet only for event-level data
to avoid pandas memory pressure on large cohorts:

1. Reads cohort_feature_importance CSVs from Step 3b/3c (refined feature list — the *filters*;
   these are not from Step 2; Step 2 is cohort creation only):
     - 3b_feature_importance_eda/outputs or DATA_ROOT/gold/feature_importance
   REQUIRED: Step 3b must run before Step 4a (will error if files not found)
2. Extracts the `feature` column (e.g., `item_99284`, `item_AMOXICILLIN`) and
   strips the `item_` prefix to get raw item codes.
3. For each (cohort_name, age_band) combination in those files, it:
   - reads Step 2 cohort parquet files (case/control membership, target dates) from local disk,
     typically under:
       PROJECT_ROOT/data/gold_cohorts/
         cohort_name={cohort_name}/event_year={year}/age_band={age_band}/cohort.parquet
   - reads gold medical / pharmacy events for the same age band and years:
       PROJECT_ROOT/data/gold_medical/age_band={age_band}/event_year={year}/*.parquet
       PROJECT_ROOT/data/gold_pharmacy/age_band={age_band}/event_year={year}/*.parquet
   - builds:
       * **cases** (target = 1):
           - patients with is_target_case = 1 in the cohort tables
           - events filtered by feature importance
             (drug_name, all ICD diagnosis columns, procedure_code)
       * **controls** (target = 0):
           - patients drawn from gold medical/pharmacy for the same age band
           - must have no opioid ICD codes in any diagnosis column across
             all their medical events, using OPIOID_ICD_CODES
           - must not appear in the case set for this cohort/age band
           - all medical + pharmacy events are kept (no FI-based filtering)
         Controls are sampled to maintain an approximate DEFAULT_SAMPLE_RATIO
         (e.g., 5:1) control:case patient ratio.
   - writes the combined events to:
       4_model_data/cohort_name={cohort_name}/age_band={age_band}/model_events.parquet
     with an event-level `target` column.
   - **Target leakage removal (Step 4):** For case events, keeps only events strictly before
     the target date (event_date < first_opioid_ed_date or first_ed_non_opioid_date). Events
     on or after the target date are dropped here (linear flow: 3b identifies leakage → 4 removes it).

This output is then used as input for:
 - FP-Growth (pattern mining on important features plus within-cohort controls)
 - BupaR (process mining / event-log analysis)
 - DTW (trajectory analysis on filtered event sequences)
 - Final models (Step 6)
"""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import duckdb
import pandas as pd

try:
    from py_helpers.common_imports import s3_client, S3_BUCKET
except ImportError:
    import boto3
    s3_client = boto3.client("s3")
    S3_BUCKET = "pgxdatalake"

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import (
    ALL_ICD_DIAGNOSIS_COLUMNS,
    DRUG_NAMES_EXCLUDED_MODEL_TRAINING,
    FEATURE_SUBSTRINGS_EXCLUDED,
    OPIOID_ICD_CODES,
    DEFAULT_SAMPLE_RATIO,
    get_opioid_icd_sql_condition,
    get_physical_age_bands_for_gold,
    get_physical_age_bands_for_medical_pharmacy,
    age_band_partition_candidates,
)
from py_helpers.env_utils import get_data_root, get_model_data_root
from py_helpers.feature_utils import feature_to_code

try:
    from py_helpers.fe_monitor import mirror_log_to_s3
except ImportError:
    mirror_log_to_s3 = None  # best-effort S3 upload

STEP3B_OUTPUTS_DIR = PROJECT_ROOT / "3b_feature_importance_eda" / "outputs"

# Explicit target date column names in model_events: first_{cohort_target}_date
# Opioid-ED: F11.20 = opioid use disorder ICD-10
# Polypharmacy: O11_P = canonical ED identifier (includes P51b, O11, P33), 21-day drug window
TARGET_DATE_COL_OPIOID = "first_f1120_date"
TARGET_DATE_COL_NON_OPIOID = "first_o11_p_date"
# Cohort (Step 2) column names we read from
COHORT_SOURCE_COL_OPIOID = "first_opioid_ed_date"
COHORT_SOURCE_COL_NON_OPIOID = "first_ed_non_opioid_date"

# Use exact cohort name so "non_opioid_ed" is not treated as opioid_ed (substring "opioid_ed" would match both)
def _is_opioid_cohort(cohort_name: str) -> bool:
    return (cohort_name or "").strip().lower() == "opioid_ed"


def _get_logger(cohort_name: str, age_band: str) -> tuple[logging.Logger, Path]:
    """Create logger with file and console handlers; log file under logs/4_model_data/."""
    logs_dir = PROJECT_ROOT / "logs" / "4_model_data"
    logs_dir.mkdir(parents=True, exist_ok=True)
    age_band_fname = age_band.replace("-", "_")
    log_path = logs_dir / f"create_model_data_{cohort_name}_{age_band_fname}.log"
    logger = logging.getLogger(f"4_model_data.{cohort_name}.{age_band_fname}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger, log_path


def get_step3b_fi_roots() -> list:
    """Roots for Step 3b cohort_feature_importance (NVMe/project). Prefer DATA_ROOT/gold/feature_importance."""
    data_root = get_data_root()
    return [
        data_root / "gold" / "feature_importance",
        STEP3B_OUTPUTS_DIR,
    ]
# Single canonical location: py_helpers.env_utils.get_model_data_root()
MODEL_DATA_ROOT = get_model_data_root()


def resolve_local_cohort_root() -> Path:
    """
    Resolve the root directory containing Step-2 **gold cohort** parquet files.

    Priority:
      1. LOCAL_DATA_PATH environment variable (if set)
      2. get_data_root()/gold/cohorts, then data/gold_cohorts alternatives
      3. PROJECT_ROOT/data/gold_cohorts (default)
    """
    env_path = os.getenv("LOCAL_DATA_PATH")
    if env_path:
        root = Path(env_path)
        print(f"[INFO] Cohort root: LOCAL_DATA_PATH={root}  exists={root.exists()}")
        if root.exists():
            return root

    data_root = get_data_root()
    candidates = [
        data_root / "gold" / "cohorts",  # Linux/EC2: /mnt/nvme/gold/cohorts
        data_root / "data" / "gold_cohorts",
        PROJECT_ROOT / "data" / "gold_cohorts",
    ]
    for path in candidates:
        if path.exists():
            print(f"[INFO] Cohort root: tried {[str(c) for c in candidates]} -> using {path} (first existing)")
            return path
    chosen = candidates[2]
    print(f"[INFO] Cohort root: tried {[str(c) for c in candidates]} (none existed) -> using default {chosen}")
    return chosen


def resolve_local_medical_root() -> Path:
    """
    Resolve the root directory containing gold medical event parquet files.

    Priority:
      1. LOCAL_MEDICAL_PATH environment variable
      2. get_data_root()/gold/medical, then data/gold_medical
      3. PROJECT_ROOT/data/gold_medical (default)
    """
    env_path = os.getenv("LOCAL_MEDICAL_PATH")
    if env_path:
        root = Path(env_path)
        print(f"[INFO] Medical root: LOCAL_MEDICAL_PATH={root}  exists={root.exists()}")
        if root.exists():
            return root

    data_root = get_data_root()
    candidates = [
        data_root / "gold" / "medical",
        data_root / "data" / "gold_medical",
        PROJECT_ROOT / "data" / "gold_medical",
    ]
    for path in candidates:
        if path.exists():
            print(f"[INFO] Medical root: tried {[str(c) for c in candidates]} -> using {path} (first existing)")
            return path
    chosen = candidates[2]
    print(f"[INFO] Medical root: tried {[str(c) for c in candidates]} (none existed) -> using default {chosen}")
    return chosen


def resolve_local_pharmacy_root() -> Path:
    """
    Resolve the root directory containing gold pharmacy event parquet files.

    Priority:
      1. LOCAL_PHARMACY_PATH environment variable
      2. get_data_root()/gold/pharmacy, then data/gold_pharmacy
      3. PROJECT_ROOT/data/gold_pharmacy (default)
    """
    env_path = os.getenv("LOCAL_PHARMACY_PATH")
    if env_path:
        root = Path(env_path)
        print(f"[INFO] Pharmacy root: LOCAL_PHARMACY_PATH={root}  exists={root.exists()}")
        if root.exists():
            return root

    data_root = get_data_root()
    candidates = [
        data_root / "gold" / "pharmacy",
        data_root / "data" / "gold_pharmacy",
        PROJECT_ROOT / "data" / "gold_pharmacy",
    ]
    for path in candidates:
        if path.exists():
            print(f"[INFO] Pharmacy root: tried {[str(c) for c in candidates]} -> using {path} (first existing)")
            return path
    chosen = candidates[2]
    print(f"[INFO] Pharmacy root: tried {[str(c) for c in candidates]} (none existed) -> using default {chosen}")
    return chosen


def parse_aggregated_filename(path: Path) -> Tuple[str, str]:
    """
    Parse cohort_name and age_band from a cohort_feature_importance CSV filename (Step 3b output).

    Expected pattern (from 3b_feature_importance_eda/outputs):
        {cohort_name}_{age_band_fname}_cohort_feature_importance.csv
    where age_band_fname is two numeric parts, e.g. 13_24 or 0_12.

    Example:
        opioid_ed_13_24_cohort_feature_importance.csv -> cohort_name=opioid_ed, age_band=13-24
        opioid_ed_0_12_cohort_feature_importance.csv  -> cohort_name=opioid_ed, age_band=0-12
    """
    stem = path.stem
    if not stem.endswith("_cohort_feature_importance"):
        raise ValueError(f"Unexpected feature importance filename format: {path.name}. Expected *_cohort_feature_importance.csv")

    prefix = stem[: -len("_cohort_feature_importance")]
    parts = prefix.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected refined filename format: {path.name} (prefix {prefix!r})")

    # Age band is always last two parts (e.g. 13_24, 0_12, 65_74)
    age_band_tokens = parts[-2:]
    cohort_name_tokens = parts[:-2]
    age_band_fname = "_".join(age_band_tokens)
    cohort_name = "_".join(cohort_name_tokens)
    age_band = age_band_fname.replace("_", "-")
    return cohort_name, age_band


def get_important_items(agg_csv: Path, cohort: Optional[str] = None) -> List[str]:
    """Read aggregated feature-importance CSV and return raw item codes for SQL matching.

    Step 3b CSVs use feature names like item_icd_F1120, item_cpt_80307, item_drug_SUBOXONE,
    and may include a raw_code column. Gold medical/pharmacy tables store raw codes:
    primary_icd_diagnosis_code='F1120', procedure_code='80307', drug_name='SUBOXONE'.
    We use raw_code when present (from 3b), else feature_to_code(feature), so that the
    filter matches all three sources; otherwise only drugs would match and ICD/CPT would
    never appear in model_events (opioid_ed would effectively get Drug-only features).

    For cohort non_opioid_ed (polypharmacy), only drug-name features are used (ICD/CPT dropped).

    Excludes drug names in DRUG_NAMES_EXCLUDED_MODEL_TRAINING (Narcan, Unknown, Fentanyl,
    1036F, T401XA1) so they are not used as features in model training."""
    df = pd.read_csv(agg_csv)
    if "feature" not in df.columns:
        raise ValueError(f"'feature' column not found in {agg_csv}")

    if cohort == "non_opioid_ed":
        from py_helpers.feature_utils import filter_fi_to_drug_only
        df = filter_fi_to_drug_only(df, feature_col="feature")

    # Prefer raw_code from Step 3b when present; else derive from feature column
    if "raw_code" in df.columns:
        raw_codes = df["raw_code"].astype(str).dropna().str.strip().replace("", pd.NA).dropna().unique().tolist()
    else:
        raw_codes = []
        for f in df["feature"].astype(str).unique().tolist():
            code = feature_to_code(f)
            if code and code.strip():
                raw_codes.append(code.strip())
    items = list(dict.fromkeys(raw_codes))  # preserve order, dedupe

    # Match case-insensitively so "NARCAN" / "Narcan" are both excluded; also exclude any item containing FEATURE_SUBSTRINGS_EXCLUDED (e.g. syringe)
    excluded_lower = {z.lower() for z in DRUG_NAMES_EXCLUDED_MODEL_TRAINING}
    filtered = [x for x in items if (x.strip().lower() if x else "") not in excluded_lower]
    before_substring = len(filtered)
    filtered = [x for x in filtered if not any((sub.lower() in (x or "").lower()) for sub in FEATURE_SUBSTRINGS_EXCLUDED)]
    if len(filtered) < len(items):
        n_removed = len(items) - len(filtered)
        removed = [x for x in items if (x.strip().lower() if x else "") in excluded_lower or any((sub.lower() in (x or "").lower()) for sub in FEATURE_SUBSTRINGS_EXCLUDED)]
        logging.getLogger(__name__).info(
            "Excluded %s item(s) from important_items (drug-name + substrings): %s", n_removed, removed[:15]
        )
    return filtered


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


def _validate_model_events_target_date_column(
    parquet_path: Path, cohort_name: str
) -> Tuple[bool, str]:
    """
    Validate that model_events.parquet has the required target date column and case rows have it set.
    Uses explicit names: first_f1120_date (opioid_ed), first_o11_p_date (non_opioid_ed).
    Accepts legacy column names as alias: first_opioid_ed_date (opioid), first_ed_non_opioid_date (non_opioid).

    Returns:
        (success: bool, message: str)
    """
    is_opioid = _is_opioid_cohort(cohort_name)
    canonical_col = TARGET_DATE_COL_OPIOID if is_opioid else TARGET_DATE_COL_NON_OPIOID
    legacy_col = COHORT_SOURCE_COL_OPIOID if is_opioid else COHORT_SOURCE_COL_NON_OPIOID
    path_str = str(parquet_path).replace("'", "''")
    con = duckdb.connect()
    try:
        schema = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{path_str}')"
        ).fetchall()
        col_names = [row[0] for row in schema]
        # Use canonical name if present, else accept legacy alias
        target_date_col = canonical_col if canonical_col in col_names else (legacy_col if legacy_col in col_names else None)
        if target_date_col is None:
            return (
                False,
                f"Output schema missing required column '{canonical_col}' (or legacy '{legacy_col}'). BupaR needs it for pre-target events.",
            )
        # For case rows (target=1), at least one must have non-null target date
        result = con.execute(
            f"""SELECT COUNT(*)::BIGINT AS n FROM read_parquet('{path_str}')
               WHERE target = 1 AND "{target_date_col}" IS NOT NULL"""
        ).fetchone()
        n_with_date = int(result[0]) if result and result[0] is not None else 0
        if n_with_date == 0:
            return (
                False,
                f"Case rows (target=1) have no non-null '{target_date_col}'; BupaR will get 0 pre-target events.",
            )
        alias_note = f" (canonical: {canonical_col})" if target_date_col != canonical_col else ""
        return True, f"Target date column '{target_date_col}'{alias_note} present; {n_with_date} case rows have it set."
    finally:
        con.close()


def load_control_exclusions(cohort_name: str, age_band: str, step3b_outputs_dir: Path) -> Optional[List[str]]:
    """
    Load control feature exclusions JSON and return list of item codes to exclude.
    
    Returns:
        List of item codes (without 'item_' prefix) to exclude for controls, or None if not found
    """
    import json
    from py_helpers.constants import age_band_to_fname
    
    age_band_fname = age_band_to_fname(age_band)
    exclusions_path = step3b_outputs_dir / cohort_name / age_band_fname / f"{cohort_name}_{age_band_fname}_control_feature_exclusions.json"
    
    if exclusions_path.exists():
        with open(exclusions_path, 'r') as f:
            exclusions_data = json.load(f)
        
        # Get features to exclude and remove 'item_' prefix
        features_to_exclude = exclusions_data.get('features_to_exclude', [])
        # Features are already normalized: item_80307, item_SUBOXONE, item_F1120
        # Just remove 'item_' prefix to get the code
        items_to_exclude = []
        for feature in features_to_exclude:
            if feature.startswith('item_'):
                code = feature[5:]  # Remove 'item_'
                items_to_exclude.append(code)
            else:
                # Already without prefix
                items_to_exclude.append(feature)
        
        return items_to_exclude
    
    return None


def filter_cohort_events_for_items(
    cohort_name: str,
    age_band: str,
    important_items: List[str],
    years: List[int],
    output_root: Path,
    local_cohort_root: Path,
    local_medical_root: Path,
    local_pharmacy_root: Path,
    sample_ratio: float = DEFAULT_SAMPLE_RATIO,
    control_exclusions: Optional[List[str]] = None,
    time_window_days: Optional[int] = None,  # Deprecated - time window now handled in Step 2
    skip_s3_download: bool = False,  # When True (e.g. 3b BupaR input), build locally only, no S3
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Build model-ready event data for a single cohort/age-band and write to 4_model_data/.

    For the given (cohort_name, age_band, years), this function:
      - reads the Step-2 cohort parquet(s) for that cohort from LOCAL storage
        (gold cohorts),
      - for **cases** (is_target_case = 1 in the cohort tables):
          * keeps only events where ANY of the item-bearing columns match an
            important item (drug_name, all ICD diagnosis columns, procedure_code),
      - for **controls** (target = 0):
          * selects patients from gold medical/pharmacy in the same age_band/years
          * excludes any patient who has an opioid ICD code in any diagnosis
            column across their medical events
          * excludes any patient whose mi_person_key is in the case set
          * samples patients to approximate `sample_ratio` controls per case
          * keeps all medical + pharmacy events for selected controls
      - writes the combined events to:
            4_model_data/cohort_name={cohort_name}/age_band={age_band}/model_events.parquet
        with an event-level `target` column.

    All heavy lifting is done in DuckDB; pandas is not used for event-level data.
    """
    # If important_items is empty, create model_events.parquet with ALL events (no filtering)
    # This allows Step 3 to run and generate feature importance, then Step 4a can be re-run with filtering
    use_all_events = len(important_items) == 0
    if use_all_events:
        print(f"[INFO] No feature importance CSVs found. Creating model_events.parquet with ALL events (no filtering) for {cohort_name}/{age_band}.")

    # Build list of local cohort parquet paths for this cohort/age_band across years.
    # For 85-114: prefer single partition age_band=85-114 when present; else use 85-94 and 95-114.
    # Try both hyphen and underscore partition names for gold data.
    cohort_parquet_paths: List[str] = []
    physical_bands = get_physical_age_bands_for_gold(age_band)
    print(f"[INFO] Cohort parquet search: root={local_cohort_root}  (age_band {age_band} -> physical {physical_bands}, try hyphen and underscore)")
    for year in years:
        added_for_year = False
        for physical in physical_bands:
            if added_for_year:
                break
            for part in age_band_partition_candidates(physical):
                p = (
                    local_cohort_root
                    / f"cohort_name={cohort_name}"
                    / f"event_year={year}"
                    / f"age_band={part}"
                    / "cohort.parquet"
                )
                if p.exists():
                    cohort_parquet_paths.append(str(p))
                    added_for_year = True
                    # For 85-114, using single partition; don't also add 85-94/95-114 this year
                    if physical == "85-114":
                        break
                    break
                print(f"[INFO]   {p}  -> MISSING")
            if added_for_year:
                print(f"[INFO]   ... (year {year} found, skip remaining physical bands)")
        if not added_for_year:
            for physical in physical_bands:
                for part in age_band_partition_candidates(physical):
                    p = (
                        local_cohort_root
                        / f"cohort_name={cohort_name}"
                        / f"event_year={year}"
                        / f"age_band={part}"
                        / "cohort.parquet"
                    )
                    print(f"[INFO]   {p}  -> MISSING")
    for p in cohort_parquet_paths:
        print(f"[INFO]   (using) {p}")

    if not cohort_parquet_paths:
        msg = (
            f"[WARN] No local cohort parquet files found for {cohort_name}/{age_band} "
            f"across years {years}. Did you run aws s3 sync into {local_cohort_root}?"
        )
        print(msg)
        if logger:
            logger.warning(msg)
        return

    # Build lists of gold medical and pharmacy parquet paths for this age_band across years.
    # For 85-114, medical/pharmacy use two sub-cohorts (85-94 and 95-114); cohort uses single 85-114.
    medical_pharmacy_bands = get_physical_age_bands_for_medical_pharmacy(age_band)
    medical_parquet_paths: List[str] = []
    pharmacy_parquet_paths: List[str] = []

    print(f"[INFO] Gold medical search: root={local_medical_root}  (age_band {age_band} -> medical/pharmacy physical {medical_pharmacy_bands})")
    print(f"[INFO] Gold pharmacy search: root={local_pharmacy_root}  (age_band {age_band} -> medical/pharmacy physical {medical_pharmacy_bands})")
    for year in years:
        for physical in medical_pharmacy_bands:
            med_files = []
            pharm_files = []
            for part in age_band_partition_candidates(physical):
                medical_parent = (
                    local_medical_root
                    / f"age_band={part}"
                    / f"event_year={year}"
                )
                pharmacy_parent = (
                    local_pharmacy_root
                    / f"age_band={part}"
                    / f"event_year={year}"
                )
                if not med_files and medical_parent.exists():
                    med_files = list(medical_parent.glob("*.parquet"))
                if not pharm_files and pharmacy_parent.exists():
                    pharm_files = list(pharmacy_parent.glob("*.parquet"))
                if med_files and pharm_files:
                    break
            print(
                f"[INFO]   medical   {local_medical_root}/age_band={physical}/event_year={year}  -> "
                + (f"{len(med_files)} file(s)" if med_files else "MISSING or no *.parquet")
            )
            print(
                f"[INFO]   pharmacy  {local_pharmacy_root}/age_band={physical}/event_year={year}  -> "
                + (f"{len(pharm_files)} file(s)" if pharm_files else "MISSING or no *.parquet")
            )

            for p in med_files:
                medical_parquet_paths.append(str(p))
            for p in pharm_files:
                pharmacy_parquet_paths.append(str(p))

    if not medical_parquet_paths:
        msg = (
            f"[WARN] No gold medical parquet files found for age_band={age_band} "
            f"across years {years}. Controls cannot be constructed; skipping."
        )
        print(msg)
        if logger:
            logger.warning(msg)
        return

    if not pharmacy_parquet_paths:
        msg = (
            f"[WARN] No gold pharmacy parquet files found for age_band={age_band} "
            f"across years {years}. Controls cannot be fully constructed; skipping."
        )
        print(msg)
        if logger:
            logger.warning(msg)
        return

    # Use DuckDB to read and filter in one pass
    con = duckdb.connect()
    cohort_paths_literal = ", ".join(f"'{p}'" for p in cohort_parquet_paths)
    gold_medical_paths_literal = ", ".join(f"'{p}'" for p in medical_parquet_paths)
    gold_pharmacy_paths_literal = ", ".join(f"'{p}'" for p in pharmacy_parquet_paths)
    all_control_paths_literal = ", ".join(
        f"'{p}'" for p in (medical_parquet_paths + pharmacy_parquet_paths)
    )

    # Build SQL filter condition for items
    # If important_items is empty, don't filter (keep all events)
    if use_all_events:
        item_filter_condition = "TRUE"  # Keep all events
    else:
        item_list_literal = ", ".join(f"'{v}'" for v in important_items)
        if not item_list_literal:
            item_list_literal = "''"  # Empty string to avoid SQL syntax error

        # Build ICD diagnosis conditions dynamically from ALL_ICD_DIAGNOSIS_COLUMNS
        icd_conditions = " OR ".join(
            f"{col} IN ({item_list_literal})" for col in ALL_ICD_DIAGNOSIS_COLUMNS
        )
        if not icd_conditions:
            icd_conditions = "FALSE"  # Empty condition
        
        # Build the full filter condition
        item_filter_condition = f"""(
            drug_name IN ({item_list_literal}) OR
            {icd_conditions} OR
            procedure_code IN ({item_list_literal})
        )"""

    if use_all_events:
        print(
            f"[INFO] Building model events for {cohort_name}/{age_band} "
            f"from {len(cohort_parquet_paths)} cohort files, "
            f"{len(medical_parquet_paths)} medical globs, "
            f"{len(pharmacy_parquet_paths)} pharmacy globs, "
            f"with ALL events (no filtering - feature importance CSVs not found)."
        )
    else:
        print(
            f"[INFO] Building model events for {cohort_name}/{age_band} "
            f"from {len(cohort_parquet_paths)} cohort files, "
            f"{len(medical_parquet_paths)} medical globs, "
            f"{len(pharmacy_parquet_paths)} pharmacy globs, "
            f"using {len(important_items)} important items."
        )

    # Write to flat path so Step 5 (PGx) and Step 6 (final model) find it:
    # 4_model_data/cohort_name={cohort_name}/age_band={age_band}/model_events.parquet
    out_dir = (
        output_root
        / f"cohort_name={cohort_name}"
        / f"age_band={age_band}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_events.parquet"
    out_path_msg = f"[INFO] Model data output (4_model_data): {out_path.resolve()}"
    print(out_path_msg)
    if logger:
        logger.info(out_path_msg)

    # S3 path aligned with Step 6 download candidates (gold/cohorts_model_data/...)
    s3_output_path = (
        f"s3://pgxdatalake/gold/cohorts_model_data/"
        f"cohort_name={cohort_name}/age_band={age_band}/model_events.parquet"
    )

    # Idempotency / Windows-friendly: if the file already exists locally, assume this
    # cohort/age_band has been built successfully and skip overwriting. This
    # avoids DuckDB attempting to delete an open file and throwing WinError 32.
    if out_path.exists():
        msg = (
            f"[INFO] model_events.parquet already exists at {out_path.resolve()}; "
            f"skipping rebuild for {cohort_name}/{age_band}."
        )
        print(msg)
        if logger:
            logger.info(msg)
        con.close()
        return

    # Check S3 and download if exists there but not locally (skip when building 3b BupaR input)
    if not skip_s3_download:
        try:
            from py_helpers.checkpoint_utils import check_s3_output_exists
            import subprocess
            import shutil

            if check_s3_output_exists(s3_output_path):
                # File exists in S3 but not locally - download it
                print(
                    f"[INFO] model_events.parquet exists in S3 but not locally. Downloading from S3..."
                )
                aws_cli = shutil.which("aws")
                if aws_cli:
                    result = subprocess.run(
                        [aws_cli, "s3", "cp", s3_output_path, str(out_path), "--no-progress"],
                        capture_output=True,
                        text=True,
                        timeout=300,
                        check=False,
                    )
                    if result.returncode == 0 and out_path.exists():
                        print(f"[INFO] Successfully downloaded from S3: {out_path}")
                        # Validate downloaded file has controls
                        validation_result = _validate_model_events_has_controls(out_path)
                        if validation_result["has_controls"]:
                            print(
                                f"[INFO] Downloaded file validated: {validation_result['n_cases']} cases, "
                                f"{validation_result['n_controls']} controls"
                            )
                            con.close()
                            return
                        else:
                            print(
                                f"[WARN] Downloaded file from S3 is missing controls! "
                                f"Cases: {validation_result['n_cases']}, Controls: {validation_result['n_controls']}. "
                                f"Will rebuild..."
                            )
                            out_path.unlink()  # Delete invalid file, will rebuild below
                    else:
                        print(f"[WARN] Failed to download from S3: {result.stderr if result.stderr else 'Unknown error'}")
                else:
                    print("[WARN] AWS CLI not found, cannot download from S3")
        except ImportError:
            pass  # Fallback to local check if checkpoint_utils not available
        except Exception as e:
            print(f"[WARN] Error checking/downloading from S3: {e}")

    # Derive a common set of columns present in both cohort and control sources,
    # so that set operations (UNION ALL) are well-defined.
    # Use union_by_name=True so multi-partition cohort (e.g. 85-94 + 95-114) exposes all columns.
    cohort_cols = [
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet([{cohort_paths_literal}], union_by_name=True)"
        ).fetchall()
    ]
    control_cols = [
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet([{all_control_paths_literal}], union_by_name=True)"
        ).fetchall()
    ]
    common_cols = [c for c in cohort_cols if c in control_cols]
    if not common_cols:
        print(
            f"[WARN] No common columns between cohort and control sources for "
            f"{cohort_name}/{age_band}; skipping."
        )
        con.close()
        return

    # Require target date column in cohort (BupaR/dashboard need it for pre-target split).
    # We read cohort source column and write explicit output column: first_f1120_date / first_o11_p_date.
    is_opioid = _is_opioid_cohort(cohort_name)
    output_col = TARGET_DATE_COL_OPIOID if is_opioid else TARGET_DATE_COL_NON_OPIOID
    source_col = COHORT_SOURCE_COL_OPIOID if is_opioid else COHORT_SOURCE_COL_NON_OPIOID
    # non_opioid_ed: allow first_opioid_ed_date as fallback if first_ed_non_opioid_date missing (legacy schema)
    if not is_opioid and source_col not in cohort_cols and COHORT_SOURCE_COL_OPIOID in cohort_cols:
        source_col = COHORT_SOURCE_COL_OPIOID
        print(f"[INFO] Using cohort column '{source_col}' AS '{output_col}' (legacy schema).")
    if source_col not in cohort_cols:
        msg = (
            f"[ERROR] Cohort schema missing required target date column '{source_col}' for {cohort_name}/{age_band}. "
            f"Cohort parquets must include this column (Step 2). Found columns: {cohort_cols[:20]}{'...' if len(cohort_cols) > 20 else ''}. "
            f"Refusing to write model_events without it (BupaR would get 0 pre-target events)."
        )
        print(msg)
        if logger:
            logger.error(msg)
        con.close()
        return
    if source_col not in common_cols:
        common_cols = list(common_cols) + [source_col]
        print(f"[INFO] Including cohort-only column in model_events for target date (output: {output_col}).")

    # Output schema uses explicit names (first_f1120_date / first_o11_p_date)
    output_common_cols = [output_col if c == source_col else c for c in common_cols]
    # Case SELECT: from cohort, alias source_col -> output_col
    case_cols_sql = ", ".join(
        f'"{source_col}" AS "{output_col}"' if c == output_col else c for c in output_common_cols
    )
    common_cols_sql_control = ", ".join(
        f"NULL AS \"{output_col}\"" if c == output_col else (f"c.{c}" if c in control_cols else f"NULL AS {c}")
        for c in output_common_cols
    )

    # 1. Case patients from gold cohorts
    # NOTE: Time window filtering is now handled in Step 2 (2_create_cohort)
    # Step 2 creates cohorts with first ED visit (HCG Setting) within 21 days of drug event; we use all target cases from the cohort
    # No need to re-filter here - the cohort definition in Step 2 is the source of truth
    if False:  # Disabled - time window filtering moved to Step 2
        # Filter target cases to only those with first ED (HCG) within time_window_days of drug events
        # Need to get gold medical/pharmacy paths for time window checking
        gold_medical_paths_literal = ", ".join(f"'{p}'" for p in medical_parquet_paths) if medical_parquet_paths else ""
        gold_pharmacy_paths_literal = ", ".join(f"'{p}'" for p in pharmacy_parquet_paths) if pharmacy_parquet_paths else ""
        
        if not gold_medical_paths_literal or not gold_pharmacy_paths_literal:
            print(f"[WARN] Cannot apply time window filtering: missing medical or pharmacy files")
            case_patients_query = f"""
                CREATE TEMP TABLE case_patients AS
                SELECT DISTINCT mi_person_key
                FROM read_parquet([{cohort_paths_literal}], union_by_name=True)
                WHERE is_target_case = 1
            """
        else:
            case_patients_query = f"""
                CREATE TEMP TABLE case_patients AS
                WITH cohort_cases AS (
                    SELECT DISTINCT mi_person_key
                    FROM read_parquet([{cohort_paths_literal}], union_by_name=True)
                    WHERE is_target_case = 1
                ),
                pharmacy_events AS (
                    SELECT
                        mi_person_key,
                        CASE 
                            WHEN LENGTH(CAST(incurred_date AS VARCHAR)) = 8 THEN 
                                CAST(SUBSTR(CAST(incurred_date AS VARCHAR), 1, 4) || '-' || 
                                     SUBSTR(CAST(incurred_date AS VARCHAR), 5, 2) || '-' || 
                                     SUBSTR(CAST(incurred_date AS VARCHAR), 7, 2) AS DATE)
                            ELSE CAST(incurred_date AS DATE)
                        END AS event_date
                    FROM read_parquet([{gold_pharmacy_paths_literal}])
                ),
                medical_events AS (
                    SELECT
                        mi_person_key,
                        CASE 
                            WHEN LENGTH(CAST(incurred_date AS VARCHAR)) = 8 THEN 
                                CAST(SUBSTR(CAST(incurred_date AS VARCHAR), 1, 4) || '-' || 
                                     SUBSTR(CAST(incurred_date AS VARCHAR), 5, 2) || '-' || 
                                     SUBSTR(CAST(incurred_date AS VARCHAR), 7, 2) AS DATE)
                            ELSE CAST(incurred_date AS DATE)
                        END AS event_date,
                        hcg_line
                    FROM read_parquet([{gold_medical_paths_literal}])
                ),
                patient_hcg_dates AS (
                    SELECT
                        me.mi_person_key,
                        me.event_date AS hcg_event_date
                    FROM medical_events me
                    WHERE me.hcg_line IN ('P51 - ER Visits and Observation Care', 'O11 - Emergency Room', 'P33 - Urgent Care Visits')
                ),
                drug_hcg_pairs AS (
                    -- Check if ANY first ED (HCG) within time_window_days of drug event
                    SELECT DISTINCT
                        pe.mi_person_key
                    FROM pharmacy_events pe
                    INNER JOIN cohort_cases cc ON pe.mi_person_key = cc.mi_person_key
                    INNER JOIN patient_hcg_dates phd ON pe.mi_person_key = phd.mi_person_key
                        AND phd.hcg_event_date >= pe.event_date
                        AND phd.hcg_event_date <= DATE_ADD(pe.event_date, INTERVAL {time_window_days} DAY)
                )
                SELECT DISTINCT mi_person_key
                FROM drug_hcg_pairs
            """
            print(f"[INFO] Filtering target cases for polypharmacy cohort with {time_window_days}-day time window")
    else:
        # Standard case: use all target cases from cohort
        case_patients_query = f"""
            CREATE TEMP TABLE case_patients AS
            SELECT DISTINCT mi_person_key
            FROM read_parquet([{cohort_paths_literal}], union_by_name=True)
            WHERE is_target_case = 1
        """
    con.execute(case_patients_query)

    # Check number of cases; if zero, skip
    n_cases = con.execute("SELECT COUNT(*) FROM case_patients").fetchone()[0]
    if n_cases == 0:
        print(
            f"[WARN] No case patients found for {cohort_name}/{age_band}; skipping."
        )
        con.close()
        return

    # 2. Control candidates from gold medical + pharmacy, excluding patients
    #    with opioid ICDs and excluding case patients.
    opioid_condition = get_opioid_icd_sql_condition(table_alias="ue")

    control_candidates_query = f"""
        CREATE TEMP TABLE control_candidates AS
        WITH unified_gold_events AS (
            SELECT
                mi_person_key,
                primary_icd_diagnosis_code,
                two_icd_diagnosis_code,
                three_icd_diagnosis_code,
                four_icd_diagnosis_code,
                five_icd_diagnosis_code,
                six_icd_diagnosis_code,
                seven_icd_diagnosis_code,
                eight_icd_diagnosis_code,
                nine_icd_diagnosis_code,
                ten_icd_diagnosis_code
            FROM read_parquet([{gold_medical_paths_literal}])
            UNION ALL
            SELECT
                mi_person_key,
                NULL AS primary_icd_diagnosis_code,
                NULL AS two_icd_diagnosis_code,
                NULL AS three_icd_diagnosis_code,
                NULL AS four_icd_diagnosis_code,
                NULL AS five_icd_diagnosis_code,
                NULL AS six_icd_diagnosis_code,
                NULL AS seven_icd_diagnosis_code,
                NULL AS eight_icd_diagnosis_code,
                NULL AS nine_icd_diagnosis_code,
                NULL AS ten_icd_diagnosis_code
            FROM read_parquet([{gold_pharmacy_paths_literal}])
        ),
        per_patient_icd_check AS (
            SELECT
                mi_person_key,
                MAX(
                    CASE
                        WHEN {opioid_condition} THEN 1
                        ELSE 0
                    END
                ) AS has_opioid_icd
            FROM unified_gold_events ue
            GROUP BY mi_person_key
        )
        SELECT
            pp.mi_person_key
        FROM per_patient_icd_check pp
        LEFT JOIN case_patients cp
            ON pp.mi_person_key = cp.mi_person_key
        WHERE
            pp.has_opioid_icd = 0
            AND cp.mi_person_key IS NULL
    """
    con.execute(control_candidates_query)

    n_candidate_controls = con.execute(
        "SELECT COUNT(*) FROM control_candidates"
    ).fetchone()[0]
    if n_candidate_controls == 0:
        print(
            f"[WARN] No eligible control patients found for {cohort_name}/{age_band}; "
            f"using cases only."
        )
        # In this degenerate case, just build case-only events (with target leakage removal).
        # Use source_col (same as main path) for leakage filter.
        leakage_condition = "TRUE"
        if "event_date" in cohort_cols and source_col in cohort_cols:
            leakage_condition = (
                f"(event_date IS NULL OR \"{source_col}\" IS NULL OR "
                f"CAST(event_date AS DATE) < CAST(\"{source_col}\" AS DATE))"
            )
        final_query = f"""
            COPY (
                SELECT
                    *,
                    1 AS target
                FROM read_parquet([{cohort_paths_literal}], union_by_name=True)
                WHERE
                    is_target_case = 1 AND {item_filter_condition} AND {leakage_condition}
            ) TO '{str(out_path)}'
            (FORMAT PARQUET)
        """
        con.execute(final_query)
        con.close()
        print(
            f"[INFO] Wrote case-only model_events.parquet for {cohort_name}/{age_band}: {out_path}"
        )
        td_ok, td_msg = _validate_model_events_target_date_column(out_path, cohort_name)
        if not td_ok:
            print(f"[ERROR] {td_msg} File: {out_path}")
            sys.exit(1)
        print(f"[INFO] {td_msg}")
        return

    # 3. Sample control patients to maintain approximate sample_ratio:1 control:case
    desired_controls = int(sample_ratio * n_cases)
    if desired_controls <= 0:
        desired_controls = n_candidate_controls
    else:
        desired_controls = min(desired_controls, n_candidate_controls)

    con.execute(
        f"""
        CREATE TEMP TABLE control_patients AS
        SELECT mi_person_key
        FROM control_candidates
        ORDER BY random()
        LIMIT {desired_controls}
        """
    )

    # 4. Construct case and control events and write to Parquet
    # Target leakage removal (Step 4): keep only events strictly before target date for cases.
    # Use source_col (cohort column name) for the filter since we read from cohort.
    leakage_condition = "TRUE"
    if "event_date" in common_cols and source_col in common_cols:
        leakage_condition = (
            f"(event_date IS NULL OR \"{source_col}\" IS NULL OR "
            f"CAST(event_date AS DATE) < CAST(\"{source_col}\" AS DATE))"
        )
        print(f"[INFO] Applying target leakage removal: keep only events before {source_col}")
    case_events_query = f"""
        SELECT
            {case_cols_sql},
            1 AS target
        FROM read_parquet([{cohort_paths_literal}], union_by_name=True)
        WHERE
            is_target_case = 1 AND {item_filter_condition} AND {leakage_condition}
    """

    # Build control exclusion filter (blacklist approach)
    # Controls keep all features EXCEPT post-target leakage features
    control_exclusion_condition = "TRUE"  # Default: no exclusions
    if control_exclusions and len(control_exclusions) > 0:
        exclusion_list_literal = ", ".join(f"'{v}'" for v in control_exclusions)
        # Build exclusion conditions for all item-bearing columns
        exclusion_icd_conditions = " OR ".join(
            f"{col} IN ({exclusion_list_literal})" for col in ALL_ICD_DIAGNOSIS_COLUMNS
        )
        control_exclusion_condition = f"""NOT (
            drug_name IN ({exclusion_list_literal}) OR
            {exclusion_icd_conditions} OR
            procedure_code IN ({exclusion_list_literal})
        )"""
        print(f"[INFO] Applying control exclusions: excluding {len(control_exclusions)} post-target leakage features")
    
    control_events_query = f"""
        SELECT
            {common_cols_sql_control},
            0 AS target
        FROM read_parquet([{all_control_paths_literal}], union_by_name=True) c
        JOIN control_patients cp
            ON c.mi_person_key = cp.mi_person_key
        WHERE {control_exclusion_condition}
    """

    final_query = f"""
        COPY (
            {case_events_query}
            UNION ALL
            {control_events_query}
        ) TO '{str(out_path)}'
        (FORMAT PARQUET)
    """

    con.execute(final_query)
    con.close()

    write_msg = f"[INFO] Wrote model_events.parquet for {cohort_name}/{age_band}: {out_path}"
    print(write_msg)
    if logger:
        logger.info(write_msg)
    
    # Validate that controls are present and ratio is approximately correct
    validation_result = _validate_model_events_has_controls(out_path)
    if not validation_result["has_controls"]:
        print(
            f"[ERROR] Generated file {out_path} is missing controls! "
            f"Cases: {validation_result['n_cases']}, Controls: {validation_result['n_controls']}"
        )
        sys.exit(1)

    # Validate target date column present and set for cases (required for BupaR pre-target split)
    td_ok, td_msg = _validate_model_events_target_date_column(out_path, cohort_name)
    if not td_ok:
        print(f"[ERROR] {td_msg} File: {out_path}")
        sys.exit(1)
    print(f"[INFO] {td_msg}")
    
    # Validate control:case ratio (should be approximately sample_ratio:1)
    n_cases = validation_result['n_cases']
    n_controls = validation_result['n_controls']
    actual_ratio = n_controls / max(n_cases, 1)
    expected_ratio = sample_ratio
    
    # Allow 20% tolerance (e.g., 4:1 to 6:1 for 5:1 target)
    tolerance = 0.2
    min_ratio = expected_ratio * (1 - tolerance)
    max_ratio = expected_ratio * (1 + tolerance)
    
    if actual_ratio < min_ratio or actual_ratio > max_ratio:
        print(
            f"[WARN] Control:case ratio is {actual_ratio:.2f}:1, expected approximately "
            f"{expected_ratio}:1 (tolerance: {min_ratio:.2f}-{max_ratio:.2f}:1). "
            f"Cases: {n_cases}, Controls: {n_controls}"
        )
        # Don't fail - this is a warning, not an error (may be due to limited control candidates)
    else:
        print(
            f"[INFO] Control:case ratio validation passed: {actual_ratio:.2f}:1 "
            f"(target: {expected_ratio}:1)"
        )
    
    print(
        f"[INFO] Validation passed: {n_cases} cases, {n_controls} controls"
    )
    
    # Upload to S3 using aws s3 sync (best-effort)
    _sync_model_events_to_s3(out_path, cohort_name, age_band)

    # Save checkpoint to S3
    try:
        from py_helpers.checkpoint_utils import save_step_checkpoint
        save_step_checkpoint(
            step_name="4_model_data",
            cohort=cohort_name,
            age_band=age_band,
            metadata={
                "n_cases": validation_result["n_cases"],
                "n_controls": validation_result["n_controls"],
                "local_path": str(out_path),
            },
            output_paths=[s3_output_path],
        )
    except ImportError:
        pass  # Checkpoint saving is optional


def _sync_model_events_to_s3(parquet_path: Path, cohort_name: str, age_band: str) -> None:
    """
    Sync model_events.parquet to S3 using aws s3 sync.
    S3 path aligned with Step 6: gold/cohorts_model_data/cohort_name={cohort_name}/age_band={age_band}/
    """
    aws_cli = shutil.which("aws")
    if not aws_cli:
        print("[WARN] AWS CLI not found, skipping S3 sync")
        return

    s3_path = (
        f"s3://pgxdatalake/gold/cohorts_model_data/"
        f"cohort_name={cohort_name}/age_band={age_band}/"
    )
    
    # Use s3 sync to upload the file (syncs the directory)
    local_dir = parquet_path.parent
    
    try:
        print(f"[INFO] Syncing to S3: {s3_path}")
        # aws s3 sync will overwrite if local file is newer or different
        # Use --delete to remove files in S3 that don't exist locally (we don't want this)
        # Just sync the specific file - it will overwrite if it exists
        result = subprocess.run(
            [aws_cli, "s3", "sync", str(local_dir), s3_path, "--exclude", "*", "--include", "model_events.parquet", "--no-progress"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            check=False,  # Don't raise on error, just log
        )
        if result.returncode == 0:
            print(f"[INFO] Successfully synced to S3: {s3_path}model_events.parquet")
        else:
            print(f"[WARN] S3 sync failed: {result.stderr if result.stderr else 'Unknown error'}")
    except subprocess.TimeoutExpired:
        print(f"[WARN] S3 sync timed out after 5 minutes")
    except Exception as e:
        print(f"[WARN] Error syncing to S3: {e}")


def _step3b_download_root() -> Path:
    """Prefer first existing Step 3b root (NVMe then project) for downloads."""
    for root in get_step3b_fi_roots():
        return root
    return STEP3B_OUTPUTS_DIR


def download_cohort_feature_importance_from_s3(cohort: Optional[str] = None, age_band: Optional[str] = None) -> List[Path]:
    """
    Download cohort_feature_importance files from S3 (Step 3b outputs).
    
    If cohort and age_band are specified, downloads only that file.
    Otherwise, lists all available files in S3.
    Writes to first Step 3b root (DATA_ROOT/gold/feature_importance when used).
    """
    downloaded_files = []
    download_root = _step3b_download_root()

    if cohort and age_band:
        # Download specific file
        age_band_fname = age_band.replace("-", "_")
        s3_key = (
            f"gold/feature_importance/{cohort}/{age_band}/"
            f"{cohort}_{age_band_fname}_cohort_feature_importance.csv"
        )
        local_path = download_root / cohort / age_band_fname / f"{cohort}_{age_band_fname}_cohort_feature_importance.csv"
        
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
            print(f"[INFO] Downloading from S3: s3://{S3_BUCKET}/{s3_key}")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            with open(local_path, 'wb') as f:
                f.write(obj['Body'].read())
            print(f"[INFO] Saved locally: {local_path}")
            downloaded_files.append(local_path)
        except Exception as e:
            print(f"[WARN] Could not download {s3_key}: {e}")
    else:
        # List all cohorts and age bands from S3
        prefix = "gold/feature_importance/"
        try:
            paginator = s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix, Delimiter="/"):
                # Process each cohort
                for prefix_info in page.get("CommonPrefixes", []):
                    cohort_prefix = prefix_info["Prefix"]
                    cohort_name = cohort_prefix.split("/")[-2]
                    
                    # List age bands for this cohort
                    for age_page in paginator.paginate(Bucket=S3_BUCKET, Prefix=cohort_prefix, Delimiter="/"):
                        for age_prefix_info in age_page.get("CommonPrefixes", []):
                            age_band = age_prefix_info["Prefix"].split("/")[-2]
                            age_band_fname = age_band.replace("-", "_")
                            
                            s3_key = f"{age_prefix_info['Prefix']}{cohort_name}_{age_band_fname}_cohort_feature_importance.csv"
                            local_path = download_root / cohort_name / age_band_fname / f"{cohort_name}_{age_band_fname}_cohort_feature_importance.csv"
                            
                            try:
                                s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
                                print(f"[INFO] Downloading from S3: s3://{S3_BUCKET}/{s3_key}")
                                local_path.parent.mkdir(parents=True, exist_ok=True)
                                obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
                                with open(local_path, 'wb') as f:
                                    f.write(obj['Body'].read())
                                print(f"[INFO] Saved locally: {local_path}")
                                downloaded_files.append(local_path)
                            except Exception as e:
                                print(f"[WARN] Could not download {s3_key}: {e}")
        except Exception as e:
            print(f"[WARN] Error listing S3 files: {e}")
    
    return downloaded_files


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create model_events.parquet files with cases and controls"
    )
    parser.add_argument(
        "--cohort",
        type=str,
        help="Process specific cohort (e.g., opioid_ed). If not specified, processes all found cohorts.",
    )
    parser.add_argument(
        "--age-band",
        type=str,
        help="Process specific age band (e.g., 13-24). Requires --cohort to be specified.",
    )
    parser.add_argument(
        "--time-window-days",
        type=int,
        default=None,
        choices=[7, 14, 21, 30, 45],
        help="DEPRECATED: Time window is now handled in Step 2 (2_create_cohort). This argument is ignored.",
    )
    args = parser.parse_args()
    
    # Ensure local directories exist (idempotent: we overwrite per file)
    # Re-resolve MODEL_DATA_ROOT in case get_data_root() behavior changed
    model_data_root = get_model_data_root()
    model_data_root.mkdir(parents=True, exist_ok=True)

    # Discover cohort_feature_importance CSVs from Step 3b (REQUIRED - no fallback)
    # Step 3b must run before Step 4a to produce refined feature importances
    # Expected location: 3b_feature_importance_eda/outputs/{cohort}/{age_band}/{cohort}_{age_band}_cohort_feature_importance.csv
    aggregated_files = []
    
    # If both cohort and age_band are specified, look for specific file
    if args.cohort and args.age_band:
        age_band_fname = args.age_band.replace("-", "_")
        fname = f"{args.cohort}_{age_band_fname}_cohort_feature_importance.csv"
        # Check for Step 3b refined feature importance (REQUIRED): NVMe then project
        refined_file = None
        for root in get_step3b_fi_roots():
            candidate = root / args.cohort / age_band_fname / fname
            if candidate.exists():
                refined_file = candidate
                break
        if refined_file is not None:
            aggregated_files.append(refined_file)
            print(f"[INFO] Found Step 3b refined feature importance: {refined_file}")
        else:
            # Try downloading from S3 if not found locally
            print(f"[INFO] Step 3b refined feature importance not found locally: {refined_file}")
            print(f"[INFO] Attempting to download from S3...")
            downloaded = download_cohort_feature_importance_from_s3(args.cohort, args.age_band)
            if downloaded and downloaded[0].exists():
                aggregated_files.append(downloaded[0])
                print(f"[INFO] Successfully downloaded from S3: {downloaded[0]}")
            else:
                # Error out - file not found locally or in S3
                print(f"[ERROR] Step 3b refined feature importance not found locally or in S3")
                expected_path = STEP3B_OUTPUTS_DIR / args.cohort / age_band_fname / fname
                print(f"[ERROR] Expected (check NVMe and project): {expected_path}")
                print(f"[ERROR] S3 path: s3://{S3_BUCKET}/gold/feature_importance/{args.cohort}/{args.age_band}/{fname}")
                print(f"[ERROR] Step 3b must run before Step 4a to produce cohort_feature_importance files")
                print(f"[ERROR] Run: python 3b_feature_importance_eda/run_feature_importance_eda.py --cohort {args.cohort} --age-band {args.age_band}")
                sys.exit(1)
    else:
        # Discover Step 3b refined files from NVMe then project roots (dedupe by cohort/age_band)
        seen = set()
        for root in get_step3b_fi_roots():
            if not root.exists():
                continue
            for cohort_dir in root.iterdir():
                if not cohort_dir.is_dir():
                    continue
                if args.cohort and cohort_dir.name != args.cohort:
                    continue
                for age_band_dir in cohort_dir.iterdir():
                    if not age_band_dir.is_dir():
                        continue
                    key = (cohort_dir.name, age_band_dir.name)
                    if key in seen:
                        continue
                    refined_files = sorted(
                        age_band_dir.glob("*_cohort_feature_importance.csv")
                    )
                    if refined_files:
                        seen.add(key)
                        aggregated_files.append(refined_files[0])
                    else:
                        cohort_name = cohort_dir.name
                        age_band = age_band_dir.name.replace("_", "-")
                        print(f"[INFO] No cohort_feature_importance.csv in {age_band_dir}")
                        print(f"[INFO] Attempting download from S3 for {cohort_name}/{age_band}...")
                        downloaded = download_cohort_feature_importance_from_s3(cohort_name, age_band)
                        if downloaded:
                            seen.add(key)
                            aggregated_files.append(downloaded[0])
                            print(f"[INFO] Downloaded from S3: {downloaded[0]}")
                        else:
                            print(f"[WARN] Could not download. Step 3b required for {cohort_name}/{age_band}")
        if not aggregated_files:
            print(f"[ERROR] No cohort_feature_importance files found in Step 3b roots (NVMe or project)")
            print(f"[ERROR] Step 3b must run before Step 4a")
            sys.exit(1)
    
    if not aggregated_files:
        print(
            f"[ERROR] No cohort_feature_importance CSVs found."
        )
        if args.cohort and args.age_band:
            age_band_fname = args.age_band.replace("-", "_")
            expected_file = STEP3B_OUTPUTS_DIR / args.cohort / age_band_fname / f"{args.cohort}_{age_band_fname}_cohort_feature_importance.csv"
            print(
                f"[ERROR] Expected file: {expected_file}"
            )
            print(
                f"[ERROR] Step 3b must run before Step 4a to produce cohort_feature_importance files."
            )
            print(
                f"[ERROR] Run: python 3b_feature_importance_eda/run_step_3b.py --cohort {args.cohort} --age-band {args.age_band}"
            )
        sys.exit(1)

    # Default years: match feature-importance temporal setup (2016–2018 train, 2019 test)
    YEARS = [2016, 2017, 2018, 2019]

    local_cohort_root = resolve_local_cohort_root()
    local_medical_root = resolve_local_medical_root()
    local_pharmacy_root = resolve_local_pharmacy_root()

    print(f"[INFO] Step 4 data roots: cohorts={local_cohort_root}, medical={local_medical_root}, pharmacy={local_pharmacy_root}")
    example_cohort = local_cohort_root / "cohort_name=opioid_ed" / "event_year=2016" / "age_band=13-24" / "cohort.parquet"
    print(f"[INFO] Example cohort path (must exist for build to run): {example_cohort}  exists={example_cohort.exists()}")

    for agg_path in aggregated_files:
        try:
            cohort_name, age_band = parse_aggregated_filename(agg_path)
        except ValueError as e:
            print(f"[WARN] Skipping {agg_path.name}: {e}")
            continue
        
        # Filter by command-line arguments if provided
        if args.cohort and cohort_name != args.cohort:
            continue
        if args.age_band and age_band != args.age_band:
            continue

        print(
            f"\n=== Processing cohort={cohort_name}, age_band={age_band} "
            f"from {agg_path.name} ==="
        )
        important_items = get_important_items(agg_path, cohort=cohort_name)
        if not important_items:
            print(
                f"[WARN] No important items extracted from {agg_path.name}; "
                f"skipping {cohort_name}/{age_band}."
            )
            continue

        # Load control exclusions (blacklist for controls); use same root as cohort_feature_importance
        step3b_root = agg_path.parent.parent.parent
        control_exclusions = load_control_exclusions(cohort_name, age_band, step3b_root)
        if control_exclusions:
            print(f"[INFO] Loaded {len(control_exclusions)} control exclusions for {cohort_name}/{age_band}")

        step4_logger, log_path = _get_logger(cohort_name, age_band)
        step4_logger.info(f"Processing cohort={cohort_name}, age_band={age_band}")

        filter_cohort_events_for_items(
            cohort_name=cohort_name,
            age_band=age_band,
            important_items=important_items,
            years=YEARS,
            output_root=model_data_root,
            local_cohort_root=local_cohort_root,
            local_medical_root=local_medical_root,
            local_pharmacy_root=local_pharmacy_root,
            sample_ratio=DEFAULT_SAMPLE_RATIO,
            control_exclusions=control_exclusions,
            logger=step4_logger,
        )

        if mirror_log_to_s3 and log_path.exists():
            try:
                mirror_log_to_s3(
                    feature_step="4_model_data",
                    cohort=cohort_name,
                    age_band=age_band,
                    log_path=log_path,
                    logger=step4_logger,
                )
            except Exception:
                pass  # best-effort; log remains local


if __name__ == "__main__":
    main()



