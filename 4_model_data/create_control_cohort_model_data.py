#!/usr/bin/env python3
"""
Create model_events.parquet for control cohort (non_opioid_non_ed).

This script creates model_events.parquet for the control cohort used in BupaR analysis.
For POLYPHARMACY COHORT (cohort_name="non_opioid_ed" in data partitions), 
the control cohort consists of patients who:
- Have drug events (pharmacy events)
- Have NO first ED visit (HCG Setting) within the time window of a drug event (no qualifying HCG line ED visits)
- Have no opioid ICD codes (non-opioid)
- Are in the same age band as the target cohort

Qualifying ED = first ED (HCG Setting) within 21 days of drug; hcg_line IN (same as 2_create_cohort):
  - 'P51 - ER Visits and Observation Care'
  - 'O11 - Emergency Room'
  - 'P33 - Urgent Care Visits'

This is a simplified version that only creates control events (target=0).
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

import duckdb

# Add project root and 4_model_data to path (for get_important_items from create_model_data)
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DATA_DIR = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MODEL_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DATA_DIR))

from py_helpers.constants import (
    ALL_ICD_DIAGNOSIS_COLUMNS,
    get_cohort_slug,
    get_opioid_icd_sql_condition,
)
from py_helpers.env_utils import get_data_root, get_model_data_root
from py_helpers.feature_importance_eda_utils import load_administrative_codes

# get_important_items from create_model_data (same as target cohort filter)
from create_model_data import get_important_items


def resolve_local_medical_root() -> Path:
    """
    Resolve the root directory containing gold medical parquet files.
    
    Priority:
      1. LOCAL_MEDICAL_PATH environment variable
      2. get_data_root()/gold/medical (Linux/EC2: /mnt/nvme/gold/medical)
      3. get_data_root()/data/gold_medical (Alternative Linux path)
      4. PROJECT_ROOT/data/gold_medical (Windows/local dev)
    """
    env_path = os.getenv("LOCAL_MEDICAL_PATH")
    if env_path:
        root = Path(env_path)
        if root.exists():
            return root
    
    # OS-aware path resolution
    data_root = get_data_root()
    candidates = [
        data_root / "gold" / "medical",  # Linux/EC2: /mnt/nvme/gold/medical
        data_root / "data" / "gold_medical",  # Alternative Linux path
        PROJECT_ROOT / "data" / "gold_medical",  # Windows/local dev
    ]
    
    # Return first existing path, or default to project root if none exists
    for path in candidates:
        if path.exists():
            return path
    
    return candidates[2]  # Default to project root


def resolve_local_pharmacy_root() -> Path:
    """
    Resolve the root directory containing gold pharmacy parquet files.
    
    Priority:
      1. LOCAL_PHARMACY_PATH environment variable
      2. get_data_root()/gold/pharmacy (Linux/EC2: /mnt/nvme/gold/pharmacy)
      3. get_data_root()/data/gold_pharmacy (Alternative Linux path)
      4. PROJECT_ROOT/data/gold_pharmacy (Windows/local dev)
    """
    env_path = os.getenv("LOCAL_PHARMACY_PATH")
    if env_path:
        root = Path(env_path)
        if root.exists():
            return root
    
    # OS-aware path resolution
    data_root = get_data_root()
    candidates = [
        data_root / "gold" / "pharmacy",  # Linux/EC2: /mnt/nvme/gold/pharmacy
        data_root / "data" / "gold_pharmacy",  # Alternative Linux path
        PROJECT_ROOT / "data" / "gold_pharmacy",  # Windows/local dev
    ]
    
    # Return first existing path, or default to project root if none exists
    for path in candidates:
        if path.exists():
            return path
    
    return candidates[2]  # Default to project root


def _item_filter_condition_sql(important_items: List[str]) -> str:
    """Build SQL WHERE condition for event-level filter (drug_name, ICD cols, procedure_code). Same logic as create_model_data.filter_cohort_events_for_items."""
    if not important_items:
        return "TRUE"
    item_list_literal = ", ".join(f"'{v}'" for v in important_items)
    icd_conditions = " OR ".join(
        f"{col} IN ({item_list_literal})" for col in ALL_ICD_DIAGNOSIS_COLUMNS
    )
    if not icd_conditions:
        icd_conditions = "FALSE"
    return f"""(
        drug_name IN ({item_list_literal}) OR
        {icd_conditions} OR
        procedure_code IN ({item_list_literal})
    )"""


def create_control_cohort_model_data(
    age_band: str,
    years: List[int] = [2016, 2017, 2018],
    sample_size: int = 10000,
    output_root: Path = None,
    target_cohort_path: Path = None,
    aggregated_fi_csv: Optional[Path] = None,
    time_window_days: int = 14,
) -> None:
    """
    Create model_events.parquet for non_opioid_non_ed control cohort.
    
    For POLYPHARMACY COHORT, controls must have drug events but NO HCG target events
    within the specified time window of their drug events.
    
    Optionally filter control events to the same feature set as target (3a aggregated FI
    minus admin codes) to reduce noise in BupaR analysis.
    
    Args:
        age_band: Age band (e.g., "13-24")
        years: List of years to include
        sample_size: Number of control patients to sample
        output_root: Root directory for output (default: get_model_data_root())
        target_cohort_path: Optional path to target cohort model_events.parquet for ratio logging
        aggregated_fi_csv: Path to 3a aggregated feature importance CSV; control events are filtered
            to the same items (with admin codes removed) as target. Required when output_root is set (Step 3b).
        time_window_days: Time window in days for checking HCG target events near drug events (default: 30)
                          Common values: 30, 60, 90, 120
    """
    if output_root is None:
        output_root = get_model_data_root()
    
    important_items: List[str] = []
    if aggregated_fi_csv and aggregated_fi_csv.exists():
        important_items = get_important_items(aggregated_fi_csv)
        admin_codes = load_administrative_codes(PROJECT_ROOT)
        if admin_codes:
            n_before = len(important_items)
            important_items = [x for x in important_items if x not in admin_codes]
            if n_before > len(important_items):
                print(f"[INFO] Filtering control events by 3a FI (admin removed): {len(important_items)} items")
        if not important_items:
            important_items = []
    item_filter_sql = _item_filter_condition_sql(important_items)
    
    local_medical_root = resolve_local_medical_root()
    local_pharmacy_root = resolve_local_pharmacy_root()
    
    cohort_name = "non_opioid_non_ed"
    
    # Build paths to medical and pharmacy parquet files
    medical_parquet_paths = []
    pharmacy_parquet_paths = []
    
    for year in years:
        # Medical files
        medical_glob = local_medical_root / f"age_band={age_band}" / f"event_year={year}" / "*.parquet"
        medical_files = list(medical_glob.parent.glob(medical_glob.name))
        medical_parquet_paths.extend(medical_files)
        
        # Pharmacy files
        pharmacy_glob = local_pharmacy_root / f"age_band={age_band}" / f"event_year={year}" / "*.parquet"
        pharmacy_files = list(pharmacy_glob.parent.glob(pharmacy_glob.name))
        pharmacy_parquet_paths.extend(pharmacy_files)
    
    if not medical_parquet_paths and not pharmacy_parquet_paths:
        print(f"[ERROR] No medical or pharmacy files found for age_band={age_band}")
        print(f"  Medical root: {local_medical_root}")
        print(f"  Pharmacy root: {local_pharmacy_root}")
        return
    
    print(f"[INFO] Found {len(medical_parquet_paths)} medical files and {len(pharmacy_parquet_paths)} pharmacy files")
    
    # Control cohort uses fixed slug "non_opioid_non_ed" (matches R/control_cohort_utils and S3)
    control_slug = "non_opioid_non_ed"
    
    # New structure: cohorts/input_model_data/cohort_name=non_opioid_non_ed/age_band={age_band}/
    out_dir = (
        output_root 
        / "cohorts" 
        / "input_model_data"
        / f"cohort_name={control_slug}" 
        / f"age_band={age_band}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_events.parquet"
    
    # Check if already exists
    if out_path.exists():
        print(f"[INFO] Control cohort model_events.parquet already exists: {out_path}")
        return
    
    con = duckdb.connect()
    
    # Get opioid ICD condition
    opioid_condition = get_opioid_icd_sql_condition("me")
    
    # Build query to:
    # 1. Load all medical and pharmacy events
    # 2. Identify patients with opioid ICD codes
    # 3. Identify patients with ED visits (hcg_line is not null for ED visits)
    # 4. Select control patients (no opioids, no ED)
    # 5. Sample control patients
    # 6. Extract all events for sampled controls
    
    medical_paths_literal = ", ".join(f"'{p}'" for p in medical_parquet_paths) if medical_parquet_paths else ""
    pharmacy_paths_literal = ", ".join(f"'{p}'" for p in pharmacy_parquet_paths) if pharmacy_parquet_paths else ""
    
    if not medical_paths_literal or not pharmacy_paths_literal:
        print(f"[ERROR] Both medical and pharmacy files are required")
        return
    
    query = f"""
    WITH     medical_events AS (
        SELECT
            mi_person_key,
            CASE 
                WHEN LENGTH(CAST(incurred_date AS VARCHAR)) = 8 THEN 
                    CAST(SUBSTR(CAST(incurred_date AS VARCHAR), 1, 4) || '-' || 
                         SUBSTR(CAST(incurred_date AS VARCHAR), 5, 2) || '-' || 
                         SUBSTR(CAST(incurred_date AS VARCHAR), 7, 2) AS DATE)
                ELSE CAST(incurred_date AS DATE)
            END AS event_date,  -- Parse YYYYMMDD format to YYYY-MM-DD
            event_year,
            NULL AS drug_name,  -- Medical files don't have drug_name
            primary_icd_diagnosis_code,
            two_icd_diagnosis_code,
            three_icd_diagnosis_code,
            four_icd_diagnosis_code,
            five_icd_diagnosis_code,
            six_icd_diagnosis_code,
            seven_icd_diagnosis_code,
            eight_icd_diagnosis_code,
            nine_icd_diagnosis_code,
            ten_icd_diagnosis_code,
            procedure_code,
            hcg_line,
            age_band
        FROM read_parquet([{medical_paths_literal}])
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
            END AS event_date,  -- Parse YYYYMMDD format to YYYY-MM-DD
            event_year,
            drug_name,  -- Pharmacy files have drug_name
            NULL AS primary_icd_diagnosis_code,
            NULL AS two_icd_diagnosis_code,
            NULL AS three_icd_diagnosis_code,
            NULL AS four_icd_diagnosis_code,
            NULL AS five_icd_diagnosis_code,
            NULL AS six_icd_diagnosis_code,
            NULL AS seven_icd_diagnosis_code,
            NULL AS eight_icd_diagnosis_code,
            NULL AS nine_icd_diagnosis_code,
            NULL AS ten_icd_diagnosis_code,
            NULL AS procedure_code,
            NULL AS hcg_line,
            age_band
        FROM read_parquet([{pharmacy_paths_literal}])
    ),
    patients_with_drug_events AS (
        -- POLYPHARMACY COHORT: Controls must have drug events (pharmacy events)
        -- This ensures controls have drug events, matching the polypharmacy cohort's focus on drug sequences
        SELECT DISTINCT mi_person_key
        FROM pharmacy_events
    ),
    patient_hcg_dates AS (
        -- Get HCG target event dates for each patient
        SELECT
            me.mi_person_key,
            me.event_date AS hcg_event_date
        FROM medical_events me
        WHERE me.hcg_line IN ('P51 - ER Visits and Observation Care', 'O11 - Emergency Room', 'P33 - Urgent Care Visits')
    ),
    drug_hcg_pairs AS (
        -- For each patient, check if ANY HCG target event occurs within {time_window_days} days of ANY drug event
        -- This creates pairs of (drug_event_date, hcg_event_date) where they're within the time window
        SELECT DISTINCT
            pe.mi_person_key,
            pe.event_date AS drug_event_date,
            phd.hcg_event_date
        FROM pharmacy_events pe
        INNER JOIN patients_with_drug_events pde ON pe.mi_person_key = pde.mi_person_key
        LEFT JOIN patient_hcg_dates phd ON pe.mi_person_key = phd.mi_person_key
            AND phd.hcg_event_date >= pe.event_date
            AND phd.hcg_event_date <= DATE_ADD(pe.event_date, INTERVAL {time_window_days} DAY)
        WHERE phd.hcg_event_date IS NOT NULL  -- Only keep pairs where HCG event is within window
    ),
    patients_with_hcg_in_window AS (
        -- Get distinct patients who have HCG target events within time window of drug events
        SELECT DISTINCT mi_person_key
        FROM drug_hcg_pairs
    ),
    per_patient_flags AS (
        -- Check ALL patients with drug events for opioid ICD codes and time-windowed HCG target events
        -- Use LEFT JOIN to check even if patient only has pharmacy events
        -- Time window: Check if HCG target event occurs within {time_window_days} days of ANY drug event
        SELECT
            pde.mi_person_key,
            COALESCE(MAX(
                CASE
                    WHEN {opioid_condition} THEN 1
                    ELSE 0
                END
            ), 0) AS has_opioid_icd,  -- Default to 0 if no medical events
            CASE
                WHEN phw.mi_person_key IS NOT NULL THEN 1
                ELSE 0
            END AS has_hcg_target_event_in_window  -- 1 if patient has HCG target event within window of any drug event
        FROM patients_with_drug_events pde
        LEFT JOIN medical_events me ON pde.mi_person_key = me.mi_person_key
        LEFT JOIN patients_with_hcg_in_window phw ON pde.mi_person_key = phw.mi_person_key
        GROUP BY pde.mi_person_key, phw.mi_person_key
    ),
    control_candidates AS (
        -- POLYPHARMACY COHORT: Controls must have drug events AND no time-windowed HCG target events
        -- Drug event requirement already enforced by patients_with_drug_events filter above
        -- has_hcg_target_event_in_window = 0 means patient has NO HCG target events within {time_window_days} days of drug events
        SELECT mi_person_key
        FROM per_patient_flags
        WHERE has_opioid_icd = 0 AND has_hcg_target_event_in_window = 0
    ),
    sampled_controls AS (
        SELECT mi_person_key
        FROM control_candidates
        ORDER BY random()
        LIMIT {sample_size}
    ),
    final_unified_events AS (
        -- Get ALL events (medical + pharmacy) for sampled controls
        -- Medical events: Include all medical events for sampled controls (if they have any)
        SELECT
            me.*
        FROM sampled_controls sc
        INNER JOIN medical_events me ON sc.mi_person_key = me.mi_person_key
        UNION ALL
        -- Pharmacy events: Include all pharmacy events for sampled controls
        SELECT
            pe.*
        FROM sampled_controls sc
        INNER JOIN pharmacy_events pe ON sc.mi_person_key = pe.mi_person_key
    )
    SELECT
        fue.*,
        0 AS target
    FROM final_unified_events fue
    WHERE {item_filter_sql}
    """
    
    try:
        # Get cohort slug based on age band
        cohort_slug = get_cohort_slug(age_band)
        print(f"[INFO] Creating control cohort model_events.parquet for {cohort_slug}/{age_band}...")
        print(f"[INFO] POLYPHARMACY COHORT control definition (time window: {time_window_days} days):")
        print(f"[INFO]   - Patients with drug events (pharmacy events)")
        print(f"[INFO]   - NO time-windowed HCG target events within {time_window_days} days of drug events")
        print(f"[INFO]   - NO opioid ICD codes")
        print(f"[INFO] Sampling {sample_size} control patients")
        
        # Diagnostic queries to understand where data is being filtered
        print(f"\n[DEBUG] Running diagnostic queries...")
        
        # Check medical events count
        diag_medical = con.execute(f"SELECT COUNT(*) as n FROM read_parquet([{medical_paths_literal}])").fetchone()[0]
        print(f"[DEBUG] Medical events: {diag_medical:,}")
        
        # Check pharmacy events count
        diag_pharmacy = con.execute(f"SELECT COUNT(*) as n FROM read_parquet([{pharmacy_paths_literal}])").fetchone()[0]
        print(f"[DEBUG] Pharmacy events: {diag_pharmacy:,}")
        
        # Check patients with drug events (pharmacy events)
        diag_drug_query = f"""
        SELECT COUNT(DISTINCT mi_person_key) as n
        FROM read_parquet([{pharmacy_paths_literal}])
        """
        diag_drug = con.execute(diag_drug_query).fetchone()[0]
        print(f"[DEBUG] Patients with drug events (pharmacy): {diag_drug:,}")
        
        # Check control candidates count - use same structure as main query (parquet has incurred_date, derive event_date)
        _event_date_sql = """CASE 
                WHEN LENGTH(CAST(incurred_date AS VARCHAR)) = 8 THEN 
                    CAST(SUBSTR(CAST(incurred_date AS VARCHAR), 1, 4) || '-' || SUBSTR(CAST(incurred_date AS VARCHAR), 5, 2) || '-' || SUBSTR(CAST(incurred_date AS VARCHAR), 7, 2) AS DATE)
                ELSE CAST(incurred_date AS DATE)
            END"""
        diag_candidates_simple = f"""
        WITH medical_events AS (
            SELECT mi_person_key, {_event_date_sql} AS event_date, primary_icd_diagnosis_code, two_icd_diagnosis_code,
                   three_icd_diagnosis_code, four_icd_diagnosis_code, five_icd_diagnosis_code,
                   six_icd_diagnosis_code, seven_icd_diagnosis_code, eight_icd_diagnosis_code,
                   nine_icd_diagnosis_code, ten_icd_diagnosis_code, hcg_line
            FROM read_parquet([{medical_paths_literal}])
        ),
        pharmacy_events AS (
            SELECT mi_person_key, {_event_date_sql} AS event_date
            FROM read_parquet([{pharmacy_paths_literal}])
        ),
        patients_with_drug_events AS (
            -- POLYPHARMACY COHORT: Controls must have drug events (pharmacy events)
            SELECT DISTINCT mi_person_key
            FROM pharmacy_events
        ),
        patient_hcg_dates AS (
            SELECT
                me.mi_person_key,
                me.event_date AS hcg_event_date
            FROM medical_events me
            WHERE me.hcg_line IN ('P51 - ER Visits and Observation Care', 'O11 - Emergency Room', 'P33 - Urgent Care Visits')
        ),
        drug_hcg_pairs AS (
            -- For each patient, check if ANY HCG target event occurs within {time_window_days} days of ANY drug event
            SELECT DISTINCT
                pe.mi_person_key,
                pe.event_date AS drug_event_date,
                phd.hcg_event_date
            FROM pharmacy_events pe
            INNER JOIN patients_with_drug_events pde ON pe.mi_person_key = pde.mi_person_key
            LEFT JOIN patient_hcg_dates phd ON pe.mi_person_key = phd.mi_person_key
                AND phd.hcg_event_date >= pe.event_date
                AND phd.hcg_event_date <= DATE_ADD(pe.event_date, INTERVAL {time_window_days} DAY)
            WHERE phd.hcg_event_date IS NOT NULL
        ),
        per_patient_flags AS (
            -- Check ALL patients with drug events for opioid ICD codes and time-windowed HCG target events
            SELECT
                pde.mi_person_key,
                COALESCE(MAX(CASE WHEN {opioid_condition} THEN 1 ELSE 0 END), 0) AS has_opioid_icd,
                CASE
                    WHEN EXISTS (
                        SELECT 1 FROM drug_hcg_pairs dhp 
                        WHERE dhp.mi_person_key = pde.mi_person_key
                    ) THEN 1
                    ELSE 0
                END AS has_hcg_target_event_in_window
            FROM patients_with_drug_events pde
            LEFT JOIN medical_events me ON pde.mi_person_key = me.mi_person_key
            GROUP BY pde.mi_person_key
        ),
        control_candidates AS (
            -- POLYPHARMACY COHORT: Controls must have drug events AND no time-windowed HCG target events
            SELECT mi_person_key
            FROM per_patient_flags
            WHERE has_opioid_icd = 0 AND has_hcg_target_event_in_window = 0
        )
        SELECT COUNT(*) as n FROM control_candidates
        """
        try:
            diag_candidates = con.execute(diag_candidates_simple).fetchone()[0]
            print(f"[DEBUG] Control candidates (have drug events, no opioid ICD, no HCG target events): {diag_candidates:,}")
            
            # Check if we're trying to sample more than available
            if sample_size > diag_candidates:
                print(f"[WARN] Requested sample size ({sample_size:,}) exceeds available candidates ({diag_candidates:,})")
                print(f"[WARN] Will sample all available candidates ({diag_candidates:,})")
                # Note: SQL LIMIT will automatically cap at available rows, but we log this for visibility
        except Exception as e:
            print(f"[DEBUG] Could not count control candidates: {e}")
        
        print(f"[DEBUG] Diagnostic queries complete.\n")
        
        con.execute(f"COPY ({query}) TO '{out_path}' (FORMAT PARQUET)")
        
        # Validate the created file
        if not out_path.exists():
            raise FileNotFoundError(f"Parquet file was not created: {out_path}")
        
        file_size = out_path.stat().st_size
        if file_size < 1000:  # Parquet files should be at least 1KB
            raise ValueError(f"Created parquet file is too small ({file_size} bytes), likely empty or corrupted")
        
        # Check result by reading the file
        try:
            result_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
            if result_count == 0:
                raise ValueError(f"Created parquet file contains 0 rows")
            
            # Count distinct patients to verify no duplicates
            distinct_controls = con.execute(f"SELECT COUNT(DISTINCT mi_person_key) FROM read_parquet('{out_path}')").fetchone()[0]
            print(f"[OK] Created control cohort model_events.parquet: {out_path}")
            print(f"[OK] File size: {file_size:,} bytes")
            print(f"[OK] Total events: {result_count:,}")
            print(f"[OK] Distinct controls: {distinct_controls:,}")
            
            # Log ratio if target cohort path is provided
            if target_cohort_path and target_cohort_path.exists():
                years_list = ','.join(map(str, years))
                try:
                    distinct_targets = con.execute(
                        f"SELECT COUNT(DISTINCT mi_person_key) FROM read_parquet('{target_cohort_path}') "
                        f"WHERE event_year IN ({years_list}) AND target = 1"
                    ).fetchone()[0]
                    if distinct_targets > 0:
                        actual_ratio = distinct_controls / distinct_targets
                        print(f"[OK] Distinct targets: {distinct_targets:,}")
                        print(f"[OK] Actual ratio: {actual_ratio:.2f}:1 (controls:targets)")
                    else:
                        print(f"[WARN] No distinct targets found in target cohort")
                except Exception as e:
                    print(f"[WARN] Could not calculate ratio: {e}")
            
            # Warn if we got fewer patients than requested (due to limited candidates)
            if diag_candidates is not None and distinct_controls < sample_size:
                print(f"[WARN] Sampled {distinct_controls:,} patients (requested {sample_size:,})")
                print(f"[WARN] Limited by available control candidates ({diag_candidates:,})")
                print(f"[WARN] This is expected when target cohort is large relative to available controls")
        except Exception as validation_error:
            # If validation fails, remove the corrupted file
            if out_path.exists():
                out_path.unlink()
            raise ValueError(f"Created parquet file is invalid: {validation_error}") from validation_error
        
    except Exception as e:
        print(f"[ERROR] Failed to create control cohort model_events.parquet: {e}")
        # Remove any partially created file
        if out_path.exists():
            try:
                out_path.unlink()
                print(f"[INFO] Removed partially created file: {out_path}")
            except:
                pass
        import traceback
        traceback.print_exc()
        raise  # Re-raise to signal failure
    finally:
        con.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create model_events.parquet for non_opioid_non_ed control cohort"
    )
    parser.add_argument(
        "--age-band",
        type=str,
        required=True,
        help="Age band (e.g., 13-24)",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2016, 2017, 2018],
        help="Years to include (default: 2016 2017 2018)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of control patients to sample (default: 10000)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root directory for output (default: 4_model_data). Use 3b_feature_importance_eda/outputs for Step 3b so control is not written to 4_model_data.",
    )
    parser.add_argument(
        "--aggregated-fi-csv",
        type=str,
        default=None,
        help="Path to 3a aggregated feature importance CSV; control events are filtered to same items (admin removed). Required when --output-root is set (Step 3b).",
    )
    args = parser.parse_args()
    
    output_root = Path(args.output_root) if args.output_root else None
    aggregated_fi_csv = Path(args.aggregated_fi_csv) if args.aggregated_fi_csv else None
    if output_root is not None and (aggregated_fi_csv is None or not aggregated_fi_csv.exists()):
        raise SystemExit("When --output-root is set (Step 3b), --aggregated-fi-csv is required and must point to an existing 3a aggregated feature importance CSV.")
    create_control_cohort_model_data(
        age_band=args.age_band,
        years=args.years,
        sample_size=args.sample_size,
        output_root=output_root,
        aggregated_fi_csv=aggregated_fi_csv,
    )


if __name__ == "__main__":
    main()
