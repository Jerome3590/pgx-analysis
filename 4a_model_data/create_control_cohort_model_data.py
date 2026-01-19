#!/usr/bin/env python3
"""
Create model_events.parquet for control cohort (non_opioid_non_ed).

This script creates model_events.parquet for the control cohort used in BupaR analysis.
The control cohort consists of patients who:
- Have no opioid ICD codes (non-opioid)
- Have no ED visits (non-ED)
- Are in the same age band as the target cohort

This is a simplified version that only creates control events (target=0).
"""

import os
import sys
from pathlib import Path
from typing import List

import duckdb

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import (
    OPIOID_ICD_CODES,
    get_opioid_icd_sql_condition,
)
from py_helpers.env_utils import get_data_root, is_linux


def get_model_data_root() -> Path:
    """Get the root directory for model data output (OS-aware)."""
    data_root = get_data_root()
    if is_linux():
        return data_root / "4a_model_data"
    else:
        return PROJECT_ROOT / "4a_model_data"


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


def create_control_cohort_model_data(
    age_band: str,
    years: List[int] = [2016, 2017, 2018],
    sample_size: int = 10000,
    output_root: Path = None,
) -> None:
    """
    Create model_events.parquet for non_opioid_non_ed control cohort.
    
    Args:
        age_band: Age band (e.g., "13-24")
        years: List of years to include
        sample_size: Number of control patients to sample
        output_root: Root directory for output (default: get_model_data_root())
    """
    if output_root is None:
        output_root = get_model_data_root()
    
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
    
    # Create output directory
    out_dir = output_root / f"cohort_name={cohort_name}" / f"age_band={age_band}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_events.parquet"
    
    # Check if already exists
    if out_path.exists():
        print(f"[INFO] Control cohort model_events.parquet already exists: {out_path}")
        return
    
    con = duckdb.connect()
    
    # Get opioid ICD condition
    opioid_condition = get_opioid_icd_sql_condition("ue")
    
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
    patients_with_both AS (
        -- Only include patients who have events in BOTH medical AND pharmacy
        SELECT DISTINCT me.mi_person_key
        FROM medical_events me
        INNER JOIN pharmacy_events pe ON me.mi_person_key = pe.mi_person_key
    ),
    unified_events AS (
        SELECT
            me.*
        FROM medical_events me
        INNER JOIN patients_with_both pwb ON me.mi_person_key = pwb.mi_person_key
        UNION ALL
        SELECT
            pe.*
        FROM pharmacy_events pe
        INNER JOIN patients_with_both pwb ON pe.mi_person_key = pwb.mi_person_key
    ),
    per_patient_flags AS (
        SELECT
            mi_person_key,
            MAX(
                CASE
                    WHEN {opioid_condition} THEN 1
                    ELSE 0
                END
            ) AS has_opioid_icd,
            MAX(
                CASE
                    WHEN hcg_line IN ('P51 - ER Visits and Observation Care', 'O11 - Emergency Room', 'P33 - Urgent Care Visits') THEN 1
                    ELSE 0
                END
            ) AS has_ed_visit
        FROM unified_events ue
        GROUP BY mi_person_key
    ),
    control_candidates AS (
        SELECT mi_person_key
        FROM per_patient_flags
        WHERE has_opioid_icd = 0 AND has_ed_visit = 0
    ),
    sampled_controls AS (
        SELECT mi_person_key
        FROM control_candidates
        ORDER BY random()
        LIMIT {sample_size}
    )
    SELECT
        ue.*,
        0 AS target
    FROM unified_events ue
    INNER JOIN sampled_controls sc ON ue.mi_person_key = sc.mi_person_key
    """
    
    try:
        print(f"[INFO] Creating control cohort model_events.parquet for {cohort_name}/{age_band}...")
        print(f"[INFO] Sampling {sample_size} control patients (non-opioid, non-ED)")
        
        # Diagnostic queries to understand where data is being filtered
        print(f"\n[DEBUG] Running diagnostic queries...")
        
        # Check medical events count
        diag_medical = con.execute(f"SELECT COUNT(*) as n FROM read_parquet([{medical_paths_literal}])").fetchone()[0]
        print(f"[DEBUG] Medical events: {diag_medical:,}")
        
        # Check pharmacy events count
        diag_pharmacy = con.execute(f"SELECT COUNT(*) as n FROM read_parquet([{pharmacy_paths_literal}])").fetchone()[0]
        print(f"[DEBUG] Pharmacy events: {diag_pharmacy:,}")
        
        # Check patients with both
        diag_both_query = f"""
        WITH medical_events AS (
            SELECT DISTINCT mi_person_key
            FROM read_parquet([{medical_paths_literal}])
        ),
        pharmacy_events AS (
            SELECT DISTINCT mi_person_key
            FROM read_parquet([{pharmacy_paths_literal}])
        )
        SELECT COUNT(DISTINCT me.mi_person_key) as n
        FROM medical_events me
        INNER JOIN pharmacy_events pe ON me.mi_person_key = pe.mi_person_key
        """
        diag_both = con.execute(diag_both_query).fetchone()[0]
        print(f"[DEBUG] Patients with both medical AND pharmacy events: {diag_both:,}")
        
        # Check control candidates count
        diag_candidates_query = f"""
        {query.replace('LIMIT ' + str(sample_size), '')}
        """
        # Extract just the control_candidates CTE
        diag_candidates_simple = f"""
        WITH medical_events AS (
            SELECT mi_person_key, incurred_date, event_year, primary_icd_diagnosis_code, two_icd_diagnosis_code,
                   three_icd_diagnosis_code, four_icd_diagnosis_code, five_icd_diagnosis_code,
                   six_icd_diagnosis_code, seven_icd_diagnosis_code, eight_icd_diagnosis_code,
                   nine_icd_diagnosis_code, ten_icd_diagnosis_code, hcg_line
            FROM read_parquet([{medical_paths_literal}])
        ),
        pharmacy_events AS (
            SELECT mi_person_key, incurred_date, event_year
            FROM read_parquet([{pharmacy_paths_literal}])
        ),
        patients_with_both AS (
            SELECT DISTINCT me.mi_person_key
            FROM medical_events me
            INNER JOIN pharmacy_events pe ON me.mi_person_key = pe.mi_person_key
        ),
        unified_events AS (
            SELECT me.mi_person_key, me.primary_icd_diagnosis_code, me.two_icd_diagnosis_code,
                   me.three_icd_diagnosis_code, me.four_icd_diagnosis_code, me.five_icd_diagnosis_code,
                   me.six_icd_diagnosis_code, me.seven_icd_diagnosis_code, me.eight_icd_diagnosis_code,
                   me.nine_icd_diagnosis_code, me.ten_icd_diagnosis_code, me.hcg_line
            FROM medical_events me
            INNER JOIN patients_with_both pwb ON me.mi_person_key = pwb.mi_person_key
        ),
        per_patient_flags AS (
            SELECT
                mi_person_key,
                MAX(CASE WHEN {opioid_condition} THEN 1 ELSE 0 END) AS has_opioid_icd,
                MAX(CASE WHEN hcg_line IS NOT NULL THEN 1 ELSE 0 END) AS has_ed_visit
            FROM unified_events ue
            GROUP BY mi_person_key
        ),
        control_candidates AS (
            SELECT mi_person_key
            FROM per_patient_flags
            WHERE has_opioid_icd = 0 AND has_ed_visit = 0
        )
        SELECT COUNT(*) as n FROM control_candidates
        """
        try:
            diag_candidates = con.execute(diag_candidates_simple).fetchone()[0]
            print(f"[DEBUG] Control candidates (no opioid ICD, no ED visit): {diag_candidates:,}")
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
            print(f"[OK] Created control cohort model_events.parquet: {out_path}")
            print(f"[OK] File size: {file_size:,} bytes")
            print(f"[OK] Total events: {result_count:,}")
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
    args = parser.parse_args()
    
    create_control_cohort_model_data(
        age_band=args.age_band,
        years=args.years,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
