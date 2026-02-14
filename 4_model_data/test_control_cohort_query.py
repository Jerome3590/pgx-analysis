#!/usr/bin/env python3
"""
Test script to validate the control cohort creation query locally.
Downloads files from S3 if not found locally.
"""

import sys
import subprocess
import shutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
from py_helpers.env_utils import get_data_root
from py_helpers.constants import get_opioid_icd_sql_condition


def resolve_local_medical_root() -> Path:
    """Resolve the root directory containing gold medical parquet files."""
    import os
    env_path = os.getenv("LOCAL_MEDICAL_PATH")
    if env_path:
        root = Path(env_path)
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
            return path
    
    return candidates[2]


def resolve_local_pharmacy_root() -> Path:
    """Resolve the root directory containing gold pharmacy parquet files."""
    import os
    env_path = os.getenv("LOCAL_PHARMACY_PATH")
    if env_path:
        root = Path(env_path)
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
            return path
    
    return candidates[2]


def download_from_s3_if_needed(s3_path: str, local_path: Path) -> bool:
    """Download file from S3 if it doesn't exist locally."""
    if local_path.exists():
        print(f"[OK] File exists locally: {local_path}")
        return True
    
    print(f"[INFO] File not found locally: {local_path}")
    print(f"[INFO] Attempting to download from S3: {s3_path}")
    
    aws_cli = shutil.which("aws")
    if not aws_cli:
        print("[WARN] AWS CLI not found. Cannot download from S3.")
        return False
    
    # Create parent directory
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        result = subprocess.run(
            [aws_cli, "s3", "cp", s3_path, str(local_path)],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        
        if result.returncode == 0 and local_path.exists():
            print(f"[OK] Successfully downloaded from S3: {local_path}")
            return True
        else:
            print(f"[ERROR] Failed to download from S3: {result.stderr if result.stderr else 'Unknown error'}")
            return False
    except Exception as e:
        print(f"[ERROR] Error downloading from S3: {e}")
        return False


def test_query(age_band: str = "85-114", sample_size: int = 100):
    """Test the control cohort creation query with a small sample."""
    print(f"\n{'='*80}")
    print(f"Testing Control Cohort Query")
    print(f"{'='*80}")
    print(f"Age band: {age_band}")
    print(f"Sample size: {sample_size} (small test)")
    print()
    
    local_medical_root = resolve_local_medical_root()
    local_pharmacy_root = resolve_local_pharmacy_root()
    
    print(f"Medical root: {local_medical_root}")
    print(f"Pharmacy root: {local_pharmacy_root}")
    print()
    
    # Build paths to medical and pharmacy parquet files
    medical_parquet_paths = []
    pharmacy_parquet_paths = []
    years = [2016, 2017, 2018]
    
    for year in years:
        # Medical files
        medical_glob = local_medical_root / f"age_band={age_band}" / f"event_year={year}" / "*.parquet"
        medical_files = list(medical_glob.parent.glob(medical_glob.name))
        medical_parquet_paths.extend(medical_files)
        
        # Pharmacy files
        pharmacy_glob = local_pharmacy_root / f"age_band={age_band}" / f"event_year={year}" / "*.parquet"
        pharmacy_files = list(pharmacy_glob.parent.glob(pharmacy_glob.name))
        pharmacy_parquet_paths.extend(pharmacy_files)
    
    # Try downloading from S3 if files not found
    if not medical_parquet_paths:
        print(f"[WARN] No medical files found locally for age_band={age_band}")
        # Try downloading one file as a test
        test_s3_path = f"s3://pgxdatalake/gold/medical/age_band={age_band}/event_year=2016/"
        print(f"[INFO] Would need to download from S3: {test_s3_path}")
        print(f"[INFO] Skipping S3 download for now - please ensure files are available")
    
    if not pharmacy_parquet_paths:
        print(f"[WARN] No pharmacy files found locally for age_band={age_band}")
        # Try downloading one file as a test
        test_s3_path = f"s3://pgxdatalake/gold/pharmacy/age_band={age_band}/event_year=2016/"
        print(f"[INFO] Would need to download from S3: {test_s3_path}")
        print(f"[INFO] Skipping S3 download for now - please ensure files are available")
    
    if not medical_parquet_paths and not pharmacy_parquet_paths:
        print(f"[ERROR] No medical or pharmacy files found for age_band={age_band}")
        return False
    
    print(f"[INFO] Found {len(medical_parquet_paths)} medical files and {len(pharmacy_parquet_paths)} pharmacy files")
    
    # Limit to first few files for testing
    test_medical = medical_parquet_paths[:3] if medical_parquet_paths else []
    test_pharmacy = pharmacy_parquet_paths[:1] if pharmacy_parquet_paths else []
    
    print(f"[INFO] Testing with {len(test_medical)} medical files and {len(test_pharmacy)} pharmacy files")
    print()
    
    medical_paths_literal = ", ".join(f"'{p}'" for p in test_medical) if test_medical else ""
    pharmacy_paths_literal = ", ".join(f"'{p}'" for p in test_pharmacy) if test_pharmacy else ""
    
    if not medical_paths_literal or not pharmacy_paths_literal:
        print(f"[ERROR] Need both medical and pharmacy files for testing")
        return False
    
    con = duckdb.connect()
    
    # Get opioid ICD condition
    opioid_condition = get_opioid_icd_sql_condition("ue")
    
    # Test query (same structure as create_control_cohort_model_data.py)
    query = f"""
    WITH medical_events AS (
        SELECT
            mi_person_key,
            CAST(incurred_date AS DATE) AS event_date,
            event_year,
            NULL AS drug_name,
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
            CAST(incurred_date AS DATE) AS event_date,
            event_year,
            drug_name,
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
                    WHEN hcg_line IS NOT NULL THEN 1
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
    LIMIT 1000
    """
    
    try:
        print("[INFO] Testing query execution...")
        print()
        
        # Test each CTE step by step
        print("Step 1: Testing medical_events CTE...")
        test_query1 = f"""
        SELECT COUNT(*) as n_medical_events
        FROM read_parquet([{medical_paths_literal}])
        """
        result1 = con.execute(test_query1).fetchone()
        print(f"  Medical events in files: {result1[0]:,}")
        
        print("\nStep 2: Testing pharmacy_events CTE...")
        test_query2 = f"""
        SELECT COUNT(*) as n_pharmacy_events
        FROM read_parquet([{pharmacy_paths_literal}])
        """
        result2 = con.execute(test_query2).fetchone()
        print(f"  Pharmacy events in files: {result2[0]:,}")
        
        print("\nStep 3: Testing medical_events with date casting...")
        test_query3 = f"""
        SELECT 
            COUNT(*) as n_events,
            COUNT(DISTINCT mi_person_key) as n_patients,
            MIN(CAST(incurred_date AS DATE)) as min_date,
            MAX(CAST(incurred_date AS DATE)) as max_date
        FROM read_parquet([{medical_paths_literal}])
        """
        result3 = con.execute(test_query3).fetchone()
        print(f"  Events: {result3[0]:,}, Patients: {result3[1]:,}")
        print(f"  Date range: {result3[2]} to {result3[3]}")
        
        print("\nStep 4: Testing pharmacy_events with date casting...")
        test_query4 = f"""
        SELECT 
            COUNT(*) as n_events,
            COUNT(DISTINCT mi_person_key) as n_patients,
            MIN(CAST(incurred_date AS DATE)) as min_date,
            MAX(CAST(incurred_date AS DATE)) as max_date
        FROM read_parquet([{pharmacy_paths_literal}])
        """
        result4 = con.execute(test_query4).fetchone()
        print(f"  Events: {result4[0]:,}, Patients: {result4[1]:,}")
        print(f"  Date range: {result4[2]} to {result4[3]}")
        
        print("\nStep 5: Testing full query...")
        result = con.execute(query).fetchall()
        print(f"[OK] Query executed successfully!")
        print(f"  Returned {len(result):,} rows")
        
        if len(result) > 0:
            print(f"  Sample row columns: {len(result[0])} columns")
            print(f"  First row sample: {result[0][:5]}...")  # Show first 5 columns
        
        con.close()
        return True
        
    except Exception as e:
        print(f"[ERROR] Query failed: {e}")
        import traceback
        traceback.print_exc()
        con.close()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test control cohort creation query")
    parser.add_argument("--age-band", type=str, default="85-114", help="Age band to test (default: 85-114)")
    parser.add_argument("--sample-size", type=int, default=100, help="Sample size for testing (default: 100)")
    
    args = parser.parse_args()
    
    success = test_query(age_band=args.age_band, sample_size=args.sample_size)
    
    if success:
        print(f"\n{'='*80}")
        print("[OK] Query test passed!")
        print(f"{'='*80}")
        sys.exit(0)
    else:
        print(f"\n{'='*80}")
        print("[ERROR] Query test failed!")
        print(f"{'='*80}")
        sys.exit(1)
