"""
Check relationship between HCG ED visit dates and drug event dates.
Samples patients with HCG ED visits and shows their drug events to verify date gap calculations.
"""

import sys
import os
# Add project root to path so we can import py_helpers
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import duckdb
import pandas as pd
from py_helpers.constants import S3_BUCKET
from py_helpers.duckdb_utils import create_duckdb_conn

# Sample age_band and event_year
age_band = "65-74"
event_year = 2019

# HCG condition for precise ED visit identification
hcg_condition = """
    (hcg_line = 'P51 - ER Visits and Observation Care' AND hcg_detail = 'P51b - PHY ED Visits and Observation Care - ED Visits')
    OR hcg_line = 'O11 - Emergency Room'
    OR hcg_line = 'P33 - Urgent Care Visits'
"""

print("=" * 100)
print("HCG ED Visit - Drug Event Date Relationship Check")
print("=" * 100)
print(f"\nHCG Condition (precise ED visits):")
print("  - P51b (ED Visits only, excludes P51a Observation Care)")
print("  - O11 (Emergency Room)")
print("  - P33 (Urgent Care Visits)")

# Create DuckDB connection
conn = create_duckdb_conn()

# Load AWS credentials
try:
    conn.sql("INSTALL httpfs; LOAD httpfs;")
    conn.sql("CALL load_aws_credentials('pgx');")
    print("\n✓ AWS credentials loaded")
except Exception as e:
    print(f"\n⚠ Warning: Could not load AWS credentials: {e}")

# Try to resolve paths
medical_path = None
pharmacy_path = None

local_medical_dir = f"/mnt/nvme/gold/medical/age_band={age_band}/event_year={event_year}"
local_pharmacy_dir = f"/mnt/nvme/gold/pharmacy/age_band={age_band}/event_year={event_year}"

if os.path.exists(local_medical_dir):
    import glob
    parquet_files = glob.glob(f"{local_medical_dir}/*.parquet")
    if parquet_files:
        medical_path = parquet_files[0]
        print(f"\n✓ Using local medical path: {medical_path}")
    else:
        medical_path = f"{local_medical_dir}/*.parquet"
else:
    medical_path = f"s3://{S3_BUCKET}/gold/medical/age_band={age_band}/event_year={event_year}/*.parquet"
    print(f"\n→ Using S3 medical path: {medical_path}")

if os.path.exists(local_pharmacy_dir):
    import glob
    parquet_files = glob.glob(f"{local_pharmacy_dir}/*.parquet")
    if parquet_files:
        pharmacy_path = parquet_files[0]
        print(f"✓ Using local pharmacy path: {pharmacy_path}")
    else:
        pharmacy_path = f"{local_pharmacy_dir}/*.parquet"
else:
    pharmacy_path = f"s3://{S3_BUCKET}/gold/pharmacy/age_band={age_band}/event_year={event_year}/*.parquet"
    print(f"→ Using S3 pharmacy path: {pharmacy_path}")

try:
    # First, check the schema and sample a row to see actual column names
    print("\n" + "=" * 100)
    print("Schema Check: Medical parquet columns")
    print("=" * 100)
    try:
        medical_sample = conn.sql(f"SELECT * FROM read_parquet('{medical_path}') LIMIT 1").fetchdf()
        print("\nMedical parquet columns (from sample row):")
        print(list(medical_sample.columns))
        # Check if incurred_date exists (the actual column in gold parquet files)
        if 'incurred_date' not in medical_sample.columns:
            print("\n⚠ 'incurred_date' not found. Looking for date columns...")
            date_cols = [col for col in medical_sample.columns if 'date' in col.lower()]
            print(f"  Found date-related columns: {date_cols}")
        else:
            print("✓ Found 'incurred_date' column in medical parquet")
    except Exception as e:
        print(f"Could not get medical sample: {e}")
    
    print("\n" + "=" * 100)
    print("Schema Check: Pharmacy parquet columns")
    print("=" * 100)
    try:
        pharmacy_sample = conn.sql(f"SELECT * FROM read_parquet('{pharmacy_path}') LIMIT 1").fetchdf()
        print("\nPharmacy parquet columns (from sample row):")
        print(list(pharmacy_sample.columns))
        # Check if incurred_date exists
        if 'incurred_date' not in pharmacy_sample.columns:
            print("\n⚠ 'incurred_date' not found. Looking for date columns...")
            date_cols = [col for col in pharmacy_sample.columns if 'date' in col.lower()]
            print(f"  Found date-related columns: {date_cols}")
    except Exception as e:
        print(f"Could not get pharmacy sample: {e}")
    
    # Query 1: Get patients with HCG ED visits and their drug events (excluding 0-day gaps - discharge prescriptions)
    print("\n" + "=" * 100)
    print("Query 1: Patients with HCG ED visits and their drug events (sample)")
    print("=" * 100)
    print("Note: Excluding 0-day gaps (likely discharge prescriptions) to focus on adverse drug event patterns")
    
    query1 = f"""
    WITH hcg_ed_events AS (
        SELECT DISTINCT
            mi_person_key,
            TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') as ed_date,
            hcg_line,
            hcg_detail,
            primary_icd_diagnosis_code
        FROM read_parquet('{medical_path}')
        WHERE {hcg_condition}
          AND incurred_date IS NOT NULL
          AND TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') IS NOT NULL
    ),
    drug_events AS (
        SELECT DISTINCT
            mi_person_key,
            TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') as drug_date,
            drug_name
        FROM read_parquet('{pharmacy_path}')
        WHERE drug_name IS NOT NULL
          AND drug_name <> ''
          AND incurred_date IS NOT NULL
          AND TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') IS NOT NULL
    ),
    patient_ed_drugs AS (
        SELECT
            hcg.mi_person_key,
            hcg.ed_date,
            hcg.hcg_line,
            hcg.hcg_detail,
            hcg.primary_icd_diagnosis_code,
            de.drug_date,
            de.drug_name,
            CAST(datediff('day', CAST(de.drug_date AS DATE), CAST(hcg.ed_date AS DATE)) AS BIGINT) as days_from_drug_to_ed
        FROM hcg_ed_events hcg
        INNER JOIN drug_events de ON hcg.mi_person_key = de.mi_person_key
        WHERE de.drug_date < hcg.ed_date
          AND CAST(datediff('day', CAST(de.drug_date AS DATE), CAST(hcg.ed_date AS DATE)) AS BIGINT) <= 45
          AND CAST(datediff('day', CAST(de.drug_date AS DATE), CAST(hcg.ed_date AS DATE)) AS BIGINT) > 0
    ),
    patient_summary AS (
        SELECT
            mi_person_key,
            MIN(ed_date) as first_ed_date,
            COUNT(DISTINCT ed_date) as ed_visit_count,
            COUNT(DISTINCT drug_date) as drug_event_count,
            MIN(days_from_drug_to_ed) as min_days_gap,
            MAX(days_from_drug_to_ed) as max_days_gap,
            AVG(days_from_drug_to_ed) as avg_days_gap
        FROM patient_ed_drugs
        GROUP BY mi_person_key
    )
    SELECT
        ps.mi_person_key,
        ps.first_ed_date,
        ps.ed_visit_count,
        ps.drug_event_count,
        ps.min_days_gap,
        ps.max_days_gap,
        CAST(ps.avg_days_gap AS DOUBLE) as avg_days_gap
    FROM patient_summary ps
    ORDER BY ps.first_ed_date
    LIMIT 20
    """
    
    print("Querying patient HCG ED visits and drug events...")
    result1 = conn.sql(query1).fetchdf()
    
    if result1.empty:
        print("\n⚠ No patients found with HCG ED visits and drug events")
    else:
        print(f"\nFound {len(result1)} patients (showing first 20):\n")
        print(result1.to_string(index=False))
    
    # Query 2: Detailed view - show actual date pairs for sample patients
    print("\n" + "=" * 100)
    print("Query 2: Detailed date pairs for sample patients")
    print("=" * 100)
    
    if not result1.empty:
        # Get first 5 patient keys
        sample_patients = result1['mi_person_key'].head(5).tolist()
        patients_tuple = tuple(sample_patients)
        
        query2 = f"""
        WITH hcg_ed_events AS (
            SELECT DISTINCT
                mi_person_key,
                TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') as ed_date,
                hcg_line,
                hcg_detail,
                primary_icd_diagnosis_code
            FROM read_parquet('{medical_path}')
            WHERE {hcg_condition}
              AND incurred_date IS NOT NULL
              AND TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') IS NOT NULL
              AND mi_person_key IN {patients_tuple}
        ),
        drug_events AS (
            SELECT DISTINCT
                mi_person_key,
                TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') as drug_date,
                drug_name
            FROM read_parquet('{pharmacy_path}')
            WHERE drug_name IS NOT NULL
              AND drug_name <> ''
              AND incurred_date IS NOT NULL
              AND TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') IS NOT NULL
              AND mi_person_key IN {patients_tuple}
        ),
        patient_ed_drugs AS (
            SELECT
                hcg.mi_person_key,
                hcg.ed_date,
                hcg.hcg_line,
                hcg.hcg_detail,
                hcg.primary_icd_diagnosis_code,
                de.drug_date,
                de.drug_name,
                CAST(datediff('day', CAST(de.drug_date AS DATE), CAST(hcg.ed_date AS DATE)) AS BIGINT) as days_from_drug_to_ed
            FROM hcg_ed_events hcg
            INNER JOIN drug_events de ON hcg.mi_person_key = de.mi_person_key
            WHERE de.drug_date < hcg.ed_date
              AND CAST(datediff('day', CAST(de.drug_date AS DATE), CAST(hcg.ed_date AS DATE)) AS BIGINT) <= 45
              AND CAST(datediff('day', CAST(de.drug_date AS DATE), CAST(hcg.ed_date AS DATE)) AS BIGINT) > 0
        )
        SELECT
            mi_person_key,
            CAST(ed_date AS VARCHAR) as ed_date_str,
            hcg_line,
            hcg_detail,
            primary_icd_diagnosis_code,
            CAST(drug_date AS VARCHAR) as drug_date_str,
            drug_name,
            days_from_drug_to_ed,
            TYPEOF(ed_date) as ed_date_type,
            TYPEOF(drug_date) as drug_date_type
        FROM patient_ed_drugs
        ORDER BY mi_person_key, ed_date, drug_date
        LIMIT 50
        """
        
        result2 = conn.sql(query2).fetchdf()
        
        if result2.empty:
            print("\n⚠ No date pairs found for sample patients")
        else:
            print(f"\nDetailed date pairs for {len(sample_patients)} sample patients:\n")
            print(result2.to_string(index=False))
    
    # Query 3: Distribution of days from drug to ED (excluding 0-day gaps)
    print("\n" + "=" * 100)
    print("Query 3: Distribution of days from drug event to ED visit")
    print("=" * 100)
    print("Note: Excluding 0-day gaps (likely discharge prescriptions) to focus on adverse drug event patterns")
    
    query3 = f"""
    WITH hcg_ed_events AS (
        SELECT DISTINCT
            mi_person_key,
            TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') as ed_date,
            hcg_line,
            hcg_detail
        FROM read_parquet('{medical_path}')
        WHERE {hcg_condition}
          AND incurred_date IS NOT NULL
          AND TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') IS NOT NULL
    ),
    drug_events AS (
        SELECT DISTINCT
            mi_person_key,
            TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') as drug_date
        FROM read_parquet('{pharmacy_path}')
        WHERE drug_name IS NOT NULL
          AND drug_name <> ''
          AND incurred_date IS NOT NULL
          AND TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') IS NOT NULL
    ),
    patient_ed_drugs AS (
        SELECT
            hcg.mi_person_key,
            hcg.ed_date,
            de.drug_date,
            CAST(datediff('day', CAST(de.drug_date AS DATE), CAST(hcg.ed_date AS DATE)) AS BIGINT) as days_from_drug_to_ed
        FROM hcg_ed_events hcg
        INNER JOIN drug_events de ON hcg.mi_person_key = de.mi_person_key
        WHERE de.drug_date < hcg.ed_date
          AND CAST(datediff('day', CAST(de.drug_date AS DATE), CAST(hcg.ed_date AS DATE)) AS BIGINT) > 0
    ),
    most_recent_drug_per_ed AS (
        SELECT
            mi_person_key,
            ed_date,
            MAX(drug_date) as most_recent_drug_date,
            CAST(datediff('day', CAST(MAX(drug_date) AS DATE), CAST(ed_date AS DATE)) AS BIGINT) as days_from_drug_to_ed
        FROM patient_ed_drugs
        GROUP BY mi_person_key, ed_date
    )
    SELECT
        CAST(COUNT(CASE WHEN days_from_drug_to_ed >= 1 AND days_from_drug_to_ed <= 7 THEN 1 END) AS BIGINT) as patients_1_to_7_days,
        CAST(COUNT(CASE WHEN days_from_drug_to_ed >= 8 AND days_from_drug_to_ed <= 14 THEN 1 END) AS BIGINT) as patients_8_to_14_days,
        CAST(COUNT(CASE WHEN days_from_drug_to_ed >= 15 AND days_from_drug_to_ed <= 21 THEN 1 END) AS BIGINT) as patients_15_to_21_days,
        CAST(COUNT(CASE WHEN days_from_drug_to_ed >= 22 AND days_from_drug_to_ed <= 30 THEN 1 END) AS BIGINT) as patients_22_to_30_days,
        CAST(COUNT(CASE WHEN days_from_drug_to_ed >= 31 AND days_from_drug_to_ed <= 45 THEN 1 END) AS BIGINT) as patients_31_to_45_days,
        CAST(COUNT(*) AS BIGINT) as total_ed_events,
        CAST(MIN(days_from_drug_to_ed) AS BIGINT) as min_days,
        CAST(MAX(days_from_drug_to_ed) AS BIGINT) as max_days,
        CAST(AVG(days_from_drug_to_ed) AS DOUBLE) as avg_days,
        CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY days_from_drug_to_ed) AS DOUBLE) as median_days
    FROM most_recent_drug_per_ed
    WHERE days_from_drug_to_ed > 0
      AND days_from_drug_to_ed <= 45
    """
    
    result3 = conn.sql(query3).fetchdf()
    
    if result3.empty:
        print("\n⚠ No date gaps found")
    else:
        dist = result3.iloc[0]
        print(f"\nDistribution of days from drug event to ED visit (excluding 0-day discharge prescriptions):")
        print(f"  1-7 days: {int(dist['patients_1_to_7_days']):,} ED events")
        print(f"  8-14 days: {int(dist['patients_8_to_14_days']):,} ED events")
        print(f"  15-21 days: {int(dist['patients_15_to_21_days']):,} ED events")
        print(f"  22-30 days: {int(dist['patients_22_to_30_days']):,} ED events")
        print(f"  31-45 days: {int(dist['patients_31_to_45_days']):,} ED events")
        print(f"  Total ED events with drugs within 45 days: {int(dist['total_ed_events']):,}")
        print(f"  Min: {int(dist['min_days']):,} days | Max: {int(dist['max_days']):,} days")
        print(f"  Avg: {float(dist['avg_days']):.1f} days | Median: {float(dist['median_days']):.1f} days")
        print(f"\nNote: 0-day gaps (likely discharge prescriptions) have been excluded to focus on adverse drug event patterns.")
    
    # Query 4: Sample of 1-7 day gap cases (likely adverse drug events)
    print("\n" + "=" * 100)
    print("Query 4: Sample of 1-7 day gap cases (likely adverse drug events)")
    print("=" * 100)
    
    query4 = f"""
    WITH hcg_ed_events AS (
        SELECT DISTINCT
            mi_person_key,
            TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') as ed_date,
            hcg_line,
            hcg_detail,
            primary_icd_diagnosis_code
        FROM read_parquet('{medical_path}')
        WHERE {hcg_condition}
          AND incurred_date IS NOT NULL
          AND TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') IS NOT NULL
    ),
    drug_events AS (
        SELECT DISTINCT
            mi_person_key,
            TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') as drug_date,
            drug_name
        FROM read_parquet('{pharmacy_path}')
        WHERE drug_name IS NOT NULL
          AND drug_name <> ''
          AND incurred_date IS NOT NULL
          AND TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') IS NOT NULL
    ),
    patient_ed_drugs AS (
        SELECT
            hcg.mi_person_key,
            hcg.ed_date,
            hcg.hcg_line,
            hcg.hcg_detail,
            hcg.primary_icd_diagnosis_code,
            de.drug_date,
            de.drug_name,
            CAST(datediff('day', CAST(de.drug_date AS DATE), CAST(hcg.ed_date AS DATE)) AS BIGINT) as days_from_drug_to_ed
        FROM hcg_ed_events hcg
        INNER JOIN drug_events de ON hcg.mi_person_key = de.mi_person_key
        WHERE de.drug_date < hcg.ed_date
          AND CAST(datediff('day', CAST(de.drug_date AS DATE), CAST(hcg.ed_date AS DATE)) AS BIGINT) > 0
    ),
    most_recent_drug_per_ed AS (
        SELECT
            mi_person_key,
            ed_date,
            MAX(drug_date) as most_recent_drug_date,
            CAST(datediff('day', CAST(MAX(drug_date) AS DATE), CAST(ed_date AS DATE)) AS BIGINT) as days_from_drug_to_ed
        FROM patient_ed_drugs
        GROUP BY mi_person_key, ed_date
    )
    SELECT
        med.mi_person_key,
        CAST(med.ed_date AS VARCHAR) as ed_date_str,
        med.hcg_line,
        med.hcg_detail,
        med.primary_icd_diagnosis_code,
        CAST(mrd.most_recent_drug_date AS VARCHAR) as drug_date_str,
        med.drug_name,
        mrd.days_from_drug_to_ed
    FROM most_recent_drug_per_ed mrd
    INNER JOIN patient_ed_drugs med ON mrd.mi_person_key = med.mi_person_key
        AND mrd.ed_date = med.ed_date
        AND mrd.most_recent_drug_date = med.drug_date
    WHERE mrd.days_from_drug_to_ed >= 1
      AND mrd.days_from_drug_to_ed <= 7
    LIMIT 20
    """
    
    result4 = conn.sql(query4).fetchdf()
    
    if result4.empty:
        print("\n⚠ No 1-7 day gap cases found")
    else:
        print(f"\nSample of {len(result4)} cases with 1-7 day gaps (likely adverse drug events):\n")
        print(result4.to_string(index=False))
        print("\nNote: These represent drugs taken 1-7 days before ED visit, suggesting potential adverse drug events.")
        print("0-day gaps (likely discharge prescriptions) have been excluded.")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    conn.close()
    print("\n" + "=" * 100)
    print("Complete")
    print("=" * 100)
