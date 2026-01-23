"""
Quick script to sample HCG field values from gold medical data.
Shows actual values for hcg_setting, hcg_line, and hcg_detail.
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

# HCG codes currently used in codebase (using hcg_detail for precision)
# P51: Only P51b (ED Visits), exclude P51a (Observation Care)
# O11: Emergency Room (all details)
# P33: Urgent Care Visits (all details)
current_hcg_condition = """
    (hcg_line = 'P51 - ER Visits and Observation Care' AND hcg_detail = 'P51b - PHY ED Visits and Observation Care - ED Visits')
    OR hcg_line = 'O11 - Emergency Room'
    OR hcg_line = 'P33 - Urgent Care Visits'
"""
current_hcg_codes = [
    "P51 - ER Visits and Observation Care (P51b only)",
    "O11 - Emergency Room",
    "P33 - Urgent Care Visits"
]

print("=" * 100)
print("HCG Field Values Sample from Gold Medical Data")
print("=" * 100)
print(f"\nLooking for HCG codes:")
for code in current_hcg_codes:
    print(f"  - '{code}'")

# Create DuckDB connection
conn = create_duckdb_conn()

# Load AWS credentials
try:
    conn.sql("INSTALL httpfs; LOAD httpfs;")
    conn.sql("CALL load_aws_credentials('pgx');")
    print("\n✓ AWS credentials loaded")
except Exception as e:
    print(f"\n⚠ Warning: Could not load AWS credentials: {e}")

# Try to resolve path
medical_path = None
local_dir = f"/mnt/nvme/gold/medical/age_band={age_band}/event_year={event_year}"
if os.path.exists(local_dir):
    import glob
    parquet_files = glob.glob(f"{local_dir}/*.parquet")
    if parquet_files:
        medical_path = parquet_files[0]
        print(f"\n✓ Using local path: {medical_path}")
    else:
        medical_path = f"{local_dir}/*.parquet"
else:
    medical_path = f"s3://{S3_BUCKET}/gold/medical/age_band={age_band}/event_year={event_year}/*.parquet"
    print(f"\n→ Using S3 path: {medical_path}")

try:
    # Query 1: Sample records with all HCG fields
    print("\n" + "=" * 100)
    print("Sample 1: Records with ANY HCG line code (showing all 3 HCG fields)")
    print("=" * 100)
    
    query1 = f"""
    SELECT 
        hcg_setting,
        hcg_line,
        hcg_detail,
        primary_icd_diagnosis_code,
        event_date
    FROM read_parquet('{medical_path}')
    WHERE hcg_line IS NOT NULL
      AND hcg_line <> ''
    LIMIT 20
    """
    
    result1 = conn.sql(query1).fetchdf()
    
    if result1.empty:
        print("\n⚠ No records with HCG line codes found")
    else:
        print(f"\nFound {len(result1)} sample records:\n")
        # Show all columns with proper formatting
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        print(result1.to_string(index=False))
    
    # Query 2: Check for current target codes
    print("\n" + "=" * 100)
    print("Sample 2: Records matching CURRENT HCG target codes")
    print("=" * 100)
    
    codes_tuple = tuple(current_hcg_codes)
    query2 = f"""
    SELECT 
        hcg_setting,
        hcg_line,
        hcg_detail,
        primary_icd_diagnosis_code,
        event_date
    FROM read_parquet('{medical_path}')
    WHERE {current_hcg_condition}
    LIMIT 10
    """
    
    result2 = conn.sql(query2).fetchdf()
    
    if result2.empty:
        print("\n⚠ No records found with current HCG target codes!")
        print("  This suggests the codes may be incorrect or formatted differently.")
    else:
        print(f"\nFound {len(result2)} records with current codes:\n")
        print(result2.to_string(index=False))
    
    # Query 3: Get distinct HCG line values (top 30)
    print("\n" + "=" * 100)
    print("Sample 3: Top 30 distinct HCG line codes (by frequency)")
    print("=" * 100)
    
    query3 = f"""
    SELECT 
        hcg_line,
        hcg_detail,
        CAST(COUNT(*) AS BIGINT) as count,
        CAST(COUNT(DISTINCT mi_person_key) AS BIGINT) as patients
    FROM read_parquet('{medical_path}')
    WHERE hcg_line IS NOT NULL
      AND hcg_line <> ''
    GROUP BY hcg_line, hcg_detail
    ORDER BY count DESC
    LIMIT 30
    """
    
    result3 = conn.sql(query3).fetchdf()
    
    if result3.empty:
        print("\n⚠ No HCG line codes found")
    else:
        print(f"\nTop {len(result3)} HCG line codes (with detail):\n")
        for idx, row in result3.iterrows():
            code = row['hcg_line']
            detail = row.get('hcg_detail', '')
            count = row['count']
            patients = row['patients']
            # Check if this matches our condition
            is_match = (
                (code == 'P51 - ER Visits and Observation Care' and detail == 'P51b - PHY ED Visits and Observation Care - ED Visits')
                or code == 'O11 - Emergency Room'
                or code == 'P33 - Urgent Care Visits'
            )
            marker = "✓" if is_match else " "
            detail_str = f" / {detail}" if detail else ""
            print(f"{marker} {code:60s}{detail_str:50s} | {count:>10,} records | {patients:>10,} patients")
    
    # Query 4: Check for ED-related codes
    print("\n" + "=" * 100)
    print("Sample 4: All ED-related HCG codes (ER, Emergency, Urgent, Observation, P51, O11, P33)")
    print("=" * 100)
    
    query4 = f"""
    SELECT DISTINCT
        hcg_setting,
        hcg_line,
        hcg_detail
    FROM read_parquet('{medical_path}')
    WHERE hcg_line IS NOT NULL
      AND hcg_line <> ''
      AND (
          UPPER(hcg_line) LIKE '%ER%'
          OR UPPER(hcg_line) LIKE '%EMERGENCY%'
          OR UPPER(hcg_line) LIKE '%URGENT%'
          OR UPPER(hcg_line) LIKE '%OBSERVATION%'
          OR UPPER(hcg_line) LIKE '%P51%'
          OR UPPER(hcg_line) LIKE '%O11%'
          OR UPPER(hcg_line) LIKE '%P33%'
      )
    ORDER BY hcg_line
    LIMIT 50
    """
    
    result4 = conn.sql(query4).fetchdf()
    
    if result4.empty:
        print("\n⚠ No ED-related HCG codes found")
    else:
        print(f"\nFound {len(result4)} ED-related HCG code combinations:\n")
        for idx, row in result4.iterrows():
            setting = row['hcg_setting'] or '(NULL)'
            line = row['hcg_line']
            detail = row['hcg_detail'] or '(NULL)'
            # Check if this matches our condition
            is_match = (
                (line == 'P51 - ER Visits and Observation Care' and detail == 'P51b - PHY ED Visits and Observation Care - ED Visits')
                or line == 'O11 - Emergency Room'
                or line == 'P33 - Urgent Care Visits'
            )
            marker = "✓" if is_match else " "
            print(f"{marker} Setting: {setting:30s} | Line: {line:50s} | Detail: {detail}")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    conn.close()
    print("\n" + "=" * 100)
    print("Complete")
    print("=" * 100)
