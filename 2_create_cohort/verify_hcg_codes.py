"""
Quick script to verify HCG target codes in gold medical data.
Queries a sample of the gold medical table to check what HCG line codes actually exist.
"""

import sys
import os
# Add project root to path so we can import py_helpers
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import duckdb
from py_helpers.constants import S3_BUCKET
from py_helpers.duckdb_utils import create_duckdb_conn

# Sample age_band and event_year - adjust if needed
age_band = "65-74"  # Example - change as needed
event_year = 2019  # Example - change as needed

# HCG codes currently used in codebase
current_hcg_codes = [
    "P51 - ER Visits and Observation Care",
    "O11 - Emergency Room",
    "P33 - Urgent Care Visits"
]

print("=" * 80)
print("HCG Target Codes Verification")
print("=" * 80)
print(f"\nCurrent HCG codes in codebase:")
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
    print("  Will try to use local paths if available")

# Try to resolve path (prefer local, fall back to S3)
medical_path = None

# Try local path first
local_dir = f"/mnt/nvme/gold/medical/age_band={age_band}/event_year={event_year}"
if os.path.exists(local_dir):
    # Find parquet files
    import glob
    parquet_files = glob.glob(f"{local_dir}/*.parquet")
    if parquet_files:
        medical_path = parquet_files[0]  # Use first file for quick check
        print(f"\n✓ Using local path: {medical_path}")
    else:
        medical_path = f"{local_dir}/*.parquet"
else:
    # Fall back to S3
    medical_path = f"s3://{S3_BUCKET}/gold/medical/age_band={age_band}/event_year={event_year}/*.parquet"
    print(f"\n→ Using S3 path: {medical_path}")

try:
    # Query 1: Check if current codes exist
    print("\n" + "=" * 80)
    print("Query 1: Checking if current HCG codes exist in data")
    print("=" * 80)
    
    codes_tuple = tuple(current_hcg_codes)
    query1 = f"""
    SELECT 
        hcg_line,
        CAST(COUNT(*) AS BIGINT) as record_count,
        CAST(COUNT(DISTINCT mi_person_key) AS BIGINT) as distinct_patients
    FROM read_parquet('{medical_path}')
    WHERE hcg_line IN {codes_tuple}
    GROUP BY hcg_line
    ORDER BY record_count DESC
    """
    
    print("Querying for current codes...")
    result1 = conn.sql(query1).fetchdf()
    
    if result1.empty:
        print("\n⚠ No records found with current HCG target codes!")
        print("  This suggests the codes may be incorrect.")
    else:
        print(f"\n✓ Found {len(result1)} of {len(current_hcg_codes)} current codes:\n")
        print(result1.to_string(index=False))
    
    # Query 2: Get sample of all HCG codes to see what actually exists
    print("\n" + "=" * 80)
    print("Query 2: Sample of all distinct HCG line codes in data")
    print("=" * 80)
    
    query2 = f"""
    SELECT DISTINCT hcg_line
    FROM read_parquet('{medical_path}')
    WHERE hcg_line IS NOT NULL
      AND hcg_line <> ''
    ORDER BY hcg_line
    LIMIT 100
    """
    
    print("Getting sample of all HCG codes...")
    result2 = conn.sql(query2).fetchdf()
    
    if result2.empty:
        print("\n⚠ No HCG line codes found in medical data")
    else:
        print(f"\nFound {len(result2)} distinct HCG line codes (showing first 100):\n")
        for idx, row in result2.iterrows():
            code = row['hcg_line']
            # Highlight if it matches current codes
            if code in current_hcg_codes:
                print(f"  ✓ {code}")
            else:
                print(f"    {code}")
        
        # Check for ED-related codes
        print("\n" + "=" * 80)
        print("ED-related HCG codes (containing ER, Emergency, Urgent, Observation, P51, O11, P33)")
        print("=" * 80)
        
        ed_keywords = ['ER', 'Emergency', 'Urgent', 'Observation', 'P51', 'O11', 'P33']
        ed_related = result2[
            result2['hcg_line'].str.contains('|'.join(ed_keywords), case=False, na=False)
        ]
        
        if not ed_related.empty:
            print(f"\nFound {len(ed_related)} potentially ED-related codes:\n")
            for idx, row in ed_related.iterrows():
                code = row['hcg_line']
                if code in current_hcg_codes:
                    print(f"  ✓ {code} (CURRENT)")
                else:
                    print(f"    {code} (NOT IN CURRENT LIST)")
        else:
            print("\nNo ED-related codes found in sample")
    
    # Query 3: Sample records with current codes
    print("\n" + "=" * 80)
    print("Query 3: Sample records with current HCG target codes")
    print("=" * 80)
    
    query3 = f"""
    SELECT 
        hcg_line,
        hcg_setting,
        hcg_detail,
        primary_icd_diagnosis_code,
        event_date
    FROM read_parquet('{medical_path}')
    WHERE hcg_line IN {codes_tuple}
    LIMIT 5
    """
    
    result3 = conn.sql(query3).fetchdf()
    
    if result3.empty:
        print("\n⚠ No sample records found with current HCG target codes")
    else:
        print(f"\nSample of {len(result3)} records:\n")
        print(result3.to_string(index=False))
    
except Exception as e:
    print(f"\n✗ Error querying data: {e}")
    import traceback
    traceback.print_exc()

finally:
    conn.close()
    print("\n" + "=" * 80)
    print("Verification complete")
    print("=" * 80)
