#!/usr/bin/env python3
"""
Global Demographic Imputation Preprocessing

This script runs BEFORE partition-based processing to globally impute missing
demographics across all pharmacy and medical data. This eliminates the need
for per-partition imputation and dramatically improves performance.

Architecture:
1. Load all pharmacy data and identify missing demographics
2. Load all medical data and identify missing demographics  
3. Run bidirectional imputation globally
4. Save imputed demographics lookup table to silver tier
5. Save pre-imputed pharmacy data with demographics filled to silver tier

The output is designed to work with the optimized clean_pharmacy.py script
for fast partition-based processing.

Usage:
    python global_imputation.py --pharmacy-input s3://pgxdatalake/bronze/pharmacy/*.parquet \
                                --medical-input s3://pgxdatalake/bronze/medical/*.parquet \
                                --output-root s3://pgxdatalake/silver/imputed \
                                --lookahead-years 5
"""

import os
import sys
import argparse
import time
import json
import duckdb
import boto3
from typing import Dict, Any
import logging

# Add project root to path (helpers folder is at pgx-analysis level)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from helpers_1997_13.logging_utils import setup_logging, save_logs_to_s3, save_logs_checkpoint
from helpers_1997_13.duckdb_utils import create_simple_duckdb_connection
from helpers_1997_13.data_utils import validate_data_for_blank_strings
from helpers_1997_13.constants import S3_BUCKET
from helpers_1997_13.s3_utils import s3_directory_exists_with_files

def get_silver_imputed_paths(output_root: str) -> dict:
    """Build proper S3 paths for silver imputed data using s3_utils conventions."""
    # Ensure output_root is in silver tier
    if not output_root.startswith(f"s3://{S3_BUCKET}/silver/"):
        if output_root.startswith("s3://"):
            # Extract bucket and path from existing s3://bucket/path
            parts = output_root[5:].split("/", 1)
            if len(parts) == 2:
                bucket, path = parts
                output_root = f"s3://{bucket}/silver/{path}"
            else:
                output_root = f"s3://{S3_BUCKET}/silver/imputed"
        else:
            # Treat as relative path
            output_root = f"s3://{S3_BUCKET}/silver/{output_root}"
    
    return {
        "demographics_lookup": f"{output_root}/mi_person_key_demographics_lookup.parquet",
        "base_path": output_root
    }

def init_duckdb(tmp_dir: str = None, s3_region: str = "us-east-1", logger=None):
    """Initialize DuckDB with simplified settings - let DuckDB auto-detect memory/threads"""
    # Use simplified DuckDB connection (lessons learned from troubleshooting)
    duckdb_conn = create_simple_duckdb_connection(logger)
    
    # Set temp directory if provided
    if tmp_dir:
        duckdb_conn.sql(f"SET temp_directory='{tmp_dir}'")
    else:
        duckdb_conn.sql("SET temp_directory='/tmp'")
    
    # Set S3 region
    duckdb_conn.sql(f"SET s3_region='{s3_region}'")
    
    # Let DuckDB handle memory and threads automatically - NO manual settings
    logger.info("✅ Simple DuckDB connection created - auto memory/threads")
    
    return duckdb_conn

def log_data_loss(step_name: str, before_count: int, after_count: int, logger, 
                  before_patients: int = None, after_patients: int = None):
    """Log data loss between steps"""
    lost_count = before_count - after_count
    loss_pct = (lost_count / before_count * 100) if before_count > 0 else 0
    
    if before_patients is not None and after_patients is not None:
        lost_patients = before_patients - after_patients
        patient_loss_pct = (lost_patients / before_patients * 100) if before_patients > 0 else 0
        logger.info(f"📊 {step_name}: {before_count:,} → {after_count:,} rows ({lost_count:,} lost, {loss_pct:.1f}%) | {before_patients:,} → {after_patients:,} patients ({lost_patients:,} lost, {patient_loss_pct:.1f}%)")
    else:
        logger.info(f"📊 {step_name}: {before_count:,} → {after_count:,} rows ({lost_count:,} lost, {loss_pct:.1f}%)")

def create_raw_silver_datasets(pharmacy_input: str, medical_input: str, output_root: str,
                               duckdb_conn, logger, log_buffer):
    """Create raw silver datasets preserving ALL original columns from bronze."""
    
    logger.info("=" * 80)
    logger.info("CREATING RAW SILVER DATASETS (All Original Columns)")
    logger.info("=" * 80)
    
    # Determine raw silver paths
    raw_base = output_root.replace("/imputed", "").replace("/silver", "/silver")
    if not raw_base.endswith("/silver"):
        raw_base = f"{raw_base}/silver" if not raw_base.endswith("/") else f"{raw_base}silver"
    
    pharmacy_raw_path = f"{raw_base}/pharmacy_raw"
    medical_raw_path = f"{raw_base}/medical_raw"
    
    logger.info(f"Pharmacy raw output: {pharmacy_raw_path}")
    logger.info(f"Medical raw output: {medical_raw_path}")
    
    # Check if raw silver datasets already exist (idempotency)
    from urllib.parse import urlparse
    import subprocess
    
    def check_s3_path_exists(s3_path):
        if s3_path.startswith('s3://'):
            parsed = urlparse(s3_path)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
            check_cmd = ["aws", "s3", "ls", f"s3://{bucket}/{key}/"]
            result = subprocess.run(check_cmd, capture_output=True, text=True)
            return result.returncode == 0
        else:
            return os.path.exists(s3_path)
    
    pharmacy_raw_exists = check_s3_path_exists(pharmacy_raw_path)
    medical_raw_exists = check_s3_path_exists(medical_raw_path)
    
    # Check what needs to be created (idempotent - only create missing datasets)
    if pharmacy_raw_exists and medical_raw_exists:
        logger.info(f"✅ Raw silver datasets already exist:")
        logger.info(f"   • Pharmacy: {pharmacy_raw_path}")
        logger.info(f"   • Medical: {medical_raw_path}")
        logger.info(f"   Skipping raw silver dataset creation (idempotent).")
        return {
            "pharmacy_raw_path": pharmacy_raw_path,
            "medical_raw_path": medical_raw_path,
            "pharmacy_raw_count": None,  # Not counted if skipped
            "medical_raw_count": None,
            "skipped": True
        }
    
    # Determine what needs to be created
    create_pharmacy = not pharmacy_raw_exists
    create_medical = not medical_raw_exists
    
    if create_pharmacy:
        logger.info(f"📊 Will create: {pharmacy_raw_path}")
    else:
        logger.info(f"✅ Already exists (skipping): {pharmacy_raw_path}")
    
    if create_medical:
        logger.info(f"📊 Will create: {medical_raw_path}")
    else:
        logger.info(f"✅ Already exists (skipping): {medical_raw_path}")
    
    # Create raw pharmacy with all original columns (only if missing)
    pharmacy_raw_count = None
    if create_pharmacy:
        logger.info("Creating raw pharmacy dataset with all original columns...")
        # Explicitly list all columns from pharmacy_head.txt to ensure correct column names
        # This prevents DuckDB from inferring column names from data values
        duckdb_conn.sql(f"""
            CREATE OR REPLACE VIEW pharmacy_raw_with_partitions AS
            SELECT 
                "Incurred Date",
                "Claim ID",
                "MI Person Key",
                "Payer LOB",
                "Payer Type",
                "Claim Status",
                "Primary Insurance Flag",
                "Member Zip Code DOS",
                "Member County DOS",
                "Member State ENROLL",
                "Member Age DOS",
                "Member Age Band DOS",
                "ADULT_FLAG",
                "Member Gender",
                "Member Race",
                "Hispanic Indicator",
                "HCG Setting",
                "HCG Line",
                "HCG Detail",
                "Therapeutic Class 1",
                "Therapeutic Class 2",
                "Therapeutic Class 3",
                "NDC",
                "Drug Code",
                "Drug Name",
                "GPI",
                "GPI Generic Name",
                "Manufacturer",
                "Strength",
                "Dosage Form",
                "Billing Provider NPI",
                "Billing Provider Specialty",
                "Billing Provider ZIP",
                "Billing Provider County",
                "Billing Provider State",
                "Billing Provider MSA",
                "Billing Provider Taxonomy",
                "Billing Provider TIN",
                "Service Provider Name",
                "Service Provider NPI",
                "Service Provider Specialty",
                "Service Provider ZIP",
                "Service Provider County",
                "Service Provider State",
                "Service Provider MSA",
                "Service Provider Taxonomy",
                "Service Provider TIN",
                "Total Allowed",
                "Total Utilization",
                "Total RX Paid",
                "Total RX Days Supply",
                -- Derive age_band from Member Age DOS
                CASE
                    WHEN TRY_CAST("Member Age DOS" AS INTEGER) BETWEEN 0  AND 12  THEN '0-12'
                    WHEN TRY_CAST("Member Age DOS" AS INTEGER) BETWEEN 13 AND 24  THEN '13-24'
                    WHEN TRY_CAST("Member Age DOS" AS INTEGER) BETWEEN 25 AND 44  THEN '25-44'
                    WHEN TRY_CAST("Member Age DOS" AS INTEGER) BETWEEN 45 AND 54  THEN '45-54'
                    WHEN TRY_CAST("Member Age DOS" AS INTEGER) BETWEEN 55 AND 64  THEN '55-64'
                    WHEN TRY_CAST("Member Age DOS" AS INTEGER) BETWEEN 65 AND 74  THEN '65-74'
                    WHEN TRY_CAST("Member Age DOS" AS INTEGER) BETWEEN 75 AND 84  THEN '75-84'
                    WHEN TRY_CAST("Member Age DOS" AS INTEGER) BETWEEN 85 AND 94  THEN '85-94'
                    WHEN TRY_CAST("Member Age DOS" AS INTEGER) BETWEEN 95 AND 114 THEN '95-114'
                    ELSE 'Other'
                END AS age_band,
                -- Derive event_year from Incurred Date
                TRY_CAST(EXTRACT(YEAR FROM TRY_STRPTIME(CAST("Incurred Date" AS VARCHAR), '%Y%m%d')) AS INTEGER) AS event_year
            FROM (
                SELECT * EXCLUDE (filename)
                FROM read_parquet('{pharmacy_input}', union_by_name=true, filename=true)
                WHERE NOT regexp_matches(filename, '\.part_[^/]*\.parquet$')
            )
            WHERE "MI Person Key" IS NOT NULL
              AND "Incurred Date" IS NOT NULL
              AND regexp_matches(CAST("Incurred Date" AS VARCHAR), '^[0-9]{{8}}$')
              AND TRY_STRPTIME(CAST("Incurred Date" AS VARCHAR), '%Y%m%d') IS NOT NULL
              AND TRY_CAST(EXTRACT(YEAR FROM TRY_STRPTIME(CAST("Incurred Date" AS VARCHAR), '%Y%m%d')) AS INTEGER) BETWEEN 2016 AND 2020
        """)
        
        pharmacy_raw_count = duckdb_conn.sql("SELECT COUNT(*) FROM pharmacy_raw_with_partitions").fetchone()[0]
        logger.info(f"Pharmacy raw records: {pharmacy_raw_count:,}")
        
        # Write partitioned parquet (exclude partition columns from data to avoid Glue duplicates)
        logger.info(f"Writing partitioned pharmacy raw data to: {pharmacy_raw_path}")
        duckdb_conn.sql(f"""
            COPY pharmacy_raw_with_partitions
            TO '{pharmacy_raw_path}'
            (FORMAT PARQUET, 
            PARTITION_BY (age_band, event_year), 
            WRITE_PARTITION_COLUMNS FALSE,
            OVERWRITE_OR_IGNORE true)
        """)
        logger.info("✅ Pharmacy raw dataset created successfully")
    else:
        logger.info("⏭️ Skipping pharmacy raw dataset (already exists)")
    
    # Create raw medical with all original columns (only if missing)
    medical_raw_count = None
    if create_medical:
        logger.info("Creating raw medical dataset with all original columns...")
        duckdb_conn.sql(f"""
            CREATE OR REPLACE VIEW medical_raw_with_partitions AS
            SELECT 
                *,
                -- Derive age_band from Member Age DOS
                CASE
                    WHEN TRY_CAST("Member Age DOS" AS INTEGER) BETWEEN 0  AND 12  THEN '0-12'
                    WHEN TRY_CAST("Member Age DOS" AS INTEGER) BETWEEN 13 AND 24  THEN '13-24'
                    WHEN TRY_CAST("Member Age DOS" AS INTEGER) BETWEEN 25 AND 44  THEN '25-44'
                    WHEN TRY_CAST("Member Age DOS" AS INTEGER) BETWEEN 45 AND 54  THEN '45-54'
                    WHEN TRY_CAST("Member Age DOS" AS INTEGER) BETWEEN 55 AND 64  THEN '55-64'
                    WHEN TRY_CAST("Member Age DOS" AS INTEGER) BETWEEN 65 AND 74  THEN '65-74'
                    WHEN TRY_CAST("Member Age DOS" AS INTEGER) BETWEEN 75 AND 84  THEN '75-84'
                    WHEN TRY_CAST("Member Age DOS" AS INTEGER) BETWEEN 85 AND 94  THEN '85-94'
                    WHEN TRY_CAST("Member Age DOS" AS INTEGER) BETWEEN 95 AND 114 THEN '95-114'
                    ELSE 'Other'
                END AS age_band,
                -- Derive event_year from Incurred Date
                TRY_CAST(EXTRACT(YEAR FROM TRY_STRPTIME(CAST("Incurred Date" AS VARCHAR), '%Y%m%d')) AS INTEGER) AS event_year
            FROM (
                SELECT * EXCLUDE (filename)
                FROM read_parquet('{medical_input}', union_by_name=true, filename=true)
                WHERE NOT regexp_matches(filename, '\.part_[^/]*\.parquet$')
            )
            WHERE "MI Person Key" IS NOT NULL
              AND "Incurred Date" IS NOT NULL
              AND regexp_matches(CAST("Incurred Date" AS VARCHAR), '^[0-9]{{8}}$')
              AND TRY_STRPTIME(CAST("Incurred Date" AS VARCHAR), '%Y%m%d') IS NOT NULL
              AND TRY_CAST(EXTRACT(YEAR FROM TRY_STRPTIME(CAST("Incurred Date" AS VARCHAR), '%Y%m%d')) AS INTEGER) BETWEEN 2016 AND 2020
        """)
        
        medical_raw_count = duckdb_conn.sql("SELECT COUNT(*) FROM medical_raw_with_partitions").fetchone()[0]
        logger.info(f"Medical raw records: {medical_raw_count:,}")
        
        # Write partitioned parquet (exclude partition columns from data to avoid Glue duplicates)
        logger.info(f"Writing partitioned medical raw data to: {medical_raw_path}")
        duckdb_conn.sql(f"""
            COPY medical_raw_with_partitions
            TO '{medical_raw_path}'
            (FORMAT PARQUET, 
            PARTITION_BY (age_band, event_year), 
            WRITE_PARTITION_COLUMNS FALSE,
            OVERWRITE_OR_IGNORE true)
        """)
        logger.info("✅ Medical raw dataset created successfully")
    else:
        logger.info("⏭️ Skipping medical raw dataset (already exists)")
    
    logger.info("=" * 80)
    logger.info("RAW SILVER DATASETS CREATION COMPLETE")
    logger.info("=" * 80)
    if pharmacy_raw_count is not None:
        logger.info(f"Pharmacy raw: {pharmacy_raw_path} ({pharmacy_raw_count:,} records)")
    else:
        logger.info(f"Pharmacy raw: {pharmacy_raw_path} (skipped - already exists)")
    
    if medical_raw_count is not None:
        logger.info(f"Medical raw: {medical_raw_path} ({medical_raw_count:,} records)")
    else:
        logger.info(f"Medical raw: {medical_raw_path} (skipped - already exists)")
    
    return {
        "pharmacy_raw_path": pharmacy_raw_path,
        "medical_raw_path": medical_raw_path,
        "pharmacy_raw_count": pharmacy_raw_count,
        "medical_raw_count": medical_raw_count,
        "skipped": not create_pharmacy and not create_medical  # True if both were skipped
    }

def run_global_imputation(pharmacy_input: str, medical_input: str, output_root: str, 
                         lookahead_years: int, tmp_dir: str, logger, log_buffer, 
                         create_demographics_lookup: bool = True, create_raw_silver: bool = False):
    """Run global demographic imputation across all data"""
    
    logger.info("🚀 Starting Global Demographic Imputation")
    logger.info("🔧 Using Version 1997 + 12 - Global Imputation (DuckDB Lessons Learned Applied)")
    logger.info("=" * 80)

    # Build proper silver paths
    silver_paths = get_silver_imputed_paths(output_root)
    logger.info(f"📁 Silver imputed paths:")
    logger.info(f"   • Base path: {silver_paths['base_path']}")
    logger.info(f"   • mi_person_key demographics lookup: {silver_paths['demographics_lookup']}")

    # Initialize helper function for checking S3 paths
    from urllib.parse import urlparse
    import subprocess
    
    def check_s3_path_exists(s3_path):
        if s3_path.startswith('s3://'):
            parsed = urlparse(s3_path)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
            check_cmd = ["aws", "s3", "ls", f"s3://{bucket}/{key}/"]
            result = subprocess.run(check_cmd, capture_output=True, text=True)
            return result.returncode == 0
        else:
            return os.path.exists(s3_path)
    
    # Step 0: Create raw silver datasets if requested (preserves all original columns)
    # This runs independently of partitioned data existence - force creation if requested
    raw_results = None
    if create_raw_silver:
        logger.info("=" * 80)
        logger.info("STEP 0: Creating Raw Silver Datasets")
        logger.info("=" * 80)
        logger.info("Note: Raw silver creation is independent of partitioned data existence")
        
        # Initialize DuckDB early for raw silver creation
        duckdb_conn = init_duckdb(tmp_dir, logger=logger)
        save_logs_checkpoint(log_buffer, "global_imputation", "global", "all", "step0a_raw_silver_started", logger=logger)
        raw_results = create_raw_silver_datasets(pharmacy_input, medical_input, output_root, duckdb_conn, logger, log_buffer)
        save_logs_checkpoint(log_buffer, "global_imputation", "global", "all", "step0b_raw_silver_complete", logger=logger)
        logger.info("✅ Raw silver datasets creation complete")
    
    # Check if imputed partitioned data already exists (more important than demographics lookup)
    pharmacy_partitioned_path = f"{silver_paths['base_path']}/pharmacy_partitioned"
    medical_partitioned_path = f"{silver_paths['base_path']}/medical_partitioned"
    
    pharmacy_exists = check_s3_path_exists(pharmacy_partitioned_path)
    medical_exists = check_s3_path_exists(medical_partitioned_path)
    
    if pharmacy_exists and medical_exists:
        logger.info(f"✅ Imputed partitioned data already exists:")
        logger.info(f"   • Pharmacy: {pharmacy_partitioned_path}")
        logger.info(f"   • Medical: {medical_partitioned_path}")
        
        # If we only wanted to create raw silver, we're done
        if create_raw_silver and not create_demographics_lookup:
            logger.info(f"📊 Raw silver creation complete. Partitioned data already exists. Exiting.")
            return
        
        if create_demographics_lookup:
            # Check if demographics lookup exists
            lookup_path = silver_paths['demographics_lookup']
            if check_s3_path_exists(lookup_path):
                logger.info(f"✅ Demographics lookup also exists at {lookup_path}. Skipping global imputation.")
                return
            else:
                logger.info(f"📊 Demographics lookup missing, will create it...")
        else:
            logger.info(f"📊 Demographics lookup not needed (create_demographics_lookup=False). Skipping global imputation.")
            return
    else:
        logger.info(f"📊 Imputed partitioned data missing, will create it...")
        if not pharmacy_exists:
            logger.info(f"   • Missing: {pharmacy_partitioned_path}")
        if not medical_exists:
            logger.info(f"   • Missing: {medical_partitioned_path}")
    
    # Initialize DuckDB with simplified settings (if not already initialized for raw silver)
    if not create_raw_silver:
    duckdb_conn = init_duckdb(tmp_dir, logger=logger)
    
    # Save initial checkpoint
    save_logs_checkpoint(log_buffer, "global_imputation", "global", "all", "step0_pipeline_started", logger=logger)
    
    # Step 1: Load and normalize pharmacy data
    logger.info("📊 Step 1: Loading and normalizing pharmacy data...")
    
    initial_pharmacy_count = duckdb_conn.sql(f"SELECT COUNT(*) FROM read_parquet('{pharmacy_input}', union_by_name=true)").fetchone()[0]
    # Track rows with missing MI Person Key (to be dropped)
    pharmacy_missing_key = duckdb_conn.sql(
        f"""
        SELECT COUNT(*) 
        FROM read_parquet('{pharmacy_input}', union_by_name=true, filename=true)
        WHERE NOT regexp_matches(filename, '\.part_[^/]*\.parquet$')
          AND "MI Person Key" IS NULL
        """
    ).fetchone()[0]
    logger.info(f"🗑️ Pharmacy rows dropped (missing MI Person Key): {pharmacy_missing_key:,}")
    
    # Parquet preserves source column names; quote exact names. Count distinct present keys
    initial_pharmacy_patients = duckdb_conn.sql(
        f"""
        SELECT COUNT(DISTINCT "MI Person Key") 
        FROM read_parquet('{pharmacy_input}', union_by_name=true, filename=true)
        WHERE NOT regexp_matches(filename, '\.part_[^/]*\.parquet$')
          AND "MI Person Key" IS NOT NULL
        """
    ).fetchone()[0]
    
    logger.info(f"📊 Initial pharmacy records: {initial_pharmacy_count:,}")
    logger.info(f"📊 Initial pharmacy patients: {initial_pharmacy_patients:,}")
    
    # Create normalized pharmacy view with safety conversions
    duckdb_conn.sql(f"""
        CREATE OR REPLACE VIEW pharmacy_normalized AS
        SELECT 
            CAST("MI Person Key" AS VARCHAR) AS mi_person_key,
            TRY_CAST(EXTRACT(YEAR FROM TRY_STRPTIME(CAST("Incurred Date" AS VARCHAR), '%Y%m%d')) AS INTEGER) AS event_year,
            CASE
                WHEN "Member Age DOS" IS NULL THEN NULL
                WHEN TRIM(CAST("Member Age DOS" AS VARCHAR)) = '' THEN NULL
                WHEN UPPER(TRIM(CAST("Member Age DOS" AS VARCHAR))) = 'NULL' THEN NULL
                ELSE TRY_CAST(TRIM(CAST("Member Age DOS" AS VARCHAR)) AS INTEGER)
            END AS member_age_dos,
            NULLIF(TRIM("Member Gender"), '') AS member_gender,
            NULLIF(TRIM("Member Race"), '') AS member_race,
            NULLIF(TRIM("Member Zip Code DOS"), '') AS member_zip_code_dos,
            NULLIF(TRIM("Member County DOS"), '') AS member_county_dos,
            NULLIF(TRIM("Payer Type"), '') AS payer_type,
            "Drug Name" AS drug_name,
            "Incurred Date" AS incurred_date,
            "Total Utilization" AS total_utilization
        FROM (
            SELECT * EXCLUDE (filename)
            FROM read_parquet('{pharmacy_input}', union_by_name=true, filename=true)
            WHERE NOT regexp_matches(filename, '\.part_[^/]*\.parquet$')
        )
        WHERE "MI Person Key" IS NOT NULL
          AND "Incurred Date" IS NOT NULL
          AND regexp_matches(CAST("Incurred Date" AS VARCHAR), '^[0-9]{{8}}$')
          AND TRY_STRPTIME(CAST("Incurred Date" AS VARCHAR), '%Y%m%d') IS NOT NULL
    """)
    
    normalized_pharmacy_count = duckdb_conn.sql("SELECT COUNT(*) FROM pharmacy_normalized").fetchone()[0]
    normalized_pharmacy_patients = duckdb_conn.sql("SELECT COUNT(DISTINCT mi_person_key) FROM pharmacy_normalized").fetchone()[0]
    log_data_loss("After pharmacy normalization", initial_pharmacy_count, normalized_pharmacy_count, 
                  logger, initial_pharmacy_patients, normalized_pharmacy_patients)
    
    # Save checkpoint after pharmacy normalization
    save_logs_checkpoint(log_buffer, "global_imputation", "global", "all", "step1_pharmacy_normalized", logger=logger)
    
    # Step 2: Load and normalize medical data
    logger.info("📊 Step 2: Loading and normalizing medical data...")
    
    # Filter out .part_*.parquet files which have incorrect schemas
    initial_medical_count = duckdb_conn.sql(f"""
        SELECT COUNT(*) 
        FROM read_parquet('{medical_input}', union_by_name=true, filename=true)
        WHERE NOT regexp_matches(filename, '\.part_[^/]*\.parquet$')
    """).fetchone()[0]
    # Track rows with missing MI Person Key (to be dropped)
    medical_missing_key = duckdb_conn.sql(
        f"""
        SELECT COUNT(*) 
        FROM read_parquet('{medical_input}', union_by_name=true, filename=true)
        WHERE NOT regexp_matches(filename, '\.part_[^/]*\.parquet$')
          AND "MI Person Key" IS NULL
        """
    ).fetchone()[0]
    logger.info(f"🗑️ Medical rows dropped (missing MI Person Key): {medical_missing_key:,}")
    
    initial_medical_patients = duckdb_conn.sql(
        f"""
        SELECT COUNT(DISTINCT "MI Person Key") 
        FROM read_parquet('{medical_input}', union_by_name=true, filename=true)
        WHERE NOT regexp_matches(filename, '\.part_[^/]*\.parquet$')
          AND "MI Person Key" IS NOT NULL
        """
    ).fetchone()[0]
    
    logger.info(f"📊 Initial medical records: {initial_medical_count:,}")
    logger.info(f"📊 Initial medical patients: {initial_medical_patients:,}")
    
    # Create normalized medical view with safety conversions
    # First, let's check what columns are actually available in the medical data
    logger.info("🔍 Checking medical data schema...")
    try:
        # Get column names as they appear in the raw Parquet file
        medical_columns = duckdb_conn.sql(f"""
            DESCRIBE SELECT * EXCLUDE (filename)
            FROM read_parquet('{medical_input}', union_by_name=true, filename=true)
            WHERE NOT regexp_matches(filename, '\.part_[^/]*\.parquet$')
            LIMIT 1
        """).fetchall()
        logger.info(f"📊 Raw medical columns (Parquet metadata): {[col[0] for col in medical_columns]}")
        
        # Show how they'll appear after DuckDB processing (lowercase)
        logger.info(f"📊 Processed medical columns (DuckDB will convert to lowercase): {[col[0].lower() for col in medical_columns]}")
    except Exception as e:
        logger.warning(f"⚠ Could not describe medical schema: {e}")
    
    # Detect available medical columns dynamically (like clean_medical.py)
    try:
        src_columns_df = duckdb_conn.sql(f"""
            SELECT * EXCLUDE (filename)
            FROM read_parquet('{medical_input}', union_by_name=true, filename=true)
            WHERE NOT regexp_matches(filename, '\.part_[^/]*\.parquet$')
            LIMIT 0
        """).df()
        src_cols = set(src_columns_df.columns)
    except Exception:
        src_cols = set()

    def pick(*candidates: str) -> str:
        for c in candidates:
            if c in src_cols:
                return f'"{c}"'
        return 'NULL'

    # Resolve schema variants for ICD columns
    two_code    = pick('2nd_ICD_Diagnosis_Code')
    two_rollup  = pick('2nd_ICD_Diagnosis_Rollup', '2nd_ICD_Rollup')
    three_code  = pick('3rd_ICD_Diagnosis_Code')
    three_rollup= pick('3rd_ICD_Diagnosis_Rollup', '3rd_ICD_Rollup')
    four_code   = pick('4th_ICD_Diagnosis_Code')
    four_rollup = pick('4th_ICD_Diagnosis_Rollup', '4th_ICD_Rollup')
    five_code   = pick('5th_ICD_Diagnosis_Code')
    five_rollup = pick('5th_ICD_Diagnosis_Rollup', '5th_ICD_Rollup')
    six_code    = pick('6th_ICD_Diagnosis_Code')
    six_rollup  = pick('6th_ICD_Diagnosis_Rollup', '6th_ICD_Rollup')
    seven_code  = pick('7th_ICD_Diagnosis_Code')
    seven_rollup= pick('7th_ICD_Diagnosis_Rollup', '7th_ICD_Rollup')
    eight_code  = pick('8th_ICD_Diagnosis_Code')
    eight_rollup= pick('8th_ICD_Diagnosis_Rollup', '8th_ICD_Rollup')
    nine_code   = pick('9th_ICD_Diagnosis_Code')
    nine_rollup = pick('9th_ICD_Diagnosis_Rollup', '9th_ICD_Rollup')
    ten_code    = pick('10th_ICD_Diagnosis_Code')
    ten_rollup  = pick('10th_ICD_Diagnosis_Rollup', '10th_ICD_Rollup')

    # Procedure code variants
    two_proc    = pick('2nd_ICD_Procedure_Code')
    three_proc  = pick('3rd_ICD_Procedure_Code')
    four_proc   = pick('4th_ICD_Procedure_Code')
    five_proc   = pick('5th_ICD_Procedure_Code')
    six_proc    = pick('6th_ICD_Procedure_Code')
    seven_proc  = pick('7th_ICD_Procedure_Code')
    eight_proc  = pick('8th_ICD_Procedure_Code')
    nine_proc   = pick('9th_ICD_Procedure_Code')
    ten_proc    = pick('10th_ICD_Procedure_Code')

    # Create a medical view using quoted source names and snake_case aliases
    duckdb_conn.sql(f"""
        CREATE OR REPLACE VIEW medical_normalized AS
        SELECT 
            CAST("MI Person Key" AS VARCHAR) AS mi_person_key,
            CAST("Claim ID" AS VARCHAR) AS claim_id,
            "Incurred Date" AS incurred_date,
            TRY_CAST(EXTRACT(YEAR FROM TRY_STRPTIME(CAST("Incurred Date" AS VARCHAR), '%Y%m%d')) AS INTEGER) AS event_year,
            CASE
                WHEN "Member Age DOS" IS NULL THEN NULL
                WHEN TRIM(CAST("Member Age DOS" AS VARCHAR)) = '' THEN NULL
                WHEN UPPER(TRIM(CAST("Member Age DOS" AS VARCHAR))) = 'NULL' THEN NULL
                ELSE TRY_CAST(TRIM(CAST("Member Age DOS" AS VARCHAR)) AS INTEGER)
            END AS member_age_dos,
            NULLIF(TRIM("Member Gender"), '') AS member_gender,
            NULLIF(TRIM("Member Race"), '') AS member_race,
            NULLIF(TRIM("Member Zip Code DOS"), '') AS member_zip_code_dos,
            NULLIF(TRIM("Member County DOS"), '') AS member_county_dos,
            NULLIF(TRIM("Payer Type"), '') AS payer_type,
            "CCHG Label" AS cchg_label,
            "CCHG Grouping" AS cchg_grouping,
            "HCG Setting" AS hcg_setting,
            "HCG Line" AS hcg_line,
            "HCG Detail" AS hcg_detail,
            "Place of Service" AS place_of_service,
            "Admit Type" AS admit_type,
            "Primary ICD Diagnosis Code" AS primary_icd_diagnosis_code,
            "Primary ICD Rollup" AS primary_icd_rollup,
            "Primary ICD CCS Level 1" AS primary_icd_ccs_level_1,
            "Primary ICD CCS Level 2" AS primary_icd_ccs_level_2,
            "Primary ICD CCS Level 3" AS primary_icd_ccs_level_3,
            {two_code}   AS two_icd_diagnosis_code,
            {two_rollup} AS two_icd_rollup,
            {three_code} AS three_icd_diagnosis_code,
            {three_rollup} AS three_icd_rollup,
            {four_code}  AS four_icd_diagnosis_code,
            {four_rollup} AS four_icd_rollup,
            {five_code}  AS five_icd_diagnosis_code,
            {five_rollup} AS five_icd_rollup,
            {six_code}   AS six_icd_diagnosis_code,
            {six_rollup} AS six_icd_rollup,
            {seven_code} AS seven_icd_diagnosis_code,
            {seven_rollup} AS seven_icd_rollup,
            {eight_code} AS eight_icd_diagnosis_code,
            {eight_rollup} AS eight_icd_rollup,
            {nine_code}  AS nine_icd_diagnosis_code,
            {nine_rollup} AS nine_icd_rollup,
            {ten_code}   AS ten_icd_diagnosis_code,
            {ten_rollup} AS ten_icd_diagnosis_rollup,
            "Procedure Code" AS procedure_code,
            "Procedure Name" AS procedure_name,
            "Procedure Family 1" AS procedure_family_1,
            "Procedure Family 2" AS procedure_family_2,
            "Procedure Family 3" AS procedure_family_3,
            {two_proc}   AS two_icd_procedure_code,
            {three_proc} AS three_icd_procedure_code,
            {four_proc}  AS four_icd_procedure_code,
            {five_proc}  AS five_icd_procedure_code,
            {six_proc}   AS six_icd_procedure_code,
            {seven_proc} AS seven_icd_procedure_code,
            {eight_proc} AS eight_icd_procedure_code,
            {nine_proc}  AS nine_icd_procedure_code,
            {ten_proc}   AS ten_icd_procedure_code,
            "CPT Mod 1 Code" AS cpt_mod_1_code,
            "CPT Mod 2 Code" AS cpt_mod_2_code,
            "Billing Provider Name" AS billing_provider_name,
            "Billing Provider ZIP" AS billing_provider_zip,
            "Billing Provider County" AS billing_provider_county,
            "Billing Provider State" AS billing_provider_state,
            "Service Provider Name" AS service_provider_name,
            "Service Provider ZIP" AS service_provider_zip,
            "Service Provider County" AS service_provider_county,
            "Service Provider State" AS service_provider_state
        FROM read_parquet('{medical_input}', union_by_name=true)
        WHERE "MI Person Key" IS NOT NULL
          AND "Incurred Date" IS NOT NULL
          AND regexp_matches(CAST("Incurred Date" AS VARCHAR), '^[0-9]{{8}}$')
          AND TRY_STRPTIME(CAST("Incurred Date" AS VARCHAR), '%Y%m%d') IS NOT NULL
    """)
    
    normalized_medical_count = duckdb_conn.sql("SELECT COUNT(*) FROM medical_normalized").fetchone()[0]
    normalized_medical_patients = duckdb_conn.sql("SELECT COUNT(DISTINCT mi_person_key) FROM medical_normalized").fetchone()[0]
    log_data_loss("After medical normalization", initial_medical_count, normalized_medical_count, 
                  logger, initial_medical_patients, normalized_medical_patients)
    
    # Save checkpoint after medical normalization
    save_logs_checkpoint(log_buffer, "global_imputation", "global", "all", "step2_medical_normalized", logger=logger)
    
    # Step 3: Identify missing demographics globally
    logger.info("📊 Step 3: Identifying missing demographics globally...")
    
    # Pharmacy missing demographics
    duckdb_conn.sql("""
    CREATE OR REPLACE VIEW pharmacy_missing AS
    SELECT DISTINCT mi_person_key, event_year,
           CASE WHEN member_age_dos IS NULL OR member_age_dos > 114 THEN 1 ELSE 0 END AS missing_age,
           CASE WHEN member_gender IS NULL THEN 1 ELSE 0 END AS missing_gender,
           CASE WHEN member_race IS NULL THEN 1 ELSE 0 END AS missing_race,
           CASE WHEN member_zip_code_dos IS NULL THEN 1 ELSE 0 END AS missing_zip,
           CASE WHEN member_county_dos IS NULL THEN 1 ELSE 0 END AS missing_county,
           CASE WHEN payer_type IS NULL THEN 1 ELSE 0 END AS missing_payer
    FROM pharmacy_normalized
    WHERE event_year IS NOT NULL
    """)
    
    # Medical missing demographics  
    duckdb_conn.sql("""
    CREATE OR REPLACE VIEW medical_missing AS
    SELECT DISTINCT mi_person_key, event_year,
           CASE WHEN member_age_dos IS NULL OR member_age_dos > 114 THEN 1 ELSE 0 END AS missing_age,
           CASE WHEN member_gender IS NULL THEN 1 ELSE 0 END AS missing_gender,
           CASE WHEN member_race IS NULL THEN 1 ELSE 0 END AS missing_race,
           CASE WHEN member_zip_code_dos IS NULL THEN 1 ELSE 0 END AS missing_zip,
           CASE WHEN member_county_dos IS NULL THEN 1 ELSE 0 END AS missing_county,
           CASE WHEN payer_type IS NULL THEN 1 ELSE 0 END AS missing_payer
    FROM medical_normalized
    WHERE event_year IS NOT NULL
    """)
    
    pharmacy_missing_count = duckdb_conn.sql("SELECT COUNT(*) FROM pharmacy_missing").fetchone()[0]
    medical_missing_count = duckdb_conn.sql("SELECT COUNT(*) FROM medical_missing").fetchone()[0]
    
    logger.info(f"📊 Pharmacy records needing imputation: {pharmacy_missing_count:,}")
    logger.info(f"📊 Medical records needing imputation: {medical_missing_count:,}")
    
    # Save checkpoint after missing demographics identification
    save_logs_checkpoint(log_buffer, "global_imputation", "global", "all", "step3_missing_identified", logger=logger)
    
    # Step 4: Create aggregated demographics by person/year
    logger.info("📊 Step 4: Creating aggregated demographics by person/year...")
    
    # Pharmacy aggregated demographics
    duckdb_conn.sql("""
    CREATE OR REPLACE VIEW pharmacy_demographics AS
    SELECT 
        mi_person_key, 
        event_year,
        ANY_VALUE(member_age_dos) AS ph_age,
        ANY_VALUE(member_gender) AS ph_gender,
        ANY_VALUE(member_race) AS ph_race,
        ANY_VALUE(member_zip_code_dos) AS ph_zip,
        ANY_VALUE(member_county_dos) AS ph_county,
        ANY_VALUE(payer_type) AS ph_payer,
        COUNT(*) AS ph_claim_count
    FROM pharmacy_normalized
    WHERE event_year IS NOT NULL
    GROUP BY mi_person_key, event_year
    """)
    
    # Medical aggregated demographics
    duckdb_conn.sql("""
    CREATE OR REPLACE VIEW medical_demographics AS
    SELECT 
        mi_person_key, 
        event_year,
        ANY_VALUE(member_age_dos) AS med_age,
        ANY_VALUE(member_gender) AS med_gender,
        ANY_VALUE(member_race) AS med_race,
        ANY_VALUE(member_zip_code_dos) AS med_zip,
        ANY_VALUE(member_county_dos) AS med_county,
        ANY_VALUE(payer_type) AS med_payer,
        COUNT(*) AS med_claim_count
    FROM medical_normalized
    WHERE event_year IS NOT NULL
    GROUP BY mi_person_key, event_year
    """)
    
    # Save checkpoint after demographics aggregation
    save_logs_checkpoint(log_buffer, "global_imputation", "global", "all", "step4_demographics_aggregated", logger=logger)
    
    # Step 5: Run global bidirectional imputation
    logger.info("📊 Step 5: Running global bidirectional imputation...")
    
    # Pharmacy → Medical imputation
    duckdb_conn.sql(f"""
    CREATE OR REPLACE VIEW pharmacy_imputed AS
    WITH age_candidates AS (
        SELECT p.mi_person_key, p.event_year, m.med_age,
               ROW_NUMBER() OVER (PARTITION BY p.mi_person_key, p.event_year ORDER BY ABS(s.ofs), m.med_claim_count DESC) AS age_rank
        FROM pharmacy_demographics p
        JOIN range(-{lookahead_years}, {lookahead_years}+1) AS s(ofs) ON TRUE
        LEFT JOIN medical_demographics m ON m.mi_person_key = p.mi_person_key 
            AND m.event_year = p.event_year + s.ofs
        WHERE p.ph_age IS NULL OR p.ph_age > 114
          AND m.med_age IS NOT NULL AND m.med_age <= 114
    ),
    demo_candidates AS (
        SELECT p.mi_person_key, p.event_year, m.med_gender, m.med_race, m.med_zip, m.med_county, m.med_payer,
               ROW_NUMBER() OVER (PARTITION BY p.mi_person_key, p.event_year ORDER BY ABS(s.ofs), m.med_claim_count DESC) AS demo_rank
        FROM pharmacy_demographics p
        JOIN range(-{lookahead_years}, {lookahead_years}+1) AS s(ofs) ON TRUE
        LEFT JOIN medical_demographics m ON m.mi_person_key = p.mi_person_key 
            AND m.event_year = p.event_year + s.ofs
        WHERE (p.ph_gender IS NULL OR p.ph_race IS NULL OR p.ph_zip IS NULL OR p.ph_county IS NULL OR p.ph_payer IS NULL)
          AND (m.med_gender IS NOT NULL OR m.med_race IS NOT NULL OR m.med_zip IS NOT NULL OR m.med_county IS NOT NULL OR m.med_payer IS NOT NULL)
    )
    SELECT 
        pn.*,
        CASE 
            WHEN pd.ph_age > 114 THEN ac.med_age
            ELSE COALESCE(pd.ph_age, ac.med_age)
        END AS age_imputed,
        NULLIF(COALESCE(pd.ph_gender, dc.med_gender), '') AS gender_imputed,
        NULLIF(COALESCE(pd.ph_race, dc.med_race), '') AS race_imputed,
        NULLIF(COALESCE(pd.ph_zip, dc.med_zip), '') AS zip_imputed,
        NULLIF(COALESCE(pd.ph_county, dc.med_county), '') AS county_imputed,
        NULLIF(COALESCE(pd.ph_payer, dc.med_payer), '') AS payer_imputed,
        CASE 
            WHEN pd.ph_age > 114 AND ac.med_age IS NOT NULL THEN 'imputed_from_medical'
            WHEN pd.ph_age IS NULL AND ac.med_age IS NOT NULL THEN 'imputed_from_medical'
            ELSE 'original'
        END AS age_source,
        CASE WHEN pd.ph_gender IS NULL AND dc.med_gender IS NOT NULL THEN 'imputed_from_medical' ELSE 'original' END AS gender_source,
        CASE WHEN pd.ph_race IS NULL AND dc.med_race IS NOT NULL THEN 'imputed_from_medical' ELSE 'original' END AS race_source,
        CASE WHEN pd.ph_zip IS NULL AND dc.med_zip IS NOT NULL THEN 'imputed_from_medical' ELSE 'original' END AS zip_source,
        CASE WHEN pd.ph_county IS NULL AND dc.med_county IS NOT NULL THEN 'imputed_from_medical' ELSE 'original' END AS county_source,
        CASE WHEN pd.ph_payer IS NULL AND dc.med_payer IS NOT NULL THEN 'imputed_from_medical' ELSE 'original' END AS payer_source
    FROM pharmacy_normalized pn
    LEFT JOIN pharmacy_demographics pd ON pn.mi_person_key = pd.mi_person_key AND pn.event_year = pd.event_year
    LEFT JOIN age_candidates ac ON pn.mi_person_key = ac.mi_person_key AND pn.event_year = ac.event_year AND ac.age_rank = 1
    LEFT JOIN demo_candidates dc ON pn.mi_person_key = dc.mi_person_key AND pn.event_year = dc.event_year AND dc.demo_rank = 1
    """)
    
    # Medical → Pharmacy imputation
    duckdb_conn.sql(f"""
    CREATE OR REPLACE VIEW medical_imputed AS
    WITH age_candidates AS (
        SELECT m.mi_person_key, m.event_year, p.ph_age,
               ROW_NUMBER() OVER (PARTITION BY m.mi_person_key, m.event_year ORDER BY ABS(s.ofs), p.ph_claim_count DESC) AS age_rank
        FROM medical_demographics m
        JOIN range(-{lookahead_years}, {lookahead_years}+1) AS s(ofs) ON TRUE
        LEFT JOIN pharmacy_demographics p ON p.mi_person_key = m.mi_person_key 
            AND p.event_year = m.event_year + s.ofs
        WHERE m.med_age IS NULL OR m.med_age > 114
          AND p.ph_age IS NOT NULL AND p.ph_age <= 114
    ),
    demo_candidates AS (
        SELECT m.mi_person_key, m.event_year, p.ph_gender, p.ph_race, p.ph_zip, p.ph_county, p.ph_payer,
               ROW_NUMBER() OVER (PARTITION BY m.mi_person_key, m.event_year ORDER BY ABS(s.ofs), p.ph_claim_count DESC) AS demo_rank
        FROM medical_demographics m
        JOIN range(-{lookahead_years}, {lookahead_years}+1) AS s(ofs) ON TRUE
        LEFT JOIN pharmacy_demographics p ON p.mi_person_key = m.mi_person_key 
            AND p.event_year = m.event_year + s.ofs
        WHERE (m.med_gender IS NULL OR m.med_race IS NULL OR m.med_zip IS NULL OR m.med_county IS NULL OR m.med_payer IS NULL)
          AND (p.ph_gender IS NOT NULL OR p.ph_race IS NOT NULL OR p.ph_zip IS NOT NULL OR p.ph_county IS NOT NULL OR p.ph_payer IS NOT NULL)
    )
    SELECT 
        mn.*,
        CASE 
            WHEN md.med_age > 114 THEN ac.ph_age
            ELSE COALESCE(md.med_age, ac.ph_age)
        END AS age_imputed,
        NULLIF(COALESCE(md.med_gender, dc.ph_gender), '') AS gender_imputed,
        NULLIF(COALESCE(md.med_race, dc.ph_race), '') AS race_imputed,
        NULLIF(COALESCE(md.med_zip, dc.ph_zip), '') AS zip_imputed,
        NULLIF(COALESCE(md.med_county, dc.ph_county), '') AS county_imputed,
        NULLIF(COALESCE(md.med_payer, dc.ph_payer), '') AS payer_imputed,
        CASE 
            WHEN md.med_age > 114 AND ac.ph_age IS NOT NULL THEN 'imputed_from_pharmacy'
            WHEN md.med_age IS NULL AND ac.ph_age IS NOT NULL THEN 'imputed_from_pharmacy'
            ELSE 'original'
        END AS age_source,
        CASE WHEN md.med_gender IS NULL AND dc.ph_gender IS NOT NULL THEN 'imputed_from_pharmacy' ELSE 'original' END AS gender_source,
        CASE WHEN md.med_race IS NULL AND dc.ph_race IS NOT NULL THEN 'imputed_from_pharmacy' ELSE 'original' END AS race_source,
        CASE WHEN md.med_zip IS NULL AND dc.ph_zip IS NOT NULL THEN 'imputed_from_pharmacy' ELSE 'original' END AS zip_source,
        CASE WHEN md.med_county IS NULL AND dc.ph_county IS NOT NULL THEN 'imputed_from_pharmacy' ELSE 'original' END AS county_source,
        CASE WHEN md.med_payer IS NULL AND dc.ph_payer IS NOT NULL THEN 'imputed_from_pharmacy' ELSE 'original' END AS payer_source
    FROM medical_normalized mn
    LEFT JOIN medical_demographics md ON mn.mi_person_key = md.mi_person_key AND mn.event_year = md.event_year
    LEFT JOIN age_candidates ac ON mn.mi_person_key = ac.mi_person_key AND mn.event_year = ac.event_year AND ac.age_rank = 1
    LEFT JOIN demo_candidates dc ON mn.mi_person_key = dc.mi_person_key AND mn.event_year = dc.event_year AND dc.demo_rank = 1
    """)
    
    # Save checkpoint after imputation
    save_logs_checkpoint(log_buffer, "global_imputation", "global", "all", "step5_imputation_complete", logger=logger)
    
    # Step 6: Save results (demographics and imputed pharmacy data)
    logger.info("📊 Step 6: Saving imputed demographics and pharmacy data to silver tier...")
    
    # Save imputed pharmacy data with partitions
    pharmacy_partitioned_path = f"{silver_paths['base_path']}/pharmacy_partitioned"
    logger.info(f"📊 Saving imputed pharmacy data with partitions to: {pharmacy_partitioned_path}")
    
    # Create view with partition columns - PARTITION_BY needs them, but WRITE_PARTITION_COLUMNS FALSE
    # prevents them from being written as data columns (avoiding Glue schema duplicates)
    duckdb_conn.sql("""
        CREATE OR REPLACE VIEW pharmacy_partitioned_temp AS
            SELECT 
                mi_person_key,
                incurred_date,
                drug_name,
                total_utilization,
                age_imputed,
                gender_imputed,
                race_imputed,
                zip_imputed,
                county_imputed,
                payer_imputed,
                age_source,
                gender_source,
                race_source,
                zip_source,
                county_source,
                payer_source,
                CASE
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 0  AND 12  THEN '0-12'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 13 AND 24  THEN '13-24'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 25 AND 44  THEN '25-44'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 45 AND 54  THEN '45-54'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 55 AND 64  THEN '55-64'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 65 AND 74  THEN '65-74'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 75 AND 84  THEN '75-84'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 85 AND 94  THEN '85-94'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 95 AND 114 THEN '95-114'
                    ELSE 'Other'
                END AS age_band,
                TRY_CAST(EXTRACT(YEAR FROM TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d')) AS INTEGER) AS event_year
            FROM pharmacy_imputed
            WHERE incurred_date IS NOT NULL
              AND LENGTH(CAST(incurred_date AS VARCHAR)) = 8
              AND regexp_matches(CAST(incurred_date AS VARCHAR), '^[0-9]{{8}}$')
              AND TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') IS NOT NULL
              AND TRY_CAST(EXTRACT(YEAR FROM TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d')) AS INTEGER) BETWEEN 2016 AND 2020
              AND age_imputed IS NOT NULL
    """)
    
    # Use view directly - WRITE_PARTITION_COLUMNS FALSE excludes partition columns from parquet data
    # but PARTITION_BY can still access them from the view
    duckdb_conn.sql(f"""
        COPY pharmacy_partitioned_temp
        TO '{pharmacy_partitioned_path}'
        (FORMAT PARQUET, 
        PARTITION_BY (age_band, event_year), 
        WRITE_PARTITION_COLUMNS FALSE,
        OVERWRITE_OR_IGNORE true)
    """)
    
    logger.info("✅ Imputed pharmacy data saved with partitions successfully")
    
    # Save checkpoint after pharmacy data saved
    save_logs_checkpoint(log_buffer, "global_imputation", "global", "all", "step6a_pharmacy_saved", logger=logger)
    
    # Save imputed medical data with partitions
    medical_partitioned_path = f"{silver_paths['base_path']}/medical_partitioned"
    logger.info(f"📊 Saving imputed medical data with partitions to: {medical_partitioned_path}")
    
    # Create view with partition columns - PARTITION_BY needs them, but WRITE_PARTITION_COLUMNS FALSE
    # prevents them from being written as data columns (avoiding Glue schema duplicates)
    duckdb_conn.sql("""
        CREATE OR REPLACE VIEW medical_partitioned_temp AS
            SELECT 
                mi_person_key,
                claim_id,
                incurred_date,
                cchg_label,
                cchg_grouping,
                hcg_setting,
                hcg_line,
                hcg_detail,
                place_of_service,
                admit_type,
                primary_icd_diagnosis_code,
                primary_icd_rollup,
                primary_icd_ccs_level_1,
                primary_icd_ccs_level_2,
                primary_icd_ccs_level_3,
                two_icd_diagnosis_code,
                two_icd_rollup,
                three_icd_diagnosis_code,
                three_icd_rollup,
                four_icd_diagnosis_code,
                four_icd_rollup,
                five_icd_diagnosis_code,
                five_icd_rollup,
                six_icd_diagnosis_code,
                six_icd_rollup,
                seven_icd_diagnosis_code,
                seven_icd_rollup,
                eight_icd_diagnosis_code,
                eight_icd_rollup,
                nine_icd_diagnosis_code,
                nine_icd_rollup,
                ten_icd_diagnosis_code,
                ten_icd_diagnosis_rollup,
                procedure_code,
                procedure_name,
                procedure_family_1,
                procedure_family_2,
                procedure_family_3,
                two_icd_procedure_code,
                three_icd_procedure_code,
                four_icd_procedure_code,
                five_icd_procedure_code,
                six_icd_procedure_code,
                seven_icd_procedure_code,
                eight_icd_procedure_code,
                nine_icd_procedure_code,
                ten_icd_procedure_code,
                cpt_mod_1_code,
                cpt_mod_2_code,
                billing_provider_name,
                billing_provider_zip,
                billing_provider_county,
                billing_provider_state,
                service_provider_name,
                service_provider_zip,
                service_provider_county,
                service_provider_state,
                age_imputed,
                gender_imputed,
                race_imputed,
                zip_imputed,
                county_imputed,
                payer_imputed,
                age_source,
                gender_source,
                race_source,
                zip_source,
                county_source,
                payer_source,
                CASE
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 0  AND 12  THEN '0-12'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 13 AND 24  THEN '13-24'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 25 AND 44  THEN '25-44'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 45 AND 54  THEN '45-54'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 55 AND 64  THEN '55-64'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 65 AND 74  THEN '65-74'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 75 AND 84  THEN '75-84'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 85 AND 94  THEN '85-94'
                    WHEN TRY_CAST(age_imputed AS INTEGER) BETWEEN 95 AND 114 THEN '95-114'
                    ELSE 'Other'
                END AS age_band,
                TRY_CAST(EXTRACT(YEAR FROM TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d')) AS INTEGER) AS event_year
            FROM medical_imputed
            WHERE incurred_date IS NOT NULL
              AND LENGTH(CAST(incurred_date AS VARCHAR)) = 8
              AND regexp_matches(CAST(incurred_date AS VARCHAR), '^[0-9]{{8}}$')
              AND TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') IS NOT NULL
              AND TRY_CAST(EXTRACT(YEAR FROM TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d')) AS INTEGER) BETWEEN 2016 AND 2020
              AND age_imputed IS NOT NULL
    """)
    
    # Use view directly - WRITE_PARTITION_COLUMNS FALSE excludes partition columns from parquet data
    # but PARTITION_BY can still access them from the view
    duckdb_conn.sql(f"""
        COPY medical_partitioned_temp
        TO '{medical_partitioned_path}'
        (FORMAT PARQUET, 
        PARTITION_BY (age_band, event_year), 
        WRITE_PARTITION_COLUMNS FALSE,
        OVERWRITE_OR_IGNORE true)
    """)
    
    logger.info("✅ Imputed medical data saved with partitions successfully")
    
    # Save checkpoint after medical data saved
    save_logs_checkpoint(log_buffer, "global_imputation", "global", "all", "step6b_medical_saved", logger=logger)
    
    # Save demographics lookup table (only if requested)
    if create_demographics_lookup:
        demographics_output_path = silver_paths["demographics_lookup"]
        logger.info("📊 Creating demographics lookup for records that needed imputation...")
        
        # Only include records that actually needed imputation (not all records)
        duckdb_conn.sql("""
            CREATE OR REPLACE TABLE demographics_lookup_temp AS
            SELECT DISTINCT
                mi_person_key,
                event_year,
                age_imputed,
                gender_imputed,
                race_imputed,
                zip_imputed,
                county_imputed,
                payer_imputed,
                age_source,
                gender_source,
                race_source,
                zip_source,
                county_source,
                payer_source
            FROM (
                -- Only pharmacy records that needed imputation
                SELECT 
                    pi.mi_person_key,
                    pi.event_year,
                    pi.age_imputed,
                    pi.gender_imputed,
                    pi.race_imputed,
                    pi.zip_imputed,
                    pi.county_imputed,
                    pi.payer_imputed,
                    pi.age_source,
                    pi.gender_source,
                    pi.race_source,
                    pi.zip_source,
                    pi.county_source,
                    pi.payer_source
                FROM pharmacy_imputed pi
                INNER JOIN pharmacy_missing pm ON pi.mi_person_key = pm.mi_person_key AND pi.event_year = pm.event_year
                WHERE pm.missing_age = 1 OR pm.missing_gender = 1 OR pm.missing_race = 1 
                   OR pm.missing_zip = 1 OR pm.missing_county = 1 OR pm.missing_payer = 1
                
                UNION ALL
                
                -- Only medical records that needed imputation
                SELECT 
                    mi.mi_person_key,
                    mi.event_year,
                    mi.age_imputed,
                    mi.gender_imputed,
                    mi.race_imputed,
                    mi.zip_imputed,
                    mi.county_imputed,
                    mi.payer_imputed,
                    mi.age_source,
                    mi.gender_source,
                    mi.race_source,
                    mi.zip_source,
                    mi.county_source,
                    mi.payer_source
                FROM medical_imputed mi
                INNER JOIN medical_missing mm ON mi.mi_person_key = mm.mi_person_key AND mi.event_year = mm.event_year
                WHERE mm.missing_age = 1 OR mm.missing_gender = 1 OR mm.missing_race = 1 
                   OR mm.missing_zip = 1 OR mm.missing_county = 1 OR mm.missing_payer = 1
            )
        """)
        
        # Create the final view
        duckdb_conn.sql("""
            CREATE OR REPLACE VIEW demographics_lookup AS
            SELECT * FROM demographics_lookup_temp
        """)
        
        # Get statistics before dropping the temporary table
        demographics_count = duckdb_conn.sql("SELECT COUNT(*) FROM demographics_lookup").fetchone()[0]
        demographics_patients = duckdb_conn.sql("SELECT COUNT(DISTINCT mi_person_key) FROM demographics_lookup").fetchone()[0]
        
        # Copy to S3 with memory management
        logger.info("📊 Copying demographics lookup to S3...")
        duckdb_conn.sql(f"""
            COPY demographics_lookup 
            TO '{demographics_output_path}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE true)
        """)
        
        # Clean up temporary table
        duckdb_conn.sql("DROP TABLE demographics_lookup_temp")
        
        # Compare with total records to show efficiency
        total_pharmacy_records = duckdb_conn.sql("SELECT COUNT(*) FROM pharmacy_normalized").fetchone()[0]
        total_medical_records = duckdb_conn.sql("SELECT COUNT(*) FROM medical_normalized").fetchone()[0]
        total_records = total_pharmacy_records + total_medical_records
        
        logger.info(f"📊 Memory efficiency: {demographics_count:,} imputed records out of {total_records:,} total records ({demographics_count/total_records*100:.1f}%)")
    else:
        logger.info("📊 Skipping demographics lookup creation (create_demographics_lookup=False)")
        demographics_output_path = None
        demographics_count = 0
        demographics_patients = 0
    
    # Imputation success statistics for pharmacy data
    pharmacy_imputation_stats = duckdb_conn.sql("""
        SELECT 
            COUNT(*) as total_records,
            COUNT(CASE WHEN age_source = 'imputed_from_medical' THEN 1 END) as age_imputed_from_medical,
            COUNT(CASE WHEN gender_source = 'imputed_from_medical' THEN 1 END) as gender_imputed_from_medical,
            COUNT(CASE WHEN race_source = 'imputed_from_medical' THEN 1 END) as race_imputed_from_medical,
            COUNT(CASE WHEN zip_source = 'imputed_from_medical' THEN 1 END) as zip_imputed_from_medical,
            COUNT(CASE WHEN county_source = 'imputed_from_medical' THEN 1 END) as county_imputed_from_medical,
            COUNT(CASE WHEN payer_source = 'imputed_from_medical' THEN 1 END) as payer_imputed_from_medical
        FROM pharmacy_imputed
    """).fetchone()
    
    # Imputation success statistics for medical data
    medical_imputation_stats = duckdb_conn.sql("""
        SELECT 
            COUNT(*) as total_records,
            COUNT(CASE WHEN age_source = 'imputed_from_pharmacy' THEN 1 END) as age_imputed_from_pharmacy,
            COUNT(CASE WHEN gender_source = 'imputed_from_pharmacy' THEN 1 END) as gender_imputed_from_pharmacy,
            COUNT(CASE WHEN race_source = 'imputed_from_pharmacy' THEN 1 END) as race_imputed_from_pharmacy,
            COUNT(CASE WHEN zip_source = 'imputed_from_pharmacy' THEN 1 END) as zip_imputed_from_pharmacy,
            COUNT(CASE WHEN county_source = 'imputed_from_pharmacy' THEN 1 END) as county_imputed_from_pharmacy,
            COUNT(CASE WHEN payer_source = 'imputed_from_pharmacy' THEN 1 END) as payer_imputed_from_pharmacy
        FROM medical_imputed
    """).fetchone()
    
    logger.info("=" * 80)
    logger.info("🎯 GLOBAL IMPUTATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"📊 Demographics lookup records: {demographics_count:,}")
    logger.info(f"📊 Demographics lookup patients: {demographics_patients:,}")
    logger.info("")
    logger.info("🔧 PHARMACY IMPUTATION SUCCESS (Fixed from Medical Data):")
    logger.info(f"   • Total pharmacy records: {pharmacy_imputation_stats[0]:,}")
    logger.info(f"   • Age imputed from medical: {pharmacy_imputation_stats[1]:,} ({pharmacy_imputation_stats[1]/pharmacy_imputation_stats[0]*100:.1f}%)")
    logger.info(f"   • Gender imputed from medical: {pharmacy_imputation_stats[2]:,} ({pharmacy_imputation_stats[2]/pharmacy_imputation_stats[0]*100:.1f}%)")
    logger.info(f"   • Race imputed from medical: {pharmacy_imputation_stats[3]:,} ({pharmacy_imputation_stats[3]/pharmacy_imputation_stats[0]*100:.1f}%)")
    logger.info(f"   • Zip imputed from medical: {pharmacy_imputation_stats[4]:,} ({pharmacy_imputation_stats[4]/pharmacy_imputation_stats[0]*100:.1f}%)")
    logger.info(f"   • County imputed from medical: {pharmacy_imputation_stats[5]:,} ({pharmacy_imputation_stats[5]/pharmacy_imputation_stats[0]*100:.1f}%)")
    logger.info(f"   • Payer imputed from medical: {pharmacy_imputation_stats[6]:,} ({pharmacy_imputation_stats[6]/pharmacy_imputation_stats[0]*100:.1f}%)")
    logger.info("")
    logger.info("🔧 MEDICAL IMPUTATION SUCCESS (Fixed from Pharmacy Data):")
    logger.info(f"   • Total medical records: {medical_imputation_stats[0]:,}")
    logger.info(f"   • Age imputed from pharmacy: {medical_imputation_stats[1]:,} ({medical_imputation_stats[1]/medical_imputation_stats[0]*100:.1f}%)")
    logger.info(f"   • Gender imputed from pharmacy: {medical_imputation_stats[2]:,} ({medical_imputation_stats[2]/medical_imputation_stats[0]*100:.1f}%)")
    logger.info(f"   • Race imputed from pharmacy: {medical_imputation_stats[3]:,} ({medical_imputation_stats[3]/medical_imputation_stats[0]*100:.1f}%)")
    logger.info(f"   • Zip imputed from pharmacy: {medical_imputation_stats[4]:,} ({medical_imputation_stats[4]/medical_imputation_stats[0]*100:.1f}%)")
    logger.info(f"   • County imputed from pharmacy: {medical_imputation_stats[5]:,} ({medical_imputation_stats[5]/medical_imputation_stats[0]*100:.1f}%)")
    logger.info(f"   • Payer imputed from pharmacy: {medical_imputation_stats[6]:,} ({medical_imputation_stats[6]/medical_imputation_stats[0]*100:.1f}%)")
    logger.info("")
    if demographics_output_path:
        logger.info(f"📁 Silver mi_person_key demographics lookup: {demographics_output_path}")
    else:
        logger.info("📁 Silver mi_person_key demographics lookup: (skipped)")
    logger.info(f"📁 Silver base path: {silver_paths['base_path']}")
    
    result = {
        "demographics_output_path": demographics_output_path,
        "base_path": silver_paths["base_path"],
        "demographics_count": demographics_count,
        "demographics_patients": demographics_patients
    }
    
    # Add raw silver results if created
    if raw_results:
        result["raw_silver"] = raw_results
        logger.info("")
        logger.info("📁 Raw Silver Datasets Created:")
        logger.info(f"   • Pharmacy raw: {raw_results['pharmacy_raw_path']} ({raw_results['pharmacy_raw_count']:,} records)")
        logger.info(f"   • Medical raw: {raw_results['medical_raw_path']} ({raw_results['medical_raw_count']:,} records)")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Global Demographic Imputation Preprocessing")
    parser.add_argument("--pharmacy-input", 
                       default=f"s3://{S3_BUCKET}/bronze/pharmacy/*.parquet",
                       help="S3 path to pharmacy parquet files (defaults to bronze)")
    parser.add_argument("--medical-input", 
                       default=f"s3://{S3_BUCKET}/bronze/medical/*.parquet",
                       help="S3 path to medical parquet files (defaults to bronze)")
    parser.add_argument("--output-root", 
                       default=f"s3://{S3_BUCKET}/silver/imputed",
                       help="S3 root path for output files (will be placed in silver tier)")
    parser.add_argument("--lookahead-years", type=int, default=5, help="Years to look ahead/behind for imputation")
    # Removed --threads and --mem-gb arguments - DuckDB auto-detects optimal settings
    parser.add_argument("--tmp-dir", help="Temporary directory for DuckDB")
    parser.add_argument("--no-demographics-lookup", action="store_true", help="Skip creating demographics lookup table (optimized mode)")
    parser.add_argument("--create-raw-silver", action="store_true", help="Also create raw silver datasets with all original columns (pharmacy_raw, medical_raw)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--aggregate-root", default="s3://pgxdatalake/pgx_pipeline/", help="S3 root for aggregated run summaries (for BI)")
    
    args = parser.parse_args()
    
    # Setup logging
    run_id = time.strftime("%Y%m%d-%H%M%S")
    logger, log_buffer = setup_logging("global_imputation", "global", run_id)
    
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    logger.info("🚀 Starting Global Demographic Imputation")
    run_started = time.time()
    agg = {"tx": "glimp", "run_id": run_id, "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "running"}
    logger.info(f"📊 Pharmacy input: {args.pharmacy_input}")
    logger.info(f"📊 Medical input: {args.medical_input}")
    logger.info(f"📊 Output root: {args.output_root}")
    logger.info(f"📊 Lookahead years: {args.lookahead_years}")
    logger.info(f"📊 Create raw silver: {args.create_raw_silver}")

    logger.info("📊 DuckDB will auto-detect optimal memory and thread settings")
    
    try:
        results = run_global_imputation(
            args.pharmacy_input,
            args.medical_input, 
            args.output_root,
            args.lookahead_years,
            args.tmp_dir,
            logger,
            log_buffer,
            create_demographics_lookup=not args.no_demographics_lookup,
            create_raw_silver=args.create_raw_silver
        )
        if results is None:
            results = {}
        
        logger.info("✅ Global imputation completed successfully!")
        
        # Save logs
        save_logs_to_s3(log_buffer, "global_imputation", "global", run_id, "apcd_input_data", logger=logger)
        # Aggregated summary
        try:
            run_finished = time.time()
            agg.update({
                "status": "success",
                "status_code": 0,
                "end_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_finished)),
                "duration_sec": round(run_finished - run_started, 3),
                "output_base": results.get("base_path"),
                "demographics_output": results.get("demographics_output_path"),
            })
            bucket, root = args.aggregate_root.replace("s3://", ""), ""
            if "/" in bucket:
                bucket, root = bucket.split("/", 1)
            key = f"{root.rstrip('/')}/global_imputation/run_id={run_id}/summary.json" if root else f"global_imputation/run_id={run_id}/summary.json"
            boto3.client('s3').put_object(Bucket=bucket, Key=key, Body=json.dumps(agg, indent=2).encode('utf-8'), ContentType='application/json')
            logger.info(f"📊 Aggregated summary saved: s3://{bucket}/{key}")
        except Exception as e:
            logger.warning(f"⚠️ Could not save aggregated summary: {e}")
        
    except Exception as e:
        logger.error(f"❌ Global imputation failed: {e}")
        save_logs_to_s3(log_buffer, "global_imputation", "global", run_id, "apcd_input_data", logger=logger)
        # Aggregated summary (error)
        try:
            run_finished = time.time()
            agg.update({
                "status": "error",
                "status_code": 1,
                "end_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_finished)),
                "duration_sec": round(run_finished - run_started, 3),
                "error": str(e),
            })
            bucket, root = args.aggregate_root.replace("s3://", ""), ""
            if "/" in bucket:
                bucket, root = bucket.split("/", 1)
            key = f"{root.rstrip('/')}/global_imputation/run_id={run_id}/summary.json" if root else f"global_imputation/run_id={run_id}/summary.json"
            boto3.client('s3').put_object(Bucket=bucket, Key=key, Body=json.dumps(agg, indent=2).encode('utf-8'), ContentType='application/json')
            logger.info(f"📊 Aggregated summary saved: s3://{bucket}/{key}")
        except Exception as e2:
            logger.warning(f"⚠️ Could not save aggregated summary: {e2}")
        raise

if __name__ == "__main__":
    main()
