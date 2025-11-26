#!/usr/bin/env python3
"""
Merge corrected data from part_files back into main bronze parquet files.

This script:
1. Validates part file schema matches expected schema
2. Validates data quality (MI Person Key, Incurred Date)
3. Checks for duplicates (by Claim ID) - makes process idempotent
4. Merges validated, non-duplicate rows into bronze main files

Note: Silver preprocessing (demographics enrichment, partitioning) happens in follow-on scripts.
      This script focuses on adding validated data to bronze only.
"""

import argparse
import boto3
import logging
from urllib.parse import urlparse
from typing import List, Tuple
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from helpers_1997_13.logging_utils import setup_logging, save_logs_to_s3
    from helpers_1997_13.duckdb_utils import create_simple_duckdb_connection
except Exception:
    setup_logging = None
    save_logs_to_s3 = None
    create_simple_duckdb_connection = None


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("merge_part_files")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    rest = uri[5:]
    parts = rest.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def s3_exists(bucket: str, key: str) -> bool:
    s3 = boto3.client("s3")
    try:
        resp = s3.head_object(Bucket=bucket, Key=key)
        return resp.get("ContentLength", 0) > 0
    except s3.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "403", "NoSuchKey"):
            return False
        raise


def find_main_file_for_part(part_file: str, bronze_root: str, dataset: str) -> str:
    """
    Find the main parquet file that corresponds to a part file.
    Example: Genomic_Screening_All_RX_Claims_Q12016.part_e76cfc78.parquet
         -> Genomic_Screening_All_RX_Claims_Q12016.parquet
    """
    filename = part_file.split('/')[-1]
    # Remove .part_XXXXX.parquet to get base name
    if '.part_' in filename:
        base_name = filename.split('.part_')[0] + '.parquet'
    else:
        return None
    
    parsed = urlparse(bronze_root)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip('/')
    main_key = f"{prefix.rstrip('/')}/{dataset}/{base_name}" if prefix else f"{dataset}/{base_name}"
    main_uri = f"s3://{bucket}/{main_key}"
    
    return main_uri if s3_exists(bucket, main_key) else None


def get_correct_columns(dataset: str) -> List[str]:
    """
    Get the correct column names from the head file.
    """
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    head_file = os.path.join(base_dir, '1_apcd_input_data', 'apcd', dataset, f'{dataset}_head.txt')
    
    if not os.path.exists(head_file):
        raise FileNotFoundError(f"Head file not found: {head_file}")
    
    with open(head_file, 'r', encoding='utf-8') as f:
        header_line = f.readline().strip()
    
    # Split by pipe delimiter
    columns = [col.strip() for col in header_line.split('|')]
    return columns


def enrich_part_file_with_demographics(part_uri: str, dataset: str, demographics_lookup_uri: str, duckdb_conn, logger: logging.Logger) -> Tuple[bool, str, str]:
    """
    Enrich a part file with demographics lookup BEFORE merging.
    
    This fills in missing demographic columns in the part file itself,
    so when we merge it, we're merging enriched data.
    
    Returns: (success, message, enriched_part_uri)
    """
    try:
        part_bucket, part_key = _parse_s3_uri(part_uri)
        
        logger.info(f"Enriching part file {part_uri} with demographics from {demographics_lookup_uri}")
        
        # Check if demographics lookup exists
        if not s3_exists(*_parse_s3_uri(demographics_lookup_uri)):
            logger.warning(f"Demographics lookup not found at {demographics_lookup_uri} - skipping enrichment")
            return False, "Demographics lookup not found", part_uri
        
        # Get correct column names
        correct_columns = get_correct_columns(dataset)
        
        # Get actual column names from part file (may be wrong)
        schema_query = f"DESCRIBE SELECT * FROM read_parquet('{part_uri}') LIMIT 0"
        part_schema = duckdb_conn.sql(schema_query).fetchall()
        part_actual_cols = [row[0] for row in part_schema]
        
        # Map part file columns (by position) to correct column names, then enrich
        part_col_selects = []
        for i, col in enumerate(correct_columns):
            if i < len(part_actual_cols):
                actual_col = part_actual_cols[i].replace('"', '""')
                if col == "Member Age DOS":
                    part_col_selects.append(f'COALESCE(p."{actual_col}", CAST(d.age_imputed AS VARCHAR)) AS "{col}"')
                elif col == "Member Gender":
                    part_col_selects.append(f'COALESCE(p."{actual_col}", d.gender_imputed) AS "{col}"')
                elif col == "Member Race":
                    part_col_selects.append(f'COALESCE(p."{actual_col}", d.race_imputed) AS "{col}"')
                elif col == "Member Zip Code DOS":
                    part_col_selects.append(f'COALESCE(p."{actual_col}", d.zip_imputed) AS "{col}"')
                elif col == "Member County DOS":
                    part_col_selects.append(f'COALESCE(p."{actual_col}", d.county_imputed) AS "{col}"')
                elif col == "Payer Type":
                    part_col_selects.append(f'COALESCE(p."{actual_col}", d.payer_imputed) AS "{col}"')
                else:
                    part_col_selects.append(f'p."{actual_col}" AS "{col}"')
            else:
                # Column missing in part file, use NULL or imputed value
                if col == "Member Age DOS":
                    part_col_selects.append(f'CAST(d.age_imputed AS VARCHAR) AS "{col}"')
                elif col == "Member Gender":
                    part_col_selects.append(f'd.gender_imputed AS "{col}"')
                elif col == "Member Race":
                    part_col_selects.append(f'd.race_imputed AS "{col}"')
                elif col == "Member Zip Code DOS":
                    part_col_selects.append(f'd.zip_imputed AS "{col}"')
                elif col == "Member County DOS":
                    part_col_selects.append(f'd.county_imputed AS "{col}"')
                elif col == "Payer Type":
                    part_col_selects.append(f'd.payer_imputed AS "{col}"')
                else:
                    part_col_selects.append(f'NULL AS "{col}"')
        
        part_col_list_str = ", ".join(part_col_selects)
        
        # Get critical column names from part file (by position)
        actual_incurred_date_col = part_actual_cols[0] if len(part_actual_cols) > 0 else None
        actual_mi_person_key_col = part_actual_cols[2] if len(part_actual_cols) > 2 else None
        
        if not actual_incurred_date_col or not actual_mi_person_key_col:
            return False, "Part file missing critical columns", part_uri
        
        # Create enriched part file (write to temporary location, then we'll use it for merging)
        # Use a temp URI in the same location
        enriched_part_uri = part_uri.replace('.part_', '.enriched.part_')
        
        enrichment_query = f"""
        COPY (
            SELECT {part_col_list_str}
            FROM read_parquet('{part_uri}') p
            LEFT JOIN read_parquet('{demographics_lookup_uri}') d
                ON CAST(p."{actual_mi_person_key_col}" AS VARCHAR) = CAST(d.mi_person_key AS VARCHAR)
                AND TRY_CAST(EXTRACT(YEAR FROM TRY_STRPTIME(CAST(p."{actual_incurred_date_col}" AS VARCHAR), '%Y%m%d')) AS INTEGER) = d.event_year
            WHERE p."{actual_mi_person_key_col}" IS NOT NULL
              AND p."{actual_incurred_date_col}" IS NOT NULL
              AND LENGTH(CAST(p."{actual_incurred_date_col}" AS VARCHAR)) = 8
              AND regexp_matches(CAST(p."{actual_incurred_date_col}" AS VARCHAR), '^[0-9]{{8}}$')
              AND TRY_STRPTIME(CAST(p."{actual_incurred_date_col}" AS VARCHAR), '%Y%m%d') IS NOT NULL
        ) TO '{enriched_part_uri}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000, OVERWRITE_OR_IGNORE true);
        """
        
        duckdb_conn.sql(enrichment_query)
        
        # Count enrichment stats
        count_query = f"""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(d.mi_person_key) as rows_with_demographics,
            COUNT(CASE WHEN p."{actual_mi_person_key_col}" IS NULL AND d.age_imputed IS NOT NULL THEN 1 END) as age_filled,
            COUNT(CASE WHEN p."{actual_mi_person_key_col}" IS NULL AND d.gender_imputed IS NOT NULL THEN 1 END) as gender_filled
        FROM read_parquet('{part_uri}') p
        LEFT JOIN read_parquet('{demographics_lookup_uri}') d
            ON CAST(p."{actual_mi_person_key_col}" AS VARCHAR) = CAST(d.mi_person_key AS VARCHAR)
            AND TRY_CAST(EXTRACT(YEAR FROM TRY_STRPTIME(CAST(p."{actual_incurred_date_col}" AS VARCHAR), '%Y%m%d')) AS INTEGER) = d.event_year
        WHERE p."{actual_mi_person_key_col}" IS NOT NULL
          AND p."{actual_incurred_date_col}" IS NOT NULL
        """
        
        stats = duckdb_conn.sql(count_query).fetchone()
        total_rows = stats[0] if stats else 0
        rows_with_demographics = stats[1] if stats else 0
        
        logger.info(f"Enriched part file: {total_rows} rows, {rows_with_demographics} matched with demographics")
        
        return True, f"enriched {rows_with_demographics} rows", enriched_part_uri
        
    except Exception as e:
        return False, f"enrichment_error: {str(e)}", part_uri


def write_to_silver(main_uri: str, dataset: str, silver_output_root: str, duckdb_conn, logger: logging.Logger) -> Tuple[bool, str]:
    """
    Write merged/enriched main file to silver layer, partitioned by age_band and event_year.
    
    Derives age_band from "Member Age DOS" and event_year from "Incurred Date",
    then writes to silver partitioned location.
    """
    try:
        logger.info(f"Writing {main_uri} to silver: {silver_output_root}")
        
        # Get correct column names
        correct_columns = get_correct_columns(dataset)
        col_list = ", ".join([f'"{col}"' for col in correct_columns])
        
        # Create view with partition columns derived from data
        # age_band derived from "Member Age DOS"
        # event_year derived from "Incurred Date"
        # Select all columns plus derived partition columns
        col_list_with_partitions = ", ".join([f'"{col}"' for col in correct_columns])
        partition_view_query = f"""
        CREATE OR REPLACE TEMP VIEW silver_partitioned_temp AS
        SELECT 
            {col_list_with_partitions},
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
            TRY_CAST(EXTRACT(YEAR FROM TRY_STRPTIME(CAST("Incurred Date" AS VARCHAR), '%Y%m%d')) AS INTEGER) AS event_year
        FROM read_parquet('{main_uri}')
        WHERE "Incurred Date" IS NOT NULL
          AND "Member Age DOS" IS NOT NULL
          AND LENGTH(CAST("Incurred Date" AS VARCHAR)) = 8
          AND regexp_matches(CAST("Incurred Date" AS VARCHAR), '^[0-9]{{8}}$')
          AND TRY_STRPTIME(CAST("Incurred Date" AS VARCHAR), '%Y%m%d') IS NOT NULL
          AND TRY_CAST(EXTRACT(YEAR FROM TRY_STRPTIME(CAST("Incurred Date" AS VARCHAR), '%Y%m%d')) AS INTEGER) BETWEEN 2016 AND 2020
        """
        
        duckdb_conn.sql(partition_view_query)
        
        # Count rows by partition
        count_query = """
        SELECT 
            age_band,
            event_year,
            COUNT(*) as row_count
        FROM silver_partitioned_temp
        GROUP BY age_band, event_year
        ORDER BY event_year, age_band
        """
        
        partition_counts = duckdb_conn.sql(count_query).fetchall()
        total_rows = sum(row[2] for row in partition_counts)
        
        logger.info(f"Writing {total_rows:,} rows to silver, partitioned by:")
        for age_band, event_year, count in partition_counts:
            logger.info(f"  age_band={age_band}, event_year={event_year}: {count:,} rows")
        
        # Write to silver with partitioning
        write_query = f"""
        COPY silver_partitioned_temp
        TO '{silver_output_root}'
        (FORMAT PARQUET, 
        COMPRESSION ZSTD,
        ROW_GROUP_SIZE 1000000,
        PARTITION_BY (age_band, event_year), 
        WRITE_PARTITION_COLUMNS FALSE,
        OVERWRITE_OR_IGNORE true)
        """
        
        duckdb_conn.sql(write_query)
        
        logger.info(f"Successfully wrote {total_rows:,} rows to silver: {silver_output_root}")
        return True, f"wrote {total_rows:,} rows to {len(partition_counts)} partitions"
        
    except Exception as e:
        return False, f"silver_write_error: {str(e)}"


def enrich_with_demographics(main_uri: str, dataset: str, demographics_lookup_uri: str, duckdb_conn, logger: logging.Logger) -> Tuple[bool, str]:
    """
    Enrich merged data with demographics lookup.
    
    Joins the main file with demographics lookup on mi_person_key and event_year,
    filling in missing demographic columns (age, gender, race, zip, county, payer).
    """
    try:
        main_bucket, main_key = _parse_s3_uri(main_uri)
        
        logger.info(f"Enriching {main_uri} with demographics from {demographics_lookup_uri}")
        
        # Check if demographics lookup exists
        if not s3_exists(*_parse_s3_uri(demographics_lookup_uri)):
            logger.warning(f"Demographics lookup not found at {demographics_lookup_uri} - skipping enrichment")
            return False, "Demographics lookup not found"
        
        # Get correct column names
        correct_columns = get_correct_columns(dataset)
        
        # Map demographics lookup columns to raw data columns
        # Raw data columns (from head files):
        # - "Member Age DOS" (index 10)
        # - "Member Gender" (index 13)
        # - "Member Race" (index 14)
        # - "Member Zip Code DOS" (index 7)
        # - "Member County DOS" (index 8)
        # - "Payer Type" (index 4)
        
        # Demographics lookup columns:
        # - age_imputed -> "Member Age DOS"
        # - gender_imputed -> "Member Gender"
        # - race_imputed -> "Member Race"
        # - zip_imputed -> "Member Zip Code DOS"
        # - county_imputed -> "Member County DOS"
        # - payer_imputed -> "Payer Type"
        
        # Count how many rows will be enriched (before enrichment)
        
        # Count how many rows were enriched
        count_query = """
        SELECT 
            COUNT(*) as total_rows,
            COUNT(d.mi_person_key) as rows_with_demographics,
            COUNT(CASE WHEN m."Member Age DOS" IS NULL AND d.age_imputed IS NOT NULL THEN 1 END) as age_filled,
            COUNT(CASE WHEN m."Member Gender" IS NULL AND d.gender_imputed IS NOT NULL THEN 1 END) as gender_filled,
            COUNT(CASE WHEN m."Member Race" IS NULL AND d.race_imputed IS NOT NULL THEN 1 END) as race_filled,
            COUNT(CASE WHEN m."Member Zip Code DOS" IS NULL AND d.zip_imputed IS NOT NULL THEN 1 END) as zip_filled,
            COUNT(CASE WHEN m."Member County DOS" IS NULL AND d.county_imputed IS NOT NULL THEN 1 END) as county_filled,
            COUNT(CASE WHEN m."Payer Type" IS NULL AND d.payer_imputed IS NOT NULL THEN 1 END) as payer_filled
        FROM read_parquet('{main_uri}') m
        LEFT JOIN read_parquet('{demographics_lookup_uri}') d
            ON CAST(m."MI Person Key" AS VARCHAR) = CAST(d.mi_person_key AS VARCHAR)
            AND TRY_CAST(EXTRACT(YEAR FROM TRY_STRPTIME(CAST(m."Incurred Date" AS VARCHAR), '%Y%m%d')) AS INTEGER) = d.event_year
        """.replace('{main_uri}', main_uri).replace('{demographics_lookup_uri}', demographics_lookup_uri)
        
        stats = duckdb_conn.sql(count_query).fetchone()
        total_rows = stats[0] if stats else 0
        rows_with_demographics = stats[1] if stats else 0
        age_filled = stats[2] if stats else 0
        gender_filled = stats[3] if stats else 0
        race_filled = stats[4] if stats else 0
        zip_filled = stats[5] if stats else 0
        county_filled = stats[6] if stats else 0
        payer_filled = stats[7] if stats else 0
        
        logger.info(f"Demographics enrichment stats:")
        logger.info(f"  Total rows: {total_rows:,}")
        logger.info(f"  Rows matched with demographics: {rows_with_demographics:,}")
        logger.info(f"  Age filled: {age_filled:,}")
        logger.info(f"  Gender filled: {gender_filled:,}")
        logger.info(f"  Race filled: {race_filled:,}")
        logger.info(f"  Zip filled: {zip_filled:,}")
        logger.info(f"  County filled: {county_filled:,}")
        logger.info(f"  Payer filled: {payer_filled:,}")
        
        # Write enriched data back to main file
        # Build SELECT list with COALESCE for demographic columns
        enriched_col_list = []
        for col in correct_columns:
            if col == "Member Age DOS":
                # age_imputed is INTEGER, need to cast to VARCHAR to match original
                enriched_col_list.append('COALESCE(m."Member Age DOS", CAST(d.age_imputed AS VARCHAR)) AS "Member Age DOS"')
            elif col == "Member Gender":
                enriched_col_list.append('COALESCE(m."Member Gender", d.gender_imputed) AS "Member Gender"')
            elif col == "Member Race":
                enriched_col_list.append('COALESCE(m."Member Race", d.race_imputed) AS "Member Race"')
            elif col == "Member Zip Code DOS":
                enriched_col_list.append('COALESCE(m."Member Zip Code DOS", d.zip_imputed) AS "Member Zip Code DOS"')
            elif col == "Member County DOS":
                enriched_col_list.append('COALESCE(m."Member County DOS", d.county_imputed) AS "Member County DOS"')
            elif col == "Payer Type":
                enriched_col_list.append('COALESCE(m."Payer Type", d.payer_imputed) AS "Payer Type"')
            else:
                enriched_col_list.append(f'm."{col}"')
        
        enriched_col_list_str = ", ".join(enriched_col_list)
        
        write_query = f"""
        COPY (
            SELECT {enriched_col_list_str}
            FROM read_parquet('{main_uri}') m
            LEFT JOIN read_parquet('{demographics_lookup_uri}') d
                ON CAST(m."MI Person Key" AS VARCHAR) = CAST(d.mi_person_key AS VARCHAR)
                AND TRY_CAST(EXTRACT(YEAR FROM TRY_STRPTIME(CAST(m."Incurred Date" AS VARCHAR), '%Y%m%d')) AS INTEGER) = d.event_year
        ) TO '{main_uri}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000, OVERWRITE_OR_IGNORE true);
        """
        
        duckdb_conn.sql(write_query)
        
        # Verify enrichment succeeded
        if not s3_exists(main_bucket, main_key):
            return False, "Main file missing after enrichment"
        
        logger.info(f"Successfully enriched {main_uri} with demographics")
        return True, f"enriched ({age_filled} age, {gender_filled} gender, {race_filled} race, {zip_filled} zip, {county_filled} county, {payer_filled} payer)"
        
    except Exception as e:
        return False, f"enrichment_error: {str(e)}"


def merge_part_to_main(part_uri: str, main_uri: str, dataset: str, duckdb_conn, logger: logging.Logger) -> Tuple[bool, str]:
    """
    Merge a part file into its corresponding main file.
    
    The part file is a parquet file but may have incorrect column names (data values as column names).
    We read the part file's actual column names and map them positionally to correct column names.
    
    CRITICAL VALIDATION: Only merges rows that have valid MI Person Key and Incurred Date.
    """
    try:
        part_bucket, part_key = _parse_s3_uri(part_uri)
        main_bucket, main_key = _parse_s3_uri(main_uri)
        
        logger.info(f"Merging {part_uri} -> {main_uri}")
        
        # Get correct column names
        correct_columns = get_correct_columns(dataset)
        n_cols = len(correct_columns)
        
        # Critical columns that must be present and valid
        # These are always at positions 0 (Incurred Date) and 2 (MI Person Key) in both schemas
        CRITICAL_COL_INDICES = {
            "incurred_date": 0,  # "Incurred Date"
            "mi_person_key": 2   # "MI Person Key"
        }
        CRITICAL_COL_NAMES = {
            "incurred_date": "Incurred Date",
            "mi_person_key": "MI Person Key"
        }
        
        # Create column list for SELECT with correct names
        col_list = ", ".join([f'"{col}"' for col in correct_columns])
        
        # Read part file schema to get actual column names (even if wrong)
        try:
            # Get actual column names from part file
            schema_query = f"DESCRIBE SELECT * FROM read_parquet('{part_uri}') LIMIT 0"
            part_schema = duckdb_conn.sql(schema_query).fetchall()
            part_actual_cols = [row[0] for row in part_schema]
            part_n_cols = len(part_actual_cols)
            
            logger.info(f"Part file has {part_n_cols} columns (expected {n_cols})")
            logger.debug(f"Part file columns (first 5): {part_actual_cols[:5]}")
            
            # Validate critical columns exist
            if part_n_cols <= CRITICAL_COL_INDICES["incurred_date"]:
                return False, f"Part file missing 'Incurred Date' column (only has {part_n_cols} columns)"
            if part_n_cols <= CRITICAL_COL_INDICES["mi_person_key"]:
                return False, f"Part file missing 'MI Person Key' column (only has {part_n_cols} columns)"
            
            # Verify schema matches expected columns (by position and name)
            # Check critical columns match expected names
            expected_incurred_date = correct_columns[CRITICAL_COL_INDICES["incurred_date"]]
            expected_mi_person_key = correct_columns[CRITICAL_COL_INDICES["mi_person_key"]]
            expected_claim_id = correct_columns[1] if len(correct_columns) > 1 else None
            
            actual_incurred_date_col = part_actual_cols[CRITICAL_COL_INDICES["incurred_date"]]
            actual_mi_person_key_col = part_actual_cols[CRITICAL_COL_INDICES["mi_person_key"]]
            actual_claim_id_col = part_actual_cols[1] if len(part_actual_cols) > 1 else None
            
            # Verify column names match expected schema
            schema_mismatches = []
            if actual_incurred_date_col != expected_incurred_date:
                schema_mismatches.append(f"Incurred Date: expected '{expected_incurred_date}', got '{actual_incurred_date_col}'")
            if actual_mi_person_key_col != expected_mi_person_key:
                schema_mismatches.append(f"MI Person Key: expected '{expected_mi_person_key}', got '{actual_mi_person_key_col}'")
            if expected_claim_id and actual_claim_id_col != expected_claim_id:
                schema_mismatches.append(f"Claim ID: expected '{expected_claim_id}', got '{actual_claim_id_col}'")
            
            if schema_mismatches:
                logger.error(f"Schema mismatch detected:")
                for mismatch in schema_mismatches:
                    logger.error(f"  - {mismatch}")
                logger.error(f"Part file schema does not match expected schema. Skipping merge to prevent data corruption.")
                return False, f"Schema mismatch: {', '.join(schema_mismatches)}"
            
            logger.info(f"Schema verified: Critical columns match expected schema")
            logger.info(f"  Incurred Date: '{actual_incurred_date_col}'")
            logger.info(f"  MI Person Key: '{actual_mi_person_key_col}'")
            if actual_claim_id_col:
                logger.info(f"  Claim ID: '{actual_claim_id_col}'")
            
            if part_n_cols != n_cols:
                logger.warning(f"Column count mismatch: part file has {part_n_cols}, expected {n_cols}")
                # Use minimum to avoid errors, pad with NULL if needed
                n_cols_to_read = min(part_n_cols, n_cols)
            else:
                n_cols_to_read = n_cols
            
            # Map part file columns (by position) to correct column names
            # Use the actual column names from the part file, but alias them to correct names
            part_col_selects = []
            for i in range(n_cols_to_read):
                # Escape the actual column name (may contain special characters)
                actual_col = part_actual_cols[i].replace('"', '""')
                correct_col = correct_columns[i]
                part_col_selects.append(f'"{actual_col}" AS "{correct_col}"')
            
            # If part file has fewer columns, pad with NULL
            if part_n_cols < n_cols:
                for i in range(part_n_cols, n_cols):
                    part_col_selects.append(f'NULL AS "{correct_columns[i]}"')
            
            part_col_list = ", ".join(part_col_selects)
            
            # Filter to only include rows with valid MI Person Key and Incurred Date
            # Validate Incurred Date format: should be 8 digits (YYYYMMDD)
            # Validate MI Person Key: should not be NULL
            # Also exclude rows that already exist in main file (duplicate check based on Claim ID if available)
            if actual_claim_id_col:
                # With duplicate check
                part_query = f"""
                SELECT {part_col_list}
                FROM read_parquet('{part_uri}') p
                WHERE p."{actual_mi_person_key_col}" IS NOT NULL
                  AND p."{actual_incurred_date_col}" IS NOT NULL
                  AND LENGTH(CAST(p."{actual_incurred_date_col}" AS VARCHAR)) = 8
                  AND regexp_matches(CAST(p."{actual_incurred_date_col}" AS VARCHAR), '^[0-9]{{8}}$')
                  AND TRY_STRPTIME(CAST(p."{actual_incurred_date_col}" AS VARCHAR), '%Y%m%d') IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 
                      FROM read_parquet('{main_uri}') m
                      WHERE CAST(m."Claim ID" AS VARCHAR) = CAST(p."{actual_claim_id_col}" AS VARCHAR)
                  )
                """
            else:
                # Without duplicate check (Claim ID not available)
                logger.warning("Claim ID column not found - skipping duplicate check. All valid rows will be merged.")
                part_query = f"""
                SELECT {part_col_list}
                FROM read_parquet('{part_uri}') p
                WHERE p."{actual_mi_person_key_col}" IS NOT NULL
                  AND p."{actual_incurred_date_col}" IS NOT NULL
                  AND LENGTH(CAST(p."{actual_incurred_date_col}" AS VARCHAR)) = 8
                  AND regexp_matches(CAST(p."{actual_incurred_date_col}" AS VARCHAR), '^[0-9]{{8}}$')
                  AND TRY_STRPTIME(CAST(p."{actual_incurred_date_col}" AS VARCHAR), '%Y%m%d') IS NOT NULL
                """
            
            # Check how many rows will be merged (for logging)
            # Count total rows, valid rows, and duplicates
            if actual_claim_id_col:
                count_query = f"""
                SELECT 
                    COUNT(*) as total_rows,
                    COUNT(CASE WHEN p."{actual_mi_person_key_col}" IS NOT NULL 
                                  AND p."{actual_incurred_date_col}" IS NOT NULL 
                                  AND LENGTH(CAST(p."{actual_incurred_date_col}" AS VARCHAR)) = 8
                                  AND regexp_matches(CAST(p."{actual_incurred_date_col}" AS VARCHAR), '^[0-9]{{8}}$')
                                  AND TRY_STRPTIME(CAST(p."{actual_incurred_date_col}" AS VARCHAR), '%Y%m%d') IS NOT NULL
                             THEN 1 END) as valid_rows,
                    COUNT(CASE WHEN p."{actual_mi_person_key_col}" IS NOT NULL 
                                  AND p."{actual_incurred_date_col}" IS NOT NULL 
                                  AND LENGTH(CAST(p."{actual_incurred_date_col}" AS VARCHAR)) = 8
                                  AND regexp_matches(CAST(p."{actual_incurred_date_col}" AS VARCHAR), '^[0-9]{{8}}$')
                                  AND TRY_STRPTIME(CAST(p."{actual_incurred_date_col}" AS VARCHAR), '%Y%m%d') IS NOT NULL
                                  AND EXISTS (
                                      SELECT 1 
                                      FROM read_parquet('{main_uri}') m
                                      WHERE CAST(m."Claim ID" AS VARCHAR) = CAST(p."{actual_claim_id_col}" AS VARCHAR)
                                  )
                             THEN 1 END) as duplicate_rows
                FROM read_parquet('{part_uri}') p
                """
            else:
                # No Claim ID, can't check duplicates
                count_query = f"""
                SELECT 
                    COUNT(*) as total_rows,
                    COUNT(CASE WHEN p."{actual_mi_person_key_col}" IS NOT NULL 
                                  AND p."{actual_incurred_date_col}" IS NOT NULL 
                                  AND LENGTH(CAST(p."{actual_incurred_date_col}" AS VARCHAR)) = 8
                                  AND regexp_matches(CAST(p."{actual_incurred_date_col}" AS VARCHAR), '^[0-9]{{8}}$')
                                  AND TRY_STRPTIME(CAST(p."{actual_incurred_date_col}" AS VARCHAR), '%Y%m%d') IS NOT NULL
                             THEN 1 END) as valid_rows,
                    0 as duplicate_rows
                FROM read_parquet('{part_uri}') p
                """
            
            count_result = duckdb_conn.sql(count_query).fetchone()
            total_rows = count_result[0] if count_result else 0
            valid_rows = count_result[1] if count_result else 0
            duplicate_rows = count_result[2] if count_result else 0
            new_rows = valid_rows - duplicate_rows
            
            if total_rows == 0:
                logger.warning(f"Part file has no rows - skipping merge")
                return False, "Part file has no rows"
            
            if valid_rows == 0:
                logger.warning(f"Part file has {total_rows} rows but none have valid MI Person Key and Incurred Date - skipping merge")
                return False, f"No valid rows (all {total_rows} rows missing critical columns)"
            
            if duplicate_rows > 0:
                logger.info(f"Found {duplicate_rows} duplicate rows (already exist in main file) - will skip these")
            
            if new_rows == 0:
                logger.warning(f"Part file has {valid_rows} valid rows, but all are duplicates. Nothing to merge.")
                return False, f"All {valid_rows} valid rows are duplicates"
            
            if valid_rows < total_rows:
                logger.warning(f"Part file has {total_rows} rows, but only {valid_rows} have valid MI Person Key and Incurred Date.")
            
            logger.info(f"Will merge {new_rows} new rows from part file (out of {valid_rows} valid rows, {duplicate_rows} duplicates skipped)")
            
        except Exception as e:
            logger.error(f"Error reading part file schema: {e}")
            return False, f"schema_error: {str(e)}"
        
        # Read main file (should have correct schema)
        main_query = f"SELECT {col_list} FROM read_parquet('{main_uri}')"
        
        # Union and write back
        query = f"""
        COPY (
            {main_query}
            UNION ALL
            {part_query}
        ) TO '{main_uri}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000, OVERWRITE_OR_IGNORE true);
        """
        
        duckdb_conn.sql(query)
        
        # Verify merge succeeded
        if not s3_exists(main_bucket, main_key):
            return False, "Main file missing after merge"
        
        logger.info(f"Successfully merged {new_rows} new rows from {part_uri} into {main_uri} ({duplicate_rows} duplicates skipped)")
        return True, f"merged {new_rows} new rows ({duplicate_rows} duplicates skipped)"
        
    except Exception as e:
        return False, f"merge_error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Merge corrected part files back into main parquet files")
    parser.add_argument("--dataset", choices=["medical", "pharmacy", "both"], default="both")
    parser.add_argument("--bronze-root", default="s3://pgxdatalake/bronze/")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be merged without actually merging")
    parser.add_argument("--delete-parts-after-merge", action="store_true", default=True, help="Delete part files after successful merge (default: True)")
    parser.add_argument("--no-delete-parts", dest="delete_parts_after_merge", action="store_false")
    # Note: Demographics enrichment and silver write are removed - those are silver preprocessing steps
    # This script focuses on adding validated data to bronze only
    
    args = parser.parse_args()
    
    logger = setup_logger()
    
    datasets = [args.dataset] if args.dataset != "both" else ["medical", "pharmacy"]
    
    print("="*80)
    print("MERGE PART FILES TO MAIN")
    print("="*80)
    print(f"Bronze root: {args.bronze_root}")
    print(f"Dataset(s): {', '.join(datasets)}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'EXECUTE'}")
    print(f"Delete parts after merge: {args.delete_parts_after_merge}")
    print(f"Note: This script adds validated data to bronze only.")
    print(f"      Silver preprocessing (demographics, partitioning) happens in follow-on scripts.")
    print("="*80)
    print()
    
    if args.dry_run:
        print("DRY-RUN MODE: No files will be merged. Use without --dry-run to actually merge.\n")
    
    # Initialize DuckDB if not dry-run
    duckdb_conn = None
    if not args.dry_run:
        if create_simple_duckdb_connection:
            duckdb_conn = create_simple_duckdb_connection(logger, None)
        else:
            import duckdb
            duckdb_conn = duckdb.connect()
            
            # Set up AWS credentials
            try:
                import boto3 as _b3
                _sess = _b3.Session()
                _creds = _sess.get_credentials()
                _region = _sess.region_name or "us-east-1"
                if _creds is not None:
                    _f = _creds.get_frozen_credentials()
                    if getattr(_f, "access_key", None) and getattr(_f, "secret_key", None):
                        duckdb_conn.sql(f"SET s3_access_key_id='{_f.access_key}'")
                        duckdb_conn.sql(f"SET s3_secret_access_key='{_f.secret_key}'")
                        if getattr(_f, "token", None):
                            duckdb_conn.sql(f"SET s3_session_token='{_f.token}'")
                duckdb_conn.sql(f"SET s3_region='{_region}'")
                duckdb_conn.sql("SET s3_url_style='path'")
            except Exception as e:
                logger.warning(f"Could not inject AWS credentials: {e}")
    
    s3_client = boto3.client("s3")
    parsed = urlparse(args.bronze_root)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip('/')
    
    total_merged = 0
    total_errors = 0
    total_deleted = 0
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Processing {dataset.upper()}")
        print(f"{'='*80}")
        
        # List part files
        part_prefix = f"{prefix.rstrip('/')}/{dataset}/part_files/" if prefix else f"{dataset}/part_files/"
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=part_prefix)
        
        part_files = []
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.parquet') and '.part_' in key:
                        part_files.append(f"s3://{bucket}/{key}")
        
        print(f"Found {len(part_files)} part files\n")
        
        if not part_files:
            print(f"No part files found for {dataset}")
            continue
        
        # Group part files by their main file
        main_to_parts = {}
        for part_uri in part_files:
            main_uri = find_main_file_for_part(part_uri, args.bronze_root, dataset)
            if main_uri:
                if main_uri not in main_to_parts:
                    main_to_parts[main_uri] = []
                main_to_parts[main_uri].append(part_uri)
            else:
                logger.warning(f"Could not find main file for {part_uri}")
        
        print(f"Found {len(main_to_parts)} main files with part files to merge\n")
        
        # Merge each main file with its parts
        for main_uri, part_uris in main_to_parts.items():
            print(f"\nMain file: {main_uri.split('/')[-1]}")
            print(f"  Part files to merge: {len(part_uris)}")
            
            if args.dry_run:
                for part_uri in part_uris:
                    print(f"    Would merge: {part_uri.split('/')[-1]}")
                total_merged += len(part_uris)
            else:
                # Process all parts for this main file
                all_success = True
                
                for part_uri in part_uris:
                    # Step 1: Validate part file schema matches expected schema
                    # Step 2: Validate data quality (MI Person Key, Incurred Date)
                    # Step 3: Check for duplicates (by Claim ID)
                    # Step 4: Merge validated, non-duplicate rows into bronze main file
                    success, message = merge_part_to_main(part_uri, main_uri, dataset, duckdb_conn, logger)
                    if success:
                        total_merged += 1
                        
                        # Delete part file after successful merge
                        if args.delete_parts_after_merge:
                            try:
                                part_bucket, part_key = _parse_s3_uri(part_uri)
                                s3_client.delete_object(Bucket=part_bucket, Key=part_key)
                                logger.info(f"Deleted part file: {part_uri}")
                                total_deleted += 1
                            except Exception as e:
                                logger.warning(f"Could not delete part file {part_uri}: {e}")
                    else:
                        logger.error(f"Failed to merge {part_uri}: {message}")
                        all_success = False
                        total_errors += 1
                        # Don't delete part file if merge failed - allows for investigation
                
                if all_success:
                    logger.info(f"Successfully merged all parts into {main_uri}")
                    logger.info(f"Bronze main file updated. Silver preprocessing will happen in follow-on scripts.")
    
    if duckdb_conn:
        try:
            duckdb_conn.close()
        except:
            pass
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total part files {'would be merged' if args.dry_run else 'merged'}: {total_merged}")
    print(f"Total part files {'would be deleted' if args.dry_run else 'deleted'}: {total_deleted}")
    print(f"Total errors: {total_errors}")
    print(f"\nNote: Merged data is in bronze. Run silver preprocessing scripts to:")
    print(f"  - Enrich with demographics")
    print(f"  - Partition by age_band and event_year")
    print(f"  - Write to silver layer")
    
    if args.dry_run:
        print("\nTo actually merge files, run without --dry-run flag")


if __name__ == "__main__":
    main()

