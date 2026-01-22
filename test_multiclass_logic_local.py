#!/usr/bin/env python3
"""
Test Phase 3 multiclass column logic locally.

This script tests the CTE calculation for multiclass target windows (7d, 14d, 21d, 30d, 45d)
to diagnose why all patient counts are identical.

Usage:
    # Test with local unified_event_fact_table if it exists
    python test_multiclass_logic_local.py --age-band 75-84 --event-year 2018

    # Download sample from S3 and test
    python test_multiclass_logic_local.py --age-band 75-84 --event-year 2018 --download-sample
"""

import argparse
import sys
import os
from pathlib import Path
import duckdb
import logging

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from py_helpers.duckdb_utils import get_duckdb_connection
from py_helpers.constants import S3_BUCKET
from py_helpers.env_utils import get_data_root

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_sample_from_s3(age_band: str, event_year: int, local_path: Path):
    """Download a sample of unified_event_fact_table from S3."""
    import boto3
    from py_helpers.s3_utils import get_unified_event_fact_table_path
    
    logger.info(f"Downloading sample from S3...")
    s3_client = boto3.client('s3')
    
    # Get S3 path for unified_event_fact_table
    s3_path = get_unified_event_fact_table_path(age_band, event_year)
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    
    logger.info(f"S3 path: {s3_path}")
    
    # Download to local
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket, key, str(local_path))
    logger.info(f"Downloaded to: {local_path}")


def resolve_unified_event_fact_table_path(age_band: str, event_year: int) -> str:
    """
    Resolve path to unified_event_fact_table, preferring local /mnt/nvme paths over S3.
    
    Priority:
    1. Local path: /mnt/nvme/gold/unified_event_fact_table/age_band={age_band}/event_year={event_year}/unified_event_fact_table.parquet
    2. S3 path: s3://{S3_BUCKET}/gold/unified_event_fact_table/age_band={age_band}/event_year={event_year}/unified_event_fact_table.parquet
    
    Returns:
        Path string (local if exists, otherwise S3)
    """
    from py_helpers.env_utils import is_linux
    
    # Check local path first (Linux/EC2: /mnt/nvme/gold/unified_event_fact_table/)
    if is_linux():
        data_root = get_data_root()
        local_path = data_root / "gold" / "unified_event_fact_table" / f"age_band={age_band}" / f"event_year={event_year}" / "unified_event_fact_table.parquet"
        if local_path.exists():
            return str(local_path)
    
    # Fall back to S3
    return f"s3://{S3_BUCKET}/gold/unified_event_fact_table/age_band={age_band}/event_year={event_year}/unified_event_fact_table.parquet"


def test_multiclass_ctes(conn, age_band: str, event_year: int, use_local: bool = False):
    """Test the multiclass CTE logic on unified_event_fact_table."""
    
    logger.info("=" * 80)
    logger.info("Testing Multiclass CTE Logic")
    logger.info("=" * 80)
    
    # Resolve path (prefers local if exists)
    table_path = resolve_unified_event_fact_table_path(age_band, event_year)
    
    if use_local and not table_path.startswith("s3://"):
        logger.info(f"Using local file: {table_path}")
    elif table_path.startswith("s3://"):
        logger.info(f"Using S3 path: {table_path}")
    else:
        logger.info(f"Using path: {table_path}")
    
    # Create a view from the parquet file
    logger.info("Loading unified_event_fact_table...")
    conn.execute(f"""
        CREATE OR REPLACE TEMP VIEW unified_event_fact_table AS
        SELECT * FROM read_parquet('{table_path}')
    """)
    
    # Count total rows
    total_rows = conn.execute("SELECT COUNT(*)::BIGINT FROM unified_event_fact_table").fetchone()[0]
    logger.info(f"Total rows in unified_event_fact_table: {int(total_rows):,}")
    
    # Materialize opioid patients (required for Phase 3 logic)
    logger.info("Materializing opioid patients...")
    from py_helpers.constants import get_opioid_icd_sql_condition
    opioid_icd_condition = get_opioid_icd_sql_condition()
    
    conn.execute(f"""
        CREATE OR REPLACE TEMP VIEW opioid_patients_materialized AS
        SELECT DISTINCT mi_person_key
        FROM unified_event_fact_table
        WHERE {opioid_icd_condition}
    """)
    opioid_count = conn.execute("SELECT COUNT(*)::BIGINT FROM opioid_patients_materialized").fetchone()[0]
    logger.info(f"Opioid patients: {int(opioid_count):,}")
    
    # Test the multiclass CTE logic
    logger.info("\n" + "=" * 80)
    logger.info("Testing Multiclass CTEs (7d, 14d, 21d, 30d, 45d)")
    logger.info("=" * 80)
    
    # Use the exact CTE logic from Phase 3
    test_query = """
    WITH hcg_target_events AS (
        SELECT mi_person_key, event_date as hcg_event_date
        FROM unified_event_fact_table uef
        WHERE event_classification = 'ed_non_opioid'
          AND NOT EXISTS (
              SELECT 1 FROM opioid_patients_materialized op
              WHERE op.mi_person_key = uef.mi_person_key
          )
    ),
    drug_events AS (
        SELECT mi_person_key, event_date as drug_event_date
        FROM unified_event_fact_table
        WHERE event_type = 'pharmacy'
    ),
    pairs_7d AS (
        SELECT DISTINCT de.mi_person_key
        FROM drug_events de
        INNER JOIN hcg_target_events hte ON de.mi_person_key = hte.mi_person_key
            AND hte.hcg_event_date >= de.drug_event_date
            AND hte.hcg_event_date <= DATE_ADD(de.drug_event_date, INTERVAL 7 DAY)
    ),
    pairs_14d AS (
        SELECT DISTINCT de.mi_person_key
        FROM drug_events de
        INNER JOIN hcg_target_events hte ON de.mi_person_key = hte.mi_person_key
            AND hte.hcg_event_date >= de.drug_event_date
            AND hte.hcg_event_date <= DATE_ADD(de.drug_event_date, INTERVAL 14 DAY)
    ),
    pairs_21d AS (
        SELECT DISTINCT de.mi_person_key
        FROM drug_events de
        INNER JOIN hcg_target_events hte ON de.mi_person_key = hte.mi_person_key
            AND hte.hcg_event_date >= de.drug_event_date
            AND hte.hcg_event_date <= DATE_ADD(de.drug_event_date, INTERVAL 21 DAY)
    ),
    pairs_30d AS (
        SELECT DISTINCT de.mi_person_key
        FROM drug_events de
        INNER JOIN hcg_target_events hte ON de.mi_person_key = hte.mi_person_key
            AND hte.hcg_event_date >= de.drug_event_date
            AND hte.hcg_event_date <= DATE_ADD(de.drug_event_date, INTERVAL 30 DAY)
    ),
    pairs_45d AS (
        SELECT DISTINCT de.mi_person_key
        FROM drug_events de
        INNER JOIN hcg_target_events hte ON de.mi_person_key = hte.mi_person_key
            AND hte.hcg_event_date >= de.drug_event_date
            AND hte.hcg_event_date <= DATE_ADD(de.drug_event_date, INTERVAL 45 DAY)
    )
    SELECT 
        CAST((SELECT COUNT(*) FROM pairs_7d) AS BIGINT) as patients_7d,
        CAST((SELECT COUNT(*) FROM pairs_14d) AS BIGINT) as patients_14d,
        CAST((SELECT COUNT(*) FROM pairs_21d) AS BIGINT) as patients_21d,
        CAST((SELECT COUNT(*) FROM pairs_30d) AS BIGINT) as patients_30d,
        CAST((SELECT COUNT(*) FROM pairs_45d) AS BIGINT) as patients_45d,
        CAST((SELECT COUNT(*) FROM pairs_14d p14 WHERE NOT EXISTS (SELECT 1 FROM pairs_7d p7 WHERE p7.mi_person_key = p14.mi_person_key)) AS BIGINT) as patients_only_14d,
        CAST((SELECT COUNT(*) FROM pairs_21d p21 WHERE NOT EXISTS (SELECT 1 FROM pairs_14d p14 WHERE p14.mi_person_key = p21.mi_person_key)) AS BIGINT) as patients_only_21d,
        CAST((SELECT COUNT(*) FROM pairs_30d p30 WHERE NOT EXISTS (SELECT 1 FROM pairs_21d p21 WHERE p21.mi_person_key = p30.mi_person_key)) AS BIGINT) as patients_only_30d,
        CAST((SELECT COUNT(*) FROM pairs_45d p45 WHERE NOT EXISTS (SELECT 1 FROM pairs_30d p30 WHERE p30.mi_person_key = p45.mi_person_key)) AS BIGINT) as patients_only_45d
    """
    
    result_df = conn.execute(test_query).fetchdf()
    if result_df.empty:
        logger.error("No results from CTE test query!")
        return
    
    counts = result_df.iloc[0]
    
    logger.info("\nCTE Patient Counts:")
    logger.info(f"  7d:  {int(counts['patients_7d']):,}")
    logger.info(f"  14d: {int(counts['patients_14d']):,}")
    logger.info(f"  21d: {int(counts['patients_21d']):,}")
    logger.info(f"  30d: {int(counts['patients_30d']):,}")
    logger.info(f"  45d: {int(counts['patients_45d']):,}")
    
    logger.info("\nPatients ONLY in longer windows (not in shorter):")
    logger.info(f"  Only in 14d (not 7d):  {int(counts['patients_only_14d']):,}")
    logger.info(f"  Only in 21d (not 14d): {int(counts['patients_only_21d']):,}")
    logger.info(f"  Only in 30d (not 21d): {int(counts['patients_only_30d']):,}")
    logger.info(f"  Only in 45d (not 30d): {int(counts['patients_only_45d']):,}")
    
    # Check if all counts are identical
    all_same = (
        counts['patients_7d'] == counts['patients_14d'] == 
        counts['patients_21d'] == counts['patients_30d'] == counts['patients_45d']
    )
    
    if all_same:
        logger.warning("\n[WARNING] All CTE patient counts are IDENTICAL!")
        logger.warning("This suggests all patients have drug-HCG pairs within 7 days.")
        logger.warning("This is the root cause of identical multiclass column counts.")
    else:
        logger.info("\n[OK] CTE patient counts differ as expected.")
        logger.info("The issue may be in how these CTEs are joined in the final cohort.")
    
    # Additional diagnostic: Check actual date differences
    logger.info("\n" + "=" * 80)
    logger.info("Checking actual date differences in drug-HCG pairs")
    logger.info("=" * 80)
    
    date_diff_query = """
    WITH hcg_target_events AS (
        SELECT mi_person_key, event_date as hcg_event_date
        FROM unified_event_fact_table uef
        WHERE event_classification = 'ed_non_opioid'
          AND NOT EXISTS (
              SELECT 1 FROM opioid_patients_materialized op
              WHERE op.mi_person_key = uef.mi_person_key
          )
    ),
    drug_events AS (
        SELECT mi_person_key, event_date as drug_event_date
        FROM unified_event_fact_table
        WHERE event_type = 'pharmacy'
    ),
    all_pairs AS (
        SELECT DISTINCT
            de.mi_person_key,
            de.drug_event_date,
            hte.hcg_event_date,
            CAST(datediff('day', de.drug_event_date, hte.hcg_event_date) AS INTEGER) as days_between
        FROM drug_events de
        INNER JOIN hcg_target_events hte ON de.mi_person_key = hte.mi_person_key
            AND hte.hcg_event_date >= de.drug_event_date
            AND hte.hcg_event_date <= DATE_ADD(de.drug_event_date, INTERVAL 45 DAY)
    )
    SELECT 
        CAST(COUNT(DISTINCT CASE WHEN days_between <= 7 THEN mi_person_key END) AS BIGINT) as patients_within_7d,
        CAST(COUNT(DISTINCT CASE WHEN days_between > 7 AND days_between <= 14 THEN mi_person_key END) AS BIGINT) as patients_8_to_14d,
        CAST(COUNT(DISTINCT CASE WHEN days_between > 14 AND days_between <= 21 THEN mi_person_key END) AS BIGINT) as patients_15_to_21d,
        CAST(COUNT(DISTINCT CASE WHEN days_between > 21 AND days_between <= 30 THEN mi_person_key END) AS BIGINT) as patients_22_to_30d,
        CAST(COUNT(DISTINCT CASE WHEN days_between > 30 AND days_between <= 45 THEN mi_person_key END) AS BIGINT) as patients_31_to_45d,
        CAST(MIN(days_between) AS INTEGER) as min_days,
        CAST(MAX(days_between) AS INTEGER) as max_days,
        CAST(AVG(days_between) AS DOUBLE) as avg_days
    FROM all_pairs
    """
    
    date_diff_df = conn.execute(date_diff_query).fetchdf()
    if not date_diff_df.empty:
        diff_counts = date_diff_df.iloc[0]
        logger.info(f"\nPatients by date range:")
        logger.info(f"  Within 7 days:    {int(diff_counts['patients_within_7d']):,}")
        logger.info(f"  8-14 days:        {int(diff_counts['patients_8_to_14d']):,}")
        logger.info(f"  15-21 days:       {int(diff_counts['patients_15_to_21d']):,}")
        logger.info(f"  22-30 days:       {int(diff_counts['patients_22_to_30d']):,}")
        logger.info(f"  31-45 days:       {int(diff_counts['patients_31_to_45d']):,}")
        logger.info(f"\nDate difference stats:")
        logger.info(f"  Min days: {int(diff_counts['min_days']) if diff_counts['min_days'] is not None else 'N/A'}")
        logger.info(f"  Max days: {int(diff_counts['max_days']) if diff_counts['max_days'] is not None else 'N/A'}")
        logger.info(f"  Avg days: {diff_counts['avg_days']:.2f if diff_counts['avg_days'] is not None else 'N/A'}")


def main():
    parser = argparse.ArgumentParser(description="Test Phase 3 multiclass CTE logic locally")
    parser.add_argument("--age-band", required=True, help="Age band (e.g., '75-84')")
    parser.add_argument("--event-year", type=int, required=True, help="Event year (e.g., 2018)")
    parser.add_argument("--use-local", action="store_true", help="Use local unified_event_fact_table if it exists")
    parser.add_argument("--download-sample", action="store_true", help="Download sample from S3 first")
    
    args = parser.parse_args()
    
    # Setup DuckDB connection
    logger.info("Setting up DuckDB connection...")
    conn = get_duckdb_connection(logger=logger)
    
    # Enable S3 if needed
    if not args.use_local or args.download_sample:
        logger.info("Enabling S3 support...")
        conn.execute("INSTALL httpfs; LOAD httpfs;")
        conn.execute("INSTALL aws; LOAD aws;")
        try:
            conn.execute("CALL load_aws_credentials();")
        except Exception as e:
            logger.warning(f"Could not load AWS credentials: {e}")
            logger.warning("You may need to configure AWS credentials for S3 access")
    
    try:
        test_multiclass_ctes(conn, args.age_band, args.event_year, use_local=args.use_local)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
