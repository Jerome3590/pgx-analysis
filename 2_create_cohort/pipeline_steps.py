"""
Optimized pipeline steps with DuckDB optimizations from APCD development.

This module contains optimized versions of all pipeline steps that apply
the DuckDB optimizations and development rules learned from 1_apcd_input_data.

Updates:
- Centralized checkpoint system at s3://pgx-repository/pgx-pipeline-status/
- Windows emoji compatibility
- Robust step-level checkpointing
"""

import os
import sys
import traceback
import logging
import platform
from datetime import datetime

# Windows emoji compatibility
IS_WINDOWS = platform.system() == 'Windows'
SYMBOLS = {
    'arrow': '->' if IS_WINDOWS else '→',
    'success': '[PASS]' if IS_WINDOWS else '✅',
    'fail': '[FAIL]' if IS_WINDOWS else '❌',
    'info': '[INFO]' if IS_WINDOWS else '📊',
    'check': '[CHECK]' if IS_WINDOWS else '🔍'
}

# Set root of project
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    sys.path.append(project_root)

from helpers.constants import (
    MAX_RETRIES,
    RETRY_DELAY,
    DICTIONARY_SIZE_LIMIT_PERCENT,
    BLOOM_FILTER_FALSE_POSITIVE_RATIO,
    OPIOID_ICD_CODES,
    S3_BUCKET
)

from helpers.s3_utils import (
    sanitize_for_s3_key,
    validate_s3_source_paths,
    s3_exists,
    save_to_s3_parquet,
    save_to_s3_json,
    save_to_s3_text,
    save_to_s3_html,
    save_pipeline_metrics,
    get_output_paths,
    acquire_lock,
    release_lock,
    get_checkpoint_path,
    checkpoint_exists,
    save_checkpoint,
    load_checkpoint,
    delete_checkpoint,
    list_checkpoints,
    cleanup_checkpoints,
    check_step_checkpoint,
    save_step_checkpoint,
    _scan_available_checkpoints
)

from helpers.data_utils import (
    collect_validation_metrics,
    validate_cohort_name,
    generate_qa_report
)
from helpers.common_imports import get_logger

# Import optimized DuckDB utilities
from helpers.duckdb_utils import (
    create_ec2_optimized_connection,
    cleanup_duckdb_temp_files,
    monitor_disk_space,
    enable_query_profiling,
    disable_query_profiling,
    force_checkpoint,
    setup_ec2_nvme_storage,
    collect_metrics_as_dict,
    get_standardized_column_order,
    generate_null_filled_select,
    extract_metrics_subset,
    validate_environment,
    get_duckdb_connection,
    setup_duckdb_environment
)

from helpers.common_imports import s3_client

import boto3
from helpers.aws_utils import notify_error, notify_success
from helpers.cohort_utils import check_cohort_exists, check_and_fix_mismatched_sets, check_cohort_exists_and_delete_message


def run_step1_lock_acquisition_optimized(context):
    """Step 1: Lock acquisition with DuckDB optimizations."""
    logger = context["logger"]
    age_band = context["age_band"]
    event_year = context["event_year"]
    
    logger.info("→ [STEP 1] Starting optimized lock acquisition...")
    
    try:
        # Acquire lock with retry logic
        lock_acquired = False
        for attempt in range(MAX_RETRIES):
            try:
                lock_acquired = acquire_lock(age_band, event_year)
                if lock_acquired:
                    break
                else:
                    logger.warning(f"→ [STEP 1] Lock acquisition attempt {attempt + 1} failed, retrying...")
                    time.sleep(RETRY_DELAY)
            except Exception as e:
                logger.warning(f"→ [STEP 1] Lock acquisition attempt {attempt + 1} failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(RETRY_DELAY)
        
        if not lock_acquired:
            raise Exception("Failed to acquire lock after maximum retries")
        
        logger.info("→ [STEP 1] Lock acquired successfully")
        
        # Save checkpoint
        save_step_checkpoint(age_band, event_year, "step1_lock_acquisition", {
            "lock_acquired": True,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"→ [STEP 1] Lock acquisition failed: {str(e)}")
        raise


def run_step2_database_setup_optimized(context):
    """Step 2: Database setup with DuckDB optimizations."""
    logger = context["logger"]
    operation_type = context.get("operation_type", "concurrent_processing")
    
    logger.info("→ [STEP 2] Starting optimized database setup...")
    
    try:
        # Create optimized DuckDB connection
        cohort_conn_duckdb = create_ec2_optimized_connection(logger, operation_type)
        
        # Install and load required extensions
        cohort_conn_duckdb.sql("INSTALL httpfs; LOAD httpfs;")
        cohort_conn_duckdb.sql("INSTALL aws; LOAD aws;")
        
        # Load AWS credentials
        cohort_conn_duckdb.sql("CALL load_aws_credentials();")
        
        # Configure S3 settings
        cohort_conn_duckdb.sql("SET s3_region='us-east-1'")
        cohort_conn_duckdb.sql("SET s3_url_style='path'")
        
        # Set HTTP timeout and retry settings for S3 operations
        cohort_conn_duckdb.sql("SET http_timeout = '600000'")  # 10 minutes
        cohort_conn_duckdb.sql("SET http_retries = 5")
        
        # Update context with optimized connection
        context["cohort_conn_duckdb"] = cohort_conn_duckdb
        
        # Monitor disk space
        monitor_disk_space(logger)
        
        logger.info("→ [STEP 2] Optimized database setup completed")
        
        # Save checkpoint
        age_band = context["age_band"]
        event_year = context["event_year"]
        save_step_checkpoint(age_band, event_year, "step2_database_setup", {
            "database_configured": True,
            "operation_type": operation_type,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"→ [STEP 2] Database setup failed: {str(e)}")
        raise


def run_step3_medical_data_loading_optimized(context):
    """Step 3: Medical data loading with DuckDB optimizations."""
    logger = context["logger"]
    cohort_conn_duckdb = context["cohort_conn_duckdb"]
    age_band = context["age_band"]
    event_year = context["event_year"]
    
    logger.info("→ [STEP 3] Starting optimized medical data loading from APCD gold tier...")
    
    try:
        # Enable query profiling for this step
        enable_query_profiling(cohort_conn_duckdb, logger, "json", f"/tmp/duckdb_profiling_step3_medical.json")
        
        # Load demographics lookup from APCD silver tier
        demographics_sql = f"""
        CREATE OR REPLACE VIEW demographics_lookup AS
        SELECT *
        FROM read_parquet('s3://pgxdatalake/silver/imputed/mi_person_key_demographics_lookup.parquet')
        WHERE event_year = {event_year};
        """
        cohort_conn_duckdb.sql(demographics_sql)
        logger.info("→ [STEP 3] Demographics lookup loaded")
        
        # Load pre-imputed medical data from APCD gold tier
        medical_sql = f"""
        CREATE OR REPLACE VIEW medical_raw AS
        SELECT *
        FROM read_parquet_auto('s3://pgxdatalake/gold/medical/age_band={age_band}/event_year={event_year}/*.parquet')
        WHERE age_band = '{age_band}'
          AND event_year = {event_year};
        """
        cohort_conn_duckdb.sql(medical_sql)
        logger.info("→ [STEP 3] Medical raw data loaded from APCD gold tier")
        
        # Create medical view with pre-imputed demographics
        medical_demographics_sql = f"""
        CREATE OR REPLACE VIEW medical_with_demographics AS
        SELECT 
            m.*,
            -- Use pre-imputed demographics from APCD processing
            COALESCE(m.age_imputed, dl.age_imputed) AS age_imputed,
            COALESCE(m.gender_imputed, dl.gender_imputed) AS gender_imputed,
            COALESCE(m.race_imputed, dl.race_imputed) AS race_imputed,
            COALESCE(m.zip_imputed, dl.zip_imputed) AS zip_imputed,
            COALESCE(m.county_imputed, dl.county_imputed) AS county_imputed,
            COALESCE(m.payer_imputed, dl.payer_imputed) AS payer_imputed,
            -- Track data sources
            COALESCE(m.age_source, dl.age_source, 'original') AS age_source,
            COALESCE(m.gender_source, dl.gender_source, 'original') AS gender_source,
            COALESCE(m.race_source, dl.race_source, 'original') AS race_source,
            COALESCE(m.zip_source, dl.zip_source, 'original') AS zip_source,
            COALESCE(m.county_source, dl.county_source, 'original') AS county_source,
            COALESCE(m.payer_source, dl.payer_source, 'original') AS payer_source
        FROM medical_raw m
        LEFT JOIN demographics_lookup dl ON m.mi_person_key = dl.mi_person_key 
            AND m.event_year = dl.event_year;
        """
        cohort_conn_duckdb.sql(medical_demographics_sql)
        logger.info("→ [STEP 3] Medical demographics view created")
        
        # Apply data quality filters
        medical_filtered_sql = f"""
        CREATE OR REPLACE VIEW medical_clean AS
        SELECT *
        FROM medical_with_demographics
        WHERE mi_person_key IS NOT NULL
          AND mi_person_key != ''
          AND event_date IS NOT NULL
          AND age_imputed IS NOT NULL
          AND age_imputed BETWEEN 1 AND 114
          AND event_date >= '{event_year}-01-01'
          AND event_date <= '{event_year}-12-31';
        """
        cohort_conn_duckdb.sql(medical_filtered_sql)
        logger.info("→ [STEP 3] Medical data filtered and cleaned")
        
        # QA checks
        total_records = cohort_conn_duckdb.sql("SELECT COUNT(*) FROM medical_clean").fetchone()[0]
        demographics_check = cohort_conn_duckdb.sql("""
        SELECT 
            COUNT(*) as total_records,
            COUNT(age_imputed) as records_with_age,
            COUNT(gender_imputed) as records_with_gender,
            COUNT(race_imputed) as records_with_race
        FROM medical_with_demographics
        """).fetchone()
        
        logger.info(f"→ [STEP 3] QA: Total medical records: {total_records:,}")
        logger.info(f"→ [STEP 3] QA: Demographics coverage: {demographics_check[1]}/{demographics_check[0]} ({demographics_check[1]/demographics_check[0]*100:.1f}%)")
        
        # Force checkpoint
        force_checkpoint(cohort_conn_duckdb, logger)
        
        # Disable query profiling
        disable_query_profiling(cohort_conn_duckdb, logger)
        
        logger.info("→ [STEP 3] Optimized medical data loading completed")
        
        # Save checkpoint
        save_step_checkpoint(age_band, event_year, "step3_medical_data_loading", {
            "medical_records_loaded": total_records,
            "demographics_coverage": demographics_check[1]/demographics_check[0] if demographics_check[0] > 0 else 0,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"→ [STEP 3] Medical data loading failed: {str(e)}")
        # Clean up temp files on error
        cleanup_duckdb_temp_files(logger)
        raise


def run_step4_pharmacy_data_loading_optimized(context):
    """Step 4: Pharmacy data loading with DuckDB optimizations."""
    logger = context["logger"]
    cohort_conn_duckdb = context["cohort_conn_duckdb"]
    age_band = context["age_band"]
    event_year = context["event_year"]
    
    logger.info("→ [STEP 4] Starting optimized pharmacy data loading from APCD gold tier...")
    
    try:
        # Enable query profiling for this step
        enable_query_profiling(cohort_conn_duckdb, logger, "json", f"/tmp/duckdb_profiling_step4_pharmacy.json")
        
        # Load pre-imputed pharmacy data from APCD gold tier
        pharmacy_sql = f"""
        CREATE OR REPLACE VIEW pharmacy_raw AS
        SELECT *
        FROM read_parquet_auto('s3://pgxdatalake/gold/pharmacy/age_band={age_band}/event_year={event_year}/*.parquet')
        WHERE age_band = '{age_band}'
          AND event_year = {event_year};
        """
        cohort_conn_duckdb.sql(pharmacy_sql)
        logger.info("→ [STEP 4] Pharmacy raw data loaded from APCD gold tier")
        
        # Create pharmacy view with pre-imputed demographics
        pharmacy_demographics_sql = f"""
        CREATE OR REPLACE VIEW pharmacy_with_demographics AS
        SELECT 
            p.*,
            -- Use pre-imputed demographics from APCD processing
            COALESCE(p.age_imputed, dl.age_imputed) AS age_imputed,
            COALESCE(p.gender_imputed, dl.gender_imputed) AS gender_imputed,
            COALESCE(p.race_imputed, dl.race_imputed) AS race_imputed,
            COALESCE(p.zip_imputed, dl.zip_imputed) AS zip_imputed,
            COALESCE(p.county_imputed, dl.county_imputed) AS county_imputed,
            COALESCE(p.payer_imputed, dl.payer_imputed) AS payer_imputed,
            -- Track data sources
            COALESCE(p.age_source, dl.age_source, 'original') AS age_source,
            COALESCE(p.gender_source, dl.gender_source, 'original') AS gender_source,
            COALESCE(p.race_source, dl.race_source, 'original') AS race_source,
            COALESCE(p.zip_source, dl.zip_source, 'original') AS zip_source,
            COALESCE(p.county_source, dl.county_source, 'original') AS county_source,
            COALESCE(p.payer_source, dl.payer_source, 'original') AS payer_source
        FROM pharmacy_raw p
        LEFT JOIN demographics_lookup dl ON p.mi_person_key = dl.mi_person_key 
            AND p.event_year = dl.event_year;
        """
        cohort_conn_duckdb.sql(pharmacy_demographics_sql)
        logger.info("→ [STEP 4] Pharmacy demographics view created")
        
        # Apply data quality filters
        pharmacy_filtered_sql = f"""
        CREATE OR REPLACE VIEW pharmacy_clean AS
        SELECT *
        FROM pharmacy_with_demographics
        WHERE mi_person_key IS NOT NULL
          AND mi_person_key != ''
          AND event_date IS NOT NULL
          AND age_imputed IS NOT NULL
          AND age_imputed BETWEEN 1 AND 114
          AND event_date >= '{event_year}-01-01'
          AND event_date <= '{event_year}-12-31'
          AND drug_name IS NOT NULL
          AND drug_name != '';
        """
        cohort_conn_duckdb.sql(pharmacy_filtered_sql)
        logger.info("→ [STEP 4] Pharmacy data filtered and cleaned")
        
        # QA checks
        total_records = cohort_conn_duckdb.sql("SELECT COUNT(*) FROM pharmacy_clean").fetchone()[0]
        demographics_check = cohort_conn_duckdb.sql("""
        SELECT 
            COUNT(*) as total_records,
            COUNT(age_imputed) as records_with_age,
            COUNT(gender_imputed) as records_with_gender,
            COUNT(race_imputed) as records_with_race
        FROM pharmacy_with_demographics
        """).fetchone()
        
        drug_name_check = cohort_conn_duckdb.sql("""
        SELECT 
            CASE 
                WHEN drug_name = '' THEN 'Empty'
                WHEN drug_name IS NULL THEN 'NULL'
                ELSE 'Valid'
            END as drug_name_status,
            COUNT(*) as count
        FROM pharmacy_with_demographics
        GROUP BY drug_name_status
        """).fetchall()
        
        logger.info(f"→ [STEP 4] QA: Total pharmacy records: {total_records:,}")
        logger.info(f"→ [STEP 4] QA: Demographics coverage: {demographics_check[1]}/{demographics_check[0]} ({demographics_check[1]/demographics_check[0]*100:.1f}%)")
        logger.info(f"→ [STEP 4] QA: Drug name validation: {dict(drug_name_check)}")
        
        # Force checkpoint
        force_checkpoint(cohort_conn_duckdb, logger)
        
        # Disable query profiling
        disable_query_profiling(cohort_conn_duckdb, logger)
        
        logger.info("→ [STEP 4] Optimized pharmacy data loading completed")
        
        # Save checkpoint
        save_step_checkpoint(age_band, event_year, "step4_pharmacy_data_loading", {
            "pharmacy_records_loaded": total_records,
            "demographics_coverage": demographics_check[1]/demographics_check[0] if demographics_check[0] > 0 else 0,
            "drug_name_validation": dict(drug_name_check),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"→ [STEP 4] Pharmacy data loading failed: {str(e)}")
        # Clean up temp files on error
        cleanup_duckdb_temp_files(logger)
        raise


def run_step7_event_features_optimized(context):
    """Step 7: Event features with DuckDB optimizations."""
    logger = context["logger"]
    cohort_conn_duckdb = context["cohort_conn_duckdb"]
    age_band = context["age_band"]
    event_year = context["event_year"]
    
    logger.info("→ [STEP 7] Starting optimized event features creation...")
    
    try:
        # Enable query profiling for this step
        enable_query_profiling(cohort_conn_duckdb, logger, "json", f"/tmp/duckdb_profiling_step7_event_features.json")
        
        # Create unified event features using pre-imputed demographics
        event_features_sql = f"""
        CREATE OR REPLACE VIEW cohort_event_features AS
        SELECT 
            mi_person_key,
            event_date,
            'medical' as event_type,
            'medical' as data_source,
            age_imputed,
            gender_imputed as member_gender,
            race_imputed as member_race,
            zip_imputed,
            county_imputed,
            payer_imputed,
            primary_icd_diagnosis_code,
            NULL as drug_name,
            NULL as therapeutic_class_1,
            -- Event classification
            CASE 
                WHEN primary_icd_diagnosis_code IN {tuple(OPIOID_ICD_CODES)} THEN 'opioid_ed'
                ELSE 'ed_non_opioid'
            END as event_classification,
            -- First event flags
            ROW_NUMBER() OVER (PARTITION BY mi_person_key ORDER BY event_date) as event_sequence
        FROM medical
        WHERE primary_icd_diagnosis_code IS NOT NULL
        
        UNION ALL
        
        SELECT 
            mi_person_key,
            event_date,
            'pharmacy' as event_type,
            'pharmacy' as data_source,
            age_imputed,
            gender_imputed as member_gender,
            race_imputed as member_race,
            zip_imputed,
            county_imputed,
            payer_imputed,
            NULL as primary_icd_diagnosis_code,
            drug_name,
            therapeutic_class_1,
            'pharmacy' as event_classification,
            ROW_NUMBER() OVER (PARTITION BY mi_person_key ORDER BY event_date) as event_sequence
        FROM pharmacy
        WHERE drug_name IS NOT NULL;
        """
        cohort_conn_duckdb.sql(event_features_sql)
        logger.info("→ [STEP 7] Event features view created")
        
        # QA checks
        total_events = cohort_conn_duckdb.sql("SELECT COUNT(*) FROM cohort_event_features").fetchone()[0]
        event_type_dist = cohort_conn_duckdb.sql("""
        SELECT event_type, COUNT(*) as count
        FROM cohort_event_features
        GROUP BY event_type
        ORDER BY count DESC
        """).fetchall()
        
        event_classification_dist = cohort_conn_duckdb.sql("""
        SELECT event_classification, COUNT(*) as count
        FROM cohort_event_features
        GROUP BY event_classification
        ORDER BY count DESC
        """).fetchall()
        
        logger.info(f"→ [STEP 7] QA: Total events: {total_events:,}")
        logger.info(f"→ [STEP 7] QA: Event type distribution: {dict(event_type_dist)}")
        logger.info(f"→ [STEP 7] QA: Event classification distribution: {dict(event_classification_dist)}")
        
        # Force checkpoint
        force_checkpoint(cohort_conn_duckdb, logger)
        
        # Disable query profiling
        disable_query_profiling(cohort_conn_duckdb, logger)
        
        logger.info("→ [STEP 7] Optimized event features creation completed")
        
        # Save checkpoint
        save_step_checkpoint(age_band, event_year, "step7_event_features", {
            "total_events": total_events,
            "event_type_distribution": dict(event_type_dist),
            "event_classification_distribution": dict(event_classification_dist),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"→ [STEP 7] Event features creation failed: {str(e)}")
        # Clean up temp files on error
        cleanup_duckdb_temp_files(logger)
        raise


def run_step13_opioid_ed_cohort_optimized(context):
    """Step 13: OPIOID_ED cohort creation with DuckDB optimizations."""
    logger = context["logger"]
    cohort_conn_duckdb = context["cohort_conn_duckdb"]
    age_band = context["age_band"]
    event_year = context["event_year"]
    
    logger.info("→ [STEP 13] Starting optimized OPIOID_ED cohort creation...")
    
    try:
        # Enable query profiling for this step
        enable_query_profiling(cohort_conn_duckdb, logger, "json", f"/tmp/duckdb_profiling_step13_opioid_ed.json")
        
        # Create OPIOID_ED cohort with 5:1 control-to-target ratio
        opioid_ed_cohort_sql = f"""
        CREATE OR REPLACE VIEW opioid_ed_cohort AS
        WITH target_cases AS (
            SELECT DISTINCT mi_person_key
            FROM cohort_event_features
            WHERE event_classification = 'opioid_ed'
        ),
        control_candidates AS (
            SELECT DISTINCT mi_person_key
            FROM cohort_event_features
            WHERE event_classification != 'opioid_ed'
              AND mi_person_key NOT IN (SELECT mi_person_key FROM target_cases)
        ),
        sampled_controls AS (
            SELECT mi_person_key
            FROM control_candidates
            ORDER BY RANDOM()
            LIMIT (SELECT COUNT(*) * 5 FROM target_cases)
        )
        SELECT 
            cef.*,
            1 as target,
            'OPIOID_ED' as cohort_name,
            CASE WHEN tc.mi_person_key IS NOT NULL THEN 1 ELSE 0 END as is_target_case
        FROM cohort_event_features cef
        LEFT JOIN target_cases tc ON cef.mi_person_key = tc.mi_person_key
        LEFT JOIN sampled_controls sc ON cef.mi_person_key = sc.mi_person_key
        WHERE tc.mi_person_key IS NOT NULL OR sc.mi_person_key IS NOT NULL;
        """
        cohort_conn_duckdb.sql(opioid_ed_cohort_sql)
        logger.info("→ [STEP 13] OPIOID_ED cohort view created")
        
        # QA checks
        total_records = cohort_conn_duckdb.sql("SELECT COUNT(*) FROM opioid_ed_cohort").fetchone()[0]
        target_cases = cohort_conn_duckdb.sql("SELECT COUNT(DISTINCT mi_person_key) FROM opioid_ed_cohort WHERE is_target_case = 1").fetchone()[0]
        control_cases = cohort_conn_duckdb.sql("SELECT COUNT(DISTINCT mi_person_key) FROM opioid_ed_cohort WHERE is_target_case = 0").fetchone()[0]
        control_ratio = control_cases / target_cases if target_cases > 0 else 0
        
        logger.info(f"→ [STEP 13] QA: Total records: {total_records:,}")
        logger.info(f"→ [STEP 13] QA: Target cases: {target_cases:,}")
        logger.info(f"→ [STEP 13] QA: Control cases: {control_cases:,}")
        logger.info(f"→ [STEP 13] QA: Control-to-target ratio: {control_ratio:.2f}:1")
        
        # Save to S3
        output_paths = get_output_paths(age_band, event_year, "opioid_ed")
        cohort_conn_duckdb.sql(f"""
        COPY opioid_ed_cohort TO '{output_paths['cohort_data']}' 
        (FORMAT PARQUET, COMPRESSION SNAPPY, PARTITION_BY (cohort_name)
        """)
        logger.info(f"→ [STEP 13] OPIOID_ED cohort saved to S3: {output_paths['cohort_data']}")
        
        # Force checkpoint
        force_checkpoint(cohort_conn_duckdb, logger)
        
        # Disable query profiling
        disable_query_profiling(cohort_conn_duckdb, logger)
        
        logger.info("→ [STEP 13] Optimized OPIOID_ED cohort creation completed")
        
        # Save checkpoint
        save_step_checkpoint(age_band, event_year, "step13_opioid_ed_cohort", {
            "total_records": total_records,
            "target_cases": target_cases,
            "control_cases": control_cases,
            "control_ratio": control_ratio,
            "output_path": output_paths['cohort_data'],
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"→ [STEP 13] OPIOID_ED cohort creation failed: {str(e)}")
        # Clean up temp files on error
        cleanup_duckdb_temp_files(logger)
        raise


def run_step14_ed_non_opioid_cohort_optimized(context):
    """Step 14: ED_NON_OPIOID cohort creation with DuckDB optimizations."""
    logger = context["logger"]
    cohort_conn_duckdb = context["cohort_conn_duckdb"]
    age_band = context["age_band"]
    event_year = context["event_year"]
    
    logger.info("→ [STEP 14] Starting optimized ED_NON_OPIOID cohort creation...")
    
    try:
        # Enable query profiling for this step
        enable_query_profiling(cohort_conn_duckdb, logger, "json", f"/tmp/duckdb_profiling_step14_ed_non_opioid.json")
        
        # Create ED_NON_OPIOID cohort with 5:1 control-to-target ratio
        ed_non_opioid_cohort_sql = f"""
        CREATE OR REPLACE VIEW ed_non_opioid_cohort AS
        WITH target_cases AS (
            SELECT DISTINCT mi_person_key
            FROM cohort_event_features
            WHERE event_classification = 'ed_non_opioid'
        ),
        control_candidates AS (
            SELECT DISTINCT mi_person_key
            FROM cohort_event_features
            WHERE event_classification != 'ed_non_opioid'
              AND mi_person_key NOT IN (SELECT mi_person_key FROM target_cases)
        ),
        sampled_controls AS (
            SELECT mi_person_key
            FROM control_candidates
            ORDER BY RANDOM()
            LIMIT (SELECT COUNT(*) * 5 FROM target_cases)
        )
        SELECT 
            cef.*,
            1 as target,
            'ED_NON_OPIOID' as cohort_name,
            CASE WHEN tc.mi_person_key IS NOT NULL THEN 1 ELSE 0 END as is_target_case
        FROM cohort_event_features cef
        LEFT JOIN target_cases tc ON cef.mi_person_key = tc.mi_person_key
        LEFT JOIN sampled_controls sc ON cef.mi_person_key = sc.mi_person_key
        WHERE tc.mi_person_key IS NOT NULL OR sc.mi_person_key IS NOT NULL;
        """
        cohort_conn_duckdb.sql(ed_non_opioid_cohort_sql)
        logger.info("→ [STEP 14] ED_NON_OPIOID cohort view created")
        
        # QA checks
        total_records = cohort_conn_duckdb.sql("SELECT COUNT(*) FROM ed_non_opioid_cohort").fetchone()[0]
        target_cases = cohort_conn_duckdb.sql("SELECT COUNT(DISTINCT mi_person_key) FROM ed_non_opioid_cohort WHERE is_target_case = 1").fetchone()[0]
        control_cases = cohort_conn_duckdb.sql("SELECT COUNT(DISTINCT mi_person_key) FROM ed_non_opioid_cohort WHERE is_target_case = 0").fetchone()[0]
        control_ratio = control_cases / target_cases if target_cases > 0 else 0
        
        logger.info(f"→ [STEP 14] QA: Total records: {total_records:,}")
        logger.info(f"→ [STEP 14] QA: Target cases: {target_cases:,}")
        logger.info(f"→ [STEP 14] QA: Control cases: {control_cases:,}")
        logger.info(f"→ [STEP 14] QA: Control-to-target ratio: {control_ratio:.2f}:1")
        
        # Save to S3
        output_paths = get_output_paths(age_band, event_year, "ed_non_opioid")
        cohort_conn_duckdb.sql(f"""
        COPY ed_non_opioid_cohort TO '{output_paths['cohort_data']}' 
        (FORMAT PARQUET, COMPRESSION SNAPPY, PARTITION_BY (cohort_name)
        """)
        logger.info(f"→ [STEP 14] ED_NON_OPIOID cohort saved to S3: {output_paths['cohort_data']}")
        
        # Force checkpoint
        force_checkpoint(cohort_conn_duckdb, logger)
        
        # Disable query profiling
        disable_query_profiling(cohort_conn_duckdb, logger)
        
        logger.info("→ [STEP 14] Optimized ED_NON_OPIOID cohort creation completed")
        
        # Save checkpoint
        save_step_checkpoint(age_band, event_year, "step14_ed_non_opioid_cohort", {
            "total_records": total_records,
            "target_cases": target_cases,
            "control_cases": control_cases,
            "control_ratio": control_ratio,
            "output_path": output_paths['cohort_data'],
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"→ [STEP 14] ED_NON_OPIOID cohort creation failed: {str(e)}")
        # Clean up temp files on error
        cleanup_duckdb_temp_files(logger)
        raise


def run_phase1_data_preparation(context):
    """Phase 1: Data Preparation with DuckDB optimizations."""
    logger = context["logger"]
    cohort_conn_duckdb = context["cohort_conn_duckdb"]
    age_band = context["age_band"]
    event_year = context["event_year"]
    pipeline_state = context.get("pipeline_state")
    
    step_name = "phase1_data_preparation"
    
    # Check if step already completed
    if pipeline_state and pipeline_state.is_step_completed(step_name):
        logger.info(f"{SYMBOLS['success']} [PHASE 1] Already completed - skipping")
        return
    
    logger.info(f"{SYMBOLS['arrow']} [PHASE 1] Starting optimized data preparation (APCD Integration)...")
    
    try:
        # Enable query profiling for this phase
        enable_query_profiling(cohort_conn_duckdb, logger, "json", f"/tmp/duckdb_profiling_phase1_data_preparation.json")
        
        # Load demographics lookup from APCD silver tier
        demographics_sql = f"""
        CREATE OR REPLACE VIEW demographics_lookup AS
        SELECT *
        FROM read_parquet('s3://pgxdatalake/silver/imputed/mi_person_key_demographics_lookup.parquet')
        WHERE event_year = {event_year};
        """
        cohort_conn_duckdb.sql(demographics_sql)
        logger.info("→ [PHASE 1] Demographics lookup loaded")
        
        # Load pre-imputed medical data from APCD gold tier
        medical_sql = f"""
        CREATE OR REPLACE VIEW medical_raw AS
        SELECT *
        FROM read_parquet_auto('s3://pgxdatalake/gold/medical/age_band={age_band}/event_year={event_year}/*.parquet')
        WHERE age_band = '{age_band}'
          AND event_year = {event_year};
        """
        cohort_conn_duckdb.sql(medical_sql)
        logger.info("→ [PHASE 1] Medical raw data loaded from APCD gold tier")
        
        # Create medical view with pre-imputed demographics
        medical_demographics_sql = f"""
        CREATE OR REPLACE VIEW medical_with_demographics AS
        SELECT 
            m.*,
            COALESCE(m.age_imputed, dl.age_imputed) AS age_imputed,
            COALESCE(m.gender_imputed, dl.gender_imputed) AS gender_imputed,
            COALESCE(m.race_imputed, dl.race_imputed) AS race_imputed,
            COALESCE(m.zip_imputed, dl.zip_imputed) AS zip_imputed,
            COALESCE(m.county_imputed, dl.county_imputed) AS county_imputed,
            COALESCE(m.payer_imputed, dl.payer_imputed) AS payer_imputed
        FROM medical_raw m
        LEFT JOIN demographics_lookup dl ON m.mi_person_key = dl.mi_person_key 
            AND m.event_year = dl.event_year;
        """
        cohort_conn_duckdb.sql(medical_demographics_sql)
        logger.info("→ [PHASE 1] Medical demographics view created")
        
        # Apply data quality filters for medical data
        medical_filtered_sql = f"""
        CREATE OR REPLACE VIEW medical_clean AS
        SELECT *
        FROM medical_with_demographics
        WHERE mi_person_key IS NOT NULL
          AND mi_person_key != ''
          AND event_date IS NOT NULL
          AND age_imputed IS NOT NULL
          AND age_imputed BETWEEN 1 AND 114
          AND event_date >= '{event_year}-01-01'
          AND event_date <= '{event_year}-12-31';
        """
        cohort_conn_duckdb.sql(medical_filtered_sql)
        logger.info("→ [PHASE 1] Medical data filtered and cleaned")
        
        # Load pre-imputed pharmacy data from APCD gold tier
        pharmacy_sql = f"""
        CREATE OR REPLACE VIEW pharmacy_raw AS
        SELECT *
        FROM read_parquet_auto('s3://pgxdatalake/gold/pharmacy/age_band={age_band}/event_year={event_year}/*.parquet')
        WHERE age_band = '{age_band}'
          AND event_year = {event_year};
        """
        cohort_conn_duckdb.sql(pharmacy_sql)
        logger.info("→ [PHASE 1] Pharmacy raw data loaded from APCD gold tier")
        
        # Create pharmacy view with pre-imputed demographics
        pharmacy_demographics_sql = f"""
        CREATE OR REPLACE VIEW pharmacy_with_demographics AS
        SELECT 
            p.*,
            COALESCE(p.age_imputed, dl.age_imputed) AS age_imputed,
            COALESCE(p.gender_imputed, dl.gender_imputed) AS gender_imputed,
            COALESCE(p.race_imputed, dl.race_imputed) AS race_imputed,
            COALESCE(p.zip_imputed, dl.zip_imputed) AS zip_imputed,
            COALESCE(p.county_imputed, dl.county_imputed) AS county_imputed,
            COALESCE(p.payer_imputed, dl.payer_imputed) AS payer_imputed
        FROM pharmacy_raw p
        LEFT JOIN demographics_lookup dl ON p.mi_person_key = dl.mi_person_key 
            AND p.event_year = dl.event_year;
        """
        cohort_conn_duckdb.sql(pharmacy_demographics_sql)
        logger.info("→ [PHASE 1] Pharmacy demographics view created")
        
        # Apply data quality filters for pharmacy data
        pharmacy_filtered_sql = f"""
        CREATE OR REPLACE VIEW pharmacy_clean AS
        SELECT *
        FROM pharmacy_with_demographics
        WHERE mi_person_key IS NOT NULL
          AND mi_person_key != ''
          AND event_date IS NOT NULL
          AND age_imputed IS NOT NULL
          AND age_imputed BETWEEN 1 AND 114
          AND event_date >= '{event_year}-01-01'
          AND event_date <= '{event_year}-12-31'
          AND drug_name IS NOT NULL
          AND drug_name != '';
        """
        cohort_conn_duckdb.sql(pharmacy_filtered_sql)
        logger.info("→ [PHASE 1] Pharmacy data filtered and cleaned")
        
        # QA checks
        medical_count = cohort_conn_duckdb.sql("SELECT COUNT(*) FROM medical_clean").fetchone()[0]
        pharmacy_count = cohort_conn_duckdb.sql("SELECT COUNT(*) FROM pharmacy_clean").fetchone()[0]
        
        logger.info(f"→ [PHASE 1] QA: Medical records: {medical_count:,}")
        logger.info(f"→ [PHASE 1] QA: Pharmacy records: {pharmacy_count:,}")
        
        # Force checkpoint
        force_checkpoint(cohort_conn_duckdb, logger)
        
        # Disable query profiling
        disable_query_profiling(cohort_conn_duckdb, logger)
        
        # Save checkpoint
        if pipeline_state:
            pipeline_state.mark_step_completed(step_name, {
                'medical_records': medical_count,
                'pharmacy_records': pharmacy_count,
                'timestamp': datetime.now().isoformat()
            })
        
        logger.info(f"{SYMBOLS['success']} [PHASE 1] Optimized data preparation completed")
        
    except Exception as e:
        logger.error(f"{SYMBOLS['fail']} [PHASE 1] Data preparation failed: {str(e)}")
        if pipeline_state:
            pipeline_state.mark_step_failed(step_name, str(e))
        cleanup_duckdb_temp_files(logger)
        raise


def run_phase2_step1_event_fact_table(context):
    """Phase 2 Step 1: Event Fact Table Creation with DuckDB optimizations."""
    logger = context["logger"]
    cohort_conn_duckdb = context["cohort_conn_duckdb"]
    age_band = context["age_band"]
    event_year = context["event_year"]
    pipeline_state = context.get("pipeline_state")
    
    step_name = "phase2_step1_event_fact_table"
    
    # Check if step already completed
    if pipeline_state and pipeline_state.is_step_completed(step_name):
        logger.info(f"{SYMBOLS['success']} [PHASE 2 STEP 1] Already completed - skipping")
        return
    
    logger.info(f"{SYMBOLS['arrow']} [PHASE 2 STEP 1] Starting optimized event fact table creation...")
    
    try:
        # Enable query profiling for this step
        enable_query_profiling(cohort_conn_duckdb, logger, "json", f"/tmp/duckdb_profiling_phase2_step1_event_fact_table.json")
        
        # Create unified event fact table
        event_fact_table_sql = f"""
        CREATE OR REPLACE VIEW unified_event_fact_table AS
        SELECT 
            mi_person_key,
            event_date,
            'medical' as event_type,
            'medical' as data_source,
            age_imputed,
            gender_imputed as member_gender,
            race_imputed as member_race,
            zip_imputed,
            county_imputed,
            payer_imputed,
            primary_icd_diagnosis_code,
            NULL as drug_name,
            NULL as therapeutic_class_1,
            -- Event classification
            CASE 
                WHEN primary_icd_diagnosis_code IN {tuple(OPIOID_ICD_CODES)} THEN 'opioid_ed'
                ELSE 'ed_non_opioid'
            END as event_classification,
            -- First event flags
            ROW_NUMBER() OVER (PARTITION BY mi_person_key ORDER BY event_date) as event_sequence
        FROM medical_clean
        WHERE primary_icd_diagnosis_code IS NOT NULL
        
        UNION ALL
        
        SELECT 
            mi_person_key,
            event_date,
            'pharmacy' as event_type,
            'pharmacy' as data_source,
            age_imputed,
            gender_imputed as member_gender,
            race_imputed as member_race,
            zip_imputed,
            county_imputed,
            payer_imputed,
            NULL as primary_icd_diagnosis_code,
            drug_name,
            therapeutic_class_1,
            'pharmacy' as event_classification,
            ROW_NUMBER() OVER (PARTITION BY mi_person_key ORDER BY event_date) as event_sequence
        FROM pharmacy_clean
        WHERE drug_name IS NOT NULL;
        """
        cohort_conn_duckdb.sql(event_fact_table_sql)
        logger.info("→ [PHASE 2 STEP 1] Unified event fact table created")
        
        # QA checks
        total_events = cohort_conn_duckdb.sql("SELECT COUNT(*) FROM unified_event_fact_table").fetchone()[0]
        event_type_dist = cohort_conn_duckdb.sql("""
        SELECT event_type, COUNT(*) as count
        FROM unified_event_fact_table
        GROUP BY event_type
        ORDER BY count DESC
        """).fetchall()
        
        logger.info(f"→ [PHASE 2 STEP 1] QA: Total events: {total_events:,}")
        logger.info(f"→ [PHASE 2 STEP 1] QA: Event type distribution: {dict(event_type_dist)}")
        
        # Force checkpoint
        force_checkpoint(cohort_conn_duckdb, logger)
        
        # Disable query profiling
        disable_query_profiling(cohort_conn_duckdb, logger)
        
        # Save checkpoint
        if pipeline_state:
            pipeline_state.mark_step_completed(step_name, {
                'total_events': total_events,
                'event_types': dict(event_type_dist),
                'timestamp': datetime.now().isoformat()
            })
        
        logger.info(f"{SYMBOLS['success']} [PHASE 2 STEP 1] Optimized event fact table creation completed")
        
    except Exception as e:
        logger.error(f"{SYMBOLS['fail']} [PHASE 2 STEP 1] Event fact table creation failed: {str(e)}")
        if pipeline_state:
            pipeline_state.mark_step_failed(step_name, str(e))
        cleanup_duckdb_temp_files(logger)
        raise


def run_phase2_step2_drug_exposure(context):
    """Phase 2 Step 2: Drug Exposure Events with DuckDB optimizations."""
    logger = context["logger"]
    cohort_conn_duckdb = context["cohort_conn_duckdb"]
    age_band = context["age_band"]
    event_year = context["event_year"]
    pipeline_state = context.get("pipeline_state")
    
    step_name = "phase2_step2_drug_exposure"
    
    # Check if step already completed
    if pipeline_state and pipeline_state.is_step_completed(step_name):
        logger.info(f"{SYMBOLS['success']} [PHASE 2 STEP 2] Already completed - skipping")
        return
    
    logger.info(f"{SYMBOLS['arrow']} [PHASE 2 STEP 2] Starting optimized drug exposure events creation...")
    
    try:
        # Enable query profiling for this step
        enable_query_profiling(cohort_conn_duckdb, logger, "json", f"/tmp/duckdb_profiling_phase2_step2_drug_exposure.json")
        
        # Create unified drug exposure view
        drug_exposure_sql = f"""
        CREATE OR REPLACE VIEW unified_drug_exposure AS
        SELECT 
            mi_person_key,
            event_date,
            drug_name,
            therapeutic_class_1,
            age_imputed,
            gender_imputed as member_gender,
            race_imputed as member_race,
            zip_imputed,
            county_imputed,
            payer_imputed,
            -- Calculate days to target event
            NULL as days_to_target_event
        FROM pharmacy_clean
        WHERE drug_name IS NOT NULL
          AND drug_name != '';
        """
        cohort_conn_duckdb.sql(drug_exposure_sql)
        logger.info("→ [PHASE 2 STEP 2] Unified drug exposure view created")
        
        # QA checks
        total_drug_events = cohort_conn_duckdb.sql("SELECT COUNT(*) FROM unified_drug_exposure").fetchone()[0]
        
        logger.info(f"→ [PHASE 2 STEP 2] QA: Total drug exposure events: {total_drug_events:,}")
        
        # Force checkpoint
        force_checkpoint(cohort_conn_duckdb, logger)
        
        # Disable query profiling
        disable_query_profiling(cohort_conn_duckdb, logger)
        
        # Save checkpoint
        if pipeline_state:
            pipeline_state.mark_step_completed(step_name, {
                'total_drug_events': total_drug_events,
                'timestamp': datetime.now().isoformat()
            })
        
        logger.info(f"{SYMBOLS['success']} [PHASE 2 STEP 2] Optimized drug exposure events creation completed")
        
    except Exception as e:
        logger.error(f"{SYMBOLS['fail']} [PHASE 2 STEP 2] Drug exposure events creation failed: {str(e)}")
        if pipeline_state:
            pipeline_state.mark_step_failed(step_name, str(e))
        cleanup_duckdb_temp_files(logger)
        raise


def run_phase3_step3_final_cohort_fact(context):
    """Phase 3 Step 3: Final Cohort Creation with 5:1 ratio and DuckDB optimizations."""
    logger = context["logger"]
    cohort_conn_duckdb = context["cohort_conn_duckdb"]
    age_band = context["age_band"]
    event_year = context["event_year"]
    pipeline_state = context.get("pipeline_state")
    
    step_name = "phase3_step3_final_cohort_fact"
    
    # Check if step already completed
    if pipeline_state and pipeline_state.is_step_completed(step_name):
        logger.info(f"{SYMBOLS['success']} [PHASE 3 STEP 3] Already completed - skipping")
        return
    
    logger.info(f"{SYMBOLS['arrow']} [PHASE 3 STEP 3] Starting optimized final cohort creation (5:1 ratio)...")
    
    try:
        # Enable query profiling for this step
        enable_query_profiling(cohort_conn_duckdb, logger, "json", f"/tmp/duckdb_profiling_phase3_step3_final_cohort_fact.json")
        
        # Create OPIOID_ED cohort with 5:1 control-to-target ratio
        opioid_ed_cohort_sql = f"""
        CREATE OR REPLACE VIEW opioid_ed_cohort AS
        WITH target_cases AS (
            SELECT DISTINCT mi_person_key
            FROM unified_event_fact_table
            WHERE event_classification = 'opioid_ed'
        ),
        control_candidates AS (
            SELECT DISTINCT mi_person_key
            FROM unified_event_fact_table
            WHERE event_classification != 'opioid_ed'
              AND mi_person_key NOT IN (SELECT mi_person_key FROM target_cases)
        ),
        sampled_controls AS (
            SELECT mi_person_key
            FROM control_candidates
            ORDER BY RANDOM()
            LIMIT (SELECT COUNT(*) * 5 FROM target_cases)
        )
        SELECT 
            uef.*,
            1 as target,
            'OPIOID_ED' as cohort_name,
            CASE WHEN tc.mi_person_key IS NOT NULL THEN 1 ELSE 0 END as is_target_case
        FROM unified_event_fact_table uef
        LEFT JOIN target_cases tc ON uef.mi_person_key = tc.mi_person_key
        LEFT JOIN sampled_controls sc ON uef.mi_person_key = sc.mi_person_key
        WHERE tc.mi_person_key IS NOT NULL OR sc.mi_person_key IS NOT NULL;
        """
        cohort_conn_duckdb.sql(opioid_ed_cohort_sql)
        logger.info("→ [PHASE 3 STEP 3] OPIOID_ED cohort created")
        
        # Create ED_NON_OPIOID cohort with 5:1 control-to-target ratio
        ed_non_opioid_cohort_sql = f"""
        CREATE OR REPLACE VIEW ed_non_opioid_cohort AS
        WITH target_cases AS (
            SELECT DISTINCT mi_person_key
            FROM unified_event_fact_table
            WHERE event_classification = 'ed_non_opioid'
        ),
        control_candidates AS (
            SELECT DISTINCT mi_person_key
            FROM unified_event_fact_table
            WHERE event_classification != 'ed_non_opioid'
              AND mi_person_key NOT IN (SELECT mi_person_key FROM target_cases)
        ),
        sampled_controls AS (
            SELECT mi_person_key
            FROM control_candidates
            ORDER BY RANDOM()
            LIMIT (SELECT COUNT(*) * 5 FROM target_cases)
        )
        SELECT 
            uef.*,
            1 as target,
            'ED_NON_OPIOID' as cohort_name,
            CASE WHEN tc.mi_person_key IS NOT NULL THEN 1 ELSE 0 END as is_target_case
        FROM unified_event_fact_table uef
        LEFT JOIN target_cases tc ON uef.mi_person_key = tc.mi_person_key
        LEFT JOIN sampled_controls sc ON uef.mi_person_key = sc.mi_person_key
        WHERE tc.mi_person_key IS NOT NULL OR sc.mi_person_key IS NOT NULL;
        """
        cohort_conn_duckdb.sql(ed_non_opioid_cohort_sql)
        logger.info("→ [PHASE 3 STEP 3] ED_NON_OPIOID cohort created")
        
        # QA checks
        opioid_ed_count = cohort_conn_duckdb.sql("SELECT COUNT(*) FROM opioid_ed_cohort").fetchone()[0]
        ed_non_opioid_count = cohort_conn_duckdb.sql("SELECT COUNT(*) FROM ed_non_opioid_cohort").fetchone()[0]
        
        # Check control ratios
        opioid_ed_ratio = cohort_conn_duckdb.sql("""
        SELECT 
            COUNT(DISTINCT CASE WHEN is_target_case = 1 THEN mi_person_key END) as target_cases,
            COUNT(DISTINCT CASE WHEN is_target_case = 0 THEN mi_person_key END) as control_cases
        FROM opioid_ed_cohort
        """).fetchone()
        
        ed_non_opioid_ratio = cohort_conn_duckdb.sql("""
        SELECT 
            COUNT(DISTINCT CASE WHEN is_target_case = 1 THEN mi_person_key END) as target_cases,
            COUNT(DISTINCT CASE WHEN is_target_case = 0 THEN mi_person_key END) as control_cases
        FROM ed_non_opioid_cohort
        """).fetchone()
        
        opioid_ed_control_ratio = opioid_ed_ratio[1] / opioid_ed_ratio[0] if opioid_ed_ratio[0] > 0 else 0
        ed_non_opioid_control_ratio = ed_non_opioid_ratio[1] / ed_non_opioid_ratio[0] if ed_non_opioid_ratio[0] > 0 else 0
        
        logger.info(f"→ [PHASE 3 STEP 3] QA: OPIOID_ED records: {opioid_ed_count:,}")
        logger.info(f"→ [PHASE 3 STEP 3] QA: ED_NON_OPIOID records: {ed_non_opioid_count:,}")
        logger.info(f"→ [PHASE 3 STEP 3] QA: OPIOID_ED control ratio: {opioid_ed_control_ratio:.2f}:1")
        logger.info(f"→ [PHASE 3 STEP 3] QA: ED_NON_OPIOID control ratio: {ed_non_opioid_control_ratio:.2f}:1")
        
        # Force checkpoint
        force_checkpoint(cohort_conn_duckdb, logger)
        
        # Disable query profiling
        disable_query_profiling(cohort_conn_duckdb, logger)
        
        # Save checkpoint
        if pipeline_state:
            pipeline_state.mark_step_completed(step_name, {
                'opioid_ed_count': opioid_ed_count,
                'ed_non_opioid_count': ed_non_opioid_count,
                'opioid_ed_control_ratio': float(opioid_ed_control_ratio),
                'ed_non_opioid_control_ratio': float(ed_non_opioid_control_ratio),
                'timestamp': datetime.now().isoformat()
            })
        
        logger.info(f"{SYMBOLS['success']} [PHASE 3 STEP 3] Optimized final cohort creation completed")
        
    except Exception as e:
        logger.error(f"{SYMBOLS['fail']} [PHASE 3 STEP 3] Final cohort creation failed: {str(e)}")
        if pipeline_state:
            pipeline_state.mark_step_failed(step_name, str(e))
        cleanup_duckdb_temp_files(logger)
        raise


def run_phase4_complete_pipeline(context):
    """Phase 4: Complete Pipeline with DuckDB optimizations."""
    logger = context["logger"]
    cohort_conn_duckdb = context["cohort_conn_duckdb"]
    age_band = context["age_band"]
    event_year = context["event_year"]
    pipeline_state = context.get("pipeline_state")
    
    step_name = "phase4_complete_pipeline"
    
    # Check if step already completed
    if pipeline_state and pipeline_state.is_step_completed(step_name):
        logger.info(f"{SYMBOLS['success']} [PHASE 4] Already completed - skipping")
        return
    
    logger.info(f"{SYMBOLS['arrow']} [PHASE 4] Starting optimized complete pipeline execution...")
    
    try:
        # Enable query profiling for this phase
        enable_query_profiling(cohort_conn_duckdb, logger, "json", f"/tmp/duckdb_profiling_phase4_complete_pipeline.json")
        
        # Final QA validation
        logger.info("→ [PHASE 4] Performing final QA validation...")
        
        # Check both cohorts exist
        opioid_ed_count = cohort_conn_duckdb.sql("SELECT COUNT(*) FROM opioid_ed_cohort").fetchone()[0]
        ed_non_opioid_count = cohort_conn_duckdb.sql("SELECT COUNT(*) FROM ed_non_opioid_cohort").fetchone()[0]
        
        logger.info(f"→ [PHASE 4] QA: OPIOID_ED cohort records: {opioid_ed_count:,}")
        logger.info(f"→ [PHASE 4] QA: ED_NON_OPIOID cohort records: {ed_non_opioid_count:,}")
        
        # Save to S3
        from helpers.s3_utils import get_output_paths
        
        # Save OPIOID_ED cohort
        opioid_ed_paths = get_output_paths(age_band, event_year, "opioid_ed")
        cohort_conn_duckdb.sql(f"""
        COPY opioid_ed_cohort TO '{opioid_ed_paths['cohort_data']}' 
        (FORMAT PARQUET, COMPRESSION SNAPPY, PARTITION_BY (cohort_name))
        """)
        logger.info(f"→ [PHASE 4] OPIOID_ED cohort saved to S3: {opioid_ed_paths['cohort_data']}")
        
        # Save ED_NON_OPIOID cohort
        ed_non_opioid_paths = get_output_paths(age_band, event_year, "ed_non_opioid")
        cohort_conn_duckdb.sql(f"""
        COPY ed_non_opioid_cohort TO '{ed_non_opioid_paths['cohort_data']}' 
        (FORMAT PARQUET, COMPRESSION SNAPPY, PARTITION_BY (cohort_name))
        """)
        logger.info(f"→ [PHASE 4] ED_NON_OPIOID cohort saved to S3: {ed_non_opioid_paths['cohort_data']}")
        
        # Final cleanup
        cleanup_duckdb_temp_files(logger)
        monitor_disk_space(logger)
        
        # Force checkpoint
        force_checkpoint(cohort_conn_duckdb, logger)
        
        # Disable query profiling
        disable_query_profiling(cohort_conn_duckdb, logger)
        
        # Save checkpoint
        if pipeline_state:
            pipeline_state.mark_step_completed(step_name, {
                'opioid_ed_count': opioid_ed_count,
                'ed_non_opioid_count': ed_non_opioid_count,
                'timestamp': datetime.now().isoformat()
            })
        
        logger.info(f"{SYMBOLS['success']} [PHASE 4] Optimized complete pipeline execution finished")
        
    except Exception as e:
        logger.error(f"{SYMBOLS['fail']} [PHASE 4] Complete pipeline execution failed: {str(e)}")
        if pipeline_state:
            pipeline_state.mark_step_failed(step_name, str(e))
        cleanup_duckdb_temp_files(logger)
        raise


def run_step15_pipeline_complete_optimized(context):
    """Step 15: Pipeline completion with DuckDB optimizations."""
    logger = context["logger"]
    cohort_conn_duckdb = context["cohort_conn_duckdb"]
    age_band = context["age_band"]
    event_year = context["event_year"]
    
    logger.info("→ [STEP 15] Starting optimized pipeline completion...")
    
    try:
        # Final QA validation
        logger.info("→ [STEP 15] Performing final QA validation...")
        
        # Check both cohorts exist
        opioid_ed_count = cohort_conn_duckdb.sql("SELECT COUNT(*) FROM opioid_ed_cohort").fetchone()[0]
        ed_non_opioid_count = cohort_conn_duckdb.sql("SELECT COUNT(*) FROM ed_non_opioid_cohort").fetchone()[0]
        
        logger.info(f"→ [STEP 15] QA: OPIOID_ED cohort records: {opioid_ed_count:,}")
        logger.info(f"→ [STEP 15] QA: ED_NON_OPIOID cohort records: {ed_non_opioid_count:,}")
        
        # Validate control ratios
        opioid_ed_ratio = cohort_conn_duckdb.sql("""
        SELECT 
            COUNT(DISTINCT CASE WHEN is_target_case = 1 THEN mi_person_key END) as target_cases,
            COUNT(DISTINCT CASE WHEN is_target_case = 0 THEN mi_person_key END) as control_cases
        FROM opioid_ed_cohort
        """).fetchone()
        
        ed_non_opioid_ratio = cohort_conn_duckdb.sql("""
        SELECT 
            COUNT(DISTINCT CASE WHEN is_target_case = 1 THEN mi_person_key END) as target_cases,
            COUNT(DISTINCT CASE WHEN is_target_case = 0 THEN mi_person_key END) as control_cases
        FROM ed_non_opioid_cohort
        """).fetchone()
        
        opioid_ed_control_ratio = opioid_ed_ratio[1] / opioid_ed_ratio[0] if opioid_ed_ratio[0] > 0 else 0
        ed_non_opioid_control_ratio = ed_non_opioid_ratio[1] / ed_non_opioid_ratio[0] if ed_non_opioid_ratio[0] > 0 else 0
        
        logger.info(f"→ [STEP 15] QA: OPIOID_ED control ratio: {opioid_ed_control_ratio:.2f}:1")
        logger.info(f"→ [STEP 15] QA: ED_NON_OPIOID control ratio: {ed_non_opioid_control_ratio:.2f}:1")
        
        # Final cleanup
        cleanup_duckdb_temp_files(logger)
        monitor_disk_space(logger)
        
        logger.info("→ [STEP 15] Optimized pipeline completion finished")
        
        # Save final checkpoint
        save_step_checkpoint(age_band, event_year, "step15_pipeline_complete", {
            "opioid_ed_records": opioid_ed_count,
            "ed_non_opioid_records": ed_non_opioid_count,
            "opioid_ed_control_ratio": opioid_ed_control_ratio,
            "ed_non_opioid_control_ratio": ed_non_opioid_control_ratio,
            "pipeline_completed": True,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"→ [STEP 15] Pipeline completion failed: {str(e)}")
        # Clean up temp files on error
        cleanup_duckdb_temp_files(logger)
        raise
