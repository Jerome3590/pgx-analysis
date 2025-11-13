"""
Phase 1: Data Preparation with DuckDB optimizations.

Loads and filters medical and pharmacy data from APCD gold tier.
"""

from .common import (
    datetime,
    SYMBOLS,
    cleanup_duckdb_temp_files,
    enable_query_profiling,
    disable_query_profiling,
    force_checkpoint,
    execute_sql_with_dev_validation,
)


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
        
        # Use GOLD final tables to create cohort inputs (preferred source)
        # Map gold medical columns to expected normalized names
        medical_sql = f"""
        CREATE OR REPLACE VIEW medical_base AS
        SELECT
            CAST(mi_person_key AS VARCHAR) AS mi_person_key,
            -- Map gold medical fields to normalized names used downstream
            member_age_dos AS age_imputed,
            member_gender AS gender_imputed,
            member_race AS race_imputed,
            member_zip_code_dos AS zip_imputed,
            member_county_dos AS county_imputed,
            payer_type AS payer_imputed,
            primary_icd_diagnosis_code,
            -- Carry forward CPT/procedure fields for event features
            procedure_code,
            cpt_mod_1_code,
            cpt_mod_2_code,
            -- HCG fields for ED visit identification
            hcg_setting,
            hcg_line,
            hcg_detail,
            event_date,
            CAST(event_year AS INTEGER) AS event_year
        FROM read_parquet('s3://pgxdatalake/gold/medical/age_band={age_band}/event_year={event_year}/medical_data.parquet')
        WHERE mi_person_key IS NOT NULL
          AND CAST(mi_person_key AS VARCHAR) <> ''
          AND event_date IS NOT NULL;
        """
        execute_sql_with_dev_validation(cohort_conn_duckdb, logger, medical_sql)
        logger.info("→ [PHASE 1] Medical data loaded from GOLD final table")
        
        # Apply additional medical filters into final view 'medical'
        medical_filtered_sql = f"""
        CREATE OR REPLACE VIEW medical AS
        SELECT *
        FROM medical_base
        WHERE age_imputed IS NOT NULL
          AND age_imputed BETWEEN 1 AND 114
          AND event_date >= '{event_year}-01-01'
          AND event_date <= '{event_year}-12-31';
        """
        execute_sql_with_dev_validation(cohort_conn_duckdb, logger, medical_filtered_sql)
        logger.info("→ [PHASE 1] Medical data filtered and cleaned")
        
        # Pharmacy: use GOLD final table; demographics may be absent -> set to NULLs where not present
        pharmacy_sql = f"""
        CREATE OR REPLACE VIEW pharmacy_base AS
        SELECT 
            CAST(mi_person_key AS VARCHAR) AS mi_person_key,
            NULL::INTEGER AS age_imputed,
            NULL::VARCHAR AS gender_imputed,
            NULL::VARCHAR AS race_imputed,
            NULL::VARCHAR AS zip_imputed,
            NULL::VARCHAR AS county_imputed,
            NULL::VARCHAR AS payer_imputed,
            drug_name,
            NULL::VARCHAR AS therapeutic_class_1,
            -- Build event_date here from incurred_date for cohort processing
            TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') AS event_date,
            CAST(event_year AS INTEGER) AS event_year
        FROM read_parquet('s3://pgxdatalake/gold/pharmacy/age_band={age_band}/event_year={event_year}/pharmacy_data.parquet')
        WHERE mi_person_key IS NOT NULL
          AND CAST(mi_person_key AS VARCHAR) <> ''
          AND incurred_date IS NOT NULL
          AND TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') IS NOT NULL;
        """
        execute_sql_with_dev_validation(cohort_conn_duckdb, logger, pharmacy_sql)
        logger.info("→ [PHASE 1] Pharmacy data loaded from GOLD final table")
        
        # Apply additional pharmacy filters into final view 'pharmacy'
        pharmacy_filtered_sql = f"""
        CREATE OR REPLACE VIEW pharmacy AS
        SELECT *
        FROM pharmacy_base
        WHERE event_date IS NOT NULL
          AND event_date >= '{event_year}-01-01'
          AND event_date <= '{event_year}-12-31'
          AND drug_name IS NOT NULL
          AND drug_name <> '';
        """
        execute_sql_with_dev_validation(cohort_conn_duckdb, logger, pharmacy_filtered_sql)
        logger.info("→ [PHASE 1] Pharmacy data filtered and cleaned")
        
        # QA checks
        medical_count = cohort_conn_duckdb.sql("SELECT COUNT(*) FROM medical").fetchone()[0]
        pharmacy_count = cohort_conn_duckdb.sql("SELECT COUNT(*) FROM pharmacy").fetchone()[0]
        
        logger.info(f"→ [PHASE 1] QA: Medical records: {medical_count:,}")
        logger.info(f"→ [PHASE 1] QA: Pharmacy records: {pharmacy_count:,}")
        
        # F1120-specific check in raw medical data
        f1120_medical = cohort_conn_duckdb.sql("""
        SELECT 
            COUNT(*) as total_f1120_records,
            COUNT(DISTINCT mi_person_key) as distinct_f1120_patients
        FROM medical
        WHERE primary_icd_diagnosis_code = 'F1120'
        """).fetchone()
        
        if f1120_medical and f1120_medical[0] > 0:
            logger.info(f"→ [PHASE 1] F1120 CHECK in medical data:")
            logger.info(f"  Total F1120 records: {f1120_medical[0]:,}")
            logger.info(f"  Distinct F1120 patients: {f1120_medical[1]:,}")
        else:
            logger.warning(f"→ [PHASE 1] F1120 CHECK: No F1120 records found in medical data")
        
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

