"""
Phase 1: Data Preparation with DuckDB optimizations.

Loads and filters medical and pharmacy data from APCD gold tier.

IMPORTANT: Phase 1 is the CANONICAL source for medical/pharmacy view definitions.
The `ensure_gold_views()` function in common.py is a FALLBACK used when Phase 1 is skipped.
If Phase 1 is skipped, ensure_gold_views() should match Phase 1 logic exactly to prevent
silent cohort semantic drift. For production runs, always run Phase 1.
"""

from .common import (
    datetime,
    SYMBOLS,
    cleanup_duckdb_temp_files,
    enable_query_profiling,
    disable_query_profiling,
    force_checkpoint,
    execute_sql_with_dev_validation,
    check_gold_data_available,
    get_gold_data_paths,
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
        # Enable query profiling for this phase (partition-safe filename)
        enable_query_profiling(cohort_conn_duckdb, logger, "json", f"/tmp/duckdb_profile_p1_{age_band}_{event_year}.json")
        
        # Resolve paths to gold medical/pharmacy (for 85-114 may be two partitions: 85-94 + 95-114)
        if not check_gold_data_available("medical", age_band, event_year):
            raise FileNotFoundError(
                f"Gold medical data not found for age_band={age_band}, event_year={event_year}. "
                f"Ensure a single 85-114 partition or both 85-94 and 95-114 exist on S3."
            )
        if not check_gold_data_available("pharmacy", age_band, event_year):
            raise FileNotFoundError(
                f"Gold pharmacy data not found for age_band={age_band}, event_year={event_year}. "
                f"Ensure a single 85-114 partition or both 85-94 and 95-114 exist on S3."
            )
        medical_paths = get_gold_data_paths("medical", age_band, event_year)
        pharmacy_paths = get_gold_data_paths("pharmacy", age_band, event_year)
        if not medical_paths or not pharmacy_paths:
            raise FileNotFoundError(f"Gold paths resolved empty for age_band={age_band}, event_year={event_year}")
        logger.info(f"→ [PHASE 1] Medical data path(s): {medical_paths}")
        logger.info(f"→ [PHASE 1] Pharmacy data path(s): {pharmacy_paths}")

        def _parquet_from(paths):
            """SQL FROM clause: single path or UNION ALL of two paths. Escape single quotes for SQL."""
            def esc(s):
                return s.replace("'", "''")
            if len(paths) == 1:
                return f"read_parquet('{esc(paths[0])}')"
            return f"(SELECT * FROM read_parquet('{esc(paths[0])}') UNION ALL SELECT * FROM read_parquet('{esc(paths[1])}'))"

        medical_from = _parquet_from(medical_paths)
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
            -- ALL ICD diagnosis codes (for ML feature discovery) - CAST to VARCHAR for codes with letters/dots
            CAST(primary_icd_diagnosis_code AS VARCHAR) AS primary_icd_diagnosis_code,
            CAST(two_icd_diagnosis_code AS VARCHAR) AS two_icd_diagnosis_code,
            CAST(three_icd_diagnosis_code AS VARCHAR) AS three_icd_diagnosis_code,
            CAST(four_icd_diagnosis_code AS VARCHAR) AS four_icd_diagnosis_code,
            CAST(five_icd_diagnosis_code AS VARCHAR) AS five_icd_diagnosis_code,
            CAST(six_icd_diagnosis_code AS VARCHAR) AS six_icd_diagnosis_code,
            CAST(seven_icd_diagnosis_code AS VARCHAR) AS seven_icd_diagnosis_code,
            CAST(eight_icd_diagnosis_code AS VARCHAR) AS eight_icd_diagnosis_code,
            CAST(nine_icd_diagnosis_code AS VARCHAR) AS nine_icd_diagnosis_code,
            CAST(ten_icd_diagnosis_code AS VARCHAR) AS ten_icd_diagnosis_code,
            -- ALL ICD procedure codes (for ML feature discovery) - CAST to VARCHAR
            CAST(two_icd_procedure_code AS VARCHAR) AS two_icd_procedure_code,
            CAST(three_icd_procedure_code AS VARCHAR) AS three_icd_procedure_code,
            CAST(four_icd_procedure_code AS VARCHAR) AS four_icd_procedure_code,
            CAST(five_icd_procedure_code AS VARCHAR) AS five_icd_procedure_code,
            CAST(six_icd_procedure_code AS VARCHAR) AS six_icd_procedure_code,
            CAST(seven_icd_procedure_code AS VARCHAR) AS seven_icd_procedure_code,
            CAST(eight_icd_procedure_code AS VARCHAR) AS eight_icd_procedure_code,
            CAST(nine_icd_procedure_code AS VARCHAR) AS nine_icd_procedure_code,
            CAST(ten_icd_procedure_code AS VARCHAR) AS ten_icd_procedure_code,
            -- CPT/procedure fields for event features
            procedure_code,
            cpt_mod_1_code,
            cpt_mod_2_code,
            -- HCG fields for ED visit identification
            hcg_setting,
            hcg_line,
            hcg_detail,
            event_date,
            CAST(event_year AS INTEGER) AS event_year
        FROM {medical_from}
        WHERE mi_person_key IS NOT NULL
          AND CAST(mi_person_key AS VARCHAR) <> ''
          AND event_date IS NOT NULL;
        """
        execute_sql_with_dev_validation(cohort_conn_duckdb, logger, medical_sql)
        logger.info("→ [PHASE 1] Medical data loaded from GOLD final table")
        
        # Apply additional medical filters into final view 'medical'
        # IMPORTANT: Calendar-year filtering enforces strict partitioning
        # Events outside the calendar year are intentionally excluded even if exposure windows cross year boundaries
        # This is consistent with Phase 2 event ordering, Phase 3 time windows, and Phase 4 Parquet partitioning
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
        pharmacy_from = _parquet_from(pharmacy_paths)
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
        FROM {pharmacy_from}
        WHERE mi_person_key IS NOT NULL
          AND CAST(mi_person_key AS VARCHAR) <> ''
          AND incurred_date IS NOT NULL
          AND TRY_STRPTIME(CAST(incurred_date AS VARCHAR), '%Y%m%d') IS NOT NULL;
        """
        execute_sql_with_dev_validation(cohort_conn_duckdb, logger, pharmacy_sql)
        logger.info("→ [PHASE 1] Pharmacy data loaded from GOLD final table")
        
        # Apply additional pharmacy filters into final view 'pharmacy'
        # IMPORTANT: Calendar-year filtering enforces strict partitioning
        # Events outside the calendar year are intentionally excluded even if exposure windows cross year boundaries
        # This is consistent with Phase 2 event ordering, Phase 3 time windows, and Phase 4 Parquet partitioning
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
        # Cast COUNT(*) to BIGINT to avoid INT32 overflow for large counts
        # Use ::BIGINT syntax and convert to int in Python to handle large values
        medical_count_result = cohort_conn_duckdb.sql("SELECT COUNT(*)::BIGINT FROM medical").fetchone()[0]
        medical_count = int(medical_count_result) if medical_count_result is not None else 0
        
        pharmacy_count_result = cohort_conn_duckdb.sql("SELECT COUNT(*)::BIGINT FROM pharmacy").fetchone()[0]
        pharmacy_count = int(pharmacy_count_result) if pharmacy_count_result is not None else 0
        
        logger.info(f"→ [PHASE 1] QA: Medical records: {medical_count:,}")
        logger.info(f"→ [PHASE 1] QA: Pharmacy records: {pharmacy_count:,}")
        
        # F1120-specific check in raw medical data
        # NOTE: This is a PRIMARY-ONLY sanity check (not full opioid detection)
        # The actual opioid detection logic (get_opioid_icd_sql_condition()) checks ALL 10 ICD diagnosis columns
        # This Phase 1 check is for data quality validation only
        f1120_medical = cohort_conn_duckdb.sql("""
        SELECT 
            COUNT(*) as total_f1120_records,
            COUNT(DISTINCT mi_person_key) as distinct_f1120_patients
        FROM medical
        WHERE primary_icd_diagnosis_code = 'F1120'
        """).fetchone()
        
        if f1120_medical and f1120_medical[0] > 0:
            logger.info(f"→ [PHASE 1] F1120 CHECK (primary column only - sanity check):")
            logger.info(f"  Total F1120 records: {f1120_medical[0]:,}")
            logger.info(f"  Distinct F1120 patients: {f1120_medical[1]:,}")
        else:
            logger.warning(f"→ [PHASE 1] F1120 CHECK: No F1120 records found in medical data (primary column)")
        
        # HCG codes of interest check (for polypharmacy cohort target identification)
        # Check for ED visit HCG line codes and details: P51b (ED Visits only), O11, P33
        # Use hcg_detail to distinguish actual ED visits from observation care
        hcg_condition = """
            (hcg_line = 'P51 - ER Visits and Observation Care' AND hcg_detail = 'P51b - PHY ED Visits and Observation Care - ED Visits')
            OR hcg_line = 'O11 - Emergency Room'
            OR hcg_line = 'P33 - Urgent Care Visits'
        """
        hcg_medical = cohort_conn_duckdb.sql(f"""
        SELECT 
            hcg_line,
            hcg_detail,
            COUNT(*) as count_by_code,
            COUNT(DISTINCT mi_person_key) as distinct_hcg_patients
        FROM medical
        WHERE {hcg_condition}
        GROUP BY hcg_line, hcg_detail
        ORDER BY count_by_code DESC
        """).fetchall()
        
        if hcg_medical:
            # Row structure: (hcg_line, hcg_detail, count_by_code, distinct_hcg_patients)
            # Sum the count_by_code (row[2]) for total HCG records
            total_hcg = sum(int(row[2]) for row in hcg_medical)
            distinct_hcg = cohort_conn_duckdb.sql(f"""
            SELECT COUNT(DISTINCT mi_person_key)
            FROM medical
            WHERE {hcg_condition}
            """).fetchone()[0]
            logger.info(f"→ [PHASE 1] HCG CODES CHECK (ED visit codes for polypharmacy cohort - using hcg_detail for precision):")
            logger.info(f"  Total HCG records: {total_hcg:,}")
            logger.info(f"  Distinct HCG patients: {distinct_hcg:,}")
            logger.info(f"  HCG codes breakdown (line + detail):")
            for row in hcg_medical:
                # Row: (hcg_line, hcg_detail, count_by_code, distinct_hcg_patients)
                logger.info(f"    '{row[0]}' / '{row[1]}': {row[2]:,} records, {row[3]:,} patients")
        else:
            logger.warning(f"→ [PHASE 1] HCG CODES CHECK: No ED visit HCG codes found in medical data")
        
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

