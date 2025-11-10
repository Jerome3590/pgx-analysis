"""
Phase 2: Event Processing with DuckDB optimizations.

Step 1: Event Fact Table Creation
Step 2: Drug Exposure Events
"""

from .common import (
    datetime,
    SYMBOLS,
    OPIOID_ICD_CODES,
    cleanup_duckdb_temp_files,
    enable_query_profiling,
    disable_query_profiling,
    force_checkpoint,
    execute_sql_with_dev_validation,
    ensure_gold_views,
)


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
        # Ensure gold-backed views exist if Phase 1 was skipped
        ensure_gold_views(cohort_conn_duckdb, logger, age_band, event_year)
        # Enable query profiling for this step
        enable_query_profiling(cohort_conn_duckdb, logger, "json", f"/tmp/duckdb_profiling_phase2_step1_event_fact_table.json")

        # Build dynamic target classification from environment variables
        import os
        target_icd_codes = [c.strip() for c in os.getenv("PGX_TARGET_ICD_CODES", "").split(',') if c.strip()]
        target_cpt_codes = [c.strip() for c in os.getenv("PGX_TARGET_CPT_CODES", "").split(',') if c.strip()]
        target_icd_prefixes = [p.strip() for p in os.getenv("PGX_TARGET_ICD_PREFIXES", "").split(',') if p.strip()]
        target_cpt_prefixes = [p.strip() for p in os.getenv("PGX_TARGET_CPT_PREFIXES", "").split(',') if p.strip()]

        # Compose SQL condition for ICD-based targeting
        icd_conditions = []
        if target_icd_codes:
            icd_conditions.append(f"primary_icd_diagnosis_code IN {tuple(target_icd_codes)}")
        for pref in target_icd_prefixes:
            # Use LIKE with ESCAPE for wildcard safe match (assumes % in prefix if desired)
            like = pref if ('%' in pref or '_' in pref) else (pref + '%')
            icd_conditions.append(f"primary_icd_diagnosis_code LIKE '{like}'")

        # Compose SQL condition for CPT-based targeting (medical rows only)
        cpt_conditions = []
        if target_cpt_codes:
            tup = tuple(target_cpt_codes)
            cpt_conditions.append(f"procedure_code IN {tup} OR cpt_mod_1_code IN {tup} OR cpt_mod_2_code IN {tup}")
        for pref in target_cpt_prefixes:
            like = pref if ('%' in pref or '_' in pref) else (pref + '%')
            cpt_conditions.append(
                f"procedure_code LIKE '{like}' OR cpt_mod_1_code LIKE '{like}' OR cpt_mod_2_code LIKE '{like}'"
            )

        # Default classification falls back to opioid_ed vs ed_non_opioid using OPIOID_ICD_CODES
        default_case = f"""
            CASE 
                WHEN primary_icd_diagnosis_code IN {tuple(OPIOID_ICD_CODES)} THEN 'opioid_ed'
                ELSE 'ed_non_opioid'
            END
        """

        # If any env targets are provided, build a generic target/non_target classification
        if icd_conditions or cpt_conditions:
            where_clause = " OR ".join(filter(None, icd_conditions + cpt_conditions)) or "1=0"
            classification_sql = f"CASE WHEN ({where_clause}) THEN 'target' ELSE 'non_target' END"
        else:
            classification_sql = default_case
        
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
            -- CPT/procedure codes (medical)
            procedure_code,
            cpt_mod_1_code,
            cpt_mod_2_code,
            -- Event classification (dynamic via env or default)
            {classification_sql} as event_classification,
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
            -- CPT/procedure codes not present in pharmacy (set NULLs)
            NULL as procedure_code,
            NULL as cpt_mod_1_code,
            NULL as cpt_mod_2_code,
            -- Use same classification expression to preserve target logic across union
            {classification_sql} as event_classification,
            ROW_NUMBER() OVER (PARTITION BY mi_person_key ORDER BY event_date) as event_sequence
        FROM pharmacy
        WHERE drug_name IS NOT NULL;
        """
        execute_sql_with_dev_validation(cohort_conn_duckdb, logger, event_fact_table_sql)
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
        # Ensure gold-backed views exist if Phase 1 was skipped
        ensure_gold_views(cohort_conn_duckdb, logger, age_band, event_year)
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
        FROM pharmacy
        WHERE drug_name IS NOT NULL
          AND drug_name != '';
        """
        execute_sql_with_dev_validation(cohort_conn_duckdb, logger, drug_exposure_sql)
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

