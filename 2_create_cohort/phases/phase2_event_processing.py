"""
Phase 2: Event Processing with DuckDB optimizations.

Step 1: Event Fact Table Creation
Step 2: Medication claims events (pharmacy lines → event fact)
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
from py_helpers.constants import get_opioid_icd_sql_condition, ALL_ICD_DIAGNOSIS_COLUMNS


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
        # Enable query profiling for this step (partition-safe filename)
        enable_query_profiling(cohort_conn_duckdb, logger, "json", f"/tmp/duckdb_profile_p2_step1_{age_band}_{event_year}.json")

        # Build dynamic target classification from environment variables
        # Use centralized config helper to reduce drift across phases
        from .common import get_dynamic_targeting_config
        config = get_dynamic_targeting_config()
        target_icd_codes = config["target_icd_codes"]
        target_cpt_codes = config["target_cpt_codes"]
        target_icd_prefixes = config["target_icd_prefixes"]
        target_cpt_prefixes = config["target_cpt_prefixes"]

        # Compose SQL condition for ICD-based targeting
        # Codes are normalized to F1120 format (no dots, no prefixes) via 7_update_codes.py
        # Verified: gold tier contains 'F1120' format codes
        icd_conditions = []
        if target_icd_codes:
            # Exact match (codes are normalized to F1120 format in gold tier)
            icd_conditions.append(f"primary_icd_diagnosis_code IN {tuple(target_icd_codes)}")
        for pref in target_icd_prefixes:
            # Normalize prefix and use LIKE with ESCAPE for wildcard safe match
            # CRITICAL: This normalization must match get_opioid_icd_sql_condition() logic
            # Both use: UPPER, remove '.', remove ' ' (spaces)
            # get_opioid_icd_sql_condition() checks codes already normalized in gold tier (F1120 format)
            # This prefix matching also normalizes to match gold tier format
            # NOTE: This normalization is duplicated in common.py ensure_unified_views() - consider centralizing
            norm_pref = pref.upper().replace('.', '').replace(' ', '')
            like = norm_pref if ('%' in norm_pref or '_' in norm_pref) else (norm_pref + '%')
            icd_conditions.append(
                f"REPLACE(REPLACE(UPPER(primary_icd_diagnosis_code), '.', ''), ' ', '') LIKE '{like}'"
            )

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

        # HCG-based ED visit identification (for ED_NON_OPIOID cohort)
        # ED visits are identified by HCG line codes and details for precision
        # Use hcg_detail to distinguish actual ED visits from observation care
        # P51a = Observation Care (exclude), P51b = ED Visits (include)
        # O11 = Emergency Department (include)
        # P33 = Urgent Care Visits (include)
        ed_hcg_condition = """
            (hcg_line = 'P51 - ER Visits and Observation Care' AND hcg_detail = 'P51b - PHY ED Visits and Observation Care - ED Visits')
            OR hcg_line = 'O11 - Emergency Room'
            OR hcg_line = 'P33 - Urgent Care Visits'
        """
        
        # Default classification falls back to opioid_ed vs ed_non_opioid
        # Priority: 1) Opioid ICD codes (ANY position) → opioid_ed, 2) HCG ED visits → ed_non_opioid, 3) Other → ed_non_opioid
        # CRITICAL: Check ALL 10 ICD diagnosis columns for opioid codes
        opioid_icd_condition = get_opioid_icd_sql_condition()
        default_case = f"""
            CASE 
                WHEN {opioid_icd_condition} THEN 'opioid_ed'
                WHEN {ed_hcg_condition} THEN 'ed_non_opioid'
                ELSE 'ed_non_opioid'
            END
        """

        # If any env targets are provided, build a generic target/non_target classification
        # Priority: 1) Target ICD/CPT codes → target, 2) HCG ED visits → ed_non_opioid, 3) Other → non_target
        # IMPORTANT: 'non_target' is intentionally excluded from ED-based cohorts in later phases
        # This classification is used for dynamic targeting scenarios, not for standard opioid/polypharmacy cohorts
        if icd_conditions or cpt_conditions:
            where_clause = " OR ".join(filter(None, icd_conditions + cpt_conditions)) or "1=0"
            classification_sql = f"""
                CASE 
                    WHEN ({where_clause}) THEN 'target'
                    WHEN {ed_hcg_condition} THEN 'ed_non_opioid'
                    ELSE 'non_target'
                END
            """
        else:
            classification_sql = default_case
        
        # CRITICAL FIX: Compute event_sequence AFTER UNION ALL to ensure global chronological ordering
        # Previously, ROW_NUMBER() was computed separately for medical and pharmacy, breaking global sequence
        # Example: Medical event Jan 10 → seq 1, Pharmacy event Jan 05 → seq 1 (both seq 1, but Jan 05 should come first)
        # This fix ensures event_sequence reflects true chronological order across all event types
        # This is essential for Phase 3 time windows, first events, and temporal analysis
        event_fact_table_sql = f"""
        CREATE OR REPLACE VIEW unified_event_fact_table AS
        WITH unified_events AS (
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
                -- ALL ICD diagnosis codes (for ML feature discovery)
                primary_icd_diagnosis_code,
                two_icd_diagnosis_code,
                three_icd_diagnosis_code,
                four_icd_diagnosis_code,
                five_icd_diagnosis_code,
                six_icd_diagnosis_code,
                seven_icd_diagnosis_code,
                eight_icd_diagnosis_code,
                nine_icd_diagnosis_code,
                ten_icd_diagnosis_code,
                -- ALL ICD procedure codes (for ML feature discovery)
                two_icd_procedure_code,
                three_icd_procedure_code,
                four_icd_procedure_code,
                five_icd_procedure_code,
                six_icd_procedure_code,
                seven_icd_procedure_code,
                eight_icd_procedure_code,
                nine_icd_procedure_code,
                ten_icd_procedure_code,
                NULL as drug_name,
                NULL as therapeutic_class_1,
                -- CPT/procedure codes (medical)
                procedure_code,
                cpt_mod_1_code,
                cpt_mod_2_code,
                -- HCG fields for ED visit identification
                hcg_setting,
                hcg_line,
                hcg_detail,
                -- Event classification (dynamic via env or default)
                {classification_sql} as event_classification
            FROM medical
            -- INTENTIONAL: Medical events without primary ICD codes are excluded
            -- Rationale: Medical events without ICD codes are not analytically meaningful for this study
            -- This creates asymmetric exposure histories (medical requires ICD, pharmacy does not)
            -- This is a design decision: we prioritize events with diagnostic information
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
                -- ICD diagnosis codes not present in pharmacy (set NULLs)
                NULL as primary_icd_diagnosis_code,
                NULL as two_icd_diagnosis_code,
                NULL as three_icd_diagnosis_code,
                NULL as four_icd_diagnosis_code,
                NULL as five_icd_diagnosis_code,
                NULL as six_icd_diagnosis_code,
                NULL as seven_icd_diagnosis_code,
                NULL as eight_icd_diagnosis_code,
                NULL as nine_icd_diagnosis_code,
                NULL as ten_icd_diagnosis_code,
                -- ICD procedure codes not present in pharmacy (set NULLs)
                NULL as two_icd_procedure_code,
                NULL as three_icd_procedure_code,
                NULL as four_icd_procedure_code,
                NULL as five_icd_procedure_code,
                NULL as six_icd_procedure_code,
                NULL as seven_icd_procedure_code,
                NULL as eight_icd_procedure_code,
                NULL as nine_icd_procedure_code,
                NULL as ten_icd_procedure_code,
                drug_name,
                therapeutic_class_1,
                -- CPT/procedure codes not present in pharmacy (set NULLs)
                NULL as procedure_code,
                NULL as cpt_mod_1_code,
                NULL as cpt_mod_2_code,
                -- HCG fields not present in pharmacy (set NULLs)
                NULL as hcg_setting,
                NULL as hcg_line,
                NULL as hcg_detail,
                -- Use same classification expression to preserve target logic across union
                {classification_sql} as event_classification
            FROM pharmacy
            WHERE drug_name IS NOT NULL
        )
        SELECT 
            *,
            -- CRITICAL: Compute event_sequence AFTER union to ensure global chronological ordering
            ROW_NUMBER() OVER (PARTITION BY mi_person_key ORDER BY event_date) as event_sequence
        FROM unified_events;
        """
        execute_sql_with_dev_validation(cohort_conn_duckdb, logger, event_fact_table_sql)
        logger.info("→ [PHASE 2 STEP 1] Unified event fact table created")
        
        # QA checks
        # Cast COUNT(*) to BIGINT to avoid INT32 overflow for large counts
        # Use ::BIGINT syntax and convert to int in Python to handle large values
        total_events_result = cohort_conn_duckdb.sql("SELECT COUNT(*)::BIGINT FROM unified_event_fact_table").fetchone()[0]
        total_events = int(total_events_result) if total_events_result is not None else 0
        event_type_dist = cohort_conn_duckdb.sql("""
        SELECT event_type, COUNT(*) as count
        FROM unified_event_fact_table
        GROUP BY event_type
        ORDER BY count DESC
        """).fetchall()
        
        logger.info(f"→ [PHASE 2 STEP 1] QA: Total events: {total_events:,}")
        logger.info(f"→ [PHASE 2 STEP 1] QA: Event type distribution: {dict(event_type_dist)}")
        
        # F1120-specific checks
        # NOTE: This QA check only inspects primary_icd_diagnosis_code for F1120
        # The actual opioid detection logic (get_opioid_icd_sql_condition()) checks ALL 10 ICD diagnosis columns
        # This is a simplified QA check - for full validation, see Phase 4 QA which checks all columns
        f1120_total = cohort_conn_duckdb.sql("""
        SELECT 
            COUNT(*) as total_f1120_records,
            COUNT(DISTINCT mi_person_key) as distinct_f1120_patients
        FROM unified_event_fact_table
        WHERE primary_icd_diagnosis_code = 'F1120'
        """).fetchone()
        
        f1120_by_class = cohort_conn_duckdb.sql("""
        SELECT 
            event_classification,
            COUNT(*) as count_by_classification
        FROM unified_event_fact_table
        WHERE primary_icd_diagnosis_code = 'F1120'
        GROUP BY event_classification
        ORDER BY count_by_classification DESC
        """).fetchall()
        
        # Expanded QA: Check all 10 ICD columns for F1120 (matches Phase 3/4 logic)
        opioid_icd_condition = get_opioid_icd_sql_condition()
        f1120_all_columns = cohort_conn_duckdb.sql(f"""
        SELECT 
            COUNT(*) as total_f1120_records_all_columns,
            COUNT(DISTINCT mi_person_key) as distinct_f1120_patients_all_columns
        FROM unified_event_fact_table
        WHERE {opioid_icd_condition}
        """).fetchone()
        
        if f1120_total and f1120_total[0] > 0:
            logger.info(f"→ [PHASE 2 STEP 1] F1120 CHECK (primary column only - simplified QA):")
            logger.info(f"  Total F1120 records (primary): {f1120_total[0]:,}")
            logger.info(f"  Distinct F1120 patients (primary): {f1120_total[1]:,}")
            if f1120_by_class:
                logger.info(f"  F1120 by classification:")
                for row in f1120_by_class:
                    logger.info(f"    '{row[0]}': {row[1]:,} records")
            
            # Log expanded check for comparison
            if f1120_all_columns and f1120_all_columns[0] > 0:
                logger.info(f"→ [PHASE 2 STEP 1] F1120 CHECK (all 10 ICD columns - matches Phase 3/4 logic):")
                logger.info(f"  Total opioid records (all columns): {f1120_all_columns[0]:,}")
                logger.info(f"  Distinct opioid patients (all columns): {f1120_all_columns[1]:,}")
                if f1120_all_columns[0] > f1120_total[0]:
                    logger.info(f"  → Note: {f1120_all_columns[0] - f1120_total[0]:,} additional records found in non-primary ICD columns")
        else:
            logger.warning(f"→ [PHASE 2 STEP 1] F1120 CHECK: No F1120 records found in unified_event_fact_table (primary column)")
            if f1120_all_columns and f1120_all_columns[0] > 0:
                logger.info(f"→ [PHASE 2 STEP 1] F1120 CHECK (all columns): Found {f1120_all_columns[0]:,} opioid records in non-primary columns")
        
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
    """Phase 2 Step 2: Medication claims events (pharmacy lines → event fact) with DuckDB optimizations."""
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
    
    logger.info(f"{SYMBOLS['arrow']} [PHASE 2 STEP 2] Starting optimized medication claims events creation...")
    
    try:
        # Ensure gold-backed views exist if Phase 1 was skipped
        ensure_gold_views(cohort_conn_duckdb, logger, age_band, event_year)
        # Enable query profiling for this step (partition-safe filename)
        enable_query_profiling(cohort_conn_duckdb, logger, "json", f"/tmp/duckdb_profile_p2_step2_{age_band}_{event_year}.json")
        
        # Create unified medication claims events view
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
        logger.info("→ [PHASE 2 STEP 2] Unified medication claims events view created")
        
        # QA checks
        # Cast COUNT(*) to BIGINT to avoid INT32 overflow for large counts
        # Use ::BIGINT syntax and convert to int in Python to handle large values
        total_drug_events_result = cohort_conn_duckdb.sql("SELECT COUNT(*)::BIGINT FROM unified_drug_exposure").fetchone()[0]
        total_drug_events = int(total_drug_events_result) if total_drug_events_result is not None else 0
        
        logger.info(f"→ [PHASE 2 STEP 2] QA: Total medication claims events: {total_drug_events:,}")
        
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
        
        logger.info(f"{SYMBOLS['success']} [PHASE 2 STEP 2] Optimized medication claims events creation completed")
        
    except Exception as e:
        logger.error(f"{SYMBOLS['fail']} [PHASE 2 STEP 2] Medication claims events creation failed: {str(e)}")
        if pipeline_state:
            pipeline_state.mark_step_failed(step_name, str(e))
        cleanup_duckdb_temp_files(logger)
        raise

