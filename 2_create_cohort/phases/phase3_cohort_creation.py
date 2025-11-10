"""
Phase 3: Final Cohort Creation with 5:1 ratio and DuckDB optimizations.

Creates OPIOID_ED and ED_NON_OPIOID cohorts with target and control groups.
"""

from .common import (
    datetime,
    SYMBOLS,
    cleanup_duckdb_temp_files,
    enable_query_profiling,
    disable_query_profiling,
    force_checkpoint,
    execute_sql_with_dev_validation,
    ensure_gold_views,
    ensure_unified_views,
)
import os


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
        # Ensure required views exist if earlier phases were skipped
        ensure_gold_views(cohort_conn_duckdb, logger, age_band, event_year)
        ensure_unified_views(cohort_conn_duckdb, logger)

        # Determine classification labels based on dynamic targeting env
        target_icd = os.getenv("PGX_TARGET_ICD_CODES", "").strip() or os.getenv("PGX_TARGET_ICD_PREFIXES", "").strip()
        target_cpt = os.getenv("PGX_TARGET_CPT_CODES", "").strip() or os.getenv("PGX_TARGET_CPT_PREFIXES", "").strip()
        dynamic_targeting = bool(target_icd or target_cpt)
        label_target = 'target' if dynamic_targeting else 'opioid_ed'
        label_nontarget = 'non_target' if dynamic_targeting else 'ed_non_opioid'
        # Enable query profiling for this step
        enable_query_profiling(cohort_conn_duckdb, logger, "json", f"/tmp/duckdb_profiling_phase3_step3_final_cohort_fact.json")
        
        # Create OPIOID_ED cohort with 5:1 control-to-target ratio
        opioid_ed_cohort_sql = f"""
        CREATE OR REPLACE VIEW opioid_ed_cohort AS
        WITH target_cases AS (
            SELECT DISTINCT mi_person_key
            FROM unified_event_fact_table
            WHERE event_classification = '{label_target}'
        ),
        control_candidates AS (
            SELECT DISTINCT mi_person_key
            FROM unified_event_fact_table
            WHERE event_classification != '{label_target}'
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
        execute_sql_with_dev_validation(cohort_conn_duckdb, logger, opioid_ed_cohort_sql)
        logger.info("→ [PHASE 3 STEP 3] OPIOID_ED cohort created")
        
        # Create ED_NON_OPIOID cohort with 5:1 control-to-target ratio
        ed_non_opioid_cohort_sql = f"""
        CREATE OR REPLACE VIEW ed_non_opioid_cohort AS
        WITH target_cases AS (
            SELECT DISTINCT mi_person_key
            FROM unified_event_fact_table
            WHERE event_classification = '{label_nontarget}'
        ),
        control_candidates AS (
            SELECT DISTINCT mi_person_key
            FROM unified_event_fact_table
            WHERE event_classification != '{label_nontarget}'
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
        execute_sql_with_dev_validation(cohort_conn_duckdb, logger, ed_non_opioid_cohort_sql)
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

