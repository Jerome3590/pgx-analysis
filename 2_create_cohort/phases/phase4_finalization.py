"""
Phase 4: Complete Pipeline with DuckDB optimizations.

Final QA validation and save outputs to S3.
Optionally trigger a separate QA notebook via papermill for additional validations.
"""

from .common import (
    datetime,
    SYMBOLS,
    cleanup_duckdb_temp_files,
    enable_query_profiling,
    disable_query_profiling,
    force_checkpoint,
    monitor_disk_space,
    ensure_gold_views,
    ensure_unified_views,
    ensure_cohort_views,
)
import os
import subprocess


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
        # Ensure required views exist if earlier phases were skipped
        ensure_gold_views(cohort_conn_duckdb, logger, age_band, event_year)
        ensure_unified_views(cohort_conn_duckdb, logger)
        ensure_cohort_views(cohort_conn_duckdb, logger)
        # Enable query profiling for this phase
        enable_query_profiling(cohort_conn_duckdb, logger, "json", f"/tmp/duckdb_profiling_phase4_complete_pipeline.json")
        
        # Final QA validation
        logger.info("→ [PHASE 4] Performing final QA validation...")
        
        # Check both cohorts exist
        opioid_ed_count = cohort_conn_duckdb.sql("SELECT COUNT(*) FROM opioid_ed_cohort").fetchone()[0]
        ed_non_opioid_count = cohort_conn_duckdb.sql("SELECT COUNT(*) FROM ed_non_opioid_cohort").fetchone()[0]
        
        logger.info(f"→ [PHASE 4] QA: OPIOID_ED cohort records: {opioid_ed_count:,}")
        logger.info(f"→ [PHASE 4] QA: ED_NON_OPIOID cohort records: {ed_non_opioid_count:,}")
        
        # F1120-specific checks in final cohorts
        f1120_opioid_final = cohort_conn_duckdb.sql("""
        SELECT 
            COUNT(*) as total_f1120_records,
            COUNT(DISTINCT mi_person_key) as distinct_f1120_patients
        FROM opioid_ed_cohort
        WHERE primary_icd_diagnosis_code = 'F1120'
        """).fetchone()
        
        f1120_ed_non_opioid_final = cohort_conn_duckdb.sql("""
        SELECT 
            COUNT(*) as total_f1120_records,
            COUNT(DISTINCT mi_person_key) as distinct_f1120_patients
        FROM ed_non_opioid_cohort
        WHERE primary_icd_diagnosis_code = 'F1120'
        """).fetchone()
        
        logger.info(f"→ [PHASE 4] F1120 IN FINAL COHORTS:")
        logger.info(f"  OPIOID_ED: {f1120_opioid_final[0]:,} records, {f1120_opioid_final[1]:,} patients")
        logger.info(f"  ED_NON_OPIOID: {f1120_ed_non_opioid_final[0]:,} records, {f1120_ed_non_opioid_final[1]:,} patients")
        
        # Warn if cohorts are empty
        if opioid_ed_count == 0:
            logger.warning(f"⚠️ [PHASE 4] WARNING: OPIOID_ED cohort is empty for {age_band}/{event_year}")
        if ed_non_opioid_count == 0:
            logger.warning(f"⚠️ [PHASE 4] WARNING: ED_NON_OPIOID cohort is empty for {age_band}/{event_year}")
        
        # Save to S3
        from helpers_1997_13.s3_utils import get_output_paths, get_cohort_parquet_path
        
        # Save OPIOID_ED cohort (always save, even if control-only)
        opioid_ed_out = get_cohort_parquet_path("opioid_ed", age_band, event_year)
        if opioid_ed_count > 0:
            cohort_conn_duckdb.sql(f"""
            COPY opioid_ed_cohort TO '{opioid_ed_out}' 
            (FORMAT PARQUET, COMPRESSION SNAPPY)
            """)
            # Check if it's control-only
            target_count_check = cohort_conn_duckdb.sql("SELECT COUNT(*) FROM opioid_ed_cohort WHERE is_target_case = 1").fetchone()[0]
            if target_count_check == 0:
                logger.info(f"→ [PHASE 4] OPIOID_ED cohort saved (CONTROL-ONLY) to S3: {opioid_ed_out}")
            else:
                logger.info(f"→ [PHASE 4] OPIOID_ED cohort saved to S3: {opioid_ed_out}")
        else:
            logger.warning(f"⚠️ [PHASE 4] Skipping save of empty OPIOID_ED cohort to {opioid_ed_out}")

        # Optional: run QA notebook for opioid_ed cohort if configured
        qa_nb = os.environ.get("PGX_QA_NOTEBOOK")
        if qa_nb:
            try:
                out_nb = f"/tmp/Cohort_QA_opioid_ed_{age_band}_{event_year}.ipynb"
                cmd = [
                    "papermill", qa_nb, out_nb,
                    "-p", "cohort_name", "opioid_ed",
                    "-p", "cohort_parquet_path", opioid_ed_out,
                    "-p", "age_band", str(age_band),
                    "-p", "event_year", str(event_year),
                ]
                logger.info(f"→ [PHASE 4] Running QA notebook: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                logger.info(f"✓ QA notebook completed: {out_nb}")
            except Exception as nb_e:
                logger.warning(f"⚠ QA notebook failed for opioid_ed: {nb_e}")
        
        # Save ED_NON_OPIOID cohort (always save, even if control-only)
        ed_non_opioid_out = get_cohort_parquet_path("ed_non_opioid", age_band, event_year)
        if ed_non_opioid_count > 0:
            cohort_conn_duckdb.sql(f"""
            COPY ed_non_opioid_cohort TO '{ed_non_opioid_out}' 
            (FORMAT PARQUET, COMPRESSION SNAPPY)
            """)
            # Check if it's control-only
            target_count_check = cohort_conn_duckdb.sql("SELECT COUNT(*) FROM ed_non_opioid_cohort WHERE is_target_case = 1").fetchone()[0]
            if target_count_check == 0:
                logger.info(f"→ [PHASE 4] ED_NON_OPIOID cohort saved (CONTROL-ONLY) to S3: {ed_non_opioid_out}")
            else:
                logger.info(f"→ [PHASE 4] ED_NON_OPIOID cohort saved to S3: {ed_non_opioid_out}")
        else:
            logger.warning(f"⚠️ [PHASE 4] Skipping save of empty ED_NON_OPIOID cohort to {ed_non_opioid_out}")

        # Optional: run QA notebook for ed_non_opioid cohort if configured
        qa_nb = os.environ.get("PGX_QA_NOTEBOOK")
        if qa_nb:
            try:
                out_nb = f"/tmp/Cohort_QA_ed_non_opioid_{age_band}_{event_year}.ipynb"
                cmd = [
                    "papermill", qa_nb, out_nb,
                    "-p", "cohort_name", "ed_non_opioid",
                    "-p", "cohort_parquet_path", ed_non_opioid_out,
                    "-p", "age_band", str(age_band),
                    "-p", "event_year", str(event_year),
                ]
                logger.info(f"→ [PHASE 4] Running QA notebook: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                logger.info(f"✓ QA notebook completed: {out_nb}")
            except Exception as nb_e:
                logger.warning(f"⚠ QA notebook failed for ed_non_opioid: {nb_e}")
        
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

