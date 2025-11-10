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
        
        # Save to S3
        from helpers_1997_13.s3_utils import get_output_paths, get_cohort_parquet_path
        
        # Save OPIOID_ED cohort
        opioid_ed_out = get_cohort_parquet_path("opioid_ed", age_band, event_year)
        cohort_conn_duckdb.sql(f"""
        COPY opioid_ed_cohort TO '{opioid_ed_out}' 
        (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)
        logger.info(f"→ [PHASE 4] OPIOID_ED cohort saved to S3: {opioid_ed_out}")

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
        
        # Save ED_NON_OPIOID cohort
        ed_non_opioid_out = get_cohort_parquet_path("ed_non_opioid", age_band, event_year)
        cohort_conn_duckdb.sql(f"""
        COPY ed_non_opioid_cohort TO '{ed_non_opioid_out}' 
        (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)
        logger.info(f"→ [PHASE 4] ED_NON_OPIOID cohort saved to S3: {ed_non_opioid_out}")

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

