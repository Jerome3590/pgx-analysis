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
from py_helpers.constants import get_opioid_icd_sql_condition
import os
import subprocess
import shutil
from pathlib import Path
from py_helpers.env_utils import is_linux


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
        
        # Note: We now write to local NVMe first, then use aws s3 sync
        # This is faster and more reliable than DuckDB's direct S3 COPY
        logger.info("→ [PHASE 4] Using local staging + aws s3 sync for cohort uploads (faster and more reliable)")
        
        # Enable query profiling with partition-safe filename (prevents overwrite in parallel runs)
        import time
        profile_filename = f"/tmp/duckdb_profiling_phase4_{age_band.replace('-', '_')}_{event_year}_{int(time.time())}.json"
        enable_query_profiling(cohort_conn_duckdb, logger, "json", profile_filename)
        
        # Cache AWS CLI discovery once (micro-optimization: avoid repeated PATH lookups)
        # Use full path to AWS CLI on EC2
        aws_cli = "/usr/local/bin/aws"
        if not Path(aws_cli).exists():
            # Fallback to PATH lookup if full path doesn't exist
            aws_cli = shutil.which("aws")
            if not aws_cli:
                logger.error("❌ [PHASE 4] AWS CLI not found, cannot sync to S3")
                raise Exception("AWS CLI not available")
        
        # Monitor disk space BEFORE writing Parquet (early warning for NVMe exhaustion)
        monitor_disk_space(logger)
        
        # Final QA validation
        logger.info("→ [PHASE 4] Performing final QA validation...")
        
        # HIGH-IMPACT FIX #1: Check cohort views exist (not just row counts)
        # This prevents silent partial pipeline success if views are missing
        # Use fetchdf() for consistency (though these counts are small)
        cohort_exists_check_df = cohort_conn_duckdb.sql("""
        SELECT 
            COUNT(*) as view_count
        FROM information_schema.tables
        WHERE table_schema = 'main'
          AND (table_name = 'opioid_ed_cohort' OR table_name = 'ed_non_opioid_cohort')
        """).fetchdf()
        cohort_exists_check = int(cohort_exists_check_df.iloc[0]['view_count']) if not cohort_exists_check_df.empty else 0
        
        if cohort_exists_check < 2:
            missing_views = []
            opioid_check_df = cohort_conn_duckdb.sql("SELECT COUNT(*) AS count FROM information_schema.tables WHERE table_schema = 'main' AND table_name = 'opioid_ed_cohort'").fetchdf()
            if int(opioid_check_df.iloc[0]['count']) if not opioid_check_df.empty else 0 == 0:
                missing_views.append("opioid_ed_cohort")
            ed_check_df = cohort_conn_duckdb.sql("SELECT COUNT(*) AS count FROM information_schema.tables WHERE table_schema = 'main' AND table_name = 'ed_non_opioid_cohort'").fetchdf()
            if int(ed_check_df.iloc[0]['count']) if not ed_check_df.empty else 0 == 0:
                missing_views.append("ed_non_opioid_cohort")
            logger.error(f"❌ [PHASE 4] Missing cohort views: {missing_views}")
            raise Exception(f"Cohort views missing: {missing_views}. Phase 3 may have failed silently.")
        
        # Check both cohorts exist and get patient counts
        # CRITICAL: Use COUNT(DISTINCT mi_person_key) instead of COUNT(*) to avoid row explosion issues
        # Event-level COUNT(*) can explode to billions of rows due to multiple time windows
        # Patient-level counts are stable and prevent INT32 overflow
        # Use fetchdf() to avoid Python connector's INT32 casting issue
        opioid_ed_count_df = cohort_conn_duckdb.sql("SELECT CAST(COUNT(DISTINCT mi_person_key) AS BIGINT) AS count FROM opioid_ed_cohort").fetchdf()
        opioid_ed_count = int(opioid_ed_count_df.iloc[0]['count']) if not opioid_ed_count_df.empty else 0
        
        ed_non_opioid_count_df = cohort_conn_duckdb.sql("SELECT CAST(COUNT(DISTINCT mi_person_key) AS BIGINT) AS count FROM ed_non_opioid_cohort").fetchdf()
        ed_non_opioid_count = int(ed_non_opioid_count_df.iloc[0]['count']) if not ed_non_opioid_count_df.empty else 0
        
        logger.info(f"→ [PHASE 4] QA: OPIOID_ED cohort patients: {opioid_ed_count:,}")
        logger.info(f"→ [PHASE 4] QA: ED_NON_OPIOID cohort patients: {ed_non_opioid_count:,}")
        
        # Cohort-specific QA checks
        # OPIOID_ED cohort: Check F1120 (opioid ICD codes) - all 10 ICD columns
        # ED_NON_OPIOID cohort: Check HCG target events (polypharmacy cohort target)
        opioid_icd_condition = get_opioid_icd_sql_condition()
        
        # OPIOID_ED: F1120 check (all 10 ICD diagnosis columns)
        # Use fetchdf() to avoid INT32 overflow in COUNT queries
        f1120_opioid_final_df = cohort_conn_duckdb.sql(f"""
        SELECT 
            CAST(COUNT(*) AS BIGINT) as total_f1120_records,
            CAST(COUNT(DISTINCT mi_person_key) AS BIGINT) as distinct_f1120_patients
        FROM opioid_ed_cohort
        WHERE {opioid_icd_condition}
        """).fetchdf()
        f1120_opioid_final = (
            int(f1120_opioid_final_df.iloc[0]['total_f1120_records']) if not f1120_opioid_final_df.empty and f1120_opioid_final_df.iloc[0]['total_f1120_records'] is not None else 0,
            int(f1120_opioid_final_df.iloc[0]['distinct_f1120_patients']) if not f1120_opioid_final_df.empty and f1120_opioid_final_df.iloc[0]['distinct_f1120_patients'] is not None else 0
        )
        
        # ED_NON_OPIOID: HCG target events check (polypharmacy cohort)
        # Check for HCG line codes and details used to identify ED visits
        # Use hcg_detail for precision: P51b = ED Visits (exclude P51a = Observation Care)
        # Also check that target cases have drug events (pharmacy events) - matches Phase 3 logic
        hcg_condition = """
            (hcg_line = 'P51 - ER Visits and Observation Care' AND hcg_detail = 'P51b - PHY ED Visits and Observation Care - ED Visits')
            OR hcg_line = 'O11 - Emergency Room'
            OR hcg_line = 'P33 - Urgent Care Visits'
        """
        
        # Use fetchdf() to avoid INT32 overflow in COUNT queries
        hcg_ed_non_opioid_final_df = cohort_conn_duckdb.sql(f"""
        SELECT 
            CAST(COUNT(*) AS BIGINT) as total_hcg_records,
            CAST(COUNT(DISTINCT mi_person_key) AS BIGINT) as distinct_hcg_patients,
            CAST(COUNT(DISTINCT CASE WHEN is_target_case = 1 THEN mi_person_key END) AS BIGINT) as hcg_target_patients,
            CAST(COUNT(DISTINCT CASE WHEN event_type = 'pharmacy' AND is_target_case = 1 THEN mi_person_key END) AS BIGINT) as hcg_target_patients_with_drugs
        FROM ed_non_opioid_cohort
        WHERE {hcg_condition}
        """).fetchdf()
        hcg_ed_non_opioid_final = (
            int(hcg_ed_non_opioid_final_df.iloc[0]['total_hcg_records']) if not hcg_ed_non_opioid_final_df.empty and hcg_ed_non_opioid_final_df.iloc[0]['total_hcg_records'] is not None else 0,
            int(hcg_ed_non_opioid_final_df.iloc[0]['distinct_hcg_patients']) if not hcg_ed_non_opioid_final_df.empty and hcg_ed_non_opioid_final_df.iloc[0]['distinct_hcg_patients'] is not None else 0,
            int(hcg_ed_non_opioid_final_df.iloc[0]['hcg_target_patients']) if not hcg_ed_non_opioid_final_df.empty and hcg_ed_non_opioid_final_df.iloc[0]['hcg_target_patients'] is not None else 0,
            int(hcg_ed_non_opioid_final_df.iloc[0]['hcg_target_patients_with_drugs']) if not hcg_ed_non_opioid_final_df.empty and hcg_ed_non_opioid_final_df.iloc[0]['hcg_target_patients_with_drugs'] is not None else 0
        )
        
        logger.info(f"→ [PHASE 4] OPIOID_ED COHORT QA (F1120 - all ICD columns):")
        logger.info(f"  Total F1120 records: {f1120_opioid_final[0]:,}")
        logger.info(f"  Distinct F1120 patients: {f1120_opioid_final[1]:,}")
        
        logger.info(f"→ [PHASE 4] ED_NON_OPIOID COHORT QA (HCG target events - polypharmacy cohort):")
        logger.info(f"  Total HCG records: {hcg_ed_non_opioid_final[0]:,}")
        logger.info(f"  Distinct HCG patients: {hcg_ed_non_opioid_final[1]:,}")
        logger.info(f"  HCG target patients: {hcg_ed_non_opioid_final[2]:,}")
        logger.info(f"  HCG target patients with drug events: {hcg_ed_non_opioid_final[3]:,}")
        
        # Verify is_target_case column exists in ED_NON_OPIOID cohort
        logger.info("→ [PHASE 4] ED_NON_OPIOID COHORT QA (Schema validation - target case column):")
        schema_check_df = cohort_conn_duckdb.sql("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'ed_non_opioid_cohort'
        ORDER BY column_name
        """).fetchdf()
        schema_columns = schema_check_df['column_name'].tolist() if not schema_check_df.empty else []
        
        required_column = 'is_target_case'
        if required_column not in schema_columns:
            logger.error(f"❌ [PHASE 4] ED_NON_OPIOID cohort missing required column: {required_column}")
            logger.error(f"   All available columns: {schema_columns}")
            raise Exception(f"ED_NON_OPIOID cohort table missing required target case column: {required_column}. Phase 3 may have failed to create this column.")
        else:
            logger.info(f"✓ Required target case column present: {required_column}")
            
            # Log counts for target case column to verify it's populated
            try:
                target_counts_df = cohort_conn_duckdb.sql("""
                SELECT 
                    CAST(COUNT(CASE WHEN is_target_case = 1 THEN 1 END) AS BIGINT) as target_cases,
                    CAST(COUNT(CASE WHEN is_target_case = 0 THEN 1 END) AS BIGINT) as control_cases,
                    CAST(COUNT(*) AS BIGINT) as total_cases
                FROM ed_non_opioid_cohort
                """).fetchdf()
                if not target_counts_df.empty:
                    counts = target_counts_df.iloc[0]
                    logger.info(f"  Target case distribution (21-day window):")
                    logger.info(f"    Target cases: {int(counts['target_cases']):,}")
                    logger.info(f"    Control cases: {int(counts['control_cases']):,}")
                    logger.info(f"    Total: {int(counts['total_cases']):,}")
            except Exception as e:
                logger.warning(f"⚠️ Could not calculate target case counts: {e}")
        
        # Warn if cohorts are empty
        if opioid_ed_count == 0:
            logger.warning(f"⚠️ [PHASE 4] WARNING: OPIOID_ED cohort is empty for {age_band}/{event_year}")
        if ed_non_opioid_count == 0:
            logger.warning(f"⚠️ [PHASE 4] WARNING: ED_NON_OPIOID cohort is empty for {age_band}/{event_year}")
        
        # Save cohorts: Write to local NVMe first, then sync to S3
        from py_helpers.s3_utils import get_output_paths, get_cohort_parquet_path
        from py_helpers.env_utils import get_data_root
        
        # Determine local staging directory (prefer NVMe on Linux)
        if is_linux():
            local_staging = Path("/mnt/nvme/cohorts_staging")
        else:
            # Windows fallback
            local_staging = Path(os.path.join(os.path.expanduser("~"), "cohorts_staging"))
        local_staging.mkdir(parents=True, exist_ok=True)
        
        # Save OPIOID_ED cohort (always save, even if control-only)
        opioid_ed_s3_path = get_cohort_parquet_path("opioid_ed", age_band, event_year)
        opioid_ed_local = None
        if opioid_ed_count > 0:
            # Write to local NVMe first (much faster)
            opioid_ed_local = local_staging / f"opioid_ed_{age_band}_{event_year}.parquet"
            logger.info(f"→ [PHASE 4] Writing OPIOID_ED cohort ({opioid_ed_count:,} patients) to local: {opioid_ed_local}")
            cohort_conn_duckdb.sql(f"""
            COPY opioid_ed_cohort TO '{opioid_ed_local}' 
            (FORMAT PARQUET, COMPRESSION SNAPPY)
            """)
            
            # Log file size before upload (helps diagnose timeouts vs IAM/network failures)
            file_size_gb = opioid_ed_local.stat().st_size / 1e9
            logger.info(f"→ [PHASE 4] OPIOID_ED cohort written to local ({file_size_gb:.2f} GB)")
            
            # Sync to S3 using aws s3 cp (more reliable for large files)
            logger.info(f"→ [PHASE 4] Syncing OPIOID_ED cohort to S3: {opioid_ed_s3_path}")
            local_file = str(opioid_ed_local)
            
            # Use cached AWS CLI (resolved once at top of phase)
            try:
                result = subprocess.run(
                    [aws_cli, "s3", "cp", local_file, opioid_ed_s3_path, "--no-progress"],
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
                if result.returncode == 0:
                    logger.info(f"→ [PHASE 4] OPIOID_ED cohort synced to S3 successfully")
                    # Clean up local file after successful sync
                    try:
                        opioid_ed_local.unlink()
                        logger.info(f"→ [PHASE 4] Cleaned up local OPIOID_ED cohort file")
                        opioid_ed_local = None  # Mark as cleaned
                    except Exception as e:
                        logger.warning(f"⚠️ [PHASE 4] Could not clean up local file: {e}")
                else:
                    logger.error(f"❌ [PHASE 4] Failed to sync OPIOID_ED cohort to S3: {result.stderr}")
                    # Keep local file for retry/debugging
                    logger.warning(f"⚠️ [PHASE 4] Keeping local file for retry: {opioid_ed_local}")
                    raise Exception(f"S3 sync failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                logger.error(f"❌ [PHASE 4] S3 sync timeout for OPIOID_ED cohort (exceeded 1 hour)")
                # Keep local file for retry
                logger.warning(f"⚠️ [PHASE 4] Keeping local file for retry: {opioid_ed_local}")
                raise
            except FileNotFoundError:
                logger.error(f"❌ [PHASE 4] AWS CLI not found at {aws_cli}, cannot sync to S3")
                raise Exception("AWS CLI not available")
            
            # Check if it's control-only
            # NOTE: 'target' column is legacy and not used in Phase 4 logic
            # Use 'is_target_case' for actual target/control distinction
            # Use patient-level count to avoid row explosion issues
            target_count_check_df = cohort_conn_duckdb.sql("SELECT CAST(COUNT(DISTINCT mi_person_key) AS BIGINT) AS count FROM opioid_ed_cohort WHERE is_target_case = 1").fetchdf()
            target_count_check = int(target_count_check_df.iloc[0]['count']) if not target_count_check_df.empty else 0
            if target_count_check == 0:
                logger.info(f"→ [PHASE 4] OPIOID_ED cohort saved (CONTROL-ONLY) to S3: {opioid_ed_s3_path}")
            else:
                logger.info(f"→ [PHASE 4] OPIOID_ED cohort saved to S3: {opioid_ed_s3_path}")
        else:
            logger.warning(f"⚠️ [PHASE 4] Skipping save of empty OPIOID_ED cohort to {opioid_ed_s3_path}")

        # Optional: run QA notebook for opioid_ed cohort if configured
        qa_nb = os.environ.get("PGX_QA_NOTEBOOK")
        if qa_nb:
            try:
                out_nb = f"/tmp/Cohort_QA_opioid_ed_{age_band}_{event_year}.ipynb"
                cmd = [
                    "papermill", qa_nb, out_nb,
                    "-p", "cohort_name", "opioid_ed",
                    "-p", "cohort_parquet_path", opioid_ed_s3_path,
                    "-p", "age_band", str(age_band),
                    "-p", "event_year", str(event_year),
                ]
                logger.info(f"→ [PHASE 4] Running QA notebook: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                logger.info(f"✓ QA notebook completed: {out_nb}")
            except Exception as nb_e:
                logger.warning(f"⚠ QA notebook failed for opioid_ed: {nb_e}")
        
        # Save ED_NON_OPIOID cohort (always save, even if control-only)
        ed_non_opioid_s3_path = get_cohort_parquet_path("ed_non_opioid", age_band, event_year)
        ed_non_opioid_local = None
        if ed_non_opioid_count > 0:
            # Write to local NVMe first (much faster, especially for large cohorts)
            ed_non_opioid_local = local_staging / f"ed_non_opioid_{age_band}_{event_year}.parquet"
            logger.info(f"→ [PHASE 4] Writing ED_NON_OPIOID cohort ({ed_non_opioid_count:,} patients) to local: {ed_non_opioid_local}")
            cohort_conn_duckdb.sql(f"""
            COPY ed_non_opioid_cohort TO '{ed_non_opioid_local}' 
            (FORMAT PARQUET, COMPRESSION SNAPPY)
            """)
            
            # Log file size before upload (helps diagnose timeouts vs IAM/network failures)
            file_size_gb = ed_non_opioid_local.stat().st_size / 1e9
            logger.info(f"→ [PHASE 4] ED_NON_OPIOID cohort written to local ({file_size_gb:.2f} GB)")
            
            # Sync to S3 using aws s3 cp (more reliable for large files, can resume on failure)
            logger.info(f"→ [PHASE 4] Syncing ED_NON_OPIOID cohort to S3: {ed_non_opioid_s3_path}")
            local_file = str(ed_non_opioid_local)
            
            # Use cached AWS CLI (resolved once at top of phase)
            try:
                result = subprocess.run(
                    [aws_cli, "s3", "cp", local_file, ed_non_opioid_s3_path, "--no-progress"],
                    capture_output=True,
                    text=True,
                    timeout=7200  # 2 hour timeout for very large cohorts
                )
                if result.returncode == 0:
                    logger.info(f"→ [PHASE 4] ED_NON_OPIOID cohort synced to S3 successfully")
                    # Clean up local file after successful sync
                    try:
                        ed_non_opioid_local.unlink()
                        logger.info(f"→ [PHASE 4] Cleaned up local ED_NON_OPIOID cohort file")
                        ed_non_opioid_local = None  # Mark as cleaned
                    except Exception as e:
                        logger.warning(f"⚠️ [PHASE 4] Could not clean up local file: {e}")
                else:
                    logger.error(f"❌ [PHASE 4] Failed to sync ED_NON_OPIOID cohort to S3: {result.stderr}")
                    # Keep local file for retry/debugging
                    logger.warning(f"⚠️ [PHASE 4] Keeping local file for retry: {ed_non_opioid_local}")
                    raise Exception(f"S3 sync failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                logger.error(f"❌ [PHASE 4] S3 sync timeout for ED_NON_OPIOID cohort (exceeded 2 hours)")
                # Keep local file for retry
                logger.warning(f"⚠️ [PHASE 4] Keeping local file for retry: {ed_non_opioid_local}")
                raise
            except FileNotFoundError:
                logger.error(f"❌ [PHASE 4] AWS CLI not found at {aws_cli}, cannot sync to S3")
                raise Exception("AWS CLI not available")
            
            # Check if it's control-only
            # NOTE: 'target' column is legacy and not used in Phase 4 logic
            # Use 'is_target_case' for actual target/control distinction
            # Use patient-level count to avoid row explosion issues
            target_count_check_df = cohort_conn_duckdb.sql("SELECT CAST(COUNT(DISTINCT mi_person_key) AS BIGINT) AS count FROM ed_non_opioid_cohort WHERE is_target_case = 1").fetchdf()
            target_count_check = int(target_count_check_df.iloc[0]['count']) if not target_count_check_df.empty else 0
            if target_count_check == 0:
                logger.info(f"→ [PHASE 4] ED_NON_OPIOID cohort saved (CONTROL-ONLY) to S3: {ed_non_opioid_s3_path}")
            else:
                logger.info(f"→ [PHASE 4] ED_NON_OPIOID cohort saved to S3: {ed_non_opioid_s3_path}")
        else:
            logger.warning(f"⚠️ [PHASE 4] Skipping save of empty ED_NON_OPIOID cohort to {ed_non_opioid_s3_path}")

        # Optional: run QA notebook for ed_non_opioid cohort if configured
        qa_nb = os.environ.get("PGX_QA_NOTEBOOK")
        if qa_nb:
            try:
                out_nb = f"/tmp/Cohort_QA_ed_non_opioid_{age_band}_{event_year}.ipynb"
                cmd = [
                    "papermill", qa_nb, out_nb,
                    "-p", "cohort_name", "ed_non_opioid",
                    "-p", "cohort_parquet_path", ed_non_opioid_s3_path,
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
        
        # Clean up staging directory if empty (all files successfully uploaded and removed)
        try:
            if local_staging.exists():
                # Check if staging directory is empty
                remaining_files = list(local_staging.glob("*.parquet"))
                if not remaining_files:
                    # Directory is empty, but keep it for future use (no need to remove)
                    logger.debug(f"→ [PHASE 4] Staging directory is empty: {local_staging}")
                else:
                    logger.warning(f"⚠️ [PHASE 4] Staging directory still contains {len(remaining_files)} file(s): {[f.name for f in remaining_files]}")
        except Exception as e:
            logger.warning(f"⚠️ [PHASE 4] Could not check staging directory: {e}")
        
        # Monitor disk space at end (already monitored before writes)
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
        
        # Clean up any remaining local files on error (optional - comment out to keep for debugging)
        # Note: Files are kept by default for retry/debugging, but can be cleaned up if desired
        try:
            if 'opioid_ed_local' in locals() and opioid_ed_local and opioid_ed_local.exists():
                logger.warning(f"⚠️ [PHASE 4] Local OPIOID_ED file remains: {opioid_ed_local} (kept for retry/debugging)")
            if 'ed_non_opioid_local' in locals() and ed_non_opioid_local and ed_non_opioid_local.exists():
                logger.warning(f"⚠️ [PHASE 4] Local ED_NON_OPIOID file remains: {ed_non_opioid_local} (kept for retry/debugging)")
        except Exception:
            pass
        
        if pipeline_state:
            pipeline_state.mark_step_failed(step_name, str(e))
        cleanup_duckdb_temp_files(logger)
        raise

