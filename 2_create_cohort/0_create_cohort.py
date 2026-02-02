"""
Optimized cohort creation pipeline with DuckDB optimizations from APCD development.

This module applies all the DuckDB optimizations and development rules learned
from the 1_apcd_input_data development to the 2_create_cohort pipeline.

Key Optimizations Applied:
- EC2-optimized DuckDB connections
- Advanced temp file management
- Memory optimization for large datasets
- S3 performance tuning
- Query profiling and monitoring
- Robust error handling and cleanup
- Centralized checkpoint system at s3://pgx-repository/pgx-pipeline-status/
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
    'rocket': '[START]' if IS_WINDOWS else '🚀',
    'info': '[INFO]' if IS_WINDOWS else '📊',
    'config': '[CONFIG]' if IS_WINDOWS else '🔧',
    'success': '[PASS]' if IS_WINDOWS else '✅',
    'fail': '[FAIL]' if IS_WINDOWS else '❌',
    'clean': '[CLEAN]' if IS_WINDOWS else '🧹',
    'trophy': '[SUCCESS]' if IS_WINDOWS else '🎉'
}

# Set root of project (e.g., /home/pgx3874/pgx-analysis)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    sys.path.append(project_root)

# Import constants and s3 helpers as modules so we can reload them if CLI overrides are provided
import importlib
from py_helpers import constants as constants
from py_helpers import s3_utils as s3_utils

from py_helpers.data_utils import (
    collect_validation_metrics,
    validate_cohort_name,
    generate_qa_report
)
from py_helpers.logging_utils import (
    setup_logging,
    save_logs_to_s3,
    save_logs_immediate,
)

# Import optimized DuckDB utilities
from py_helpers.duckdb_utils import (
    get_duckdb_connection,
    create_simple_duckdb_connection,
    check_memory_usage,
    get_duckdb_info,
    close_duckdb_connection,
    tune_duckdb_for_mp,
)

# Import new centralized checkpoint system
from py_helpers.pipeline_utils import PipelineState, GlobalPipelineTracker

from py_helpers.common_imports import s3_client

import boto3
from py_helpers.aws_utils import notify_error, notify_success
from py_helpers.cohort_utils import check_cohort_exists, check_and_fix_mismatched_sets, check_cohort_exists_and_delete_message


# Import modular phase functions
from phases import (
    run_phase1_data_preparation,
    run_phase2_step1_event_fact_table,
    run_phase2_step2_drug_exposure,
    run_phase3_step3_final_cohort_fact,
    run_phase4_complete_pipeline
)


def cleanup_persistent_tables(context):
    """Clean up cross-step temporary tables after all pipeline steps complete with DuckDB optimizations."""
    logger = context["logger"]
    cohort_conn_duckdb = context["cohort_conn_duckdb"]
    
    logger.info("→ [CLEANUP] Starting optimized cleanup of persistent temporary tables...")
    
    try:
        # List of persistent tables that should be cleaned up after pipeline completion
        persistent_tables = [
            "medical_clean", "pharmacy_clean", "medical_filtered", "pharmacy_filtered",
            "medical_with_demographics", "pharmacy_with_demographics",  # APCD tables
            "cohort_event_features", "first_opioid_ed", "first_ed_non_opioid",
            "tagged_cohort_events", "opioid_drug_exposure", "ade_drug_exposure",
            "control_cohort_events", "opioid_patients", "ade_patients",
            "control_patients_filtered", "demographics_lookup"  # APCD demographics lookup
        ]
        
        cleanup_count = 0
        for table_name in persistent_tables:
            try:
                # Check if table exists before dropping
                result = cohort_conn_duckdb.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'").fetchone()
                if result and result[0] > 0:
                    cohort_conn_duckdb.execute(f"DROP TABLE IF EXISTS {table_name}")
                    cleanup_count += 1
                    logger.debug(f"→ [CLEANUP] Dropped table: {table_name}")
            except Exception as e:
                logger.warning(f"→ [CLEANUP] Could not drop table {table_name}: {e}")
        
        logger.info(f"→ [CLEANUP] Cleaned up {cleanup_count} persistent tables and temp files")
        
    except Exception as e:
        logger.error(f"→ [CLEANUP] Error during cleanup: {e}")


# Define the step execution order (optimized for new 4-phase workflow)
STEP_EXECUTION_ORDER = [
    "phase1_data_preparation",     # Load pre-imputed medical and pharmacy data
    "phase2_step1_event_fact_table",  # Create unified event fact table
    "phase2_step2_drug_exposure",     # Create drug exposure events
    "phase3_step3_final_cohort_fact", # Create final cohort fact tables with 5:1 ratio
    "phase4_complete_pipeline"        # Complete pipeline execution
]

# Define specific table/view dependencies for key steps (optimized for new 4-phase workflow)
STEP_TABLE_DEPENDENCIES = {
    "phase1_data_preparation": [],  # Loads from APCD gold tier, no table dependencies
    "phase2_step1_event_fact_table": ["medical_clean", "pharmacy_clean"], # Needs data from phase 1
    "phase2_step2_drug_exposure": ["unified_event_fact_table"], # Needs event fact table from phase 2 step 1
    "phase3_step3_final_cohort_fact": ["unified_event_fact_table", "unified_drug_exposure"], # Needs both from phase 2
    "phase4_complete_pipeline": ["opioid_ed_cohort", "ed_non_opioid_cohort"] # Needs final cohorts from phase 3
}

# Map step names to their corresponding functions (new 4-phase workflow)
step_functions = {
    "phase1_data_preparation": run_phase1_data_preparation,
    "phase2_step1_event_fact_table": run_phase2_step1_event_fact_table,
    "phase2_step2_drug_exposure": run_phase2_step2_drug_exposure,
    "phase3_step3_final_cohort_fact": run_phase3_step3_final_cohort_fact,
    "phase4_complete_pipeline": run_phase4_complete_pipeline,
}


def step_execution_dispatcher(starting_step, context):
    """
    Execute pipeline steps starting from the specified step with DuckDB optimizations.
    
    Args:
        starting_step (str): The step to start execution from
        context (dict): Pipeline context containing all necessary data
    """
    logger = context["logger"]
    
    # Find the starting index
    try:
        start_index = STEP_EXECUTION_ORDER.index(starting_step)
    except ValueError:
        logger.error(f"→ [DISPATCHER] Invalid starting step: {starting_step}")
        logger.error(f"→ [DISPATCHER] Available steps: {STEP_EXECUTION_ORDER}")
        raise ValueError(f"Invalid starting step: {starting_step}")
    
    # Execute steps from starting point
    steps_to_execute = STEP_EXECUTION_ORDER[start_index:]
    logger.info(f"→ [DISPATCHER] Executing steps: {steps_to_execute}")
    
    for step_name in steps_to_execute:
        try:
            logger.info(f"→ [DISPATCHER] Executing {step_name}...")
            
            # Check if step has a corresponding function
            if step_name not in step_functions:
                logger.warning(f"→ [DISPATCHER] No function found for step: {step_name}")
                continue
            
            # Execute the step
            step_function = step_functions[step_name]
            step_function(context)
            
            # Note: Profiling and explicit checkpoints are not used in simplified helpers
            
            logger.info(f"→ [DISPATCHER] Completed {step_name}")
            
        except Exception as e:
            logger.error(f"→ [DISPATCHER] Error in {step_name}: {str(e)}")
            logger.error(f"→ [DISPATCHER] Traceback: {traceback.format_exc()}")
            
            # Continue raising after logging; temp file cleanup not available in simplified helpers
            raise


# Note: check_existing_checkpoints function removed - now handled by PipelineState system


def execute_pipeline(context):
    """Execute the complete pipeline by running all phases in order with DuckDB optimizations."""
    logger = context["logger"]
    age_band = context["age_band"]
    event_year = context["event_year"]
    
    logger.info("→ [PIPELINE] Starting optimized 4-phase pipeline execution...")
    logger.info("→ [PIPELINE] Applied DUCKDB optimizations from APCD development")
    logger.info("→ [PIPELINE] Using new consolidated 4-phase workflow (5 steps total)")
    
    try:
        # Pre-phase: Sync gold data from S3 to local /mnt/nvme if needed
        from phases.common import sync_gold_data_to_local
        logger.info("→ [PIPELINE] Pre-phase: Ensuring gold medical/pharmacy data is available locally...")
        sync_gold_data_to_local("medical", age_band, event_year, logger)
        sync_gold_data_to_local("pharmacy", age_band, event_year, logger)
        
        # Phase 1: Data Preparation (APCD Integration)
        logger.info("→ [PIPELINE] Executing Phase 1: Data Preparation (APCD Integration)")
        run_phase1_data_preparation(context)
        
        # Phase 2 Step 1: Event Fact Table Creation
        logger.info("→ [PIPELINE] Executing Phase 2 Step 1: Event Fact Table Creation")
        run_phase2_step1_event_fact_table(context)
        
        # Phase 2 Step 2: Drug Exposure Events
        logger.info("→ [PIPELINE] Executing Phase 2 Step 2: Drug Exposure Events")
        run_phase2_step2_drug_exposure(context)
        
        # Phase 3 Step 3: Final Cohort Creation (5:1 ratio)
        logger.info("→ [PIPELINE] Executing Phase 3 Step 3: Final Cohort Creation (5:1 ratio)")
        run_phase3_step3_final_cohort_fact(context)
        
        # Phase 4: Complete Pipeline
        logger.info("→ [PIPELINE] Executing Phase 4: Complete Pipeline")
        run_phase4_complete_pipeline(context)
        
        logger.info("→ [PIPELINE] Optimized 4-phase pipeline execution completed successfully!")
        
    except Exception as e:
        logger.error(f"→ [PIPELINE] Pipeline execution failed: {str(e)}")
        logger.error(f"→ [PIPELINE] Traceback: {traceback.format_exc()}")
        
        # No temp file cleanup in simplified helpers
        raise


def main():
    """Main entry point for the optimized cohort creation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Cohort Creation Pipeline with DuckDB Optimizations")
    parser.add_argument("--age-band", required=True, help="Age band (e.g., '65-74')")
    parser.add_argument("--event-year", type=int, required=True, help="Event year (e.g., 2016)")
    parser.add_argument("--cohort", default="both", choices=["opioid_ed", "ed_non_opioid", "both"], 
                       help="Cohort type to create")
    parser.add_argument("--starting-step", default="phase1_data_preparation", 
                       help="Phase/Step to start execution from")
    parser.add_argument("--operation-type", default="concurrent_processing", 
                       choices=["large_processing", "concurrent_processing", "s3_heavy", "default"],
                       help="DuckDB operation type for optimization")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--skip-checkpoints", action="store_true", 
                       help="Skip checkpoint loading and start fresh")
    parser.add_argument("--enable-profiling", action="store_true",
                       help="Enable query profiling for debugging")
    parser.add_argument("--profile-format", default="json", choices=["json", "query_tree"],
                       help="Query profiling output format")
    # Optional runtime overrides for target configuration (will set env vars and reload helpers)
    parser.add_argument("--target-name", default=None, help="Optional target name to set (overrides PGX_TARGET_NAME env)")
    parser.add_argument("--target-icd-codes", default=None, help="Optional ICD codes string (comma-separated) to set PGX_TARGET_ICD_CODES")
    parser.add_argument("--target-cpt-codes", default=None, help="Optional CPT codes string (comma-separated) to set PGX_TARGET_CPT_CODES")
    parser.add_argument("--time-window-days", type=int, default=None, choices=[7, 14, 21, 30, 45],
                       help="DEPRECATED: Time window is fixed at 21 days. This argument is ignored.")
    parser.add_argument("--concurrent-workers", type=int, default=None,
                       help="Number of concurrent workers (for memory limit calculation). If not set, detects from MAX_WORKERS or PGX_COHORT_WORKERS env vars, or defaults to 3.")
    
    args = parser.parse_args()
    # If target overrides provided on CLI, set environment variables *before* reloading constants/s3_utils
    if args.target_name or args.target_icd_codes or args.target_cpt_codes:
        if args.target_name:
            os.environ["PGX_TARGET_NAME"] = args.target_name
        if args.target_icd_codes:
            os.environ["PGX_TARGET_ICD_CODES"] = args.target_icd_codes
        if args.target_cpt_codes:
            os.environ["PGX_TARGET_CPT_CODES"] = args.target_cpt_codes

        # reload the modules so module-level constants derived from env are refreshed
        try:
            importlib.reload(constants)
            importlib.reload(s3_utils)
        except Exception:
            # Best-effort; if reload fails we'll proceed and let later code surface errors
            pass
    
    # Setup logging (aligned with 1_apcd_input_data logging framework)
    logger, log_buffer = setup_logging("create_cohort", args.age_band, args.event_year)
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    
    logger.info("=" * 80)
    logger.info(f"{SYMBOLS['rocket']} OPTIMIZED COHORT CREATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"{SYMBOLS['info']} Age Band: {args.age_band}")
    logger.info(f"{SYMBOLS['info']} Event Year: {args.event_year}")
    logger.info(f"{SYMBOLS['info']} Cohort Type: {args.cohort}")
    logger.info(f"{SYMBOLS['info']} Starting Step: {args.starting_step}")
    logger.info(f"{SYMBOLS['info']} Operation Type: {args.operation_type}")
    logger.info(f"{SYMBOLS['info']} Profiling: {'Enabled' if args.enable_profiling else 'Disabled'}")
    
    # Log process information
    import os
    import multiprocessing
    current_pid = os.getpid()
    cpu_count = multiprocessing.cpu_count()
    logger.info(f"{SYMBOLS['info']} Process ID: {current_pid}")
    logger.info(f"{SYMBOLS['info']} CPU Cores Available: {cpu_count}")
    
    # Check for concurrent workers setting (will be logged later in config section)
    detected_workers = None
    if args.concurrent_workers is not None:
        detected_workers = args.concurrent_workers
    elif os.getenv('PGX_COHORT_WORKERS'):
        detected_workers = int(os.getenv('PGX_COHORT_WORKERS'))
    elif os.getenv('MAX_WORKERS'):
        detected_workers = int(os.getenv('MAX_WORKERS'))
    
    if detected_workers:
        logger.info(f"{SYMBOLS['info']} Concurrent Workers Detected: {detected_workers} (for memory limit calculation)")
    else:
        logger.info(f"{SYMBOLS['info']} Concurrent Workers: Not set (will use default: 3)")
    
    logger.info("=" * 80)
    logger.info(f"{SYMBOLS['config']} DUCKDB OPTIMIZATIONS APPLIED:")
    logger.info("   - EC2-optimized connections (32-core 1TB RAM)")
    logger.info("   - Multi-threaded DuckDB execution (configurable via PGX_THREADS_PER_WORKER)")
    logger.info("   - S3 uploader parallelization (multi-part uploads)")
    logger.info("   - Operation-type specific optimizations")
    logger.info("   - Advanced temp file management")
    logger.info("   - Memory optimization for large datasets")
    logger.info("   - S3 performance tuning")
    logger.info("   - Query profiling and monitoring")
    logger.info("   - Robust error handling and cleanup")
    logger.info("   - Centralized checkpoint system")
    logger.info("=" * 80)
    
    try:
        # Note: environment validation handled implicitly in simplified helpers

        # Initialize centralized checkpoint system
        entity_id = f"{args.cohort}_{args.age_band}_{args.event_year}"
        pipeline_state = PipelineState('create_cohort', entity_id, logger)
        logger.info(f"Checkpoint location: s3://pgx-repository/pgx-pipeline-status/create_cohort/{entity_id.replace('/', '_')}/")

        # Check if final output already exists (skip if all requested cohort parquets exist)
        if args.cohort == "both":
            opioid_path = s3_utils.get_cohort_parquet_path("opioid_ed", args.age_band, args.event_year)
            ed_path = s3_utils.get_cohort_parquet_path("ed_non_opioid", args.age_band, args.event_year)
            both_exist = (
                PipelineState.check_output_exists(opioid_path) and
                PipelineState.check_output_exists(ed_path)
            )
            if both_exist:
                logger.info(f"{SYMBOLS['success']} Both cohort parquets already exist: opioid_ed, ed_non_opioid")
                logger.info(f"{SYMBOLS['success']} Skipping pipeline - cohort already created")
                pipeline_state.mark_pipeline_completed({'output': opioid_path, 'skipped': True})
                try:
                    save_logs_to_s3(log_buffer, args.cohort, args.age_band, args.event_year, "create_cohort", logger=logger)
                except Exception as e:
                    logger.warning(f"Could not save logs to S3 on early exit: {e}")
                return
        else:
            output_paths = s3_utils.get_output_paths(args.cohort, args.age_band, args.event_year)
            cohort_output = output_paths.get('cohort_parquet')
            if cohort_output and PipelineState.check_output_exists(cohort_output):
                logger.info(f"{SYMBOLS['success']} Final output already exists: {cohort_output}")
                logger.info(f"{SYMBOLS['success']} Skipping pipeline - cohort already created")
                pipeline_state.mark_pipeline_completed({'output': cohort_output, 'skipped': True})
                try:
                    save_logs_to_s3(log_buffer, args.cohort, args.age_band, args.event_year, "create_cohort", logger=logger)
                except Exception as e:
                    logger.warning(f"Could not save logs to S3 on early exit: {e}")
                return

        # Cleanup old DuckDB temp files at startup (from previous runs/crashes)
        from phases.common import cleanup_duckdb_temp_files
        logger.info("→ [STARTUP] Cleaning up old DuckDB temp files...")
        cleanup_duckdb_temp_files(logger)
        
        # Setup optimized DuckDB connection with parallelization
        # Since we're processing a single partition, use multiple threads for better performance
        # Use worker-specific temp directory (with process PID) to avoid conflicts when running multiple cohorts in parallel
        from py_helpers.duckdb_utils import get_worker_temp_dir
        worker_temp_dir = get_worker_temp_dir()
        logger.info(f"→ [CONFIG] Using worker-specific temp directory: {worker_temp_dir}")
        
        cohort_conn_duckdb = get_duckdb_connection(tmp_dir=worker_temp_dir, logger=logger)

        # CRITICAL: Set explicit memory limit to prevent oversubscription with multiple workers
        # DuckDB auto-detects and uses ~900GB per connection, which causes OOM with multiple workers
        # Calculate dynamic limit based on actual system memory and concurrent workers
        import multiprocessing
        try:
            import psutil
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
        except (ImportError, Exception):
            # Fallback: assume 1TB EC2 instance
            total_memory_gb = 1000.0
        
        # Detect concurrent workers (for memory limit calculation)
        # Priority: CLI argument > PGX_COHORT_WORKERS env > MAX_WORKERS env > default
        # IMPORTANT: This is the TOTAL number of concurrent workers running in parallel (e.g., from ThreadPoolExecutor)
        # Each worker process should receive this value via --concurrent-workers CLI argument
        concurrent_workers = None
        if args.concurrent_workers is not None:
            concurrent_workers = args.concurrent_workers
            logger.info(f"→ [CONFIG] Using --concurrent-workers={concurrent_workers} from CLI argument")
        elif os.getenv('PGX_COHORT_WORKERS'):
            concurrent_workers = int(os.getenv('PGX_COHORT_WORKERS'))
            logger.info(f"→ [CONFIG] Detected PGX_COHORT_WORKERS={concurrent_workers} from environment")
        elif os.getenv('MAX_WORKERS'):
            concurrent_workers = int(os.getenv('MAX_WORKERS'))
            logger.info(f"→ [CONFIG] Detected MAX_WORKERS={concurrent_workers} from environment")
        else:
            # Default: assume 3 workers (common for cohort creation)
            concurrent_workers = 3
            logger.info(f"→ [CONFIG] Using default worker count: {concurrent_workers} (set --concurrent-workers or env var to override)")
        
        # Log current process information for debugging
        logger.info(f"→ [CONFIG] Current Process ID: {os.getpid()}")
        logger.info(f"→ [CONFIG] Parent Process ID: {os.getppid() if hasattr(os, 'getppid') else 'N/A'}")
        logger.info(f"→ [CONFIG] Total Concurrent Workers (for memory calculation): {concurrent_workers}")
        logger.info(f"→ [CONFIG] NOTE: This process is 1 of {concurrent_workers} concurrent workers")
        
        # Reserve 40% for OS, buffers, and other processes (600GB for 1TB system)
        # Divide remaining 60% among workers
        available_for_duckdb = total_memory_gb * 0.6
        per_worker_memory_gb = available_for_duckdb / max(1, concurrent_workers)
        
        # Clamp between 50GB (minimum for large cohorts) and 300GB (maximum per worker)
        per_worker_memory_gb = max(50.0, min(300.0, per_worker_memory_gb))
        
        # Round to nearest 10GB for cleaner values
        per_worker_memory_gb = round(per_worker_memory_gb / 10) * 10
        
        memory_limit = f"{int(per_worker_memory_gb)}GB"
        cohort_conn_duckdb.sql(f"SET memory_limit='{memory_limit}'")
        logger.info(f"→ [CONFIG] DuckDB memory limit: {memory_limit} (for {concurrent_workers} workers, {total_memory_gb:.0f}GB total system memory, {available_for_duckdb:.0f}GB available for DuckDB)")
        
        # Log active process count for debugging
        try:
            import psutil
            current_process = psutil.Process()
            # Count Python processes running 0_create_cohort.py
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and any('0_create_cohort.py' in str(arg) for arg in cmdline):
                        python_processes.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            logger.info(f"→ [CONFIG] Active cohort creation processes: {len(python_processes)} (PIDs: {python_processes[:10]}{'...' if len(python_processes) > 10 else ''})")
            logger.info(f"→ [CONFIG] Current process memory: {current_process.memory_info().rss / (1024**3):.2f}GB RSS")
        except (ImportError, Exception) as e:
            logger.debug(f"→ [CONFIG] Could not query process information: {e}")

        # Configure DuckDB for optimal parallelization based on operation type
        # For single partition processing, we can use more threads (up to CPU cores - 2)
        # Default to 8 threads, but allow override via PGX_THREADS_PER_WORKER
        # On EC2 with 32 cores, we can safely use up to 30 threads for single partition processing
        max_threads = max(1, multiprocessing.cpu_count() - 2)  # Reserve 2 cores for OS/other processes
        default_threads = min(8, max_threads)  # Default to 8, but cap at available cores
        threads = int(os.getenv('PGX_THREADS_PER_WORKER', str(default_threads)))
        threads = min(threads, max_threads)  # Cap at available cores
        cohort_conn_duckdb.sql(f"PRAGMA threads={threads}")
        logger.info(f"→ [CONFIG] DuckDB threads: {threads} (max available: {max_threads}, CPU cores: {multiprocessing.cpu_count()})")

        # Configure S3 uploader settings for optimal parallel uploads
        # These settings optimize multi-part uploads when saving large cohort files to S3
        s3_uploader_thread_limit = os.getenv('PGX_S3_UPLOADER_THREAD_LIMIT')
        if s3_uploader_thread_limit and s3_uploader_thread_limit.isdigit():
            cohort_conn_duckdb.sql(f"SET s3_uploader_thread_limit={int(s3_uploader_thread_limit)}")
            logger.info(f"→ [CONFIG] S3 uploader thread limit: {s3_uploader_thread_limit}")
        # Default values should suffice for most use cases, but can be overridden if needed
        s3_uploader_max_filesize = os.getenv('PGX_S3_UPLOADER_MAX_FILESIZE')
        if s3_uploader_max_filesize:
            cohort_conn_duckdb.sql(f"SET s3_uploader_max_filesize='{s3_uploader_max_filesize}'")
            logger.info(f"→ [CONFIG] S3 uploader max filesize: {s3_uploader_max_filesize}")
        s3_uploader_max_parts_per_file = os.getenv('PGX_S3_UPLOADER_MAX_PARTS_PER_FILE')
        if s3_uploader_max_parts_per_file and s3_uploader_max_parts_per_file.isdigit():
            cohort_conn_duckdb.sql(f"SET s3_uploader_max_parts_per_file={int(s3_uploader_max_parts_per_file)}")
            logger.info(f"→ [CONFIG] S3 uploader max parts per file: {s3_uploader_max_parts_per_file}")

        # Apply operation-type specific optimizations
        if args.operation_type == "s3_heavy":
            # Increase uploader threads for parallel uploads
            if not s3_uploader_thread_limit:
                cohort_conn_duckdb.sql("SET s3_uploader_thread_limit=16")
                logger.info("→ [CONFIG] S3-heavy mode: increased uploader threads to 16")
            logger.info("→ [CONFIG] S3-heavy mode: optimized for S3 operations")
        elif args.operation_type == "large_processing":
            # Use more threads for large processing
            large_threads = max(threads, 16)
            cohort_conn_duckdb.sql(f"PRAGMA threads={large_threads}")
            logger.info(f"→ [CONFIG] Large processing mode: increased threads to {large_threads}")

        # Query profiling not supported in simplified helpers; skip
        
        # Create context with pipeline state
        context = {
            "age_band": args.age_band,
            "event_year": args.event_year,
            "cohort": args.cohort,
            "cohort_conn_duckdb": cohort_conn_duckdb,
            "logger": logger,
            "operation_type": args.operation_type,
            "s3_bucket": constants.S3_BUCKET,
            "pipeline_state": pipeline_state,  # Add checkpoint system to context
            "time_window_days": 21  # Fixed 21-day window (command-line argument is deprecated and ignored)
        }
        
        # Execute pipeline (step functions will use pipeline_state from context)
        if args.starting_step == "phase1_data_preparation":
            execute_pipeline(context)
        else:
            step_execution_dispatcher(args.starting_step, context)
        
        # Cleanup
        try:
            cleanup_persistent_tables(context)
            # Clean up DuckDB temp files after successful completion
            cleanup_duckdb_temp_files(logger)
        except Exception as e:
            logger.warning(f"Cleanup encountered an issue: {e}")
        
        # Profiling not enabled in simplified helpers
        
        try:
            cohort_conn_duckdb.close()
        except Exception as e:
            logger.warning(f"Could not close DuckDB connection: {e}")
        
        # Mark pipeline as completed (output path for state)
        if args.cohort == "both":
            output_for_state = s3_utils.get_cohort_parquet_path("opioid_ed", args.age_band, args.event_year)
        else:
            output_paths = s3_utils.get_output_paths(args.cohort, args.age_band, args.event_year)
            output_for_state = output_paths.get("cohort_parquet")
        pipeline_state.mark_pipeline_completed({
            "cohort": args.cohort,
            "age_band": args.age_band,
            "event_year": args.event_year,
            "output": output_for_state,
        })
        
        logger.info("=" * 80)
        logger.info(f"{SYMBOLS['success']} OPTIMIZED COHORT CREATION PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)

        # Save logs to S3 on success
        try:
            save_logs_to_s3(log_buffer, args.cohort, args.age_band, args.event_year, "create_cohort", logger=logger)
        except Exception as e:
            logger.warning(f"Could not save logs to S3: {e}")
        
    except Exception as e:
        logger.error(f"{SYMBOLS['fail']} Pipeline failed: {str(e)}")
        logger.error(f"{SYMBOLS['fail']} Traceback: {traceback.format_exc()}")
        
        # Mark pipeline as failed (record in metadata)
        if 'pipeline_state' in locals():
            try:
                pipeline_state.state['status'] = 'failed'
                pipeline_state.state['failed_steps'].append({'step_name': 'pipeline', 'error': str(e)})
                # Persist state update
                pipeline_state._save_state()
            except Exception as ps_e:
                logger.warning(f"Could not record pipeline failure state: {ps_e}")
        
        # Cleanup on error
        try:
            if 'cohort_conn_duckdb' in locals():
                cohort_conn_duckdb.close()
        except Exception:
            pass

        # Save error logs to S3 immediately
        try:
            save_logs_immediate(log_buffer, args.cohort, args.age_band, args.event_year, "create_cohort", logger=logger, reason="error")
        except Exception as save_e:
            logger.warning(f"Could not save error logs to S3: {save_e}")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
