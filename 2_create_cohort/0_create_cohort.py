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
from helpers_1997_13 import constants as constants
from helpers_1997_13 import s3_utils as s3_utils

from helpers_1997_13.data_utils import (
    collect_validation_metrics,
    validate_cohort_name,
    generate_qa_report
)
from helpers_1997_13.logging_utils import (
    setup_logging,
    save_logs_to_s3,
    save_logs_immediate,
)

# Import optimized DuckDB utilities
from helpers_1997_13.duckdb_utils import (
    get_duckdb_connection,
    create_simple_duckdb_connection,
    check_memory_usage,
    get_duckdb_info,
    close_duckdb_connection,
    tune_duckdb_for_mp,
)

# Import new centralized checkpoint system
from helpers_1997_13.pipeline_utils import PipelineState, GlobalPipelineTracker

from helpers_1997_13.common_imports import s3_client

import boto3
from helpers_1997_13.aws_utils import notify_error, notify_success
from helpers_1997_13.cohort_utils import check_cohort_exists, check_and_fix_mismatched_sets, check_cohort_exists_and_delete_message


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
    
    logger.info("→ [PIPELINE] Starting optimized 4-phase pipeline execution...")
    logger.info("→ [PIPELINE] Applied DUCKDB optimizations from APCD development")
    logger.info("→ [PIPELINE] Using new consolidated 4-phase workflow (5 steps total)")
    
    try:
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
    logger.info("=" * 80)
    logger.info(f"{SYMBOLS['config']} DUCKDB OPTIMIZATIONS APPLIED:")
    logger.info("   - EC2-optimized connections (32-core 1TB RAM)")
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

        # Check if final output already exists
        output_paths = s3_utils.get_output_paths(args.cohort, args.age_band, args.event_year)
        cohort_output = output_paths.get('cohort_parquet')
        if cohort_output and PipelineState.check_output_exists(cohort_output):
            logger.info(f"{SYMBOLS['success']} Final output already exists: {cohort_output}")
            logger.info(f"{SYMBOLS['success']} Skipping pipeline - cohort already created")
            pipeline_state.mark_pipeline_completed({'output': cohort_output, 'skipped': True})
            # Persist logs to S3 before exit (consistent with APCD logging)
            try:
                save_logs_to_s3(log_buffer, args.cohort, args.age_band, args.event_year, "create_cohort", logger=logger)
            except Exception as e:
                logger.warning(f"Could not save logs to S3 on early exit: {e}")
            return

        # Setup simplified DuckDB connection (helpers_1997_13)
        cohort_conn_duckdb = get_duckdb_connection(logger=logger)
        
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
            "pipeline_state": pipeline_state  # Add checkpoint system to context
        }
        
        # Execute pipeline (step functions will use pipeline_state from context)
        if args.starting_step == "phase1_data_preparation":
            execute_pipeline(context)
        else:
            step_execution_dispatcher(args.starting_step, context)
        
        # Cleanup
        try:
            cleanup_persistent_tables(context)
        except Exception as e:
            logger.warning(f"Cleanup encountered an issue: {e}")
        
        # Profiling not enabled in simplified helpers
        
        try:
            cohort_conn_duckdb.close()
        except Exception as e:
            logger.warning(f"Could not close DuckDB connection: {e}")
        
        # Mark pipeline as completed
        pipeline_state.mark_pipeline_completed({
            'cohort': args.cohort,
            'age_band': args.age_band,
            'event_year': args.event_year,
            'output': cohort_output
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
