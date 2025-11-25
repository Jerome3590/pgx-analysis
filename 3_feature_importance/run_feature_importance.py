"""
Feature Importance Analysis with Monte Carlo Cross-Validation

This script provides a Python wrapper for the R-based feature importance analysis,
with comprehensive logging setup aligned with the 2_create_cohort pipeline.

Key Features:
- Monte Carlo Cross-Validation with CatBoost and Random Forest
- Comprehensive logging to console, file, and S3
- Windows emoji compatibility
- Error handling and log persistence
- Integration with R notebook execution
"""

import os
import sys
import traceback
import logging
import platform
import subprocess
from datetime import datetime
from pathlib import Path

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

# Import constants and s3 helpers as modules
import importlib
from helpers_1997_13 import constants as constants
from helpers_1997_13 import s3_utils as s3_utils

# Import logging utilities
from helpers_1997_13.logging_utils import (
    setup_logging,
    save_logs_to_s3,
    save_logs_immediate,
)


def main():
    """Main entry point for feature importance analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Feature Importance Analysis with Monte Carlo Cross-Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run analysis for opioid_ed cohort, age 25-44, year 2016
  python run_feature_importance.py --cohort opioid_ed --age-band 25-44 --event-year 2016

  # Run in debug mode (5 splits, quick test)
  python run_feature_importance.py --cohort opioid_ed --age-band 25-44 --event-year 2016 --debug

  # Run with custom number of splits
  python run_feature_importance.py --cohort opioid_ed --age-band 25-44 --event-year 2016 --n-splits 1000

  # Run with logloss scaling instead of recall
  python run_feature_importance.py --cohort opioid_ed --age-band 25-44 --event-year 2016 --scaling-metric logloss
        """
    )
    
    parser.add_argument(
        '--cohort',
        type=str,
        required=True,
        choices=['opioid_ed', 'non_opioid_ed'],
        help='Cohort name: opioid_ed or non_opioid_ed'
    )
    
    parser.add_argument(
        '--age-band',
        type=str,
        required=True,
        help='Age band (e.g., "25-44", "45-54")'
    )
    
    parser.add_argument(
        '--event-year',
        type=int,
        required=True,
        help='Event year (e.g., 2016, 2017)'
    )
    
    parser.add_argument(
        '--n-splits',
        type=int,
        default=200,
        help='Number of MC-CV splits (default: 200, use 5 for debug, 1000 for production)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (5 splits, quick test)'
    )
    
    parser.add_argument(
        '--scaling-metric',
        type=str,
        default='recall',
        choices=['recall', 'logloss'],
        help='Scaling metric for feature importance (default: recall)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--notebook-path',
        type=str,
        default=None,
        help='Path to feature_importance_mc_cv.ipynb (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Setup logging (aligned with 2_create_cohort logging framework)
    logger, log_buffer = setup_logging("feature_importance", args.age_band, args.event_year)
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    
    logger.info("=" * 80)
    logger.info(f"{SYMBOLS['rocket']} FEATURE IMPORTANCE ANALYSIS - MONTE CARLO CROSS-VALIDATION")
    logger.info("=" * 80)
    logger.info(f"{SYMBOLS['info']} Cohort: {args.cohort}")
    logger.info(f"{SYMBOLS['info']} Age Band: {args.age_band}")
    logger.info(f"{SYMBOLS['info']} Event Year: {args.event_year}")
    logger.info(f"{SYMBOLS['info']} MC-CV Splits: {5 if args.debug else args.n_splits}")
    logger.info(f"{SYMBOLS['info']} Scaling Metric: {args.scaling_metric}")
    logger.info(f"{SYMBOLS['info']} Debug Mode: {'Enabled' if args.debug else 'Disabled'}")
    logger.info("=" * 80)
    logger.info(f"{SYMBOLS['config']} CONFIGURATION:")
    logger.info("   - Monte Carlo Cross-Validation with stratified sampling")
    logger.info("   - CatBoost and Random Forest models")
    logger.info("   - Feature importance scaled by MC-CV performance")
    logger.info("   - Parallel processing with furrr/future")
    logger.info("   - Comprehensive logging to console, file, and S3")
    logger.info("=" * 80)
    
    try:
        # Determine notebook path
        script_dir = Path(__file__).parent
        if args.notebook_path:
            notebook_path = Path(args.notebook_path)
        else:
            notebook_path = script_dir / "feature_importance_mc_cv.ipynb"
        
        if not notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")
        
        logger.info(f"{SYMBOLS['info']} Notebook path: {notebook_path}")
        
        # Check if R is available
        try:
            r_version = subprocess.check_output(['R', '--version'], stderr=subprocess.STDOUT, text=True)
            logger.info(f"{SYMBOLS['info']} R version detected:\n{r_version.split(chr(10))[0]}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning(f"{SYMBOLS['fail']} R not found in PATH. Please ensure R is installed and available.")
            raise
        
        # Check if required R packages are available
        logger.info(f"{SYMBOLS['info']} Checking R package dependencies...")
        check_r_packages_script = """
        required_packages <- c("here", "dplyr", "readr", "tidyr", "tibble", "purrr", 
                               "catboost", "randomForest", "rsample", "furrr", "future", 
                               "progressr", "duckdb", "DBI")
        missing_packages <- required_packages[!required_packages %in% installed.packages()[,"Package"]]
        if (length(missing_packages) > 0) {
            cat("MISSING_PACKAGES:", paste(missing_packages, collapse=","), "\n")
            quit(status=1)
        } else {
            cat("ALL_PACKAGES_INSTALLED\n")
        }
        """
        
        try:
            result = subprocess.run(
                ['R', '--slave', '--no-restore', '--no-save'],
                input=check_r_packages_script,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                logger.warning(f"{SYMBOLS['fail']} Some R packages may be missing. Check output:\n{result.stdout}\n{result.stderr}")
            else:
                logger.info(f"{SYMBOLS['success']} All required R packages are installed")
        except subprocess.TimeoutExpired:
            logger.warning(f"{SYMBOLS['fail']} R package check timed out")
        except Exception as e:
            logger.warning(f"{SYMBOLS['fail']} Could not check R packages: {e}")
        
        # Prepare R script to execute notebook cells
        # Note: This is a simplified approach - in practice, you might want to use
        # jupyter nbconvert or a more sophisticated notebook execution method
        logger.info(f"{SYMBOLS['info']} Preparing to execute R notebook...")
        # Log configuration that will be used
        logger.info(f"{SYMBOLS['config']} R Configuration:")
        logger.info(f"   COHORT_NAME <- '{args.cohort}'")
        logger.info(f"   AGE_BAND <- '{args.age_band}'")
        logger.info(f"   EVENT_YEAR <- {args.event_year}")
        logger.info(f"   DEBUG_MODE <- {str(args.debug).upper()}")
        logger.info(f"   N_SPLITS <- {5 if args.debug else args.n_splits}")
        logger.info(f"   SCALING_METRIC <- '{args.scaling_metric}'")
        
        logger.info("=" * 80)
        logger.info(f"{SYMBOLS['info']} Notebook execution:")
        logger.info(f"   Path: {notebook_path}")
        logger.info(f"   To execute: Open in RStudio or Jupyter and run all cells")
        logger.info(f"   Configuration: Set variables in Cell 3 as shown above")
        logger.info("=" * 80)
        
        logger.info(f"{SYMBOLS['success']} Logging setup complete. Ready for analysis execution.")
        
        # Save logs to S3 on completion
        try:
            save_logs_to_s3(log_buffer, args.cohort, args.age_band, args.event_year, "feature_importance", logger=logger)
        except Exception as e:
            logger.warning(f"Could not save logs to S3: {e}")
        
    except Exception as e:
        logger.error(f"{SYMBOLS['fail']} Script failed: {str(e)}")
        logger.error(f"{SYMBOLS['fail']} Traceback: {traceback.format_exc()}")
        
        # Save error logs to S3 immediately
        try:
            save_logs_immediate(log_buffer, args.cohort, args.age_band, args.event_year, "feature_importance", logger=logger, reason="error")
        except Exception as save_e:
            logger.warning(f"Could not save error logs to S3: {save_e}")
        
        sys.exit(1)


if __name__ == "__main__":
    main()

