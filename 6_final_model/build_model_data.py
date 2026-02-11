#!/usr/bin/env python3
"""
Build model data train and test datasets with all feature engineering steps.

This script orchestrates the complete model data build process:
1. Regenerates FP-Growth features (with match-only, no individual support features)
2. Regenerates BupaR features (with target and control patients)
3. Regenerates DTW features (with target and control patients)
4. Regenerates PGx features (with target and control patients)
5. Regenerates predictive time features (with target and control patients)
6. Merges all features into final feature table
7. Removes target leakage features
8. Builds train/test splits with temporal validation

Usage:
    python build_model_data.py --cohort-name opioid_ed --age-band 0-12
"""

import argparse
import sys
import subprocess
from pathlib import Path
import logging
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging

# Setup simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_script(script_path: Path, args: list, description: str) -> bool:
    """Run a Python script and return success status."""
    logger.info(f"Running: {description}")
    logger.info(f"  Script: {script_path}")
    logger.info(f"  Args: {' '.join(args)}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)] + args,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"✓ {description} completed successfully")
        if result.stdout:
            logger.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed")
        logger.error(f"  Error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"✗ {description} failed with exception: {e}")
        return False


def build_model_data(
    project_root: Path,
    cohort_name: str,
    age_band: str,
    skip_feature_engineering: bool = False,
    skip_final_merge: bool = False,
    skip_train_test: bool = False,
) -> None:
    """
    Build model data by running all feature engineering steps and creating train/test splits.
    
    Parameters
    ----------
    project_root : Path
        Project root directory
    cohort_name : str
        Cohort name (e.g., "opioid_ed")
    age_band : str
        Age band (e.g., "0-12")
    skip_feature_engineering : bool
        Skip individual feature engineering steps (use existing features)
    skip_final_merge : bool
        Skip final feature table merge (use existing merged features)
    skip_train_test : bool
        Skip train/test split (use existing splits)
    """
    age_band_fname = age_band.replace("-", "_")
    
    logger.info("=" * 80)
    logger.info(f"Building Model Data: {cohort_name} / {age_band}")
    logger.info("=" * 80)
    
    # Step 1: Feature Engineering Steps
    if not skip_feature_engineering:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Regenerating Feature Engineering Steps")
        logger.info("=" * 80)
        
        # 1.1 FP-Growth Features (with updated match-only logic)
        logger.info("\n--- FP-Growth Features ---")
        fpgrowth_create = PROJECT_ROOT / "4_fpgrowth_analysis" / "create_fpgrowth_features.py"
        if fpgrowth_create.exists():
            success = run_script(
                fpgrowth_create,
                ["--cohort", cohort_name, "--age_band", age_band],
                "Create FP-Growth features (match-only, no individual support)"
            )
            if not success:
                logger.warning("FP-Growth feature creation failed, continuing...")
        else:
            logger.warning(f"FP-Growth script not found: {fpgrowth_create}")
        
        fpgrowth_add = PROJECT_ROOT / "4_fpgrowth_analysis" / "add_fpgrowth_features_to_model_data.py"
        if fpgrowth_add.exists():
            # Check what arguments this script expects
            success = run_script(
                fpgrowth_add,
                ["--cohort-name", cohort_name, "--age-band", age_band],
                "Add FP-Growth features to model data"
            )
            if not success:
                logger.warning("FP-Growth feature addition failed, continuing...")
        
        # 1.2 BupaR Features (with target and control)
        logger.info("\n--- BupaR Features ---")
        bupar_create = PROJECT_ROOT / "5a_bupaR_analysis" / "create_sequence_features.R"
        if bupar_create.exists():
            success = run_script(
                bupar_create,
                ["--cohort", cohort_name, "--age-band", age_band],
                "Create BupaR sequence features (target and control)"
            )
            if not success:
                logger.warning("BupaR sequence feature creation failed, continuing...")
        
        bupar_add = PROJECT_ROOT / "5a_bupaR_analysis" / "add_bupar_features_to_model_data.R"
        if bupar_add.exists():
            success = run_script(
                bupar_add,
                ["--cohort-name", cohort_name, "--age-band", age_band],
                "Add BupaR features to model data"
            )
            if not success:
                logger.warning("BupaR feature addition failed, continuing...")
        
        # 1.3 DTW Features (with target and control)
        logger.info("\n--- DTW Features ---")
        dtw_create = PROJECT_ROOT / "6_dtw_analysis" / "create_dtw_features.py"
        if dtw_create.exists():
            success = run_script(
                dtw_create,
                ["--cohort", cohort_name, "--age_band", age_band],
                "Create DTW features (target and control)"
            )
            if not success:
                logger.warning("DTW feature creation failed, continuing...")
        
        dtw_visuals = PROJECT_ROOT / "9_risk_dashboard" / "visualizations" / "dtw" / "create_dtw_visuals.py"
        if dtw_visuals.exists():
            success = run_script(
                dtw_visuals,
                ["--cohort-name", cohort_name, "--age-band", age_band],
                "Create DTW visuals (publish for dashboard)"
            )
            if not success:
                logger.warning("DTW visuals creation failed, continuing...")
        
        # 1.4 Predictive Time Features (with target and control)
        logger.info("\n--- Predictive Time Features ---")
        time_create = PROJECT_ROOT / "6_dtw_analysis" / "create_predictive_time_features.py"
        if time_create.exists():
            success = run_script(
                time_create,
                ["--cohort-name", cohort_name, "--age-band", age_band],
                "Create predictive time features (target and control)"
            )
            if not success:
                logger.warning("Predictive time feature creation failed, continuing...")
        
        # 1.5 PGx Features (with target and control)
        logger.info("\n--- PGx Features ---")
        pgx_create = PROJECT_ROOT / "7_pgx_analysis" / "create_pgx_features_patient_level.py"
        if pgx_create.exists():
            success = run_script(
                pgx_create,
                ["--cohort", cohort_name, "--age_band", age_band],
                "Create PGx features (target and control)"
            )
            if not success:
                logger.warning("PGx feature creation failed, continuing...")
        
        pgx_add = PROJECT_ROOT / "7_pgx_analysis" / "add_pgx_features_to_model_data.py"
        if pgx_add.exists():
            success = run_script(
                pgx_add,
                ["--cohort-name", cohort_name, "--age-band", age_band],
                "Add PGx features to model data"
            )
            if not success:
                logger.warning("PGx feature addition failed, continuing...")
    else:
        logger.info("Skipping feature engineering steps (using existing features)")
    
    # Step 2: Build Final Feature Table
    if not skip_final_merge:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Building Final Feature Table")
        logger.info("=" * 80)
        
        build_final = PROJECT_ROOT / "8_final_model" / "build_final_cohort_model_features.py"
        if build_final.exists():
            success = run_script(
                build_final,
                ["--cohort-name", cohort_name, "--age-band", age_band],
                "Build final feature table (merge all features)"
            )
            if not success:
                logger.error("Final feature table build failed!")
                return
        else:
            logger.error(f"Final feature table script not found: {build_final}")
            return
    else:
        logger.info("Skipping final feature table merge (using existing merged features)")
    
    # Step 3: Remove Target Leakage
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Removing Target Leakage Features")
    logger.info("=" * 80)
    
    remove_leakage = PROJECT_ROOT / "8_final_model" / "remove_target_leakage.py"
    if remove_leakage.exists():
        success = run_script(
            remove_leakage,
            ["--cohort-name", cohort_name, "--age-band", age_band],
            "Remove target leakage features"
        )
        if not success:
            logger.warning("Target leakage removal failed, continuing...")
    else:
        logger.warning(f"Target leakage removal script not found: {remove_leakage}")
    
    # Step 4: Prepare Train/Test Splits
    if not skip_train_test:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Preparing Train/Test Splits")
        logger.info("=" * 80)
        
        prepare_splits = PROJECT_ROOT / "8_final_model" / "prepare_train_test_s3.py"
        if prepare_splits.exists():
            success = run_script(
                prepare_splits,
                ["--cohort-name", cohort_name, "--age-band", age_band],
                "Prepare train/test splits (temporal validation)"
            )
            if not success:
                logger.error("Train/test split preparation failed!")
                return
        else:
            logger.error(f"Train/test split script not found: {prepare_splits}")
            return
    else:
        logger.info("Skipping train/test split (using existing splits)")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ Model Data Build Complete!")
    logger.info("=" * 80)
    logger.info(f"\nSummary:")
    logger.info(f"  Cohort: {cohort_name}")
    logger.info(f"  Age Band: {age_band}")
    logger.info(f"\nOutput Locations:")
    logger.info(f"  Final Features: 8_final_model/outputs/{cohort_name}/{age_band_fname}/")
    logger.info(f"  Train/Test: 8_final_model/inputs/{cohort_name}/{age_band_fname}/")


def main():
    parser = argparse.ArgumentParser(
        description="Build model data train and test datasets with updated feature engineering"
    )
    parser.add_argument(
        "--cohort-name",
        type=str,
        required=True,
        help="Cohort name (e.g., opioid_ed)",
    )
    parser.add_argument(
        "--age-band",
        type=str,
        required=True,
        help="Age band (e.g., 0-12)",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root path (default: current directory)",
    )
    parser.add_argument(
        "--skip-feature-engineering",
        action="store_true",
        help="Skip individual feature engineering steps (use existing features)",
    )
    parser.add_argument(
        "--skip-final-merge",
        action="store_true",
        help="Skip final feature table merge (use existing merged features)",
    )
    parser.add_argument(
        "--skip-train-test",
        action="store_true",
        help="Skip train/test split (use existing splits)",
    )
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    if project_root.name != "pgx-analysis":
        # Try to find pgx-analysis in parents
        for parent in project_root.parents:
            if parent.name == "pgx-analysis":
                project_root = parent
                break
    
    build_model_data(
        project_root=project_root,
        cohort_name=args.cohort_name,
        age_band=args.age_band,
        skip_feature_engineering=args.skip_feature_engineering,
        skip_final_merge=args.skip_final_merge,
        skip_train_test=args.skip_train_test,
    )


if __name__ == "__main__":
    main()

