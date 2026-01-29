#!/usr/bin/env python3
"""
Module-style orchestration script for DTW feature engineering.

This script runs the DTW workflow:
1. Create predictive time-window features (optional but recommended)
2. Create DTW trajectory features
3. Merge DTW features into a final feature file (via add_dtw_features_to_model_data.py)

Usage:
    python 4b_event_filter/run_analysis.py --cohort-name opioid_ed --age-band 0-12
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.fe_monitor import (  # noqa: E402
    detect_runtime_environment,
    function_block,
    module_block,
    step_block,
    mirror_log_to_s3,
)


def _get_logger(cohort_name: str, age_band: str) -> tuple[logging.Logger, Path]:
    """Create a module-level logger with both console and file handlers."""
    logs_dir = PROJECT_ROOT / "logs" / "feature_engineering" / "6_dtw"
    logs_dir.mkdir(parents=True, exist_ok=True)

    age_band_fname = age_band.replace("-", "_")
    log_path = logs_dir / f"dtw_{cohort_name}_{age_band_fname}.log"

    logger = logging.getLogger(f"dtw.{cohort_name}.{age_band_fname}")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
        )

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.propagate = False
    return logger, log_path


def create_predictive_features(
    cohort_name: str,
    age_band: str,
    logger: logging.Logger,
) -> None:
    """
    Step 0 (DTW filter + features): Create predictive time-window features
    for drugs/ICD/CPT prior to both event filtering (4b_event_filter) and
    downstream DTW feature-add (5d_dtw_analysis).
    """
    with step_block("6_dtw", "create_predictive_time_features", logger=logger):
        logger.info(
            "Creating predictive time-window features for %s / %s",
            cohort_name,
            age_band,
        )
        # Use the 5d_dtw_analysis implementation as the authoritative source.
        script_path = PROJECT_ROOT / "5d_dtw_analysis" / "create_predictive_time_features.py"

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--cohort-name",
                    cohort_name,
                    "--age-band",
                    age_band,
                    "--project-root",
                    str(PROJECT_ROOT),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("Predictive time-window features created")
            if result.stdout:
                logger.info("Predictive stdout:\n%s", result.stdout)
            if result.stderr:
                logger.info("Predictive stderr:\n%s", result.stderr)
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Predictive feature creation failed (returncode=%s)", exc.returncode
            )
            if exc.stderr:
                logger.error("stderr:\n%s", exc.stderr)
            logger.warning(
                "Continuing without predictive time-window features for DTW step"
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Predictive feature creation failed with exception: %s", exc)
            logger.warning(
                "Continuing without predictive time-window features for DTW step"
            )


def create_dtw_features_step(
    cohort_name: str,
    age_band: str,
    logger: logging.Logger,
) -> bool:
    """Step 1: Create DTW trajectory features (delegates to 5d_dtw_analysis)."""
    with step_block("6_dtw", "create_dtw_features", logger=logger):
        logger.info("Creating DTW trajectory features for %s / %s", cohort_name, age_band)
        script_path = PROJECT_ROOT / "5d_dtw_analysis" / "create_dtw_features.py"

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--cohort",
                    cohort_name,
                    "--age_band",
                    age_band,
                    "--split_type",
                    "target",
                    "--event_year",
                    "train",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("DTW trajectory features created")
            if result.stdout:
                logger.info("DTW stdout:\n%s", result.stdout)
            if result.stderr:
                logger.info("DTW stderr:\n%s", result.stderr)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error("DTW feature creation failed (returncode=%s)", exc.returncode)
            if exc.stderr:
                logger.error("stderr:\n%s", exc.stderr)
            return False
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("DTW feature creation failed with exception: %s", exc)
            return False


def add_dtw_features_to_model_data(
    cohort_name: str,
    age_band: str,
    logger: logging.Logger,
) -> bool:
    """Step 2: Merge DTW features into final DTW feature table (5d_dtw_analysis)."""
    with step_block("6_dtw", "add_dtw_features_to_model_data", logger=logger):
        logger.info(
            "Adding DTW features to model data for %s / %s",
            cohort_name,
            age_band,
        )
        script_path = PROJECT_ROOT / "5d_dtw_analysis" / "add_dtw_features_to_model_data.py"

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--cohort-name",
                    cohort_name,
                    "--age-band",
                    age_band,
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("DTW features added to model data")
            if result.stdout:
                logger.info("Merge stdout:\n%s", result.stdout)
            if result.stderr:
                logger.info("Merge stderr:\n%s", result.stderr)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error("DTW feature merge failed (returncode=%s)", exc.returncode)
            if exc.stderr:
                logger.error("stderr:\n%s", exc.stderr)
            return False
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("DTW feature merge failed with exception: %s", exc)
            return False


def run_dtw_analysis(
    cohort_name: str,
    age_band: str,
    skip_predictive_features: bool = False,
) -> bool:
    """
    Run complete DTW analysis workflow as a module-style function.

    This function is idempotent with respect to its outputs: rerunning will
    regenerate the same CSVs, and downstream code reads the most recent files.
    """
    logger, log_path = _get_logger(cohort_name, age_band)

    env = detect_runtime_environment(PROJECT_ROOT)
    logger.info(
        "Runtime environment: os=%s logical_cores=%s ram_gb=%s fast_root=%s",
        env.os_name,
        env.logical_cores,
        env.ram_gb,
        env.fast_root,
    )

    with function_block("6_dtw", "run_dtw_analysis", logger=logger):
        logger.info("Starting DTW analysis for %s / %s", cohort_name, age_band)

        if not skip_predictive_features:
            create_predictive_features(cohort_name, age_band, logger=logger)
        else:
            logger.info(
                "Skipping predictive time-window features (using existing features if present)"
            )

        if not create_dtw_features_step(cohort_name, age_band, logger=logger):
            logger.error("DTW feature creation failed; aborting module")
            mirror_log_to_s3("6_dtw", cohort_name, age_band, log_path, logger)
            return False

        if not add_dtw_features_to_model_data(cohort_name, age_band, logger=logger):
            logger.error("DTW feature merge failed; aborting module")
            mirror_log_to_s3("6_dtw", cohort_name, age_band, log_path, logger)
            return False

        logger.info("DTW analysis completed for %s / %s", cohort_name, age_band)

    mirror_log_to_s3("6_dtw", cohort_name, age_band, log_path, logger)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run complete DTW analysis workflow"
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
        "--skip-predictive-features",
        action="store_true",
        help="Skip predictive time-window feature creation",
    )

    args = parser.parse_args()

    with module_block("6_dtw"):
        success = run_dtw_analysis(
            cohort_name=args.cohort_name,
            age_band=args.age_band,
            skip_predictive_features=args.skip_predictive_features,
        )

    sys.exit(0 if success else 1)

