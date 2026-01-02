#!/usr/bin/env python3
"""
Module-style orchestration script for BupaR process-mining features.

This script runs the BupaR workflow for a given cohort and age band:
1. Create BupaR outputs and plots via R scripts
2. Merge BupaR features into a final feature table

Outputs:
- Features: 5a_bupaR_analysis/outputs/feature_engineering/bupaR_added_features_{cohort}_{age_band_fname}.csv
- Mirrored features and plots:
  feature_engineering_outputs/5_bupar/{cohort}/{age_band}/[features,plots]
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
    logs_dir = PROJECT_ROOT / "logs" / "feature_engineering" / "5_bupar"
    logs_dir.mkdir(parents=True, exist_ok=True)

    age_band_fname = age_band.replace("-", "_")
    log_path = logs_dir / f"bupar_{cohort_name}_{age_band_fname}.log"

    logger = logging.getLogger(f"bupar.{cohort_name}.{age_band_fname}")
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


def create_bupar_outputs(
    cohort_name: str,
    age_band: str,
    logger: logging.Logger,
) -> bool:
    """Step 1: Run the R script that builds BupaR event logs, features, and plots."""
    with step_block("5_bupar", "create_bupar_outputs", logger=logger):
        age_band_arg = age_band

        if cohort_name == "opioid_ed":
            r_script = PROJECT_ROOT / "5a_bupaR_analysis" / "create_bupar_outputs_opioid_ed.R"
        elif cohort_name == "non_opioid_ed":
            r_script = PROJECT_ROOT / "5a_bupaR_analysis" / "create_bupar_outputs_non_opioid_ed.R"
        else:
            logger.error("Unsupported cohort for BupaR: %s", cohort_name)
            return False

        logger.info(
            "Running BupaR outputs script %s for %s / %s",
            r_script,
            cohort_name,
            age_band_arg,
        )

        try:
            result = subprocess.run(
                ["Rscript", str(r_script), age_band_arg],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("BupaR outputs created")
            if result.stdout:
                logger.info("BupaR stdout:\n%s", result.stdout)
            if result.stderr:
                logger.info("BupaR stderr:\n%s", result.stderr)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error("BupaR outputs script failed (returncode=%s)", exc.returncode)
            if exc.stderr:
                logger.error("stderr:\n%s", exc.stderr)
            return False
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("BupaR outputs script failed with exception: %s", exc)
            return False


def merge_bupar_features(
    cohort_name: str,
    age_band: str,
    logger: logging.Logger,
) -> bool:
    """Step 2: Merge per-patient BupaR features into a final feature table."""
    with step_block("5_bupar", "add_bupar_features_to_model_data", logger=logger):
        r_script = PROJECT_ROOT / "5a_bupaR_analysis" / "add_bupar_features_to_model_data.R"

        logger.info(
            "Merging BupaR features for %s / %s using %s",
            cohort_name,
            age_band,
            r_script,
        )

        try:
            result = subprocess.run(
                [
                    "Rscript",
                    str(r_script),
                    "--project-root",
                    str(PROJECT_ROOT),
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
            logger.info("Merged BupaR features successfully")
            if result.stdout:
                logger.info("Merge stdout:\n%s", result.stdout)
            if result.stderr:
                logger.info("Merge stderr:\n%s", result.stderr)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error(
                "BupaR feature merge script failed (returncode=%s)", exc.returncode
            )
            if exc.stderr:
                logger.error("stderr:\n%s", exc.stderr)
            return False
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("BupaR feature merge script failed with exception: %s", exc)
            return False


def run_bupar_analysis(
    cohort_name: str,
    age_band: str,
) -> bool:
    """
    Run complete BupaR analysis workflow as a module-style function.
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

    with function_block("5_bupar", "run_bupar_analysis", logger=logger):
        logger.info("Starting BupaR analysis for %s / %s", cohort_name, age_band)

        if not create_bupar_outputs(cohort_name, age_band, logger=logger):
            logger.error("BupaR outputs step failed; aborting module")
            mirror_log_to_s3("5_bupar", cohort_name, age_band, log_path, logger)
            return False

        if not merge_bupar_features(cohort_name, age_band, logger=logger):
            logger.error("BupaR merge step failed; aborting module")
            mirror_log_to_s3("5_bupar", cohort_name, age_band, log_path, logger)
            return False

        logger.info("BupaR analysis completed for %s / %s", cohort_name, age_band)

    mirror_log_to_s3("5_bupar", cohort_name, age_band, log_path, logger)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run complete BupaR analysis workflow"
    )
    parser.add_argument(
        "--cohort-name",
        type=str,
        required=True,
        help="Cohort name (e.g., opioid_ed or non_opioid_ed)",
    )
    parser.add_argument(
        "--age-band",
        type=str,
        required=True,
        help="Age band (e.g., 0-12)",
    )

    args = parser.parse_args()

    with module_block("5_bupar"):
        success = run_bupar_analysis(
            cohort_name=args.cohort_name,
            age_band=args.age_band,
        )

    sys.exit(0 if success else 1)

