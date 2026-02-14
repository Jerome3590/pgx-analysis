#!/usr/bin/env python3
"""
Create FP-Growth visuals for the dashboard.

We do NOT add FP-Growth (or DTW) features to model data. This workflow is for
dashboard visualization only.

Runs the complete FP-Growth workflow:
1. Ensure FP-Growth itemsets exist (target split, TRAIN years)
2. Create FP-Growth patient-level features
3. Merge features into standalone CSV for dashboard (mirror, S3); not added to model data
4. Create visualizations (plots, network HTML)

Usage (Windows or Linux, from project root):
    python 9_dashboard_visuals/fpgrowth/create_fpgrowth_visuals.py --cohort-name opioid_ed --age-band 0-12
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Step folder (9_dashboard_visuals) and repo root; outputs go to 10_risk_dashboard/visualizations/fpgrowth
REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_FPGROWTH_OUT = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "fpgrowth"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_helpers.fe_monitor import (  # noqa: E402
    detect_runtime_environment,
    function_block,
    module_block,
    step_block,
    mirror_log_to_s3,
)


def _get_logger(cohort_name: str, age_band: str) -> tuple[logging.Logger, Path]:
    """Create a module-level logger with both console and file handlers."""
    logs_dir = REPO_ROOT / "9_dashboard_visuals" / "logs" / "feature_engineering" / "4_fpgrowth"
    logs_dir.mkdir(parents=True, exist_ok=True)

    age_band_fname = age_band.replace("-", "_")
    log_path = logs_dir / f"fpgrowth_{cohort_name}_{age_band_fname}.log"

    logger = logging.getLogger(f"fpgrowth.{cohort_name}.{age_band_fname}")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if called multiple times
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


def ensure_itemsets(
    cohort_name: str,
    age_band: str,
    logger: logging.Logger,
) -> None:
    """Step 0: Ensure FP-Growth itemsets exist for this cohort/age_band."""
    age_band_fname = age_band.replace("-", "_")
    itemsets_dir = (
        DASHBOARD_FPGROWTH_OUT
        / "outputs"
        / cohort_name
        / "target"
        / age_band_fname
        / "train"
    )
    itemsets_exist = itemsets_dir.exists() and any(
        itemsets_dir.glob("*_itemsets*.json")
    )

    with step_block("4_fpgrowth", "ensure_itemsets", logger=logger):
        if itemsets_exist:
            logger.info("Itemsets already exist at %s; skipping creation", itemsets_dir)
            return

        logger.info("Creating FP-Growth itemsets for %s / %s", cohort_name, age_band)
        script_path = PROJECT_ROOT / "fpgrowth" / "run_single_cohort_fpgrowth.py"

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--cohort-name",
                    cohort_name,
                    "--age-band",
                    age_band,
                    "--event-year",
                    "train",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("FP-Growth itemsets created successfully")
            if result.stdout:
                logger.info("Itemset stdout:\n%s", result.stdout)
            if result.stderr:
                logger.info("Itemset stderr:\n%s", result.stderr)
        except subprocess.CalledProcessError as exc:
            logger.error("Itemset creation failed (returncode=%s)", exc.returncode)
            if exc.stderr:
                logger.error("stderr:\n%s", exc.stderr)
            logger.warning(
                "Continuing; itemsets may already exist locally or be available from S3"
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Itemset creation failed with exception: %s", exc)
            logger.warning(
                "Continuing; itemsets may already exist locally or be available from S3"
            )


def create_features(
    cohort_name: str,
    age_band: str,
    logger: logging.Logger,
) -> bool:
    """Step 1: Create FP-Growth patient-level features."""
    with step_block("4_fpgrowth", "create_features", logger=logger):
        logger.info("Creating FP-Growth features for %s / %s", cohort_name, age_band)
        script_path = PROJECT_ROOT / "fpgrowth" / "create_fpgrowth_features.py"

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--cohort",
                    cohort_name,
                    "--age_band",
                    age_band,
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("FP-Growth features created")
            if result.stdout:
                logger.info("Feature stdout:\n%s", result.stdout)
            if result.stderr:
                logger.info("Feature stderr:\n%s", result.stderr)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error("Feature creation failed (returncode=%s)", exc.returncode)
            if exc.stderr:
                logger.error("stderr:\n%s", exc.stderr)
            return False
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Feature creation failed with exception: %s", exc)
            return False


def add_features_to_model_data(
    cohort_name: str,
    age_band: str,
    logger: logging.Logger,
) -> bool:
    """Step 2: Add FP-Growth features to model data."""
    with step_block("4_fpgrowth", "add_features_to_model_data", logger=logger):
        logger.info(
            "Adding FP-Growth features to model data for %s / %s",
            cohort_name,
            age_band,
        )
        script_path = (
            PROJECT_ROOT / "fpgrowth" / "add_fpgrowth_features_to_model_data.py"
        )

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
            logger.info("FP-Growth features added to model data")
            if result.stdout:
                logger.info("Merge stdout:\n%s", result.stdout)
            if result.stderr:
                logger.info("Merge stderr:\n%s", result.stderr)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error("Feature merge failed (returncode=%s)", exc.returncode)
            if exc.stderr:
                logger.error("stderr:\n%s", exc.stderr)
            return False
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Feature merge failed with exception: %s", exc)
            return False


def create_visualizations(
    cohort_name: str,
    age_band: str,
    logger: logging.Logger,
) -> bool:
    """Step 3: Create FP-Growth visualizations under 10_risk_dashboard/visualizations/fpgrowth/outputs."""
    with step_block("4_fpgrowth", "create_visualizations", logger=logger):
        logger.info("Creating FP-Growth visualizations for %s / %s", cohort_name, age_band)
        script_path = PROJECT_ROOT / "fpgrowth" / "create_plots.py"
        plots_output_dir = (
            REPO_ROOT
            / "10_risk_dashboard"
            / "visualizations"
            / "fpgrowth"
            / "outputs"
            / cohort_name
            / age_band
            / "plots"
        )
        plots_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--cohort-name",
                    cohort_name,
                    "--age-band",
                    age_band,
                    "--output-dir",
                    str(plots_output_dir),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("Visualizations created")
            if result.stdout:
                logger.info("Plots stdout:\n%s", result.stdout)
            if result.stderr:
                logger.info("Plots stderr:\n%s", result.stderr)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error("Visualization creation failed (returncode=%s)", exc.returncode)
            if exc.stderr:
                logger.error("stderr:\n%s", exc.stderr)
            return False
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Visualization creation failed with exception: %s", exc)
            return False


def create_fpgrowth_visuals(
    cohort_name: str,
    age_band: str,
    skip_feature_engineering: bool = False,
    skip_visualizations: bool = False,
    force: bool = False,
) -> bool:
    """
    Create FP-Growth visuals for the dashboard: itemsets, features, merge, and plots.
    FP-Growth features are not added to model data; they are for dashboard visualization only.

    Idempotent with respect to itemset creation; downstream scripts overwrite CSV outputs.
    If force is False and the output CSV already exists, skips (idempotent).
    """
    age_band_fname = age_band.replace("-", "_")
    out_csv = (
        DASHBOARD_FPGROWTH_OUT
        / "outputs"
        / "feature_engineering"
        / f"fpgrowth_added_features_{cohort_name}_{age_band_fname}.csv"
    )
    if not force and out_csv.exists():
        logger_skip = logging.getLogger(f"fpgrowth.{cohort_name}.{age_band_fname}")
        if not logger_skip.handlers:
            logger_skip.addHandler(logging.StreamHandler(sys.stdout))
        logger_skip.info("Output exists at %s; skipping (use --force to re-run)", out_csv)
        return True

    logger, log_path = _get_logger(cohort_name, age_band)

    env = detect_runtime_environment(REPO_ROOT)
    logger.info(
        "Runtime environment: os=%s logical_cores=%s ram_gb=%s fast_root=%s",
        env.os_name,
        env.logical_cores,
        env.ram_gb,
        env.fast_root,
    )

    with function_block("4_fpgrowth", "create_fpgrowth_visuals", logger=logger):
        logger.info("Starting FP-Growth visuals for %s / %s", cohort_name, age_band)

        ensure_itemsets(cohort_name, age_band, logger=logger)

        if not skip_feature_engineering:
            if not create_features(cohort_name, age_band, logger=logger):
                logger.error("FP-Growth feature creation failed; aborting")
                mirror_log_to_s3("4_fpgrowth", cohort_name, age_band, log_path, logger)
                return False

            if not add_features_to_model_data(cohort_name, age_band, logger=logger):
                logger.error("FP-Growth feature merge failed; aborting")
                mirror_log_to_s3("4_fpgrowth", cohort_name, age_band, log_path, logger)
                return False
        else:
            logger.info("Skipping feature engineering; assuming features already present")

        if not skip_visualizations:
            ok = create_visualizations(cohort_name, age_band, logger=logger)
            if not ok:
                logger.error("Visualization step failed")
        else:
            logger.info("Skipping visualization creation")

        logger.info("FP-Growth visuals completed for %s / %s", cohort_name, age_band)

    mirror_log_to_s3("4_fpgrowth", cohort_name, age_band, log_path, logger)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create FP-Growth visuals for the dashboard"
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
        "--skip-feature-engineering",
        action="store_true",
        help="Skip feature engineering steps",
    )
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip visualization creation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if output already exists (default: skip when idempotent)",
    )

    args = parser.parse_args()

    with module_block("4_fpgrowth"):
        success = create_fpgrowth_visuals(
            cohort_name=args.cohort_name,
            age_band=args.age_band,
            skip_feature_engineering=args.skip_feature_engineering,
            skip_visualizations=args.skip_visualizations,
            force=args.force,
        )

    sys.exit(0 if success else 1)
