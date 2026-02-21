#!/usr/bin/env python3
"""
Create FP-Growth visuals for the dashboard.

We do NOT add FP-Growth (or DTW) features to model data. This workflow produces
itemsets and plots only (no feature-engineering steps).

Runs:
1. Ensure FP-Growth itemsets exist (target split, TRAIN years)
2. Create visualizations (plots, network HTML)

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
)
from py_helpers.pipeline_logger import (  # noqa: E402
    setup_pipeline_logger,
    log_step_start,
    log_step_complete,
    PipelineLogger,
)


def ensure_itemsets(
    cohort_name: str,
    age_band: str,
    logger: logging.Logger,
    force: bool = False,
) -> bool:
    """Step 0: Ensure FP-Growth itemsets exist for this cohort/age_band. If force=True, re-run even when they exist.
    Returns True if itemsets exist (pre-existing or created); False if creation was attempted and failed (no outputs).
    """
    age_band_fname = age_band.replace("-", "_")
    # Visualization artifacts: cohort then age_band only (no target/combined/train)
    itemsets_dir = (
        DASHBOARD_FPGROWTH_OUT
        / "outputs"
        / cohort_name
        / age_band_fname
    )
    itemsets_exist = itemsets_dir.exists() and any(
        itemsets_dir.glob("*_itemsets*.json")
    )

    with step_block("6_fpgrowth", "ensure_itemsets", logger=logger):
        if itemsets_exist and not force:
            logger.info("Itemsets already exist at %s; skipping creation (use --force to re-run)", itemsets_dir)
            return True
        if itemsets_exist and force:
            logger.info("Itemsets exist at %s; re-running due to --force", itemsets_dir)

        logger.info("="*60)
        logger.info("CREATING FP-GROWTH ITEMSETS: %s / %s", cohort_name, age_band)
        logger.info("="*60)
        logger.info("Processing 4 item types: drug_name, icd_code, cpt_code, medical_code")
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
                    "--project-root",
                    str(REPO_ROOT),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("="*60)
            logger.info("✓ FP-Growth itemsets created successfully for %s / %s", cohort_name, age_band)
            logger.info("="*60)
            if result.stdout:
                # Parse stdout for [OK] lines to show summary
                ok_lines = [line for line in result.stdout.splitlines() if "[OK]" in line]
                if ok_lines:
                    logger.info("Item type results:")
                    for line in ok_lines:
                        logger.info("  %s", line.strip())
                logger.info("Full itemset stdout:\n%s", result.stdout)
            if result.stderr:
                logger.info("Itemset stderr:\n%s", result.stderr)
            # Verify outputs exist after run
            itemsets_now = itemsets_dir.exists() and any(itemsets_dir.glob("*_itemsets*.json"))
            if itemsets_now:
                itemset_files = list(itemsets_dir.glob("*_itemsets*.json"))
                logger.info("Created %d itemset files in %s", len(itemset_files), itemsets_dir)
            return itemsets_now
        except subprocess.CalledProcessError as exc:
            logger.error("="*60)
            logger.error("✗ Itemset creation FAILED for %s / %s (returncode=%s)", cohort_name, age_band, exc.returncode)
            logger.error("="*60)
            # Log first [ERROR] / [ERROR_PARAMS] line from runner so reason is visible even if full stdout is truncated
            if exc.stdout:
                error_summary = []
                for line in exc.stdout.splitlines():
                    line = line.strip()
                    if line.startswith("[ERROR]") or line.startswith("[ERROR_PARAMS]"):
                        error_summary.append(line)
                        logger.error("runner: %s", line)
                        if line.startswith("[ERROR]") and "No frequent itemsets" not in line:
                            break
                if error_summary:
                    logger.error("Error summary: %d error lines found", len(error_summary))
                logger.error("stdout (errors/params from runner):\n%s", exc.stdout)
            if exc.stderr:
                logger.error("stderr:\n%s", exc.stderr)
            logger.warning(
                "Continuing; itemsets may already exist locally or be available from S3"
            )
            itemsets_now = itemsets_dir.exists() and any(itemsets_dir.glob("*_itemsets*.json"))
            return itemsets_now
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Itemset creation failed with exception: %s", exc)
            logger.warning(
                "Continuing; itemsets may already exist locally or be available from S3"
            )
            itemsets_now = itemsets_dir.exists() and any(itemsets_dir.glob("*_itemsets*.json"))
            return itemsets_now


def create_visualizations(
    cohort_name: str,
    age_band: str,
    logger: logging.Logger,
) -> bool:
    """Step 3: Create FP-Growth visualizations under 10_risk_dashboard/visualizations/fpgrowth/outputs."""
    with step_block("6_fpgrowth", "create_visualizations", logger=logger):
        logger.info("Creating FP-Growth visualizations for %s / %s", cohort_name, age_band)
        script_path = PROJECT_ROOT / "fpgrowth" / "create_plots.py"
        age_band_fname = age_band.replace("-", "_")
        # Visualization artifacts: cohort then age_band only
        plots_output_dir = (
            REPO_ROOT
            / "10_risk_dashboard"
            / "visualizations"
            / "fpgrowth"
            / "outputs"
            / cohort_name
            / age_band_fname
            / "plots"
        )
        plots_output_dir.mkdir(parents=True, exist_ok=True)

        fpgrowth_outputs_root = DASHBOARD_FPGROWTH_OUT / "outputs"
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--base-dir",
                    str(fpgrowth_outputs_root),
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
    skip_visualizations: bool = False,
    force: bool = False,
) -> bool:
    """
    Create FP-Growth visuals for the dashboard: itemsets and plots only.
    We do not create or merge FP-Growth features (no feature engineering in this pipeline).

    Idempotent: if force is False and itemsets already exist for this cohort/age_band, skips.
    """
    age_band_fname = age_band.replace("-", "_")
    # Visualization artifacts: cohort then age_band only
    itemsets_dir = (
        DASHBOARD_FPGROWTH_OUT
        / "outputs"
        / cohort_name
        / age_band_fname
    )
    itemsets_exist = itemsets_dir.exists() and any(
        itemsets_dir.glob("*_itemsets*.json")
    )
    if not force and itemsets_exist:
        logger_skip = logging.getLogger(f"fpgrowth.{cohort_name}.{age_band_fname}")
        if not logger_skip.handlers:
            logger_skip.addHandler(logging.StreamHandler(sys.stdout))
        logger_skip.info(
            "Itemsets already exist at %s; skipping (use --force to re-run)",
            itemsets_dir,
        )
        return True

    logger = setup_pipeline_logger(
        step_name="6_fpgrowth",
        cohort=cohort_name,
        age_band=age_band,
        script_name="create_fpgrowth_visuals"
    )

    env = detect_runtime_environment(REPO_ROOT)
    logger.info(
        "Runtime environment: os=%s logical_cores=%s ram_gb=%s fast_root=%s",
        env.os_name,
        env.logical_cores,
        env.ram_gb,
        env.fast_root,
    )

    with function_block("6_fpgrowth", "create_fpgrowth_visuals", logger=logger.logger):
        logger.info("")
        logger.info("#" * 70)
        logger.info("#  FP-GROWTH VISUAL WORKFLOW: %s / %s", cohort_name, age_band)
        logger.info("#" * 70)
        logger.info("")

        logger.info("[STEP 1/2] Creating itemsets...")
        itemsets_ok = ensure_itemsets(cohort_name, age_band, logger=logger.logger, force=force)
        if not itemsets_ok:
            logger.warning(
                "⚠️  No itemsets produced for %s / %s (e.g. model_data missing or no transactions). Check log.",
                cohort_name,
                age_band,
            )
        else:
            logger.info("✓ Itemsets ready")

        if not skip_visualizations:
            logger.info("[STEP 2/2] Creating visualizations...")
            ok = create_visualizations(cohort_name, age_band, logger=logger.logger)
            if not ok:
                logger.error("✗ Visualization step failed")
            else:
                logger.info("✓ Visualizations complete")
        else:
            logger.info("[STEP 2/2] Skipping visualization creation")

        logger.info("")
        logger.info("#" * 70)
        logger.info("#  FP-GROWTH COMPLETED: %s / %s", cohort_name, age_band)
        logger.info("#" * 70)
        logger.info("")

    logger.log_summary()
    # Exit 0 only when we have itemsets (so notebook "exit 0" matches real completion)
    return itemsets_ok


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
            skip_visualizations=args.skip_visualizations,
            force=args.force,
        )

    sys.exit(0 if success else 1)
