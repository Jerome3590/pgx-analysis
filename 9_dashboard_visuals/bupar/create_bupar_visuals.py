#!/usr/bin/env python3
"""
Create BupaR visuals for the dashboard.

We do NOT add BupaR (or DTW or FP-Growth) features to model data. This workflow
is for dashboard visualization only.

Runs the BupaR workflow for a given cohort and age band:
1. Create BupaR outputs and plots via R scripts
2. Merge BupaR features into a standalone feature table (dashboard only; not added to model data)
3. Upload plot PNGs to the dashboard bucket

Outputs:
- Features: 10_risk_dashboard/visualizations/bupar/outputs/feature_engineering/bupaR_added_features_{cohort}_{age_band_fname}.csv
- Mirrored features and plots: feature_engineering_outputs/5_bupar/{cohort}/{age_band}/[features,plots]
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Step folder (9_dashboard_visuals) and repo root; outputs go to 10_risk_dashboard/visualizations/bupar
REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 9_dashboard_visuals
BUPAR_CODE_DIR = Path(__file__).resolve().parent  # 9_dashboard_visuals/bupar (R scripts live here)
DASHBOARD_BUPAR_OUT = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "bupar"
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
        age_band_fname = age_band.replace("-", "_")

        # Write SHAP/FFA allowed codes for BupaR (filter original data to model-important items)
        try:
            from py_helpers.shap_ffa_fpgrowth_utils import write_shap_ffa_allowed_codes_for_bupar

            out_dir = DASHBOARD_BUPAR_OUT / "outputs"
            allowed_path = out_dir / f"allowed_codes_shap_ffa_{cohort_name}_{age_band_fname}.json"
            if write_shap_ffa_allowed_codes_for_bupar(
                cohort_name, age_band, allowed_path, top_n=500, project_root=REPO_ROOT
            ):
                logger.info("Wrote SHAP/FFA allowed codes for BupaR to %s", allowed_path)
            else:
                logger.info("No SHAP/FFA codes found; BupaR will use FP-Growth itemsets if present")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not write SHAP/FFA allowed codes for BupaR: %s", exc)

        if cohort_name == "opioid_ed":
            r_script = BUPAR_CODE_DIR / "create_bupar_outputs_opioid_ed.R"
        elif cohort_name == "non_opioid_ed":
            r_script = BUPAR_CODE_DIR / "create_bupar_outputs_non_opioid_ed.R"
        else:
            logger.error("Unsupported cohort for BupaR: %s", cohort_name)
            return False

        logger.info(
            "Running BupaR outputs script %s for %s / %s (cwd=%s)",
            r_script,
            cohort_name,
            age_band_arg,
            REPO_ROOT,
        )

        try:
            result = subprocess.run(
                ["Rscript", str(r_script), age_band_arg],
                cwd=str(REPO_ROOT),
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
    """Step 2: Merge per-patient BupaR features into a standalone CSV for dashboard (not added to model data)."""
    with step_block("5_bupar", "add_bupar_features_to_model_data", logger=logger):
        r_script = BUPAR_CODE_DIR / "add_bupar_features_to_model_data.R"

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
                    str(REPO_ROOT),
                    "--cohort-name",
                    cohort_name,
                    "--age-band",
                    age_band,
                ],
                cwd=str(REPO_ROOT),
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


def upload_bupar_plots_to_dashboard_s3(
    cohort_name: str,
    age_band: str,
    logger: logging.Logger,
) -> bool:
    """Upload BupaR plot PNGs to the dashboard bucket (same as FP-Growth) under bupar/{cohort}/{age_band}/plots/."""
    age_band_fname = age_band.replace("-", "_")
    plots_dir = DASHBOARD_BUPAR_OUT / "outputs" / cohort_name / age_band_fname / "plots"
    if not plots_dir.exists() or not list(plots_dir.glob("*.png")):
        logger.warning("No BupaR plots directory or no PNGs at %s; skipping S3 upload", plots_dir)
        return True

    s3_bucket = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
    dashboard_prefix = os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator")
    s3_prefix = f"{dashboard_prefix.rstrip('/')}/bupar/{cohort_name}/{age_band}/plots"

    try:
        from py_helpers.checkpoint_utils import upload_file_to_s3
    except ImportError:
        logger.warning("checkpoint_utils not available; skipping BupaR plot upload to dashboard S3")
        return True

    uploaded = 0
    for p in plots_dir.glob("*.png"):
        key = f"{s3_prefix}/{p.name}"
        s3_path = f"s3://{s3_bucket}/{key}"
        if upload_file_to_s3(p, s3_path, logger=logger, check_exists=True):
            uploaded += 1
    if uploaded:
        logger.info("Uploaded %s BupaR plot(s) to s3://%s/%s/", uploaded, s3_bucket, s3_prefix)
    return True


def create_bupar_visuals(
    cohort_name: str,
    age_band: str,
    force: bool = False,
) -> bool:
    """
    Create BupaR visuals for the dashboard: outputs, feature merge, and plot upload.
    BupaR features are not added to model data; they are for dashboard visualization only.
    If force is False and the output CSV already exists, skips (idempotent).
    """
    age_band_fname = age_band.replace("-", "_")
    out_csv = DASHBOARD_BUPAR_OUT / "outputs" / "feature_engineering" / f"bupaR_added_features_{cohort_name}_{age_band_fname}.csv"
    if not force and out_csv.exists():
        logger_bupar = logging.getLogger(f"bupar.{cohort_name}.{age_band_fname}")
        if not logger_bupar.handlers:
            logger_bupar.addHandler(logging.StreamHandler(sys.stdout))
        logger_bupar.info("Output exists at %s; skipping (use --force to re-run)", out_csv)
        return True

    logger, log_path = _get_logger(cohort_name, age_band)

    env = detect_runtime_environment(PROJECT_ROOT)
    logger.info(
        "Runtime environment: os=%s logical_cores=%s ram_gb=%s fast_root=%s",
        env.os_name,
        env.logical_cores,
        env.ram_gb,
        env.fast_root,
    )

    with function_block("5_bupar", "create_bupar_visuals", logger=logger):
        logger.info("Starting BupaR visuals for %s / %s", cohort_name, age_band)

        if not create_bupar_outputs(cohort_name, age_band, logger=logger):
            logger.error("BupaR outputs step failed; aborting")
            mirror_log_to_s3("5_bupar", cohort_name, age_band, log_path, logger)
            return False

        if not merge_bupar_features(cohort_name, age_band, logger=logger):
            logger.error("BupaR merge step failed; aborting")
            mirror_log_to_s3("5_bupar", cohort_name, age_band, log_path, logger)
            return False

        upload_bupar_plots_to_dashboard_s3(cohort_name, age_band, logger=logger)

        logger.info("BupaR visuals completed for %s / %s", cohort_name, age_band)

    mirror_log_to_s3("5_bupar", cohort_name, age_band, log_path, logger)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create BupaR visuals for the dashboard"
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if output already exists (default: skip when idempotent)",
    )

    args = parser.parse_args()

    with module_block("5_bupar"):
        success = create_bupar_visuals(
            cohort_name=args.cohort_name,
            age_band=args.age_band,
            force=args.force,
        )

    sys.exit(0 if success else 1)
