#!/usr/bin/env python3
"""
Create BupaR visuals for the dashboard.

We do NOT add BupaR (or DTW or FP-Growth) features to model data. This workflow
is for dashboard visualization only.

Runs the BupaR workflow for a given cohort and age band (outputs and plots only; no feature engineering):
1. Create BupaR outputs and plots via R scripts
2. Upload interactive HTML and static PNG plots to the dashboard bucket

Outputs:
- Interactive plots: 3 HTML files with year dropdown filtering (activity_frequency, trace_explorer, process_matrix)
- Static plots: 3 PNG fallback files
"""

import argparse
import json
import logging
import os
import shutil
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
from py_helpers.model_data_paths import resolve_model_events_paths  # noqa: E402


def _find_rscript() -> str | None:
    """Return path to Rscript executable, or None if not found.
    Checks PATH first, then R_HOME, then common Windows install paths.
    """
    rscript = shutil.which("Rscript")
    if rscript:
        return rscript
    # R_HOME (e.g. set by user or R installer)
    r_home = os.environ.get("R_HOME")
    if r_home:
        cand = Path(r_home) / "bin" / "Rscript"
        if cand.suffix != ".exe" and sys.platform == "win32":
            cand = Path(str(cand) + ".exe")
        if cand.exists():
            return str(cand)
    # Windows: common install path
    if sys.platform == "win32":
        pf = Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "R"
        if pf.exists():
            for r_dir in sorted(pf.glob("R-*"), reverse=True):
                exe = r_dir / "bin" / "Rscript.exe"
                if exe.exists():
                    return str(exe)
    return None


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

        # Write SHAP/FFA allowed codes for BupaR (required: event log = dataset filtered by causal codes + dates)
        out_dir = DASHBOARD_BUPAR_OUT / "outputs"
        allowed_path = out_dir / f"allowed_codes_shap_ffa_{cohort_name}_{age_band_fname}.json"
        try:
            from py_helpers.shap_ffa_fpgrowth_utils import write_shap_ffa_allowed_codes_for_bupar

            if not write_shap_ffa_allowed_codes_for_bupar(
                cohort_name, age_band, allowed_path, top_n=500, project_root=REPO_ROOT
            ):
                logger.error(
                    "SHAP/FFA allowed codes file required for BupaR is missing or empty. "
                    "Run SHAP/FFA analysis (7_shap_analysis) for cohort=%s age_band=%s first.",
                    cohort_name,
                    age_band,
                )
                return False
            logger.info("Wrote SHAP/FFA allowed codes for BupaR to %s", allowed_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not write SHAP/FFA allowed codes for BupaR: %s", exc)
            return False

        # Require artifact exists and has at least one code before calling R
        if not allowed_path.exists():
            logger.error(
                "SHAP/FFA allowed codes file not found at %s; aborting BupaR script.",
                allowed_path,
            )
            return False
        try:
            with open(allowed_path, encoding="utf-8") as f:
                codes = json.load(f)
            if not codes or (isinstance(codes, list) and len(codes) == 0):
                logger.error(
                    "SHAP/FFA allowed codes file is empty at %s; run SHAP/FFA for %s / %s first.",
                    allowed_path,
                    cohort_name,
                    age_band,
                )
                return False
        except (json.JSONDecodeError, TypeError) as e:
            logger.error("Invalid SHAP/FFA allowed codes JSON at %s: %s", allowed_path, e)
            return False

        # Require model data exists before calling R (single path or 85-114 = 85-94 + 95-114 union)
        model_paths = resolve_model_events_paths(REPO_ROOT, cohort_name, age_band)
        if not model_paths or not all(p.exists() for p in model_paths):
            logger.error(
                "Model data (model_events.parquet) not found for cohort=%s age_band=%s. "
                "Run 3b/4_model_data for this cohort/age band first.",
                cohort_name,
                age_band,
            )
            return False
        if len(model_paths) == 2:
            logger.info("Model data found (85-114 = 85-94 + 95-114): %s, %s", model_paths[0], model_paths[1])
        else:
            logger.info("Model data found: %s", model_paths[0])

        if cohort_name == "opioid_ed":
            r_script = BUPAR_CODE_DIR / "create_bupar_outputs_opioid_ed.R"
        elif cohort_name == "non_opioid_ed":
            r_script = BUPAR_CODE_DIR / "create_bupar_outputs_non_opioid_ed.R"
        else:
            logger.error("Unsupported cohort for BupaR: %s", cohort_name)
            return False

        rscript = _find_rscript()
        if not rscript:
            logger.error(
                "Rscript not found. Install R (https://cran.r-project.org/) and add it to PATH, "
                "or set R_HOME to your R install directory, or run on a machine where R is installed (e.g. EC2)."
            )
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
                [rscript, str(r_script), age_band_arg],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("BupaR outputs created")
            if result.stdout:
                logger.info("BupaR stdout:\n%s", result.stdout)
            if result.stderr:
                logger.warning("BupaR stderr (check for EMPTY EVENT LOG or min() warnings):\n%s", result.stderr)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error("BupaR outputs script failed (returncode=%s)", exc.returncode)
            if exc.stderr:
                logger.error("stderr:\n%s", exc.stderr)
            return False
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("BupaR outputs script failed with exception: %s", exc)
            return False


def upload_bupar_plots_to_dashboard_s3(
    cohort_name: str,
    age_band: str,
    logger: logging.Logger,
) -> bool:
    """Upload BupaR plot PNGs and interactive HTML files to the dashboard bucket under bupar/{cohort}/{age_band}/plots/."""
    age_band_fname = age_band.replace("-", "_")
    plots_dir = DASHBOARD_BUPAR_OUT / "outputs" / cohort_name / age_band_fname / "plots"
    if not plots_dir.exists():
        logger.warning(
            "No BupaR plots directory at %s; skipping S3 upload. "
            "Check R stdout/stderr above for EMPTY EVENT LOG or fallback to FP-Growth.",
            plots_dir,
        )
        return True

    s3_bucket = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
    dashboard_prefix = os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator")
    s3_prefix = f"{dashboard_prefix.rstrip('/')}/bupar/{cohort_name}/{age_band}/plots"

    try:
        from py_helpers.checkpoint_utils import upload_file_to_s3
    except ImportError:
        logger.warning("checkpoint_utils not available; skipping BupaR plot upload to dashboard S3")
        return True

    # Upload both PNG (legacy) and HTML (interactive) files
    uploaded = 0
    for pattern in ["*.png", "*.html"]:
        for p in plots_dir.glob(pattern):
            key = f"{s3_prefix}/{p.name}"
            s3_path = f"s3://{s3_bucket}/{key}"
            if upload_file_to_s3(p, s3_path, logger=logger, check_exists=True):
                uploaded += 1
    if uploaded:
        logger.info("Uploaded %s BupaR file(s) (PNG + HTML) to s3://%s/%s/", uploaded, s3_bucket, s3_prefix)
    else:
        logger.warning("No PNG or HTML files found in %s", plots_dir)
    return True


def create_bupar_visuals(
    cohort_name: str,
    age_band: str,
    force: bool = False,
) -> bool:
    """
    Create BupaR visuals for the dashboard: outputs and plot upload only.
    We do not create or merge BupaR features (no feature engineering in this pipeline).
    If force is False and plots already exist, skips (idempotent).
    """
    age_band_fname = age_band.replace("-", "_")
    plots_dir = DASHBOARD_BUPAR_OUT / "outputs" / cohort_name / age_band_fname / "plots"
    if not force and plots_dir.exists() and list(plots_dir.glob("*.png")):
        logger_bupar = logging.getLogger(f"bupar.{cohort_name}.{age_band_fname}")
        if not logger_bupar.handlers:
            logger_bupar.addHandler(logging.StreamHandler(sys.stdout))
        logger_bupar.info("Output exists at %s; skipping (use --force to re-run)", plots_dir)
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
