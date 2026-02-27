#!/usr/bin/env python3
"""
Create BupaR visuals for the dashboard.

We do NOT add BupaR (or DTW or FP-Growth) features to model data. This workflow
is for dashboard visualization only.

Runs the BupaR workflow for a given cohort and age band (outputs and plots only; no feature engineering):
1. Create BupaR outputs and plots via R scripts
2. Upload only RQ-used artifacts to the dashboard bucket (see RESEARCH_QUESTIONS_ARTIFACTS.md)

We only produce/save artifacts tied to research questions (N2, N6). Upload allowlist: activity frequency
JSON/PNG/HTML, trace explorer pre-target, process_matrix_drug_drug, activity_sequence_top, and plots/lib/.
Archived artifacts (process_matrix.png, frequency_map.png, trace_explorer.png, post-target trace, etc.)
may still be written by R locally but are not uploaded.
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
)
from py_helpers.pipeline_logger import (  # noqa: E402
    setup_pipeline_logger,
    log_step_start,
    log_step_complete,
    PipelineLogger,
)
from py_helpers.model_data_paths import get_model_events_paths_checked, get_path_check_listings, resolve_model_events_paths  # noqa: E402


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


def create_bupar_outputs(
    cohort_name: str,
    age_band: str,
    logger: logging.Logger,
    local_test: bool = False,
) -> bool:
    """Step 1: Run the R script that builds BupaR event logs, features, and plots."""
    with step_block("4_bupar", "create_bupar_outputs", logger=logger):
        age_band_arg = age_band
        age_band_fname = age_band.replace("-", "_")

        # Write SHAP/FFA allowed codes for BupaR (required: event log = dataset filtered by causal codes + dates)
        out_dir = DASHBOARD_BUPAR_OUT / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        allowed_path = out_dir / f"allowed_codes_shap_ffa_{cohort_name}_{age_band_fname}.json"
        if local_test:
            # Local test: use empty list so R uses all codes; no SHAP/FFA required
            allowed_path.write_text("[]", encoding="utf-8")
            logger.info("Local test: wrote empty allowed_codes (R will use all codes) to %s", allowed_path)
        else:
            try:
                from py_helpers.shap_ffa_fpgrowth_utils import write_shap_ffa_allowed_codes_for_bupar

                try:
                    from py_helpers.env_utils import get_data_root
                    data_root = get_data_root()
                except Exception:
                    data_root = None
                if not write_shap_ffa_allowed_codes_for_bupar(
                    cohort_name,
                    age_band,
                    allowed_path,
                    top_n=500,
                    project_root=REPO_ROOT,
                    data_root=data_root,
                ):
                    logger.error(
                        "SHAP/FFA allowed codes file required for BupaR is missing or empty. "
                        "Run SHAP/FFA analysis (7_shap_analysis) for cohort=%s age_band=%s first.",
                        cohort_name,
                        age_band,
                    )
                    return False
                logger.info("Wrote SHAP/FFA allowed codes for BupaR to %s", allowed_path)
                # Mirror to S3 so others can download via sync_visualization_data_from_s3.py --allowed-codes-only
                _upload_allowed_codes_to_s3(allowed_path, logger)
            except Exception as exc:  # noqa: BLE001
                logger.error("Could not write SHAP/FFA allowed codes for BupaR: %s", exc)
                return False

        # Require artifact exists; allow empty list only for local_test (R uses all codes)
        if not allowed_path.exists():
            logger.error(
                "SHAP/FFA allowed codes file not found at %s; aborting BupaR script.",
                allowed_path,
            )
            return False
        try:
            with open(allowed_path, encoding="utf-8") as f:
                codes = json.load(f)
            if not local_test and (not codes or (isinstance(codes, list) and len(codes) == 0)):
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

        # Require model data exists before calling R; confirm path exists with objects before continuing
        model_paths = resolve_model_events_paths(REPO_ROOT, cohort_name, age_band)
        if not model_paths or not all(p.exists() for p in model_paths):
            paths_checked = get_model_events_paths_checked(REPO_ROOT, cohort_name, age_band)
            path_listings = get_path_check_listings(paths_checked) if paths_checked else []
            logger.error(
                "Model data (model_events.parquet) not found for cohort=%s age_band=%s. "
                "Run 3b/4_model_data for this cohort/age band first.",
                cohort_name,
                age_band,
            )
            logger.error(
                "[ERROR_PARAMS] step=5_bupar cohort_name=%s age_band=%s error=model_data not found paths_checked=%s",
                cohort_name, age_band, " | ".join(paths_checked) if paths_checked else "(none)",
            )
            if path_listings:
                logger.error(
                    "[ERROR_PARAMS] step=5_bupar path_listings: %s",
                    " ; ".join(path_listings),
                )
            return False
        from py_helpers.model_data_paths import confirm_paths_exist_with_listings
        all_ok, confirm_listings = confirm_paths_exist_with_listings(list(model_paths))
        for line in confirm_listings:
            logger.info("[PATH_CONFIRM] %s", line)
        if not all_ok:
            logger.error("Model data path(s) missing or empty; aborting.")
            logger.error("[ERROR_PARAMS] step=5_bupar path_listings: %s", " ; ".join(confirm_listings))
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
            # Log HTML outputs for troubleshooting empty visuals
            age_band_fname = age_band_arg.replace("-", "_")
            plots_dir = DASHBOARD_BUPAR_OUT / "outputs" / cohort_name / age_band_fname / "plots"
            if plots_dir.exists():
                for pattern in ["*.html", "*.png"]:
                    for p in sorted(plots_dir.glob(pattern)):
                        try:
                            size = p.stat().st_size
                            logger.info(
                                "BupaR output file: %s size=%s bytes (%s)",
                                p.name,
                                f"{size:,}" if size is not None else "N/A",
                                "EMPTY - check R diagnostic logs" if (size is not None and size < 500) else "ok",
                            )
                        except OSError as e:
                            logger.warning("BupaR output file %s: could not stat: %s", p.name, e)
            else:
                logger.warning("BupaR plots dir missing after R run: %s", plots_dir)
            # Deploy check: verify expected visualization and feature paths
            features_dir = DASHBOARD_BUPAR_OUT / "outputs" / cohort_name / age_band_fname / "features"
            miss_req, miss_opt, found_list = check_bupar_paths(plots_dir, features_dir, cohort_name, age_band_fname, logger=logger)
            if miss_req:
                logger.warning("BupaR missing required paths (dashboard may show gaps): %s", ", ".join(miss_req))
            logger.info("BupaR path check: %d found, %d required missing, %d optional missing", len(found_list), len(miss_req), len(miss_opt))
            return True
        except subprocess.CalledProcessError as exc:
            logger.error("BupaR outputs script failed (returncode=%s)", exc.returncode)
            if exc.stderr:
                logger.error("stderr:\n%s", exc.stderr)
            return False
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("BupaR outputs script failed with exception: %s", exc)
            return False


ALLOWED_CODES_S3_BUCKET = os.environ.get("PGX_S3_BUCKET", "pgxdatalake")
ALLOWED_CODES_S3_PREFIX = "gold/bupar/allowed_codes"

# RQ-only artifacts (RESEARCH_QUESTIONS_ARTIFACTS.md): we only produce/save these for the dashboard.
# Archived artifacts (process_matrix.png, frequency_map.png, trace_explorer.png, post trace, etc.) are not uploaded.
def _bupar_rq_artifact_basenames(cohort_name: str, age_band_fname: str) -> set:
    """Return set of plot dir basenames we upload (RQ-only). Includes lib/ for interactive HTML deps."""
    pre = "pre_f1120" if cohort_name == "opioid_ed" else "pre_hcg"
    base = f"{cohort_name}_{age_band_fname}"
    return {
        f"{base}_activity_frequency.json",
        f"{base}_pre_target_activity_frequency.json",
        f"{base}_post_target_activity_frequency.json",
        f"{base}_overall_activity_frequency.png",
        f"{base}_{pre}_activity_frequency.png",
        f"{base}_trace_explorer_{pre}.png",
        f"{base}_trace_explorer_plot.json",
        f"{base}_process_matrix_drug_drug.png",
        f"{base}_process_matrix_drug_drug.json",
        f"{base}_activity_sequence_top.png",
        f"{base}_activity_sequence_top.json",
    }


# Expected BupaR outputs for deploy/check. Aligned to RQ artifacts only (see RESEARCH_QUESTIONS_ARTIFACTS.md).
def _bupar_expected_plots(cohort_name: str, age_band_fname: str):
    """Return (filename, required) for plots_dir. Only RQ-used artifacts; archived ones omitted."""
    pre = "pre_f1120" if cohort_name == "opioid_ed" else "pre_hcg"
    base = f"{cohort_name}_{age_band_fname}"
    return [
        (f"{base}_overall_activity_frequency.png", True),
        (f"{base}_activity_frequency.json", True),
        (f"{base}_pre_target_activity_frequency.json", True),
        (f"{base}_{pre}_activity_frequency.png", True),
        (f"{base}_trace_explorer_{pre}.png", True),
        (f"{base}_process_matrix_drug_drug.png", True),
        (f"{base}_post_target_activity_frequency.json", False),
        (f"{base}_trace_explorer_plot.json", False),
        (f"{base}_process_matrix_drug_drug.json", False),
        (f"{base}_activity_sequence_top.png", False),
        (f"{base}_activity_sequence_top.json", False),
    ]


def _bupar_expected_feature_csv_patterns(cohort_name: str, age_band_fname: str):
    """Return expected CSV basenames in features/ (optional)."""
    pre = "pre_f1120" if cohort_name == "opioid_ed" else "pre_hcg"
    base = f"{cohort_name}_{age_band_fname}"
    return [
        f"{base}_train_target_traces_bupar.csv",
        f"{base}_train_target_traces_top_bupar.csv",
        f"{base}_train_target_traces_rare_bupar.csv",
        f"{base}_train_target_{pre}_traces_bupar.csv",
        f"{base}_train_target_{pre}_traces_top_bupar.csv",
        f"{base}_train_target_{pre}_traces_rare_bupar.csv",
        f"{base}_train_target_{pre}_patient_features_bupar.csv",
        f"{base}_train_target_time_to_f1120_features_bupar.csv" if cohort_name == "opioid_ed" else f"{base}_train_target_time_to_hcg_features_bupar.csv",
    ]


def check_bupar_paths(
    plots_dir: Path,
    features_dir: Path,
    cohort_name: str,
    age_band_fname: str,
    logger=None,
):
    """
    Verify expected BupaR visualization and feature paths exist.
    Returns (missing_required, missing_optional, found).
    """
    missing_required = []
    missing_optional = []
    found = []
    expected = _bupar_expected_plots(cohort_name, age_band_fname)
    found_rel = set()
    for filename, required in expected:
        path = plots_dir / filename
        if path.exists() and path.is_file():
            found.append("plots/" + filename)
            found_rel.add(filename)
        else:
            if required:
                missing_required.append("plots/" + filename)
            else:
                missing_optional.append("plots/" + filename)
    # Count other files under plots (e.g. lib/, type-pair PNGs) as found
    for p in plots_dir.rglob("*"):
        if p.is_file() and p.suffix in (".png", ".html", ".json", ".css", ".js"):
            rel = p.relative_to(plots_dir).as_posix()
            if rel not in found_rel:
                found.append("plots/" + rel)
                found_rel.add(rel)
    # Feature CSVs (all optional for dashboard deploy)
    for basename in _bupar_expected_feature_csv_patterns(cohort_name, age_band_fname):
        path = features_dir / basename
        if path.exists() and path.is_file():
            found.append(f"features/{basename}")
        else:
            missing_optional.append(f"features/{basename}")
    if logger:
        if missing_required:
            logger.warning("BupaR missing required: %s", ", ".join(missing_required))
        if missing_optional:
            logger.debug("BupaR missing optional: %s", ", ".join(missing_optional))
        logger.info("BupaR path check: %d found, %d required missing, %d optional missing", len(found), len(missing_required), len(missing_optional))
    return missing_required, missing_optional, found


def export_bupar_feature_csvs_to_json(
    plots_dir: Path,
    features_dir: Path,
    cohort_name: str,
    age_band_fname: str,
    logger=None,
):
    """
    Export key BupaR feature CSVs to JSON in plots_dir so they are uploaded with plots
    and can be served as JSON by the API if needed. Returns number of JSON files written.
    """
    import csv as csv_module
    base = f"{cohort_name}_{age_band_fname}"
    csv_to_json = [
        (f"{base}_train_target_traces_top_bupar.csv", f"{base}_traces_top.json"),
        (f"{base}_train_target_traces_rare_bupar.csv", f"{base}_traces_rare.json"),
        (f"{base}_train_target_traces_bupar.csv", f"{base}_traces.json"),
    ]
    pre = "pre_f1120" if cohort_name == "opioid_ed" else "pre_hcg"
    csv_to_json.extend([
        (f"{base}_train_target_{pre}_traces_top_bupar.csv", f"{base}_pre_target_traces_top.json"),
        (f"{base}_train_target_{pre}_traces_rare_bupar.csv", f"{base}_pre_target_traces_rare.json"),
    ])
    written = 0
    for csv_basename, json_basename in csv_to_json:
        csv_path = features_dir / csv_basename
        json_path = plots_dir / json_basename
        if not csv_path.exists() or not csv_path.is_file():
            continue
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = list(csv_module.DictReader(f))
            with open(json_path, "w", encoding="utf-8") as out:
                json.dump(rows, out, indent=2)
            written += 1
            if logger:
                logger.debug("Exported %s -> %s", csv_basename, json_basename)
        except Exception as e:
            if logger:
                logger.warning("CSV to JSON failed for %s: %s", csv_basename, e)
    return written


def _upload_allowed_codes_to_s3(allowed_path: Path, logger: logging.Logger) -> None:
    """Upload allowed_codes JSON to s3://pgxdatalake/gold/bupar/allowed_codes/ for download via --allowed-codes-only."""
    if not allowed_path.exists() or allowed_path.stat().st_size == 0:
        return
    try:
        from py_helpers.checkpoint_utils import upload_file_to_s3
    except ImportError:
        return
    s3_path = f"s3://{ALLOWED_CODES_S3_BUCKET}/{ALLOWED_CODES_S3_PREFIX}/{allowed_path.name}"
    if upload_file_to_s3(allowed_path, s3_path, logger=logger, check_exists=True):
        logger.info("Uploaded allowed_codes to %s", s3_path)


def upload_bupar_plots_to_dashboard_s3(
    cohort_name: str,
    age_band: str,
    logger: logging.Logger,
) -> bool:
    """Upload only RQ-used BupaR artifacts to the dashboard bucket (see RESEARCH_QUESTIONS_ARTIFACTS.md).
    Includes: activity frequency JSON/PNG/HTML, trace explorer pre-target, process_matrix_drug_drug, lib/ for HTML deps.
    Archived artifacts (process_matrix.png, frequency_map.png, trace_explorer.png, post trace, etc.) are not uploaded.
    When SKIP_DASHBOARD_S3_UPLOAD=1, no upload (notebook 5 Step 6 is the single sync step)."""
    if (os.environ.get("SKIP_DASHBOARD_S3_UPLOAD", "") or "").strip().lower() in ("1", "true", "yes"):
        if logger:
            logger.debug("SKIP_DASHBOARD_S3_UPLOAD set; BupaR S3 upload skipped (notebook 5 Step 6 syncs from local).")
        return True
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
    use_builds = (os.environ.get("S3_VISUALIZATIONS_BUILDS", "") or "").strip().lower() in ("1", "true", "yes")
    builds_suffix = "/builds" if use_builds else ""
    s3_prefix = f"{dashboard_prefix.rstrip('/')}/visualizations/bupar{builds_suffix}/{cohort_name}/{age_band}/plots"
    allowed_basenames = _bupar_rq_artifact_basenames(cohort_name, age_band_fname)

    try:
        from py_helpers.checkpoint_utils import upload_file_to_s3
    except ImportError:
        logger.warning("checkpoint_utils not available; skipping BupaR plot upload to dashboard S3")
        return True

    uploaded = 0
    skipped = 0
    for p in sorted(plots_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(plots_dir)
        # Allow anything under lib/ (interactive HTML deps) and only RQ artifact basenames in plots root
        if len(rel.parts) >= 1 and rel.parts[0] == "lib":
            pass  # allow
        elif len(rel.parts) == 1 and rel.name in allowed_basenames:
            pass  # allow
        else:
            skipped += 1
            logger.debug("Skipping non-RQ artifact (not uploaded): %s", rel.as_posix())
            continue
        try:
            size = p.stat().st_size
            if size is not None and size < 500 and p.suffix == ".html":
                logger.warning(
                    "BupaR HTML file very small (likely empty content): %s size=%s bytes",
                    p.name,
                    size,
                )
        except OSError:
            pass
        key = f"{s3_prefix}/{rel.as_posix()}"
        s3_path = f"s3://{s3_bucket}/{key}"
        if upload_file_to_s3(p, s3_path, logger=logger, check_exists=True):
            uploaded += 1
            logger.debug("Uploaded %s to %s", rel.as_posix(), s3_path)
    if uploaded:
        msg = f"Uploaded {uploaded} BupaR file(s) (RQ artifacts + lib) to s3://{s3_bucket}/{s3_prefix}"
        if skipped:
            msg += f" (skipped {skipped} non-RQ)"
        logger.info("%s", msg)
    else:
        logger.warning("No RQ artifact files found in %s (check R stdout for BupaR diagnostic and empty event log)", plots_dir)
    return True


def create_bupar_visuals(
    cohort_name: str,
    age_band: str,
    force: bool = False,
    local_test: bool = False,
    export_csv_to_json: bool = False,
) -> bool:
    """
    Create BupaR visuals for the dashboard: outputs and plot upload only.
    We do not create or merge BupaR features (no feature engineering in this pipeline).
    If force is False and plots already exist, skips (idempotent).
    """
    age_band_fname = age_band.replace("-", "_")
    plots_dir = DASHBOARD_BUPAR_OUT / "outputs" / cohort_name / age_band_fname / "plots"
    if not force and plots_dir.exists() and list(plots_dir.glob("*.png")):
        logger_bupar = setup_pipeline_logger(
            step_name="4_bupar",
            cohort=cohort_name,
            age_band=age_band,
            script_name="create_bupar_visuals_skip",
            mirror_to_s3=False
        )
        logger_bupar.info("Output exists at %s; skipping (use --force to re-run)", plots_dir)
        return True

    logger = setup_pipeline_logger(
        step_name="4_bupar",
        cohort=cohort_name,
        age_band=age_band,
        script_name="create_bupar_visuals"
    )

    env = detect_runtime_environment(PROJECT_ROOT)
    logger.info(
        "Runtime environment: os=%s logical_cores=%s ram_gb=%s fast_root=%s",
        env.os_name,
        env.logical_cores,
        env.ram_gb,
        env.fast_root,
    )

    with function_block("4_bupar", "create_bupar_visuals", logger=logger.logger):
        logger.info("Starting BupaR visuals for %s / %s", cohort_name, age_band)

        if not create_bupar_outputs(cohort_name, age_band, logger=logger.logger, local_test=local_test):
            logger.error("BupaR outputs step failed; aborting")
            if not local_test:
                logger.log_summary()
            return False

        if not local_test:
            # Optional: export feature CSVs to JSON in plots/ so they upload and can be served as JSON
            if export_csv_to_json:
                features_dir = DASHBOARD_BUPAR_OUT / "outputs" / cohort_name / age_band_fname / "features"
                n = export_bupar_feature_csvs_to_json(plots_dir, features_dir, cohort_name, age_band_fname, logger=logger.logger)
                if n:
                    logger.logger.info("Exported %s BupaR feature CSV(s) to JSON in plots/", n)
            upload_bupar_plots_to_dashboard_s3(cohort_name, age_band, logger=logger.logger)
        else:
            logger.info("Local test: skipping S3 upload")

        logger.info("BupaR visuals completed for %s / %s", cohort_name, age_band)

    if not local_test:
        logger.log_summary()
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
    parser.add_argument(
        "--local-test",
        action="store_true",
        help="One age-band local test: skip SHAP/FFA allowed codes (use all codes); only model data required",
    )
    parser.add_argument(
        "--export-csv-to-json",
        action="store_true",
        help="Export key feature CSVs to JSON in plots/ (traces_top, traces_rare, etc.) so they upload with plots and can be served as JSON",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only run path check for existing outputs (no R, no upload). Exit 0 if no required paths missing.",
    )

    args = parser.parse_args()

    if args.check_only:
        age_band_fname = args.age_band.replace("-", "_")
        plots_dir = DASHBOARD_BUPAR_OUT / "outputs" / args.cohort_name / age_band_fname / "plots"
        features_dir = DASHBOARD_BUPAR_OUT / "outputs" / args.cohort_name / age_band_fname / "features"
        miss_req, miss_opt, found_list = check_bupar_paths(plots_dir, features_dir, args.cohort_name, age_band_fname)
        print(f"BupaR path check: {len(found_list)} found, {len(miss_req)} required missing, {len(miss_opt)} optional missing")
        if miss_req:
            print("Missing required:", ", ".join(miss_req))
        if miss_opt:
            print("Missing optional:", ", ".join(miss_opt))
        sys.exit(0 if not miss_req else 1)

    with module_block("5_bupar"):
        success = create_bupar_visuals(
            cohort_name=args.cohort_name,
            age_band=args.age_band,
            force=args.force,
            local_test=args.local_test,
            export_csv_to_json=args.export_csv_to_json,
        )

    sys.exit(0 if success else 1)
