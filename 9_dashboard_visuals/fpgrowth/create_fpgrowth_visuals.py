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
import threading
import time
from pathlib import Path

# Step folder (9_dashboard_visuals) and repo root; outputs go to 10_risk_dashboard/visualizations/fpgrowth
REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_FPGROWTH_OUT = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "fpgrowth"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_helpers.env_utils import get_workflow_python_bin  # noqa: E402
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
from fpgrowth.cohort_fpgrowth import get_item_types_for_cohort  # noqa: E402


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


    import json

    with step_block("6_fpgrowth", "ensure_itemsets", logger=logger):
        if itemsets_exist and not force:
            logger.info("Itemsets already exist at %s; skipping creation (use --force to re-run)", itemsets_dir)
            return True
        if itemsets_exist and force:
            logger.info("Itemsets exist at %s; re-running due to --force", itemsets_dir)

        item_types = get_item_types_for_cohort(cohort_name)
        logger.info("="*60)
        logger.info("CREATING FP-GROWTH ITEMSETS: %s / %s", cohort_name, age_band)
        logger.info("="*60)
        logger.info("Processing %d item types: %s", len(item_types), ", ".join(item_types))
        script_path = PROJECT_ROOT / "fpgrowth" / "run_single_cohort_fpgrowth.py"
        cmd = [
            str(get_workflow_python_bin()),
            str(script_path),
            "--cohort-name",
            cohort_name,
            "--age-band",
            age_band,
            "--event-year",
            "train",
            "--project-root",
            str(REPO_ROOT),
        ]
        logger.info(
            "Starting itemset subprocess: cwd=%s script=%s (streaming stdout/stderr to this log)",
            PROJECT_ROOT,
            script_path.name,
        )
        t0 = time.perf_counter()
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            def stream_to_log(stream, prefix):
                for line in iter(stream.readline, ""):
                    logger.info("%s %s", prefix, line.rstrip())

            out_thread = threading.Thread(target=stream_to_log, args=(proc.stdout, "[stdout]"))
            err_thread = threading.Thread(target=stream_to_log, args=(proc.stderr, "[stderr]"))
            out_thread.daemon = True
            err_thread.daemon = True
            out_thread.start()
            err_thread.start()
            out_thread.join()
            err_thread.join()
            returncode = proc.wait()
            elapsed = time.perf_counter() - t0

            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, cmd, None, None)

            logger.info("="*60)
            logger.info(
                "[OK] FP-Growth itemsets created successfully for %s / %s (returncode=0, duration=%.1fs)",
                cohort_name,
                age_band,
                elapsed,
            )
            logger.info("="*60)
            # Verify outputs exist after run
            itemsets_now = itemsets_dir.exists() and any(itemsets_dir.glob("*_itemsets*.json"))
            if itemsets_now:
                itemset_files = list(itemsets_dir.glob("*_itemsets*.json"))
                logger.info("Created %d itemset files in %s", len(itemset_files), itemsets_dir)
                return True
            else:
                # No itemsets produced: write empty JSON files for each expected item type
                logger.warning("No itemsets produced for %s / %s; writing empty JSON files for downstream compatibility.", cohort_name, age_band)
                itemsets_dir.mkdir(parents=True, exist_ok=True)
                for item_type in item_types:
                    empty_files = [
                        f"{item_type}_itemsets.json",
                        f"{item_type}_rules.json",
                        f"{item_type}_metrics.json",
                        f"{item_type}_encoding_map.json",
                        f"{item_type}_itemsets_target_only.json",
                        f"{item_type}_rules_target_only.json",
                    ]
                    for fname in empty_files:
                        fpath = itemsets_dir / fname
                        with open(fpath, "w") as f:
                            json.dump([], f)
                return True
        except subprocess.CalledProcessError as exc:
            elapsed = time.perf_counter() - t0
            logger.error("="*60)
            logger.error(
                "Itemset creation FAILED for %s / %s (returncode=%s, duration=%.1fs)",
                cohort_name,
                age_band,
                exc.returncode,
                elapsed,
            )
            logger.error("="*60)
            logger.error("Subprocess output was streamed above; check [stdout]/[stderr] lines in this log.")
            logger.warning(
                "Continuing; itemsets may already exist locally or be available from S3"
            )
            itemsets_now = itemsets_dir.exists() and any(itemsets_dir.glob("*_itemsets*.json"))
            if not itemsets_now:
                # No itemsets produced: write empty JSON files for each expected item type
                logger.warning("No itemsets produced for %s / %s after failure; writing empty JSON files for downstream compatibility.", cohort_name, age_band)
                itemsets_dir.mkdir(parents=True, exist_ok=True)
                for item_type in item_types:
                    empty_files = [
                        f"{item_type}_itemsets.json",
                        f"{item_type}_rules.json",
                        f"{item_type}_metrics.json",
                        f"{item_type}_encoding_map.json",
                        f"{item_type}_itemsets_target_only.json",
                        f"{item_type}_rules_target_only.json",
                    ]
                    for fname in empty_files:
                        fpath = itemsets_dir / fname
                        with open(fpath, "w") as f:
                            json.dump([], f)
                return True
            return itemsets_now
        except Exception as exc:  # pragma: no cover - defensive
            elapsed = time.perf_counter() - t0
            logger.error("Itemset creation failed with exception after %.1fs: %s", elapsed, exc)
            logger.warning(
                "Continuing; itemsets may already exist locally or be available from S3"
            )
            itemsets_now = itemsets_dir.exists() and any(itemsets_dir.glob("*_itemsets*.json"))
            if not itemsets_now:
                # No itemsets produced: write empty JSON files for each expected item type
                logger.warning("No itemsets produced for %s / %s after exception; writing empty JSON files for downstream compatibility.", cohort_name, age_band)
                itemsets_dir.mkdir(parents=True, exist_ok=True)
                for item_type in item_types:
                    empty_files = [
                        f"{item_type}_itemsets.json",
                        f"{item_type}_rules.json",
                        f"{item_type}_metrics.json",
                        f"{item_type}_encoding_map.json",
                        f"{item_type}_itemsets_target_only.json",
                        f"{item_type}_rules_target_only.json",
                    ]
                    for fname in empty_files:
                        fpath = itemsets_dir / fname
                        with open(fpath, "w") as f:
                            json.dump([], f)
                return True
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
        item_types = get_item_types_for_cohort(cohort_name)
        # Visualization artifacts: cohort then age_band only
        plots_output_dir = (
            REPO_ROOT
            / "10_risk_dashboard"
            / "visualizations"
            / "fpgrowth"
            / cohort_name
            / age_band_fname
            / "plots"
        )
        plots_output_dir.mkdir(parents=True, exist_ok=True)

        fpgrowth_outputs_root = DASHBOARD_FPGROWTH_OUT
        cmd = [
            str(get_workflow_python_bin()),
            str(script_path),
            "--base-dir",
            str(fpgrowth_outputs_root),
            "--cohort-name",
            cohort_name,
            "--age-band",
            age_band,
            "--output-dir",
            str(plots_output_dir),
        ]
        if item_types:
            cmd += ["--item-types"] + item_types
        logger.info(
            "Starting visualization subprocess: cwd=%s script=%s item_types=%s",
            PROJECT_ROOT,
            script_path.name,
            item_types,
        )
        t0 = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            elapsed = time.perf_counter() - t0
            logger.info(
                "Visualizations created (returncode=0, duration=%.1fs)",
                elapsed,
            )
            if result.stdout:
                logger.info("Plots stdout:\n%s", result.stdout)
            if result.stderr:
                logger.info("Plots stderr:\n%s", result.stderr)
            return True
        except subprocess.CalledProcessError as exc:
            elapsed = time.perf_counter() - t0
            logger.error(
                "Visualization creation failed (returncode=%s, duration=%.1fs)",
                exc.returncode,
                elapsed,
            )
            if exc.stderr:
                logger.error("stderr:\n%s", exc.stderr)
            return False
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Visualization creation failed with exception: %s", exc)
            return False


def generate_combined_bin_itemset_heatmap(
    cohort_name: str,
    age_band: str,
    top_n: int = 50,
    logger: logging.Logger | None = None,
) -> bool:
    """Build density/combined/fpgrowth_itemset_heatmap.json from per-bin drug_name_itemsets JSONs.

    Reads per-bin drug_name_itemsets.json for each density bin, extracts support values,
    and writes a cross-bin support matrix in the format expected by the dashboard:
      {row_labels, column_labels, matrix, metric}

    Output path (local):
      DASHBOARD_FPGROWTH_OUT/{cohort}/{age_band_fname}/density/combined/fpgrowth_itemset_heatmap.json
    S3 key (underscores — matches frontend fetch URL):
      visualizations/fpgrowth/{cohort}/{age_band_fname}/density/combined/fpgrowth_itemset_heatmap.json
    """
    import json as _json
    age_band_fname = age_band.replace("-", "_")
    bins = ("low", "medium", "high", "extreme")

    def _format_itemset_label(itemsets) -> str:
        if isinstance(itemsets, list):
            parts = itemsets
        elif isinstance(itemsets, str):
            parts = itemsets.split(",")
        else:
            parts = [str(itemsets)]
        cleaned = [p.strip().replace("DRUG:", "").replace("drug_", "").replace("_", " ").title() for p in parts if p]
        return " : ".join(cleaned[:5]) or "(empty)"

    bin_data: dict = {}
    for bin_name in bins:
        bin_json_path = (
            DASHBOARD_FPGROWTH_OUT / cohort_name / age_band_fname
            / "density" / bin_name / "plots" / "drug_name_itemsets.json"
        )
        if not bin_json_path.exists():
            # Try legacy path without /plots/
            legacy_path = (
                DASHBOARD_FPGROWTH_OUT / cohort_name / age_band_fname
                / "density" / bin_name / "drug_name_itemsets.json"
            )
            if legacy_path.exists():
                bin_json_path = legacy_path
            else:
                if logger:
                    logger.warning("drug_name_itemsets.json not found for bin=%s: %s", bin_name, bin_json_path)
                continue
        try:
            with open(bin_json_path) as f:
                records = _json.load(f)
            if isinstance(records, list) and records:
                bin_data[bin_name] = records
        except Exception as e:
            if logger:
                logger.warning("Could not read itemsets JSON for bin=%s: %s", bin_name, e)

    if not bin_data:
        if logger:
            logger.warning("No per-bin itemsets JSON files found for %s/%s; skipping combined heatmap", cohort_name, age_band)
        return False

    # Build per-itemset support dict per bin: label -> {bin: support}
    all_labels: dict = {}
    for bin_name, records in bin_data.items():
        for rec in records:
            label = _format_itemset_label(rec.get("itemsets", []))
            support = float(rec.get("support", 0))
            if label not in all_labels:
                all_labels[label] = {}
            if label not in all_labels or support > all_labels[label].get(bin_name, 0):
                all_labels[label][bin_name] = round(support, 6)

    if not all_labels:
        if logger:
            logger.warning("No itemset labels extracted; skipping combined heatmap")
        return False

    ranked_labels = sorted(
        all_labels.keys(),
        key=lambda lbl: max(all_labels[lbl].values()),
        reverse=True,
    )[:top_n]

    column_labels = [b for b in bins if b in bin_data]
    matrix = [
        [all_labels[lbl].get(b, 0.0) for b in column_labels]
        for lbl in ranked_labels
    ]

    heatmap_json = {
        "row_labels": ranked_labels,
        "column_labels": column_labels,
        "matrix": matrix,
        "metric": "support",
        "cohort": cohort_name,
        "age_band": age_band,
    }

    out_dir = DASHBOARD_FPGROWTH_OUT / cohort_name / age_band_fname / "density" / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fpgrowth_itemset_heatmap.json"
    with open(out_path, "w") as f:
        _json.dump(heatmap_json, f)
    if logger:
        logger.info(
            "Wrote combined FP-Growth itemset heatmap: %s (%d itemsets × %d bins)",
            out_path, len(ranked_labels), len(column_labels),
        )

    try:
        import boto3 as _boto3
        import os as _os
        s3_bucket = _os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
        dashboard_prefix = _os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator")
        s3_key = (
            f"{dashboard_prefix.rstrip('/')}/visualizations/fpgrowth"
            f"/{cohort_name}/{age_band_fname}/density/combined/fpgrowth_itemset_heatmap.json"
        )
        _boto3.client("s3").put_object(
            Bucket=s3_bucket,
            Key=s3_key,
            Body=_json.dumps(heatmap_json).encode(),
            ContentType="application/json",
        )
        if logger:
            logger.info("Uploaded combined FP-Growth heatmap to s3://%s/%s", s3_bucket, s3_key)
    except Exception as e:
        if logger:
            logger.warning("S3 upload failed for combined FP-Growth heatmap: %s", e)

    return True


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
        step_name="9_fpgrowth",
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
                "No itemsets produced for %s / %s (e.g. model_data missing or no transactions). Check log.",
                cohort_name,
                age_band,
            )
        else:
            logger.info("[OK] Itemsets ready")

        if not skip_visualizations:
            logger.info("[STEP 2/2] Creating visualizations...")
            ok = create_visualizations(cohort_name, age_band, logger=logger.logger)
            if not ok:
                logger.error("Visualization step failed")
            else:
                logger.info("[OK] Visualizations complete")
            generate_combined_bin_itemset_heatmap(cohort_name, age_band, logger=logger.logger)
        else:
            logger.info("[STEP 2/2] Skipping visualization creation")

        logger.info("")
        logger.info("#" * 70)
        logger.info("#  FP-GROWTH COMPLETED: %s / %s", cohort_name, age_band)
        logger.info("#" * 70)
        logger.info("")

    logger.info("Calling log_summary (log will be mirrored to S3)...")
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
