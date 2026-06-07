#!/usr/bin/env python3
"""
PGx Dashboard Visuals – create all dashboard visualization artifacts.

This script uses the VS Code Jupyter format (# %% cells) so you can run it
as a normal Python script or run cells interactively in VS Code / Cursor.

Steps:
1. Setup: resolve repo root and paths (outputs under 10_risk_dashboard/visualizations/)
2. BupaR: process mining sequences and plots (SHAP/FFA-filtered)
3. DTW: trajectory features and plots (SHAP/FFA-filtered)
4. Extreme-density: extract top ~5% by medical_code density, then DTW for subgroup (parallel; set EXTREME_COMBINATIONS=[] to skip)
5. FP-Growth: itemsets, rules, network plots (SHAP/FFA-filtered)
6. Lambda/API: document endpoints and deployment
7. Deploy Lambda / frontend: skipped by default; run once in 5_build_and_deploy.ipynb (set DEPLOY_LAMBDA=1 / DEPLOY_FRONTEND=1 to run from this script)

Run from repo root (pgx-analysis). Prerequisites: 4_model_data, 7_shap_analysis,
8_ffa_analysis for SHAP/FFA-driven filtering; R and bupaR for BupaR step.
"""

# %%


# --- Setup: paths for dashboard visual pipelines ---
import sys
import subprocess
import os
from pathlib import Path
import numpy as np
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_helpers.constants import AGE_BANDS, COHORT_NAMES  # noqa: E402
from py_helpers.env_utils import get_workflow_python_bin  # noqa: E402

VISUAL_ROOT = REPO_ROOT / "10_risk_dashboard" / "visualizations"
BUPAR_VISUALS_SCRIPT = VISUAL_ROOT / "bupar" / "create_bupar_visuals.py"
DTW_VISUALS_SCRIPT = VISUAL_ROOT / "dtw" / "create_dtw_visuals.py"
EXTREME_EXTRACT_SCRIPT = VISUAL_ROOT / "dtw" / "extract_extreme_density_cohort.py"
FPGROWTH_VISUALS_SCRIPT = VISUAL_ROOT / "fpgrowth" / "create_fpgrowth_visuals.py"

print(f"Repo root: {REPO_ROOT}")
print(f"Visualizations: {VISUAL_ROOT}")

# %%
# --- Config: cohorts and age bands to process ---
COHORTS_TO_RUN = []   # e.g. ["opioid_ed"] or [] for all
AGE_BANDS_TO_RUN = [] # e.g. ["0-12", "13-24"] or [] for all

if not COHORTS_TO_RUN:
    COHORTS_TO_RUN = COHORT_NAMES.copy()
if not AGE_BANDS_TO_RUN:
    AGE_BANDS_TO_RUN = AGE_BANDS.copy()

print(f"Cohorts: {COHORTS_TO_RUN}")
print(f"Age bands: {AGE_BANDS_TO_RUN}")
combinations = [(c, ab) for c in COHORTS_TO_RUN for ab in AGE_BANDS_TO_RUN]
print(f"Total combinations: {len(combinations)}")

# Idempotent: skip when output exists. Set FORCE_RERUN=True to re-run all.
FORCE_RERUN = False
# Parallel workers for BupaR, DTW, and extreme-density (FP-Growth stays sequential for memory).
PARALLEL_WORKERS = 32
# Extreme-density: same (cohort, age_band) as main pipeline, or [] to skip.
EXTREME_COMBINATIONS = combinations.copy()

# %%
# --- Run BupaR process mining (event logs, traces, plots; SHAP/FFA-filtered when available) ---
# Parallel; idempotent unless FORCE_RERUN.
FAIL_FAST = True  # set False to continue on first failure

force_flag = ["--force"] if FORCE_RERUN else []

def _run_bupar_one(cohort_name, age_band):
    return (cohort_name, age_band, subprocess.run(
        [str(get_workflow_python_bin()), str(BUPAR_VISUALS_SCRIPT), "--cohort-name", cohort_name, "--age-band", age_band] + force_flag,
        cwd=str(REPO_ROOT),
        capture_output=False,
    ).returncode)

with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
    futures = {ex.submit(_run_bupar_one, c, ab): (c, ab) for c, ab in combinations}
    for fut in as_completed(futures):
        cohort_name, age_band, code = fut.result()
        print(f"  [BupaR] {cohort_name} / {age_band} -> exit {code}")
        if code != 0 and FAIL_FAST:
            raise RuntimeError(f"BupaR failed: {cohort_name} / {age_band}")
print("BupaR done.")

# %%

# --- Run DTW visuals only (parallel; idempotent unless FORCE_RERUN). We do not create DTW features here. ---

def save_time_between_events_histogram(cohort, age_band, bins=30):
    """
    Generate and save time-between-events histogram JSON for a given cohort and age band.
    Reads DTW features parquet, computes histogram, writes JSON to outputs dir.
    """
    # File paths
    age_band_fname = age_band.replace("-", "_")
    features_path = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "dtw" / "feature_engineering" / f"dtw_features_{cohort}_{age_band_fname}.parquet"
    output_dir = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "dtw" / "outputs" / cohort / age_band_fname
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "time_between_events_histogram.json"
    if not features_path.exists():
        print(f"[histogram] Features file missing: {features_path}")
        return False
    try:
        df = pd.read_parquet(features_path)
        if "mean_days_between_events" not in df.columns:
            print(f"[histogram] Column mean_days_between_events missing in {features_path}")
            return False
        values = df["mean_days_between_events"].dropna().values
        if len(values) == 0:
            print(f"[histogram] No values for mean_days_between_events in {features_path}")
            return False
        counts, bin_edges = np.histogram(values, bins=bins)
        histogram_json = {
            "type": "histogram",
            "cohort": cohort,
            "age_band": age_band,
            "bin_edges": bin_edges.tolist(),
            "counts": counts.tolist(),
            "n": int(len(values)),
            "x_label": "Mean days between events",
            "y_label": "Patient count"
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(histogram_json, f, indent=2)
        print(f"[histogram] Saved: {json_path}")
        return True
    except Exception as e:
        print(f"[histogram] Error for {cohort}/{age_band}: {e}")
        return False

def _run_dtw_one(cohort_name, age_band):
    r = subprocess.run(
        [str(get_workflow_python_bin()), str(DTW_VISUALS_SCRIPT), "--cohort-name", cohort_name, "--age-band", age_band] + force_flag,
        cwd=str(REPO_ROOT),
        capture_output=False,
    )
    # After DTW visuals, generate histogram JSON
    hist_ok = save_time_between_events_histogram(cohort_name, age_band)
    return (cohort_name, age_band, r.returncode, hist_ok)

with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
    futures = {ex.submit(_run_dtw_one, c, ab): (c, ab) for c, ab in combinations}
    for fut in as_completed(futures):
        cohort_name, age_band, code, hist_ok = fut.result()
        print(f"  [DTW] {cohort_name} / {age_band} -> exit {code} | histogram: {'ok' if hist_ok else 'fail'}")
        if code != 0 and FAIL_FAST:
            raise RuntimeError(f"DTW create_dtw_visuals failed: {cohort_name} / {age_band}")
print("DTW + histogram done.")

# %%
# --- Run extreme-density extract + DTW visuals only (parallel; idempotent unless FORCE_RERUN) ---
# Extract top ~5% by medical_code density into {cohort}_extreme_density, then run DTW visuals for that subgroup.
def _run_extreme_one(cohort_name, age_band):
    r0 = subprocess.run(
        [str(get_workflow_python_bin()), str(EXTREME_EXTRACT_SCRIPT), "--cohort-name", cohort_name, "--age-band", age_band],
        cwd=str(REPO_ROOT),
        capture_output=False,
    )
    if r0.returncode != 0:
        return (cohort_name, age_band, r0.returncode, None)
    extreme_name = f"{cohort_name}_extreme_density"
    r2 = subprocess.run(
        [str(get_workflow_python_bin()), str(DTW_VISUALS_SCRIPT), "--cohort-name", extreme_name, "--age-band", age_band,
         "--project-root", str(REPO_ROOT)] + force_flag,
        cwd=str(REPO_ROOT),
        capture_output=False,
    )
    return (cohort_name, age_band, r0.returncode, r2.returncode)

if EXTREME_COMBINATIONS:
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
        futures = {ex.submit(_run_extreme_one, c, ab): (c, ab) for c, ab in EXTREME_COMBINATIONS}
        for fut in as_completed(futures):
            cohort_name, age_band, c0, c2 = fut.result()
            print(f"  [Extreme] {cohort_name} / {age_band} -> extract={c0}, dtw_vis={c2}")
            if c0 != 0 and FAIL_FAST:
                raise RuntimeError(f"Extract extreme cohort failed: {cohort_name} / {age_band}")
            if c2 is not None and c2 != 0 and FAIL_FAST:
                raise RuntimeError(f"DTW create_dtw_visuals failed: {cohort_name}_extreme_density / {age_band}")
    print(f"Extreme-density done ({len(EXTREME_COMBINATIONS)} combinations).")
else:
    print("EXTREME_COMBINATIONS empty; skipping extreme-density.")

# %%
# --- Run FP-Growth (itemsets, rules, plots; sequential for memory; idempotent unless FORCE_RERUN) ---
for cohort_name, age_band in combinations:
    print(f"\n[FP-Growth] {cohort_name} / {age_band}")
    result = subprocess.run(
        [str(get_workflow_python_bin()), str(FPGROWTH_VISUALS_SCRIPT), "--cohort-name", cohort_name, "--age-band", age_band] + force_flag,
        cwd=str(REPO_ROOT),
        capture_output=False,
    )
    if result.returncode != 0 and FAIL_FAST:
        raise RuntimeError(f"FP-Growth failed: {cohort_name} / {age_band}")
    print(f"  -> exit code {result.returncode}")

# %%
# --- Lambda / API Gateway: endpoints used by dashboard visuals ---
# The dashboard frontend calls these endpoints. Ensure API Gateway has proxy
# to Lambda and Lambda has access to S3 paths below.
#
# GET /visualizations/scenario?cohort=...&age_band=...[&drugs=...&icds=...&cpts=...]
#   -> Causal + SHAP importance (filtered by codes or top SHAP/FFA when no selection)
#
# GET /visualizations/bupar?cohort=...&age_band=...
#   -> S3 paths to BupaR PNGs (gold/feature_importance/{cohort}/{age_band}/plots/)
#
# GET /visualizations/dtw?cohort=...&age_band=...
#   -> S3 paths to DTW images + routine_comparison / high_risk_trajectories chart data
#
# GET /visualizations/fpgrowth?cohort=...&age_band=...&item_type=...
#   -> S3 paths to FP-Growth itemsets/support/network (gold/fpgrowth/{cohort}/{age_band}/plots/)
#
# To (re)deploy API: see utility_scripts/create_api_gateway_pgx_risk_calculator.sh
# and 10_risk_dashboard/backend/README.md. Lambda reads from S3 bucket (PGX_RESULTS_BUCKET).
print("Dashboard visualization endpoints are documented in 10_risk_dashboard/backend/README.md")
print("To update API Gateway: utility_scripts/create_api_gateway_pgx_risk_calculator.sh")

# %%
# --- Deploy Lambda: build image, push ECR, update function ---
# Build and deploy run once in 5_build_and_deploy.ipynb. This script skips deploy by default; set DEPLOY_LAMBDA=1 to run from here.
DASHBOARD_DIR = REPO_ROOT / "10_risk_dashboard"
SKIP_DEPLOY_LAMBDA = os.environ.get("DEPLOY_LAMBDA", "").strip() not in ("1", "true", "yes")
docker_script = DASHBOARD_DIR / "deployment" / "docker_build.sh"
LAMBDA_NAME = "pgx-risk-calculator"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

if not SKIP_DEPLOY_LAMBDA and docker_script.exists():
    print("Deploy Lambda: building image and pushing to ECR...")
    r = subprocess.run(["bash", str(docker_script)], cwd=str(DASHBOARD_DIR))
    if r.returncode != 0:
        print("Docker build/push failed.")
    else:
        acc = subprocess.run(
            ["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"],
            capture_output=True, text=True
        )
        if acc.returncode == 0:
            ecr_uri = f"{acc.stdout.strip()}.dkr.ecr.{AWS_REGION}.amazonaws.com/pgx-risk-calculator:latest"
            print("Updating Lambda function...")
            r2 = subprocess.run(
                ["aws", "lambda", "update-function-code", "--function-name", LAMBDA_NAME,
                 "--image-uri", ecr_uri, "--region", AWS_REGION]
            )
            if r2.returncode == 0:
                subprocess.run(
                    ["aws", "lambda", "wait", "function-updated", "--function-name", LAMBDA_NAME, "--region", AWS_REGION],
                    capture_output=True
                )
                print("Lambda updated.")
            else:
                print("Lambda update failed.")
        else:
            print("Could not get AWS account ID.")
elif not SKIP_DEPLOY_LAMBDA:
    print("Docker script not found:", docker_script)

# %%
# --- Deploy frontend: sync frontend to S3 ---
# Build and deploy run once in 5_build_and_deploy.ipynb. This script skips by default; set DEPLOY_FRONTEND=1 to run from here.
SKIP_DEPLOY_FRONTEND = os.environ.get("DEPLOY_FRONTEND", "").strip() not in ("1", "true", "yes")
frontend_dir = DASHBOARD_DIR / "frontend"
s3_bucket = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
s3_prefix = os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator")
s3_uri = f"s3://{s3_bucket}/{s3_prefix}/"

if not SKIP_DEPLOY_FRONTEND and frontend_dir.exists():
    print(f"Syncing frontend to {s3_uri}")
    r = subprocess.run(["aws", "s3", "sync", str(frontend_dir), s3_uri, "--region", "us-east-1"])
    if r.returncode == 0:
        print("Frontend synced.")
        # Upload feature importance heatmaps (notebook 4 copies 3a → visualizations/feature_importance; sync from there)
        # Match pattern: S3_VISUALIZATIONS_BUILDS=1 → upload to .../builds/ (notebook 4); else final (notebook 5)
        fi_base = VISUAL_ROOT / "feature_importance"
        prefix_clean = s3_prefix.strip("/")
        use_fi_builds = (os.environ.get("S3_VISUALIZATIONS_BUILDS", "") or "").strip().lower() in ("1", "true", "yes")
        fi_builds_suffix = "/builds" if use_fi_builds else ""
        fi_prefix = f"{prefix_clean}/visualizations/feature_importance{fi_builds_suffix}"
        uploaded_fi = 0
        for cohort in COHORT_NAMES:
            cohort_dir = fi_base / cohort
            local_png = cohort_dir / "aggregated_fi_heatmap.png"
            local_json = cohort_dir / "aggregated_fi_heatmap.json"
            if local_png.exists():
                s3_key = f"{fi_prefix}/{cohort}/aggregated_fi_heatmap.png"
                r2 = subprocess.run(
                    ["aws", "s3", "cp", str(local_png), f"s3://{s3_bucket}/{s3_key}", "--region", "us-east-1"],
                    capture_output=True, text=True,
                )
                if r2.returncode == 0:
                    uploaded_fi += 1
                    print(f"  Uploaded FI heatmap: {s3_key}")
            if local_json.exists():
                s3_key_json = f"{fi_prefix}/{cohort}/aggregated_fi_heatmap.json"
                r2j = subprocess.run(
                    ["aws", "s3", "cp", str(local_json), f"s3://{s3_bucket}/{s3_key_json}", "--region", "us-east-1"],
                    capture_output=True, text=True,
                )
                if r2j.returncode == 0:
                    uploaded_fi += 1
                    print(f"  Uploaded FI heatmap data: {s3_key_json}")
            # Per-model heatmap JSONs (same pattern: row_labels, column_labels, matrix) for model filter
            for model in ("catboost", "xgboost", "xgboost_rf"):
                m_json = cohort_dir / f"{model}_fi_heatmap.json"
                if m_json.exists():
                    m_key = f"{fi_prefix}/{cohort}/{model}_fi_heatmap.json"
                    r2m = subprocess.run(
                        ["aws", "s3", "cp", str(m_json), f"s3://{s3_bucket}/{m_key}", "--region", "us-east-1"],
                        capture_output=True, text=True,
                    )
                    if r2m.returncode == 0:
                        uploaded_fi += 1
                        print(f"  Uploaded FI heatmap data: {m_key}")
        combined_png = fi_base / "combined_cohorts_feature_importance_heatmap.png"
        if combined_png.exists():
            s3_key_combined = f"{fi_prefix}/combined_cohorts_feature_importance_heatmap.png"
            r3 = subprocess.run(
                ["aws", "s3", "cp", str(combined_png), f"s3://{s3_bucket}/{s3_key_combined}", "--region", "us-east-1"],
                capture_output=True, text=True,
            )
            if r3.returncode == 0:
                uploaded_fi += 1
                print(f"  Uploaded FI heatmap: {s3_key_combined}")
        # Combined cohort heatmap JSON for dashboard Plotly (GET ?cohort=combined)
        combined_json = fi_base / "combined" / "aggregated_fi_heatmap.json"  # unchanged path under viz
        if combined_json.exists():
            s3_key_combined_json = f"{fi_prefix}/combined/aggregated_fi_heatmap.json"
            r3j = subprocess.run(
                ["aws", "s3", "cp", str(combined_json), f"s3://{s3_bucket}/{s3_key_combined_json}", "--region", "us-east-1"],
                capture_output=True, text=True,
            )
            if r3j.returncode == 0:
                uploaded_fi += 1
                print(f"  Uploaded FI heatmap data: {s3_key_combined_json}")
        if uploaded_fi:
            print(f"  Feature importance: {uploaded_fi} heatmap(s) uploaded.")
        elif fi_base.exists():
            print("  No feature importance heatmaps found under 10_risk_dashboard/visualizations/feature_importance (run notebook 4 FI heatmaps + copy).")
        # Upload scenario dashboard JSON (Scenario Analysis tab): visualizations/scenario -> S3 visualizations/scenario/
        scenario_script = DASHBOARD_DIR / "data_preparation" / "upload_scenario_outputs_to_s3.py"
        if scenario_script.exists():
            r_scenario = subprocess.run([str(get_workflow_python_bin()), str(scenario_script)], cwd=str(REPO_ROOT), capture_output=True, text=True)
            if r_scenario.returncode == 0 and r_scenario.stdout:
                for line in r_scenario.stdout.strip().split("\n"):
                    print(f"  {line}")
            elif r_scenario.returncode != 0 and r_scenario.stderr:
                print("  Causal upload:", r_scenario.stderr.strip() or "failed")
    else:
        print("S3 sync failed.")
elif not SKIP_DEPLOY_FRONTEND:
    print("Frontend dir not found:", frontend_dir)

# %%
# When run as script (python pgx_dashboard_visuals.py), the full file runs top-to-bottom
# so the BupaR/DTW/FP-Growth cells above execute. In VS Code/Cursor, run by cell (# %%) instead.
if __name__ == "__main__":
    print("Pipeline complete. Upload outputs to S3 for Lambda (see 10_risk_dashboard/backend/README.md).")
