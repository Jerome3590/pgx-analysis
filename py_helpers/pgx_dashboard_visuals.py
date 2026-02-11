#!/usr/bin/env python3
"""
PGx Dashboard Visuals – create all dashboard visualization artifacts.

This script uses the VS Code Jupyter format (# %% cells) so you can run it
as a normal Python script or run cells interactively in VS Code / Cursor.

Steps:
1. Setup: resolve repo root, create symlinks 10b/10c/10d at repo root if needed
2. BupaR: process mining sequences and plots (SHAP/FFA-filtered)
3. DTW: trajectory features and plots (SHAP/FFA-filtered)
4. FP-Growth: itemsets, rules, network plots (SHAP/FFA-filtered)
5. Lambda/API: document endpoints and deployment
6. Deploy Lambda / frontend: skipped by default; run once in 5_build_and_deploy.ipynb (set DEPLOY_LAMBDA=1 / DEPLOY_FRONTEND=1 to run from this script)

Run from repo root (pgx-analysis). Prerequisites: 4_model_data, 7_shap_analysis,
8_ffa_analysis for SHAP/FFA-driven filtering; R and bupaR for BupaR step.
"""

# %%
# --- Setup: paths and symlinks for dashboard visual pipelines ---
import os
import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_helpers.constants import AGE_BANDS, COHORT_NAMES  # noqa: E402

VISUAL_ROOT = REPO_ROOT / "9_risk_dashboard" / "visualizations"
BUPAR_VISUALS_SCRIPT = VISUAL_ROOT / "bupar" / "create_bupar_visuals.py"
DTW_FEATURES_SCRIPT = VISUAL_ROOT / "dtw" / "create_dtw_features.py"
DTW_VISUALS_SCRIPT = VISUAL_ROOT / "dtw" / "create_dtw_visuals.py"
FPGROWTH_VISUALS_SCRIPT = VISUAL_ROOT / "fpgrowth" / "create_fpgrowth_visuals.py"

print(f"Repo root: {REPO_ROOT}")
print(f"Visualizations: {VISUAL_ROOT}")

# %%
# --- Create symlinks 10b, 10c, 10d (idempotent: no-op if present) ---
def ensure_dashboard_symlinks():
    """Create 10b/10c/10d at repo root and under visualizations so R and Python scripts find them."""
    # At repo root: so R (cwd=REPO_ROOT) finds 10c_bupaR_dashboard_visual/outputs, 4a_model_data, etc.
    repo_links = [
        ("10c_bupaR_dashboard_visual", "9_risk_dashboard/visualizations/bupar"),
        ("10b_fpgrowth_dashboard_visual", "9_risk_dashboard/visualizations/fpgrowth"),
        ("10d_dtw_dashboard_visual", "9_risk_dashboard/visualizations/dtw"),
    ]
    for name, target in repo_links:
        path = REPO_ROOT / name
        target_path = REPO_ROOT / target
        if path.exists():
            print(f"  [repo] {name} exists")
            continue
        if not target_path.exists():
            print(f"  [repo] Skip {name}: target not found")
            continue
        try:
            path.symlink_to(target_path.relative_to(path.parent))
            print(f"  [repo] Created: {name} -> {target}")
        except OSError as e:
            if os.name == "nt":
                print(f"  [repo] Windows: create junction: mklink /J \"{path}\" \"{target_path}\"")
            else:
                print(f"  [repo] {name}: {e}")
    # Under visualizations: so create_*_visuals.py (PROJECT_ROOT=visualizations) finds 10c/10b/10d
    for name, subdir in [("10c_bupaR_dashboard_visual", "bupar"), ("10b_fpgrowth_dashboard_visual", "fpgrowth"), ("10d_dtw_dashboard_visual", "dtw")]:
        path = VISUAL_ROOT / name
        if path.exists():
            print(f"  [visual] {name} exists")
            continue
        target = VISUAL_ROOT / subdir
        if not target.exists():
            continue
        try:
            path.symlink_to(subdir)
            print(f"  [visual] Created: {name} -> {subdir}")
        except OSError as e:
            if os.name == "nt":
                print(f"  [visual] Windows: mklink /J \"{path}\" \"{target}\"")
            else:
                print(f"  [visual] {name}: {e}")

ensure_dashboard_symlinks()

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
# Parallel workers for BupaR and DTW (FP-Growth stays sequential for memory).
PARALLEL_WORKERS = 32

# %%
# --- Run BupaR process mining (event logs, traces, plots; SHAP/FFA-filtered when available) ---
# Parallel; idempotent unless FORCE_RERUN.
FAIL_FAST = True  # set False to continue on first failure
from concurrent.futures import ThreadPoolExecutor, as_completed

force_flag = ["--force"] if FORCE_RERUN else []

def _run_bupar_one(cohort_name, age_band):
    return (cohort_name, age_band, subprocess.run(
        [sys.executable, str(BUPAR_VISUALS_SCRIPT), "--cohort-name", cohort_name, "--age-band", age_band] + force_flag,
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
# --- Run DTW trajectory features and publish (parallel; idempotent unless FORCE_RERUN) ---
def _run_dtw_one(cohort_name, age_band):
    r1 = subprocess.run(
        [sys.executable, str(DTW_FEATURES_SCRIPT), "--cohort", cohort_name, "--age_band", age_band] + force_flag,
        cwd=str(REPO_ROOT),
        capture_output=False,
    )
    if r1.returncode != 0:
        return (cohort_name, age_band, r1.returncode, None)
    r2 = subprocess.run(
        [sys.executable, str(DTW_VISUALS_SCRIPT), "--cohort-name", cohort_name, "--age-band", age_band] + force_flag,
        cwd=str(REPO_ROOT),
        capture_output=False,
    )
    return (cohort_name, age_band, r1.returncode, r2.returncode)

with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
    futures = {ex.submit(_run_dtw_one, c, ab): (c, ab) for c, ab in combinations}
    for fut in as_completed(futures):
        cohort_name, age_band, c1, c2 = fut.result()
        print(f"  [DTW] {cohort_name} / {age_band} -> exit {c1}, {c2}")
        if c2 is None and FAIL_FAST:
            raise RuntimeError(f"DTW create_dtw_features failed: {cohort_name} / {age_band}")
        if c2 is not None and c2 != 0 and FAIL_FAST:
            raise RuntimeError(f"DTW create_dtw_visuals failed: {cohort_name} / {age_band}")
print("DTW done.")

# %%
# --- Run FP-Growth (itemsets, rules, plots; sequential for memory; idempotent unless FORCE_RERUN) ---
for cohort_name, age_band in combinations:
    print(f"\n[FP-Growth] {cohort_name} / {age_band}")
    result = subprocess.run(
        [sys.executable, str(FPGROWTH_VISUALS_SCRIPT), "--cohort-name", cohort_name, "--age-band", age_band] + force_flag,
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
# GET /visualizations/causal?cohort=...&age_band=...[&drugs=...&icds=...&cpts=...]
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
# and 9_risk_dashboard/backend/README.md. Lambda reads from S3 bucket (PGX_RESULTS_BUCKET).
print("Dashboard visualization endpoints are documented in 9_risk_dashboard/backend/README.md")
print("To update API Gateway: utility_scripts/create_api_gateway_pgx_risk_calculator.sh")

# %%
# --- Deploy Lambda: build image, push ECR, update function ---
# Build and deploy run once in 5_build_and_deploy.ipynb. This script skips deploy by default; set DEPLOY_LAMBDA=1 to run from here.
DASHBOARD_DIR = REPO_ROOT / "9_risk_dashboard"
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
    else:
        print("S3 sync failed.")
elif not SKIP_DEPLOY_FRONTEND:
    print("Frontend dir not found:", frontend_dir)

# %%
# When run as script (python pgx_dashboard_visuals.py), the full file runs top-to-bottom
# so the BupaR/DTW/FP-Growth cells above execute. In VS Code/Cursor, run by cell (# %%) instead.
if __name__ == "__main__":
    print("Pipeline complete. Upload outputs to S3 for Lambda (see 9_risk_dashboard/backend/README.md).")
