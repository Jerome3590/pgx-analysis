# -*- coding: utf-8 -*-
# Auto-generated from 0_config_and_pipeline.ipynb (VS Code Python notebook script format)

# %% [markdown]
# # Config & Pipeline Run Guide
#
# This notebook supports:
# 1. **Environment checks**: Python, R, and **Docker** (for risk calculator Lambda container).
# 2. **Reset**: Run the **cleanup script** to clear checkpoints and S3/local outputs before a workflow run. **Default:** Feature importance is **preserved** (notebook 2 only adds missing). Use `--clear-feature-importance` for full FI recompute.
# 3. **Optional manual clear**: EC2 NVMe and project pipeline output dirs (alternative to the script).
# 4. **Risk calculator infrastructure**: **API Gateway** (PGx Risk Calculator API) and **ECR** repo build/push for the Lambda image.
# 5. **Instructions for running the pipeline** (notebooks 1 → 5 in order).
#
# See [docs/README_checkpoints_and_workflow_resets.md](docs/README_checkpoints_and_workflow_resets.md) for what gets cleared and options (`--clear-feature-importance`, `--skip-checkpoints`, `--skip-s3`, `--skip-local`, `--yes`).

# %% [markdown]
# ## Python & R versions and dependencies
#
# Run this cell to print Python and R versions and to check that required packages are installed.

# %%
import sys
import subprocess
from pathlib import Path

# Project root (for Rscript cwd)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Python version ---
print("=== Python ===")
print(f"Version: {sys.version}")
print(f"Executable: {sys.executable}")

# --- Python dependencies (used by workflow notebooks and step scripts) ---
PYTHON_PACKAGES = [
    "boto3",
    "duckdb",
    "pandas",
    "numpy",
    "pyarrow",
    "sklearn",
    "joblib",
    "catboost",
    "xgboost",
    "matplotlib",
    "seaborn",
    "tenacity",
    "certifi",
]
print("\n=== Python packages ===")
ok = []
missing = []
for pkg in PYTHON_PACKAGES:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "?")
        ok.append((pkg, ver))
    except ImportError:
        missing.append(pkg)
for pkg, ver in ok:
    print(f"  {pkg}: {ver}")
for pkg in missing:
    print(f"  {pkg}: NOT INSTALLED")
if missing:
    print(f"\n  Install missing packages, e.g.: pip install {' '.join(missing)}")

# --- R version ---
print("\n=== R ===")
R_version = None
for cmd in [["R", "--version"], ["Rscript", "-e", "cat(R.version.string)"], ["Rscript", "-e", "cat(paste(R.version$major, R.version$minor, sep='.'))"]]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10, cwd=_PROJECT_ROOT)
        if r.returncode == 0 and (r.stdout or r.stderr):
            out = (r.stdout or r.stderr).strip()
            if "R version" in out or out.startswith("R version"):
                R_version = out
            elif R_version is None and out:
                R_version = f"R {out}"
            if R_version:
                break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        continue
if R_version:
    print(f"Version: {R_version}")
else:
    print("R not found on PATH (optional for pipeline; required for R-based feature importance / visualizations).")

# --- R dependencies (used by r_helpers and 3b BupaR) ---
R_PACKAGES = [
    "dplyr", "readr", "tidyr", "tibble", "purrr",
    "catboost", "randomForest", "rsample", "furrr", "future", "progressr",
    "duckdb", "DBI", "bupaR",
    "tidyverse", "ggplot2", "here",
]
if R_version:
    print("\n=== R packages ===")
    script = "for (p in c(" + ",".join(repr(p) for p in R_PACKAGES) + ")) { v <- tryCatch(packageVersion(p), error=function(e) NULL); cat(p, if(is.null(v)) 'NOT INSTALLED' else as.character(v), sep=': '); cat(\"\\n\") }"
    try:
        r = subprocess.run(
            ["Rscript", "-e", script],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=_PROJECT_ROOT,
        )
        if r.returncode == 0:
            for line in (r.stdout or "").strip().split("\n"):
                if line.strip():
                    print(f"  {line.strip()}")
        else:
            print("  (Rscript check failed:", (r.stderr or r.stdout or "")[:200], ")")
    except Exception as e:
        print(f"  Error checking R packages: {e}")

# %% [markdown]
# ## Docker check
#
# Run this cell to verify Docker is installed and the daemon is running. Required for building and pushing the risk calculator Lambda container image to ECR.

# %%
# Docker: version and daemon
import subprocess
import sys

def run_cmd(cmd, timeout=15):
    try:
        r = subprocess.run(
            cmd if isinstance(cmd, list) else cmd.split(),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=isinstance(cmd, str),
        )
        return r.returncode == 0, (r.stdout or "").strip(), (r.stderr or "").strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return False, "", str(e)

print("=== Docker ===")
ok_ver, out_ver, err_ver = run_cmd(["docker", "--version"])
if ok_ver and out_ver:
    print(f"Version: {out_ver}")
else:
    print("Docker not found or not on PATH. Install Docker Desktop (Windows/Mac) or docker.io (Linux).")

ok_info, out_info, err_info = run_cmd(["docker", "info"], timeout=10)
if ok_info:
    print("Daemon: running")
else:
    print("Daemon: not running or not accessible. Start Docker Desktop or the docker service.")
    if err_info and "Cannot connect" in err_info:
        print("  (Cannot connect to the Docker daemon)")

# %% [markdown]
# ## Setup

# %%
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.env_utils import get_data_root, is_linux

DATA_ROOT = get_data_root()

# Subdirs under DATA_ROOT (NVMe on EC2: /mnt/nvme) that MAY be cleared for a fresh run.
# NEVER cleared: gold/medical, gold/pharmacy (initial load only; ~80 GB + ~5 GB — do not delete).
NVME_SUBDIRS = [
    "gold/cohorts",
    "cohorts_staging",
    "4_model_data",  # model-training data (model_events.parquet) written by Step 4 on Linux
]

# Project-relative paths that hold pipeline outputs (clearing removes contents, not step source code)
PROJECT_OUTPUT_DIRS = [
    "2_create_cohort/cohort_metrics",
    "4_model_data/cohort_name=opioid_ed",
    "4_model_data/cohort_name=non_opioid_ed",
    "5_pgx_analysis/outputs",
    "6_final_model/outputs",
    "7_shap_analysis/outputs",
    "8_ffa_analysis/results",
    "10_risk_dashboard/outputs",
    "feature_encoding_outputs",
    "logs",
]

# Staging dir used by cohort phase4 when not on Linux (e.g. Windows: ~/cohorts_staging)
if is_linux():
    EXTRA_STAGING = DATA_ROOT / "cohorts_staging"  # already in NVME_SUBDIRS
else:
    EXTRA_STAGING = Path.home() / "cohorts_staging"

print(f"Project root: {PROJECT_ROOT}")
print(f"Data root (NVMe/local): {DATA_ROOT}")
print(f"NVMe subdirs to clear: {NVME_SUBDIRS}")
print(f"Project output dirs: {PROJECT_OUTPUT_DIRS}")

# %% [markdown]
# ## Full reset: run cleanup script
#
# The script **`utility_scripts/cleanup_cohort_data.sh`** clears checkpoints, S3 artifacts (pgxdatalake + pgx-repository), and EC2/local pipeline outputs so you can run the workflow from scratch. It does **not** delete gold medical/pharmacy tables.
#
# - **Options:** `--skip-checkpoints` (keep S3 checkpoints), `--skip-s3` (only local), `--skip-local` (only S3), `--yes` (skip confirmation).
# - **Details:** [docs/README_checkpoints_and_workflow_resets.md](docs/README_checkpoints_and_workflow_resets.md)
#
# Run the cell below to execute the script (with `--yes` for non-interactive). On Windows, run the same command in Git Bash or WSL from the project root.

# %%
# Full reset: run cleanup script (checkpoints + S3 + EC2/local)
# Set to True to run the script; False to only print the command
RUN_CLEANUP_SCRIPT = True

import subprocess
from pathlib import Path

_root = Path(PROJECT_ROOT) if "PROJECT_ROOT" in dir() and PROJECT_ROOT is not None else Path(__file__).resolve().parents[2]
if not (_root / "py_helpers").exists():
    _root = Path(__file__).resolve().parents[2]
_script = _root / "utility_scripts" / "cleanup_cohort_data.sh"
_cmd = ["bash", str(_script), "--yes"]  # --yes skips confirmation

if not _script.exists():
    print(f"Script not found: {_script}")
    print("From project root in a terminal, run: ./utility_scripts/cleanup_cohort_data.sh --yes")
elif not RUN_CLEANUP_SCRIPT:
    print("RUN_CLEANUP_SCRIPT is False; skipping. Set to True to run the cleanup script.")
    print(f"Or from project root in a terminal run:\n  {' '.join(_cmd)}")
else:
    print("Running cleanup script (checkpoints + S3 + EC2/local)...")
    r = subprocess.run(_cmd, cwd=str(_root), timeout=600)
    if r.returncode == 0:
        print("Cleanup script finished successfully.")
    else:
        print(f"Cleanup script exited with code {r.returncode}. Check output above.")

# %% [markdown]
# ## Cohort → model-training data flow (NVMe)
#
# On EC2, **DATA_ROOT** = `/mnt/nvme`. The pipeline uses these NVMe paths for cohort and model-training data:
#
# **Never cleared (initial load only):** `gold/medical` (~80 GB) and `gold/pharmacy` (~5 GB) are loaded once and must not be deleted by any clear operation. To verify gold pharmacy (or medical) is complete after sync, run:  
# `python 4_model_data/check_gold_pharmacy_completeness.py --medical`  
# (see `4_model_data/README_model_data.md`).
#
# 1. **Step 2 (Cohort creation)**  
#    - Writes cohort parquets to **local staging** (`cohorts_staging`), then syncs to S3:  
#      `s3://pgxdatalake/gold/cohorts/cohort_name={cohort}/event_year={year}/age_band={age_band}/cohort.parquet`  
#    - To use cohorts locally, sync from S3 into **DATA_ROOT/gold/cohorts** (same layout).
#
# 2. **Inputs for model-training data (Step 4)**  
#    - **Cohorts**: `DATA_ROOT/gold/cohorts/cohort_name={cohort}/event_year={year}/age_band={age_band}/cohort.parquet`  
#    - **Gold medical**: `DATA_ROOT/gold/medical/age_band={band}/event_year={year}/*.parquet`  
#    - **Gold pharmacy**: `DATA_ROOT/gold/pharmacy/age_band={band}/event_year={year}/*.parquet`  
#    - **Step 3b outputs** (refined feature importance): `PROJECT_ROOT/3b_feature_importance_eda/outputs/{cohort}/{age_band}/`
#
# 3. **Step 4 (Model data)**  
#    - Reads cohorts + gold medical/pharmacy + Step 3b feature lists.  
#    - **Writes** (on Linux) to **DATA_ROOT/4_model_data** (the **model-training** data):  
#      `DATA_ROOT/4_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet`  
#    - On Windows, Step 4 writes to `PROJECT_ROOT/4_model_data` instead.
#
# 4. **Ingestion into final model training (Step 6)**  
#    - Step 6 (and prepare_train_test_s3) read **model_events.parquet** from:  
#      - `PROJECT_ROOT/4_model_data/...` or  
#      - `DATA_ROOT/4_model_data/...` (when on Linux).  
#    - Train/test splits are then written to `6_final_model/outputs` and optionally to S3 `gold/final_model/`.
#
# So on NVMe, **4_model_data** is the model-training folder: it holds the event-level data ingested by final model training (Step 6). Clearing NVMe (including `4_model_data`) forces Step 4 and downstream steps to regenerate that data.

# %%
def dir_size(p: Path) -> int:
    if not p.exists() or not p.is_dir():
        return 0
    total = 0
    try:
        for f in p.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except Exception:
        pass
    return total

def fmt_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"

print("=== Data root (NVMe) subdirs ===")
for sub in NVME_SUBDIRS:
    d = DATA_ROOT / sub
    if d.exists():
        n = dir_size(d)
        print(f"  {d}  ({fmt_size(n)})")
    else:
        print(f"  {d}  (absent)")

if not is_linux() and EXTRA_STAGING.exists():
    n = dir_size(EXTRA_STAGING)
    print(f"  {EXTRA_STAGING}  ({fmt_size(n)}) [staging on non-Linux]")

print("\n=== Project output dirs ===")
for sub in PROJECT_OUTPUT_DIRS:
    d = PROJECT_ROOT / sub
    if d.exists():
        n = dir_size(d)
        print(f"  {d}  ({fmt_size(n)})")
    else:
        print(f"  {d}  (absent)")

# %% [markdown]
# ## Clear EC2 NVMe / data root (optional manual clear)
#
# Removes contents of each `DATA_ROOT` subdir in **NVME_SUBDIRS** (gold/cohorts, cohorts_staging, 4_model_data). **Does not** remove `gold/medical` or `gold/pharmacy` — those are never cleared. For a **full reset** (including checkpoints and S3), use the **cleanup script** above instead. Set `DO_CLEAR_NVME = True` to perform the deletion here.

# %%
DO_CLEAR_NVME = True  # Set to True to actually delete

if not DO_CLEAR_NVME:
    print("DO_CLEAR_NVME is False; skipping. Set to True to clear data root subdirs.")
else:
    for sub in NVME_SUBDIRS:
        d = DATA_ROOT / sub
        if d.exists():
            try:
                shutil.rmtree(d)
                print(f"Removed: {d}")
            except Exception as e:
                print(f"Error removing {d}: {e}")
        else:
            print(f"Absent (skip): {d}")
    if not is_linux() and EXTRA_STAGING.exists():
        try:
            shutil.rmtree(EXTRA_STAGING)
            print(f"Removed: {EXTRA_STAGING}")
        except Exception as e:
            print(f"Error removing {EXTRA_STAGING}: {e}")
    print("Done clearing data root.")

# %% [markdown]
# ## Clear project pipeline output directories
#
# Removes contents of the project output dirs listed above (e.g. `4_model_data` partitions, `6_final_model/outputs`, **logs**). Set `DO_CLEAR_PROJECT = True` to perform the deletion.

# %%
DO_CLEAR_PROJECT = True  # Set to True to actually delete

if not DO_CLEAR_PROJECT:
    print("DO_CLEAR_PROJECT is False; skipping. Set to True to clear project output dirs.")
else:
    for sub in PROJECT_OUTPUT_DIRS:
        d = PROJECT_ROOT / sub
        if d.exists():
            try:
                shutil.rmtree(d)
                print(f"Removed: {d}")
            except Exception as e:
                print(f"Error removing {d}: {e}")
        else:
            print(f"Absent (skip): {d}")
    print("Done clearing project outputs.")

# %% [markdown]
# ## S3 checkpoints (optional: list or clear from notebook)
#
# The **cleanup script** (run above without `--skip-checkpoints`) already clears `s3://pgx-repository/pipeline_checkpoints/` and `pgx-pipeline-status/`. The cell below lets you **list** checkpoint objects or **delete** only checkpoints from this notebook (e.g. if you ran the script with `--skip-checkpoints`). Set `DO_CLEAR_S3_CHECKPOINTS = True` to delete. Requires `boto3` and AWS credentials with delete access to the `pgx-repository` bucket.

# %% [markdown]
# ---
#
# # How to run the pipeline
#
# The pipeline has **10 phases** (steps) and is run via **six notebooks** in order: 0_config_and_pipeline, 1_cohort_workflow, 2_feature_importance, 3_model_train_shap_ffa, 4_dashboard_visuals, 5_build_and_deploy. Each notebook syncs required inputs from S3 to local/NVMe and uses S3 checkpoints so completed steps are skipped.
#
# ## Prerequisites
#
# - **Python** and project dependencies: `pip install -r requirements.txt`
# - **AWS credentials** configured (e.g. `aws configure` or instance profile on EC2) for S3 and, if used, `pgx-repository` checkpoints.
# - **Data root**: On EC2, NVMe is used (`/mnt/nvme`). On Windows, data root is `%USERPROFILE%\pgx_data`. Override with `PGX_DATA_ROOT` if needed.
#
# ## Order of execution
#
# 1. **1_cohort_workflow.ipynb** (Steps 1–2)  
#    - Step 1a: APCD input data (bronze → silver → gold). Often run via scripts in `1a_apcd_input_data/`.
#    - Step 1b: Event filter (ICD/administrative codes) in `1b_apcd_event_filter/`.
#    - Step 2: Cohort creation (5:1 target:control) in `2_create_cohort/`.
#
# 2. **2_feature_importance.ipynb** (Steps 3a–3b)  
#    - Step 3a: Feature importance (MC-CV, CatBoost/XGBoost) in `3a_feature_importance/`.
#    - Step 3b: Feature Importance EDA (BupaR, code research) in `3b_feature_importance_eda/`; produces refined `cohort_feature_importance.csv`.
#
# 3. **3_model_train_shap_ffa.ipynb** → **4_dashboard_visuals.ipynb** → **5_build_and_deploy.ipynb** (model train + SHAP/FFA → visuals → build and deploy)  
#    - Step 4: Model data (`4_model_data/`) → `model_events.parquet`.
#    - Step 5: PGx feature engineering (`5_pgx_analysis/`).
#    - Step 6: Final model training and selection (`6_final_model/`).
#    - Step 7: SHAP analysis (`7_shap_analysis/`).
#    - Step 8: FFA analysis (`8_ffa_analysis/`).
#    - Step 9: Risk dashboard artifacts (SHAP/CPIC, etc. in `10_risk_dashboard/`).
#    - **Step 10:** Dashboard visuals (BupaR, DTW, FP-Growth) — **4_dashboard_visuals.ipynb**; see `9_dashboard_visuals/README.md`.
#
# ## Cohorts and age bands
#
# - **opioid_ed**: 13-24, 25-44, 45-54, 55-64  
# - **non_opioid_ed**: 65-74, 75-84, 85-94  
#
# Run the notebooks in order; later notebooks depend on outputs from earlier ones. Steps are idempotent: completed steps are skipped when S3 checkpoints exist.

# %%
# S3 checkpoint bucket/prefix used by py_helpers.checkpoint_utils (workflow notebooks)
S3_CHECKPOINT_BUCKET = "pgx-repository"
S3_CHECKPOINT_PREFIX = "pipeline_checkpoints/"

DO_CLEAR_S3_CHECKPOINTS = True  # Set to True to actually delete

try:
    import boto3
    s3 = boto3.client("s3")
except Exception as e:
    print(f"boto3 not available: {e}. Install with: pip install boto3")
    s3 = None

if s3 is None:
    print("Skipping S3 checkpoint clear (no boto3).")
else:
    # List all objects under the checkpoint prefix
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_CHECKPOINT_BUCKET, Prefix=S3_CHECKPOINT_PREFIX):
        for obj in page.get("Contents", []):
            keys.append({"Key": obj["Key"]})
    print(f"Found {len(keys)} checkpoint object(s) under s3://{S3_CHECKPOINT_BUCKET}/{S3_CHECKPOINT_PREFIX}")
    if keys:
        for i, k in enumerate(keys[:20]):
            print(f"  {k['Key']}")
        if len(keys) > 20:
            print(f"  ... and {len(keys) - 20} more")
    if not DO_CLEAR_S3_CHECKPOINTS:
        print("\nDO_CLEAR_S3_CHECKPOINTS is False; skipping delete. Set to True to clear S3 checkpoints.")
    else:
        deleted = 0
        for i in range(0, len(keys), 1000):
            chunk = keys[i : i + 1000]
            s3.delete_objects(Bucket=S3_CHECKPOINT_BUCKET, Delete={"Objects": chunk})
            deleted += len(chunk)
        print(f"\nDeleted {deleted} checkpoint object(s). Done clearing S3 checkpoints.")

# %% [markdown]
# ## Risk calculator: Docker, ECR & API Gateway
#
# Infrastructure for the **PGx Risk Calculator API** (Step 9):
#
# - **Dashboard (frontend) S3:** `s3://jerome-dixon.io/vcu/pgx-risk-calculator/` — hosted frontend and static assets.
# - **ECR:** Repository `pgx-risk-dashboard` — container image for the Lambda (models + CPIC data). URI: `535362115856.dkr.ecr.us-east-1.amazonaws.com/pgx-risk-dashboard`. Build from `10_risk_dashboard/` and push to ECR, then create/update a Lambda function using that image.
# - **API Gateway** (REST, Edge-optimized): `pgx-calculator-api` — API id `4ynsaq4dyk`. Exposes the calculator as an HTTP API; add resource (e.g. `/predict`), POST method, and Lambda integration in the console.
#
# **Create ECR + API (if not already created):** Run from project root (WSL or Git Bash):
#
# ```bash
# bash utility_scripts/create_ecr_and_apigateway.sh --profile mushin
# ```
#
# - **On EC2:** No profile or credentials file; script and notebook use the **instance role**.
# - **Off EC2:** Default profile `mushin`; shared credentials at `C:\Projects\credentials` when that file exists (override with `--profile` or `AWS_PROFILE`).
# - **Region:** `us-east-1` (or set `AWS_REGION`).
#
# Run the cell below to **check** Docker, **verify** the ECR repository and API Gateway, and **print** build/push commands for the Lambda image.

# %%
# Risk calculator: Docker check, ECR repo, and API Gateway / build commands
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_ROOT = PROJECT_ROOT / "10_risk_dashboard"

AWS_REGION = "us-east-1"
ECR_REPOSITORY = "pgx-risk-dashboard"
API_NAME = "pgx-calculator-api"
API_DESCRIPTION = "PGx Risk Calculator API"
API_ID = "4ynsaq4dyk"  # Created by utility_scripts/create_ecr_and_apigateway.sh
ECR_URI = "535362115856.dkr.ecr.us-east-1.amazonaws.com/pgx-risk-dashboard"
# On EC2: no profile or credentials file (instance role). Off EC2: profile mushin + C:\Projects\credentials if present.
creds = PROJECT_ROOT.parent / "credentials"
if "AWS_SHARED_CREDENTIALS_FILE" not in os.environ and creds.exists():
    os.environ["AWS_SHARED_CREDENTIALS_FILE"] = str(creds)
AWS_PROFILE = "mushin" if creds.exists() else ""  # empty on EC2 → boto3 uses instance role

# Dashboard frontend (static site)
DASHBOARD_S3_URI = "s3://jerome-dixon.io/vcu/pgx-risk-calculator/"

def run_cmd(cmd, timeout=15):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=False)
        return r.returncode == 0, (r.stdout or "").strip(), (r.stderr or "").strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, "", ""

# --- Dashboard S3 ---
print("=== Risk calculator dashboard (frontend) ===")
print(f"  S3: {DASHBOARD_S3_URI}")
print("  Deploy: aws s3 sync 10_risk_dashboard/frontend/ s3://jerome-dixon.io/vcu/pgx-risk-calculator/")

# --- Docker check ---
print("\n=== Docker (for ECR build) ===")
ok, out, err = run_cmd(["docker", "info"])
if ok:
    print("  Docker daemon: running")
else:
    print("  Docker daemon: not running. Start Docker before building the Lambda image.")

# --- ECR repo (verify or create) ---
print("\n=== ECR repository ===")
print(f"  URI: {ECR_URI}")
try:
    import boto3
    session = boto3.Session(profile_name=AWS_PROFILE) if AWS_PROFILE else boto3.Session()
    ecr = session.client("ecr", region_name=AWS_REGION)
    ecr.describe_repositories(repositoryNames=[ECR_REPOSITORY])
    print(f"  Repository exists: {ECR_REPOSITORY}")
except Exception as e:
    if "RepositoryNotFoundException" in str(type(e).__name__) or "not found" in str(e).lower():
        print(f"  Repository {ECR_REPOSITORY} not found. Create with:")
        print(f"    bash utility_scripts/create_ecr_and_apigateway.sh --profile {AWS_PROFILE}")
    else:
        print(f"  Error: {e}")

# --- API Gateway ---
print("\n=== API Gateway ===")
print(f"  API: {API_NAME} (id: {API_ID}, Edge-optimized)")
print("  Console: add resource (e.g. /predict), POST method, integrate with Lambda (container image from ECR).")
print("  To create if missing: bash utility_scripts/create_ecr_and_apigateway.sh --profile", AWS_PROFILE)

# --- Build & push (reference) ---
print("\n=== Build and push Lambda image ===")
print(f"  ECR URI: {ECR_URI}")
print(f"  From project root: cd 10_risk_dashboard/deployment && ./docker_build.sh")
print(f"  Or set: AWS_REGION={AWS_REGION}, ECR_REPOSITORY={ECR_REPOSITORY}, AWS_ACCOUNT_ID=535362115856")

# %%
