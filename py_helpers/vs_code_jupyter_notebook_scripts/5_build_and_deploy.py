# -*- coding: utf-8 -*-
# Auto-generated from 5_build_and_deploy.ipynb (VS Code Python notebook script format)

# %% [markdown]
# # 5. Build and deploy
#
#
#
# **Purpose:** Prepare Lambda directory, build Docker image, push to ECR, update Lambda, and sync dashboard assets to S3 (frontend, metadata, feature importance, Cohort PGx, and causal dashboard JSON for the Causal Analysis tab). Run once after [3_model_train_shap_ffa.ipynb](3_model_train_shap_ffa.ipynb) and [4_dashboard_visuals.ipynb](4_dashboard_visuals.ipynb).
#
#
#
# **Prerequisites:** Notebook 3 (models, metadata, SHAP/FFA combined) and notebook 4 (dashboard visuals) completed. Run from repo root.

# %%
# Setup: paths and project root
import sys
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))
from py_helpers.env_utils import get_data_root, get_workflow_python_bin
from py_helpers.workflow_sync_checkpoint import sync_s3_to_local, check_step_checkpoint_exists, save_step_checkpoint
PYTHON_BIN = get_workflow_python_bin()  # EC2: jupyter-env; local: sys.executable

DASHBOARD_DIR = PROJECT_ROOT / "10_risk_dashboard"
DATA_PREP_DIR = DASHBOARD_DIR / "data_preparation"
DEPLOY_DIR = DASHBOARD_DIR / "deployment"
S3_BUCKET = os.environ.get("PGX_S3_BUCKET", "pgxdatalake")
DATA_ROOT = get_data_root()
AWS_PROFILE = os.environ.get("AWS_PROFILE")

print("PGx Risk Calculator Workflow")
print("=" * 60)
print(f"Project root: {PROJECT_ROOT}")
print(f"Dashboard dir: {DASHBOARD_DIR}")
print(f"Data prep: {DATA_PREP_DIR}")
print(f"Data root (NVMe/local): {DATA_ROOT}")
print("=" * 60)

def _confirm(prompt: str, default: bool = False) -> bool:
    """Return True if user confirms. Non-interactive safe by default.

    - Set env PGX_ASSUME_YES=1 to auto-confirm.
    - If stdin is not a TTY, returns `default`.
    """
    if (os.environ.get("PGX_ASSUME_YES") or "").strip().lower() in ("1", "true", "yes", "y"):
        return True
    try:
        if not sys.stdin.isatty():
            return default
    except Exception:
        return default
    resp = input(f"{prompt} (y/n): ").strip().lower()
    return resp == "y"

# %%
# Configuration: PGx cohorts and age bands (each cohort has all age bands; from py_helpers.constants)
from py_helpers.constants import REQUIRED_COHORTS

# Input dirs (required for pipeline Step 4–6)
# Cohorts: Step 2 cohort.parquet files (create_model_data reads case/control and target dates from here).
COHORTS_ROOT = DATA_ROOT / "gold" / "cohorts"
# Feature importance: Step 3/3b outputs — cohort_feature_importance.csv and feature_filtering_summary.json per cohort/age_band.
FI_ROOT = DATA_ROOT / "gold" / "feature_importance"
STEP3_OUTPUTS = STEP3B_OUTPUTS = FI_ROOT
# Model data: single canonical location (Step 4 output, Step 5/6 input).
from py_helpers.env_utils import get_model_data_root
MODEL_DATA_ROOT = get_model_data_root()

# Output dirs (Step 6 final model outputs; data prep and Lambda read from these)
FINAL_MODEL_OUTPUTS = PROJECT_ROOT / "6_final_model" / "outputs"
FINAL_MODEL_OUTPUTS_ALT = DATA_ROOT / "6_final_model" / "outputs"
FINAL_MODEL_GOLD = DATA_ROOT / "gold" / "final_model"  # S3 layout: cohort/13-24/*.joblib

print("Cohorts and age bands:")
for cohort, bands in REQUIRED_COHORTS.items():
    print(f"  {cohort}: {bands}")

print("\nInput dirs (for Step 4–6):")
print(f"  Cohorts (2):               {COHORTS_ROOT}")
print(f"  Feature importance (3/3b): {FI_ROOT}  (CSVs + feature_filtering_summary.json)")
print(f"  Model data (4; in/out):    {MODEL_DATA_ROOT}")

print("\nOutput dirs (Step 6):")
print(f"  Project:    {FINAL_MODEL_OUTPUTS}")
print(f"  NVMe:       {FINAL_MODEL_OUTPUTS_ALT}")
print(f"  gold/NVMe:  {FINAL_MODEL_GOLD}")

# %% [markdown]
# ## Step 3: Build and deploy risk calculator
#
# Build the Docker image and push to ECR; then update API Gateway/Lambda. Use the deployment script in `10_risk_dashboard/deployment`.
#
# **Order:** Run [4_dashboard_visuals.ipynb](4_dashboard_visuals.ipynb) before this notebook so BupaR, DTW, FP-Growth artifacts exist in S3 before deploy.

# %% [markdown]
# ### AWS infrastructure configuration
#
# Current deployment (run this cell so Verify infrastructure and Build/deploy use these values):
#
# - **Region:** us-east-1  
# - **Account:** 535362115856  
# - **ECR:** pgx-risk-calculator  
# - **Lambda:** pgx-risk-calculator (role: pgx-lambda-role)  
# - **API Gateway:** pgx-risk-calculator (id: cmv0qislq3)  
# - **Invoke URL:** https://cmv0qislq3.execute-api.us-east-1.amazonaws.com/prod  
# - **S3 dashboard:** s3://jerome-dixon.io/vcu/pgx-risk-calculator/ (Lambda has write; upload HTML here)  
# - **Lambda env:** PGX_RESULTS_BUCKET=pgxdatalake, MODEL_CACHE_TTL=3600

# %%
# AWS infrastructure (set env so Verify infrastructure and Build/deploy use these)
import os
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID", "535362115856")
ECR_REPOSITORY = os.environ.get("ECR_REPOSITORY", "pgx-risk-calculator")
ECR_URI = f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPOSITORY}:latest"
LAMBDA_FUNCTION_NAME = os.environ.get("LAMBDA_FUNCTION_NAME", "pgx-risk-calculator")
LAMBDA_ROLE_NAME = os.environ.get("LAMBDA_ROLE_NAME", "pgx-lambda-role")
LAMBDA_ROLE_ARN = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/{LAMBDA_ROLE_NAME}"
PGX_API_GATEWAY_NAME = os.environ.get("PGX_API_GATEWAY_NAME", "pgx-risk-calculator")
API_GATEWAY_ID = os.environ.get("API_GATEWAY_ID", "cmv0qislq3")
API_INVOKE_URL = f"https://{API_GATEWAY_ID}.execute-api.{AWS_REGION}.amazonaws.com/prod"
S3_DASHBOARD_BUCKET = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
S3_DASHBOARD_PREFIX = os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator")
S3_DASHBOARD_PATH = f"s3://{S3_DASHBOARD_BUCKET}/{S3_DASHBOARD_PREFIX}/"

os.environ["AWS_REGION"] = AWS_REGION
os.environ["AWS_ACCOUNT_ID"] = str(AWS_ACCOUNT_ID)
os.environ["ECR_REPOSITORY"] = ECR_REPOSITORY
os.environ["LAMBDA_FUNCTION_NAME"] = LAMBDA_FUNCTION_NAME
os.environ["LAMBDA_ROLE_NAME"] = LAMBDA_ROLE_NAME
os.environ["PGX_API_GATEWAY_NAME"] = PGX_API_GATEWAY_NAME
os.environ["API_GATEWAY_ID"] = API_GATEWAY_ID
os.environ["S3_DASHBOARD_BUCKET"] = S3_DASHBOARD_BUCKET
os.environ["S3_DASHBOARD_PREFIX"] = S3_DASHBOARD_PREFIX

print(f"Region: {AWS_REGION}  Account: {AWS_ACCOUNT_ID}")
print(f"ECR: {ECR_URI}")
print(f"Lambda: {LAMBDA_FUNCTION_NAME}  Role: {LAMBDA_ROLE_NAME}")
print(f"API: {PGX_API_GATEWAY_NAME} (id: {API_GATEWAY_ID})")
print(f"Invoke URL: {API_INVOKE_URL}")
print(f"S3 dashboard: {S3_DASHBOARD_PATH}")

# %% [markdown]
# ### Verify infrastructure (Docker, ECR, API Gateway, Lambda)
#
# Verifies Docker, ECR repo, API Gateway, and Lambda function. Run the AWS infrastructure configuration cell first so names/IDs match our deployment.

# %%
# Docker, ECR, API Gateway checks for PGx dashboard
import subprocess
import os

print(f"\n{'=' * 80}")
print("Verify infrastructure (Docker, ECR, API Gateway)")
print(f"{'=' * 80}\n")

# 1. Docker
print("1. Docker")
print("-" * 40)
try:
    r = subprocess.run(["docker", "ps"], capture_output=True, text=True, timeout=5)
    if r.returncode == 0:
        print("✓ Docker is running")
    else:
        print("⚠ Docker not accessible:", r.stderr.strip() or r.stdout.strip())
        if "permission denied" in (r.stderr or "").lower():
            print("  Linux: sudo usermod -aG docker $USER && newgrp docker")
except FileNotFoundError:
    print("✗ Docker not found — install Docker first")
except Exception as e:
    print(f"⚠ Error: {e}")
print()

# 2. AWS / ECR
print("2. ECR (AWS credentials + repository)")
print("-" * 40)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
ECR_REPOSITORY = os.environ.get("ECR_REPOSITORY", "pgx-risk-calculator")
try:
    # Get caller identity (proves credentials work)
    r = subprocess.run(
        ["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"],
        capture_output=True, text=True, timeout=10
    )
    if r.returncode != 0:
        print("⚠ AWS CLI not configured or no credentials:", (r.stderr or r.stdout or "").strip())
    else:
        account = r.stdout.strip()
        print(f"✓ AWS identity: account {account}")

    # ECR repository exists?
    r2 = subprocess.run(
        ["aws", "ecr", "describe-repositories", "--repository-names", ECR_REPOSITORY, "--region", AWS_REGION],
        capture_output=True, text=True, timeout=10
    )
    if r2.returncode == 0:
        print(f"✓ ECR repository exists: {ECR_REPOSITORY}")
    else:
        print(f"⚠ ECR repository '{ECR_REPOSITORY}' not found (create it or set ECR_REPOSITORY)")
        print("  docker_build.sh can create it automatically on first push")
except FileNotFoundError:
    print("✗ AWS CLI not found")
except Exception as e:
    print(f"⚠ Error: {e}")
print()

# 3. API Gateway
print("3. API Gateway")
print("-" * 40)
print("Note: API Gateway should already be set up; this step only verifies.")
API_NAME = os.environ.get("PGX_API_GATEWAY_NAME", "pgx-risk-calculator")
try:
    r = subprocess.run(
        [
            "aws", "apigateway", "get-rest-apis",
            "--region", AWS_REGION,
            "--query", f"items[?name=='{API_NAME}'].id",
            "--output", "text",
        ],
        capture_output=True, text=True, timeout=10
    )
    if r.returncode == 0 and r.stdout.strip():
        api_id = r.stdout.strip().split()[0]
        api_url = f"https://{api_id}.execute-api.{AWS_REGION}.amazonaws.com/prod"
        print(f"✓ API Gateway found: {API_NAME}")
        print(f"  API ID: {api_id}")
        print(f"  Base URL: {api_url}")

        # Optional: test endpoint (e.g. GET /metadata)
        try:
            import urllib.request
            import urllib.error
            import ssl

            req = urllib.request.Request(f"{api_url}/metadata", method="GET")
            ctx = ssl.create_default_context()

            try:
                with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                    code = resp.getcode()
            except urllib.error.URLError as ue:
                if "CERTIFICATE_VERIFY_FAILED" in str(getattr(ue, "reason", "")) or "SSL" in str(getattr(ue, "reason", "")):
                    ctx = ssl._create_unverified_context()
                    with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                        code = resp.getcode()
                else:
                    raise

            if code in (200, 301, 302):
                print("  ✓ API is responding")
            else:
                print(f"  ⚠ API returned HTTP {code}")

        except Exception as e:
            if hasattr(e, "code"):
                print(f"  ⚠ API endpoint test: HTTP {e.code} (Lambda may need deployment)")
            else:
                err = getattr(e, "status", None) or str(e)[:80]
                print(f"  ⚠ API endpoint test inconclusive ({err})")
    else:
        print(f"⚠ API Gateway '{API_NAME}' not found or AWS CLI not configured")
        print("  Set up API Gateway and link to Lambda, then re-run this check.")
except Exception as e:
    print(f"⚠ Could not verify API Gateway: {e}")
    print("  Assuming API Gateway is already set up or will be configured after deploy.")
print()

# 4. Lambda
print("4. Lambda")
print("-" * 40)
LAMBDA_FUNCTION_NAME = os.environ.get("LAMBDA_FUNCTION_NAME", "pgx-risk-calculator")
try:
    r = subprocess.run(
        ["aws", "lambda", "get-function", "--function-name", LAMBDA_FUNCTION_NAME, "--region", AWS_REGION],
        capture_output=True, text=True, timeout=10
    )
    if r.returncode == 0:
        print(f"✓ Lambda function exists: {LAMBDA_FUNCTION_NAME}")
    else:
        print(f"⚠ Lambda '{LAMBDA_FUNCTION_NAME}' not found (create via Console/CLI or workflow)")
except Exception as e:
    print(f"⚠ Could not verify Lambda: {e}")

print(f"\n{'=' * 80}")

# %% [markdown]
# ### Prepare Models
#
# Step 6 (notebook 3 or `6_final_model/run_final_model.py`) saves models to `6_final_model/outputs/{cohort}/{age_band}/models/`. This step runs `prepare_models.py`, which reads from there and writes to `10_risk_dashboard/outputs/models/` so `prepare_lambda_dir.py` and Docker can find them. **Checkpoint:** step is skipped if S3 checkpoint exists.

# %%
import logging
logger = logging.getLogger(__name__)

# Set to True to run prepare_models regardless of checkpoint; False to skip when checkpoint exists
force = False

# Set to True to upload prepared dashboard models to S3 (s3://pgxdatalake/gold/dashboard/models)
upload_s3 = True

if not force and check_step_checkpoint_exists("9_dashboard_models", "all", "all", logger):
    print("Step 3 (prepare models) already completed (checkpoint exists). Skipping.")
else:
    # -u: unbuffered stdout so progress appears in notebook
    cmd = [str(PYTHON_BIN), "-u", "prepare_models.py", "--all"]
    if upload_s3:
        cmd.append("--upload-s3")
    if force:
        cmd.append("--force")
    r = subprocess.run(cmd, cwd=DATA_PREP_DIR)
    if r.returncode == 0:
        save_step_checkpoint("9_dashboard_models", "all", "all", logger=logger)
    if r.returncode != 0:
        raise SystemExit(r.returncode)

# %% [markdown]
# ### Prepare CPIC Data

# %%
import logging
logger = logging.getLogger(__name__)
if check_step_checkpoint_exists("9_dashboard_cpic", "all", "all", logger):
    print("Step 3 (prepare cpic data) already completed (checkpoint exists). Skipping.")
else:
    r = subprocess.run([str(PYTHON_BIN), "prepare_cpic_data.py", "--all"], cwd=DATA_PREP_DIR)
    if r.returncode == 0:
        save_step_checkpoint("9_dashboard_cpic", "all", "all", logger=logger)
    if r.returncode != 0:
        raise SystemExit(r.returncode)

# %% [markdown]
# ### Prepare Lambda directory
#
# Assemble `lambda_dir` under `10_risk_dashboard` for Docker build (models, metadata, CPIC data). **Prerequisites:** Run Step 1a (metadata), Step 3 (prepare models), and `prepare_cpic_data.py` (in data_preparation) so `outputs/models`, `outputs/metadata`, and `outputs/cpic` exist. The first cell copies those into `lambda_dir`; the second verifies.

# %%
# Prepare first: copy outputs/models, outputs/metadata, outputs/cpic into lambda_dir (required before verify)
r = subprocess.run([str(PYTHON_BIN), "prepare_lambda_dir.py"], cwd=DEPLOY_DIR)
if r.returncode != 0:
    raise SystemExit(r.returncode)

# %%
# Verify lambda_dir has models/, metadata/, data/ (run after prepare cell above)
subprocess.run([str(PYTHON_BIN), "prepare_lambda_dir.py", "--verify-only"], cwd=DEPLOY_DIR, check=True)
print("Lambda directory prepared.")

# %% [markdown]
# ### Docker Build

# %%
import subprocess

# Variables for Docker build (DASHBOARD_DIR from setup cell)
risk_dashboard_dir = DASHBOARD_DIR
needs_prepare = True  # Set True to prompt for build; False to skip rebuild

print(f"\n{'=' * 80}")
print("Step 3: Build and Push Docker Image")
print(f"{'=' * 80}")

docker_script = risk_dashboard_dir / "deployment" / "docker_build.sh"
needs_docker_build = needs_prepare

if docker_script.exists():
    print("\nDocker build strategy:")
    print(f"  - Lambda directory was {'updated' if needs_prepare else 'unchanged'}")
    print(f"  - Docker image will be {'built' if needs_docker_build else 'skipped (use --force to rebuild)'}")
    print("-" * 80)

    print("\nThis will:")
    print("  1. Build Docker image with models and dependencies")
    print("  2. Push image to AWS ECR (Elastic Container Registry)")
    print("-" * 80)

    print("\nNote: This requires:")
    print("  - Docker installed and running")
    print("  - AWS CLI configured with ECR permissions")
    print("  - AWS credentials with push access to ECR")
    print("-" * 80)

    if needs_docker_build:
        response = "y" if _confirm("Proceed with Docker build?", default=False) else "n"
    else:
        print("\nSkipping Docker build (Lambda directory unchanged)")
        print("  To force rebuild, run: ./deployment/docker_build.sh")
        response = "n"

    if response == "y":
        try:
            result = subprocess.run(
                ["bash", str(docker_script)],
                cwd=str(risk_dashboard_dir),
                capture_output=False,
                text=True
            )

            if result.returncode == 0:
                print("\nDocker image built and pushed successfully!")
                print("  Next: Get ECR URI from output above and use it to update Lambda")
            else:
                print(f"\nDocker build exited with code: {result.returncode}")
        except Exception as e:
            print(f"\nError building Docker image: {e}")
            # Don't hard-fail if logger isn't defined in this notebook
            if "logger" in globals():
                logger.error("Error building Docker image", exc_info=True)
else:
    print(f"\nDocker build script not found: {docker_script}")
    print("  Expected location: 10_risk_dashboard/deployment/docker_build.sh")

print(f"\n{'=' * 80}")

# %% [markdown]
# ### Check dashboard artifact paths (before Lambda / Step 6)
#
# Verify that EC2 paths and artifacts from **README_dashboard_visual_artifact_paths.md** (and RESEARCH_QUESTIONS_ARTIFACTS.md) are present before **updating Lambda** or **syncing S3**. Run the cell below; it checks Feature Importance, Causal, BupaR, DTW, FP-Growth, PGx Cohort, and metadata/frontend. This single check covers frontend, metadata, FI, Causal, BupaR, DTW, FP-Growth, and PGx (replacing the former "Dashboard requirement check"). See **TEST_PLAN_FINAL_DASHBOARD.md** Section 1.3 & 7. **Step 6** uploads causal JSON from `10_risk_dashboard/visualizations/causal/{cohort}/{age_band_fname}/dashboard_data.json` (generate via notebook 3 or combine_shap_ffa_results). Use `--strict` so the notebook fails if any required path is missing.

# %%
# Check dashboard artifact paths before Lambda update / Step 6 sync
# See 10_risk_dashboard/docs/README_dashboard_visual_artifact_paths.md
check_script = DASHBOARD_DIR / "data_preparation" / "check_dashboard_artifact_paths.py"
if check_script.exists():
    result = subprocess.run(
        [str(PYTHON_BIN), str(check_script), "--project-root", str(PROJECT_ROOT), "--strict"],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        print("\n⚠ Artifact path check failed. Fix missing paths before Step 6 or Lambda deploy.")
else:
    print(f"Script not found: {check_script}")
    print("  Run: python 10_risk_dashboard/data_preparation/check_dashboard_artifact_paths.py --project-root <repo_root>")

# %% [markdown]
# ### Dashboard visual objects checklist (from notebook 4) with S3 paths
#
# Notebook 4 writes `10_risk_dashboard/visualizations/dashboard_visual_objects.json` listing all dashboard visuals (path, s3_path, tab, notes). The table below shows the same checklist with **S3 destination paths**. Items marked **Step 6** are synced/uploaded in the "Step 6: Sync dashboard frontend and assets to S3" cell; items marked **Notebook 4** are uploaded by the pipeline when you run notebook 4 (BupaR, DTW, FP-Growth scripts upload to the dashboard bucket directly).

# %%
# Load dashboard visual objects from notebook 4 manifest (single source of truth)
import json
from pathlib import Path

manifest_path = DASHBOARD_DIR / "visualizations" / "dashboard_visual_objects.json"
s3_bucket = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")

print("Dashboard visual objects checklist (from manifest)")
print("=" * 100)
if manifest_path.exists():
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    objects = data.get("visual_objects", [])
    for i, obj in enumerate(objects, 1):
        s3_key = obj.get("s3_path", "")
        s3_display = f"s3://{s3_bucket}/{s3_key}" if s3_key else "(no s3_path in manifest)"
        print(f"{i}. {obj.get('visual_name', '')}")
        print(f"   Tab:          {obj.get('dashboard_tab', '')}")
        print(f"   Local path:   {obj.get('path', '')}")
        print(f"   S3 path:      {s3_display}")
        if obj.get("notes"):
            print(f"   Notes:        {obj['notes']}")
        print()
    print("=" * 100)
    print(f"Manifest: {manifest_path}")
    print("Run Step 6 below to sync frontend, metadata, FI heatmaps/JSON, Cohort PGx, and causal data to S3.")
else:
    print(f"Manifest not found: {manifest_path}")
    print("Run notebook 4 (4_dashboard_visuals.ipynb) and execute the cell that writes dashboard_visual_objects.json, then re-run this cell.")

# %% [markdown]
# ### Update Lambda

# %%
import subprocess

# Step 4: Update Lambda Function (Idempotent - only if Docker image was updated)
print(f"\n{'=' * 80}")
print("Step 4: Update Lambda Function")
print(f"{'=' * 80}")

lambda_function_name = "pgx-risk-calculator"
region = "us-east-1"

# Check if Lambda function exists
lambda_exists = False
try:
    result = subprocess.run(
        ["aws", "lambda", "get-function", "--function-name", lambda_function_name, "--region", region],
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0:
        lambda_exists = True
        print(f"\n✓ Lambda function exists: {lambda_function_name}")
    else:
        # If it failed, show a hint (often permissions or wrong name/region)
        msg = (result.stderr or result.stdout or "").strip()
        if msg:
            print(f"\n⚠ Could not confirm Lambda exists: {msg}")
except FileNotFoundError:
    print("\n⚠ AWS CLI not found (cannot check Lambda function status)")
except Exception as e:
    print("\n⚠ Could not check Lambda function status")
    print(f"  {e}")

# Get AWS account ID and ECR URI (for update)
account_id = None
ecr_uri = None
try:
    result = subprocess.run(
        ["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"],
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0:
        account_id = result.stdout.strip()
        ecr_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/pgx-risk-calculator:latest"
        print(f"  AWS Account ID: {account_id}")
        print(f"  ECR URI: {ecr_uri}")
except FileNotFoundError:
    print("\n⚠ AWS CLI not found (cannot resolve account/ECR URI)")
except Exception:
    # keep quiet like your original code
    pass

if lambda_exists and needs_docker_build and account_id and ecr_uri:
    print("\nLambda function will be updated with new Docker image")
    print("-" * 80)
    response = "y" if _confirm("Proceed with Lambda update?", default=False) else "n"

    if response == "y":
        try:
            result = subprocess.run(
                [
                    "aws", "lambda", "update-function-code",
                    "--function-name", lambda_function_name,
                    "--image-uri", ecr_uri,
                    "--region", region
                ],
                capture_output=False,
                text=True
            )

            if result.returncode == 0:
                print("\n✓ Lambda function updated successfully!")
                subprocess.run(
                    ["aws", "lambda", "wait", "function-updated", "--function-name", lambda_function_name, "--region", region],
                    capture_output=True,
                    text=True
                )
                print("  ✓ Lambda function is ready")
            else:
                print(f"\n⚠ Lambda update exited with code: {result.returncode}")

        except FileNotFoundError:
            print("\n✗ AWS CLI not found (cannot update Lambda)")
        except Exception as e:
            print(f"\n✗ Error updating Lambda: {e}")
    else:
        print("\nSkipping Lambda update")

elif not lambda_exists:
    print(f"\n⚠ Lambda function '{lambda_function_name}' does not exist")
    print("  Create it via AWS Console or CLI, then use this cell to update image.")

elif not needs_docker_build:
    print("\nSkipping Lambda update (Docker image unchanged)")

else:
    print("\nTo update Lambda function manually:")
    print("-" * 80)
    print("\n1. Get your AWS Account ID:")
    print("   AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)")
    print("\n2. Construct ECR URI:")
    print("   ECR_URI=\"${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/pgx-risk-calculator:latest\"")
    print("\n3. Update Lambda function:")
    print("   aws lambda update-function-code \\")
    print("       --function-name pgx-risk-calculator \\")
    print("       --image-uri ${ECR_URI} \\")
    print("       --region us-east-1")

# Set memory and timeout (30s helps cold starts; 512 MB is enough per CloudWatch Max Memory Used)
if lambda_exists:
    try:
        cfg = subprocess.run(
            [
                "aws", "lambda", "update-function-configuration",
                "--function-name", lambda_function_name,
                "--timeout", "30",
                "--memory-size", "512",
                "--region", region
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        if cfg.returncode == 0:
            print("\n✓ Lambda configuration updated (timeout=30s, memory=512 MB)")
            subprocess.run(
                ["aws", "lambda", "wait", "function-updated", "--function-name", lambda_function_name, "--region", region],
                capture_output=True,
                text=True
            )
        else:
            print(f"\n⚠ Lambda configuration update failed: {cfg.stderr or cfg.stdout or 'unknown'}")
    except FileNotFoundError:
        print("\n⚠ AWS CLI not found (cannot update Lambda configuration)")
    except Exception as e:
        print(f"\n⚠ Error updating Lambda configuration: {e}")

print(f"\n{'=' * 80}")

# %% [markdown]
# ### Step 6: Sync dashboard frontend and assets to S3
#
# All dashboard visualization data lives under the S3 **visualizations/** prefix. **Notebook 4 writes to local only** (`SKIP_DASHBOARD_S3_UPLOAD=1`). **Step 6 is the single place that syncs to S3** (idempotent): local → S3 final for all viz types.
#
# Syncs the dashboard frontend and uploads assets required by each tab:
# - **Frontend:** HTML/JS (10_risk_dashboard/frontend → S3 prefix root)
# - **Metadata:** model_performance_metrics.json, cohort metadata (Documentation and dropdowns)
# - **Feature importance:** aggregated heatmaps (PNG) and JSON (aggregated_fi_heatmap.json per cohort and combined) for Feature Importance tab
# - **Cohort PGx:** network topology HTML (PGx Cohort tab)
# - **Causal data:** dashboard_data.json from 10_risk_dashboard/visualizations/causal/{cohort}/{age_band_fname}/ → S3 visualizations/causal/{cohort}/{age_band}/causal_data.json (Causal Analysis tab; upload_causal_outputs_to_s3.py)
#
# **Dashboard tabs ↔ data sources (S3 prefix = `vcu/pgx-risk-calculator/`).** EC2 paths are relative to repo root (e.g. `/home/pgx3874/pgx-analysis`).
#
# | Tab | EC2 folder (deployed from) | API / data source | S3 path (dashboard bucket) | Uploaded in Step 6? |
# |-----|----------------------------|-------------------|----------------------------|---------------------|
# | Risk Assessment | 10_risk_dashboard/outputs/models/, outputs/metadata/ (→ Lambda) | POST /risk, GET /metadata | — (Lambda + container) | — |
# | Drugs / ICD Codes / CPT Codes | 10_risk_dashboard/outputs/metadata/ | GET /metadata | metadata/*.json | ✓ metadata |
# | PGx Card | 10_risk_dashboard/outputs/cpic/ (→ Lambda) | POST /pgx/card | — (Lambda + CPIC) | — |
# | Documentation | 10_risk_dashboard/outputs/metadata/model_performance_metrics.json | Same-origin JSON | metadata/model_performance_metrics.json | ✓ metrics |
# | **Feature Importance** | 3a_feature_importance/outputs/{cohort}/plots/, outputs/plots/ | GET /visualizations/feature_importance?cohort= | visualizations/feature_importance/{cohort}/, .../combined_* | ✓ FI heatmaps |
# | **Causal Analysis** | 10_risk_dashboard/visualizations/causal/{cohort}/{age_band_fname}/dashboard_data.json | GET /visualizations/causal?cohort=&age_band= | visualizations/causal/{cohort}/{age_band}/causal_data.json (S3: hyphen) | ✓ upload_causal_outputs_to_s3 |
# | **BupaR Process Mining** | (notebook 4 → S3 builds) | GET /visualizations/bupar, ... | visualizations/bupar/{cohort}/{age_band}/plots/ | ✓ Step 6 promotes builds→final |
# | **DTW Trajectories** | (notebook 4 → S3 builds) | GET /visualizations/dtw?cohort=&age_band= | visualizations/dtw/{cohort}/{age_band}/chart_data.json, sequence_heatmap.json, plots/ | ✓ Step 6 promotes builds→final |
# | **FP-Growth Patterns** | (notebook 4 → S3 builds) | GET /visualizations/fpgrowth, /fpgrowth/network_html | visualizations/fpgrowth/{cohort}/{age_band}/plots/, data/ | ✓ Step 6 promotes builds→final |
# | **PGx Cohort** | (notebook 4 → S3 builds) | GET /visualizations/cohort_pgx?cohort=&age_band= | visualizations/cohort_pgx/networks/{cohort}/{age_band}/ (S3: hyphen) | ✓ Step 6 promotes builds + sync_cohort_pgx_to_s3 |

# %%
import os
import subprocess

# Step 6: Sync frontend HTML to S3 (local EC2 directory -> S3)
# Dashboard HTML is tracked in git (10_risk_dashboard/frontend/*.html) so local changes deploy here.
print(f"\n{'=' * 80}")
print("Step 6: Sync Dashboard Frontend to S3")
print(f"{'=' * 80}")

frontend_dir = risk_dashboard_dir / "frontend"
s3_bucket = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
s3_prefix = os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator")
s3_uri = f"s3://{s3_bucket}/{s3_prefix}/"

if frontend_dir.exists() and frontend_dir.is_dir():
    print(f"\nSource (local): {frontend_dir}")
    print(f"Destination:   {s3_uri}")
    print("  (Only changed or new files are uploaded. Cache-Control: no-cache so browsers see updates.)")
    print("-" * 80)

    response = "y" if _confirm("Proceed with S3 sync?", default=False) else "n"
    if response == "y":
        # Apply CORS to bucket (idempotent) so direct S3 URL fetches (dtw, bupar, fpgrowth, etc.) work from dashboard origin
        cors_script = risk_dashboard_dir / "deployment" / "apply_dashboard_bucket_cors.py"
        if cors_script.exists():
            cors_result = subprocess.run(
                [str(PYTHON_BIN), str(cors_script), "--bucket", s3_bucket, "--region", "us-east-1"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )
            if cors_result.returncode == 0:
                print(cors_result.stdout.strip() or "✓ Dashboard bucket CORS applied")
            else:
                print(f"  ⚠ CORS apply failed (non-fatal): {cors_result.stderr or cors_result.stdout}")
        try:
            # Cache-Control so browsers don't serve stale HTML/JS after deploy
            cache_control = "max-age=0, no-cache, no-store, must-revalidate"
            result = subprocess.run(
                ["aws", "s3", "sync", str(frontend_dir), s3_uri, "--region", "us-east-1", "--cache-control", cache_control],
                capture_output=False,
                text=True
            )

            if result.returncode == 0:
                print("\n✓ Frontend synced to S3")
                print(f"  Dashboard URL: https://{s3_bucket}.s3.us-east-1.amazonaws.com/{s3_prefix}/index.html")

                # Upload dashboard visual manifest so Documentation tab (index.html) can load it same-origin
                manifest_path = risk_dashboard_dir / "visualizations" / "dashboard_visual_objects.json"
                if manifest_path.exists():
                    manifest_key = f"{s3_prefix.rstrip('/')}/visualizations/dashboard_visual_objects.json"
                    try:
                        subprocess.run(
                            ["aws", "s3", "cp", str(manifest_path), f"s3://{s3_bucket}/{manifest_key}", "--content-type", "application/json", "--region", "us-east-1"],
                            check=True, capture_output=True, text=True
                        )
                        print(f"  ✓ Manifest uploaded: {manifest_key}")
                    except (subprocess.CalledProcessError, FileNotFoundError) as e:
                        print(f"  ⚠ Manifest upload failed: {e}")
                else:
                    print(f"  ⚠ Manifest not found: {manifest_path} — run notebook 4 and execute the cell that writes dashboard_visual_objects.json.")

                # Upload metrics JSON so Documentation tab loads it same-origin (no API call)
                metrics_path = risk_dashboard_dir / "outputs" / "metadata" / "model_performance_metrics.json"
                if metrics_path.exists():
                    metrics_key = f"{s3_prefix.rstrip('/')}/metadata/model_performance_metrics.json"
                    try:
                        subprocess.run(
                            [
                                "aws", "s3", "cp",
                                str(metrics_path),
                                f"s3://{s3_bucket}/{metrics_key}",
                                "--content-type", "application/json",
                                "--region", "us-east-1"
                            ],
                            check=True,
                            capture_output=True,
                            text=True
                        )
                        print(f"  ✓ Metrics uploaded to s3://{s3_bucket}/{metrics_key}")
                    except (subprocess.CalledProcessError, FileNotFoundError) as e:
                        print(f"  ⚠ Metrics upload failed: {e}")
                else:
                    print(f"  ⚠ Metrics file not found: {metrics_path} — run data_preparation/generate_metrics.py to create it.")

                # Upload cohort metadata JSONs so dropdowns load same-origin (no API call)
                metadata_dir = risk_dashboard_dir / "outputs" / "metadata"
                for cohort_name, s3_suffix in [("opioid_ed", "opioid_ed"), ("non_opioid_ed", "non_opioid_ed")]:
                    meta_path = metadata_dir / f"metadata_{cohort_name}.json"
                    if meta_path.exists():
                        meta_key = f"{s3_prefix.rstrip('/')}/metadata/{s3_suffix}.json"
                        try:
                            subprocess.run(
                                [
                                    "aws", "s3", "cp",
                                    str(meta_path),
                                    f"s3://{s3_bucket}/{meta_key}",
                                    "--content-type", "application/json",
                                    "--region", "us-east-1"
                                ],
                                check=True,
                                capture_output=True,
                                text=True
                            )
                            print(f"  ✓ Metadata uploaded: {meta_key}")
                        except (subprocess.CalledProcessError, FileNotFoundError) as e:
                            print(f"  ⚠ Metadata upload failed ({cohort_name}): {e}")

                if not (metadata_dir / "metadata_opioid_ed.json").exists() or not (metadata_dir / "metadata_non_opioid_ed.json").exists():
                    print(f"  ⚠ Some metadata files missing in {metadata_dir} — run data_preparation/generate_metadata.py --all to create them.")

                # Upload feature importance heatmaps (notebook 4 copies 3a → visualizations/feature_importance)
                fi_base = PROJECT_ROOT / "10_risk_dashboard" / "visualizations" / "feature_importance"
                fi_prefix = f"{s3_prefix.rstrip('/')}/visualizations/feature_importance"

                for cohort in ("opioid_ed", "non_opioid_ed"):
                    local_png = fi_base / cohort / "aggregated_fi_heatmap.png"
                    if local_png.exists():
                        s3_key = f"{fi_prefix}/{cohort}/aggregated_fi_heatmap.png"
                        try:
                            subprocess.run(
                                ["aws", "s3", "cp", str(local_png), f"s3://{s3_bucket}/{s3_key}", "--region", "us-east-1"],
                                check=True,
                                capture_output=True,
                                text=True
                            )
                            print(f"  ✓ Feature importance heatmap: {s3_key}")
                        except (subprocess.CalledProcessError, FileNotFoundError) as e:
                            print(f"  ⚠ FI heatmap upload failed ({cohort}): {e}")
                    local_json = fi_base / cohort / "aggregated_fi_heatmap.json"
                    if local_json.exists():
                        s3_key_json = f"{fi_prefix}/{cohort}/aggregated_fi_heatmap.json"
                        try:
                            subprocess.run(
                                ["aws", "s3", "cp", str(local_json), f"s3://{s3_bucket}/{s3_key_json}", "--content-type", "application/json", "--region", "us-east-1"],
                                check=True,
                                capture_output=True,
                                text=True
                            )
                            print(f"  ✓ Feature importance heatmap (JSON): {s3_key_json}")
                        except (subprocess.CalledProcessError, FileNotFoundError) as e:
                            print(f"  ⚠ FI heatmap JSON upload failed ({cohort}): {e}")

                combined_png = fi_base / "combined_cohorts_feature_importance_heatmap.png"
                if combined_png.exists():
                    s3_key_combined = f"{fi_prefix}/combined_cohorts_feature_importance_heatmap.png"
                    try:
                        subprocess.run(
                            ["aws", "s3", "cp", str(combined_png), f"s3://{s3_bucket}/{s3_key_combined}", "--region", "us-east-1"],
                            check=True,
                            capture_output=True,
                            text=True
                        )
                        print(f"  ✓ Feature importance heatmap: {s3_key_combined}")
                    except (subprocess.CalledProcessError, FileNotFoundError) as e:
                        print(f"  ⚠ FI combined heatmap upload failed: {e}")
                combined_json = fi_base / "combined" / "aggregated_fi_heatmap.json"
                if combined_json.exists():
                    s3_key_combined_json = f"{fi_prefix}/combined/aggregated_fi_heatmap.json"
                    try:
                        subprocess.run(
                            ["aws", "s3", "cp", str(combined_json), f"s3://{s3_bucket}/{s3_key_combined_json}", "--content-type", "application/json", "--region", "us-east-1"],
                            check=True,
                            capture_output=True,
                            text=True
                        )
                        print(f"  ✓ Feature importance heatmap (JSON combined): {s3_key_combined_json}")
                    except (subprocess.CalledProcessError, FileNotFoundError) as e:
                        print(f"  ⚠ FI combined JSON upload failed: {e}")

                # Sync Cohort PGx to S3 (script maps EC2 25_44 -> S3 25-44)
                cohort_pgx_local = risk_dashboard_dir / "visualizations" / "cohort_pgx"
                if cohort_pgx_local.exists():
                    try:
                        subprocess.run(
                            [str(PYTHON_BIN), str(risk_dashboard_dir / "deployment" / "sync_cohort_pgx_to_s3.py"), "--local-dir", str(cohort_pgx_local)],
                            check=True,
                            capture_output=True,
                            text=True,
                            cwd=str(PROJECT_ROOT)
                        )
                        print("  ✓ Cohort PGx (PGx Cohort tab) synced to S3")
                    except (subprocess.CalledProcessError, FileNotFoundError) as e:
                        print(f"  ⚠ Cohort PGx sync failed: {e}")
                else:
                    print("  ⚠ Cohort PGx not found (run 4_dashboard_visuals Cohort PGx); PGx Cohort tab will 404 until synced.")

                # Upload causal dashboard JSON (Causal Analysis tab): 10_risk_dashboard/visualizations/causal/ -> S3 visualizations/causal/
                upload_causal_script = risk_dashboard_dir / "data_preparation" / "upload_causal_outputs_to_s3.py"
                if upload_causal_script.exists():
                    try:
                        result_causal = subprocess.run(
                            [str(PYTHON_BIN), str(upload_causal_script)],
                            cwd=str(PROJECT_ROOT),
                            capture_output=True,
                            text=True,
                        )
                        if result_causal.returncode == 0 and result_causal.stdout:
                            for line in result_causal.stdout.strip().split("\n"):
                                print(f"  {line}")
                        elif result_causal.returncode != 0 and result_causal.stderr:
                            print(f"  ⚠ Causal upload: {result_causal.stderr.strip() or 'failed'}")
                    except Exception as e:
                        print(f"  ⚠ Causal upload failed: {e}")
                else:
                    print("  ⚠ Causal upload script not found; Causal Analysis tab may have no data until dashboard_data.json is uploaded to S3.")

                # Manifest-driven upload: only upload static_files listed in dashboard_visual_objects.json.
                # This prevents debug artifacts (trajectory_status_*.json, dtw_model_events_diagnostics_*.json,
                # *.csv, *.parquet, Rplots.pdf, feature_engineering/, etc.) from polluting S3.
                # Covers: BupaR, DTW, FP-Growth, Feature Importance, Cohort PGx.
                # Causal Analysis is handled above by upload_causal_outputs_to_s3.py (rename-on-upload).
                sync_visuals_script = risk_dashboard_dir / "deployment" / "sync_visuals_to_s3.py"
                if sync_visuals_script.exists():
                    try:
                        result_visuals = subprocess.run(
                            [
                                str(PYTHON_BIN),
                                str(sync_visuals_script),
                                "--bucket", s3_bucket,
                                "--prefix", s3_prefix.strip("/"),
                                "--region", "us-east-1",
                            ],
                            cwd=str(PROJECT_ROOT),
                            capture_output=True,
                            text=True,
                        )
                        for line in (result_visuals.stdout or "").strip().split("\n"):
                            if line.strip():
                                print(f"  {line}")
                        if result_visuals.returncode not in (0, 1):
                            print(f"  ⚠ sync_visuals_to_s3 exited {result_visuals.returncode}")
                            if result_visuals.stderr:
                                print(f"    {result_visuals.stderr.strip()}")
                    except Exception as e:
                        print(f"  ⚠ Manifest-driven visual sync failed: {e}")
                else:
                    print("  ⚠ sync_visuals_to_s3.py not found; falling back to broad aws s3 sync")
                    viz_prefix = f"{s3_prefix.rstrip('/')}/visualizations"
                    bupar_out = risk_dashboard_dir / "visualizations" / "bupar"
                    if bupar_out.exists():
                        for cohort_dir in sorted(bupar_out.iterdir()):
                            if not cohort_dir.is_dir(): continue
                            for age_dir in sorted(cohort_dir.iterdir()):
                                if not age_dir.is_dir(): continue
                                plots_dir = age_dir / "plots"
                                if not plots_dir.exists(): continue
                                age_band = age_dir.name.replace("_", "-")
                                dest = f"s3://{s3_bucket}/{viz_prefix}/bupar/{cohort_dir.name}/{age_band}/plots/"
                                subprocess.run(["aws", "s3", "sync", str(plots_dir), dest, "--region", "us-east-1", "--exclude", "Rplots.pdf"], text=True)
                    dtw_out = risk_dashboard_dir / "visualizations" / "dtw"
                    if dtw_out.exists():
                        for cohort_dir in sorted(dtw_out.iterdir()):
                            if not cohort_dir.is_dir(): continue
                            if cohort_dir.name == "feature_engineering": continue
                            for age_dir in sorted(cohort_dir.iterdir()):
                                if not age_dir.is_dir(): continue
                                if not (age_dir / "chart_data.json").exists() and not (age_dir / "plots").exists(): continue
                                age_band = age_dir.name.replace("_", "-")
                                dest = f"s3://{s3_bucket}/{viz_prefix}/dtw/{cohort_dir.name}/{age_band}/"
                                subprocess.run(["aws", "s3", "sync", str(age_dir), dest, "--region", "us-east-1", "--exclude", "*.csv", "--exclude", "*checkpoint*"], text=True)
                    fpgrowth_out = risk_dashboard_dir / "visualizations" / "fpgrowth"
                    if fpgrowth_out.exists():
                        for cohort_dir in sorted(fpgrowth_out.iterdir()):
                            if not cohort_dir.is_dir(): continue
                            for age_dir in sorted(cohort_dir.iterdir()):
                                if not age_dir.is_dir(): continue
                                if not (age_dir / "plots").exists() and not (age_dir / "data").exists(): continue
                                age_band = age_dir.name.replace("_", "-")
                                dest = f"s3://{s3_bucket}/{viz_prefix}/fpgrowth/{cohort_dir.name}/{age_band}/"
                                subprocess.run(["aws", "s3", "sync", str(age_dir), dest, "--region", "us-east-1", "--exclude", "*checkpoint*"], text=True)

            else:
                print(f"\n⚠ S3 sync exited with code: {result.returncode}")

        except Exception as e:
            print(f"\n✗ Error syncing to S3: {e}")
    else:
        print("\nSkipping S3 sync")
else:
    print(f"\n⚠ Frontend directory not found: {frontend_dir}")
    print("  Expected: 10_risk_dashboard/frontend/ (index.html and assets; tracked in git)")

print(f"\n{'=' * 80}")

# %% [markdown]
# ### Invalidate CloudFront distribution (before EC2 shutdown)
#
# After syncing the frontend and assets to S3, invalidate the CloudFront cache so the dashboard at **jerome-dixon.io** serves the latest content. Run this step before shutting down the EC2 instance.

# %%
# CloudFront invalidation (jerome-dixon.io distribution)
# Distribution ID for jerome-dixon.io HTTPS website (origin: jerome-dixon.io.s3-website-us-east-1.amazonaws.com)
CLOUDFRONT_DISTRIBUTION_ID = os.environ.get("CLOUDFRONT_DISTRIBUTION_ID", "E3MZK5HYTJ14P3")

import subprocess
import shutil

aws_cmd = shutil.which("aws") or next(
    (p for p in ["/usr/local/bin/aws", "/usr/bin/aws", "/home/ec2-user/.local/bin/aws"] if os.path.exists(p)),
    None
)
if aws_cmd:
    invalidation_cmd = [aws_cmd, "cloudfront", "create-invalidation", "--distribution-id", CLOUDFRONT_DISTRIBUTION_ID, "--paths", "/*"]
    print(f"Running: {' '.join(invalidation_cmd)}")
    result = subprocess.run(invalidation_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("CloudFront invalidation created successfully. Cache will clear in a few minutes.")
        if result.stdout:
            print(result.stdout.strip())
    else:
        print(f"Invalidation failed (exit code {result.returncode}): {result.stderr or result.stdout}")
else:
    print("AWS CLI not found. Skipping CloudFront invalidation. Run manually: aws cloudfront create-invalidation --distribution-id E3MZK5HYTJ14P3 --paths '/*'")

# %% [markdown]
# # Shutdown EC2

# %%
# -------------------------------------------------------------------
# Optional EC2 Auto-Shutdown
# -------------------------------------------------------------------

SHUTDOWN_EC2 = True  # Set to False to disable auto-shutdown

print("=" * 80)
print("Final Step: EC2 Instance Shutdown (Optional)")
print("=" * 80)

if SHUTDOWN_EC2:
    print("\nShutting down EC2 instance...")
    print("-" * 80)

    import subprocess
    import shutil
    import os

    try:
        # Retrieve EC2 instance ID from metadata service
        result = subprocess.run(
            ["curl", "-s", "http://169.254.169.254/latest/meta-data/instance-id"],
            capture_output=True,
            text=True,
            timeout=5
        )

        instance_id = result.stdout.strip()

        if instance_id:
            print(f"Instance ID: {instance_id}")

            # Locate AWS CLI
            aws_cmd = shutil.which("aws")
            if not aws_cmd:
                for path in [
                    "/usr/local/bin/aws",
                    "/usr/bin/aws",
                    "/home/ec2-user/.local/bin/aws"
                ]:
                    if os.path.exists(path):
                        aws_cmd = path
                        break

            if not aws_cmd:
                print("\nWarning: AWS CLI not found. Cannot stop instance.")
                print("Install AWS CLI or ensure it is in your PATH.")
                logger.warning("AWS CLI not found; cannot stop EC2 instance")
            else:
                shutdown_cmd = [
                    aws_cmd,
                    "ec2",
                    "stop-instances",
                    "--instance-ids",
                    instance_id
                ]

                print(f"Running: {' '.join(shutdown_cmd)}")
                result = subprocess.run(
                    shutdown_cmd,
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0:
                    print("\nEC2 stop command sent successfully.")
                    print("Instance will stop shortly.")
                    print("Note: This is a STOP (not terminate).")
                    logger.info(
                        f"EC2 instance {instance_id} stop command issued"
                    )
                else:
                    print(
                        f"\nWarning: EC2 stop command failed "
                        f"(exit code {result.returncode})"
                    )
                    if result.stderr:
                        print(f"Error: {result.stderr.strip()}")
                    logger.warning(
                        f"EC2 stop command failed: {result.stderr}"
                    )
        else:
            print("\nWarning: Instance ID not found. Skipping shutdown.")
            print("Manual shutdown command:")
            print("  aws ec2 stop-instances --instance-ids <instance-id>")
            logger.warning("EC2 instance ID could not be determined")

    except subprocess.TimeoutExpired:
        print("\nWarning: Timeout contacting EC2 metadata service.")
        logger.warning("Timeout retrieving EC2 instance ID")

    except Exception as e:
        print(f"\nWarning: Error during EC2 shutdown: {e}")
        logger.warning(f"EC2 shutdown exception: {e}")

else:
    print("\nEC2 Auto-Shutdown: DISABLED")
    print("Set SHUTDOWN_EC2 = True to enable it.")

print("\n" + "=" * 80)
print("Workflow Complete!")
print("=" * 80)

# %%
