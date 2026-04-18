# Deployment

## Overview

Scripts and configurations for deploying the dashboard to AWS.

## Files

- **`docker_build.sh`** - Build and push Docker image to ECR
- **`prepare_lambda_dir.py`** - Prepare Lambda deployment package
- **`apply_dashboard_bucket_cors.py`** - Apply CORS to the dashboard S3 bucket (idempotent). Run by Step 6 in notebook 5; can also be run standalone or with `--check` to print current CORS.
- **`apply_api_gateway_cors.py`** - Add CORS headers to API Gateway **gateway responses** (4XX, 5XX, DEFAULT) so the browser receives `Access-Control-Allow-Origin` even when Lambda errors or times out. Run once per API (e.g. `python apply_api_gateway_cors.py --api-id cmv0qislq3`). See [CORS blocked from jerome-dixon.io](#cors-blocked-from-jeromedixonio) below.
- **`scripts/`** - Additional deployment helper scripts

**Dashboard tabs ↔ data sources:** See [../docs/DASHBOARD_TABS.md](../docs/DASHBOARD_TABS.md) for which API/S3 path feeds each tab (Feature Importance, Causal Analysis, BupaR, DTW, FP-Growth, PGx Cohort, etc.). That doc also documents the **required S3 URL format** for assets: path-style `https://s3.{region}.amazonaws.com/{bucket}/{prefix}/{key}`. **Age bands:** EC2 paths use underscore (e.g. `25_44`); S3 paths use hyphen (e.g. `25-44`). Use `sync_cohort_pgx_to_s3.py` for Cohort PGx so S3 keys use hyphen.

**Target S3 path for BupaR (Lambda reads this):** The API expects BupaR plots under `{prefix}/visualizations/bupar/{cohort}/{age_band}/plots/`. Example: `s3://jerome-dixon.io/vcu/pgx-risk-calculator/visualizations/bupar/opioid_ed/45-54/plots/`. Step 6 in notebook 5 promotes from `.../visualizations/bupar/builds/` to this path.

**Static-first JSON (fast/cheap):** Pre-built JSON (metadata, feature importance) is loaded from same-origin paths (S3/CloudFront) first; Lambda API is fallback. See [../docs/STATIC_FIRST_JSON.md](../docs/STATIC_FIRST_JSON.md) for the pattern and S3 layout.

**S3 CORS and public read:** When the frontend fetches direct S3 URLs (e.g. `causal_data_url`), the dashboard bucket must have (1) **CORS** configured and (2) **bucket policy** allowing public `GetObject` for the dashboard prefix (or serve assets only via CloudFront). See [../docs/S3_CORS_SETUP.md](../docs/S3_CORS_SETUP.md) (CORS + 403 troubleshooting) and `../docs/s3-public-read-policy.json`. **CORS is applied automatically** in the deployment workflow: notebook 5 **Step 6** runs `apply_dashboard_bucket_cors.py` before syncing frontend/assets so the bucket CORS is idempotent and repeatable for production and new visuals.

## EC2 vs Local (Windows) Deployment Workflows

### When to use each

| Task | EC2 | Windows (Local) |
|---|---|---|
| Retrain models (notebook 3) | ✅ Required | ❌ |
| Regenerate visualization artifacts (notebook 4) | ✅ Required | ❌ |
| Sync artifacts to S3 (notebook 5 Step 6) | ✅ Preferred | ✅ (if data already on S3) |
| Full Docker rebuild + ECR push | ✅ Preferred (fast, local models) | ✅ (slow, pulls models from S3) |
| `lambda_function.py` code-only change | ✅ | ✅ (use S3 update path — no rebuild) |
| Frontend `index.html` change | ✅ | ✅ |

---

### EC2 Full Deployment

**Prerequisites:** SSH to EC2, activate the project env, navigate to repo root.

```bash
# 1. Regenerate per-bin visualization artifacts (if pipeline changed)
#    Run notebook 4 FP-Growth / BupaR / DTW / Cohort PGx cells as needed

# 2. Sync artifacts to S3
cd /home/pgx3874/pgx-analysis
python3 10_risk_dashboard/deployment/sync_visuals_to_s3.py

# 3. Full Docker rebuild, push to ECR, update Lambda
bash 10_risk_dashboard/deployment/docker_build.sh
```

`docker_build.sh` auto-detects the Python binary (`jupyter-env/bin/python3.11` → `python3` → `python`) and uses `--no-s3` (local model files) automatically when EC2 training outputs are present.

---

### Windows (Local) — Full Docker Rebuild

**Prerequisites:** Docker Desktop running, AWS CLI configured (`aws sts get-caller-identity` to verify).

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Projects\pgx-analysis\10_risk_dashboard\deployment\scripts\build_and_push.ps1"
```

`build_and_push.ps1` uses the dashboard root as Docker build context (required — Dockerfile references `COPY backend/...` and `COPY lambda_dir/...` relative to dashboard root). `prepare_lambda_dir.py` pulls models from S3 since EC2 training outputs are not present locally.

**Note:** Full rebuild takes longer on Windows because models must be downloaded from S3. Prefer the S3 code-only update (below) for pure Python changes.

---

### Windows (Local) — Code-Only Lambda Update (Fastest)

Use when only `lambda_function.py` changed — no model or artifact changes:

```powershell
# 1. Upload updated code to S3
aws s3 cp C:\Projects\pgx-analysis\10_risk_dashboard\backend\lambda_function.py `
    s3://pgxdatalake/gold/dashboard/code/lambda_function.py

# 2. Trigger cold start (Lambda downloads new code on next invocation)
aws lambda update-function-configuration `
    --function-name pgx-risk-calculator `
    --environment 'Variables={S3_BUCKET=pgxdatalake,CODE_S3_KEY=gold/dashboard/code/lambda_function.py,PREFER_S3=false,PGX_RESULTS_BUCKET=pgxdatalake}'
```

Lambda cold start takes ~20 s. Verify with an API call after waiting:
```bash
curl -s "https://cmv0qislq3.execute-api.us-east-1.amazonaws.com/prod/available" | python3 -c "import sys,json; d=json.load(sys.stdin); print('causal combos:', sum(len(v) for v in d.get('causal',{}).values()))"
```

---

### How `docker_build.sh` decides EC2 vs Windows/CI

```bash
FINAL_MODEL_OUTPUTS="${DASHBOARD_ROOT}/../6_final_model/outputs"
if [ -d "/mnt/nvme" ] || [ -d "${FINAL_MODEL_OUTPUTS}" ]; then
    # EC2: use local model files (fast, no S3 download)
    PREPARE_FLAGS="--no-s3"
else
    # Windows/CI: pull models from S3
    PREPARE_FLAGS=""
fi
```

---

### Verify deployment from anywhere

```python
import urllib.request, json
base = "https://cmv0qislq3.execute-api.us-east-1.amazonaws.com/prod"
av = json.loads(urllib.request.urlopen(f"{base}/available").read())
print("causal:", sum(len(v) for v in av.get("causal",{}).values()), "combos")
print("fpgrowth:", sum(len(v) for v in av.get("fpgrowth",{}).values()), "combos")
print("fpgrowth_per_bin:", sum(len(bins) for c in av.get("fpgrowth_per_bin",{}).values() for bins in c.values()), "bin-combos")
```

Expected: `causal ≥ 14`, `fpgrowth = 14`, `fpgrowth_per_bin ≥ 20`.

---

## Deployment Steps

1. **Prepare Models and Metadata**:
   ```bash
   cd ../data_preparation
   python prepare_models.py --all
   python generate_metadata.py --all
   python prepare_cpic_data.py
   ```

2. **Build Docker Image**:
   ```bash
   ./docker_build.sh
   ```

3. **Create Lambda Function**:
   - Use ECR image created in step 2
   - Configure API Gateway integration
   - Set environment variables

4. **Deploy Frontend**:
   - **S3 location:** `s3://jerome-dixon.io/vcu/pgx-risk-calculator/`
   - Upload `../frontend/index.html` and assets to that prefix (e.g. `aws s3 sync ../frontend/ s3://jerome-dixon.io/vcu/pgx-risk-calculator/`)
   - Upload model performance metrics:  
     `aws s3 cp ../outputs/metadata/model_performance_metrics.json s3://jerome-dixon.io/vcu/pgx-risk-calculator/metadata/model_performance_metrics.json --content-type application/json`
   - Upload cohort metadata for dropdowns (same-origin, no API call):  
     `aws s3 cp ../outputs/metadata/metadata_opioid_ed.json s3://jerome-dixon.io/vcu/pgx-risk-calculator/metadata/opioid_ed.json --content-type application/json`  
     `aws s3 cp ../outputs/metadata/metadata_non_opioid_ed.json s3://jerome-dixon.io/vcu/pgx-risk-calculator/metadata/non_opioid_ed.json --content-type application/json`  
     (The 5_build_and_deploy notebook uploads metrics and metadata automatically after frontend sync.)
   - Configure the bucket for static website hosting (or use CloudFront with that origin)
   - (Optional) Set up CloudFront distribution

## Architecture

```
User Browser
    ↓
S3 Static Website (frontend/index.html)
    ↓
API Gateway
    ↓
Lambda Function (ECR Container)
    ├── Models (from /var/task/models/)
    ├── Metadata (from /var/task/metadata/)
    └── CPIC Data (from /var/task/cpic/)
```

## Troubleshooting: "Error loading metadata" / "Failed to fetch" / "Not secure"

### Why VCU fails but UVA works (different origins)

- **VCU (failing):** `https://jerome-dixon.io.s3.us-east-1.amazonaws.com/vcu/pgx-risk-calculator/index.html`  
  Page origin = `https://jerome-dixon.io.s3.us-east-1.amazonaws.com` (raw S3 REST hostname).
- **UVA (working):** `https://jerome-dixon.io/uva/phts-risk-calculator/index.html`  
  Page origin = `https://jerome-dixon.io` (custom domain).

The dashboard calls API Gateway from the **page origin**. The Lambda returns `Access-Control-Allow-Origin: *`, but:

1. **CORS / "Failed to fetch"** – Browsers can still block or surface errors when the page is served from the S3 bucket hostname (e.g. preflight, strict transport, or resource policies). So the same API can work from `jerome-dixon.io` and fail from `jerome-dixon.io.s3.us-east-1.amazonaws.com`.
2. **"Not secure"** – The VCU URL is the **S3 REST endpoint**, not your custom domain. Browsers often show "Not secure" for that hostname (or when mixed content / failed requests occur). The UVA app is served under `jerome-dixon.io`, which is typically behind CloudFront with a proper certificate, so it shows as secure.

### Recommended fix: serve VCU from the same domain as UVA

Serve the PGx calculator from your custom domain so it uses the same origin and HTTPS setup as the working UVA app:

- **Target URL:** `https://jerome-dixon.io/vcu/pgx-risk-calculator/index.html`
- **How:** Upload the frontend to the same bucket/prefix used for the domain (e.g. under `vcu/pgx-risk-calculator/` in the bucket that backs `jerome-dixon.io`), so that `jerome-dixon.io` (and CloudFront, if used) serves both `uva/` and `vcu/`. Do **not** open the app via the raw S3 URL `jerome-dixon.io.s3.us-east-1.amazonaws.com`.
- **Result:** Same origin as UVA, one certificate, no cross-origin surprise, and "Not secure" goes away.

### CORS blocked from jerome-dixon.io

If the browser shows: *"Access to fetch at 'https://...execute-api.../prod/visualizations/dtw?...' from origin 'https://jerome-dixon.io' has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header"* — the API response is not including CORS headers. Apply gateway response CORS once (no redeploy):

```bash
python 10_risk_dashboard/deployment/apply_api_gateway_cors.py --api-id cmv0qislq3 --region us-east-1
```

This adds `Access-Control-Allow-Origin` (and related headers) to API Gateway’s DEFAULT_4XX, DEFAULT_5XX, and DEFAULT responses so even errors/timeouts return CORS. Lambda already sends CORS on 200 responses.

### If you must keep the S3 hostname URL

1. **Ensure API Gateway can invoke Lambda**  
   From project root:  
   - **Windows (PowerShell):** `.\utility_scripts\create_api_gateway_pgx_risk_calculator.ps1`  
   - **Linux / macOS (bash):** `bash utility_scripts/create_api_gateway_pgx_risk_calculator.sh`  
   This (re)creates the API and adds Lambda invoke permission.

   **Optional – PHTS-style resources (per-tab paths):** To have explicit resources like the PHTS calculator (`/metadata`, `/risk`, `/risk/comparison`, `/causal/importance`, `/causal/interactions`) instead of only `/` and `/{proxy+}`, run after the create script:  
   - **Windows:** `.\utility_scripts\add_api_gateway_resources_phts_style.ps1`  
   - **Linux / macOS:** `bash utility_scripts/add_api_gateway_resources_phts_style.sh`  
   Same Lambda handles all paths; the new resources take precedence over the proxy.

2. **Test the API** (replace with your API id if different):
   ```bash
   # CORS preflight
   curl -s -o /dev/null -w "%{http_code}" -X OPTIONS "https://cmv0qislq3.execute-api.us-east-1.amazonaws.com/prod/metadata"
   # Should return 200

   # Metadata
   curl -s "https://cmv0qislq3.execute-api.us-east-1.amazonaws.com/prod/metadata?cohort=opioid_ed"
   ```
   If OPTIONS or GET fails, fix API Gateway/Lambda before relying on the dashboard.

3. **Add CORS to API Gateway gateway responses** (fixes "No 'Access-Control-Allow-Origin' header" when calling from `https://jerome-dixon.io`). Lambda already returns CORS on 200, but if Lambda fails or times out, API Gateway returns 502/503 without CORS unless gateway responses are set. Run once:
   ```bash
   python 10_risk_dashboard/deployment/apply_api_gateway_cors.py --api-id cmv0qislq3 --region us-east-1
   ```
   (Use `--profile PROFILE` if needed.) No redeploy needed; takes effect immediately.

4. Confirm there is **no API Gateway resource policy** (or WAF) that restricts `Origin` to only `https://jerome-dixon.io`; Lambda already sends `Access-Control-Allow-Origin: *`.

5. **Redeploy Lambda** after backend code changes (e.g. CORS): rebuild image, push to ECR, then update Lambda code (notebook "Update Lambda" or `aws lambda update-function-code --image-uri ...`).

6. **Set Lambda memory and timeout** (30 s helps cold starts; 512 MB is sufficient per logs). Replace `YOUR_FUNCTION_NAME` with your PGx dashboard Lambda name:
   ```bash
   aws lambda update-function-configuration \
     --function-name YOUR_FUNCTION_NAME \
     --timeout 30 \
     --memory-size 512
   ```
   Then wait for the function to finish updating (or use `--no-cli-pager` and run updates before redeploying code).

### Still getting "Failed to fetch" for metadata?

1. **Verify the API from the command line** (use your actual API id if different):
   ```bash
   curl -s -w "\nHTTP_CODE:%{http_code}\n" "https://cmv0qislq3.execute-api.us-east-1.amazonaws.com/prod/metadata?cohort=opioid_ed"
   ```
   You should see JSON and `HTTP_CODE:200`. If you get 403/502/504, fix API Gateway and Lambda first.

2. **Confirm which API URL the dashboard uses**  
   After deploying the updated frontend, the error message will include `[API: https://...]`. Open the dashboard with an explicit base if needed:  
   `https://jerome-dixon.io/vcu/pgx-risk-calculator/index.html?apiBase=https://cmv0qislq3.execute-api.us-east-1.amazonaws.com/prod`

3. **Deploy the correct file**  
   The live app must use `frontend/index.html` (which has the real API id). Do **not** deploy `pgx_dashboard.html` as the main page—it contains the placeholder `YOUR_API.execute-api...`.

4. **Check the browser Network tab**  
   Find the request to `.../metadata?cohort=...`. If it is blocked (CORS) you’ll see a red entry and no response headers. If it returns 4xx/5xx, the response body will show the backend error.

---

## Lessons Learned (2026-04-18)

### 1 — `python: command not found` in `docker_build.sh` on EC2

**Symptom:** `docker_build.sh` line 38 fails with `python: command not found` on EC2.  
**Root cause:** EC2 uses a virtualenv (`/home/pgx3874/jupyter-env/bin/python3.11`); `python` is not on PATH.  
**Fix:** Auto-detect the Python binary at the top of `docker_build.sh`:
```bash
if [ -x "/home/pgx3874/jupyter-env/bin/python3.11" ]; then
    PYTHON_BIN="/home/pgx3874/jupyter-env/bin/python3.11"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi
${PYTHON_BIN} deployment/prepare_lambda_dir.py ${PREPARE_FLAGS}
```

---

### 2 — Bare `except` in boto3 S3 checks silently swallows `NoCredentialsError`

**Symptom:** `s3.head_object()` in a `try/except Exception` block reported every S3 key as MISSING when run on a machine without AWS credentials.  
**Root cause:** `NoCredentialsError` is a subclass of `Exception`. A bare `except` catches it and falls through to the "missing" branch, producing completely false results without any warning.  
**Fix:** Always catch specific errors from boto3:
```python
from botocore.exceptions import ClientError, NoCredentialsError
try:
    s3.head_object(Bucket=bucket, Key=key)
    return True
except ClientError as e:
    if e.response["Error"]["Code"] in ("404", "NoSuchKey", "403", "AccessDenied"):
        return False
    raise
except NoCredentialsError:
    raise  # Never silently swallow credential errors
```
**Never** use `try/except Exception` or bare `except` around S3 existence probes.

---

### 3 — FP-Growth per-bin path mismatch (local vs S3)

**Symptom:** Lambda returned `full_cohort_fallback` for all bins even though `available.json` showed per-bin data present.  
**Root cause:** `_save_per_density_fpgrowth_outputs` saved JSON locally to `density/{bin}/drug_name_itemsets.json` (no `plots/` subdir) but the inline S3 upload and Lambda probe both expected `density/{bin}/plots/drug_name_itemsets.json`. Step 6 sync mirrored the local path, so S3 had files at the wrong location.  
**Fix:** Changed local save to `density/{bin}/plots/` (matching the S3/Lambda canonical path). Added backward-compat Lambda probe that also checks the legacy `density/{bin}/` path during transition.  
**Rule:** Local save path and S3 upload path in `_save_per_density_fpgrowth_outputs` must always be identical so Step 6 sync produces the correct S3 layout.

---

### 4 — Manifest-based sync (`sync_visuals_to_s3.py`) does not auto-discover per-bin subdirs

**Symptom:** Per-bin FP-Growth files existed on EC2 disk but were never uploaded to S3.  
**Root cause:** `sync_visuals_to_s3.py` only uploads files explicitly listed in `static_files` per manifest entry. Per-bin subdirs (`density/{bin}/plots/`) were not in the manifest so they were silently skipped.  
**Fix:** Added 4 per-bin manifest entries to `dashboard_visual_objects.json` (one per density level: `low`, `medium`, `high`, `extreme`), each with `s3_path` pointing to `density/{bin}/plots/`.  
**Rule:** Any new S3 path that Lambda reads must have a corresponding manifest entry. The manifest is the single source of truth for what gets synced.

---

### 5 — `_resolve_local` only captured the first path component after `{age_band}`

**Symptom:** Per-bin manifest entries (e.g. `s3_path: ".../density/low/plots/"`) resolved to the wrong local directory (`density/` only, losing `low/plots/`).  
**Root cause:** `_resolve_local` split on `/` and took only `parts[0]` after `{age_band}`, truncating multi-level fixed subpaths.  
**Fix:** Changed to capture all fixed components between `{age_band}` and the next `{` placeholder (or end of path):
```python
parts = raw.split("/")
fixed_parts = [p for p in itertools.takewhile(lambda p: p and "{" not in p, parts)]
after_age_band = "/".join(fixed_parts)  # e.g. "density/low/plots"
```

---

### 6 — `empty_state.json` must not short-circuit nearest-bin fallback

**Symptom:** When a requested bin had insufficient transactions (`empty_state.json`), the user saw "no data" even though an adjacent bin had real patterns.  
**Root cause:** Lambda checked `empty_state.json` existence first and immediately returned it, skipping the nearest-bin search.  
**Fix:** Priority order is now:
1. Requested bin has `itemsets.json` → `per_bin`
2. Nearest bin with real `itemsets.json` exists → `nearest_bin_fallback`
3. Requested bin has `empty_state.json`, no adjacent bins have data → return empty state
4. Nothing → `full_cohort_fallback`

---

### 7 — Lambda code-only update via S3 (no Docker rebuild required)

The Lambda `entrypoint.sh` downloads `lambda_function.py` from S3 on every cold start when `CODE_S3_KEY` is set. Use this for fast code-only iterations:
```bash
# Upload new code
aws s3 cp backend/lambda_function.py s3://pgxdatalake/gold/dashboard/code/lambda_function.py

# Trigger cold start by touching env vars
aws lambda update-function-configuration \
    --function-name pgx-risk-calculator \
    --environment 'Variables={S3_BUCKET=pgxdatalake,CODE_S3_KEY=gold/dashboard/code/lambda_function.py,PREFER_S3=false,PGX_RESULTS_BUCKET=pgxdatalake}'
```
Cold start takes ~20 s. This is ~10× faster than a full Docker rebuild for pure Python changes.

---

### 8 — `build_and_push.ps1` Docker build context

**Symptom:** Docker image digest unchanged after editing `lambda_function.py`; changes not picked up.  
**Root cause:** `build_and_push.ps1` passed `backend/` as the Docker build context, but the `Dockerfile` expects the **dashboard root** as context (it references `COPY backend/lambda_function.py`). With the wrong context, Docker can't see the changed file and reuses cached layers.  
**Fix:** The Dockerfile must be built from the dashboard root (`10_risk_dashboard/`):
```powershell
docker build -t pgx-risk-calculator:latest `
    -f "$DashboardRoot\backend\Dockerfile" `
    "$DashboardRoot"
```

## See Also

- **Backend README**: `../backend/README.md`
- **Frontend README**: `../frontend/README.md`
- **Main Dashboard README**: `../README.md`
