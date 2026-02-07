# Deployment

## Overview

Scripts and configurations for deploying the dashboard to AWS.

## Files

- **`docker_build.sh`** - Build and push Docker image to ECR
- **`prepare_lambda_dir.py`** - Prepare Lambda deployment package
- **`scripts/`** - Additional deployment helper scripts

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

3. Confirm there is **no API Gateway resource policy** (or WAF) that restricts `Origin` to only `https://jerome-dixon.io`; Lambda already sends `Access-Control-Allow-Origin: *`.

4. **Redeploy Lambda** after backend code changes (e.g. CORS): rebuild image, push to ECR, then update Lambda code (notebook "Update Lambda" or `aws lambda update-function-code --image-uri ...`).

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

## See Also

- **Backend README**: `../backend/README.md`
- **Frontend README**: `../frontend/README.md`
- **Main Dashboard README**: `../README.md`
