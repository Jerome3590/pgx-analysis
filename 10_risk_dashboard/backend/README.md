# Backend API (Lambda Function)

## Overview

Lambda receives **user input** (cohort, age_band, model/feature selections) and **filters** only—it does not process or generate visualization data. All visuals are prebuilt on EC2 and saved to S3; Lambda returns URLs to those prebuilt assets. Risk inference uses the ensemble with user-provided features; visualization endpoints return prebuilt S3 URLs filtered by cohort/age_band. No analytics or chart building runs in Lambda.

## Files

- **`lambda_function.py`** - Main Lambda handler with all API endpoints
- **`requirements.txt`** - Python dependencies
- **`Dockerfile`** - Docker container configuration for Lambda (ECR)
- **`lambda_api_template.py`** - Template for API Gateway integration

## API Endpoints

### Core Endpoints

- **`GET /metadata`** - Get valid codes for dropdowns (filter by cohort). **Fallback only:** the frontend loads from same-origin `metadata/{cohort}.json` (deployed with the dashboard). If missing, the frontend calls this endpoint.
  - Query params: `cohort` (opioid_ed | non_opioid_ed)
  - Returns: Age bands and code lists (drugs, ICDs, CPTs)

- **`POST /risk`** - Risk score from ensemble, filtered by user-selected cohort and features
  - Body: `{cohort, age_band, drugs[], icds[], cpts[]}`
  - Returns: Risk score, risk band, model breakdown

- **`POST /risk/comparison`** - Compare risk for user-provided scenarios (filter by selection)
  - Body: `{base: {...}, scenarios: [...]}`
  - Returns: Risk scores for base and scenarios

- **`POST /pgx/card`** - Generate PGx patient card
  - Body: `{patient_id?, variants: [{gene, variants[]}]}`
  - Returns: PGx card data with drug-gene interactions

- **`GET /metrics`** - Return prebuilt model performance metrics (Documentation tab). **Fallback only:** the frontend loads metrics from the same-origin static asset `metadata/model_performance_metrics.json` (deployed with the dashboard to the dashboard bucket). If that file is missing (e.g. local dev), the frontend calls this endpoint. Lambda reads from S3 (`gold/dashboard/metadata/model_performance_metrics.json`) or container bundle; no recomputation.

### Visualization Endpoints (filter only; return prebuilt S3 URLs)

- **`GET /visualizations/causal`** - Return causal/SHAP data filtered by user selection
  - Query params: `cohort`, `age_band`; optional `drugs`, `icds`, `cpts` to filter to selected codes
  - Returns: Prebuilt causal factors and SHAP importance (filtered when codes provided)

- **`GET /visualizations/dtw`** - Return URLs to prebuilt DTW assets (no processing)
  - Query params: `cohort`, `age_band`
  - Returns: `overview_image`, `sample_trajectories_image`, `chart_data_url`, `sequence_heatmap_url` (S3 URLs)

- **`GET /visualizations/fpgrowth`** - Return URLs to prebuilt FP-Growth assets (drug names only)
  - Query params: `cohort`, `age_band` (item_type is fixed to `drug_name`)
  - Returns: S3 URLs to drug itemsets, network HTML, and itemsets JSON when present

- **`GET /visualizations/bupar`** - Return URLs to prebuilt BupaR assets
  - Query params: `cohort`, `age_band`
  - Returns: S3 URLs to BupaR plot images; process matrix type-pair is **Drug × Drug** only (`process_matrix_drug_drug.png`)

- **`GET /visualizations/bupar/activity_frequency`** - Return activity frequency JSON for bar charts
  - Query params: `cohort`, `age_band`
  - Returns: `{ overall, pre_target, post_target }` (each with `year_labels` and `data`); frontend builds Chart.js bar charts with year filter

## Model Loading

Models are loaded from:
1. **Container filesystem** (`/var/task/models/`) - Primary source (ECR)
2. **S3** (`s3://pgxdatalake/gold/dashboard/models/`) - Fallback

## Environment Variables

- `PGX_RESULTS_BUCKET` - S3 bucket for data/models (default: `pgxdatalake`)
- `S3_DASHBOARD_BUCKET` - Bucket where the dashboard frontend is deployed; FP-Growth assets are uploaded here (default: `jerome-dixon.io`)
- `S3_DASHBOARD_PREFIX` - Key prefix for the dashboard app in that bucket (default: `vcu/pgx-risk-calculator`). FP-Growth, BupaR, and DTW URLs use `{prefix}/fpgrowth/`, `{prefix}/bupar/`, and `{prefix}/dtw/{cohort}/{age_band}/plots/`.
- `MODEL_BASE_PATH` - Path to models in container (default: `/var/task/models`)
- `MODEL_CACHE_TTL` - Model cache TTL in seconds (default: `3600`)

**Empty visualization responses:** Notebook 4 (**4_dashboard_visuals.ipynb**) and **run_dashboard_visuals.py** both **build** the artifacts (BupaR, DTW, FP-Growth) and **upload** them to the dashboard bucket using the same `S3_DASHBOARD_BUCKET` and `S3_DASHBOARD_PREFIX`. Lambda reads from that same bucket/prefix. So in the normal flow, notebook 4 builds and uploads; the API then returns URLs to those objects. If you still see empty responses: (1) Lambda may not have permission to read the bucket (403) — we return 200 with empty payload instead of 500 so the frontend shows "not available"; (2) Lambda env `S3_DASHBOARD_BUCKET` / `S3_DASHBOARD_PREFIX` must match the bucket/prefix used when running the notebook on EC2; (3) **PGx Cohort network visuals**: Built in notebook 4 (fetch_vip_reports + build_network_topology); each build uploads to the dashboard bucket (same as BupaR/DTW/FP-Growth). Notebook 5 (Step 6: Sync Dashboard Frontend) syncs `10_risk_dashboard/visualizations/cohort_pgx/` to S3 when you deploy. Lambda returns `network_topology_url` when `{prefix}/cohort_pgx/networks/{cohort}/{age_band_fname}/network_topology.html` exists.

## Generating visualization artifacts

All visualization **artifacts** (BupaR plots, DTW images and chart data, FP-Growth itemsets/plots) are **built and uploaded to S3** by notebook 4 (or run_dashboard_visuals.py) on EC2. The scripts use `S3_DASHBOARD_BUCKET` and `S3_DASHBOARD_PREFIX`; Lambda uses the same env vars to return URLs. The API returns only URLs to these prebuilt assets (no computation at request time). To (re)generate from repo root:

- **Notebook:** `4_dashboard_visuals.ipynb` (run from repo root)
- **Script (VS Code Jupyter format):** `pgx_dashboard_visuals.py` (run as script or by cell with `# %%`)

Both run BupaR, DTW, and FP-Growth for configured cohorts/age bands. **FP-Growth**, **BupaR**, and **DTW** assets are uploaded to the **dashboard bucket** (e.g. `jerome-dixon.io`) under `{S3_DASHBOARD_PREFIX}/fpgrowth/`, `{S3_DASHBOARD_PREFIX}/bupar/`, and `{S3_DASHBOARD_PREFIX}/dtw/`; the dashboard loads them directly from S3 (or via API URL responses). **Redeploy the Lambda image** only when backend code changes.

## Deployment

See `../deployment/README.md` for deployment instructions. To (re)create the API Gateway REST API and wire it to Lambda: `utility_scripts/create_api_gateway_pgx_risk_calculator.sh` (or `.ps1` on Windows).
