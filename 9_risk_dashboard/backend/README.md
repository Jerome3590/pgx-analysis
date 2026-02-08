# Backend API (Lambda Function)

## Overview

AWS Lambda function that provides the API backend for the risk dashboard. Handles model inference, metadata retrieval, and visualization data serving.

## Files

- **`lambda_function.py`** - Main Lambda handler with all API endpoints
- **`requirements.txt`** - Python dependencies
- **`Dockerfile`** - Docker container configuration for Lambda (ECR)
- **`lambda_api_template.py`** - Template for API Gateway integration

## API Endpoints

### Core Endpoints

- **`GET /metadata`** - Get valid codes for dropdowns
  - Query params: `cohort` (opioid_ed | non_opioid_ed)
  - Returns: Age bands and code lists (drugs, ICDs, CPTs)

- **`POST /risk`** - Calculate risk score
  - Body: `{cohort, age_band, drugs[], icds[], cpts[]}`
  - Returns: Risk score, risk band, model breakdown

- **`POST /risk/comparison`** - Compare risk scenarios
  - Body: `{base: {...}, scenarios: [...]}`
  - Returns: Risk scores for base and scenarios

- **`POST /pgx/card`** - Generate PGx patient card
  - Body: `{patient_id?, variants: [{gene, variants[]}]}`
  - Returns: PGx card data with drug-gene interactions

### Visualization Endpoints

- **`GET /visualizations/causal`** - Get causal analysis data
  - Query params: `cohort`, `age_band`; optional `drugs`, `icds`, `cpts` (each comma-separated) to filter to user-selected codes
  - Returns: Causal factors and SHAP importance (filtered to selected codes when provided), plus `filtered_by_codes`: boolean

- **`GET /visualizations/dtw`** - Get DTW visualization data
  - Query params: `cohort`, `age_band`
  - Returns: S3 paths to DTW images (`overview_image`, `sample_trajectories_image`), `metrics`, and when DTW feature data exists in S3 (`gold/feature_engineering/6_dtw/{cohort}/{age_band}/`):
    - **`routine_comparison`** – Chart data: outcome rate by trajectory intensity (Low/Medium/High event count), proxy for routine vs non-routine care
    - **`high_risk_trajectories`** – Chart data: outcome rate by trajectory archetype (quartiles of DTW distance or length)

- **`GET /visualizations/fpgrowth`** - Get FP-Growth visualization paths
  - Query params: `cohort`, `age_band`, `item_type`
  - Returns: S3 paths to FP-Growth visualization images

- **`GET /visualizations/bupar`** - Get BupaR visualization paths
  - Query params: `cohort`, `age_band`
  - Returns: S3 paths to BupaR visualization images

## Model Loading

Models are loaded from:
1. **Container filesystem** (`/var/task/models/`) - Primary source (ECR)
2. **S3** (`s3://pgxdatalake/gold/dashboard/models/`) - Fallback

## Environment Variables

- `PGX_RESULTS_BUCKET` - S3 bucket name (default: `pgxdatalake`)
- `MODEL_BASE_PATH` - Path to models in container (default: `/var/task/models`)
- `MODEL_CACHE_TTL` - Model cache TTL in seconds (default: `3600`)

## Generating visualization artifacts

Visualization **artifacts** (BupaR plots, DTW features/plots, FP-Growth itemsets/plots) are produced by pipeline scripts, not by Lambda. To (re)generate them from repo root, use:

- **Notebook:** `4_pgx_dashboard_visuals.ipynb` (run from repo root)
- **Script (VS Code Jupyter format):** `pgx_dashboard_visuals.py` (run as script or by cell with `# %%`)

Both run BupaR, DTW, and FP-Growth for configured cohorts/age bands and document Lambda/API Gateway endpoints. Upload outputs to S3 so Lambda can serve paths (e.g. `gold/feature_importance/`, `gold/fpgrowth/`, `gold/feature_engineering/6_dtw/`). **Redeploy the Lambda image** only when backend code changes (e.g. Causal tab now defaults to top 500 SHAP/FFA features when no user selection—redeploy to get that in production).

## Deployment

See `../deployment/README.md` for deployment instructions. To (re)create the API Gateway REST API and wire it to Lambda: `utility_scripts/create_api_gateway_pgx_risk_calculator.sh` (or `.ps1` on Windows).
