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
  - Query params: `cohort`, `age_band`
  - Returns: Causal factors and SHAP importance

- **`GET /visualizations/dtw`** - Get DTW visualization paths
  - Query params: `cohort`, `age_band`
  - Returns: S3 paths to DTW visualization images

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

## Deployment

See `../deployment/README.md` for deployment instructions.
