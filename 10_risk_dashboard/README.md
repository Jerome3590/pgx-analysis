## 10_results – Final Results & Dashboard

This directory contains the **production-ready risk assessment dashboard** and deployment artifacts for the PGx analysis pipeline.

### Quick Overview

The dashboard provides two main capabilities:

1. **Risk Assessment Dashboard** - Predict opioid ED visit risk (ages 13-64) or polypharmacy risk (ages 65-114)
2. **PGx Patient Card Generator** - Generate pharmacogenomic cards from genetic variants

### Core Components

- **`index.html`** - Frontend dashboard (HTML/JavaScript)
- **`lambda_function.py`** - AWS Lambda handler (API endpoints)
- **`generate_metadata.py`** - Extract valid codes for dropdowns
- **`prepare_models.py`** - Package models for Lambda deployment
- **`prepare_cpic_data.py`** - Prepare CPIC data for PGx cards
- **`combine_shap_ffa_results.py`** - Combine SHAP and FFA analysis for consensus features
- **`Dockerfile`** - Container image for Lambda (ECR)
- **`requirements.txt`** - Python dependencies

### Quick Start

```bash
# 1. Generate metadata for dropdowns
python generate_metadata.py --all

# 2. Prepare models for deployment
python prepare_models.py --all

# 3. Prepare CPIC data
python prepare_cpic_data.py

# 4. Combine SHAP and FFA results (optional, for comprehensive explanations)
python combine_shap_ffa_results.py --cohort non_opioid_ed --age-band 65-74

# 5. Build Docker container
./docker_build.sh
```

### Documentation

For detailed documentation, see [`docs/Step10_Results/`](../docs/Step10_Results/):

**Main Documentation:**
- **[README_results_dashboard.md](../docs/Step10_Results/README_results_dashboard.md)** - Complete dashboard system overview
- **[README_results_value_proposition.md](../docs/Step10_Results/README_results_value_proposition.md)** - Business value and use cases
- **[README_results_deployment.md](../docs/Step10_Results/README_results_deployment.md)** - Complete deployment guide (architecture, steps, security)
- **[README_results_prediction.md](../docs/Step10_Results/README_results_prediction.md)** - Prediction workflow and technical details
- **[README_results_quickstart.md](../docs/Step10_Results/README_results_quickstart.md)** - Quick start guide for predictions

**Feature Documentation:**
- **[README_results_pgx_card.md](../docs/Step10_Results/README_results_pgx_card.md)** - PGx Patient Card feature
- **[README_results_ensemble.md](../docs/Step10_Results/README_results_ensemble.md)** - Ensemble model approach
- **[README_results_model_weights.md](../docs/Step10_Results/README_results_model_weights.md)** - Performance-based model weighting
- **[README_SHAP_FFA_COMBINATION.md](README_SHAP_FFA_COMBINATION.md)** - SHAP + FFA combination and consensus analysis

**Deployment Guides:**
- **[README_results_deployment_ecr.md](../docs/Step10_Results/README_results_deployment_ecr.md)** - Lambda ECR container deployment
- **[README_results_deployment_cpic.md](../docs/Step10_Results/README_results_deployment_cpic.md)** - CPIC data deployment

**Reference:**
- **[README_results_storage.md](../docs/Step10_Results/README_results_storage.md)** - Storage analysis and container sizing
- **[README_results_age_bands.md](../docs/Step10_Results/README_results_age_bands.md)** - Supported age bands and mappings

See [`docs/Step10_Results/README.md`](../docs/Step10_Results/README.md) for complete documentation index.

### Architecture

```
User Browser → S3 Static Site → API Gateway → Lambda (ECR) → Models/Data
```

### Key Features

- **Ensemble Models**: CatBoost + XGBoost + XGBoost RF with performance-based weighting
- **Age-Based Selection**: Automatically selects appropriate model based on age
- **Feature-Driven Inputs**: Dropdowns populated from actual feature importances
- **Privacy-First PGx Cards**: Anonymous, generic cards with optional patient ID
- **SHAP + FFA Combination**: Comprehensive patient-level explanations combining quantitative (SHAP) and logical (FFA) methods
- **Consensus Features**: High-confidence features identified by both SHAP and FFA analysis

### API Endpoints

- `GET /metadata` - Get valid age bands and valid codes for dropdowns.
  - Returns, per cohort, the supported age bands and code lists for the **Drugs / CPT / ICD** tabs.
  - The dashboard uses these to populate the cohort grid (e.g., 13-24, 25-44, 45-54, 65-74, 75-84, 85-94) and the tab-specific grids.
- `POST /risk` - Calculate risk score for a given `(cohort, age_band)` and selected codes.
  - Dashboard sends a JSON body like:
    ```json
    {
      "cohort": "opioid_ed",
      "age_band": "25-44",
      "drugs": ["DRUG_NAME_1", "DRUG_NAME_2"],
      "icds": ["F1120", "R51"],
      "cpts": ["80305", "99213"]
    }
    ```
  - Lambda builds a feature vector using `feature_schema.json` (prepared by `prepare_models.py`) and returns ensemble risk plus per-model breakdown for visualization.
- `POST /risk/comparison` - Compare risk scenarios
- `POST /pgx/card` - Generate PGx card

See [README_results_dashboard.md](../docs/Step10_Results/README_results_dashboard.md) for complete API documentation.
