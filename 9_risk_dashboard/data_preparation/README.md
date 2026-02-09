# Data Preparation

## Overview

Scripts to prepare models, metadata, and data files for dashboard deployment.

## Scripts

### `prepare_models.py`

Packages trained models for Lambda deployment.

**Usage:**
```bash
python prepare_models.py --cohort opioid_ed
python prepare_models.py --cohort non_opioid_ed
python prepare_models.py --all
```

**What it does:**
1. Loads models from `6_final_model/outputs/`
2. Extracts feature schemas from training data
3. Calculates model weights based on MC-CV performance
4. Saves models and schemas to `../outputs/models/`

**Outputs:**
- `{cohort}/{age_band}/catboost.joblib`
- `{cohort}/{age_band}/xgboost.joblib`
- `{cohort}/{age_band}/xgboost_rf.joblib`
- `{cohort}/{age_band}/feature_schema.json`

### `generate_metadata.py`

Generates metadata JSON files with valid codes for dropdowns.

**Usage:**
```bash
python generate_metadata.py --cohort opioid_ed
python generate_metadata.py --cohort non_opioid_ed
python generate_metadata.py --all
```

**What it does:**
1. Loads Step 3b `cohort_feature_importance` files (or Step 3 aggregated as fallback)
2. Extracts codes (drugs, ICDs, CPTs) from feature importances
3. Creates metadata JSON with code lists and importance scores
4. Saves to `../outputs/metadata/`

**Outputs:**
- `metadata_opioid_ed.json`
- `metadata_non_opioid_ed.json`

### `generate_metrics.py`

Generates `model_performance_metrics.json` for the Documentation tab (model performance and at-risk identification by cohort/age band). The file is written to `../outputs/metadata/` and bundled into the Lambda/ECR image so the dashboard serves it without S3 calls.

**Usage:**
```bash
python generate_metrics.py
python generate_metrics.py --download-s3   # Fallback to S3 if local CSVs missing
```

**What it does:**
1. Reads `model_metrics_summary.csv` from `6_final_model/outputs/{cohort}/{age_band}/` (or S3)
2. Aggregates into a single JSON
3. Writes `model_performance_metrics.json` to `../outputs/metadata/`

**Outputs:**
- `model_performance_metrics.json` (bundled at `/var/task/metadata/` in the Lambda container)

### `prepare_cpic_data.py`

Prepares CPIC (Clinical Pharmacogenomics Implementation Consortium) data for PGx cards.

**Usage:**
```bash
python prepare_cpic_data.py
```

**What it does:**
1. Loads CPIC master Excel file
2. Processes gene-drug pairs
3. Prepares data for Lambda function
4. Saves to `../outputs/cpic/`

**Outputs:**
- CPIC data files for Lambda container

### `combine_shap_ffa_results.py`

Combines SHAP and FFA analysis results for consensus features.

**Usage:**
```bash
python combine_shap_ffa_results.py --cohort opioid_ed --age-band 25-44
```

**What it does:**
1. Loads SHAP importance from Step 7
2. Loads FFA causal importance from Step 8
3. Identifies consensus features (high importance in both)
4. Creates combined analysis files

**Outputs:**
- Combined SHAP/FFA analysis files

## Input Sources

- **Models**: `6_final_model/outputs/{cohort}/{age_band}/`
- **Feature Importances**: `3b_feature_importance_eda/outputs/{cohort}/{age_band}/` (Step 3b)
- **SHAP Results**: `s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/`
- **FFA Results**: `s3://pgxdatalake/gold/ffa_analysis/{cohort}/{age_band}/`

## Output Locations

All outputs go to `../outputs/`:
- `outputs/models/` - Prepared models for Lambda
- `outputs/metadata/` - Metadata JSON files
- `outputs/cpic/` - CPIC data files

These outputs are then packaged into the Lambda container during deployment.
