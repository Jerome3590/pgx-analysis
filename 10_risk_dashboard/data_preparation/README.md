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
1. Loads models from `6_final_model/outputs/{cohort}/{age_band_fname}/models/` (e.g. `xgboost.joblib`, `catboost.joblib`)
2. Reads MC-CV results from `6_final_model/outputs/{cohort}/{age_band_fname}/{cohort}_{age_band_fname}_mc_cv_results.csv` to select the best model per cohort/age_band (weight 1.0 for best, 0 for others)
3. Extracts feature schemas from training data: prefers **Parquet** at `6_final_model/outputs/.../inputs/model_train/final_features.parquet`, else the legacy CSV. Uses **DuckDB** when available for efficient reads and percentiles; falls back to pandas otherwise.
4. Writes to `10_risk_dashboard/outputs/models/` (used by `prepare_lambda_dir.py` and Docker build)

**Parallelism:** Uses all available CPU cores: age_bands are processed in parallel per cohort (ProcessPoolExecutor); S3 upload (when used) is parallel (ThreadPoolExecutor). The 2019 distribution step runs once per cohort and uses ProcessPoolExecutor over all cohort/age_band pairs.
**Progress:** Progress is printed as each age_band completes. The 2019 distribution subprocess has a 600s timeout.

**Outputs:** (under `10_risk_dashboard/outputs/models/{cohort}/{age_band_fname}/`)
- `catboost.joblib`, `xgboost.joblib`, optionally `xgboost_rf.joblib`
- `feature_schema.json` (includes `patient_bucket_thresholds` for n_events and n_drugs only — 33rd/67th percentiles from training — used for risk bucket low/medium/high; n_drugs is built in the PGx analysis step; n_pgx_drugs is a separate input)
- `risk_distribution_2019.json` (built idempotently after models; 2019 holdout predicted-probability distribution for the risk histogram)

At the end of each cohort run, the script invokes `prepare_risk_distribution_2019.py` for that cohort so the 2019 distribution is available in the same pipeline.

### `prepare_risk_distribution_2019.py`

Builds the 2019 holdout risk distribution used by the dashboard risk tab (idempotent). Run automatically as part of `prepare_models.py --all`, or standalone:

```bash
python prepare_risk_distribution_2019.py --cohort opioid_ed
python prepare_risk_distribution_2019.py --all
```

**Inputs:**
- `6_final_model/outputs/{cohort}/{age_band_fname}/inputs/model_test/final_features.parquet` (2019 test set)
- `10_risk_dashboard/outputs/models/{cohort}/{age_band_fname}/feature_schema.json` and model joblibs (must exist; run `prepare_models.py` first)

**Outputs:** (idempotent overwrite)
- `10_risk_dashboard/outputs/models/{cohort}/{age_band_fname}/risk_distribution_2019.json` (`bins`, `counts`, `n_patients`, `baseline_risk`, `risk_band_thresholds` (33rd/67th %ile of 2019 predictions), `description`, `bin_edges_pct`)

Lambda includes this in the POST /risk response as `dist` when present so the UI can show "Risk Distribution (2019 holdout)". When the user enters **no** Drug, ICD, or CPT codes, the API returns `risk_score` = `baseline_risk` (actual 2019 outcome rate); as the user adds codes, risk is the model's classification probability. **Risk band label (Low/Medium/High)** is based on **absolute cutoffs** in the API (low &lt;20%, medium 20–50%, high ≥50%), not the 33rd/67th percentiles in this file; those percentiles remain in the JSON for the histogram and reference only.

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
- Local: `../outputs/metadata/metadata_opioid_ed.json`, `metadata_non_opioid_ed.json`
- Dashboard deploy (5_build_and_deploy) uploads these to the dashboard bucket as `{prefix}/metadata/opioid_ed.json` and `{prefix}/metadata/non_opioid_ed.json` so the frontend can fetch them same-origin (no API call).

### `generate_metrics.py`

Generates `model_performance_metrics.json` for the Documentation tab from **existing** 6_final_model artifacts (no recomputation). Writes to `../outputs/metadata/` and uploads to S3 at `gold/dashboard/metadata/model_performance_metrics.json`. Lambda GET /metrics returns this prebuilt artifact from S3 (same pattern as other visuals); container bundle is an optional fallback.

**Usage:**
```bash
python generate_metrics.py
python generate_metrics.py --download-s3   # Fallback to S3 if local CSVs missing
```

**What it does:**
1. Reads existing `model_metrics_summary.csv` from `6_final_model/outputs/{cohort}/{age_band_fname}/` (or S3)
2. Aggregates into a single JSON (no recomputation)
3. Writes `model_performance_metrics.json` to `../outputs/metadata/` and uploads to S3 at `gold/dashboard/metadata/model_performance_metrics.json`

**Outputs:**
- Local: `../outputs/metadata/model_performance_metrics.json`
- S3 (pgxdatalake): `gold/dashboard/metadata/model_performance_metrics.json` (Lambda fallback)
- Dashboard deploy (5_build_and_deploy) uploads this file to the **dashboard bucket** at `{S3_DASHBOARD_PREFIX}/metadata/model_performance_metrics.json` so the frontend can fetch it same-origin (no API call; better performance).

### `prepare_cpic_data.py`

Prepares CPIC (Clinical Pharmacogenomics Implementation Consortium) data for PGx cards.

**Usage:**
```bash
python prepare_cpic_data.py
```

**What it does:**
1. Prefers official CPIC Excel (or downloads it); fallback: CSV from `5_pgx_analysis/data/` (read with **DuckDB** when available, else pandas)
2. Copies or converts to `cpic_gene-drug_pairs.xlsx` for the Lambda container
3. Writes a **Parquet** copy (`cpic_gene-drug_pairs.parquet`) alongside the Excel for efficient downstream use

**Outputs:**
- `outputs/cpic/cpic_gene-drug_pairs.xlsx` (Lambda container)
- `outputs/cpic/cpic_gene-drug_pairs.parquet` (when DuckDB or pandas+pyarrow available)

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
- `dashboard_data.json` (PHTS-style) and CSV under `--output-dir/{cohort}/{age_band_fname}/` (EC2: underscore; default output-dir = `10_risk_dashboard/outputs`)

**Dashboard S3 (optional):** Use `--upload-to-dashboard` to upload `dashboard_data.json` as `causal_data.json` to the dashboard bucket (`S3_DASHBOARD_BUCKET` / `S3_DASHBOARD_PREFIX`). The Causal Analysis tab then loads it via `GET /visualizations/causal` (prebuilt JSON pattern, same as DTW).

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
