# Pipeline Data Locations - S3 and Local

## Summary

This document maps where each pipeline step stores and expects to find data files, both in S3 and locally.

## Step 6: Final Model Training

### Expected Local Locations:
1. **Primary (Parquet)**: `6_final_model/outputs/{cohort}/{age_band}/inputs/model_train/final_features.parquet`
2. **Primary (CSV)**: `6_final_model/outputs/{cohort}/{age_band}/{cohort}_{age_band}_train_final_features_no_leakage.csv`

### Expected S3 Locations:
1. **Primary (Parquet)**: `s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/inputs/model_train/final_features.parquet`
2. **Primary (CSV)**: `s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/inputs/model_train/{cohort}_{age_band}_train_final_features_no_leakage.csv`

### Current S3 Status:
- **pgxdatalake bucket:**
  - ✅ Model JSONs exist: `gold/final_model/opioid_ed/13-24/opioid_ed_13_24_best_xgboost_model.json`
  - ❌ Training data NOT found in: `gold/final_model/opioid_ed/13-24/inputs/` (directory doesn't exist)
  - ❌ Training data NOT found in any alternative locations checked
- **pgx-repository bucket:**
  - ❌ Training data NOT found in any checked locations
  - ✅ Bucket accessible but no training data files found

## Step 7: SHAP Analysis

### Expected Local Locations:
1. **Global Importance**: `7_shap_analysis/outputs/{cohort}/{age_band}/{cohort}_{age_band}_shap_global_importance_{model_type}.csv`
2. **Sample Values**: `7_shap_analysis/outputs/{cohort}/{age_band}/{cohort}_{age_band}_shap_sample_values_{model_type}.parquet`

### Expected S3 Locations:
1. **Global Importance**: `s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/{cohort}_{age_band}_shap_global_importance_{model_type}.csv`
2. **Sample Values**: `s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/{cohort}_{age_band}_shap_sample_values_{model_type}.parquet`

### Current S3 Status:
- ✅ Both files exist and are accessible

## Step 8: FFA Analysis

### Expected Local Locations:
1. **AXP Explanations**: `8_ffa_analysis/outputs/{cohort}/{age_band}/{model_type}/axp_explanations.parquet`
2. **Feature Importance**: `8_ffa_analysis/outputs/{cohort}/{age_band}/{model_type}/feature_importance_axp.parquet`
3. **Causal Importance**: `8_ffa_analysis/outputs/{cohort}/{age_band}/{model_type}/causal_importance.parquet`

### Expected S3 Locations:
1. **AXP Explanations**: `s3://pgxdatalake/gold/ffa_analysis/{cohort}/{age_band}/{model_type}/axp_explanations.parquet`
2. **Feature Importance**: `s3://pgxdatalake/gold/ffa_analysis/{cohort}/{age_band}/{model_type}/feature_importance_axp.parquet`
3. **Causal Importance**: `s3://pgxdatalake/gold/ffa_analysis/{cohort}/{age_band}/{model_type}/causal_importance.parquet`

### Current S3 Status:
- ✅ All files exist and are accessible

## Data Folder Locations (Alternative)

### Local `data/` Directory Structure:
The code now checks multiple locations in the `data/` folder:

1. **Direct**: `data/{cohort}/{age_band}/final_features.parquet`
2. **Direct CSV**: `data/{cohort}/{age_band}/{cohort}_{age_band}_train_final_features_no_leakage.csv`
3. **Root CSV**: `data/{cohort}_{age_band}_train_final_features_no_leakage.csv`
4. **Cohorts Structure**: `data/cohorts/cohort_name={cohort}/event_year={year}/age_band={age_band}/final_features.parquet`
5. **Gold Cohorts**: `data/gold_cohorts/cohort_name={cohort}/{cohort}_{age_band}_train_final_features_no_leakage.csv`
6. **Gold Cohorts Root**: `data/gold_cohorts/{cohort}_{age_band}_train_final_features_no_leakage.csv`

### Current Local Status:
- ❌ Training data NOT found in any of the expected `data/` locations
- ✅ Raw cohort data exists: `data/gold_cohorts/cohort_name=opioid_ed/event_year={year}/age_band=13-24/cohort.parquet`
- ⚠️ Raw cohort data is NOT the processed training data (needs feature engineering)

## Issue: Missing Training Data File

### Problem:
The training data file (`final_features.parquet` or `{cohort}_{age_band}_train_final_features_no_leakage.csv`) is **NOT** found in:
- ❌ S3: `gold/final_model/opioid_ed/13-24/inputs/` (directory doesn't exist)
- ❌ Local: `6_final_model/outputs/opioid_ed/13_24/inputs/model_train/` (directory exists but empty)
- ❌ Local: Any location in `data/` folder

### What Exists:
- ✅ Raw cohort data: `data/gold_cohorts/cohort_name=opioid_ed/event_year={year}/age_band=13-24/cohort.parquet`
- ✅ Model JSONs: `6_final_model/outputs/opioid_ed/13_24/final_model_json/opioid_ed_13_24_best_xgboost_model.json`
- ✅ SHAP outputs: Both local and S3
- ✅ FFA outputs: Both local and S3

### Solution Options:

1. **Run Step 6** to generate the training data file:
   - Step 6 should create the processed training data with features
   - Should save to: `6_final_model/outputs/{cohort}/{age_band}/inputs/model_train/final_features.parquet`
   - Should upload to: `s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/inputs/model_train/final_features.parquet`

2. **Check if training data is in a different S3 location**:
   - May be in: `gold/cohorts/{cohort}/{age_band}/`
   - May be in: `gold/gold_cohorts/{cohort}/{age_band}/`
   - May be in: `gold/cohorts_model_data/{cohort}/{age_band}/` (checked - not found)

3. **Create training data from raw cohort data**:
   - Use each cohort’s `model_events` (each has within-cohort target and control; control = target=0, not the other cohort).
   - Apply feature engineering pipeline and save to expected location.

## Code Configuration

The code in `run_full_ffa_analysis.py` and `download_and_test_ffa.py` now checks all these locations in order:
1. Primary local location (6_final_model outputs)
2. Alternative local locations (data folder variants including data/gold_cohorts)
3. Primary S3 location (gold/final_model)
4. Alternative S3 locations (gold/cohorts, gold/gold_cohorts, gold/cohorts_model_data)

If the file is not found, the analysis will fail with a clear error message listing all checked locations.
