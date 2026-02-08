# Implementation Summary: Workflow and Model Updates

## Overview

This document summarizes the major updates made to align the workflow with the new pipeline requirements.

## Completed Updates

### 1. Workflow Script Updates ✅

**File**: Workflow runs via the five notebooks: `1_cohort_workflow.ipynb`, `2_feature_importance.ipynb`, `3_model_train_shap_ffa.ipynb`, `4_dashboard_visuals.ipynb`, `5_build_and_deploy.ipynb`. Legacy notebooks and scripts are in `archived/`.

**Changes**:
- Removed steps 5a (BupaR), 5b (FP-Growth), 5d (DTW) from main pipeline
- Pipeline: Feature Importance (Step 2) → Model Data (Step 4) → PGx (Step 5) → Final Model (Step 6) → SHAP (Step 7) → FFA (Step 8) → Dashboard (Step 9)
- Event filtering is in Step 1b; BupaR, DTW, and FP-Growth are dashboard-only visualizations
- Updated step descriptions to reflect new requirements

### 2. Final Model Script Updates ✅

**File**: `6_final_model/run_final_model.py`

**Major Changes**:

#### Feature Loading
- ✅ Removed FP-Growth, BupaR, DTW feature loading
- ✅ Only loads PGx features now
- ✅ Keeps patient-level aggregated features (drug/ICD/CPT encodings) - these are the "aggregated features"

#### Model Training
- ✅ Added XGBoost RF training in MC-CV loop
- ✅ Tracks metrics separately for XGBoost and XGBoost RF
- ✅ Both variants trained with same hyperparameters (except RF uses XGBRFClassifier)

#### Model Selection
- ✅ Implements selection logic based on **Recall** (primary) and **AUC-PR** (secondary)
- ✅ Compares XGBoost vs XGBoost RF after MC-CV
- ✅ Saves model selection metadata JSON with selection rationale

#### Model Export
- ✅ Exports **best CatBoost model** as `.cbm` binary (for SHAP analysis)
- ✅ Exports **best XGBoost model** as JSON (for FFA analysis)
- ✅ File naming: `best_catboost_model.cbm` and `best_xgboost_model.json`
- ✅ Model selection metadata saved to `model_selection_metadata.json`

### 3. SHAP Analysis Updates ✅

**File**: `8_shap_analysis/run_shap_analysis.py`

**Changes**:
- ✅ Added `_load_best_models()` function to load best CatBoost binary (.cbm)
- ✅ Updated `_fit_models_for_shap()` to use loaded best CatBoost model instead of refitting
- ✅ Checks multiple locations for best CatBoost model (final_model_json and model_outputs)
- ✅ Provides helpful error messages if best model not found

### 4. FFA Analysis Updates ✅

**File**: `8_ffa_analysis/run_full_ffa_analysis.py`

**Changes**:
- ✅ Updated to load `best_xgboost_model.json` instead of separate xgb/xgb_rf models
- ✅ Checks model selection metadata to determine which variant was selected
- ✅ Falls back to model_outputs location if not found in final_model_json
- ✅ Updated "all" model type to only analyze best XGBoost variant (not both)

## Model Selection Logic

The final model script now implements the following selection criteria:

1. **Primary Metric**: Recall (mean across MC-CV runs)
2. **Secondary Metric**: AUC-PR (mean across MC-CV runs) - used as tiebreaker

**Selection Process**:
```python
if xgb_recall_mean > xgb_rf_recall_mean:
    best_variant = "xgb"
elif xgb_recall_mean < xgb_rf_recall_mean:
    best_variant = "xgb_rf"
else:
    # Tie on recall, use AUC-PR
    if xgb_pr_auc_mean >= xgb_rf_pr_auc_mean:
        best_variant = "xgb"
    else:
        best_variant = "xgb_rf"
```

## Output Files

### Final Model Training Outputs

**Location**: `6_final_model/outputs/{cohort}/{age_band_fname}/`

1. **Model Selection Metadata**:
   - `{cohort}_{age_band_fname}_model_selection_metadata.json`
   - Contains: best variant, metrics, selection reason

2. **Best CatBoost Model** (for SHAP):
   - `final_model_json/{cohort}_{age_band_fname}_best_catboost_model.cbm` (binary)
   - `final_model_json/{cohort}_{age_band_fname}_best_catboost_model.json` (JSON)

3. **Best XGBoost Model** (for FFA):
   - `final_model_json/{cohort}_{age_band_fname}_best_xgboost_model.json`
   - Contains: model_type, variant, feature_names, trees, selection_metadata

4. **Model Outputs** (also copied to `model_outputs/`):
   - Same files copied for downstream consumption

### SHAP Analysis Inputs

- **Best CatBoost Binary**: `6_final_model/outputs/{cohort}/{age_band_fname}/final_model_json/{cohort}_{age_band_fname}_best_catboost_model.cbm`

### FFA Analysis Inputs

- **Best XGBoost JSON**: `6_final_model/outputs/{cohort}/{age_band_fname}/final_model_json/{cohort}_{age_band_fname}_best_xgboost_model.json`

## Workflow Pipeline

The updated workflow now follows this sequence:

1. **Step 3**: Feature Importance (idempotent - checks for existing aggregated results)
2. **Step 4**: Model Data (model events; event filtering done in Step 1b)
3. **Step 5**: PGx Feature Engineering (ONLY feature engineering step)
4. **Step 6**: Final Model Training
   - Uses aggregated features + PGx features
   - Trains CatBoost, XGBoost, XGBoost RF
   - Selects best XGBoost variant (recall/AUC-PR)
   - Exports best CatBoost binary and best XGBoost JSON
5. **Step 7**: SHAP Analysis (uses best CatBoost binary, produces SHAP importance)
6. **Step 8**: FFA Analysis (uses best XGBoost JSON + SHAP importance from Step 7). Rule selection: union of (1) first 100 matched rules, (2) random sample of 100 matched rules, and (3) all rules with SHAP > 0
7. **Step 9**: Risk Dashboard (BupaR/DTW/FP-Growth visualizations)
8. **Deploy**: S3/AWS Lambda as needed

## Testing Checklist

- [ ] Final model script trains XGBoost RF correctly
- [ ] Model selection logic works correctly
- [ ] Best CatBoost binary saved correctly
- [ ] Best XGBoost JSON saved correctly
- [ ] Model selection metadata saved correctly
- [ ] SHAP analysis can load best CatBoost binary
- [ ] FFA analysis can load best XGBoost JSON
- [ ] Workflow script runs end-to-end without errors

## Notes

- BupaR, DTW, and FP-Growth are no longer used as model features
- These methods are now used only for dashboard visualizations
- The workflow is streamlined to focus on PGx analysis as the primary feature engineering step
- Model selection is now automated based on performance metrics

