# Dashboard Build Requirements

This document outlines what outputs are needed from each cohort/age_band combination to build the risk dashboard.

## Overview

The dashboard build process (`build_dashboard.sh`) requires outputs from multiple pipeline steps. **All cohorts and age bands must be completed before the dashboard can be built.**

### Required Cohorts and Age Bands

- **opioid_ed**: 13-24, 25-44, 45-54, 55-64 (4 age bands)
- **non_opioid_ed**: 65-74, 75-84, 85-94 (3 age bands)

**Total: 7 cohort/age_band combinations required**

## Required Outputs by Step

### Step 3: Feature Importance (Required for Metadata)

**Location:** `3_feature_importance/outputs/`

**Required Files:**
- `{cohort}_{age_band}_aggregated_feature_importance.csv`
  - Used by: `generate_metadata.py`
  - Purpose: Extract top ICD/CPT/Drug codes for dashboard metadata

**Example:**
```
3_feature_importance/outputs/opioid_ed_13_24_aggregated_feature_importance.csv
```

---

### Step 6: Final Model Training (Required for Models)

**Location:** `6_final_model/outputs/{cohort}/{age_band}/`

**Required Files:**

1. **Model Files:**
   - `models/catboost.joblib` (or `models/catboost.cbm`)
   - `models/xgboost.joblib` (or `models/xgboost_model.ubj`)
   - Used by: `prepare_models.py`
   - Purpose: Trained models for risk prediction

2. **Training Data:**
   - `{cohort}_{age_band}_train_final_features_no_leakage.csv`
   - Used by: `prepare_models.py`
   - Purpose: Extract feature schema and default values

3. **Model Performance Metrics:**
   - `models/{cohort}_{age_band}_mc_cv_results.csv`
   - Used by: `prepare_models.py`
   - Purpose: Calculate model weights based on MC-CV performance

**Example:**
```
6_final_model/outputs/opioid_ed/13_24/
├── models/
│   ├── catboost.joblib
│   ├── xgboost.joblib
│   └── opioid_ed_13_24_mc_cv_results.csv
└── opioid_ed_13_24_train_final_features_no_leakage.csv
```

---

### Step 7: SHAP Analysis (Optional but Recommended)

**Location:** `7_shap_analysis/outputs/{cohort}/{age_band}/`

**Required Files:**
- `{cohort}_{age_band}_shap_global_importance_xgboost.csv`
- `{cohort}_{age_band}_shap_sample_values_xgboost.parquet`
- Used by: Dashboard for feature explanations
- Purpose: SHAP values for individual patient explanations

**Example:**
```
7_shap_analysis/outputs/opioid_ed/13_24/
├── opioid_ed_13_24_shap_global_importance_xgboost.csv
└── opioid_ed_13_24_shap_sample_values_xgboost.parquet
```

---

### Step 8: FFA Analysis (Optional but Recommended)

**Location:** `8_ffa_analysis/outputs/{cohort}/{age_band}/xgboost/`

**Required Files:**
- `axp_explanations.parquet`
- `feature_importance_axp.parquet`
- `causal_importance.parquet` (optional)
- Used by: Dashboard for causal explanations
- Purpose: Formal Feature Attribution explanations and causal importance

**Example:**
```
8_ffa_analysis/outputs/opioid_ed/13_24/xgboost/
├── axp_explanations.parquet
├── feature_importance_axp.parquet
└── causal_importance.parquet
```

---

## Minimum Requirements

**To build a basic dashboard (models only):**
- ✅ Step 3: Feature importance CSV
- ✅ Step 6: Model files + training data + MC-CV results

**To build a full-featured dashboard (with explanations):**
- ✅ Step 3: Feature importance CSV
- ✅ Step 6: Model files + training data + MC-CV results
- ✅ Step 7: SHAP outputs
- ✅ Step 8: FFA outputs

---

## Build Process

1. **Prepare Models:**
   ```bash
   python 9_risk_dashboard/prepare_models.py --cohort opioid_ed
   ```
   - Reads from: Step 6 outputs
   - Creates: `10_results/models/{cohort}/{age_band}/`

2. **Generate Metadata:**
   ```bash
   python 9_risk_dashboard/generate_metadata.py --cohort opioid_ed
   ```
   - Reads from: Step 3 outputs
   - Creates: `10_results/metadata/metadata_{cohort}.json`

3. **Build Dashboard:**
   ```bash
   bash utility_scripts/build_dashboard.sh
   ```
   - Processes all available cohorts
   - Skips missing cohorts gracefully

---

## File Structure Summary

For each cohort/age_band combination, you need:

```
{cohort}/{age_band}/
├── Step 3: Feature Importance
│   └── {cohort}_{age_band}_aggregated_feature_importance.csv
│
├── Step 6: Final Model
│   ├── models/
│   │   ├── catboost.joblib
│   │   ├── xgboost.joblib
│   │   └── {cohort}_{age_band}_mc_cv_results.csv
│   └── {cohort}_{age_band}_train_final_features_no_leakage.csv
│
├── Step 7: SHAP Analysis (optional)
│   ├── {cohort}_{age_band}_shap_global_importance_xgboost.csv
│   └── {cohort}_{age_band}_shap_sample_values_xgboost.parquet
│
└── Step 8: FFA Analysis (optional)
    └── xgboost/
        ├── axp_explanations.parquet
        ├── feature_importance_axp.parquet
        └── causal_importance.parquet
```

---

## Checking Availability

To check what's available for dashboard build:

```bash
# Check Step 6 outputs (required)
ls -la 6_final_model/outputs/opioid_ed/*/models/*.joblib

# Check Step 3 outputs (required for metadata)
ls -la 3_feature_importance/outputs/*aggregated_feature_importance.csv

# Check Step 7 outputs (optional)
ls -la 7_shap_analysis/outputs/opioid_ed/*/*shap*.csv

# Check Step 8 outputs (optional)
ls -la 8_ffa_analysis/outputs/opioid_ed/*/xgboost/*.parquet
```

---

## Notes

- **All cohorts and age bands are required** - the build will fail if any are missing
- Step 6 outputs are **required** - without them, the dashboard cannot make predictions
- Step 3 outputs are **required** for metadata generation
- Steps 7 and 8 are **optional** but recommended for full explanation features
- The build script validates that all required cohorts/age_bands are present before building
