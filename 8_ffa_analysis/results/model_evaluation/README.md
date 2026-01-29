# Model Evaluation Results

This directory contains comprehensive model evaluation results for all cohorts on **2019 test data**. All models were evaluated with calibration, feature importance, and SHAP analysis.

## Overview

**Total Evaluations: 14** (7 cohorts × 2 model types)

All models were evaluated on **2019 test data** (unseen during training) with:
- **Calibration**: Isotonic regression calibration for probability calibration
- **Performance Metrics**: Recall, Precision, F1, Accuracy, ROC-AUC, PR-AUC, LogLoss, Brier Score
- **Feature Importance**: Model-specific feature importance scores
- **SHAP Analysis**: SHAP values computed on 1000 samples (memory-efficient using DuckDB)

## Completion Status

### Opioid ED Cohort (`opioid_ed`)
- **13-24**: ✅ Complete (XGBoost + CatBoost)
- **25-44**: ✅ Complete (XGBoost + CatBoost)
- **45-54**: ✅ Complete (XGBoost + CatBoost)
- **55-64**: ✅ Complete (XGBoost + CatBoost)

### Non-Opioid ED Cohort (`non_opioid_ed`)
- **65-74**: ✅ Complete (XGBoost + CatBoost)
- **75-84**: ✅ Complete (XGBoost + CatBoost)
- **85-94**: ✅ Complete (XGBoost + CatBoost)

**All cohorts evaluated on 2026-01-14**

## Files Structure

For each cohort/age_band/model_type combination, the following files are generated:

### Summary File
- **test_evaluation_summary.csv** - Combined summary of all evaluations

### Per-Model Files (14 models × 5 files = 70 files)
- **`{cohort}_{age_band}_{model_type}_test_metrics.json`** - Performance metrics (raw and calibrated)
- **`{cohort}_{age_band}_{model_type}_test_feature_importance.csv`** - Feature importance scores
- **`{cohort}_{age_band}_{model_type}_test_shap_importance.csv`** - SHAP-based feature importance
- **`{cohort}_{age_band}_{model_type}_test_shap_values.parquet`** - SHAP values for 1000 samples
- **`{cohort}_{age_band}_{model_type}_test_predictions.parquet`** - Raw and calibrated predictions

## Performance Summary

### Overall Performance (Calibrated)

| Cohort | Age Band | Model | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|--------|----------|-------|--------|-----------|----|---------|--------|
| **non_opioid_ed** | 65-74 | XGBoost | **0.974** | 0.829 | 0.896 | **0.987** | **0.930** |
| **non_opioid_ed** | 65-74 | CatBoost | **0.972** | 0.824 | 0.892 | **0.986** | 0.925 |
| **non_opioid_ed** | 75-84 | XGBoost | **0.977** | **0.904** | 0.939 | **0.989** | **0.960** |
| **non_opioid_ed** | 75-84 | CatBoost | **0.975** | 0.899 | 0.936 | **0.988** | 0.956 |
| **non_opioid_ed** | 85-94 | XGBoost | **0.971** | **0.924** | 0.947 | **0.984** | **0.967** |
| **non_opioid_ed** | 85-94 | CatBoost | **0.969** | 0.920 | 0.944 | **0.983** | 0.962 |
| **opioid_ed** | 13-24 | XGBoost | 0.851 | 0.619 | 0.717 | **0.971** | **0.867** |
| **opioid_ed** | 13-24 | CatBoost | 0.853 | 0.522 | 0.647 | **0.961** | 0.838 |
| **opioid_ed** | 25-44 | XGBoost | 0.856 | 0.577 | 0.689 | **0.954** | **0.841** |
| **opioid_ed** | 25-44 | CatBoost | 0.855 | 0.532 | 0.656 | **0.947** | 0.823 |
| **opioid_ed** | 45-54 | XGBoost | **0.916** | 0.520 | 0.664 | **0.966** | **0.861** |
| **opioid_ed** | 45-54 | CatBoost | 0.881 | 0.489 | 0.629 | **0.955** | 0.827 |
| **opioid_ed** | 55-64 | XGBoost | 0.862 | **0.667** | 0.752 | **0.973** | **0.888** |
| **opioid_ed** | 55-64 | CatBoost | 0.829 | 0.644 | 0.725 | **0.957** | 0.850 |

### Key Performance Insights

1. **Non-Opioid ED Models**: Excellent performance across all age bands
   - ROC-AUC: 0.984-0.989 (outstanding discrimination)
   - PR-AUC: 0.930-0.967 (excellent precision-recall balance)
   - Recall: 0.969-0.977 (captures most positive cases)
   - Precision: 0.824-0.924 (low false positive rate)

2. **Opioid ED Models**: Strong performance with age-dependent patterns
   - ROC-AUC: 0.947-0.973 (very good discrimination)
   - PR-AUC: 0.823-0.888 (good precision-recall balance)
   - Recall: 0.829-0.916 (good sensitivity)
   - Precision: 0.489-0.667 (moderate precision, expected for rare events)

3. **Model Comparison**:
   - **XGBoost** generally performs slightly better than CatBoost
   - Both models show consistent performance across cohorts
   - Calibration improves recall at the cost of precision (as expected)

## Key Findings

### Universal Top Features (Across Cohorts)

Based on feature importance and SHAP analysis:

1. **n_events** - Number of events (consistently top feature)
2. **pgx_num_drugs** - PGx drug count (high importance)
3. **item_drug_GABAPENTIN** - Present in most cohorts
4. **item_drug_NARCAN** - Opioid-related (opioid_ed cohorts)
5. **item_drug_BUPRENORPHINE_HYDROCHLORI** - Opioid-related (opioid_ed cohorts)

### Age-Dependent Patterns

- **Younger cohorts (13-24, 25-44)**: More psychiatric medications (TRAZODONE, QUETIAPINE, SERTRALINE)
- **Middle-aged (45-54, 55-64)**: More chronic conditions (hypertension, pain management)
- **Older cohorts (65-94)**: More preventive care (vaccines, screenings)

### Calibration Impact

- **Before calibration**: Models often had high precision but lower recall
- **After calibration**: Improved recall (captures more true positives) with adjusted thresholds
- **Optimal thresholds**: Range from 0.11-0.40 depending on cohort and model

## Data Schema

### Metrics JSON Files
```json
{
  "cohort": "opioid_ed",
  "age_band": "13-24",
  "model_type": "xgboost",
  "n_test_samples": 6640,
  "n_features": 11056,
  "optimal_threshold": 0.1286,
  "recall_calibrated": 0.8513,
  "precision_calibrated": 0.6189,
  "f1_calibrated": 0.7167,
  "roc_auc_calibrated": 0.9708,
  "pr_auc_calibrated": 0.8672,
  ...
}
```

### Feature Importance CSV
Columns:
- `feature` - Feature name
- `importance_gain` / `importance` - Feature importance score
- `importance_gain_norm` / `importance_norm` - Normalized importance (0-1)

### SHAP Importance CSV
Columns:
- `feature` - Feature name
- `mean_abs_shap` - Mean absolute SHAP value (importance)
- `mean_shap` - Mean SHAP value (direction)

### SHAP Values Parquet
- Rows: 1000 samples (or all if dataset < 1000)
- Columns: One per feature (SHAP values)
- Index: Sample indices

### Predictions Parquet
Columns:
- `y_true` - True labels
- `y_proba_raw` - Raw model probabilities
- `y_proba_calibrated` - Calibrated probabilities
- `y_pred_raw` - Raw predictions (threshold 0.5)
- `y_pred_calibrated` - Calibrated predictions (optimal threshold)

## Reading the Files

### Using Python with Pandas:
```python
import pandas as pd
import json

# Load summary
summary = pd.read_csv('test_evaluation_summary.csv')

# Load metrics for specific cohort
with open('opioid_ed_13_24_xgboost_test_metrics.json', 'r') as f:
    metrics = json.load(f)

# Load feature importance
fi_df = pd.read_csv('opioid_ed_13_24_xgboost_test_feature_importance.csv')

# Load SHAP values
shap_df = pd.read_parquet('opioid_ed_13_24_xgboost_test_shap_values.parquet')

# Load predictions
pred_df = pd.read_parquet('opioid_ed_13_24_xgboost_test_predictions.parquet')
```

### Using DuckDB (Memory-Efficient):
```python
import duckdb

con = duckdb.connect()

# Query SHAP values without loading full dataset
shap_df = con.execute("""
    SELECT feature_1, feature_2, ...
    FROM read_parquet('opioid_ed_13_24_xgboost_test_shap_values.parquet')
    WHERE mean_abs_shap > 0.1
""").df()

con.close()
```

## Related Documents

- **FFA_RESULTS_SUMMARY.md** - Causal factors and interactions analysis
- **README.md** (parent directory) - FFA analysis results overview
- **MODEL_PERFORMANCE_SUMMARY.md** - Model training methodology and expected metrics

## Technical Notes

### Calibration Method
- **Isotonic Regression**: Non-parametric calibration that maps raw probabilities to calibrated probabilities
- **Optimal Threshold**: Selected using ROC curve (maximizes TPR - FPR)
- **Impact**: Improves recall while maintaining reasonable precision

### SHAP Analysis
- **Method**: TreeExplainer for both XGBoost and CatBoost
- **Samples**: 1000 samples (or all if dataset < 1000)
- **Memory Optimization**: Uses DuckDB for efficient sampling from large datasets
- **Purpose**: Provides model-agnostic feature importance and explanations

### Model Formats
- **XGBoost**: Loaded from `.ubj` (Universal Binary JSON) or `.joblib` files
- **CatBoost**: Loaded from `.cbm` (CatBoost Model) files
- **Test Data**: Loaded from S3 parquet files (2019 test data)

## Last Updated
2026-01-14

## Evaluation Script
Results generated using `utility_scripts/evaluate_models_test_data.py` with DuckDB memory optimization.
