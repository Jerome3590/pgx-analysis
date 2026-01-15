# Model Evaluation on Test Data (2019)

This directory contains scripts to evaluate models on 2019 test data with calibration, performance metrics, feature importance, and SHAP analysis.

## Scripts

### Main Evaluation Script
- **`evaluate_models_test_data.py`**: Evaluates a single cohort/age_band combination
  - Loads models from S3
  - Loads 2019 test data from S3
  - Evaluates performance (Recall, AUC-PR, etc.)
  - Calibrates models using isotonic regression
  - Computes feature importance
  - Computes SHAP values
  - Saves results

### Wrapper Scripts (Run Each Cohort Separately)
- **`run_model_evaluation_all_cohorts.sh`**: Bash wrapper to run all cohorts sequentially
- **`run_model_evaluation_all_cohorts.py`**: Python wrapper to run all cohorts sequentially

## Usage

### Run Single Cohort/Age Band

```bash
# Evaluate XGBoost model for opioid_ed/13-24
python utility_scripts/evaluate_models_test_data.py \
    --cohort opioid_ed \
    --age-band 13-24 \
    --model-type xgboost \
    --n-shap-samples 1000

# Evaluate CatBoost model
python utility_scripts/evaluate_models_test_data.py \
    --cohort opioid_ed \
    --age-band 13-24 \
    --model-type catboost \
    --n-shap-samples 1000

# Evaluate both models
python utility_scripts/evaluate_models_test_data.py \
    --cohort opioid_ed \
    --age-band 13-24 \
    --model-type both \
    --n-shap-samples 1000
```

### Run All Cohorts (Sequentially)

**Bash version:**
```bash
# Run all cohorts with default settings
./utility_scripts/run_model_evaluation_all_cohorts.sh

# Customize SHAP samples
N_SHAP_SAMPLES=500 ./utility_scripts/run_model_evaluation_all_cohorts.sh

# Evaluate only XGBoost models
MODEL_TYPE=xgboost ./utility_scripts/run_model_evaluation_all_cohorts.sh
```

**Python version:**
```bash
# Run all cohorts
python utility_scripts/run_model_evaluation_all_cohorts.py

# Customize parameters
python utility_scripts/run_model_evaluation_all_cohorts.py \
    --model-type xgboost \
    --n-shap-samples 500 \
    --delay 3.0

# Run specific cohort
python utility_scripts/run_model_evaluation_all_cohorts.py \
    --cohort opioid_ed \
    --model-type both
```

## Output Files

Results are saved to: `8_ffa_analysis/results/model_evaluation/`

For each cohort/age_band/model_type combination:

1. **Metrics JSON**: `{cohort}_{age_band}_{model_type}_test_metrics.json`
   - Performance metrics (Recall, Precision, AUC-PR, ROC-AUC, etc.)
   - Before and after calibration
   - Confusion matrices

2. **Feature Importance CSV**: `{cohort}_{age_band}_{model_type}_test_feature_importance.csv`
   - Feature importance rankings
   - Normalized importance scores

3. **SHAP Importance CSV**: `{cohort}_{age_band}_{model_type}_test_shap_importance.csv`
   - Global SHAP importance per feature
   - Mean absolute SHAP values

4. **SHAP Values Parquet**: `{cohort}_{age_band}_{model_type}_test_shap_values.parquet`
   - Row-level SHAP values for sampled patients
   - Can be combined with FFA analysis results

5. **Predictions Parquet**: `{cohort}_{age_band}_{model_type}_test_predictions.parquet`
   - Raw and calibrated predictions
   - True labels

6. **Combined Summary CSV**: `test_evaluation_summary.csv`
   - Summary of all evaluations
   - Easy comparison across cohorts

## Performance Metrics

The script calculates the following metrics (both raw and calibrated):

- **Recall** (Sensitivity): TP / (TP + FN)
- **Precision**: TP / (TP + FP)
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under Precision-Recall curve (better for imbalanced data)
- **LogLoss**: Logarithmic loss (calibration quality)
- **Brier Score**: Mean squared error of probabilities

## Model Calibration

Models are calibrated using **isotonic regression**:
- Fits isotonic regression on raw predictions vs true labels
- Transforms probabilities to be better calibrated
- Finds optimal threshold using ROC curve (maximizing TPR - FPR)

## Memory Management

- **SHAP Sampling**: By default, uses 1000 samples for SHAP analysis (configurable)
- **Sequential Processing**: Wrapper scripts run each cohort separately to manage memory
- **Caching**: Models and test data are cached locally after first download

## Combining with FFA Analysis

Results can be combined with FFA analysis results:

1. **Feature Importance**: Compare model feature importance with FFA causal importance
2. **SHAP Values**: Combine SHAP values with FFA interaction analysis
3. **Performance Metrics**: Add model performance context to FFA results

## Example: Run One Cohort at a Time

```bash
# Process opioid_ed cohorts one by one
python utility_scripts/evaluate_models_test_data.py --cohort opioid_ed --age-band 13-24 --model-type both
python utility_scripts/evaluate_models_test_data.py --cohort opioid_ed --age-band 25-44 --model-type both
python utility_scripts/evaluate_models_test_data.py --cohort opioid_ed --age-band 45-54 --model-type both
python utility_scripts/evaluate_models_test_data.py --cohort opioid_ed --age-band 55-64 --model-type both

# Process non_opioid_ed cohorts
python utility_scripts/evaluate_models_test_data.py --cohort non_opioid_ed --age-band 65-74 --model-type both
python utility_scripts/evaluate_models_test_data.py --cohort non_opioid_ed --age-band 75-84 --model-type both
python utility_scripts/evaluate_models_test_data.py --cohort non_opioid_ed --age-band 85-94 --model-type both
```

## Notes

- **EC2**: Scripts auto-detect EC2 environment and use IAM roles (no profile needed)
- **Local**: Uses AWS_PROFILE env var or defaults to 'mushin' profile
- **Models**: Loads best models from S3 (selected by final model training)
- **Test Data**: Uses 2019 test data from S3 (never seen during training)
