# Model Performance Summary

**Date**: 2026-01-14  
**Status**: Performance metrics files not found in S3 or local directories

## Model Training Methodology

### Temporal Validation Strategy

All models follow a strict temporal validation approach:

- **Training Data**: Years 2016-2018 (full training set)
- **Test Data**: Year 2019 (holdout set, never used for training)
- **Excluded**: Year 2020 (COVID-19 pandemic year)

**Rationale**:
1. Prevents data leakage - 2019 data is never seen during training
2. Maintains temporal order - train on past data, test on future data
3. Avoids COVID impact - 2020 excluded due to pandemic-related changes
4. Consistent with feature importance - same train/test split ensures features generalize

### Model Selection Process

Models are trained and compared using **Monte Carlo Cross-Validation (MC-CV)**:

1. **Model Types Evaluated**:
   - CatBoost
   - XGBoost
   - XGBoost RF (Random Forest mode)

2. **MC-CV Configuration**:
   - **Target splits**: ~1000 MC-CV splits (for publication-grade estimates)
   - **Train/Test Split**: 80% train, 20% test (within 2016-2018 training data)
   - **Temporal Structure**: Each split trains on 80% sample of 2016-2018 and evaluates on 2019 (per README)

3. **Performance Metrics Calculated**:
   - **Recall** (Sensitivity) - Primary metric for model selection
   - **Precision** - Positive predictive value
   - **PR-AUC** (Average Precision) - Precision-Recall Area Under Curve
   - **LogLoss** - Logarithmic loss (calibration quality)

4. **Model Selection Criteria**:
   - Models ranked by composite score: `0.5 * PR-AUC + 0.5 * (1/(1+logloss))`
   - Best model selected based on composite score
   - Model selection prioritizes both discrimination (PR-AUC) and calibration (LogLoss)

### Expected Performance Files

The following files should contain model performance metrics:

**MC-CV Results** (per cohort/age_band):
- `{cohort}_{age_band}_mc_cv_results.csv` - Raw per-split MC-CV metrics
  - Columns: `split`, `model`, `recall`, `precision`, `logloss`, `pr_auc`
  - Contains metrics for each MC-CV split across all model types

**Model Summary** (per cohort/age_band):
- `{cohort}_{age_band}_model_summary.txt` - Summary of best model performance
  - Best model name
  - Mean LogLoss and PR-AUC across MC-CV splits
  - Composite score

**Model Selection Metadata** (per cohort/age_band):
- `{cohort}_{age_band}_model_selection_metadata.json` - Model selection details
  - Best model variant (xgb vs xgb_rf vs catboost)
  - Performance metrics
  - Selection rationale

## Expected S3 Locations

Performance files should be located at:
```
s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/outputs/
s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/model_outputs/
```

Or locally at:
```
6_final_model/outputs/{cohort}/{age_band_fname}/
6_final_model/model_outputs/{cohort}/{age_band_fname}/
```

## Current Status

**⚠️ Performance files not found**: No MC-CV results or model summary files were found in:
- S3: `s3://pgxdatalake/gold/final_model/`
- Local: `6_final_model/outputs/` or `6_final_model/model_outputs/`

**Possible reasons**:
1. Model training outputs may be stored in a different location
2. Performance metrics may not have been uploaded to S3
3. Model training may need to be re-run to generate performance files

## What We Know About Model Performance

### Training Process

Based on the code and documentation:

1. **MC-CV Evaluation**:
   - Models evaluated on multiple random splits of training data (2016-2018)
   - Each split: 80% train, 20% test (within training window)
   - Metrics calculated: Recall, Precision, LogLoss, PR-AUC

2. **Model Selection**:
   - Best model selected by composite score combining PR-AUC and LogLoss
   - Composite score: `0.5 * PR-AUC + 0.5 * (1/(1+logloss))`
   - Higher composite score = better model

3. **Final Model**:
   - Best model trained on full 2016-2018 dataset
   - Model exported for use in FFA analysis
   - Model used for predictions on 2019 test data in FFA analysis

### Test Data (2019) Performance

**Important**: The FFA analysis uses models trained on 2016-2018 and validates on 2019 test data. However, explicit test set performance metrics (accuracy, recall, precision, AUC on 2019) are not currently available in the expected locations.

**To obtain 2019 test set performance**:
1. Load the trained model from S3
2. Load 2019 test data: `s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/inputs/model_test/final_features.parquet`
3. Generate predictions on 2019 test set
4. Calculate metrics: Accuracy, Recall, Precision, F1, ROC-AUC, PR-AUC, LogLoss

## Next Steps

To complete the model performance summary:

1. **Locate performance files**: Check alternative S3 paths or local directories
2. **Re-evaluate models**: If files not found, re-run model evaluation on 2019 test data
3. **Generate summary**: Create comprehensive performance report for all cohorts

## Model Performance Metrics Explained

### Recall (Sensitivity)
- **Definition**: True Positive Rate = TP / (TP + FN)
- **Interpretation**: Fraction of actual positives correctly identified
- **Clinical Meaning**: Ability to catch all at-risk patients
- **Target**: High recall is critical for risk prediction (minimize false negatives)

### Precision (Positive Predictive Value)
- **Definition**: TP / (TP + FP)
- **Interpretation**: Fraction of predicted positives that are actually positive
- **Clinical Meaning**: When model predicts risk, how often is it correct
- **Trade-off**: Higher precision often means lower recall

### PR-AUC (Average Precision)
- **Definition**: Area under Precision-Recall curve
- **Interpretation**: Overall model performance balancing precision and recall
- **Range**: 0.0 to 1.0 (higher is better)
- **Use Case**: Better than ROC-AUC for imbalanced datasets (like ours with 5:1 control:case ratio)

### LogLoss (Logarithmic Loss)
- **Definition**: Measures calibration quality of probability predictions
- **Interpretation**: Lower is better (perfect = 0.0)
- **Clinical Meaning**: How well-calibrated are the risk probabilities
- **Use Case**: Important for clinical decision-making based on probability thresholds

### Composite Score
- **Definition**: `0.5 * PR-AUC + 0.5 * (1/(1+logloss))`
- **Interpretation**: Balanced measure of discrimination (PR-AUC) and calibration (LogLoss)
- **Range**: 0.0 to 1.0 (higher is better)
- **Use Case**: Single metric for model selection
