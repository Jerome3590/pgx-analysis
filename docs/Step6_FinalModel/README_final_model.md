# Final Model Development - PGx Analysis

This module hosts the final prediction model pipeline for patient-level classification. **Authoritative description:** see `6_final_model/README.md` in the repo.

## Overview

**Current pipeline:** Feature engineering for the final model **never generates** trajectory, sequence, or itemset features. The model uses only:
- **n_events** (event count)
- **item_*** (binary drug/ICD/CPT indicators from aggregated feature importance, e.g. SHAP/FFA)
- **PGx counts** (e.g. pgx_num_drugs, pgx_num_cpic_drugs; **n_drugs** is built in the PGx analysis step)
- Demographics (e.g. age) and other non-item schema features

**FPGrowth, BupaR, and DTW** are used for **dashboard visualizations** (and DTW for **protocol filtering** only). They do not produce columns in the final model feature table.

## Goals

- Build cohort-level prediction models for target outcomes (opioid dependence, ED visits)
- Use only n_events, item_*, PGx counts, and other schema features (no trajectory/sequence/itemset)
- Use DTW for protocol filtering; FPGrowth and BupaR for dashboard visualizations only
- Standardize feature extraction across pharmacy (drug_name) and medical (ICD/CPT) domains
- Produce model explanations to guide feature reduction and clinical review

## Feature Schema

The complete feature schema is defined in `final_feature_schema.json` (JSON Schema Draft 7). **Feature categories in the actual pipeline:**

| Category | Description |
|----------|-------------|
| **n_events** | Event count (pre-target) |
| **item_*** | Binary indicators for drugs/ICD/CPT from aggregated feature importance (SHAP/FFA) |
| **PGx** | e.g. pgx_num_drugs, pgx_num_cpic_drugs; n_drugs from PGx analysis step |
| **Demographics** | Age and other non-item schema features |

Trajectory, sequence, and itemset features are **not** produced by feature engineering and are **not** in the model schema. FPGrowth/BupaR/DTW outputs are used for visualizations and (DTW) protocol filtering only.

## Data Inputs

### Base Cohort Data
- Gold cohort partitions: `s3://pgxdatalake/gold/cohorts/{cohort_name}/{age_band}/{event_year}/`

### Model Features
- Final model features come from **model data + feature importance** (n_events, item_*, PGx, etc.). Feature engineering does **not** produce trajectory/sequence/itemset columns.

### FPGrowth / BupaR / DTW (Visualization and Filtering Only)
- FPGrowth and BupaR outputs are used for **dashboard visualizations**, not as model feature columns.
- DTW is used for **protocol filtering** (preprocessing) only. See `6_final_model/README.md` for paths and details.

## Feature Engineering Pipeline

The pipeline builds the final feature table from model data and feature importance (n_events, item_*, PGx, etc.). It does **not** load FPGrowth/BupaR/DTW as feature columns. Example flow:

```python
# 1. Load base cohort / model data
cohort_df = load_cohort_data(cohort_name, age_band, event_year)

# 2. Build item features from refined feature list (Step 3c)
# 3. Add n_events, PGx counts, demographics; no trajectory/sequence/itemset

# 4. Load DTW features
dtw_features = load_dtw_features(cohort_name, age_band, event_year)

# 5. Merge all features
final_features = (
    cohort_df[['mi_person_key', 'is_target_case', ...]]
    .merge(fpgrowth_features, on='mi_person_key', how='left')
    .merge(bupar_features, on='mi_person_key', how='left')
    .merge(dtw_features, on='mi_person_key', how='left')
)

# 6. Prepare for model training
X = final_features.drop(['mi_person_key', 'is_target_case'], axis=1)
y = final_features['is_target_case']
```

## Model Training and Selection

Final model development trains **four candidates** per `(cohort, age_band)`:

- **XGBoost** (gradient boosting)
- **XGBoost RF mode** (random forest-style XGBoost)
- **CatBoost** (symmetric-tree gradient boosting with native categorical support)
- **Ensemble** (probability average of XGBoost + CatBoost)

These candidates are compared with **Monte Carlo Cross-Validation (MC-CV)** on the training window (2016–2018). The best candidate is selected and all component models are deployed for weighted Lambda inference.

### MC-CV and Model Selection (2016–2018 Train Window)

The `7_final_model/final_model.ipynb` notebook:

- Loads patient-level features from `7_final_model/outputs/{cohort}/{age_band}/..._train_final_features.csv`
- Splits into `X` (features) and `y` (binary target)
- Runs MC-CV across:
  - **CatBoost**
  - **XGBoost**
  - **XGBoost RF mode**
- Aggregates per-split metrics (`pr_auc`, `recall`, `auc`, `logloss`) and computes:
  - Mean and standard deviation per candidate
  - **Primary selection criterion:** highest mean **PR-AUC** (robust to 5:1 class imbalance)
  - **Secondary tiebreaker:** highest mean **Recall**
  - Ensemble is eligible for selection

The winning candidate is recorded in `model_metrics_summary.csv` (`selected=True`) and `model_selection_metadata.json`.

### Optuna Hyperparameter Optimization

Optuna runs automatically inside `train_and_evaluate()` when `n_runs` is passed. All candidate models are tuned:

- **Objective:** maximize mean **PR-AUC** over `StratifiedKFold` CV
- **Search space (examples):**
  - CatBoost: `iterations`, `learning_rate`, `depth`, `l2_leaf_reg`
  - XGBoost / XGBoost RF: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
- **Output:** best trial parameters → used for final model refit

Per-bin models (`train_per_bin()`) also run Optuna automatically, one study per density bin.

### Platt Scaling Calibration (OOF-based)

After MC-CV completes, a **Platt scaler** (logistic regression) is fitted per model type using the concatenated out-of-fold (OOF) predictions accumulated during MC-CV:

```
OOF probabilities from all MC test folds  →  LogisticRegression(C=1.0, solver='lbfgs')
                                           →  calibration_{model_type}.joblib
```

**Why OOF:** Each MC test fold is out-of-fold — the model predicting those rows was never trained on them. Concatenating all OOF folds gives ideal second-stage calibration data without holdout leakage.

**Diagnostics saved to** `calibration_diagnostics.json`:

| Field | Description |
|-------|-------------|
| `n_oof_samples` | Total OOF rows used for fitting |
| `mean_raw_proba` | Mean uncalibrated probability |
| `mean_calibrated_proba` | Mean post-calibration probability |
| `observed_rate` | Actual event rate (ground truth) |
| `calibrator_coef` / `intercept` | LogReg sigmoid parameters |

**Lambda inference:** `load_calibration_model(bin_name=...)` loads the scaler; if missing, raw model probability is used unchanged.

**The 2019 holdout is never used** in tuning or calibration — reserved exclusively for final evaluation.

### Final Model Artifacts and S3 Layout

For each cohort and age band, the notebook writes the following artifacts:

- **Local (under `7_final_model/outputs/{cohort}/{age_band_fname}/`):**
  - `{cohort}_{age_band_fname}_mc_cv_results.csv` – raw per-split MC-CV metrics
  - `{cohort}_{age_band_fname}_final_model_{best_model_name}.joblib` – tuned, uncalibrated model
  - `{cohort}_{age_band_fname}_final_model_{best_model_name}_calibrated.joblib` – calibrated model wrapper
  - `{cohort}_{age_band_fname}_final_model_catboost.cbm` / `.json` – CatBoost native exports (when CatBoost wins)
  - `{cohort}_{age_band_fname}_final_model_xgboost*.json` – XGBoost booster JSON (when XGBoost/XGBoost RF wins)

- **S3 Gold (per-cohort, per-age-band, per-event_year=train):**

  - `s3://pgxdatalake/gold/final_model/cohort_name={cohort}/age_band={age_band_fname}/event_year=train/models/`
    - All of the above artifacts are uploaded here:
      - `*_mc_cv_results.csv`
      - `*_final_model_*.joblib`
      - `*_final_model_*.cbm` / `*_final_model_*.json`

This layout is aligned with the broader visualization and causal analysis outputs documented in
`README_data_visualizations.md` and `8_ffa_analysis`.

## Notebooks and Scripts

- `7_final_model/final_model.ipynb`: MC-CV comparison (CatBoost, XGBoost, XGBoost RF), Optuna tuning, temporal calibration, and final model export.
- `6_final_model/build_final_cohort_model_features.py`: Builds the final feature table (n_events, item_*, PGx, etc.). Feature engineering does not use BupaR/DTW/FPGrowth as feature columns; those are visualization/filtering only.

## Feature Validation

### Missing Values
- **Categorical**: Use "unknown" or mode imputation
- **Continuous**: Use median or mean imputation
- **Binary**: Use 0 (absence) for missing

### Feature Scaling
- **CatBoost**: No scaling needed (handles categoricals natively)
- **Random Forest**: No scaling needed (tree-based)
- **Logistic Regression**: Standardize continuous features

### Expected Feature Importance
- **High importance**: item_* (drug/ICD/CPT from SHAP/FFA), n_events, PGx-related counts. FPGrowth, DTW, and BupaR are not used as model features (visualization/filtering only).

## Important Notes

1. **No trajectory/sequence/itemset**: Feature engineering never produces these; model uses n_events, item_*, PGx, and other schema features only.
2. **Cohort-Specific**: Some features are cohort-specific (e.g., drug-only for polypharmacy).
3. **DTW/BupaR/FPGrowth**: Used for dashboard visualizations and (DTW) protocol filtering only, not as model feature columns.

## TODOs

- [ ] Feature importance exploration: identify which item_* and count features most strongly predict target outcomes
- [ ] Use model-based importance and SHAP summaries to filter to manageable feature set
- [ ] Post-model: revisit ICD/CPT/Drug heatmaps with top features only

## References

- **Feature Schema**: `final_feature_schema.json` - Complete JSON Schema definition
- **Model data**: `4_model_data/`; **FP-Growth (dashboard)**: `10_risk_dashboard/visualizations/fpgrowth/`
- **BupaR Analysis**: `../5_bupaR_analysis/`
- **DTW Analysis**: `../6_dtw_analysis/DTW_FEATURE_EXTRACTION.md`
- **Notebook Integration**: See `../docs/README_notebook_calls.md` for calls and patterns


