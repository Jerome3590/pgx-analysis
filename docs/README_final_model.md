# Final Model Development - PGx Analysis

This module hosts the final prediction model pipeline combining features from FPGrowth, BupaR, and DTW analyses for patient-level classification.

## Overview

The final model integrates three complementary analysis methods to create comprehensive patient-level features:

1. **FPGrowth** - Frequent pattern mining (itemsets, association rules)
2. **BupaR** - Process mining (sequence patterns, temporal flows)
3. **DTW** - Trajectory analysis (patient clustering, similarity scores)

## Goals

- Build cohort-level prediction models for target outcomes (opioid dependence, ED visits)
- Integrate features from FPGrowth, BupaR, and DTW analyses
- Standardize feature extraction across pharmacy (drug_name) and medical (ICD/CPT) domains
- Produce model explanations to guide feature reduction and clinical review

## Feature Schema

The complete feature schema is defined in `final_feature_schema.json` (JSON Schema Draft 7).

### Feature Categories

| Category | Feature Count | Description |
|----------|---------------|-------------|
| **FPGrowth** | ~100-500 | Frequent itemsets, association rules, itemset metrics |
| **BupaR** | ~50-200 | Process flow patterns, sequence features, temporal metrics |
| **DTW** | ~20-25 | Trajectory clusters, similarity scores, temporal characteristics |
| **Demographics** | ~10-15 | Age, gender, race, location, payer |
| **Temporal** | ~5-10 | Event dates, temporal windows, seasonality |
| **Total** | **~185-750** | Patient-level features for classification |

### Key Features

#### FPGrowth Features
- **Frequent Itemsets**: Binary features for each frequent itemset (drugs, ICD codes, CPT codes)
- **Association Rules**: Rule matching counts, confidence, and lift metrics
  - `rules_target_icd_match`: Number of opioid dependence prediction rules matched
  - `rules_target_ed_match`: Number of ED visit prediction rules matched
  - `max_rule_confidence_target_icd`: Maximum confidence of matched rules
  - `max_rule_lift_target_icd`: Maximum lift of matched rules
- **Itemset Metrics**: Aggregated statistics (total unique items, avg support)
- **Drug Encoding**: Global drug encoding for CatBoost categorical features

#### BupaR Features
- **Process Flow**: Path length, unique activities, path diversity
- **Temporal**: Throughput time, waiting time, active time, avg time between activities
- **Activity Frequencies**: Counts for each activity type
- **Sequence Patterns**: Repetition indicators, complexity measures, common pattern matching
- **Drug Sequences**: Sequence length, drug switches, concurrent drugs

#### DTW Features
- **Trajectory Clusters**: Cluster membership for drugs, ICD codes, CPT codes
- **Trajectory Characteristics**: 
  - Length (number of events)
  - Diversity (unique items)
  - Temporal span (days between first and last event)
  - **Temporal density (events per month)** - clinically interpretable scale
- **Similarity Scores**: Distance to archetypes, distance to target cases
- **Cluster Properties**: Target rates, cluster sizes
- **Multi-Modal**: Cross-modal cluster alignment (drug-ICD, drug-CPT)

## Data Inputs

### Base Cohort Data
- Gold cohort partitions: `s3://pgxdatalake/gold/cohorts/{cohort_name}/{age_band}/{event_year}/`

### FPGrowth Features
- **Source**: `s3://pgxdatalake/gold/fpgrowth/global/{item_type}/`
- **Files**: 
  - `rules_TARGET_ICD.json` - Opioid dependence prediction rules
  - `rules_TARGET_ED.json` - ED visit prediction rules
  - `rules_CONTROL.json` - Baseline/control rules
  - `frequent_itemsets.parquet` - Frequent itemsets

### BupaR Features
- **Source**: `s3://pgxdatalake/gold/bupar/{cohort_name}/{age_band}/{event_year}/`
- **Files**:
  - `process_flow_features.parquet`
  - `sequence_patterns.parquet`
  - `activity_frequencies.parquet`

### DTW Features
- **Source**: `s3://pgxdatalake/dtw_trajectories/{cohort_name}/{age_band}/{event_year}/`
- **Files**:
  - `trajectory_results_{item_type}.json`
  - `patient_trajectories_{item_type}.parquet`

## Feature Engineering Pipeline

```python
# 1. Load base cohort data
cohort_df = load_cohort_data(cohort_name, age_band, event_year)

# 2. Load FPGrowth features
fpgrowth_features = load_fpgrowth_features(cohort_name, age_band, event_year)

# 3. Load BupaR features
bupar_features = load_bupar_features(cohort_name, age_band, event_year)

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

Final model development uses the **same three-model ensemble** as feature importance:

- **CatBoost** (gradient boosting on categorical features)
- **XGBoost (boosted trees)**
- **XGBoost RF mode** (random forest-style XGBoost)

These models are compared with **Monte Carlo Cross-Validation (MC-CV)** on the training window (2016–2018),
then the best-performing base model is further tuned and calibrated before being evaluated on a strict 2019 holdout.

### MC-CV and Model Selection (2016–2018 Train Window)

The `7_final_model/final_model.ipynb` notebook:

- Loads patient-level features from `7_final_model/outputs/{cohort}/{age_band}/..._train_final_features.csv`
- Splits into `X` (features) and `y` (binary target)
- Runs MC-CV across:
  - **CatBoost**
  - **XGBoost**
  - **XGBoost RF mode**
- Aggregates per-split metrics (`roc_auc`, `logloss`, `recall`) and computes:
  - Mean and standard deviation by model
  - **Model selection criterion:** highest mean **Recall**

The model with the highest mean Recall is chosen as the **base final model** for that cohort/age-band.

### Optuna Hyperparameter Optimization

Once the best base model is identified, the notebook runs an **Optuna** study on the 2016–2018 training window:

- **Objective:** maximize mean Recall over 5-fold `StratifiedKFold` CV
- **Search space (examples):**
  - CatBoost: `iterations`, `learning_rate`, `depth`, `l2_leaf_reg`
  - XGBoost / XGBoost RF: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
- **Output:** best trial parameters and recall score

The tuned hyperparameters are merged with sensible defaults and used to fit a **tuned final model**.

### Temporal Probability Calibration (2016–2018 Only)

To ensure well-calibrated probabilities **without leaking test data**, we use a **temporal calibration strategy**:

1. Use `model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet` to determine each patient’s
   latest `event_year` within 2016–2018.
2. Define:
   - **Train-for-calibration:** patients with `max_event_year` in **2016 or 2017**
   - **Calibration set:** patients with `max_event_year == 2018`
3. Refit the tuned model on the 2016–2017 group.
4. Wrap it in `CalibratedClassifierCV` (`method="isotonic"`, `cv="prefit"`) and fit on the 2018 calibration group.
5. Report a **Brier score** on the 2018 calibration set as a calibration quality check.

The **2019 holdout (true test set) is never used** in tuning or calibration. It is reserved for final performance
and calibration diagnostics.

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
- `7_final_model/build_final_features_opioid_ed_0_12.py`: Builds the cohort 1, age 0–12 final feature table from `model_data`, BupaR, and DTW outputs.

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
- **High importance**: 
  - FPGrowth: `rules_target_icd_match`, `max_rule_lift_target_icd`
  - DTW: `trajectory_cluster_drug`, `cluster_target_rate_drug`
  - BupaR: `path_length`, `throughput_time_days`

## Important Notes

1. **Temporal Density**: Always calculated as **events per month** (not per day) for clinical interpretability
2. **Cohort-Specific**: Some features are cohort-specific (e.g., `days_to_target_event` only for ED_NON_OPIOID)
3. **Null Handling**: Many DTW features may be null if patient not assigned to cluster or trajectory unavailable
4. **Feature Count**: Actual feature count varies based on:
   - Number of frequent itemsets discovered
   - Number of activities in process flows
   - Number of trajectory clusters

## TODOs

- [ ] Implement feature engineering pipeline script
- [ ] Create feature extraction utilities for FPGrowth, BupaR, DTW
- [ ] Feature importance exploration: identify which features most strongly predict target outcomes
- [ ] Use model-based importance and SHAP summaries to filter to manageable feature set
- [ ] Post-model: revisit ICD/CPT/Drug heatmaps with top features only

## References

- **Feature Schema**: `final_feature_schema.json` - Complete JSON Schema definition
- **FPGrowth Analysis**: `../4_fpgrowth_analysis/`
- **BupaR Analysis**: `../5_bupaR_analysis/`
- **DTW Analysis**: `../6_dtw_analysis/DTW_FEATURE_EXTRACTION.md`
- **Notebook Integration**: See `../docs/notebook_calls.md` for calls and patterns


