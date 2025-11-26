# Final Model Development - PGx Analysis

This module hosts the final prediction model pipeline combining features from FPGrowth, BupaR, and DTW analyses for patient-level classification.

## Overview

The final model integrates three complementary analysis methods to create comprehensive patient-level features:

1. **FPGrowth** - Frequent pattern mining (itemsets, association rules)
2. **BupaR** - Process mining (sequence patterns, temporal flows)
3. **DTW** - Trajectory analysis (patient clustering, similarity scores)

### Temporal Validation Strategy

**Important:** The final model uses a strict temporal validation approach consistent with feature importance analysis:

- **Training Data:** Years 2016-2018 (full training set)
- **Test Data:** Year 2019 (holdout set, never used for training)
- **Excluded:** Year 2020 (COVID-19 pandemic year)

**Rationale:**
1. **Prevents Data Leakage:** 2019 data is never seen during training, ensuring true temporal validation
2. **Maintains Temporal Order:** Train on past data, test on future data
3. **Avoids COVID Impact:** 2020 excluded due to pandemic-related changes in healthcare patterns
4. **Consistent with Feature Importance:** Same train/test split as feature importance analysis ensures selected features generalize well

**Note:** This validation strategy matches the feature importance analysis pipeline, ensuring that features identified as important during MC-CV will perform well in the final model.

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

## Model Training

### CatBoost Integration

```python
from catboost import CatBoostClassifier, Pool

# Identify categorical features
categorical_features = [
    'cohort_name',
    'age_band',
    'member_gender',
    'member_race',
    'trajectory_cluster_drug',
    'trajectory_cluster_icd',
    'trajectory_cluster_cpt',
    'encoded_drug_names',  # From FPGrowth
    'most_frequent_activity',  # From BupaR
    # ... other categorical features
]

# Create CatBoost pool
train_pool = Pool(X_train, y_train, cat_features=categorical_features)

# Train model
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    cat_features=categorical_features,
    verbose=100
)
model.fit(train_pool)
```

### Random Forest Integration

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# One-hot encode categorical features
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_features])

# Combine with continuous features
X_continuous = X.drop(categorical_features, axis=1)
X_final = np.hstack([X_continuous, X_encoded])

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_final, y)
```

## Scripts

- `run_ade_targets.py`: Orchestrates ADE/ED CatBoost runs across age bands/years
- `run_catboost_*.py`: Entry points for specific target setups (opioid ED, ADE ED, etc.)
- `feature_importance_scaled.py`: Feature importance calculation with MC-CV

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


