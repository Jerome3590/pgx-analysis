## 6b_final_model_selection – Final Model Training and Export

This step trains the final prediction models for each `(cohort, age_band)`, using
the leakage-filtered feature tables built from event-level model data and upstream
feature engineering steps (DTW, BupaR, FP-Growth, PGx).

### Implementation

The core implementation resides in this step:

- `6b_final_model_selection/run_final_model.py`
  - Loads event-level model data from `4a_model_data/.../model_events.parquet`.
  - Merges FP-Growth, BupaR, DTW, and PGx patient-level feature tables.
  - Applies target-leakage removal (post-event, time-to-target, DTW-derived features, etc.).
  - Restricts to numeric features and runs Monte-Carlo CV for:
    - XGBoost (GPU if available).
    - CatBoost with `grow_policy="SymmetricTree"` (oblivious trees).
    - Simple ensemble (average of XGBoost + CatBoost probabilities).
  - Exports:
    - Leakage-filtered final feature table for FFA.
    - FFA-friendly JSON models for XGBoost and CatBoost.
    - XGBoost feature importance CSV.

Outputs are written under:

- `6_final_model/outputs/{cohort}/{age_band_fname}/`
  - `{cohort}_{age_band_fname}_train_final_features_no_leakage.csv`
  - `final_model_json/{cohort}_{age_band_fname}_final_model_{xgboost,catboost}.json`
  - `{cohort}_{age_band_fname}_xgboost_feature_importance.csv`

### Relationship to 6a_feature_encoding

Before running 6b, you should run step 6a for the same `(cohort, age_band)`:

1. `6_final_model/create_feature_lookup.py`
2. `6_final_model/create_drug_codebook.py`

These scripts populate `feature_encoding_outputs/{cohort}/{age_band_fname}/` with:

- A feature index→name/description lookup table.
- A numeric drug-name codebook.

These artifacts are then used to interpret the models, feature importances, SHAP
values, and FFA outputs produced by the 6b final model step.

