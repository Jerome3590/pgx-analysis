# 6_final_model – Final Model Training and Export

This step trains the final prediction models for each `(cohort, age_band)`, using
the leakage-filtered feature tables built from event-level model data and upstream
feature engineering (PGx; BupaR/DTW/FP-Growth are dashboard-only).

### Implementation

The core implementation resides in this step:

- `6_final_model/run_final_model.py`
  - Loads event-level model data from `4_model_data/.../model_events.parquet`.
  - Uses aggregated feature importances (Step 3a) and PGx patient-level features (Step 5).
  - Applies target-leakage removal (post-event, time-to-target, etc.).
  - Restricts to numeric features and runs Monte-Carlo CV for:
    - XGBoost and XGBoost RF (selects best by recall/AUC-PR).
    - CatBoost with `grow_policy="SymmetricTree"` (oblivious trees).
  - Exports:
    - Leakage-filtered final feature table for FFA.
    - Best CatBoost binary (.cbm) for SHAP; best XGBoost JSON for FFA.
    - Model selection metadata.

Outputs are written under:

- `6_final_model/outputs/{cohort}/{age_band_fname}/`
  - `{cohort}_{age_band_fname}_train_final_features_no_leakage.csv`
  - `final_model_json/{cohort}_{age_band_fname}_best_catboost_model.cbm`
  - `final_model_json/{cohort}_{age_band_fname}_best_xgboost_model.json`
  - `{cohort}_{age_band_fname}_xgboost_feature_importance.csv`

### Relationship to feature encoding

Before or as part of the final model step, feature lookups and codebooks may be used:

1. `6_final_model/create_feature_lookup.py`
2. `6_final_model/create_drug_codebook.py`

These populate feature encoding outputs used to interpret models, feature importances, SHAP values, and FFA outputs.
