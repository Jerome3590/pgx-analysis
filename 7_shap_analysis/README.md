## 7_shap_analysis – SHAP-Based Post-Model Analysis

This module runs **SHAP analysis** on the final models for each `(cohort, age_band)` to
quantify per-feature and per-patient contributions, complementing the structural
explanations from `8_ffa_analysis`.

### Goals

- Compute **global SHAP importances** for XGBoost and CatBoost final models.
- Compute **local SHAP values** for sampled patients to support individual-level risk
  explanations.
- Align SHAP outputs with the **same numeric feature set** used in `6_final_model`
  (after target-leakage removal).
- Use the **feature lookup** from `6a_feature_encoding` so SHAP plots and tables show
  human-readable feature labels (including FP-Growth itemsets and structural ICD/CPT
  encodings).

### Inputs

For each `(cohort, age_band)`:

- Final feature table (no leakage):  
  - `6_final_model/outputs/{cohort}/{age_band_fname}/{cohort}_{age_band_fname}_train_final_features_no_leakage.csv`
- Feature lookup (6a):  
  - `feature_encoding_outputs/{cohort}/{age_band_fname}/{cohort}_{age_band_fname}_feature_lookup.csv`
- Final models (6b):  
  - XGBoost and CatBoost final models, either refit in-place or reloaded from
    `6_final_model/outputs/{cohort}/{age_band_fname}/final_model_json/`.

### Outputs

- `7_shap_analysis/outputs/{cohort}/{age_band_fname}/`:
  - `*_shap_global_importance_{model}.csv` – mean absolute SHAP value per feature.
  - `*_shap_sample_values_{model}.parquet` – SHAP values for a sampled subset of patients.
  - Summary plots (bar + beeswarm) per model:
    - `*_shap_summary_bar_{model}.png`
    - `*_shap_summary_beeswarm_{model}.png`
- Optional S3 mirror (when AWS CLI is configured):  
  - `s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/...`

### Script (planned)

- `7_shap_analysis/run_shap_analysis.py`:
  - CLI: `--cohort`, `--age_band`, `--n_background` (background sample size), `--n_eval` (eval sample size).
  - Loads final features, fits or reloads XGBoost and CatBoost models with the same
    hyperparameters used in `6_final_model`.
  - Uses `shap.TreeExplainer` for XGBoost and CatBoost (or CatBoost’s native SHAP
    implementation when available).
  - Aggregates SHAP values to global importances and saves per-feature tables with
    enriched labels via the feature lookup.

