## 9_combined_shap_ffa – Combined SHAP + FFA Consensus Analysis

This module merges **distributional** explanations from SHAP (`8_shap_analysis`) with
the **structural** explanations from FFA (`7_ffa_analysis`) to produce a unified
view of feature importance for each `(cohort, age_band)`.

### Goals

- Align feature names and groups across SHAP and FFA using the **feature lookup** from
  step 6a (`feature_encoding_outputs/`).
- Produce a **consensus table** that includes:
  - SHAP global mean |SHAP| scores and ranks.
  - FFA combined weighted importance and ranks.
  - Simple combined scores (e.g., average of normalized SHAP + normalized FFA).
  - Feature metadata: group, description, itemset type, and itemset contents for FP-Growth
    itemset features.
- Export CSV + plots suitable for:
  - Final model documentation and clinical review.
  - Use by the risk calculator / dashboard to highlight stable, high-signal features.

### Inputs

Per `(cohort, age_band)`:

- SHAP global importance (currently XGBoost):  
  - `8_shap_analysis/outputs/{cohort}/{age_band_fname}/{cohort}_{age_band_fname}_shap_global_importance_xgboost.csv`
- FFA combined weighted importance:  
  - `7_ffa_analysis/outputs/{cohort}/{age_band_fname}/visualizations/combined_weighted_feature_importance.csv`
- Feature lookup from 6a:  
  - `feature_encoding_outputs/{cohort}/{age_band_fname}/{cohort}_{age_band_fname}_feature_lookup.csv`

### Outputs

- `9_combined_shap_ffa/outputs/{cohort}/{age_band_fname}/`:
  - `{cohort}_{age_band_fname}_combined_shap_ffa_importance.csv` – one row per feature with:
    - `feature`, `group`, `description`, `itemset_type`, `itemset_items`
    - `shap_mean_abs`, `shap_rank`, `shap_norm`
    - `ffa_weighted_importance`, `ffa_rank`, `ffa_norm`, `ffa_weighted_coverage`, `ffa_model_count`
    - `combined_score` and `combined_rank`
  - Optional bar plots for the top‑k consensus features (local only; S3 mirroring can be
    added similar to FFA/SHAP visualizations if needed).

### Script

- `9_combined_shap_ffa/run_combined_shap_ffa.py`:
  - CLI: `--cohort`, `--age_band`, `--top_k`.
  - Loads the SHAP and FFA importance tables and the feature lookup.
  - Performs left-join on `feature` and computes normalized scores and ranks.
  - Writes the combined CSV and (optionally) a simple top‑k bar plot for documentation.

