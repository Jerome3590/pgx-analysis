# Optuna Implementation & Artifact Validation

## 1. Optuna implementation completeness

| Plan item | Status | Location / notes |
|-----------|--------|------------------|
| **Constants** N_MCCV_HPO=5, N_OPTUNA_TRIALS=50, RANDOM_STATE=1997 | Done | `run_final_model.py` top |
| **generate_mc_splits** (X, y, n_splits, test_size, random_state) | Done | `_generate_mc_splits()` |
| **build_model_from_trial** (xgb, xgb_rf, cat search spaces) | Done | `_build_model_from_trial()` |
| **select_best_trial_from_pareto** (auprc / recall / recall_threshold) | Done | `_select_best_trial_from_pareto()` |
| **Multi-objective** (mean_recall, mean_auprc), create_study(directions=[maximize, maximize]) | Done | `_optuna_objective()`, `study.optimize()` |
| **25-split MCCV** with best config after Optuna (same metrics structure) | Done | Single loop over n_runs with selected model (Optuna params) + others (defaults) |
| **Best XGB for FFA** always trained and saved | Done | `best_xgb_variant`, xgb_final → best_xgboost_model.json |
| **Final training** uses Optuna best params when optuna_used | Done | `_build_model_from_params(optuna_best_params, ...)` for xgb_final, cb_final |
| **selection_metadata** includes optuna_used, optuna_best_params, best_pr_auc, best_recall | Done | Written to `*_model_selection_metadata.json` |
| **Legacy fallback** when Optuna not installed or best_trial is None | Done | `if not optuna_used:` runs fixed-hyperparameter loop and selection |
| **requirements.txt** optuna>=3.0 | Done | Already present |

Implementation is **complete** per plan. Optional: RECALL_MIN / strategy "recall_threshold" is implemented in `_select_best_trial_from_pareto` but default is "auprc".

---

## 2. Artifacts produced by run_final_model.py (unchanged paths)

All paths under `6_final_model/outputs/{cohort}/{age_band_fname}/` (and mirrored to `6_final_model/model_outputs/` and S3):

| Artifact | Path | Used by |
|----------|------|--------|
| Model selection metadata | `{cohort}_{age_band_fname}_model_selection_metadata.json` | SHAP (metadata), reporting |
| MC CV results (per split/model) | `{cohort}_{age_band_fname}_mc_cv_results.csv` | Reporting |
| Model metrics summary | `{cohort}_{age_band_fname}_model_metrics_summary.csv` | Reporting |
| Final features (no leakage) | `{cohort}_{age_band_fname}_train_final_features_no_leakage.csv` | FFA data, checkpoint |
| XGBoost feature importance | `{cohort}_{age_band_fname}_xgboost_feature_importance.csv` | Reporting, FI viz |
| CatBoost feature importance | `{cohort}_{age_band_fname}_catboost_feature_importance.csv` | Reporting (added with Optuna work) |
| **Best XGBoost JSON (FFA)** | `final_model_json/{cohort}_{age_band_fname}_best_xgboost_model.json` | **FFA** (rule extraction), contains `feature_names`, `trees`, `selection_metadata` |
| **Best CatBoost binary** | `final_model_json/{cohort}_{age_band_fname}_best_catboost_model.cbm` | **SHAP** (CatBoost explainer) |
| Best CatBoost JSON | `final_model_json/{cohort}_{age_band_fname}_best_catboost_model.json` | Fallback load |
| XGBoost joblib / binary | `models/xgboost.joblib`, `models/xgboost_model.ubj` | **SHAP** (XGB explainer) |
| CatBoost joblib / binary | `models/catboost.joblib`, `models/catboost_model.cbm` | **SHAP** |
| Train/test parquet | `inputs/model_train/final_features.parquet`, `inputs/model_test/...` | SHAP/FFA data (via prepare_train_test_s3) |

Same artifact set is produced for both **Optuna path** and **legacy path**; only the way the best model and params are chosen differs.

---

## 3. SHAP analysis (7_shap_analysis) – compatibility

- **Model selection metadata:** `6_final_model/outputs/{cohort}/{age_band_fname}/{cohort}_{age_band_fname}_model_selection_metadata.json` — **used**; structure unchanged (selected_model, best_xgb_variant, means, etc.); extra keys `optuna_used`, `optuna_best_params` are additive.
- **CatBoost:** Load order: `outputs/.../models/catboost_model.cbm` → `model_outputs/.../models/catboost_model.cbm` → `outputs/.../final_model_json/..._best_catboost_model.cbm` → `model_outputs/.../..._best_catboost_model.cbm` → `..._best_catboost_model.json`. **All produced by run_final_model.**
- **XGBoost:** Load order: `outputs/.../models/xgboost_model.ubj` → `model_outputs/.../models/xgboost_model.ubj` → joblib fallbacks. **All produced by run_final_model.**

**Conclusion:** SHAP receives the same artifacts at the same paths; no change required for Optuna.

---

## 4. FFA analysis (8_ffa_analysis) – compatibility

- **XGBoost JSON for FFA:** FFA expects the best XGBoost model as JSON with `feature_names`, `trees`, `selection_metadata`. Step 6 writes `final_model_json/{cohort}_{age_band_fname}_best_xgboost_model.json` with that structure (and `model_type` normalized to `xgboost` / `xgboost_rf`). **Compatible.**
- **Data:** `train_final_features_no_leakage.csv` and/or `inputs/model_train/final_features.parquet` — produced by Step 6 and prepare_train_test_s3. **Compatible.**
- **check_inputs.py:** Updated to look for `{cohort}_{age_band_fname}_best_xgboost_model.json` and `{cohort}_{age_band_fname}_best_catboost_model.cbm` in `final_model_json/` (was incorrectly checking `xgboost_model.json`).
- **download_and_test_ffa.py:** Already uses S3 keys `..._best_xgboost_model.json` and `..._best_catboost_model.json`. **Compatible.**
- **combined_causal_analysis.py:** Looks for `final_model_json/{cohort}_{age_band_fname}_final_model_{model_type}.json` (e.g. `final_model_xgboost.json`). That naming is **different** from Step 6’s `best_xgboost_model.json` / `best_catboost_model.json`. If FFA rule extraction is driven by the single best XGB JSON (best_xgboost_model.json), then the main FFA pipeline is compatible; any code that expects `final_model_*.json` per model type is legacy and would need to be updated or supplied via a different path.

**Conclusion:** The artifacts that SHAP and the main FFA flow (best XGB JSON, CatBoost, data) use are produced at the same paths and with the same structure. check_inputs was updated to match. combined_causal_analysis’s `final_model_{model_type}.json` naming is a separate convention and may require a follow-up if that script is still in use.

---

## 5. selection_metadata structure (for downstream)

Current keys (unchanged plus Optuna extras):

- `selected_model`, `best_xgb_variant`, `selection_reason`
- `xgb_recall_mean`, `xgb_pr_auc_mean`, `xgb_rf_recall_mean`, `xgb_rf_pr_auc_mean`
- `catboost_recall_mean`, `catboost_pr_auc_mean`, `catboost_auc_mean`, `catboost_logloss_mean` (if CatBoost ran)
- `best_pr_auc`, `best_recall`
- `optuna_used` (bool), `optuna_best_params` (dict, when optuna_used)

SHAP and FFA do not depend on the new keys for loading models or data; they are for documentation and reporting.

---

## 6. Summary

- **Optuna:** Implemented as planned (multi-objective, Pareto, 25-split MCCV with best config, best XGB always saved for FFA, same artifact paths).
- **Artifacts:** Unchanged set of outputs; same paths and shapes; selection_metadata extended with optuna_* and best_pr_auc/best_recall.
- **SHAP:** Uses same paths; fully compatible.
- **FFA:** Uses same best XGB JSON and data paths; compatible. check_inputs.py updated to the correct filenames. combined_causal_analysis’s `final_model_{model_type}.json` remains a separate convention if that script is used.
