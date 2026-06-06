# 6_final_model – Final Model Training and Export

This step trains the final prediction models for each `(cohort, age_band)`, using
the leakage-filtered feature tables built from event-level model data and upstream
feature engineering (PGx; BupaR/DTW/FP-Growth are dashboard-only).

### Current notebook workflow

The production entry point is `3_model_train_shap_ffa.ipynb`.

- Step 4 runs for both `opioid_ed` and `non_opioid_ed`, but skips existing
  `model_events.parquet` unless `FORCE_STEP4_ALL` or `FORCE_STEP4_NON_OPIOID`
  is explicitly enabled.
- Step 5 runs for both cohorts and should normally use `FORCE_STEP5 = False`
  so checkpoint-aware logic can skip completed PGx outputs.
- Step 6 invokes `run_final_model.py` for both cohorts. With
  `CLEAN_STEP6_DOWNSTREAM_ARTIFACTS = False`, `FORCE_STEP6 = False`, and
  `FORCE_STEP6_REBUILT_ONLY = False`, existing complete model artifacts are not
  retrained. Missing or incomplete cohort/age-band artifacts regenerate.
- Downstream SHAP, FFA, and combine use `DOWNSTREAM_ANALYSIS_COHORTS` and run
  for both cohorts after Step 6 artifacts are available.

### Implementation

The core implementation resides in this step:

- `6_final_model/run_final_model.py`
  - **Training mode (CLI):** `--train-mode per_bin` (**default**), `aggregate`, or `both`. Per-bin trains one model per event-density bin (`low` / `medium` / `high` / `extreme`) under `outputs/.../bin_models/{bin}/`, then **mirrors** the first available bin in preference order (`medium`, then the rest) to the cohort-level `outputs/.../{age_band_fname}/` tree so `prepare_models.py` and deploy paths keep working. `aggregate` trains only the cohort-wide model (legacy). `both` runs cohort-wide then per-bin (no mirror; aggregate outputs stay from the unified run).
  - Loads event-level model data from `4_model_data/.../model_events.parquet`.
  - Uses aggregated feature importances (Step 3a) and PGx patient-level features (Step 5).
  - Applies target-leakage removal (post-event, time-to-target, etc.).
  - Performs a temporal split: training rows are restricted to patients with max event year `2016-2018`; `2019` is reserved for holdout evaluation.
  - In `per_bin` mode, checks class support per density bin. Sparse bins receive full-cohort pooled fallback artifacts and an `INFERENCE_SOURCE.txt` marker instead of independently trained bin models.
  - Includes CatBoost preflight/runtime guards so all-constant categorical features or CatBoost failures do not abort the full pipeline.
  - Restricts to numeric features and runs Monte-Carlo CV for:
    - XGBoost and XGBoost RF (selects best by PR-AUC, then recall).
    - CatBoost with `grow_policy="SymmetricTree"` (oblivious trees).
  - **Platt calibration (OOF):** accumulates out-of-fold predictions across all MC-CV splits, then fits a `LogisticRegression(C=1)` calibrator per model type on the concatenated OOF probabilities vs actual outcomes. This corrects systematic over/under-prediction so dashboard risk scores match observed event rates. Saved as `models/calibration_{model_type}.joblib`. **Re-running model training regenerates calibration files.**
  - Exports:
    - Leakage-filtered final feature table for FFA.
    - Best CatBoost binary (.cbm) for SHAP; best XGBoost JSON for FFA.
    - Model selection metadata.
    - `models/calibration_{xgboost,xgboost_rf,catboost}.joblib` — Platt calibrators.
    - `models/calibration_diagnostics.json` — per-model: n_OOF samples, raw mean, calibrated mean, observed rate, residual.

Outputs are written under:

- `6_final_model/outputs/{cohort}/{age_band_fname}/`
  - `{cohort}_{age_band_fname}_train_final_features_no_leakage.csv`
  - `final_model_json/{cohort}_{age_band_fname}_best_catboost_model.cbm`
  - `final_model_json/{cohort}_{age_band_fname}_best_xgboost_model.json`
  - `{cohort}_{age_band_fname}_xgboost_feature_importance.csv`
  - `models/calibration_xgboost.joblib` — Platt calibrator for XGBoost
  - `models/calibration_xgboost_rf.joblib` — Platt calibrator for XGBoost RF
  - `models/calibration_catboost.joblib` — Platt calibrator for CatBoost
  - `models/calibration_diagnostics.json` — calibration diagnostics (raw→calibrated→observed rate per model)
  - `n_event_bin_thresholds.json` — P25/P50/P95 cut-points for event density binning
- `6_final_model/outputs/{cohort}/{age_band_fname}/bin_models/{low|medium|high|extreme}/`
  - Per-bin or fallback model artifacts and metrics.
  - Fallback bins include `INFERENCE_SOURCE.txt`.

> **⚠️ Calibration files require a training run.** If `models/calibration_*.joblib` is missing for a cohort/age_band (e.g. the model was trained before this feature was added), run `run_final_model.py` for that cohort/age_band. The Lambda falls back to raw probabilities when calibration files are absent — functional but uncalibrated.

### No redundant calibration elsewhere

- `8_ffa_analysis/ffa_analysis.py` has a `calibrate_model()` method — this is **threshold calibration** (finding an optimal decision boundary via Youden's J), not probability calibration. No conflict.
- The old `README.md` references "temporal probability calibration (train on 2016–2017, calibrate on 2018)" — that approach was superseded by **MC-CV OOF Platt scaling** which uses all `n_runs × 30%` test folds as out-of-fold data. This is preferable because it uses more calibration data and is consistent with the same training data distribution.

### Relationship to feature encoding

Before or as part of the final model step, feature lookups and codebooks may be used:

1. `6_final_model/create_feature_lookup.py`
2. `6_final_model/create_drug_codebook.py`

These populate feature encoding outputs used to interpret models, feature importances, SHAP values, and FFA outputs.
