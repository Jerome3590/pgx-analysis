# Step 7: SHAP Analysis

**Code:** `7_shap_analysis/run_shap_analysis.py`
**Run by:** `3_model_train_shap_ffa.ipynb` (Phase 3, after Step 6)

SHAP (SHapley Additive exPlanations) analysis produces global and row-level feature attributions for both XGBoost and CatBoost models. These results are used by Step 8 (FFA) for rule prioritization and by the dashboard Causal Analysis tab.

---

## Key design decisions

**Both models always run — model selection is ignored.**
SHAP is computed for both XGBoost (`.ubj` binary) and CatBoost (`.cbm` binary) regardless of which model was selected in Step 6. This provides cross-model consensus for the dashboard and for FFA filtering.

**Two-pass memory-efficient approach.**
Healthcare feature matrices can be very wide (1000+ features) and tall (100k+ rows). Processing all features for all rows simultaneously would exceed EC2 RAM. The two-pass approach avoids this:

1. **Pass 1 — Global signal sweep:** Stream all rows in chunks of 500; accumulate `mean_abs_shap` and `mean_shap` per feature. O(features) memory.
2. **Pass 2 — Row-level SHAP for selected features only:** Select top-K features by `mean_abs_shap` (default K=500); stream all rows in chunks of 200, slice only those columns, stream to Parquet via DuckDB. O(rows × K) memory instead of O(rows × all_features).

**Native model binaries preferred.**
Loading XGBoost from `.ubj` and CatBoost from `.cbm` avoids serialization artifacts (e.g. `base_score` string-array format in joblib models that corrupts SHAP values). A fallback to joblib with auto-repair of `base_score` is implemented for legacy models.

---

## How SHAP is computed per model

### XGBoost — `pred_contribs=True`

```python
booster = xgb_clf.get_booster()
dmatrix = xgb.DMatrix(X_chunk, feature_names=booster.feature_names)
contrib = booster.predict(dmatrix, pred_contribs=True)
# shape: (n_rows, n_features + 1)
# contrib[:, :-1] = feature SHAP values
# contrib[:, -1]  = bias (base score / expected value)
```

XGBoost's `pred_contribs=True` uses **exact TreeSHAP** (Lundberg et al. 2018) — not an approximation. Values are in log-odds space (pre-sigmoid). Each feature's SHAP value is the marginal contribution of that feature to the model's output, relative to the expected prediction over the training set.

Feature alignment is enforced: `X.reindex(columns=booster.feature_names, fill_value=0)` before any SHAP call to prevent column mismatch errors.

### CatBoost — `get_feature_importance(type="ShapValues")`

```python
from catboost import Pool
pool = Pool(X_chunk, y_chunk, cat_features=cat_feature_indices)
shap_chunk = model.get_feature_importance(type="ShapValues", data=pool)
# Binary classification shape: (n_samples, n_features + 1)
# Multiclass shape:             (n_samples, n_classes, n_features + 1)
# Last column = expected value (E[f(x)]) — NOT a feature SHAP
shap_feat = shap_chunk[:, :-1]          # binary
# shap_feat = shap_chunk[:,:,:-1].mean(axis=1)  # multiclass collapse
```

Key CatBoost SHAP specifics (per CatBoost SHAP tutorial):
- **Pool objects are required** — raw numpy/pandas arrays are not accepted for `type="ShapValues"`. A `Pool` wraps data with optional label and cat_feature indices.
- **Last column is E[f(x)]** (the expected value / bias), not a feature. Must be sliced off before computing statistics.
- **`item_*` features are marked as categorical** (`cat_features` argument) to match how the model was trained. These are binary indicator features (0/1) for drug names, ICD codes, CPT codes.
- CatBoost produces exact TreeSHAP values via its own internal implementation, equivalent to the SHAP library's `TreeExplainer`.

### Feature selection after Pass 1

Two strategies are implemented; Top-K is the default:

| Strategy | Function | Default |
|---|---|---|
| **Top-K** (default) | `select_signal_features_topk(global_df, k=500)` | K=500 |
| **Threshold** | `select_signal_features_threshold(global_df, min_mean_abs=0.0005)` | — |

Top-K ensures a fixed output size; threshold adapts to model signal density. Either can be swapped by changing the call in `run_shap_analysis()`.

---

## Per-bin SHAP

When `--bin {low|medium|high|extreme}` is passed, the CSV rows are filtered to that `n_event_bin` before SHAP computation, and the best per-bin model is loaded from `bin_models/{bin_name}/`:

```bash
python 7_shap_analysis/run_shap_analysis.py \
    --cohort opioid_ed --age-band 25-44 --bin low
```

Output goes to `7_shap_analysis/outputs/{cohort}/{ab}/bin_models/{bin_name}/`. This is run for each of the four bins (low / medium / high / extreme) by the notebook.

---

## Inputs

| Input | Source |
|---|---|
| Feature matrix | `6_final_model/outputs/{cohort}/{ab}/{cohort}_{ab}_train_final_features_no_leakage.csv` |
| XGBoost model | `6_final_model/outputs/{cohort}/{ab}/models/xgboost_model.ubj` (preferred) or `.joblib` |
| CatBoost model | `6_final_model/outputs/{cohort}/{ab}/models/catboost_model.cbm` (preferred) or `.json` |
| Model selection metadata | `6_final_model/outputs/{cohort}/{ab}/{cohort}_{ab}_model_selection_metadata.json` |

Per-bin inputs use `bin_models/{bin_name}/` prefix within the Step 6 output directory.

---

## Outputs

All outputs are written to `7_shap_analysis/outputs/{cohort}/{age_band_fname}/`:

| File | Description |
|---|---|
| `{cohort}_{ab}_shap_global_importance_xgboost.csv` | Per-feature `mean_abs_shap` and `mean_shap` (signed) for XGBoost. Sorted by `mean_abs_shap` desc. Features with `mean_abs_shap = 0` excluded. |
| `{cohort}_{ab}_shap_global_importance_catboost.csv` | Same for CatBoost. |
| `{cohort}_{ab}_shap_sample_values_xgboost.parquet` | Row-level SHAP values for top-500 features (XGBoost). Columns: `mi_person_key`, top-500 feature names, `bias`. |
| `{cohort}_{ab}_shap_sample_values_catboost.parquet` | Same for CatBoost. |
| `{cohort}_{ab}_shap_summary_bar_xgboost.png` | Bar chart of top features by mean absolute SHAP (XGBoost). |
| `{cohort}_{ab}_shap_summary_beeswarm_xgboost.png` | Beeswarm plot showing direction and magnitude per feature (XGBoost). |
| `{cohort}_{ab}_shap_summary_bar_catboost.png` | Bar chart (CatBoost). |
| `{cohort}_{ab}_shap_summary_beeswarm_catboost.png` | Beeswarm plot (CatBoost). |

Per-bin outputs use the same naming under `bin_models/{bin_name}/` subdirectory.

### Output schema — global importance CSV

| Column | Type | Description |
|---|---|---|
| `feature` | str | Feature name (matches training feature column name) |
| `mean_abs_shap` | float64 | Mean absolute SHAP value across all training rows |
| `mean_shap` | float64 | Mean signed SHAP value (positive = increases risk prediction) |

### Output schema — sample values Parquet

| Column | Type | Description |
|---|---|---|
| `mi_person_key` | int/str | Patient identifier from original data |
| `{feature_name}` × 500 | float32 | Per-row SHAP contribution (log-odds) for that feature |
| `bias` | float32 | Model base score / expected value for that row |

---

## S3 paths

```
s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/
  {cohort}_{ab}_shap_global_importance_xgboost.csv
  {cohort}_{ab}_shap_global_importance_catboost.csv
  {cohort}_{ab}_shap_sample_values_xgboost.parquet
  {cohort}_{ab}_shap_sample_values_catboost.parquet
```

Logs mirror to `s3://pgx-repository/7_shap_analysis_log/{cohort}/{age_band}/`.

---

## Downstream consumers

| Step | Uses SHAP output |
|---|---|
| **Step 8 (FFA)** | `shap_global_importance_xgboost.csv` — filters FFA rules to top SHAP features; only rules touching SHAP-important features are kept |
| **Combine SHAP/FFA** (`combine_shap_ffa_results.py`) | Both CSVs merged into `combined_importance.csv` for dashboard causal tab |
| **BupaR / DTW** | `combined_importance.csv` → `allowed_codes_shap_ffa_{cohort}_{ab}.json` drives event filtering in Step 9 visuals |
| **Dashboard Causal tab** | `combined_importance.csv` served by Lambda `/visualizations/causal` endpoint |

---

## Running

```bash
# Full cohort/age-band (both models)
python 7_shap_analysis/run_shap_analysis.py \
    --cohort opioid_ed --age-band 25-44

# Specific event density bin
python 7_shap_analysis/run_shap_analysis.py \
    --cohort non_opioid_ed --age-band 85-114 --bin extreme

# Verify outputs
ls 7_shap_analysis/outputs/opioid_ed/25_44/
```

---

## Logging

- **Local:** `logs/7_shap_analysis/{cohort}_{ab}[_{bin}].log` (repo root)
- **S3:** `s3://pgx-repository/7_shap_analysis_log/{cohort}/{age_band}/{filename}`

---

## Dependencies

- `xgboost` — `pred_contribs=True` on `xgb.DMatrix`
- `catboost` — `Pool`, `CatBoostClassifier.get_feature_importance(type="ShapValues")`
- `shap` — `shap.summary_plot` for visualization only (not for value computation)
- `duckdb` — memory-efficient CSV read and chunked Parquet write
- `joblib` — fallback model loading

**Reference:** [CatBoost SHAP Values Tutorial](https://github.com/catboost/catboost/blob/master/catboost/tutorials/model_analysis/shap_values_tutorial.ipynb)
