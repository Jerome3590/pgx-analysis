# Workflow Execution Status

**Cohort:** `opioid_ed`  
**Age Band:** `0-12`  

---

## Previous End-to-End Run (2025-12-10) – LEGACY PIPELINE

Status: ✅ Complete (for historical reference only; used `4_model_data` and older step layout).  
Model performance and step details are preserved below but superseded by the refactored run.

### Summary (Legacy)
- Step 3: Feature Importance – complete, artifacts preserved under `3_feature_importance/outputs/`
- Step 4: FP-Growth Analysis – complete
- Step 5: BupaR Analysis – complete
- Step 6: DTW Analysis – complete
- Step 7: PGx Analysis – complete
- Step 8: Final Model – complete (XGBoost RF best model, ~155 features)

> See earlier sections of this file for full legacy metrics and notes.

---

## Current Workflow (4a_model_data + Central Outputs)

**As of:** 2025-12-31  
**Status:** ✅ Canonical workflow definition (see historical section above for last full run)

### Step 3: Feature Importance ✅
- **Status:** ✅ Complete (reused from previous run)
- **Outputs:** `3_feature_importance/outputs/`
- **Notes:** Not regenerated; serves as input to `4a_model_data/create_model_data.py` and PGx scripts.

### Step 4a: Model Data Extraction (4a_model_data) ⏳
- **Goal:** Build within-cohort model-ready event data (cases + clean controls) for `opioid_ed / 0-12`.
- **Script:** `4a_model_data/create_model_data.py`
- **Outputs:**  
  - `4a_model_data/cohort_name=opioid_ed/age_band=0-12/model_events.parquet`
- **Status:** ⏳ To (re)run with updated DuckDB-only implementation.

### Step 4b: DTW Protocol Filtering (`4b_dtw_filter`) ⏳
- **Goal:** Create protocol-filtered event data to reduce noise from routine care sequences.
- **Script:** `4b_dtw_filter/filter_protocol_events.py`
- **Inputs:**  
  - `4a_model_data/cohort_name=opioid_ed/age_band=0-12/model_events.parquet`
- **Outputs:**  
  - `4a_model_data/cohort_name=opioid_ed/age_band=0-12/model_events_no_protocols.parquet`
- **Status:** ⏳ Pending after Step 4a completes.

### Step 4c: Extreme-Density Cohort Split (`5b_fpgrowth_analysis/extract_extreme_density_cohort.py`) ⏳
- **Goal:** Identify patients with extremely dense medical_code trajectories (ICD + procedure_code over TRAIN years) and:
  - move them into a separate `_extreme_density` cohort, and  
  - remove them from the main `4a_model_data` event set so they do not dominate FP-Growth, BupaR, DTW, and final models.
- **Script:** `5b_fpgrowth_analysis/extract_extreme_density_cohort.py`
- **Inputs:**  
  - `4a_model_data/cohort_name={cohort_name}/age_band={age_band}/model_events.parquet` (after Steps 4a and 4b as applicable)
- **Outputs:**  
  - `4a_model_data/cohort_name={cohort_name}_extreme_density/age_band={age_band}/model_events.parquet`  
  - `4a_model_data/cohort_name={cohort_name}/age_band={age_band}/model_events.parquet` rewritten with extreme-density patients removed  
  - `4a_model_data/cohort_name={cohort_name}/age_band={age_band}/extreme_density_patients_{age_band_fname}.csv`
- **Status:** ⏳ Run once per `(cohort, age_band)` before `5b_fpgrowth_analysis/run_analysis.py`; when complete, main and `_extreme_density` cohorts can each run through Steps 5–8 independently.

### Step 5b: FP-Growth Analysis (Pattern Mining + Features + Plots) ⏳
- **Goal:** Rebuild FP-Growth itemsets, rules, features, and visualizations using the new layout.
- **Script:** `5b_fpgrowth_analysis/run_analysis.py`
- **Inputs:**  
  - `4a_model_data/.../model_events[_no_protocols].parquet` (via `create_fpgrowth_features.py`)  
  - FP-Growth itemsets JSONs under `5b_fpgrowth_analysis/outputs/...`
- **Outputs (authoritative):**  
  - `5b_fpgrowth_analysis/outputs/...` (itemsets, rules, metrics, encoding maps)  
  - `5b_fpgrowth_analysis/outputs/feature_engineering/fpgrowth_features_*`  
  - `5b_fpgrowth_analysis/outputs/feature_engineering/fpgrowth_added_features_*`
- **Outputs (mirrored for convenience):**  
  - `feature_engineering_outputs/4_fpgrowth/opioid_ed/0-12/fpgrowth_features_opioid_ed_0_12.csv`  
  - `feature_engineering_outputs/4_fpgrowth/opioid_ed/0-12/fpgrowth_added_features_opioid_ed_0_12.csv`  
  - `feature_engineering_outputs/4_fpgrowth/opioid_ed/0-12/plots/` (all FP-Growth plots)
- **Status:** ⏳ Pending (to be rerun end‑to‑end).

### Step 5a: BupaR Process Mining ⏳
- **Goal:** Rebuild BupaR event logs, pre-/post-target features, and merged patient-level table.
- **Scripts:**  
  - `5a_bupaR_analysis/create_bupar_outputs_opioid_ed.R`  
  - `5a_bupaR_analysis/add_bupar_features_to_model_data.R`
- **Inputs:**  
  - `4a_model_data/.../model_events[_no_protocols].parquet` (via create_bupar scripts)  
  - FP-Growth target-only itemsets from `5b_fpgrowth_analysis/outputs/...`
- **Outputs (authoritative):**  
  - `5a_bupaR_analysis/outputs/opioid_ed/0_12/features/...`  
  - `5a_bupaR_analysis/outputs/feature_engineering/bupaR_added_features_opioid_ed_0_12.csv`
- **Outputs (mirrored for convenience):**  
  - `feature_engineering_outputs/5_bupar/opioid_ed/0-12/bupaR_added_features_opioid_ed_0_12.csv`  
  - `feature_engineering_outputs/5_bupar/opioid_ed/0-12/sequence_features_opioid_ed_0_12.csv` (if present)  
  - `feature_engineering_outputs/5_bupar/opioid_ed/0-12/plots/` (Gantt charts, activity plots, etc.)
- **Status:** ⏳ Pending (to be rerun with new event data + central mirroring).

### Step 6: DTW Trajectory Features ⏳
- **Goal:** Rebuild DTW trajectory features and merge them into a single patient-level block.
- **Scripts:**  
  - `5d_dtw_analysis/create_dtw_features.py`  
  - `5d_dtw_analysis/add_dtw_features_to_model_data.py`
- **Inputs:**  
  - Protocol-filtered model data from Step 4b  
  - FP-Growth itemsets (`5b_fpgrowth_analysis/outputs/...`)
- **Outputs (authoritative):**  
  - `5d_dtw_analysis/outputs/feature_engineering/dtw_features_opioid_ed_0_12.csv`  
  - `5d_dtw_analysis/outputs/feature_engineering/dtw_added_features_opioid_ed_0_12.csv`
- **Outputs (mirrored for convenience):**  
  - `feature_engineering_outputs/6_dtw/opioid_ed/0-12/dtw_features_opioid_ed_0_12.csv`  
  - `feature_engineering_outputs/6_dtw/opioid_ed/0-12/dtw_added_features_opioid_ed_0_12.csv`
- **Status:** ⏳ Pending.

### Step 5c: PGx Feature Engineering ⏳
- **Goal:** Rebuild PGx patient-level and added-features tables, and mirror to central outputs.
- **Scripts:**  
  - `5c_pgx_analysis/create_pgx_features.py`  
  - `5c_pgx_analysis/add_pgx_features_to_model_data.py`
- **Inputs:**  
  - Aggregated feature importance (`3_feature_importance/outputs/...`)  
  - Drug–gene mappings, allele frequencies (`7_pgx_analysis/outputs/...`)  
  - Model events from `4a_model_data/...` (for exposure linking)
- **Outputs (authoritative):**  
  - `5c_pgx_analysis/outputs/feature_engineering/pgx_features_opioid_ed_0_12.csv`  
  - `5c_pgx_analysis/outputs/feature_engineering/pgx_added_features_opioid_ed_0_12.csv`
- **Outputs (mirrored for convenience):**  
  - `feature_engineering_outputs/7_pgx/opioid_ed/0-12/pgx_features_opioid_ed_0_12.csv`  
  - `feature_engineering_outputs/7_pgx/opioid_ed/0-12/pgx_added_features_opioid_ed_0_12.csv`
- **Status:** ⏳ Pending.

### Step 6 Final Model: GPU-Preferred XGBoost / RF ⏳
- **Goal:** Use the refactored `6b_final_model_selection/run_final_model.py` to assemble the final feature matrix
  from `4a_model_data` + mirrored feature blocks and train/evaluate a GPU-accelerated classifier when available.
- **Script:** `6b_final_model_selection/run_final_model.py`
- **Inputs:**  
  - `4a_model_data/cohort_name=opioid_ed/age_band=0-12/model_events.parquet`  
  - `feature_engineering_outputs/4_fpgrowth/opioid_ed/0-12/fpgrowth_added_features_...csv`  
  - `feature_engineering_outputs/5_bupar/opioid_ed/0-12/bupaR_added_features_...csv`  
  - `feature_engineering_outputs/6_dtw/opioid_ed/0-12/dtw_added_features_...csv`  
  - `feature_engineering_outputs/7_pgx/opioid_ed/0-12/pgx_added_features_...csv`
- **Outputs:**  
  - Updated evaluation metrics (AUC, PR-AUC, logloss, classification report)
  - Model object (in-memory) and any CSV/plots written by `run_final_model.py`
- **Status:** ⏳ Pending.

---

## Per-Cohort Checkpoints

For each `(cohort, age_band)` we track the following high-level checkpoints:

1. **Feature engineering complete (no skipped steps)**  
   - FP-Growth (`4_fpgrowth`), BupaR (`5_bupar`), DTW (`6_dtw`), and PGx (`7_pgx`) all present under `5_feature_engineering/feature_engineering_outputs/{step}/{cohort}/{age_band}/`.  
   - If a step yields zero or trivial features, annotate the reason (e.g., no events, cohort too small).
2. **Final model selection complete**  
   - `6_final_model/outputs/{cohort}/{age_band_fname}/...` present, including `*_train_final_features_no_leakage.csv` and `final_model_json/*.json`.  
3. **FFA analysis complete**  
   - `7_ffa_analysis/outputs/{cohort}/{age_band_fname}/...` populated (AXP explanations, feature_importance_axp, causal_importance, visualizations).  
4. **SHAP value analysis complete**  
   - `8_shap_analysis/outputs/{cohort}/{age_band_fname}/...` populated with SHAP global importances and value arrays for both XGBoost and CatBoost (for example, `*_shap_global_importance_xgboost.csv`, `*_shap_global_importance_catboost.csv`, and their corresponding `*_shap_sample_values_*.parquet` plus summary plots).  
5. **Final model artifacts for risk dashboard saved**  
   - `6_final_model/model_outputs/{cohort}/{age_band_fname}/...` contains XGBoost and CatBoost JSON + CBM (and/or joblib) for use by FFA, SHAP, and the risk dashboard.

These checkpoints are applied across Cohort 1 (`opioid_ed`) and Cohort 2 (`non_opioid_ed`) for each age band in the modeling grid.

---

### Per-Cohort Grid – Main Cohorts (Current Status After Output Reset)

Legend:  
- `PENDING` – Step not yet (re)run on the refactored pipeline / outputs cleared.  
- `DONE` – Confirmed by presence of the expected artifacts.  
- `TEST` – Smoke-test only; not required for production risk dashboard.  
- `IGNORED` – Cohort/age band intentionally out of scope for modeling.

**Cohort 1 – Opioid ED (`opioid_ed`)**

| Cohort      | Age Band | 1. Feature Engineering | 2. Final Model Selection | 3. FFA Analysis | 4. SHAP Analysis | 5. Dashboard Artifacts | Notes                                  |
|------------|----------|------------------------|--------------------------|-----------------|------------------|------------------------|----------------------------------------|
| opioid_ed  | 0-12     | TEST                   | TEST                     | TEST            | TEST             | TEST                   | Test-only cohort; pipeline smoke test. |
| opioid_ed  | 13-24    | DONE                   | DONE                     | DONE            | DONE             | DONE                   | DTW and FP-Growth yielded trivial/empty features; BupaR, PGx, final model, FFA, SHAP, and combined SHAP+FFA all completed on DTW-filtered 4a_model_data. |
| opioid_ed  | 25-44    | DONE                   | DONE                     | DONE            | DONE             | DONE                   | DTW and FP-Growth yielded trivial/empty features; BupaR, PGx, final model, FFA, SHAP, and combined SHAP+FFA all completed on DTW-filtered 4a_model_data. |
| opioid_ed  | 45-54    | DONE                   | DONE                     | DONE            | DONE             | DONE                   | DTW and FP-Growth yielded trivial/empty features; BupaR, PGx, final model, FFA, SHAP (XGBoost + CatBoost), and combined SHAP+FFA all completed on DTW-filtered 4a_model_data. |
| opioid_ed  | 55-64    | PENDING                | PENDING                  | PENDING         | PENDING          | PENDING                | Planned production cohort.             |

**Cohort 2 – Polypharmacy ED (`non_opioid_ed`)**

| Cohort         | Age Band | 1. Feature Engineering | 2. Final Model Selection | 3. FFA Analysis | 4. SHAP Analysis | 5. Dashboard Artifacts | Notes                      |
|---------------|----------|------------------------|--------------------------|-----------------|------------------|------------------------|----------------------------|
| non_opioid_ed | 65-74    | PENDING                | PENDING                  | PENDING         | PENDING          | PENDING                | Primary production cohort. |
| non_opioid_ed | 75-84    | PENDING                | PENDING                  | PENDING         | PENDING          | PENDING                | Primary production cohort. |
| non_opioid_ed | 85-94    | PENDING                | PENDING                  | PENDING         | PENDING          | PENDING                | Primary production cohort. |
| non_opioid_ed | 95-114   | IGNORED                | IGNORED                  | IGNORED         | IGNORED          | IGNORED                | Explicitly excluded cohort.|

> As of the latest reset, all downstream feature engineering outputs and final model artifacts have been cleared, so all non-test, non-ignored cells are marked `PENDING`. As runs complete and artifacts appear on disk, update the corresponding cells in this table to `DONE` together with brief notes (e.g., command used, commit hash, or run date).

---

### Extreme-Density Cohort Grid (Parallel Checklist)

For each `(cohort, age_band)` where the FP-Growth / process-mining pipeline is run, we also maintain a **parallel grid for the derived extreme-density cohorts**. These cohorts capture the top ~5% of patients by medical_code transaction density and are modeled separately so they do not drive the main cohort models.

Legend (same as above):  
- `PENDING` – Extreme-density split and/or downstream analysis not yet run.  
- `DONE` – Extreme-density split + all downstream steps completed.  
- `IGNORED` – Extreme-density cohort intentionally not modeled for this cell.

**Cohort 1 – Opioid ED Extreme-Density (`opioid_ed_extreme_density`)**

| Cohort                 | Age Band | 1. Feature Engineering | 2. Final Model Selection | 3. FFA Analysis | 4. SHAP Analysis | 5. Dashboard Artifacts | Notes                                                           |
|------------------------|----------|------------------------|--------------------------|-----------------|------------------|------------------------|-----------------------------------------------------------------|
| opioid_ed_extreme_density | 13-24 | PENDING                | PENDING                  | PENDING         | PENDING          | PENDING                | Optional; can mirror main-cohort pipeline if density cohort is of interest. |
| opioid_ed_extreme_density | 25-44 | PENDING                | PENDING                  | PENDING         | PENDING          | PENDING                | Historical extreme-density split exists in legacy layout; refactor to 4a_model_data as needed. |
| opioid_ed_extreme_density | 45-54 | PENDING                | PENDING                  | PENDING         | PENDING          | PENDING                | Planned once main 45–54 refactor is complete.                  |
| opioid_ed_extreme_density | 55-64 | PENDING                | PENDING                  | PENDING         | PENDING          | PENDING                | Extreme-density split implemented in `4a_model_data`; rerun Steps 5–8 on this cohort. |

**Cohort 2 – Polypharmacy ED Extreme-Density (`non_opioid_ed_extreme_density`)**

| Cohort                        | Age Band | 1. Feature Engineering | 2. Final Model Selection | 3. FFA Analysis | 4. SHAP Analysis | 5. Dashboard Artifacts | Notes                                                           |
|-------------------------------|----------|------------------------|--------------------------|-----------------|------------------|------------------------|-----------------------------------------------------------------|
| non_opioid_ed_extreme_density | 65-74   | PENDING                | PENDING                  | PENDING         | PENDING          | PENDING                | To be created via `extract_extreme_density_cohort.py` prior to FP-Growth. |
| non_opioid_ed_extreme_density | 75-84   | PENDING                | PENDING                  | PENDING         | PENDING          | PENDING                | To be created via `extract_extreme_density_cohort.py` prior to FP-Growth. |
| non_opioid_ed_extreme_density | 85-94   | PENDING                | PENDING                  | PENDING         | PENDING          | PENDING                | To be created via `extract_extreme_density_cohort.py` prior to FP-Growth. |

---

## Execution Log

### 2025-12-31 – Workflow Layout Updated
- ✅ Preserve Step 3 Feature Importance artifacts under `3_feature_importance/outputs/`.
- ✅ Update scripts and paths to use `4a_model_data/` and `feature_engineering_outputs/{step}/{cohort}/{age_band}/`.

