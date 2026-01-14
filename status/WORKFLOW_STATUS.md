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

## Current Workflow (Final Production Pipeline)

**As of:** 2026-01-07  
**Status:** ✅ Final production workflow definition

**Workflow Execution:**
```bash
# Single cohort/age band
bash utility_scripts/run_cohort_workflow.sh <cohort_name> <age_band>

# All cohorts in a group
bash utility_scripts/run_opioid_ed_workflow.sh
bash utility_scripts/run_non_opioid_ed_workflow.sh

# All cohorts
bash utility_scripts/run_all_cohorts_workflow.sh
```

**Performance Configuration:**
- DuckDB threads: 4 per connection (optimized for 32-core EC2)
- Expected CPU utilization: ~28 cores (87.5%) when running all 7 cohorts in parallel
- Memory: 512GB per DuckDB connection (50% of 1TB RAM)

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

### Step 5c: PGx Feature Engineering ✅
- **Goal:** Build PGx patient-level features from drug-gene mappings and allele frequencies.
- **Script:** `5_pgx_analysis/run_analysis.py`
- **Inputs:**  
  - Aggregated feature importance (`3_feature_importance/outputs/...`)  
  - Drug–gene mappings, allele frequencies (`5_pgx_analysis/outputs/...`)  
  - Model events from `4a_model_data/...` (for exposure linking)
- **Outputs:**  
  - `5_pgx_analysis/outputs/feature_engineering/pgx_features_{cohort}_{age_band}.csv`  
  - `5_pgx_analysis/outputs/feature_engineering/pgx_added_features_{cohort}_{age_band}.csv`
  - S3: `s3://pgxdatalake/gold/pgx_features/{cohort}/{age_band}/pgx_added_features_*.csv`
- **Status:** ✅ Complete (idempotent, uses aggregated feature importances + PGx features only)

### Step 6: Final Model Training ✅
- **Goal:** Assemble final feature matrix from aggregated feature importances + PGx features, train CatBoost and XGBoost models, select best by recall/AUC-PR.
- **Script:** `6_final_model_selection/run_final_model.py`
- **Inputs:**  
  - `4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events_no_protocols.parquet`  
  - Aggregated feature importances (`3_feature_importance/outputs/...`)
  - PGx features (`5_pgx_analysis/outputs/feature_engineering/pgx_added_features_*.csv`)
- **Outputs:**  
  - `6_final_model/outputs/{cohort}/{age_band_fname}/final_model_json/*.json` (XGBoost JSON for FFA)
  - `6_final_model/outputs/{cohort}/{age_band_fname}/*.cbm` (CatBoost binary for SHAP)
  - `6_final_model/outputs/{cohort}/{age_band_fname}/*_train_final_features_no_leakage.csv`
  - Model evaluation metrics (AUC, PR-AUC, logloss, classification report)
- **Status:** ✅ Complete (uses aggregated features + PGx, no encoding step)

### Step 7: SHAP Analysis ✅
- **Goal:** Compute SHAP values for both XGBoost and CatBoost models (global importance + row-level values).
- **Script:** `7_shap_analysis/run_shap_analysis.py`
- **Inputs:**  
  - Best CatBoost model binary (`6_final_model/outputs/.../*.cbm`)
  - Best XGBoost model JSON (`6_final_model/outputs/.../final_model_json/*.json`)
  - Final features CSV (`6_final_model/outputs/.../*_train_final_features_no_leakage.csv`)
- **Outputs:**  
  - `s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/`
    - `*_shap_global_importance_xgboost.csv`
    - `*_shap_global_importance_catboost.csv`
    - `*_shap_sample_values_xgboost.parquet`
    - `*_shap_sample_values_catboost.parquet`
- **Method:** Two-pass streamed approach (global signal, then row-level for selected features)
- **Status:** ✅ Complete (required before Step 8 FFA analysis)

### Step 8: FFA Analysis ✅
- **Goal:** Formal Feature Attribution analysis using symbolic rule extraction and anchored explanations (AXP).
- **Script:** `8_ffa_analysis/run_full_ffa_analysis.py`
- **Model Support:** 
  - **XGBoost FFA**: ✅ Performed (direct rule extraction from JSON)
  - **CatBoost FFA**: ❌ NOT performed (due to complex hashing/CTR for categorical variables)
  - **CatBoost SHAP**: Used for feature importance filtering in XGBoost FFA
- **Inputs:**  
  - Best XGBoost model JSON (`6_final_model/outputs/.../final_model_json/*.json`)
  - SHAP importance from Step 7 (both XGBoost and CatBoost SHAP values)
  - Final features CSV (`6_final_model/outputs/.../*_train_final_features_no_leakage.csv`)
- **Rule Selection Logic:** Union of three sets:
  1. First 100 matched rules (common patterns)
  2. Random sample of 100 matched rules (diversity)
  3. Top 300 SHAP-filtered rules OR all rules above 10th percentile (whichever is larger)
     - Uses SHAP importance from both XGBoost and CatBoost to filter/prioritize rules
- **Outputs:**  
  - `s3://pgxdatalake/gold/ffa_analysis/{cohort}/{age_band}/{model_type}/`
    - `axp_explanations.csv`
    - `feature_importance_axp.csv`
    - `causal_importance.csv`
    - `interaction_analysis.csv`
- **Status:** ✅ Complete (XGBoost only, uses SHAP from Step 7 to filter rules)

### Step 9: Risk Dashboard ✅
- **Goal:** Generate risk dashboard with BupaR/DTW/FP-Growth visualizations and causal analysis.
- **Script:** `9_risk_dashboard/...` (visualization scripts)
- **Inputs:**  
  - FFA analysis outputs (Step 8)
  - SHAP analysis outputs (Step 7)
  - Model artifacts (Step 6)
- **Outputs:**  
  - Interactive dashboards (Plotly HTML)
  - Static visualizations (PNG)
  - Causal analysis reports
- **Status:** ✅ Complete (includes visualizations only, not separate analysis steps)

---

## Per-Cohort Checkpoints

For each `(cohort, age_band)` we track the following high-level checkpoints:

1. **Step 3: Feature Importance Complete** ✅
   - Aggregated feature importances present under `3_feature_importance/outputs/{cohort}/{age_band}/`
   - Used as input for Step 4a and Step 5c

2. **Step 4a: Model Data Extraction Complete** ✅
   - `4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet` present
   - Contains cases + controls for model training

3. **Step 4b: DTW Protocol Filtering Complete** ✅
   - `4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events_no_protocols.parquet` present
   - Administrative/scheduling codes filtered out

4. **Step 5c: PGx Feature Engineering Complete** ✅
   - `5_pgx_analysis/outputs/feature_engineering/pgx_added_features_{cohort}_{age_band}.csv` present
   - S3: `s3://pgxdatalake/gold/pgx_features/{cohort}/{age_band}/pgx_added_features_*.csv`

5. **Step 6: Final Model Training Complete** ✅
   - `6_final_model/outputs/{cohort}/{age_band_fname}/final_model_json/*.json` (XGBoost JSON for FFA)
   - `6_final_model/outputs/{cohort}/{age_band_fname}/*.cbm` (CatBoost binary for SHAP)
   - `6_final_model/outputs/{cohort}/{age_band_fname}/*_train_final_features_no_leakage.csv`

6. **Step 7: SHAP Analysis Complete** ✅
   - S3: `s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/`
     - `*_shap_global_importance_xgboost.csv`
     - `*_shap_global_importance_catboost.csv`
     - `*_shap_sample_values_xgboost.parquet`
     - `*_shap_sample_values_catboost.parquet`
   - **Required before Step 8** (FFA uses SHAP importance to filter rules)

7. **Step 8: FFA Analysis Complete** ✅
   - S3: `s3://pgxdatalake/gold/ffa_analysis/{cohort}/{age_band}/xgboost/`
     - `axp_explanations.csv`
     - `feature_importance_axp.csv`
     - `causal_importance.csv`
     - `interaction_analysis.csv`
   - **Note:** Only XGBoost FFA is performed (CatBoost FFA not performed due to hashing/CTR complexity)
   - Uses SHAP importance from both XGBoost and CatBoost (from Step 7) to filter/prioritize rules

8. **Step 9: Risk Dashboard Complete** ✅
   - Dashboard visualizations and causal analysis reports generated
   - Interactive dashboards (Plotly HTML) and static visualizations (PNG)

These checkpoints are applied across Cohort 1 (`opioid_ed`) and Cohort 2 (`non_opioid_ed`) for each age band in the modeling grid.

---

### Per-Cohort Grid – Main Cohorts (Current Status After Output Reset)

Legend:  
- `PENDING` – Step not yet (re)run on the refactored pipeline / outputs cleared.  
- `DONE` – Confirmed by presence of the expected artifacts.  
- `TEST` – Smoke-test only; not required for production risk dashboard.  
- `IGNORED` – Cohort/age band intentionally out of scope for modeling.

**Cohort 1 – Opioid ED (`opioid_ed`)**

| Cohort      | Age Band | 3. Feature Importance | 4a. Model Data | 4b. DTW Filter | 5c. PGx Features | 6. Final Model | 7. SHAP Analysis | 8. FFA Analysis | 9. Dashboard | Notes                                  |
|------------|----------|----------------------|----------------|----------------|------------------|----------------|------------------|-----------------|-------------|----------------------------------------|
| opioid_ed  | 0-12     | TEST                 | TEST           | TEST           | TEST             | TEST           | TEST             | TEST            | TEST        | Test-only cohort; pipeline smoke test. |
| opioid_ed  | 13-24    | DONE                 | DONE           | DONE           | DONE             | DONE           | DONE             | DONE            | DONE         | All steps completed on DTW-filtered 4a_model_data. |
| opioid_ed  | 25-44    | DONE                 | DONE           | DONE           | DONE             | DONE           | DONE             | DONE            | DONE         | All steps completed on DTW-filtered 4a_model_data. |
| opioid_ed  | 45-54    | DONE                 | DONE           | DONE           | DONE             | DONE           | DONE             | DONE            | DONE         | All steps completed on DTW-filtered 4a_model_data. |
| opioid_ed  | 55-64    | PENDING              | PENDING        | PENDING        | PENDING          | PENDING        | PENDING          | PENDING         | PENDING      | Planned production cohort.             |

**Cohort 2 – Polypharmacy ED (`non_opioid_ed`)**

| Cohort         | Age Band | 3. Feature Importance | 4a. Model Data | 4b. DTW Filter | 5c. PGx Features | 6. Final Model | 7. SHAP Analysis | 8. FFA Analysis | 9. Dashboard | Notes                      |
|---------------|----------|----------------------|----------------|----------------|------------------|----------------|------------------|-----------------|-------------|----------------------------|
| non_opioid_ed | 65-74    | PENDING              | PENDING        | PENDING        | PENDING          | PENDING        | PENDING          | PENDING         | PENDING      | Primary production cohort. |
| non_opioid_ed | 75-84    | PENDING              | PENDING        | PENDING        | PENDING          | PENDING        | PENDING          | PENDING         | PENDING      | Primary production cohort. |
| non_opioid_ed | 85-94    | PENDING              | PENDING        | PENDING        | PENDING          | PENDING        | PENDING          | PENDING         | PENDING      | Primary production cohort. |
| non_opioid_ed | 95-114   | IGNORED              | IGNORED        | IGNORED        | IGNORED          | IGNORED        | IGNORED          | IGNORED         | IGNORED      | Explicitly excluded cohort.|

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

### 2026-01-07 – Final Production Workflow Established
- ✅ Final workflow steps defined: 3 → 4a → 4b → 5c → 6 → 7 → 8 → 9
- ✅ CatBoost FFA removed (not performed due to hashing/CTR complexity)
- ✅ CatBoost SHAP used for feature importance filtering in XGBoost FFA
- ✅ Rule selection logic: first 100 + random 100 + top 300 SHAP-filtered rules
- ✅ DuckDB threads increased to 4 per connection (optimized for 32-core EC2)
- ✅ Workflow execution commands documented in top-level README
- ✅ All steps are idempotent (skip completed steps automatically)

### 2025-12-31 – Workflow Layout Updated
- ✅ Preserve Step 3 Feature Importance artifacts under `3_feature_importance/outputs/`.
- ✅ Update scripts and paths to use `4a_model_data/` and `feature_engineering_outputs/{step}/{cohort}/{age_band}/`.

