# Workflow Execution Status

**Cohort:** `opioid_ed`  
**Age Band:** `0-12`  

---

## Previous End-to-End Run (2025-12-10) – LEGACY PIPELINE

Status: ✅ Complete (for historical reference only; used `4_model_data` and older step layout).  
Model performance and step details are preserved below but superseded by the refactored run.

### Summary (Legacy)
- Step 3: Feature Importance – complete, artifacts preserved under `3a_feature_importance/outputs/`
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

**Workflow Execution:** Run via the three workflow notebooks (`1_cohort_workflow.ipynb`, `2_feature_importance.ipynb`, `3_pgx_calculator_workflow.ipynb`). Legacy shell scripts are in `archived/utility_scripts/`:
```bash
# Single cohort/age band (legacy)
bash archived/utility_scripts/run_cohort_workflow.sh <cohort_name> <age_band>

# All cohorts (legacy)
bash archived/utility_scripts/run_opioid_ed_workflow.sh
bash archived/utility_scripts/run_non_opioid_ed_workflow.sh
bash archived/utility_scripts/run_all_cohorts_workflow.sh
```

**Performance Configuration:**
- DuckDB threads: 4 per connection (optimized for 32-core EC2)
- Expected CPU utilization: ~28 cores (87.5%) when running all 7 cohorts in parallel
- Memory: 512GB per DuckDB connection (50% of 1TB RAM)

### Step 3: Feature Importance ✅
- **Status:** ✅ Complete (reused from previous run)
- **Outputs:** `3a_feature_importance/outputs/`
- **Notes:** Not regenerated; serves as input to `4_model_data/create_model_data.py` and PGx scripts.

### Step 4: Model Data (4_model_data) ⏳
- **Goal:** Build within-cohort model-ready event data (cases + clean controls) for `opioid_ed / 0-12`.
- **Script:** `4_model_data/create_model_data.py`
- **Outputs:**  
  - `4_model_data/cohort_name=opioid_ed/age_band=0-12/model_events.parquet`
- **Status:** ⏳ To (re)run with updated DuckDB-only implementation.
- **Note:** Event filtering (administrative/scheduling codes) is in Step 1b (`1b_apcd_event_filter`). DTW protocol filtering is for dashboard visualizations only (Step 9).

### Step 4 (optional): Extreme-Density Cohort Split ⏳
- **Goal:** Optional extraction of patients with extremely dense medical_code trajectories for dashboard visualizations (not part of main pipeline).
- **Inputs/Outputs:** If used, extreme-density outputs live under `4_model_data/` or dashboard-specific paths.
- **Status:** Optional; main pipeline uses `4_model_data/` only.

**Note:** FP-Growth, BupaR, and DTW are used for dashboard visualizations only (Step 9, `9_risk_dashboard/`), not as pipeline steps. Main pipeline: 3a → 4 → 5 → 6 → 7 → 8 → 9.

### Step 5: PGx Feature Engineering ✅
- **Goal:** Build PGx patient-level features from drug-gene mappings and allele frequencies.
- **Script:** `5_pgx_analysis/run_analysis.py`
- **Inputs:**  
  - Aggregated feature importance (`3a_feature_importance/outputs/...`)  
  - Drug–gene mappings, allele frequencies (`5_pgx_analysis/outputs/...`)  
  - Model events from `4_model_data/...` (for exposure linking)
- **Outputs:**  
  - `5_pgx_analysis/outputs/feature_engineering/pgx_features_{cohort}_{age_band}.csv`  
  - `5_pgx_analysis/outputs/feature_engineering/pgx_added_features_{cohort}_{age_band}.csv`
  - S3: `s3://pgxdatalake/gold/pgx_features/{cohort}/{age_band}/pgx_added_features_*.csv`
- **Status:** ✅ Complete (idempotent, uses aggregated feature importances + PGx features only)

### Step 6: Final Model Training ✅
- **Goal:** Assemble final feature matrix from aggregated feature importances + PGx features, train CatBoost and XGBoost models, select best by recall/AUC-PR.
- **Script:** `6_final_model/run_final_model.py`
- **Inputs:**  
  - `4_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet` (event filtering in Step 1b)  
  - Aggregated feature importances (`3a_feature_importance/outputs/...`)
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
   - Aggregated feature importances present under `3a_feature_importance/outputs/{cohort}/{age_band}/`
   - Used as input for Step 4 and Step 5

2. **Step 4: Model Data Complete** ✅
   - `4_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet` present
   - Contains cases + controls for model training (event filtering in Step 1b)

3. **Step 5: PGx Feature Engineering Complete** ✅
   - `5_pgx_analysis/outputs/feature_engineering/pgx_added_features_{cohort}_{age_band}.csv` present
   - S3: `s3://pgxdatalake/gold/pgx_features/{cohort}/{age_band}/pgx_added_features_*.csv`

5. **Step 6: Final Model Training Complete** ✅
   - `6_final_model/outputs/{cohort}/{age_band_fname}/final_model_json/*.json` (XGBoost JSON for FFA)
   - `6_final_model/outputs/{cohort}/{age_band_fname}/*.cbm` (CatBoost binary for SHAP)
   - `6_final_model/outputs/{cohort}/{age_band_fname}/*_train_final_features_no_leakage.csv`

5. **Step 7: SHAP Analysis Complete** ✅
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

7. **Step 9: Risk Dashboard Complete** ✅
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
| opioid_ed  | 13-24    | DONE                 | DONE           | DONE           | DONE             | DONE           | DONE             | DONE            | DONE         | All steps completed (4_model_data). |
| opioid_ed  | 25-44    | DONE                 | DONE           | DONE           | DONE             | DONE           | DONE             | DONE            | DONE         | All steps completed (4_model_data). |
| opioid_ed  | 45-54    | DONE                 | DONE           | DONE           | DONE             | DONE           | DONE             | DONE            | DONE         | All steps completed (4_model_data). |
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
| opioid_ed_extreme_density | 25-44 | PENDING   | PENDING | PENDING | PENDING | PENDING | Historical extreme-density split; refactor to 4_model_data as needed. |
| opioid_ed_extreme_density | 45-54 | PENDING   | PENDING | PENDING | PENDING | PENDING | Planned once main 45–54 refactor is complete. |
| opioid_ed_extreme_density | 55-64 | PENDING   | PENDING | PENDING | PENDING | PENDING | Extreme-density in `4_model_data`; rerun Steps 5–8 if used. |

**Cohort 2 – Polypharmacy ED Extreme-Density (`non_opioid_ed_extreme_density`)**

| Cohort                        | Age Band | 1. Feature Engineering | 2. Final Model Selection | 3. FFA Analysis | 4. SHAP Analysis | 5. Dashboard Artifacts | Notes                                                           |
|-------------------------------|----------|------------------------|--------------------------|-----------------|------------------|------------------------|-----------------------------------------------------------------|
| non_opioid_ed_extreme_density | 65-74   | PENDING                | PENDING                  | PENDING         | PENDING          | PENDING                | To be created via `extract_extreme_density_cohort.py` prior to FP-Growth. |
| non_opioid_ed_extreme_density | 75-84   | PENDING                | PENDING                  | PENDING         | PENDING          | PENDING                | To be created via `extract_extreme_density_cohort.py` prior to FP-Growth. |
| non_opioid_ed_extreme_density | 85-94   | PENDING                | PENDING                  | PENDING         | PENDING          | PENDING                | To be created via `extract_extreme_density_cohort.py` prior to FP-Growth. |

---

## Execution Log

### 2026-01-07 – Final Production Workflow Established
- ✅ Final workflow steps: 3 → 4 → 5 → 6 → 7 → 8 → 9 (notebooks: 1_cohort_workflow, 2_feature_importance, 3_pgx_calculator_workflow)
- ✅ Event filtering in Step 1b (`1b_apcd_event_filter`); model data in `4_model_data/`
- ✅ CatBoost FFA removed (not performed); CatBoost SHAP used for feature importance filtering in XGBoost FFA
- ✅ Rule selection logic: first 100 + random 100 + top 300 SHAP-filtered rules
- ✅ DuckDB threads increased to 4 per connection (optimized for 32-core EC2)
- ✅ Workflow execution via three notebooks; legacy scripts in `archived/utility_scripts/`
- ✅ All steps are idempotent (skip completed steps automatically)

### 2025-12-31 – Workflow Layout Updated
- ✅ Step 3 Feature Importance artifacts under `3a_feature_importance/outputs/`.
- ✅ Scripts and paths use `4_model_data/`; BupaR/DTW/FP-Growth for dashboard only (Step 9).

