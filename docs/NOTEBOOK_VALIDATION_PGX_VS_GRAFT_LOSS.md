# PGx Calculator Workflow – Validation vs Graft-Loss Calculator

This document validates `3_pgx_calculator_workflow.ipynb` against the structure and practices of the graft-loss calculator (`C:\Projects\phts\graft-loss\cohort_analysis\calculator`).

**Reference:**  
- Graft-loss: `calculator_workflow.ipynb`, `NOTEBOOK_STRUCTURE_CHECK.md`, `PATH_VERIFICATION.md`  
- PGx: `3_pgx_calculator_workflow.ipynb`

---

## 1. Workflow step mapping

| Graft-loss calculator | PGx calculator workflow | Notes |
|------------------------|--------------------------|--------|
| **1. Input features / setup** | Setup + config (paths, cohorts, DATA_ROOT, FI_ROOT, MODEL_DATA_ROOT) | ✅ PGx has config cell and cohort/age_band table |
| **2. Sync / inputs** | Sync from S3 (cohorts, feature_importance, final_model) | ✅ Idempotent sync cell |
| **3a. Train Baseline** | Pipeline Step 4 → 5 → 6 (model data, PGx analysis, final model) | ✅ Per cohort/age_band; PGx uses single “best” model per band |
| **3b. Train Extended** | (N/A – PGx has no baseline/extended variants) | Different design: PGx = one model per cohort/age_band |
| **4a. SHAP/FFA Baseline** | Step 7 (SHAP) + Step 8 (FFA) run elsewhere; **Step 2: Combine SHAP/FFA** in notebook | ✅ Combine cell calls `combine_shap_ffa_results.py` per cohort/age_band |
| **4b. SHAP/FFA Extended** | (N/A) | — |
| **5. Results inspection** | No dedicated “inspect results” section | ⚠ Gap: no cell to view top causal factors / dashboard_data.json per cohort |
| **6. Deploy** | Step 1 (metadata) → Step 3 (prepare models) → Step 4 (Lambda dir) → Step 5 (verify) → Step 6 (build/deploy) | ✅ All present; order in notebook: Combine SHAP/FFA (Step 2) then metadata (Step 1) then prepare models (Step 3) |

---

## 2. Required elements (from graft-loss NOTEBOOK_STRUCTURE_CHECK)

### Training

| Requirement | PGx | Status |
|-------------|-----|--------|
| Train models per cohort/variant | Pipeline Step 4–6 per (cohort, age_band) | ✅ |
| Feature importance available | Step 3/3b FI as input; Step 6 writes feature importance CSVs | ✅ |
| Best model / selection | Step 6 selects best XGBoost variant (xgb vs xgb_rf), saves as xgboost.joblib | ✅ |

### SHAP/FFA

| Requirement | PGx | Status |
|-------------|-----|--------|
| SHAP/FFA run and combined | Step 7 & 8 run externally; **Step 2: Combine SHAP/FFA** in notebook | ✅ |
| Dashboard-oriented output | combine_shap_ffa_results.py → outputs (e.g. dashboard_data.json, top factors) | ✅ |
| Per cohort/age_band | Loop over REQUIRED_COHORTS in combine cell | ✅ |

### Feature importance

| Requirement | PGx | Status |
|-------------|-----|--------|
| Check model/importance paths | Step 0 verifies FI (and cohorts, model data, Step 6); no “Section 5a” style importance view | ⚠ Step 0 checks existence only; no “display top features” cell |

### Results inspection

| Requirement | PGx | Status |
|-------------|-----|--------|
| Inspect causal factors / dashboard_data | No cell that loads and displays dashboard_data.json or top_causal_factors per cohort | ⚠ Gap |

### Deployment

| Requirement | PGx | Status |
|-------------|-----|--------|
| Prepare models for dashboard | Step 3: prepare_models.py --all | ✅ |
| Lambda directory | Step 4: prepare_lambda_dir.py | ✅ |
| Verify then build/deploy | Step 5 (verify), Step 6 (build/deploy) | ✅ |

---

## 3. Output directory structure (PGx vs graft-loss)

### Graft-loss

- Models: `outputs/models/{COHORT}_base/`, `outputs/models/{COHORT}_enhanced/`
- SHAP/FFA: `outputs/shap_ffa/{COHORT}_base/`, `outputs/shap_ffa/{COHORT}_enhanced/`  
  - dashboard_data.json, top_causal_factors.csv, combined_shap_importance.csv

### PGx

- Models: `6_final_model/outputs/{cohort}/{age_band_fname}/` (and gold/final_model); dashboard copy: `9_risk_dashboard/outputs/models/{cohort}/{age_band_fname}/`
- SHAP/FFA combined: `9_risk_dashboard/outputs/` (or path from combine_shap_ffa_results --output-dir) per cohort/age_band
- Lambda: assembled under `9_risk_dashboard` (deployment/lambda_dir or similar)

Layout is consistent with PGx’s multi-cohort, multi–age-band design; no structural mismatch.

---

## 4. Step numbering and order (PGx notebook)

Current section headers:

1. Sync (no number)  
2. **Step 0:** Verify inputs  
3. **Pipeline Step 4:** Model data  
4. **Pipeline Step 5:** PGx analysis  
5. **Pipeline Step 6:** Final model deployment (contains “Step 1: Train Models” as first sub-heading – this is the Step 6 pipeline run)  
6. **Step 2:** Combine SHAP and FFA results  
7. (Unnumbered) Generate metadata – checkpoint message says “Step 1”  
8. **Step 3:** Prepare models  
9. **Step 4:** Prepare Lambda directory  
10. **Step 5:** Verify Lambda directory  
11. **Step 6:** Build and deploy  

Observation: Dashboard steps are labeled 1–6 in the docs/overview, but in the notebook “Step 1” (metadata) appears after “Step 2” (Combine SHAP/FFA). For alignment with graft-loss and clarity, consider reordering so that “Step 1: Generate metadata” comes before “Step 2: Combine SHAP/FFA”, or explicitly document that “Step 2” is required before “Step 1” (e.g. “Step 2: Combine SHAP/FFA (required before metadata)”).

---

## 5. Gaps and recommendations (implemented)

| Gap | Recommendation | Status |
|-----|----------------|--------|
| No “Results inspection” section | Add a cell (or small section) that, for one or all cohort/age_bands, loads and prints or displays top causal factors / dashboard_data.json from the combine output directory. | **Done:** "Results inspection" section added. |
| No “Feature importance” display | Optional: add a cell that reads Step 6 (or Step 3b) feature importance for a chosen cohort/age_band and shows top N features. | **Done:** "Feature importance display" section added. |
| Step order (metadata vs combine) | Either move the “Step 1: Generate metadata” cell above “Step 2: Combine SHAP/FFA”, or add a short note in the markdown that Combine (Step 2) must run before metadata (Step 1) if causal data is needed for metadata. | **Done:** Step 1 moved above Step 2. |
| Combine script dependency | Ensure `combine_shap_ffa_results.py` is robust (indentation, args). | Already fixed in repo. |

---

## 6. Verification checklist (PGx)

- [x] Sync from S3 (cohorts, feature importance, final model)
- [x] Step 0: Verify inputs (FI required; cohorts, model data, Step 6 informational)
- [x] Pipeline Step 4: Model data (create_model_data.py)
- [x] Pipeline Step 5: PGx analysis (run_analysis.py)
- [x] Pipeline Step 6: Final model (run_final_model.py)
- [x] Step 2: Combine SHAP/FFA (combine_shap_ffa_results.py per cohort/age_band)
- [x] Step 1: Generate metadata (checkpoint; generate_metadata.py --all)
- [x] Step 3: Prepare models (checkpoint; prepare_models.py --all)
- [x] Step 4: Prepare Lambda directory
- [x] Step 5: Verify Lambda directory
- [x] Step 6: Build and deploy
- [x] Optional: Results inspection cell (top causal/consensus features, combined_importance, summary)
- [x] Optional: Feature importance display cell (Step 6 or Step 3b, top N)

---

## 7. Summary

- **Structure:** PGx workflow matches the graft-loss calculator pattern (sync → verify → train → SHAP/FFA combine → metadata → prepare models → Lambda → verify → deploy). The main difference is design (multiple cohort/age_bands and single best model per band vs baseline/enhanced variants).
- **Coverage:** All required training, SHAP/FFA combine, and deployment steps are present. Gaps are optional “inspection” and “feature importance” cells and minor step-order/naming clarity.
- **Outputs:** Output directories and roles align with graft-loss expectations, adapted to PGx paths and naming.

Validation date: 2026-02. Reference: `C:\Projects\phts\graft-loss\cohort_analysis\calculator`.
