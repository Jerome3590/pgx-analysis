# PGx Dashboard & Manuscript Run Plan

**Created:** 2026-03-29  
**Purpose:** Track EC2 notebook run order, code changes deployed, and post-run manuscript extraction steps.

---

## Code Changes Deployed (Local → Git → EC2)

All changes below are committed and ready. EC2 pulls from git before running.

| File | Change | Status |
|:-----|:-------|:------:|
| `10_risk_dashboard/backend/lambda_function.py` | Drop `n_events` + `n_event_bin` from feature schema at load time (`_normalize_feature_schema_for_training`) | ✅ |
| `10_risk_dashboard/backend/lambda_function.py` | `build_feature_vector`: new `n_event_bin` param → sets `n_event_bin_ordinal` after defaults (fixes low=0.0 override bug) | ✅ |
| `10_risk_dashboard/backend/lambda_function.py` | `build_feature_vector`: `auto_pgx_num_drugs` → derives `pgx_num_drugs` from `len(drugs)` | ✅ |
| `10_risk_dashboard/backend/lambda_function.py` | `handle_risk`: computes `n_event_bin_value` BEFORE building feature vector; passes `n_event_bin` through | ✅ |
| `10_risk_dashboard/backend/lambda_function.py` | `handle_risk_comparison`: computes bin per-scenario; passes `n_event_bin` to both base and scenario feature vectors | ✅ |
| `10_risk_dashboard/backend/lambda_function.py` | `handle_drug_contributions`: new `POST /risk/drug_contributions` endpoint for per-drug leave-one-out Δp̂ | ✅ |
| `10_risk_dashboard/backend/lambda_function.py` | `lambda_handler`: routing fixes — `/risk/drug_contributions` and `/risk/comparison` now reachable before `/risk` catch-all | ✅ |
| `6_final_model/run_final_model.py` | `_EXCLUDE_FROM_FEATURES` += `"n_events"` — models train without continuous claim count | ✅ |
| `6_final_model/build_final_cohort_model_features.py` | Drops `n_events` and `n_event_bin` (string) from saved training parquet before writing | ✅ |
| `10_risk_dashboard/data_preparation/prepare_models.py` | `_extract_feature_schema_duckdb` `exclude_cols` += `"n_events"` — `feature_schema.json` never includes it | ✅ |
| `4_dashboard_visuals.ipynb` | Added manuscript checkpoint writer cell (cells 51–52): writes FPG/DTW/SHAP/FFA/PGx summaries to `s3://pgxdatalake/gold/manuscript_checkpoints/` for all cohorts × age bands × bins | ✅ |
| `manuscript/scripts/extract_visual_manuscript.py` | New script: reads S3 manuscript checkpoints → writes `visual_manuscript_data.json`, `pgx_coverage.json`, `shap_top_features.json` | ✅ |

---

## EC2 Notebook Run Order

Run in this exact sequence. Each step is a prerequisite for the next.

### Step 1 — Notebook 3: Model Training
**File:** `final_model_cohort_runner.ipynb`  
**Script:** `6_final_model/run_final_model.py`

- [ ] Run notebook 3 (all cohorts, all age bands)
- Trains per-bin models (low/medium/high/extreme) **without** `n_events` feature
- Saves per-bin model artifacts to `6_final_model/outputs/{cohort}/{ab}/bin_models/{bin}/`
- Writes `n_event_bin_thresholds.json` (required by notebook 4)
- Pushes artifacts to `s3://pgxdatalake/gold/final_model/`

**Verify:** `6_final_model/outputs/non_opioid_ed/65_74/bin_models/low/models/` exists

---

### Step 2 — Notebook 4: Dashboard Visuals + Manuscript Checkpoints
**File:** `4_dashboard_visuals.ipynb`  
**Scripts:** `9_dashboard_visuals/{bupar,dtw,fpgrowth}/`, `8_ffa_analysis/`

- [ ] Run notebook 4 (all cohorts, all age bands)
- Builds BupaR, DTW, FP-Growth, SHAP+FFA causal visuals
- Uploads causal dashboard JSON to S3
- **NEW:** Writes manuscript checkpoints to `s3://pgxdatalake/gold/manuscript_checkpoints/`:

| Checkpoint | S3 path | Covers |
|:-----------|:--------|:-------|
| FP-Growth top rules | `fpgrowth/{cohort}/{ab}/{bin}/fpgrowth_manuscript_summary.json` | 2 cohorts × 8 bands × 4 bins |
| DTW trajectory stats | `dtw/{cohort}/{ab}/{bin}/dtw_manuscript_summary.json` | 2 cohorts × 8 bands × 4 bins |
| SHAP top-10 features | `shap/{cohort}/{ab}/{bin}/shap_manuscript_summary.json` | 2 cohorts × 8 bands × 4 bins |
| FFA causal features | `ffa/{cohort}/{ab}/{bin}/ffa_manuscript_summary.json` | 2 cohorts × 8 bands × 4 bins |
| PGx coverage % | `pgx/{cohort}/{ab}/pgx_manuscript_summary.json` | 2 cohorts × 8 bands |

**Prerequisite check:** notebook 4 will raise `RuntimeError` if `n_event_bin_thresholds.json` is missing → must run notebook 3 first.

---

### Step 3 — Notebook 5: Build & Deploy
**File:** `5_build_and_deploy.ipynb`  
**Script:** `10_risk_dashboard/data_preparation/prepare_models.py`

- [ ] Run notebook 5
- Runs `prepare_models.py` → generates clean `feature_schema.json` (no `n_events`, no `n_event_bin`)
- Syncs all visualizations (BupaR/DTW/FP-Growth/causal/cohort_pgx) to S3
- Builds and deploys Lambda container with new models

**Verify:** Lambda `POST /risk` returns a prediction for a test drug list

---

## Post-Run: Local Manuscript Extraction

Run **after** EC2 notebooks complete. All scripts read from S3.

```powershell
cd c:\Projects\pgx-analysis\manuscript\scripts

# 1. Model calibration metrics (Brier score, ICI)
python compute_brier_ici.py
# → updates manuscript/brier_ici_results.json

# 2. FFA causal importance data (top drugs, IE scores, rule counts)
python extract_ffa_manuscript.py
# → updates manuscript/ffa_manuscript_data.json

# 3. Visual pipeline data (FP-Growth, DTW, SHAP top-10, PGx coverage)
python extract_visual_manuscript.py
# → writes manuscript/visual_manuscript_data.json
# → writes manuscript/pgx_coverage.json
# → writes manuscript/shap_top_features.json
```

Then rebuild manuscript PDFs:
```powershell
cd c:\Projects\pgx-analysis\manuscript
.\build.ps1
```

---

## Manuscript Placeholder Status

| Placeholder | Chapter(s) | Source | Status |
|:-----------|:-----------|:-------|:------:|
| Total opioid ED N (2,106,127 cases) | CH_2, CH_3, CH_6 | `4_model_data` checkpoints | ✅ Filled |
| Total polypharmacy ED N (55,248 cases) | CH_2, CH_4, CH_6 | `4_model_data` checkpoints | ✅ Filled |
| Opioid ED AUROC 0.9572–0.9924 | CH_3 | `model_performance_metrics.json` | ✅ Filled — will update after retrain |
| Polypharmacy ED AUROC 0.9729–0.9989 | CH_4 | `model_performance_metrics.json` | ✅ Filled — will update after retrain |
| Opioid ED AUC-PR 0.8401–0.9786 | CH_3 | `model_performance_metrics.json` | ✅ Filled — will update after retrain |
| Polypharmacy ED AUC-PR 0.9081–0.9966 | CH_4 | `model_performance_metrics.json` | ✅ Filled — will update after retrain |
| Brier score / ICI per cohort | CH_3, CH_4 | `brier_ici_results.json` | ⏳ Rerun `compute_brier_ici.py` post-retrain |
| FFA top drugs / IE scores / DDI pairs | CH_4, CH_5 | `ffa_manuscript_data.json` | ⏳ Rerun `extract_ffa_manuscript.py` post-retrain |
| SHAP top-10 features per cohort/bin | CH_3, CH_4, CH_5 | `shap_top_features.json` | ⏳ Extract after notebook 4 |
| FP-Growth top drug-pair associations | CH_3, CH_4 | `visual_manuscript_data.json` | ⏳ Extract after notebook 4 |
| DTW cluster N / trajectory counts | CH_3, CH_4 | `visual_manuscript_data.json` | ⏳ Extract after notebook 4 |
| PGx feature coverage % | CH_5 | `pgx_coverage.json` | ⏳ Extract after notebook 4 |
| Lambda inference latency `[XXX ms]` | CH_5, CH_6 | **CloudWatch Logs** | ⏳ Manual — see below |
| `opioid_ed/0-12` model metrics | CH_3 | `gold/final_model` | ❌ Excluded (N=893, below threshold) |

---

## Manual: Lambda Latency (CloudWatch)

Not producible from notebooks. Pull from AWS CloudWatch after deployment:

```powershell
aws logs filter-log-events `
  --log-group-name "/aws/lambda/pgx-risk-dashboard" `
  --filter-pattern "REPORT" `
  --start-time (Get-Date).AddDays(-7).ToUniversalTime() `
  | ConvertFrom-Json | Select-Object -ExpandProperty events `
  | ForEach-Object { $_.message } `
  | Select-String "Duration"
```

Report format: `REPORT RequestId: ... Duration: XXX.XX ms  Billed Duration: XXX ms  Memory Size: XXXX MB  Max Memory Used: XXX MB`

---

## S3 Output Locations Reference

| Data | S3 Path |
|:-----|:--------|
| Trained models (per-bin) | `s3://pgxdatalake/gold/final_model/{cohort}/{ab}/bin_models/{bin}/` |
| Deployed Lambda models | `s3://pgxdatalake/gold/dashboard/models/{cohort}/{ab}/bin_models/{bin}/` |
| Model performance metrics | `s3://pgxdatalake/gold/dashboard/metadata/model_performance_metrics.json` |
| FFA causal factors | `s3://pgxdatalake/gold/ffa_analysis/{cohort}/{ab}/bin_models/{bin}/{model}/` |
| Causal dashboard JSON | `s3://jerome-dixon.io/vcu/pgx-risk-calculator/visualizations/causal/{cohort}/{ab}/{bin}/causal_data.json` |
| **Manuscript checkpoints** | `s3://pgxdatalake/gold/manuscript_checkpoints/{type}/{cohort}/{ab}/{bin}/` |
