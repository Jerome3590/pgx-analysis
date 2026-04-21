# Workflow Execution TODO List

This document provides a step-by-step checklist for executing the complete workflow and tracks status for a full end-to-end run.

---

## Workflow run status (full end-to-end)

**Run goal:** Full pipeline with cohort/age-band updates: both cohorts use full age bands (0-12 through 85-114), single band 85-114, dashboard cohort-from-tab (Opioid ED / Polypharmacy).

**Last reset:** 2026-02-14

| Phase | Notebook | Status | Notes |
|-------|----------|--------|--------|
| 0 | **0_config_and_pipeline.ipynb** | ⬜ Not started | **Reset:** env checks, run cleanup script. **Default:** feature importance preserved; Phase 2 only adds missing. Use `--clear-feature-importance` for full recompute. |
| 1 | **1_cohort_workflow.ipynb** | ⬜ Not started | Steps 1–2: Cohorts (all bands, 85-114) |
| 2 | **2_feature_importance.ipynb** | ⬜ Not started | Steps 3a–3c: additive — skips existing (cohort, age_band); only adds missing. Keep `FORCE_FEATURE_IMPORTANCE = False`. |
| 3 | **3_model_train_shap_ffa.ipynb** | ⬜ Not started | Steps 4–8: Model data → PGx → final model → SHAP/FFA |
| 4 | **4_dashboard_visuals.ipynb** | ⬜ Not started | Step 9: BupaR, DTW, FP-Growth |
| 5 | **5_build_and_deploy.ipynb** | ⬜ Not started | Lambda + S3 frontend deploy |

**How to update:** Change ⬜ to ✅ when that phase's notebook’s run is complete.

**Default behavior:** Feature importance is **preserved**. Run Phase 0 and execute the cleanup script (no extra flags). Step 3a/3b outputs and `gold/feature_importance` are **not** deleted; Phase 2 **skips** (cohort, age_band) that already have checkpoints/outputs and **only runs for missing combinations**—e.g. new age bands from the full-grid update. Then Phase 1 → 2 → 3 → 4 → 5.

**Optional — full recompute of feature importance:** Run the cleanup script with **`--clear-feature-importance`** to also delete Step 3a/3b and `gold/feature_importance`; Phase 2 will then recompute all (cohort, age_band).

---

## Prerequisites

- EC2 instance with access to S3 (`pgxdatalake` and `pgx-repository` buckets) via **IAM instance profile** (do not set `AWS_PROFILE` on EC2).
- **Local/dev:** Use profile `mushin` for AWS CLI (e.g. `.vscode/settings.json` sets `AWS_PROFILE=mushin` for integrated terminals).
- **Full reset:** To run the workflow from scratch (no checkpoints, no S3/EC2 artifacts), run `./utility_scripts/cleanup_cohort_data.sh` first. See [docs/CrossStep_Workflow/README_checkpoints_and_workflow_resets.md](docs/CrossStep_Workflow/README_checkpoints_and_workflow_resets.md).
- Python 3.11+ with jupyter-env activated
- R with required packages (bupaR, edeaR, etc.)
- AWS CLI configured
- Project cloned: `~/pgx-analysis`

**Default:** Run [0_config_and_pipeline.ipynb](0_config_and_pipeline.ipynb) (Phase 0) first to clear checkpoints and S3/local artifacts; **feature importance is preserved** (Phase 2 only adds missing). Then run phases 1 → 5. Use cleanup script with `--clear-feature-importance` only when you need a full recompute of feature importance.

### Key phases (run from repo root) — full fresh run: Phase 0 → 1 → 2 → 3 → 4 → 5

| Phase | Notebook | Purpose |
|-------|----------|---------|
| 0 | [0_config_and_pipeline.ipynb](0_config_and_pipeline.ipynb) | **Reset:** env checks, run cleanup script. **Default:** FI preserved; Phase 2 only adds missing. Use `--clear-feature-importance` for full FI recompute. |
| 1 | [1_cohort_workflow.ipynb](1_cohort_workflow.ipynb) | Step 2: Create cohorts (OPIOID_ED, POLYPHARMACY) |
| 2 | [2_feature_importance.ipynb](2_feature_importance.ipynb) | Steps 3a–3c: additive — only add missing (cohort, age_band); keep `FORCE_FEATURE_IMPORTANCE = False`. |
| 3 | [3_model_train_shap_ffa.ipynb](3_model_train_shap_ffa.ipynb) | Model data → PGx → final model → SHAP/FFA → combine (no deploy) |
| 4 | [4_dashboard_visuals.ipynb](4_dashboard_visuals.ipynb) | Dashboard visuals: BupaR, DTW, FP-Growth (SHAP/FFA-driven) |
| 5 | [5_build_and_deploy.ipynb](5_build_and_deploy.ipynb) | Build and deploy: prepare Lambda → Docker → ECR → Lambda → S3 frontend |

Alternative to Phase 4: run `pgx_dashboard_visuals.py` (same steps, VS Code `# %%` or CLI).

---

## Step 1: Clear Cohort Data (utility – full reset only) ✅

**Purpose**: Utility script to remove old cohort data (e.g. before the time-windowed logic migration). Only run when you need a full reset; not part of the normal pipeline.

### 1.1 Run Cleanup Script

```bash
cd ~/pgx-analysis

# Make script executable (if not already)
chmod +x utility_scripts/cleanup_cohort_data.sh

# Run cleanup (default: preserves feature importance; Phase 2 only adds missing)
./utility_scripts/cleanup_cohort_data.sh

# Optional: also clear feature importance for full recompute
# ./utility_scripts/cleanup_cohort_data.sh --clear-feature-importance

# Review the log: ./utility_scripts/check_cleanup_log.sh  (or see repo root for cleanup_cohort_data_YYYYMMDD_HHMMSS.log)
```

**What gets cleared (default):**
- Step 2: Cohort parquet files (S3 and local)
- Step 4a: Model data
- Step 6: Trained models
- Checkpoints (optional)
- **Step 3b/3a feature importance:** **NOT** cleared by default (preserved; Phase 2 only adds missing).

**Options:**
- `--skip-checkpoints` - Keep checkpoints
- `--skip-s3` - Only delete local files
- `--skip-local` - Only delete S3 files
- **`--clear-feature-importance`** - Also delete Step 3a/3b outputs and `gold/feature_importance` (full recompute in Phase 2). Omit this flag for default (preserve FI, add missing only).

---

## Step 2: Create Cohorts ✅

**Purpose**: Create cohorts with first ED visit (HCG Setting) within 21 days of a prescription drug event (polypharmacy target).

### 2.1 Create Cohorts for Each Age Band

**For Polypharmacy Cohort (non_opioid_ed / ED_NON_OPIOID):**

```bash
# Age bands: 65-74, 75-84, 85-114 (full set for non_opioid_ed)
# Years: 2016, 2017, 2018, 2019

# Example for 65-74 (polypharmacy uses fixed 21-day window; --time-window-days is deprecated and ignored):
python 2_create_cohort/0_create_cohort.py \
    --age-band 65-74 \
    --event-year 2016 \
    --cohort ed_non_opioid

# Repeat for each year (2016, 2017, 2018, 2019)
# Repeat for each age band (see py_helpers.constants.AGE_BANDS; includes 85-114)
```

**For Opioid ED Cohort (opioid_ed / OPIOID_ED):**

```bash
# Age bands: full set (py_helpers.constants.AGE_BANDS): 0-12, 13-24, 25-44, 45-54, 55-64, 65-74, 75-84, 85-114
# Years: 2016, 2017, 2018, 2019
# Note: opioid_ed uses F11.20 target only. Polypharmacy (ed_non_opioid) uses first ED (HCG) within 21 days of drug event.

# Example for 13-24:
python 2_create_cohort/0_create_cohort.py \
    --age-band 13-24 \
    --event-year 2016 \
    --cohort opioid_ed

# Repeat for each year and each age band (full set)
```

### 2.2 Verify Cohort Creation

```bash
# Check S3 for cohort files
aws s3 ls s3://pgxdatalake/gold/cohorts/cohort_name=ed_non_opioid/ --recursive
aws s3 ls s3://pgxdatalake/gold/cohorts/cohort_name=opioid_ed/ --recursive

```

**Expected Output:**
- `s3://pgxdatalake/gold/cohorts/cohort_name={cohort}/age_band={age_band}/event_year={year}/cohort.parquet`

---

## Step 3: Filter Features by BupaR (Feature Importance EDA) ✅

**Purpose**: Use BupaR post-target analysis to identify and filter post-target leakage features, creating refined `cohort_feature_importance.csv` files.

### 3.1 Run Feature Importance EDA

**For Each Cohort and Age Band:**

```bash
# Polypharmacy cohort example (non_opioid_ed):
python 3b_feature_importance_eda/run_feature_importance_eda.py \
    --cohort non_opioid_ed \
    --age-band 65-74

# Opioid ED cohort example:
python 3b_feature_importance_eda/run_feature_importance_eda.py \
    --cohort opioid_ed \
    --age-band 13-24
```

**Or use interactive notebooks:**

```bash
# For polypharmacy cohort (cohorts 5, 6, 7):
jupyter notebook 3b_feature_importance_eda/step3b_interactive_analysis_cohort5.ipynb
jupyter notebook 3b_feature_importance_eda/step3b_interactive_analysis_cohort6.ipynb
jupyter notebook 3b_feature_importance_eda/step3b_interactive_analysis_cohort7.ipynb

# For opioid ED cohort (cohorts 1, 2, 3, 4):
jupyter notebook 3b_feature_importance_eda/step3b_interactive_analysis_cohort1.ipynb
jupyter notebook 3b_feature_importance_eda/step3b_interactive_analysis_cohort2.ipynb
jupyter notebook 3b_feature_importance_eda/step3b_interactive_analysis_cohort3.ipynb
jupyter notebook 3b_feature_importance_eda/step3b_interactive_analysis_cohort4.ipynb
```

### 3.2 Verify Feature Importance EDA Outputs

```bash
# Check for refined feature importance files
ls -lh 3b_feature_importance_eda/outputs/{cohort}/{age_band}/*cohort_feature_importance.csv

# Check S3
aws s3 ls s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/ --recursive
```

**Expected Output:**
- `{cohort}_{age_band}_cohort_feature_importance.csv` - Refined features (post-target leakage filtered)
- `{cohort}_{age_band}_bupar_post_target_analysis.csv` - Post-target analysis results
- BupaR visualizations in `outputs/{cohort}/{age_band}/plots/`

---

## Step 4: Run Model Analysis ✅

**Purpose**: Create model data, filter protocols, add PGx features, and train models.

### 4.1 Step 4a: Create Model Data

**Uses refined features from Step 3b (REQUIRED - no fallback):**

```bash
# For each cohort and age band:
python 4_model_data/create_model_data.py \
    --cohort non_opioid_ed \
    --age-band 65-74
```

**Verify:**
```bash
# Check model_events.parquet created (path: PGX_DATA_ROOT or /mnt/nvme on EC2)
ls -lh /mnt/nvme/4_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet
```

### 4.2 Step 4b: Event Filtering (Protocol Filtering)

**Removes administrative codes and post-event leakage:**

```bash
# For each cohort and age band:
python 1b_apcd_event_filter/filter_protocol_events.py \
    --cohort non_opioid_ed \
    --age-band 65-74
```

**Verify:**
```bash
# Check model_events_no_protocols.parquet created
ls -lh /mnt/nvme/4_model_data/cohort_name={cohort}/age_band={age_band}/model_events_no_protocols.parquet
```

### 4.3 Step 5: PGx Feature Engineering

**Adds PGx drug counts and drug counts:**

```bash
# For each cohort and age band:
python 5_pgx_analysis/run_analysis.py \
    --cohort-name non_opioid_ed \
    --age-band 65-74
```

**Verify:**
```bash
# Check PGx features added
ls -lh 5_pgx_analysis/outputs/{cohort}/{age_band}/pgx_added_features_*.csv
```

### 4.4 Step 6: Final Model Training

**Trains four candidates: XGBoost, XGBoost RF, CatBoost, and Ensemble (probability average of XGB + CatBoost):**

```bash
# For each cohort and age band:
python 6_final_model/run_final_model.py \
    --cohort non_opioid_ed \
    --age-band 65-74
```

**Verify:**
```bash
# Check per-bin models trained (age_band_fname uses underscore, e.g. 25_44)
# Bins: low / medium / high / extreme
ls -lh 6_final_model/outputs/{cohort}/{age_band_fname}/bin_models/
ls -lh 6_final_model/outputs/{cohort}/{age_band_fname}/bin_models/{bin}/{cohort}_{age_band_fname}_model_metrics_summary.csv
ls -lh 6_final_model/outputs/{cohort}/{age_band_fname}/bin_models/{bin}/{cohort}_{age_band_fname}_mc_cv_results.csv
# S3 path:
# s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/bin_models/{bin}/
```

### 4.5 Step 7: SHAP Analysis

**SHAP values for CatBoost and XGBoost:**

```bash
# For each cohort and age band:
python 7_shap_analysis/run_shap_analysis.py \
    --cohort non_opioid_ed \
    --age-band 65-74
```

**Verify:**
```bash
# Check SHAP outputs
ls -lh 7_shap_analysis/outputs/{cohort}/{age_band}/
```

### 4.6 Step 8: FFA Analysis

**Formal Feature Attribution for XGBoost (uses SHAP from Step 7):**

```bash
# For each cohort and age band:
python 8_ffa_analysis/run_full_ffa_analysis.py \
    --cohort non_opioid_ed \
    --age-band 65-74
```

**Verify:**
```bash
# Check FFA outputs
ls -lh 8_ffa_analysis/outputs/{cohort}/{age_band}/
```

---

## Step 5: Generate Dashboard Visuals ✅

**Purpose**: Generate all dashboard visualization artifacts. Distinct step before building and deploying the dashboard.

**Run from repo root.** All artifacts are **SHAP/FFA-driven** (filter original data to model-important features from Step 7 / Step 8). The notebook cell sequence is:

1. **Setup** — repo paths, S3 config, `S3_VISUALIZATIONS_BUILDS=1` (uploads to `.../builds/`)
2. **BupaR** — process mining sequences and plots, per cohort/age_band; outputs to `10_risk_dashboard/visualizations/bupar/`
3. **DTW** — trajectory features, bins, and plots, per cohort/age_band; outputs to `10_risk_dashboard/visualizations/dtw/`
4. **FP-Growth** — drug itemsets, association rules, network HTML, per cohort/age_band; outputs to `10_risk_dashboard/visualizations/fpgrowth/`
5. **Causal** — FFA + SHAP combined dashboard JSON; outputs to `10_risk_dashboard/visualizations/causal/`
6. **Cohort PGx** — per cohort/age_band (full cohort + per-bin low/medium/high/extreme):
   - Fetch PharmGKB VIP reports (`fetch_vip_reports.py`)
   - Fetch PubMed citations + radar data (`fetch_pubmed_citations.py`)
   - Build Gene–Drug–Phenotype network topology HTML (`build_network_topology.py`)
   - Outputs to `10_risk_dashboard/visualizations/cohort_pgx/networks/`
7. **PGx patient card** — download CPIC Excel, fetch VIP JSON, generate QR codes, build PGx database; outputs to `pgx-patient-card/data/` and `pgx-patient-card/qr_codes/`
8. **Dashboard visual manifest** — writes `10_risk_dashboard/visualizations/dashboard_visual_objects.json` for notebook 5
9. **Time-between-events histogram** — per cohort/age_band JSON for DTW tab
10. **Manuscript checkpoint writer** — validates all outputs for every cohort/age_band/bin; uploads to `s3://pgxdatalake/gold/manuscript_checkpoints/`

**Option A – Notebook:**

```bash
cd ~/pgx-analysis
jupyter notebook 4_dashboard_visuals.ipynb
# Run all cells in order (Setup → BupaR → DTW → FP-Growth → Causal → Cohort PGx → PGx Card → Manifest → Checkpoints)
```

**Option B – Python script (VS Code Jupyter `# %%` or CLI):**

```bash
cd ~/pgx-analysis
python pgx_dashboard_visuals.py
# Set SKIP_DEPLOY_LAMBDA=1 / SKIP_DEPLOY_FRONTEND=1 to skip deploy when needed
```

**Prerequisites:** Step 4 (model data), Step 7 (SHAP), Step 8 (FFA) for SHAP/FFA-driven filtering; R and bupaR for BupaR; AWS credentials for Cohort PGx VIP/PubMed fetches and S3 uploads.

**Verify (manuscript checkpoint output):**

The final cell prints a per-cohort/age_band/bin summary. A passing run looks like:

```
── non_opioid_ed/65-74 ──
  [PGx] coverage=78.54%  ✓
  [CohortPGx] network_html=✓
  [BupaR/low] n_activities=40  top=DRUG:AMOXICILLIN  ✓
  [FPG/low] total_rules=0  ✓
  [DTW/low] trajectories=8203  rapid_onset_n=2053  chronic_escalation_n=2051  ✓
  [CohortPGx/low] network_html=✓
  [SHAP/low] top=pgx_num_drugs  ✓
  [FFA/low] features=39  ✓
  ...
Manuscript checkpoints written: 416  errors: 0
```

**Local output directories:**
- BupaR: `10_risk_dashboard/visualizations/bupar/`
- DTW: `10_risk_dashboard/visualizations/dtw/`
- FP-Growth: `10_risk_dashboard/visualizations/fpgrowth/`
- Cohort PGx: `10_risk_dashboard/visualizations/cohort_pgx/networks/`
- Causal: `10_risk_dashboard/visualizations/causal/`

See [9_dashboard_visuals/README.md](9_dashboard_visuals/README.md) for the full checkpoint field reference.

---

## Step 6: Build Risk Dashboard ✅

**Purpose**: Deploy production dashboard with frontend, backend API, and visualizations.

### 6.1 Prepare Dashboard Data

**Ensure all required data is available:**
- Model artifacts from Step 4.4 (final model)
- SHAP values from Step 7
- FFA results from Step 8
- Dashboard visuals from **Step 5** (BupaR, DTW, FP-Growth) and Causal (served from SHAP/FFA by Lambda)

### 6.2 Build and Deploy Dashboard

```bash
cd ~/pgx-analysis

# 1. Copy per-bin models + Lambda code into deploy directory
python 10_risk_dashboard/deployment/prepare_lambda_dir.py

# 2. Build Docker image and push to ECR (requires Docker)
bash 10_risk_dashboard/deployment/docker_build.sh

# 3. Sync static frontend (HTML/CSS/JS) to S3
python 10_risk_dashboard/deployment/sync_frontend_to_s3.py

# 4. Sync visualization artifacts to S3
python 10_risk_dashboard/deployment/sync_visuals_to_s3.py

# 5. Invalidate CloudFront cache
aws cloudfront create-invalidation \
    --distribution-id E3MZK5HYTJ14P3 \
    --paths "/vcu/pgx-risk-calculator/*"
```

**Verify:**
- Frontend accessible at `https://jerome-dixon.io/vcu/pgx-risk-calculator/index.html`
- Backend API responding via API Gateway (`https://cmv0qislq3.execute-api.us-east-1.amazonaws.com/prod`)
- All visualization tabs working (Causal Analysis, BupaR, DTW, FP-Growth)

**Do you need to update the Lambda image for dashboard visuals?**
- **BupaR, DTW, FP-Growth:** No Lambda code change. Lambda only returns S3 paths to artifacts. Run **Phase 4** (or `pgx_dashboard_visuals.py`); upload outputs to S3. Build and deploy run in **Phase 5** only.
- **Causal tab:** The Lambda was updated to default to **top 500 SHAP/FFA important features** when the user does not select drugs/ICDs/CPTs. To get that behavior in production, **redeploy the Lambda** (rebuild the Docker image and update the Lambda function with the current `10_risk_dashboard/backend/lambda_function.py`). See `10_risk_dashboard/deployment/README.md` and `utility_scripts/create_api_gateway_pgx_risk_calculator.sh`.

---

## Quick Reference: All Cohorts and Age Bands

### Opioid ED Cohorts (F11.20 target)
- `opioid_ed`: full set 0-12, 13-24, 25-44, 45-54, 55-64, 65-74, 75-84, 85-114

### Polypharmacy Cohorts (First ED within 21d of drug event)
- `non_opioid_ed`: full set (same as opioid_ed); last band 85-114

---

## Automation Option

**Run via the workflow notebooks:**
- Workflow: `1_cohort_workflow.ipynb` → `2_feature_importance.ipynb` → `3_model_train_shap_ffa.ipynb` → `4_dashboard_visuals.ipynb` → `5_build_and_deploy.ipynb`
- Dashboard visuals (alternative to Phase 4): `pgx_dashboard_visuals.py` (from repo root)

**Note**: After automated Steps 3–8, you still need to:
1. Run Step 1 (cleanup) manually
2. Run Step 2 (cohort creation) manually (polypharmacy uses fixed 21-day window)
3. Run **3** → **4** → **5** (model train/SHAP/FFA → dashboard visuals → build and deploy)

---

## Final documented workflow (phases 0–5)

**Full fresh run:** **Phase 0** → 1 → 2 → **3** → **4** → **5** (Phase 0 clears checkpoints; then one phase per stage; build and deploy once in Phase 5).

| Phase | Notebook | Covers |
|-------|----------|--------|
| 0 | **`0_config_and_pipeline.ipynb`** | Clear checkpoints + S3/local (cleanup script); env checks. Run first for full E2E. |
| 1 | **`1_cohort_workflow.ipynb`** | Step 2: Cohort creation (OPIOID_ED and POLYPHARMACY). |
| 2 | **`2_feature_importance.ipynb`** | Steps 3a–3c: Feature importance, BupaR leakage, refine features. |
| 3 | **`3_model_train_shap_ffa.ipynb`** | Model data → PGx → final model → SHAP/FFA → combine. No deploy. |
| 4 | **`4_dashboard_visuals.ipynb`** or **`pgx_dashboard_visuals.py`** | Dashboard visuals: BupaR, DTW, FP-Growth (SHAP/FFA-driven). |
| 5 | **`5_build_and_deploy.ipynb`** | Prepare models/Lambda → Docker → ECR → update Lambda → S3 frontend. Run once. |

**Prerequisites before Phase 1:** Run Phase 0 to clear checkpoints (and optionally full cleanup); Step 1a/1b data as needed.

---

## Checklist (reset for full end-to-end run 2026-02-14)

- [ ] **Phase 0**: Clear checkpoints and start fresh — `0_config_and_pipeline.ipynb` (run cleanup script to clear checkpoints + S3/local; run first for full E2E)
- [ ] **Phase 1**: Create cohorts — `1_cohort_workflow.ipynb` (OPIOID_ED + POLYPHARMACY, all age bands including 85-114)
- [ ] **Phase 2**: Feature importance + EDA + 3c — `2_feature_importance.ipynb` (Steps 3a–3c; only adds missing — do not delete existing FI; keep `FORCE_FEATURE_IMPORTANCE = False`)
- [ ] **Phase 3**: Model train + SHAP/FFA — `3_model_train_shap_ffa.ipynb` (model data, PGx, final model, SHAP, FFA, combine)
- [ ] **Phase 4**: Dashboard visuals — `4_dashboard_visuals.ipynb` or `pgx_dashboard_visuals.py` (BupaR, DTW, FP-Growth)
- [ ] **Phase 5**: Build and deploy — `5_build_and_deploy.ipynb` (Lambda, ECR, S3 frontend) — run once

---

## Notes

- **Feature importance (default):** Cleanup script **preserves** Step 3a/3b and `gold/feature_importance` by default. Phase 2 skips (cohort, age_band) that already have checkpoints/outputs and only runs for missing combinations. Keep **`FORCE_FEATURE_IMPORTANCE = False`** in Phase 2. Use **`--clear-feature-importance`** only when you need a full recompute.
- **Time Windows**: Polypharmacy (ed_non_opioid) target = first ED visit (HCG Setting) within 21 days of a prescription drug event. See `2_create_cohort/README.md` and `py_helpers/constants.py` (NON_OPIOID_ED_TARGET_DESCRIPTION). Opioid_ed uses F11.20 target only.
- **SHAP/FFA-driven visuals**: BupaR, DTW, FP-Growth, and Causal dashboard visuals use model-important features (Step 7 SHAP, Step 8 FFA). Run Step 7 and 8 before generating dashboard artifacts for best results.
- **Idempotent**: All scripts are idempotent and will skip completed steps
- **Checkpoints**: Pipeline uses S3 checkpoints to track progress
- **Logs**: Check log files in `logs/{step_name}/` at repo root (e.g. `logs/6_final_model/`, `logs/7_shap_analysis/`, `logs/9_dtw/`); all steps mirror to `s3://pgx-repository/{step_name}_log/` on completion
