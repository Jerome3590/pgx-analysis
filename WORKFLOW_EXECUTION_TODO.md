# Workflow Execution TODO List

This document provides a step-by-step checklist for executing the complete workflow after the time-windowed HCG event logic migration.

## Prerequisites

- EC2 instance with access to S3 (`pgxdatalake` and `pgx-repository` buckets)
- **Full reset:** To run the workflow from scratch (no checkpoints, no S3/EC2 artifacts), run `./utility_scripts/cleanup_cohort_data.sh` first. See [docs/CLEAR_WORKFLOW_FOR_FULL_RUN.md](docs/CLEAR_WORKFLOW_FOR_FULL_RUN.md).
- Python 3.11+ with jupyter-env activated
- R with required packages (bupaR, edeaR, etc.)
- AWS CLI configured
- Project cloned: `~/pgx-analysis`

**Optional:** [0_config_and_pipeline.ipynb](0_config_and_pipeline.ipynb) — env checks (Python/R/Docker), run cleanup script, and pipeline instructions. Not required for the workflow below.

### Key notebooks (run from repo root) — efficient flow: 1 → 2 → 3 → 4 → 5

| # | Notebook | Purpose |
|---|----------|---------|
| 1 | [1_cohort_workflow.ipynb](1_cohort_workflow.ipynb) | Step 2: Create cohorts (OPIOID_ED, POLYPHARMACY) |
| 2 | [2_feature_importance.ipynb](2_feature_importance.ipynb) | Steps 3a–3c: Feature importance, BupaR leakage, refine features |
| 3 | [3_model_train_shap_ffa.ipynb](3_model_train_shap_ffa.ipynb) | Model data → PGx → final model → SHAP/FFA → combine (no deploy) |
| 4 | [4_dashboard_visuals.ipynb](4_dashboard_visuals.ipynb) | Dashboard visuals: BupaR, DTW, FP-Growth (SHAP/FFA-driven) |
| 5 | [5_build_and_deploy.ipynb](5_build_and_deploy.ipynb) | Build and deploy: prepare Lambda → Docker → ECR → Lambda → S3 frontend |

Alternative to notebook 4: run `pgx_dashboard_visuals.py` (same steps, VS Code `# %%` or CLI).

---

## Step 1: Clear Cohort Data (utility – full reset only) ✅

**Purpose**: Utility script to remove old cohort data (e.g. before the time-windowed logic migration). Only run when you need a full reset; not part of the normal pipeline.

### 1.1 Run Cleanup Script

```bash
cd ~/pgx-analysis

# Make script executable (if not already)
chmod +x utility_scripts/cleanup_cohort_data.sh

# Run cleanup (will scan and log all existing data, then delete)
./utility_scripts/cleanup_cohort_data.sh

# Review the log: ./utility_scripts/check_cleanup_log.sh  (or see repo root for cleanup_cohort_data_YYYYMMDD_HHMMSS.log)
```

**What gets cleared:**
- Step 2: Cohort parquet files (S3 and local)
- Step 3b: Feature importance outputs
- Step 4a: Model data
- Step 6: Trained models
- Checkpoints (optional)

**Options:**
- `--skip-checkpoints` - Keep checkpoints
- `--skip-s3` - Only delete local files
- `--skip-local` - Only delete S3 files

---

## Step 2: Create Cohorts ✅

**Purpose**: Create cohorts with new time-windowed HCG event logic.

### 2.1 Create Cohorts for Each Age Band

**For Polypharmacy Cohort (non_opioid_ed / ED_NON_OPIOID):**

```bash
# Age bands: 65-74, 75-84, 85-94
# Years: 2016, 2017, 2018, 2019

# Example for 65-74 (polypharmacy uses fixed 21-day window; --time-window-days is deprecated and ignored):
python 2_create_cohort/0_create_cohort.py \
    --age-band 65-74 \
    --event-year 2016 \
    --cohort ed_non_opioid

# Repeat for each year (2016, 2017, 2018, 2019)
# Repeat for each age band (65-74, 75-84, 85-94)
```

**For Opioid ED Cohort (opioid_ed / OPIOID_ED):**

```bash
# Age bands: 13-24, 25-44, 45-54, 55-64
# Years: 2016, 2017, 2018, 2019
# Note: opioid_ed uses F11.20 target only (no time window). Polypharmacy (ed_non_opioid) uses fixed 21-day window.

# Example for 13-24:
python 2_create_cohort/0_create_cohort.py \
    --age-band 13-24 \
    --event-year 2016 \
    --cohort opioid_ed

# Repeat for each year and age band
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
# Check model_events.parquet created
ls -lh /mnt/nvme/4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet
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

**Trains CatBoost, XGBoost, and XGBoost RF models:**

```bash
# For each cohort and age band:
python 6_final_model/run_final_model.py \
    --cohort non_opioid_ed \
    --age-band 65-74
```

**Verify:**
```bash
# Check models trained
ls -lh 6_final_model/models/{cohort}/{age_band}/
ls -lh 6_final_model/outputs/{cohort}/{age_band}/*model_metrics_summary.csv
ls -lh 6_final_model/outputs/{cohort}/{age_band}/*mc_cv_results.csv
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

**Purpose**: Generate all dashboard visualization artifacts (BupaR, DTW, FP-Growth). Distinct step before building and deploying the dashboard.

**Run from repo root.** Produces BupaR, DTW, and FP-Growth artifacts; all are **SHAP/FFA-driven** (filter original data to model-important features from Step 7 / Step 8). Causal tab uses the same importance data. The notebook/script also includes Deploy Lambda and Deploy frontend cells (idempotent; run to refresh live backend and S3 frontend).

**Option A – Notebook:**

```bash
cd ~/pgx-analysis
jupyter notebook 4_dashboard_visuals.ipynb
# Run cells: Setup → Symlinks → Config → BupaR → DTW → FP-Growth → Deploy Lambda → Deploy frontend
```

**Option B – Python script (VS Code Jupyter `# %%` or CLI):**

```bash
cd ~/pgx-analysis
python pgx_dashboard_visuals.py
# Set SKIP_DEPLOY_LAMBDA=1 / SKIP_DEPLOY_FRONTEND=1 to skip deploy when needed
```

**Symlinks:** If R or scripts expect `10c_bupaR_dashboard_visual`, `10b_fpgrowth_dashboard_visual`, `10d_dtw_dashboard_visual` at repo root, the notebook/script create them (idempotent). See `pgx_dashboard_visuals.py` setup cell.

**Prerequisites:** Step 4 (model data), Step 7 (SHAP), Step 8 (FFA) for SHAP/FFA-driven filtering; R and bupaR for BupaR.

**Verify:**
- BupaR: `9_risk_dashboard/visualizations/bupar` outputs (or `10c_bupaR_dashboard_visual/outputs`) and S3 `gold/bupar/`
- DTW: `gold/feature_engineering/6_dtw/{cohort}/{age_band}/` and feature_importance plots
- FP-Growth: `9_risk_dashboard/visualizations/fpgrowth` outputs (or `10b_fpgrowth_dashboard_visual/outputs`) and S3 `gold/fpgrowth/`

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
cd ~/pgx-analysis/9_risk_dashboard

# Build frontend
cd frontend
npm install
npm run build

# Deploy backend (Lambda function)
# Follow 9_risk_dashboard/deployment/README.md

# Deploy to S3
aws s3 sync dist/ s3://{your-dashboard-bucket}/

# Configure API Gateway (if not already)
# utility_scripts/create_api_gateway_pgx_risk_calculator.sh or .ps1
# See 9_risk_dashboard/backend/README.md
```

**Verify:**
- Frontend accessible via S3/CloudFront
- Backend API responding via API Gateway
- All visualization tabs working (Causal Analysis, BupaR, DTW, FP-Growth)

**Do you need to update the Lambda image for dashboard visuals?**
- **BupaR, DTW, FP-Growth:** No Lambda code change. Lambda only returns S3 paths to artifacts. Run **notebook 6** (or `pgx_dashboard_visuals.py`); upload outputs to S3. Build and deploy run in **notebook 7** only.
- **Causal tab:** The Lambda was updated to default to **top 500 SHAP/FFA important features** when the user does not select drugs/ICDs/CPTs. To get that behavior in production, **redeploy the Lambda** (rebuild the Docker image and update the Lambda function with the current `9_risk_dashboard/backend/lambda_function.py`). See `9_risk_dashboard/deployment/README.md` and `utility_scripts/create_api_gateway_pgx_risk_calculator.sh`.

---

## Quick Reference: All Cohorts and Age Bands

### Opioid ED Cohorts (F11.20 target)
- `opioid_ed`: 13-24, 25-44, 45-54, 55-64

### Polypharmacy Cohorts (Time-windowed HCG target)
- `non_opioid_ed`: 65-74, 75-84, 85-94

---

## Automation Option

**Run via the workflow notebooks:**
- Workflow: `1_cohort_workflow.ipynb` → `2_feature_importance.ipynb` → `3_model_train_shap_ffa.ipynb` → `4_dashboard_visuals.ipynb` → `5_build_and_deploy.ipynb`
- Dashboard visuals (alternative to notebook 4): `pgx_dashboard_visuals.py` (from repo root)

Legacy shell scripts are in `archived/utility_scripts/` (if present):

```bash
# For a single cohort/age band (legacy):
bash archived/utility_scripts/run_cohort_workflow.sh non_opioid_ed 65-74
# ... etc.
```

**Note**: After automated Steps 3–8, you still need to:
1. Run Step 1 (cleanup) manually
2. Run Step 2 (cohort creation) manually (polypharmacy uses fixed 21-day window)
3. Run **3** → **4** → **5** (model train/SHAP/FFA → dashboard visuals → build and deploy)

---

## Final documented workflow (notebooks)

**Most efficient flow:** 1 → 2 → **3** → **4** → **5** (one notebook per stage; build and deploy once in 5).

| Order | Notebook | Covers |
|-------|----------|--------|
| 1 | **`1_cohort_workflow.ipynb`** | Step 2: Cohort creation (OPIOID_ED and POLYPHARMACY). |
| 2 | **`2_feature_importance.ipynb`** | Steps 3a–3c: Feature importance, BupaR leakage, refine features. |
| 3 | **`3_model_train_shap_ffa.ipynb`** | Model data → PGx → final model → SHAP/FFA → combine. No deploy. |
| 4 | **`4_dashboard_visuals.ipynb`** or **`pgx_dashboard_visuals.py`** | Dashboard visuals: BupaR, DTW, FP-Growth (SHAP/FFA-driven). |
| 5 | **`5_build_and_deploy.ipynb`** | Prepare models/Lambda → Docker → ECR → update Lambda → S3 frontend. Run once. |

**Prerequisites before notebook 1:** Step 1 (cleanup) and Step 1a/1b data as needed.

---

## Checklist

- [ ] **Step 1 (utility)**: Clear cohort data for full reset only — `utility_scripts/cleanup_cohort_data.sh`
- [ ] **Step 2**: Create cohorts with time-windowed logic for polypharmacy cohort (`1_cohort_workflow.ipynb`)
- [ ] **Step 2**: Create cohorts for opioid ED cohort (`1_cohort_workflow.ipynb`)
- [ ] **Step 3**: Run Feature Importance EDA (BupaR post-target analysis) for all cohorts (`2_feature_importance.ipynb` or `3b_feature_importance_eda/step3b_interactive_analysis_cohort*.ipynb`)
- [ ] **Notebook 3**: Model train + SHAP/FFA — `3_model_train_shap_ffa.ipynb` (model data, PGx, final model, SHAP, FFA, combine)
- [ ] **Notebook 4**: Dashboard visuals — `4_dashboard_visuals.ipynb` or `pgx_dashboard_visuals.py` (BupaR, DTW, FP-Growth)
- [ ] **Notebook 5**: Build and deploy — `5_build_and_deploy.ipynb` (Lambda, ECR, S3 frontend) — run once

---

## Notes

- **Time Windows**: Polypharmacy (ed_non_opioid) uses a **fixed 21-day window** for adverse drug event identification (~90.5% capture). See `2_create_cohort/README.md`. Opioid_ed has no time window (F11.20 target only).
- **SHAP/FFA-driven visuals**: BupaR, DTW, FP-Growth, and Causal dashboard visuals use model-important features (Step 7 SHAP, Step 8 FFA). Run Step 7 and 8 before generating dashboard artifacts for best results.
- **Idempotent**: All scripts are idempotent and will skip completed steps
- **Checkpoints**: Pipeline uses S3 checkpoints to track progress
- **Logs**: Check log files in each step's output directory for detailed execution logs
