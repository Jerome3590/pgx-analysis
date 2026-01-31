# Workflow Execution TODO List

This document provides a step-by-step checklist for executing the complete workflow after the time-windowed HCG event logic migration.

## Prerequisites

- EC2 instance with access to S3 (`pgxdatalake` and `pgx-repository` buckets)
- Python 3.11+ with jupyter-env activated
- R with required packages (bupaR, edeaR, etc.)
- AWS CLI configured
- Project cloned: `~/pgx-analysis`

---

## Step 1: Clear Cohort Data ✅

**Purpose**: Remove old cohort data that doesn't have the new time-windowed logic and multiclass target columns.

### 1.1 Run Cleanup Script

```bash
cd ~/pgx-analysis

# Make script executable (if not already)
chmod +x 2_create_cohort/cleanup_cohort_data.sh

# Run cleanup (will scan and log all existing data, then delete)
./2_create_cohort/cleanup_cohort_data.sh

# Review the log file created:
# ~/pgx-analysis/cleanup_cohort_data_YYYYMMDD_HHMMSS.log
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

**Purpose**: Create cohorts with new time-windowed HCG event logic and multiclass target columns.

### 2.1 Create Cohorts for Each Age Band

**For Polypharmacy Cohort (non_opioid_ed / ED_NON_OPIOID):**

```bash
# Age bands: 65-74, 75-84, 85-94
# Years: 2016, 2017, 2018, 2019

# Example for 65-74:
python 2_create_cohort/0_create_cohort.py \
    --age-band 65-74 \
    --event-year 2016 \
    --cohort ed_non_opioid \
    --time-window-days 14

# Repeat for each year (2016, 2017, 2018, 2019)
# Repeat for each age band (65-74, 75-84, 85-94)
```

**For Opioid ED Cohort (opioid_ed / OPIOID_ED):**

```bash
# Age bands: 13-24, 25-44, 45-54, 55-64
# Years: 2016, 2017, 2018, 2019
# Note: time-window-days doesn't apply to opioid_ed (only uses F11.20 target)

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

# Verify multiclass columns exist (for polypharmacy cohort)
# Check that parquet files have: is_target_case_7d, is_target_case_14d, is_target_case_21d, is_target_case_30d, is_target_case_45d
```

**Expected Output:**
- `s3://pgxdatalake/gold/cohorts/cohort_name={cohort}/age_band={age_band}/event_year={year}/cohort.parquet`
- Each polypharmacy cohort should have multiclass target columns

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

## Step 5: Build Risk Dashboard ✅

**Purpose**: Deploy production dashboard with frontend, backend API, and visualizations.

### 5.1 Prepare Dashboard Data

**Ensure all required data is available:**
- Model artifacts from Step 6
- SHAP values from Step 7
- FFA results from Step 8
- BupaR visualizations from Step 3b
- FP-Growth patterns (if available)
- DTW trajectories (if available)

### 5.2 Build and Deploy Dashboard

```bash
cd 9_risk_dashboard

# Build frontend
cd frontend
npm install
npm run build

# Deploy backend (Lambda function)
# Follow instructions in 9_risk_dashboard/README.md

# Deploy to S3
aws s3 sync dist/ s3://{your-dashboard-bucket}/

# Configure API Gateway
# Follow instructions in 9_risk_dashboard/README.md
```

**Verify:**
- Frontend accessible via S3/CloudFront
- Backend API responding via API Gateway
- All visualization tabs working (Causal Analysis, DTW, FP-Growth, BupaR)

---

## Quick Reference: All Cohorts and Age Bands

### Opioid ED Cohorts (F11.20 target)
- `opioid_ed`: 13-24, 25-44, 45-54, 55-64

### Polypharmacy Cohorts (Time-windowed HCG target)
- `non_opioid_ed`: 65-74, 75-84, 85-94

**Note**: Polypharmacy cohorts have multiclass target columns (7d, 14d, 21d, 30d, 45d) for analysis.

---

## Automation Option

**Run via the three workflow notebooks** (`1_cohort_workflow.ipynb`, `2_feature_importance.ipynb`, `3_pgx_calculator_workflow.ipynb`). Legacy shell scripts are in `archived/utility_scripts/`:

```bash
# For a single cohort/age band (legacy):
bash archived/utility_scripts/run_cohort_workflow.sh non_opioid_ed 65-74

# For all polypharmacy cohorts (legacy):
bash archived/utility_scripts/run_non_opioid_ed_workflow.sh

# For all opioid ED cohorts (legacy):
bash archived/utility_scripts/run_opioid_ed_workflow.sh

# For all cohorts (legacy):
bash archived/utility_scripts/run_all_cohorts_workflow.sh
```

**Note**: The workflow script will automatically run Steps 3-8 in sequence. You still need to:
1. Run Step 1 (cleanup) manually
2. Run Step 2 (cohort creation) manually with `--time-window-days` parameter
3. Run Step 9 (dashboard) manually

---

## Checklist

- [ ] **Step 1**: Clear cohort data using cleanup script
- [ ] **Step 2**: Create cohorts with time-windowed logic for polypharmacy cohort
- [ ] **Step 2**: Create cohorts for opioid ED cohort
- [ ] **Step 3**: Run Feature Importance EDA (BupaR post-target analysis) for all cohorts
- [ ] **Step 4a**: Create model data for all cohorts
- [ ] **Step 4b**: Filter protocols for all cohorts
- [ ] **Step 5**: Add PGx features for all cohorts
- [ ] **Step 6**: Train models for all cohorts
- [ ] **Step 7**: Run SHAP analysis for all cohorts
- [ ] **Step 8**: Run FFA analysis for all cohorts
- [ ] **Step 9**: Build and deploy Risk Dashboard

---

## Notes

- **Time Windows**: Polypharmacy cohorts support 7, 14, 21, 30, 45 day windows (default: 14)
- **Multiclass Analysis**: Polypharmacy cohorts have `is_target_case_7d`, `is_target_case_14d`, etc. for multiclass analysis
- **Idempotent**: All scripts are idempotent and will skip completed steps
- **Checkpoints**: Pipeline uses S3 checkpoints to track progress
- **Logs**: Check log files in each step's output directory for detailed execution logs
