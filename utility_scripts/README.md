# Utility Scripts

This directory contains utility scripts for managing, monitoring, and rerunning specific sections of the PGx analysis pipeline. All scripts support idempotent operations and checkpoint-based skipping.

## 📋 Table of Contents

- [Workflow Execution](#workflow-execution)
- [Clearing Outputs](#clearing-outputs)
- [Regenerating Stale Outputs](#regenerating-stale-outputs)
- [Status Checking](#status-checking)
- [Other Utilities](#other-utilities)

---

## Workflow Execution

For complete workflow execution scripts, see [`README_workflow_scripts.md`](README_workflow_scripts.md).

### Quick Reference

```bash
# Single cohort/age band
bash utility_scripts/run_cohort_workflow.sh <cohort> <age_band> [--skip-steps STEP1,STEP2]

# All opioid_ed cohorts
bash utility_scripts/run_opioid_ed_workflow.sh [--skip-steps STEP1,STEP2]

# All non_opioid_ed cohorts
bash utility_scripts/run_non_opioid_ed_workflow.sh [--skip-steps STEP1,STEP2]

# All cohorts
bash utility_scripts/run_all_cohorts_workflow.sh [--skip-steps STEP1,STEP2]
```

**Pipeline Steps:**
- **3**: Feature Importance (Monte Carlo CV)
- **4a**: Model Data Extraction (`model_events.parquet`)
- **4b**: DTW Protocol Filtering (`model_events_no_protocols.parquet`)
- **5a**: BupaR Process Mining
- **5b**: FP-Growth Analysis
- **5c**: PGx Feature Engineering
- **5d**: DTW Trajectory Analysis (optional)
- **6**: Final Model Training
- **7**: SHAP Analysis
- **8**: FFA Analysis

---

## Clearing Outputs

### Clear Model Outputs (Steps 6, 7, 8)

**Script:** `clear_models.sh`

Clears model training and downstream analysis outputs (Steps 6, 7, 8) to force regeneration.

**Usage:**
```bash
# Clear all cohorts/age bands (default)
bash utility_scripts/clear_models.sh [--s3] [--no-s3]

# Clear specific cohort/age band
bash utility_scripts/clear_models.sh --cohort <cohort> --age-band <age_band> [--s3] [--no-s3]
```

**Options:**
- `--cohort <cohort>`: Clear specific cohort (requires `--age-band`)
- `--age-band <age_band>`: Clear specific age band (requires `--cohort`)
- `--s3`: Clear S3 outputs (default: enabled)
- `--no-s3`: Skip clearing S3 outputs (only clear local)

**What it clears:**

**Step 6 (Final Model):**
- Model selection metadata JSON
- Final features CSV (`train_final_features_no_leakage.csv`)
- Final features Parquet (`inputs/model_train/final_features.parquet`)
- Model JSON files (XGBoost, CatBoost)
- Model binaries (`.joblib`, `.cbm`)
- Feature importance CSV files

**Step 7 (SHAP Analysis):**
- Global importance CSV files (XGBoost, CatBoost)
- Sample values Parquet files (XGBoost, CatBoost)
- Summary plots (PNG)

**Step 8 (FFA Analysis):**
- `axp_explanations.parquet`
- `feature_importance_axp.parquet`
- `causal_importance.parquet`
- `interaction_analysis.parquet`

**Examples:**
```bash
# Clear all model outputs (local + S3)
bash utility_scripts/clear_models.sh

# Clear only local files (keep S3)
bash utility_scripts/clear_models.sh --no-s3

# Clear specific cohort/age band
bash utility_scripts/clear_models.sh --cohort opioid_ed --age-band 13-24

# Clear specific cohort/age band, local only
bash utility_scripts/clear_models.sh --cohort opioid_ed --age-band 13-24 --no-s3
```

**S3 Locations Cleared:**
- `s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/`
- `s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/`
- `s3://pgxdatalake/gold/ffa_analysis/{cohort}/{age_band}/`

---

### Clear Step 5 (PGx Analysis) Outputs

**Script:** `clear_pgx_step5_outputs.py`

Clears Step 5 (PGx feature engineering) outputs from S3 and local checkpoints.

**Usage:**
```bash
python utility_scripts/clear_pgx_step5_outputs.py --cohort <cohort> --age-band <age_band> [--local] [--s3]
```

**Options:**
- `--cohort <cohort>`: Cohort name (required)
- `--age-band <age_band>`: Age band (required)
- `--local`: Also clear local files
- `--s3`: Clear S3 outputs (default: enabled)

**Example:**
```bash
# Clear Step 5 outputs for opioid_ed 13-24
python utility_scripts/clear_pgx_step5_outputs.py --cohort opioid_ed --age-band 13-24
```

---

## Regenerating Stale Outputs

### Regenerate SHAP/FFA if Stale

**Script:** `regenerate_ffa_shap_if_stale.py`

Automatically regenerates Step 7 (SHAP) and Step 8 (FFA) outputs if Step 6 model outputs are newer than 5 minutes (configurable). This ensures downstream analyses stay in sync with model updates.

**Usage:**
```bash
# Single cohort/age band
python utility_scripts/regenerate_ffa_shap_if_stale.py --cohort <cohort> --age-band <age_band> [options]

# All cohorts
python utility_scripts/regenerate_ffa_shap_if_stale.py --all [options]
```

**Options:**
- `--cohort <cohort>`: Cohort name (required unless `--all`)
- `--age-band <age_band>`: Age band (required unless `--all`)
- `--all`: Process all cohort/age_band combinations found in Step 6 outputs
- `--stale-threshold-minutes <N>`: Minimum age difference to consider stale (default: 5 minutes)
- `--force`: Force regeneration even if outputs are not stale
- `--no-s3`: Skip clearing S3 outputs (only clear local)
- `--no-regenerate`: Only clear stale outputs, do not regenerate

**How it works:**
1. Checks Step 6 output timestamps (model files, feature tables)
2. Compares with Step 7 (SHAP) and Step 8 (FFA) output timestamps
3. If Step 6 is newer than threshold → clears stale outputs and regenerates
4. Step 7 runs before Step 8 (FFA depends on SHAP)

**Examples:**
```bash
# Check and regenerate if stale (5 minute threshold)
python utility_scripts/regenerate_ffa_shap_if_stale.py --cohort opioid_ed --age-band 13-24

# Force regeneration regardless of timestamps
python utility_scripts/regenerate_ffa_shap_if_stale.py --cohort opioid_ed --age-band 13-24 --force

# Use 10 minute threshold
python utility_scripts/regenerate_ffa_shap_if_stale.py --cohort opioid_ed --age-band 13-24 --stale-threshold-minutes 10

# Process all cohorts
python utility_scripts/regenerate_ffa_shap_if_stale.py --all

# Only clear stale outputs, don't regenerate
python utility_scripts/regenerate_ffa_shap_if_stale.py --cohort opioid_ed --age-band 13-24 --no-regenerate
```

**Output Files Checked:**

**Step 7 (SHAP):**
- `{cohort}_{age_band}_shap_global_importance_xgboost.csv` (required)
- `{cohort}_{age_band}_shap_sample_values_xgboost.parquet` (required)
- `{cohort}_{age_band}_shap_global_importance_catboost.csv` (optional)
- `{cohort}_{age_band}_shap_sample_values_catboost.parquet` (optional)

**Step 8 (FFA):**
- `xgboost/axp_explanations.parquet` (required)
- `xgboost/feature_importance_axp.parquet` (required)

---

## Status Checking

### Check S3 Checkpoints and Outputs

**Script:** `check_s3_checkpoints.py`

Checks checkpoint metadata and output file status for all pipeline steps across all cohorts/age bands.

**Usage:**
```bash
python utility_scripts/check_s3_checkpoints.py [--cohort <cohort>] [--age-band <age_band>] [--step <step_name>]
```

**Options:**
- `--cohort <cohort>`: Filter by cohort
- `--age-band <age_band>`: Filter by age band
- `--step <step_name>`: Filter by step (e.g., `7_shap_analysis`, `8_ffa_analysis`)

**Example:**
```bash
# Check all steps for all cohorts
python utility_scripts/check_s3_checkpoints.py

# Check specific cohort/age band
python utility_scripts/check_s3_checkpoints.py --cohort opioid_ed --age-band 13-24

# Check specific step
python utility_scripts/check_s3_checkpoints.py --step 7_shap_analysis
```

**Output:**
- Checkpoint status (exists/missing, completion timestamp)
- Output file status (exists/missing)
- Summary counts (completed vs. total)

**Steps Checked:**
- `4a_model_data`: `model_events.parquet`
- `4b_dtw_filter`: `model_events_no_protocols.parquet`, `protocol_summary_*.csv`, `event_intervals_*.parquet`
- `5_pgx_analysis`: PGx feature files
- `6_final_model`: Model artifacts and feature tables
- `7_shap_analysis`: SHAP global importance CSV, sample values Parquet (XGBoost, CatBoost)
- `8_ffa_analysis`: FFA Parquet outputs (`axp_explanations.parquet`, `feature_importance_axp.parquet`)

---

### Check SHAP Files in S3

**Script:** `check_shap_s3_files.py`

Quick check for SHAP analysis output files in S3 for a specific cohort/age band.

**Usage:**
```bash
python utility_scripts/check_shap_s3_files.py --cohort <cohort> --age-band <age_band>
```

**Example:**
```bash
python utility_scripts/check_shap_s3_files.py --cohort opioid_ed --age-band 13-24
```

**Output:**
- XGBoost Global Importance CSV: [EXISTS] / [MISSING]
- XGBoost Sample Values Parquet: [EXISTS] / [MISSING]
- CatBoost Global Importance CSV: [EXISTS] / [MISSING]
- CatBoost Sample Values Parquet: [EXISTS] / [MISSING]

---

### Check Cohort Status

**Script:** `check_cohort_status.py`

Lightweight status check for feature importance and local cohort/model data.

**Usage:**
```bash
python utility_scripts/check_cohort_status.py
```

**What it checks:**
- Aggregated feature importance in S3 (test year 2019)
- Local cohort parquet files under `data/cohorts_F1120`

**Output:**
- ASCII summary table showing status for each cohort/age band combination

---

## Other Utilities

### Download S3 Artifacts

**Script:** `download_s3_artifacts.py`

Download model artifacts and outputs from S3 to local.

**Usage:**
```bash
python utility_scripts/download_s3_artifacts.py --cohort <cohort> --age-band <age_band> [--step <step>]
```

---

### Monitor Resources

**Script:** `monitor_resources.sh`

Monitor system resources (CPU, memory, disk) during pipeline execution.

**Usage:**
```bash
bash utility_scripts/monitor_resources.sh
```

---

### Kill All Cohorts

**Script:** `kill_all_cohorts.sh`

Kill all running cohort workflow processes.

**Usage:**
```bash
bash utility_scripts/kill_all_cohorts.sh
```

**Warning:** This will terminate all running pipeline processes.

---

## Common Use Cases

### Rerun Steps 6-8 After Model Changes

```bash
# 1. Clear model outputs
bash utility_scripts/clear_models.sh --cohort opioid_ed --age-band 13-24

# 2. Rerun workflow (will skip Steps 1-5 if outputs exist)
bash utility_scripts/run_cohort_workflow.sh opioid_ed 13-24
```

### Regenerate SHAP/FFA After Model Update

```bash
# Automatically detects if Step 6 is newer and regenerates
python utility_scripts/regenerate_ffa_shap_if_stale.py --cohort opioid_ed --age-band 13-24
```

### Force Regeneration of SHAP/FFA

```bash
# Force regeneration regardless of timestamps
python utility_scripts/regenerate_ffa_shap_if_stale.py --cohort opioid_ed --age-band 13-24 --force
```

### Check Pipeline Status

```bash
# Check all steps for all cohorts
python utility_scripts/check_s3_checkpoints.py

# Check specific cohort/age band
python utility_scripts/check_s3_checkpoints.py --cohort opioid_ed --age-band 13-24

# Check SHAP files specifically
python utility_scripts/check_shap_s3_files.py --cohort opioid_ed --age-band 13-24
```

### Clear and Rerun Single Step

```bash
# Example: Clear and rerun Step 5 (PGx)
python utility_scripts/clear_pgx_step5_outputs.py --cohort opioid_ed --age-band 13-24
bash utility_scripts/run_cohort_workflow.sh opioid_ed 13-24 --skip-steps 6,7,8
```

### Rerun from Specific Step

```bash
# Rerun from Step 6 onwards (skips Steps 1-5)
bash utility_scripts/run_cohort_workflow.sh opioid_ed 13-24 --skip-steps 3,4a,4b,5a,5b,5c,5d
```

---

## Idempotency and Checkpoints

All pipeline steps use a unified checkpoint/idempotency system:

1. **Local File Checks:** Each step checks for local outputs first
2. **S3 Output Checks:** If local files missing, checks S3 for outputs
3. **Checkpoint Metadata:** Saves checkpoint JSON to S3 after completion
4. **File Format Awareness:** Idempotency checks match actual output formats

**Checkpoint Location:**
```
s3://pgx-repository/pipeline_checkpoints/{step_name}/{cohort}/{age_band}/checkpoint.json
```

**Idempotency Behavior:**
- ✅ If outputs exist locally → Skip step, optionally upload to S3
- ✅ If outputs exist in S3 → Download to local, skip step
- ✅ If checkpoint exists → Skip step (even if some files missing)
- ✅ If nothing exists → Run step, upload outputs, save checkpoint

**File Format Matching:**
- **Step 4b:** Checks `.parquet` and `.csv` (matches actual outputs)
- **Step 7:** Checks `.parquet` for sample values, `.csv` for global importance
- **Step 8:** Checks `.parquet` only (all outputs are Parquet)

---

## Troubleshooting

### Script Permission Errors

```bash
chmod +x utility_scripts/*.sh
```

### Line Ending Errors (Windows)

```bash
sed -i 's/\r$//' utility_scripts/*.sh
```

### Python Import Errors

Make sure you're running from the project root:
```bash
cd /path/to/pgx-analysis
python utility_scripts/check_s3_checkpoints.py
```

### S3 Access Errors

Ensure AWS credentials are configured:
```bash
aws configure
# Or set environment variables:
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

### Checkpoint Not Found

If checkpoints are missing but outputs exist, the step will still skip (idempotent). To force rerun:
```bash
# Clear outputs first
bash utility_scripts/clear_models.sh --cohort <cohort> --age-band <age_band>

# Then rerun
bash utility_scripts/run_cohort_workflow.sh <cohort> <age_band>
```

---

## Related Documentation

- [`README_workflow_scripts.md`](README_workflow_scripts.md) - Complete workflow execution scripts
- [`../docs/README_analysis_workflow.md`](../docs/README_analysis_workflow.md) - Detailed pipeline documentation
- [`../docs/CrossStep_Development/README_data_pipeline_architecture.md`](../docs/CrossStep_Development/README_data_pipeline_architecture.md) - Pipeline architecture and optimization

---

## Notes

- All scripts are idempotent and will skip completed steps
- S3 operations require AWS credentials configured
- Scripts check both local files and S3 for outputs
- Parquet files are preferred over CSV for new operations
- CatBoost outputs are optional (model might not be available)

