# Pipeline Checkpoints, Refresh Mechanisms, and Workflow Resets

## Overview

The PGx Analysis pipeline uses **S3-based checkpointing** to enable resumable, idempotent workflow execution. This document covers:

1. **Checkpoint Implementation** - How checkpoints work and enable workflow resumability
2. **Refresh Mechanism** - How to update existing outputs and avoid accidental overwrites
3. **Clearing Workflow** - How to fully reset for fresh runs

**Key Principle:** By default, **feature importance is preserved** across runs. Notebook 2 only adds missing (cohort, age_band) combinations. Use `--clear-feature-importance` only when you need a full recompute.

---

## Table of Contents

- [Pipeline Checkpoints, Refresh Mechanisms, and Workflow Resets](#pipeline-checkpoints-refresh-mechanisms-and-workflow-resets)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Quick Start: Clearing for a Fresh Run](#quick-start-clearing-for-a-fresh-run)
  - [Checkpoint Implementation](#checkpoint-implementation)
    - [How Checkpoints Work](#how-checkpoints-work)
    - [Checkpoint Storage](#checkpoint-storage)
    - [Implementation by Step](#implementation-by-step)
    - [Helper Module: checkpoint\_utils.py](#helper-module-checkpoint_utilspy)
    - [Usage Pattern](#usage-pattern)
  - [Refresh Mechanism](#refresh-mechanism)
    - [Idempotent Upload Behavior](#idempotent-upload-behavior)
    - [How to Refresh Files](#how-to-refresh-files)
      - [Option 1: Delete Before Re-running (Recommended)](#option-1-delete-before-re-running-recommended)
      - [Option 2: Force Overwrite (Modify Code)](#option-2-force-overwrite-modify-code)
      - [Option 3: Manual S3 Deletion](#option-3-manual-s3-deletion)
    - [File Locations](#file-locations)
      - [SHAP Analysis (Step 7)](#shap-analysis-step-7)
      - [FFA Analysis (Step 8)](#ffa-analysis-step-8)
      - [Checking File Existence](#checking-file-existence)
  - [Clearing Workflow for Full Run](#clearing-workflow-for-full-run)
    - [What Gets Cleared](#what-gets-cleared)
      - [1. Checkpoints (S3: pgx-repository)](#1-checkpoints-s3-pgx-repository)
      - [2. S3 Artifacts (pgxdatalake)](#2-s3-artifacts-pgxdatalake)
      - [3. EC2 / Local Artifacts](#3-ec2--local-artifacts)
    - [What Is Preserved](#what-is-preserved)
    - [Clearing Options](#clearing-options)
  - [Notebook 0: Config and Pipeline](#notebook-0-config-and-pipeline)
  - [Best Practices](#best-practices)
    - [Before Running Pipeline](#before-running-pipeline)
    - [During Development](#during-development)
    - [For Production](#for-production)
    - [After Pipeline Runs](#after-pipeline-runs)
  - [Related Files](#related-files)
    - [Core Implementation](#core-implementation)
    - [Scripts](#scripts)
    - [Documentation](#documentation)
    - [Notebooks](#notebooks)
  - [Summary](#summary)

---

## Quick Start: Clearing for a Fresh Run

**Default behavior:** Feature importance (Step 3a/3b and `gold/feature_importance`) is **preserved**. Notebook 2 will only add missing (cohort, age_band) combinations.

```bash
cd ~/pgx-analysis   # or your project root
chmod +x utility_scripts/cleanup_cohort_data.sh

# Default: preserves feature importance; notebook 2 only adds missing
./utility_scripts/cleanup_cohort_data.sh

# Full reset including feature importance (full recompute in notebook 2)
./utility_scripts/cleanup_cohort_data.sh --clear-feature-importance

# Skip confirmation prompt
./utility_scripts/cleanup_cohort_data.sh --yes
```

After clearing, run the workflow: **0** → **1** → **2** → **3** → **4** → **5**

---

## Checkpoint Implementation

### How Checkpoints Work

All pipeline steps (Steps 1b, 4-10) support S3-based checkpointing and idempotency checks. This ensures:

1. **Resumability** - Steps can be safely interrupted and resumed
2. **Idempotency** - Completed steps are skipped on re-run
3. **Durability** - Outputs are persisted to S3, surviving instance termination
4. **Cross-instance** - Checkpoints work across different EC2 instances
5. **Efficiency** - Avoids redundant computation

### Checkpoint Storage

Checkpoints are stored at:

```
s3://pgx-repository/pipeline_checkpoints/{step_name}/{cohort}/{age_band}/checkpoint.json
```

**Checkpoint JSON structure:**

```json
{
  "step_name": "4_model_data",
  "cohort": "opioid_ed",
  "age_band": "13-24",
  "completed_at": "2026-01-04T03:30:00Z",
  "status": "completed",
  "metadata": {
    "n_cases": 311228,
    "n_controls": 837267
  },
  "output_paths": [
    "s3://pgxdatalake/gold/cohorts_model_data/..."
  ]
}
```

### Implementation by Step

| Step | Status | S3 Check | S3 Upload | Checkpoint Location |
|------|--------|----------|-----------|---------------------|
| **1b: Event Filter** | ✅ | Yes | Yes | `pipeline_checkpoints/1b_apcd_event_filter/{cohort}/{age_band}/` |
| **4: Model Data** | ✅ | Yes (`model_events.parquet`) | Yes | `pipeline_checkpoints/4_model_data/{cohort}/{age_band}/` |
| **5: PGx Features** | ⏳ | Local only | TODO | `pipeline_checkpoints/5_pgx_analysis/{cohort}/{age_band}/` |
| **6: Final Model** | ⏳ | S3 check added | Upload TODO | `pipeline_checkpoints/6_final_model/{cohort}/{age_band}/` |
| **7: SHAP** | ⏳ | Local binary | TODO | `pipeline_checkpoints/7_shap_analysis/{cohort}/{age_band}/` |
| **8: FFA** | ⏳ | Local JSON | TODO | `pipeline_checkpoints/8_ffa_analysis/{cohort}/{age_band}/` |
| **9: Combined Analysis** | ⏳ | Not yet implemented | TODO | `pipeline_checkpoints/combined_analysis/{cohort}/{age_band}/` |
| **10: Dashboard Visuals** | ✅ | Yes (DTW) | Yes | `pipeline_checkpoints/9_dashboard_visuals/{cohort}/{age_band}/` |

**Note:** Step 10 (dashboard visuals) uses checkpoint name `9_dashboard_visuals` for historical reasons.

### Helper Module: checkpoint_utils.py

Location: `py_helpers/checkpoint_utils.py`

**Core Functions:**

```python
# Check if S3 outputs exist
check_s3_output_exists(s3_path: str) -> bool
check_step_outputs_exist(s3_paths: List[str], logger=None) -> bool

# Check if checkpoint exists
check_step_checkpoint_exists(step_name, cohort, age_band, logger=None) -> bool

# Upload files to S3
upload_file_to_s3(local_path, s3_path, logger=None, check_exists=True) -> bool

# Save checkpoint metadata
save_step_checkpoint(step_name, cohort, age_band, metadata, output_paths, logger=None) -> bool
```

### Usage Pattern

Each step follows this pattern:

```python
# 1. Check S3 for existing outputs (idempotency)
try:
    from py_helpers.checkpoint_utils import check_step_outputs_exist, check_step_checkpoint_exists
    
    s3_output_paths = [
        f"s3://pgxdatalake/gold/{step}/{cohort}/{age_band}/output1.parquet",
        f"s3://pgxdatalake/gold/{step}/{cohort}/{age_band}/output2.csv",
    ]
    
    if check_step_outputs_exist(s3_output_paths, logger) or \
       check_step_checkpoint_exists(step_name, cohort, age_band, logger):
        logger.info(f"Step {step_name} outputs already exist in S3; skipping.")
        return
except ImportError:
    pass  # Fallback to local-only if checkpoint_utils not available

# 2. Run the step (existing logic)
# ... perform computation ...

# 3. Upload outputs to S3 and save checkpoint
try:
    from py_helpers.checkpoint_utils import upload_file_to_s3, save_step_checkpoint
    
    s3_outputs = []
    if local_output.exists():
        s3_path = f"s3://pgxdatalake/gold/{step}/{cohort}/{age_band}/output.parquet"
        if upload_file_to_s3(local_output, s3_path, logger):
            s3_outputs.append(s3_path)
    
    save_step_checkpoint(
        step_name=step_name,
        cohort=cohort,
        age_band=age_band,
        metadata={"key": "value"},
        output_paths=s3_outputs,
        logger=logger,
    )
except ImportError:
    pass  # Checkpoint saving is optional
```

---

## Refresh Mechanism

### Idempotent Upload Behavior

The `upload_file_to_s3()` function uses **idempotent uploads** by default:

```python
def upload_file_to_s3(
    local_path: Path, 
    s3_path: str, 
    logger: Optional[logging.Logger] = None, 
    check_exists: bool = True
) -> bool:
```

**Default behavior (`check_exists=True`):**

- Checks if file already exists in S3 using `head_object()`
- If file exists → **skips upload** (returns `True`)
- If file doesn't exist → uploads the file
- **Result:** Files are **NOT overwritten** on subsequent runs

**Why idempotent?**

- ✅ **Prevents accidental overwrites** - Protects existing results
- ✅ **Enables resumable pipelines** - Can re-run steps without losing data
- ✅ **Cost savings** - Avoids unnecessary S3 PUT operations
- ✅ **Faster runs** - Skips uploads if files already exist

### How to Refresh Files

#### Option 1: Delete Before Re-running (Recommended)

Use the utility script to clear outputs before re-running:

```bash
# Clear Step 7 (SHAP) outputs
python archived/utility_scripts/regenerate_ffa_shap_if_stale.py \
    --cohort opioid_ed \
    --age-band 13-24 \
    --clear-step7

# Clear Step 8 (FFA) outputs
python archived/utility_scripts/regenerate_ffa_shap_if_stale.py \
    --cohort opioid_ed \
    --age-band 13-24 \
    --clear-step8
```

This script:
- Deletes local output files
- Deletes S3 files under the cohort/age_band prefix
- Clears checkpoint metadata

#### Option 2: Force Overwrite (Modify Code)

To force overwrite, modify the upload call to set `check_exists=False`:

```python
upload_file_to_s3(local_path, s3_path, logger, check_exists=False)
```

⚠️ **Note:** This is not recommended for production as it bypasses the safety mechanism.

#### Option 3: Manual S3 Deletion

Delete files directly from S3 using AWS CLI or console:

```bash
# Delete specific file
aws s3 rm s3://pgxdatalake/gold/shap_analysis/opioid_ed/13-24/opioid_ed_13_24_shap_global_importance_catboost.csv

# Delete all SHAP files for a cohort/age_band
aws s3 rm s3://pgxdatalake/gold/shap_analysis/opioid_ed/13-24/ --recursive

# Delete all FFA files for a cohort/age_band
aws s3 rm s3://pgxdatalake/gold/ffa_analysis/opioid_ed/13-24/ --recursive
```

### File Locations

#### SHAP Analysis (Step 7)

**Base path:** `s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/`

**Files:**
- `{cohort}_{age_band_fname}_shap_global_importance_xgboost.csv`
- `{cohort}_{age_band_fname}_shap_sample_values_xgboost.parquet`
- `{cohort}_{age_band_fname}_shap_global_importance_catboost.csv`
- `{cohort}_{age_band_fname}_shap_sample_values_catboost.parquet`

#### FFA Analysis (Step 8)

**Base path:** `s3://pgxdatalake/gold/ffa_analysis/{cohort}/{age_band}/{model_type}/`

**Files:**
- `axp_explanations.csv`
- `feature_importance_axp.csv`
- `causal_importance.csv`
- `interaction_analysis.csv` (if available)

#### Checking File Existence

```bash
# Check SHAP files
python archived/utility_scripts/check_shap_s3_files.py \
    --cohort opioid_ed \
    --age-band 13-24

# Or use Python directly
from py_helpers.checkpoint_utils import check_s3_output_exists

s3_path = "s3://pgxdatalake/gold/shap_analysis/opioid_ed/13-24/opioid_ed_13_24_shap_global_importance_catboost.csv"
exists = check_s3_output_exists(s3_path)
```

---

## Clearing Workflow for Full Run

### What Gets Cleared

Use `./utility_scripts/cleanup_cohort_data.sh` to clear:

#### 1. Checkpoints (S3: pgx-repository)

| Location | Purpose |
|----------|---------|
| `s3://pgx-repository/pipeline_checkpoints/` | Step checkpoints used by `py_helpers.checkpoint_utils` (1b, 4_model_data, 6, etc.). Steps skip if checkpoint exists. |
| `s3://pgx-repository/pgx-pipeline-status/` | Legacy/alternate pipeline status (create_cohort, feature_importance_eda, model_data, final_model). |

Clearing these forces steps to re-run (unless they also check for output files in S3).

#### 2. S3 Artifacts (pgxdatalake)

| Prefix | Step | Contents |
|--------|------|----------|
| `gold/cohorts/` | 2 | Cohort parquet (cohort_name=non_opioid_ed, cohort_name=opioid_ed). |
| `gold/cohorts_model_data/` | 4 | Model data (model_events.parquet) — current path. |
| `gold/model_data/` | 4 | Model data — alternate path. |
| `gold/event_filter/` | 1b | model_events_no_protocols.parquet, protocol summaries. |
| `gold/feature_importance/` | 3a | Aggregated FI CSVs (and `_baseline/` subfolder). **Preserved by default.** |
| `gold/bupar/` | 3b | BupaR / feature importance EDA outputs. |
| `gold/pgx_features/` | 5 | PGx feature engineering outputs. |
| `gold/final_model/` | 6 | Trained model binaries and metadata. |
| `gold/shap_analysis/` | 7 | SHAP outputs. |
| `gold/ffa_analysis/` | 8 | FFA (AXP) outputs. |
| `gold/combined_analysis/` | 9 | Combined risk dashboard inputs. |
| `gold/models/` | 6 (legacy) | Legacy trained models path. |
| `gold/4a_model_data/` | 4 (legacy) | Legacy model data path. |

#### 3. EC2 / Local Artifacts

**Data root** (default on Linux: `PGX_DATA_ROOT` or `/mnt/nvme`):

| Path | Step | Contents |
|------|------|----------|
| `$PGX_DATA_ROOT/gold/cohorts/` | 2 | Synced cohort parquet. |
| `$PGX_DATA_ROOT/4_model_data/` or `4a_model_data/` | 4 | Model data by cohort/age_band. |

**Project root** (e.g. `~/pgx-analysis`):

| Path | Step | Contents |
|------|------|----------|
| `data/gold/cohorts/` | 2 | Project-local copy of cohort parquet (same layout as S3 gold/cohorts). |
| `2_create_cohort/` (cohort_metrics, etc.) | 2 | Local cohort metrics. |
| `3b_feature_importance_eda/outputs/` | 3b | Feature importance EDA. **Preserved by default.** |
| `3_feature_importance/outputs/` | 3a | Feature importance (legacy naming). **Preserved by default.** |
| `3a_feature_importance/outputs/` | 3a | Feature importance MC CV + aggregated FI. **Preserved by default.** |
| `1b_apcd_event_filter/outputs/` | 1b | Event filter outputs, for_review. |
| `4_model_data/` (local outputs under project) | 4 | model_events*.parquet if written to project. |
| `5_pgx_analysis/` outputs | 5 | PGx feature files. |
| `6_final_model/models/` | 6 | Trained model files. |
| `7_shap_analysis/` outputs | 7 | SHAP outputs. |
| `8_ffa_analysis/` outputs | 8 | FFA outputs. |
| `10_risk_dashboard/` outputs | 9 | Dashboard outputs. |

### What Is Preserved

**Always preserved (never deleted):**

1. **Source data:**
   - `/mnt/nvme/gold/medical` and `/mnt/nvme/gold/pharmacy` - Never deleted
   - Step 1a/1b input data

2. **Feature importance (by default):**
   - `gold/feature_importance/` (including `_baseline/` subfolder)
   - `3a_feature_importance/outputs/`
   - `3b_feature_importance_eda/outputs/`
   - `pgx-pipeline-status/feature_importance_eda`
   - **Use `--clear-feature-importance` flag to clear these**

3. **Historical feature importance bucket:**
   - `s3://pgx-repository/pgx-analysis/3_feature_importance/outputs/`
   - Has versioning enabled - never deleted
   - Step 1b reads from this when local/pgxdatalake FI is missing

4. **Baseline aggregated feature importances:**
   - `_baseline/` under `gold/feature_importance/` in pgxdatalake
   - Overwritten when you re-run Step 3a with `--baseline`

### Clearing Options

```bash
# Default: preserves feature importance; notebook 2 only adds missing
./utility_scripts/cleanup_cohort_data.sh

# Clear checkpoints but keep S3 (steps will still skip if outputs exist)
./utility_scripts/cleanup_cohort_data.sh --skip-checkpoints

# Only delete local/EC2 files
./utility_scripts/cleanup_cohort_data.sh --skip-s3

# Only delete S3 (keep EC2/local)
./utility_scripts/cleanup_cohort_data.sh --skip-local

# Full reset including feature importance (full recompute in notebook 2)
./utility_scripts/cleanup_cohort_data.sh --clear-feature-importance

# Skip confirmation prompt
./utility_scripts/cleanup_cohort_data.sh --yes
```

---

## Notebook 0: Config and Pipeline

**[0_config_and_pipeline.ipynb](../0_config_and_pipeline.ipynb)** is the entry point for clearing checkpoints and resetting the workflow.

**Purpose:**
- Clear EC2 NVMe and project pipeline output directories
- Run cleanup script with appropriate flags
- Verify environment and dependencies
- Contains step-by-step instructions for running the pipeline

**Execution order:**

1. Run **Notebook 0** to clear checkpoints (and optionally full cleanup)
2. Run notebooks **1** → **2** → **3** → **4** → **5** in sequence
3. Each notebook syncs required inputs from S3 to local via `aws s3 sync` (idempotent)
4. Each notebook uses S3 checkpoints to skip completed steps

**Default behavior in Notebook 0:**

- Runs cleanup script **without** `--clear-feature-importance`
- Feature importance is **preserved**
- Notebook 2 only adds missing (cohort, age_band) combinations
- Set `FORCE_FEATURE_IMPORTANCE = False` in Notebook 2

**Full reset in Notebook 0:**

- Run cleanup script **with** `--clear-feature-importance`
- All feature importance outputs are cleared
- Notebook 2 will recompute all (cohort, age_band) combinations
- Set `FORCE_FEATURE_IMPORTANCE = True` in Notebook 2 if needed

---

## Best Practices

### Before Running Pipeline

1. **Check what exists:** Use utility scripts to verify current state
   ```bash
   python py_helpers/checkpoint_utils.py  # List checkpoint status
   python archived/utility_scripts/check_shap_s3_files.py --cohort opioid_ed --age-band 13-24
   ```

2. **Clear only what's needed:**
   - **Default (incremental):** Run cleanup script without flags - preserves feature importance
   - **Full recompute:** Use `--clear-feature-importance` when models or features changed
   - **Selective clearing:** Use S3 CLI or utility scripts for specific steps

3. **Verify cleanup:**
   ```bash
   # Review the log
   ./utility_scripts/check_cleanup_log.sh
   # Or check: cleanup_cohort_data_YYYYMMDD_HHMMSS.log in repo root
   ```

### During Development

1. **Keep idempotent behavior** (default) - prevents accidental overwrites
2. **Use development flags** when testing:
   - `--clear-step7` / `--clear-step8` for SHAP/FFA refresh
   - Set `FORCE_RERUN = True` in notebooks to bypass checkpoints

### For Production

1. **Full reset before major runs:**
   ```bash
   ./utility_scripts/cleanup_cohort_data.sh --clear-feature-importance --yes
   ```

2. **Incremental updates (default):**
   ```bash
   ./utility_scripts/cleanup_cohort_data.sh --yes
   ```

3. **Monitor checkpoints:**
   - Check S3 checkpoint times to verify completion
   - Validate output file existence in S3

### After Pipeline Runs

1. **Baseline feature importance** is expected on S3 (`gold/feature_importance/.../_baseline/`)
2. If you **did not** clear `gold/feature_importance/` (or kept `_baseline`), Step 1b will use it
3. If you **cleared everything**, run Step 3a with `--baseline` first for each cohort/age_band
4. Then run 1b, then 3a without `--baseline`

---

## Related Files

### Core Implementation
- [py_helpers/checkpoint_utils.py](../py_helpers/checkpoint_utils.py) - Upload and checkpoint functions
- [py_helpers/s3_utils.py](../py_helpers/s3_utils.py) - S3 utility functions

### Scripts
- [utility_scripts/cleanup_cohort_data.sh](../utility_scripts/cleanup_cohort_data.sh) - Main cleanup script
- [archived/utility_scripts/regenerate_ffa_shap_if_stale.py](../archived/utility_scripts/regenerate_ffa_shap_if_stale.py) - Clear SHAP/FFA outputs
- [archived/utility_scripts/check_shap_s3_files.py](../archived/utility_scripts/check_shap_s3_files.py) - Check file existence

### Documentation
- [README_file_resolver.md](README_file_resolver.md) - Universal file resolution across storage layers
- [README_feature_engineering_and_analysis.md](README_feature_engineering_and_analysis.md) - Feature engineering pipeline
- [WORKFLOW_EXECUTION_TODO.md](../WORKFLOW_EXECUTION_TODO.md) - Pipeline execution order and checklist
- [CrossStep_Development/README_data_pipeline_architecture.md](CrossStep_Development/README_data_pipeline_architecture.md) - Data pipeline architecture

### Notebooks
- [0_config_and_pipeline.ipynb](../0_config_and_pipeline.ipynb) - Config and cleanup entry point
- [1_cohort_workflow.ipynb](../1_cohort_workflow.ipynb) - Cohort creation (Step 2)
- [2_feature_importance.ipynb](../2_feature_importance.ipynb) - Feature importance (Steps 3a/3b)
- [3_model_train_shap_ffa.ipynb](../3_model_train_shap_ffa.ipynb) - Model training and analysis (Steps 4-8)
- [4_dashboard_visuals.ipynb](../4_dashboard_visuals.ipynb) - Dashboard visuals (Step 10)
- [5_build_and_deploy.ipynb](../5_build_and_deploy.ipynb) - Build and deploy risk calculator

---

## Summary

**Checkpoint system provides:**
- ✅ Resumable workflows across interruptions
- ✅ Idempotent execution (skip completed steps)
- ✅ Durable S3 storage with cross-instance support
- ✅ Efficient resource usage (avoid redundant computation)

**Refresh mechanism provides:**
- ✅ Protection against accidental overwrites (default)
- ✅ Explicit control over file updates (when needed)
- ✅ Multiple options for refreshing outputs

**Clearing workflow provides:**
- ✅ Default incremental mode (preserves feature importance)
- ✅ Full reset mode (clear everything including FI)
- ✅ Fine-grained control over what gets cleared
- ✅ Safe cleanup with confirmation prompts

**Default behavior:** Run cleanup script without flags to **preserve feature importance** and only add missing (cohort, age_band) in notebook 2. Use `--clear-feature-importance` only when you need a full recompute of feature importance.
