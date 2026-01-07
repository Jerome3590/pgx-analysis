# S3 File Refresh Mechanism

## Overview

SHAP and FFA analysis outputs are saved to S3 with **idempotent uploads** - files are **not automatically overwritten** on new runs. This document explains how files are managed and refreshed.

## Current Behavior

### Upload Mechanism

The `upload_file_to_s3()` function in `py_helpers/checkpoint_utils.py` uses **idempotent uploads** by default:

```python
def upload_file_to_s3(local_path: Path, s3_path: str, logger: Optional[logging.Logger] = None, check_exists: bool = True) -> bool:
```

**Default behavior (`check_exists=True`):**
- Checks if file already exists in S3 using `head_object()`
- If file exists → **skips upload** (returns `True`)
- If file doesn't exist → uploads the file
- **Result**: Files are NOT overwritten on subsequent runs

### Why Idempotent?

- **Prevents accidental overwrites**: Protects existing results
- **Enables resumable pipelines**: Can re-run steps without losing data
- **Cost savings**: Avoids unnecessary S3 PUT operations
- **Faster runs**: Skips uploads if files already exist

## How to Refresh Files

### Option 1: Delete Before Re-running (Recommended)

Use the utility script to clear outputs before re-running:

```bash
# Clear Step 7 (SHAP) outputs
python utility_scripts/regenerate_ffa_shap_if_stale.py \
    --cohort opioid_ed \
    --age-band 13-24 \
    --clear-step7

# Clear Step 8 (FFA) outputs
python utility_scripts/regenerate_ffa_shap_if_stale.py \
    --cohort opioid_ed \
    --age-band 13-24 \
    --clear-step8
```

This script:
- Deletes local output files
- Deletes S3 files under the cohort/age_band prefix
- Clears checkpoint metadata

### Option 2: Force Overwrite (Modify Code)

To force overwrite, modify the upload call to set `check_exists=False`:

```python
upload_file_to_s3(local_path, s3_path, logger, check_exists=False)
```

**Note**: This is not recommended for production as it bypasses the safety mechanism.

### Option 3: Manual S3 Deletion

Delete files directly from S3 using AWS CLI or console:

```bash
# Delete specific file
aws s3 rm s3://pgxdatalake/gold/shap_analysis/opioid_ed/13-24/opioid_ed_13_24_shap_global_importance_catboost.csv

# Delete all SHAP files for a cohort/age_band
aws s3 rm s3://pgxdatalake/gold/shap_analysis/opioid_ed/13-24/ --recursive

# Delete all FFA files for a cohort/age_band
aws s3 rm s3://pgxdatalake/gold/ffa_analysis/opioid_ed/13-24/ --recursive
```

## File Locations

### SHAP Analysis (Step 7)

**Base path**: `s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/`

**Files**:
- `{cohort}_{age_band_fname}_shap_global_importance_xgboost.csv`
- `{cohort}_{age_band_fname}_shap_sample_values_xgboost.parquet`
- `{cohort}_{age_band_fname}_shap_global_importance_catboost.csv`
- `{cohort}_{age_band_fname}_shap_sample_values_catboost.parquet`

### FFA Analysis (Step 8)

**Base path**: `s3://pgxdatalake/gold/ffa_analysis/{cohort}/{age_band}/{model_type}/`

**Files**:
- `axp_explanations.csv`
- `feature_importance_axp.csv`
- `causal_importance.csv`
- `interaction_analysis.csv` (if available)

## Checking File Existence

Use the utility script to check if files exist:

```bash
python utility_scripts/check_shap_s3_files.py \
    --cohort opioid_ed \
    --age-band 13-24
```

Or use the Python function directly:

```python
from py_helpers.checkpoint_utils import check_s3_output_exists

s3_path = "s3://pgxdatalake/gold/shap_analysis/opioid_ed/13-24/opioid_ed_13_24_shap_global_importance_catboost.csv"
exists = check_s3_output_exists(s3_path)
```

## Best Practices

1. **Before re-running analysis**: Clear outputs if you want fresh results
2. **For production**: Keep idempotent behavior (default) to prevent accidental overwrites
3. **For development**: Use `--clear-step7` or `--clear-step8` flags when needed
4. **Check before clearing**: Use `check_shap_s3_files.py` to verify what exists

## Related Files

- `py_helpers/checkpoint_utils.py` - Upload and checkpoint functions
- `utility_scripts/regenerate_ffa_shap_if_stale.py` - Clear outputs utility
- `utility_scripts/check_shap_s3_files.py` - Check file existence utility
- `py_helpers/s3_utils.py` - S3 utility functions (including `s3_delete_object_if_exists`)

