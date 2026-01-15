# Model Save Locations

## Where Scripts Say They Save Models

Based on `6_final_model_selection/run_final_model.py`:

### Local Save Locations

**Base directory**: `6_final_model/outputs/{cohort}/{age_band_fname}/`

**Model files saved to**:
1. **Best XGBoost JSON**:
   - Path: `6_final_model/outputs/{cohort}/{age_band_fname}/final_model_json/{cohort}_{age_band_fname}_best_xgboost_model.json`
   - Print statement: `"Saved BEST XGBoost model JSON ({best_xgb_variant}) to {xgb_json_path}"`

2. **Best CatBoost JSON**:
   - Path: `6_final_model/outputs/{cohort}/{age_band_fname}/final_model_json/{cohort}_{age_band_fname}_best_catboost_model.json`
   - Print statement: `"Saved BEST CatBoost model JSON to {cb_json_path}"`

3. **Best CatBoost Binary (.cbm)**:
   - Path: `6_final_model/outputs/{cohort}/{age_band_fname}/final_model_json/{cohort}_{age_band_fname}_best_catboost_model.cbm`
   - Print statement: `"Saved BEST CatBoost model binary to {cb_binary_path} (for SHAP analysis)"`

4. **Model Selection Metadata**:
   - Path: `6_final_model/outputs/{cohort}/{age_band_fname}/{cohort}_{age_band_fname}_model_selection_metadata.json`
   - Contains: best variant, recall, PR-AUC, selection reason

### S3 Save Locations

**Base S3 path**: `s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/`

**Model files uploaded to**:
1. **Best XGBoost JSON**:
   - S3: `s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{cohort}_{age_band_fname}_best_xgboost_model.json`

2. **Best CatBoost JSON**:
   - S3: `s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{cohort}_{age_band_fname}_best_catboost_model.json`

3. **Best CatBoost Binary (.cbm)**:
   - S3: `s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{cohort}_{age_band_fname}_best_catboost_model.cbm`

4. **Model Selection Metadata**:
   - S3: `s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{cohort}_{age_band_fname}_model_selection_metadata.json`

### Save Function Logic

The script uses `save_model_idempotent()` function (lines 1778-1810) which:
1. **Checks S3 first**: If file exists in S3, skips save (idempotent)
2. **Checks local**: If file exists locally, skips save
3. **Saves locally**: Calls save function to create local file
4. **Uploads to S3**: Uploads the saved file to S3

**Key code**:
```python
def save_model_idempotent(local_path: Path, s3_path: str, save_func, *save_args, **save_kwargs):
    """Save model file idempotently: check S3 first, then local, then save and upload."""
    # Check S3 first
    if s3_exists(s3_path):
        print(f"[INFO] Model already exists in S3: {s3_path}; skipping save.")
        return
    
    # Check local
    if local_path.exists():
        print(f"[INFO] Model already exists locally: {local_path}; skipping save.")
        # Still upload to S3 if not there
        if not s3_exists(s3_path):
            upload_file_to_s3(local_path, s3_path)
        return
    
    # Save locally
    save_func(*save_args, **save_kwargs)
    print(f"Saved model to {local_path}")
    
    # Upload to S3
    upload_file_to_s3(local_path, s3_path)
    print(f"Uploaded to S3: {s3_path}")
```

## Current Status

**Local models found**:
- ✅ `opioid_ed/13-24`: Both XGBoost and CatBoost models found locally

**S3 models checked**:
- ❌ Models NOT found in expected S3 locations: `s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/`

**Possible explanations**:
1. Models were saved but later deleted/moved
2. Models exist in S3 but in different location than expected
3. Models were never uploaded to S3 (upload failed silently)
4. Models were saved to a different S3 bucket/path

## Expected vs Actual

**Expected S3 path** (from script):
```
s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{cohort}_{age_band_fname}_best_xgboost_model.json
```

**Example for `opioid_ed/13-24`**:
```
s3://pgxdatalake/gold/final_model/opioid_ed/13-24/opioid_ed_13_24_best_xgboost_model.json
```

**Note**: The script uses `age_band` (with hyphens) in S3 paths, but `age_band_fname` (with underscores) in filenames.
