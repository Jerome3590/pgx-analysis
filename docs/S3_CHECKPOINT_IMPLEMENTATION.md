# S3 Checkpoint and Idempotency Implementation

## Overview

All pipeline steps (4-9) now support S3-based checkpointing and idempotency checks. This ensures that:
1. Steps can be safely interrupted and resumed
2. Completed steps are skipped on re-run
3. Outputs are persisted to S3 for durability
4. Checkpoints are stored in S3 for cross-instance resumability

## Implementation Status

### ✅ Step 4: Model Data Creation
- **S3 Check**: Checks for `model_events.parquet` in S3 before running
- **Upload**: Uses `aws s3 sync` to upload after completion
- **Checkpoint**: Saves checkpoint metadata to S3
- **Location**: `s3://pgxdatalake/gold/cohorts_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet` (or `gold/model_data/{cohort}/{age_band}/model_events.parquet`)

**Note:** Event filtering (administrative/scheduling codes) is in Step 1b. DTW protocol filtering is for dashboard visualizations only (Step 9), not a separate pipeline checkpoint.

### ⏳ Step 5: PGx Feature Engineering
- **Status**: Already has local idempotency (checks for existing mapping files)
- **TODO**: Add S3 checks and uploads for:
  - `drug_gene_mappings.csv`
  - `allele_frequencies.csv`
  - `pgx_added_features_{cohort}_{age_band}.csv`
- **Location**: `s3://pgxdatalake/gold/pgx_features/{cohort}/{age_band}/`

### ⏳ Step 6: Final Model Training
- **Status**: S3 check added at start of `main()`
- **TODO**: Add S3 uploads after model training completes
- **Location**: `s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/`
- **Files**: Best XGBoost JSON, Best CatBoost CBM, model selection metadata

### ⏳ Step 7: SHAP Analysis
- **Status**: Checks for model binary locally
- **TODO**: Add S3 checks for:
  - Model binary in S3
  - SHAP outputs (values, importance, plots)
- **Location**: `s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/`

### ⏳ Step 8: FFA Analysis
- **Status**: Checks for model JSON locally
- **TODO**: Add S3 checks for:
  - Model JSON in S3
  - FFA outputs (AXP explanations, importance)
- **Location**: `s3://pgxdatalake/gold/ffa_analysis/{cohort}/{age_band}/`

### ⏳ Step 9: Risk Dashboard (combined SHAP + FFA)
- **Status**: Not yet implemented
- **TODO**: Add S3 checks and uploads for combined analysis outputs
- **Location**: `s3://pgxdatalake/gold/combined_analysis/{cohort}/{age_band}/`

## Helper Module: `py_helpers/checkpoint_utils.py`

Provides utilities for S3 checkpoint management:

### Functions

- `check_s3_output_exists(s3_path: str) -> bool`
  - Check if a single S3 object exists

- `check_step_outputs_exist(s3_paths: List[str], logger=None) -> bool`
  - Check if all step outputs exist in S3

- `check_step_checkpoint_exists(step_name, cohort, age_band, logger=None) -> bool`
  - Check if checkpoint metadata exists in S3

- `upload_file_to_s3(local_path, s3_path, logger=None) -> bool`
  - Upload a local file to S3

- `save_step_checkpoint(step_name, cohort, age_band, metadata, output_paths, logger=None) -> bool`
  - Save checkpoint metadata to S3

### Checkpoint Storage

Checkpoints are stored at:
```
s3://pgx-repository/pipeline_checkpoints/{step_name}/{cohort}/{age_band}/checkpoint.json
```

Checkpoint JSON structure:
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

## Usage Pattern

Each step follows this pattern:

```python
# 1. Check S3 for existing outputs (idempotency)
try:
    from py_helpers.checkpoint_utils import check_step_outputs_exist, check_step_checkpoint_exists
    
    s3_output_paths = [
        f"s3://pgxdatalake/gold/{step}/{cohort}/{age_band}/output1.parquet",
        f"s3://pgxdatalake/gold/{step}/{cohort}/{age_band}/output2.csv",
    ]
    
    if check_step_outputs_exist(s3_output_paths, logger) or check_step_checkpoint_exists(step_name, cohort, age_band, logger):
        logger.info(f"Step {step_name} outputs already exist in S3; skipping.")
        return
except ImportError:
    pass  # Fallback to local-only if checkpoint_utils not available

# 2. Run the step (existing logic)

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

## Benefits

1. **Resumability**: Steps can be interrupted and resumed without losing progress
2. **Idempotency**: Re-running a completed step skips work automatically
3. **Durability**: Outputs are persisted to S3, surviving instance termination
4. **Cross-instance**: Checkpoints work across different EC2 instances
5. **Efficiency**: Avoids redundant computation

## Next Steps

1. Complete S3 checkpoint implementation for Steps 5, 7, 8, and 9
2. Add checkpoint verification utilities
3. Add checkpoint cleanup utilities for failed runs
4. Add monitoring/alerting for checkpoint failures

