# S3 Path Migration Guide

This guide documents the S3 path updates to match the new folder structure and how to migrate existing artifacts.

## Summary of Changes

### Checkpoint Paths (s3://pgx-repository/pipeline_checkpoints/)

**Old → New:**
- `pipeline_checkpoints/5c_pgx_analysis/` → `pipeline_checkpoints/5_pgx_analysis/`
- `pipeline_checkpoints/6b_final_model_selection/` → `pipeline_checkpoints/6_final_model/`

### Output Paths (s3://pgxdatalake/gold/)

**PGx Features:**
- **Primary location**: `gold/pgx_features/{cohort}/{age_band}/` (new, preferred)
- **Legacy location**: `gold/feature_engineering/7_pgx/{cohort}/{age_band}/` (maintained for backward compatibility)

**Aggregated Feature Importance:**
- **Location**: `gold/feature_importance/{cohort}/{age_band}/` (no change)
- **File pattern**: `{cohort}_{age_band_fname}_aggregated_feature_importance.csv`

## Migration Steps

### 1. Run Migration Script (Dry Run First)

```bash
# Dry run to see what will be migrated
python utility_scripts/migrate_s3_paths.py

# Verify aggregated feature importance access
python utility_scripts/migrate_s3_paths.py --verify-fi

# Execute the migration
python utility_scripts/migrate_s3_paths.py --execute
```

### 2. Verify Migration

```bash
# Check S3 checkpoint and output status
python utility_scripts/check_s3_checkpoints.py
```

### 3. Update Code References

All code references have been updated to:
- Use `5_pgx_analysis` instead of `5c_pgx_analysis`
- Use `6_final_model` instead of `6b_final_model_selection`
- Check both `gold/pgx_features/` (primary) and `gold/feature_engineering/7_pgx/` (legacy) for PGx features

## What Gets Migrated

### Checkpoints
- Copies checkpoints from old step names to new step names
- Preserves original checkpoints (does not delete)
- Only copies if new checkpoint doesn't already exist

### PGx Features
- Script identifies files in legacy location (`gold/feature_engineering/7_pgx/`)
- Copies them to primary location (`gold/pgx_features/`)
- Maintains both locations for backward compatibility

### Aggregated Feature Importance
- Verification only (no migration needed)
- Confirms files are accessible at: `gold/feature_importance/{cohort}/{age_band}/`

## Code Updates

### Updated Files

1. **Checkpoint Utilities** (`py_helpers/checkpoint_utils.py`)
   - Uses `5_pgx_analysis` and `6_final_model` as step names

2. **PGx Analysis Scripts** (`5_pgx_analysis/`)
   - Upload to both primary and legacy S3 locations
   - Check primary location first, fallback to legacy

3. **Checkpoint Checker** (`utility_scripts/check_s3_checkpoints.py`)
   - Updated step definitions
   - Checks both primary and legacy locations for PGx features

4. **Workflow Scripts** (`utility_scripts/run_cohort_workflow.sh`)
   - References updated step names

## Backward Compatibility

- **PGx Features**: Code checks both `gold/pgx_features/` (primary) and `gold/feature_engineering/7_pgx/` (legacy)
- **Checkpoints**: Old checkpoints remain in place; new checkpoints use updated names
- **Feature Importance**: No changes to paths

## Troubleshooting

### Missing Aggregated Feature Importance Files

If verification shows missing files:
1. Run Step 3 (Feature Importance) to generate them:
   ```bash
   python 3_feature_importance/run_mc_feature_importance.py --cohort opioid_ed --age-band 13-24
   ```

### Checkpoint Migration Fails

- Check AWS credentials and permissions
- Verify S3 bucket access
- Check that old checkpoints exist before migrating

### PGx Features Not Found

- Code checks both primary and legacy locations
- If files are missing, re-run Step 5 (PGx Analysis):
   ```bash
   python 5_pgx_analysis/run_analysis.py --cohort-name opioid_ed --age-band 13-24
   ```

## Post-Migration Verification

After migration, verify:

1. **Checkpoints exist in new locations:**
   ```bash
   aws s3 ls s3://pgx-repository/pipeline_checkpoints/5_pgx_analysis/ --recursive
   aws s3 ls s3://pgx-repository/pipeline_checkpoints/6_final_model/ --recursive
   ```

2. **PGx features accessible:**
   ```bash
   aws s3 ls s3://pgxdatalake/gold/pgx_features/ --recursive
   ```

3. **Aggregated feature importance accessible:**
   ```bash
   aws s3 ls s3://pgxdatalake/gold/feature_importance/ --recursive
   ```

4. **Run workflow to verify end-to-end:**
   ```bash
   bash utility_scripts/run_cohort_workflow.sh opioid_ed 13-24
   ```

