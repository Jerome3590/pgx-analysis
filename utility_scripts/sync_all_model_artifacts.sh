#!/bin/bash
# Sync all model artifacts, input, and test data from S3 for offline analysis
# Usage: bash sync_all_model_artifacts.sh

set -e

S3_BUCKET=s3://pgx-repository
LOCAL_DATA_DIR=data_offline

# Model artifact and data prefixes (add/adjust as needed)
PREFIXES=(
  4_model_data_artifacts
  5_pgx_analysis_artifacts
  6_final_model_artifacts
  7_shap_analysis_artifacts
  8_ffa_analysis_artifacts
  9_dtw_artifacts
  9_fpgrowth_artifacts
  9_bupar_artifacts
  9_cohort_pgx_artifacts
  gold/cohorts
  gold/cohorts_model_data
  gold/cohorts_F1120
  gold/input_data
  gold/test_data
)

mkdir -p "$LOCAL_DATA_DIR"

for prefix in "${PREFIXES[@]}"; do
  echo "Syncing $S3_BUCKET/$prefix ..."
  aws s3 sync "$S3_BUCKET/$prefix" "$LOCAL_DATA_DIR/$prefix" --no-sign-request || echo "Warning: $prefix may not exist or is incomplete."
done

echo "All available model artifacts and data have been synced to $LOCAL_DATA_DIR/"
