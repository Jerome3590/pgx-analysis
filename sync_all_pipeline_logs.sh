#!/bin/bash
# Sync all pipeline logs from S3 to local logs/ directory for offline analysis
# Usage: bash sync_all_pipeline_logs.sh

set -e

S3_BUCKET=s3://pgx-repository
LOCAL_LOGS_DIR=logs

# Step log prefixes
LOG_PREFIXES=(
  4_model_data_log
  5_pgx_analysis_log
  6_final_model_log
  7_shap_analysis_log
  8_ffa_analysis_log
  9_dtw_log
  9_fpgrowth_log
  9_bupar_log
  9_cohort_pgx_log
)

mkdir -p "$LOCAL_LOGS_DIR"

for prefix in "${LOG_PREFIXES[@]}"; do
  echo "Syncing $S3_BUCKET/$prefix ..."
  aws s3 sync "$S3_BUCKET/$prefix" "$LOCAL_LOGS_DIR/$prefix" --no-sign-request || echo "Warning: $prefix may not exist or is incomplete."
done

echo "All available logs have been synced to $LOCAL_LOGS_DIR/"
