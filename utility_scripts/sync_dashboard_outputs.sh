#!/bin/bash
# Sync dashboard outputs and visualizations for offline manuscript work
# Usage: bash sync_dashboard_outputs.sh

set -e

S3_BUCKET=s3://pgx-repository
LOCAL_DASHBOARD_DIR=dashboard_offline

# Add/adjust prefixes as needed for your outputs and visualizations
PREFIXES=(
  10_risk_dashboard/outputs
  10_risk_dashboard/visualizations
  10_risk_dashboard/frontend
  10_risk_dashboard/backend
)

mkdir -p "$LOCAL_DASHBOARD_DIR"

for prefix in "${PREFIXES[@]}"; do
  echo "Syncing $S3_BUCKET/$prefix ..."
  aws s3 sync "$S3_BUCKET/$prefix" "$LOCAL_DASHBOARD_DIR/$prefix" --no-sign-request || echo "Warning: $prefix may not exist or is incomplete."
done

echo "All dashboard outputs and visualizations have been synced to $LOCAL_DASHBOARD_DIR/"
