#!/usr/bin/env bash
# Idempotent Step 6 upload to S3 (explicit keys via upload_file_to_s3 — no aws s3 sync).
#
# Usage (EC2, from repo root):
#   chmod +x utility_scripts/sync_step6_final_model_to_s3.sh
#   PGX_REPO_ROOT=/path/to/pgx-analysis ./utility_scripts/sync_step6_final_model_to_s3.sh
#
# Optional args are passed through to the Python module, e.g.:
#   ./utility_scripts/sync_step6_final_model_to_s3.sh --cohort opioid_ed --age-band 13-24
#   ./utility_scripts/sync_step6_final_model_to_s3.sh --all

set -euo pipefail
REPO="${PGX_REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export PGX_REPO_ROOT="${PGX_REPO_ROOT:-$REPO}"
PY="${PGX_PYTHON:-$(command -v python3 || command -v python)}"
cd "$REPO"
# No args → upload every cohort/age_band found under 6_final_model/outputs
if [[ $# -eq 0 ]]; then
  exec "$PY" -m py_helpers.final_model_s3_upload --all
fi
exec "$PY" -m py_helpers.final_model_s3_upload "$@"
