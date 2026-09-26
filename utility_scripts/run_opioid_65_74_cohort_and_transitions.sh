#!/usr/bin/env bash
# One-band job for the missing opioid_ed / 65-74 DTW bin transitions.
# Builds gold/cohorts only (2016-2019). Does not create model_events.
#
#   bash aws-pgx-setup/ec2/scripts/bash/run_ec2_analysis_session.sh \
#     --job-name "opioid_ed 65-74 gold cohorts + bin transitions" -- \
#     bash utility_scripts/run_opioid_65_74_cohort_and_transitions.sh
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PGX_DATA_ROOT="${PGX_DATA_ROOT:-/mnt/nvme}"
export HOME="${HOME:-/home/pgx3874}"

PY="${PY:-${HOME}/jupyter-env/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3.11 || command -v python3 || command -v python)"
fi

COHORT="opioid_ed"
AGE_BAND="65-74"
YEARS=(2016 2017 2018 2019)

MOUNT_SH="${MOUNT_NVME_SH:-$REPO/aws-pgx-setup/ec2/scripts/bash/mount_nvme.sh}"
if [[ -f "$MOUNT_SH" ]]; then
  bash "$MOUNT_SH"
else
  echo "WARN: $MOUNT_SH missing; sudo mkdir only (no format)"
  sudo mkdir -p /mnt/nvme
  sudo chown -R "$(id -u)":"$(id -g)" /mnt/nvme || true
fi
mkdir -p "$PGX_DATA_ROOT/gold/cohorts" "$PGX_DATA_ROOT/duckdb_tmp" \
         "$PGX_DATA_ROOT/pgx-analysis/logs"
df -h /mnt/nvme || df -h /

for y in "${YEARS[@]}"; do
  echo "==== CREATE COHORT ${COHORT} ${AGE_BAND} ${y} $(date -u) ===="
  "$PY" "$REPO/2_create_cohort/0_create_cohort.py" \
    --cohort "$COHORT" \
    --age-band "$AGE_BAND" \
    --event-year "$y" \
    --concurrent-workers 1
done

echo "==== BIN TRANSITIONS ${COHORT} ${AGE_BAND} $(date -u) ===="
"$PY" "$REPO/9_dashboard_visuals/dtw/create_bin_transitions.py" \
  --cohort "$COHORT" \
  --age-band "$AGE_BAND" \
  --force

echo "==== JOB DONE $(date -u) ===="
