#!/usr/bin/env bash
# Fill missing gold/cohorts extracts for dashboard bands that already have
# DTW bin transitions from dtw_filter / model_events.
# One --cohort opioid_ed run writes both opioid_ed and non_opioid_ed partitions.
#
#   bash aws-pgx-setup/ec2/scripts/bash/run_ec2_analysis_session.sh \
#     --job-name "gold cohort gaps 55-64 85-114" -- \
#     bash utility_scripts/run_gold_cohort_gaps.sh
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PGX_DATA_ROOT="${PGX_DATA_ROOT:-/mnt/nvme}"
export HOME="${HOME:-/home/ec2-user}"

PY="${PY:-${HOME}/jupyter-env/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3.11 || command -v python3 || command -v python)"
fi

COHORT="${COHORT:-opioid_ed}"
YEARS=(2016 2017 2018 2019)
# Live S3 audit 2026-09-26: these bands have no gold/cohorts parquet.
AGE_BANDS=("55-64" "85-114")

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

for age in "${AGE_BANDS[@]}"; do
  for y in "${YEARS[@]}"; do
    echo "==== CREATE COHORT ${COHORT} ${age} ${y} $(date -u) ===="
    "$PY" "$REPO/2_create_cohort/0_create_cohort.py" \
      --cohort "$COHORT" \
      --age-band "$age" \
      --event-year "$y" \
      --concurrent-workers 1
  done
  echo "==== BIN TRANSITIONS ${COHORT} ${age} $(date -u) ===="
  "$PY" "$REPO/9_dashboard_visuals/dtw/create_bin_transitions.py" \
    --cohort "$COHORT" \
    --age-band "$age" \
    --force
  echo "==== BIN TRANSITIONS non_opioid_ed ${age} $(date -u) ===="
  "$PY" "$REPO/9_dashboard_visuals/dtw/create_bin_transitions.py" \
    --cohort non_opioid_ed \
    --age-band "$age" \
    --force
done

echo "==== JOB DONE $(date -u) ===="
