#!/usr/bin/env bash
# Orchestrate CH4 util-free sensitivity on EC2 with SES emails + stop.
# Steps: check model data → (if needed) sync gold → cohorts → model data → sensitivity → stop.
set -uo pipefail

INSTANCE_ID="${INSTANCE_ID:-i-0e7d1bd469620c0bb}"
REPO="${REPO:-/home/pgx3874/pgx-analysis}"
PY="${PY:-/home/pgx3874/jupyter-env/bin/python3.11}"
LOG_DIR="${LOG_DIR:-/mnt/nvme/pgx-analysis/logs}"
NVME="${NVME:-/mnt/nvme}"
COHORT="non_opioid_ed"
AGE_BANDS=(0-12 13-24 25-44 45-54 55-64 65-74 75-84 85-114)
EVENT_YEARS=(2016 2017 2018 2019)

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/sensitivity_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

cd "$REPO" || exit 1
export PYTHONPATH="$REPO"
export PGX_DATA_ROOT="$NVME"
export HOME=/home/pgx3874
umask 022

email() {
  local subject="$1"
  local body="$2"
  SUBJECT="$subject" BODY="$body" "$PY" - <<'PY'
import os
from py_helpers.aws_utils import send_status_email_ses
ok = send_status_email_ses(os.environ["SUBJECT"], os.environ["BODY"])
print("SES:", ok, flush=True)
PY
}

model_events_path() {
  echo "$NVME/4_model_data/cohort_name=${COHORT}/age_band=$1/model_events.parquet"
}

list_missing_bands() {
  local b
  for b in "${AGE_BANDS[@]}"; do
    local p
    p="$(model_events_path "$b")"
    if [[ ! -s "$p" ]]; then
      printf '%s\n' "$b"
    fi
  done
}

# Physical gold medical age bands needed for a logical band (85-114 joins 85-94+95-114).
physical_bands_for() {
  case "$1" in
    85-114) printf '%s\n' 85-94 95-114 ;;
    *) printf '%s\n' "$1" ;;
  esac
}

echo "==== PIPELINE START $(date -u) ===="
echo "Repo=$(git -C "$REPO" rev-parse --short HEAD) LOG=$LOG"

mapfile -t MISSING < <(list_missing_bands)
MISS_STR="${MISSING[*]:-}"
echo "Missing model_events bands: '${MISS_STR:-none}'"

if [[ ${#MISSING[@]} -eq 0 ]]; then
  email "[pgx-analysis-1a] STEP2 COMPLETE: model data available" \
"Instance: ${INSTANCE_ID}
All modeled age bands have model_events under ${NVME}/4_model_data/cohort_name=${COHORT}/.
Skipping cohort + model-data rebuild.
Next: run sensitivity analysis.
Log: ${LOG}"
else
  email "[pgx-analysis-1a] STEP2 COMPLETE: model data INCOMPLETE — rebuild required" \
"Instance: ${INSTANCE_ID}
Missing model_events for age bands: ${MISS_STR}
Present bands will be reused.
Next: sync gold medical/pharmacy (+ FI) for missing bands, then create cohorts + model data.
Log: ${LOG}"

  echo "==== SYNC gold medical/pharmacy for missing (+ FI) ===="
  mkdir -p "$NVME/gold/medical" "$NVME/gold/pharmacy" "$NVME/gold/feature_importance" \
           "$NVME/gold/cohorts" "$NVME/4_model_data" "$NVME/duckdb_tmp"

  # Collect unique physical age bands to sync
  declare -A NEED_PHYS=()
  for b in "${MISSING[@]}"; do
    while read -r pb; do
      NEED_PHYS["$pb"]=1
    done < <(physical_bands_for "$b")
  done

  for pb in "${!NEED_PHYS[@]}"; do
    echo "---- sync medical age_band=${pb} ----"
    aws s3 sync "s3://pgxdatalake/gold/medical/age_band=${pb}/" \
      "$NVME/gold/medical/age_band=${pb}/" --only-show-errors
    echo "---- sync pharmacy age_band=${pb} ----"
    aws s3 sync "s3://pgxdatalake/gold/pharmacy/age_band=${pb}/" \
      "$NVME/gold/pharmacy/age_band=${pb}/" --only-show-errors
  done

  aws s3 sync "s3://pgxdatalake/gold/feature_importance/${COHORT}/" \
    "$NVME/gold/feature_importance/${COHORT}/" --only-show-errors

  # FI layout for create_model_data: prefer /mnt/nvme/gold/feature_importance/{cohort}/{age_band}/
  # S3 keys already use hyphen age bands under non_opioid_ed/{age_band}/

  email "[pgx-analysis-1a] STEP2b COMPLETE: gold inputs synced to NVMe" \
"Synced gold medical/pharmacy for physical bands: ${!NEED_PHYS[*]}
Also synced gold/feature_importance/${COHORT}
df:
$(df -h ${NVME} | tail -1)
Next: create cohorts for missing age bands: ${MISS_STR}
Log: ${LOG}"

  echo "==== CREATE COHORTS (ed_non_opioid) for missing bands ===="
  for b in "${MISSING[@]}"; do
    for y in "${EVENT_YEARS[@]}"; do
      echo "---- cohort ed_non_opioid age_band=${b} event_year=${y} $(date -u) ----"
      if ! "$PY" "$REPO/2_create_cohort/0_create_cohort.py" \
          --age-band "$b" \
          --event-year "$y" \
          --cohort ed_non_opioid; then
        email "[pgx-analysis-1a] ERROR: cohort create failed ${b} ${y}" "See ${LOG}"
        exit 1
      fi
    done
  done

  email "[pgx-analysis-1a] STEP3 COMPLETE: cohorts created for missing bands" \
"Created ed_non_opioid cohorts for: ${MISS_STR}
Years: ${EVENT_YEARS[*]}
Next: create model data (Step 4).
Log: ${LOG}"

  echo "==== CREATE MODEL DATA for missing bands ===="
  for b in "${MISSING[@]}"; do
    echo "---- create_model_data ${COHORT} ${b} $(date -u) ----"
    if ! "$PY" "$REPO/4_model_data/create_model_data.py" \
        --cohort "$COHORT" \
        --age-band "$b"; then
      email "[pgx-analysis-1a] ERROR: model data failed ${b}" "See ${LOG}"
      exit 1
    fi
  done

  email "[pgx-analysis-1a] STEP4 COMPLETE: model data created" \
"model_events written for: ${MISS_STR}
Next: run sensitivity analysis (all age bands).
Log: ${LOG}"
fi

mapfile -t MISSING2 < <(list_missing_bands)
if [[ ${#MISSING2[@]} -gt 0 ]]; then
  email "[pgx-analysis-1a] ERROR: model data still missing before sensitivity" \
"Still missing: ${MISSING2[*]}
Log: ${LOG}"
  exit 1
fi

echo "==== RUN SENSITIVITY $(date -u) ===="
if ! "$PY" "$REPO/6_final_model/run_sensitivity_util_free.py"; then
  email "[pgx-analysis-1a] ERROR: sensitivity analysis failed" "See ${LOG}"
  exit 1
fi

SUMMARY="$REPO/manuscript/data/supplementary/ch04_util_free_sensitivity/sensitivity_summary_all_bands.json"
SUMMARY_NOTE="(summary file missing)"
if [[ -f "$SUMMARY" ]]; then
  SUMMARY_NOTE="$("$PY" -c "import json; p=json.load(open(r'$SUMMARY')); print(p if not isinstance(p, dict) else 'keys='+ ','.join(sorted(p.keys())))" 2>/dev/null || echo present)"
fi

email "[pgx-analysis-1a] STEP5 COMPLETE: sensitivity analysis OK" \
"Sensitivity finished with no errors.
Summary path: ${SUMMARY}
Summary: ${SUMMARY_NOTE}
Next: stop EC2 instance ${INSTANCE_ID}.
Log: ${LOG}"

echo "==== STOP INSTANCE $(date -u) ===="
aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region us-east-1
sleep 8
STATE="$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region us-east-1 \
  --query 'Reservations[0].Instances[0].State.Name' --output text || echo unknown)"

email "[pgx-analysis-1a] FINAL: sensitivity OK + EC2 shutdown initiated" \
"Sensitivity analysis completed with no errors.
EC2 stop-instances issued for ${INSTANCE_ID}.
State shortly after stop: ${STATE}
Repo commit: $(git -C "$REPO" rev-parse --short HEAD)
Log: ${LOG}
Summary artifact: ${SUMMARY}"

echo "==== PIPELINE DONE $(date -u) state=${STATE} ===="
