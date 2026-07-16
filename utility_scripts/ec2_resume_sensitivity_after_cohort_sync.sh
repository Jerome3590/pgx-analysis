#!/usr/bin/env bash
# Resume after Step 3: sync cohorts from S3 → NVMe, rebuild missing model_events, sensitivity, stop.
set -uo pipefail

INSTANCE_ID="${INSTANCE_ID:-i-0e7d1bd469620c0bb}"
REPO="${REPO:-/home/pgx3874/pgx-analysis}"
PY="${PY:-/home/pgx3874/jupyter-env/bin/python3.11}"
LOG_DIR="${LOG_DIR:-/mnt/nvme/pgx-analysis/logs}"
NVME="${NVME:-/mnt/nvme}"
COHORT="non_opioid_ed"
AGE_BANDS=(0-12 13-24 25-44 45-54 55-64 65-74 75-84 85-114)

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/sensitivity_resume_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

cd "$REPO" || exit 1
export PYTHONPATH="$REPO"
export PGX_DATA_ROOT="$NVME"
export HOME=/home/pgx3874

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

echo "==== RESUME START $(date -u) ===="
echo "Repo=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) LOG=$LOG"

mapfile -t MISSING < <(list_missing_bands)
MISS_STR="${MISSING[*]:-}"
echo "Missing model_events: '${MISS_STR:-none}'"

email "[pgx-analysis-1a] RESUME: sync cohorts + rebuild model data" \
"Prior run failed Step 4: cohorts existed on S3 but not on /mnt/nvme/gold/cohorts.
Missing model_events bands: ${MISS_STR:-none}
Next: aws s3 sync gold/cohorts → NVMe, create_model_data (verify parquet), sensitivity, stop.
Log: ${LOG}"

echo "==== SYNC gold/cohorts (non_opioid_ed) from S3 ===="
mkdir -p "$NVME/gold/cohorts"
aws s3 sync "s3://pgxdatalake/gold/cohorts/cohort_name=${COHORT}/" \
  "$NVME/gold/cohorts/cohort_name=${COHORT}/" --only-show-errors
# Also pull opioid_ed if present (create_model_data may inspect sibling paths)
aws s3 sync "s3://pgxdatalake/gold/cohorts/cohort_name=opioid_ed/" \
  "$NVME/gold/cohorts/cohort_name=opioid_ed/" --only-show-errors || true

echo "Local cohort parquet count:"
find "$NVME/gold/cohorts/cohort_name=${COHORT}" -name 'cohort.parquet' 2>/dev/null | wc -l

# Spot-check one path
sample="$NVME/gold/cohorts/cohort_name=${COHORT}/event_year=2019/age_band=75-84/cohort.parquet"
if [[ ! -s "$sample" ]]; then
  email "[pgx-analysis-1a] ERROR: cohort sync incomplete" \
"Expected ${sample} after S3 sync.
Log: ${LOG}"
  exit 1
fi

email "[pgx-analysis-1a] RESUME: cohorts synced to NVMe" \
"Synced s3://pgxdatalake/gold/cohorts/cohort_name=${COHORT}/ → ${NVME}/gold/cohorts/
Sample OK: ${sample}
Next: create_model_data for: ${MISS_STR}
Log: ${LOG}"

if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "==== CREATE MODEL DATA (verified) ===="
  for b in "${MISSING[@]}"; do
    echo "---- create_model_data ${COHORT} ${b} $(date -u) ----"
    if ! "$PY" "$REPO/4_model_data/create_model_data.py" \
        --cohort "$COHORT" \
        --age-band "$b"; then
      email "[pgx-analysis-1a] ERROR: model data failed ${b}" "See ${LOG}"
      exit 1
    fi
    out="$(model_events_path "$b")"
    if [[ ! -s "$out" ]]; then
      email "[pgx-analysis-1a] ERROR: model_events missing after create_model_data ${b}" \
"Expected non-empty: ${out}
create_model_data exited 0 but wrote nothing (cohort path / FI issue).
Log: ${LOG}"
      exit 1
    fi
    echo "OK model_events: $(ls -lh "$out")"
  done

  email "[pgx-analysis-1a] STEP4 COMPLETE: model data created (resume)" \
"model_events verified for: ${MISS_STR}
Next: sensitivity analysis (all age bands).
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
  SUMMARY_NOTE="$("$PY" -c "import json; p=json.load(open(r'$SUMMARY')); print(p if not isinstance(p, dict) else 'keys='+','.join(sorted(map(str,p.keys()))))" 2>/dev/null || echo present)"
fi

email "[pgx-analysis-1a] STEP5 COMPLETE: sensitivity analysis OK" \
"Sensitivity finished with no errors.
Summary: ${SUMMARY}
${SUMMARY_NOTE}
Next: stop EC2 ${INSTANCE_ID}.
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
Log: ${LOG}
Summary: ${SUMMARY}"

echo "==== RESUME DONE $(date -u) state=${STATE} ===="
