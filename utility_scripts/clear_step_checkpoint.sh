#!/usr/bin/env bash
# Clear checkpoint for a specific step to force rerun

if [ $# -lt 3 ]; then
    echo "Usage: $0 <cohort_name> <age_band> <step_number>"
    echo "Example: $0 opioid_ed 13-24 3b"
    exit 1
fi

COHORT_NAME="$1"
AGE_BAND="$2"
STEP_NUM="$3"
AGE_BAND_FNAME=$(echo "$AGE_BAND" | tr '-' '_')

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
TIME_LOG_FILE="$PROJECT_ROOT/logs/time_tracking/${COHORT_NAME}_${AGE_BAND_FNAME}.json"

if [ ! -f "$TIME_LOG_FILE" ]; then
    echo "Checkpoint file not found: $TIME_LOG_FILE"
    exit 1
fi

python3 <<EOF
import json
with open('$TIME_LOG_FILE', 'r') as f:
    data = json.load(f)

if 'step_times' in data and '$STEP_NUM' in data['step_times']:
    data['step_times']['$STEP_NUM']['completed'] = False
    print(f"Cleared checkpoint for Step $STEP_NUM")
else:
    print(f"Step $STEP_NUM checkpoint not found (may not have run yet)")

with open('$TIME_LOG_FILE', 'w') as f:
    json.dump(data, f, indent=2)
EOF
