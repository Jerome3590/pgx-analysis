#!/usr/bin/env bash
# Clear all step checkpoints for all cohorts/age bands

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
TIME_LOG_DIR="$PROJECT_ROOT/logs/time_tracking"

if [ ! -d "$TIME_LOG_DIR" ]; then
    echo "No checkpoint directory found: $TIME_LOG_DIR"
    echo "Nothing to clear."
    exit 0
fi

echo "Clearing all step checkpoints..."
echo "Checkpoint directory: $TIME_LOG_DIR"
echo ""

# Find all JSON checkpoint files
CHECKPOINT_FILES=$(find "$TIME_LOG_DIR" -name "*.json" -type f 2>/dev/null)

if [ -z "$CHECKPOINT_FILES" ]; then
    echo "No checkpoint files found."
    exit 0
fi

# Clear all step checkpoints
CLEARED_COUNT=0
for checkpoint_file in $CHECKPOINT_FILES; do
    cohort_age=$(basename "$checkpoint_file" .json)
    echo "Clearing checkpoints for: $cohort_age"
    
    python3 <<EOF
import json
import sys

try:
    with open('$checkpoint_file', 'r') as f:
        data = json.load(f)
    
    if 'step_times' in data:
        # Clear all step completion flags
        for step_key in data['step_times']:
            data['step_times'][step_key]['completed'] = False
        print(f"  Cleared {len(data['step_times'])} step checkpoints")
        CLEARED_COUNT += 1
    else:
        print(f"  No step checkpoints found")
    
    with open('$checkpoint_file', 'w') as f:
        json.dump(data, f, indent=2)
except Exception as e:
    print(f"  Error: {e}")
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        CLEARED_COUNT=$((CLEARED_COUNT + 1))
    fi
done

echo ""
echo "Cleared checkpoints for $CLEARED_COUNT cohort/age_band combinations"
echo "All steps will rerun on next workflow execution."
