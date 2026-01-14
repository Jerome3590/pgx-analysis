#!/bin/bash
# Clear all Step 8 (FFA Analysis) outputs for all cohorts
# This allows workflows to restart at Step 8

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Detect Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    error "Python not found"
    exit 1
fi

# Check for EC2 jupyter-env
if [ -f "/home/pgx3874/jupyter-env/bin/python3.11" ]; then
    PYTHON_CMD="/home/pgx3874/jupyter-env/bin/python3.11"
fi

log "=========================================="
log "Clear All Step 8 (FFA Analysis) Outputs"
log "=========================================="
log ""

# Step 1: Clear all Step 8 outputs (local + S3)
log "Step 1: Clearing all Step 8 outputs (local files, S3 checkpoints, S3 outputs)..."
if [ "${1:-}" = "--dry-run" ]; then
    warn "DRY RUN MODE - No files will be deleted"
    $PYTHON_CMD utility_scripts/clear_step8_outputs.py --all-cohorts --dry-run
else
    $PYTHON_CMD utility_scripts/clear_step8_outputs.py --all-cohorts
fi
log ""

# Step 2: Clear Step 8 completion flags from time logs
log "Step 2: Clearing Step 8 completion flags from time logs..."
cleared_count=$($PYTHON_CMD -c "
import json
from pathlib import Path

log_dir = Path('logs/time_tracking')
cleared = 0
for log_file in log_dir.glob('*.json'):
    try:
        with open(log_file, 'r') as f:
            data = json.load(f)
        if 'step_8' in data.get('steps', {}):
            del data['steps']['step_8']
            with open(log_file, 'w') as f:
                json.dump(data, f, indent=2)
            cleared += 1
            print(f'  Cleared Step 8 from {log_file.name}')
    except Exception as e:
        print(f'  Error processing {log_file.name}: {e}')
print(cleared)
" 2>&1 | tail -1)

if [ "$cleared_count" -gt 0 ]; then
    log "Cleared Step 8 from $cleared_count time log files"
else
    warn "No Step 8 entries found in time logs (or error occurred)"
fi
log ""

log "=========================================="
log "Clear Complete!"
log "=========================================="
log ""
log "Next steps: Run each cohort in separate terminals:"
log ""
log "Terminal 1: ./utility_scripts/run_cohort_workflow.sh opioid_ed 13-24"
log "Terminal 2: ./utility_scripts/run_cohort_workflow.sh opioid_ed 25-44"
log "Terminal 3: ./utility_scripts/run_cohort_workflow.sh opioid_ed 45-54"
log "Terminal 4: ./utility_scripts/run_cohort_workflow.sh opioid_ed 55-64"
log "Terminal 5: ./utility_scripts/run_cohort_workflow.sh non_opioid_ed 65-74"
log "Terminal 6: ./utility_scripts/run_cohort_workflow.sh non_opioid_ed 75-84"
log "Terminal 7: ./utility_scripts/run_cohort_workflow.sh non_opioid_ed 85-94"
log ""
