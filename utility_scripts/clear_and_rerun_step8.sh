#!/bin/bash
# Clear all Step 8 outputs and rerun workflow starting at Step 8

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
log "Clear All Step 8 Outputs and Rerun Workflow"
log "=========================================="
log ""

# Step 1: Clear all Step 8 outputs
log "Step 1: Clearing all Step 8 outputs..."
if [ "${1:-}" = "--dry-run" ]; then
    warn "DRY RUN MODE - No files will be deleted"
    $PYTHON_CMD utility_scripts/clear_step8_outputs.py --all-cohorts --dry-run
else
    $PYTHON_CMD utility_scripts/clear_step8_outputs.py --all-cohorts
fi
log ""

# Step 2: Clear Step 8 completion flags from time logs
log "Step 2: Clearing Step 8 completion flags from time logs..."
$PYTHON_CMD -c "
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
print(f'Cleared Step 8 from {cleared} time log files')
"
log ""

# Step 3: Run workflow for all cohorts
log "Step 3: Running workflow for all cohorts (will start at Step 8)..."
log ""

COHORTS=(
    "opioid_ed:13-24"
    "opioid_ed:25-44"
    "opioid_ed:45-54"
    "opioid_ed:55-64"
    "non_opioid_ed:65-74"
    "non_opioid_ed:75-84"
    "non_opioid_ed:85-94"
)

for combo in "${COHORTS[@]}"; do
    cohort=$(echo "$combo" | cut -d':' -f1)
    age_band=$(echo "$combo" | cut -d':' -f2)
    
    log "Running workflow for $cohort/$age_band..."
    if [ "${1:-}" = "--dry-run" ]; then
        warn "  [DRY RUN] Would run: ./utility_scripts/run_cohort_workflow.sh $cohort $age_band"
    else
        ./utility_scripts/run_cohort_workflow.sh "$cohort" "$age_band" || {
            error "Failed for $cohort/$age_band"
            warn "Continuing with other cohorts..."
        }
    fi
    log ""
done

log "=========================================="
log "Complete!"
log "=========================================="
