#!/usr/bin/env bash
set -euo pipefail

##############################################
# Run Full Workflow for All Non-Opioid ED Cohorts
##############################################
#
# This script runs the complete workflow for all non_opioid_ed age bands:
# - 65-74
# - 75-84
# - 85-94
#
# Usage:
#   ./run_non_opioid_ed_workflow.sh [--skip-steps STEP1,STEP2]
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

# Check that run_cohort_workflow.sh exists
if [ ! -f "./run_cohort_workflow.sh" ]; then
    error "run_cohort_workflow.sh not found in $SCRIPT_DIR"
    exit 1
fi

if [ ! -x "./run_cohort_workflow.sh" ]; then
    warn "run_cohort_workflow.sh is not executable, making it executable..."
    chmod +x ./run_cohort_workflow.sh
fi

# Verify Python is available (early check)
detect_python() {
    local python_cmd=""
    if command -v python3 &> /dev/null; then
        python_cmd="python3"
    elif command -v python &> /dev/null; then
        python_cmd="python"
    fi
    
    # Check for virtual environments
    if [ -f "$HOME/jupyter-env/bin/python3" ]; then
        python_cmd="$HOME/jupyter-env/bin/python3"
    elif [ -n "${VIRTUAL_ENV:-}" ] && [ -f "$VIRTUAL_ENV/bin/python3" ]; then
        python_cmd="$VIRTUAL_ENV/bin/python3"
    fi
    
    if [ -z "$python_cmd" ] || ! $python_cmd --version &> /dev/null; then
        error "Python not found or not working. Please install Python 3 or activate your virtual environment."
        exit 1
    fi
    
    log "Python detected: $($python_cmd --version)"
}

detect_python

COHORT_NAME="non_opioid_ed"
AGE_BANDS=("65-74" "75-84" "85-94")

log "=========================================="
log "Running Non-Opioid ED Workflow"
log "=========================================="
log "Cohort: $COHORT_NAME"
log "Age Bands: ${AGE_BANDS[*]}"
log ""

# Parse skip-steps if provided
SKIP_ARGS=""
if [ $# -ge 1 ] && [ "$1" = "--skip-steps" ]; then
    if [ $# -ge 2 ]; then
        SKIP_ARGS="--skip-steps $2"
    fi
fi

FAILED=()
SUCCESS=()

for age_band in "${AGE_BANDS[@]}"; do
    log ""
    log "=========================================="
    log "Processing: $COHORT_NAME / $age_band"
    log "=========================================="
    
    if ./run_cohort_workflow.sh "$COHORT_NAME" "$age_band" $SKIP_ARGS; then
        SUCCESS+=("$age_band")
        log "✅ Completed: $COHORT_NAME / $age_band"
    else
        FAILED+=("$age_band")
        log "❌ Failed: $COHORT_NAME / $age_band"
    fi
done

log ""
log "=========================================="
log "Summary"
log "=========================================="
log "Successful: ${#SUCCESS[@]}"
for band in "${SUCCESS[@]}"; do
    log "  ✅ $band"
done

if [ ${#FAILED[@]} -gt 0 ]; then
    log ""
    log "Failed: ${#FAILED[@]}"
    for band in "${FAILED[@]}"; do
        log "  ❌ $band"
    done
    exit 1
else
    log ""
    log "✅ All age bands completed successfully!"
    exit 0
fi

