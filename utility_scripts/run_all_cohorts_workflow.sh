#!/usr/bin/env bash
set -euo pipefail

##############################################
# Run Full Workflow for All Cohorts
##############################################
#
# This script runs the complete workflow for all cohorts and age bands:
# - opioid_ed: 13-24, 25-44, 45-54, 55-64
# - non_opioid_ed: 65-74, 75-84, 85-94
#
# Usage:
#   ./run_all_cohorts_workflow.sh [--skip-steps STEP1,STEP2]
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

# Check that required scripts exist
for script in run_opioid_ed_workflow.sh run_non_opioid_ed_workflow.sh; do
    if [ ! -f "./$script" ]; then
        error "$script not found in $SCRIPT_DIR"
        exit 1
    fi
    if [ ! -x "./$script" ]; then
        warn "$script is not executable, making it executable..."
        chmod +x "./$script"
    fi
done

# Verify Python is available (early check)
detect_python() {
    local python_cmd=""
    if command -v python3 &> /dev/null; then
        python_cmd="python3"
    elif command -v python &> /dev/null; then
        python_cmd="python"
    fi
    
    # Check for virtual environments (EC2 path first)
    if [ -f "/home/pgx3874/jupyter-env/bin/python3.11" ]; then
        python_cmd="/home/pgx3874/jupyter-env/bin/python3.11"
    elif [ -f "$HOME/jupyter-env/bin/python3" ]; then
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

log "=========================================="
log "Running Complete Workflow for All Cohorts"
log "=========================================="
log ""

# Parse skip-steps if provided
SKIP_ARGS=""
if [ $# -ge 1 ] && [ "$1" = "--skip-steps" ]; then
    if [ $# -ge 2 ]; then
        SKIP_ARGS="--skip-steps $2"
    fi
fi

# Run opioid_ed workflow
log "Starting Opioid ED workflow..."
if ./run_opioid_ed_workflow.sh $SKIP_ARGS; then
    log "✅ Opioid ED workflow completed"
else
    log "❌ Opioid ED workflow failed"
    exit 1
fi

log ""
log "=========================================="
log ""

# Run non_opioid_ed workflow
log "Starting Non-Opioid ED workflow..."
if ./run_non_opioid_ed_workflow.sh $SKIP_ARGS; then
    log "✅ Non-Opioid ED workflow completed"
else
    log "❌ Non-Opioid ED workflow failed"
    exit 1
fi

log ""
log "=========================================="
log "✅ All workflows completed successfully!"
log "=========================================="

