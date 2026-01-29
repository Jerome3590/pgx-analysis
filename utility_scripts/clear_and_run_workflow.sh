#!/usr/bin/env bash
# Clear all models and checkpoints, then run workflow
# Usage: ./clear_and_run_workflow.sh <cohort_name> <age_band> [--skip-steps STEP1,STEP2]

set -euo pipefail

##############################################
# Clear All Models and Checkpoints, Then Run Workflow
##############################################
#
# This script:
# 1. Clears all model outputs (Steps 6, 7, 8)
# 2. Clears all checkpoints (local and S3)
# 3. Runs the full workflow
#
# Usage:
#   ./clear_and_run_workflow.sh <cohort_name> <age_band> [--skip-steps STEP1,STEP2]
#
# Examples:
#   ./clear_and_run_workflow.sh opioid_ed 13-24
#   ./clear_and_run_workflow.sh non_opioid_ed 65-74 --skip-steps 5d

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

# Parse arguments
if [ $# -lt 2 ]; then
    error "Usage: $0 <cohort_name> <age_band> [--skip-steps STEP1,STEP2]"
    error "  Cohorts: opioid_ed, non_opioid_ed"
    error "  Age bands:"
    error "    opioid_ed: 13-24, 25-44, 45-54, 55-64"
    error "    non_opioid_ed: 65-74, 75-84, 85-94"
    exit 1
fi

COHORT_NAME="$1"
AGE_BAND="$2"
SKIP_STEPS=""

# Parse skip-steps if provided
if [ $# -ge 3 ] && [ "$3" = "--skip-steps" ]; then
    if [ $# -ge 4 ]; then
        SKIP_STEPS="$4"
    else
        error "--skip-steps requires a value"
        exit 1
    fi
fi

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

log "=========================================="
log "Clear and Run Workflow"
log "=========================================="
log "Cohort: $COHORT_NAME"
log "Age Band: $AGE_BAND"
if [ -n "$SKIP_STEPS" ]; then
    log "Skip Steps: $SKIP_STEPS"
fi
log ""

# Step 1: Clear all models (Steps 6, 7, 8)
log "Step 1: Clearing all model outputs (Steps 6, 7, 8)..."
if [ -f "utility_scripts/clear_models.sh" ]; then
    if bash utility_scripts/clear_models.sh --cohort "$COHORT_NAME" --age-band "$AGE_BAND" --s3; then
        log "✓ Models cleared"
    else
        warn "Model clearing had issues (continuing anyway)"
    fi
else
    warn "clear_models.sh not found, skipping model clearing"
fi
log ""

# Step 2: Clear local checkpoints
log "Step 2: Clearing local checkpoints..."
if [ -f "utility_scripts/clear_all_checkpoints.py" ]; then
    if python3 utility_scripts/clear_all_checkpoints.py 2>/dev/null; then
        log "✓ Local checkpoints cleared"
    else
        warn "Local checkpoint clearing had issues (may not exist, continuing anyway)"
    fi
else
    warn "clear_all_checkpoints.py not found, skipping local checkpoint clearing"
fi
log ""

# Step 3: Clear S3 checkpoints
log "Step 3: Clearing S3 checkpoints..."
if [ -f "utility_scripts/clear_s3_checkpoints.py" ]; then
    # Clear checkpoints for this specific cohort/age_band
    if python3 utility_scripts/clear_s3_checkpoints.py --cohort "$COHORT_NAME" --age-band "$AGE_BAND" <<< "yes" 2>/dev/null; then
        log "✓ S3 checkpoints cleared for $COHORT_NAME/$AGE_BAND"
    else
        warn "S3 checkpoint clearing had issues (continuing anyway)"
    fi
else
    warn "clear_s3_checkpoints.py not found, skipping S3 checkpoint clearing"
fi
log ""

# Step 4: Run the workflow
log "=========================================="
log "Step 4: Running workflow..."
log "=========================================="
log ""

if [ -n "$SKIP_STEPS" ]; then
    bash utility_scripts/run_cohort_workflow.sh "$COHORT_NAME" "$AGE_BAND" --skip-steps "$SKIP_STEPS"
else
    bash utility_scripts/run_cohort_workflow.sh "$COHORT_NAME" "$AGE_BAND"
fi

log ""
log "=========================================="
log "Workflow Complete"
log "=========================================="
