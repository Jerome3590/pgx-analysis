#!/usr/bin/env bash
# Build and deploy dashboard incrementally (works with available cohorts)
# This script can be run at any time - it will build with whatever cohorts are available

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

DASHBOARD_DIR="$PROJECT_ROOT/10_risk_dashboard"
cd "$DASHBOARD_DIR"

log "=========================================="
log "Building Dashboard (Incremental)"
log "=========================================="
log "This will build with whatever cohorts/age_bands are available"
log "Missing cohorts will be skipped gracefully"
log ""

# Check what cohorts/age_bands are available
log "Checking available models..."
AVAILABLE_COHORTS=()

if [ -d "$PROJECT_ROOT/6_final_model/outputs/opioid_ed" ]; then
    OPIOID_AGE_BANDS=$(find "$PROJECT_ROOT/6_final_model/outputs/opioid_ed" -mindepth 1 -maxdepth 1 -type d -name "*_*" | wc -l)
    if [ "$OPIOID_AGE_BANDS" -gt 0 ]; then
        AVAILABLE_COHORTS+=("opioid_ed")
        log "  ✓ opioid_ed: $OPIOID_AGE_BANDS age band(s) available"
    fi
fi

if [ -d "$PROJECT_ROOT/6_final_model/outputs/non_opioid_ed" ]; then
    NON_OPIOID_AGE_BANDS=$(find "$PROJECT_ROOT/6_final_model/outputs/non_opioid_ed" -mindepth 1 -maxdepth 1 -type d -name "*_*" | wc -l)
    if [ "$NON_OPIOID_AGE_BANDS" -gt 0 ]; then
        AVAILABLE_COHORTS+=("non_opioid_ed")
        log "  ✓ non_opioid_ed: $NON_OPIOID_AGE_BANDS age band(s) available"
    fi
fi

if [ ${#AVAILABLE_COHORTS[@]} -eq 0 ]; then
    warn "No completed cohorts found. Dashboard cannot be built yet."
    warn "Please complete Step 6 (Final Model Training) for at least one cohort/age_band."
    exit 1
fi

log ""
log "Preparing models for available cohorts..."

# Prepare models for each available cohort
for cohort in "${AVAILABLE_COHORTS[@]}"; do
    log "  Processing cohort: $cohort"
    python prepare_models.py --cohort "$cohort" || {
        warn "  Warning: Some age bands may be missing for $cohort"
    }
done

log ""
log "Generating metadata for available cohorts..."

# Generate metadata for each available cohort
for cohort in "${AVAILABLE_COHORTS[@]}"; do
    log "  Processing cohort: $cohort"
    python generate_metadata.py --cohort "$cohort" || {
        warn "  Warning: Some metadata may be missing for $cohort"
    }
done

log ""
log "=========================================="
log "Dashboard Preparation Complete"
log "=========================================="
log ""
log "Available cohorts: ${AVAILABLE_COHORTS[*]}"
log ""
log "Next steps:"
log "  1. Review models/ directory: $DASHBOARD_DIR/models/"
log "  2. Review metadata/ directory: $DASHBOARD_DIR/../10_results/metadata/"
log "  3. Build Docker image: cd $DASHBOARD_DIR && ./docker_build.sh"
log "  4. Deploy to AWS Lambda/ECR"
log ""
log "Note: Dashboard will work with available cohorts only."
log "      Missing cohorts will show appropriate error messages in the UI."

