#!/usr/bin/env bash
# Build and deploy dashboard (requires all cohorts and age bands)
# This script requires all cohorts to be completed before building

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

DASHBOARD_DIR="$PROJECT_ROOT/10_risk_dashboard"
cd "$DASHBOARD_DIR"

# Required cohorts and age bands
REQUIRED_COHORTS=("opioid_ed" "non_opioid_ed")
OPIOID_ED_AGE_BANDS=("13-24" "25-44" "45-54" "55-64")
NON_OPIOID_ED_AGE_BANDS=("65-74" "75-84" "85-94")

log "=========================================="
log "Building Dashboard (All Cohorts Required)"
log "=========================================="
log "This requires all cohorts and age bands to be completed"
log ""

# Check what cohorts/age_bands are available
log "Checking required models..."
MISSING_COHORTS=()
MISSING_AGE_BANDS=()

# Check opioid_ed cohort
if [ ! -d "$PROJECT_ROOT/6_final_model/outputs/opioid_ed" ]; then
    MISSING_COHORTS+=("opioid_ed")
    error "  ✗ opioid_ed: cohort directory not found"
else
    log "  ✓ opioid_ed: cohort directory found"
    for age_band in "${OPIOID_ED_AGE_BANDS[@]}"; do
        age_band_fname=$(echo "$age_band" | tr '-' '_')
        if [ ! -d "$PROJECT_ROOT/6_final_model/outputs/opioid_ed/$age_band_fname" ]; then
            MISSING_AGE_BANDS+=("opioid_ed/$age_band")
            error "    ✗ opioid_ed/$age_band: missing"
        else
            log "    ✓ opioid_ed/$age_band: found"
        fi
    done
fi

# Check non_opioid_ed cohort
if [ ! -d "$PROJECT_ROOT/6_final_model/outputs/non_opioid_ed" ]; then
    MISSING_COHORTS+=("non_opioid_ed")
    error "  ✗ non_opioid_ed: cohort directory not found"
else
    log "  ✓ non_opioid_ed: cohort directory found"
    for age_band in "${NON_OPIOID_ED_AGE_BANDS[@]}"; do
        age_band_fname=$(echo "$age_band" | tr '-' '_')
        if [ ! -d "$PROJECT_ROOT/6_final_model/outputs/non_opioid_ed/$age_band_fname" ]; then
            MISSING_AGE_BANDS+=("non_opioid_ed/$age_band")
            error "    ✗ non_opioid_ed/$age_band: missing"
        else
            log "    ✓ non_opioid_ed/$age_band: found"
        fi
    done
fi

# Fail if any cohorts or age bands are missing
if [ ${#MISSING_COHORTS[@]} -gt 0 ] || [ ${#MISSING_AGE_BANDS[@]} -gt 0 ]; then
    error ""
    error "=========================================="
    error "Dashboard build failed - missing requirements"
    error "=========================================="
    
    if [ ${#MISSING_COHORTS[@]} -gt 0 ]; then
        error "Missing cohorts:"
        for cohort in "${MISSING_COHORTS[@]}"; do
            error "  - $cohort"
        done
    fi
    
    if [ ${#MISSING_AGE_BANDS[@]} -gt 0 ]; then
        error "Missing age bands:"
        for age_band in "${MISSING_AGE_BANDS[@]}"; do
            error "  - $age_band"
        done
    fi
    
    error ""
    error "Please complete Step 6 (Final Model Training) for all required cohorts/age_bands."
    error "Required:"
    error "  - opioid_ed: ${OPIOID_ED_AGE_BANDS[*]}"
    error "  - non_opioid_ed: ${NON_OPIOID_ED_AGE_BANDS[*]}"
    exit 1
fi

log ""
log "✓ All required cohorts and age bands are present"

log ""
log "Preparing models for all cohorts..."

# Prepare models for each required cohort
for cohort in "${REQUIRED_COHORTS[@]}"; do
    log "  Processing cohort: $cohort"
    if ! python prepare_models.py --cohort "$cohort"; then
        error "  Failed to prepare models for $cohort"
        exit 1
    fi
done

log ""
log "Generating metadata for all cohorts..."

# Generate metadata for each required cohort
for cohort in "${REQUIRED_COHORTS[@]}"; do
    log "  Processing cohort: $cohort"
    if ! python generate_metadata.py --cohort "$cohort"; then
        error "  Failed to generate metadata for $cohort"
        exit 1
    fi
done

log ""
log "=========================================="
log "Dashboard Preparation Complete"
log "=========================================="
log ""
log "All cohorts processed: ${REQUIRED_COHORTS[*]}"
log ""
log "Next steps:"
log "  1. Review models/ directory: $DASHBOARD_DIR/models/"
log "  2. Review metadata/ directory: $DASHBOARD_DIR/../10_results/metadata/"
log "  3. Build Docker image: cd $DASHBOARD_DIR && ./docker_build.sh"
log "  4. Deploy to AWS Lambda/ECR"

