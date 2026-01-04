#!/usr/bin/env bash
set -euo pipefail

##############################################
# Run Full Workflow for a Single Cohort/Age Band
##############################################
#
# Usage:
#   ./run_cohort_workflow.sh <cohort_name> <age_band> [--skip-steps STEP1,STEP2]
#
# Examples:
#   ./run_cohort_workflow.sh opioid_ed 13-24
#   ./run_cohort_workflow.sh non_opioid_ed 65-74 --skip-steps 5d
#
# Steps:
#   3: Feature Importance (check for completed aggregated feature importances)
#   4a: Model Data Creation (generate model_events.parquet with cases + controls)
#   4b: DTW Protocol Filtering (administrative/scheduling/non-medical codes, keep all surgeries)
#   5c: PGx Feature Engineering (only feature engineering step)
#   6: Final Model Training (use aggregated features + PGx, no encoding, select best by recall/AUC-PR)
#   7: FFA Analysis (use best XGBoost model JSON)
#   8: SHAP Analysis (use best CatBoost model binary)
#   9: Combined SHAP + FFA
#   10: Risk Dashboard (BupaR/DTW/FP-Growth visualizations + causal analysis)
#   11: Deploy to S3/AWS Lambda
#

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
    fi
fi

# Validate cohort
if [[ ! "$COHORT_NAME" =~ ^(opioid_ed|non_opioid_ed)$ ]]; then
    error "Invalid cohort: $COHORT_NAME"
    error "Valid cohorts: opioid_ed, non_opioid_ed"
    exit 1
fi

# Get project root (assume script is in utility_scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Detect Python executable
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    error "Python not found. Please install Python 3 or activate your virtual environment."
    exit 1
fi

# Check for virtual environment in common locations
if [ -f "$HOME/jupyter-env/bin/python3" ]; then
    PYTHON_CMD="$HOME/jupyter-env/bin/python3"
    log "Using Python from jupyter-env: $PYTHON_CMD"
elif [ -f "$PROJECT_ROOT/venv/bin/python3" ]; then
    PYTHON_CMD="$PROJECT_ROOT/venv/bin/python3"
    log "Using Python from project venv: $PYTHON_CMD"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -f "$VIRTUAL_ENV/bin/python3" ]; then
    PYTHON_CMD="$VIRTUAL_ENV/bin/python3"
    log "Using Python from active virtualenv: $PYTHON_CMD"
else
    log "Using system Python: $PYTHON_CMD"
fi

# Verify Python works
if ! $PYTHON_CMD --version &> /dev/null; then
    error "Python command '$PYTHON_CMD' is not working. Please check your Python installation."
    exit 1
fi

log "Python version: $($PYTHON_CMD --version)"

# Convert age band to filename format
AGE_BAND_FNAME=$(echo "$AGE_BAND" | tr '-' '_')

log "Starting workflow for cohort: $COHORT_NAME, age_band: $AGE_BAND"
log "Project root: $PROJECT_ROOT"
log "Skip steps: ${SKIP_STEPS:-none}"

should_skip() {
    local step="$1"
    if [ -n "$SKIP_STEPS" ]; then
        echo "$SKIP_STEPS" | grep -q "$step"
    else
        return 1
    fi
}

run_step() {
    local step_num="$1"
    local step_name="$2"
    local command="$3"
    
    if should_skip "$step_num"; then
        warn "Skipping step $step_num: $step_name"
        return 0
    fi
    
    log "=========================================="
    log "Step $step_num: $step_name"
    log "=========================================="
    
    # Replace 'python ' or 'python3 ' at the start of command with detected Python
    # Only replace if command starts with "python " or "python3 " (not already a full path)
    local cmd_substituted="$command"
    if [[ "$cmd_substituted" == python\ * ]]; then
        # Replace "python " at the start - remove "python " and prepend PYTHON_CMD with space
        cmd_substituted="${PYTHON_CMD} ${cmd_substituted#python }"
    elif [[ "$cmd_substituted" == python3\ * ]]; then
        # Replace "python3 " at the start - remove "python3 " and prepend PYTHON_CMD with space
        cmd_substituted="${PYTHON_CMD} ${cmd_substituted#python3 }"
    fi
    
    log "Running: $cmd_substituted"
    
    if eval "$cmd_substituted"; then
        log "✅ Step $step_num completed successfully"
    else
        error "❌ Step $step_num failed"
        exit 1
    fi
}

# Step 3: Feature Importance (check for completed aggregated feature importances)
# This step is idempotent - will skip if results already exist
run_step "3" "Feature Importance (Check/Generate Aggregated)" \
    "python 3_feature_importance/run_mc_feature_importance.py --cohort $COHORT_NAME --age_band $AGE_BAND"

# Step 4a: Model Data Creation (with controls)
# Generate model_events.parquet with cases AND controls from gold medical/pharmacy data
# This must run BEFORE Step 4b to ensure local files with controls exist
# (Otherwise Step 4b will download from S3, which has files without controls)
# The script validates controls and uploads to S3 automatically
run_step "4a" "Model Data Creation (Cases + Controls)" \
    "python 4a_model_data/create_model_data.py --cohort $COHORT_NAME --age-band $AGE_BAND"

# Validate Step 4a output (check that controls are present and file exists)
if ! should_skip "4a"; then
    log "Validating Step 4a output (checking for controls)..."
    
    # Check if file was actually generated
    data_root=$(python3 -c "import sys; sys.path.insert(0, '.'); from py_helpers.env_utils import get_data_root; print(get_data_root())" 2>/dev/null || echo "$PROJECT_ROOT")
    expected_file_linux="$data_root/4a_model_data/cohort_name=$COHORT_NAME/age_band=$AGE_BAND/model_events.parquet"
    expected_file_local="$PROJECT_ROOT/4a_model_data/cohort_name=$COHORT_NAME/age_band=$AGE_BAND/model_events.parquet"
    
    if [ ! -f "$expected_file_linux" ] && [ ! -f "$expected_file_local" ]; then
        error "Step 4a validation failed: model_events.parquet was not generated"
        error "Expected at: $expected_file_linux or $expected_file_local"
        error "Please check Step 4a output above for errors."
        exit 1
    fi
    
    # Check for controls
    if $PYTHON_CMD utility_scripts/validate_model_data_controls.py \
        --cohort "$COHORT_NAME" --age-band "$AGE_BAND" 2>&1 | grep -q "WARNING: Missing controls"; then
        error "Step 4a validation failed: Missing controls in model_events.parquet"
        error "Please check the output above and regenerate model data."
        exit 1
    else
        log "✓ Step 4a validation passed: Controls present"
    fi
fi

# Step 4b: DTW Protocol Filtering
# Filter administrative/scheduling/non-medical codes, keep all surgeries
run_step "4b" "DTW Protocol Filtering (Admin/Scheduling Filter, Keep Surgeries)" \
    "python 4b_dtw_filter/filter_protocol_events.py --cohort-name $COHORT_NAME --age-band $AGE_BAND"

# Step 5c: PGx Feature Engineering (ONLY feature engineering step)
# Note: BupaR, FP-Growth, and DTW are now used only for dashboard visualizations
run_step "5c" "PGx Feature Engineering" \
    "python 5c_pgx_analysis/run_analysis.py --cohort-name $COHORT_NAME --age-band $AGE_BAND"

# Step 6: Final Model Training
# Uses aggregated feature importances directly + PGx features (no encoding)
# Trains CatBoost and XGBoost (XGBoost vs XGBoost RF), selects best by recall/AUC-PR
# Outputs: best CatBoost binary (for SHAP), best XGBoost JSON (for FFA)
run_step "6" "Final Model Training (Aggregated Features + PGx, No Encoding)" \
    "python 6b_final_model_selection/run_final_model.py --cohort $COHORT_NAME --age_band $AGE_BAND"

# Step 7: FFA Analysis (uses best XGBoost model JSON)
if ! should_skip "7"; then
    log "=========================================="
    log "Step 7: FFA Analysis (Best XGBoost Model)"
    log "=========================================="
    if $PYTHON_CMD 7_ffa_analysis/run_full_ffa_analysis.py --cohort-name $COHORT_NAME --age-band $AGE_BAND; then
        log "✅ Step 7 completed successfully"
    else
        warn "Step 7 failed (check if best XGBoost model JSON exists)"
    fi
fi

# Step 8: SHAP Analysis (uses best CatBoost model binary)
run_step "8" "SHAP Analysis (Best CatBoost Model)" \
    "python 8_shap_analysis/run_shap_analysis.py --cohort $COHORT_NAME --age_band $AGE_BAND"

# Step 9: Combined SHAP + FFA
if ! should_skip "9"; then
    log "=========================================="
    log "Step 9: Combined SHAP + FFA"
    log "=========================================="
    if $PYTHON_CMD 9_combined_shap_ffa/combine_shap_ffa_analysis.py --cohort $COHORT_NAME --age_band $AGE_BAND; then
        log "✅ Step 9 completed successfully"
    else
        warn "Step 9 failed (check if required inputs exist)"
    fi
fi

# Step 10: Risk Dashboard (BupaR/DTW/FP-Growth visualizations + causal analysis)
if ! should_skip "10"; then
    log "=========================================="
    log "Step 10: Risk Dashboard Preparation"
    log "=========================================="
    log "Note: BupaR, DTW, and FP-Growth are now used for dashboard visualizations only"
    # TODO: Add dashboard preparation script when available
    # if $PYTHON_CMD 10_risk_dashboard/prepare_dashboard.py --cohort-name $COHORT_NAME --age-band $AGE_BAND; then
    #     log "✅ Step 10 completed successfully"
    # else
    #     warn "Step 10 failed (check if required inputs exist)"
    # fi
    warn "Step 10: Dashboard preparation script not yet implemented"
fi

# Step 11: Deploy to S3/AWS Lambda
if ! should_skip "11"; then
    log "=========================================="
    log "Step 11: Deploy to S3/AWS Lambda"
    log "=========================================="
    # TODO: Add deployment script when available
    # if $PYTHON_CMD 10_risk_dashboard/deploy_dashboard.py --cohort-name $COHORT_NAME --age-band $AGE_BAND; then
    #     log "✅ Step 11 completed successfully"
    # else
    #     warn "Step 11 failed (check deployment configuration)"
    # fi
    warn "Step 11: Deployment script not yet implemented"
fi

log "=========================================="
log "✅ Workflow completed successfully!"
log "=========================================="
log "Cohort: $COHORT_NAME"
log "Age Band: $AGE_BAND"
log "All steps completed"

