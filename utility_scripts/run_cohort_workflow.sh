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
#   7: SHAP Analysis (use best CatBoost model binary)
#   8: FFA Analysis (use best XGBoost model JSON and SHAP importance from Step 7)
#   Note: Step 9 (Combined SHAP + FFA) removed - consensus is already reflected in FFA's causal importance
#   9: Risk Dashboard (BupaR/DTW/FP-Growth visualizations + causal analysis)
#   10: Deploy to S3/AWS Lambda (if applicable)
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
if [ -f "/home/pgx3874/jupyter-env/bin/python3.11" ]; then
    PYTHON_CMD="/home/pgx3874/jupyter-env/bin/python3.11"
    log "Using Python from EC2 jupyter-env: $PYTHON_CMD"
elif [ -f "$HOME/jupyter-env/bin/python3" ]; then
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

# Time tracking file (persists across restarts)
TIME_LOG_DIR="$PROJECT_ROOT/logs/time_tracking"
mkdir -p "$TIME_LOG_DIR"
TIME_LOG_FILE="$TIME_LOG_DIR/${COHORT_NAME}_${AGE_BAND_FNAME}.json"

# Load previous time tracking if exists
if [ -f "$TIME_LOG_FILE" ]; then
    PREV_START_TIME=$(python3 -c "import json; d=json.load(open('$TIME_LOG_FILE')); print(d.get('workflow_start_time', ''))" 2>/dev/null || echo "")
    PREV_STEP_TIMES=$(python3 -c "import json; d=json.load(open('$TIME_LOG_FILE')); print(json.dumps(d.get('step_times', {})))" 2>/dev/null || echo "{}")
    PREV_TOTAL_TIME=$(python3 -c "import json; d=json.load(open('$TIME_LOG_FILE')); print(d.get('total_elapsed_seconds', 0))" 2>/dev/null || echo "0")
    
    if [ -n "$PREV_START_TIME" ] && [ "$PREV_START_TIME" != "None" ]; then
        log "Resuming time tracking from previous run (started: $PREV_START_TIME)"
        log "Previous total elapsed time: $(python3 -c "print(f'{int($PREV_TOTAL_TIME)//3600}h {int($PREV_TOTAL_TIME)%3600//60}m {int($PREV_TOTAL_TIME)%60}s')" 2>/dev/null || echo 'unknown')"
    fi
else
    PREV_START_TIME=""
    PREV_STEP_TIMES="{}"
    PREV_TOTAL_TIME=0
fi

# Initialize or update workflow start time
WORKFLOW_START_TIME=$(date +%s)
if [ -z "$PREV_START_TIME" ] || [ "$PREV_START_TIME" = "None" ]; then
    python3 -c "
import json
from datetime import datetime
data = {
    'cohort': '$COHORT_NAME',
    'age_band': '$AGE_BAND',
    'workflow_start_time': $WORKFLOW_START_TIME,
    'workflow_start_time_iso': datetime.fromtimestamp($WORKFLOW_START_TIME).isoformat(),
    'step_times': {},
    'total_elapsed_seconds': 0,
    'last_updated': datetime.now().isoformat()
}
with open('$TIME_LOG_FILE', 'w') as f:
    json.dump(data, f, indent=2)
" 2>/dev/null || true
    log "Started new time tracking: $(date -d @$WORKFLOW_START_TIME 2>/dev/null || date -r $WORKFLOW_START_TIME 2>/dev/null || echo 'now')"
else
    # Use previous start time for cumulative tracking
    WORKFLOW_START_TIME=$(python3 -c "print(int('$PREV_START_TIME'))" 2>/dev/null || echo "$(date +%s)")
fi

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
    
    # Check if step was already completed (from previous run)
    STEP_COMPLETED=$(python3 -c "
import json
try:
    with open('$TIME_LOG_FILE') as f:
        data = json.load(f)
        step_times = data.get('step_times', {})
        step_key = '$step_num'
        if step_key in step_times and step_times[step_key].get('completed', False):
            print('yes')
        else:
            print('no')
except:
    print('no')
" 2>/dev/null || echo "no")
    
    if [ "$STEP_COMPLETED" = "yes" ]; then
        # For Step 6, verify essential files exist before skipping
        # (Python script handles S3 downloads, but we need to let it run if files are missing)
        if [ "$step_num" = "6" ]; then
            AGE_BAND_FNAME=$(echo "$AGE_BAND" | tr '-' '_')
            FEATURES_CSV="$PROJECT_ROOT/6_final_model/outputs/$COHORT_NAME/$AGE_BAND_FNAME/${COHORT_NAME}_${AGE_BAND_FNAME}_train_final_features_no_leakage.csv"
            if [ ! -f "$FEATURES_CSV" ]; then
                log "Step 6 marked as completed but features CSV missing. Re-running to download/regenerate..."
                # Clear the completion flag so we run the step
                python3 -c "
import json
try:
    with open('$TIME_LOG_FILE', 'r') as f:
        data = json.load(f)
    if 'step_times' in data and '6' in data['step_times']:
        data['step_times']['6']['completed'] = False
    with open('$TIME_LOG_FILE', 'w') as f:
        json.dump(data, f, indent=2)
except:
    pass
" 2>/dev/null || true
                STEP_COMPLETED="no"
            fi
        fi
        
        if [ "$STEP_COMPLETED" = "yes" ]; then
            PREV_DURATION=$(python3 -c "
import json
try:
    with open('$TIME_LOG_FILE') as f:
        data = json.load(f)
        step_times = data.get('step_times', {})
        step_key = '$step_num'
        duration = step_times.get(step_key, {}).get('duration_seconds', 0)
        print(int(duration))
except:
    print(0)
" 2>/dev/null || echo "0")
            log "Step $step_num already completed in previous run (duration: ${PREV_DURATION}s)"
            return 0
        fi
    fi
    
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
    
    # Record step start time
    STEP_START_TIME=$(date +%s)
    
    if eval "$cmd_substituted"; then
        # Record step end time and duration
        STEP_END_TIME=$(date +%s)
        STEP_DURATION=$((STEP_END_TIME - STEP_START_TIME))
        
        # Update time log
        python3 -c "
import json
from datetime import datetime
try:
    with open('$TIME_LOG_FILE', 'r') as f:
        data = json.load(f)
except:
    data = {'step_times': {}, 'total_elapsed_seconds': 0}
    
if 'step_times' not in data:
    data['step_times'] = {}

step_key = '$step_num'
data['step_times'][step_key] = {
    'step_name': '$step_name',
    'start_time': $STEP_START_TIME,
    'start_time_iso': datetime.fromtimestamp($STEP_START_TIME).isoformat(),
    'end_time': $STEP_END_TIME,
    'end_time_iso': datetime.fromtimestamp($STEP_END_TIME).isoformat(),
    'duration_seconds': $STEP_DURATION,
    'duration_formatted': f'{int($STEP_DURATION)//3600}h {int($STEP_DURATION)%3600//60}m {int($STEP_DURATION)%60}s',
    'completed': True
}

# Update total elapsed time
data['total_elapsed_seconds'] = data.get('total_elapsed_seconds', 0) + $STEP_DURATION
data['last_updated'] = datetime.now().isoformat()

with open('$TIME_LOG_FILE', 'w') as f:
    json.dump(data, f, indent=2)
" 2>/dev/null || true
        
        # Calculate and display cumulative time
        TOTAL_ELAPSED=$(python3 -c "
import json
try:
    with open('$TIME_LOG_FILE') as f:
        data = json.load(f)
        total = data.get('total_elapsed_seconds', 0)
        hours = int(total) // 3600
        minutes = (int(total) % 3600) // 60
        seconds = int(total) % 60
        print(f'{hours}h {minutes}m {seconds}s')
except:
    print('0h 0m 0s')
" 2>/dev/null || echo "0h 0m 0s")
        
        log "✅ Step $step_num completed successfully (duration: ${STEP_DURATION}s, total elapsed: $TOTAL_ELAPSED)"
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
# Always validate, even if step was skipped due to checkpoint
log "Validating Step 4a output (checking for controls)..."

# Check if file was actually generated
data_root=$(python3 -c "import sys; sys.path.insert(0, '.'); from py_helpers.env_utils import get_data_root; print(get_data_root())" 2>/dev/null || echo "$PROJECT_ROOT")
expected_file_linux="$data_root/4a_model_data/cohort_name=$COHORT_NAME/age_band=$AGE_BAND/model_events.parquet"
expected_file_local="$PROJECT_ROOT/4a_model_data/cohort_name=$COHORT_NAME/age_band=$AGE_BAND/model_events.parquet"

if [ ! -f "$expected_file_linux" ] && [ ! -f "$expected_file_local" ]; then
    # File missing - check if checkpoint says it's complete
    STEP_COMPLETED=$(python3 -c "
import json
try:
    with open('$TIME_LOG_FILE') as f:
        data = json.load(f)
        step_times = data.get('step_times', {})
        step_key = '4a'
        if step_key in step_times and step_times[step_key].get('completed', False):
            print('yes')
        else:
            print('no')
except:
    print('no')
" 2>/dev/null || echo "no")
    
    if [ "$STEP_COMPLETED" = "yes" ]; then
        warn "Step 4a marked as completed but output file is missing. Clearing checkpoint and re-running..."
        # Clear the completion flag
        python3 -c "
import json
try:
    with open('$TIME_LOG_FILE', 'r') as f:
        data = json.load(f)
    if 'step_times' in data and '4a' in data['step_times']:
        data['step_times']['4a']['completed'] = False
    with open('$TIME_LOG_FILE', 'w') as f:
        json.dump(data, f, indent=2)
except:
    pass
" 2>/dev/null || true
        # Re-run Step 4a
        log "Re-running Step 4a..."
        $PYTHON_CMD 4a_model_data/create_model_data.py --cohort "$COHORT_NAME" --age-band "$AGE_BAND" || {
            error "Step 4a failed to generate model_events.parquet"
            exit 1
        }
    else
        error "Step 4a validation failed: model_events.parquet was not generated"
        error "Expected at: $expected_file_linux or $expected_file_local"
        error "Please check Step 4a output above for errors."
        exit 1
    fi
fi

# Check for controls (only if not explicitly skipped via SKIP_STEPS)
if ! should_skip "4a"; then
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

# Step 5: PGx Feature Engineering (ONLY feature engineering step)
# Note: BupaR, FP-Growth, and DTW are now used only for dashboard visualizations
run_step "5" "PGx Feature Engineering" \
    "python 5_pgx_analysis/run_analysis.py --cohort-name $COHORT_NAME --age-band $AGE_BAND"

# Step 6: Final Model Training
# Uses aggregated feature importances directly + PGx features (no encoding)
# Trains CatBoost and XGBoost (XGBoost vs XGBoost RF), selects best by recall/AUC-PR
# Outputs: best CatBoost binary (for Step 7 SHAP), best XGBoost JSON (for Step 8 FFA)
# Step 6: Final Model Training
# Uses aggregated feature importances directly + PGx features (no encoding)
# Trains CatBoost and XGBoost (XGBoost vs XGBoost RF), selects best by recall/AUC-PR
# Outputs: best CatBoost binary (for Step 7 SHAP), best XGBoost JSON (for Step 8 FFA)
# Note: run_step will verify features CSV exists before skipping
run_step "6" "Final Model Training (Aggregated Features + PGx, No Encoding)" \
    "python 6_final_model_selection/run_final_model.py --cohort $COHORT_NAME --age_band $AGE_BAND"

# Step 7: SHAP Analysis (uses best CatBoost model binary)
# Must run before Step 8 (FFA) since FFA uses SHAP values to prioritize rules
run_step "7" "SHAP Analysis (Best CatBoost Model)" \
    "python 7_shap_analysis/run_shap_analysis.py --cohort $COHORT_NAME --age_band $AGE_BAND"

# Step 8: FFA Analysis (uses best XGBoost model JSON and SHAP importance from Step 7)
run_step "8" "FFA Analysis (Best XGBoost Model, uses SHAP from Step 7)" \
    "python 8_ffa_analysis/run_full_ffa_analysis.py --cohort-name $COHORT_NAME --age-band $AGE_BAND"

# Step 9: Risk Dashboard (BupaR/DTW/FP-Growth visualizations + causal analysis)
# Prepare models and metadata for this specific cohort/age_band
# This step can run independently for each cohort/age_band - doesn't require all cohorts
if ! should_skip "9"; then
    log "=========================================="
    log "Step 9: Risk Dashboard Preparation"
    log "=========================================="
    log "Note: BupaR, DTW, and FP-Growth are now used for dashboard visualizations only"
    log "Preparing dashboard artifacts for cohort: $COHORT_NAME, age_band: $AGE_BAND"
    
    # Check if required inputs exist (Step 6 models)
    MODEL_DIR="$PROJECT_ROOT/6_final_model/outputs/$COHORT_NAME/${AGE_BAND_FNAME}/models"
    if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
        warn "Step 9 skipped: Models not found at $MODEL_DIR"
        warn "Step 6 must complete first for this cohort/age_band"
    else
        # Prepare models for this specific cohort/age_band only
        if $PYTHON_CMD 10_risk_dashboard/prepare_models.py --cohort "$COHORT_NAME" --age-band "$AGE_BAND"; then
            log "✅ Model preparation completed for $COHORT_NAME/$AGE_BAND"
        else
            warn "Step 9: Model preparation had warnings (check logs)"
        fi
        
        # Generate metadata for this specific cohort/age_band only
        if $PYTHON_CMD 10_risk_dashboard/generate_metadata.py --cohort "$COHORT_NAME" --age-band "$AGE_BAND"; then
            log "✅ Metadata generation completed for $COHORT_NAME/$AGE_BAND"
        else
            warn "Step 9: Metadata generation had warnings (check logs)"
        fi
        
        log "✅ Step 9 completed successfully for $COHORT_NAME/$AGE_BAND"
        log "Note: Dashboard can be built incrementally as cohorts complete"
    fi
fi

# Step 11: Deploy to S3/AWS Lambda
# Note: Deployment should be done separately after all cohorts are ready (or when ready)
# Use: ./utility_scripts/build_dashboard.sh to build incrementally
# Use: cd 10_risk_dashboard && ./docker_build.sh to build Docker image
if ! should_skip "11"; then
    log "=========================================="
    log "Step 11: Deploy to S3/AWS Lambda"
    log "=========================================="
    log "Note: Dashboard deployment is done separately using build_dashboard.sh"
    log "      This ensures all available cohorts are included in the build"
    log ""
    log "To build dashboard with available cohorts:"
    log "  ./utility_scripts/build_dashboard.sh"
    log ""
    log "To build Docker image for deployment:"
    log "  cd 10_risk_dashboard && ./docker_build.sh"
    log ""
    warn "Step 11: Skipped (use build_dashboard.sh when ready to deploy)"
fi

log "=========================================="
# Calculate and display final statistics
FINAL_STATS=$(python3 -c "
import json
from datetime import datetime
try:
    with open('$TIME_LOG_FILE') as f:
        data = json.load(f)
        total_seconds = data.get('total_elapsed_seconds', 0)
        workflow_start_iso = data.get('workflow_start_time_iso', 'unknown')
        
        hours = int(total_seconds) // 3600
        minutes = (int(total_seconds) % 3600) // 60
        seconds = int(total_seconds) % 60
        
        stats = []
        stats.append(f'Total execution time: {hours}h {minutes}m {seconds}s')
        stats.append(f'Workflow started: {workflow_start_iso}')
        
        # Show step breakdown
        step_times = data.get('step_times', {})
        if step_times:
            stats.append('')
            stats.append('Step breakdown:')
            for step_num in sorted(step_times.keys()):
                step_info = step_times[step_num]
                if step_info.get('completed', False):
                    duration = step_info.get('duration_seconds', 0)
                    step_name = step_info.get('step_name', 'Unknown')
                    d_hours = int(duration) // 3600
                    d_minutes = (int(duration) % 3600) // 60
                    d_seconds = int(duration) % 60
                    stats.append(f'  Step {step_num} ({step_name}): {d_hours}h {d_minutes}m {d_seconds}s')
        
        print('\\n'.join(stats))
except Exception as e:
    print(f'Time tracking statistics unavailable: {e}')
" 2>/dev/null || echo "Time tracking statistics unavailable")

log "✅ Workflow completed successfully!"
log ""
log "=========================================="
log "Time Tracking Summary"
log "=========================================="
log "$FINAL_STATS"
log ""
log "Time tracking log saved to: $TIME_LOG_FILE"
log "=========================================="
log "Cohort: $COHORT_NAME"
log "Age Band: $AGE_BAND"
log "All steps completed"

