#!/bin/bash
#
# Cleanup script for full workflow reset: checkpoints, S3 artifacts, EC2/local outputs.
#
# This script clears:
# - Step 2: Cohort parquet (S3 + local)
# - Step 1b: Event filter outputs (S3 + local)
# - Step 3b/3a: Feature importance outputs (S3 + local; _baseline preserved, overwritten on re-run)
# - Step 4/4a: Model data (S3 + NVMe + project)
# - Step 5–9: PGx features, final model, SHAP, FFA, combined (S3)
# - Step 6: Trained models (local + S3)
# - Checkpoints: pipeline_checkpoints + pgx-pipeline-status (optional)
#
# IMPORTANT: Does NOT delete gold medical/pharmacy tables
# (/mnt/nvme/gold/medical/, /mnt/nvme/gold/pharmacy/). Does NOT delete historical FI:
# s3://pgx-repository/pgx-analysis/3_feature_importance/outputs/ (1b reads from here; bucket has versioning so overwrites are safe).
# See docs/CLEAR_WORKFLOW_FOR_FULL_RUN.md.
#
# Usage: ./cleanup_cohort_data.sh [--skip-checkpoints] [--skip-s3] [--skip-local] [--yes]
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
SKIP_CHECKPOINTS=false
SKIP_S3=false
SKIP_LOCAL=false
AUTO_CONFIRM=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-checkpoints)
            SKIP_CHECKPOINTS=true
            shift
            ;;
        --skip-s3)
            SKIP_S3=true
            shift
            ;;
        --skip-local)
            SKIP_LOCAL=true
            shift
            ;;
        --yes)
            AUTO_CONFIRM=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-checkpoints] [--skip-s3] [--skip-local] [--yes]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Cohort Data Cleanup Script"
echo "=========================================="
echo ""
echo "This script will clear:"
echo "  - Step 2: Cohort parquet files"
echo "  - Step 1b: Event filter outputs"
echo "  - Step 3b / 3a: Feature importance outputs"
echo "  - Step 4/4a: Model data"
echo "  - Step 5–9: PGx features, final model, SHAP, FFA, combined (S3)"
echo "  - Step 6: Trained models (local)"
if [ "$SKIP_CHECKPOINTS" = false ]; then
    echo "  - Checkpoints (S3: pipeline_checkpoints + pgx-pipeline-status)"
fi
echo ""
echo -e "${GREEN}NOTE: Gold medical/pharmacy tables are preserved${NC}"
echo "  (These are shared across workers and should not be deleted)"
echo ""
echo -e "${YELLOW}WARNING: This will delete data!${NC}"
echo ""

if [ "$AUTO_CONFIRM" = false ]; then
    read -p "Are you sure you want to continue? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        echo "Cleanup cancelled."
        exit 0
    fi
else
    echo "Auto-confirmation enabled (--yes flag). Proceeding with cleanup..."
fi

# S3 bucket
S3_BUCKET="pgxdatalake"
S3_REPO_BUCKET="pgx-repository"

# Local paths (EC2). PROJECT_ROOT = repo root (parent of utility_scripts).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# NEVER delete gold/medical or gold/pharmacy (only gold/cohorts and other step outputs).
NVME_ROOT="/mnt/nvme"
DATA_ROOT="${NVME_ROOT}/gold"
LOCAL_COHORT_ROOT="${DATA_ROOT}/cohorts"  # Only gold/cohorts; never ${DATA_ROOT}/medical or ${DATA_ROOT}/pharmacy
MODEL_DATA_ROOT="${NVME_ROOT}/4a_model_data"
MODEL_DATA_ROOT_4="${NVME_ROOT}/4_model_data"  # Current step name
STEP3B_OUTPUTS="${PROJECT_ROOT}/3b_feature_importance_eda/outputs"
STEP3A_OUTPUTS="${PROJECT_ROOT}/3a_feature_importance/outputs"
STEP3_OUTPUTS="${PROJECT_ROOT}/3_feature_importance/outputs"
STEP1B_OUTPUTS="${PROJECT_ROOT}/1b_apcd_event_filter/outputs"
STEP6_MODELS="${PROJECT_ROOT}/6_final_model/models"
# Project pipeline output dirs (Steps 2, 5–9) — same as notebook PROJECT_OUTPUT_DIRS
PROJECT_OUTPUT_DIRS=(
    "2_create_cohort/cohort_metrics"
    "4_model_data/cohort_name=opioid_ed"
    "4_model_data/cohort_name=non_opioid_ed"
    "5_pgx_analysis/outputs"
    "6_final_model/outputs"
    "6_final_model/model_outputs"
    "7_shap_analysis/outputs"
    "8_ffa_analysis/results"
    "9_risk_dashboard/outputs"
    "feature_encoding_outputs"
    "logs"
)

# Counter for deleted items
DELETED_COUNT=0

# Log file
LOG_FILE="${PROJECT_ROOT}/cleanup_cohort_data_$(date +%Y%m%d_%H%M%S).log"

# Function to log message
log_message() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Function to check and log S3 path
# Returns 0 if path exists, 0 if missing (not an error), 0 if skipped
check_s3_path() {
    local path=$1
    local description=$2
    
    if [ "$SKIP_S3" = true ]; then
        log_message "[SKIP S3] $description"
        return 0  # Not an error, just skipped
    fi
    
    # Temporarily disable set -e to prevent script exit on error
    set +e
    aws s3 ls "$path" &>/dev/null
    local ls_status=$?
    set -e
    
    if [ $ls_status -eq 0 ]; then
        # Get size and count (may fail, but that's OK)
        set +e
        local size=$(aws s3 ls "$path" --recursive --summarize 2>/dev/null | grep "Total Size" | awk '{print $3, $4}')
        local count=$(aws s3 ls "$path" --recursive 2>/dev/null | wc -l)
        set -e
        log_message "[S3 EXISTS] $description"
        log_message "           Path: $path"
        if [ -n "$count" ]; then
            log_message "           Files: $count"
        fi
        if [ -n "$size" ]; then
            log_message "           Size: $size"
        fi
        return 0
    else
        log_message "[S3 MISSING] $description"
        log_message "            Path: $path"
        return 0  # Missing path is not an error, just informational
    fi
}

# Function to check and log local path
# Returns 0 if path exists, 0 if missing (not an error), 0 if skipped
check_local_path() {
    local path=$1
    local description=$2
    
    if [ "$SKIP_LOCAL" = true ]; then
        log_message "[SKIP LOCAL] $description"
        return 0  # Not an error, just skipped
    fi
    
    if [ -d "$path" ] || [ -f "$path" ]; then
        if [ -d "$path" ]; then
            local size=$(du -sh "$path" 2>/dev/null | awk '{print $1}')
            local count=$(find "$path" -type f 2>/dev/null | wc -l)
            log_message "[LOCAL EXISTS] $description"
            log_message "              Path: $path"
            log_message "              Files: $count"
            log_message "              Size: $size"
        else
            local size=$(du -sh "$path" 2>/dev/null | awk '{print $1}')
            log_message "[LOCAL EXISTS] $description"
            log_message "              Path: $path"
            log_message "              Size: $size"
        fi
        return 0
    else
        log_message "[LOCAL MISSING] $description"
        log_message "               Path: $path"
        return 0  # Missing path is not an error, just informational
    fi
}

# Function to delete S3 path
delete_s3_path() {
    local path=$1
    local description=$2
    
    if [ "$SKIP_S3" = true ]; then
        echo -e "${YELLOW}[SKIP S3]${NC} $description"
        return
    fi
    
    echo -e "${YELLOW}[S3]${NC} Deleting: $description"
    log_message "[S3 DELETE] $description"
    log_message "           Path: $path"
    # Temporarily disable set -e to prevent script exit on error
    set +e
    aws s3 ls "$path" &>/dev/null
    local ls_status=$?
    set -e
    
    if [ $ls_status -eq 0 ]; then
        # Get size before deletion (may fail, but that's OK)
        set +e
        local size_before=$(aws s3 ls "$path" --recursive --summarize 2>/dev/null | grep "Total Size" | awk '{print $3, $4}')
        set -e
        if [ -n "$size_before" ]; then
            log_message "           Size before: $size_before"
        fi
        # Temporarily disable set -e for deletion command to prevent script exit on error
        set +e
        aws s3 rm "$path" --recursive
        local delete_status=$?
        set -e
        if [ $delete_status -eq 0 ]; then
            echo -e "${GREEN}[S3]${NC} Deleted: $description"
            log_message "           Status: DELETED"
            # Use set +e around arithmetic to prevent failure
            set +e
            ((DELETED_COUNT++))
            set -e
        else
            echo -e "${YELLOW}[S3]${NC} Deletion may have failed (check logs): $description"
            log_message "           Status: DELETION ATTEMPTED (exit code: $delete_status)"
        fi
    else
        echo -e "${YELLOW}[S3]${NC} Path not found (may already be deleted): $description"
        log_message "           Status: NOT FOUND (already deleted or doesn't exist)"
    fi
    log_message ""
}

# Delete S3 prefix but preserve keys containing _baseline/ (baseline aggregated FI; overwritten on re-run)
delete_s3_prefix_exclude_baseline() {
    local prefix=$1
    local description=$2
    if [ "$SKIP_S3" = true ]; then
        echo -e "${YELLOW}[SKIP S3]${NC} $description"
        return
    fi
    echo -e "${YELLOW}[S3]${NC} Deleting (preserving _baseline): $description"
    log_message "[S3 DELETE EXCLUDE BASELINE] $description"
    log_message "           Prefix: s3://${S3_BUCKET}/${prefix}"
    set +e
    local deleted=0
    while IFS= read -r key; do
        [ -z "$key" ] && continue
        if [[ "$key" != *"_baseline/"* ]]; then
            aws s3 rm "s3://${S3_BUCKET}/${key}" 2>/dev/null && ((deleted++)) || true
        fi
    done < <(aws s3 ls "s3://${S3_BUCKET}/${prefix}" --recursive 2>/dev/null | awk '{print $4}')
    set -e
    echo -e "${GREEN}[S3]${NC} Deleted $deleted objects (baseline preserved): $description"
    log_message "           Status: DELETED $deleted objects (baseline preserved)"
    [ "$deleted" -gt 0 ] && { set +e; ((DELETED_COUNT++)); set -e; }
    log_message ""
}

# Function to delete local path
delete_local_path() {
    local path=$1
    local description=$2
    
    if [ "$SKIP_LOCAL" = true ]; then
        echo -e "${YELLOW}[SKIP LOCAL]${NC} $description"
        return
    fi
    
    echo -e "${YELLOW}[LOCAL]${NC} Deleting: $description"
    log_message "[LOCAL DELETE] $description"
    log_message "              Path: $path"
    if [ -d "$path" ] || [ -f "$path" ]; then
        if [ -d "$path" ]; then
            local size_before=$(du -sh "$path" 2>/dev/null | awk '{print $1}')
            local count_before=$(find "$path" -type f 2>/dev/null | wc -l)
            log_message "              Size before: $size_before"
            log_message "              Files before: $count_before"
        else
            local size_before=$(du -sh "$path" 2>/dev/null | awk '{print $1}')
            log_message "              Size before: $size_before"
        fi
        # Temporarily disable set -e for rm command
        set +e
        rm -rf "$path"
        local rm_status=$?
        set -e
        if [ $rm_status -eq 0 ]; then
            echo -e "${GREEN}[LOCAL]${NC} Deleted: $description"
            log_message "              Status: DELETED"
            set +e
            ((DELETED_COUNT++))
            set -e
        else
            echo -e "${YELLOW}[LOCAL]${NC} Deletion may have failed: $description"
            log_message "              Status: DELETION ATTEMPTED (exit code: $rm_status)"
        fi
    else
        echo -e "${YELLOW}[LOCAL]${NC} Path not found (may already be deleted): $description"
        log_message "              Status: NOT FOUND (already deleted or doesn't exist)"
    fi
    log_message ""
}

# Initialize log file
log_message "=========================================="
log_message "Cohort Data Cleanup Log"
log_message "Started: $(date)"
log_message "=========================================="
log_message ""

echo ""
echo "=========================================="
echo "Scanning existing data..."
echo "=========================================="
log_message "--- Scanning existing data ---"
echo ""

# Step 2: Cohort parquet files
echo "--- Step 2: Cohort Data ---"
log_message "--- Step 2: Cohort Data ---"
# New format: s3://pgxdatalake/gold/cohorts/
# NOTE: S3 uses normalized cohort names: ed_non_opioid -> non_opioid_ed (see COHORT_ALIASES in s3_utils.py)
check_s3_path "s3://${S3_BUCKET}/gold/cohorts/cohort_name=non_opioid_ed/" "Step 2: ED_NON_OPIOID cohorts (S3 - new format)"
check_s3_path "s3://${S3_BUCKET}/gold/cohorts/cohort_name=opioid_ed/" "Step 2: OPIOID_ED cohorts (S3 - new format)"
# Legacy paths (old format) - for cleanup
check_s3_path "s3://${S3_BUCKET}/gold/cohorts_F1120/" "Step 2: Legacy cohorts_F1120 (S3 - old format)"
check_s3_path "s3://${S3_BUCKET}/gold/cohorts_model_data/" "Step 2: Legacy cohorts_model_data (S3 - old format)"
check_local_path "${LOCAL_COHORT_ROOT}/cohort_name=ed_non_opioid" "Step 2: ED_NON_OPIOID cohorts (local)"
check_local_path "${LOCAL_COHORT_ROOT}/cohort_name=opioid_ed" "Step 2: OPIOID_ED cohorts (local)"
check_local_path "${LOCAL_COHORT_ROOT}/cohort_name=non_opioid_ed" "Step 2: NON_OPIOID_ED cohorts (local)"
check_local_path "${NVME_ROOT}/cohorts_staging" "Step 2: Cohorts staging (NVMe)"
for _rel in "${PROJECT_OUTPUT_DIRS[@]}"; do
    check_local_path "${PROJECT_ROOT}/${_rel}" "Project: ${_rel}"
done
if [ -d "${PROJECT_ROOT}/data/gold/cohorts" ]; then
    check_local_path "${PROJECT_ROOT}/data/gold/cohorts/cohort_name=ed_non_opioid" "Step 2: ED_NON_OPIOID cohorts (project data)"
    check_local_path "${PROJECT_ROOT}/data/gold/cohorts/cohort_name=opioid_ed" "Step 2: OPIOID_ED cohorts (project data)"
    check_local_path "${PROJECT_ROOT}/data/gold/cohorts/cohort_name=non_opioid_ed" "Step 2: NON_OPIOID_ED cohorts (project data)"
fi

echo ""

# Step 1b: Event filter outputs
echo "--- Step 1b: Event Filter Outputs ---"
log_message "--- Step 1b: Event Filter Outputs ---"
check_local_path "${STEP1B_OUTPUTS}" "Step 1b: Event filter outputs (local)"
check_s3_path "s3://${S3_BUCKET}/gold/event_filter/" "Step 1b: Event filter (S3)"

echo ""

# Step 3b + 3a: Feature importance outputs
echo "--- Step 3b / 3a: Feature Importance Outputs ---"
log_message "--- Step 3b / 3a: Feature Importance Outputs ---"
check_local_path "${STEP3B_OUTPUTS}/ed_non_opioid" "Step 3b: ED_NON_OPIOID feature importance"
check_local_path "${STEP3B_OUTPUTS}/opioid_ed" "Step 3b: OPIOID_ED feature importance"
check_local_path "${STEP3A_OUTPUTS}" "Step 3a: MC feature importance outputs"
check_local_path "${STEP3_OUTPUTS}" "Step 3: Legacy feature importance outputs"
check_s3_path "s3://${S3_BUCKET}/gold/bupar/ed_non_opioid/" "Step 3b: ED_NON_OPIOID BupaR outputs (S3)"
check_s3_path "s3://${S3_BUCKET}/gold/bupar/opioid_ed/" "Step 3b: OPIOID_ED BupaR outputs (S3)"
check_s3_path "s3://${S3_BUCKET}/gold/feature_importance/" "Step 3a: Feature importance (S3, _baseline preserved)"

echo ""

# Step 4a: Model data
echo "--- Step 4a: Model Data ---"
log_message "--- Step 4a: Model Data ---"
# Local paths (old format)
check_local_path "${MODEL_DATA_ROOT}/cohort_name=ed_non_opioid" "Step 4a: ED_NON_OPIOID model data (NVMe - old format)"
check_local_path "${MODEL_DATA_ROOT}/cohort_name=opioid_ed" "Step 4a: OPIOID_ED model data (NVMe - old format)"
check_local_path "${MODEL_DATA_ROOT_4}/cohort_name=non_opioid_ed" "Step 4: POLYPHARMACY model data (NVMe - 4_model_data)"
check_local_path "${MODEL_DATA_ROOT_4}/cohort_name=opioid_ed" "Step 4: OPIOID_ED model data (NVMe - 4_model_data)"
# Local paths (new format with slugs)
check_local_path "${MODEL_DATA_ROOT}/cohorts/input_model_data/cohort_name=polypharmacy" "Step 4a: POLYPHARMACY model data (NVMe - new format)"
check_local_path "${MODEL_DATA_ROOT}/cohorts/input_model_data/cohort_name=opioid" "Step 4a: OPIOID model data (NVMe - new format)"
if [ -d "${PROJECT_ROOT}/4a_model_data" ] || [ -d "${PROJECT_ROOT}/4_model_data" ]; then
    check_local_path "${PROJECT_ROOT}/4a_model_data/cohort_name=ed_non_opioid" "Step 4a: ED_NON_OPIOID model data (project - old format)"
    check_local_path "${PROJECT_ROOT}/4a_model_data/cohort_name=opioid_ed" "Step 4a: OPIOID_ED model data (project - old format)"
    check_local_path "${PROJECT_ROOT}/4a_model_data/cohorts/input_model_data/cohort_name=polypharmacy" "Step 4a: POLYPHARMACY model data (project - new format)"
    check_local_path "${PROJECT_ROOT}/4a_model_data/cohorts/input_model_data/cohort_name=opioid" "Step 4a: OPIOID model data (project - new format)"
    [ -d "${PROJECT_ROOT}/4_model_data" ] && check_local_path "${PROJECT_ROOT}/4_model_data/cohort_name=non_opioid_ed" "Step 4: POLYPHARMACY model data (project - 4_model_data)"
    [ -d "${PROJECT_ROOT}/4_model_data" ] && check_local_path "${PROJECT_ROOT}/4_model_data/cohort_name=opioid_ed" "Step 4: OPIOID_ED model data (project - 4_model_data)"
fi
# S3 paths (old format with cohort names)
check_s3_path "s3://${S3_BUCKET}/gold/cohorts/input_model_data/cohort_name=ed_non_opioid/" "Step 4a: ED_NON_OPIOID model data (S3 - old format)"
check_s3_path "s3://${S3_BUCKET}/gold/cohorts/input_model_data/cohort_name=opioid_ed/" "Step 4a: OPIOID_ED model data (S3 - old format)"
# S3 paths (new format with slugs)
check_s3_path "s3://${S3_BUCKET}/gold/cohorts/input_model_data/cohort_name=polypharmacy/" "Step 4a: POLYPHARMACY model data (S3 - new format)"
check_s3_path "s3://${S3_BUCKET}/gold/cohorts/input_model_data/cohort_name=opioid/" "Step 4a: OPIOID model data (S3 - new format)"
# Legacy path (very old format) - for cleanup
check_s3_path "s3://${S3_BUCKET}/gold/4a_model_data/cohort_name=ed_non_opioid/" "Step 4a: ED_NON_OPIOID model data (S3 - legacy)"
check_s3_path "s3://${S3_BUCKET}/gold/4a_model_data/cohort_name=opioid_ed/" "Step 4a: OPIOID_ED model data (S3 - legacy)"
check_s3_path "s3://${S3_BUCKET}/gold/model_data/" "Step 4: Model data (S3 - alternate path)"
check_s3_path "s3://${S3_BUCKET}/gold/pgx_features/" "Step 5: PGx features (S3)"
check_s3_path "s3://${S3_BUCKET}/gold/final_model/" "Step 6: Final model (S3)"
check_s3_path "s3://${S3_BUCKET}/gold/shap_analysis/" "Step 7: SHAP analysis (S3)"
check_s3_path "s3://${S3_BUCKET}/gold/ffa_analysis/" "Step 8: FFA analysis (S3)"
check_s3_path "s3://${S3_BUCKET}/gold/combined_analysis/" "Step 9: Combined analysis (S3)"

echo ""

# Step 6: Trained models
echo "--- Step 6: Trained Models ---"
log_message "--- Step 6: Trained Models ---"
check_local_path "${STEP6_MODELS}" "Step 6: All trained models"
check_s3_path "s3://${S3_BUCKET}/gold/models/" "Step 6: Trained models (S3)"

echo ""

# Checkpoints (optional)
if [ "$SKIP_CHECKPOINTS" = false ]; then
    echo "--- Checkpoints ---"
    log_message "--- Checkpoints ---"
    check_s3_path "s3://${S3_REPO_BUCKET}/pipeline_checkpoints/" "All step checkpoints (1b, 4, 6, etc.)"
    check_s3_path "s3://${S3_REPO_BUCKET}/pgx-pipeline-status/create_cohort/" "Step 2: Cohort creation checkpoints"
    check_s3_path "s3://${S3_REPO_BUCKET}/pgx-pipeline-status/feature_importance_eda/" "Step 3b: Feature importance checkpoints"
    check_s3_path "s3://${S3_REPO_BUCKET}/pgx-pipeline-status/model_data/" "Step 4a: Model data checkpoints"
    check_s3_path "s3://${S3_REPO_BUCKET}/pgx-pipeline-status/final_model/" "Step 6: Model training checkpoints"
fi

log_message ""
log_message "--- End of scan ---"
log_message ""

echo ""
echo "=========================================="
echo "Starting cleanup..."
echo "=========================================="
log_message "=========================================="
log_message "Starting cleanup..."
log_message "=========================================="
log_message ""
echo ""

# Step 2: Cohort parquet files
echo "--- Step 2: Cohort Data ---"
# New format: s3://pgxdatalake/gold/cohorts/
# NOTE: S3 uses normalized cohort names: ed_non_opioid -> non_opioid_ed (see COHORT_ALIASES in s3_utils.py)
delete_s3_path "s3://${S3_BUCKET}/gold/cohorts/cohort_name=non_opioid_ed/" "Step 2: ED_NON_OPIOID cohorts (S3 - new format)"
delete_s3_path "s3://${S3_BUCKET}/gold/cohorts/cohort_name=opioid_ed/" "Step 2: OPIOID_ED cohorts (S3 - new format)"
# Legacy paths (old format) - for cleanup
delete_s3_path "s3://${S3_BUCKET}/gold/cohorts_F1120/" "Step 2: Legacy cohorts_F1120 (S3 - old format)"
delete_s3_path "s3://${S3_BUCKET}/gold/cohorts_model_data/" "Step 2: Legacy cohorts_model_data (S3 - old format)"
delete_local_path "${LOCAL_COHORT_ROOT}/cohort_name=ed_non_opioid" "Step 2: ED_NON_OPIOID cohorts (local)"
delete_local_path "${LOCAL_COHORT_ROOT}/cohort_name=opioid_ed" "Step 2: OPIOID_ED cohorts (local)"
delete_local_path "${LOCAL_COHORT_ROOT}/cohort_name=non_opioid_ed" "Step 2: NON_OPIOID_ED cohorts (local)"
delete_local_path "${NVME_ROOT}/cohorts_staging" "Step 2: Cohorts staging (NVMe)"
# Project pipeline output dirs (Steps 2, 5–9)
for _rel in "${PROJECT_OUTPUT_DIRS[@]}"; do
    delete_local_path "${PROJECT_ROOT}/${_rel}" "Project: ${_rel}"
done

# Project data dir: gold/cohorts layout (never gold/medical or gold/pharmacy)
if [ -d "${PROJECT_ROOT}/data/gold/cohorts" ]; then
    delete_local_path "${PROJECT_ROOT}/data/gold/cohorts/cohort_name=ed_non_opioid" "Step 2: ED_NON_OPIOID cohorts (project data)"
    delete_local_path "${PROJECT_ROOT}/data/gold/cohorts/cohort_name=opioid_ed" "Step 2: OPIOID_ED cohorts (project data)"
    delete_local_path "${PROJECT_ROOT}/data/gold/cohorts/cohort_name=non_opioid_ed" "Step 2: NON_OPIOID_ED cohorts (project data)"
fi

echo ""

# Step 1b: Event filter outputs
echo "--- Step 1b: Event Filter Outputs ---"
delete_local_path "${STEP1B_OUTPUTS}" "Step 1b: Event filter outputs (local)"
delete_s3_path "s3://${S3_BUCKET}/gold/event_filter/" "Step 1b: Event filter (S3)"

echo ""

# Step 3b + 3a: Feature importance outputs (preserve _baseline; overwritten on 3a --baseline re-run)
echo "--- Step 3b / 3a: Feature Importance Outputs ---"
delete_local_path "${STEP3B_OUTPUTS}/ed_non_opioid" "Step 3b: ED_NON_OPIOID feature importance"
delete_local_path "${STEP3B_OUTPUTS}/opioid_ed" "Step 3b: OPIOID_ED feature importance"
# Clear 3a local outputs but preserve _baseline subdir (overwritten on 3a --baseline re-run)
if [ "$SKIP_LOCAL" = false ]; then
    for _out_root in "${STEP3A_OUTPUTS}" "${STEP3_OUTPUTS}"; do
        [ -d "$_out_root" ] || continue
        for _cohort_dir in "${_out_root}"/*/; do
            [ -d "$_cohort_dir" ] || continue
            for _item in "${_cohort_dir}"*; do
                [ -e "$_item" ] || continue
                case "$_item" in
                    *"/_baseline" ) ;;
                    * ) rm -rf "$_item" 2>/dev/null || true ;;
                esac
            done
        done
    done
    echo -e "${GREEN}[LOCAL]${NC} Cleared 3a/3 feature importance (baseline preserved)"
    log_message "[LOCAL] Cleared 3a/3 feature importance outputs (baseline preserved)"
fi
delete_s3_path "s3://${S3_BUCKET}/gold/bupar/ed_non_opioid/" "Step 3b: ED_NON_OPIOID BupaR outputs (S3)"
delete_s3_path "s3://${S3_BUCKET}/gold/bupar/opioid_ed/" "Step 3b: OPIOID_ED BupaR outputs (S3)"
delete_s3_prefix_exclude_baseline "gold/feature_importance/" "Step 3a: Feature importance (S3, _baseline preserved)"

echo ""

# Step 4a: Model data
echo "--- Step 4a: Model Data ---"
# Local paths (old format)
delete_local_path "${MODEL_DATA_ROOT}/cohort_name=ed_non_opioid" "Step 4a: ED_NON_OPIOID model data (NVMe - old format)"
delete_local_path "${MODEL_DATA_ROOT}/cohort_name=opioid_ed" "Step 4a: OPIOID_ED model data (NVMe - old format)"
delete_local_path "${MODEL_DATA_ROOT_4}/cohort_name=non_opioid_ed" "Step 4: POLYPHARMACY model data (NVMe - 4_model_data)"
delete_local_path "${MODEL_DATA_ROOT_4}/cohort_name=opioid_ed" "Step 4: OPIOID_ED model data (NVMe - 4_model_data)"
# Local paths (new format with slugs)
delete_local_path "${MODEL_DATA_ROOT}/cohorts/input_model_data/cohort_name=polypharmacy" "Step 4a: POLYPHARMACY model data (NVMe - new format)"
delete_local_path "${MODEL_DATA_ROOT}/cohorts/input_model_data/cohort_name=opioid" "Step 4a: OPIOID model data (NVMe - new format)"
if [ -d "${PROJECT_ROOT}/4a_model_data" ]; then
    delete_local_path "${PROJECT_ROOT}/4a_model_data/cohort_name=ed_non_opioid" "Step 4a: ED_NON_OPIOID model data (project - old format)"
    delete_local_path "${PROJECT_ROOT}/4a_model_data/cohort_name=opioid_ed" "Step 4a: OPIOID_ED model data (project - old format)"
    delete_local_path "${PROJECT_ROOT}/4a_model_data/cohorts/input_model_data/cohort_name=polypharmacy" "Step 4a: POLYPHARMACY model data (project - new format)"
    delete_local_path "${PROJECT_ROOT}/4a_model_data/cohorts/input_model_data/cohort_name=opioid" "Step 4a: OPIOID model data (project - new format)"
fi
if [ -d "${PROJECT_ROOT}/4_model_data" ]; then
    delete_local_path "${PROJECT_ROOT}/4_model_data/cohort_name=non_opioid_ed" "Step 4: POLYPHARMACY model data (project - 4_model_data)"
    delete_local_path "${PROJECT_ROOT}/4_model_data/cohort_name=opioid_ed" "Step 4: OPIOID_ED model data (project - 4_model_data)"
fi
# S3 paths (old format with cohort names)
delete_s3_path "s3://${S3_BUCKET}/gold/cohorts/input_model_data/cohort_name=ed_non_opioid/" "Step 4a: ED_NON_OPIOID model data (S3 - old format)"
delete_s3_path "s3://${S3_BUCKET}/gold/cohorts/input_model_data/cohort_name=opioid_ed/" "Step 4a: OPIOID_ED model data (S3 - old format)"
# S3 paths (new format with slugs)
delete_s3_path "s3://${S3_BUCKET}/gold/cohorts/input_model_data/cohort_name=polypharmacy/" "Step 4a: POLYPHARMACY model data (S3 - new format)"
delete_s3_path "s3://${S3_BUCKET}/gold/cohorts/input_model_data/cohort_name=opioid/" "Step 4a: OPIOID model data (S3 - new format)"
# Legacy path (very old format) - for cleanup
delete_s3_path "s3://${S3_BUCKET}/gold/4a_model_data/cohort_name=ed_non_opioid/" "Step 4a: ED_NON_OPIOID model data (S3 - legacy)"
delete_s3_path "s3://${S3_BUCKET}/gold/4a_model_data/cohort_name=opioid_ed/" "Step 4a: OPIOID_ED model data (S3 - legacy)"
delete_s3_path "s3://${S3_BUCKET}/gold/model_data/" "Step 4: Model data (S3 - alternate path)"
delete_s3_path "s3://${S3_BUCKET}/gold/pgx_features/" "Step 5: PGx features (S3)"
delete_s3_path "s3://${S3_BUCKET}/gold/final_model/" "Step 6: Final model (S3)"
delete_s3_path "s3://${S3_BUCKET}/gold/shap_analysis/" "Step 7: SHAP analysis (S3)"
delete_s3_path "s3://${S3_BUCKET}/gold/ffa_analysis/" "Step 8: FFA analysis (S3)"
delete_s3_path "s3://${S3_BUCKET}/gold/combined_analysis/" "Step 9: Combined analysis (S3)"

echo ""

# Step 6: Trained models
echo "--- Step 6: Trained Models ---"
delete_local_path "${STEP6_MODELS}" "Step 6: All trained models"
delete_s3_path "s3://${S3_BUCKET}/gold/models/" "Step 6: Trained models (S3)"

echo ""

# Checkpoints (optional)
if [ "$SKIP_CHECKPOINTS" = false ]; then
    echo "--- Checkpoints ---"
    delete_s3_path "s3://${S3_REPO_BUCKET}/pipeline_checkpoints/" "All step checkpoints (1b, 4, 6, etc.)"
    delete_s3_path "s3://${S3_REPO_BUCKET}/pgx-pipeline-status/create_cohort/" "Step 2: Cohort creation checkpoints"
    delete_s3_path "s3://${S3_REPO_BUCKET}/pgx-pipeline-status/feature_importance_eda/" "Step 3b: Feature importance checkpoints"
    delete_s3_path "s3://${S3_REPO_BUCKET}/pgx-pipeline-status/model_data/" "Step 4a: Model data checkpoints"
    delete_s3_path "s3://${S3_REPO_BUCKET}/pgx-pipeline-status/final_model/" "Step 6: Model training checkpoints"
fi

log_message ""
log_message "=========================================="
log_message "Cleanup completed: $(date)"
log_message "=========================================="
log_message "Deleted $DELETED_COUNT items"
log_message ""
log_message "Log file saved to: $LOG_FILE"
log_message ""

echo ""
echo "=========================================="
echo -e "${GREEN}Cleanup completed!${NC}"
echo "=========================================="
echo "Deleted $DELETED_COUNT items"
echo ""
echo -e "${GREEN}Log file saved to: ${LOG_FILE}${NC}"
echo ""
echo "Next steps:"
echo "  1. Rerun Step 2 to create cohorts with new time-windowed logic:"
echo "     python 2_create_cohort/0_create_cohort.py --age-band <age_band> --event-year <year> --cohort ed_non_opioid"
echo ""
echo "  2. Rerun Step 3b for feature importance:"
echo "     python 3b_feature_importance_eda/run_feature_importance_eda.py --cohort ed_non_opioid --age-band <age_band>"
echo ""
echo "  3. Rerun Step 4a to create model data:"
echo "     python 4a_model_data/create_model_data.py --cohort ed_non_opioid --age-band <age_band>"
echo ""
echo "  4. Rerun Step 6 to train models:"
echo "     python 6_final_model/train_models.py --cohort ed_non_opioid --age-band <age_band>"
echo ""
