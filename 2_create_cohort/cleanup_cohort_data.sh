#!/bin/bash
#
# Cleanup script for cohort data and related outputs
# 
# This script clears:
# - Step 2: Cohort parquet files (S3 and local)
# - Step 3b: Feature importance outputs
# - Step 4a: Model data
# - Step 6: Trained models
# - Checkpoints (optional)
#
# Usage: ./cleanup_cohort_data.sh [--skip-checkpoints] [--skip-s3] [--skip-local]
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
echo "  - Step 3b: Feature importance outputs"
echo "  - Step 4a: Model data"
echo "  - Step 6: Trained models"
if [ "$SKIP_CHECKPOINTS" = false ]; then
    echo "  - Checkpoints (optional)"
fi
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

# Local paths (EC2)
PROJECT_ROOT="${HOME}/pgx-analysis"
DATA_ROOT="/mnt/nvme/gold"
MODEL_DATA_ROOT="/mnt/nvme/4a_model_data"
LOCAL_COHORT_ROOT="${DATA_ROOT}/cohorts"  # If synced locally
STEP3B_OUTPUTS="${PROJECT_ROOT}/3b_feature_importance_eda/outputs"
STEP6_MODELS="${PROJECT_ROOT}/6_final_model/models"

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
    
    if aws s3 ls "$path" &>/dev/null; then
        local size=$(aws s3 ls "$path" --recursive --summarize 2>/dev/null | grep "Total Size" | awk '{print $3, $4}')
        local count=$(aws s3 ls "$path" --recursive 2>/dev/null | wc -l)
        log_message "[S3 EXISTS] $description"
        log_message "           Path: $path"
        log_message "           Files: $count"
        log_message "           Size: $size"
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
    if aws s3 ls "$path" &>/dev/null; then
        local size_before=$(aws s3 ls "$path" --recursive --summarize 2>/dev/null | grep "Total Size" | awk '{print $3, $4}')
        log_message "           Size before: $size_before"
        aws s3 rm "$path" --recursive
        echo -e "${GREEN}[S3]${NC} Deleted: $description"
        log_message "           Status: DELETED"
        ((DELETED_COUNT++))
    else
        echo -e "${YELLOW}[S3]${NC} Path not found (may already be deleted): $description"
        log_message "           Status: NOT FOUND (already deleted or doesn't exist)"
    fi
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
        rm -rf "$path"
        echo -e "${GREEN}[LOCAL]${NC} Deleted: $description"
        log_message "              Status: DELETED"
        ((DELETED_COUNT++))
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
check_s3_path "s3://${S3_BUCKET}/gold/cohorts/cohort_name=ed_non_opioid/" "Step 2: ED_NON_OPIOID cohorts (S3 - new format)"
check_s3_path "s3://${S3_BUCKET}/gold/cohorts/cohort_name=opioid_ed/" "Step 2: OPIOID_ED cohorts (S3 - new format)"
# Legacy paths (old format)
check_s3_path "s3://${S3_BUCKET}/gold/cohorts_F1120/" "Step 2: Legacy cohorts_F1120 (S3 - old format)"
check_s3_path "s3://${S3_BUCKET}/gold/cohorts_model_data/" "Step 2: Legacy cohorts_model_data (S3 - old format)"
check_local_path "${LOCAL_COHORT_ROOT}/cohort_name=ed_non_opioid" "Step 2: ED_NON_OPIOID cohorts (local)"
check_local_path "${LOCAL_COHORT_ROOT}/cohort_name=opioid_ed" "Step 2: OPIOID_ED cohorts (local)"
if [ -d "${PROJECT_ROOT}/data/gold_cohorts" ]; then
    check_local_path "${PROJECT_ROOT}/data/gold_cohorts/cohort_name=ed_non_opioid" "Step 2: ED_NON_OPIOID cohorts (project data)"
    check_local_path "${PROJECT_ROOT}/data/gold_cohorts/cohort_name=opioid_ed" "Step 2: OPIOID_ED cohorts (project data)"
fi

echo ""

# Step 3b: Feature importance outputs
echo "--- Step 3b: Feature Importance Outputs ---"
log_message "--- Step 3b: Feature Importance Outputs ---"
check_local_path "${STEP3B_OUTPUTS}/ed_non_opioid" "Step 3b: ED_NON_OPIOID feature importance"
check_local_path "${STEP3B_OUTPUTS}/opioid_ed" "Step 3b: OPIOID_ED feature importance"
check_s3_path "s3://${S3_BUCKET}/gold/bupar/ed_non_opioid/" "Step 3b: ED_NON_OPIOID BupaR outputs (S3)"
check_s3_path "s3://${S3_BUCKET}/gold/bupar/opioid_ed/" "Step 3b: OPIOID_ED BupaR outputs (S3)"
check_s3_path "s3://${S3_BUCKET}/gold/feature_importance/ed_non_opioid/" "Step 3b: ED_NON_OPIOID feature importance (S3)"
check_s3_path "s3://${S3_BUCKET}/gold/feature_importance/opioid_ed/" "Step 3b: OPIOID_ED feature importance (S3)"

echo ""

# Step 4a: Model data
echo "--- Step 4a: Model Data ---"
log_message "--- Step 4a: Model Data ---"
check_local_path "${MODEL_DATA_ROOT}/cohort_name=ed_non_opioid" "Step 4a: ED_NON_OPIOID model data (NVMe)"
check_local_path "${MODEL_DATA_ROOT}/cohort_name=opioid_ed" "Step 4a: OPIOID_ED model data (NVMe)"
if [ -d "${PROJECT_ROOT}/4a_model_data" ]; then
    check_local_path "${PROJECT_ROOT}/4a_model_data/cohort_name=ed_non_opioid" "Step 4a: ED_NON_OPIOID model data (project)"
    check_local_path "${PROJECT_ROOT}/4a_model_data/cohort_name=opioid_ed" "Step 4a: OPIOID_ED model data (project)"
fi
# New format: s3://pgxdatalake/gold/cohorts/input_model_data
check_s3_path "s3://${S3_BUCKET}/gold/cohorts/input_model_data/cohort_name=ed_non_opioid/" "Step 4a: ED_NON_OPIOID model data (S3 - new format)"
check_s3_path "s3://${S3_BUCKET}/gold/cohorts/input_model_data/cohort_name=opioid_ed/" "Step 4a: OPIOID_ED model data (S3 - new format)"
# Legacy path (old format) - for cleanup
check_s3_path "s3://${S3_BUCKET}/gold/4a_model_data/cohort_name=ed_non_opioid/" "Step 4a: ED_NON_OPIOID model data (S3 - legacy)"
check_s3_path "s3://${S3_BUCKET}/gold/4a_model_data/cohort_name=opioid_ed/" "Step 4a: OPIOID_ED model data (S3 - legacy)"

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
delete_s3_path "s3://${S3_BUCKET}/gold/cohorts/cohort_name=ed_non_opioid/" "Step 2: ED_NON_OPIOID cohorts (S3 - new format)"
delete_s3_path "s3://${S3_BUCKET}/gold/cohorts/cohort_name=opioid_ed/" "Step 2: OPIOID_ED cohorts (S3 - new format)"
# Legacy paths (old format) - for cleanup
delete_s3_path "s3://${S3_BUCKET}/gold/cohorts_F1120/" "Step 2: Legacy cohorts_F1120 (S3 - old format)"
delete_s3_path "s3://${S3_BUCKET}/gold/cohorts_model_data/" "Step 2: Legacy cohorts_model_data (S3 - old format)"
delete_local_path "${LOCAL_COHORT_ROOT}/cohort_name=ed_non_opioid" "Step 2: ED_NON_OPIOID cohorts (local)"
delete_local_path "${LOCAL_COHORT_ROOT}/cohort_name=opioid_ed" "Step 2: OPIOID_ED cohorts (local)"

# Also check project root data directory
if [ -d "${PROJECT_ROOT}/data/gold_cohorts" ]; then
    delete_local_path "${PROJECT_ROOT}/data/gold_cohorts/cohort_name=ed_non_opioid" "Step 2: ED_NON_OPIOID cohorts (project data)"
    delete_local_path "${PROJECT_ROOT}/data/gold_cohorts/cohort_name=opioid_ed" "Step 2: OPIOID_ED cohorts (project data)"
fi

echo ""

# Step 3b: Feature importance outputs
echo "--- Step 3b: Feature Importance Outputs ---"
delete_local_path "${STEP3B_OUTPUTS}/ed_non_opioid" "Step 3b: ED_NON_OPIOID feature importance"
delete_local_path "${STEP3B_OUTPUTS}/opioid_ed" "Step 3b: OPIOID_ED feature importance"
delete_s3_path "s3://${S3_BUCKET}/gold/bupar/ed_non_opioid/" "Step 3b: ED_NON_OPIOID BupaR outputs (S3)"
delete_s3_path "s3://${S3_BUCKET}/gold/bupar/opioid_ed/" "Step 3b: OPIOID_ED BupaR outputs (S3)"
delete_s3_path "s3://${S3_BUCKET}/gold/feature_importance/ed_non_opioid/" "Step 3b: ED_NON_OPIOID feature importance (S3)"
delete_s3_path "s3://${S3_BUCKET}/gold/feature_importance/opioid_ed/" "Step 3b: OPIOID_ED feature importance (S3)"

echo ""

# Step 4a: Model data
echo "--- Step 4a: Model Data ---"
delete_local_path "${MODEL_DATA_ROOT}/cohort_name=ed_non_opioid" "Step 4a: ED_NON_OPIOID model data (NVMe)"
delete_local_path "${MODEL_DATA_ROOT}/cohort_name=opioid_ed" "Step 4a: OPIOID_ED model data (NVMe)"
if [ -d "${PROJECT_ROOT}/4a_model_data" ]; then
    delete_local_path "${PROJECT_ROOT}/4a_model_data/cohort_name=ed_non_opioid" "Step 4a: ED_NON_OPIOID model data (project)"
    delete_local_path "${PROJECT_ROOT}/4a_model_data/cohort_name=opioid_ed" "Step 4a: OPIOID_ED model data (project)"
fi
# New format: s3://pgxdatalake/gold/cohorts/input_model_data
delete_s3_path "s3://${S3_BUCKET}/gold/cohorts/input_model_data/cohort_name=ed_non_opioid/" "Step 4a: ED_NON_OPIOID model data (S3 - new format)"
delete_s3_path "s3://${S3_BUCKET}/gold/cohorts/input_model_data/cohort_name=opioid_ed/" "Step 4a: OPIOID_ED model data (S3 - new format)"
# Legacy path (old format) - for cleanup
delete_s3_path "s3://${S3_BUCKET}/gold/4a_model_data/cohort_name=ed_non_opioid/" "Step 4a: ED_NON_OPIOID model data (S3 - legacy)"
delete_s3_path "s3://${S3_BUCKET}/gold/4a_model_data/cohort_name=opioid_ed/" "Step 4a: OPIOID_ED model data (S3 - legacy)"

echo ""

# Step 6: Trained models
echo "--- Step 6: Trained Models ---"
delete_local_path "${STEP6_MODELS}" "Step 6: All trained models"
delete_s3_path "s3://${S3_BUCKET}/gold/models/" "Step 6: Trained models (S3)"

echo ""

# Checkpoints (optional)
if [ "$SKIP_CHECKPOINTS" = false ]; then
    echo "--- Checkpoints ---"
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
echo "     python 2_create_cohort/0_create_cohort.py --age-band <age_band> --event-year <year> --cohort ed_non_opioid --time-window-days 14"
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
