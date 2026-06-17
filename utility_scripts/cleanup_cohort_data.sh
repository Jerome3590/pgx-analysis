#!/usr/bin/env bash
#
# Project-specific cleanup for pgx-analysis.
#
# Clears generated pipeline artifacts for the opioid_ed/non_opioid_ed workflow:
# - Whitelisted pgx-analysis S3 artifact prefixes
# - Whitelisted pgx-analysis S3 checkpoint prefixes
# - Generated local project output folders under this repository root
# - Generated EC2/NVMe project output folders under /mnt/nvme/pgx-analysis/
#
# IMPORTANT: This script only deletes local paths under this repository root or the
# project-specific NVMe root, and only deletes S3 paths from the explicit pgx-analysis
# allowlist below. It does not delete shared EC2/NVMe paths such as /mnt/nvme/gold/medical,
# /mnt/nvme/gold/pharmacy, /mnt/nvme/gold/cohorts, /mnt/nvme/cohorts_staging, or
# /mnt/nvme/4_model_data.
#
# Usage:
#   ./utility_scripts/cleanup_cohort_data.sh [--skip-checkpoints] [--skip-s3] [--skip-local] [--clear-feature-importance] [--yes]
#
# Defaults:
#   PGX_S3_BUCKET=pgxdatalake
#   PGX_CHECKPOINT_BUCKET=pgx-repository
#   PGX_PROJECT_SLUG=pgx-analysis
#   PGX_NVME_ROOT=/mnt/nvme
#
# By default, Step 3 feature-importance outputs are preserved so downstream steps can reuse
# existing selected features. Pass --clear-feature-importance for a full Step 3 recompute.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SKIP_CHECKPOINTS=false
SKIP_S3=false
SKIP_LOCAL=false
CLEAR_FEATURE_IMPORTANCE=false
AUTO_CONFIRM=false

usage() {
    echo "Usage: $0 [--skip-checkpoints] [--skip-s3] [--skip-local] [--clear-feature-importance] [--yes]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
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
        --skip-feature-importance)
            # Backward compatible no-op; preserving feature importance is the default.
            shift
            ;;
        --clear-feature-importance)
            CLEAR_FEATURE_IMPORTANCE=true
            shift
            ;;
        --yes)
            AUTO_CONFIRM=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_NAME="pgx-analysis"
PROJECT_SLUG="${PGX_PROJECT_SLUG:-pgx-analysis}"
S3_BUCKET="${PGX_S3_BUCKET:-pgxdatalake}"
CHECKPOINT_BUCKET="${PGX_CHECKPOINT_BUCKET:-pgx-repository}"
NVME_ROOT="${PGX_NVME_ROOT:-/mnt/nvme}"
PROJECT_NVME_ROOT="${PGX_PROJECT_NVME_ROOT:-${NVME_ROOT}/${PROJECT_SLUG}}"

if [ "$PROJECT_NVME_ROOT" = "/" ] || [ "$PROJECT_NVME_ROOT" = "$NVME_ROOT" ]; then
    echo "Invalid project NVMe root: $PROJECT_NVME_ROOT"
    echo "Set PGX_PROJECT_NVME_ROOT to a project-specific child, for example: ${NVME_ROOT}/${PROJECT_SLUG}"
    exit 1
fi

COHORTS=("opioid_ed" "non_opioid_ed")
AGE_BANDS=("0-12" "13-24" "25-44" "45-54" "55-64" "65-74" "75-84" "85-114")

LOG_FILE="${PROJECT_ROOT}/cleanup_cohort_data_$(date +%Y%m%d_%H%M%S).log"
DELETED_COUNT=0

PROJECT_OUTPUT_DIRS=(
    "2_create_cohort/cohort_metrics"
    "4_model_data/cohort_name=opioid_ed"
    "4_model_data/cohort_name=non_opioid_ed"
    "4a_model_data/cohort_name=opioid_ed"
    "4a_model_data/cohort_name=non_opioid_ed"
    "5_pgx_analysis/outputs"
    "6_final_model/outputs"
    "6_final_model/model_outputs"
    "7_shap_analysis/outputs"
    "8_ffa_analysis/results"
    "10_risk_dashboard/outputs"
    "feature_encoding_outputs"
    "logs"
)

FEATURE_IMPORTANCE_LOCAL_DIRS=(
    "3a_feature_importance/outputs"
    "3b_feature_importance_eda/outputs"
    "3_feature_importance/outputs"
)

LOCAL_ALWAYS_CLEAN=(
    "${PROJECT_ROOT}/data/gold/cohorts"
    "${PROJECT_NVME_ROOT}/gold/cohorts"
    "${PROJECT_NVME_ROOT}/cohorts_staging"
    "${PROJECT_NVME_ROOT}/4_model_data"
    "${PROJECT_NVME_ROOT}/4a_model_data"
)

S3_ALWAYS_CLEAN=(
    "s3://${S3_BUCKET}/gold/cohorts/"
    "s3://${S3_BUCKET}/gold/cohorts_F1120/"
    "s3://${S3_BUCKET}/gold/cohorts_model_data/"
    "s3://${S3_BUCKET}/gold/cohorts/input_model_data/"
    "s3://${S3_BUCKET}/gold/4a_model_data/"
    "s3://${S3_BUCKET}/gold/model_data/"
    "s3://${S3_BUCKET}/gold/event_filter/"
    "s3://${S3_BUCKET}/gold/pgx_features/"
    "s3://${S3_BUCKET}/gold/final_model/"
    "s3://${S3_BUCKET}/gold/models/"
    "s3://${S3_BUCKET}/gold/shap_analysis/"
    "s3://${S3_BUCKET}/gold/ffa_analysis/"
    "s3://${S3_BUCKET}/gold/combined_analysis/"
)

S3_PRESERVED_BY_DEFAULT=(
    "s3://${S3_BUCKET}/gold/feature_importance/"
    "s3://${S3_BUCKET}/gold/bupar/"
)

S3_CHECKPOINTS=(
    "s3://${CHECKPOINT_BUCKET}/pipeline_checkpoints/"
    "s3://${CHECKPOINT_BUCKET}/pgx-pipeline-status/create_cohort/"
    "s3://${CHECKPOINT_BUCKET}/pgx-pipeline-status/model_data/"
    "s3://${CHECKPOINT_BUCKET}/pgx-pipeline-status/final_model/"
)

S3_FEATURE_IMPORTANCE_CHECKPOINTS=(
    "s3://${CHECKPOINT_BUCKET}/pgx-pipeline-status/feature_importance_eda/"
)

log_message() {
    echo "$1" | tee -a "$LOG_FILE"
}

increment_deleted_count() {
    DELETED_COUNT=$((DELETED_COUNT + 1))
}

array_contains_prefix() {
    local path=$1
    shift
    local prefix

    for prefix in "$@"; do
        case "$path" in
            "$prefix"|"$prefix"*) return 0 ;;
        esac
    done
    return 1
}

is_project_s3_path() {
    local path=$1

    array_contains_prefix \
        "$path" \
        "${S3_ALWAYS_CLEAN[@]}" \
        "${S3_PRESERVED_BY_DEFAULT[@]}" \
        "${S3_CHECKPOINTS[@]}" \
        "${S3_FEATURE_IMPORTANCE_CHECKPOINTS[@]}"
}

is_project_local_path() {
    local path=$1

    case "$path" in
        "${PROJECT_ROOT}/"*|"${PROJECT_NVME_ROOT}/"*) return 0 ;;
        *) return 1 ;;
    esac
}

check_s3_path() {
    local path=$1
    local description=$2

    if [ "$SKIP_S3" = true ]; then
        log_message "[SKIP S3] $description"
        return 0
    fi

    set +e
    aws s3 ls "$path" &>/dev/null
    local ls_status=$?
    set -e

    if [ "$ls_status" -eq 0 ]; then
        set +e
        local count
        count=$(aws s3 ls "$path" --recursive 2>/dev/null | wc -l | tr -d ' ')
        local size
        size=$(aws s3 ls "$path" --recursive --summarize 2>/dev/null | awk '/Total Size/ {print $3}')
        set -e
        log_message "[S3 EXISTS] $description"
        log_message "           Path: $path"
        log_message "           Files: ${count:-unknown}"
        [ -n "${size:-}" ] && log_message "           Size: ${size} bytes"
    else
        log_message "[S3 MISSING] $description"
        log_message "            Path: $path"
    fi
}

check_local_path() {
    local path=$1
    local description=$2

    if [ "$SKIP_LOCAL" = true ]; then
        log_message "[SKIP LOCAL] $description"
        return 0
    fi

    if [ -d "$path" ] || [ -f "$path" ]; then
        log_message "[LOCAL EXISTS] $description"
        log_message "              Path: $path"
        if [ -d "$path" ]; then
            local count
            count=$(find "$path" -type f 2>/dev/null | wc -l | tr -d ' ')
            local size
            size=$(du -sh "$path" 2>/dev/null | awk '{print $1}')
            log_message "              Files: ${count:-unknown}"
            [ -n "${size:-}" ] && log_message "              Size: $size"
        else
            local size
            size=$(du -sh "$path" 2>/dev/null | awk '{print $1}')
            [ -n "${size:-}" ] && log_message "              Size: $size"
        fi
    else
        log_message "[LOCAL MISSING] $description"
        log_message "               Path: $path"
    fi
}

delete_s3_path() {
    local path=$1
    local description=$2

    if [ "$SKIP_S3" = true ]; then
        echo -e "${YELLOW}[SKIP S3]${NC} $description"
        return 0
    fi

    if ! is_project_s3_path "$path"; then
        echo -e "${RED}[S3 BLOCKED]${NC} Refusing to delete non-project S3 path: $path"
        log_message "[S3 BLOCKED] $description"
        log_message "             Path: $path"
        log_message "             Reason: outside pgx-analysis S3 allowlist"
        log_message ""
        return 1
    fi

    echo -e "${YELLOW}[S3]${NC} Deleting: $description"
    log_message "[S3 DELETE] $description"
    log_message "           Path: $path"

    set +e
    aws s3 ls "$path" &>/dev/null
    local ls_status=$?
    set -e

    if [ "$ls_status" -ne 0 ]; then
        echo -e "${YELLOW}[S3]${NC} Path not found: $description"
        log_message "           Status: NOT FOUND"
        log_message ""
        return 0
    fi

    set +e
    aws s3 rm "$path" --recursive
    local delete_status=$?
    set -e

    if [ "$delete_status" -eq 0 ]; then
        echo -e "${GREEN}[S3]${NC} Deleted: $description"
        log_message "           Status: DELETED"
        increment_deleted_count
    else
        echo -e "${YELLOW}[S3]${NC} Deletion may have failed: $description"
        log_message "           Status: DELETION ATTEMPTED (exit code: $delete_status)"
    fi
    log_message ""
}

delete_local_path() {
    local path=$1
    local description=$2

    if [ "$SKIP_LOCAL" = true ]; then
        echo -e "${YELLOW}[SKIP LOCAL]${NC} $description"
        return 0
    fi

    if ! is_project_local_path "$path"; then
        echo -e "${RED}[LOCAL BLOCKED]${NC} Refusing to delete non-project local path: $path"
        log_message "[LOCAL BLOCKED] $description"
        log_message "                Path: $path"
        log_message "                Reason: outside project root and project NVMe root"
        log_message ""
        return 1
    fi

    echo -e "${YELLOW}[LOCAL]${NC} Deleting: $description"
    log_message "[LOCAL DELETE] $description"
    log_message "              Path: $path"

    if [ ! -d "$path" ] && [ ! -f "$path" ]; then
        echo -e "${YELLOW}[LOCAL]${NC} Path not found: $description"
        log_message "              Status: NOT FOUND"
        log_message ""
        return 0
    fi

    rm -rf "$path"
    echo -e "${GREEN}[LOCAL]${NC} Deleted: $description"
    log_message "              Status: DELETED"
    increment_deleted_count
    log_message ""
}

print_summary() {
    echo "=========================================="
    echo "PGx Analysis Cleanup"
    echo "=========================================="
    echo "Project root:      $PROJECT_ROOT"
    echo "Project slug:      $PROJECT_SLUG"
    echo "S3 bucket:         $S3_BUCKET"
    echo "Checkpoint bucket: $CHECKPOINT_BUCKET"
    echo "Project NVMe root: $PROJECT_NVME_ROOT"
    echo ""
    echo "This script will clear generated artifacts for cohorts: ${COHORTS[*]}"
    echo "Age bands: ${AGE_BANDS[*]}"
    echo ""
    echo "Preserved always:"
    echo "  - Any local path outside ${PROJECT_ROOT} and ${PROJECT_NVME_ROOT}"
    echo "  - Any S3 path outside the pgx-analysis S3 allowlist in this script"
    if [ "$CLEAR_FEATURE_IMPORTANCE" = false ]; then
        echo "  - Step 3 feature-importance outputs and checkpoints"
    fi
    echo ""
    echo -e "${YELLOW}WARNING: This deletes generated local and S3 data for this project.${NC}"
    echo ""
}

print_summary

if [ "$AUTO_CONFIRM" = false ]; then
    read -r -p "Are you sure you want to continue? Type yes: " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Cleanup cancelled."
        exit 0
    fi
else
    echo "Auto-confirmation enabled (--yes). Proceeding with cleanup..."
fi

log_message "=========================================="
log_message "${PROJECT_NAME} Cleanup Log"
log_message "Started: $(date)"
log_message "Project root: $PROJECT_ROOT"
log_message "Project slug: $PROJECT_SLUG"
log_message "S3 bucket: $S3_BUCKET"
log_message "Checkpoint bucket: $CHECKPOINT_BUCKET"
log_message "=========================================="
log_message ""

echo ""
echo "=========================================="
echo "Scanning existing data..."
echo "=========================================="
log_message "--- Scanning existing data ---"

for path in "${S3_ALWAYS_CLEAN[@]}"; do
    check_s3_path "$path" "Project artifact: $path"
done

if [ "$CLEAR_FEATURE_IMPORTANCE" = true ]; then
    for path in "${S3_PRESERVED_BY_DEFAULT[@]}"; do
        check_s3_path "$path" "Feature importance artifact: $path"
    done
fi

if [ "$SKIP_CHECKPOINTS" = false ]; then
    for path in "${S3_CHECKPOINTS[@]}"; do
        check_s3_path "$path" "Project checkpoint: $path"
    done

    if [ "$CLEAR_FEATURE_IMPORTANCE" = true ]; then
        for path in "${S3_FEATURE_IMPORTANCE_CHECKPOINTS[@]}"; do
            check_s3_path "$path" "Feature importance checkpoint: $path"
        done
    fi
fi

for path in "${LOCAL_ALWAYS_CLEAN[@]}"; do
    check_local_path "$path" "Generated local data: $path"
done

for rel_path in "${PROJECT_OUTPUT_DIRS[@]}"; do
    check_local_path "${PROJECT_ROOT}/${rel_path}" "Project output: ${rel_path}"
done

if [ "$CLEAR_FEATURE_IMPORTANCE" = true ]; then
    for rel_path in "${FEATURE_IMPORTANCE_LOCAL_DIRS[@]}"; do
        check_local_path "${PROJECT_ROOT}/${rel_path}" "Feature importance output: ${rel_path}"
    done
fi

log_message ""
log_message "--- End of scan ---"
log_message ""

echo ""
echo "=========================================="
echo "Starting cleanup..."
echo "=========================================="
log_message "--- Starting cleanup ---"

for path in "${S3_ALWAYS_CLEAN[@]}"; do
    delete_s3_path "$path" "Project artifact: $path"
done

if [ "$CLEAR_FEATURE_IMPORTANCE" = true ]; then
    for path in "${S3_PRESERVED_BY_DEFAULT[@]}"; do
        delete_s3_path "$path" "Feature importance artifact: $path"
    done
else
    echo "--- Step 3 Feature Importance (preserved) ---"
    log_message "Step 3 feature-importance outputs preserved; use --clear-feature-importance to clear them."
fi

if [ "$SKIP_CHECKPOINTS" = false ]; then
    for path in "${S3_CHECKPOINTS[@]}"; do
        delete_s3_path "$path" "Project checkpoint: $path"
    done

    if [ "$CLEAR_FEATURE_IMPORTANCE" = true ]; then
        for path in "${S3_FEATURE_IMPORTANCE_CHECKPOINTS[@]}"; do
            delete_s3_path "$path" "Feature importance checkpoint: $path"
        done
    else
        echo "  (preserving feature_importance_eda checkpoint so existing Step 3b work is reusable)"
        log_message "Feature importance checkpoint preserved; use --clear-feature-importance to clear it."
    fi
fi

for path in "${LOCAL_ALWAYS_CLEAN[@]}"; do
    delete_local_path "$path" "Generated local data: $path"
done

for rel_path in "${PROJECT_OUTPUT_DIRS[@]}"; do
    delete_local_path "${PROJECT_ROOT}/${rel_path}" "Project output: ${rel_path}"
done

if [ "$CLEAR_FEATURE_IMPORTANCE" = true ]; then
    for rel_path in "${FEATURE_IMPORTANCE_LOCAL_DIRS[@]}"; do
        delete_local_path "${PROJECT_ROOT}/${rel_path}" "Feature importance output: ${rel_path}"
    done
fi

log_message ""
log_message "=========================================="
log_message "Cleanup completed: $(date)"
log_message "Deleted $DELETED_COUNT item groups"
log_message "Log file saved to: $LOG_FILE"
log_message "=========================================="

echo ""
echo "=========================================="
echo -e "${GREEN}Cleanup completed!${NC}"
echo "=========================================="
echo "Deleted $DELETED_COUNT item groups"
echo "Log file saved to: $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. Re-run Step 2:"
echo "     python 2_create_cohort/0_create_cohort.py --age-band <age_band> --event-year <year> --cohort both"
echo "  2. Re-run Step 3b as needed:"
echo "     python 3b_feature_importance_eda/run_feature_importance_eda.py --cohort opioid_ed --age-band <age_band>"
echo "     python 3b_feature_importance_eda/run_feature_importance_eda.py --cohort non_opioid_ed --age-band <age_band>"
echo "  3. Re-run Step 4:"
echo "     python 4_model_data/create_model_data.py --cohort opioid_ed --age-band <age_band>"
echo "     python 4_model_data/create_model_data.py --cohort non_opioid_ed --age-band <age_band>"
echo "  4. Re-run Step 6:"
echo "     python 6_final_model/train_models.py --cohort opioid_ed --age-band <age_band>"
echo "     python 6_final_model/train_models.py --cohort non_opioid_ed --age-band <age_band>"
