#!/bin/bash
# Clear Step 8 (FFA Analysis) outputs to allow workflow restart
# Removes: local files, S3 checkpoints, and S3 outputs

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
COHORT_NAME="${1:-opioid_ed}"
AGE_BAND="${2:-13-24}"
MODEL_TYPE="${3:-xgboost}"
DRY_RUN="${4:-false}"

AGE_BAND_FNAME=$(echo "$AGE_BAND" | tr '-' '_')

echo "=========================================="
echo "Clear Step 8 (FFA Analysis) Outputs"
echo "=========================================="
echo "Cohort: $COHORT_NAME"
echo "Age Band: $AGE_BAND"
echo "Model Type: $MODEL_TYPE"
echo "Mode: $([ "$DRY_RUN" = "true" ] && echo "DRY RUN (preview only)" || echo "DELETE")"
echo "=========================================="
echo

# Local files directory
LOCAL_OUTPUT_DIR="$PROJECT_ROOT/8_ffa_analysis/outputs/$COHORT_NAME/$AGE_BAND_FNAME/$MODEL_TYPE"
LOCAL_FILES=(
    "axp_explanations.parquet"
    "feature_importance_axp.parquet"
    "causal_importance.parquet"
    "interaction_analysis.parquet"
    "analysis_summary.json"
)

# S3 paths
S3_CHECKPOINT="s3://pgx-repository/pipeline_checkpoints/8_ffa_analysis/$COHORT_NAME/$AGE_BAND_FNAME/checkpoint.json"
S3_OUTPUT_BASE="s3://pgxdatalake/gold/ffa_analysis/$COHORT_NAME/$AGE_BAND/$MODEL_TYPE"
S3_FILES=(
    "axp_explanations.parquet"
    "feature_importance_axp.parquet"
    "causal_importance.parquet"
    "interaction_analysis.parquet"
)

DELETED=0
SKIPPED=0

# Function to check if AWS CLI is available
check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        echo -e "${YELLOW}[WARNING] AWS CLI not found. S3 operations will be skipped.${NC}"
        return 1
    fi
    return 0
}

# Function to delete S3 file
delete_s3_file() {
    local s3_path=$1
    if [ "$DRY_RUN" = "true" ]; then
        echo -e "${YELLOW}[DRY RUN] Would delete: $s3_path${NC}"
        return 0
    fi
    
    if aws s3 ls "$s3_path" &>/dev/null; then
        if aws s3 rm "$s3_path" 2>&1; then
            echo -e "${GREEN}✓ Deleted: $s3_path${NC}"
            DELETED=$((DELETED + 1))
            return 0
        else
            echo -e "${RED}✗ Failed to delete: $s3_path${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}[SKIP] Not found: $s3_path${NC}"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi
}

# Clear local files
echo "LOCAL FILES"
echo "----------------------------------------"
if [ -d "$LOCAL_OUTPUT_DIR" ]; then
    for file in "${LOCAL_FILES[@]}"; do
        file_path="$LOCAL_OUTPUT_DIR/$file"
        if [ -f "$file_path" ]; then
            if [ "$DRY_RUN" = "true" ]; then
                echo -e "${YELLOW}[DRY RUN] Would delete: $file_path${NC}"
            else
                if rm -f "$file_path"; then
                    echo -e "${GREEN}✓ Deleted: $file${NC}"
                    DELETED=$((DELETED + 1))
                else
                    echo -e "${RED}✗ Failed to delete: $file${NC}"
                fi
            fi
        else
            echo -e "${YELLOW}[SKIP] Not found: $file${NC}"
            SKIPPED=$((SKIPPED + 1))
        fi
    done
else
    echo -e "${YELLOW}[SKIP] Directory does not exist: $LOCAL_OUTPUT_DIR${NC}"
fi
echo

# Clear S3 checkpoint
echo "S3 CHECKPOINT"
echo "----------------------------------------"
if check_aws_cli; then
    delete_s3_file "$S3_CHECKPOINT"
else
    echo -e "${YELLOW}[SKIP] AWS CLI not available${NC}"
fi
echo

# Clear S3 output files
echo "S3 OUTPUT FILES"
echo "----------------------------------------"
if check_aws_cli; then
    for file in "${S3_FILES[@]}"; do
        s3_path="$S3_OUTPUT_BASE/$file"
        delete_s3_file "$s3_path"
    done
else
    echo -e "${YELLOW}[SKIP] AWS CLI not available${NC}"
fi
echo

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
if [ "$DRY_RUN" = "true" ]; then
    echo -e "${YELLOW}DRY RUN MODE: No files were actually deleted${NC}"
else
    echo -e "${GREEN}Deleted: $DELETED files${NC}"
    echo -e "${YELLOW}Skipped (not found): $SKIPPED files${NC}"
    echo
    echo "Step 8 outputs cleared. Workflow will restart at Step 8."
fi
echo "=========================================="
