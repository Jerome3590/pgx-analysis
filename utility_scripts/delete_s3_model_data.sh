#!/usr/bin/env bash
# Delete model_events.parquet files from S3 that are missing controls
# This should be run BEFORE regenerating files with controls

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

S3_BUCKET="pgxdatalake"
S3_PREFIX="gold/cohorts_model_data"

# Parse arguments
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Delete S3 Model Data (Missing Controls)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "S3 Bucket: ${YELLOW}${S3_BUCKET}${NC}"
echo -e "Prefix: ${YELLOW}${S3_PREFIX}${NC}"
echo -e "Mode: ${YELLOW}${DRY_RUN:+DRY RUN (preview only)}${DRY_RUN:-LIVE (will delete)}${NC}"
echo ""

if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI not found. Please install AWS CLI.${NC}"
    exit 1
fi

# List all model_events.parquet files in S3
echo -e "${GREEN}Scanning S3 for model_events.parquet files...${NC}"
FILES=$(aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/" --recursive | grep "model_events.parquet" | awk '{print $4}')

if [ -z "$FILES" ]; then
    echo -e "${YELLOW}No model_events.parquet files found in S3.${NC}"
    exit 0
fi

FILE_COUNT=$(echo "$FILES" | wc -l)
echo -e "${GREEN}Found ${FILE_COUNT} file(s)${NC}"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN - Files that would be deleted:${NC}"
    echo "$FILES" | while read -r file; do
        echo -e "  ${YELLOW}s3://${S3_BUCKET}/${file}${NC}"
    done
    echo ""
    echo -e "${GREEN}DRY RUN: No files were deleted.${NC}"
    echo -e "${GREEN}Run without --dry-run to actually delete these files.${NC}"
    exit 0
fi

# Confirm deletion
echo -e "${RED}WARNING: This will delete ${FILE_COUNT} file(s) from S3!${NC}"
echo -e "${YELLOW}These files will be regenerated with controls by Step 4a.${NC}"
echo ""
read -p "Are you sure you want to continue? (yes/no): " -r
echo ""

if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo -e "${GREEN}Cancelled. No files were deleted.${NC}"
    exit 0
fi

# Delete files
DELETED=0
FAILED=0

echo "$FILES" | while read -r file; do
    s3_uri="s3://${S3_BUCKET}/${file}"
    if aws s3 rm "$s3_uri" 2>/dev/null; then
        echo -e "${GREEN}✓ Deleted: ${s3_uri}${NC}"
        ((DELETED++))
    else
        echo -e "${RED}✗ Failed to delete: ${s3_uri}${NC}"
        ((FAILED++))
    fi
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Deletion complete!${NC}"
echo -e "  Deleted: ${DELETED}"
echo -e "  Failed: ${FAILED}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Run: python 4a_model_data/create_model_data.py"
echo -e "  2. This will regenerate model_events.parquet files with controls"
echo -e "  3. Files will be uploaded to S3 automatically"
echo -e "${BLUE}========================================${NC}"

