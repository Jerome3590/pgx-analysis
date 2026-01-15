#!/bin/bash
# Sync best models from EC2 local storage to S3
# This script uploads model files from 6_final_model/outputs/ to S3

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# S3 bucket
S3_BUCKET="pgxdatalake"
S3_PROFILE="${AWS_PROFILE:-mushin}"

# Project root (adjust if needed)
PROJECT_ROOT="${PROJECT_ROOT:-/home/pgx3874/pgx-analysis}"
if [ ! -d "$PROJECT_ROOT" ]; then
    # Try alternative locations
    if [ -d "/mnt/nvme/pgx-analysis" ]; then
        PROJECT_ROOT="/mnt/nvme/pgx-analysis"
    elif [ -d "$HOME/pgx-analysis" ]; then
        PROJECT_ROOT="$HOME/pgx-analysis"
    else
        echo -e "${RED}Error: PROJECT_ROOT not found. Please set PROJECT_ROOT environment variable.${NC}"
        exit 1
    fi
fi

MODEL_BASE_DIR="$PROJECT_ROOT/6_final_model/outputs"

# Cohorts and age bands
declare -A COHORTS
COHORTS[opioid_ed]="13-24 25-44 45-54 55-64"
COHORTS[non_opioid_ed]="65-74 75-84 85-94"

echo "================================================================================"
echo "Sync Models from EC2 to S3"
echo "================================================================================"
echo ""
echo "Project Root: $PROJECT_ROOT"
echo "Model Base Dir: $MODEL_BASE_DIR"
echo "S3 Bucket: $S3_BUCKET"
echo "AWS Profile: $S3_PROFILE"
echo ""

# Check if AWS CLI is available
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI not found. Please install AWS CLI.${NC}"
    exit 1
fi

# Check if model base directory exists
if [ ! -d "$MODEL_BASE_DIR" ]; then
    echo -e "${RED}Error: Model base directory not found: $MODEL_BASE_DIR${NC}"
    exit 1
fi

# Function to sync a single file to S3
sync_file_to_s3() {
    local local_file="$1"
    local s3_path="$2"
    local file_type="$3"
    
    if [ ! -f "$local_file" ]; then
        echo -e "${YELLOW}[SKIP] File not found: $local_file${NC}"
        return 1
    fi
    
    # Check if file already exists in S3
    if aws s3 ls "$s3_path" --profile "$S3_PROFILE" &> /dev/null; then
        echo -e "${YELLOW}[SKIP] Already exists in S3: $s3_path${NC}"
        return 0
    fi
    
    # Upload file
    echo -e "${GREEN}[UPLOAD]${NC} $file_type"
    echo "  Local:  $local_file"
    echo "  S3:     $s3_path"
    
    if aws s3 cp "$local_file" "$s3_path" --profile "$S3_PROFILE"; then
        echo -e "${GREEN}[OK]${NC} Uploaded successfully"
        return 0
    else
        echo -e "${RED}[ERROR]${NC} Failed to upload"
        return 1
    fi
}

# Counters
TOTAL_FILES=0
UPLOADED=0
SKIPPED=0
ERRORS=0

# Process each cohort
for cohort in "${!COHORTS[@]}"; do
    echo ""
    echo "================================================================================"
    echo "Processing Cohort: $cohort"
    echo "================================================================================"
    
    age_bands=(${COHORTS[$cohort]})
    
    for age_band in "${age_bands[@]}"; do
        age_band_fname=$(echo "$age_band" | tr '-' '_')
        
        echo ""
        echo "Age Band: $age_band ($age_band_fname)"
        
        # Local directory
        local_dir="$MODEL_BASE_DIR/$cohort/$age_band_fname/final_model_json"
        
        if [ ! -d "$local_dir" ]; then
            echo -e "${YELLOW}[SKIP] Directory not found: $local_dir${NC}"
            continue
        fi
        
        # Files to sync
        declare -a files_to_sync=(
            "$local_dir/${cohort}_${age_band_fname}_best_xgboost_model.json|gold/final_model/${cohort}/${age_band}/${cohort}_${age_band_fname}_best_xgboost_model.json|Best XGBoost JSON"
            "$local_dir/${cohort}_${age_band_fname}_best_catboost_model.json|gold/final_model/${cohort}/${age_band}/${cohort}_${age_band_fname}_best_catboost_model.json|Best CatBoost JSON"
            "$local_dir/${cohort}_${age_band_fname}_best_catboost_model.cbm|gold/final_model/${cohort}/${age_band}/${cohort}_${age_band_fname}_best_catboost_model.cbm|Best CatBoost Binary"
        )
        
        # Also sync model selection metadata
        metadata_local="$MODEL_BASE_DIR/$cohort/$age_band_fname/${cohort}_${age_band_fname}_model_selection_metadata.json"
        metadata_s3="gold/final_model/${cohort}/${age_band}/${cohort}_${age_band_fname}_model_selection_metadata.json"
        if [ -f "$metadata_local" ]; then
            files_to_sync+=("$metadata_local|$metadata_s3|Model Selection Metadata")
        fi
        
        # Sync each file
        for file_info in "${files_to_sync[@]}"; do
            IFS='|' read -r local_file s3_key file_type <<< "$file_info"
            s3_path="s3://${S3_BUCKET}/${s3_key}"
            
            TOTAL_FILES=$((TOTAL_FILES + 1))
            
            if sync_file_to_s3 "$local_file" "$s3_path" "$file_type"; then
                if aws s3 ls "$s3_path" --profile "$S3_PROFILE" &> /dev/null; then
                    SKIPPED=$((SKIPPED + 1))
                else
                    UPLOADED=$((UPLOADED + 1))
                fi
            else
                ERRORS=$((ERRORS + 1))
            fi
        done
    done
done

# Summary
echo ""
echo "================================================================================"
echo "Summary"
echo "================================================================================"
echo "Total files processed: $TOTAL_FILES"
echo -e "${GREEN}Uploaded: $UPLOADED${NC}"
echo -e "${YELLOW}Skipped (already exists): $SKIPPED${NC}"
echo -e "${RED}Errors: $ERRORS${NC}"
echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All files synced successfully!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some errors occurred. Please check the output above.${NC}"
    exit 1
fi
