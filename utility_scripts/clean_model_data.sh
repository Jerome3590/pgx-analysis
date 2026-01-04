#!/usr/bin/env bash
# Clean model_events.parquet files to force rebuild with controls
# This is useful when gold data paths change or controls need to be regenerated

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DRY_RUN=false
COHORT_FILTER=""
AGE_BAND_FILTER=""
DATA_ROOT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --cohort)
            COHORT_FILTER="$2"
            shift 2
            ;;
        --age-band)
            AGE_BAND_FILTER="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Clean model_events.parquet files to force rebuild with controls."
            echo ""
            echo "Options:"
            echo "  --dry-run              Show what would be deleted without actually deleting"
            echo "  --cohort COHORT        Only clean files for specific cohort (e.g., opioid_ed)"
            echo "  --age-band AGE_BAND    Only clean files for specific age band (e.g., 13-24)"
            echo "  --data-root PATH       Custom data root path (default: auto-detect)"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --dry-run                                    # Preview all deletions"
            echo "  $0 --cohort opioid_ed                          # Clean all opioid_ed files"
            echo "  $0 --cohort opioid_ed --age-band 13-24        # Clean specific cohort/age_band"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Detect data root (same logic as Python scripts)
if [ -z "$DATA_ROOT" ]; then
    # Check if we're on Linux/EC2
    if [ -d "/mnt/nvme" ]; then
        DATA_ROOT="/mnt/nvme"
    else
        # Default to project root (Windows/local dev)
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
        DATA_ROOT="$PROJECT_ROOT"
    fi
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Model Data Cleanup Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Data root: ${YELLOW}$DATA_ROOT${NC}"
echo -e "Cohort filter: ${YELLOW}${COHORT_FILTER:-all}${NC}"
echo -e "Age band filter: ${YELLOW}${AGE_BAND_FILTER:-all}${NC}"
echo -e "Mode: ${YELLOW}${DRY_RUN:+DRY RUN (preview only)}${DRY_RUN:-LIVE (will delete)}${NC}"
echo ""

# Find model_events.parquet files
CANDIDATE_ROOTS=(
    "$DATA_ROOT/4a_model_data"
    "$DATA_ROOT/../4a_model_data"  # If DATA_ROOT is /mnt/nvme, check project root
    "$(pwd)/4a_model_data"  # Current directory
)

MODEL_DATA_DIR=""
for root in "${CANDIDATE_ROOTS[@]}"; do
    if [ -d "$root" ]; then
        MODEL_DATA_DIR="$root"
        break
    fi
done

if [ -z "$MODEL_DATA_DIR" ]; then
    echo -e "${RED}Error: Could not find 4a_model_data directory${NC}"
    echo "Searched in:"
    for root in "${CANDIDATE_ROOTS[@]}"; do
        echo "  - $root"
    done
    exit 1
fi

echo -e "${GREEN}Found model data directory: $MODEL_DATA_DIR${NC}"
echo ""

# Build find command
FIND_CMD="find \"$MODEL_DATA_DIR\" -name \"model_events.parquet\" -type f"

# Apply filters
if [ -n "$COHORT_FILTER" ]; then
    FIND_CMD="$FIND_CMD | grep \"cohort_name=$COHORT_FILTER\""
fi

if [ -n "$AGE_BAND_FILTER" ]; then
    FIND_CMD="$FIND_CMD | grep \"age_band=$AGE_BAND_FILTER\""
fi

# Execute find and collect files
FILES_TO_DELETE=()
while IFS= read -r file; do
    if [ -n "$file" ] && [ -f "$file" ]; then
        FILES_TO_DELETE+=("$file")
    fi
done < <(eval "$FIND_CMD")

if [ ${#FILES_TO_DELETE[@]} -eq 0 ]; then
    echo -e "${YELLOW}No model_events.parquet files found matching criteria.${NC}"
    exit 0
fi

echo -e "${BLUE}Found ${#FILES_TO_DELETE[@]} file(s) to clean:${NC}"
echo ""
for file in "${FILES_TO_DELETE[@]}"; do
    # Extract cohort and age_band from path
    cohort=$(echo "$file" | grep -oP 'cohort_name=\K[^/]+' || echo "unknown")
    age_band=$(echo "$file" | grep -oP 'age_band=\K[^/]+' || echo "unknown")
    
    # Get file size
    if command -v du &> /dev/null; then
        size=$(du -h "$file" | cut -f1)
    else
        size="unknown"
    fi
    
    echo -e "  ${YELLOW}$file${NC}"
    echo -e "    Cohort: $cohort, Age band: $age_band, Size: $size"
done
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${GREEN}DRY RUN: No files were deleted.${NC}"
    echo -e "${GREEN}Run without --dry-run to actually delete these files.${NC}"
    exit 0
fi

# Confirm deletion
echo -e "${RED}WARNING: This will delete ${#FILES_TO_DELETE[@]} file(s)!${NC}"
echo -e "${YELLOW}These files will be regenerated when you run Step 4a (create_model_data.py).${NC}"
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

for file in "${FILES_TO_DELETE[@]}"; do
    if rm -f "$file"; then
        echo -e "${GREEN}✓ Deleted: $file${NC}"
        ((DELETED++))
    else
        echo -e "${RED}✗ Failed to delete: $file${NC}"
        ((FAILED++))
    fi
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Cleanup complete!${NC}"
echo -e "  Deleted: $DELETED"
echo -e "  Failed: $FAILED"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Run: python 4a_model_data/create_model_data.py"
echo -e "  2. This will regenerate model_events.parquet files with controls"
echo -e "${BLUE}========================================${NC}"

