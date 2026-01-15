#!/bin/bash
# Run model evaluation for each cohort separately
# This script runs evaluation for each cohort/age_band combination individually
# to manage memory and handle different models

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
SCRIPT_PATH="$PROJECT_ROOT/utility_scripts/evaluate_models_test_data.py"

# Cohorts and age bands
declare -A COHORTS
COHORTS[opioid_ed]="13-24 25-44 45-54 55-64"
COHORTS[non_opioid_ed]="65-74 75-84 85-94"

# Default parameters
N_SHAP_SAMPLES="${N_SHAP_SAMPLES:-1000}"
MODEL_TYPE="${MODEL_TYPE:-both}"

echo "================================================================================"
echo "Model Evaluation - Running Each Cohort Separately"
echo "================================================================================"
echo ""
echo "Project Root: $PROJECT_ROOT"
echo "Script: $SCRIPT_PATH"
echo "SHAP Samples: $N_SHAP_SAMPLES"
echo "Model Type: $MODEL_TYPE"
echo ""

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${RED}Error: Evaluation script not found: $SCRIPT_PATH${NC}"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# Track results
TOTAL=0
SUCCESS=0
FAILED=0

# Process each cohort
for cohort in "${!COHORTS[@]}"; do
    echo ""
    echo "================================================================================"
    echo "Processing Cohort: $cohort"
    echo "================================================================================"
    
    age_bands=(${COHORTS[$cohort]})
    
    for age_band in "${age_bands[@]}"; do
        TOTAL=$((TOTAL + 1))
        
        echo ""
        echo "-------------------------------------------------------------------------------"
        echo "Evaluating: $cohort / $age_band"
        echo "-------------------------------------------------------------------------------"
        echo ""
        
        # Run evaluation for this cohort/age_band
        if python3 "$SCRIPT_PATH" \
            --cohort "$cohort" \
            --age-band "$age_band" \
            --model-type "$MODEL_TYPE" \
            --n-shap-samples "$N_SHAP_SAMPLES"; then
            echo ""
            echo -e "${GREEN}[OK]${NC} Completed evaluation for $cohort/$age_band"
            SUCCESS=$((SUCCESS + 1))
        else
            echo ""
            echo -e "${RED}[ERROR]${NC} Failed evaluation for $cohort/$age_band"
            FAILED=$((FAILED + 1))
        fi
        
        # Small delay between runs to allow cleanup
        sleep 2
    done
done

# Summary
echo ""
echo "================================================================================"
echo "Summary"
echo "================================================================================"
echo "Total cohorts processed: $TOTAL"
echo -e "${GREEN}Successful: $SUCCESS${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All evaluations completed successfully!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some evaluations failed. Check the output above.${NC}"
    exit 1
fi
