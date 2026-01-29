#!/bin/bash
# Run multiple Jupyter notebooks in parallel using nbconvert
# Usage: ./run_notebooks_parallel.sh

set -e

PROJECT_ROOT="/home/pgx3874/pgx-analysis"
cd "$PROJECT_ROOT"

# Notebooks to run (cohorts 5, 6, 7)
NOTEBOOKS=(
    "3b_feature_importance_eda/step3b_interactive_analysis_cohort5.ipynb"
    "3b_feature_importance_eda/step3b_interactive_analysis_cohort6.ipynb"
    "3b_feature_importance_eda/step3b_interactive_analysis_cohort7.ipynb"
)

# Create log directory
LOG_DIR="$PROJECT_ROOT/3b_feature_importance_eda/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Running Multiple Notebooks in Parallel"
echo "=========================================="
echo "Notebooks: ${NOTEBOOKS[@]}"
echo "Log directory: $LOG_DIR"
echo ""

# Function to run a single notebook
run_notebook() {
    local notebook="$1"
    local notebook_name=$(basename "$notebook" .ipynb)
    local log_file="$LOG_DIR/${notebook_name}.log"
    local output_dir="$PROJECT_ROOT/3b_feature_importance_eda/outputs/${notebook_name}"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $notebook_name" | tee -a "$log_file"
    
    # Use jupyter nbconvert to execute notebook
    jupyter nbconvert \
        --to notebook \
        --execute \
        --inplace \
        --ExecutePreprocessor.timeout=3600 \
        --ExecutePreprocessor.kernel_name=python3 \
        "$notebook" \
        2>&1 | tee -a "$log_file"
    
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Completed $notebook_name" | tee -a "$log_file"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Failed $notebook_name (exit code: $exit_code)" | tee -a "$log_file"
    fi
    
    return $exit_code
}

# Export function so it can be used by parallel processes
export -f run_notebook
export PROJECT_ROOT LOG_DIR

# Run notebooks in parallel (limit to 3 concurrent jobs)
echo "Starting parallel execution..."
for notebook in "${NOTEBOOKS[@]}"; do
    run_notebook "$notebook" &
done

# Wait for all background jobs to complete
wait

echo ""
echo "=========================================="
echo "All notebooks completed"
echo "=========================================="
echo "Check logs in: $LOG_DIR"
