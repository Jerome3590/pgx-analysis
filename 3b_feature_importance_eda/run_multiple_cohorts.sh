#!/bin/bash
# Run multiple cohorts in parallel on EC2
# Usage: ./run_multiple_cohorts.sh

set -e

PROJECT_ROOT="/home/pgx3874/pgx-analysis"
cd "$PROJECT_ROOT"

# Cohorts to run (cohort 5, 6, 7 = non_opioid_ed age bands 65-74, 75-84, 85-94)
COHORTS=(
    "non_opioid_ed:65-74"
    "non_opioid_ed:75-84"
    "non_opioid_ed:85-94"
)

# Create log directory
LOG_DIR="$PROJECT_ROOT/3b_feature_importance_eda/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Running Multiple Cohorts in Parallel"
echo "=========================================="
echo "Cohorts: ${COHORTS[@]}"
echo "Log directory: $LOG_DIR"
echo ""

# Function to run a single cohort
run_cohort() {
    local cohort_age="$1"
    IFS=':' read -r cohort age_band <<< "$cohort_age"
    local log_file="$LOG_DIR/${cohort}_${age_band//-/_}.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $cohort / $age_band" | tee -a "$log_file"
    
    python3 "$PROJECT_ROOT/3b_feature_importance_eda/run_step_3b.py" \
        --cohort "$cohort" \
        --age-band "$age_band" \
        2>&1 | tee -a "$log_file"
    
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Completed $cohort / $age_band" | tee -a "$log_file"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Failed $cohort / $age_band (exit code: $exit_code)" | tee -a "$log_file"
    fi
    
    return $exit_code
}

# Export function so it can be used by parallel processes
export -f run_cohort
export PROJECT_ROOT LOG_DIR

# Run cohorts in parallel (limit to 3 concurrent jobs to avoid resource exhaustion)
echo "Starting parallel execution..."
for cohort_age in "${COHORTS[@]}"; do
    run_cohort "$cohort_age" &
done

# Wait for all background jobs to complete
wait

echo ""
echo "=========================================="
echo "All cohorts completed"
echo "=========================================="
echo "Check logs in: $LOG_DIR"
