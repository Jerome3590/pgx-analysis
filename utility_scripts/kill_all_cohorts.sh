#!/bin/bash
# Safely stop all running cohort workflows

echo "Stopping all running cohort workflows..."
echo ""

# Find all running cohort processes
PIDS=$(ps aux | grep -E "run_cohort_workflow\.sh|run_final_model|filter_protocol|run_analysis|run_shap|run_full_ffa|combine_shap_ffa" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "No cohort processes found running."
    exit 0
fi

echo "Found processes:"
ps aux | grep -E "run_cohort_workflow\.sh|run_final_model|filter_protocol|run_analysis|run_shap|run_full_ffa|combine_shap_ffa" | grep -v grep
echo ""

read -p "Kill these processes? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "$PIDS" | xargs kill -TERM 2>/dev/null
    sleep 2
    # Force kill if still running
    REMAINING=$(ps aux | grep -E "run_cohort_workflow\.sh|run_final_model|filter_protocol|run_analysis|run_shap|run_full_ffa|combine_shap_ffa" | grep -v grep | awk '{print $2}')
    if [ ! -z "$REMAINING" ]; then
        echo "Force killing remaining processes..."
        echo "$REMAINING" | xargs kill -KILL 2>/dev/null
    fi
    echo "Done."
else
    echo "Cancelled."
fi

