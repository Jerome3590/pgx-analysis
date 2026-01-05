#!/bin/bash
# Monitor all running cohort workflows

echo "=========================================="
echo "Parallel Cohort Execution Monitor"
echo "=========================================="
echo ""

# Check running cohort workflows
echo "=== Running Cohort Processes ==="
ps aux | grep -E "run_cohort_workflow\.sh|run_final_model|filter_protocol|run_analysis|run_shap|run_full_ffa|combine_shap_ffa" | grep -v grep | while read line; do
    pid=$(echo $line | awk '{print $2}')
    cpu=$(echo $line | awk '{print $3}')
    mem=$(echo $line | awk '{print $4}')
    cmd=$(echo $line | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
    echo "PID: $pid | CPU: ${cpu}% | MEM: ${mem}% | $cmd"
done

echo ""
echo "=== System Resources ==="
echo "CPU Usage:"
top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print "  Idle: " 100 - $1 "%"}'
echo ""
echo "Memory Usage:"
free -h | grep -E "Mem|Swap" | awk '{print "  " $1 ": " $3 " / " $2 " (" $3/$2*100 "%)"}'
echo ""
echo "Disk Usage (NVMe):"
df -h /mnt/nvme 2>/dev/null | tail -1 | awk '{print "  /mnt/nvme: " $3 " / " $2 " (" $5 " used)"}'
echo ""

# Count running processes by step
echo "=== Process Count by Step ==="
echo "  Step 4a (create_model_data): $(ps aux | grep -c 'create_model_data' | grep -v grep || echo 0)"
echo "  Step 4b (filter_protocol): $(ps aux | grep -c 'filter_protocol' | grep -v grep || echo 0)"
echo "  Step 5c (pgx_analysis): $(ps aux | grep -c 'pgx_analysis' | grep -v grep || echo 0)"
echo "  Step 6 (run_final_model): $(ps aux | grep -c 'run_final_model' | grep -v grep || echo 0)"
echo "  Step 7 (ffa_analysis): $(ps aux | grep -c 'ffa_analysis' | grep -v grep || echo 0)"
echo "  Step 8 (shap_analysis): $(ps aux | grep -c 'shap_analysis' | grep -v grep || echo 0)"
echo "  Step 9 (combine_shap_ffa): $(ps aux | grep -c 'combine_shap_ffa' | grep -v grep || echo 0)"
echo ""

# Check recent log files
echo "=== Recent Log Activity (last 5 minutes) ==="
find . -name "*.log" -type f -mmin -5 2>/dev/null | head -10 | while read logfile; do
    echo "  $logfile:"
    tail -3 "$logfile" 2>/dev/null | sed 's/^/    /'
done

echo ""
echo "=== Quick Status Check ==="
echo "Run this to check S3 checkpoint status:"
echo "  python utility_scripts/check_s3_checkpoints.py"
echo ""
echo "To view a specific cohort's progress:"
echo "  tail -f logs/<cohort>_<age_band>.log"

