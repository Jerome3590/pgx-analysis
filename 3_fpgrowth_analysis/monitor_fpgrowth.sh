#!/bin/bash
# Monitor FPGrowth execution progress

clear
echo "=========================================="
echo "FPGrowth Analysis Monitor"
echo "=========================================="
echo ""

while true; do
    clear
    echo "Last updated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "=========================================="
    echo "Recent Log Output:"
    echo "=========================================="
    tail -40 fpgrowth_execution.log 2>/dev/null || echo "Waiting for log file..."
    echo ""
    echo "=========================================="
    echo "Press Ctrl+C to exit monitor"
    echo "=========================================="
    sleep 10
done

