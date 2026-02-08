#!/bin/bash
# Quick script to find and display the most recent cleanup log file.
# Run from repo root or any directory; PROJECT_ROOT = repo root (parent of utility_scripts).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Find the most recent log file
LATEST_LOG=$(ls -t "${PROJECT_ROOT}"/cleanup_cohort_data_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "No cleanup log files found in ${PROJECT_ROOT}"
    exit 1
fi

echo "Most recent log file: $LATEST_LOG"
echo "=========================================="
echo ""

# Show last 50 lines (or tail -f for live monitoring)
if [ "$1" == "-f" ] || [ "$1" == "--follow" ]; then
    echo "Following log file (Ctrl+C to exit)..."
    tail -f "$LATEST_LOG"
else
    echo "Last 100 lines:"
    echo "=========================================="
    tail -100 "$LATEST_LOG"
    echo ""
    echo "=========================================="
    echo "Full log file: $LATEST_LOG"
    echo "To view full log: cat $LATEST_LOG"
    echo "To follow log: tail -f $LATEST_LOG"
fi
