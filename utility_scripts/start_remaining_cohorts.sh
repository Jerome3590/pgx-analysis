#!/usr/bin/env bash
# Start remaining cohort workflows in parallel
# Currently running: opioid_ed 13-24, non_opioid_ed 75-84, non_opioid_ed 85-94

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting remaining cohort workflows...${NC}"
echo ""

# Remaining cohorts to run:
# - opioid_ed: 25-44, 45-54, 55-64
# - non_opioid_ed: 65-74

# Start opioid_ed cohorts (3 remaining)
echo -e "${YELLOW}Starting opioid_ed cohorts...${NC}"
nohup ./run_cohort_workflow.sh opioid_ed 25-44 > ../logs/opioid_ed_25-44.log 2>&1 &
echo "  Started: opioid_ed 25-44 (PID: $!)"

nohup ./run_cohort_workflow.sh opioid_ed 45-54 > ../logs/opioid_ed_45-54.log 2>&1 &
echo "  Started: opioid_ed 45-54 (PID: $!)"

nohup ./run_cohort_workflow.sh opioid_ed 55-64 > ../logs/opioid_ed_55-64.log 2>&1 &
echo "  Started: opioid_ed 55-64 (PID: $!)"

echo ""

# Start non_opioid_ed cohort (1 remaining)
echo -e "${YELLOW}Starting non_opioid_ed cohort...${NC}"
nohup ./run_cohort_workflow.sh non_opioid_ed 65-74 > ../logs/non_opioid_ed_65-74.log 2>&1 &
echo "  Started: non_opioid_ed 65-74 (PID: $!)"

echo ""
echo -e "${GREEN}All remaining cohorts started!${NC}"
echo ""
echo "Monitor progress with:"
echo "  ./monitor_resources.sh"
echo ""
echo "Check logs:"
echo "  tail -f ../logs/opioid_ed_25-44.log"
echo "  tail -f ../logs/opioid_ed_45-54.log"
echo "  tail -f ../logs/opioid_ed_55-64.log"
echo "  tail -f ../logs/non_opioid_ed_65-74.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep run_cohort_workflow"

