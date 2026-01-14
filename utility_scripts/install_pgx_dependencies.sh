#!/usr/bin/env bash
# Install Python dependencies for PGx analysis

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Installing PGx analysis dependencies...${NC}"

# Detect Python executable (same logic as run_cohort_workflow.sh)
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found"
    exit 1
fi

# Check for virtual environment in common locations (EC2 path first)
if [ -f "/home/pgx3874/jupyter-env/bin/python3.11" ]; then
    PYTHON_CMD="/home/pgx3874/jupyter-env/bin/python3.11"
    PIP_CMD="/home/pgx3874/jupyter-env/bin/pip3.11"
    echo -e "${GREEN}Using Python from EC2 jupyter-env: $PYTHON_CMD${NC}"
elif [ -f "$HOME/jupyter-env/bin/python3" ]; then
    PYTHON_CMD="$HOME/jupyter-env/bin/python3"
    PIP_CMD="$HOME/jupyter-env/bin/pip3"
    echo -e "${GREEN}Using Python from jupyter-env: $PYTHON_CMD${NC}"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -f "$VIRTUAL_ENV/bin/python3" ]; then
    PYTHON_CMD="$VIRTUAL_ENV/bin/python3"
    PIP_CMD="$VIRTUAL_ENV/bin/pip3"
    echo -e "${GREEN}Using Python from active virtualenv: $PYTHON_CMD${NC}"
else
    PIP_CMD="${PYTHON_CMD} -m pip"
    echo -e "${GREEN}Using system Python: $PYTHON_CMD${NC}"
fi

# Required dependencies
echo -e "\n${YELLOW}Installing required packages...${NC}"
$PIP_CMD install rapidfuzz

# Optional but recommended dependencies
echo -e "\n${YELLOW}Installing optional packages (recommended)...${NC}"
$PIP_CMD install openpyxl biopython

echo -e "\n${GREEN}✅ Installation complete!${NC}"
echo -e "\nInstalled packages:"
$PIP_CMD list | grep -E "(rapidfuzz|openpyxl|biopython)" || echo "  (packages may be listed differently)"

