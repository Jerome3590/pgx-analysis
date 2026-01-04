#!/usr/bin/env bash
# Check if PGx analysis dependencies are installed

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect Python executable (same logic as run_cohort_workflow.sh)
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found"
    exit 1
fi

# Check for virtual environment in common locations
if [ -f "$HOME/jupyter-env/bin/python3" ]; then
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

echo ""
echo -e "${YELLOW}Checking PGx analysis dependencies...${NC}"
echo ""

# Required packages
REQUIRED_PACKAGES=("rapidfuzz")
OPTIONAL_PACKAGES=("openpyxl" "biopython")

# Check required packages
echo -e "${YELLOW}Required packages:${NC}"
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if $PIP_CMD show "$pkg" &> /dev/null; then
        version=$($PIP_CMD show "$pkg" 2>/dev/null | grep "^Version:" | awk '{print $2}')
        echo -e "  ${GREEN}✓${NC} $pkg (version: $version)"
    else
        echo -e "  ${RED}✗${NC} $pkg (NOT INSTALLED)"
    fi
done

# Check optional packages
echo ""
echo -e "${YELLOW}Optional packages (recommended):${NC}"
for pkg in "${OPTIONAL_PACKAGES[@]}"; do
    if $PIP_CMD show "$pkg" &> /dev/null; then
        version=$($PIP_CMD show "$pkg" 2>/dev/null | grep "^Version:" | awk '{print $2}')
        echo -e "  ${GREEN}✓${NC} $pkg (version: $version)"
    else
        echo -e "  ${YELLOW}○${NC} $pkg (not installed - optional)"
    fi
done

# Quick test import
echo ""
echo -e "${YELLOW}Testing imports...${NC}"
if $PYTHON_CMD -c "import rapidfuzz" 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} rapidfuzz imports successfully"
else
    echo -e "  ${RED}✗${NC} rapidfuzz import failed"
fi

if $PYTHON_CMD -c "import openpyxl" 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} openpyxl imports successfully"
else
    echo -e "  ${YELLOW}○${NC} openpyxl not available (optional)"
fi

if $PYTHON_CMD -c "import Bio" 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} biopython imports successfully"
else
    echo -e "  ${YELLOW}○${NC} biopython not available (optional)"
fi

echo ""
echo -e "${YELLOW}To install missing packages, run:${NC}"
echo "  ./utility_scripts/install_pgx_dependencies.sh"
echo ""
echo -e "${YELLOW}Or manually:${NC}"
echo "  $PIP_CMD install rapidfuzz openpyxl biopython"

