#!/usr/bin/env python3
"""
Prepare CPIC master Excel file for Lambda deployment.

Copies the master Excel file from 7_pgx_analysis to 10_results/data/
for inclusion in the Docker container.
"""

import sys
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).parent.parent

# Source and destination paths
SOURCE_EXCEL = PROJECT_ROOT / "7_pgx_analysis" / "cpic" / "cpic_gene-drug_pairs.xlsx"
DEST_DIR = PROJECT_ROOT / "10_results" / "data"
DEST_EXCEL = DEST_DIR / "cpic_gene-drug_pairs.xlsx"


def prepare_cpic_data():
    """Copy CPIC master Excel file to Lambda deployment directory."""
    
    # Create destination directory
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Copy master Excel file
    if SOURCE_EXCEL.exists():
        print(f"Copying master Excel file: {SOURCE_EXCEL} -> {DEST_EXCEL}")
        shutil.copy2(SOURCE_EXCEL, DEST_EXCEL)
        print(f"OK: Copied {SOURCE_EXCEL.name} ({DEST_EXCEL.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"WARNING: Master Excel file not found at {SOURCE_EXCEL}")
        print("  Download from: https://files.cpicpgx.org/data/report/current/pair/cpic_gene-drug_pairs.xlsx")
        sys.exit(1)
    
    print(f"\nOK: CPIC data prepared in {DEST_DIR}")
    print("  File will be included in Docker container at /var/task/data/")


if __name__ == "__main__":
    prepare_cpic_data()

