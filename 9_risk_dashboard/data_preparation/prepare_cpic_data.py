#!/usr/bin/env python3
"""
Prepare CPIC master Excel file for Lambda deployment.

Copies the master Excel file from 5_pgx_analysis/cpic/ to outputs/cpic/ for the
Docker container. If the Excel file is missing, falls back to 5_pgx_analysis/data/
CSV (cpicPairs.csv or cpic.csv) and converts to the expected .xlsx for Lambda.
"""

import sys
from pathlib import Path
import shutil

# This script is in 9_risk_dashboard/data_preparation/
# Project root is 3 levels up
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Source paths (try Excel first, then CSV fallbacks)
SOURCE_EXCEL = PROJECT_ROOT / "5_pgx_analysis" / "cpic" / "cpic_gene-drug_pairs.xlsx"
SOURCE_CSV_PAIRS = PROJECT_ROOT / "5_pgx_analysis" / "data" / "cpicPairs.csv"
SOURCE_CSV = PROJECT_ROOT / "5_pgx_analysis" / "data" / "cpic.csv"

DEST_DIR = PROJECT_ROOT / "9_risk_dashboard" / "outputs" / "cpic"
DEST_EXCEL = DEST_DIR / "cpic_gene-drug_pairs.xlsx"

CPIC_XLSX_URL = "https://files.cpicpgx.org/data/report/current/pair/cpic_gene-drug_pairs.xlsx"


def _download_cpic_excel() -> bool:
    """Download official CPIC Excel (more recent and accurate) to 5_pgx_analysis/cpic/."""
    try:
        import urllib.request
        SOURCE_EXCEL.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading official CPIC Excel (recommended source): {CPIC_XLSX_URL}")
        urllib.request.urlretrieve(CPIC_XLSX_URL, SOURCE_EXCEL)
        if SOURCE_EXCEL.exists() and SOURCE_EXCEL.stat().st_size > 0:
            print(f"  Saved to {SOURCE_EXCEL} ({SOURCE_EXCEL.stat().st_size / 1024:.1f} KB)")
            return True
    except Exception as e:
        print(f"  Download failed: {e}")
    return False


def prepare_cpic_data():
    """Copy or build CPIC gene-drug file for Lambda deployment. Prefers official Excel (more recent and accurate)."""
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Prefer official Excel if present (or try to download it once)
    if not SOURCE_EXCEL.exists():
        _download_cpic_excel()
    if SOURCE_EXCEL.exists():
        print(f"Using official Excel: {SOURCE_EXCEL} -> {DEST_EXCEL}")
        shutil.copy2(SOURCE_EXCEL, DEST_EXCEL)
        print(f"OK: Copied {SOURCE_EXCEL.name} ({DEST_EXCEL.stat().st_size / 1024:.1f} KB)")
        print(f"\nOK: CPIC data prepared in {DEST_DIR}")
        print("  File will be included in Docker container at /var/task/data/")
        return

    # 2) Fallback: CSV from 5_pgx_analysis/data/ -> convert to .xlsx for Lambda (less preferred)
    for csv_path in (SOURCE_CSV_PAIRS, SOURCE_CSV):
        if not csv_path.exists():
            continue
        try:
            import pandas as pd
            print(f"Using CSV fallback (Excel is preferred when available): {csv_path}")
            df = pd.read_csv(csv_path)
            # Lambda expects columns with gene/drug (case-insensitive); ensure we have something
            if df.empty:
                continue
            df.to_excel(DEST_EXCEL, index=False, engine="openpyxl")
            print(f"OK: Wrote {DEST_EXCEL} ({DEST_EXCEL.stat().st_size / 1024:.1f} KB) from {csv_path.name}")
            print(f"\nOK: CPIC data prepared in {DEST_DIR}")
            print("  File will be included in Docker container at /var/task/data/")
            return
        except Exception as e:
            print(f"WARNING: Failed to convert {csv_path} to Excel: {e}")
            continue

    print("WARNING: CPIC source not found. Excel is preferred (more recent and accurate).")
    print(f"  Download Excel: {CPIC_XLSX_URL}")
    print(f"  Save as: {SOURCE_EXCEL}")
    print(f"  Or ensure 5_pgx_analysis/data/cpicPairs.csv (or cpic.csv) exists as fallback.")
    sys.exit(1)


if __name__ == "__main__":
    prepare_cpic_data()

