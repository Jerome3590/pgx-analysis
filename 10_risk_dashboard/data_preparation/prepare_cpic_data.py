#!/usr/bin/env python3
"""
Prepare CPIC master Excel file for Lambda deployment.

Copies the master Excel file from 5_pgx_analysis/cpic/ to outputs/cpic/ for the
Docker container. If the Excel file is missing, falls back to 5_pgx_analysis/data/
CSV (cpicPairs.csv or cpic.csv) and converts to the expected .xlsx for Lambda.

Uses DuckDB for CSV reads when available; writes a Parquet copy alongside the
Excel for efficient downstream use where supported.
"""

import sys
from pathlib import Path
import shutil

# This script is in 10_risk_dashboard/data_preparation/
# Project root is 3 levels up
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Source paths (try Excel first, then CSV fallbacks)
SOURCE_EXCEL = PROJECT_ROOT / "5_pgx_analysis" / "cpic" / "cpic_gene-drug_pairs.xlsx"
SOURCE_CSV_PAIRS = PROJECT_ROOT / "5_pgx_analysis" / "data" / "cpicPairs.csv"
SOURCE_CSV = PROJECT_ROOT / "5_pgx_analysis" / "data" / "cpic.csv"

DEST_DIR = PROJECT_ROOT / "10_risk_dashboard" / "outputs" / "cpic"
DEST_EXCEL = DEST_DIR / "cpic_gene-drug_pairs.xlsx"
DEST_PARQUET = DEST_DIR / "cpic_gene-drug_pairs.parquet"

CPIC_XLSX_URL = "https://files.cpicpgx.org/data/report/current/pair/cpic_gene-drug_pairs.xlsx"

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


def _read_csv_with_duckdb(csv_path: Path):
    """Read CSV via DuckDB when available; returns DataFrame or None on failure."""
    if not DUCKDB_AVAILABLE:
        return None
    try:
        con = duckdb.connect(":memory:")
        path_str = str(csv_path.resolve())
        df = con.execute("SELECT * FROM read_csv_auto(?)", [path_str]).fetchdf()
        con.close()
        return df
    except Exception:
        return None


def _write_parquet_copy(df, dest_parquet: Path) -> bool:
    """Write DataFrame to Parquet; use DuckDB when available else pandas. Returns True if written."""
    dest_parquet = Path(dest_parquet)
    if DUCKDB_AVAILABLE:
        try:
            con = duckdb.connect(":memory:")
            con.register("cpic_df", df)
            # DuckDB COPY TO expects path as literal; path is from our DEST_PARQUET
            path_str = str(dest_parquet.resolve().as_posix()).replace("'", "''")
            con.execute(f"COPY cpic_df TO '{path_str}' (FORMAT PARQUET)")
            con.close()
            return True
        except Exception:
            pass
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            df.to_parquet(dest_parquet, index=False)
            return True
    except Exception:
        pass
    return False


def _download_cpic_excel() -> bool:
    """Download official CPIC Excel (more recent and accurate) to 5_pgx_analysis/cpic/."""
    import ssl
    import urllib.request
    import urllib.error

    SOURCE_EXCEL.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official CPIC Excel (recommended source): {CPIC_XLSX_URL}")
    req = urllib.request.Request(CPIC_XLSX_URL)
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
            with open(SOURCE_EXCEL, "wb") as f:
                f.write(resp.read())
    except urllib.error.URLError as ue:
        if "CERTIFICATE_VERIFY_FAILED" in str(ue.reason) or "SSL" in str(ue.reason):
            ctx = ssl._create_unverified_context()
            try:
                with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                    with open(SOURCE_EXCEL, "wb") as f:
                        f.write(resp.read())
            except Exception as e2:
                print(f"  Download failed: {e2}")
                return False
        else:
            print(f"  Download failed: {ue}")
            return False
    except Exception as e:
        print(f"  Download failed: {e}")
        return False
    if SOURCE_EXCEL.exists() and SOURCE_EXCEL.stat().st_size > 0:
        print(f"  Saved to {SOURCE_EXCEL} ({SOURCE_EXCEL.stat().st_size / 1024:.1f} KB)")
        return True
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
        import pandas as pd
        try:
            df = pd.read_excel(DEST_EXCEL, engine="openpyxl")
            if not df.empty and _write_parquet_copy(df, DEST_PARQUET):
                print(f"  Parquet copy: {DEST_PARQUET.name}")
        except Exception as e:
            print(f"  (Parquet copy skipped: {e})")
        print(f"\nOK: CPIC data prepared in {DEST_DIR}")
        print("  File will be included in Docker container at /var/task/data/")
        return

    # 2) Fallback: CSV from 5_pgx_analysis/data/ -> convert to .xlsx for Lambda (less preferred)
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        openpyxl = None
    import pandas as pd
    for csv_path in (SOURCE_CSV_PAIRS, SOURCE_CSV):
        if not csv_path.exists():
            continue
        if openpyxl is None:
            print("WARNING: openpyxl not installed; cannot convert CSV to Excel.")
            print("  Install with: pip install openpyxl")
            break
        try:
            print(f"Using CSV fallback (Excel is preferred when available): {csv_path}")
            df = _read_csv_with_duckdb(csv_path)
            if df is None:
                df = pd.read_csv(csv_path)
            else:
                print("  (read via DuckDB)")
            # Lambda expects columns with gene/drug (case-insensitive); ensure we have something
            if df.empty:
                continue
            df.to_excel(DEST_EXCEL, index=False, engine="openpyxl")
            print(f"OK: Wrote {DEST_EXCEL} ({DEST_EXCEL.stat().st_size / 1024:.1f} KB) from {csv_path.name}")
            if _write_parquet_copy(df, DEST_PARQUET):
                print(f"  Parquet copy: {DEST_PARQUET.name}")
            print(f"\nOK: CPIC data prepared in {DEST_DIR}")
            print("  File will be included in Docker container at /var/task/data/")
            return
        except Exception as e:
            print(f"WARNING: Failed to convert {csv_path} to Excel: {e}")
            continue

    print("WARNING: CPIC source not found. Excel is preferred (more recent and accurate).")
    print(f"  Download Excel: {CPIC_XLSX_URL}")
    print(f"  Save as: {SOURCE_EXCEL}")
    print("  Or ensure 5_pgx_analysis/data/cpicPairs.csv (or cpic.csv) exists and install: pip install openpyxl")
    sys.exit(1)


if __name__ == "__main__":
    prepare_cpic_data()

