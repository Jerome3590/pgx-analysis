#!/usr/bin/env python3
"""
Download model_events.parquet for non_opioid_ed 55-64 from S3 and confirm schema
includes the target-date column (first_o11_p_date or legacy first_ed_non_opioid_date).
Run from WSL or external terminal if Cursor proxy blocks AWS.

Usage:
  python 4_model_data/download_and_check_model_data_non_opioid_ed_55_64.py
"""
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIR = PROJECT_ROOT / "4_model_data" / "cohort_name=non_opioid_ed" / "age_band=55-64"
LOCAL_PATH = LOCAL_DIR / "model_events.parquet"
S3_URI = "s3://pgxdatalake/gold/cohorts_model_data/cohort_name=non_opioid_ed/age_band=55-64/model_events.parquet"
# Step 4 writes first_o11_p_date (O11_P includes P51b, O11, P33); accept legacy name for older parquets
REQUIRED_COL = "first_o11_p_date"
LEGACY_COL = "first_ed_non_opioid_date"


def download_via_aws_cli() -> bool:
    """Use aws s3 cp; requires AWS CLI and network (run from WSL/external if proxy blocks)."""
    import subprocess
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [
            "aws", "s3", "cp",
            S3_URI,
            str(LOCAL_PATH),
            "--only-show-errors",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if r.returncode != 0:
        print(f"[ERROR] aws s3 cp failed: {r.stderr or r.stdout}")
        return False
    print(f"[OK] Downloaded to {LOCAL_PATH}")
    return True


def check_schema(path: Path) -> None:
    """Print schema and confirm target-date column (first_o11_p_date or legacy) is present."""
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        print("  Run this script from a terminal where 'aws s3 cp' works (e.g. WSL).")
        return
    path_str = str(path.resolve()).replace("\\", "/")
    con = duckdb.connect()
    schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path_str}')").fetchall()
    col_names = [row[0] for row in schema]
    con.close()

    print("Schema columns:", ", ".join(col_names))
    if REQUIRED_COL in col_names:
        print(f"[OK] Column '{REQUIRED_COL}' is present. BupaR pre/post target split can use it.")
    elif LEGACY_COL in col_names:
        print(f"[OK] Column '{LEGACY_COL}' is present (legacy). BupaR can use it.")
    else:
        print(f"[MISSING] Column '{REQUIRED_COL}' (or legacy '{LEGACY_COL}') not in schema. Pre-HCG/Post-HCG event logs will be empty.")
        if "first_opioid_ed_date" in col_names:
            print("  (Schema has first_opioid_ed_date; non_opioid_ed needs first_o11_p_date or first_ed_non_opioid_date.)")


def main() -> int:
    import sys
    check_only = "--check-only" in sys.argv
    if check_only:
        print("Checking schema only (use without --check-only to download first).")
        check_schema(LOCAL_PATH)
        return 0
    print("Downloading model_events for non_opioid_ed 55-64 from S3...")
    if not download_via_aws_cli():
        if LOCAL_PATH.exists():
            print("Download failed; checking existing local file.")
            check_schema(LOCAL_PATH)
        return 1
    print("\nChecking schema...")
    check_schema(LOCAL_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
