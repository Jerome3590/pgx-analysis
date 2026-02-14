#!/usr/bin/env python3
"""
Check that gold/pharmacy (and optionally gold/medical) on NVMe are complete.

Expected layout (same as create_model_data.py):
  DATA_ROOT/gold/pharmacy/age_band={band}/event_year={year}/*.parquet
  DATA_ROOT/gold/medical/age_band={band}/event_year={year}/*.parquet

Expected cells for Step 4:
  - Age bands: 13-24, 25-44, 45-54, 55-64, 65-74, 75-84, 85-114
  - Event years: 2016, 2017, 2018, 2019

Run from project root:
  python 4_model_data/check_gold_pharmacy_completeness.py
  python 4_model_data/check_gold_pharmacy_completeness.py --medical  # also check gold/medical
  python 4_model_data/check_gold_pharmacy_completeness.py --s3      # compare to S3 (requires boto3)
"""

import argparse
import sys
from pathlib import Path

# Project root (path must be set before py_helpers import)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.env_utils import get_data_root  # noqa: E402

# Same as create_model_data.py
EXPECTED_AGE_BANDS = ["13-24", "25-44", "45-54", "55-64", "65-74", "75-84", "85-114"]
EXPECTED_YEARS = [2016, 2017, 2018, 2019]


def check_gold_dir(name: str, root: Path, check_s3: bool = False, bucket: str = "pgxdatalake") -> None:
    """Check completeness of gold/name (pharmacy or medical)."""
    total_files = 0
    total_bytes = 0
    missing = []
    present = []

    for age_band in EXPECTED_AGE_BANDS:
        for year in EXPECTED_YEARS:
            cell_dir = root / f"age_band={age_band}" / f"event_year={year}"
            parquets = list(cell_dir.glob("*.parquet")) if cell_dir.exists() else []
            cell_size = sum(f.stat().st_size for f in parquets)
            if parquets:
                present.append((age_band, year, len(parquets), cell_size))
                total_files += len(parquets)
                total_bytes += cell_size
            else:
                missing.append((age_band, year))

    # Report
    print(f"\n=== gold/{name} ===")
    print(f"  Root: {root}")
    print(f"  Expected cells: {len(EXPECTED_AGE_BANDS) * len(EXPECTED_YEARS)} (age_band × event_year)")
    print(f"  Present: {len(present)} cells, {total_files} parquet files, {_fmt(total_bytes)} total")
    if missing:
        print(f"  Missing cells: {len(missing)}")
        for age_band, year in missing:
            print(f"    - age_band={age_band} event_year={year}")
    else:
        print("  All expected cells present.")

    # Per-cell summary (optional, if not too many)
    if present and len(present) <= 40:
        print("  Per-cell (age_band, year): files, size")
        for age_band, year, n, size in sorted(present):
            print(f"    {age_band} {year}: {n} files, {_fmt(size)}")

    if check_s3:
        try:
            import boto3
            s3 = boto3.client("s3")
            prefix = f"gold/{name}/"
            paginator = s3.get_paginator("list_objects_v2")
            s3_count = 0
            s3_size = 0
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    s3_count += 1
                    s3_size += obj.get("Size", 0)
            print(f"  S3 s3://{bucket}/{prefix}: {s3_count} objects, {_fmt(s3_size)} total")
            if total_files > 0 and s3_count > 0:
                pct = 100.0 * total_files / s3_count if s3_count else 0
                print(f"  Local has {pct:.1f}% of S3 file count (by object count).")
        except Exception as e:
            print(f"  S3 check skipped: {e}")


def _fmt(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def main():
    parser = argparse.ArgumentParser(description="Check gold/pharmacy (and optionally gold/medical) completeness on NVMe.")
    parser.add_argument("--medical", action="store_true", help="Also check gold/medical")
    parser.add_argument("--s3", action="store_true", help="Compare to S3 (list object count/size)")
    args = parser.parse_args()

    data_root = get_data_root()
    pharmacy_root = data_root / "gold" / "pharmacy"
    print(f"Data root: {data_root}")

    check_gold_dir("pharmacy", pharmacy_root, check_s3=args.s3)
    if args.medical:
        medical_root = data_root / "gold" / "medical"
        check_gold_dir("medical", medical_root, check_s3=args.s3)


if __name__ == "__main__":
    main()
