#!/usr/bin/env python3
"""
Download FP-Growth logs from S3 to local logs dir for inspection.

S3 layout (see py_helpers.fe_monitor.mirror_log_to_s3):
  s3://{bucket}/4_fpgrowth_log/{cohort}/{age_band}/fpgrowth_{cohort}_{age_band_fname}.log

Use this to inspect why FP-Growth is not running all age bands for each cohort or
missing the polypharmacy (non_opioid_ed) cohort. Logs are written to:
  9_dashboard_visuals/logs/fpgrowth/

Usage (from repo root):
  python 9_dashboard_visuals/download_fpgrowth_logs.py
  python 9_dashboard_visuals/download_fpgrowth_logs.py --profile my-aws-profile
  python 9_dashboard_visuals/download_fpgrowth_logs.py --cohort non_opioid_ed --age-band 55-64
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from py_helpers.constants import REQUIRED_COHORTS
except ImportError:
    REQUIRED_COHORTS = {
        "opioid_ed": ["0-12", "13-24", "25-44", "45-54", "55-64", "65-74", "75-84", "85-114"],
        "non_opioid_ed": ["0-12", "13-24", "25-44", "45-54", "55-64", "65-74", "75-84", "85-114"],
    }

S3_BUCKET = os.environ.get("PGX_S3_BUCKET", "pgx-repository")
S3_PREFIX = "4_fpgrowth_log"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download FP-Growth logs from S3 to 9_dashboard_visuals/logs/fpgrowth/"
    )
    ap.add_argument(
        "--profile",
        default=os.environ.get("AWS_PROFILE"),
        help="AWS profile for S3 (default: AWS_PROFILE)",
    )
    ap.add_argument(
        "--cohort",
        action="append",
        dest="cohorts",
        help="Cohort to download (repeatable); default: all (opioid_ed, non_opioid_ed)",
    )
    ap.add_argument(
        "--age-band",
        action="append",
        dest="age_bands",
        help="Age band to download (repeatable); default: all",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print S3 keys that would be downloaded",
    )
    args = ap.parse_args()

    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        print("boto3 is required. Install with: pip install boto3")
        return 1

    # Combinations: same as run_dashboard_visuals / REQUIRED_COHORTS
    if args.cohorts and args.age_bands:
        combinations = [(c, ab) for c in args.cohorts for ab in args.age_bands]
    elif args.cohorts:
        combinations = [
            (c, ab) for c in args.cohorts for ab in REQUIRED_COHORTS.get(c, [])
        ]
    elif args.age_bands:
        combinations = [
            (c, ab)
            for c, bands in REQUIRED_COHORTS.items()
            for ab in args.age_bands
            if ab in bands
        ]
    else:
        combinations = [
            (c, ab)
            for c, bands in REQUIRED_COHORTS.items()
            for ab in bands
        ]

    logs_dir = REPO_ROOT / "9_dashboard_visuals" / "logs" / "fpgrowth"
    logs_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        print(f"Downloading to {logs_dir}")

    session_kw = {}
    if args.profile:
        session_kw["profile_name"] = args.profile
    s3 = boto3.client("s3", **session_kw)

    downloaded = 0
    missing = []
    errors = []

    for cohort_name, age_band in combinations:
        age_band_fname = age_band.replace("-", "_")
        log_fname = f"fpgrowth_{cohort_name}_{age_band_fname}.log"
        s3_key = f"{S3_PREFIX}/{cohort_name}/{age_band}/{log_fname}"
        local_path = logs_dir / log_fname

        if args.dry_run:
            print(s3_key)
            continue

        try:
            s3.download_file(S3_BUCKET, s3_key, str(local_path))
            print(f"  OK  {cohort_name}/{age_band} -> {local_path.name}")
            downloaded += 1
        except ClientError as e:
            if e.response["Error"]["Code"] == "404" or "Not Found" in str(e):
                missing.append(f"{cohort_name}/{age_band}")
            else:
                errors.append(f"{cohort_name}/{age_band}: {e}")
        except Exception as e:
            errors.append(f"{cohort_name}/{age_band}: {e}")

    if args.dry_run:
        print(f"Would download {len(combinations)} log(s).")
        return 0

    print(f"\nDownloaded: {downloaded} log(s).")
    if missing:
        print(f"Missing on S3 ({len(missing)}): {', '.join(missing)}")
        print("  -> FP-Growth may not have been run for these cohort/age_band combinations.")
    if errors:
        print("Errors:")
        for err in errors:
            print(f"  {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
