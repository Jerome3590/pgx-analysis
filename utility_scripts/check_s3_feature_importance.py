#!/usr/bin/env python3
r"""
Local utility: check S3 feature importance output from this machine.

Modes:
  list    - List all objects under pgxdatalake/gold/feature_importance/
  filters - Check expected feature filter files per cohort/age_band (exist in S3 or not)

Usage (from repo root or utility_scripts/):
    python utility_scripts/check_s3_feature_importance.py [--profile NAME] [--bucket NAME]
    python utility_scripts/check_s3_feature_importance.py filters [--profile NAME] [--bucket NAME]
    --profile: AWS profile (default: AWS_PROFILE or default)
    --bucket: S3 bucket (default: pgxdatalake)
    Local: if <repo_parent>/credentials exists (e.g. C:\Projects\credentials),
    uses it as AWS_SHARED_CREDENTIALS_FILE.
"""

import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# Repo root (parent of utility_scripts)
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Use <repo_parent>/credentials when present (local runs)
_creds_file = _repo_root.parent / "credentials"
if _creds_file.exists() and not os.environ.get("AWS_SHARED_CREDENTIALS_FILE"):
    os.environ["AWS_SHARED_CREDENTIALS_FILE"] = str(_creds_file)

BUCKET_DEFAULT = os.environ.get("PGX_S3_BUCKET", "pgxdatalake")
PREFIX = "gold/feature_importance"

# Expected Step 3b feature filter artifacts per cohort/age_band (uploaded to S3 by filter_and_refine or workflow)
REQUIRED_COHORTS = {
    "opioid_ed": ["13-24", "25-44", "45-54", "55-64"],
    "non_opioid_ed": ["65-74", "75-84", "85-94"],
}
# Files we expect in S3 (filter_and_refine_features.py uploads cohort_feature_importance.csv and feature_filtering_summary.json)
# safe_feature_filter and control_feature_exclusions are written locally but not uploaded by pipeline; we still check if present
EXPECTED_FILTER_FILES = [
    "cohort_feature_importance.csv",
    "feature_filtering_summary.json",
    "safe_feature_filter.json",
    "control_feature_exclusions.json",
]


def list_feature_importance_outputs(s3_client, bucket: str, prefix: str = PREFIX):
    """List all objects under gold/feature_importance/ with key, size, LastModified."""
    results = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            results.append({
                "key": key,
                "size": obj.get("Size", 0),
                "last_modified": obj.get("LastModified"),
            })
    return results


def s3_exists(s3_client, bucket: str, key: str) -> bool:
    """Return True if object exists."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def check_feature_filters_s3(s3_client, bucket: str) -> None:
    """Check expected feature filter files for each cohort/age_band; print table of presence in S3."""
    print("Expected feature filter files in S3 (gold/feature_importance/{cohort}/{age_band}/)")
    print("Cohorts:", list(REQUIRED_COHORTS.keys()))
    print()
    missing = []
    for cohort, age_bands in REQUIRED_COHORTS.items():
        for age_band in age_bands:
            age_band_fname = age_band.replace("-", "_")
            base = f"{cohort}_{age_band_fname}"
            for suffix in EXPECTED_FILTER_FILES:
                key = f"gold/feature_importance/{cohort}/{age_band}/{base}_{suffix}"
                exists = s3_exists(s3_client, bucket, key)
                status = "OK" if exists else "MISSING"
                if not exists:
                    missing.append((cohort, age_band, suffix))
                print(f"  {cohort}/{age_band}  {suffix:<45}  {status}")
    print()
    if missing:
        print(f"Missing {len(missing)} file(s) in S3.")
        print("Note: safe_feature_filter.json and control_feature_exclusions.json are not uploaded by the pipeline;")
        print("      upload them manually if you want them in S3, or run filter_and_refine and sync.")
    else:
        print("All expected feature filter files present in S3.")
    return


def main():
    parser = argparse.ArgumentParser(
        description="Check S3 feature importance output from local device"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="list",
        choices=["list", "filters"],
        help="list = list all objects; filters = check expected filter files per cohort/age_band",
    )
    parser.add_argument(
        "--bucket",
        default=BUCKET_DEFAULT,
        help=f"S3 bucket (default: {BUCKET_DEFAULT})",
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("AWS_PROFILE"),
        help="AWS profile (default: AWS_PROFILE or default)",
    )
    args = parser.parse_args()

    session_kw = {}
    if args.profile:
        session_kw["profile_name"] = args.profile
    session = boto3.Session(**session_kw)
    s3 = session.client("s3")

    if args.mode == "filters":
        print("=" * 80)
        print("FEATURE FILTER FILES IN S3")
        print(f"Bucket: {args.bucket}  Prefix: {PREFIX}/")
        print("=" * 80)
        check_feature_filters_s3(s3, args.bucket)
        return 0

    print("=" * 80)
    print("FEATURE IMPORTANCE S3 OUTPUT (pgxdatalake)")
    print(f"Bucket: {args.bucket}  Prefix: {PREFIX}/")
    print("=" * 80)

    items = list_feature_importance_outputs(s3, args.bucket)
    if not items:
        print("No objects found.")
        return 0

    # Sort by key for stable output
    items.sort(key=lambda x: x["key"])

    def size_fmt(n):
        if n is None or n == 0:
            return "—"
        if n < 1024:
            return f"{n} B"
        if n < 1024 * 1024:
            return f"{n / 1024:.1f} KB"
        return f"{n / (1024 * 1024):.2f} MB"

    def date_fmt(dt):
        if dt is None:
            return "—"
        return dt.strftime("%Y-%m-%d %H:%M UTC")

    print(f"\n{len(items)} object(s):\n")
    print(f"{'Key':<70} {'Size':>12} {'LastModified':<20}")
    print("-" * 80)
    for r in items:
        print(f"{r['key']:<70} {size_fmt(r['size']):>12} {date_fmt(r['last_modified']):<20}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
