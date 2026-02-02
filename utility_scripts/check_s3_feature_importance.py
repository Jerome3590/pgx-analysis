#!/usr/bin/env python3
r"""
Local utility: check S3 feature importance output from this machine.

Lists objects under pgxdatalake/gold/feature_importance/ so you can verify
what's in S3 without running on EC2.

Usage (from repo root or utility_scripts/):
    python utility_scripts/check_s3_feature_importance.py [--profile NAME] [--bucket NAME]
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


def main():
    parser = argparse.ArgumentParser(
        description="Check S3 feature importance output from local device"
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
