#!/usr/bin/env python3
"""
Apply CORS configuration to the dashboard S3 bucket (idempotent).

Ensures the bucket allows cross-origin GET/HEAD from the dashboard origin
(e.g. https://jerome-dixon.io) so the frontend can fetch direct S3 URLs
(dtw/chart_data.json, causal_data_url, BupaR/FP-Growth/cohort_pgx assets)
without CORS errors.

Run as part of the deployment workflow (notebook 5 Step 6) or standalone:
  python apply_dashboard_bucket_cors.py
  python apply_dashboard_bucket_cors.py --check   # print current CORS and exit

Config source: 10_risk_dashboard/docs/s3-cors-config.json (CORSRules format).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Project root: this script is in 10_risk_dashboard/deployment/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_CORS_CONFIG_PATH = PROJECT_ROOT / "10_risk_dashboard" / "docs" / "s3-cors-config.json"


def load_cors_config(config_path: Path) -> dict:
    """Load CORS config JSON (must have CORSRules key for put_bucket_cors)."""
    if not config_path.exists():
        raise FileNotFoundError(f"CORS config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    if "CORSRules" not in data:
        raise ValueError("CORS config must contain 'CORSRules' key. See S3_CORS_SETUP.md.")
    return data


def apply_cors(bucket: str, config_path: Path, region: str = "us-east-1") -> bool:
    """Apply CORS configuration to the bucket. Idempotent."""
    import boto3
    config = load_cors_config(config_path)
    client = boto3.client("s3", region_name=region)
    client.put_bucket_cors(Bucket=bucket, CORSConfiguration=config)
    return True


def get_current_cors(bucket: str, region: str = "us-east-1") -> dict | None:
    """Return current CORS config or None if not set."""
    import boto3
    from botocore.exceptions import ClientError
    client = boto3.client("s3", region_name=region)
    try:
        return client.get_bucket_cors(Bucket=bucket)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "NoSuchCORSConfiguration":
            return None
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply CORS to dashboard S3 bucket (idempotent). Used in deploy workflow."
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="S3 bucket name (default: S3_DASHBOARD_BUCKET env or jerome-dixon.io)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CORS_CONFIG_PATH,
        help=f"Path to CORS config JSON (default: {DEFAULT_CORS_CONFIG_PATH})",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region for the bucket (default: us-east-1)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only print current CORS config and exit (no apply)",
    )
    args = parser.parse_args()

    bucket = args.bucket
    if bucket is None:
        import os
        bucket = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")

    try:
        if args.check:
            current = get_current_cors(bucket, args.region)
            if current is None:
                print(f"Bucket {bucket}: no CORS configuration")
            else:
                print(json.dumps(current, indent=2))
            return 0

        apply_cors(bucket, args.config, args.region)
        print(f"✓ CORS applied to bucket {bucket} (visualization folders can pass/receive objects)")
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error applying CORS: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
