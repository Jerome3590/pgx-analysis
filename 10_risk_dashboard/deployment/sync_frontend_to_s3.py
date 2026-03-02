#!/usr/bin/env python3
"""
Sync the dashboard frontend (index.html and assets) to the dashboard S3 bucket.

Uses S3_DASHBOARD_BUCKET and S3_DASHBOARD_PREFIX (defaults: jerome-dixon.io, vcu/pgx-risk-calculator).
Run from repo root or from this directory.

Usage:
    python sync_frontend_to_s3.py
    python sync_frontend_to_s3.py --dry-run

    # Override bucket/prefix via env or args:
    S3_DASHBOARD_BUCKET=my-bucket S3_DASHBOARD_PREFIX=my/prefix python sync_frontend_to_s3.py
    python sync_frontend_to_s3.py --bucket my-bucket --prefix my/prefix
"""

import os
import subprocess
import sys
from pathlib import Path

# Script is in 10_risk_dashboard/deployment/; frontend is 10_risk_dashboard/frontend/
SCRIPT_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = SCRIPT_DIR.parent
FRONTEND_DIR = DASHBOARD_DIR / "frontend"


def main():
    import argparse
    p = argparse.ArgumentParser(description="Sync dashboard frontend to S3")
    p.add_argument("--dry-run", action="store_true", help="Show what would be synced (aws s3 sync --dryrun)")
    p.add_argument("--bucket", default=os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io"), help="S3 bucket (default: S3_DASHBOARD_BUCKET or jerome-dixon.io)")
    p.add_argument("--prefix", default=os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator"), help="S3 key prefix (default: S3_DASHBOARD_PREFIX or vcu/pgx-risk-calculator)")
    p.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"), help="AWS region")
    args = p.parse_args()

    if not FRONTEND_DIR.exists():
        print(f"[ERROR] Frontend dir not found: {FRONTEND_DIR}", file=sys.stderr)
        sys.exit(1)

    prefix = args.prefix.strip("/")
    s3_uri = f"s3://{args.bucket}/{prefix}/"
    print(f"Syncing {FRONTEND_DIR} -> {s3_uri}" + (" (dry run)" if args.dry_run else ""))

    cmd = ["aws", "s3", "sync", str(FRONTEND_DIR), s3_uri, "--region", args.region]
    if args.dry_run:
        cmd.append("--dryrun")

    r = subprocess.run(cmd)
    if r.returncode == 0:
        print("Frontend sync complete." if not args.dry_run else "Dry run complete.")
    else:
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
