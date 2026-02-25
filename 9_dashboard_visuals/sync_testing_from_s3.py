#!/usr/bin/env python3
"""
Sync the dashboard S3 'testing' prefix to the local project for analysis.

Production path for Cohort PGx is cohort_pgx/networks/... (Lambda uses that).
The testing prefix (vcu/pgx-risk-calculator/testing/) is only for syncing a copy
to local for inspection; this script pulls that testing prefix to a local folder.

Usage (from repo root):
  python 9_dashboard_visuals/sync_testing_from_s3.py [--profile PROFILE] [--list-only]

Local destination:
  10_risk_dashboard/visualizations/s3_sync_testing/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUCKET = "jerome-dixon.io"
PREFIX = "vcu/pgx-risk-calculator/testing"
LOCAL_DEST = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "s3_sync_testing"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync s3://jerome-dixon.io/vcu/pgx-risk-calculator/testing/ to local for analysis"
    )
    parser.add_argument("--profile", default=None, help="AWS CLI profile")
    parser.add_argument("--list-only", action="store_true", help="Only list S3 objects (no download)")
    args = parser.parse_args()

    s3_uri = f"s3://{BUCKET}/{PREFIX}/"

    if args.list_only:
        cmd = ["aws", "s3", "ls", s3_uri, "--recursive"]
        if args.profile:
            cmd.extend(["--profile", args.profile])
        r = subprocess.run(cmd, cwd=str(REPO_ROOT))
        return r.returncode

    LOCAL_DEST.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "sync", s3_uri, str(LOCAL_DEST)]
    if args.profile:
        cmd.extend(["--profile", args.profile])
    print(f"Syncing {s3_uri} -> {LOCAL_DEST}")
    r = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if r.returncode == 0:
        print(f"Done. Inspect: {LOCAL_DEST}")
        print("Note: Production API uses cohort_pgx/networks/...; testing prefix is for local analysis only. See 10_risk_dashboard/docs/S3_TESTING_PREFIX.md.")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
