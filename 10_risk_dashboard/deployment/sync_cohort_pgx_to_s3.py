#!/usr/bin/env python3
"""
Sync Cohort PGx network HTML from EC2 to S3 with age-band path convention.

EC2 paths use underscore (e.g. networks/opioid_ed/25_44/network_topology.html).
S3 paths use hyphen (e.g. cohort_pgx/networks/opioid_ed/25-44/network_topology.html).

Usage (from repo root):
    python 10_risk_dashboard/deployment/sync_cohort_pgx_to_s3.py
    python 10_risk_dashboard/deployment/sync_cohort_pgx_to_s3.py --local-dir 10_risk_dashboard/visualizations/cohort_pgx

Environment:
    S3_DASHBOARD_BUCKET (default: jerome-dixon.io)
    S3_DASHBOARD_PREFIX (default: vcu/pgx-risk-calculator)
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "cohort_pgx"
NETWORKS_SUBDIR = "networks"


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Sync Cohort PGx to S3 with hyphen age-band paths")
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL, help="Local cohort_pgx dir (contains networks/)")
    args = parser.parse_args()
    local_base = args.local_dir.resolve()
    networks_dir = local_base / NETWORKS_SUBDIR
    if not networks_dir.exists():
        print(f"No {NETWORKS_SUBDIR}/ under {local_base}; nothing to sync.")
        return 0
    try:
        import boto3
    except ImportError:
        print("boto3 not available; pip install boto3", file=sys.stderr)
        return 1
    bucket = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
    prefix = (os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator") or "").strip("/")
    s3_prefix = f"{prefix}/cohort_pgx/{NETWORKS_SUBDIR}"
    s3 = boto3.client("s3")
    uploaded = 0
    for cohort_dir in sorted(networks_dir.iterdir()):
        if not cohort_dir.is_dir() or cohort_dir.name.startswith("."):
            continue
        cohort = cohort_dir.name
        for age_dir in sorted(cohort_dir.iterdir()):
            if not age_dir.is_dir() or age_dir.name.startswith("."):
                continue
            age_band_fname = age_dir.name  # EC2: 25_44
            age_band_s3 = age_band_fname.replace("_", "-")  # S3: 25-44
            for f in age_dir.rglob("*"):
                if not f.is_file():
                    continue
                rel = f.relative_to(age_dir)
                key = f"{s3_prefix}/{cohort}/{age_band_s3}/{rel.as_posix()}"
                try:
                    extra = {"ContentType": "text/html"} if f.suffix.lower() == ".html" else {}
                    s3.upload_file(str(f), bucket, key, ExtraArgs=extra)
                    print(f"  ✓ {cohort}/{age_band_s3}/{rel} -> s3://{bucket}/{key}")
                    uploaded += 1
                except Exception as e:
                    print(f"  ⚠ Upload failed {key}: {e}", file=sys.stderr)
    if uploaded:
        print(f"Cohort PGx: {uploaded} file(s) synced to S3 (age-band paths use hyphen).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
