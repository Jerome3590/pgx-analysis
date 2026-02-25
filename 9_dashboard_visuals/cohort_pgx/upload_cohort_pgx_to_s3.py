#!/usr/bin/env python3
"""
Upload Cohort PGx network topology outputs to the dashboard S3 bucket.

Run after building networks (notebook 4 or build_network_topology.py). Uses the same
S3_DASHBOARD_BUCKET and S3_DASHBOARD_PREFIX as BupaR/DTW/FP-Growth so Lambda can
return network_topology_url for the PGx Cohort tab.

Usage:
  python upload_cohort_pgx_to_s3.py [--project-root PATH] [--dry-run]

Environment:
  S3_DASHBOARD_BUCKET  (default: jerome-dixon.io)
  S3_DASHBOARD_PREFIX  (default: vcu/pgx-risk-calculator)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Repo root
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Local output dir (same as Lambda key structure under prefix)
VISUAL_COHORT_PGX = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "cohort_pgx"
NETWORKS_DIR = VISUAL_COHORT_PGX / "networks"


def upload_cohort_pgx_to_dashboard_s3(project_root: Path | None = None, dry_run: bool = False) -> int:
    """Upload 10_risk_dashboard/visualizations/cohort_pgx/networks/ to dashboard S3.
    Returns number of files uploaded (0 on dry run or skip)."""
    root = project_root or REPO_ROOT
    networks_dir = root / "10_risk_dashboard" / "visualizations" / "cohort_pgx" / "networks"
    if not networks_dir.exists():
        print(f"No networks dir at {networks_dir}; nothing to upload.")
        return 0

    s3_bucket = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
    dashboard_prefix = (os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator") or "").strip("/")
    s3_base = f"{dashboard_prefix}/cohort_pgx/networks"

    try:
        from py_helpers.checkpoint_utils import upload_file_to_s3
    except ImportError:
        print("py_helpers.checkpoint_utils not available; use: aws s3 sync 10_risk_dashboard/visualizations/cohort_pgx/ s3://{bucket}/{prefix}/cohort_pgx/")
        return 0

    uploaded = 0
    for path in sorted(networks_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(networks_dir)
        key = f"{s3_base}/{rel.as_posix()}"
        s3_path = f"s3://{s3_bucket}/{key}"
        if dry_run:
            print(f"[dry-run] would upload {rel} -> {s3_path}")
            uploaded += 1
        elif upload_file_to_s3(path, s3_path, logger=None, check_exists=True):
            uploaded += 1
            print(f"Uploaded {rel} -> {s3_path}")

    if uploaded and not dry_run:
        print(f"Uploaded {uploaded} Cohort PGx file(s) to s3://{s3_bucket}/{s3_base}/")
    return uploaded


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload Cohort PGx network outputs to dashboard S3")
    ap.add_argument("--project-root", type=Path, default=REPO_ROOT, help="Repo root (default: auto)")
    ap.add_argument("--dry-run", action="store_true", help="Print uploads without uploading")
    args = ap.parse_args()
    n = upload_cohort_pgx_to_dashboard_s3(project_root=args.project_root, dry_run=args.dry_run)
    if args.dry_run:
        print(f"Dry run: would upload {n} file(s).")
    sys.exit(0 if n >= 0 else 1)


if __name__ == "__main__":
    main()
