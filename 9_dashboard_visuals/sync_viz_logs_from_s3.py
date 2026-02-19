#!/usr/bin/env python3
"""
Sync visualization logs from S3 into logs/viz_sync/ for local inspection.

S3 layout (from fe_monitor.mirror_log_to_s3):
  s3://pgx-repository/{step}_log/{cohort}/{age_band}/{filename}

Steps: 4_model_data_log, 4_fpgrowth_log, 5_bupar_log, 6_dtw_log,
  final_model_log (model training).

Usage (from repo root):
  python 9_dashboard_visuals/sync_viz_logs_from_s3.py [--profile PROFILE]

  --profile PROFILE   AWS CLI profile (e.g. mushin). Optional.

Requires AWS CLI configured with access to pgx-repository.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # 9_dashboard_visuals -> repo root
VIZ_SYNC_DIR = REPO_ROOT / "logs" / "viz_sync"
BUCKET = "pgx-repository"
PREFIXES = (
    "4_model_data_log",
    "4_fpgrowth_log",
    "5_bupar_log",
    "6_dtw_log",
    "final_model_log",  # Step 6 model training (run_final_model.py)
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync visualization logs from S3")
    parser.add_argument("--profile", default=None, help="AWS CLI profile (e.g. mushin)")
    args = parser.parse_args()

    VIZ_SYNC_DIR.mkdir(parents=True, exist_ok=True)
    profile_msg = f" (profile {args.profile})" if args.profile else ""
    print(f"Syncing viz logs from s3://{BUCKET}/ to {VIZ_SYNC_DIR}{profile_msg}")
    cmd_base = ["aws", "s3", "sync"]
    if args.profile:
        cmd_base.extend(["--profile", args.profile])
    for prefix in PREFIXES:
        local_dir = VIZ_SYNC_DIR / prefix
        local_dir.mkdir(parents=True, exist_ok=True)
        uri = f"s3://{BUCKET}/{prefix}/"
        print(f"  {uri} -> {local_dir}")
        r = subprocess.run(
            [*cmd_base, uri, str(local_dir)],
            cwd=str(REPO_ROOT),
        )
        if r.returncode != 0:
            print(f"  Warning: sync returned {r.returncode}", file=sys.stderr)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
