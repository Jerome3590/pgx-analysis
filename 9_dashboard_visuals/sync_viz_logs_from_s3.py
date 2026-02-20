#!/usr/bin/env python3
"""
Sync visualization logs and pipeline status from S3 into logs/viz_sync/ for local inspection.

Syncs:
1. Step logs (from mirror_log_to_s3):
   s3://pgx-repository/{step}_log/{cohort}/{age_band}/{filename}
   Steps: 4_model_data_log, 4_fpgrowth_log, 5_bupar_log, 6_dtw_log, final_model_log.
2. Pipeline checkpoints (status):
   s3://pgx-repository/pipeline_checkpoints/{step}/{cohort}/{age_band}/checkpoint.json
   -> logs/viz_sync/pipeline_checkpoints/

After sync, prints a short summary of ERROR/FATAL lines found in synced log files.

Usage (from repo root):
  python 9_dashboard_visuals/sync_viz_logs_from_s3.py [--profile PROFILE] [--no-errors-summary]

  --profile PROFILE     AWS CLI profile (e.g. mushin). Optional.
  --no-errors-summary   Skip scanning logs for ERROR/FATAL after sync.

Requires AWS CLI configured with access to pgx-repository.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # 9_dashboard_visuals -> repo root
VIZ_SYNC_DIR = REPO_ROOT / "logs" / "viz_sync"
BUCKET = "pgx-repository"
LOG_PREFIXES = (
    "4_model_data_log",
    "4_fpgrowth_log",
    "5_bupar_log",
    "6_dtw_log",
    "final_model_log",  # Step 6 model training (run_final_model.py)
)
STATUS_PREFIX = "pipeline_checkpoints"


def _sync_one(cmd_base: list, uri: str, local_dir: Path) -> bool:
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"  {uri} -> {local_dir}")
    r = subprocess.run([*cmd_base, uri, str(local_dir)], cwd=str(REPO_ROOT))
    if r.returncode != 0:
        print(f"  Warning: sync returned {r.returncode}", file=sys.stderr)
        return False
    return True


def _scan_errors(sync_dir: Path) -> list[tuple[str, int, str]]:
    """Scan .txt and .log under sync_dir for lines containing ERROR or FATAL. Return (path, line_no, line)."""
    hits = []
    err_pat = re.compile(r"ERROR|FATAL", re.I)
    for path in sync_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in (".txt", ".log") and path.name != "log":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        try:
            rel = path.relative_to(sync_dir)
        except ValueError:
            rel = path
        for i, line in enumerate(text.splitlines(), start=1):
            if err_pat.search(line):
                hits.append((str(rel), i, line.strip()[:120]))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync viz logs and pipeline status from S3")
    parser.add_argument("--profile", default=None, help="AWS CLI profile (e.g. mushin)")
    parser.add_argument("--no-errors-summary", action="store_true", help="Skip ERROR/FATAL summary after sync")
    args = parser.parse_args()

    VIZ_SYNC_DIR.mkdir(parents=True, exist_ok=True)
    profile_msg = f" (profile {args.profile})" if args.profile else ""
    cmd_base = ["aws", "s3", "sync"]
    if args.profile:
        cmd_base.extend(["--profile", args.profile])

    print("Syncing viz logs and status from s3://{}/ to {}{}".format(BUCKET, VIZ_SYNC_DIR, profile_msg))
    print("-" * 60)
    for prefix in LOG_PREFIXES:
        local_dir = VIZ_SYNC_DIR / prefix
        uri = f"s3://{BUCKET}/{prefix}/"
        _sync_one(cmd_base, uri, local_dir)
    # Pipeline checkpoints (status)
    status_local = VIZ_SYNC_DIR / STATUS_PREFIX
    uri = f"s3://{BUCKET}/{STATUS_PREFIX}/"
    _sync_one(cmd_base, uri, status_local)
    print("-" * 60)
    print("Done.")

    if not args.no_errors_summary:
        hits = _scan_errors(VIZ_SYNC_DIR)
        if hits:
            print("\nERROR/FATAL lines in synced logs (first 50):")
            for path, line_no, snippet in hits[:50]:
                print("  {}:{}  {}".format(path, line_no, snippet))
            if len(hits) > 50:
                print("  ... and {} more.".format(len(hits) - 50))
        else:
            print("\nNo ERROR/FATAL lines found in synced logs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
