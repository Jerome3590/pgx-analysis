#!/usr/bin/env python3
"""
Local Python workflow for dashboard visuals (pipeline step 9).

Emulates [4_dashboard_visuals.ipynb](../4_dashboard_visuals.ipynb): optional S3 sync of model data,
then BupaR, DTW, and FP-Growth for configured cohort/age_band combinations. Run from repo root
(e.g. from VS Code or terminal).

Usage:
  # Sync from S3 then run all visuals (default combinations)
  python 9_dashboard_visuals/run_dashboard_visuals.py

  # Only sync model data + feature importance from S3, then exit
  python 9_dashboard_visuals/run_dashboard_visuals.py --sync-only

  # Skip sync (data already local), run visuals
  python 9_dashboard_visuals/run_dashboard_visuals.py --no-sync

  # One cohort/age_band, force re-run
  python 9_dashboard_visuals/run_dashboard_visuals.py --no-sync --cohort opioid_ed --age-band 13-24 --force

  # AWS profile for sync
  python 9_dashboard_visuals/run_dashboard_visuals.py --profile my-profile
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from py_helpers.constants import REQUIRED_COHORTS
except ImportError:
    _all_bands = ["0-12", "13-24", "25-44", "45-54", "55-64", "65-74", "75-84", "85-114"]
    REQUIRED_COHORTS = {"opioid_ed": _all_bands, "non_opioid_ed": _all_bands}


def run_sync(profile: str | None) -> bool:
    """Run sync_visualization_data_from_s3.py; return True on success."""
    script = REPO_ROOT / "9_dashboard_visuals" / "sync_visualization_data_from_s3.py"
    if not script.exists():
        print("sync_visualization_data_from_s3.py not found; skip sync.")
        return False
    cmd = [sys.executable, str(script)]
    if profile:
        cmd.extend(["--profile", profile])
    r = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser(
        description="Run dashboard visuals workflow (sync optional, then BupaR, DTW, FP-Growth)."
    )
    ap.add_argument("--sync-only", action="store_true", help="Only sync model data + feature_importance from S3, then exit")
    ap.add_argument("--no-sync", action="store_true", help="Do not sync; assume data already local")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"), help="AWS profile for S3 sync")
    ap.add_argument("--cohort", action="append", dest="cohorts", help="Cohort to run (repeatable); default: all REQUIRED_COHORTS")
    ap.add_argument("--age-band", action="append", dest="age_bands", help="Age band to run (repeatable); default: per-cohort from REQUIRED_COHORTS")
    ap.add_argument("--force", action="store_true", help="Pass --force to BupaR, DTW, FP-Growth")
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers for BupaR and DTW (default 4)")
    ap.add_argument("--fail-fast", action="store_true", default=True, help="Stop on first failure (default True)")
    args = ap.parse_args()

    # Combinations (same logic as 4_dashboard_visuals.ipynb)
    if args.cohorts and args.age_bands:
        combinations = [(c, ab) for c in args.cohorts for ab in args.age_bands]
    elif args.cohorts:
        combinations = [(c, ab) for c in args.cohorts for ab in REQUIRED_COHORTS.get(c, [])]
    elif args.age_bands:
        combinations = [(c, ab) for c, bands in REQUIRED_COHORTS.items() for ab in bands if ab in args.age_bands]
    else:
        combinations = [(c, ab) for c, bands in REQUIRED_COHORTS.items() for ab in bands]

    if not combinations:
        print("No cohort/age_band combinations; check --cohort and --age-band.")
        sys.exit(2)

    print("Dashboard visuals workflow (step 9)")
    print("=" * 60)
    print(f"Repo root: {REPO_ROOT}")
    print(f"Combinations: {len(combinations)}")
    print()

    # Creation code lives in 9_dashboard_visuals (step 9); outputs go to 10_risk_dashboard/visualizations
    step9_root = REPO_ROOT / "9_dashboard_visuals"
    bupar_script = step9_root / "bupar" / "create_bupar_visuals.py"
    dtw_visuals_script = step9_root / "dtw" / "create_dtw_visuals.py"
    fpgrowth_script = step9_root / "fpgrowth" / "create_fpgrowth_visuals.py"
    force_flag = ["--force"] if args.force else []

    # Sync
    if not args.no_sync:
        print("Syncing model data and feature_importance from S3...")
        if not run_sync(args.profile):
            print("Sync failed; exiting.")
            sys.exit(1)
        print()
    if args.sync_only:
        print("--sync-only: done.")
        sys.exit(0)

    # BupaR
    print("BupaR")
    print("-" * 40)
    if not bupar_script.exists():
        print("  BupaR script not found; skip.")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(
                    subprocess.run,
                    [sys.executable, str(bupar_script), "--cohort-name", c, "--age-band", ab] + force_flag,
                    cwd=str(REPO_ROOT),
                    capture_output=False,
                ): (c, ab)
                for c, ab in combinations
            }
            for fut in as_completed(futures):
                c, ab = futures[fut]
                r = fut.result()
                print(f"  [BupaR] {c} / {ab} -> exit {r.returncode}")
                if r.returncode != 0 and args.fail_fast:
                    sys.exit(1)
    print()

    # DTW (visuals only; we do not create DTW features in this pipeline)
    print("DTW")
    print("-" * 40)
    if not dtw_visuals_script.exists():
        print("  DTW visuals script not found; skip.")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(
                    subprocess.run,
                    [sys.executable, str(dtw_visuals_script), "--cohort-name", c, "--age-band", ab, "--project-root", str(REPO_ROOT)] + force_flag,
                    cwd=str(REPO_ROOT),
                    capture_output=False,
                ): (c, ab)
                for c, ab in combinations
            }
            for fut in as_completed(futures):
                c, ab = futures[fut]
                r = fut.result()
                print(f"  [DTW] {c} / {ab} -> exit {r.returncode}")
                if r.returncode != 0 and args.fail_fast:
                    sys.exit(1)
    print()

    # FP-Growth (sequential for memory)
    print("FP-Growth")
    print("-" * 40)
    if not fpgrowth_script.exists():
        print("  FP-Growth script not found; skip.")
    else:
        for c, ab in combinations:
            r = subprocess.run(
                [sys.executable, str(fpgrowth_script), "--cohort-name", c, "--age-band", ab] + force_flag,
                cwd=str(REPO_ROOT),
                capture_output=False,
            )
            print(f"  [FP-Growth] {c} / {ab} -> exit {r.returncode}")
            if r.returncode != 0 and args.fail_fast:
                sys.exit(1)

    print()
    print("Dashboard visuals workflow done.")


if __name__ == "__main__":
    main()
