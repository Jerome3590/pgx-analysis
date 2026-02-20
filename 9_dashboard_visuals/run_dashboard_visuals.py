#!/usr/bin/env python3
"""
Local Python workflow for dashboard visuals (pipeline step 9).

Emulates [4_dashboard_visuals.ipynb](../4_dashboard_visuals.ipynb): optional S3 sync of model data,
then BupaR, DTW (trajectories then visuals), and FP-Growth for configured cohort/age_band combinations.
DTW trajectories (create_dtw_trajectories.py) produce the features CSV including N3 time-between metrics;
DTW visuals (create_dtw_visuals.py) produce plots and chart_data.json. Run from repo root.

Usage:
  # Sync from S3 then run all visuals (default combinations)
  python 9_dashboard_visuals/run_dashboard_visuals.py

  # Only sync model data + feature importance from S3, then exit
  python 9_dashboard_visuals/run_dashboard_visuals.py --sync-only

  # Skip sync (data already local), run visuals
  python 9_dashboard_visuals/run_dashboard_visuals.py --no-sync

  # One cohort/age_band, force re-run
  python 9_dashboard_visuals/run_dashboard_visuals.py --no-sync --cohort opioid_ed --age-band 13-24 --force

  # EC2: use more workers to utilize CPU (default is min(32, cpu_count))
  python 9_dashboard_visuals/run_dashboard_visuals.py --no-sync --workers 32

  # AWS profile for sync
  python 9_dashboard_visuals/run_dashboard_visuals.py --profile my-profile

  # Quick test: one age band, both cohorts (DTW only)
  python 9_dashboard_visuals/run_dtw_test_one_age_band.py --age-band 25-44
"""

from __future__ import annotations

import argparse
import json
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


def check_shap_ffa_allowed_codes_prerequisite(combinations: list, repo_root: Path) -> tuple[bool, str | None]:
    """
    Require SHAP/FFA combined allowed codes files for all (cohort, age_band).
    Returns (True, None) on success, (False, error_message) if any file is missing or empty.
    """
    bupar_outputs = repo_root / "10_risk_dashboard" / "visualizations" / "bupar" / "outputs"
    missing = []
    empty = []
    for cohort_name, age_band in combinations:
        age_band_fname = age_band.replace("-", "_")
        path = bupar_outputs / f"allowed_codes_shap_ffa_{cohort_name}_{age_band_fname}.json"
        if not path.exists():
            missing.append(f"{cohort_name}/{age_band} ({path.name})")
        else:
            try:
                with open(path, encoding="utf-8") as f:
                    codes = json.load(f)
                if not codes or (isinstance(codes, list) and len(codes) == 0):
                    empty.append(f"{cohort_name}/{age_band} ({path.name})")
            except Exception as e:
                empty.append(f"{cohort_name}/{age_band} ({path.name}): {e}")
    if missing or empty:
        msg = "SHAP/FFA combined allowed codes are required for BupaR, DTW, and FP-Growth (prerequisite).\n"
        if missing:
            msg += f"  Missing: {', '.join(missing)}\n"
        if empty:
            msg += f"  Empty or invalid: {', '.join(empty)}\n"
        msg += "  Allowed codes are created on EC2; locally run: python 9_dashboard_visuals/sync_visualization_data_from_s3.py --allowed-codes-only"
        return False, msg
    return True, None


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
    _ncpu = getattr(os, "cpu_count", lambda: 4)() or 4
    _default_workers = min(32, max(4, _ncpu))
    ap.add_argument("--workers", type=int, default=_default_workers,
                    help="Parallel workers for BupaR and DTW (default: min(32, cpu_count), use e.g. 32 on EC2)")
    ap.add_argument("--fpgrowth-workers", type=int, default=None,
                    help="Parallel workers for FP-Growth (default: min(8, workers)); lower than BupaR/DTW to limit memory")
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

    # Ensure allowed-codes files exist: try to generate from SHAP/FFA when missing or empty
    try:
        from py_helpers.env_utils import get_data_root
        data_root = get_data_root()
        data_root = Path(data_root) if data_root else None
    except Exception:
        data_root = None
    bupar_outputs = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "bupar" / "outputs"
    bupar_outputs.mkdir(parents=True, exist_ok=True)
    try:
        from py_helpers.shap_ffa_fpgrowth_utils import write_shap_ffa_allowed_codes_for_bupar
    except ImportError:
        write_shap_ffa_allowed_codes_for_bupar = None
    if write_shap_ffa_allowed_codes_for_bupar:
        for cohort_name, age_band in combinations:
            age_band_fname = age_band.replace("-", "_")
            path = bupar_outputs / f"allowed_codes_shap_ffa_{cohort_name}_{age_band_fname}.json"
            if not path.exists() or path.stat().st_size == 0:
                if write_shap_ffa_allowed_codes_for_bupar(
                    cohort_name, age_band, path, top_n=500,
                    project_root=REPO_ROOT, data_root=data_root,
                ):
                    print(f"  Wrote allowed codes: {path.name}")
                elif path.exists() and path.stat().st_size == 0:
                    path.unlink(missing_ok=True)

    ok, err = check_shap_ffa_allowed_codes_prerequisite(combinations, REPO_ROOT)
    if not ok:
        print(err, file=sys.stderr)
        sys.exit(1)

    print("Dashboard visuals workflow (step 9)")
    print("=" * 60)
    print(f"Repo root: {REPO_ROOT}")
    print(f"Combinations: {len(combinations)}")
    fpgrowth_w = args.fpgrowth_workers if args.fpgrowth_workers is not None else min(8, args.workers)
    print(f"Parallel workers: BupaR/DTW={args.workers}, FP-Growth={fpgrowth_w}")
    print()

    # Creation code lives in 9_dashboard_visuals (step 9); outputs go to 10_risk_dashboard/visualizations
    step9_root = REPO_ROOT / "9_dashboard_visuals"
    bupar_script = step9_root / "bupar" / "create_bupar_visuals.py"
    dtw_trajectories_script = step9_root / "dtw" / "create_dtw_trajectories.py"
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

    # DTW: 1) Create trajectory features CSV (N3 time-between metrics, etc.), 2) Create plots and chart_data
    print("DTW trajectories (create_dtw_trajectories.py)")
    print("-" * 40)
    if not dtw_trajectories_script.exists():
        print("  create_dtw_trajectories.py not found; skip.")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(
                    subprocess.run,
                    [sys.executable, str(dtw_trajectories_script), "--cohort", c, "--age-band", ab] + force_flag,
                    cwd=str(REPO_ROOT),
                    capture_output=False,
                ): (c, ab)
                for c, ab in combinations
            }
            for fut in as_completed(futures):
                c, ab = futures[fut]
                r = fut.result()
                print(f"  [DTW trajectories] {c} / {ab} -> exit {r.returncode}")
                if r.returncode != 0 and args.fail_fast:
                    sys.exit(1)
    print()

    print("DTW visuals (create_dtw_visuals.py)")
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
                print(f"  [DTW visuals] {c} / {ab} -> exit {r.returncode}")
                if r.returncode != 0 and args.fail_fast:
                    sys.exit(1)
    print()

    # FP-Growth (parallel; fewer workers than BupaR/DTW to balance memory)
    print("FP-Growth")
    print("-" * 40)
    if not fpgrowth_script.exists():
        print("  FP-Growth script not found; skip.")
    else:
        print(f"  Processing {len(combinations)} cohort/age_band combinations with {fpgrowth_w} parallel workers...")
        with ThreadPoolExecutor(max_workers=fpgrowth_w) as ex:
            futures = {
                ex.submit(
                    subprocess.run,
                    [sys.executable, str(fpgrowth_script), "--cohort-name", c, "--age-band", ab] + force_flag,
                    cwd=str(REPO_ROOT),
                    capture_output=False,
                ): (c, ab)
                for c, ab in combinations
            }
            completed = 0
            for fut in as_completed(futures):
                c, ab = futures[fut]
                r = fut.result()
                completed += 1
                print(f"  [FP-Growth {completed}/{len(combinations)}] {c} / {ab} -> exit {r.returncode}")
                if r.returncode != 0 and args.fail_fast:
                    sys.exit(1)

    print()
    print("Dashboard visuals workflow done.")


if __name__ == "__main__":
    main()
