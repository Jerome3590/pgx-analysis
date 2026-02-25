#!/usr/bin/env python3
"""
Local Python workflow for dashboard visuals (pipeline step 9).

Emulates [4_dashboard_visuals.ipynb](../4_dashboard_visuals.ipynb): optional S3 sync of model data,
then BupaR, DTW (trajectories then visuals), FP-Growth, and Cohort PGx for configured cohort/age_band combinations.
DTW trajectories (create_dtw_trajectories.py) produce the features CSV including N3 time-between metrics;
DTW visuals (create_dtw_visuals.py) produce plots and chart_data.json. Cohort PGx: fetch VIP reports,
build network topology, upload to dashboard S3. Run from repo root.

Usage:
  # Sync from S3 then run all visuals (default combinations)
  python 9_dashboard_visuals/run_dashboard_visuals.py

  # Only sync model data + feature importance from S3, then exit
  python 9_dashboard_visuals/run_dashboard_visuals.py --sync-only

  # Skip sync (data already local), run visuals
  python 9_dashboard_visuals/run_dashboard_visuals.py --no-sync

  # One cohort/age_band, force re-run
  python 9_dashboard_visuals/run_dashboard_visuals.py --no-sync --cohort opioid_ed --age-band 13-24 --force

  # Default: all cohorts and age bands, one worker per combo (capped by CPU). Override workers:
  python 9_dashboard_visuals/run_dashboard_visuals.py --no-sync --workers 32

  # AWS profile for sync
  python 9_dashboard_visuals/run_dashboard_visuals.py --profile my-profile

  # Quick test: one age band, both cohorts (DTW only)
  python 9_dashboard_visuals/run_dtw_test_one_age_band.py --age-band 25-44

  # Skip Cohort PGx (no PharmGKB fetch / network build)
  python 9_dashboard_visuals/run_dashboard_visuals.py --skip-cohort-pgx

  # Run Cohort PGx fetch and build but do not upload to S3
  python 9_dashboard_visuals/run_dashboard_visuals.py --no-cohort-pgx-upload
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
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel workers for BupaR and DTW (default: one per cohort/age_band combo, capped by CPU count)")
    ap.add_argument("--fpgrowth-workers", type=int, default=None,
                    help="Parallel workers for FP-Growth (default: one per cohort/age_band combo, capped by CPU count)")
    ap.add_argument("--fail-fast", action="store_true", default=True, help="Stop on first failure (default True)")
    ap.add_argument("--export-bupar-csv-to-json", action="store_true", help="Pass --export-csv-to-json to BupaR so feature CSVs are exported as JSON in plots/ and uploaded")
    ap.add_argument("--skip-cohort-pgx", action="store_true", help="Skip Cohort PGx (fetch VIP reports, build network topology, upload to S3)")
    ap.add_argument("--no-cohort-pgx-upload", action="store_true", help="Run Cohort PGx fetch and build but do not upload to S3")
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

    # Default: one worker per (cohort, age_band) combo, capped by CPU count
    if args.workers is None:
        args.workers = min(_ncpu, len(combinations))
    # FP-Growth: default = all combos in parallel (max EC2 capacity); each subprocess uses 3 DuckDB threads per item type
    fpgrowth_w = args.fpgrowth_workers if args.fpgrowth_workers is not None else len(combinations)

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
    print(f"Parallel workers: BupaR/DTW={args.workers}, FP-Growth={fpgrowth_w}")
    print()

    # Creation code lives in 9_dashboard_visuals (step 9); outputs go to 10_risk_dashboard/visualizations
    step9_root = REPO_ROOT / "9_dashboard_visuals"
    bupar_script = step9_root / "bupar" / "create_bupar_visuals.py"
    dtw_trajectories_script = step9_root / "dtw" / "create_dtw_trajectories.py"
    dtw_features_script = step9_root / "dtw" / "create_dtw_features.py"
    dtw_visuals_script = step9_root / "dtw" / "create_dtw_visuals.py"
    fpgrowth_script = step9_root / "fpgrowth" / "create_fpgrowth_visuals.py"
    force_flag = ["--force"] if args.force else []
    bupar_extra = ["--export-csv-to-json"] if getattr(args, "export_bupar_csv_to_json", False) else []

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
                    [sys.executable, str(bupar_script), "--cohort-name", c, "--age-band", ab] + force_flag + bupar_extra,
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

    # DTW: 1) Trajectories CSV, 2) Alignment (DTW distances + common sequences), 3) Visuals
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

    print("DTW alignment (create_dtw_features.py)")
    print("-" * 40)
    if not dtw_features_script.exists():
        print("  create_dtw_features.py not found; skip.")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(
                    subprocess.run,
                    [sys.executable, str(dtw_features_script), "--cohort", c, "--age-band", ab] + force_flag,
                    cwd=str(REPO_ROOT),
                    capture_output=False,
                ): (c, ab)
                for c, ab in combinations
            }
            for fut in as_completed(futures):
                c, ab = futures[fut]
                r = fut.result()
                print(f"  [DTW alignment] {c} / {ab} -> exit {r.returncode}")
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

    # Cohort PGx: fetch VIP reports, then build network topology (upload is inside build script, same as BupaR/DTW/FP-Growth)
    if not args.skip_cohort_pgx:
        cohort_pgx_dir = step9_root / "cohort_pgx"
        fetch_pgx_script = cohort_pgx_dir / "fetch_vip_reports.py"
        build_pgx_script = cohort_pgx_dir / "build_network_topology.py"
        reports_dir = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "cohort_pgx" / "reports"
        networks_dir = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "cohort_pgx" / "networks"
        reports_dir.mkdir(parents=True, exist_ok=True)
        networks_dir.mkdir(parents=True, exist_ok=True)

        if fetch_pgx_script.exists() and build_pgx_script.exists():
            print("Cohort PGx: fetch VIP reports")
            print("-" * 40)
            pgx_fetch_workers = min(2, len(combinations))  # API rate limit
            with ThreadPoolExecutor(max_workers=pgx_fetch_workers) as ex:
                pgx_fetch_force = ["--force"] if args.force else []
                futures = {
                    ex.submit(
                        subprocess.run,
                        [
                            sys.executable, str(fetch_pgx_script),
                            "--cohort", c, "--age-band", ab,
                            "--top-n", "50",
                            "--project-root", str(REPO_ROOT),
                            "--output-dir", str(reports_dir),
                        ] + pgx_fetch_force,
                        cwd=str(REPO_ROOT),
                        capture_output=False,
                    ): (c, ab)
                    for c, ab in combinations
                }
                for fut in as_completed(futures):
                    c, ab = futures[fut]
                    r = fut.result()
                    print(f"  [Cohort PGx fetch] {c} / {ab} -> exit {r.returncode}")
                    if r.returncode != 0 and args.fail_fast:
                        sys.exit(1)
            print()

            print("Cohort PGx: build network topology (upload to S3 inside script)")
            print("-" * 40)
            pgx_build_workers = min(4, len(combinations))
            pgx_no_upload = ["--no-upload"] if args.no_cohort_pgx_upload else []
            with ThreadPoolExecutor(max_workers=pgx_build_workers) as ex:
                futures = {}
                for c, ab in combinations:
                    age_band_fname = ab.replace("-", "_")
                    reports_file = reports_dir / f"{c}_{age_band_fname}_vip_reports.json"
                    out_dir = networks_dir / c / age_band_fname
                    if not reports_file.exists():
                        print(f"  [Cohort PGx build] {c} / {ab} -> skip (no reports file)")
                        continue
                    futures[
                        ex.submit(
                            subprocess.run,
                            [
                                sys.executable, str(build_pgx_script),
                                "--reports", str(reports_file),
                                "--output-dir", str(out_dir),
                                "--cohort", c, "--age-band", ab,
                            ] + pgx_no_upload,
                            cwd=str(REPO_ROOT),
                            capture_output=False,
                        )
                    ] = (c, ab)
                for fut in as_completed(futures):
                    c, ab = futures[fut]
                    r = fut.result()
                    print(f"  [Cohort PGx build] {c} / {ab} -> exit {r.returncode}")
                    if r.returncode != 0 and args.fail_fast:
                        sys.exit(1)
            print()
        else:
            print("Cohort PGx: scripts not found; skip.")
            print()

    print()
    print("Dashboard visuals workflow done.")


if __name__ == "__main__":
    main()
