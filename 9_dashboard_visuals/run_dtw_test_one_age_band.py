#!/usr/bin/env python3
"""
Run DTW pipeline (trajectories + visuals) for one age band and both cohorts.

Use for a quick smoke test: one age band, opioid_ed and non_opioid_ed.
Requires: 4_model_data (model_events), SHAP/FFA allowed_codes under bupar/outputs.

Usage (from repo root):
  python 9_dashboard_visuals/run_dtw_test_one_age_band.py
  python 9_dashboard_visuals/run_dtw_test_one_age_band.py --age-band 25-44 --force
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STEP9 = REPO_ROOT / "9_dashboard_visuals"
DTW_TRAJECTORIES = STEP9 / "dtw" / "create_dtw_trajectories.py"
DTW_VISUALS = STEP9 / "dtw" / "create_dtw_visuals.py"
BUPAR_OUTPUTS = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "bupar" / "outputs"
COHORTS = ["opioid_ed", "non_opioid_ed"]


def check_prereqs(age_band: str) -> tuple[bool, str | None]:
    """Require allowed_codes and trajectory script. Return (ok, error_message)."""
    age_fname = age_band.replace("-", "_")
    missing = []
    if not DTW_TRAJECTORIES.exists():
        missing.append(str(DTW_TRAJECTORIES))
    if not DTW_VISUALS.exists():
        missing.append(str(DTW_VISUALS))
    for c in COHORTS:
        path = BUPAR_OUTPUTS / f"allowed_codes_shap_ffa_{c}_{age_fname}.json"
        if not path.exists():
            missing.append(f"allowed_codes {c}/{age_band} ({path.name})")
    if missing:
        return False, "Missing: " + "; ".join(missing) + ". Run BupaR or sync allowed codes first."
    return True, None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run DTW trajectories + visuals for one age band and both cohorts (opioid_ed, non_opioid_ed)."
    )
    ap.add_argument(
        "--age-band",
        default="25-44",
        help="Age band to run (default: 25-44)",
    )
    ap.add_argument("--force", action="store_true", help="Pass --force to both scripts")
    args = ap.parse_args()
    age_band = args.age_band
    force_flag = ["--force"] if args.force else []

    ok, err = check_prereqs(age_band)
    if not ok:
        print(err, file=sys.stderr)
        return 1

    print("DTW test: one age band, both cohorts")
    print("=" * 50)
    print(f"Age band: {age_band}")
    print(f"Cohorts: {COHORTS}")
    print()

    failed = []
    for cohort in COHORTS:
        # 1) Trajectories
        r1 = subprocess.run(
            [sys.executable, str(DTW_TRAJECTORIES), "--cohort", cohort, "--age-band", age_band] + force_flag,
            cwd=str(REPO_ROOT),
            capture_output=False,
        )
        if r1.returncode != 0:
            failed.append(f"{cohort} trajectories")
            continue
        print(f"  [OK] {cohort} trajectories")
        # 2) Visuals
        r2 = subprocess.run(
            [sys.executable, str(DTW_VISUALS), "--cohort-name", cohort, "--age-band", age_band, "--project-root", str(REPO_ROOT)] + force_flag,
            cwd=str(REPO_ROOT),
            capture_output=False,
        )
        if r2.returncode != 0:
            failed.append(f"{cohort} visuals")
        else:
            print(f"  [OK] {cohort} visuals")

    if failed:
        print()
        print("Failed:", ", ".join(failed), file=sys.stderr)
        return 1
    print()
    print("All DTW steps completed for", age_band, "and both cohorts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
