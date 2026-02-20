#!/usr/bin/env python3
"""
Use logs/85_114 (combined_importance and other Combine outputs) to run BupaR and FP-Growth
for age band 85-114 for both cohorts (opioid_ed, non_opioid_ed).

Copies logs/85_114/opioid_ed -> 10_risk_dashboard/outputs/opioid_ed/85_114/
       logs/85_114/polypharmacy -> 10_risk_dashboard/outputs/non_opioid_ed/85_114/
then runs BupaR and FP-Growth for (opioid_ed, 85-114) and (non_opioid_ed, 85-114) with --skip-sync
(allowed_codes are built from the copied combined_importance.csv).

Requires model_events.parquet for both cohort/85-114 (e.g. from sync or 4_model_data).
Run from repo root:
  python 9_dashboard_visuals/run_bupar_fpgrowth_85_114_from_logs.py
  python 9_dashboard_visuals/run_bupar_fpgrowth_85_114_from_logs.py --force
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

STEP9_ROOT = Path(__file__).resolve().parent
REPO_ROOT = STEP9_ROOT.parent
LOGS_85_114 = REPO_ROOT / "logs" / "85_114"
DASHBOARD_OUTPUTS = REPO_ROOT / "10_risk_dashboard" / "outputs"
RUNNER = STEP9_ROOT / "run_one_age_band_bupar_fpgrowth.py"

# logs/85_114 uses "polypharmacy" for the POLYPHARMACY COHORT; partition name is non_opioid_ed
COHORT_COPY = [
    ("opioid_ed", "opioid_ed"),
    ("polypharmacy", "non_opioid_ed"),
]


def main() -> int:
    if not LOGS_85_114.exists():
        print(f"Missing {LOGS_85_114}; nothing to copy.")
        return 1
    if not RUNNER.exists():
        print(f"Missing {RUNNER}")
        return 1

    # Copy logs/85_114 -> 10_risk_dashboard/outputs so allowed_codes can be built from combined_importance
    print("Copying logs/85_114 -> 10_risk_dashboard/outputs/...")
    for src_cohort, dest_cohort in COHORT_COPY:
        src_dir = LOGS_85_114 / src_cohort
        dest_dir = DASHBOARD_OUTPUTS / dest_cohort / "85_114"
        if not src_dir.exists():
            print(f"  Skip copy: {src_dir} not found")
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in src_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, dest_dir / f.name)
                print(f"  Copied {f.name} -> {dest_dir.relative_to(REPO_ROOT)}")
    print("Done copying.\n")

    # Run BupaR + FP-Growth for 85-114 for both cohorts (--skip-sync: use local combined_importance)
    force = ["--force"] if "--force" in sys.argv else []
    for cohort in ("opioid_ed", "non_opioid_ed"):
        cmd = [
            sys.executable,
            str(RUNNER),
            "--cohort", cohort,
            "--age-band", "85-114",
            "--skip-sync",
        ] + force
        print(f"Running: {' '.join(cmd)}")
        r = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if r.returncode != 0:
            print(f"Failed (exit {r.returncode})")
            return r.returncode
        print()
    print("BupaR + FP-Growth for 85-114 (opioid_ed, non_opioid_ed) done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
