#!/usr/bin/env python3
"""
Run BupaR and FP-Growth visuals for one cohort/age_band that is already local.

Always downloads allowed_codes from S3 first (never run with all codes). Then runs BupaR, then FP-Growth.

Requires:
- model_events.parquet for the cohort/age_band (e.g. 4_model_data/cohort_name=opioid_ed/age_band=85-114/)
- AWS access to s3://pgxdatalake/gold/bupar/allowed_codes/ (allowed codes are created on EC2)

Usage (from repo root):
  python 9_dashboard_visuals/run_one_age_band_bupar_fpgrowth.py --cohort opioid_ed --age-band 85-114
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STEP9_ROOT = Path(__file__).resolve().parent  # 9_dashboard_visuals
REPO_ROOT = STEP9_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SYNC_SCRIPT = STEP9_ROOT / "sync_visualization_data_from_s3.py"
BUPAR_SCRIPT = STEP9_ROOT / "bupar" / "create_bupar_visuals.py"
FPGROWTH_SCRIPT = STEP9_ROOT / "fpgrowth" / "create_fpgrowth_visuals.py"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download allowed_codes from S3, then run BupaR and FP-Growth for one cohort/age_band."
    )
    ap.add_argument("--cohort", required=True, help="Cohort (e.g. opioid_ed)")
    ap.add_argument("--age-band", required=True, help="Age band (e.g. 85-114, more events)")
    ap.add_argument("--skip-sync", action="store_true", help="Skip downloading allowed_codes from S3 (use if already present)")
    ap.add_argument("--force", action="store_true", help="Pass --force to BupaR and FP-Growth")
    args = ap.parse_args()

    cwd = str(REPO_ROOT)

    # 0. Download allowed_codes from S3 (never run with all codes)
    if not args.skip_sync:
        sync_cmd = [sys.executable, str(SYNC_SCRIPT), "--allowed-codes-only"]
        print("[0/3] Downloading allowed_codes from S3: %s" % " ".join(sync_cmd))
        r0 = subprocess.run(sync_cmd, cwd=cwd)
        if r0.returncode != 0:
            print("Sync failed (exit %s). Fix AWS/S3 access and re-run." % r0.returncode)
            return r0.returncode
        print("Allowed codes sync OK.\n")
    else:
        print("[0/3] Skipping allowed_codes sync (--skip-sync).\n")

    # 1. BupaR
    bupar_cmd = [
        sys.executable,
        str(BUPAR_SCRIPT),
        "--cohort-name", args.cohort,
        "--age-band", args.age_band,
    ]
    if args.force:
        bupar_cmd.append("--force")

    print("[1/3] BupaR: %s" % " ".join(bupar_cmd))
    r = subprocess.run(bupar_cmd, cwd=cwd)
    if r.returncode != 0:
        print("BupaR failed (exit %s). Fix errors above, then re-run." % r.returncode)
        return r.returncode
    print("BupaR OK.\n")

    # 2. FP-Growth
    fpgrowth_cmd = [
        sys.executable,
        str(FPGROWTH_SCRIPT),
        "--cohort-name", args.cohort,
        "--age-band", args.age_band,
    ]
    if args.force:
        fpgrowth_cmd.append("--force")

    print("[2/3] FP-Growth: %s" % " ".join(fpgrowth_cmd))
    r2 = subprocess.run(fpgrowth_cmd, cwd=cwd)
    if r2.returncode != 0:
        print("FP-Growth failed (exit %s). Re-run without --skip-sync to refresh allowed_codes from S3." % r2.returncode)
        return r2.returncode
    print("FP-Growth OK.")

    print("\n[3/3] Done. Outputs under 10_risk_dashboard/visualizations/bupar/outputs/ and .../fpgrowth/outputs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
