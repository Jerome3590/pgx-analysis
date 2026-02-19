#!/usr/bin/env python3
"""
Sync dashboard visualization artifacts FROM the live dashboard S3 bucket (jerome.dixon.io)
into the repo so you have local copies of BupaR, DTW, and FP-Growth plots/HTML.

S3 layout (same as upload): s3://jerome-dixon.io/vcu/pgx-risk-calculator/
  - bupar/{cohort}/{age_band}/plots/  (includes plots/lib/ for interactive HTML deps)
  - dtw/{cohort}/{age_band}/plots/ and chart_data.json
  - fpgrowth/{cohort}/{age_band}/plots/

Local layout (canonical):
  - 10_risk_dashboard/visualizations/bupar/outputs/{cohort}/{age_band_fname}/plots/
  - 10_risk_dashboard/visualizations/dtw/outputs/{cohort}/{age_band_fname}/plots/ (+ chart_data at parent)
  - 10_risk_dashboard/visualizations/fpgrowth/outputs/{cohort}/{age_band_fname}/plots/

Age band: S3 uses hyphen (0-12); local uses underscore (0_12).

Usage (from repo root):
  python 9_dashboard_visuals/sync_visuals_from_dashboard_s3.py [--profile PROFILE] [--bupar-only | --dtw-only | --fpgrowth-only]
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUCKET = "jerome-dixon.io"
PREFIX = "vcu/pgx-risk-calculator"
COHORTS = ("opioid_ed", "non_opioid_ed")
AGE_BANDS = ("0-12", "13-24", "25-44", "45-54", "55-64", "65-74", "75-84", "85-114")


def age_band_fname(age_band: str) -> str:
    return age_band.replace("-", "_")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync dashboard visuals from jerome.dixon.io S3")
    parser.add_argument("--profile", default=None, help="AWS CLI profile (e.g. mushin)")
    parser.add_argument("--bupar-only", action="store_true", help="Only sync BupaR")
    parser.add_argument("--dtw-only", action="store_true", help="Only sync DTW")
    parser.add_argument("--fpgrowth-only", action="store_true", help="Only sync FP-Growth")
    args = parser.parse_args()

    cmd_base = ["aws", "s3", "sync"]
    if args.profile:
        cmd_base.extend(["--profile", args.profile])

    sync_bupar = not (args.dtw_only or args.fpgrowth_only)
    sync_dtw = not (args.bupar_only or args.fpgrowth_only)
    sync_fpgrowth = not (args.bupar_only or args.dtw_only)

    for cohort in COHORTS:
        for age_band in AGE_BANDS:
            ab_fname = age_band_fname(age_band)
            if sync_bupar:
                s3_uri = f"s3://{BUCKET}/{PREFIX}/bupar/{cohort}/{age_band}/plots/"
                local = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "bupar" / "outputs" / cohort / ab_fname / "plots"
                local.mkdir(parents=True, exist_ok=True)
                r = subprocess.run([*cmd_base, s3_uri, str(local)], cwd=str(REPO_ROOT))
                if r.returncode != 0:
                    print(f"  [warn] bupar {cohort}/{age_band} sync returned {r.returncode}", file=sys.stderr)
            if sync_dtw:
                s3_uri = f"s3://{BUCKET}/{PREFIX}/dtw/{cohort}/{age_band}/"
                local = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "dtw" / "outputs" / cohort / ab_fname
                local.mkdir(parents=True, exist_ok=True)
                r = subprocess.run([*cmd_base, s3_uri, str(local)], cwd=str(REPO_ROOT))
                if r.returncode != 0:
                    print(f"  [warn] dtw {cohort}/{age_band} sync returned {r.returncode}", file=sys.stderr)
            if sync_fpgrowth:
                s3_uri = f"s3://{BUCKET}/{PREFIX}/fpgrowth/{cohort}/{age_band}/plots/"
                local = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "fpgrowth" / "outputs" / cohort / ab_fname / "plots"
                local.mkdir(parents=True, exist_ok=True)
                r = subprocess.run([*cmd_base, s3_uri, str(local)], cwd=str(REPO_ROOT))
                if r.returncode != 0:
                    print(f"  [warn] fpgrowth {cohort}/{age_band} sync returned {r.returncode}", file=sys.stderr)

    print(f"Done syncing dashboard visuals from s3://{BUCKET}/{PREFIX}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
