#!/usr/bin/env python3
"""
Upload causal dashboard JSON to S3 for the Causal Analysis tab.

Scans 10_risk_dashboard/visualizations/causal/{cohort}/{age_band_fname}/ for dashboard_data.json (EC2 uses underscore)
and uploads each to the dashboard bucket as visualizations/causal/{cohort}/{age_band}/causal_data.json (S3 uses hyphen).
Lambda (GET /visualizations/causal) reads from S3; this script is run during deployment
(5_build_and_deploy.ipynb or pgx_dashboard_visuals.py) so the tab has data.

Usage (from repo root):
    python 10_risk_dashboard/data_preparation/upload_causal_outputs_to_s3.py

Environment:
    S3_DASHBOARD_BUCKET (default: jerome-dixon.io)
    S3_DASHBOARD_PREFIX (default: vcu/pgx-risk-calculator)
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CAUSAL_VISUALS_DIR = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "causal"


def main() -> int:
    bucket = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
    prefix = (os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator") or "").strip("/")

    try:
        import boto3
    except ImportError:
        print("boto3 not available; install with: pip install boto3", file=sys.stderr)
        return 1

    if not OUTPUTS_DIR.exists():
        print(f"No outputs dir: {OUTPUTS_DIR}; nothing to upload.")
        return 0

    uploaded = 0
    s3 = boto3.client("s3")
    for cohort_dir in OUTPUTS_DIR.iterdir():
        if not cohort_dir.is_dir() or cohort_dir.name.startswith("."):
            continue
        cohort = cohort_dir.name
        for age_dir in cohort_dir.iterdir():
            if not age_dir.is_dir() or age_dir.name.startswith("."):
                continue
            age_band_fname = age_dir.name  # EC2 dir: 25_44
            age_band_s3 = age_band_fname.replace("_", "-")  # S3 path: 25-44
            json_path = age_dir / "dashboard_data.json"
            if not json_path.exists():
                continue
            key = f"{prefix}/visualizations/causal/{cohort}/{age_band_s3}/causal_data.json"
            try:
                s3.upload_file(
                    str(json_path),
                    bucket,
                    key,
                    ExtraArgs={"ContentType": "application/json"},
                )
                print(f"  ✓ Causal data: {cohort}/{age_band_s3} -> s3://{bucket}/{key}")
                uploaded += 1
            except Exception as e:
                print(f"  ⚠ Upload failed {cohort}/{age_band_s3}: {e}", file=sys.stderr)

    if uploaded:
        print(f"Causal dashboard JSON: {uploaded} file(s) uploaded to S3.")
    else:
        print("No dashboard_data.json found under 10_risk_dashboard/visualizations/causal/ (run combine_shap_ffa_results or run_shap_ffa_workflow first).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
