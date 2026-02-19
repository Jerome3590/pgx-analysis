#!/usr/bin/env python3
"""
Download one cohort/age_band of model_events.parquet from S3 to local 4_model_data.

S3 path: s3://pgxdatalake/gold/cohorts_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet
Local:   4_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet

Usage (from repo root):
  python 4_model_data/download_one_age_band_from_s3.py --cohort opioid_ed --age-band 0-12
  python 4_model_data/download_one_age_band_from_s3.py --cohort non_opioid_ed --age-band 65-74 --profile mushin
"""

import argparse
import subprocess
import sys
from pathlib import Path

BUCKET = "pgxdatalake"
S3_PREFIX = "gold/cohorts_model_data"
REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download one age band of model_events from S3")
    parser.add_argument("--cohort", required=True, help="e.g. opioid_ed or non_opioid_ed")
    parser.add_argument("--age-band", required=True, help="e.g. 0-12 or 65-74")
    parser.add_argument("--profile", default=None, help="AWS CLI profile (e.g. mushin)")
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT, help="Repo root (default: parent of 4_model_data)")
    args = parser.parse_args()

    cohort = args.cohort
    age_band = args.age_band
    project_root = args.project_root.resolve()
    model_data_root = project_root / "4_model_data"
    local_dir = model_data_root / f"cohort_name={cohort}" / f"age_band={age_band}"
    local_file = local_dir / "model_events.parquet"

    s3_uri = f"s3://{BUCKET}/{S3_PREFIX}/cohort_name={cohort}/age_band={age_band}/model_events.parquet"

    local_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["aws", "s3", "cp", s3_uri, str(local_file)]
    if args.profile:
        cmd.extend(["--profile", args.profile])

    print(f"Downloading {s3_uri} -> {local_file}")
    result = subprocess.run(cmd, cwd=str(project_root))
    if result.returncode != 0:
        print(f"Failed: aws s3 cp returned {result.returncode}", file=sys.stderr)
        return result.returncode
    print(f"Done. Run BupaR for this cell: python 9_dashboard_visuals/bupar/create_bupar_visuals.py --cohort-name {cohort} --age-band {age_band} --force --local-test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
