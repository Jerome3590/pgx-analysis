#!/usr/bin/env python3
"""
Generate model_performance_metrics.json for the dashboard Documentation tab.

Reads existing model_metrics_summary.csv from 6_final_model/outputs (or S3) per cohort/age_band,
aggregates into a single JSON (no recomputation of metrics), writes locally and uploads to S3.
Lambda GET /metrics returns this prebuilt artifact from S3 (same pattern as other visuals).

Usage:
    python generate_metrics.py
    python generate_metrics.py --download-s3   # Prefer S3 if local CSVs missing
    python generate_metrics.py --no-upload     # Skip S3 upload (local only)
"""

import sys
import json
import csv
from pathlib import Path
from io import StringIO
from typing import Dict, List, Any

# Project root: this script is in 9_risk_dashboard/data_preparation/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

FINAL_MODEL_DIR = PROJECT_ROOT / "6_final_model" / "outputs"
OUTPUT_DIR = PROJECT_ROOT / "9_risk_dashboard" / "outputs" / "metadata"
METRICS_FILENAME = "model_performance_metrics.json"

REQUIRED_COHORTS = {
    "opioid_ed": ["13-24", "25-44", "45-54", "55-64"],
    "non_opioid_ed": ["65-74", "75-84", "85-94"],
}

S3_BUCKET = "pgxdatalake"
S3_PREFIX = "gold/final_model"
METRICS_S3_KEY = "gold/dashboard/metadata/model_performance_metrics.json"


def _read_csv_row(row: Dict[str, str]) -> Dict[str, Any]:
    """Convert a CSV row to typed dict for JSON."""
    out = {}
    for k, v in row.items():
        if v == "" or v is None:
            out[k] = None
        elif k in ("recall_mean", "pr_auc_mean", "auc_mean", "logloss_mean"):
            try:
                out[k] = round(float(v), 4)
            except (ValueError, TypeError):
                out[k] = None
        elif k == "n_runs":
            try:
                out[k] = int(float(v))
            except (ValueError, TypeError):
                out[k] = None
        elif k == "selected":
            out[k] = str(v).strip().lower() in ("true", "1", "yes")
        else:
            out[k] = v
    return out


def _load_summary_local(cohort: str, age_band: str) -> List[Dict[str, Any]]:
    """Load model_metrics_summary.csv from local 6_final_model/outputs."""
    age_band_fname = age_band.replace("-", "_")
    path = (
        FINAL_MODEL_DIR
        / cohort
        / age_band_fname
        / f"{cohort}_{age_band_fname}_model_metrics_summary.csv"
    )
    if not path.exists():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(_read_csv_row(row))
    return rows


def _load_summary_s3(cohort: str, age_band: str) -> List[Dict[str, Any]]:
    """Load model_metrics_summary.csv from S3."""
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        return []
    age_band_fname = age_band.replace("-", "_")
    key = f"{S3_PREFIX}/{cohort}/{age_band}/{cohort}_{age_band_fname}_model_metrics_summary.csv"
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        body = obj["Body"].read().decode("utf-8")
        rows = []
        for row in csv.DictReader(StringIO(body)):
            rows.append(_read_csv_row(row))
        return rows
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
            return []
        raise
    except Exception:
        return []


def generate_metrics(download_s3: bool = False) -> Dict[str, Any]:
    """Build by_cohort metrics from local and optionally S3."""
    by_cohort: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for cohort, age_bands in REQUIRED_COHORTS.items():
        by_cohort[cohort] = {}
        for age_band in age_bands:
            rows = _load_summary_local(cohort, age_band)
            if not rows and download_s3:
                rows = _load_summary_s3(cohort, age_band)
            by_cohort[cohort][age_band] = rows
    return {"by_cohort": by_cohort, "source": "s3"}


def upload_metrics_to_s3(out_path: Path) -> bool:
    """Upload prebuilt model_performance_metrics.json to S3 for Lambda (no recomputation)."""
    try:
        import boto3
    except ImportError:
        print("  boto3 not available; skipping S3 upload.")
        return False
    try:
        s3 = boto3.client("s3")
        s3.upload_file(str(out_path), S3_BUCKET, METRICS_S3_KEY, ExtraArgs={"ContentType": "application/json"})
        print(f"  Uploaded to s3://{S3_BUCKET}/{METRICS_S3_KEY}")
        return True
    except Exception as e:
        print(f"  Warning: S3 upload failed: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate model_performance_metrics.json from existing 6_final_model artifacts; upload to S3."
    )
    parser.add_argument("--download-s3", action="store_true", help="Fallback to S3 if local CSV missing")
    parser.add_argument("--no-upload", action="store_true", help="Do not upload to S3 (local only)")
    args = parser.parse_args()

    payload = generate_metrics(download_s3=args.download_s3)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / METRICS_FILENAME
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")
    n_bands = sum(len(b) for b in payload["by_cohort"].values())
    n_rows = sum(
        len(rows)
        for cohort_bands in payload["by_cohort"].values()
        for rows in cohort_bands.values()
    )
    print(f"  Cohorts: {len(payload['by_cohort'])}, age bands: {n_bands}, metric rows: {n_rows}")
    if not getattr(args, "no_upload", False):
        upload_metrics_to_s3(out_path)


if __name__ == "__main__":
    main()
