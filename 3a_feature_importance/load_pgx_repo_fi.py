#!/usr/bin/env python3
"""
Load and display aggregated feature importance from pgx-repository for a given cohort/age_band.
Check all cohort/age_bands: --all
Check S3 object versions (previous versions) for a key: --versions --cohort X --age_band Y
Usage: python 3a_feature_importance/load_pgx_repo_fi.py [--cohort opioid_ed] [--age_band 25-44]
"""
import argparse
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from py_helpers.constants import age_band_to_fname

try:
    from py_helpers.common_imports import s3_client
except ImportError:
    import boto3
    s3_client = boto3.client("s3")

PGX_REPO_BUCKET = "pgx-repository"
PGX_REPO_FI_PREFIX = "pgx-analysis/3_feature_importance/outputs"

# All cohort / age_band combinations used in Step 3a
ALL_COHORT_AGE_BANDS = [
    ("opioid_ed", "13-24"),
    ("opioid_ed", "25-44"),
    ("opioid_ed", "45-54"),
    ("opioid_ed", "55-64"),
    ("non_opioid_ed", "65-74"),
    ("non_opioid_ed", "75-84"),
    ("non_opioid_ed", "85-94"),
]


def load_aggregated_fi(cohort: str, age_band: str, version_id: str | None = None):
    age_band_fname = age_band_to_fname(age_band)
    filename = f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
    s3_key = f"{PGX_REPO_FI_PREFIX}/{filename}"
    kwargs = {"Bucket": PGX_REPO_BUCKET, "Key": s3_key}
    if version_id:
        kwargs["VersionId"] = version_id
    obj = s3_client.get_object(**kwargs)
    return pd.read_csv(io.BytesIO(obj["Body"].read())), s3_key


def check_all():
    """Load each cohort/age_band file and print row count and unique feature count."""
    print("pgx-repository aggregated feature importance files:\n")
    for cohort, age_band in ALL_COHORT_AGE_BANDS:
        try:
            df, key = load_aggregated_fi(cohort, age_band)
            n_rows = len(df)
            n_features = df["feature"].nunique() if "feature" in df.columns else 0
            sample = list(df["feature"].head(3).values) if "feature" in df.columns else []
            print(f"  {cohort}/{age_band}: {n_rows} rows, {n_features} unique features  sample={sample}")
        except Exception as e:
            print(f"  {cohort}/{age_band}: FAILED - {e}")
    print()


def get_baseline_summary_df():
    """Load each cohort/age_band file from pgx-repository and return a DataFrame summary for display in notebooks."""
    rows = []
    for cohort, age_band in ALL_COHORT_AGE_BANDS:
        try:
            df, _ = load_aggregated_fi(cohort, age_band)
            n_rows = len(df)
            n_features = df["feature"].nunique() if "feature" in df.columns else 0
            sample = list(df["feature"].head(3).astype(str).values) if "feature" in df.columns else []
            sample_str = ", ".join(sample) if sample else ""
            rows.append({
                "cohort": cohort,
                "age_band": age_band,
                "rows": n_rows,
                "unique_features": n_features,
                "sample": sample_str,
            })
        except Exception as e:
            rows.append({
                "cohort": cohort,
                "age_band": age_band,
                "rows": None,
                "unique_features": None,
                "sample": str(e),
            })
    return pd.DataFrame(rows)


def check_versions(cohort: str, age_band: str):
    """Check bucket versioning and list object versions for the given key (to find previous versions)."""
    age_band_fname = age_band_to_fname(age_band)
    filename = f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
    s3_key = f"{PGX_REPO_FI_PREFIX}/{filename}"
    print(f"Bucket: {PGX_REPO_BUCKET}")
    try:
        v = s3_client.get_bucket_versioning(Bucket=PGX_REPO_BUCKET)
        status = v.get("Status", "Not set")
        print(f"Versioning: {status}")
    except Exception as e:
        print(f"Versioning check failed: {e}")
        return
    try:
        resp = s3_client.list_object_versions(Bucket=PGX_REPO_BUCKET, Prefix=s3_key, MaxKeys=20)
        versions = resp.get("Versions", []) or []
        if not versions:
            print(f"No versions found for {s3_key} (or prefix list returned no Versions).")
            return
        print(f"Object key: {s3_key}")
        print(f"Found {len(versions)} version(s):")
        for v in versions:
            vid = v.get("VersionId", "")
            size = v.get("Size", 0)
            last_modified = v.get("LastModified", "")
            is_latest = v.get("IsLatest", False)
            marker = " (current)" if is_latest else ""
            print(f"  VersionId={vid}  Size={size}  LastModified={last_modified}{marker}")
    except Exception as e:
        print(f"list_object_versions failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Load aggregated FI from pgx-repository")
    parser.add_argument("--cohort", default="opioid_ed", help="Cohort name (e.g. opioid_ed)")
    parser.add_argument("--age_band", default="25-44", help="Age band (e.g. 25-44)")
    parser.add_argument("--all", action="store_true", help="Check all cohort/age_band files (row and feature count)")
    parser.add_argument("--versions", action="store_true", help="Check S3 object versions for this key (previous versions)")
    parser.add_argument("--version_id", default=None, help="Load a specific S3 object version (e.g. previous baseline)")
    args = parser.parse_args()
    if args.all:
        check_all()
        return
    if args.versions:
        check_versions(args.cohort, args.age_band)
        return
    df, key = load_aggregated_fi(args.cohort, args.age_band, version_id=args.version_id)
    print(f"Loaded: s3://{PGX_REPO_BUCKET}/{key}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    if "feature" in df.columns:
        print(f"Unique features: {df['feature'].nunique()}")
        print("\nFirst 20 rows:")
        print(df.head(20).to_string())
        print("\n...")
        print(f"\nLast 5 rows:")
        print(df.tail(5).to_string())
    else:
        print(df.head(30).to_string())
    return df


if __name__ == "__main__":
    main()
