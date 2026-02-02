#!/usr/bin/env python3
"""
Restore aggregated feature importance in pgx-repository to a previous (full baseline) version.
For opioid_ed 25-44, 45-54, 55-64 the current object is a bad 1-row (n_events) run; this script
copies the previous version (full item-level baseline) to become the current object.

Usage: python 3a_feature_importance/restore_pgx_repo_fi.py [--dry_run]
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import age_band_to_fname

try:
    from py_helpers.common_imports import s3_client
except ImportError:
    import boto3
    s3_client = boto3.client("s3")

PGX_REPO_BUCKET = "pgx-repository"
PGX_REPO_FI_PREFIX = "pgx-analysis/3_feature_importance/outputs"

# opioid_ed age_bands that had bad 1-row overwrites (we restore from previous version)
RESTORE_COHORT_AGE_BANDS = [
    ("opioid_ed", "25-44"),
    ("opioid_ed", "45-54"),
    ("opioid_ed", "55-64"),
]


def get_versions(cohort: str, age_band: str):
    age_band_fname = age_band_to_fname(age_band)
    filename = f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
    s3_key = f"{PGX_REPO_FI_PREFIX}/{filename}"
    resp = s3_client.list_object_versions(Bucket=PGX_REPO_BUCKET, Prefix=s3_key, MaxKeys=10)
    versions = resp.get("Versions", []) or []
    return s3_key, [v for v in versions if v.get("Key") == s3_key]


def restore_one(cohort: str, age_band: str, dry_run: bool):
    s3_key, versions = get_versions(cohort, age_band)
    if len(versions) < 2:
        print(f"  {cohort}/{age_band}: only one version, nothing to restore")
        return
    # Current is IsLatest=True; we want the previous (full baseline) = largest non-current
    by_size = sorted(versions, key=lambda v: v.get("Size", 0), reverse=True)
    current = next((v for v in versions if v.get("IsLatest")), None)
    # Prefer restoring the version that is NOT current and has larger size (full baseline)
    candidates = [v for v in versions if not v.get("IsLatest") and v.get("Size", 0) > 1000]
    if not candidates:
        # Restore the largest previous version
        previous = next((v for v in by_size if not v.get("IsLatest")), None)
        if not previous:
            print(f"  {cohort}/{age_band}: no previous version to restore")
            return
    else:
        previous = max(candidates, key=lambda v: v.get("Size", 0))
    version_id = previous["VersionId"]
    size = previous.get("Size", 0)
    print(f"  {cohort}/{age_band}: restoring VersionId={version_id} (Size={size}) -> current")
    if dry_run:
        print(f"    [DRY RUN] would copy s3://{PGX_REPO_BUCKET}/{s3_key}?versionId={version_id} -> same key")
        return
    copy_source = {"Bucket": PGX_REPO_BUCKET, "Key": s3_key, "VersionId": version_id}
    s3_client.copy_object(
        Bucket=PGX_REPO_BUCKET,
        Key=s3_key,
        CopySource=copy_source,
    )
    print(f"    Restored.")


def main():
    parser = argparse.ArgumentParser(description="Restore pgx-repository aggregated FI to previous version")
    parser.add_argument("--dry_run", action="store_true", help="Print what would be done, do not copy")
    args = parser.parse_args()
    print("Restoring opioid_ed 25-44, 45-54, 55-64 aggregated FI in pgx-repository to previous (full) version.")
    if args.dry_run:
        print("[DRY RUN] No changes will be made.\n")
    for cohort, age_band in RESTORE_COHORT_AGE_BANDS:
        try:
            restore_one(cohort, age_band, args.dry_run)
        except Exception as e:
            print(f"  {cohort}/{age_band}: FAILED - {e}")
    print("\nDone. Verify with: python 3a_feature_importance/load_pgx_repo_fi.py --all")


if __name__ == "__main__":
    main()
