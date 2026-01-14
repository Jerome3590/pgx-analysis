import os
from pathlib import Path

import boto3

from py_helpers.constants import S3_BUCKET, COHORT_NAMES, AGE_BANDS


def _s3_client():
    return boto3.client("s3")


def _download_prefix(bucket: str, prefix: str, dest_root: Path) -> int:
    """
    Download all objects under an S3 prefix into dest_root, preserving
    the key structure under that prefix.

    Returns the number of files downloaded.
    """
    s3 = _s3_client()
    dest_root.mkdir(parents=True, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    downloaded = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Skip "directories"
            if key.endswith("/"):
                continue

            rel = key[len(prefix) :].lstrip("/")
            if not rel:
                continue

            dest_path = dest_root / rel
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"downloading s3://{bucket}/{key} -> {dest_path}")
            s3.download_file(bucket, key, str(dest_path))
            downloaded += 1

    return downloaded


def download_feature_importance(
    local_root: Path = Path("3_feature_importance") / "from_s3",
) -> None:
    """
    Download all feature-importance artifacts for both cohorts from S3.

    This covers both the newer layout:
        gold/feature_importance/{cohort}/{age_band}/...
    and the earlier partitioned layout:
        gold/feature_importance/cohort_name={cohort}/age_band={age_band}/event_year={year}/...
    """
    bucket = S3_BUCKET
    local_root.mkdir(parents=True, exist_ok=True)

    total = 0

    # Newer, simpler layout used by python feature_importance_utils.py
    for cohort in COHORT_NAMES:
        prefix = f"gold/feature_importance/{cohort}/"
        dest = local_root / "by_cohort" / cohort
        print(f"\n=== Syncing feature importance (cohort layout): s3://{bucket}/{prefix} -> {dest}")
        total += _download_prefix(bucket, prefix, dest)

    # Older layout used by R visualizations and some QA utilities
    for cohort in COHORT_NAMES:
        for age_band in AGE_BANDS:
            prefix = f"gold/feature_importance/cohort_name={cohort}/age_band={age_band}/"
            dest = local_root / "by_partition" / f"cohort_name={cohort}" / f"age_band={age_band}"
            print(
                f"\n=== Syncing feature importance (partition layout): "
                f"s3://{bucket}/{prefix} -> {dest}"
            )
            total += _download_prefix(bucket, prefix, dest)

    print(f"\nFeature importance download complete ({total} files).")


def download_model_data(
    cohorts=None,
    age_bands=None,
    local_root: Path = Path("model_data"),
) -> None:
    """
    Download model_events.parquet for the requested cohorts/age-bands.

    Preferred layout in pgxdatalake:
      gold/cohorts_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet

    We also try a couple of legacy layouts as fallbacks:
      - gold/model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet
      - gold/model_data/{cohort}/{age_band}/model_events.parquet

    All are mirrored under the local `model_data/` folder as:
      model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet
    """
    bucket = S3_BUCKET
    cohorts = cohorts or COHORT_NAMES
    age_bands = age_bands or AGE_BANDS

    local_root.mkdir(parents=True, exist_ok=True)
    s3 = _s3_client()

    downloaded = 0

    for cohort in cohorts:
        for age_band in age_bands:
            print(f"\n=== Checking model data for cohort={cohort}, age_band={age_band}")

            # Layout 1 (preferred): cohorts_model_data with cohort_name/age_band partitions
            key1 = (
                f"gold/cohorts_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet"
            )
            # Layout 2: legacy partitioned layout
            key2 = (
                f"gold/model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet"
            )
            # Layout 3: simpler cohort/age_band path
            key3 = f"gold/model_data/{cohort}/{age_band}/model_events.parquet"

            for key in (key1, key2, key3):
                try:
                    s3.head_object(Bucket=bucket, Key=key)
                except s3.exceptions.ClientError as e:
                    # 404/403 → treat as "not found" and try the next layout
                    code = e.response.get("Error", {}).get("Code")
                    if code in ("404", "403", "NoSuchKey"):
                        continue
                    # Anything else: re-raise
                    raise
                else:
                    # Found – download to a clean, human-friendly local layout
                    dest = (
                        local_root
                        / f"cohort_name={cohort}"
                        / f"age_band={age_band}"
                        / "model_events.parquet"
                    )
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    print(f"downloading s3://{bucket}/{key} -> {dest}")
                    s3.download_file(bucket, key, str(dest))
                    downloaded += 1
                    break  # no need to try the second layout once one worked

    if downloaded == 0:
        print(
            "\nNo model_events.parquet files were found with the expected layouts.\n"
            "Double-check the S3 prefixes for model_data in your environment."
        )
    else:
        print(f"\nModel data download complete ({downloaded} files).")


def download_feature_engineering(
    cohorts=None,
    age_bands=None,
    local_root: Path = Path("feature_engineering") / "from_s3",
) -> None:
    """
    Download FP-Growth, BupaR, DTW, and PGx feature engineering outputs from S3.

    Expected S3 layout (per WORKFLOW_COMPLETE_SUMMARY):
      s3://{bucket}/gold/feature_engineering/{step}/{cohort}/{age_band}/...
        where step in {4_fpgrowth, 5_bupar, 6_dtw, 7_pgx}
    """
    bucket = S3_BUCKET
    cohorts = cohorts or COHORT_NAMES
    age_bands = age_bands or AGE_BANDS

    steps = ["4_fpgrowth", "5_bupar", "6_dtw", "7_pgx"]

    total = 0
    for step in steps:
        for cohort in cohorts:
            for age_band in age_bands:
                prefix = f"gold/feature_engineering/{step}/{cohort}/{age_band}/"
                dest = local_root / step / cohort / age_band
                print(
                    f"\n=== Syncing feature engineering step={step}, "
                    f"cohort={cohort}, age_band={age_band}"
                )
                total += _download_prefix(bucket, prefix, dest)

    print(f"\nFeature engineering download complete ({total} files).")


if __name__ == "__main__":
    """
    Convenience CLI:

      python download_s3_artifacts.py           # download FI + model data
      python download_s3_artifacts.py fi        # feature importance only
      python download_s3_artifacts.py model     # model data only
      python download_s3_artifacts.py features  # feature engineering (FP-Growth/BupaR/DTW/PGx)
      python download_s3_artifacts.py all       # FI + model data + feature engineering
    """
    import sys

    args = sys.argv[1:]
    modes = {"fi", "model", "features", "all"}

    if not args:
        # Default: FI + model data
        download_feature_importance()
        download_model_data()
    else:
        arg = args[0].lower()
        if arg not in modes:
            print("Unrecognized mode '{0}'. Expected one of: fi, model, features, all.".format(arg))
            sys.exit(1)
        if arg == "fi":
            download_feature_importance()
        elif arg == "model":
            download_model_data()
        elif arg == "features":
            download_feature_engineering()
        elif arg == "all":
            download_feature_importance()
            download_model_data()
            download_feature_engineering()


