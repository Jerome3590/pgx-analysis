from __future__ import annotations

import sys
from typing import Iterable

import boto3


def find_model_events(bucket: str, prefixes: Iterable[str] | None = None) -> None:
    """
    Scan an S3 bucket for keys ending in 'model_events.parquet' and print them.

    This is a diagnostic helper to see where model_data was written in a given
    environment (e.g., 'pgx-repository' vs 'pgxdatalake').
    """
    s3 = boto3.client("s3")
    if prefixes is None:
        prefixes = ["gold/", "model_data/", ""]

    print("Searching bucket '{0}' for keys ending with 'model_events.parquet'.".format(bucket))
    found_any = False

    for prefix in prefixes:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("model_events.parquet"):
                    print("s3://{0}/{1}".format(bucket, key))
                    found_any = True

    if not found_any:
        print("No keys ending with 'model_events.parquet' found in bucket '{0}'.".format(bucket))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_model_events_in_bucket.py <bucket-name> [prefix1 prefix2 ...]")
        sys.exit(1)

    bucket_name = sys.argv[1]
    extra_prefixes = sys.argv[2:] or None
    find_model_events(bucket_name, extra_prefixes)


