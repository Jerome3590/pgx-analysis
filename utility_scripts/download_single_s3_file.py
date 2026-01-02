from __future__ import annotations

import sys
from pathlib import Path

import boto3


def download_s3_file(s3_uri: str, local_path: str | None = None) -> None:
    """
    Download a single S3 object to a local path.

    Example:
      python download_single_s3_file.py s3://pgx-repository/pgx-analysis/4a_model_data/create_model_data.py
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError("S3 URI must start with s3://")

    # Parse bucket and key
    without_scheme = s3_uri[len("s3://") :]
    bucket, key = without_scheme.split("/", 1)

    if local_path is None:
        local_path = Path(key).name

    dest = Path(local_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")
    print("downloading {0} -> {1}".format(s3_uri, dest))
    s3.download_file(bucket, key, str(dest))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_single_s3_file.py s3://bucket/key [local_path]")
        sys.exit(1)

    s3_uri_arg = sys.argv[1]
    local_arg = sys.argv[2] if len(sys.argv) > 2 else None
    download_s3_file(s3_uri_arg, local_arg)


