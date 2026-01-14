from __future__ import annotations

import sys

import boto3


def list_python_files(bucket: str, prefix: str) -> None:
    """
    List all .py objects under a given S3 prefix.
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    print("Listing *.py under s3://{0}/{1}".format(bucket, prefix))
    found_any = False

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".py"):
                print("s3://{0}/{1}".format(bucket, key))
                found_any = True

    if not found_any:
        print("No Python files found under s3://{0}/{1}".format(bucket, prefix))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python list_python_in_bucket_prefix.py <bucket-name> <prefix>")
        sys.exit(1)

    list_python_files(sys.argv[1], sys.argv[2])


