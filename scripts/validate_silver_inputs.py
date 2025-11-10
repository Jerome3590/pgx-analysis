#!/usr/bin/env python3
"""
Validate and select partitioned silver input paths for APCD orchestrator jobs.

Usage examples:
  python scripts/validate_silver_inputs.py --bucket pgxdatalake --base-prefix silver --job medical
  python scripts/validate_silver_inputs.py --bucket pgxdatalake --base-prefix silver --job pharmacy --check-previous

Outputs on success (exit 0): prints selected paths as JSON to stdout, e.g.:
  {"raw_medical": "s3://pgxdatalake/silver/imputed/medical_partitioned/*.parquet", "raw_pharmacy": null}

Exit code non-zero if no valid inputs found.
"""
import argparse
import json
import sys
import boto3
from botocore.config import Config


def s3_prefix_exists(s3, bucket, prefix, min_keys=1):
    paginator = s3.get_paginator("list_objects_v2")
    kwargs = {"Bucket": bucket, "Prefix": prefix}
    count = 0
    try:
        for page in paginator.paginate(**kwargs):
            if "Contents" in page:
                count += len(page["Contents"])
                if count >= min_keys:
                    return True
    except Exception as e:
        print(f"ERROR listing s3://{bucket}/{prefix}: {e}", file=sys.stderr)
        return False
    return False


def find_preferred_paths(bucket, base_prefix, job):
    # For medical: prefer imputed/medical_partitioned/, then silver/medical/
    # For pharmacy: prefer imputed/pharmacy_partitioned/, then silver/pharmacy/
    s3 = boto3.client("s3", config=Config(retries={"max_attempts": 3}))
    result = {"raw_medical": None, "raw_pharmacy": None}

    if job in ("medical", "both"):
        pref = f"{base_prefix}/imputed/medical_partitioned/"
        if s3_prefix_exists(s3, bucket, pref):
            result["raw_medical"] = f"s3://{bucket}/{pref}*.parquet"
        else:
            fallback = f"{base_prefix}/medical/"
            if s3_prefix_exists(s3, bucket, fallback):
                result["raw_medical"] = f"s3://{bucket}/{fallback}*.parquet"

    if job in ("pharmacy", "both"):
        pref = f"{base_prefix}/imputed/pharmacy_partitioned/"
        if s3_prefix_exists(s3, bucket, pref):
            result["raw_pharmacy"] = f"s3://{bucket}/{pref}*.parquet"
        else:
            fallback = f"{base_prefix}/pharmacy/"
            if s3_prefix_exists(s3, bucket, fallback):
                result["raw_pharmacy"] = f"s3://{bucket}/{fallback}*.parquet"

    return result


def check_previous_workflow(bucket, job):
    # Check for existing orchestrator logs under pgx-repository/build_logs/apcd_input_data/orchestrator/{job}/
    repo_bucket = "pgx-repository"
    prefix = f"build_logs/apcd_input_data/orchestrator/{job}/"
    s3 = boto3.client("s3", config=Config(retries={"max_attempts": 3}))
    exists = s3_prefix_exists(s3, repo_bucket, prefix)
    return exists


def main():
    parser = argparse.ArgumentParser(description="Validate and select silver partitioned input paths")
    parser.add_argument("--bucket", required=True, help="S3 bucket containing silver data (e.g. pgxdatalake)")
    parser.add_argument("--base-prefix", default="silver", help="Base prefix inside the bucket (default: silver)")
    parser.add_argument("--job", choices=["medical", "pharmacy", "both"], default="medical")
    parser.add_argument("--check-previous", action="store_true", help="Check for previous orchestrator workflow logs")
    args = parser.parse_args()

    paths = find_preferred_paths(args.bucket, args.base_prefix, args.job)

    if args.check_previous:
        prev = check_previous_workflow(args.bucket, args.job if args.job != 'both' else 'medical')
        if prev:
            print(f"INFO: Found previous orchestrator logs for job '{args.job}'", file=sys.stderr)
        else:
            print(f"WARN: No previous orchestrator logs found for job '{args.job}'", file=sys.stderr)

    # If job == both, we return both keys. Otherwise return only the requested job's keys.
    if args.job == "medical":
        out = {"raw_medical": paths.get("raw_medical")}
    elif args.job == "pharmacy":
        out = {"raw_pharmacy": paths.get("raw_pharmacy")}
    else:
        out = paths

    print(json.dumps(out))

    # Exit non-zero if requested paths are None
    missing = [k for k, v in out.items() if v is None]
    if missing:
        print(f"ERROR: Missing inputs for: {', '.join(missing)}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
