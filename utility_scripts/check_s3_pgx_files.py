#!/usr/bin/env python3
"""
Quick script to check S3 for PGx prerequisite files.
"""

import sys
import io
import boto3
from botocore.exceptions import ClientError

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

S3_BUCKET = "pgxdatalake"
s3_client = boto3.client("s3")


def check_s3_object_exists(bucket: str, key: str) -> tuple[bool, int]:
    """Check if an S3 object exists and return its size."""
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        return True, response.get("ContentLength", 0)
    except ClientError:
        return False, 0


def list_s3_prefix(bucket: str, prefix: str) -> list[str]:
    """List all objects with a given prefix."""
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        objects = []
        for page in pages:
            if "Contents" in page:
                objects.extend([obj["Key"] for obj in page["Contents"]])
        return objects
    except Exception as e:
        print(f"Error listing {prefix}: {e}")
        return []


print("=" * 70)
print("Checking S3 for PGx prerequisite files...")
print("=" * 70)

# Check global cache paths
print("\n1. Global Cache Files:")
print("-" * 70)

global_mappings_key = "gold/pgx_features/global/pgx_drug_gene_mappings_global.csv"
global_freq_key = "gold/pgx_features/global/pgx_allele_frequencies_global.csv"

exists, size = check_s3_object_exists(S3_BUCKET, global_mappings_key)
if exists:
    print(f"[OK] Global drug-gene mappings: s3://{S3_BUCKET}/{global_mappings_key}")
    print(f"  Size: {size:,} bytes ({size/1024:.2f} KB)")
else:
    print(f"[NOT FOUND] Global drug-gene mappings")

exists, size = check_s3_object_exists(S3_BUCKET, global_freq_key)
if exists:
    print(f"[OK] Global allele frequencies: s3://{S3_BUCKET}/{global_freq_key}")
    print(f"  Size: {size:,} bytes ({size/1024:.2f} KB)")
else:
    print(f"[NOT FOUND] Global allele frequencies")

# Check cohort-level paths
print("\n2. Cohort-Level Files (opioid_ed):")
print("-" * 70)

cohort_mappings_key = "gold/pgx_features/opioid_ed/opioid_ed_drug_gene_mappings.csv"
cohort_freq_key = "gold/pgx_features/opioid_ed/opioid_ed_allele_frequencies.csv"

exists, size = check_s3_object_exists(S3_BUCKET, cohort_mappings_key)
if exists:
    print(f"[OK] Cohort drug-gene mappings: s3://{S3_BUCKET}/{cohort_mappings_key}")
    print(f"  Size: {size:,} bytes ({size/1024:.2f} KB)")
else:
    print(f"[NOT FOUND] Cohort drug-gene mappings")

exists, size = check_s3_object_exists(S3_BUCKET, cohort_freq_key)
if exists:
    print(f"[OK] Cohort allele frequencies: s3://{S3_BUCKET}/{cohort_freq_key}")
    print(f"  Size: {size:,} bytes ({size/1024:.2f} KB)")
else:
    print(f"[NOT FOUND] Cohort allele frequencies")

# Check age-band specific paths (legacy)
print("\n3. Age-Band Specific Files (legacy, 13-24):")
print("-" * 70)

age_band_mappings_key = "gold/pgx_features/opioid_ed/13-24/opioid_ed_drug_gene_mappings.csv"
age_band_freq_key = "gold/pgx_features/opioid_ed/13-24/opioid_ed_allele_frequencies.csv"

exists, size = check_s3_object_exists(S3_BUCKET, age_band_mappings_key)
if exists:
    print(f"[OK] Age-band drug-gene mappings: s3://{S3_BUCKET}/{age_band_mappings_key}")
    print(f"  Size: {size:,} bytes ({size/1024:.2f} KB)")
else:
    print(f"[NOT FOUND] Age-band drug-gene mappings")

exists, size = check_s3_object_exists(S3_BUCKET, age_band_freq_key)
if exists:
    print(f"[OK] Age-band allele frequencies: s3://{S3_BUCKET}/{age_band_freq_key}")
    print(f"  Size: {size:,} bytes ({size/1024:.2f} KB)")
else:
    print(f"[NOT FOUND] Age-band allele frequencies")

# Search for any drug_gene or allele_frequency files
print("\n4. Searching for any PGx mapping/frequency files:")
print("-" * 70)

all_pgx_files = list_s3_prefix(S3_BUCKET, "gold/pgx_features/")
mapping_files = [f for f in all_pgx_files if "drug_gene" in f.lower()]
freq_files = [f for f in all_pgx_files if "allele_freq" in f.lower()]

if mapping_files:
    print(f"\nFound {len(mapping_files)} drug-gene mapping file(s):")
    for f in mapping_files[:10]:  # Show first 10
        exists, size = check_s3_object_exists(S3_BUCKET, f)
        print(f"  - s3://{S3_BUCKET}/{f} ({size:,} bytes)")
    if len(mapping_files) > 10:
        print(f"  ... and {len(mapping_files) - 10} more")
else:
    print("No drug-gene mapping files found")

if freq_files:
    print(f"\nFound {len(freq_files)} allele frequency file(s):")
    for f in freq_files[:10]:  # Show first 10
        exists, size = check_s3_object_exists(S3_BUCKET, f)
        print(f"  - s3://{S3_BUCKET}/{f} ({size:,} bytes)")
    if len(freq_files) > 10:
        print(f"  ... and {len(freq_files) - 10} more")
else:
    print("No allele frequency files found")

# Summary
print("\n" + "=" * 70)
print("Summary:")
has_global_mappings = check_s3_object_exists(S3_BUCKET, global_mappings_key)[0]
has_cohort_mappings = check_s3_object_exists(S3_BUCKET, cohort_mappings_key)[0]
has_any_mappings = len(mapping_files) > 0

has_global_freq = check_s3_object_exists(S3_BUCKET, global_freq_key)[0]
has_cohort_freq = check_s3_object_exists(S3_BUCKET, cohort_freq_key)[0]
has_any_freq = len(freq_files) > 0

if has_global_mappings or has_cohort_mappings or has_any_mappings:
    print("[OK] Drug-gene mappings found on S3")
else:
    print("[NOT FOUND] Drug-gene mappings NOT found on S3")

if has_global_freq or has_cohort_freq or has_any_freq:
    print("[OK] Allele frequencies found on S3")
else:
    print("[NOT FOUND] Allele frequencies NOT found on S3")

