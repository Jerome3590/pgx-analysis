#!/usr/bin/env python3
"""
Check if PGx prerequisite files (drug-gene mappings and allele frequencies) exist on S3,
and optionally download them if they're missing locally.

Usage:
    python utility_scripts/check_download_pgx_prerequisites.py --cohort opioid_ed --age-band 13-24 [--download]
"""

import argparse
import sys
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

PROJECT_ROOT = Path(__file__).parent.parent
S3_BUCKET = "pgxdatalake"


def check_s3_object_exists(bucket: str, key: str) -> bool:
    """Check if an S3 object exists."""
    s3_client = boto3.client("s3")
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def download_from_s3(bucket: str, key: str, local_path: Path) -> bool:
    """Download a file from S3."""
    s3_client = boto3.client("s3")
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket, key, str(local_path))
        print(f"✓ Downloaded: {local_path}")
        return True
    except ClientError as e:
        print(f"✗ Failed to download {key}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Check and optionally download PGx prerequisite files from S3"
    )
    parser.add_argument("--cohort", required=True, help="Cohort name (e.g., opioid_ed)")
    parser.add_argument("--age-band", required=True, help="Age band (e.g., 13-24)")
    parser.add_argument(
        "--download", action="store_true", help="Download missing files from S3"
    )

    args = parser.parse_args()
    cohort_name = args.cohort
    age_band = args.age_band
    age_band_fname = age_band.replace("-", "_")

    print(f"\nChecking PGx prerequisites for {cohort_name} / {age_band}...")
    print("=" * 70)

    # Define S3 paths
    # Note: These files are GLOBAL (or cohort-level), not age-band specific
    # Check global cache paths first (preferred)
    s3_global_mappings_key = "gold/pgx_features/global/pgx_drug_gene_mappings_global.csv"
    s3_global_freq_key = "gold/pgx_features/global/pgx_allele_frequencies_global.csv"
    
    # Also check cohort-level paths (fallback)
    s3_cohort_mappings_key = (
        f"gold/pgx_features/{cohort_name}/{cohort_name}_drug_gene_mappings.csv"
    )
    s3_cohort_freq_key = (
        f"gold/pgx_features/{cohort_name}/{cohort_name}_allele_frequencies.csv"
    )
    
    # Legacy age-band specific paths (for backward compatibility)
    s3_legacy_mappings_key = (
        f"gold/pgx_features/{cohort_name}/{age_band}/{cohort_name}_drug_gene_mappings.csv"
    )
    s3_legacy_freq_key = (
        f"gold/pgx_features/{cohort_name}/{age_band}/{cohort_name}_allele_frequencies.csv"
    )

    # Define local paths
    cohort_out_dir = PROJECT_ROOT / "5_pgx_analysis" / "outputs" / cohort_name
    mappings_path = cohort_out_dir / f"{cohort_name}_drug_gene_mappings.csv"
    freq_path = cohort_out_dir / f"{cohort_name}_allele_frequencies.csv"

    global_out_dir = PROJECT_ROOT / "5_pgx_analysis" / "outputs" / "global"
    global_mappings_path = global_out_dir / "pgx_drug_gene_mappings_global.csv"
    global_freq_path = global_out_dir / "pgx_allele_frequencies_global.csv"

    # Check drug-gene mappings (prefer global, then cohort-level, then legacy)
    print("\n1. Drug-Gene Mappings:")
    
    # Check global cache first (preferred)
    print(f"   Global cache: {global_mappings_path}")
    global_local_exists = global_mappings_path.exists()
    global_s3_exists = check_s3_object_exists(S3_BUCKET, s3_global_mappings_key)

    if global_local_exists:
        print(f"   ✓ Global cache exists locally ({global_mappings_path.stat().st_size} bytes)")
    else:
        print("   ✗ Global cache missing locally")

    if global_s3_exists:
        print(f"   ✓ Global cache exists on S3: s3://{S3_BUCKET}/{s3_global_mappings_key}")
        if args.download and not global_local_exists:
            download_from_s3(S3_BUCKET, s3_global_mappings_key, global_mappings_path)
    else:
        print(f"   ✗ Global cache missing on S3: s3://{S3_BUCKET}/{s3_global_mappings_key}")
    
    # Check cohort-level (fallback)
    print(f"\n   Cohort-level: {mappings_path}")
    local_exists = mappings_path.exists()
    cohort_s3_exists = check_s3_object_exists(S3_BUCKET, s3_cohort_mappings_key)

    if local_exists:
        print(f"   ✓ Cohort-level file exists locally ({mappings_path.stat().st_size} bytes)")
    else:
        print("   ✗ Cohort-level file missing locally")

    if cohort_s3_exists:
        print(f"   ✓ Cohort-level file exists on S3: s3://{S3_BUCKET}/{s3_cohort_mappings_key}")
        if args.download and not local_exists and not global_local_exists:
            download_from_s3(S3_BUCKET, s3_cohort_mappings_key, mappings_path)
    else:
        print(f"   ✗ Cohort-level file missing on S3: s3://{S3_BUCKET}/{s3_cohort_mappings_key}")
    
    # Check legacy age-band path (for reference)
    legacy_s3_exists = check_s3_object_exists(S3_BUCKET, s3_legacy_mappings_key)
    if legacy_s3_exists:
        print(f"\n   Note: Legacy age-band file found on S3: s3://{S3_BUCKET}/{s3_legacy_mappings_key}")

    # Check allele frequencies (prefer global, then cohort-level, then legacy)
    print("\n2. Allele Frequencies:")
    
    # Check global cache first (preferred)
    print(f"   Global cache: {global_freq_path}")
    global_freq_local_exists = global_freq_path.exists()
    global_freq_s3_exists = check_s3_object_exists(S3_BUCKET, s3_global_freq_key)

    if global_freq_local_exists:
        print(f"   ✓ Global cache exists locally ({global_freq_path.stat().st_size} bytes)")
    else:
        print("   ✗ Global cache missing locally")

    if global_freq_s3_exists:
        print(f"   ✓ Global cache exists on S3: s3://{S3_BUCKET}/{s3_global_freq_key}")
        if args.download and not global_freq_local_exists:
            download_from_s3(S3_BUCKET, s3_global_freq_key, global_freq_path)
    else:
        print(f"   ✗ Global cache missing on S3: s3://{S3_BUCKET}/{s3_global_freq_key}")
    
    # Check cohort-level (fallback)
    print(f"\n   Cohort-level: {freq_path}")
    local_freq_exists = freq_path.exists()
    cohort_freq_s3_exists = check_s3_object_exists(S3_BUCKET, s3_cohort_freq_key)

    if local_freq_exists:
        print(f"   ✓ Cohort-level file exists locally ({freq_path.stat().st_size} bytes)")
    else:
        print("   ✗ Cohort-level file missing locally")

    if cohort_freq_s3_exists:
        print(f"   ✓ Cohort-level file exists on S3: s3://{S3_BUCKET}/{s3_cohort_freq_key}")
        if args.download and not local_freq_exists and not global_freq_local_exists:
            download_from_s3(S3_BUCKET, s3_cohort_freq_key, freq_path)
    else:
        print(f"   ✗ Cohort-level file missing on S3: s3://{S3_BUCKET}/{s3_cohort_freq_key}")
    
    # Check legacy age-band path (for reference)
    legacy_freq_s3_exists = check_s3_object_exists(S3_BUCKET, s3_legacy_freq_key)
    if legacy_freq_s3_exists:
        print(f"\n   Note: Legacy age-band file found on S3: s3://{S3_BUCKET}/{s3_legacy_freq_key}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    has_mappings = (
        global_local_exists
        or local_exists
        or global_s3_exists
        or cohort_s3_exists
        or legacy_s3_exists
    )
    has_frequencies = (
        global_freq_local_exists
        or local_freq_exists
        or global_freq_s3_exists
        or cohort_freq_s3_exists
        or legacy_freq_s3_exists
    )

    if has_mappings:
        print("✓ Drug-gene mappings available")
    else:
        print("✗ Drug-gene mappings NOT found (need to run map_drugs_to_genes.py)")

    if has_frequencies:
        print("✓ Allele frequencies available")
    else:
        print("✗ Allele frequencies NOT found (need to run add_allele_frequencies.py or build_global_pgx_cache.py)")

    if has_mappings and has_frequencies:
        print("\n✓ All prerequisites available - PGx analysis can proceed")
        return 0
    else:
        print("\n✗ Missing prerequisites - PGx analysis will create empty features")
        if not args.download:
            print("\nTip: Run with --download to automatically fetch files from S3")
        return 1


if __name__ == "__main__":
    sys.exit(main())

