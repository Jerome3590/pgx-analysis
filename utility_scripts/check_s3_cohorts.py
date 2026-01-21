#!/usr/bin/env python3
"""
Check S3 gold/cohorts folder for completed cohorts.

Lists all cohort parquet files in s3://pgxdatalake/gold/cohorts/
"""

import sys
from pathlib import Path
from collections import defaultdict

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("Error: boto3 is required. Install with: pip install boto3")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

S3_BUCKET = "pgxdatalake"
S3_PREFIX = "gold/cohorts/"


def list_cohorts_in_s3() -> dict:
    """List all cohort parquet files in S3."""
    s3_client = boto3.client("s3")
    cohorts = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX, Delimiter="/"):
            # Process common prefixes (directories)
            for prefix_info in page.get("CommonPrefixes", []):
                prefix = prefix_info["Prefix"]
                # Extract cohort_name from prefix like "gold/cohorts/cohort_name=opioid_ed/"
                if "cohort_name=" in prefix:
                    cohort_name = prefix.split("cohort_name=")[1].rstrip("/")
                    
                    # List event_year subdirectories
                    for year_page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix, Delimiter="/"):
                        for year_prefix_info in year_page.get("CommonPrefixes", []):
                            year_prefix = year_prefix_info["Prefix"]
                            # Extract event_year from prefix like "gold/cohorts/cohort_name=opioid_ed/event_year=2019/"
                            if "event_year=" in year_prefix:
                                event_year = year_prefix.split("event_year=")[1].rstrip("/")
                                
                                # List age_band subdirectories
                                for age_page in paginator.paginate(Bucket=S3_BUCKET, Prefix=year_prefix, Delimiter="/"):
                                    for age_prefix_info in age_page.get("CommonPrefixes", []):
                                        age_prefix = age_prefix_info["Prefix"]
                                        # Extract age_band from prefix like ".../age_band=25-44/"
                                        if "age_band=" in age_prefix:
                                            age_band = age_prefix.split("age_band=")[1].rstrip("/")
                                            
                                            # Check for cohort.parquet file
                                            for file_page in paginator.paginate(Bucket=S3_BUCKET, Prefix=age_prefix):
                                                for obj in file_page.get("Contents", []):
                                                    key = obj["Key"]
                                                    if key.endswith("cohort.parquet"):
                                                        size = obj["Size"]
                                                        last_modified = obj["LastModified"]
                                                        cohorts[cohort_name][event_year][age_band].append({
                                                            "key": key,
                                                            "size": size,
                                                            "last_modified": last_modified,
                                                        })
    except ClientError as e:
        print(f"Error listing cohorts: {e}")
        return {}
    
    return cohorts


def main():
    print("=" * 80)
    print("Checking S3 for completed cohorts")
    print("=" * 80)
    print(f"Bucket: {S3_BUCKET}")
    print(f"Prefix: {S3_PREFIX}")
    print()
    
    cohorts = list_cohorts_in_s3()
    
    if not cohorts:
        print("No cohorts found in S3.")
        return
    
    total_cohorts = 0
    for cohort_name in sorted(cohorts.keys()):
        print(f"\n{'=' * 80}")
        print(f"Cohort: {cohort_name}")
        print(f"{'=' * 80}")
        
        for event_year in sorted(cohorts[cohort_name].keys()):
            print(f"\n  Event Year: {event_year}")
            for age_band in sorted(cohorts[cohort_name][event_year].keys()):
                files = cohorts[cohort_name][event_year][age_band]
                if files:
                    file_info = files[0]
                    size_mb = file_info["size"] / (1024 * 1024)
                    last_modified = file_info["last_modified"]
                    print(f"    {age_band:10s} - {size_mb:8.2f} MB - {last_modified}")
                    total_cohorts += 1
    
    print(f"\n{'=' * 80}")
    print(f"Total completed cohorts: {total_cohorts}")
    print(f"{'=' * 80}")
    
    # Summary by cohort
    print("\nSummary by cohort:")
    for cohort_name in sorted(cohorts.keys()):
        count = sum(len(age_bands) for age_bands in cohorts[cohort_name].values())
        print(f"  {cohort_name}: {count} age_band/year combinations")


if __name__ == "__main__":
    main()
