#!/usr/bin/env python3
"""
Download latest logs from S3 build_logs directory.

This script downloads logs from s3://pgx-repository/build_logs/ for a given
pipeline phase, cohort, age band, and event year (or all if not specified).

Usage:
    python utility_scripts/download_logs_from_s3.py --phase 4a_model_data --cohort opioid_ed --age-band 0-12
    python utility_scripts/download_logs_from_s3.py --phase 4b_event_filter --cohort opioid_ed --age-band 0-12 --year 2016
    python utility_scripts/download_logs_from_s3.py --phase 4a_model_data  # Downloads all logs for this phase
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
from datetime import datetime

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("Error: boto3 is required. Install with: pip install boto3")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

S3_BUCKET = "pgx-repository"
S3_PREFIX = "build_logs"


def list_logs_in_s3(
    phase: Optional[str] = None,
    cohort: Optional[str] = None,
    age_band: Optional[str] = None,
    year: Optional[str] = None,
) -> List[dict]:
    """List log files in S3 matching the given criteria."""
    s3_client = boto3.client("s3")
    logs = []
    
    # Build prefix path
    prefix_parts = [S3_PREFIX]
    if phase:
        prefix_parts.append(phase)
    if cohort:
        prefix_parts.append(cohort)
    if age_band:
        prefix_parts.append(age_band)
    if year:
        prefix_parts.append(year)
    
    prefix = "/".join(prefix_parts) + "/"
    
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".txt") or key.endswith(".log"):
                    logs.append({
                        "key": key,
                        "size": obj["Size"],
                        "last_modified": obj["LastModified"],
                    })
    except ClientError as e:
        print(f"Error listing logs: {e}")
        return []
    
    # Sort by last modified (newest first)
    logs.sort(key=lambda x: x["last_modified"], reverse=True)
    return logs


def download_logs(
    phase: Optional[str] = None,
    cohort: Optional[str] = None,
    age_band: Optional[str] = None,
    year: Optional[str] = None,
    output_dir: Optional[Path] = None,
    max_logs: int = 10,
) -> int:
    """Download latest logs from S3."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "logs" / "from_s3"
    else:
        output_dir = Path(output_dir)
    
    # List logs matching criteria
    logs = list_logs_in_s3(phase, cohort, age_band, year)
    
    if not logs:
        print(f"No logs found matching criteria:")
        if phase:
            print(f"  Phase: {phase}")
        if cohort:
            print(f"  Cohort: {cohort}")
        if age_band:
            print(f"  Age band: {age_band}")
        if year:
            print(f"  Year: {year}")
        return 0
    
    print(f"Found {len(logs)} log files. Downloading up to {max_logs} most recent...")
    
    # Download up to max_logs most recent files
    s3_client = boto3.client("s3")
    downloaded = 0
    
    for log_info in logs[:max_logs]:
        s3_key = log_info["key"]
        # Build local path preserving S3 structure
        relative_path = s3_key[len(S3_PREFIX) + 1:]  # Remove "build_logs/"
        local_path = output_dir / relative_path
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"Downloading: s3://{S3_BUCKET}/{s3_key}")
            print(f"  -> {local_path}")
            s3_client.download_file(S3_BUCKET, s3_key, str(local_path))
            print(f"  [OK] Downloaded ({log_info['size']:,} bytes, modified: {log_info['last_modified']})")
            downloaded += 1
        except Exception as e:
            print(f"  [ERROR] Error downloading {s3_key}: {e}")
    
    print(f"\nDownloaded {downloaded} log file(s) to {output_dir}")
    return downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Download latest logs from S3 build_logs directory"
    )
    parser.add_argument(
        "--phase",
        type=str,
        help="Pipeline phase (e.g., 4a_model_data, 4b_event_filter)",
    )
    parser.add_argument(
        "--cohort",
        type=str,
        help="Cohort name (e.g., opioid_ed, non_opioid_ed)",
    )
    parser.add_argument(
        "--age-band",
        type=str,
        help="Age band (e.g., 0-12, 13-24)",
    )
    parser.add_argument(
        "--year",
        type=str,
        help="Event year (e.g., 2016, 2017)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: PROJECT_ROOT/logs/from_s3)",
    )
    parser.add_argument(
        "--max-logs",
        type=int,
        default=10,
        help="Maximum number of logs to download (default: 10)",
    )
    
    args = parser.parse_args()
    
    # Convert age_band format if needed (e.g., 0_12 -> 0-12)
    age_band = args.age_band
    if age_band and "_" in age_band:
        age_band = age_band.replace("_", "-")
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    downloaded = download_logs(
        phase=args.phase,
        cohort=args.cohort,
        age_band=age_band,
        year=args.year,
        output_dir=output_dir,
        max_logs=args.max_logs,
    )
    
    sys.exit(0 if downloaded > 0 else 1)


if __name__ == "__main__":
    main()
