"""
Download only aggregated feature importance files from S3 to 3_feature_importance/outputs.

This script downloads:
- s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/{cohort}_{age_band}_aggregated_feature_importance.csv
- To: 3_feature_importance/outputs/{cohort}/{age_band}/{cohort}_{age_band}_aggregated_feature_importance.csv
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import boto3

from py_helpers.constants import S3_BUCKET, COHORT_NAMES, AGE_BANDS


def download_aggregated_fi():
    """Download aggregated feature importance files only."""
    s3_client = boto3.client("s3")
    bucket = S3_BUCKET
    
    output_root = Path("3_feature_importance") / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    skipped = 0
    
    for cohort in COHORT_NAMES:
        for age_band in AGE_BANDS:
            age_band_fname = age_band.replace("-", "_")
            
            # S3 key
            s3_key = (
                f"gold/feature_importance/{cohort}/{age_band}/"
                f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
            )
            
            # Local destination
            dest_dir = output_root / cohort / age_band
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
            
            # Check if file exists in S3
            try:
                s3_client.head_object(Bucket=bucket, Key=s3_key)
            except s3_client.exceptions.ClientError as e:
                code = e.response.get("Error", {}).get("Code")
                if code in ("404", "403", "NoSuchKey"):
                    print(f"  [SKIP] Not found in S3: s3://{bucket}/{s3_key}")
                    skipped += 1
                    continue
                raise
            
            # Download if local file doesn't exist or is older
            if dest_file.exists():
                print(f"  [SKIP] Already exists locally: {dest_file}")
                skipped += 1
            else:
                print(f"  [DOWNLOAD] s3://{bucket}/{s3_key} -> {dest_file}")
                s3_client.download_file(bucket, s3_key, str(dest_file))
                downloaded += 1
    
    print(f"\n{'='*80}")
    print(f"Download complete: {downloaded} files downloaded, {skipped} skipped/not found")
    print(f"{'='*80}")


if __name__ == "__main__":
    download_aggregated_fi()
