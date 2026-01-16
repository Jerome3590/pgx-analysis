#!/usr/bin/env python3
"""
Sync Step 3b files from S3 to local storage.

Checks S3 for:
- cohort_feature_importance.csv files
- BupaR analysis results
- DTW analysis results
- Feature filtering summaries
- Visualizations (plots)
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import subprocess

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import age_band_to_fname

try:
    from py_helpers.common_imports import s3_client, S3_BUCKET
except ImportError:
    import boto3
    s3_client = boto3.client("s3")
    S3_BUCKET = "pgxdatalake"

# Cohorts and age bands
COHORTS = {
    'opioid_ed': ['13-24', '25-44', '45-54', '55-64'],
    'non_opioid_ed': ['65-74', '75-84', '85-94']
}


def check_s3_file_exists(s3_path: str) -> bool:
    """Check if a file exists in S3."""
    try:
        if s3_path.startswith("s3://"):
            s3_path = s3_path[5:]
        parts = s3_path.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except s3_client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ["404", "NoSuchKey"]:
            return False
        raise
    except Exception:
        return False


def download_from_s3(s3_path: str, local_path: Path, profile: Optional[str] = None) -> bool:
    """Download file from S3 using AWS CLI."""
    try:
        cmd = ['aws', 's3', 'cp', s3_path, str(local_path)]
        if profile:
            cmd.extend(['--profile', profile])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True
        else:
            print(f"  [ERROR] Download failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  [ERROR] Failed to download {s3_path}: {e}")
        return False


def get_expected_s3_files(cohort: str, age_band: str) -> List[tuple[str, Path]]:
    """Get list of expected S3 files and their local paths."""
    age_band_fname = age_band_to_fname(age_band)
    base_s3_path = f"s3://{S3_BUCKET}/gold/feature_importance/{cohort}/{age_band}"
    base_local_dir = PROJECT_ROOT / "3b_feature_importance_eda" / "outputs" / cohort / age_band_fname
    
    files = [
        # Main outputs
        (f"{base_s3_path}/{cohort}_{age_band_fname}_cohort_feature_importance.csv",
         base_local_dir / f"{cohort}_{age_band_fname}_cohort_feature_importance.csv"),
        
        # Analysis results
        (f"{base_s3_path}/{cohort}_{age_band_fname}_bupar_post_target_analysis.csv",
         base_local_dir / f"{cohort}_{age_band_fname}_bupar_post_target_analysis.csv"),
        
        (f"{base_s3_path}/{cohort}_{age_band_fname}_dtw_trajectory_analysis.csv",
         base_local_dir / f"{cohort}_{age_band_fname}_dtw_trajectory_analysis.csv"),
        
        # Summary
        (f"{base_s3_path}/{cohort}_{age_band_fname}_feature_filtering_summary.json",
         base_local_dir / f"{cohort}_{age_band_fname}_feature_filtering_summary.json"),
    ]
    
    # Visualizations
    plots_local_dir = base_local_dir / "plots"
    visualization_files = [
        f"{cohort}_{age_band_fname}_overall_activity_frequency.png",
        f"{cohort}_{age_band_fname}_activity_milestones_gantt.png",
        f"{cohort}_{age_band_fname}_activity_sequence_top.png",
        f"{cohort}_{age_band_fname}_gantt_icd.png",
        f"{cohort}_{age_band_fname}_pre_f1120_activity_frequency.png",
        f"{cohort}_{age_band_fname}_pre_f1120_gantt.png",
        f"{cohort}_{age_band_fname}_post_f1120_activity_frequency.png",
        f"{cohort}_{age_band_fname}_post_f1120_gantt.png",
        f"{cohort}_{age_band_fname}_post_f1120_gantt_icd.png",
    ]
    
    for viz_file in visualization_files:
        files.append((
            f"{base_s3_path}/plots/{viz_file}",
            plots_local_dir / viz_file
        ))
    
    return files


def sync_cohort_age_band(cohort: str, age_band: str, dry_run: bool = False, profile: Optional[str] = None) -> dict:
    """Sync files for a single cohort/age_band."""
    age_band_fname = age_band_to_fname(age_band)
    
    print(f"\n{'='*80}")
    print(f"Checking: {cohort} / {age_band}")
    print(f"{'='*80}")
    
    expected_files = get_expected_s3_files(cohort, age_band)
    
    stats = {
        'total': len(expected_files),
        'exists_s3': 0,
        'exists_local': 0,
        'missing_local': 0,
        'downloaded': 0,
        'skipped': 0
    }
    
    for s3_path, local_path in expected_files:
        exists_s3 = check_s3_file_exists(s3_path)
        exists_local = local_path.exists()
        
        stats['exists_s3'] += exists_s3
        stats['exists_local'] += exists_local
        
        if exists_s3 and not exists_local:
            stats['missing_local'] += 1
            print(f"  [MISSING] {local_path.name}")
            if not dry_run:
                # Create parent directory
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download file
                if download_from_s3(s3_path, local_path, profile):
                    stats['downloaded'] += 1
                    print(f"    [OK] Downloaded")
                else:
                    print(f"    [ERROR] Download failed")
        elif exists_local:
            stats['skipped'] += 1
            if exists_s3:
                print(f"  [OK] {local_path.name} (exists locally and in S3)")
            else:
                print(f"  [WARN] {local_path.name} (exists locally but not in S3)")
        elif not exists_s3:
            print(f"  [SKIP] {local_path.name} (not in S3)")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Sync Step 3b files from S3 to local storage"
    )
    parser.add_argument(
        "--cohort",
        type=str,
        choices=list(COHORTS.keys()),
        help="Sync specific cohort (default: all)"
    )
    parser.add_argument(
        "--age-band",
        type=str,
        help="Sync specific age band (requires --cohort)"
    )
    parser.add_argument(
        "--all-cohorts",
        action="store_true",
        help="Sync all cohorts and age bands"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading"
    )
    parser.add_argument(
        "--profile",
        type=str,
        help="AWS profile to use (optional)"
    )
    
    args = parser.parse_args()
    
    # Determine cohorts to process
    if args.all_cohorts or (not args.cohort and not args.age_band):
        cohorts_to_process = []
        for cohort, age_bands in COHORTS.items():
            for age_band in age_bands:
                cohorts_to_process.append((cohort, age_band))
    elif args.cohort and args.age_band:
        cohorts_to_process = [(args.cohort, args.age_band)]
    elif args.cohort:
        cohorts_to_process = [
            (args.cohort, age_band) 
            for age_band in COHORTS[args.cohort]
        ]
    else:
        print("[ERROR] Must specify --cohort and --age-band, or --all-cohorts")
        sys.exit(1)
    
    print("=" * 80)
    print("Step 3b S3 Sync")
    print("=" * 80)
    if args.dry_run:
        print("DRY RUN MODE - No files will be downloaded")
    print(f"Processing {len(cohorts_to_process)} cohort/age_band combinations")
    print()
    
    total_stats = {
        'total': 0,
        'exists_s3': 0,
        'exists_local': 0,
        'missing_local': 0,
        'downloaded': 0,
        'skipped': 0
    }
    
    for cohort, age_band in cohorts_to_process:
        stats = sync_cohort_age_band(cohort, age_band, dry_run=args.dry_run, profile=args.profile)
        
        # Aggregate stats
        for key in total_stats:
            total_stats[key] += stats[key]
    
    # Summary
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total files checked: {total_stats['total']}")
    print(f"Files in S3: {total_stats['exists_s3']}")
    print(f"Files already local: {total_stats['exists_local']}")
    print(f"Files missing locally: {total_stats['missing_local']}")
    if not args.dry_run:
        print(f"Files downloaded: {total_stats['downloaded']}")
    print(f"Files skipped: {total_stats['skipped']}")
    print()


if __name__ == "__main__":
    main()
