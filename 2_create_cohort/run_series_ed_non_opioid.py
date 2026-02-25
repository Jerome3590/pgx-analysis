#!/usr/bin/env python3
"""
Wrapper script to run cohort creation for ed_non_opioid (polypharmacy) cohort
across all age bands and event years, processing heavy partitions first.

This script:
1. Creates cohort parquets (all age_band × event_year) via 0_create_cohort.py
2. Runs the event filter (Step 1b) for each age_band, producing model_events_no_protocols.parquet

Event filter requires baseline aggregated feature importance (Step 3a) for each age_band.
Run Step 3a with --baseline first if needed.

Usage:
    python run_series_ed_non_opioid.py [--skip-existing] [--no-event-filter]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from py_helpers.cohort_utils import check_existing_cohorts
from py_helpers.env_utils import get_workflow_python_bin

# Recommended order: heavy partitions first
AGE_BANDS_ORDERED = [
    "25-44",  # Heaviest
    "65-74",  # Second heaviest
    "45-54",
    "55-64",
    "75-84",
    "85-114",  # Combined former 85-94 and 95-114
    "13-24",
    "0-12",
]

EVENT_YEARS = [2016, 2017, 2018, 2019]


def main():
    parser = argparse.ArgumentParser(
        description="Run ed_non_opioid cohort creation for all partitions (heavy first)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip partitions that already exist in S3"
    )
    parser.add_argument(
        "--concurrent-workers",
        type=int,
        default=1,
        help="Number of concurrent workers (for memory limit calculation, default: 1)"
    )
    parser.add_argument(
        "--python-bin",
        default=None,
        help="Python executable path (default: EC2 jupyter-env or current interpreter)"
    )
    args = parser.parse_args()
    if args.python_bin is None:
        args.python_bin = str(get_workflow_python_bin())

    script_path = Path(__file__).parent / "0_create_cohort.py"
    if not script_path.exists():
        print(f"❌ Error: {script_path} not found")
        sys.exit(1)

    # Get list of jobs that need processing
    if args.skip_existing:
        print("🔍 Checking for existing cohorts in S3...")
        jobs_to_process = check_existing_cohorts(
            age_bands=AGE_BANDS_ORDERED,
            event_years=EVENT_YEARS
        )
        # Filter to only ed_non_opioid (check_existing_cohorts returns jobs for both cohorts)
        # We'll process all jobs since 0_create_cohort.py with --cohort ed_non_opioid will handle it
        print(f"✓ Found {len(jobs_to_process)} partitions that need processing")
    else:
        # Process all partitions
        jobs_to_process = []
        for band in AGE_BANDS_ORDERED:
            for year in EVENT_YEARS:
                jobs_to_process.append({"age_band": band, "event_year": year})
        print(f"📋 Will process {len(jobs_to_process)} partitions (all partitions)")

    if not jobs_to_process:
        print("✅ All partitions already exist. Nothing to do.")
        return

    # Process in order (heavy first)
    print(f"\n🚀 Starting ed_non_opioid cohort creation (heavy partitions first)")
    print(f"   Processing {len(jobs_to_process)} partitions sequentially")
    print(f"   Order: {AGE_BANDS_ORDERED}")
    print(f"   Years: {EVENT_YEARS}\n")

    success_count = 0
    failed_count = 0
    skipped_count = 0

    for i, job in enumerate(jobs_to_process, 1):
        age_band = job["age_band"]
        event_year = job["event_year"]
        job_id = f"{age_band}/{event_year}"

        print(f"\n{'='*80}")
        print(f"[{i}/{len(jobs_to_process)}] Processing ed_non_opioid: {job_id}")
        print(f"{'='*80}")

        cmd = [
            args.python_bin,
            str(script_path),
            "--cohort", "ed_non_opioid",
            "--age-band", age_band,
            "--event-year", str(event_year),
            "--starting-step", "phase1_data_preparation",
            "--operation-type", "concurrent_processing",
            "--log-level", "INFO",
            "--concurrent-workers", str(args.concurrent_workers),
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            if result.returncode == 0:
                success_count += 1
                print(f"✅ Successfully processed {job_id}")
            else:
                failed_count += 1
                print(f"❌ Failed to process {job_id} (return code: {result.returncode})")
        except subprocess.CalledProcessError as e:
            failed_count += 1
            print(f"❌ Error processing {job_id}: {e}")
        except KeyboardInterrupt:
            print(f"\n⚠️ Interrupted by user. Processed {i-1}/{len(jobs_to_process)} partitions.")
            print(f"   Success: {success_count}, Failed: {failed_count}, Skipped: {skipped_count}")
            sys.exit(1)

    # Summary
    print(f"\n{'='*80}")
    print(f"📊 COHORT CREATION SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"⏭️  Skipped: {skipped_count}")
    print(f"📋 Total: {len(jobs_to_process)}")
    print(f"{'='*80}")

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
