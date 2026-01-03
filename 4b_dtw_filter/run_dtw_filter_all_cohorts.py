#!/usr/bin/env python3
"""
Run DTW protocol filter for all cohorts and age bands.

This script runs the DTW filter (Step 4b) for all cohort/age band combinations
to prepare data for research and downstream feature engineering.
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import COHORT_NAMES, AGE_BANDS

def run_dtw_filter(
    cohort_name: str,
    age_band: str,
    min_interval_days: int = 1,
    keep_first_event: bool = True,
    protocol_threshold_pct: float = 0.5,
) -> Tuple[bool, str]:
    """
    Run DTW filter for a single cohort/age band combination.
    
    Returns:
        (success: bool, message: str)
    """
    script_path = PROJECT_ROOT / "4b_dtw_filter" / "filter_protocol_events.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--cohort-name", cohort_name,
        "--age-band", age_band,
        "--min-interval-days", str(min_interval_days),
    ]
    
    if keep_first_event:
        cmd.append("--keep-first-event")
    
    cmd.extend([
        "--protocol-threshold-pct", str(protocol_threshold_pct)
    ])
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        return True, f"Success: {cohort_name} / {age_band}"
    except subprocess.CalledProcessError as e:
        return False, f"Error: {cohort_name} / {age_band} - {e.stderr[:200]}"


def main():
    """Run DTW filter for all cohorts."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run DTW protocol filter for all cohorts and age bands"
    )
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=COHORT_NAMES,
        help=f"Cohort names to process (default: {COHORT_NAMES})"
    )
    parser.add_argument(
        "--age-bands",
        nargs="+",
        default=AGE_BANDS,
        help=f"Age bands to process (default: {AGE_BANDS})"
    )
    parser.add_argument(
        "--min-interval-days",
        type=int,
        default=1,
        help="Minimum interval (days) for time window research (default: 1, used for research analysis only)"
    )
    parser.add_argument(
        "--protocol-threshold-pct",
        type=float,
        default=0.5,
        help="Keep all events if patient has > this % protocol events (default: 0.5)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cohorts that already have filtered data"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DTW Protocol Filter - Batch Run for All Cohorts")
    print("=" * 80)
    print(f"Cohorts: {args.cohorts}")
    print(f"Age Bands: {args.age_bands}")
    print(f"Min Interval Days: {args.min_interval_days}")
    print(f"Protocol Threshold %: {args.protocol_threshold_pct}")
    print("=" * 80)
    print()
    
    total = len(args.cohorts) * len(args.age_bands)
    completed = 0
    successful = 0
    failed = 0
    skipped = 0
    
    for cohort_name in args.cohorts:
        for age_band in args.age_bands:
            completed += 1
            
            # Check if filtered data already exists
            if args.skip_existing:
                model_data_dir = (
                    PROJECT_ROOT
                    / "4a_model_data"
                    / f"cohort_name={cohort_name}"
                    / f"age_band={age_band}"
                )
                filtered_path = model_data_dir / "model_events_no_protocols.parquet"
                if filtered_path.exists():
                    print(f"[{completed}/{total}] SKIP (exists): {cohort_name} / {age_band}")
                    skipped += 1
                    continue
            
            print(f"[{completed}/{total}] Processing: {cohort_name} / {age_band}...", end=" ")
            
            success, message = run_dtw_filter(
                cohort_name=cohort_name,
                age_band=age_band,
                min_interval_days=args.min_interval_days,
                protocol_threshold_pct=args.protocol_threshold_pct,
            )
            
            if success:
                print(f"✓ {message}")
                successful += 1
            else:
                print(f"✗ {message}")
                failed += 1
    
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total: {total}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"⊘ Skipped: {skipped}")
    print("=" * 80)


if __name__ == "__main__":
    main()
