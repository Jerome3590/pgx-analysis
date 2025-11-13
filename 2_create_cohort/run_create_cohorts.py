#!/usr/bin/env python3
"""
Orchestration script to run create_cohort.py across age bands/year combos.
Improved: cross-platform live output capture using threads, and robust S3 head_object checks.
"""

import sys
import subprocess
import concurrent.futures
import boto3
import traceback
import os
import threading
import queue
import time

from helpers_1997_13 import constants
from helpers_1997_13.cohort_utils import check_existing_cohorts as cu_check_existing_cohorts, run_cohort as cu_run_cohort
import functools

# Script configuration
# Adjust this path to where 0_create_cohort.py lives on the target host
script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '0_create_cohort.py'))
python_bin = sys.executable


# Use shared helpers from helpers_1997_13.cohort_utils to avoid duplication.
def check_existing_cohorts():
    return cu_check_existing_cohorts()


# Wrapper that binds script_path and python_bin into the shared runner.
def run_cohort(job):
    target_icd = os.environ.get('PGX_TARGET_ICD_CODES', 'F1120')
    cb = functools.partial(cu_run_cohort, script_path=script_path, python_bin=python_bin, target_icd=target_icd)
    return cb(job)


# ----- Batch processing orchestration -----

if __name__ == '__main__':
    jobs_to_process = check_existing_cohorts()

    if not jobs_to_process:
        print("\nAll cohorts already exist or are locked. No jobs to run.")
        sys.exit(0)

    MAX_WORKERS = min(2, len(jobs_to_process))
    print(f"\nStarting {len(jobs_to_process)} cohort processing jobs with {MAX_WORKERS} parallel workers...", flush=True)
    print(f"{'='*80}", flush=True)

    BATCH_SIZE = 2
    all_job_batches = [jobs_to_process[i:i+BATCH_SIZE] for i in range(0, len(jobs_to_process), BATCH_SIZE)]
    all_results = []
    all_job_statuses = {}

    for batch_num, job_batch in enumerate(all_job_batches, 1):
        print(f"\nProcessing batch {batch_num}/{len(all_job_batches)} with {len(job_batch)} jobs...", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_job = {executor.submit(run_cohort, job): job for job in job_batch}
            total_jobs = len(future_to_job)
            completed = 0
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                job_id = f"{job['age_band']}/{job['event_year']}"
                completed += 1
                try:
                    result = future.result()
                    all_results.append(result)
                    all_job_statuses[job_id] = result
                    print(f"\n[{completed}/{total_jobs}] Job status: {result}", flush=True)
                except Exception as e:
                    error_msg = f"Job execution error for {job_id}: {str(e)}"
                    print(f"\n[{completed}/{total_jobs}] {error_msg}", flush=True)
                    print(traceback.format_exc(), flush=True)
                    all_results.append(f"ERROR: {job_id} - {str(e)}")
                    all_job_statuses[job_id] = f"ERROR: {str(e)}"

                print(f"\nProgress: {completed}/{total_jobs} jobs completed ({100*completed/total_jobs:.1f}%)", flush=True)
                print(f"{'='*80}", flush=True)

        print(f"\nBatch {batch_num}/{len(all_job_batches)} complete.", flush=True)
        if batch_num < len(all_job_batches):
            print("Pausing briefly before starting next batch...", flush=True)
            time.sleep(5)

    print("\n" + "="*80)
    print("FINAL SUMMARY OF RESULTS:")
    print("="*80)
    success_count = sum(1 for r in all_results if r.startswith("SUCCESS"))
    locked_count = sum(1 for r in all_results if r.startswith("SKIPPED_LOCKED"))
    failed_count = sum(1 for r in all_results if r.startswith("FAILED") or r.startswith("ERROR"))

    print(f"✓ Successful: {success_count}")
    print(f"⚠ Skipped (locked): {locked_count}")
    print(f"✗ Failed: {failed_count}")

    if all_job_statuses:
        print("\nDetailed status by job:")
        for job_id, status in sorted(all_job_statuses.items()):
            if status.startswith("SUCCESS"):
                status_icon = "✓"
            elif status.startswith("SKIPPED"):
                status_icon = "⚠"
            else:
                status_icon = "✗"
            print(f"{status_icon} {job_id}: {status}")

    print("="*80)
