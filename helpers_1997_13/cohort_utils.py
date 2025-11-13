"""
Cohort processing utilities.
"""

import sys
import os
import re
import time
import subprocess
import boto3
import concurrent.futures
from typing import List, Optional
import logging
import traceback


# Set root of project (e.g., /home/pgx3874/pgx-analysis)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if project_root not in sys.path:
    sys.path.append(project_root)

from helpers_1997_13.common_imports import (
    s3_client, 
    S3_BUCKET, 
    pd
)

from helpers_1997_13.s3_utils import (
    parse_s3_path,
    parse_path_params,
    save_to_s3_parquet,
    get_output_paths,
    s3_exists
)

from helpers_1997_13.fpgrowth_utils import (
    save_feature_artifacts
)

from helpers_1997_13.data_utils import (
    safe_concat_columns
)

# Local fallbacks to avoid dependency on advanced duckdb_utils helpers
def get_standardized_column_order(sample_df, root_cols):
    """Return a stable column order: root columns first, then remaining sorted."""
    try:
        sample_columns = list(getattr(sample_df, 'columns', []) or [])
    except Exception:
        sample_columns = []
    ordered = list(root_cols)
    for col in sorted(c for c in sample_columns if c not in root_cols):
        ordered.append(col)
    return ordered

def generate_null_filled_select(source_alias: str, tagged_cols: list[str], override_map: dict[str, str]):
    """Generate a SELECT clause ensuring every tagged column is present.

    - If override_map provides an expression for a column, use it as the value.
    - Otherwise, default to NULL for safety (avoids referencing missing columns).
    """
    select_exprs: list[str] = []
    for col in tagged_cols:
        expr = override_map.get(col)
        if expr is None:
            expr = "NULL"
        # Ensure aliasing to the target column name
        if " as " not in expr.lower():
            expr = f"{expr} AS {col}"
        select_exprs.append(expr)
    return ",\n".join(select_exprs)

from helpers_1997_13.constants import BASE_PATH_COHORT, MAX_RETRIES


def get_cohort_paths(cohort_name, age_band, event_year, logger: Optional[logging.Logger] = None, bucket_name="pgxdatalake"):
    try:
        if isinstance(age_band, str) and age_band.startswith('age_band='):
            age_band = age_band.replace('age_band=', '')

        partitions = f"cohort_name={cohort_name}/event_year={event_year}/age_band={age_band}"
        s3_path = f"s3://{bucket_name}/cohorts/{partitions}/cohort.parquet"
        return s3_path
    except Exception as e:
        if logger:
            logger.error(f"Error generating cohort path: {str(e)}")
        else:
            print(f"Error generating cohort path: {str(e)}")
        raise


def list_cohort_input(cohort_name: str = "ed_non_opioid", bucket_name: str = "pgxdatalake", logger: Optional[logging.Logger] = None) -> List[str]:
    prefix = f"cohorts/cohort_name={cohort_name}/"
    results = []

    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("cohort.parquet"):
                path = f"s3://{bucket_name}/{key}"
                results.append(path)
                if logger:
                    logger.debug(f"Discovered cohort file: {path}")
    return results


def list_fpgrowth_cohort_output(cohort_name: str = "ed_non_opioid", logger: Optional[logging.Logger] = None, bucket_name: str = "pgxdatalake") -> List[str]:
    prefix = f"fpgrowth_features/cohort_name={cohort_name}/"
    result_paths = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("fpgrowth_features.parquet"):
                path = f"s3://{bucket_name}/{key}"
                result_paths.append(path)
                if logger:
                    logger.debug(f"Discovered output file: {path}")
    return result_paths


def check_cohort_needs_processing(s3_path, bucket_name: str = "pgxdatalake", logger: Optional[logging.Logger] = None):
    try:
        path_info = parse_path_params(s3_path)
        if not path_info:
            msg = f"Unable to parse path: {s3_path}"
            if logger:
                logger.error(msg)
            else:
                print(msg)
            return None

        cohort_name = path_info.get('cohort_name') or path_info.get('cohort_type')
        age_band = path_info.get('age_band')
        event_year = path_info.get('event_year')

        cohort_id = f"{cohort_name}/{age_band}/{event_year}"

        paths = get_output_paths(cohort_name, age_band, event_year, bucket_name=bucket_name)

        needs_processing = not all(s3_exists(path) for path in paths.values())

        msg = f"→ {cohort_id}: Needs feature engineering" if needs_processing else f"✓ {cohort_id}: Already processed"
        if logger:
            logger.info(msg)
        else:
            print(msg)

        return {
            'cohort_path': s3_path,
            'cohort_name': cohort_name,
            'age_band': age_band,
            'event_year': event_year,
            'needs_processing': needs_processing,
            'cohort_id': cohort_id
        }

    except Exception as e:
        msg = f"✗ Error checking {s3_path}: {str(e)}"
        if logger is None:
            # Fallback to a simple local logger to avoid import cycles
            logger = logging.getLogger("cohort_utils")
            if not logger.handlers:
                _handler = logging.StreamHandler()
                _handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                logger.addHandler(_handler)
            logger.setLevel(logging.INFO)
        logger.error(msg)
        return None


def check_cohort_exists(age_band, event_year, cohort_name, logger: Optional[logging.Logger] = None):
    key = f"cohorts/cohort_name={cohort_name}/age_band={age_band}/event_year={event_year}/cohort.parquet"
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        raise


def check_cohort_exists_and_delete_message(age_band, event_year, sqs_queue_url, receipt_handle, logger: Optional[logging.Logger] = None):
    try:
        if check_cohort_exists(age_band, event_year, "opioid_ed") and check_cohort_exists(age_band, event_year, "ed_non_opioid"):
            sqs = boto3.client('sqs')
            sqs.delete_message(
                QueueUrl=sqs_queue_url,
                ReceiptHandle=receipt_handle
            )
            return True
        return False
    except Exception as e:
        if logger:
            logger.error(f"Error checking cohorts: {str(e)}")
        else:
            print(f"Error checking cohorts: {str(e)}")
        return False


def check_and_fix_mismatched_sets(age_band, event_year, logger: Optional[logging.Logger] = None):
    try:
        opioid_ed_exists = check_cohort_exists(age_band, event_year, "opioid_ed", logger)
        ed_non_opioid_exists = check_cohort_exists(age_band, event_year, "ed_non_opioid", logger)

        if opioid_ed_exists and not ed_non_opioid_exists:
            if logger:
                logger.info("Only opioid_ed cohort exists, will process ed_non_opioid cohort")
            return "ed_non_opioid"
        elif not opioid_ed_exists and ed_non_opioid_exists:
            if logger:
                logger.info("Only ed_non_opioid cohort exists, will process opioid_ed cohort")
            return "opioid_ed"
        elif not opioid_ed_exists and not ed_non_opioid_exists:
            if logger:
                logger.info("Neither cohort exists, will process both")
            return None
        else:
            if logger:
                logger.info("Both cohorts exist, no processing needed")
            return None
    except Exception as e:
        if logger:
            logger.error(f"Error checking mismatched sets: {str(e)}")
        return None


def handle_empty_filtered_cohort(df, cohort_name, band, year, paths, logger: Optional[logging.Logger] = None, TOP_K=25):
    logger.warning(f"Skipping {cohort_name} {band} {year} — no valid rows after filtering")

    placeholder_columns = ["mi_person_key", "target", "drug_tokens", "tokens"]
    placeholder = pd.DataFrame(columns=placeholder_columns)

    padding = {
        f"pattern_{i+1}": ["missing"] for i in range(TOP_K)
    }
    padding.update({f"support_{i+1}": [0.0] for i in range(TOP_K)})
    padding.update({f"confidence_{i+1}": [0.0] for i in range(TOP_K)})
    padding.update({f"lift_{i+1}": [0.0] for i in range(TOP_K)})
    padding.update({f"certainty_{i+1}": [0.0] for i in range(TOP_K)})

    placeholder = safe_concat_columns(placeholder, padding)
    save_to_s3_parquet(placeholder, paths['fpgrowth_features'])
    logger.info(f"✓ Saved placeholder enhanced dataset with 0 valid rows: {paths['fpgrowth_features']}")

    save_feature_artifacts(placeholder, pd.DataFrame(), pd.DataFrame(), {}, paths, logger)
    return True


def check_and_reprocess_all_cohorts(
    base_path: Optional[str] = None,
    cohort_type: Optional[str] = None,
    age_band: Optional[str] = None,
    event_year: Optional[str] = None,
    ratio_threshold: float = 5.0,
    dry_run: bool = False,
    max_workers: int = 2,
    max_retries: int = MAX_RETRIES,
    logger: Optional[logging.Logger] = None
) -> tuple[List[str], List[str]]:
    
    
    """
    Identify and reprocess cohorts where the control/case ratio exceeds threshold.
    """
    

    def list_cohort_paths():
        prefix = BASE_PATH_COHORT.replace("s3://pgxdatalake/", "")
        paginator = s3_client.get_paginator("list_objects_v2")
        paths = []
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("cohort.parquet"):
                    if cohort_type and f"cohort_name={cohort_type}/" not in key:
                        continue
                    if age_band and f"age_band={age_band}/" not in key:
                        continue
                    if event_year and f"event_year={event_year}/" not in key:
                        continue
                    paths.append(f"s3://{S3_BUCKET}/{key}")
        if logger:
            logger.debug(f"Discovered cohort file: s3://{S3_BUCKET}/{key}")

        return paths


    def check_cohort_ratio(s3_path: str):
        try:
            path_info = parse_path_params(s3_path)
            if not path_info:
                return None

            df = pd.read_parquet(s3_path)
            counts = df["target"].value_counts()
            control = counts.get(0, 0)
            case = counts.get(1, 0)
            if case == 0:
                return None
            ratio = control / case
            return {
                "cohort_path": s3_path,
                "cohort_id": f"{path_info['cohort_name']}/{path_info['age_band']}/{path_info['event_year']}",
                "control": control,
                "case": case,
                "ratio": ratio,
                "needs_reprocessing": ratio > ratio_threshold
            }
        except Exception as e:
            if logger:
                logger.error(f"Failed to check cohort ratio for {s3_path}: {e}")
            return None

    def delete_cohort_file(cohort_path: str):
        try:
            bucket, key = parse_s3_path(cohort_path)
            s3_client.delete_object(Bucket=bucket, Key=key)
            lock_key = re.sub(r"/cohort\\.parquet$", "/lock", key)
            try:
                s3_client.delete_object(Bucket=bucket, Key=lock_key)
            except Exception:
                pass
            return True
        except Exception:
            return False


    def reprocess_cohort(cohort_path: str):
        path_info = parse_path_params(cohort_path)
        if not path_info:
            return False

        cohort_type = path_info["cohort_name"]
        age_band = path_info["age_band"]
        event_year = path_info["event_year"]

        if dry_run:
            logger.warning(f"[DRY RUN] Would reprocess {cohort_type}/{age_band}/{event_year}")
            return True

        deleted = delete_cohort_file(cohort_path)
        if not deleted:
            logger.error(f"Failed to delete {cohort_path}")
            return False

        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "create_cohort.py"),
            "--cohort", cohort_type,
            "--age-band", age_band,
            "--event-year", event_year
        ]

        for _ in range(max_retries):
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info(f"✓ Reprocessed {cohort_type}/{age_band}/{event_year}")
                return True
            except subprocess.CalledProcessError as e:
                logger.warning(f"Retrying {cohort_type}/{age_band}/{event_year} due to: {e.stderr}")
                time.sleep(2)

        logger.error(f"✗ Failed to reprocess {cohort_type}/{age_band}/{event_year}")
        return False

    base_path = base_path or BASE_PATH_COHORT
    cohort_paths = list_cohort_paths()
    if not cohort_paths:
        print("No cohorts found.")
        return [], []

    print(f"→ Found {len(cohort_paths)} cohort files")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_cohort_ratio, path): path for path in cohort_paths}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result and result.get("needs_reprocessing"):
                results.append(result)

    reprocessed, failed = [], []
    if not dry_run:
        for r in results:
            success = reprocess_cohort(r["cohort_path"])
            (reprocessed if success else failed).append(r["cohort_id"])
            time.sleep(2)

    return reprocessed, failed


def build_tagged_column_schema(conn, root_cols):
    sample_df = conn.sql("SELECT * FROM tagged_cohort_events LIMIT 1").df()
    return get_standardized_column_order(sample_df, root_cols)


def build_opioid_ed_union_query(tagged_cols, age_band, event_year):
    override_map = {
        "mi_person_key": "o.mi_person_key",
        "event_date": "o.opioid_ed_date",
        "age_band": f"'{age_band}'",
        "event_year": f"{event_year}",
        "drug_name": "o.drug_name",
        "therapeutic_class_1": "o.therapeutic_class_1",
        "therapeutic_class_2": "o.therapeutic_class_2",
        "therapeutic_class_3": "o.therapeutic_class_3",
        "first_opioid_ed_date": "o.opioid_ed_date",
        "First_Event": "'OPIOID_ED'",
        "days_to_opioid_ed": "o.days_to_opioid_ed",
        "data_source": "'drug_exposure'",
        "Event": "'Drug_Prescription'"
    }
    return generate_null_filled_select("o", tagged_cols, override_map)

    # ---------------------------------------------------------------------------
    # Orchestration helpers migrated from 2_create_cohort/calls/by_cohort.txt
    # These helpers are intentionally repository-agnostic and prefer existing
    # project-level constants (S3_BUCKET, project_root) defined above.
    # ---------------------------------------------------------------------------

import threading
import queue
import subprocess


def _enqueue_output(stream, q, tag):
    """Reader thread target: read lines from stream and push (tag,line) into queue.

    Keeps the orchestration script platform-independent (no select()).
    """
    try:
        for line in iter(stream.readline, ''):
            if not line:
                break
            q.put((tag, line))
    except Exception:
        q.put((tag, f"<<reader error: {traceback.format_exc()}>>\n"))
    finally:
        try:
            stream.close()
        except Exception:
            pass


def check_existing_cohorts(age_bands=None, event_years=None, bucket_name: str = None):
    """Check S3 for existing cohorts and return jobs that need processing.

    Returns a list of dicts: [{'age_band':..., 'event_year':...}, ...]
    """
    s3 = boto3.client("s3")
    age_bands = age_bands or [
        "0-12", "13-24", "25-44", "45-54", "55-64",
        "65-74", "75-84", "85-94", "95-114"
    ]
    event_years = event_years or [2016, 2017, 2018, 2019, 2020]
    bucket_name = bucket_name or S3_BUCKET

    jobs_to_process = []
    existing_cohorts = []
    processed_combinations = set()

    print("\nChecking existing cohorts and lock files in S3...")
    for band in age_bands:
        for year in event_years:
            combo_key = f"{band}_{year}"
            if combo_key in processed_combinations:
                continue

            opioid_ed_key = f"cohorts/cohort_name=opioid_ed/age_band={band}/event_year={year}/cohort.parquet"
            ed_non_opioid_key = f"cohorts/cohort_name=ed_non_opioid/age_band={band}/event_year={year}/cohort.parquet"
            lock_key = f"cohorts/locks/{band}_{year}.lock"

            opioid_ed_exists = False
            ed_non_opioid_exists = False
            lock_exists = False

            # Check lock
            try:
                s3.head_object(Bucket=bucket_name, Key=lock_key)
                lock_exists = True
                print(f"✗ Lock file exists for {band}/{year}, skipping...")
            except s3.exceptions.ClientError as e:
                if e.response['Error']['Code'] not in ("404", "NotFound"):
                    raise

            if lock_exists:
                continue

            # Check opioid_ed
            try:
                s3.head_object(Bucket=bucket_name, Key=opioid_ed_key)
                opioid_ed_exists = True
                print(f"✓ Opioid ED cohort exists for {band}/{year}")
            except s3.exceptions.ClientError as e:
                if e.response['Error']['Code'] in ("404", "NotFound"):
                    print(f"→ Missing opioid ED cohort for {band}/{year}")
                else:
                    raise

            # Check ed_non_opioid
            try:
                s3.head_object(Bucket=bucket_name, Key=ed_non_opioid_key)
                ed_non_opioid_exists = True
                print(f"✓ ED non-opioid cohort exists for {band}/{year}")
            except s3.exceptions.ClientError as e:
                if e.response['Error']['Code'] in ("404", "NotFound"):
                    print(f"→ Missing ED non-opioid cohort for {band}/{year}")
                else:
                    raise

            if opioid_ed_exists and ed_non_opioid_exists:
                existing_cohorts.append((band, year))
            else:
                jobs_to_process.append({
                    "age_band": band,
                    "event_year": year
                })
                processed_combinations.add(combo_key)

    print(f"\nSummary:")
    print(f"✓ {len(existing_cohorts)} age_band/year combinations fully processed")
    print(f"→ {len(jobs_to_process)} combinations need processing")

    if existing_cohorts:
        print("\nExisting cohorts:")
        for band, year in existing_cohorts:
            print(f"  - {band}/{year}")

    return jobs_to_process


def run_cohort(job, script_path, python_bin=sys.executable, target_icd=None):
    """Run the cohort entrypoint (script_path) for a single job and stream output.

    Returns one of: 'SUCCESS: id', 'FAILED: id (code)', 'SKIPPED_LOCKED: id', 'ERROR: id - msg'
    """
    target_icd = target_icd or os.environ.get("PGX_TARGET_ICD_CODES", "F1120")
    cmd = [
        python_bin, script_path,
        "--age-band", job["age_band"],
        "--event-year", str(job["event_year"]),
        "--cohort", "both",
        "--target-icd-codes", target_icd,
        "--starting-step", "phase1_data_preparation",
        "--operation-type", "concurrent_processing",
        "--log-level", "INFO",
    ]

    job_id = f"{job['age_band']}/{job['event_year']}"
    print(f"\nStarting job: {job_id}", flush=True)
    try:
        print(f"Running command: {' '.join(cmd)}", flush=True)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        print(f"\n{'='*30} Live Output for {job_id} {'='*30}", flush=True)

        q = queue.Queue()
        t_out = threading.Thread(target=_enqueue_output, args=(process.stdout, q, 'OUT'), daemon=True)
        t_err = threading.Thread(target=_enqueue_output, args=(process.stderr, q, 'ERR'), daemon=True)
        t_out.start()
        t_err.start()

        lock_check_buffer = ""
        has_lock_message = False

        while process.poll() is None or not q.empty():
            try:
                tag, line = q.get(timeout=0.5)
            except queue.Empty:
                continue

            if tag == 'OUT':
                print(line, end='', flush=True)
                if not has_lock_message and "Lock already exists" in line:
                    has_lock_message = True
                if len(lock_check_buffer) < 1000:
                    lock_check_buffer += line
            else:
                print(f"STDERR: {line}", end='', flush=True)

        t_out.join(timeout=1)
        t_err.join(timeout=1)

        returncode = process.returncode

        print(f"\n{'='*30} End of Output for {job_id} {'='*30}", flush=True)
        print(f"Return code: {returncode}", flush=True)

        if has_lock_message or "Lock already exists" in lock_check_buffer:
            print(f"✓ Job for {job_id} skipped due to existing lock", flush=True)
            return f"SKIPPED_LOCKED: {job_id}"
        elif returncode != 0:
            print(f"✗ Job for {job_id} failed with code {returncode}", flush=True)
            return f"FAILED: {job_id} (code: {returncode})"
        else:
            print(f"✓ Job for {job_id} completed successfully with code {returncode}", flush=True)
            return f"SUCCESS: {job_id}"
    except Exception as e:
        error_msg = f"✗ Job for {job_id} raised exception: {str(e)}"
        print(error_msg, flush=True)
        print(traceback.format_exc(), flush=True)
        return f"ERROR: {job_id} - {str(e)}"



def build_ed_non_opioid_union_query(tagged_cols, age_band, event_year):
    override_map = {
        "mi_person_key": "a.mi_person_key",
        "event_date": "a.ade_date",
        "age_band": f"'{age_band}'",
        "event_year": f"{event_year}",
        "drug_name": "a.drug_name",
        "therapeutic_class_1": "a.therapeutic_class_1",
        "therapeutic_class_2": "a.therapeutic_class_2",
        "therapeutic_class_3": "a.therapeutic_class_3",
        "first_ed_non_opioid_date": "a.ade_date",
        "First_Event": "'ED_NON_OPIOID'",
        "days_to_ade": "a.days_to_ade",
        "data_source": "'drug_exposure'",
        "Event": "'Drug_Prescription'"
    }
    return generate_null_filled_select("a", tagged_cols, override_map)

