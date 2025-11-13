import argparse
import os
import sys
import logging
import multiprocessing as mp
import time
from typing import List, Iterable, Tuple, Optional
import boto3
import json
import psutil

# Add project root to path (helpers folder is at pgx-analysis level)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from helpers_1997_13.logging_utils import (
        setup_logging,
        save_logs_to_s3,
        setup_mp_logging,
        get_worker_queue_logger,
        stop_mp_logging,
    )
    from helpers_1997_13.duckdb_utils import create_simple_duckdb_connection
    from helpers_1997_13.pipeline_utils import PipelineState
except Exception:
    # Fallback if helpers path not available; will use local logger
    setup_logging = None
    save_logs_to_s3 = None
    setup_mp_logging = None
    get_worker_queue_logger = None
    stop_mp_logging = None
    create_simple_duckdb_connection = None
    PipelineState = None



# Reuse one DuckDB connection per worker process to reduce setup latency
_WORKER_CONN = None

def _log_system_metrics(logger: logging.Logger, label: str) -> None:
    try:
        if psutil is None:
            return
        cpu = psutil.cpu_percent(interval=0.1)
        vm = psutil.virtual_memory()
        try:
            net = psutil.net_io_counters()
            net_str = f"NET sent={net.bytes_sent/1_000_000:.1f}MB recv={net.bytes_recv/1_000_000:.1f}MB"
        except Exception:
            net_str = "NET n/a"
        try:
            dio = psutil.disk_io_counters()
            io_str = f"DISK r={dio.read_bytes/1_000_000:.1f}MB w={dio.write_bytes/1_000_000:.1f}MB"
        except Exception:
            io_str = "DISK n/a"
        logger.info(f"METRICS {label}: CPU={cpu:.1f}% MEM={vm.percent:.1f}% ({vm.available/1_073_741_824:.1f}GiB free) {net_str} {io_str}")
    except Exception:
        pass

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("bronze_ingest")
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
def _load_expected_header(dataset: str) -> List[str]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if dataset == 'medical':
        path = os.path.join(base_dir, 'apcd', 'medical', 'medical_head.txt')
    else:
        path = os.path.join(base_dir, 'apcd', 'pharmacy', 'pharmacy_head.txt')
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\n')
    return header.split('|')


def _dataset_from_uri(uri: str) -> str:
    low = uri.lower()
    if '/pharmacy/' in low or low.endswith('/pharmacy'):
        return 'pharmacy'
    return 'medical'



def list_s3_txt_files(bucket: str, prefix: str) -> Iterable[str]:
    s3 = boto3.client("s3")
    continuation_token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".txt") or key.lower().endswith(".csv"):
                yield key
        if resp.get("IsTruncated"):
            continuation_token = resp.get("NextContinuationToken")
        else:
            break


def s3_path(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def output_key_for_bronze(input_key: str, dataset: str, bronze_root: str) -> Tuple[str, str]:
    base = os.path.basename(input_key)
    name, _ext = os.path.splitext(base)
    out_prefix = f"{bronze_root.strip('/').rstrip('/')}/{dataset.lower()}".strip('/')
    # Return (out_bucket, out_key)
    parts = out_prefix.split('/', 1)
    out_bucket = parts[0]
    out_key_prefix = parts[1] if len(parts) > 1 else ''
    out_key = f"{out_key_prefix}/{name}.parquet" if out_key_prefix else f"{name}.parquet"
    return out_bucket, out_key

def list_staged_good_files(bronze_root: str, dataset: str) -> Iterable[str]:
    """Yield s3:// URIs for staged good files under bronze/_staging/<dataset>/*.good.txt"""
    s3 = boto3.client("s3")
    bucket, root_key = _parse_s3_uri(bronze_root)
    stage_prefix = f"{root_key.rstrip('/')}/_staging/{dataset}/" if root_key else f"_staging/{dataset}/"
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": stage_prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".good.txt"):
                yield f"s3://{bucket}/{key}"
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break


def s3_exists(bucket: str, key: str) -> bool:
    s3 = boto3.client("s3")
    try:
        resp = s3.head_object(Bucket=bucket, Key=key)
        # Treat zero-length objects as non-existent for idempotency purposes
        return resp.get("ContentLength", 0) > 0
    except s3.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        # If 404/NoSuchKey → definitely absent; if 403, assume present (skip) to avoid reprocessing
        if code in ("404", "NoSuchKey"):
            return False
        if code == "403":
            # Assume object exists but HEAD is forbidden; err on the side of idempotent skip
            return True
        raise


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    rest = uri[5:]
    parts = rest.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def convert_one_with_duckdb(input_uri: str, output_uri: str, logger: Optional[logging.Logger] = None, duckdb_threads: int = 1, tmp_dir: Optional[str] = None, proceed_on_errors: bool = False) -> None:
    if logger is None:
        logger = setup_logger()
    # Helper to quietly sample first bytes to infer delimiter when needed
    def _infer_delim_counts() -> Tuple[int, int]:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(input_uri)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=bucket, Key=key, Range="bytes=0-65535")
            raw = obj["Body"].read()
            text = raw.decode("latin1", errors="ignore")
            first_non_empty = next((ln for ln in text.splitlines() if ln.strip()), "")
            return first_non_empty.count('\t'), first_non_empty.count('|')
        except Exception:
            return 0, 0

    # Consistent DuckDB setup with helpers; reuse a single connection per worker
    global _WORKER_CONN
    if _WORKER_CONN is None:
        if create_simple_duckdb_connection is None:
            import duckdb
            _WORKER_CONN = duckdb.connect()
        else:
            _WORKER_CONN = create_simple_duckdb_connection(logger, tmp_dir)
        _log_system_metrics(logger, "worker_init")
        # Control threads per process to avoid oversubscription
        try:
            _WORKER_CONN.sql(f"PRAGMA threads={int(duckdb_threads)}")
        except Exception:
            _WORKER_CONN.sql("PRAGMA threads=1")
        # Boost S3 and temp spill performance where supported
        try:
            _WORKER_CONN.sql("PRAGMA s3_max_io_threads=64")
            _WORKER_CONN.sql("PRAGMA s3_max_connections=64")
        except Exception:
            pass
        try:
            if tmp_dir:
                _WORKER_CONN.sql(f"PRAGMA temp_directory='{tmp_dir}'")
            _WORKER_CONN.sql("PRAGMA memory_limit='80%'")
        except Exception:
            pass
        # Inject AWS credentials from boto3 into DuckDB (robust in EC2 workers)
        try:
            import boto3 as _b3
            _sess = _b3.Session()
            _creds = _sess.get_credentials()
            _region = _sess.region_name or "us-east-1"
            if _creds is not None:
                _f = _creds.get_frozen_credentials()
                if getattr(_f, "access_key", None) and getattr(_f, "secret_key", None):
                    _WORKER_CONN.sql(f"SET s3_access_key_id='{_f.access_key}'")
                    _WORKER_CONN.sql(f"SET s3_secret_access_key='{_f.secret_key}'")
                    if getattr(_f, "token", None):
                        _WORKER_CONN.sql(f"SET s3_session_token='{_f.token}'")
            _WORKER_CONN.sql(f"SET s3_region='{_region}'")
            _WORKER_CONN.sql("SET s3_url_style='path'")
        except Exception as _inj_e:
            if logger:
                logger.warning(f"Could not inject AWS credentials into DuckDB: {_inj_e}")

    con = _WORKER_CONN

    logger.info(f"Converting → {input_uri} → {output_uri}")
    _log_system_metrics(logger, "before_copy")

    # Convert with robust fallbacks for encoding/delimiter issues
    def _exec_copy(sql: str) -> None:
        con.sql(sql)
        _log_system_metrics(logger, "after_copy")

    def make_copy_sql_forced(delim: str = '|', enc: str = 'CP1252') -> str:
        ds = _dataset_from_uri(input_uri)
        # Predefined columns per dataset (do not read from disk)
        MEDICAL_HEADER = (
            "Incurred Date|Claim ID|MI Person Key|Payer LOB|Payer Type|Claim Status|Primary Insurance Flag|"
            "Member Zip Code DOS|Member County DOS|Member State ENROLL|Member Age DOS|Member Age Band DOS|ADULT_FLAG|"
            "Member Gender|Member Race|Hispanic Indicator|CCHG Label|CCHG Grouping|HCG Setting|HCG Line|HCG Detail|"
            "Place of Service|Bill Type ID|Bill Type Class|Bill Type Description|Revenue Code|Admit Type|"
            "Primary ICD Diagnosis Code|Primary ICD Diagnosis Desc|Primary ICD Rollup|"
            "2nd ICD Diagnosis Code|2nd ICD Diagnosis Desc|2nd ICD Rollup|"
            "3rd ICD Diagnosis Code|3rd ICD Diagnosis Desc|3rd ICD Rollup|"
            "4th ICD Diagnosis Code|4th ICD Diagnosis Desc|4th ICD Rollup|"
            "5th ICD Diagnosis Code|5th ICD Diagnosis Desc|5th ICD Rollup|"
            "6th ICD Diagnosis Code|6th ICD Diagnosis Desc|6th ICD Rollup|"
            "7th ICD Diagnosis Code|7th ICD Diagnosis Desc|7th ICD Rollup|"
            "8th ICD Diagnosis Code|8th ICD Diagnosis Desc|8th ICD Rollup|"
            "9th ICD Diagnosis Code|9th ICD Diagnosis Desc|9th ICD Rollup|"
            "10th ICD Diagnosis Code|10th ICD Diagnosis Desc|10th ICD Rollup|"
            "Primary ICD CCS Level 1|Primary ICD CCS Level 2|Primary ICD CCS Level 3|"
            "Procedure Code|Procedure Name|Procedure Family 1|Procedure Family 2|Procedure Family 3|"
            "Primary ICD Procedure Code|Primary ICD Procedure Desc|"
            "2nd ICD Procedure Code|2nd ICD Procedure Desc|"
            "3rd ICD Procedure Code|3rd ICD Procedure Desc|"
            "4th ICD Procedure Code|4th ICD Procedure Desc|"
            "5th ICD Procedure Code|5th ICD Procedure Desc|"
            "6th ICD Procedure Code|6th ICD Procedure Desc|"
            "7th ICD Procedure Code|7th ICD Procedure Desc|"
            "8th ICD Procedure Code|8th ICD Procedure Desc|"
            "9th ICD Procedure Code|9th ICD Procedure Desc|"
            "10th ICD Procedure Code|10th ICD Procedure Desc|"
            "CPT Mod 1 Code|CPT Mod 1 Desc|CPT Mod 2 Code|CPT Mod 2 Desc|"
            "DRG Code|DRG Desc|DRG Type|"
            "Billing Provider Name|Billing Provider NPI|Billing Provider Specialty|Billing Provider ZIP|"
            "Billing Provider County|Billing Provider State|Billing Provider MSA|Billing Provider Taxonomy|Billing Provider TIN|"
            "Service Provider Name|Service Provider NPI|Service Provider Specialty|Service Provider ZIP|"
            "Service Provider County|Service Provider State|Service Provider MSA|Service Provider Taxonomy|Service Provider TIN|"
            "Total Allowed|Total Utilization|Total Paid"
        )
        PHARMACY_HEADER = (
            "Incurred Date|Claim ID|MI Person Key|Payer LOB|Payer Type|Claim Status|Primary Insurance Flag|"
            "Member Zip Code DOS|Member County DOS|Member State ENROLL|Member Age DOS|Member Age Band DOS|ADULT_FLAG|"
            "Member Gender|Member Race|Hispanic Indicator|HCG Setting|HCG Line|HCG Detail|"
            "Therapeutic Class 1|Therapeutic Class 2|Therapeutic Class 3|NDC|Drug Code|Drug Name|"
            "GPI|GPI Generic Name|Manufacturer|Strength|Dosage Form|"
            "Billing Provider NPI|Billing Provider Specialty|Billing Provider ZIP|Billing Provider County|Billing Provider State|"
            "Billing Provider MSA|Billing Provider Taxonomy|Billing Provider TIN|"
            "Service Provider Name|Service Provider NPI|Service Provider Specialty|Service Provider ZIP|"
            "Service Provider County|Service Provider State|Service Provider MSA|Service Provider Taxonomy|Service Provider TIN|"
            "Total Allowed|Total Utilization|Total RX Paid|Total RX Days Supply"
        )
        header = PHARMACY_HEADER if ds == 'pharmacy' else MEDICAL_HEADER
        names = header.split('|')
        # Build columns map: all VARCHAR
        if names:
            def esc(s: str) -> str:
                return s.replace("'", "''")
            cols_map = ", ".join([f"'{esc(col)}': 'VARCHAR'" for col in names])
            columns_clause = f"COLUMNS={{ {cols_map} }}"
        else:
            columns_clause = "ALL_VARCHAR=TRUE"
        opts = [
            f"DELIM='{delim}'",
            "HEADER=TRUE",
            f"ENCODING='{enc}'",
            columns_clause,
            "SAMPLE_SIZE=100000",
        ]
        opts_str = ",\n                         ".join(opts)
        return f"""
        COPY (
          SELECT *
          FROM read_csv('{input_uri}',
                         {opts_str})
        ) TO '{output_uri}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 5000000, OVERWRITE_OR_IGNORE true);
        """

    def make_copy_sql(enc: str | None = None, delim: str | None = None, all_varchar: bool = False) -> str:
        opts = [
            "SAMPLE_SIZE=2000000",
            "HEADER=TRUE",
            "AUTO_DETECT=TRUE",
        ]
        if proceed_on_errors:
            opts.append("IGNORE_ERRORS=TRUE")
        if enc:
            opts.append(f"ENCODING='{enc}'")
        if delim:
            opts.append(f"DELIM='{delim}'")
        if all_varchar:
            opts.append("ALL_VARCHAR=TRUE")
        opts_str = ",\n                         ".join(opts)
        return f"""
        COPY (
          SELECT *
          FROM read_csv_auto('{input_uri}',
                             {opts_str})
        ) TO '{output_uri}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000, OVERWRITE_OR_IGNORE true);
        """

    # Attempt 0: If path indicates known layout, force fast path without auto-detect
    try:
        ds = _dataset_from_uri(input_uri)
        if ds == 'pharmacy':
            logger.info("Using forced layout for Pharmacy: DELIM '|' ENCODING CP1252, ALL_VARCHAR")
            _exec_copy(make_copy_sql_forced(delim='|', enc='CP1252'))
            return
        if ds == 'medical':
            logger.info("Using forced layout for Medical: DELIM '|' ENCODING CP1252, ALL_VARCHAR")
            _exec_copy(make_copy_sql_forced(delim='|', enc='CP1252'))
            return
    except Exception as forced_e:
        logger.warning(f"Forced layout path failed, falling back ({str(forced_e)[:220]}…)")

    # Attempt 1: Default (UTF-8, auto delimiter). If it fails, log a concise sample of the error.
    try:
        _exec_copy(make_copy_sql())
        return
    except Exception as e1:
        msg = str(e1)
        logger.warning(f"Default read_csv_auto failed, retrying with encoding + delimiter based on preview... ({msg[:220]}…)")

    # Use quiet heuristic for delimiter preference (no preview logging)
    tcnt, pcnt = _infer_delim_counts()
    preferred_delim = '|' if pcnt >= tcnt else '\t'
    alt_delim = '\t' if preferred_delim == '|' else '|'

    # Attempt 2/3: Try supported encodings with preferred delimiter
    for enc in ("ISO8859_1", "CP1252"):
        try:
            _exec_copy(make_copy_sql(enc=enc, delim=preferred_delim))
            return
        except Exception as e_enc_pref:
            logger.warning(f"{enc} + '{preferred_delim}' failed ({str(e_enc_pref)[:120]}…). Trying ALL_VARCHAR…")
            try:
                _exec_copy(make_copy_sql(enc=enc, delim=preferred_delim, all_varchar=True))
                return
            except Exception:
                pass

    # Attempt 4/5: Try supported encodings with alternate delimiter
    for enc in ("ISO8859_1", "CP1252"):
        try:
            _exec_copy(make_copy_sql(enc=enc, delim=alt_delim))
            return
        except Exception as e_enc_alt:
            logger.warning(f"{enc} + '{alt_delim}' failed ({str(e_enc_alt)[:120]}…). Trying ALL_VARCHAR…")
            try:
                _exec_copy(make_copy_sql(enc=enc, delim=alt_delim, all_varchar=True))
                return
            except Exception:
                pass

    # Final fallback: try CP1252 with no delimiter override but ALL_VARCHAR
    _exec_copy(make_copy_sql(enc="CP1252", all_varchar=True))


def _split_and_stage_good(input_uri: str, dataset: str, bronze_root: str, logger: logging.Logger) -> Tuple[str, str]:
    """Stream-split a large TXT into good/rejects in S3 and return staged good/reject URIs."""
    s3 = boto3.client("s3")
    in_bucket, in_key = _parse_s3_uri(input_uri)
    _log_system_metrics(logger, "split_start")
    split_t0 = time.time()
    # Destinations
    dest = bronze_root.replace("s3://", "")
    dest_bucket = dest.split("/", 1)[0]
    dest_prefix = dest.split("/", 1)[1] if "/" in dest else ""
    base = os.path.basename(in_key)
    name, _ = os.path.splitext(base)
    good_key = f"{dest_prefix}_staging/{dataset}/{name}.good.txt" if dest_prefix else f"_staging/{dataset}/{name}.good.txt"
    bad_key = f"{dest_prefix}_rejects/{dataset}/{name}.rejects.txt" if dest_prefix else f"_rejects/{dataset}/{name}.rejects.txt"

    # Stream
    obj = s3.get_object(Bucket=in_bucket, Key=in_key)
    stream = obj["Body"]
    import io
    good_buf = io.BytesIO()
    bad_buf = io.BytesIO()
    delim = b"|"
    header = next(iter(stream.iter_lines(chunk_size=1024*1024)))
    expected = header.count(delim) if header else 0
    good_buf.write(header + b"\n")
    line_no = 1
    bytes_read = len(header) + 1
    next_report = time.time() + 30
    for chunk in stream.iter_lines(chunk_size=8*1024*1024):
        line_no += 1
        if not chunk:
            continue
        bytes_read += len(chunk) + 1
        if chunk.count(delim) == expected:
            good_buf.write(chunk + b"\n")
        else:
            bad_buf.write(chunk[:200] + b"\n")
        now = time.time()
        if now >= next_report:
            mb = bytes_read / (1024*1024)
            rate = mb / max(1e-6, (now - split_t0))
            logger.info(f"SPLIT_PROGRESS {name}: lines={line_no:,} bytes={bytes_read:,} (~{mb:.1f} MB) rate={rate:.1f} MB/s")
            _log_system_metrics(logger, "split_progress")
            next_report = now + 30

    # Upload
    good_buf.seek(0)
    s3.upload_fileobj(good_buf, dest_bucket, good_key)
    bad_buf.seek(0)
    s3.upload_fileobj(bad_buf, dest_bucket, bad_key)
    split_elapsed = time.time() - split_t0
    if split_elapsed > 0:
        logger.info(f"SPLIT_DONE {name}: lines={line_no:,} bytes={bytes_read:,} in {split_elapsed:.1f}s ({bytes_read/1024/1024/split_elapsed:.1f} MB/s)")
    _log_system_metrics(logger, "split_done")
    return s3_path(dest_bucket, good_key), s3_path(dest_bucket, bad_key)


def _derive_staged_good_uri(input_uri: str, dataset: str, bronze_root: str) -> Tuple[str, str]:
    """Compute expected staged good and rejects URIs for an input file without performing work."""
    in_bucket, in_key = _parse_s3_uri(input_uri)
    dest = bronze_root.replace("s3://", "")
    dest_bucket = dest.split("/", 1)[0]
    dest_prefix = dest.split("/", 1)[1] if "/" in dest else ""
    base = os.path.basename(in_key)
    name, _ = os.path.splitext(base)
    good_key = f"{dest_prefix}_staging/{dataset}/{name}.good.txt" if dest_prefix else f"_staging/{dataset}/{name}.good.txt"
    bad_key = f"{dest_prefix}_rejects/{dataset}/{name}.rejects.txt" if dest_prefix else f"_rejects/{dataset}/{name}.rejects.txt"
    return s3_path(dest_bucket, good_key), s3_path(dest_bucket, bad_key)

def _sanitize_to_s3(input_uri: str, dataset: str, bronze_root: str, logger: logging.Logger) -> Tuple[str, str, dict]:
    """Sanitize a delimited TXT by enforcing expected column count using header map.

    Writes fixed lines to _fixed, unfixable to _rejects2. Returns (fixed_uri, rejects2_uri, stats).
    """
    # Column expectations
    if dataset == 'pharmacy':
        header = (
            "Incurred Date|Claim ID|MI Person Key|Payer LOB|Payer Type|Claim Status|Primary Insurance Flag|"
            "Member Zip Code DOS|Member County DOS|Member State ENROLL|Member Age DOS|Member Age Band DOS|ADULT_FLAG|"
            "Member Gender|Member Race|Hispanic Indicator|HCG Setting|HCG Line|HCG Detail|"
            "Therapeutic Class 1|Therapeutic Class 2|Therapeutic Class 3|NDC|Drug Code|Drug Name|"
            "GPI|GPI Generic Name|Manufacturer|Strength|Dosage Form|"
            "Billing Provider NPI|Billing Provider Specialty|Billing Provider ZIP|Billing Provider County|Billing Provider State|"
            "Billing Provider MSA|Billing Provider Taxonomy|Billing Provider TIN|"
            "Service Provider Name|Service Provider NPI|Service Provider Specialty|Service Provider ZIP|"
            "Service Provider County|Service Provider State|Service Provider MSA|Service Provider Taxonomy|Service Provider TIN|"
            "Total Allowed|Total Utilization|Total RX Paid|Total RX Days Supply"
        )
        tail_numeric = 4
    else:
        header = (
            "Incurred Date|Claim ID|MI Person Key|Payer LOB|Payer Type|Claim Status|Primary Insurance Flag|"
            "Member Zip Code DOS|Member County DOS|Member State ENROLL|Member Age DOS|Member Age Band DOS|ADULT_FLAG|"
            "Member Gender|Member Race|Hispanic Indicator|CCHG Label|CCHG Grouping|HCG Setting|HCG Line|HCG Detail|"
            "Place of Service|Bill Type ID|Bill Type Class|Bill Type Description|Revenue Code|Admit Type|"
            "Primary ICD Diagnosis Code|Primary ICD Diagnosis Desc|Primary ICD Rollup|"
            "2nd ICD Diagnosis Code|2nd ICD Diagnosis Desc|2nd ICD Rollup|"
            "3rd ICD Diagnosis Code|3rd ICD Diagnosis Desc|3rd ICD Rollup|"
            "4th ICD Diagnosis Code|4th ICD Diagnosis Desc|4th ICD Rollup|"
            "5th ICD Diagnosis Code|5th ICD Diagnosis Desc|5th ICD Rollup|"
            "6th ICD Diagnosis Code|6th ICD Diagnosis Desc|6th ICD Rollup|"
            "7th ICD Diagnosis Code|7th ICD Diagnosis Desc|7th ICD Rollup|"
            "8th ICD Diagnosis Code|8th ICD Diagnosis Desc|8th ICD Rollup|"
            "9th ICD Diagnosis Code|9th ICD Diagnosis Desc|9th ICD Rollup|"
            "10th ICD Diagnosis Code|10th ICD Diagnosis Desc|10th ICD Rollup|"
            "Primary ICD CCS Level 1|Primary ICD CCS Level 2|Primary ICD CCS Level 3|"
            "Procedure Code|Procedure Name|Procedure Family 1|Procedure Family 2|Procedure Family 3|"
            "Primary ICD Procedure Code|Primary ICD Procedure Desc|"
            "2nd ICD Procedure Code|2nd ICD Procedure Desc|"
            "3rd ICD Procedure Code|3rd ICD Procedure Desc|"
            "4th ICD Procedure Code|4th ICD Procedure Desc|"
            "5th ICD Procedure Code|5th ICD Procedure Desc|"
            "6th ICD Procedure Code|6th ICD Procedure Desc|6th ICD Rollup|"
            "7th ICD Procedure Code|7th ICD Procedure Desc|"
            "8th ICD Procedure Code|8th ICD Procedure Desc|"
            "9th ICD Procedure Code|9th ICD Procedure Desc|"
            "10th ICD Procedure Code|10th ICD Procedure Desc|"
            "CPT Mod 1 Code|CPT Mod 1 Desc|CPT Mod 2 Code|CPT Mod 2 Desc|"
            "DRG Code|DRG Desc|DRG Type|"
            "Billing Provider Name|Billing Provider NPI|Billing Provider Specialty|Billing Provider ZIP|"
            "Billing Provider County|Billing Provider State|Billing Provider MSA|Billing Provider Taxonomy|Billing Provider TIN|"
            "Service Provider Name|Service Provider NPI|Service Provider Specialty|Service Provider ZIP|"
            "Service Provider County|Service Provider State|Service Provider MSA|Service Provider Taxonomy|Service Provider TIN|"
            "Total Allowed|Total Utilization|Total Paid"
        )
        tail_numeric = 3

    expected = header.split('|')
    n_expected = len(expected)
    head_n = n_expected - tail_numeric
    # choose a spill column near tail: prefer Desc/Name
    spill_idx = head_n - 1
    for i in range(head_n - 1, -1, -1):
        nm = expected[i].lower()
        if 'desc' in nm or 'name' in nm:
            spill_idx = i
            break

    in_bucket, in_key = _parse_s3_uri(input_uri)
    dest = bronze_root.replace('s3://', '')
    out_bucket = dest.split('/', 1)[0]
    out_prefix = dest.split('/', 1)[1] if '/' in dest else ''
    base = os.path.basename(in_key)
    stem = os.path.splitext(base)[0]
    fixed_key = f"{out_prefix}_fixed/{dataset}/{stem}.fixed.txt" if out_prefix else f"_fixed/{dataset}/{stem}.fixed.txt"
    rej2_key = f"{out_prefix}_rejects2/{dataset}/{stem}.rejects2.txt" if out_prefix else f"_rejects2/{dataset}/{stem}.rejects2.txt"

    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=in_bucket, Key=in_key)
    stream = obj['Body']
    import io
    fixed_buf = io.BytesIO()
    rej2_buf = io.BytesIO()
    fixed_buf.write((header + '\n').encode('utf-8'))
    stats = {"total": 0, "fixed": 0, "padded": 0, "spilled": 0, "unfixable": 0}

    for raw in stream.iter_lines(chunk_size=8*1024*1024):
        if not raw:
            continue
        stats["total"] += 1
        try:
            line = raw.decode('cp1252', errors='ignore')
        except Exception:
            line = raw.decode('latin1', errors='ignore')
        parts = line.rstrip('\n').split('|')
        m = len(parts)
        if m == n_expected:
            fixed_buf.write((line.rstrip('\n') + '\n').encode('utf-8'))
            stats["fixed"] += 1
            continue
        if m < n_expected:
            parts = parts + [''] * (n_expected - m)
            fixed_buf.write(('|'.join(parts) + '\n').encode('utf-8'))
            stats["padded"] += 1
            continue
        # m > n_expected: right-anchor tail
        if m >= tail_numeric:
            tail = parts[-tail_numeric:]
            head = parts[:m - tail_numeric]
            if len(head) >= head_n:
                left = head[:spill_idx]
                spill = '|'.join(head[spill_idx:len(head) - (head_n - spill_idx - 1)])
                right = head[len(head) - (head_n - spill_idx - 1):head_n]
                new_head = left + [spill] + right
            else:
                new_head = head + [''] * (head_n - len(head))
            rebuilt = new_head + tail
            if len(rebuilt) == n_expected:
                fixed_buf.write(('|'.join(rebuilt) + '\n').encode('utf-8'))
                stats["spilled"] += 1
                continue
        rej2_buf.write((line.rstrip('\n') + '\n').encode('utf-8'))
        stats["unfixable"] += 1

    # upload artifacts
    fixed_buf.seek(0)
    s3.upload_fileobj(fixed_buf, out_bucket, fixed_key)
    rej2_buf.seek(0)
    if rej2_buf.getbuffer().nbytes > 0:
        s3.upload_fileobj(rej2_buf, out_bucket, rej2_key)

    fixed_uri = f"s3://{out_bucket}/{fixed_key}"
    rej2_uri = f"s3://{out_bucket}/{rej2_key}"
    logger.info(f"SANITIZE_DONE {stem}: stats={stats} fixed={fixed_uri} rejects2={rej2_uri}")
    return fixed_uri, rej2_uri, stats


def _convert_worker(task: Tuple[str, str, bool, int, Optional[str], bool, Optional[mp.Queue], str, str, bool]) -> Tuple[str, bool, str]:
    """Worker-safe split+convert with idempotent skip.

    Returns (output_uri, success, message)
    """
    input_uri, output_uri, overwrite, duckdb_threads, tmp_dir, proceed_on_errors, log_queue, dataset, bronze_root, split_rejects = task
    try:
        logger = None
        if log_queue is not None and get_worker_queue_logger is not None:
            logger = get_worker_queue_logger(log_queue, name="txt_to_parquet.worker")
        else:
            logger = setup_logger()
        out_bucket, out_key = _parse_s3_uri(output_uri)
        if s3_exists(out_bucket, out_key) and not overwrite:
            return output_uri, True, "skipped_exists"

        # Optional split in worker
        source_uri = input_uri
        if split_rejects:
            try:
                # If staged good already exists, reuse it to ensure idempotency and skip splitting
                exp_good, exp_bad = _derive_staged_good_uri(input_uri, dataset, bronze_root)
                g_bucket, g_key = _parse_s3_uri(exp_good)
                if s3_exists(g_bucket, g_key):
                    logger.info(f"Reusing existing staged good: {exp_good}")
                    source_uri = exp_good
                else:
                    # Try split; on split error, attempt sanitize+convert as fallback
                    try:
                        staged_good, staged_bad = _split_and_stage_good(input_uri, dataset, bronze_root, logger)
                        logger.info(f"Staged good rows: {staged_good}; rejects: {staged_bad}")
                        source_uri = staged_good
                    except Exception as split_err:
                        logger.warning(f"split_error: {split_err}; attempting sanitize fallback…")
                        fixed_uri, rej2_uri, st = _sanitize_to_s3(input_uri, dataset, bronze_root, logger)
                        logger.info(f"Sanitized: {fixed_uri} (unfixable: {st['unfixable']})")
                        source_uri = fixed_uri
            except Exception as se:
                return output_uri, False, f"split_error: {se}"

        convert_one_with_duckdb(source_uri, output_uri, logger=logger, duckdb_threads=duckdb_threads, tmp_dir=tmp_dir, proceed_on_errors=proceed_on_errors)
        return output_uri, True, "converted"
    except Exception as e:
        return output_uri, False, f"error: {e}"


def process_dataset(dataset: str, input_root: str, bronze_root: str, limit: int, dry_run: bool, overwrite: bool, logger: logging.Logger, workers: int, duckdb_threads: int, tmp_dir: Optional[str], proceed_on_errors: bool, split_rejects: bool, shard_count: int = 0, shard_index: int = 0, log_queue: Optional[mp.Queue] = None) -> dict:
    # Parse bucket and prefix
    if not input_root.startswith("s3://"):
        raise ValueError("input_root must be an s3:// URI")
    inp = input_root.replace("s3://", "")
    in_bucket, in_prefix = (inp.split("/", 1) + [""])[0:2]

    # Bronze bucket from bronze_root
    if not bronze_root.startswith("s3://"):
        raise ValueError("bronze_root must be an s3:// URI")

    # Pre-pass: convert any existing staged good files that are missing Parquet
    backlog_tasks: List[Tuple[str, str, bool, int, Optional[str], bool, Optional[mp.Queue], str, str, bool]] = []
    staged_count = 0
    for good_uri in list_staged_good_files(bronze_root, dataset):
        staged_count += 1
        # derive output uri from staged base name
        _b, g_key = _parse_s3_uri(good_uri)
        base = os.path.basename(g_key)
        stem = base[:-9] if base.lower().endswith('.good.txt') else os.path.splitext(base)[0]
        out_bucket, out_key = _parse_s3_uri(bronze_root)[0], ''
        # rebuild bronze path using output_key_for_bronze logic by faking input_key
        out_b2, out_k2 = output_key_for_bronze(stem + '.txt', dataset, bronze_root.replace("s3://", ""))
        out_uri = s3_path(out_b2, out_k2)
        out_b_chk, out_k_chk = _parse_s3_uri(out_uri)
        if s3_exists(out_b_chk, out_k_chk) and not overwrite:
            continue
        backlog_tasks.append((good_uri, out_uri, overwrite, duckdb_threads, tmp_dir, proceed_on_errors, log_queue, dataset, bronze_root, False))
    if staged_count:
        logger.info(f"Backlog: {len(backlog_tasks)}/{staged_count} staged good files need conversion for {dataset}")

    # Build task list for raw inputs
    tasks: List[Tuple[str, str, bool, int, Optional[str], bool, Optional[mp.Queue], str, str, bool]] = []
    import hashlib
    planned = 0
    for key in list_s3_txt_files(in_bucket, in_prefix):
        # Deterministic sharding by filename to split work across multiple invocations
        if shard_count and shard_count > 1:
            h = int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)
            if (h % shard_count) != shard_index:
                continue
        in_uri = s3_path(in_bucket, key)
        out_bucket, out_key = output_key_for_bronze(key, dataset, bronze_root.replace("s3://", ""))
        out_uri = s3_path(out_bucket, out_key)
        # Early skip
        check_uri = out_uri
        exists = s3_exists(out_bucket, out_key)
        logger.info(f"Idempotency check → {check_uri} exists={exists} overwrite={overwrite}")
        if exists and not overwrite:
            logger.info(f"Skip (exists): {out_uri}")
            continue
        logger.info(f"Plan: {in_uri} → {out_uri}")
        tasks.append((in_uri, out_uri, overwrite, duckdb_threads, tmp_dir, proceed_on_errors, log_queue, dataset, bronze_root, split_rejects))
        if limit and len(tasks) >= limit:
            break
        planned += 1

    if dry_run:
        return {"dataset": dataset, "planned": len(backlog_tasks) + len(tasks), "converted": 0, "skipped": 0, "errors": 0}

    if not backlog_tasks and not tasks:
        return {"dataset": dataset, "planned": 0, "converted": 0, "skipped": 0, "errors": 0}

    processed = 0
    skipped = 0
    errors = 0
    # First process backlog (convert existing staged goods) if any
    if backlog_tasks:
        logger.info(f"Converting staged backlog ({len(backlog_tasks)}) before processing new splits…")
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=workers or 1) as ex:
            for output_uri, success, msg in ex.map(_convert_worker, backlog_tasks):
                if success:
                    if msg == "converted":
                        logger.info(f"✓ {output_uri}")
                        processed += 1

                    elif msg == "skipped_exists":
                        logger.info(f"↷ Skipped (exists): {output_uri}")
                        skipped += 1
                else:
                    logger.error(f"✗ {output_uri} -> {msg}")
                    errors += 1

    # Now process new tasks
    if workers and workers > 1:
        import concurrent.futures as cf
        logger.info(f"Starting parallel conversion with {workers} workers…")
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            for output_uri, success, msg in ex.map(_convert_worker, tasks):
                if success:
                    if msg == "converted":
                        logger.info(f"✓ {output_uri}")
                        processed += 1

                    elif msg == "skipped_exists":
                        logger.info(f"↷ Skipped (exists): {output_uri}")
                        skipped += 1
                else:
                    logger.error(f"✗ {output_uri} -> {msg}")
                    errors += 1
    else:
        for t in tasks:
            output_uri, success, msg = _convert_worker(t)
            if success:
                if msg == "converted":
                    logger.info(f"✓ {output_uri}")
                    processed += 1

                elif msg == "skipped_exists":
                    logger.info(f"↷ Skipped (exists): {output_uri}")
                    skipped += 1
            else:
                logger.error(f"✗ {output_uri} -> {msg}")
                errors += 1

    return {"dataset": dataset, "planned": len(backlog_tasks) + len(tasks), "converted": processed, "skipped": skipped, "errors": errors}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert raw Medical/Pharmacy TXT/CSV to Parquet bronze on S3 (one file at a time)")
    parser.add_argument("--dataset", choices=["medical", "pharmacy", "both"], default="both", help="Dataset to process")
    parser.add_argument("--medical-input", default="s3://pgxdatalake/Medical/", help="S3 input prefix for Medical")
    parser.add_argument("--pharmacy-input", default="s3://pgxdatalake/Pharmacy/", help="S3 input prefix for Pharmacy")
    parser.add_argument("--bronze-root", default="s3://pgxdatalake/bronze/", help="S3 root for bronze outputs")
    parser.add_argument("--limit", type=int, default=0, help="Max files per dataset (0 = no limit)")
    parser.add_argument("--dry-run", action="store_true", help="List planned conversions only")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Parquet outputs (default: skip if exists)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes")
    parser.add_argument("--duckdb-threads", type=int, default=1, help="DuckDB threads per worker process")
    parser.add_argument("--tmp-dir", help="Temporary directory for DuckDB")
    parser.add_argument("--proceed-on-errors", action="store_true", help="Proceed with conversion even if some rows are malformed (IGNORE_ERRORS)")
    parser.add_argument("--split-rejects", action="store_true", help="Pre-validate and split into good and rejects before converting good rows")
    parser.add_argument("--shard-count", type=int, default=0, help="Optional total number of shards for deterministic filename sharding")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index [0..shard-count-1] to process in this run")
    parser.add_argument("--aggregate-root", default="s3://pgxdatalake/pgx_pipeline/", help="S3 root for aggregated run summaries (for BI)")

    args = parser.parse_args(argv)

    # Unified logging setup (consistent with apcd_clean.py)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    if setup_logging:
        logger, log_buffer = setup_logging("txt_to_parquet", args.dataset, run_id)
        log_level = getattr(logging, getattr(args, "log_level", "INFO").upper(), logging.INFO)
        logger.setLevel(log_level)
    else:
        logger = setup_logger()
        log_buffer = []

    # Ensure tmp_dir is valid before starting
    if args.tmp_dir:
        import os as _os
        if not _os.path.exists(args.tmp_dir):
            _os.makedirs(args.tmp_dir, exist_ok=True)
            logger.info(f"Created tmp_dir: {args.tmp_dir}")
        if not _os.access(args.tmp_dir, _os.W_OK):
            raise PermissionError(f"tmp_dir not writable: {args.tmp_dir}")
        logger.info(f"Using tmp_dir: {args.tmp_dir}")

    try:
        logger.info("🚀 Starting TXT→Parquet bronze conversion")
        logger.info(f"📁 Medical input: {args.medical_input}")
        logger.info(f"📁 Pharmacy input: {args.pharmacy_input}")
        logger.info(f"📁 Bronze root: {args.bronze_root}")
        logger.info(f"⚙️  Workers={args.workers}, DuckDB threads/worker={args.duckdb_threads}, Limit={args.limit or 'none'}, Overwrite={args.overwrite}, DryRun={args.dry_run}, SplitRejects={args.split_rejects}, ProceedOnErrors={args.proceed_on_errors}, Shard={args.shard_index}/{args.shard_count}")

        _run_started = time.time()
        run_summary = {
            "tx": "txtpq",
            "run_id": run_id,
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "datasets": [],
            "totals": {"planned": 0, "converted": 0, "skipped": 0, "errors": 0}
        }
        # PipelineState per run (optional)
        if PipelineState is not None:
            ps = PipelineState('txt_to_parquet', f"{args.dataset}_{run_id}", logger)
            ps.mark_step_completed('start', {"workers": args.workers, "split_rejects": args.split_rejects})
        else:
            ps = None
        # Setup parent-side multiprocessing logging (real-time flush)
        if setup_mp_logging is not None:
            parent_logger, parent_buffer, log_queue, listener = setup_mp_logging("txt_to_parquet", args.dataset, run_id)
            # Attach parent handlers to our main logger for unified output
            for h in list(parent_logger.handlers):
                logger.addHandler(h)
        else:
            log_queue = None
            listener = None

        if args.dataset in ("medical", "both"):
            result_med = process_dataset(
                dataset="medical",
                input_root=args.medical_input,
                bronze_root=args.bronze_root.replace("\\", "/"),
                limit=args.limit,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
                logger=logger,
                workers=args.workers,
                duckdb_threads=args.duckdb_threads,
                tmp_dir=args.tmp_dir,
                proceed_on_errors=args.proceed_on_errors,
                split_rejects=args.split_rejects,
                shard_count=args.shard_count,
                shard_index=args.shard_index,
                log_queue=log_queue,

            )
            run_summary["datasets"].append(result_med)
            for k in ("planned", "converted", "skipped", "errors"):
                run_summary["totals"][k] += result_med[k]
            if ps:
                ps.mark_step_completed('medical_convert', result_med)
        if args.dataset in ("pharmacy", "both"):
            result_ph = process_dataset(
                dataset="pharmacy",
                input_root=args.pharmacy_input,
                bronze_root=args.bronze_root.replace("\\", "/"),
                limit=args.limit,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
                logger=logger,
                workers=args.workers,
                duckdb_threads=args.duckdb_threads,
                tmp_dir=args.tmp_dir,
                proceed_on_errors=args.proceed_on_errors,
                split_rejects=args.split_rejects,
                shard_count=args.shard_count,
                shard_index=args.shard_index,
                log_queue=log_queue,

            )
            run_summary["datasets"].append(result_ph)
            for k in ("planned", "converted", "skipped", "errors"):
                run_summary["totals"][k] += result_ph[k]
            if ps:
                ps.mark_step_completed('pharmacy_convert', result_ph)

        # Finish timing and status
        _run_finished = time.time()
        run_summary["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(_run_finished))
        run_summary["duration_sec"] = round(_run_finished - _run_started, 3)
        if run_summary['totals']['errors'] > 0:
            run_summary["status"] = "error"
            run_summary["status_code"] = 1
        else:
            run_summary["status"] = "success"
            run_summary["status_code"] = 0

        logger.info(f"✅ Completed. Files processed: {run_summary['totals']['converted']}")
        if listener and stop_mp_logging is not None:
            stop_mp_logging(listener)

        # Persist logs to S3 if available
        if save_logs_to_s3 and setup_logging:
            try:
                save_logs_to_s3(log_buffer, "txt_to_parquet", args.dataset, run_id, "apcd_input_data", logger=logger)
            except Exception:
                pass
        # Aggregate summary to pgxdatalake
        try:
            agg_bucket, agg_key_root = args.aggregate_root.replace("s3://", "").split("/", 1)
        except ValueError:
            agg_bucket, agg_key_root = args.aggregate_root.replace("s3://", ""), ""
        agg_key = f"{agg_key_root.rstrip('/')}/txt_to_parquet/run_id={run_id}/summary.json" if agg_key_root else f"txt_to_parquet/run_id={run_id}/summary.json"
        try:
            boto3.client('s3').put_object(Bucket=agg_bucket, Key=agg_key, Body=json.dumps(run_summary, indent=2).encode('utf-8'), ContentType='application/json')
            logger.info(f"📊 Aggregated summary saved: s3://{agg_bucket}/{agg_key}")
        except Exception as e:
            logger.warning(f"⚠️ Could not save aggregated summary: {e}")
        if ps:
            ps.mark_pipeline_completed(run_summary["totals"]) 
        return 0
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        if save_logs_to_s3 and setup_logging:
            try:
                save_logs_to_s3(log_buffer, "txt_to_parquet", args.dataset, run_id, "apcd_input_data", logger=logger)
            except Exception:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())




