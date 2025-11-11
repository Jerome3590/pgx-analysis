#!/usr/bin/env python3
"""
7_update_codes.py

Apply JSON code mappings to normalize and correct ICD/CPT (and optionally drug_name) codes
across the gold datasets. Uses idempotent overwrites (S3 versioning recommended).

Behavior:
- For medical gold: updates ICD diagnosis columns and CPT columns in-place per partition.
- For pharmacy gold (optional): updates drug_name using mapping.

Inputs:
- --icd-map: JSON mapping (local path or s3://) of variant -> canonical ICD-10-CM
- --cpt-map: JSON mapping (local path or s3://) of variant -> canonical CPT
- --drug-map: JSON mapping (local path or s3://) of variant -> canonical drug name
- --years: optional comma-separated years filter (e.g., 2016,2017)
- --age-bands: optional comma-separated age bands filter (e.g., 0-12,13-24)
- --apply: actually write updates (default true). If omitted, dry-run prints sample preview.

Notes:
- Mapping application is idempotent; running again yields same result.
- ICD normalization extracts an ICD-10 token and removes dots (F11.20 -> F1120).
- CPT normalization strips spaces/dots and uppercases before mapping.
"""

import argparse
import json
import os
import sys
import math
import glob as _glob
from typing import Dict, List, Optional

# Add project root to path (helpers folder is at pgx-analysis level)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import duckdb
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import traceback
import platform
import time

# Import utility modules
from helpers_1997_13.logging_utils import log_cpu_context
from helpers_1997_13.duckdb_utils import (
    create_duckdb_conn,
    get_worker_temp_dir,
    calculate_memory_limit_per_worker,
    load_mapping_into_duckdb_chunked,
    load_mapping_from_file_into_duckdb,
    cleanup_old_duckdb_temp_dirs
)
from helpers_1997_13.pipeline_utils import (
    get_multiprocessing_context,
    persist_mappings_to_temp,
    load_mappings_from_temp,
    get_retry_attempts,
    get_timeout_seconds
)


GOLD_MEDICAL_GLOB = "s3://pgxdatalake/gold/medical/age_band=*/event_year=*/medical_data.parquet"
GOLD_PHARMACY_GLOB = "s3://pgxdatalake/gold/pharmacy/age_band=*/event_year=*/pharmacy_data.parquet"
DEFAULT_MAPPING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'claim_mappings')
DEFAULT_TARGET_MAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'target_mapping')


# Note: Worker temp directory, multiprocessing context, and mapping persistence
# functions are now imported from helpers_1997_13 utility modules


# Note: Mapping loading functions are now imported from helpers_1997_13.duckdb_utils


# Note: DuckDB connection and memory limit functions are now imported from helpers_1997_13.duckdb_utils


def _marker_path_for(filename: str, marker_suffix: str) -> str:
    """
    Return a sidecar checkpoint path for a given data file.
    - For s3://bucket/key, returns s3://bucket/key{marker_suffix}
    - For local paths, returns {filename}{marker_suffix}
    """
    if filename.startswith("s3://"):
        return filename + marker_suffix
    return filename + marker_suffix


def _marker_exists(filename: str, marker_suffix: str) -> bool:
    marker = _marker_path_for(filename, marker_suffix)
    try:
        if marker.startswith("s3://"):
            import boto3  # type: ignore
            from helpers_1997_13.s3_utils import s3_path_to_bucket_key
            bkt, key = s3_path_to_bucket_key(marker)
            s3 = boto3.client("s3")
            s3.head_object(Bucket=bkt, Key=key)
            return True
        return os.path.exists(marker)
    except Exception:
        return False


def _write_marker(filename: str, marker_suffix: str, payload: Dict[str, str]) -> None:
    import json as _json
    marker = _marker_path_for(filename, marker_suffix)
    body = _json.dumps(payload).encode("utf-8")
    if marker.startswith("s3://"):
        import boto3  # type: ignore
        from helpers_1997_13.s3_utils import s3_path_to_bucket_key
        bkt, key = s3_path_to_bucket_key(marker)
        s3 = boto3.client("s3")
        s3.put_object(Bucket=bkt, Key=key, Body=body, ContentType="application/json")
    else:
        with open(marker, "wb") as f:
            f.write(body)


def _staging_prefix_for(filename: str, staging_suffix: str) -> str:
    """Return a staging prefix to hold chunk files for a given data file."""
    # Ensure suffix ends with '/'
    if not staging_suffix.endswith('/'):
        staging_suffix = staging_suffix + '/'
    return filename + staging_suffix


def _chunk_path_for(filename: str, staging_suffix: str, chunk_index: int) -> str:
    prefix = _staging_prefix_for(filename, staging_suffix)
    return f"{prefix}chunk_{chunk_index:05d}.parquet"


def _get_max_chunks_per_batch() -> int:
    """
    Get the maximum number of chunks to merge per batch.
    Reads from PGX_MAX_CHUNKS_PER_BATCH env var, defaults to 5-8 range.
    """
    env_val = os.getenv('PGX_MAX_CHUNKS_PER_BATCH', '')
    if env_val and env_val.isdigit():
        return max(1, int(env_val))  # Ensure at least 1
    return 8  # Default: 8 (balanced between memory and I/O)


def _merge_chunks_incremental(con: duckdb.DuckDBPyConnection, staging_prefix: str, output_filename: str, max_chunks_per_batch: Optional[int] = None) -> None:
    """
    Merge staging chunks incrementally to avoid loading all chunks into memory at once.
    Uses UNION ALL in batches to reduce memory pressure.
    
    Args:
        con: DuckDB connection
        staging_prefix: S3 or local prefix for staging chunks
        output_filename: Final output file path
        max_chunks_per_batch: Maximum chunks per batch (if None, reads from env)
    """
    # Get batch size from parameter or environment
    if max_chunks_per_batch is None:
        max_chunks_per_batch = _get_max_chunks_per_batch()
    
    # List all chunk files
    chunk_files = []
    if staging_prefix.startswith("s3://"):
        import boto3
        from helpers_1997_13.s3_utils import s3_path_to_bucket_key
        bkt, key = s3_path_to_bucket_key(staging_prefix)
        if not key.endswith("/"):
            key = key + "/"
        s3 = boto3.client("s3")
        token = None
        while True:
            kwargs = {"Bucket": bkt, "Prefix": key}
            if token:
                kwargs["ContinuationToken"] = token
            resp = s3.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                if obj["Key"].endswith(".parquet") and "chunk_" in obj["Key"]:
                    chunk_files.append(f"s3://{bkt}/{obj['Key']}")
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
            else:
                break
        # Sort by chunk index
        chunk_files.sort(key=lambda x: int(x.split("chunk_")[1].split(".")[0]))
    else:
        # Local filesystem
        import glob
        pattern = staging_prefix.rstrip("/") + "/*.parquet"
        chunk_files = sorted(glob.glob(pattern), key=lambda x: int(x.split("chunk_")[1].split(".")[0]))
    
    if not chunk_files:
        return
    
    # Merge in batches to avoid loading all chunks at once
    if len(chunk_files) <= max_chunks_per_batch:
        # Small number of chunks - merge all at once
        chunk_list = "', '".join(chunk_files)
        con.sql(
            f"""
            COPY (
              SELECT * FROM read_parquet(['{chunk_list}'])
            ) TO '{output_filename}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE)
            """
        )
    else:
        # Large number of chunks - merge incrementally using temp files
        temp_files = []
        batch_num = 0
        for i in range(0, len(chunk_files), max_chunks_per_batch):
            batch = chunk_files[i:i + max_chunks_per_batch]
            batch_list = "', '".join(batch)
            temp_file = f"{staging_prefix}temp_batch_{batch_num:05d}.parquet"
            temp_files.append(temp_file)
            con.sql(
                f"""
                COPY (
                  SELECT * FROM read_parquet(['{batch_list}'])
                ) TO '{temp_file}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE)
                """
            )
            batch_num += 1
        
        # Merge temp batches into final file
        if len(temp_files) == 1:
            # Only one batch, just rename/move
            import shutil
            if output_filename.startswith("s3://"):
                # For S3, copy the temp file to final location
                import boto3
                from helpers_1997_13.s3_utils import s3_path_to_bucket_key
                src_bkt, src_key = s3_path_to_bucket_key(temp_files[0])
                dst_bkt, dst_key = s3_path_to_bucket_key(output_filename)
                s3 = boto3.client("s3")
                copy_source = {"Bucket": src_bkt, "Key": src_key}
                s3.copy_object(CopySource=copy_source, Bucket=dst_bkt, Key=dst_key)
                s3.delete_object(Bucket=src_bkt, Key=src_key)
            else:
                shutil.move(temp_files[0], output_filename)
        else:
            # Multiple batches - merge them
            temp_list = "', '".join(temp_files)
            con.sql(
                f"""
                COPY (
                  SELECT * FROM read_parquet(['{temp_list}'])
                ) TO '{output_filename}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE)
                """
            )
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    if temp_file.startswith("s3://"):
                        import boto3
                        from helpers_1997_13.s3_utils import s3_path_to_bucket_key
                        bkt, key = s3_path_to_bucket_key(temp_file)
                        s3 = boto3.client("s3")
                        s3.delete_object(Bucket=bkt, Key=key)
                    else:
                        os.remove(temp_file)
                except Exception:
                    pass


def _object_exists(path: str) -> bool:
    try:
        if path.startswith("s3://"):
            import boto3  # type: ignore
            from helpers_1997_13.s3_utils import s3_path_to_bucket_key
            bkt, key = s3_path_to_bucket_key(path)
            s3 = boto3.client("s3")
            s3.head_object(Bucket=bkt, Key=key)
            return True
        return os.path.exists(path)
    except Exception:
        return False


def _upload_to_s3_with_retry(local_path: str, s3_path: str, max_retries: int = 3) -> None:
    """
    Upload a local file to S3 with retry logic and multipart support.
    Uses boto3's managed transfer for reliability and performance.
    """
    import boto3
    from botocore.exceptions import ClientError
    from helpers_1997_13.s3_utils import s3_path_to_bucket_key
    
    bucket, key = s3_path_to_bucket_key(s3_path)
    s3 = boto3.client("s3")
    
    # Configure transfer for large files
    # Tune max_concurrency to avoid saturating network I/O when many workers upload simultaneously
    # Scale down based on total workers to prevent network saturation
    base_concurrency = int(os.getenv('PGX_MAX_UPLOAD_CONCURRENCY', '10'))
    
    # Get total workers to scale concurrency appropriately
    total_workers_env = os.getenv('PGX_TOTAL_WORKERS', '')
    if total_workers_env and total_workers_env.isdigit():
        total_workers = int(total_workers_env)
        # Scale down concurrency for high worker counts:
        # ≥24 workers: cap at 4
        # ≥12 workers: cap at 6
        # <12 workers: use base (up to 10)
        if total_workers >= 24:
            max_upload_concurrency = min(4, base_concurrency)
        elif total_workers >= 12:
            max_upload_concurrency = min(6, base_concurrency)
        else:
            max_upload_concurrency = min(10, base_concurrency)
    else:
        # No total workers info, use base with cap
        max_upload_concurrency = min(10, base_concurrency)
    
    from boto3.s3.transfer import TransferConfig
    config = TransferConfig(
        multipart_threshold=100 * 1024 * 1024,  # 100MB - use multipart for files > 100MB
        max_concurrency=max_upload_concurrency,  # Parallel upload threads (tunable via env)
        multipart_chunksize=25 * 1024 * 1024,  # 25MB chunks
        use_threads=True,
        max_io_queue=1000
    )
    
    for attempt in range(max_retries):
        try:
            s3.upload_file(local_path, bucket, key, Config=config)
            return
        except ClientError as e:
            if attempt == max_retries - 1:
                raise
            # Exponential backoff
            wait_time = 2 ** attempt
            time.sleep(wait_time)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


def _get_local_staging_dir() -> str:
    """
    Get the local staging directory from environment or use default.
    Prefer fast local NVMe storage over network storage.
    """
    staging_dir = os.getenv('PGX_LOCAL_STAGING_DIR', '/mnt/nvme/pgx_staging')
    os.makedirs(staging_dir, exist_ok=True)
    return staging_dir


def _local_staging_path(s3_path: str) -> str:
    """
    Generate a local staging path for an S3 file.
    Uses a hash to avoid path length issues and collisions.
    """
    import hashlib
    staging_dir = _get_local_staging_dir()
    # Use hash of full S3 path to ensure uniqueness
    path_hash = hashlib.md5(s3_path.encode()).hexdigest()[:16]
    filename = os.path.basename(s3_path)
    return os.path.join(staging_dir, f"{path_hash}_{filename}")


def _cleanup_orphaned_staging_files(max_age_hours: int = 24) -> int:
    """
    Clean up orphaned staging files from previous failed runs.
    Removes files older than max_age_hours that are likely from crashed processes.
    
    Returns:
        Number of files cleaned up
    """
    staging_dir = _get_local_staging_dir()
    if not os.path.exists(staging_dir):
        return 0
    
    import time
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    cleaned_count = 0
    
    try:
        for filename in os.listdir(staging_dir):
            file_path = os.path.join(staging_dir, filename)
            if os.path.isfile(file_path) and filename.endswith('.parquet'):
                # Check file age
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                    except Exception:
                        pass
    except Exception:
        pass
    
    if cleaned_count > 0:
        print(f"🧹 Cleaned up {cleaned_count} orphaned staging files older than {max_age_hours} hours")
    
    return cleaned_count


def _get_object_size_bytes(path: str) -> int:
    try:
        if path.startswith("s3://"):
            import boto3  # type: ignore
            from helpers_1997_13.s3_utils import s3_path_to_bucket_key
            bkt, key = s3_path_to_bucket_key(path)
            s3 = boto3.client("s3")
            head = s3.head_object(Bucket=bkt, Key=key)
            return int(head.get("ContentLength") or 0)
        return os.path.getsize(path)
    except Exception:
        return 0


def _list_staging_chunks(prefix: str) -> int:
    """
    Return count of chunk parquet objects under a staging prefix.
    """
    if prefix.startswith("s3://"):
        try:
            import boto3  # type: ignore
            from helpers_1997_13.s3_utils import s3_path_to_bucket_key
            bkt, key = s3_path_to_bucket_key(prefix)
            if not key.endswith("/"):
                key = key + "/"
            s3 = boto3.client("s3")
            token = None
            count = 0
            while True:
                kwargs = {"Bucket": bkt, "Prefix": key}
                if token:
                    kwargs["ContinuationToken"] = token
                resp = s3.list_objects_v2(**kwargs)
                for obj in resp.get("Contents", []):
                    if obj["Key"].endswith(".parquet"):
                        count += 1
                if resp.get("IsTruncated"):
                    token = resp.get("NextContinuationToken")
                else:
                    break
            return count
        except Exception:
            return 0
    # local
    try:
        if not prefix.endswith("/"):
            prefix = prefix + "/"
        return len(_glob.glob(prefix + "*.parquet"))
    except Exception:
        return 0


def _enumerate_partitions(src_glob: str):
    """
    Enumerate partitions from S3 glob pattern.
    Uses streaming fetch to avoid loading all paths into memory at once.
    """
    conn = create_duckdb_conn(threads=1)
    try:
        result = conn.sql(
            """
            WITH f AS (
              SELECT file AS filename FROM glob(?)
            ), p AS (
              SELECT
                regexp_extract(filename, 'age_band=([^/]+)', 1) AS age_band,
                CAST(regexp_extract(filename, 'event_year=([0-9]{4})', 1) AS INTEGER) AS event_year,
                filename
              FROM f
            )
            SELECT * FROM p
            """,
            params=[src_glob],
        )
        
        # Use fetch_arrow_table() for streaming (more memory-efficient than .df())
        # Falls back to .df() if arrow is not available
        try:
            import pyarrow as pa
            arrow_table = result.fetch_arrow_table()
            parts = arrow_table.to_pandas()
        except (ImportError, Exception):
            # Fallback to .df() if arrow not available or fails
            parts = result.df()
        
        return parts
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _get_parquet_columns(con: duckdb.DuckDBPyConnection, filename: str) -> List[str]:
    """
    Get all column names from a Parquet file.
    Returns list of column names.
    """
    try:
        # Use DESCRIBE to get column names without reading data
        result = con.sql(f"DESCRIBE SELECT * FROM read_parquet('{filename}') LIMIT 0")
        columns = [row[0] for row in result.fetchall()]
        return columns
    except Exception:
        # Fallback: try to read schema from parquet_metadata
        try:
            result = con.sql(f"SELECT column_name FROM parquet_schema('{filename}')")
            columns = [row[0] for row in result.fetchall()]
            return columns
        except Exception:
            # Last resort: read one row to infer schema (expensive but works)
            try:
                df = con.sql(f"SELECT * FROM read_parquet('{filename}') LIMIT 0").df()
                return list(df.columns)
            except Exception:
                return []


def _build_optimized_read_parquet_query(filename: str, columns_to_read: List[str], all_columns: List[str], limit: int, offset: int) -> str:
    """
    Build an optimized read_parquet query that only reads columns being modified.
    
    Args:
        filename: Path to Parquet file
        columns_to_read: List of column names to read (columns being modified)
        all_columns: List of all column names in the file
        limit: LIMIT clause value
        offset: OFFSET clause value
    
    Returns:
        SQL query string
    """
    # If we're modifying all columns or most columns, just read all
    if len(columns_to_read) >= len(all_columns) * 0.8:
        return f"read_parquet('{filename}')"
    
    # Otherwise, use columns parameter to only read what we need
    # Note: We still need all columns for output, so we'll use a subquery approach
    # But actually, DuckDB's read_parquet with columns parameter is efficient
    # and we can use SELECT * REPLACE on the result
    columns_str = ", ".join([f"'{col}'" for col in columns_to_read])
    return f"read_parquet('{filename}', columns=[{columns_str}])"


def _compute_rowgroup_chunks(con: duckdb.DuckDBPyConnection, filename: str, chunk_rows: int):
    """
    Plan chunk offsets and limits aligned to parquet row groups.
    Returns list of (offset, limit) tuples and total_rows.
    Falls back to uniform LIMIT/OFFSET if metadata is unavailable or too large.
    
    For files with >50k row groups, uses streaming iteration to avoid loading
    all metadata into memory at once.
    """
    if not chunk_rows or chunk_rows <= 0:
        try:
            total_rows = con.sql(f"SELECT COALESCE(SUM(num_rows), 0) FROM parquet_metadata('{filename}')").fetchone()[0]
        except Exception:
            total_rows = con.sql(f"SELECT COUNT(*) FROM read_parquet('{filename}')").fetchone()[0]
        if not total_rows:
            return [], 0
        return [(0, total_rows)], total_rows

    try:
        # First check row group count to decide on approach
        rg_count_result = con.sql(f"SELECT COUNT(*) FROM parquet_metadata('{filename}')").fetchone()
        rg_count = rg_count_result[0] if rg_count_result else 0
        
        # For files with >50k row groups, use streaming approach
        if rg_count > 50000:
            # Use streaming: iterate through row groups one at a time
            chunks = []
            total_rows = 0
            current_offset = 0
            current_size = 0
            
            # Stream row groups using a cursor-like approach
            result = con.sql(f"SELECT num_rows FROM parquet_metadata('{filename}')")
            for row in result:
                size = row[0]
                total_rows += size
                if current_size == 0:
                    # start new chunk
                    current_offset = total_rows - size
                    current_size = size
                else:
                    if current_size + size <= chunk_rows:
                        current_size += size
                    else:
                        chunks.append((current_offset, current_size))
                        current_offset = total_rows - size
                        current_size = size
            
            if current_size > 0:
                chunks.append((current_offset, current_size))
            
            if not chunks:
                raise RuntimeError("no row groups")
            return chunks, total_rows
        else:
            # For smaller files, use the original approach (faster for <50k groups)
            rg_sizes = [r[0] for r in con.sql(f"SELECT num_rows FROM parquet_metadata('{filename}')").fetchall()]
            if not rg_sizes:
                raise RuntimeError("no row groups")
            chunks = []
            total_rows = 0
            current_offset = 0
            current_size = 0
            for size in rg_sizes:
                total_rows += size
                if current_size == 0:
                    # start new chunk
                    current_offset = total_rows - size
                    current_size = size
                else:
                    if current_size + size <= chunk_rows:
                        current_size += size
                    else:
                        chunks.append((current_offset, current_size))
                        current_offset = total_rows - size
                        current_size = size
            if current_size > 0:
                chunks.append((current_offset, current_size))
            return chunks, total_rows
    except Exception:
        # Fallback: uniform chunks via COUNT
        total_rows = con.sql(f"SELECT COUNT(*) FROM read_parquet('{filename}')").fetchone()[0]
        if not total_rows:
            return [], 0
        num_chunks = max(1, math.ceil(total_rows / float(chunk_rows)))
        chunks = []
        for i in range(num_chunks):
            offset = i * chunk_rows
            limit = min(chunk_rows, total_rows - offset)
            chunks.append((offset, limit))
        return chunks, total_rows


def _final_chunk_path_for(filename: str, chunk_index: int) -> str:
    """
    Derive a deterministic final chunk filename next to the original file:
    .../medical_data.parquet -> .../medical_data-00000.parquet
    """
    if filename.endswith(".parquet"):
        base = filename[:-8]
    else:
        base = filename
    return f"{base}-{chunk_index:05d}.parquet"


def load_mapping(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    try:
        if path.startswith("s3://"):
            import boto3
            from botocore.exceptions import ClientError  # type: ignore
            from helpers_1997_13.s3_utils import s3_path_to_bucket_key

            s3 = boto3.client("s3")
            bucket, key = s3_path_to_bucket_key(path)
            # Only warn if object actually exists but cannot be read
            try:
                s3.head_object(Bucket=bucket, Key=key)
            except ClientError as ce:
                if ce.response.get('Error', {}).get('Code') in ("404", "NoSuchKey"):
                    return {}
                raise
            obj = s3.get_object(Bucket=bucket, Key=key)
            return json.loads(obj["Body"].read().decode("utf-8"))
        # Local path
        if not os.path.isfile(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load mapping {path}: {e}")
        return {}


def normalize_icd_sql(col: str) -> str:
    # Extract token and remove dots to prefer undotted canonical (e.g., F11.20 -> F1120)
    return (
        f"(CASE WHEN {col} IS NULL OR TRIM(CAST({col} AS VARCHAR))='' THEN NULL "
        f"ELSE REPLACE(regexp_extract(UPPER(CAST({col} AS VARCHAR)), '([A-Z][0-9]{{2}}(\\.?[A-Z0-9]{{1,4}})?)', 1), '.', '') END)"
    )


def normalize_cpt_sql(col: str) -> str:
    # Uppercase, strip spaces and dots
    return f"REPLACE(REPLACE(UPPER(CAST({col} AS VARCHAR)),'.',''),' ','')"


# Note: Retry and timeout functions are now imported from helpers_1997_13.pipeline_utils


# Note: CPU context logging is now imported from helpers_1997_13.logging_utils


def _process_medical_partition(filename: str, icd_map: Dict[str, str], cpt_map: Dict[str, str], icd_target_map: Dict[str, str], apply: bool, threads: int, log_cpu: bool = False, log_s3: bool = False, resume: bool = False, marker_suffix: str = ".codes_updated.ok", chunked: bool = False, chunk_rows: int = 0, staging_suffix: str = ".codes_updated.staging/", total_workers: Optional[int] = None, use_temp_db: bool = False, mapping_temp_dir: Optional[str] = None) -> str:
    import sys
    import traceback as tb_module
    
    try:
        worker_start_time = time.time()
        print(f"[medical-worker] 🚀 START processing: {os.path.basename(filename)}")
        sys.stdout.flush()  # Ensure worker output is immediately visible
        
        # Initialize metrics tracking
        metrics = {
            "records_processed": 0,
            "chunks_processed": 0,
            "file_size_bytes": 0,
            "duration_seconds": 0.0,
            "mapping_load_time": 0.0,
            "processing_time": 0.0,
            "upload_time": 0.0
        }
        
        # Load mappings from temp files if provided (reduces memory duplication in spawn mode)
        if mapping_temp_dir:
            print(f"[medical-worker] 📥 Loading mappings from temp directory...")
            sys.stdout.flush()
            try:
                icd_map, cpt_map, icd_target_map, _ = load_mappings_from_temp(mapping_temp_dir)
                print(f"[medical-worker] ✅ Mappings loaded from temp directory")
                sys.stdout.flush()
            except Exception as e:
                print(f"[medical-worker] ❌ ERROR loading mappings: {e}", file=sys.stderr)
                print(tb_module.format_exc(), file=sys.stderr)
                sys.stderr.flush()
                raise
        
        # Enforce threads=1 to avoid oversubscription (process-level parallelism is sufficient)
        if threads > 1:
            print(f"[medical-worker] ⚠️ Warning: threads={threads} > 1 may cause CPU oversubscription. Using 1 thread per worker.")
            sys.stdout.flush()
            threads = 1
        
        print(f"[medical-worker] 🔌 Creating DuckDB connection...")
        sys.stdout.flush()
        try:
            con = create_duckdb_conn(threads=threads, total_workers=total_workers, use_temp_db=use_temp_db)
            print(f"[medical-worker] ✅ DuckDB connection established and configured")
            sys.stdout.flush()
        except Exception as e:
            print(f"[medical-worker] ❌ ERROR creating DuckDB connection: {e}", file=sys.stderr)
            print(tb_module.format_exc(), file=sys.stderr)
            sys.stderr.flush()
            raise
            
    except Exception as e:
        # If we can't even initialize, something is very wrong - try to log to stderr
        print(f"[medical-worker] FATAL: Worker failed to initialize: {e}", file=sys.stderr)
        print(tb_module.format_exc(), file=sys.stderr)
        sys.stderr.flush()
        raise
    
    try:
        # Checkpoint short-circuit
        if resume and _marker_exists(filename, marker_suffix):
            print(f"[medical-worker] ↩ Skipped (checkpoint exists): {os.path.basename(filename)}")
            return f"↩ Skipped (checkpoint) {filename}"
        if log_cpu:
            log_cpu_context(prefix="[medical-worker]", threads=threads)
        s3_info_pre = None
        if log_s3 and filename.startswith("s3://"):
            try:
                import boto3  # type: ignore
                from helpers_1997_13.s3_utils import s3_path_to_bucket_key
                bkt, key = s3_path_to_bucket_key(filename)
                s3 = boto3.client('s3')
                pre = s3.head_object(Bucket=bkt, Key=key)
                s3_info_pre = {
                    'size_bytes': pre.get('ContentLength'),
                    'etag': pre.get('ETag'),
                }
            except Exception:
                s3_info_pre = None
        # Register mapping tables using chunked streaming to avoid materializing full lists
        print(f"[medical-worker] 📋 Loading mapping dictionaries into DuckDB...")
        mapping_start = time.time()
        # Note: With multiprocessing 'spawn' context, each worker gets a copy of mappings.
        # For very large mappings, consider using shared memory or loading from S3 per worker.
        if icd_map:
            print(f"[medical-worker]   → Loading ICD map ({len(icd_map)} entries)...")
            load_mapping_into_duckdb_chunked(con, icd_map, "icd_map", chunk_size=10000)
        if icd_target_map:
            print(f"[medical-worker]   → Loading ICD target map ({len(icd_target_map)} entries)...")
            # Transform keys: uppercase, remove dots and spaces
            def transform_icd_target_key(k):
                return k.upper().replace('.', '').replace(' ', '')
            load_mapping_into_duckdb_chunked(con, icd_target_map, "icd_target_map", transform_key=transform_icd_target_key, chunk_size=10000)
        if cpt_map:
            print(f"[medical-worker]   → Loading CPT map ({len(cpt_map)} entries)...")
            load_mapping_into_duckdb_chunked(con, cpt_map, "cpt_map", chunk_size=10000)
        mapping_elapsed = time.time() - mapping_start
        metrics["mapping_load_time"] = mapping_elapsed
        print(f"[medical-worker] ✅ Mappings loaded in {mapping_elapsed:.2f}s")

        icd_cols = [
            "primary_icd_diagnosis_code",
            "two_icd_diagnosis_code",
            "three_icd_diagnosis_code",
            "four_icd_diagnosis_code",
            "five_icd_diagnosis_code",
            "six_icd_diagnosis_code",
            "seven_icd_diagnosis_code",
            "eight_icd_diagnosis_code",
            "nine_icd_diagnosis_code",
            "ten_icd_diagnosis_code",
        ]
        cpt_cols = ["procedure_code", "cpt_mod_1_code", "cpt_mod_2_code"]

        # Build explicit LEFT JOIN query instead of correlated subqueries
        # This is much more efficient and avoids 4GB buffer allocation issues
        # Strategy: Create normalized CTE, then use explicit LEFT JOINs with mapping tables
        def build_join_query(read_expr: str, limit: Optional[int] = None, offset: Optional[int] = None) -> str:
            """Build query with explicit LEFT JOINs for efficient mapping"""
            limit_clause = f" LIMIT {limit} OFFSET {offset}" if limit and offset is not None else ""
            
            # Get all columns
            all_cols_df = con.sql(f"DESCRIBE SELECT * FROM {read_expr} LIMIT 1").df()
            all_cols = all_cols_df['column_name'].tolist()
            
            # Build normalized CTE with all computed normalized values
            # This allows us to JOIN efficiently on pre-computed values
            norm_computations = []
            for col in icd_cols:
                raw_flat = f"REPLACE(REPLACE(UPPER(CAST({col} AS VARCHAR)),'.',''),' ','') AS raw_flat_{col}"
                normalized = f"REPLACE(regexp_extract(UPPER(CAST({col} AS VARCHAR)), '([A-Z][0-9]{{2}}(\\.?[A-Z0-9]{{1,4}})?)', 1), '.', '') AS norm_{col}"
                norm_computations.extend([raw_flat, normalized])
            
            for col in cpt_cols:
                normalized = f"REPLACE(REPLACE(UPPER(CAST({col} AS VARCHAR)),'.',''),' ','') AS norm_{col}"
                norm_computations.append(normalized)
            
            # Build JOIN clauses - one per column to mapping tables
            # We use computed expressions in ON clause which DuckDB can optimize
            joins = []
            join_aliases = {}  # Track aliases to avoid duplicates
            
            # For each ICD column, create JOINs to target_map and icd_map
            for col in icd_cols:
                if icd_target_map:
                    alias = f"tgt_{col}"
                    join_aliases[alias] = True
                    raw_flat_expr = f"REPLACE(REPLACE(UPPER(CAST(norm.{col} AS VARCHAR)),'.',''),' ','')"
                    joins.append(f"LEFT JOIN icd_target_map AS {alias} ON {alias}.variant = {raw_flat_expr}")
                
                if icd_map:
                    alias = f"icd_{col}"
                    join_aliases[alias] = True
                    norm_expr = f"REPLACE(regexp_extract(UPPER(CAST(norm.{col} AS VARCHAR)), '([A-Z][0-9]{{2}}(\\.?[A-Z0-9]{{1,4}})?)', 1), '.', '')"
                    joins.append(f"LEFT JOIN icd_map AS {alias} ON {alias}.variant = {norm_expr}")
            
            # For each CPT column, create JOIN to cpt_map
            for col in cpt_cols:
                if cpt_map:
                    alias = f"cpt_{col}"
                    join_aliases[alias] = True
                    norm_expr = f"REPLACE(REPLACE(UPPER(CAST(norm.{col} AS VARCHAR)),'.',''),' ','')"
                    joins.append(f"LEFT JOIN cpt_map AS {alias} ON {alias}.variant = {norm_expr}")
            
            # Build final SELECT with COALESCE using joined values
            final_selects = []
            for col in icd_cols:
                tgt_alias = f"tgt_{col}"
                icd_alias = f"icd_{col}"
                norm_expr = f"REPLACE(regexp_extract(UPPER(CAST(norm.{col} AS VARCHAR)), '([A-Z][0-9]{{2}}(\\.?[A-Z0-9]{{1,4}})?)', 1), '.', '')"
                
                if icd_target_map and icd_map:
                    mapped = f"COALESCE({tgt_alias}.canonical, {icd_alias}.canonical, {norm_expr}, CAST(norm.{col} AS VARCHAR))"
                elif icd_target_map:
                    mapped = f"COALESCE({tgt_alias}.canonical, {norm_expr}, CAST(norm.{col} AS VARCHAR))"
                elif icd_map:
                    mapped = f"COALESCE({icd_alias}.canonical, {norm_expr}, CAST(norm.{col} AS VARCHAR))"
                else:
                    mapped = f"COALESCE({norm_expr}, CAST(norm.{col} AS VARCHAR))"
                
                final_selects.append(f"CASE WHEN norm.{col} IS NULL OR TRIM(CAST(norm.{col} AS VARCHAR))='' THEN NULL ELSE {mapped} END AS {col}")
            
            for col in cpt_cols:
                cpt_alias = f"cpt_{col}"
                norm_expr = f"REPLACE(REPLACE(UPPER(CAST(norm.{col} AS VARCHAR)),'.',''),' ','')"
                
                if cpt_map:
                    mapped = f"COALESCE({cpt_alias}.canonical, {norm_expr}, CAST(norm.{col} AS VARCHAR))"
                else:
                    mapped = f"COALESCE({norm_expr}, CAST(norm.{col} AS VARCHAR))"
                
                final_selects.append(f"CASE WHEN norm.{col} IS NULL OR TRIM(CAST(norm.{col} AS VARCHAR))='' THEN NULL ELSE {mapped} END AS {col}")
            
            # Add all other columns
            other_cols = [col for col in all_cols if col not in icd_cols and col not in cpt_cols]
            for col in other_cols:
                final_selects.append(f"norm.{col}")
            
            # Build complete query with normalized CTE and explicit JOINs
            query = f"""
            WITH norm AS (
                SELECT 
                    *,
                    {', '.join(norm_computations)}
                FROM {read_expr}{limit_clause}
            )
            SELECT
                {', '.join(final_selects)}
            FROM norm
            {' '.join(joins)}
            """
            return query
        
        def get_read_parquet_expr(cols_to_read: List[str]) -> str:
            # Always read all columns
            return f"read_parquet('{filename}')"
        
        # Build change conditions for pre-check (simplified - just check if normalization changes value)
        change_conds = []
        for col in icd_cols:
            norm_expr = normalize_icd_sql(col)
            change_conds.append(f"({col} IS DISTINCT FROM {norm_expr})")
        for col in cpt_cols:
            norm_expr = normalize_cpt_sql(col)
            change_conds.append(f"({col} IS DISTINCT FROM {norm_expr})")

        # Pre-check: skip if no changes (skip global pre-scan when chunked mode is enabled)
        # Use sampling to avoid loading entire file into memory
        skip_sample_check = os.getenv("PGX_SKIP_SAMPLE_CHECK", "0") == "1"
        if apply and change_conds and not (chunked and chunk_rows and chunk_rows > 0) and not skip_sample_check:
            print(f"[medical-worker] 🔍 Starting pre-check (sampling for changes)...")
            precheck_start = time.time()
            or_cond = " OR ".join(change_conds)
            # Sample first 100k rows to check for changes (much faster and lower memory)
            # If changes found in sample, proceed; if not, do full scan
            sample_size = 100000
            try:
                needs_update = con.sql(
                    f"SELECT EXISTS(SELECT 1 FROM read_parquet('{filename}') WHERE {or_cond} LIMIT {sample_size})"
                ).fetchone()[0]
                # If no changes in sample, do a lightweight count check on full dataset
                if not needs_update:
                    # Check if file is small enough to scan fully, otherwise skip pre-check
                    file_size_mb = _get_object_size_bytes(filename) / (1024 * 1024)
                    if file_size_mb < 100:  # Only full scan if file < 100MB
                        needs_update = con.sql(
                            f"SELECT EXISTS(SELECT 1 FROM read_parquet('{filename}') WHERE {or_cond})"
                        ).fetchone()[0]
                    # For large files, assume changes exist if sample found none (conservative)
            except Exception:
                # If sampling fails, fall back to full scan (but this is rare)
                needs_update = con.sql(
                    f"SELECT EXISTS(SELECT 1 FROM read_parquet('{filename}') WHERE {or_cond})"
                ).fetchone()[0]
            if not needs_update:
                if resume:
                    try:
                        _write_marker(filename, marker_suffix, {
                            "status": "skipped_no_changes",
                            "ts": str(int(time.time()))
                        })
                    except Exception:
                        pass
                return f"↩ Skipped (no changes) {filename}"
        if apply and chunked:
            print(f"[medical-worker] 📦 Starting chunked processing mode...")
            chunked_start = time.time()
            # Use local staging for better performance and reliability
            use_local_staging = os.getenv("PGX_USE_LOCAL_STAGING", "1") == "1"
            
            # Row-group aligned chunk planning
            print(f"[medical-worker]   → Computing chunk plan...")
            # Derive chunk_rows from target size if not provided
            # Reduced default chunk size for faster processing (since we're only replacing column values)
            effective_chunk_rows = chunk_rows
            if not effective_chunk_rows or effective_chunk_rows <= 0:
                total_rows_try = 0
                try:
                    total_rows_try = con.sql(f"SELECT COALESCE(SUM(num_rows), 0) FROM parquet_metadata('{filename}')").fetchone()[0]
                except Exception:
                    total_rows_try = con.sql(f"SELECT COUNT(*) FROM read_parquet('{filename}')").fetchone()[0]
                size_b = _get_object_size_bytes(filename)
                avg_bpr = (size_b / max(1, total_rows_try)) if total_rows_try else 1024.0
                target_mb = float(os.getenv("PGX_TARGET_FILE_SIZE_MB", "0"))
                # Reduced default to 256 MB for faster processing (was 512 MB)
                if target_mb <= 0:
                    target_mb = 256.0
                target_bytes = target_mb * 1024 * 1024
                effective_chunk_rows = max(50000, int(target_bytes / max(1.0, avg_bpr)))  # Min 50k rows (was 10k)
            chunks, total_rows = _compute_rowgroup_chunks(con, filename, effective_chunk_rows)
            if chunks:
                total_chunks = len(chunks)
                metrics["records_processed"] = total_rows
                metrics["chunks_processed"] = total_chunks
                metrics["file_size_bytes"] = _get_object_size_bytes(filename)
                print(f"[medical-worker]   → Chunk plan: {total_chunks} chunks, {total_rows:,} total rows")
                print(f"[medical-worker] 🚀 Starting chunk processing ({total_chunks} chunks)...")
                chunk_processing_start = time.time()
                chunks_completed = 0
                for i, (offset, limit) in enumerate(chunks):
                    # Decide destination: staging (for merge) or final chunk (no-merge)
                    no_merge = os.getenv("PGX_NO_MERGE", "0") == "1"
                    s3_dest_path = _final_chunk_path_for(filename, i) if no_merge else _chunk_path_for(filename, staging_suffix, i)
                    
                    if resume and _object_exists(s3_dest_path):
                        print(f"[medical-worker] ↩ Skipped chunk {i+1}/{total_chunks} (exists) {filename}")
                        chunks_completed += 1
                        continue
                    
                    # Write to local staging first if enabled
                    if use_local_staging and s3_dest_path.startswith("s3://"):
                        local_dest = _local_staging_path(s3_dest_path)
                        chunk_start = time.time()
                        print(f"[medical-worker]   [Chunk {i+1}/{total_chunks}] ▶ Writing to local staging (rows={limit:,}, offset={offset:,})...")
                        
                        try:
                            # Write to local disk (fast) - using explicit LEFT JOINs
                            read_expr = get_read_parquet_expr([])
                            query = build_join_query(read_expr, limit=limit, offset=offset)
                            con.sql(f"COPY ({query}) TO '{local_dest}' (FORMAT PARQUET)")
                            write_elapsed = time.time() - chunk_start
                            print(f"[medical-worker]   [Chunk {i+1}/{total_chunks}] ✅ Local write complete ({write_elapsed:.2f}s)")
                            
                            # Upload to S3 (with retry)
                            upload_start = time.time()
                            print(f"[medical-worker]   [Chunk {i+1}/{total_chunks}] ↗ Uploading to S3...")
                            _upload_to_s3_with_retry(local_dest, s3_dest_path)
                            upload_elapsed = time.time() - upload_start
                            print(f"[medical-worker]   [Chunk {i+1}/{total_chunks}] ✅ Upload complete ({upload_elapsed:.2f}s)")
                            chunks_completed += 1
                            metrics["upload_time"] += upload_elapsed
                        except Exception as chunk_err:
                            # Check for DuckDB buffer allocation error (4GB limit)
                            error_msg = str(chunk_err)
                            if "4294967296" in error_msg or "Information loss on integer cast" in error_msg or "outside of target range" in error_msg:
                                print(f"[medical-worker] ⚠️ Buffer allocation error detected. This may indicate chunk size is too large for JOIN operations.")
                                print(f"[medical-worker]   Error: {error_msg[:200]}")
                                # Re-raise to trigger retry with smaller chunk size
                                raise
                            else:
                                # Re-raise other errors
                                raise
                        finally:
                            # Always clean up local file, even if upload fails
                            try:
                                if os.path.exists(local_dest):
                                    os.remove(local_dest)
                            except Exception as cleanup_err:
                                print(f"[medical-worker] ⚠️ Warning: Could not clean up {local_dest}: {cleanup_err}")
                    else:
                        # Direct write (original behavior) - using explicit LEFT JOINs
                        chunk_start = time.time()
                        read_expr = get_read_parquet_expr([])
                        print(f"[medical-worker]   [Chunk {i+1}/{total_chunks}] ▶ Writing directly to S3 (rows={limit:,}, offset={offset:,})...")
                        query = build_join_query(read_expr, limit=limit, offset=offset)
                        con.sql(f"COPY ({query}) TO '{s3_dest_path}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE)")
                        chunk_elapsed = time.time() - chunk_start
                        print(f"[medical-worker]   [Chunk {i+1}/{total_chunks}] ✅ Write complete ({chunk_elapsed:.2f}s)")
                        chunks_completed += 1
                # Merge only if not in no-merge mode
                # Use incremental merge to avoid loading all chunks into memory at once
                no_merge = os.getenv("PGX_NO_MERGE", "0") == "1"
                if not no_merge:
                    print(f"[medical-worker] 🔗 Starting chunk merge...")
                    merge_start = time.time()
                    staging_prefix = _staging_prefix_for(filename, staging_suffix)
                    _merge_chunks_incremental(con, staging_prefix, filename, max_chunks_per_batch=None)
                    merge_elapsed = time.time() - merge_start
                    print(f"[medical-worker] ✅ Chunk merge complete ({merge_elapsed:.2f}s)")
                else:
                    print(f"[medical-worker] ⏭️  Skipping merge (PGX_NO_MERGE=1)")
                chunk_processing_elapsed = time.time() - chunk_processing_start
                chunked_elapsed = time.time() - chunked_start
                metrics["processing_time"] = chunk_processing_elapsed
                metrics["chunks_processed"] = chunks_completed
                print(f"[medical-worker] ✅ Chunk processing complete ({chunk_processing_elapsed:.2f}s)")
                print(f"[medical-worker] ✅ Chunked processing complete ({chunked_elapsed:.2f}s)")
            else:
                # Nothing to write
                print(f"[medical-worker] ⏭️  No chunks to process")
        elif apply:
            # Use local staging for non-chunked mode too
            print(f"[medical-worker] 📝 Starting non-chunked processing mode...")
            nonchunked_start = time.time()
            use_local_staging = os.getenv("PGX_USE_LOCAL_STAGING", "1") == "1"
            
            if use_local_staging and filename.startswith("s3://"):
                # Write to local first, then upload
                local_dest = _local_staging_path(filename)
                print(f"[medical-worker] ▶ Writing to local staging: {os.path.basename(local_dest)}")
                
                try:
                    # Write to local staging - using explicit LEFT JOINs
                    write_start = time.time()
                    read_expr = get_read_parquet_expr([])
                    print(f"[medical-worker]   → Writing to local disk...")
                    query = build_join_query(read_expr)
                    con.sql(f"COPY ({query}) TO '{local_dest}' (FORMAT PARQUET)")
                    write_elapsed = time.time() - write_start
                    print(f"[medical-worker] ✅ Local write complete ({write_elapsed:.2f}s)")
                    
                    # Upload to S3
                    upload_start = time.time()
                    print(f"[medical-worker] ↗ Uploading to S3...")
                    _upload_to_s3_with_retry(local_dest, filename)
                    upload_elapsed = time.time() - upload_start
                    print(f"[medical-worker] ✅ Upload complete ({upload_elapsed:.2f}s)")
                finally:
                    # Always clean up local file, even if upload fails
                    try:
                        if os.path.exists(local_dest):
                            os.remove(local_dest)
                    except Exception as cleanup_err:
                        print(f"[medical-worker] ⚠️ Warning: Could not clean up {local_dest}: {cleanup_err}")
            else:
                # Direct write (original behavior) - using explicit LEFT JOINs
                write_start = time.time()
                read_expr = get_read_parquet_expr([])
                print(f"[medical-worker] ▶ Writing directly to S3...")
                query = build_join_query(read_expr)
                con.sql(f"COPY ({query}) TO '{filename}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE)")
                write_elapsed = time.time() - write_start
                print(f"[medical-worker] ✅ Write complete ({write_elapsed:.2f}s)")
            nonchunked_elapsed = time.time() - nonchunked_start
            metrics["processing_time"] = nonchunked_elapsed
            print(f"[medical-worker] ✅ Non-chunked processing complete ({nonchunked_elapsed:.2f}s)")
            t0 = nonchunked_start
            t1 = time.time()
            if log_s3 and filename.startswith("s3://"):
                try:
                    import boto3  # type: ignore
                    from helpers_1997_13.s3_utils import s3_path_to_bucket_key
                    bkt, key = s3_path_to_bucket_key(filename)
                    s3 = boto3.client('s3')
                    post = s3.head_object(Bucket=bkt, Key=key)
                    size_b = post.get('ContentLength') or 0
                    metrics["file_size_bytes"] = size_b
                    elapsed = max(1e-6, t1 - t0)
                    mbps = (size_b / (1024 * 1024)) / elapsed
                    print(f"[medical-worker] S3 write: key={key}, size_bytes={size_b}, elapsed_sec={elapsed:.3f}, approx_MBps={mbps:.2f}, etag={post.get('ETag')}")
                    if s3_info_pre and s3_info_pre.get('size_bytes') is not None:
                        print(f"[medical-worker] S3 pre-size={s3_info_pre['size_bytes']}, post-size={size_b}")
                except Exception as _e:
                    print(f"[medical-worker] S3 throughput logging failed: {_e}")
            
            # Get record count if not already set
            if metrics["records_processed"] == 0:
                try:
                    metrics["records_processed"] = con.sql(f"SELECT COUNT(*) FROM read_parquet('{filename}')").fetchone()[0]
                except Exception:
                    pass
            
            # Get file size if not already set
            if metrics["file_size_bytes"] == 0:
                metrics["file_size_bytes"] = _get_object_size_bytes(filename)
            
            # Update final metrics
            metrics["duration_seconds"] = time.time() - worker_start_time
            metrics["mapping_load_time"] = mapping_elapsed if 'mapping_elapsed' in locals() else 0.0
            
            if resume:
                print(f"[medical-worker] 📌 Writing checkpoint marker...")
                try:
                    # Extract entity_id from filename (age_band/event_year pattern)
                    entity_id = "UNKNOWN"
                    try:
                        import re
                        age_match = re.search(r'age_band=([^/]+)', filename)
                        year_match = re.search(r'event_year=(\d{4})', filename)
                        if age_match and year_match:
                            entity_id = f"MEDICAL_{age_match.group(1)}_{year_match.group(1)}"
                    except Exception:
                        pass
                    
                    checkpoint_data = {
                        "pipeline": "update_codes",
                        "entity_id": entity_id,
                        "phase": "code_mapping",
                        "status": "completed",
                        "ts": str(int(time.time())),
                        "metrics": metrics
                    }
                    _write_marker(filename, marker_suffix, checkpoint_data)
                    print(f"[medical-worker] ✅ Checkpoint marker written")
                except Exception as e:
                    print(f"[medical-worker] ⚠️ Warning: Could not write checkpoint marker: {e}")
        else:
            # Dry-run mode - using explicit LEFT JOINs
            print(f"[medical-worker] 🔍 Dry-run mode: Previewing changes...")
            read_expr = get_read_parquet_expr([])
            query = build_join_query(read_expr, limit=1, offset=0)
            con.sql(query).fetchall()
            if resume:
                print(f"[medical-worker] 📌 Writing dry-run checkpoint marker...")
                try:
                    # Extract entity_id from filename
                    entity_id = "UNKNOWN"
                    try:
                        import re
                        age_match = re.search(r'age_band=([^/]+)', filename)
                        year_match = re.search(r'event_year=(\d{4})', filename)
                        if age_match and year_match:
                            entity_id = f"MEDICAL_{age_match.group(1)}_{year_match.group(1)}"
                    except Exception:
                        pass
                    
                    _write_marker(filename, marker_suffix, {
                        "pipeline": "update_codes",
                        "entity_id": entity_id,
                        "phase": "code_mapping",
                        "status": "dry_run_previewed",
                        "ts": str(int(time.time())),
                        "metrics": {"duration_seconds": time.time() - worker_start_time}
                    })
                    print(f"[medical-worker] ✅ Dry-run checkpoint marker written")
                except Exception as e:
                    print(f"[medical-worker] ⚠️ Warning: Could not write checkpoint marker: {e}")
        
        # Final completion message
        worker_elapsed = time.time() - worker_start_time
        print(f"[medical-worker] ✅ COMPLETE: {os.path.basename(filename)} (total time: {worker_elapsed:.2f}s)")
        return f"✓ Updated {filename}"
    finally:
        # Explicitly close DuckDB connection and clean up
        try:
            if con:
                con.close()
                # Explicitly delete connection reference before GC for clarity
                del con
                # Force garbage collection to ensure connection is fully released
                import gc
                gc.collect()
        except Exception:
            pass
        # Note: Worker temp directory cleanup is handled by atexit (once per process, not per partition)


def apply_mappings_to_medical(
    icd_map: Dict[str, str],
    cpt_map: Dict[str, str],
    icd_target_map: Dict[str, str],
    years: Optional[List[int]] = None,
    age_bands: Optional[List[str]] = None,
    apply: bool = True,
    workers: int = 8,
    threads_per_worker: int = 1,
    log_cpu: bool = False,
    log_s3: bool = False,
    resume: bool = False,
    marker_suffix: str = ".codes_updated.ok",
    chunked: bool = False,
    chunk_rows: int = 0,
    staging_suffix: str = ".codes_updated.staging/",
):
    print(f"\n{'='*80}")
    print(f"[medical] 🚀 STARTING medical code updates")
    print(f"[medical]   Workers: {workers}, Chunked: {chunked}, Apply: {apply}")
    if years:
        print(f"[medical]   Years filter: {years}")
    if age_bands:
        print(f"[medical]   Age bands filter: {age_bands}")
    print(f"{'='*80}\n")
    medical_start_time = time.time()
    
    # Build partition filter by paths if needed
    src_glob = GOLD_MEDICAL_GLOB
    if years:
        years_set = set(int(y) for y in years)
    else:
        years_set = None
    if age_bands:
        ab_set = set(age_bands)
    else:
        ab_set = None

    # Enumerate partitions via DuckDB glob
    conn = create_duckdb_conn(threads=1)
    try:
        result = conn.sql(
            """
            WITH f AS (
              SELECT file AS filename FROM glob(?)
            ), p AS (
              SELECT
                regexp_extract(filename, 'age_band=([^/]+)', 1) AS age_band,
                CAST(regexp_extract(filename, 'event_year=([0-9]{4})', 1) AS INTEGER) AS event_year,
                filename
              FROM f
            )
            SELECT * FROM p
            """,
            params=[src_glob],
        )
        
        # Use fetch_arrow_table() for streaming (more memory-efficient than .df())
        # Falls back to .df() if arrow is not available
        try:
            import pyarrow as pa
            arrow_table = result.fetch_arrow_table()
            parts = arrow_table.to_pandas()
        except (ImportError, Exception):
            # Fallback to .df() if arrow not available or fails
            parts = result.df()
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if parts.empty:
        print("[medical] ⚠️  No medical gold files found.")
        return
    
    print(f"[medical] 📊 Found {len(parts)} partition(s) to process")

    # Filter partitions if requested
    if years_set is not None:
        parts = parts[parts["event_year"].isin(years_set)]
    if ab_set is not None:
        parts = parts[parts["age_band"].isin(ab_set)]

    # Progress snapshot
    if resume:
        completed = 0
        pending = 0
        chunk_existing = 0
        chunk_expected = 0
        # Note: _marker_exists() calls boto3.head_object() which may be throttled for 1k+ partitions.
        # If latency is observed, consider async batching (e.g., using ThreadPoolExecutor with batch_size=100)
        for _, row in parts.iterrows():
            fname = row["filename"]
            if _marker_exists(fname, marker_suffix):
                completed += 1
            else:
                pending += 1
                if chunked and chunk_rows and chunk_rows > 0:
                    pref = _staging_prefix_for(fname, staging_suffix)
                    chunk_existing += _list_staging_chunks(pref)
                    try:
                        conn2 = create_duckdb_conn(threads=1)
                        try:
                            total_rows = conn2.sql(f"SELECT COALESCE(SUM(num_rows), 0) FROM parquet_metadata('{fname}')").fetchone()[0]
                        except Exception:
                            total_rows = conn2.sql(f"SELECT COUNT(*) FROM read_parquet('{fname}')").fetchone()[0]
                        try:
                            conn2.close()
                        except Exception:
                            pass
                        chunk_expected += math.ceil(total_rows / float(chunk_rows)) if total_rows else 0
                    except Exception:
                        pass
        if chunked and chunk_rows and chunk_rows > 0:
            print(f"[medical] Progress: partitions completed={completed}, pending={pending}, chunks existing={chunk_existing}, expected={chunk_expected}")
        else:
            print(f"[medical] Progress: partitions completed={completed}, pending={pending}")

    # Parallel processing of partitions
    results = []
    errors = []
    
    # Surface PGX_TOTAL_WORKERS to workers (for upload concurrency scaling)
    os.environ.setdefault('PGX_TOTAL_WORKERS', str(workers))
    
    # Choose multiprocessing context (fork on Linux for better performance, spawn otherwise)
    ctx, mp_method = get_multiprocessing_context()
    if mp_method == 'fork':
        print(f"[medical] Using 'fork' multiprocessing (faster startup, shared memory)")
    else:
        print(f"[medical] Using 'spawn' multiprocessing (slower startup, copies environment)")
    
    # Check if temp DB should be used (for high memory pressure scenarios)
    # Default to enabled (1) for better stability with multiprocessing
    use_temp_db = os.getenv('PGX_USE_TEMP_DB', '1') == '1'
    if use_temp_db:
        print(f"[medical] Using disk-backed DuckDB (PGX_USE_TEMP_DB=1) - reduces memory pressure")
    else:
        print(f"[medical] Using in-memory DuckDB (PGX_USE_TEMP_DB=0) - faster but higher memory usage")
    
    # Option to persist mappings to temp files to reduce memory duplication in spawn mode
    # Default to enabled (1) when using spawn mode for better memory efficiency
    persist_mappings_default = '1' if mp_method == 'spawn' else '0'
    persist_mappings = os.getenv('PGX_PERSIST_MAPPINGS', persist_mappings_default) == '1' and mp_method == 'spawn'
    mapping_temp_dir = None
    if persist_mappings:
        mapping_temp_dir = persist_mappings_to_temp(icd_map, cpt_map, icd_target_map)
        if mapping_temp_dir:
            print(f"[medical] Mappings persisted to {mapping_temp_dir} (reducing memory duplication)")
            # Pass empty dicts to workers - they'll load from temp files
            icd_map_for_workers = {}
            cpt_map_for_workers = {}
            icd_target_map_for_workers = {}
        else:
            # Fallback: use original mappings
            icd_map_for_workers = icd_map
            cpt_map_for_workers = cpt_map
            icd_target_map_for_workers = icd_target_map
    else:
        icd_map_for_workers = icd_map
        cpt_map_for_workers = cpt_map
        icd_target_map_for_workers = icd_target_map
    
    # Ensure threads_per_worker is 1 to avoid oversubscription (unless explicitly set higher)
    if threads_per_worker > 1:
        print(f"⚠️ Warning: threads_per_worker={threads_per_worker} > 1 may cause CPU oversubscription. Consider using 1 thread per worker with process-level parallelism.")
    
    if workers and workers > 1:
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
        try:
            # Get retry and timeout settings
            max_retries = get_retry_attempts()
            timeout_seconds = get_timeout_seconds()
            
            print(f"[medical] Submitting {len(parts)} partitions to {workers} workers...")
            future_to_file = {}
            for _, row in parts.iterrows():
                # Submit with retry wrapper
                future = executor.submit(_process_medical_partition, row["filename"], icd_map_for_workers, cpt_map_for_workers, icd_target_map_for_workers, apply, threads_per_worker, log_cpu, log_s3, resume, marker_suffix, chunked, chunk_rows, staging_suffix, workers, use_temp_db, mapping_temp_dir)
                future_to_file[future] = row
            
            print(f"[medical] All {len(future_to_file)} partitions submitted, waiting for completion...")
            import sys
            sys.stdout.flush()  # Ensure output is flushed
            
            # Track retry attempts per partition
            partition_retries = {row["filename"]: 0 for _, row in parts.iterrows()}
            retry_queue = []  # Queue for partitions that need retry
            
            # Process all futures with progress tracking
            completed_count = 0
            total_count = len(future_to_file)
            
            for future in as_completed(future_to_file):
                completed_count += 1
                row = future_to_file[future]
                filename = row["filename"]
                
                # Log progress every completion
                print(f"[medical] Progress: {completed_count}/{total_count} partitions completed")
                sys.stdout.flush()
                
                try:
                    msg = future.result()
                    print(msg)
                    sys.stdout.flush()
                    results.append(msg)
                except BrokenProcessPool as e:
                    # Worker process was killed (likely OOM or segfault)
                    # Try to get more details from the exception
                    error_details = str(e)
                    if hasattr(e, '__cause__') and e.__cause__:
                        error_details += f"\n  Cause: {e.__cause__}"
                    if hasattr(e, '__context__') and e.__context__:
                        error_details += f"\n  Context: {e.__context__}"
                    
                    partition_retries[filename] = partition_retries.get(filename, 0) + 1
                    attempt = partition_retries[filename]
                    
                    if attempt < max_retries:
                        wait_time = 2 ** (attempt - 1)  # Exponential backoff: 1s, 2s, 4s
                        print(f"⚠️ Worker process crashed for {filename} (attempt {attempt}/{max_retries})")
                        print(f"   Error: {error_details}")
                        print(f"   This usually indicates Out of Memory (OOM) or a segmentation fault.")
                        print(f"   Check system logs: dmesg | tail -20 or journalctl -k | tail -20")
                        print(f"   Retrying in {wait_time} seconds...")
                        sys.stdout.flush()
                        time.sleep(wait_time)
                        retry_queue.append((filename, row, attempt))
                    else:
                        print(f"✗ Worker process crashed after {max_retries} attempts for {filename}")
                        print(f"   Error: {error_details}")
                        print(f"  Check system logs: dmesg | tail -20 or journalctl -k | tail -20")
                        print(f"  Consider: reducing workers, reducing PGX_DUCKDB_MEMORY_LIMIT, or using spawn mode")
                        sys.stdout.flush()
                        errors.append(filename)
                except Exception as e:
                    partition_retries[filename] = partition_retries.get(filename, 0) + 1
                    attempt = partition_retries[filename]
                    
                    if attempt < max_retries:
                        wait_time = 2 ** (attempt - 1)  # Exponential backoff: 1s, 2s, 4s
                        tb = traceback.format_exc()
                        print(f"⚠️ Error processing {filename} (attempt {attempt}/{max_retries}): {e}")
                        print(f"   Traceback:\n{tb}")
                        print(f"   Retrying in {wait_time} seconds...")
                        sys.stdout.flush()
                        time.sleep(wait_time)
                        retry_queue.append((filename, row, attempt))
                    else:
                        tb = traceback.format_exc()
                        print(f"✗ Error processing {filename} after {max_retries} attempts: {e}\n{tb}")
                        sys.stdout.flush()
                        errors.append(filename)
            
            # Process retries
            while retry_queue:
                filename, row, attempt = retry_queue.pop(0)
                print(f"[medical] Retrying {filename} (attempt {attempt + 1}/{max_retries})...")
                try:
                    future = executor.submit(_process_medical_partition, filename, icd_map_for_workers, cpt_map_for_workers, icd_target_map_for_workers, apply, threads_per_worker, log_cpu, log_s3, resume, marker_suffix, chunked, chunk_rows, staging_suffix, workers, use_temp_db, mapping_temp_dir)
                    msg = future.result(timeout=timeout_seconds if timeout_seconds > 0 else None)
                    print(msg)
                    results.append(msg)
                except Exception as e:
                    if attempt + 1 < max_retries:
                        wait_time = 2 ** attempt
                        print(f"⚠️ Retry {attempt + 1} failed for {filename}, will retry again in {wait_time}s...")
                        time.sleep(wait_time)
                        retry_queue.append((filename, row, attempt + 1))
                    else:
                        tb = traceback.format_exc()
                        print(f"✗ Error processing {filename} after {max_retries} attempts: {e}\n{tb}")
                        errors.append(filename)
        finally:
            # Explicitly shutdown executor to ensure proper cleanup
            executor.shutdown(wait=True)
            # Clean up mapping temp directory if used
            if mapping_temp_dir and os.path.exists(mapping_temp_dir):
                try:
                    import shutil
                    shutil.rmtree(mapping_temp_dir)
                except Exception:
                    pass
    else:
        # Default to enabled (1) for better stability
        use_temp_db = os.getenv('PGX_USE_TEMP_DB', '1') == '1'
        for _, row in parts.iterrows():
            try:
                msg = _process_medical_partition(row["filename"], icd_map_for_workers, cpt_map_for_workers, icd_target_map_for_workers, apply, threads_per_worker, log_cpu, log_s3, resume, marker_suffix, chunked, chunk_rows, staging_suffix, workers, use_temp_db, mapping_temp_dir)
                print(msg)
                results.append(msg)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"✗ Error processing {row['filename']}: {e}\n{tb}")
                errors.append(row['filename'])
        # Clean up mapping temp directory if used
        if mapping_temp_dir and os.path.exists(mapping_temp_dir):
            try:
                import shutil
                shutil.rmtree(mapping_temp_dir)
            except Exception:
                pass
    medical_elapsed = time.time() - medical_start_time
    print(f"\n{'='*80}")
    print(f"[medical] ✅ COMPLETE: {len(results)} successful, {len(errors)} errors (total time: {medical_elapsed:.2f}s)")
    print(f"{'='*80}\n")
    if errors:
        print(f"[medical] ⚠️  Completed with {len(errors)} errors.")


def _process_pharmacy_partition(filename: str, drug_map: Dict[str, str], apply: bool, threads: int, log_cpu: bool = False, log_s3: bool = False, resume: bool = False, marker_suffix: str = ".codes_updated.ok", chunked: bool = False, chunk_rows: int = 0, staging_suffix: str = ".codes_updated.staging/", total_workers: Optional[int] = None, use_temp_db: bool = False, mapping_temp_dir: Optional[str] = None) -> str:
    import sys
    worker_start_time = time.time()
    print(f"[pharmacy-worker] 🚀 START processing: {os.path.basename(filename)}")
    sys.stdout.flush()  # Ensure worker output is immediately visible
    
    # Initialize metrics tracking
    metrics = {
        "records_processed": 0,
        "chunks_processed": 0,
        "file_size_bytes": 0,
        "duration_seconds": 0.0,
        "mapping_load_time": 0.0,
        "processing_time": 0.0,
        "upload_time": 0.0
    }
    
    # Load mappings from temp files if provided (reduces memory duplication in spawn mode)
    if mapping_temp_dir:
        print(f"[pharmacy-worker] 📥 Loading mappings from temp directory...")
        _, _, _, drug_map = load_mappings_from_temp(mapping_temp_dir)
        print(f"[pharmacy-worker] ✅ Mappings loaded from temp directory")
    
    # Enforce threads=1 to avoid oversubscription (process-level parallelism is sufficient)
    if threads > 1:
        print(f"[pharmacy-worker] ⚠️ Warning: threads={threads} > 1 may cause CPU oversubscription. Using 1 thread per worker.")
        threads = 1
    
    print(f"[pharmacy-worker] 🔌 Creating DuckDB connection...")
    con = create_duckdb_conn(threads=threads, total_workers=total_workers, use_temp_db=use_temp_db)
    print(f"[pharmacy-worker] ✅ DuckDB connection established")
    
    try:
        if resume and _marker_exists(filename, marker_suffix):
            print(f"[pharmacy-worker] ↩ Skipped (checkpoint exists): {os.path.basename(filename)}")
            return f"↩ Skipped (checkpoint) {filename}"
        if log_cpu:
            log_cpu_context(prefix="[pharmacy-worker]", threads=threads)
        s3_info_pre = None
        if log_s3 and filename.startswith("s3://"):
            try:
                import boto3  # type: ignore
                from helpers_1997_13.s3_utils import s3_path_to_bucket_key
                bkt, key = s3_path_to_bucket_key(filename)
                s3 = boto3.client('s3')
                pre = s3.head_object(Bucket=bkt, Key=key)
                s3_info_pre = {
                    'size_bytes': pre.get('ContentLength'),
                    'etag': pre.get('ETag'),
                }
            except Exception:
                s3_info_pre = None
        if drug_map:
            print(f"[pharmacy-worker] 📋 Loading drug mapping dictionary into DuckDB ({len(drug_map)} entries)...")
            mapping_start = time.time()
            # Use chunked streaming to avoid materializing full list
            load_mapping_into_duckdb_chunked(con, drug_map, "drug_map", chunk_size=10000)
            mapping_elapsed = time.time() - mapping_start
            metrics["mapping_load_time"] = mapping_elapsed
            print(f"[pharmacy-worker] ✅ Drug mapping loaded in {mapping_elapsed:.2f}s")

        mapped = (
            "CASE WHEN drug_name IS NULL OR TRIM(CAST(drug_name AS VARCHAR))='' THEN NULL "
            "ELSE COALESCE((SELECT canonical FROM drug_map WHERE variant = LOWER(TRIM(drug_name)) LIMIT 1), LOWER(TRIM(drug_name))) END"
        )
        
        # Note: We always read all columns because SELECT * REPLACE requires all columns to be present
        # Column-level optimization would require restructuring the query, which isn't worth the complexity
        def get_read_parquet_expr_pharmacy(cols_to_read: List[str]) -> str:
            # Always read all columns when using SELECT * REPLACE
            return f"read_parquet('{filename}')"

        # Pre-check: skip if no changes (use sampling for large files to save memory)
        skip_sample_check = os.getenv("PGX_SKIP_SAMPLE_CHECK", "0") == "1"
        if apply and not chunked and not skip_sample_check:
            print(f"[pharmacy-worker] 🔍 Starting pre-check (sampling for changes)...")
            precheck_start = time.time()
            change_cond = f"(drug_name IS DISTINCT FROM {mapped})"
            # Use sampling to avoid loading entire file into memory
            sample_size = 100000
            try:
                needs_update = con.sql(
                    f"SELECT EXISTS(SELECT 1 FROM read_parquet('{filename}') WHERE {change_cond} LIMIT {sample_size})"
                ).fetchone()[0]
                # If no changes in sample, check file size before full scan
                if not needs_update:
                    file_size_mb = _get_object_size_bytes(filename) / (1024 * 1024)
                    if file_size_mb < 100:  # Only full scan if file < 100MB
                        needs_update = con.sql(
                            f"SELECT EXISTS(SELECT 1 FROM read_parquet('{filename}') WHERE {change_cond})"
                        ).fetchone()[0]
            except Exception:
                # Fall back to full scan if sampling fails
                needs_update = con.sql(
                    f"SELECT EXISTS(SELECT 1 FROM read_parquet('{filename}') WHERE {change_cond})"
                ).fetchone()[0]
            precheck_elapsed = time.time() - precheck_start
            if not needs_update:
                print(f"[pharmacy-worker] ↩ Pre-check complete ({precheck_elapsed:.2f}s): No changes detected, skipping")
                if resume:
                    try:
                        # Extract entity_id from filename
                        entity_id = "UNKNOWN"
                        try:
                            import re
                            age_match = re.search(r'age_band=([^/]+)', filename)
                            year_match = re.search(r'event_year=(\d{4})', filename)
                            if age_match and year_match:
                                entity_id = f"PHARMACY_{age_match.group(1)}_{year_match.group(1)}"
                        except Exception:
                            pass
                        
                        _write_marker(filename, marker_suffix, {
                            "pipeline": "update_codes",
                            "entity_id": entity_id,
                            "phase": "code_mapping",
                            "status": "skipped_no_changes",
                            "ts": str(int(time.time())),
                            "metrics": {"duration_seconds": time.time() - worker_start_time}
                        })
                    except Exception:
                        pass
                return f"↩ Skipped (no changes) {filename}"
            print(f"[pharmacy-worker] ✅ Pre-check complete ({precheck_elapsed:.2f}s): Changes detected, proceeding")
        if apply and chunked:
            print(f"[pharmacy-worker] 📦 Starting chunked processing mode...")
            chunked_start = time.time()
            # Use local staging for better performance and reliability
            use_local_staging = os.getenv("PGX_USE_LOCAL_STAGING", "1") == "1"
            
            # Row-group aligned chunk planning
            print(f"[pharmacy-worker]   → Computing chunk plan...")
            effective_chunk_rows = chunk_rows
            if not effective_chunk_rows or effective_chunk_rows <= 0:
                total_rows_try = 0
                try:
                    total_rows_try = con.sql(f"SELECT COALESCE(SUM(num_rows), 0) FROM parquet_metadata('{filename}')").fetchone()[0]
                except Exception:
                    total_rows_try = con.sql(f"SELECT COUNT(*) FROM read_parquet('{filename}')").fetchone()[0]
                size_b = _get_object_size_bytes(filename)
                avg_bpr = (size_b / max(1, total_rows_try)) if total_rows_try else 1024.0
                target_mb = float(os.getenv("PGX_TARGET_FILE_SIZE_MB", "0"))
                if target_mb <= 0:
                    target_mb = 512.0
                target_bytes = target_mb * 1024 * 1024
                effective_chunk_rows = max(10000, int(target_bytes / max(1.0, avg_bpr)))
            chunks, total_rows = _compute_rowgroup_chunks(con, filename, effective_chunk_rows)
            if chunks:
                total_chunks = len(chunks)
                metrics["records_processed"] = total_rows
                metrics["chunks_processed"] = total_chunks
                metrics["file_size_bytes"] = _get_object_size_bytes(filename)
                print(f"[pharmacy-worker]   → Chunk plan: {total_chunks} chunks, {total_rows:,} total rows")
                print(f"[pharmacy-worker] 🚀 Starting chunk processing ({total_chunks} chunks)...")
                chunk_processing_start = time.time()
                chunks_completed = 0
                for i, (offset, limit) in enumerate(chunks):
                    no_merge = os.getenv("PGX_NO_MERGE", "0") == "1"
                    s3_dest_path = _final_chunk_path_for(filename, i) if no_merge else _chunk_path_for(filename, staging_suffix, i)
                    
                    if resume and _object_exists(s3_dest_path):
                        print(f"[pharmacy-worker] ↩ Skipped chunk {i+1}/{total_chunks} (exists) {filename}")
                        chunks_completed += 1
                        continue
                    
                    # Write to local staging first if enabled
                    if use_local_staging and s3_dest_path.startswith("s3://"):
                        local_dest = _local_staging_path(s3_dest_path)
                        chunk_start = time.time()
                        print(f"[pharmacy-worker]   [Chunk {i+1}/{total_chunks}] ▶ Writing to local staging (rows={limit:,}, offset={offset:,})...")
                        
                        try:
                            # Write to local disk (fast)
                            read_expr = get_read_parquet_expr_pharmacy([])
                            con.sql(
                                f"""
                                COPY (
                                  SELECT * REPLACE (
                                    {mapped} AS drug_name
                                  )
                                  FROM {read_expr}
                                  LIMIT {limit} OFFSET {offset}
                                ) TO '{local_dest}' (FORMAT PARQUET)
                                """
                            )
                            write_elapsed = time.time() - chunk_start
                            print(f"[pharmacy-worker]   [Chunk {i+1}/{total_chunks}] ✅ Local write complete ({write_elapsed:.2f}s)")
                            
                            # Upload to S3 (with retry)
                            upload_start = time.time()
                            print(f"[pharmacy-worker]   [Chunk {i+1}/{total_chunks}] ↗ Uploading to S3...")
                            _upload_to_s3_with_retry(local_dest, s3_dest_path)
                            upload_elapsed = time.time() - upload_start
                            print(f"[pharmacy-worker]   [Chunk {i+1}/{total_chunks}] ✅ Upload complete ({upload_elapsed:.2f}s)")
                            chunks_completed += 1
                            metrics["upload_time"] += upload_elapsed
                        finally:
                            # Always clean up local file, even if upload fails
                            try:
                                if os.path.exists(local_dest):
                                    os.remove(local_dest)
                            except Exception as cleanup_err:
                                print(f"[pharmacy-worker] ⚠️ Warning: Could not clean up {local_dest}: {cleanup_err}")
                    else:
                        # Direct write (original behavior)
                        chunk_start = time.time()
                        read_expr = get_read_parquet_expr_pharmacy([])
                        print(f"[pharmacy-worker]   [Chunk {i+1}/{total_chunks}] ▶ Writing directly to S3 (rows={limit:,}, offset={offset:,})...")
                        con.sql(
                            f"""
                            COPY (
                              SELECT * REPLACE (
                                {mapped} AS drug_name
                              )
                              FROM {read_expr}
                              LIMIT {limit} OFFSET {offset}
                            ) TO '{s3_dest_path}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE)
                            """
                        )
                        chunk_elapsed = time.time() - chunk_start
                        print(f"[pharmacy-worker]   [Chunk {i+1}/{total_chunks}] ✅ Write complete ({chunk_elapsed:.2f}s)")
                        chunks_completed += 1
                # Merge only if not in no-merge mode
                no_merge = os.getenv("PGX_NO_MERGE", "0") == "1"
                if not no_merge:
                    print(f"[pharmacy-worker] 🔗 Starting chunk merge...")
                    merge_start = time.time()
                    staging_prefix = _staging_prefix_for(filename, staging_suffix)
                    _merge_chunks_incremental(con, staging_prefix, filename, max_chunks_per_batch=None)
                    merge_elapsed = time.time() - merge_start
                    print(f"[pharmacy-worker] ✅ Chunk merge complete ({merge_elapsed:.2f}s)")
                else:
                    print(f"[pharmacy-worker] ⏭️  Skipping merge (PGX_NO_MERGE=1)")
                chunk_processing_elapsed = time.time() - chunk_processing_start
                chunked_elapsed = time.time() - chunked_start
                metrics["processing_time"] = chunk_processing_elapsed
                metrics["chunks_processed"] = chunks_completed
                print(f"[pharmacy-worker] ✅ Chunk processing complete ({chunk_processing_elapsed:.2f}s)")
                print(f"[pharmacy-worker] ✅ Chunked processing complete ({chunked_elapsed:.2f}s)")
            else:
                # Nothing to write
                print(f"[pharmacy-worker] ⏭️  No chunks to process")
        elif apply and not (chunked and chunk_rows and chunk_rows > 0):
            # Use local staging for non-chunked mode too
            print(f"[pharmacy-worker] 📝 Starting non-chunked processing mode...")
            nonchunked_start = time.time()
            use_local_staging = os.getenv("PGX_USE_LOCAL_STAGING", "1") == "1"
            
            if use_local_staging and filename.startswith("s3://"):
                # Write to local first, then upload
                local_dest = _local_staging_path(filename)
                print(f"[pharmacy-worker] ▶ Writing to local staging: {os.path.basename(local_dest)}")
                
                try:
                    # Write to local staging
                    write_start = time.time()
                    read_expr = get_read_parquet_expr_pharmacy([])
                    print(f"[pharmacy-worker]   → Writing to local disk...")
                    con.sql(
                        f"""
                        COPY (
                          SELECT * REPLACE (
                            {mapped} AS drug_name
                          )
                          FROM {read_expr}
                        ) TO '{local_dest}' (FORMAT PARQUET)
                        """
                    )
                    write_elapsed = time.time() - write_start
                    print(f"[pharmacy-worker] ✅ Local write complete ({write_elapsed:.2f}s)")
                    
                    # Upload to S3
                    upload_start = time.time()
                    print(f"[pharmacy-worker] ↗ Uploading to S3...")
                    _upload_to_s3_with_retry(local_dest, filename)
                    upload_elapsed = time.time() - upload_start
                    print(f"[pharmacy-worker] ✅ Upload complete ({upload_elapsed:.2f}s)")
                finally:
                    # Always clean up local file, even if upload fails
                    try:
                        if os.path.exists(local_dest):
                            os.remove(local_dest)
                    except Exception as cleanup_err:
                        print(f"[pharmacy-worker] ⚠️ Warning: Could not clean up {local_dest}: {cleanup_err}")
            else:
                # Direct write (original behavior)
                write_start = time.time()
                read_expr = get_read_parquet_expr_pharmacy([])
                print(f"[pharmacy-worker] ▶ Writing directly to S3...")
                con.sql(
                    f"""
                    COPY (
                      SELECT * REPLACE (
                        {mapped} AS drug_name
                      )
                      FROM {read_expr}
                    ) TO '{filename}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE)
                    """
                )
                write_elapsed = time.time() - write_start
                print(f"[pharmacy-worker] ✅ Write complete ({write_elapsed:.2f}s)")
            nonchunked_elapsed = time.time() - nonchunked_start
            metrics["processing_time"] = nonchunked_elapsed
            print(f"[pharmacy-worker] ✅ Non-chunked processing complete ({nonchunked_elapsed:.2f}s)")
            t0 = nonchunked_start
            t1 = time.time()
            if log_s3 and filename.startswith("s3://"):
                try:
                    import boto3  # type: ignore
                    from helpers_1997_13.s3_utils import s3_path_to_bucket_key
                    bkt, key = s3_path_to_bucket_key(filename)
                    s3 = boto3.client('s3')
                    post = s3.head_object(Bucket=bkt, Key=key)
                    size_b = post.get('ContentLength') or 0
                    metrics["file_size_bytes"] = size_b
                    elapsed = max(1e-6, t1 - t0)
                    mbps = (size_b / (1024 * 1024)) / elapsed
                    print(f"[pharmacy-worker] S3 write: key={key}, size_bytes={size_b}, elapsed_sec={elapsed:.3f}, approx_MBps={mbps:.2f}, etag={post.get('ETag')}")
                    if s3_info_pre and s3_info_pre.get('size_bytes') is not None:
                        print(f"[pharmacy-worker] S3 pre-size={s3_info_pre['size_bytes']}, post-size={size_b}")
                except Exception as _e:
                    print(f"[pharmacy-worker] S3 throughput logging failed: {_e}")
            
            # Get record count if not already set
            if metrics["records_processed"] == 0:
                try:
                    metrics["records_processed"] = con.sql(f"SELECT COUNT(*) FROM read_parquet('{filename}')").fetchone()[0]
                except Exception:
                    pass
            
            # Get file size if not already set
            if metrics["file_size_bytes"] == 0:
                metrics["file_size_bytes"] = _get_object_size_bytes(filename)
            
            # Update final metrics
            metrics["duration_seconds"] = time.time() - worker_start_time
            metrics["mapping_load_time"] = mapping_elapsed if 'mapping_elapsed' in locals() else 0.0
            
            if resume:
                print(f"[pharmacy-worker] 📌 Writing checkpoint marker...")
                try:
                    # Extract entity_id from filename (age_band/event_year pattern)
                    entity_id = "UNKNOWN"
                    try:
                        import re
                        age_match = re.search(r'age_band=([^/]+)', filename)
                        year_match = re.search(r'event_year=(\d{4})', filename)
                        if age_match and year_match:
                            entity_id = f"PHARMACY_{age_match.group(1)}_{year_match.group(1)}"
                    except Exception:
                        pass
                    
                    checkpoint_data = {
                        "pipeline": "update_codes",
                        "entity_id": entity_id,
                        "phase": "code_mapping",
                        "status": "completed",
                        "ts": str(int(time.time())),
                        "metrics": metrics
                    }
                    _write_marker(filename, marker_suffix, checkpoint_data)
                    print(f"[pharmacy-worker] ✅ Checkpoint marker written")
                except Exception as e:
                    print(f"[pharmacy-worker] ⚠️ Warning: Could not write checkpoint marker: {e}")
        else:
            # Dry-run mode
            print(f"[pharmacy-worker] 🔍 Dry-run mode: Previewing changes...")
            con.sql(
                f"""
                SELECT * REPLACE (
                  {mapped} AS drug_name
                )
                FROM read_parquet('{filename}')
                LIMIT 1
                """
            ).fetchall()
            if resume:
                print(f"[pharmacy-worker] 📌 Writing dry-run checkpoint marker...")
                try:
                    # Extract entity_id from filename
                    entity_id = "UNKNOWN"
                    try:
                        import re
                        age_match = re.search(r'age_band=([^/]+)', filename)
                        year_match = re.search(r'event_year=(\d{4})', filename)
                        if age_match and year_match:
                            entity_id = f"PHARMACY_{age_match.group(1)}_{year_match.group(1)}"
                    except Exception:
                        pass
                    
                    _write_marker(filename, marker_suffix, {
                        "pipeline": "update_codes",
                        "entity_id": entity_id,
                        "phase": "code_mapping",
                        "status": "dry_run_previewed",
                        "ts": str(int(time.time())),
                        "metrics": {"duration_seconds": time.time() - worker_start_time}
                    })
                    print(f"[pharmacy-worker] ✅ Dry-run checkpoint marker written")
                except Exception as e:
                    print(f"[pharmacy-worker] ⚠️ Warning: Could not write checkpoint marker: {e}")
        
        # Final completion message
        worker_elapsed = time.time() - worker_start_time
        print(f"[pharmacy-worker] ✅ COMPLETE: {os.path.basename(filename)} (total time: {worker_elapsed:.2f}s)")
        return f"✓ Updated {filename}"
    finally:
        # Explicitly close DuckDB connection and clean up
        try:
            if con:
                con.close()
                # Explicitly delete connection reference before GC for clarity
                del con
                # Force garbage collection to ensure connection is fully released
                import gc
                gc.collect()
        except Exception:
            pass
        # Note: Worker temp directory cleanup is handled by atexit (once per process, not per partition)


def apply_mappings_to_pharmacy(
    drug_map: Dict[str, str],
    years: Optional[List[int]] = None,
    age_bands: Optional[List[str]] = None,
    apply: bool = True,
    workers: int = 8,
    threads_per_worker: int = 1,
    log_cpu: bool = False,
    log_s3: bool = False,
    resume: bool = False,
    marker_suffix: str = ".codes_updated.ok",
    chunked: bool = False,
    chunk_rows: int = 0,
    staging_suffix: str = ".codes_updated.staging/",
):
    print(f"\n{'='*80}")
    print(f"[pharmacy] 🚀 STARTING pharmacy code updates")
    print(f"[pharmacy]   Workers: {workers}, Chunked: {chunked}, Apply: {apply}")
    if years:
        print(f"[pharmacy]   Years filter: {years}")
    if age_bands:
        print(f"[pharmacy]   Age bands filter: {age_bands}")
    print(f"{'='*80}\n")
    pharmacy_start_time = time.time()
    
    if not drug_map:
        print("[pharmacy] ⚠️  No drug map provided, skipping pharmacy updates")
        return

    src_glob = GOLD_PHARMACY_GLOB
    years_set = set(int(y) for y in years) if years else None
    ab_set = set(age_bands) if age_bands else None

    conn = create_duckdb_conn(threads=1)
    try:
        result = conn.sql(
            """
            WITH f AS (
              SELECT file AS filename FROM glob(?)
            ), p AS (
              SELECT
                regexp_extract(filename, 'age_band=([^/]+)', 1) AS age_band,
                CAST(regexp_extract(filename, 'event_year=([0-9]{4})', 1) AS INTEGER) AS event_year,
                filename
              FROM f
            )
            SELECT * FROM p
            """,
            params=[src_glob],
        )
        
        # Use fetch_arrow_table() for streaming (more memory-efficient than .df())
        # Falls back to .df() if arrow is not available
        try:
            import pyarrow as pa
            arrow_table = result.fetch_arrow_table()
            parts = arrow_table.to_pandas()
        except (ImportError, Exception):
            # Fallback to .df() if arrow not available or fails
            parts = result.df()
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if parts.empty:
        print("[pharmacy] ⚠️  No pharmacy gold files found.")
        return
    
    print(f"[pharmacy] 📊 Found {len(parts)} partition(s) to process")

    if years_set is not None:
        parts = parts[parts["event_year"].isin(years_set)]
    if ab_set is not None:
        parts = parts[parts["age_band"].isin(ab_set)]

    # Progress snapshot
    if resume:
        completed = 0
        pending = 0
        chunk_existing = 0
        chunk_expected = 0
        # Note: _marker_exists() calls boto3.head_object() which may be throttled for 1k+ partitions.
        # If latency is observed, consider async batching (e.g., using ThreadPoolExecutor with batch_size=100)
        for _, row in parts.iterrows():
            fname = row["filename"]
            if _marker_exists(fname, marker_suffix):
                completed += 1
            else:
                pending += 1
                if chunked and chunk_rows and chunk_rows > 0:
                    pref = _staging_prefix_for(fname, staging_suffix)
                    chunk_existing += _list_staging_chunks(pref)
                    try:
                        conn2 = create_duckdb_conn(threads=1)
                        try:
                            total_rows = conn2.sql(f"SELECT COALESCE(SUM(num_rows), 0) FROM parquet_metadata('{fname}')").fetchone()[0]
                        except Exception:
                            total_rows = conn2.sql(f"SELECT COUNT(*) FROM read_parquet('{fname}')").fetchone()[0]
                        try:
                            conn2.close()
                        except Exception:
                            pass
                        chunk_expected += math.ceil(total_rows / float(chunk_rows)) if total_rows else 0
                    except Exception:
                        pass
        if chunked and chunk_rows and chunk_rows > 0:
            print(f"[pharmacy] Progress: partitions completed={completed}, pending={pending}, chunks existing={chunk_existing}, expected={chunk_expected}")
        else:
            print(f"[pharmacy] Progress: partitions completed={completed}, pending={pending}")

    # Parallel processing of partitions
    results = []
    errors = []
    
    # Surface PGX_TOTAL_WORKERS to workers (for upload concurrency scaling)
    os.environ.setdefault('PGX_TOTAL_WORKERS', str(workers))
    
    # Choose multiprocessing context (fork on Linux for better performance, spawn otherwise)
    ctx, mp_method = get_multiprocessing_context()
    if mp_method == 'fork':
        print(f"[pharmacy] Using 'fork' multiprocessing (faster startup, shared memory)")
    else:
        print(f"[pharmacy] Using 'spawn' multiprocessing (slower startup, copies environment)")
    
    # Check if temp DB should be used (for high memory pressure scenarios)
    # Default to enabled (1) for better stability with multiprocessing
    use_temp_db = os.getenv('PGX_USE_TEMP_DB', '1') == '1'
    if use_temp_db:
        print(f"[pharmacy] Using disk-backed DuckDB (PGX_USE_TEMP_DB=1) - reduces memory pressure")
    else:
        print(f"[pharmacy] Using in-memory DuckDB (PGX_USE_TEMP_DB=0) - faster but higher memory usage")
    
    # Option to persist mappings to temp files to reduce memory duplication in spawn mode
    # Default to enabled (1) when using spawn mode for better memory efficiency
    persist_mappings_default = '1' if mp_method == 'spawn' else '0'
    persist_mappings = os.getenv('PGX_PERSIST_MAPPINGS', persist_mappings_default) == '1' and mp_method == 'spawn'
    mapping_temp_dir = None
    if persist_mappings:
        mapping_temp_dir = persist_mappings_to_temp(None, None, None, drug_map)
        if mapping_temp_dir:
            print(f"[pharmacy] Mappings persisted to {mapping_temp_dir} (reducing memory duplication)")
            # Pass empty dict to workers - they'll load from temp files
            drug_map_for_workers = {}
        else:
            # Fallback: use original mappings
            drug_map_for_workers = drug_map
    else:
        drug_map_for_workers = drug_map
    
    # Ensure threads_per_worker is 1 to avoid oversubscription (unless explicitly set higher)
    if threads_per_worker > 1:
        print(f"⚠️ Warning: threads_per_worker={threads_per_worker} > 1 may cause CPU oversubscription. Consider using 1 thread per worker with process-level parallelism.")
    
    if workers and workers > 1:
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
        try:
            # Get retry and timeout settings
            max_retries = get_retry_attempts()
            timeout_seconds = get_timeout_seconds()
            
            print(f"[pharmacy] Submitting {len(parts)} partitions to {workers} workers...")
            future_to_file = {}
            for _, row in parts.iterrows():
                future = executor.submit(_process_pharmacy_partition, row["filename"], drug_map_for_workers, apply, threads_per_worker, log_cpu, log_s3, resume, marker_suffix, chunked, chunk_rows, staging_suffix, workers, use_temp_db, mapping_temp_dir)
                future_to_file[future] = row
            
            print(f"[pharmacy] All {len(future_to_file)} partitions submitted, waiting for completion...")
            import sys
            sys.stdout.flush()  # Ensure output is flushed
            
            # Track retry attempts per partition
            partition_retries = {row["filename"]: 0 for _, row in parts.iterrows()}
            retry_queue = []  # Queue for partitions that need retry
            
            # Process all futures with progress tracking
            completed_count = 0
            total_count = len(future_to_file)
            
            for future in as_completed(future_to_file):
                completed_count += 1
                row = future_to_file[future]
                filename = row["filename"]
                
                # Log progress every completion
                print(f"[pharmacy] Progress: {completed_count}/{total_count} partitions completed")
                sys.stdout.flush()
                
                try:
                    msg = future.result()
                    print(msg)
                    sys.stdout.flush()
                    results.append(msg)
                except BrokenProcessPool as e:
                    # Worker process was killed (likely OOM or segfault)
                    partition_retries[filename] = partition_retries.get(filename, 0) + 1
                    attempt = partition_retries[filename]
                    
                    if attempt < max_retries:
                        wait_time = 2 ** (attempt - 1)  # Exponential backoff: 1s, 2s, 4s
                        print(f"⚠️ Worker process crashed for {filename} (attempt {attempt}/{max_retries})")
                        print(f"   This usually indicates Out of Memory (OOM) or a segmentation fault.")
                        print(f"   Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        retry_queue.append((filename, row, attempt))
                    else:
                        print(f"✗ Worker process crashed after {max_retries} attempts for {filename}")
                        print(f"  Check system logs: dmesg | tail -20 or journalctl -k | tail -20")
                        print(f"  Consider: reducing workers, reducing PGX_DUCKDB_MEMORY_LIMIT, or enabling PGX_USE_TEMP_DB=1")
                        errors.append(filename)
                except Exception as e:
                    partition_retries[filename] = partition_retries.get(filename, 0) + 1
                    attempt = partition_retries[filename]
                    
                    if attempt < max_retries:
                        wait_time = 2 ** (attempt - 1)  # Exponential backoff: 1s, 2s, 4s
                        print(f"⚠️ Error processing {filename} (attempt {attempt}/{max_retries}): {e}")
                        print(f"   Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        retry_queue.append((filename, row, attempt))
                    else:
                        tb = traceback.format_exc()
                        print(f"✗ Error processing {filename} after {max_retries} attempts: {e}\n{tb}")
                        errors.append(filename)
            
            # Process retries
            while retry_queue:
                filename, row, attempt = retry_queue.pop(0)
                print(f"[pharmacy] Retrying {filename} (attempt {attempt + 1}/{max_retries})...")
                try:
                    future = executor.submit(_process_pharmacy_partition, filename, drug_map_for_workers, apply, threads_per_worker, log_cpu, log_s3, resume, marker_suffix, chunked, chunk_rows, staging_suffix, workers, use_temp_db, mapping_temp_dir)
                    msg = future.result(timeout=timeout_seconds if timeout_seconds > 0 else None)
                    print(msg)
                    results.append(msg)
                except Exception as e:
                    if attempt + 1 < max_retries:
                        wait_time = 2 ** attempt
                        print(f"⚠️ Retry {attempt + 1} failed for {filename}, will retry again in {wait_time}s...")
                        time.sleep(wait_time)
                        retry_queue.append((filename, row, attempt + 1))
                    else:
                        tb = traceback.format_exc()
                        print(f"✗ Error processing {filename} after {max_retries} attempts: {e}\n{tb}")
                        errors.append(filename)
        finally:
            # Explicitly shutdown executor to ensure proper cleanup
            executor.shutdown(wait=True)
            # Clean up mapping temp directory if used
            if mapping_temp_dir and os.path.exists(mapping_temp_dir):
                try:
                    import shutil
                    shutil.rmtree(mapping_temp_dir)
                except Exception:
                    pass
    else:
        # Default to enabled (1) for better stability
        use_temp_db = os.getenv('PGX_USE_TEMP_DB', '1') == '1'
        for _, row in parts.iterrows():
            try:
                msg = _process_pharmacy_partition(row["filename"], drug_map_for_workers, apply, threads_per_worker, log_cpu, log_s3, resume, marker_suffix, chunked, chunk_rows, staging_suffix, workers, use_temp_db, mapping_temp_dir)
                print(msg)
                results.append(msg)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"✗ Error processing {row['filename']}: {e}\n{tb}")
                errors.append(row['filename'])
        # Clean up mapping temp directory if used
        if mapping_temp_dir and os.path.exists(mapping_temp_dir):
            try:
                import shutil
                shutil.rmtree(mapping_temp_dir)
            except Exception:
                pass
    pharmacy_elapsed = time.time() - pharmacy_start_time
    print(f"\n{'='*80}")
    print(f"[pharmacy] ✅ COMPLETE: {len(results)} successful, {len(errors)} errors (total time: {pharmacy_elapsed:.2f}s)")
    print(f"{'='*80}\n")
    if errors:
        print(f"[pharmacy] ⚠️  Completed with {len(errors)} errors.")


def main():
    parser = argparse.ArgumentParser(description="Apply JSON mappings to normalize codes in gold data")
    parser.add_argument("--icd-map", help="Path to ICD mapping JSON (local or s3://)", default=os.path.join(DEFAULT_MAPPING_DIR, 'icd_mappings.json'))
    parser.add_argument("--cpt-map", help="Path to CPT mapping JSON (local or s3://)", default=os.path.join(DEFAULT_MAPPING_DIR, 'cpt_mappings.json'))
    parser.add_argument("--use-suggested", action="store_true", help="Load *_mappings_suggested.json instead of canonical files")
    parser.add_argument("--icd-target-map", help="Path to target ICD mapping JSON (local or s3://)", default=os.path.join(DEFAULT_TARGET_MAP_DIR, 'target_icd_mapping.json'))
    parser.add_argument("--drug-map", help="Path to drug mapping JSON (local or s3://)")
    parser.add_argument("--years", help="Comma-separated years filter", default="")
    parser.add_argument("--age-bands", help="Comma-separated age bands filter", default="")
    parser.add_argument("--no-apply", action="store_true", help="Dry-run (do not write)")
    parser.add_argument("--threads", type=int, help="DuckDB threads per worker (default from PGX_THREADS_PER_WORKER or 1)")
    parser.add_argument("--fail-on-missing-mapping", action="store_true", help="Exit early if mapping files are missing or empty")
    parser.add_argument("--workers", type=int, help="Global override: workers for both datasets (processes)")
    parser.add_argument("--workers-medical", type=int, help="Workers for medical updates (default from PGX_WORKERS_MEDICAL or 16)")
    parser.add_argument("--workers-pharmacy", type=int, help="Workers for pharmacy updates (default from PGX_WORKERS_PHARMACY or 48)")
    parser.add_argument("--log-cpu", action="store_true", help="Log per-worker CPU affinity/context (requires psutil for best detail)")
    parser.add_argument("--log-s3", action="store_true", help="Log per-partition S3 throughput (size, elapsed, MB/s, ETag)")
    parser.add_argument("--resume", action="store_true", help="Skip partitions that already have a checkpoint marker")
    parser.add_argument("--checkpoint-suffix", default=".codes_updated.ok", help="Suffix for per-file checkpoint markers (default .codes_updated.ok)")
    parser.add_argument("--chunked", action="store_true", help="Enable chunked intra-file processing with per-chunk resume")
    parser.add_argument("--chunk-rows", type=int, default=0, help="Number of rows per chunk when --chunked is enabled")
    parser.add_argument("--staging-suffix", default=".codes_updated.staging/", help="Suffix for per-file staging prefix to hold chunks")
    parser.add_argument("--duckdb-mem-limit", default="", help="Optional DuckDB memory_limit (e.g., 1GB) to keep memory low")
    parser.add_argument("--show-progress", action="store_true", help="Show progress (completed/pending partitions, chunk counts) and exit")
    parser.add_argument("--progress-deep", action="store_true", help="Compute expected chunk counts by scanning row counts (slower)")
    parser.add_argument("--no-merge", action="store_true", help="Keep chunk files as final outputs per partition (no merge back to single file)")
    parser.add_argument("--target-file-size-mb", type=int, default=512, help="Target file size per output chunk when deriving chunk rows (default 512MB)")
    args = parser.parse_args()

    years = [int(y.strip()) for y in args.years.split(",") if y.strip()] if args.years else None
    age_bands = [ab.strip() for ab in args.age_bands.split(",") if ab.strip()] if args.age_bands else None

    # Resolve mapping file paths (suggested vs canonical)
    icd_path = args.icd_map
    cpt_path = args.cpt_map
    if args.use_suggested:
        icd_path = os.path.join(DEFAULT_MAPPING_DIR, 'icd_mappings_suggested.json')
        cpt_path = os.path.join(DEFAULT_MAPPING_DIR, 'cpt_mappings_suggested.json')

    # Clean up orphaned staging files from previous failed runs
    if os.getenv("PGX_USE_LOCAL_STAGING", "1") == "1":
        _cleanup_orphaned_staging_files(max_age_hours=24)
    
    # Clean up old DuckDB temp directories from previous failed runs
    cleanup_old_duckdb_temp_dirs(max_age_hours=1)
    
    icd_map = load_mapping(icd_path)
    cpt_map = load_mapping(cpt_path)
    drug_map = load_mapping(args.drug_map)
    icd_target_map = load_mapping(args.icd_target_map)

    # Validate mapping structures early to fail fast and surface mapping format errors
    def _validate_map(name: str, path: str, mapping: dict) -> list:
        errors = []
        if path and not mapping:
            # Allow empty CPT and Drug mapping files: it's valid to run without these mappings
            # (e.g., when no normalization for that code family is desired). Treat as a warning, not an error.
            if name.upper() in ('CPT', 'DRUG'):
                print(f"ℹ️ {name} mapping file provided but empty: {path} — continuing without {name} normalization")
                return errors
            errors.append(f"Mapping file provided but could not be loaded or is empty: {path}")
            return errors
        if mapping:
            if not isinstance(mapping, dict):
                errors.append(f"Mapping at {path} is not a JSON object/dict")
                return errors
            for k, v in list(mapping.items())[:20]:
                if not isinstance(k, str) or not (isinstance(v, str) or isinstance(v, (int, float))):
                    errors.append(f"Invalid mapping entry (non-string key or value): {k} -> {v}")
                    break
        return errors

    mapping_errors = []
    mapping_errors += _validate_map('ICD', icd_path, icd_map)
    mapping_errors += _validate_map('CPT', cpt_path, cpt_map)
    mapping_errors += _validate_map('Drug', args.drug_map, drug_map)
    mapping_errors += _validate_map('ICD Target', args.icd_target_map, icd_target_map)

    if mapping_errors:
        err_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '1_apcd_input_data', '7_update_codes_mapping_errors.json')
        try:
            # Ensure dir exists
            os.makedirs(os.path.dirname(err_path), exist_ok=True)
            with open(err_path, 'w', encoding='utf-8') as ef:
                json.dump({'errors': mapping_errors}, ef, indent=2)
            print(f"❌ Mapping validation failed; details written to {err_path}")
        except Exception:
            print("❌ Mapping validation failed; could not write error details to file")
        # If the user explicitly requested to fail on missing mapping, exit now
        if getattr(args, 'fail_on_missing_mapping', False):
            print("Exiting due to mapping validation failure (--fail-on-missing-mapping)")
            sys.exit(2)
        else:
            print("⚠️ Mapping validation warnings detected; continuing (use --fail-on-missing-mapping to abort)")

    print("🔧 Mappings loaded:")
    print(f"  ICD: {len(icd_map)} entries")
    print(f"  CPT: {len(cpt_map)} entries")
    print(f"  Drug: {len(drug_map)} entries")
    if icd_target_map:
        print(f"  ICD Target Map: {len(icd_target_map)} entries")

    try:
        # Resolve threads per worker from CLI or env
        threads_env = os.getenv('PGX_THREADS_PER_WORKER')
        threads_per_worker = args.threads if (args.threads and args.threads > 0) else int(threads_env) if threads_env and threads_env.isdigit() and int(threads_env) > 0 else 1

        # Optional memory cap to keep memory low in DuckDB
        if args.duckdb_mem_limit:
            os.environ['PGX_DUCKDB_MEMORY_LIMIT'] = str(args.duckdb_mem_limit)
        # Control no-merge and target size via env for workers
        os.environ['PGX_NO_MERGE'] = "1" if args.no_merge else "0"
        os.environ['PGX_TARGET_FILE_SIZE_MB'] = str(int(args.target_file_size_mb)) if args.target_file_size_mb and args.target_file_size_mb > 0 else "512"

        # Resolve workers by dataset (CLI > env > defaults)
        workers_override = args.workers if (args.workers and args.workers > 0) else None
        med_env = os.getenv('PGX_WORKERS_MEDICAL')
        ph_env  = os.getenv('PGX_WORKERS_PHARMACY')
        workers_medical = (
            workers_override or
            (args.workers_medical if (args.workers_medical and args.workers_medical > 0) else int(med_env) if med_env and med_env.isdigit() and int(med_env) > 0 else 16)
        )
        workers_pharmacy = (
            workers_override or
            (args.workers_pharmacy if (args.workers_pharmacy and args.workers_pharmacy > 0) else int(ph_env) if ph_env and ph_env.isdigit() and int(ph_env) > 0 else 48)
        )

        if args.show_progress:
            # Just show progress and exit
            # Medical partitions
            med_parts = _enumerate_partitions(GOLD_MEDICAL_GLOB)
            if years:
                med_parts = med_parts[med_parts["event_year"].isin(set(years))] if not med_parts.empty else med_parts
            if age_bands:
                med_parts = med_parts[med_parts["age_band"].isin(set(age_bands))] if not med_parts.empty else med_parts
            completed = 0; pending = 0; chunk_existing = 0; chunk_expected = 0
            for _, row in med_parts.iterrows():
                fname = row["filename"]
                if _marker_exists(fname, args.checkpoint_suffix):
                    completed += 1
                else:
                    pending += 1
                    if args.chunked and args.chunk_rows and args.chunk_rows > 0:
                        pref = _staging_prefix_for(fname, args.staging_suffix)
                        chunk_existing += _list_staging_chunks(pref)
                        if args.progress_deep:
                            try:
                                conn2 = create_duckdb_conn(threads=1)
                                total_rows = conn2.sql(f"SELECT COUNT(*) FROM read_parquet('{fname}')").fetchone()[0]
                                try:
                                    conn2.close()
                                except Exception:
                                    pass
                                chunk_expected += math.ceil(total_rows / float(args.chunk_rows)) if total_rows else 0
                            except Exception:
                                pass
            if args.chunked and args.chunk_rows and args.chunk_rows > 0:
                print(f"[medical] Progress: partitions completed={completed}, pending={pending}, chunks existing={chunk_existing}" + (f", expected={chunk_expected}" if args.progress_deep else ""))
            else:
                print(f"[medical] Progress: partitions completed={completed}, pending={pending}")
            # Pharmacy partitions (only if drug_map provided)
            if drug_map:
                ph_parts = _enumerate_partitions(GOLD_PHARMACY_GLOB)
                if years:
                    ph_parts = ph_parts[ph_parts["event_year"].isin(set(years))] if not ph_parts.empty else ph_parts
                if age_bands:
                    ph_parts = ph_parts[ph_parts["age_band"].isin(set(age_bands))] if not ph_parts.empty else ph_parts
                completed = 0; pending = 0; chunk_existing = 0; chunk_expected = 0
                for _, row in ph_parts.iterrows():
                    fname = row["filename"]
                    if _marker_exists(fname, args.checkpoint_suffix):
                        completed += 1
                    else:
                        pending += 1
                        if args.chunked and args.chunk_rows and args.chunk_rows > 0:
                            pref = _staging_prefix_for(fname, args.staging_suffix)
                            chunk_existing += _list_staging_chunks(pref)
                            if args.progress_deep:
                                try:
                                    conn2 = create_duckdb_conn(threads=1)
                                    total_rows = conn2.sql(f"SELECT COUNT(*) FROM read_parquet('{fname}')").fetchone()[0]
                                    try:
                                        conn2.close()
                                    except Exception:
                                        pass
                                    chunk_expected += math.ceil(total_rows / float(args.chunk_rows)) if total_rows else 0
                                except Exception:
                                    pass
                if args.chunked and args.chunk_rows and args.chunk_rows > 0:
                    print(f"[pharmacy] Progress: partitions completed={completed}, pending={pending}, chunks existing={chunk_existing}" + (f", expected={chunk_expected}" if args.progress_deep else ""))
                else:
                    print(f"[pharmacy] Progress: partitions completed={completed}, pending={pending}")
            return

        apply_mappings_to_medical(
            icd_map, cpt_map, icd_target_map,
            years=years, age_bands=age_bands,
            apply=(not args.no_apply),
            workers=workers_medical,
            threads_per_worker=threads_per_worker,
            log_cpu=args.log_cpu,
            log_s3=args.log_s3,
            resume=args.resume,
            marker_suffix=args.checkpoint_suffix,
            chunked=args.chunked,
            chunk_rows=args.chunk_rows,
            staging_suffix=args.staging_suffix,
        )
        apply_mappings_to_pharmacy(
            drug_map,
            years=years, age_bands=age_bands,
            apply=(not args.no_apply),
            workers=workers_pharmacy,
            threads_per_worker=threads_per_worker,
            log_cpu=args.log_cpu,
            log_s3=args.log_s3,
            resume=args.resume,
            marker_suffix=args.checkpoint_suffix,
            chunked=args.chunked,
            chunk_rows=args.chunk_rows,
            staging_suffix=args.staging_suffix,
        )
        print("✅ Completed code updates.")
    finally:
        pass


if __name__ == "__main__":
    main()


