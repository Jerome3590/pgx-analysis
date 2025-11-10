#!/usr/bin/env python3
"""
Simplified DuckDB Utilities - Version 1997 + 12
Removed complex chaining to fix memory_limit issues
"""

import os
import tempfile
import atexit
import uuid
import time
import glob
import shutil
import duckdb
from typing import Optional, Dict, Any

# Global state for worker temp directory management
_cleanup_registered = False
_worker_temp_dir_cache = None

def create_simple_duckdb_connection(logger, tmp_dir: Optional[str] = None, s3_region: str = "us-east-1"):
    """Create a simple DuckDB connection without complex chaining"""
    try:
        # Create basic connection
        conn = duckdb.connect(database=':memory:')
        
        # Basic S3 setup only
        conn.sql("INSTALL httpfs; LOAD httpfs;")
        conn.sql("INSTALL aws; LOAD aws;")
        conn.sql("CALL load_aws_credentials();")
        conn.sql(f"SET s3_region='{s3_region}'")
        conn.sql("SET s3_url_style='path'")
        
        # Configure S3 uploader settings for large files
        # ⚠️ DISABLED: These parameters cause memory issues - let DuckDB auto-configure
        #conn.sql("SET s3_uploader_max_filesize='5368709120'")  # 5GB max file size
        #conn.sql("SET s3_uploader_max_parts_per_file='10000'")  # Max parts per file
        
        # Set temp directory if provided
        if tmp_dir:
            os.makedirs(tmp_dir, exist_ok=True)
            conn.sql(f"SET temp_directory = '{tmp_dir}'")
        
        # Set threads to 1 for multiprocessing environments (multiple workers)
        # Each worker process gets 1 thread to avoid over-subscription
        conn.sql("SET threads = 1")
        
        # Let DuckDB auto-detect memory limit
        logger.info("✅ Simple DuckDB connection created - 1 thread per worker (for multiprocessing)")
        return conn
        
    except Exception as e:
        logger.error(f"❌ Failed to create DuckDB connection: {e}")
        raise

def get_duckdb_connection(tmp_dir: Optional[str] = None, s3_region: str = "us-east-1", logger=None):
    """Get a simple DuckDB connection - wrapper for backward compatibility"""
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    return create_simple_duckdb_connection(logger, tmp_dir, s3_region)

def check_memory_usage(conn, logger, step_name: str = "Memory check"):
    """Check current DuckDB memory usage (bytes) and log it."""
    try:
        # Use duckdb_memory() function to get accurate memory usage
        result = conn.sql("SELECT SUM(memory_usage_bytes) as total_memory FROM duckdb_memory()").fetchone()
        if result and result[0] is not None:
            memory_usage = result[0]
            # Convert bytes to human readable format
            if memory_usage >= 1024**3:  # GB
                memory_str = f"{memory_usage / (1024**3):.1f} GiB"
            elif memory_usage >= 1024**2:  # MB
                memory_str = f"{memory_usage / (1024**2):.1f} MiB"
            elif memory_usage >= 1024:  # KB
                memory_str = f"{memory_usage / 1024:.1f} KiB"
            else:
                memory_str = f"{memory_usage} bytes"
            
            logger.info(f"📊 {step_name}: Memory usage: {memory_str}")
            return (memory_usage,)  # match original return style (tuple with one element)
        else:
            logger.warning(f"⚠️ Could not get memory usage from duckdb_memory()")
            return (0,)
    except Exception as e:
        logger.warning(f"⚠️ Memory check failed: {e}")
        return (0,)

def get_duckdb_info(conn, logger):
    """Get basic DuckDB configuration info"""
    try:
        # Get DuckDB version
        version = conn.sql("SELECT version()").fetchone()[0]
        logger.info(f"📊 DuckDB version: {version}")
        
        # Get memory usage
        check_memory_usage(conn, logger, "Current memory usage")
        
        # Get thread count
        threads = conn.sql("SELECT current_setting('threads')").fetchone()[0]
        logger.info(f"📊 Threads: {threads}")
        
        # Get memory limit
        memory_limit = conn.sql("SELECT current_setting('memory_limit')").fetchone()[0]
        logger.info(f"📊 Memory limit: {memory_limit}")
        
        return {
            "version": version,
            "threads": threads,
            "memory_limit": memory_limit
        }
        
    except Exception as e:
        logger.warning(f"⚠️ Could not get DuckDB info: {e}")
        return {}

def close_duckdb_connection(conn, logger):
    """Close DuckDB connection safely"""
    try:
        if conn:
            conn.close()
            logger.info("✅ DuckDB connection closed")
    except Exception as e:
        logger.warning(f"⚠️ Could not close DuckDB connection: {e}")

# Legacy function names for backward compatibility
# Removed init_duckdb() to avoid name collision with clean_medical.py

def tune_duckdb_for_mp(conn, logger, memory_limit: str = "2GB", threads: int = 1):
    """Legacy function - no-op for simplified version"""
    logger.info("📊 Simplified DuckDB - no manual tuning needed")
    return conn


# ===== Worker Temp Directory Management =====

def _cleanup_worker_temp_dir():
    """
    Clean up worker temp directory and temp DB files.
    This is registered with atexit to run once per worker process on exit.
    Uses the cached worker temp dir if available, otherwise tries to find it.
    """
    try:
        global _worker_temp_dir_cache
        
        # Use cached worker temp dir if available
        if _worker_temp_dir_cache and os.path.exists(_worker_temp_dir_cache):
            worker_temp_dir = _worker_temp_dir_cache
        else:
            # Fallback: try to find worker temp dir by pattern (less reliable)
            # This shouldn't normally be needed, but provides a safety net
            return
        
        if os.path.exists(worker_temp_dir):
            # Remove temp DB file if it exists
            temp_db_path = os.path.join(worker_temp_dir, "duckdb_temp.db")
            if os.path.exists(temp_db_path):
                try:
                    os.remove(temp_db_path)
                except Exception:
                    pass
            # Remove temp DB WAL file if it exists
            temp_db_wal = os.path.join(worker_temp_dir, "duckdb_temp.db.wal")
            if os.path.exists(temp_db_wal):
                try:
                    os.remove(temp_db_wal)
                except Exception:
                    pass
            # Remove any other temp DB files (with UUID names)
            try:
                for db_file in glob.glob(os.path.join(worker_temp_dir, "duckdb_temp_*.db")):
                    try:
                        os.remove(db_file)
                    except Exception:
                        pass
                for wal_file in glob.glob(os.path.join(worker_temp_dir, "duckdb_temp_*.db.wal")):
                    try:
                        os.remove(wal_file)
                    except Exception:
                        pass
            except Exception:
                pass
            shutil.rmtree(worker_temp_dir)
    except Exception:
        pass


def get_worker_temp_dir() -> str:
    """
    Get or create a worker-specific temp directory for DuckDB.
    Prefers NVMe if available, falls back to /tmp.
    Uses PID + unique identifier to ensure uniqueness even with fork multiprocessing.
    Registers cleanup with atexit to run once per worker process on exit.
    Caches result per process to avoid repeated calls.
    """
    global _cleanup_registered, _worker_temp_dir_cache
    
    # Return cached value if already computed
    if _worker_temp_dir_cache is not None:
        return _worker_temp_dir_cache
    
    # Generate a unique identifier for this worker (PID + timestamp + random to ensure uniqueness)
    # This is especially important with fork multiprocessing where PIDs might be reused
    unique_id = f"{os.getpid()}_{int(time.time() * 1000000)}_{uuid.uuid4().hex[:8]}"
    
    # Prefer NVMe temp directory if available
    nvme_tmp_base = "/mnt/nvme/duckdb_tmp"
    if os.path.exists("/mnt/nvme"):
        # Ensure base directory exists
        os.makedirs(nvme_tmp_base, exist_ok=True)
        worker_temp_dir = os.path.join(nvme_tmp_base, f"worker_{unique_id}")
    else:
        # Fall back to /tmp
        worker_temp_dir = f"/tmp/duckdb_worker_{unique_id}"
    
    # Clean up any existing directory or temp DB files to avoid lock conflicts
    if os.path.exists(worker_temp_dir):
        try:
            # Remove temp DB file if it exists (to avoid lock conflicts)
            temp_db_path = os.path.join(worker_temp_dir, "duckdb_temp.db")
            if os.path.exists(temp_db_path):
                try:
                    os.remove(temp_db_path)
                except Exception:
                    pass
            # Remove temp DB WAL file if it exists
            temp_db_wal = os.path.join(worker_temp_dir, "duckdb_temp.db.wal")
            if os.path.exists(temp_db_wal):
                try:
                    os.remove(temp_db_wal)
                except Exception:
                    pass
            shutil.rmtree(worker_temp_dir)
        except Exception:
            pass
    
    os.makedirs(worker_temp_dir, exist_ok=True)
    
    # Cache the result
    _worker_temp_dir_cache = worker_temp_dir
    
    # Register cleanup to run once per worker process on exit (not per partition)
    if not _cleanup_registered:
        atexit.register(_cleanup_worker_temp_dir)
        _cleanup_registered = True
    
    return worker_temp_dir


def cleanup_old_duckdb_temp_dirs(max_age_hours: int = 1) -> int:
    """
    Clean up old DuckDB worker temp directories from previous failed runs.
    Returns number of directories cleaned.
    """
    cleaned_count = 0
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    # Clean up from NVMe
    nvme_tmp_base = "/mnt/nvme/duckdb_tmp"
    if os.path.exists(nvme_tmp_base):
        try:
            for item in os.listdir(nvme_tmp_base):
                item_path = os.path.join(nvme_tmp_base, item)
                if os.path.isdir(item_path) and item.startswith("worker_"):
                    try:
                        # Check age
                        if os.path.getmtime(item_path) < current_time - max_age_seconds:
                            shutil.rmtree(item_path)
                            cleaned_count += 1
                    except Exception:
                        pass
        except Exception:
            pass
    
    # Clean up from /tmp
    try:
        old_dirs = glob.glob("/tmp/duckdb_worker_*")
        for old_dir in old_dirs:
            if os.path.isdir(old_dir):
                try:
                    # Check age
                    if os.path.getmtime(old_dir) < current_time - max_age_seconds:
                        shutil.rmtree(old_dir)
                        cleaned_count += 1
                except Exception:
                    pass
    except Exception:
        pass
    
    if cleaned_count > 0:
        print(f"🧹 Cleaned up {cleaned_count} old DuckDB temp directories")
    
    return cleaned_count


# ===== Memory Limit Calculation =====

def calculate_memory_limit_per_worker(total_workers: Optional[int] = None, total_memory_gb: Optional[float] = None) -> str:
    """
    Calculate per-worker memory limit to avoid aggregate oversubscription.
    
    Args:
        total_workers: Total number of workers (if None, tries to detect from env)
        total_memory_gb: Total available memory in GB (if None, tries to detect or uses conservative default)
    
    Returns:
        Memory limit string (e.g., '2GB', '1GB')
    """
    # Check if explicitly set via environment
    explicit_limit = os.getenv('PGX_DUCKDB_MEMORY_LIMIT')
    if explicit_limit:
        return explicit_limit
    
    # Try to detect worker count
    if total_workers is None:
        # Try to get from environment (medical or pharmacy workers)
        medical_workers = os.getenv('PGX_WORKERS_MEDICAL')
        pharmacy_workers = os.getenv('PGX_WORKERS_PHARMACY')
        if medical_workers and medical_workers.isdigit():
            total_workers = int(medical_workers)
        elif pharmacy_workers and pharmacy_workers.isdigit():
            total_workers = int(pharmacy_workers)
        else:
            # Conservative default: assume 24 workers
            total_workers = 24
    
    # Try to detect total available memory (container-aware)
    if total_memory_gb is None:
        try:
            # First, try to read cgroup memory limit (container-aware)
            cgroup_memory_limit = None
            try:
                # Try cgroup v2 first (newer systems)
                try:
                    with open('/sys/fs/cgroup/memory.max', 'r') as f:
                        cgroup_limit_str = f.read().strip()
                        if cgroup_limit_str != 'max' and cgroup_limit_str.isdigit():
                            cgroup_memory_limit = int(cgroup_limit_str) / (1024**3)
                except (FileNotFoundError, ValueError, IOError):
                    # Try cgroup v1
                    try:
                        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                            cgroup_limit_bytes = int(f.read().strip())
                            # cgroup limit of 9223372036854771712 (max int64) means unlimited, use host memory
                            if cgroup_limit_bytes < 9223372036854771712:
                                cgroup_memory_limit = cgroup_limit_bytes / (1024**3)
                    except (FileNotFoundError, ValueError, IOError):
                        pass
            except Exception:
                pass
                
            # Use cgroup limit if available, otherwise fall back to psutil
            if cgroup_memory_limit and cgroup_memory_limit > 0:
                total_memory_gb = cgroup_memory_limit
            else:
                import psutil
                total_memory_gb = psutil.virtual_memory().total / (1024**3)
        except (ImportError, Exception):
            # Conservative default: assume 64GB total
            total_memory_gb = 64.0
    
    # Calculate per-worker limit: reserve 20% for OS/system, divide rest by workers
    # Use 80% of total memory, divided by workers, with a minimum of 512MB and maximum of 4GB
    available_memory_gb = total_memory_gb * 0.8
    per_worker_gb = available_memory_gb / max(1, total_workers)
    
    # Clamp between 512MB and 4GB (respects container limits via total_memory_gb)
    per_worker_gb = max(0.5, min(4.0, per_worker_gb))
    
    # Round to nearest 0.5GB for cleaner values
    per_worker_gb = round(per_worker_gb * 2) / 2
    
    return f"{per_worker_gb:.1f}GB"


# ===== Advanced DuckDB Connection Creation =====

def create_duckdb_conn(threads: Optional[int] = 1, tmp_dir: Optional[str] = None, 
                       total_workers: Optional[int] = None, use_temp_db: bool = False):
    """
    Create a DuckDB connection with proper temp directory setup and dynamic memory limits.
    
    Args:
        threads: Number of threads for DuckDB (default 1)
        tmp_dir: Optional temp directory. If None, uses worker-specific temp dir.
        total_workers: Total number of workers for dynamic memory calculation
        use_temp_db: If True, use temp file-based DB instead of :memory: (for high memory pressure)
    """
    # Use provided tmp_dir or get worker-specific one
    if tmp_dir is None:
        tmp_dir = get_worker_temp_dir()
    
    # Set environment variable for DuckDB
    os.environ["DUCKDB_TMP_DIRECTORY"] = tmp_dir
    
    # Enable DuckDB profiling if requested (for optimization diagnostics)
    enable_profiling = os.getenv('PGX_ENABLE_DUCKDB_PROFILING', '0') == '1'
    
    # Choose database: temp file for high memory pressure, or in-memory
    if use_temp_db:
        # Use a unique DB file per connection to avoid lock conflicts when processing multiple partitions
        # This is especially important with fork multiprocessing where workers might share state
        unique_db_name = f"duckdb_temp_{uuid.uuid4().hex[:16]}.db"
        db_path = os.path.join(tmp_dir, unique_db_name)
        # Clean up any existing temp DB file to avoid lock conflicts (especially with fork mode)
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass
        # Also clean up WAL file if it exists
        wal_path = db_path + ".wal"
        if os.path.exists(wal_path):
            try:
                os.remove(wal_path)
            except Exception:
                pass
        con = duckdb.connect(database=db_path)
    else:
        con = duckdb.connect(database=":memory:")
    
    # Load extensions (already installed, just need to load per-connection)
    # Note: INSTALL is not needed - extensions are pre-installed on the system
    try:
        import sys
        worker_pid = os.getpid()
        
        # Load httpfs extension (fast, per-connection operation)
        con.sql("LOAD httpfs;")
        print(f"[duckdb-{worker_pid}] ✅ httpfs loaded", flush=True)
        
        # Load aws extension (fast, per-connection operation)
        con.sql("LOAD aws;")
        print(f"[duckdb-{worker_pid}] ✅ aws loaded", flush=True)
        
        # Load AWS credentials (per-connection operation)
        print(f"[duckdb-{worker_pid}] Loading AWS credentials...", flush=True)
        con.sql("CALL load_aws_credentials();")
        print(f"[duckdb-{worker_pid}] ✅ AWS credentials loaded", flush=True)
        
        # Set S3 configuration (per-connection)
        con.sql("SET s3_region='us-east-1'")
        con.sql("SET s3_url_style='path'")
        print(f"[duckdb-{worker_pid}] ✅ S3 configuration set", flush=True)
    except Exception as e:
        import sys
        worker_pid = os.getpid()
        print(f"[duckdb-{worker_pid}] ❌ ERROR during extension/credential setup: {e}", file=sys.stderr, flush=True)
        raise
    try:
        worker_pid = os.getpid()
        import sys
        
        # Set temp directory in DuckDB
        print(f"[duckdb-{worker_pid}] Setting temp directory...", flush=True)
        os.makedirs(tmp_dir, exist_ok=True)
        con.sql(f"SET temp_directory = '{tmp_dir}'")
        print(f"[duckdb-{worker_pid}] ✅ Temp directory set: {tmp_dir}", flush=True)
        
        # Disable progress bar to avoid noisy ETA lines in logs/notebooks
        con.sql("PRAGMA enable_progress_bar=false")
        
        # Enable profiling if requested (for optimization diagnostics)
        if enable_profiling:
            con.sql("PRAGMA enable_profiling='json'")
            print(f"[duckdb-{worker_pid}] Profiling enabled (JSON mode) for optimization diagnostics", flush=True)
        
        # Default to 1 thread per worker to prevent over-subscription; override if provided
        if threads is None or threads <= 0:
            threads = 1
        print(f"[duckdb-{worker_pid}] Setting threads={threads}...", flush=True)
        con.sql(f"PRAGMA threads={threads}")
        print(f"[duckdb-{worker_pid}] ✅ Threads set", flush=True)
        
        # Note: s3_max_connections is not a valid DuckDB parameter (removed)
        # S3 connection concurrency is managed automatically by DuckDB
        
        # S3 reliability tuning - increase timeouts and retries for large writes
        print(f"[duckdb-{worker_pid}] Configuring S3 timeouts and retries...", flush=True)
        con.sql("SET http_timeout=300000")  # 5 minute timeout (up from 30s default)
        con.sql("SET http_retries=5")  # Retry failed requests
        con.sql("SET http_retry_wait_ms=1000")  # Wait 1s between retries
        con.sql("SET s3_uploader_max_filesize='100GB'")  # Support very large files
        print(f"[duckdb-{worker_pid}] ✅ S3 configuration complete", flush=True)
        
        # Dynamic memory limit calculation to avoid aggregate oversubscription
        print(f"[duckdb-{worker_pid}] Calculating memory limit...", flush=True)
        mem_limit = calculate_memory_limit_per_worker(total_workers=total_workers)
        con.sql(f"SET memory_limit='{mem_limit}'")
        print(f"[duckdb-{worker_pid}] ✅ Memory limit set: {mem_limit}", flush=True)
        
        # Log memory limit for debugging (only in worker processes, not enumeration)
        if total_workers and total_workers > 1:
            # Also log system memory info for diagnostics
            try:
                import psutil
                mem = psutil.virtual_memory()
                print(f"[duckdb-{worker_pid}] System memory: total={mem.total/(1024**3):.1f}GB, available={mem.available/(1024**3):.1f}GB", flush=True)
            except Exception:
                pass
    except Exception as e:
        import sys
        worker_pid = os.getpid()
        print(f"[duckdb-{worker_pid}] ❌ ERROR during DuckDB configuration: {e}", file=sys.stderr, flush=True)
        # Don't silently pass - re-raise to see the error
        raise
    return con


# ===== Mapping Loading Functions =====

def load_mapping_into_duckdb_chunked(con: duckdb.DuckDBPyConnection, mapping: Dict[str, str], table_name: str, transform_key=None, chunk_size: int = 10000) -> None:
    """
    Load a mapping dictionary into DuckDB using chunked streaming inserts to avoid
    materializing the full list in memory.
    
    Args:
        con: DuckDB connection
        mapping: Dictionary mapping variant -> canonical
        table_name: Name of the temp table to create
        transform_key: Optional function to transform keys before insertion
        chunk_size: Number of items to insert per batch (default 10k)
    """
    con.sql(f"DROP TABLE IF EXISTS {table_name}")
    con.sql(f"CREATE TEMP TABLE {table_name}(variant VARCHAR, canonical VARCHAR)")
    
    # Use chunked streaming to avoid materializing full list
    items_iter = mapping.items()
    chunk = []
    
    for key, value in items_iter:
        if transform_key:
            key = transform_key(key)
        chunk.append((key, value))
        
        if len(chunk) >= chunk_size:
            con.executemany(f"INSERT INTO {table_name} VALUES (?, ?)", chunk)
            chunk.clear()  # Free memory immediately
    
    # Insert remaining items
    if chunk:
        con.executemany(f"INSERT INTO {table_name} VALUES (?, ?)", chunk)
        chunk.clear()


def load_mapping_from_file_into_duckdb(con: duckdb.DuckDBPyConnection, file_path: str, table_name: str, transform_key=None) -> None:
    """
    Load a mapping directly from JSON/Parquet file into DuckDB without loading into Python memory.
    This is the most memory-efficient approach for very large mappings.
    
    Args:
        con: DuckDB connection
        file_path: Path to JSON or Parquet file (local or s3://)
        table_name: Name of the temp table to create
        transform_key: Optional SQL expression to transform keys (e.g., "UPPER(REPLACE(REPLACE(variant, '.', ''), ' ', ''))")
    """
    con.sql(f"DROP TABLE IF EXISTS {table_name}")
    
    if file_path.endswith('.parquet'):
        # Load from Parquet
        if transform_key:
            con.sql(f"""
                CREATE TEMP TABLE {table_name} AS
                SELECT {transform_key} AS variant, canonical
                FROM read_parquet('{file_path}')
            """)
        else:
            con.sql(f"""
                CREATE TEMP TABLE {table_name} AS
                SELECT variant, canonical
                FROM read_parquet('{file_path}')
            """)
    else:
        # Load from JSON (assumes JSON is array of objects with variant/canonical keys, or object with key-value pairs)
        # Try to auto-detect structure
        if transform_key:
            # For JSON objects (key-value pairs), we need to use json_each or similar
            # DuckDB's read_json_auto can handle both arrays and objects
            con.sql(f"""
                CREATE TEMP TABLE {table_name} AS
                SELECT {transform_key} AS variant, canonical
                FROM read_json_auto('{file_path}')
            """)
        else:
            con.sql(f"""
                CREATE TEMP TABLE {table_name} AS
                SELECT variant, canonical
                FROM read_json_auto('{file_path}')
            """)
