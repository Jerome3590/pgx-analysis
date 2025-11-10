# apcd_clean.py
import os, sys, time, argparse, concurrent.futures as cf
from typing import List, Tuple
import importlib, subprocess
import logging
# Removed duckdb import - using pure S3 listing for discovery

# Project path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)


# ----------------- Optional import-mode targets -----------------
try:
    from pharmacy_clean import main as run_pharmacy  # optional (import mode)
except Exception:
    run_pharmacy = None
try:
    from medical_clean import main as run_medical    # optional (import mode)
except Exception:
    run_medical = None

# ----------------- Logging (S3-capable) -----------------
# Uses your helper to create an in-memory buffer + console and save to S3.
from helpers_1997_13.logging_utils import setup_logging, save_logs_to_s3, save_logs_immediate
from helpers_1997_13.s3_utils import convert_raw_to_imputed_path


# make sure print/logging is line-buffered even without a TTY
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# Global logger handle used by helpers defined below; will be replaced inside main()
log = logging.getLogger("orchestrator")

# ----------------- Config -----------------
DEFAULT_RAW_PHARMACY = "s3://pgxdatalake/silver/pharmacy/*.parquet"
DEFAULT_RAW_MEDICAL  = "s3://pgxdatalake/silver/medical/*.parquet"

AGE_BANDS = ["0-12","13-24","25-44","45-54","55-64","65-74","75-84","85-94","95-114"]

AGE_BAND_CASE = """
CASE
  WHEN TRY_CAST(member_age_dos AS INTEGER) BETWEEN 0  AND 12  THEN '0-12'
  WHEN TRY_CAST(member_age_dos AS INTEGER) BETWEEN 13 AND 24  THEN '13-24'
  WHEN TRY_CAST(member_age_dos AS INTEGER) BETWEEN 25 AND 44  THEN '25-44'
  WHEN TRY_CAST(member_age_dos AS INTEGER) BETWEEN 45 AND 54  THEN '45-54'
  WHEN TRY_CAST(member_age_dos AS INTEGER) BETWEEN 55 AND 64  THEN '55-64'
  WHEN TRY_CAST(member_age_dos AS INTEGER) BETWEEN 65 AND 74  THEN '65-74'
  WHEN TRY_CAST(member_age_dos AS INTEGER) BETWEEN 75 AND 84  THEN '75-84'
  WHEN TRY_CAST(member_age_dos AS INTEGER) BETWEEN 85 AND 94  THEN '85-94'
  WHEN TRY_CAST(member_age_dos AS INTEGER) BETWEEN 95 AND 114 THEN '95-114'
  ELSE 'Other'
END
"""

# ----------------- Discovery -----------------
def _build_year_where(min_year: int | None, max_year: int | None) -> str:
    if min_year is not None and max_year is not None:
        return f"AND SUBSTR(CAST(incurred_date AS VARCHAR),1,4) BETWEEN '{min_year}' AND '{max_year}'"
    if min_year is not None:
        return f"AND SUBSTR(CAST(incurred_date AS VARCHAR),1,4) >= '{min_year}'"
    if max_year is not None:
        return f"AND SUBSTR(CAST(incurred_date AS VARCHAR),1,4) <= '{max_year}'"
    return ""


def discover_from_pharmacy(raw_path: str, min_year: int = None, max_year: int = None, min_rows: int = 1) -> List[Tuple[str, int, int]]:
    """Discover age_band/year pairs from silver/imputed partitioned data using S3 listing"""
    log.info("Discovery (pharmacy) starting...")
    t_start = time.perf_counter()
    
    try:
        import boto3
        from urllib.parse import urlparse
        
        # Convert raw path to silver/imputed path using centralized utility
        # raw: s3://pgxdatalake/silver/pharmacy/*.parquet
        # silver: s3://pgxdatalake/silver/imputed/pharmacy_partitioned
        try:
            silver_path = convert_raw_to_imputed_path(raw_path, 'pharmacy')
        except Exception:
            # Fallback to original path if conversion fails
            silver_path = raw_path
        
        log.info(f"🔍 Raw path: {raw_path}")
        log.info(f"🔍 Looking for partitions in: {silver_path}")
        
        # Parse S3 URL
        parsed = urlparse(silver_path)
        bucket = parsed.netloc
        prefix = parsed.path.lstrip('/')
        
        log.info(f"🔍 S3 bucket: {bucket}, prefix: {prefix}")
        
        # Create S3 client
        s3_client = boto3.client('s3')
        
        # List objects with pagination
        pairs = []
        paginator = s3_client.get_paginator('list_objects_v2')
        
        total_objects = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                total_objects += len(page['Contents'])
                for obj in page['Contents']:
                    key = obj['Key']
                    
                    # Extract age_band and event_year from partitioned path
                    # Expected format: age_band=XX-YY/event_year=YYYY/...
                    if 'age_band=' in key and 'event_year=' in key:
                        try:
                            # Extract age_band (e.g., "65-74")
                            age_start = key.find('age_band=') + 9
                            age_end = key.find('/', age_start)
                            age_band = key[age_start:age_end]
                            
                            # Extract event_year (e.g., "2019")
                            year_start = key.find('event_year=') + 11
                            year_end = key.find('/', year_start)
                            event_year = int(key[year_start:year_end])
                            
                            # Filter by year range (handle None values)
                            if min_year is not None and event_year < min_year:
                                continue
                            if max_year is not None and event_year > max_year:
                                continue
                            pairs.append((age_band, event_year))
                                
                        except (ValueError, IndexError) as e:
                            # Skip malformed paths
                            log.debug(f"Skipping malformed path: {key} ({e})")
                            continue
        
        log.info(f"🔍 Found {total_objects} total objects, extracted {len(pairs)} partition pairs")
        
        # Remove duplicates and sort
        pairs = sorted(list(set(pairs)))
        
        # Estimate row counts (simplified - in practice you might want to sample)
        estimated_rows = [(ab, y, 1000000) for ab, y in pairs]  # Placeholder count
        
        log.info(f"Discovery (pharmacy) returned {len(pairs)} partitions in {time.perf_counter() - t_start:.2f}s")
        return estimated_rows
        
    except Exception as e:
        log.error(f"Discovery failed: {e}", exc_info=True)
        return []


def discover_from_medical(raw_path: str, min_year: int=None, max_year: int=None, min_rows: int=1):
    """Discover age_band/year pairs from silver/imputed partitioned data using S3 listing"""
    log.info("Discovery (medical) starting...")
    t_start = time.perf_counter()
    
    try:
        import boto3
        from urllib.parse import urlparse
        
        # Convert raw path to silver/imputed path using centralized utility
        # raw: s3://pgxdatalake/silver/medical/*.parquet
        # silver: s3://pgxdatalake/silver/imputed/medical_partitioned
        try:
            silver_path = convert_raw_to_imputed_path(raw_path, 'medical')
        except Exception:
            # Fallback to original path if conversion fails
            silver_path = raw_path
        
        log.info(f"🔍 Looking for partitions in: {silver_path}")
        
        # Parse S3 URL
        parsed = urlparse(silver_path)
        bucket = parsed.netloc
        prefix = parsed.path.lstrip('/')
        
        log.info(f"🔍 S3 bucket: {bucket}, prefix: {prefix}")
        
        # Create S3 client
        s3_client = boto3.client('s3')
        
        # List objects with pagination
        pairs = []
        paginator = s3_client.get_paginator('list_objects_v2')
        
        total_objects = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                total_objects += len(page['Contents'])
                for obj in page['Contents']:
                    key = obj['Key']
                    
                    # Extract age_band and event_year from partitioned path
                    # Expected format: age_band=XX-YY/event_year=YYYY/...
                    if 'age_band=' in key and 'event_year=' in key:
                        try:
                            # Extract age_band (e.g., "65-74")
                            age_start = key.find('age_band=') + 9
                            age_end = key.find('/', age_start)
                            age_band = key[age_start:age_end]
                            
                            # Extract event_year (e.g., "2019")
                            year_start = key.find('event_year=') + 11
                            year_end = key.find('/', year_start)
                            event_year = int(key[year_start:year_end])
                            
                            # Filter by year range (handle None values)
                            if min_year is not None and event_year < min_year:
                                continue
                            if max_year is not None and event_year > max_year:
                                continue
                            pairs.append((age_band, event_year))
                                
                        except (ValueError, IndexError) as e:
                            # Skip malformed paths
                            log.debug(f"Skipping malformed path: {key} ({e})")
                            continue
        
        log.info(f"🔍 Found {total_objects} total objects, extracted {len(pairs)} partition pairs")
        
        # Remove duplicates and sort
        pairs = sorted(list(set(pairs)))
        
        # Estimate row counts (simplified - in practice you might want to sample)
        estimated_rows = [(ab, y, 1000000) for ab, y in pairs]  # Placeholder count
        
        log.info(f"Discovery (medical) returned {len(pairs)} partitions in {time.perf_counter() - t_start:.2f}s")
        return estimated_rows
        
    except Exception as e:
        log.error(f"Discovery failed: {e}", exc_info=True)
        return []


# ----------------- Pairs parsing -----------------
def parse_pairs_line(line: str) -> Tuple[str, int]:
    s = line.strip()
    if not s or s.startswith("#"):
        raise ValueError("skip")
    t = s.replace(" ", "").replace("\t", "")
    if "age_band=" in t and "/event_year=" in t:
        left, right = t.split("/event_year=")
        ab = left.split("age_band=")[1]
        return ab, int(right)
    if "," in t:
        ab, yr = t.split(",", 1)
        return ab, int(yr)
    if "/" in t:
        ab, yr = t.split("/", 1)
        return ab, int(yr)
    parts = s.split()
    if len(parts) == 2:
        return parts[0], int(parts[1])
    raise ValueError(f"Unrecognized pair format: {line!r}")


def load_pairs_override(pairs: str=None, pairs_file: str=None) -> List[Tuple[str,int]]:
    out: List[Tuple[str,int]] = []
    if pairs_file:
        with open(pairs_file, "r") as f:
            for ln in f:
                try:
                    out.append(parse_pairs_line(ln))
                except Exception:
                    continue
    if pairs:
        for chunk in pairs.split(";"):
            if chunk.strip():
                try:
                    out.append(parse_pairs_line(chunk))
                except Exception:
                    continue
    # de-dup while preserving order
    seen = set(); uniq = []
    for ab, yr in out:
        key = (ab, int(yr))
        if key not in seen:
            seen.add(key); uniq.append(key)
    return uniq


# ----------------- Worker task -----------------
def run_task(job: str, ab: str, yr: int, lookahead: int,
             retries: int = 1, backoff: float = 2.0,
             run_mode: str = "import",
             pharmacy_module: str = "pharmacy_clean",
             medical_module: str = "medical_clean",
             pharmacy_script: str = "",
             medical_script: str = "",
             demographics_lookup: str = "",
             medical_input: str = "",
             pharmacy_input: str = "",
             output_root: str = "",
             python_bin: str = sys.executable):
    # Use central logger for all subprocesses
    from helpers_1997_13.logging_utils import setup_logging
    worker_log, _ = setup_logging("orchestrator_worker", ab, yr)
    worker_log.info(f"START {job} {ab}/{yr} mode={run_mode}")
    # Ensure worker isolation with better temp directory setup
    import tempfile
    import shutil
    
    # Create unique temp directory for this worker (replace all special chars with underscores)
    safe_ab = ab.replace('-', '_').replace('+', '_').replace(' ', '_')
    worker_temp_dir = f"/tmp/duckdb_worker_{os.getpid()}_{safe_ab}_{yr}"
    
    # Clean up any existing files in the temp directory
    if os.path.exists(worker_temp_dir):
        try:
            shutil.rmtree(worker_temp_dir)
            print(f"🧹 Cleaned up existing tmp directory: {worker_temp_dir}")
        except Exception as e:
            print(f"⚠️ Warning: Could not clean tmp directory {worker_temp_dir}: {e}")
    
    # Also clean up any old duckdb worker directories to free space
    try:
        import glob
        old_dirs = glob.glob("/tmp/duckdb_worker_*")
        for old_dir in old_dirs:
            if os.path.exists(old_dir):
                try:
                    shutil.rmtree(old_dir)
                    print(f"🧹 Cleaned up old tmp directory: {old_dir}")
                except Exception:
                    pass  # Ignore errors for old directories
        
        # Also clean up any old duckdb temp files in /tmp (only old ones)
        temp_files = glob.glob("/tmp/duckdb_temp_*")
        for temp_file in temp_files:
            try:
                # Only clean files older than 1 hour to avoid race conditions
                if os.path.getmtime(temp_file) < time.time() - 3600:
                    os.unlink(temp_file)
                    print(f"🧹 Cleaned up old temp file: {temp_file}")
            except Exception:
                pass  # Ignore errors for temp files
        
        # Clean up any duckdb temp files in the worker's specific directory
        worker_temp_files = glob.glob(f"{worker_temp_dir}/duckdb_temp_*")
        for temp_file in worker_temp_files:
            try:
                os.unlink(temp_file)
                print(f"🧹 Cleaned up worker temp file: {temp_file}")
            except Exception:
                pass  # Ignore errors for temp files
                
    except Exception as e:
        print(f"⚠️ Warning: Could not clean old tmp directories: {e}")
    
    # Check available disk space before proceeding
    try:
        import shutil
        import subprocess
        total, used, free = shutil.disk_usage("/tmp")
        free_gb = free // (1024**3)
        if free_gb < 2:  # Less than 2GB free
            worker_log.warning(f"⚠️ Low disk space: {free_gb}GB free in /tmp")
            # Try to clean up more aggressively
            try:
                subprocess.run(["find", "/tmp", "-name", "duckdb*", "-type", "f", "-mtime", "+0", "-delete"], 
                             capture_output=True, timeout=30)
                subprocess.run(["find", "/tmp", "-name", "duckdb*", "-type", "d", "-mtime", "+0", "-exec", "rm", "-rf", "{}", "+"], 
                             capture_output=True, timeout=30)
                worker_log.info("🧹 Aggressive cleanup completed")
            except Exception as e:
                worker_log.warning(f"⚠️ Aggressive cleanup failed: {e}")
        else:
            worker_log.info(f"💾 Disk space OK: {free_gb}GB free in /tmp")
    except Exception as e:
        worker_log.warning(f"⚠️ Could not check disk space: {e}")
    
    # Log system resources for debugging
    try:
        import psutil
        memory = psutil.virtual_memory()
        worker_log.info(f"💾 System memory: {memory.percent}% used, {memory.available // (1024**3)}GB available")
        cpu_count = psutil.cpu_count()
        worker_log.info(f"🖥️ CPU cores: {cpu_count}")
    except ImportError:
        worker_log.debug("psutil not available for resource monitoring")
    except Exception as e:
        worker_log.debug(f"Could not check system resources: {e}")
    
    os.makedirs(worker_temp_dir, exist_ok=True)
    os.environ["DUCKDB_TMP_DIRECTORY"] = worker_temp_dir
    
    # Set thread controls for OpenMP/MKL (NOT DuckDB - those are separate libraries)
    os.environ.setdefault("OMP_NUM_THREADS", "4")  # Control OpenMP threads
    os.environ.setdefault("MKL_NUM_THREADS", "4")  # Control Intel MKL threads
    
    worker_log.info(f"Using temp directory: {worker_temp_dir}")
    attempt = 0

    while True:
        try:
            if run_mode == "import":
                mod_name = pharmacy_module if job == "pharmacy" else medical_module
                mod = importlib.import_module(mod_name)
                mod.main(ab, int(yr), lookahead)
                worker_log.info(f"OK    {job} {ab}/{yr}")
                
                # Clean up temp directory on success
                try:
                    if os.path.exists(worker_temp_dir):
                        shutil.rmtree(worker_temp_dir)
                        worker_log.info(f"Cleaned up temp directory: {worker_temp_dir}")
                except Exception as cleanup_e:
                    worker_log.warning(f"Could not clean up temp directory: {cleanup_e}")
                
                return  # <-- important: stop retry loop after success

            elif run_mode == "subprocess":
                script = pharmacy_script if job == "pharmacy" else medical_script
                if not script:
                    raise RuntimeError(
                        f"Missing --{'pharmacy' if job=='pharmacy' else 'medical'}-script path for subprocess mode"
                    )

                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"  # unbuffered child

                # Build base command with common arguments
                cmd = [
                    python_bin, "-u", script,         # -u = unbuffered stdio
                    "--age-band", ab,
                    "--event-year", str(int(yr)),
                    "--tmp-dir", worker_temp_dir,     # pass worker-specific temp directory
                ]
                
                # Add job-specific arguments
                if job == "pharmacy":
                    # Pharmacy jobs need pharmacy-input, demographics-lookup, and output-root
                    if pharmacy_input:
                        cmd.extend(["--pharmacy-input", pharmacy_input])
                    if demographics_lookup:
                        cmd.extend(["--demographics-lookup", demographics_lookup])
                    if output_root:
                        cmd.extend(["--output-root", output_root])
                elif job == "medical":
                    # Medical jobs need medical-input, demographics-lookup, output-root, lookahead-years
                    # medical_input is required for medical jobs
                    if not medical_input:
                        raise RuntimeError(f"Missing medical_input for medical job {ab}/{yr}")
                    cmd.extend(["--medical-input", medical_input])
                    if demographics_lookup:
                        cmd.extend(["--demographics-lookup", demographics_lookup])
                    if output_root:
                        cmd.extend(["--output-root", output_root])
                    cmd.extend(["--lookahead-years", str(int(lookahead))])

                with subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=1,          # line-buffer
                    text=True,
                    encoding="utf-8",
                    errors="replace",   # robust to odd bytes
                    env=env,
                ) as p:
                    # Capture both stdout and stderr
                    stdout_lines = []
                    stderr_lines = []
                    
                    # Read stdout
                    for line in p.stdout:
                        line = line.rstrip()
                        stdout_lines.append(line)
                        worker_log.info(f"[{job} {ab}/{yr}] {line}")
                    
                    # Read stderr
                    for line in p.stderr:
                        line = line.rstrip()
                        stderr_lines.append(line)
                        worker_log.error(f"[{job} {ab}/{yr}] STDERR: {line}")
                    
                    rc = p.wait()
                    if rc != 0:
                        # Log the full error details
                        worker_log.error(f"Process failed with return code {rc}")
                        worker_log.error(f"Command: {' '.join(cmd)}")
                        if stderr_lines:
                            worker_log.error(f"STDERR output: {' '.join(stderr_lines)}")
                        raise subprocess.CalledProcessError(rc, cmd, output='\n'.join(stdout_lines), stderr='\n'.join(stderr_lines))

                worker_log.info(f"OK    {job} {ab}/{yr}")
                
                # Clean up temp directory on success
                try:
                    if os.path.exists(worker_temp_dir):
                        shutil.rmtree(worker_temp_dir)
                        worker_log.info(f"Cleaned up temp directory: {worker_temp_dir}")
                except Exception as cleanup_e:
                    worker_log.warning(f"Could not clean up temp directory: {cleanup_e}")
                
                return  # <-- important

            else:
                raise ValueError(f"Unknown run_mode={run_mode!r}")

        except Exception as e:
            attempt += 1
            
            # Enhanced error logging for debugging
            if isinstance(e, subprocess.CalledProcessError):
                worker_log.warning(f"RETRY {job} {ab}/{yr} attempt={attempt} err={type(e).__name__}({e.returncode}, {e.cmd})")
                if hasattr(e, 'stderr') and e.stderr:
                    worker_log.warning(f"STDERR: {e.stderr}")
                if hasattr(e, 'output') and e.output:
                    worker_log.warning(f"OUTPUT: {e.output}")
            else:
                worker_log.warning(f"RETRY {job} {ab}/{yr} attempt={attempt} err={repr(e)}")
            
            if attempt > retries:
                worker_log.error(f"FAIL  {job} {ab}/{yr} err={repr(e)}")
                
                # Clean up temp directory on final failure
                try:
                    if os.path.exists(worker_temp_dir):
                        shutil.rmtree(worker_temp_dir)
                        worker_log.info(f"Cleaned up temp directory after failure: {worker_temp_dir}")
                except Exception as cleanup_e:
                    worker_log.warning(f"Could not clean up temp directory after failure: {cleanup_e}")
                
                raise
            time.sleep(backoff ** attempt)

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser(description="Parallel orchestrator for pharmacy or medical ETL")
    ap.add_argument("--job", choices=["pharmacy","medical","global-imputation"], required=True)
    ap.add_argument("--raw-pharmacy", default=DEFAULT_RAW_PHARMACY)
    ap.add_argument("--raw-medical",  default=DEFAULT_RAW_MEDICAL)
    ap.add_argument("--pharmacy-input", help="S3 path to pharmacy parquet files")
    ap.add_argument("--medical-input", help="S3 path to medical parquet files")
    ap.add_argument("--output-root", help="S3 root path for output files")
    # Removed --threads and --mem-gb arguments - DuckDB auto-detects optimal settings
    ap.add_argument("--tmp-dir", help="Temporary directory for DuckDB (for global-imputation)")
    ap.add_argument("--min-year", type=int)
    ap.add_argument("--max-year", type=int)
    ap.add_argument("--min-rows", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--pairs")
    ap.add_argument("--pairs-file")

    ap.add_argument("--workers", type=int, default=min(48, (os.cpu_count() or 8) * 1.5), 
                    help="Number of parallel workers (optimized for partitioned data: 48 workers × 1 thread × 2GB)")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--lookahead-years", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-mode", choices=["import","subprocess"], default="import")
    ap.add_argument("--pharmacy-module", default="pharmacy_clean")
    ap.add_argument("--medical-module",  default="medical_clean")
    ap.add_argument("--pharmacy-script", default="")
    ap.add_argument("--medical-script",  default="")
    ap.add_argument("--demographics-lookup", help="S3 path to mi_person_key demographics lookup table (for medical job)")
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    args = ap.parse_args()

    # Version logging
    print("🔧 Using Version 1997 - APCD Clean Orchestrator")

    # ---- Orchestrator run-level logger (S3-capable) ----
    run_id = time.strftime("%Y%m%d-%H%M%S")
    orc_logger, orc_buf = setup_logging("orchestrator", args.job, run_id)
    
    # Apply log level from args
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    orc_logger.setLevel(log_level)
    
    logging.getLogger("duckdb").setLevel(logging.WARNING)

    # make helpers use this logger
    global log
    log = orc_logger

    log.info("Args: %s", vars(args))



    # Handle global imputation job
    if args.job == "global-imputation":
        log.info("🚀 Running Global Demographic Imputation")
        try:
            # Call global imputation as subprocess with proper arguments
            
            cmd = [
                args.python_bin, "-u", "/home/pgx3874/pgx-analysis/1_apcd_input_data/global_imputation.py",
                "--pharmacy-input", args.pharmacy_input or args.raw_pharmacy,
                "--medical-input", args.medical_input or args.raw_medical,
                "--output-root", args.output_root or "s3://pgxdatalake/silver/imputed",
                "--lookahead-years", str(args.lookahead_years),
                # Removed --threads and --mem-gb - global_imputation.py auto-detects
                "--log-level", args.log_level
            ]
            
            # Add tmp-dir if provided
            if args.tmp_dir:
                cmd.extend(["--tmp-dir", args.tmp_dir])
            
            log.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=False, cwd="/home/pgx3874/pgx-analysis")
            log.info("✅ Global imputation completed successfully!")
            return
        except subprocess.CalledProcessError as e:
            log.error(f"❌ Global imputation failed with exit code {e.returncode}")
            raise
        except Exception as e:
            log.error(f"❌ Global imputation failed: {e}")
            raise

    # Validate execution mode early
    if args.run_mode == "subprocess":
        target = args.pharmacy_script if args.job == "pharmacy" else args.medical_script
        if not target:
            save_logs_immediate(orc_buf, "orchestrator", args.job, run_id, "apcd_input_data", logger=log, reason="missing_script")
            raise SystemExit("subprocess mode requires --pharmacy-script or --medical-script (depending on --job)")
        if not os.path.exists(target):
            save_logs_immediate(orc_buf, "orchestrator", args.job, run_id, "apcd_input_data", logger=log, reason="script_not_found")
            raise SystemExit(f"Script not found: {target}")
        log.info("Using subprocess mode: %s %s", args.python_bin, target)
    else:
        # Validate importability
        mod_name = args.pharmacy_module if args.job == "pharmacy" else args.medical_module
        try:
            imported = importlib.import_module(mod_name)
            log.info("Import mode: %s (%s)", mod_name, getattr(imported, "__file__", ""))
        except Exception as e:
            save_logs_immediate(orc_buf, "orchestrator", args.job, run_id, "apcd_input_data", logger=log, reason="import_failed")
            raise SystemExit(f"Cannot import module '{mod_name}': {e}")

    # Build (age_band, year) pairs
    t0 = time.perf_counter()
    override_pairs = load_pairs_override(args.pairs, args.pairs_file)
    if override_pairs:
        pairs_with_counts = [(ab, yr, -1) for (ab, yr) in override_pairs]
        log.info("Loaded %d override pairs in %.2fs", len(pairs_with_counts), time.perf_counter() - t0)
    else:
        if args.job == "pharmacy":
            pairs_with_counts = discover_from_pharmacy(args.pharmacy_input or args.raw_pharmacy, args.min_year, args.max_year, args.min_rows)
        else:
            pairs_with_counts = discover_from_medical(args.medical_input or args.raw_medical, args.min_year, args.max_year, args.min_rows)
        log.info("Discovery total time: %.2fs", time.perf_counter() - t0)

    # Early validation of dataset paths to fail fast and populate errors early
    try:
        from helpers_1997_13.s3_utils import validate_input_dataset_paths
        pharma_path = args.pharmacy_input or args.raw_pharmacy
        med_path = args.medical_input or args.raw_medical
        val = validate_input_dataset_paths(pharma_path, med_path, log)
        if not val.get('pharmacy', False) and not val.get('medical', False):
            log.error("❌ Neither pharmacy nor medical input paths validated. Aborting to surface errors early.")
            save_logs_to_s3(orc_buf, "orchestrator", args.job, run_id, "apcd_input_data", logger=log)
            sys.exit(2)
    except Exception as _e:
        log.warning(f"Could not validate input dataset paths (continuing): {_e}")

    if args.limit is not None:
        pairs_with_counts = pairs_with_counts[: int(args.limit)]

    if not pairs_with_counts:
        log.warning("No (age_band,event_year) partitions discovered (or provided). Exiting.")
        save_logs_to_s3(orc_buf, "orchestrator", args.job, run_id, "apcd_input_data", logger=log)
        sys.exit(0)

    # Preview to stdout
    log.info(f"Discovered {len(pairs_with_counts)} partitions for job={args.job}:")
    for ab, yr, n in pairs_with_counts[:20]:
        n_str = f"~{n} rows" if n >= 0 else "(manual)"
        log.info(f"  {ab}/{yr} {n_str}")
    if len(pairs_with_counts) > 20:
        log.info(f"  … +{len(pairs_with_counts)-20} more")
    if args.dry_run:
        log.info("Dry run requested; exiting before execution.")
        save_logs_to_s3(orc_buf, "orchestrator", args.job, run_id, "apcd_input_data", logger=log)
        return

    # Sort by descending row count (better load balance)
    pairs_sorted = sorted(pairs_with_counts, key=lambda t: t[2], reverse=True)

    # Safer multiprocessing start method
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        log.debug("Multiprocessing start method set to 'spawn'")
    except Exception as e:
        log.debug("Could not set start method: %s", e)

    successes, failures = 0, 0
    fail_list: list[tuple[str,int,str]] = []

    log.info("Submitting %d partitions with %d workers…", len(pairs_sorted), args.workers)
    log.info("🚀 Parallel Configuration: %d workers × 1 thread × 2GB RAM each", args.workers)
    t_start = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = dict()
        batch_size = 5
        total_jobs = len(pairs_sorted)
        for i in range(0, total_jobs, batch_size):
            batch = pairs_sorted[i:i+batch_size]
            for (ab, yr, _) in batch:
                fut = ex.submit(
                    run_task,
                    args.job, ab, yr, args.lookahead_years,
                    retries=args.retries,
                    run_mode=("subprocess" if args.run_mode == "subprocess" else "import"),
                    pharmacy_module=args.pharmacy_module,
                    medical_module=args.medical_module,
                    pharmacy_script=args.pharmacy_script,
                    medical_script=args.medical_script,
                    demographics_lookup=args.demographics_lookup,
                    medical_input=args.medical_input or args.raw_medical,
                    pharmacy_input=args.pharmacy_input or args.raw_pharmacy,
                    output_root=args.output_root,
                    python_bin=args.python_bin
                )
                futs[fut] = (ab, yr)
            log.info(f"Submitted batch {i//batch_size+1} ({len(batch)} jobs)")
            if i + batch_size < total_jobs:
                log.info("Sleeping 30 seconds to allow /tmp cleanup...")
                time.sleep(30)

        if not futs:
            save_logs_immediate(orc_buf, "orchestrator", args.job, run_id, "apcd_input_data", logger=log, reason="no_futures")
            raise SystemExit("No futures submitted — check run_mode and inputs.")

        try:
            for fut in cf.as_completed(list(futs.keys())):
                ab, yr = futs[fut]
                try:
                    fut.result()
                    successes += 1
                    log.info(f"✓ {ab}/{yr}")
                except Exception as e:
                    failures += 1
                    err = repr(e)
                    fail_list.append((ab, yr, err))
                    log.error(f"✗ {ab}/{yr} -> {err}")
        except KeyboardInterrupt:
            log.warning("KeyboardInterrupt received; cancelling remaining futures…")
            for f in futs:
                f.cancel()
            raise
    elapsed = time.perf_counter() - t_start

    # Enhanced Summary
    total_partitions = len(pairs_sorted)
    success_rate = 100 * successes / total_partitions if total_partitions > 0 else 0
    
    log.info("=" * 80)
    log.info(f"🎯 ORCHESTRATOR FINAL SUMMARY - {args.job.upper()}")
    log.info("=" * 80)
    log.info(f"📊 PROCESSING RESULTS:")
    log.info(f"   • Total partitions processed: {total_partitions:,}")
    log.info(f"   • Successful partitions: {successes:,}")
    log.info(f"   • Failed partitions: {failures:,}")
    log.info(f"   • Success rate: {success_rate:.1f}%")
    log.info(f"   • Total processing time: {elapsed:.1f}s")
    log.info(f"   • Average time per partition: {elapsed/total_partitions:.1f}s")
    log.info(f"")
    log.info(f"🔧 EXECUTION DETAILS:")
    log.info(f"   • Job type: {args.job}")
    log.info(f"   • Run mode: {args.run_mode}")
    log.info(f"   • Workers: {args.workers}")
    log.info(f"   • Year range: {args.min_year or 'all'} - {args.max_year or 'all'}")
    log.info(f"   • Lookahead years: {args.lookahead_years}")
    
    if failures:
        log.info(f"")
        log.error(f"❌ FAILED PARTITIONS ({failures}):")
        for ab, yr, err in fail_list[:10]:
            log.error(f"   • {ab}/{yr}: {err}")
        if len(fail_list) > 10:
            log.error(f"   • ... and {len(fail_list) - 10} more failures")
    else:
        log.info(f"")
        log.info(f"🏆 ALL PARTITIONS PROCESSED SUCCESSFULLY!")
    
    log.info("=" * 80)

    # Save orchestrator log to S3
    save_logs_to_s3(orc_buf, "orchestrator", args.job, run_id, logger=log)

    if failures:
        sys.exit(1)

if __name__ == "__main__":
    main()
