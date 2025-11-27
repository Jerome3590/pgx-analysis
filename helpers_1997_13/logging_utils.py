import logging
import io
import sys
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from logging.handlers import QueueHandler, QueueListener
import multiprocessing as mp

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import s3_client directly to avoid circular import
try:
    from helpers_1997_13.common_imports import s3_client
except ImportError:
    # Fallback for when common_imports is not available
    import boto3
    s3_client = boto3.client('s3')


class AutoFlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()  # force flush after each log message


class TimeoutException(Exception):
    pass

def setup_logging(cohort_name, band, year, pipeline_phase="apcd_input_data"):
    # Create unique logger name with timestamp and process ID to prevent collisions
    import time
    import os
    timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
    process_id = os.getpid()  # Process ID for additional uniqueness
    logger_name = f"logger_{cohort_name}_{band}_{year}_{timestamp}_{process_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    log_buffer = io.StringIO()

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Memory buffer
    buffer_handler = logging.StreamHandler(log_buffer)
    buffer_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(buffer_handler)

    # Console (with auto flush)
    console_handler = AutoFlushHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    # File handler with unique filename to prevent conflicts
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    process_id_str = str(process_id)
    
    # Create logs directory if it doesn't exist
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Use cohort_name to determine the correct log file prefix
    if cohort_name == "medical" or "medical" in str(cohort_name).lower():
        log_prefix = "medical_clean_output"
    elif cohort_name == "pharmacy" or "pharmacy" in str(cohort_name).lower():
        log_prefix = "pharmacy_clean_output"
    else:
        log_prefix = f"{cohort_name}_clean_output"
    
    output_log_path = logs_dir / f"{log_prefix}_{timestamp_str}_{process_id_str}.txt"
    file_handler = logging.FileHandler(str(output_log_path), mode="w")  # Use 'w' instead of 'a' for unique files
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    return logger, log_buffer


def create_fpgrowth_logger(name="fpgrowth", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = FlushStreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


def log_cpu_context(prefix: str, threads: int):
    """
    Log CPU context information for debugging multiprocessing and thread affinity.
    
    Args:
        prefix: Prefix string for log messages
        threads: Number of DuckDB threads configured
    """
    import os
    try:
        import psutil  # type: ignore
        p = psutil.Process()
        try:
            affinity = p.cpu_affinity()
        except Exception:
            affinity = None
        print(f"{prefix} CPU context: pid={p.pid}, logical_cpus={os.cpu_count()}, affinity={affinity}, duckdb_threads={threads}")
    except Exception:
        # Fallback: Linux-only sched_getaffinity
        try:
            cpus = None
            if hasattr(os, 'sched_getaffinity'):
                cpus = sorted(list(os.sched_getaffinity(0)))  # type: ignore
            print(f"{prefix} CPU context: pid={os.getpid()}, logical_cpus={os.cpu_count()}, affinity={cpus}, duckdb_threads={threads}")
        except Exception:
            print(f"{prefix} CPU context: pid={os.getpid()}, logical_cpus={os.cpu_count()}, duckdb_threads={threads}")


def save_logs_to_s3(log_buffer, cohort_name, band, year, pipeline_phase="apcd_input_data", logger=None, checkpoint_name=None):
    """Save captured logs to S3 using standard text writer."""
    try:
        # Validate input parameters to prevent None values in S3 paths
        if cohort_name is None or cohort_name == "":
            raise ValueError("cohort_name cannot be None or empty")
        if band is None or band == "":
            raise ValueError("band cannot be None or empty")
        if year is None or year == "":
            raise ValueError("year cannot be None or empty")

        # Clean up band and year values if they contain prefixes
        if isinstance(band, str) and band.startswith('age_band='):
            band = band.replace('age_band=', '')
        if isinstance(year, str) and year.startswith('event_year='):
            year = year.replace('event_year=', '')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create checkpoint-specific filename if provided
        if checkpoint_name:
            log_path = f"s3://pgx-repository/build_logs/{pipeline_phase}/{cohort_name}/{band}/{year}/log_{timestamp}_{checkpoint_name}.txt"
        else:
            log_path = f"s3://pgx-repository/build_logs/{pipeline_phase}/{cohort_name}/{band}/{year}/log_{timestamp}.txt"

        log_content = log_buffer.getvalue()
        save_to_s3_text(log_content, log_path, logger=logger)

        if logger:
            logger.info(f"✓ Logs saved to S3: {log_path}")
        else:
            print(f"✓ Logs saved to S3: {log_path}")

    except Exception as e:
        if logger:
            logger.warning(f"⚠ Warning: Could not save logs to S3: {str(e)}")
        else:
            print(f"⚠ Warning: Could not save logs to S3: {str(e)}")


def save_logs_checkpoint(log_buffer, cohort_name, band, year, step_name, pipeline_phase="apcd_input_data", logger=None):
    """Save logs at a specific checkpoint during pipeline execution."""
    try:
        # Validate input parameters to prevent None values in S3 paths
        if cohort_name is None or cohort_name == "":
            raise ValueError("cohort_name cannot be None or empty")
        if band is None or band == "":
            raise ValueError("band cannot be None or empty")
        if year is None or year == "":
            raise ValueError("year cannot be None or empty")
        if step_name is None or step_name == "":
            raise ValueError("step_name cannot be None or empty")

        # Clean up band and year values if they contain prefixes
        if isinstance(band, str) and band.startswith('age_band='):
            band = band.replace('age_band=', '')
        if isinstance(year, str) and year.startswith('event_year='):
            year = year.replace('event_year=', '')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Clean and standardize the step name
        step_name_clean = step_name.lower().replace(' ', '_').replace('→', '').replace(':', '').replace('-', '_').replace('__', '_')

        # Extract step number and description more robustly
        step_num = "1"  # Default
        description = step_name_clean

        # Look for step pattern like "step1", "step2", etc.
        step_match = re.search(r'step(\d+)', step_name_clean)
        if step_match:
            step_num = step_match.group(1)
            # Remove the step pattern and clean up
            description = re.sub(r'step\d+_?', '', step_name_clean)
            # Clean up any leading/trailing underscores
            description = description.strip('_')

        # If no step number found, try to extract from description
        if step_num == "1" and description:
            # Look for any remaining step patterns and remove them
            description = re.sub(r'step\d+_?', '', description)
            description = description.strip('_')

        checkpoint_name = f"step{step_num}_{description}"
        log_path = f"s3://pgx-repository/build_logs/{pipeline_phase}/{cohort_name}/{band}/{year}/log_{timestamp}_{checkpoint_name}.txt"

        log_content = log_buffer.getvalue()
        save_to_s3_text(log_content, log_path, logger=logger)

        if logger:
            logger.info(f"✓ Checkpoint logs saved: {log_path}")
        else:
            print(f"✓ Checkpoint logs saved: {log_path}")

    except Exception as e:
        if logger:
            logger.warning(f"⚠ Warning: Could not save checkpoint logs to S3: {str(e)}")
        else:
            print(f"⚠ Warning: Could not save checkpoint logs to S3: {str(e)}")


def save_to_s3_text(
    text: str,
    s3_path: str,
    logger: Optional[logging.Logger] = None,
    partition_cols: Optional[List[str]] = None
) -> None:
    """Save text data to S3, optionally excluding lines containing partition column keys."""
    try:
        # Remove partition columns from text (if applicable)
        if partition_cols:
            # Apply line filtering: remove any line that contains a partition column key
            lines = text.splitlines()
            filtered_lines = [
                line for line in lines
                if not any(col in line for col in partition_cols)
            ]
            text = "\n".join(filtered_lines)

        bucket, key = s3_path.replace("s3://", "").split("/", 1)

        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=text.encode('utf-8'),
            ContentType='text/plain'
        )

        if logger:
            logger.info(f"✓ Saved text file to {s3_path}")
        else:
            print(f"✓ Saved text file to {s3_path}")
    except Exception as e:
        msg = f"✗ Error saving text file to {s3_path}: {str(e)}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        raise


def save_logs_immediate(log_buffer, cohort_name, band, year, pipeline_phase="apcd_input_data", logger=None, reason="immediate"):
    """Save logs immediately for critical situations (crashes, errors, etc.)."""
    try:
        # Validate input parameters
        if cohort_name is None or cohort_name == "":
            raise ValueError("cohort_name cannot be None or empty")
        if band is None or band == "":
            raise ValueError("band cannot be None or empty")
        if year is None or year == "":
            raise ValueError("year cannot be None or empty")

        # Clean up band and year values if they contain prefixes
        if isinstance(band, str) and band.startswith('age_band='):
            band = band.replace('age_band=', '')
        if isinstance(year, str) and year.startswith('event_year='):
            year = year.replace('event_year=', '')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = f"s3://pgx-repository/build_logs/{pipeline_phase}/{cohort_name}/{band}/{year}/log_{timestamp}_{reason}.txt"

        log_content = log_buffer.getvalue()
        save_to_s3_text(log_content, log_path, logger=logger)

        if logger:
            logger.info(f"✓ Immediate logs saved to S3: {log_path}")
        else:
            print(f"✓ Immediate logs saved to S3: {log_path}")

    except Exception as e:
        if logger:
            logger.warning(f"⚠ Warning: Could not save immediate logs to S3: {str(e)}")
        else:
            print(f"⚠ Warning: Could not save immediate logs to S3: {str(e)}")


def log_file_sizing_optimization(logger, operation_context, table_name, sizing_params, s3_path):
    """
    Comprehensive logging for file sizing optimization operations.

    Args:
        logger: Logger instance
        operation_context (str): Description of the operation being performed
        table_name (str): Name of the table being processed
        sizing_params (dict): Dictionary containing optimization parameters
        s3_path (str): S3 path where the file will be saved
    """
    try:
        if logger is None:
            return

        logger.info(f"📊 File Sizing Optimization for {operation_context}")
        logger.info(f"   Table: {table_name}")
        logger.info(f"   Target Path: {s3_path}")

        # Log table statistics
        if 'total_rows' in sizing_params:
            logger.info(f"   Total Rows: {sizing_params['total_rows']:,}")

        if 'column_count' in sizing_params:
            logger.info(f"   Column Count: {sizing_params['column_count']}")

        # Log row group optimization details
        if 'avg_row_size_bytes' in sizing_params:
            logger.info(f"   Avg Row Size: {sizing_params['avg_row_size_bytes']:,} bytes")

        if 'rows_per_group' in sizing_params:
            logger.info(f"   Rows per Group: {sizing_params['rows_per_group']:,}")

        if 'actual_size_mb' in sizing_params:
            logger.info(f"   Actual Size: {sizing_params['actual_size_mb']:.2f} MB")

        # Log file count estimates
        if 'estimated_file_count' in sizing_params:
            logger.info(f"   Estimated Files: {sizing_params['estimated_file_count']}")

        # Log storage efficiency metrics
        if 'total_rows' in sizing_params and 'rows_per_group' in sizing_params:
            efficiency = (sizing_params['total_rows'] / sizing_params['rows_per_group']) if sizing_params['rows_per_group'] > 0 else 0
            logger.info(f"   Storage Efficiency: {efficiency:.2f} groups")

        # Log any errors or warnings
        if 'error' in sizing_params:
            logger.warning(f"   ⚠ Optimization Warning: {sizing_params['error']}")

        logger.info("   ✓ Optimization parameters calculated successfully")

    except Exception as e:
        if logger:
            logger.error(f"   ✗ Error in file sizing optimization logging: {str(e)}")
        else:
            print(f"   ✗ Error in file sizing optimization logging: {str(e)}")


# ===== Multiprocessing Logging Utilities (Queue-based real-time flush) =====
def setup_mp_logging(component: str,
                     context: str,
                     run_id: Optional[str] = None,
                     level: int = logging.INFO):
    """Initialize queue-based logging for parent/orchestrator.

    Returns (logger, log_buffer, log_queue, listener):
      - logger: parent logger for local logs
      - log_buffer: in-memory buffer for later S3 save
      - log_queue: multiprocessing.Queue to pass to workers
      - listener: QueueListener to stop() at shutdown
    """
    # Parent logger with memory buffer
    logger_name = f"mp_{component}_{context}_{run_id or datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    log_buffer = io.StringIO()
    buffer_handler = logging.StreamHandler(log_buffer)
    buffer_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(buffer_handler)

    # Real-time streaming via QueueListener to stdout
    log_queue: mp.Queue = mp.Queue()
    stream_handler = AutoFlushHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    stream_handler.setLevel(level)
    listener = QueueListener(log_queue, stream_handler, respect_handler_level=True)
    listener.start()

    return logger, log_buffer, log_queue, listener


def get_logger(name: str, band: Optional[str] = "all", year: Optional[str] = "all"):
    """Compatibility helper that returns a logger instance.

    This preserves the older get_logger(name, band, year) API used across scripts.
    It prefers setup_logging when cohort-like args are passed; otherwise it falls
    back to a simple stream logger.
    """
    try:
        # Try to use setup_logging if it accepts these args (returns logger, buffer)
        logger, _ = setup_logging(name, band, year)
        return logger
    except Exception:
        # Fallback: create lightweight logger
        return create_fpgrowth_logger(name)


def get_worker_queue_logger(log_queue: mp.Queue,
                            name: Optional[str] = None,
                            level: int = logging.INFO) -> logging.Logger:
    """Create a per-worker logger that emits records into the parent's queue.

    Usage in worker:
        worker_logger = get_worker_queue_logger(log_queue, name='txt_to_parquet.worker')
        worker_logger.info('...')
    """
    import os
    logger_name = name or f"worker_{os.getpid()}"
    worker_logger = logging.getLogger(logger_name)
    worker_logger.setLevel(level)

    # Replace existing handlers with a QueueHandler
    for h in list(worker_logger.handlers):
        worker_logger.removeHandler(h)
    qh = QueueHandler(log_queue)
    worker_logger.addHandler(qh)
    worker_logger.propagate = False
    return worker_logger


def stop_mp_logging(listener: QueueListener) -> None:
    """Stop the QueueListener safely (no-op if already stopped)."""
    try:
        if listener:
            listener.stop()
    except Exception:
        pass


# ============================================================================
# Feature Importance Logging Utilities
# ============================================================================

def setup_r_logging(cohort_name: str, age_band: str, event_year: int) -> dict:
    """
    Setup logging for feature importance analysis (R-style API)
    
    Args:
        cohort_name: Name of the cohort
        age_band: Age band (e.g., "25-44")
        event_year: Event year
        
    Returns:
        Dictionary with 'logger' and 'log_file_path'
    """
    # Create log directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"feature_importance_{cohort_name}_{age_band}_{event_year}_{timestamp}.log"
    log_file_path = str(log_dir / log_file_name)
    
    # Create logger
    logger_name = f"feature_importance_{cohort_name}_{age_band}_{event_year}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return {
        'logger': logger,
        'log_file_path': log_file_path
    }


def save_logs_to_s3_r(log_file_path: str, cohort_name: str, age_band: str, event_year: int, logger):
    """
    Save log file to S3 (R-style API)
    
    Args:
        log_file_path: Path to local log file
        cohort_name: Name of the cohort
        age_band: Age band
        event_year: Event year
        logger: Logger instance
    """
    if not os.path.exists(log_file_path):
        logger.warning("Log file not found: %s", log_file_path)
        return
    
    try:
        from helpers_1997_13.constants import S3_BUCKET
        s3_key = f"logs/feature_importance/cohort_name={cohort_name}/age_band={age_band}/event_year={event_year}/{os.path.basename(log_file_path)}"
        
        with open(log_file_path, 'rb') as f:
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=f
            )
        
        logger.info("Saved log to S3: s3://%s/%s", S3_BUCKET, s3_key)
    except Exception as e:
        logger.error("Failed to save log to S3: %s", str(e))


def check_memory_usage_r(logger, label: str):
    """
    Check and log memory usage (R-style API)
    
    Args:
        logger: Logger instance
        label: Label for this memory check
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.info("Memory usage [%s]: %.1f MB", label, memory_mb)
    except ImportError:
        pass  # psutil not available
    except Exception as e:
        logger.warning("Could not check memory usage: %s", str(e))


