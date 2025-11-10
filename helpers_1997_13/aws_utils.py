import boto3
from botocore.config import Config
import requests
import socket
import threading
import os
import sys
import signal
import traceback
import psutil
from contextlib import contextmanager
from typing import Optional, Dict, Any
import certifi

# Set root of project for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

AWS_REGION = "us-east-1"

# Configure boto3 with retries
boto3_config = Config(
    retries = dict(
        max_attempts = 3,
        mode = 'adaptive'
    ),
    connect_timeout = 5,
    read_timeout = 10,
    region_name = AWS_REGION
)

ses_client = boto3.client('ses', region_name=AWS_REGION, config=boto3_config)

# Global connection pool for multiprocessing
_connection_pool = {}
_pool_lock = threading.Lock()

# Signal debugging
_signal_debug_enabled = False
_signal_debug_logger = None

def enable_signal_debugging(logger=None):
    """
    Enable detailed signal debugging to trace Signal 15 issues.

    Args:
        logger: Logger instance for debug output
    """
    global _signal_debug_enabled, _signal_debug_logger
    _signal_debug_enabled = True
    _signal_debug_logger = logger

    def debug_signal_handler(signum, frame):
        """Enhanced signal handler with detailed debugging"""
        import traceback
        import psutil

        # Get current process info
        process = psutil.Process()
        memory_info = process.memory_info()

        # Get current stack trace
        stack_trace = traceback.format_stack(frame)

        # Log detailed signal information
        if _signal_debug_logger:
            _signal_debug_logger.error(f"🔍 SIGNAL DEBUG: Received signal {signum} (SIGTERM={signal.SIGTERM})")
            _signal_debug_logger.error(f"🔍 Process ID: {process.pid}")
            _signal_debug_logger.error(f"🔍 Memory Usage: {memory_info.rss / 1024 / 1024:.2f} MB")
            _signal_debug_logger.error(f"🔍 CPU Usage: {process.cpu_percent()}%")
            _signal_debug_logger.error(f"🔍 Open Files: {len(process.open_files())}")
            _signal_debug_logger.error(f"🔍 Threads: {process.num_threads()}")
            _signal_debug_logger.error(f"🔍 Stack Trace:")
            for i, line in enumerate(stack_trace[-10:]):  # Last 10 lines
                _signal_debug_logger.error(f"🔍   {i}: {line.strip()}")

        # Call original signal handler if it exists
        if hasattr(debug_signal_handler, 'original_handler'):
            debug_signal_handler.original_handler(signum, frame)

    # Register signal handlers
    try:
        # Store original handlers
        debug_signal_handler.original_handler = signal.signal(signal.SIGTERM, debug_signal_handler)
        signal.signal(signal.SIGINT, debug_signal_handler)
        if _signal_debug_logger:
            _signal_debug_logger.info("🔍 Signal debugging enabled")
    except Exception as e:
        if _signal_debug_logger:
            _signal_debug_logger.warning(f"Could not register signal debug handlers: {e}")

def get_system_resource_status():
    """
    Get current system resource status for debugging.

    Returns:
        Dict with system resource information
    """
    try:
        process = psutil.Process()
        memory = psutil.virtual_memory()

        return {
            'process_memory_mb': process.memory_info().rss / 1024 / 1024,
            'system_memory_percent': memory.percent,
            'process_cpu_percent': process.cpu_percent(),
            'open_files': len(process.open_files()),
            'threads': process.num_threads(),
            'connections': len(process.net_connections()) if hasattr(process, 'net_connections') else 0
        }
    except Exception as e:
        return {'error': str(e)}


def get_shared_s3_client(worker_id: Optional[str] = None) -> Any:
    """
    Get a shared S3 client for multiprocessing environments.

    Args:
        worker_id: Optional worker identifier for tracking

    Returns:
        boto3 S3 client with proper configuration
    """
    # Create a unique key for this worker's connection
    worker_key = worker_id or f"worker_{threading.get_ident()}"

    with _pool_lock:
        if worker_key not in _connection_pool:
            # Create new S3 client with optimized configuration for multiprocessing
            config = Config(
                retries=dict(
                    max_attempts=3,
                    mode='adaptive'
                ),
                connect_timeout=10,
                read_timeout=30,
                max_pool_connections=10,  # Limit connection pool size
                region_name='us-east-1'
            )

            _connection_pool[worker_key] = boto3.client(
                "s3",
                config=config,
                verify=certifi.where()
            )

        return _connection_pool[worker_key]


def get_shared_duckdb_connection(worker_id: Optional[str] = None, logger=None):
    """
    Get a shared DuckDB connection with AWS credentials for multiprocessing.
    
    Args:
        worker_id: Optional worker identifier for tracking
        logger: Optional logger for tracking
        
    Returns:
        DuckDB connection with AWS credentials loaded
    """
    import duckdb
    
    # Create a unique key for this worker's connection
    worker_key = worker_id or f"worker_{threading.get_ident()}"
    
    with _pool_lock:
        if f"duckdb_{worker_key}" not in _connection_pool:
            try:
                # Create new DuckDB connection using standardized function
                from helpers_1997_13.duckdb_utils import get_shared_duckdb_connection
                conn = get_shared_duckdb_connection(worker_key, logger)
                
                _connection_pool[f"duckdb_{worker_key}"] = conn
                
                if logger:
                    logger.info(f"Created shared DuckDB connection for worker {worker_key}")
                    
            except Exception as e:
                if logger:
                    logger.error(f"Failed to create shared DuckDB connection: {str(e)}")
                raise
                
        return _connection_pool[f"duckdb_{worker_key}"]


@contextmanager
def shared_aws_connection(worker_id: Optional[str] = None, logger=None):
    """
    Context manager for shared AWS connections in multiprocessing.
    
    Args:
        worker_id: Optional worker identifier for tracking
        logger: Optional logger for tracking
        
    Yields:
        Tuple of (s3_client, duckdb_connection)
    """
    s3_client = None
    duckdb_conn = None
    
    try:
        # Get shared connections
        s3_client = get_shared_s3_client(worker_id)
        duckdb_conn = get_shared_duckdb_connection(worker_id, logger)
        
        yield s3_client, duckdb_conn
        
    except Exception as e:
        if logger:
            logger.error(f"Error in shared AWS connection: {str(e)}")
        raise
    finally:
        # Note: We don't close the connections here as they're shared
        # The connections will be cleaned up when the process ends
        pass


def cleanup_shared_connections(worker_id: Optional[str] = None, logger=None):
    """
    Clean up shared connections for a specific worker or all workers.
    
    Args:
        worker_id: Optional worker identifier. If None, cleans up all connections.
        logger: Optional logger for tracking
    """
    with _pool_lock:
        if worker_id:
            # Clean up specific worker connections
            keys_to_remove = [k for k in _connection_pool.keys() 
                             if worker_id in k or k == worker_id]
        else:
            # Clean up all connections
            keys_to_remove = list(_connection_pool.keys())
        
        for key in keys_to_remove:
            try:
                conn = _connection_pool[key]
                if hasattr(conn, 'close'):
                    conn.close()
                del _connection_pool[key]
                
                if logger:
                    logger.info(f"Cleaned up shared connection: {key}")
                    
            except Exception as e:
                if logger:
                    logger.warning(f"Error cleaning up connection {key}: {str(e)}")


def get_connection_pool_status() -> Dict[str, Any]:
    """
    Get status of the shared connection pool.
    
    Returns:
        Dictionary with pool status information
    """
    with _pool_lock:
        return {
            'total_connections': len(_connection_pool),
            'connection_keys': list(_connection_pool.keys()),
            's3_clients': len([k for k in _connection_pool.keys() if not k.startswith('duckdb_')]),
            'duckdb_connections': len([k for k in _connection_pool.keys() if k.startswith('duckdb_')])
        }


def setup_worker_environment(worker_id: str, logger=None):
    """
    Set up the environment for a specific worker process.
    
    Args:
        worker_id: Worker identifier
        logger: Optional logger for tracking
        
    Returns:
        Tuple of (s3_client, duckdb_connection)
    """
    if logger:
        logger.info(f"Setting up worker environment for {worker_id}")
    
    try:
        # Get shared connections for this worker
        s3_client = get_shared_s3_client(worker_id)
        duckdb_conn = get_shared_duckdb_connection(worker_id, logger)
        
        if logger:
            logger.info(f"Worker {worker_id} environment setup complete")
        
        return s3_client, duckdb_conn
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to setup worker {worker_id} environment: {str(e)}")
        raise


def cleanup_worker_environment(worker_id: str, logger=None):
    """
    Clean up the environment for a specific worker process.
    
    Args:
        worker_id: Worker identifier
        logger: Optional logger for tracking
    """
    if logger:
        logger.info(f"Cleaning up worker environment for {worker_id}")
    
    try:
        cleanup_shared_connections(worker_id, logger)
        
        if logger:
            logger.info(f"Worker {worker_id} environment cleanup complete")
            
    except Exception as e:
        if logger:
            logger.error(f"Failed to cleanup worker {worker_id} environment: {str(e)}")
        raise


def get_instance_id():
    """Get the EC2 instance ID using the instance metadata service.

    Returns:
        str: EC2 instance ID or 'local' if not running on EC2
    """
    try:
        # Try to get instance ID from EC2 metadata service
        response = requests.get(
            'http://169.254.169.254/latest/meta-data/instance-id',
            timeout=1
        )
        if response.status_code == 200:
            return response.text
    except (requests.RequestException, socket.timeout):
        # If we can't reach the metadata service, we're probably not on EC2
        pass

    # Fallback to hostname if not on EC2
    try:
        return socket.gethostname()
    except Exception:
        return 'local'


def notify_error(step_name, error_msg, logger):
    """Send email notification for errors and save logs immediately."""
    subject = f"Error in step {step_name}"
    body = f"""
    Error occurred during pipeline execution:

    {error_msg}

    Please check logs for more details.
    """

    logger.error(f"Error in step {step_name}: {error_msg}")

    # Try to save logs immediately if the logger has the save_logs_now method
    try:
        if hasattr(logger, 'save_logs_now'):
            logger.save_logs_now(f"error_{step_name}")
            logger.info("✓ Logs saved immediately due to error")
    except Exception as log_error:
        logger.warning(f"Could not save logs immediately: {log_error}")

    send_email(subject, body)


def notify_success(age_band, event_year, cohort_name, logger):
    """Send email notification for successful pipeline completion."""
    subject = f"Pipeline Success: {cohort_name} cohort for {age_band}/{event_year}"
    body = f"""
    Pipeline completed successfully!

    Cohort: {cohort_name}
    Age Band: {age_band}
    Event Year: {event_year}

    The cohort has been created and saved to S3.
    You can now proceed with downstream analysis.
    """

    logger.info(f"Pipeline success notification sent for {cohort_name} cohort ({age_band}/{event_year})")
    send_email(subject, body)


def send_email(subject, body):
    try:
        response = ses_client.send_email(
            Source='jerome@mushinsolutions.com',
            Destination={
                'ToAddresses': ['dixonrj@vcu.edu']
            },
            Message={
                'Subject': {
                    'Data': subject
                },
                'Body': {
                    'Text': {
                        'Data': body
                    }
                }
            }
        )
        print(f"Email sent successfully: {response['MessageId']}")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")