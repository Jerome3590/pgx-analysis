"""
Robust logging utilities for PGx pipeline steps.

Provides:
1. Consistent log formatting across all pipeline scripts
2. File + console logging with rotation
3. Automatic S3 mirroring for pipeline logs
4. Progress tracking and step timing
5. Error aggregation and summary reporting
6. Context managers for step logging

Usage:
    from py_helpers.pipeline_logger import setup_pipeline_logger, log_step_start, log_step_complete
    
    # Basic usage
    logger = setup_pipeline_logger(
        step_name="4_model_data",
        cohort="opioid_ed",
        age_band="25-44",
        script_name="create_model_data"
    )
    
    # With context manager for automatic completion logging
    with log_step_context(logger, "Building model features"):
        # Your code here
        pass  # Automatically logs success/failure and timing
    
    # Manual step logging
    log_step_start(logger, "Loading data from S3")
    # ... do work ...
    log_step_complete(logger, "Loading data from S3", records_loaded=150000)
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Import S3 mirroring if available
try:
    from py_helpers.fe_monitor import mirror_log_to_s3
    HAS_S3_MIRROR = True
except ImportError:
    HAS_S3_MIRROR = False
    mirror_log_to_s3 = None


class PipelineLogger:
    """
    Enhanced logger for pipeline steps with automatic S3 mirroring and progress tracking.
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        step_name: str,
        cohort: str,
        age_band: str,
        log_file_path: Path,
        mirror_to_s3: bool = True
    ):
        self.logger = logger
        self.step_name = step_name
        self.cohort = cohort
        self.age_band = age_band
        self.log_file_path = log_file_path
        self.mirror_to_s3 = mirror_to_s3 and HAS_S3_MIRROR
        self.start_time = time.time()
        self.step_times: Dict[str, float] = {}
        self.errors: list[str] = []
        self.warnings: list[str] = []
        
    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        self.logger.info(msg, *args, **kwargs)
        
    def warning(self, msg: str, *args, **kwargs):
        """Log warning and track it."""
        self.logger.warning(msg, *args, **kwargs)
        self.warnings.append(msg % args if args else msg)
        
    def error(self, msg: str, *args, **kwargs):
        """Log error and track it."""
        self.logger.error(msg, *args, **kwargs)
        self.errors.append(msg % args if args else msg)
        
    def exception(self, msg: str, *args, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(msg, *args, **kwargs)
        self.errors.append(msg % args if args else msg)
        
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message."""
        self.logger.debug(msg, *args, **kwargs)
        
    def sync_to_s3(self):
        """Mirror current log file to S3."""
        if self.mirror_to_s3 and self.log_file_path.exists():
            try:
                mirror_log_to_s3(
                    self.step_name,
                    self.cohort,
                    self.age_band,
                    self.log_file_path,
                    self.logger
                )
            except Exception as e:
                self.logger.warning("Failed to mirror log to S3: %s", e)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since logger creation in seconds."""
        return time.time() - self.start_time
    
    def format_elapsed_time(self, seconds: Optional[float] = None) -> str:
        """Format elapsed time as human-readable string."""
        if seconds is None:
            seconds = self.get_elapsed_time()
        
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
    
    def log_summary(self):
        """Log summary of warnings, errors, and timing."""
        elapsed = self.get_elapsed_time()
        
        self.info("=" * 80)
        self.info("PIPELINE STEP SUMMARY")
        self.info("=" * 80)
        self.info("Step:     %s", self.step_name)
        self.info("Cohort:   %s", self.cohort)
        self.info("Age Band: %s", self.age_band)
        self.info("Elapsed:  %s", self.format_elapsed_time(elapsed))
        self.info("Warnings: %d", len(self.warnings))
        self.info("Errors:   %d", len(self.errors))
        
        if self.warnings:
            self.info("\nWarnings:")
            for i, warn in enumerate(self.warnings[:10], 1):
                self.info("  %d. %s", i, warn[:200])
            if len(self.warnings) > 10:
                self.info("  ... and %d more warnings", len(self.warnings) - 10)
        
        if self.errors:
            self.info("\nErrors:")
            for i, err in enumerate(self.errors[:10], 1):
                self.info("  %d. %s", i, err[:200])
            if len(self.errors) > 10:
                self.info("  ... and %d more errors", len(self.errors) - 10)
        
        self.info("=" * 80)
        
        # Always print completion summary to console for visibility
        status = "ERROR" if self.errors else ("WARN" if self.warnings else "OK")
        print(f"[{self.step_name}] {self.cohort}/{self.age_band} done "
              f"({self.format_elapsed_time(elapsed)}) — {status} "
              f"| warnings={len(self.warnings)} errors={len(self.errors)}")
        if self.errors:
            for err in self.errors[:3]:
                print(f"  ERROR: {err[:120]}")
        
        # Sync final log to S3
        self.sync_to_s3()


def setup_pipeline_logger(
    step_name: str,
    cohort: str,
    age_band: str,
    script_name: str,
    log_dir: Optional[Path] = None,
    console_level: int = logging.WARNING,
    file_level: int = logging.DEBUG,
    mirror_to_s3: bool = True
) -> PipelineLogger:
    """
    Set up a comprehensive logger for a pipeline step.
    
    Args:
        step_name: Pipeline step identifier (e.g., "4_model_data", "5_bupar")
        cohort: Cohort name (e.g., "opioid_ed")
        age_band: Age band (e.g., "25-44")
        script_name: Script/module name for log filename
        log_dir: Optional log directory (default: 9_dashboard_visuals/logs/{step_name})
        console_level: Logging level for console output (default: INFO)
        file_level: Logging level for file output (default: DEBUG)
        mirror_to_s3: Whether to mirror logs to S3 (default: True)
    
    Returns:
        PipelineLogger instance with enhanced functionality
    """
    # Determine repo root (walk up from this file)
    current_file = Path(__file__).resolve()
    repo_root = current_file.parents[1]  # py_helpers is one level down from repo root
    
    # Default log directory — all steps use logs/{step_name}/ at repo root
    if log_dir is None:
        log_dir = repo_root / "logs" / step_name
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    age_band_fname = age_band.replace("-", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{script_name}_{cohort}_{age_band_fname}_{timestamp}.log"
    log_file_path = log_dir / log_filename
    
    # Create logger
    logger_name = f"{step_name}.{cohort}.{age_band}.{script_name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler - shorter format for readability
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_format = logging.Formatter(
        '%(levelname)-8s %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler - detailed format with timestamps
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(file_level)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    
    # Wrap in PipelineLogger
    pipeline_logger = PipelineLogger(
        logger=logger,
        step_name=step_name,
        cohort=cohort,
        age_band=age_band,
        log_file_path=log_file_path,
        mirror_to_s3=mirror_to_s3
    )
    
    # Log initial header to file; print summary line to console for visibility
    pipeline_logger.info("=" * 80)
    pipeline_logger.info("PIPELINE STEP: %s", step_name)
    pipeline_logger.info("=" * 80)
    pipeline_logger.info("Script:   %s", script_name)
    pipeline_logger.info("Cohort:   %s", cohort)
    pipeline_logger.info("Age Band: %s", age_band)
    pipeline_logger.info("Started:  %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    pipeline_logger.info("Log File: %s", log_file_path)
    pipeline_logger.info("=" * 80)
    pipeline_logger.info("")
    print(f"[{step_name}] {cohort}/{age_band} started — log: {log_file_path}")
    
    return pipeline_logger


def log_step_start(logger: PipelineLogger, step_description: str, **metadata):
    """
    Log the start of a processing step with optional metadata.
    
    Args:
        logger: PipelineLogger instance
        step_description: Human-readable step description
        **metadata: Additional key-value pairs to log
    """
    logger.info("▶ START: %s", step_description)
    if metadata:
        for key, value in metadata.items():
            logger.info("  %s: %s", key, value)
    logger.step_times[step_description] = time.time()


def log_step_complete(logger: PipelineLogger, step_description: str, **metadata):
    """
    Log the completion of a processing step with timing and metadata.
    
    Args:
        logger: PipelineLogger instance
        step_description: Human-readable step description (must match log_step_start)
        **metadata: Additional key-value pairs to log (e.g., records_processed=1000)
    """
    if step_description in logger.step_times:
        elapsed = time.time() - logger.step_times[step_description]
        elapsed_str = logger.format_elapsed_time(elapsed)
    else:
        elapsed_str = "unknown"
    
    logger.info("✓ COMPLETE: %s (took %s)", step_description, elapsed_str)
    if metadata:
        for key, value in metadata.items():
            logger.info("  %s: %s", key, value)
    logger.info("")


def log_step_failed(logger: PipelineLogger, step_description: str, error: Exception):
    """
    Log a step failure with exception details.
    
    Args:
        logger: PipelineLogger instance
        step_description: Human-readable step description
        error: Exception that caused the failure
    """
    logger.error("✗ FAILED: %s", step_description)
    logger.exception("  Exception: %s", str(error))
    logger.info("")


@contextmanager
def log_step_context(logger: PipelineLogger, step_description: str, **start_metadata):
    """
    Context manager for automatic step logging with success/failure tracking.
    
    Usage:
        with log_step_context(logger, "Processing batch") as ctx:
            # Do work
            ctx.metadata['records_processed'] = 5000
    
    Args:
        logger: PipelineLogger instance
        step_description: Human-readable step description
        **start_metadata: Metadata to log at step start
    """
    class StepContext:
        def __init__(self):
            self.metadata: Dict[str, Any] = {}
    
    ctx = StepContext()
    log_step_start(logger, step_description, **start_metadata)
    
    try:
        yield ctx
        log_step_complete(logger, step_description, **ctx.metadata)
    except Exception as e:
        log_step_failed(logger, step_description, e)
        raise


def setup_simple_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a simple logger for utility scripts that don't need full pipeline logging.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter('%(levelname)-8s %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    return logger
