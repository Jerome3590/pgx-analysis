"""
S3-based checkpoint and idempotency utilities for pipeline steps.

This module provides functions to:
1. Check if step outputs exist in S3 (idempotency)
2. Upload step outputs to S3 after completion
3. Save checkpoint metadata to S3
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    from py_helpers.common_imports import s3_client, S3_BUCKET
except ImportError:
    import boto3
    s3_client = boto3.client("s3")
    S3_BUCKET = "pgxdatalake"


def check_s3_output_exists(s3_path: str) -> bool:
    """
    Check if an S3 object exists.
    
    Args:
        s3_path: Full S3 path (e.g., s3://bucket/key)
    
    Returns:
        True if object exists, False otherwise
    """
    try:
        bucket, key = _parse_s3_path(s3_path)
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except s3_client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ["404", "NoSuchKey"]:
            return False
        raise
    except Exception:
        return False


def check_step_outputs_exist(s3_paths: List[str], logger: Optional[logging.Logger] = None) -> bool:
    """
    Check if all step outputs exist in S3.
    
    Args:
        s3_paths: List of S3 paths to check
        logger: Optional logger
    
    Returns:
        True if all outputs exist, False otherwise
    """
    if not s3_paths:
        return False
    
    for s3_path in s3_paths:
        if not check_s3_output_exists(s3_path):
            if logger:
                logger.debug(f"Output not found in S3: {s3_path}")
            return False
    
    if logger:
        logger.info(f"✓ All {len(s3_paths)} outputs exist in S3, step can be skipped")
    return True


def upload_file_to_s3(local_path: Path, s3_path: str, logger: Optional[logging.Logger] = None) -> bool:
    """
    Upload a local file to S3.
    
    Args:
        local_path: Local file path
        s3_path: S3 destination path
        logger: Optional logger
    
    Returns:
        True if upload successful, False otherwise
    """
    if not local_path.exists():
        if logger:
            logger.warning(f"Local file does not exist: {local_path}")
        return False
    
    try:
        bucket, key = _parse_s3_path(s3_path)
        s3_client.upload_file(str(local_path), bucket, key)
        if logger:
            logger.info(f"✓ Uploaded to S3: {s3_path}")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Failed to upload {local_path} to {s3_path}: {e}")
        return False


def save_step_checkpoint(
    step_name: str,
    cohort: str,
    age_band: str,
    metadata: Optional[Dict[str, Any]] = None,
    output_paths: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Save checkpoint metadata to S3.
    
    Args:
        step_name: Name of the step (e.g., "4a_model_data", "4b_dtw_filter")
        cohort: Cohort name
        age_band: Age band
        metadata: Optional metadata dictionary
        output_paths: Optional list of S3 output paths
        logger: Optional logger
    
    Returns:
        True if checkpoint saved successfully
    """
    checkpoint_data = {
        "step_name": step_name,
        "cohort": cohort,
        "age_band": age_band,
        "completed_at": datetime.utcnow().isoformat(),
        "status": "completed",
        "metadata": metadata or {},
        "output_paths": output_paths or [],
    }
    
    # S3 checkpoint path: s3://pgx-repository/pipeline_checkpoints/{step_name}/{cohort}/{age_band}/checkpoint.json
    checkpoint_key = (
        f"pipeline_checkpoints/{step_name}/{cohort}/{age_band.replace('-', '_')}/checkpoint.json"
    )
    
    try:
        s3_client.put_object(
            Bucket="pgx-repository",
            Key=checkpoint_key,
            Body=json.dumps(checkpoint_data, indent=2),
            ContentType="application/json"
        )
        if logger:
            logger.info(f"✓ Saved checkpoint to s3://pgx-repository/{checkpoint_key}")
        return True
    except Exception as e:
        if logger:
            logger.warning(f"Failed to save checkpoint: {e}")
        return False


def check_step_checkpoint_exists(
    step_name: str,
    cohort: str,
    age_band: str,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Check if a step checkpoint exists in S3.
    
    Args:
        step_name: Name of the step
        cohort: Cohort name
        age_band: Age band
        logger: Optional logger
    
    Returns:
        True if checkpoint exists, False otherwise
    """
    checkpoint_key = (
        f"pipeline_checkpoints/{step_name}/{cohort}/{age_band.replace('-', '_')}/checkpoint.json"
    )
    
    try:
        s3_client.head_object(Bucket="pgx-repository", Key=checkpoint_key)
        if logger:
            logger.info(f"✓ Checkpoint exists: s3://pgx-repository/{checkpoint_key}")
        return True
    except s3_client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ["404", "NoSuchKey"]:
            return False
        raise
    except Exception:
        return False


def _parse_s3_path(s3_path: str) -> tuple[str, str]:
    """Parse S3 path into bucket and key."""
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}")
    
    parts = s3_path[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    
    return bucket, key

