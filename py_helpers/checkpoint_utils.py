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
        logger.info(f"[OK] All {len(s3_paths)} outputs exist in S3, step can be skipped")
    return True


def upload_file_to_s3(
    local_path: Path,
    s3_path: str,
    logger: Optional[logging.Logger] = None,
    check_exists: bool = True,
    tracker: Optional[Any] = None,
    viz_type: Optional[str] = None,
    cohort: Optional[str] = None,
    age_band: Optional[str] = None,
    item_type: Optional[str] = None
) -> bool:
    """
    Upload a local file to S3 (idempotent - skips if already exists).
    
    Args:
        local_path: Local file path
        s3_path: S3 destination path
        logger: Optional logger
        check_exists: If True, check if file already exists in S3 before uploading (idempotent)
        tracker: Optional S3UploadTracker instance for local tracking
        viz_type: Visualization type (for tracking)
        cohort: Cohort name (for tracking)
        age_band: Age band (for tracking)
        item_type: Item type (for tracking, FP-Growth only)
    
    Returns:
        True if upload successful or file already exists, False otherwise
    """
    if not local_path.exists():
        if logger:
            logger.warning(f"Local file does not exist: {local_path}")
        if tracker and viz_type and cohort and age_band:
            from py_helpers.s3_upload_tracker import get_file_size_mb
            tracker.log_upload(
                local_path=str(local_path),
                s3_path=s3_path,
                visualization_type=viz_type,
                cohort=cohort,
                age_band=age_band,
                item_type=item_type,
                file_size_mb=0.0,
                success=False,
                error="Local file does not exist"
            )
        return False
    
    try:
        bucket, key = _parse_s3_path(s3_path)
        
        # Check if file already exists in S3 (idempotent)
        if check_exists:
            try:
                s3_client.head_object(Bucket=bucket, Key=key)
                if logger:
                    logger.info(f"[OK] File already exists in S3: {s3_path} (skipping upload)")
                if tracker and viz_type and cohort and age_band:
                    from py_helpers.s3_upload_tracker import get_file_size_mb
                    tracker.log_upload(
                        local_path=str(local_path),
                        s3_path=s3_path,
                        visualization_type=viz_type,
                        cohort=cohort,
                        age_band=age_band,
                        item_type=item_type,
                        file_size_mb=get_file_size_mb(local_path),
                        success=True,
                        metadata={"skipped": True, "reason": "Already exists in S3"}
                    )
                return True
            except s3_client.exceptions.ClientError as e:
                if e.response["Error"]["Code"] not in ["404", "NoSuchKey"]:
                    # If it's not a 404, re-raise (might be permission issue)
                    raise
        
        # Upload file
        s3_client.upload_file(str(local_path), bucket, key)
        if logger:
            logger.info(f"[OK] Uploaded to S3: {s3_path}")
        
        # Track successful upload
        if tracker and viz_type and cohort and age_band:
            from py_helpers.s3_upload_tracker import get_file_size_mb
            tracker.log_upload(
                local_path=str(local_path),
                s3_path=s3_path,
                visualization_type=viz_type,
                cohort=cohort,
                age_band=age_band,
                item_type=item_type,
                file_size_mb=get_file_size_mb(local_path),
                success=True
            )
        return True
    except Exception as e:
        if logger:
            logger.error(f"Failed to upload {local_path} to {s3_path}: {e}")
        if tracker and viz_type and cohort and age_band:
            from py_helpers.s3_upload_tracker import get_file_size_mb
            tracker.log_upload(
                local_path=str(local_path),
                s3_path=s3_path,
                visualization_type=viz_type,
                cohort=cohort,
                age_band=age_band,
                item_type=item_type,
                file_size_mb=get_file_size_mb(local_path),
                success=False,
                error=str(e)
            )
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
        step_name: Name of the step (e.g., "4_model_data", "4b_dtw_filter")
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
            logger.info(f"[OK] Saved checkpoint to s3://pgx-repository/{checkpoint_key}")
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
            logger.info(f"[OK] Checkpoint exists: s3://pgx-repository/{checkpoint_key}")
        return True
    except s3_client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ["404", "NoSuchKey"]:
            return False
        raise
    except Exception:
        return False


def delete_step_checkpoint(
    step_name: str,
    cohort: str,
    age_band: str,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Delete a step checkpoint from S3 so the step will run again.

    Args:
        step_name: Name of the step (e.g., "3b_feature_importance_eda")
        cohort: Cohort name (e.g., "opioid_ed")
        age_band: Age band (e.g., "13-24")
        logger: Optional logger

    Returns:
        True if deleted or already missing, False on error
    """
    checkpoint_key = (
        f"pipeline_checkpoints/{step_name}/{cohort}/{age_band.replace('-', '_')}/checkpoint.json"
    )
    try:
        s3_client.delete_object(Bucket="pgx-repository", Key=checkpoint_key)
        if logger:
            logger.info(f"[OK] Deleted checkpoint: s3://pgx-repository/{checkpoint_key}")
        return True
    except s3_client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ["404", "NoSuchKey"]:
            if logger:
                logger.info(f"Checkpoint already missing: s3://pgx-repository/{checkpoint_key}")
            return True
        if logger:
            logger.warning(f"Failed to delete checkpoint: {e}")
        return False
    except Exception as e:
        if logger:
            logger.warning(f"Failed to delete checkpoint: {e}")
        return False


def clear_step_checkpoints(
    step_name: str,
    cohort: str,
    age_bands: List[str],
    logger: Optional[logging.Logger] = None
) -> int:
    """
    Delete checkpoints for a step/cohort for the given age bands.
    Returns the number of checkpoints deleted (or already missing).
    """
    deleted = 0
    for age_band in age_bands:
        if delete_step_checkpoint(step_name, cohort, age_band, logger):
            deleted += 1
    return deleted


def _parse_s3_path(s3_path: str) -> tuple[str, str]:
    """Parse S3 path into bucket and key."""
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}")
    
    parts = s3_path[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    
    return bucket, key

