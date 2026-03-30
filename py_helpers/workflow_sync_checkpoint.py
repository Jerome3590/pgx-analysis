"""
S3-to-NVMe sync and checkpoint helpers for workflow notebooks.

Use these to:
1. Sync required inputs from S3 to local/NVMe before each phase (idempotent: aws s3 sync).
2. Check/save S3 checkpoints so steps are idempotent (skip if already completed).

Usage in notebooks:
    from py_helpers.workflow_sync_checkpoint import (
        sync_s3_to_local,
        get_data_root,
        check_step_checkpoint_exists,
        save_step_checkpoint,
    )
    sync_s3_to_local("s3://pgxdatalake/gold/cohorts/", get_data_root() / "gold" / "cohorts")
    if check_step_checkpoint_exists("1b_apcd_event_filter", cohort, age_band):
        print("Step 1b already done, skipping")
    else:
        # run step ...
        save_step_checkpoint("1b_apcd_event_filter", cohort, age_band)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence

from py_helpers.env_utils import get_data_root
from py_helpers.checkpoint_utils import (
    check_step_checkpoint_exists,
    save_step_checkpoint,
    delete_step_checkpoint,
    clear_step_checkpoints,
)

__all__ = [
    "sync_s3_to_local",
    "sync_local_to_s3",
    "get_data_root",
    "check_step_checkpoint_exists",
    "save_step_checkpoint",
    "delete_step_checkpoint",
    "clear_step_checkpoints",
]

logger = logging.getLogger(__name__)


def sync_s3_to_local(
    s3_prefix: str,
    local_dir: Path,
    *,
    profile: Optional[str] = None,
    no_progress: bool = True,
    timeout_seconds: int = 7200,
) -> bool:
    """
    Sync S3 prefix to local directory (idempotent: aws s3 sync only updates changed/missing files).

    Args:
        s3_prefix: S3 URI (e.g. s3://pgxdatalake/gold/cohorts/)
        local_dir: Local destination directory (e.g. get_data_root() / "gold" / "cohorts")
        profile: Optional AWS CLI profile (e.g. mushin)
        no_progress: If True, pass --no-progress to aws s3 sync
        timeout_seconds: Max time for sync (default 2 hours)

    Returns:
        True if sync succeeded (or local already sufficient), False on failure.
    """
    if not s3_prefix.endswith("/"):
        s3_prefix += "/"
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    aws = os.environ.get("AWS_EXECUTABLE") or shutil.which("aws") or "aws"
    cmd = [aws, "s3", "sync", s3_prefix, str(local_dir)]
    if no_progress:
        cmd.append("--no-progress")
    if profile:
        cmd.extend(["--profile", profile])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if result.returncode == 0:
            logger.info("Sync completed: %s -> %s", s3_prefix, local_dir)
            return True
        logger.warning("Sync failed (exit %s): %s", result.returncode, result.stderr)
        return False
    except subprocess.TimeoutExpired:
        logger.error("Sync timed out after %s s: %s -> %s", timeout_seconds, s3_prefix, local_dir)
        return False
    except FileNotFoundError:
        logger.warning("AWS CLI not found; skip sync. Set AWS_EXECUTABLE or ensure 'aws' is on PATH.")
        return False
    except Exception as e:
        logger.exception("Sync error: %s", e)
        return False


def sync_local_to_s3(
    local_dir: Path,
    s3_prefix: str,
    *,
    profile: Optional[str] = None,
    no_progress: bool = True,
    timeout_seconds: int = 7200,
    extra_args: Optional[Sequence[str]] = None,
) -> bool:
    """
    Sync a local directory to an S3 prefix (idempotent ``aws s3 sync``).

    Only **uploads** new or changed objects (size / etag); does **not** pass
    ``--delete``, so objects that exist only in S3 are left unchanged — no need
    to clear old files before syncing.

    Args:
        local_dir: Local source directory (must exist; use trailing path style
            consistent with AWS CLI; contents are uploaded under ``s3_prefix``).
        s3_prefix: Destination S3 URI (e.g. ``s3://pgxdatalake/gold/final_model/cohort/13-24/``).
        profile: Optional AWS CLI profile.
        no_progress: If True, pass ``--no-progress``.
        timeout_seconds: Max time for sync.
        extra_args: Extra CLI tokens (e.g. ``--exclude``, ``--include`` filters).

    Returns:
        True if the sync subprocess exited 0, False otherwise.
    """
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        logger.warning("sync_local_to_s3: not a directory: %s", local_dir)
        return False

    if not s3_prefix.endswith("/"):
        s3_prefix += "/"

    aws = os.environ.get("AWS_EXECUTABLE") or shutil.which("aws") or "aws"
    cmd: List[str] = [aws, "s3", "sync", str(local_dir), s3_prefix]
    if no_progress:
        cmd.append("--no-progress")
    if profile:
        cmd.extend(["--profile", profile])
    if extra_args:
        cmd.extend(list(extra_args))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if result.returncode == 0:
            logger.info("Sync completed: %s -> %s", local_dir, s3_prefix)
            return True
        logger.warning("Sync failed (exit %s): %s", result.returncode, result.stderr)
        return False
    except subprocess.TimeoutExpired:
        logger.error("Sync timed out after %s s: %s -> %s", timeout_seconds, local_dir, s3_prefix)
        return False
    except FileNotFoundError:
        logger.warning("AWS CLI not found; skip sync. Set AWS_EXECUTABLE or ensure 'aws' is on PATH.")
        return False
    except Exception as e:
        logger.exception("Sync error: %s", e)
        return False
