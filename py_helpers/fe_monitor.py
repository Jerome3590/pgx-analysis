"""
Feature-engineering pipeline monitoring utilities.

This module provides:
- Lightweight OS/runtime detection
- Context managers for module / function / step resource logging
- Best-effort S3 mirroring for checkpoints and logs

All console output is ASCII-only for Windows compatibility.
"""

import os
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterator

import psutil  # type: ignore

try:
    import boto3  # type: ignore
except Exception:  # pragma: no cover - boto3 may not be installed in all envs
    boto3 = None  # type: ignore


# ---------------------------------------------------------------------------
# OS / runtime detection
# ---------------------------------------------------------------------------


@dataclass
class RuntimeEnvironment:
    os_name: str
    logical_cores: int
    ram_gb: Optional[int]
    fast_root: Path


def detect_runtime_environment(project_root: Optional[Path] = None) -> RuntimeEnvironment:
    """
    Detect basic runtime characteristics and choose a fast local root.

    Rules (can be overridden via env vars):
    - On Windows: assume 14 cores / 64 GB, use project_root as fast_root.
    - On Linux: assume 32 cores / 1 TB, prefer NVMe-like mount if present,
      falling back to project_root.
    """
    import platform

    project_root = project_root or Path.cwd()
    system = platform.system().lower()
    logical_cores = os.cpu_count() or 1

    # Allow explicit override of fast root
    fast_root_env = os.environ.get("PGX_FAST_ROOT")
    if fast_root_env:
        fast_root = Path(fast_root_env)
    else:
        if system == "windows":
            fast_root = project_root
        else:
            # Prefer common NVMe-style mount locations on Linux
            candidates = [
                Path("/mnt/nvme"),
                Path("/mnt/nvme0"),
                Path("/mnt/nvme0n1"),
                Path("/nvme"),
            ]
            fast_root = next((p for p in candidates if p.exists()), project_root)

    if system == "windows":
        ram_gb = 64
    elif system == "linux":
        ram_gb = 1024
    else:
        ram_gb = None

    return RuntimeEnvironment(
        os_name=system,
        logical_cores=logical_cores,
        ram_gb=ram_gb,
        fast_root=fast_root,
    )


# ---------------------------------------------------------------------------
# Resource-tracking context manager
# ---------------------------------------------------------------------------


@dataclass
class BlockStats:
    scope: str
    module: str
    name: str
    start_time: float
    end_time: float
    duration_sec: float
    mem_start_mb: float
    mem_end_mb: float
    mem_max_mb: float
    cpu_start_pct: float
    cpu_end_pct: float
    cpu_max_pct: float


class ResourceBlock:
    """
    Context manager that captures wall-clock time, system memory, and CPU usage.

    Scope should be one of: "module", "function", "step".
    Measurements are system-wide (not per-process) so that child processes
    (e.g., R scripts) are included in the aggregate.
    """

    def __init__(
        self,
        module: str,
        name: str,
        scope: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.module = module
        self.name = name
        self.scope = scope
        self.logger = logger or logging.getLogger(__name__)
        self._start_time: float = 0.0
        self._start_mem_mb: float = 0.0
        self._start_cpu_pct: float = 0.0

    def __enter__(self) -> "ResourceBlock":
        vm = psutil.virtual_memory()
        self._start_time = time.time()
        self._start_mem_mb = vm.used / (1024 * 1024)
        # First call to cpu_percent establishes the baseline without blocking.
        self._start_cpu_pct = psutil.cpu_percent(interval=None)

        self._log(
            "START",
            mem_mb=self._start_mem_mb,
            cpu_pct=self._start_cpu_pct,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        vm = psutil.virtual_memory()
        end_time = time.time()
        end_mem_mb = vm.used / (1024 * 1024)
        end_cpu_pct = psutil.cpu_percent(interval=None)

        duration = end_time - self._start_time
        mem_max_mb = max(self._start_mem_mb, end_mem_mb)
        cpu_max_pct = max(self._start_cpu_pct, end_cpu_pct)

        stats = BlockStats(
            scope=self.scope,
            module=self.module,
            name=self.name,
            start_time=self._start_time,
            end_time=end_time,
            duration_sec=duration,
            mem_start_mb=self._start_mem_mb,
            mem_end_mb=end_mem_mb,
            mem_max_mb=mem_max_mb,
            cpu_start_pct=self._start_cpu_pct,
            cpu_end_pct=end_cpu_pct,
            cpu_max_pct=cpu_max_pct,
        )

        level = "END_OK" if exc_type is None else "END_ERROR"
        self._log(
            level,
            mem_mb=end_mem_mb,
            cpu_pct=end_cpu_pct,
            extra=stats,
        )
        if exc_type is not None and exc_val is not None:
            self.logger.error(
                "[%s][%s][%s] exception: %s",
                self.scope.upper(),
                self.module,
                self.name,
                exc_val,
                exc_info=(exc_type, exc_val, exc_tb),
            )

    def _log(
        self,
        phase: str,
        mem_mb: float,
        cpu_pct: float,
        extra: Optional[BlockStats] = None,
    ) -> None:
        prefix = f"[{self.scope.upper()}][{self.module}][{self.name}] {phase}"
        msg = (
            f"{prefix} time={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} "
            f"mem_mb={mem_mb:.1f} cpu_pct={cpu_pct:.1f}"
        )
        if extra is not None:
            msg += (
                f" duration_sec={extra.duration_sec:.1f}"
                f" mem_start_mb={extra.mem_start_mb:.1f}"
                f" mem_end_mb={extra.mem_end_mb:.1f}"
                f" mem_max_mb={extra.mem_max_mb:.1f}"
                f" cpu_start_pct={extra.cpu_start_pct:.1f}"
                f" cpu_end_pct={extra.cpu_end_pct:.1f}"
                f" cpu_max_pct={extra.cpu_max_pct:.1f}"
            )
        self.logger.info(msg)


def module_block(module: str, logger: Optional[logging.Logger] = None) -> Iterator[ResourceBlock]:
    """Convenience wrapper for module-level blocks."""
    return ResourceBlock(module=module, name="module", scope="module", logger=logger)


def function_block(
    module: str, name: str, logger: Optional[logging.Logger] = None
) -> Iterator[ResourceBlock]:
    """Convenience wrapper for function-level blocks."""
    return ResourceBlock(module=module, name=name, scope="function", logger=logger)


def step_block(
    module: str, name: str, logger: Optional[logging.Logger] = None
) -> Iterator[ResourceBlock]:
    """Convenience wrapper for step-level blocks."""
    return ResourceBlock(module=module, name=name, scope="step", logger=logger)


# ---------------------------------------------------------------------------
# Checkpoint and log mirroring to S3 (best-effort)
# ---------------------------------------------------------------------------


S3_BUCKET = os.environ.get("PGX_S3_BUCKET", "pgx-repository")


def _put_s3_object(key: str, data: bytes, logger: Optional[logging.Logger] = None) -> None:
    if boto3 is None:
        if logger:
            logger.warning("boto3 is not available; skipping S3 upload for key=%s", key)
        else:
            print("boto3 is not available; skipping S3 upload for key={0}".format(key))
        return

    try:
        client = boto3.client("s3")
        client.put_object(Bucket=S3_BUCKET, Key=key, Body=data)
        if logger:
            logger.info("Uploaded to s3://%s/%s", S3_BUCKET, key)
        else:
            print("Uploaded to s3://{0}/{1}".format(S3_BUCKET, key))
    except Exception as exc:  # pragma: no cover - best-effort path
        if logger:
            logger.warning("Could not upload to s3://%s/%s: %s", S3_BUCKET, key, exc)
        else:
            print("Could not upload to s3://{0}/{1}: {2}".format(S3_BUCKET, key, exc))


def mirror_checkpoint_to_s3(
    feature_step: str,
    cohort: str,
    age_band: str,
    local_path: Path,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Mirror a checkpoint artifact to S3.

    S3 layout:
      s3://pgx-repository/{feature_step}_checkpoint/{cohort}/{age_band}/{filename}
    """
    if not local_path.exists():
        if logger:
            logger.warning("Checkpoint path does not exist; skipping S3: %s", local_path)
        else:
            print("Checkpoint path does not exist; skipping S3: {0}".format(local_path))
        return

    key = "{step}_checkpoint/{cohort}/{age_band}/{name}".format(
        step=feature_step,
        cohort=cohort,
        age_band=age_band,
        name=local_path.name,
    )
    data = local_path.read_bytes()
    _put_s3_object(key, data, logger=logger)


def mirror_log_to_s3(
    feature_step: str,
    cohort: str,
    age_band: str,
    log_path: Path,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Mirror a log file to S3.

    S3 layout:
      s3://pgx-repository/{feature_step}_log/{cohort}/{age_band}/{filename}
    """
    if not log_path.exists():
        if logger:
            logger.warning("Log path does not exist; skipping S3: %s", log_path)
        else:
            print("Log path does not exist; skipping S3: {0}".format(log_path))
        return

    key = "{step}_log/{cohort}/{age_band}/{name}".format(
        step=feature_step,
        cohort=cohort,
        age_band=age_band,
        name=log_path.name,
    )
    data = log_path.read_bytes()
    _put_s3_object(key, data, logger=logger)


