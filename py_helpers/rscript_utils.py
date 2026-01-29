"""
Rscript path finding utilities.

Provides functions to find Rscript executable across different platforms
and environments.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Optional


def find_rscript(configured: Optional[Path] = None) -> Optional[str]:
    """
    Return a path to Rscript if found, else None.
    
    Checks in order:
      1) configured path (if provided)
      2) PATH
      3) common locations
    
    Args:
        configured: Optional configured path to Rscript (e.g., from environment variable)
    
    Returns:
        Path to Rscript executable as string, or None if not found
    """
    # 1) Configured path
    if configured is not None:
        configured = Path(configured)
        if configured.exists() and configured.is_file():
            return str(configured)

    # 2) PATH
    p = shutil.which("Rscript")
    if p:
        return p

    # 3) Common locations
    common_paths = [
        Path("/usr/local/bin/Rscript"),  # EC2 default
        Path("/usr/bin/Rscript"),
        Path("C:/Program Files/R/R-4.5.0/bin/Rscript.exe"),
        Path("C:/Program Files/R/R-4.4.0/bin/Rscript.exe"),
        Path("C:/Program Files (x86)/R/R-4.5.0/bin/Rscript.exe"),
        Path("C:/Program Files (x86)/R/R-4.4.0/bin/Rscript.exe"),
    ]
    for path in common_paths:
        if path.exists() and path.is_file():
            return str(path)

    return None


def print_rscript_version(rscript_path: str) -> None:
    """
    Print Rscript version. Note: Rscript often writes version to stderr.
    
    Args:
        rscript_path: Path to Rscript executable
    """
    try:
        result = subprocess.run(
            [rscript_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        version_line = out.splitlines()[0] if out else (err.splitlines()[0] if err else "Unknown")
        print(f"Rscript: {rscript_path}")
        print(f"Version: {version_line}")
    except Exception as e:
        print(f"Rscript: {rscript_path}")
        print(f"Could not check version: {e}")
