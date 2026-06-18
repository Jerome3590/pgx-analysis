#!/usr/bin/env python3
"""Thin shim for notebook output sync commands."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    cursor_setup = Path(__file__).resolve().with_name("cursor_setup.py")
    raise SystemExit(
        subprocess.run(
            [sys.executable, str(cursor_setup), "push-outputs", *sys.argv[1:]],
            check=False,
        ).returncode
    )
