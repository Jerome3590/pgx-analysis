#!/usr/bin/env python3
"""Project-local entrypoint for the shared Cursor utility script."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


PROJECT_METADATA_DEFAULTS = {
    "S3_BUCKET": "mushin-solutions-project-metadata",
    "S3_PREFIX": "notebooks",
    "PROJECT_SLUG": "pgx-analysis",
}

for key, value in PROJECT_METADATA_DEFAULTS.items():
    os.environ.setdefault(key, value)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _find_shared_script() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "project_utility_scripts" / "cursor_setup.py",
        here.parent / "project_utility_scripts" / "cursor_setup.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise SystemExit(
        "Shared cursor_setup.py was not found. Expected one of:\n"
        f"{searched}\n"
        "Keep c:\\Projects\\project_utility_scripts next to pgx-analysis, "
        "or replace this shim with the full shared script."
    )


if __name__ == "__main__":
    script = _find_shared_script()
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")
