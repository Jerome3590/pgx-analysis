#!/usr/bin/env python3
"""Remove AWS CLI leftover temp files from bupar plots (e.g. *.png.15C56520, *.html.CFBb4CCb)."""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUPAR_PLOTS = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "bupar" / "outputs"
# AWS CLI temp suffix: .png or .html followed by dot and 6+ alphanumeric
TEMP_PATTERN = re.compile(r"\.(png|html)\.[0-9A-Za-z]{6,}$", re.IGNORECASE)


def main() -> None:
    removed = 0
    for f in BUPAR_PLOTS.rglob("*"):
        if not f.is_file():
            continue
        if TEMP_PATTERN.search(f.name):
            f.unlink()
            removed += 1
            print(f"Removed: {f.relative_to(REPO_ROOT)}")
    print(f"Done. Removed {removed} temp file(s).")


if __name__ == "__main__":
    main()
