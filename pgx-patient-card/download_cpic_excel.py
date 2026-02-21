#!/usr/bin/env python3
"""
Download CPIC gene-drug pairs Excel file with SSL fallback.

In environments where the system CA bundle is missing or incomplete
(e.g. minimal Linux, containers), the first request may fail with
CERTIFICATE_VERIFY_FAILED. This script retries with an unverified SSL context.
"""

import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

CPIC_XLSX_URL = "https://files.cpicpgx.org/data/report/current/pair/cpic_gene-drug_pairs.xlsx"


def download_cpic_excel(dest_path: Path, timeout: int = 60) -> bool:
    """Download CPIC Excel to dest_path. Returns True on success."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(CPIC_XLSX_URL)
    ctx_secure = ssl.create_default_context()
    ctx_insecure = ssl._create_unverified_context()

    for ctx, label in [(ctx_secure, "default"), (ctx_insecure, "no verify")]:
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                dest_path.write_bytes(resp.read())
            if dest_path.stat().st_size > 0:
                print(f"✓ Downloaded ({label}): {dest_path} ({dest_path.stat().st_size / 1024:.1f} KB)")
                return True
        except urllib.error.URLError as ue:
            if "CERTIFICATE_VERIFY_FAILED" in str(ue.reason) or "SSL" in str(ue.reason):
                continue
            print(f"✗ Download failed: {ue}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"✗ Download failed: {e}", file=sys.stderr)
            return False

    print("✗ Download failed (tried default and no-verify SSL)", file=sys.stderr)
    return False


def main() -> int:
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    dest = data_dir / "cpic_gene-drug_pairs.xlsx"

    if dest.exists() and dest.stat().st_size > 0:
        print(f"✓ CPIC Excel already exists: {dest} ({dest.stat().st_size / 1024:.1f} KB)")
        return 0

    print(f"Downloading CPIC Excel from {CPIC_XLSX_URL}...")
    if download_cpic_excel(dest):
        return 0
    print("  Try manual download:", file=sys.stderr)
    print(f"  wget {CPIC_XLSX_URL} -O {dest}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
