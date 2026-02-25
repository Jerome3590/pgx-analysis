#!/usr/bin/env python3
"""
Test PharmGKB API locally for throttling (429) and rate limits.

Calls the same endpoint used by fetch_vip_reports: GET /data/gene?symbol=...
Runs multiple requests and logs status code, elapsed time, and rate-limit headers.
Use this to confirm whether the API is throttling before running the full pipeline.

Usage (from repo root):
  python 9_dashboard_visuals/cohort_pgx/test_pharmgkb_throttle.py
  python 9_dashboard_visuals/cohort_pgx/test_pharmgkb_throttle.py --requests 20 --delay 0.2
  python 9_dashboard_visuals/cohort_pgx/test_pharmgkb_throttle.py --inspect CYP2D6   # show full response
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Repo root for imports if needed
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parents[1]) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parents[1]))

import requests

PHARMGKB_API_BASE = "https://api.pharmgkb.org/v1"
# Genes used by Cohort PGx (subset for quick test)
TEST_GENES = [
    "CYP2D6", "CYP2C19", "SLCO1B1", "CYP3A5", "DPYD",
    "TPMT", "UGT1A1", "CYP2C9", "NUDT15", "G6PD",
]


def test_pharmgkb_throttle(
    num_requests: int = 10,
    delay_seconds: float = 0.5,
    genes: list[str] | None = None,
) -> None:
    genes = genes or TEST_GENES
    # Cycle through genes if num_requests > len(genes)
    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "PGx-Analysis-Cohort/1.0 (throttle-test)",
    })

    url = f"{PHARMGKB_API_BASE}/data/gene"
    throttles = 0
    errors = 0
    ok = 0

    print(f"PharmGKB API throttle check")
    print(f"  Base: {PHARMGKB_API_BASE}")
    print(f"  Requests: {num_requests} (delay={delay_seconds}s between)")
    print(f"  Genes: {genes[:5]}{'...' if len(genes) > 5 else ''}")
    print()

    for i in range(num_requests):
        symbol = genes[i % len(genes)]
        params = {"symbol": symbol}
        t0 = time.perf_counter()
        try:
            r = session.get(url, params=params, timeout=30)
            elapsed = time.perf_counter() - t0
        except requests.exceptions.RequestException as e:
            elapsed = time.perf_counter() - t0
            print(f"  [{i+1}/{num_requests}] {symbol}  ERROR: {e}  ({elapsed:.2f}s)")
            errors += 1
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            continue

        # Rate-limit headers (PharmGKB may send these)
        headers_of_interest = {
            k: v for k, v in r.headers.items()
            if k.lower() in ("retry-after", "x-ratelimit-limit", "x-ratelimit-remaining", "x-rate-limit-remaining")
        }

        if r.status_code == 429:
            throttles += 1
            retry = r.headers.get("Retry-After", "?")
            print(f"  [{i+1}/{num_requests}] {symbol}  429 THROTTLED  ({elapsed:.2f}s)  Retry-After: {retry}  {headers_of_interest}")
        elif r.status_code >= 400:
            errors += 1
            print(f"  [{i+1}/{num_requests}] {symbol}  {r.status_code}  ({elapsed:.2f}s)  {headers_of_interest or r.text[:80]}")
        else:
            ok += 1
            if headers_of_interest:
                print(f"  [{i+1}/{num_requests}] {symbol}  {r.status_code}  ({elapsed:.2f}s)  {headers_of_interest}")
            else:
                print(f"  [{i+1}/{num_requests}] {symbol}  {r.status_code}  ({elapsed:.2f}s)")

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    print()
    print(f"Summary: {ok} OK, {throttles} throttled (429), {errors} other errors")
    if throttles > 0:
        print("  -> API is throttling; increase delay between requests or reduce batch size.")
    elif errors > 0:
        print("  -> Some requests failed (not throttling). Check status codes above.")
    else:
        print("  -> No throttling observed at this rate.")


def inspect_response(gene: str = "CYP2D6", save_path: Path | None = None) -> None:
    """Fetch one gene and print/save the actual response body."""
    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "PGx-Analysis-Cohort/1.0 (throttle-test)",
    })
    url = f"{PHARMGKB_API_BASE}/data/gene"
    print(f"GET {url}?symbol={gene}")
    r = session.get(url, params={"symbol": gene}, timeout=30)
    print(f"Status: {r.status_code}")
    print(f"Content-Type: {r.headers.get('Content-Type', '')}")
    print()

    try:
        data = r.json()
    except Exception as e:
        print(f"Body (not JSON): {r.text[:1000]}")
        return

    # Top-level keys
    print("Top-level keys:", list(data.keys()))
    for key in data:
        val = data[key]
        if isinstance(val, list):
            print(f"  {key}: list of length {len(val)}")
            if val and isinstance(val[0], dict):
                print(f"    first item keys: {list(val[0].keys())}")
        elif isinstance(val, dict):
            print(f"  {key}: dict with keys: {list(val.keys())[:15]}")
        else:
            print(f"  {key}: {type(val).__name__} = {str(val)[:80]}")

    # If data.data is the gene object, show key fields and vipSummary presence
    payload = data.get("data")
    if isinstance(payload, list) and payload:
        gene_obj = payload[0]
        print()
        print("First gene object (data[0]) summary:")
        for k in ("id", "name", "symbol", "vipId", "vipTier", "cpicGene", "hasCpicDosingGuideline"):
            print(f"  {k}: {gene_obj.get(k)}")
        vip = gene_obj.get("vipSummary")
        if isinstance(vip, dict):
            print(f"  vipSummary: dict with keys {list(vip.keys())}")
            for k, v in vip.items():
                if isinstance(v, str):
                    print(f"    vipSummary.{k}: {len(v)} chars  {repr(v[:100])}...")
                else:
                    print(f"    vipSummary.{k}: {type(v).__name__}")
        elif vip is None:
            print("  vipSummary: None (missing)")
        else:
            print(f"  vipSummary: {type(vip).__name__}")

    if save_path:
        save_path = Path(save_path)
        save_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print()
        print(f"Full response saved to: {save_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Test PharmGKB API for throttling (429)")
    ap.add_argument("--requests", "-n", type=int, default=10, help="Number of requests (default 10)")
    ap.add_argument("--delay", "-d", type=float, default=0.5, help="Seconds between requests (default 0.5)")
    ap.add_argument("--genes", nargs="+", default=None, help="Gene symbols (default: CYP2D6, CYP2C19, ...)")
    ap.add_argument("--inspect", metavar="GENE", nargs="?", const="CYP2D6", default=None,
                    help="Fetch one gene and print response structure (default CYP2D6)")
    ap.add_argument("--save", metavar="FILE", default=None, help="With --inspect: save full JSON to FILE")
    args = ap.parse_args()

    if args.inspect is not None:
        inspect_response(gene=args.inspect, save_path=Path(args.save) if args.save else SCRIPT_DIR / "pharmgkb_response_sample.json")
        return 0

    test_pharmgkb_throttle(
        num_requests=args.requests,
        delay_seconds=args.delay,
        genes=args.genes,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
