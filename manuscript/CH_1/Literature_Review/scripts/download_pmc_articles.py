"""
download_pmc_articles.py
Downloads PMC BioC JSON full-text for all articles with real PMC IDs.
Saves each JSON to {topic_dir}/pubmed_json_files/{pmc_id}.json

Run from: manuscript/CH_1/Literature_Review/
  python scripts/download_pmc_articles.py [--api-key KEY] [--workers N]
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────
ARTICLES_CSV  = Path("data/ontology/articles_tagged.csv")
BIOC_URL      = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmc_id}/unicode"
LOG_CSV       = Path("scripts/download_log.csv")
SLEEP_NO_KEY  = 0.35   # ≈ 2.86 req/sec (NCBI limit 3/sec without key)
SLEEP_API_KEY = 0.11   # ≈ 9 req/sec   (NCBI limit 10/sec with key)
MAX_RETRIES   = 3


def dest_dir(source_file: str) -> Path:
    """Return {topic_dir}/pubmed_json_files/ for a given source CSV path."""
    return Path(source_file).parent / "pubmed_json_files"


def download_one(pmc_id: str, out_path: Path, sleep_s: float) -> dict:
    url = BIOC_URL.format(pmc_id=pmc_id)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 100:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(r.content)
                time.sleep(sleep_s)
                return {"pmc_id": pmc_id, "status": "ok", "bytes": len(r.content)}
            elif r.status_code == 404:
                return {"pmc_id": pmc_id, "status": "not_found", "bytes": 0}
            else:
                time.sleep(sleep_s * attempt)  # back-off
        except requests.RequestException as e:
            time.sleep(sleep_s * attempt)
            if attempt == MAX_RETRIES:
                return {"pmc_id": pmc_id, "status": f"error:{e}", "bytes": 0}
    return {"pmc_id": pmc_id, "status": "failed_retries", "bytes": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None, help="NCBI API key (optional)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (keep 1 without API key)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N downloads (for testing)")
    args = parser.parse_args()

    sleep_s = SLEEP_API_KEY if args.api_key else SLEEP_NO_KEY
    if args.workers > 1 and not args.api_key:
        print("WARNING: parallel workers without API key will hit NCBI rate limits. "
              "Setting --workers 1. Use --api-key to enable parallel downloads.")
        args.workers = 1

    # ── Load articles ─────────────────────────────────────────────────────────
    if not ARTICLES_CSV.exists():
        sys.exit(f"ERROR: {ARTICLES_CSV} not found. Run organize_by_ontology.R first.")

    with open(ARTICLES_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Keep only rows with real PMC IDs (not HSH stubs)
    pmc_rows = [r for r in rows if r["pmc_id"].startswith("PMC")]
    print(f"Total articles:           {len(rows):,}")
    print(f"With real PMC ID:         {len(pmc_rows):,}")

    # Build download queue: skip already-downloaded non-empty files
    queue = []
    already_done = 0
    for r in pmc_rows:
        out_path = dest_dir(r["source_file"]) / f"{r['pmc_id']}.json"
        if out_path.exists() and out_path.stat().st_size > 100:
            already_done += 1
        else:
            queue.append((r["pmc_id"], out_path))

    # Deduplicate (same PMC ID may appear in multiple search results)
    seen = set()
    queue_dedup = []
    for pmc_id, out_path in queue:
        if pmc_id not in seen:
            seen.add(pmc_id)
            queue_dedup.append((pmc_id, out_path))
    queue = queue_dedup

    if args.limit:
        queue = queue[:args.limit]

    print(f"Already downloaded:       {already_done:,}")
    print(f"To download:              {len(queue):,}")
    print(f"Sleep per request:        {sleep_s}s  (workers={args.workers})")
    print()

    if not queue:
        print("Nothing to download. All PMC JSONs already present.")
        return

    # ── Download ──────────────────────────────────────────────────────────────
    results = []
    log_rows = []
    n_ok = n_not_found = n_error = 0

    def report(i, total, res):
        nonlocal n_ok, n_not_found, n_error
        symbol = "✓" if res["status"] == "ok" else ("✗" if "not_found" in res["status"] else "!")
        if res["status"] == "ok":
            n_ok += 1
        elif "not_found" in res["status"]:
            n_not_found += 1
        else:
            n_error += 1
        if i % 50 == 0 or i <= 5:
            print(f"  [{i:>5}/{total}] {symbol} {res['pmc_id']:15s}  {res['status']:12s}  {res['bytes']:,} bytes")
        log_rows.append(res)

    if args.workers == 1:
        for i, (pmc_id, out_path) in enumerate(queue, 1):
            res = download_one(pmc_id, out_path, sleep_s)
            report(i, len(queue), res)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(download_one, pmc_id, out_path, sleep_s): (i+1)
                       for i, (pmc_id, out_path) in enumerate(queue)}
            for fut in as_completed(futures):
                i = futures[fut]
                report(i, len(queue), fut.result())

    # ── Write log ─────────────────────────────────────────────────────────────
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pmc_id", "status", "bytes"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\n── Download complete ────────────────────────────────────")
    print(f"  Success:    {n_ok:,}")
    print(f"  Not found:  {n_not_found:,}  (article not in PMC OA)")
    print(f"  Errors:     {n_error:,}")
    print(f"  Log:        {LOG_CSV}")
    print(f"\nRe-run prisma_tracker.R to refresh PRISMA counts.")


if __name__ == "__main__":
    main()
