"""
zotero_import.py
Bulk-imports included articles (human_decision == "include") into a Zotero library
via the Zotero Web API, then triggers metadata retrieval via CrossRef.

Prerequisites:
  1. Zotero account at https://www.zotero.org/user/login
  2. API key:  https://www.zotero.org/settings/security → "Create new private key"
     - Scope: Personal Library → Read/Write
  3. Add to secrets/secrets.txt:
       zotero_api_key=YOUR_KEY_HERE
       zotero_user_id=6037399
  Env vars ZOTERO_API_KEY / ZOTERO_USER_ID are used as fallback if not in secrets.txt.

Run from: manuscript/CH_1/Literature_Review/
  python scripts/zotero_import.py [--dry-run] [--screened]
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

SECRETS_FILE = Path("secrets/secrets.txt")


def load_secrets() -> dict:
    """Parse secrets/secrets.txt (key=value, # comments). Same format as vcu_download.js."""
    cfg = {}
    if not SECRETS_FILE.exists():
        return cfg
    for line in SECRETS_FILE.read_text(encoding="utf-8").splitlines():
        m = re.match(r'^\s*([^#=\s][^=]*)=(.*)$', line)
        if m:
            cfg[m.group(1).strip()] = m.group(2).strip()
    return cfg

SCREENED_CSV = Path("data/ontology/articles_screened.csv")
TAGGED_CSV   = Path("data/ontology/articles_tagged.csv")
LOG_CSV      = Path("scripts/zotero_import_log.csv")
BATCH_SIZE   = 50   # Zotero API max items per POST
SLEEP_S      = 0.2  # polite rate limit


def get_zotero_headers(api_key: str) -> dict:
    return {
        "Zotero-API-Key": api_key,
        "Zotero-API-Version": "3",
        "Content-Type": "application/json",
    }


def build_library_url(user_id: str, group_id: str) -> str:
    if group_id:
        return f"https://api.zotero.org/groups/{group_id}"
    return f"https://api.zotero.org/users/{user_id}"


def row_to_zotero_item(row: dict) -> dict:
    """Convert a CSV row to a Zotero journalArticle item."""
    title   = row.get("title", "").strip().title()
    authors_str = row.get("authors", "")
    year    = str(row.get("pubdate", "")).strip()[:4]
    pmc_id  = row.get("pmc_id", "").strip()
    doi     = row.get("doi", "").strip()   # populated by unpaywall_lookup if run first

    # Parse "Last, First, Last2, First2, ..." author string
    creators = []
    for name in authors_str.split(","):
        name = name.strip()
        if not name:
            continue
        parts = name.rsplit(" ", 1)
        if len(parts) == 2:
            creators.append({"creatorType": "author",
                              "lastName": parts[0], "firstName": parts[1]})
        else:
            creators.append({"creatorType": "author", "name": name})

    extra_parts = []
    if pmc_id:
        extra_parts.append(f"PMCID: {pmc_id}")
    if doi:
        extra_parts.append(f"DOI: {doi}")

    item = {
        "itemType": "journalArticle",
        "title": title,
        "creators": creators,
        "date": year,
        "extra": "\n".join(extra_parts),
        "tags": [{"tag": row.get("ooda_phase_primary", "")},
                 {"tag": row.get("node_primary", "")},
                 {"tag": "pgx-lit-review"}],
        "collections": [],
        "relations": {},
    }
    if doi:
        item["DOI"] = doi
    if pmc_id and pmc_id.startswith("PMC"):
        item["url"] = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/"

    return item


def post_batch(items: list, base_url: str, headers: dict, dry_run: bool) -> list[dict]:
    """POST a batch of items to Zotero. Returns list of result dicts."""
    if dry_run:
        return [{"key": f"DRY{i:04d}", "status": "dry_run"} for i in range(len(items))]
    url = f"{base_url}/items"
    r = requests.post(url, headers=headers, data=json.dumps(items), timeout=30)
    if r.status_code not in (200, 201):
        return [{"key": None, "status": f"http_{r.status_code}", "error": r.text[:200]}
                for _ in items]
    data = r.json()
    results = []
    for k, v in data.get("success", {}).items():
        results.append({"key": v, "status": "created"})
    for k, v in data.get("unchanged", {}).items():
        results.append({"key": v, "status": "unchanged"})
    for k, v in data.get("failed", {}).items():
        results.append({"key": None, "status": "failed", "error": str(v)})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate without posting to Zotero")
    parser.add_argument("--screened", action="store_true",
                        help="Use articles_screened.csv (human_decision==include); "
                             "default uses articles_tagged.csv (all 9,454)")
    parser.add_argument("--collection", default="",
                        help="Zotero collection key to add items to")
    args = parser.parse_args()

    secrets  = load_secrets()
    api_key  = secrets.get("zotero_api_key") or os.environ.get("ZOTERO_API_KEY", "")
    user_id  = secrets.get("zotero_user_id") or os.environ.get("ZOTERO_USER_ID", "")
    group_id = secrets.get("zotero_group_id") or os.environ.get("ZOTERO_GROUP_ID", "")

    if not args.dry_run and (not api_key or api_key == "YOUR_KEY_HERE"):
        sys.exit(
            "ERROR: Zotero API key not set.\n"
            f"  Edit {SECRETS_FILE} and set: zotero_api_key=YOUR_KEY_HERE\n"
            "  Get key at: https://www.zotero.org/settings/security\n"
            "  (Personal Library → Read/Write)"
        )
    if not args.dry_run and not user_id and not group_id:
        sys.exit(
            "ERROR: Zotero user ID not set.\n"
            f"  Edit {SECRETS_FILE} and set: zotero_user_id=6037399"
        )

    # ── Load articles ─────────────────────────────────────────────────────────
    if args.screened and SCREENED_CSV.exists():
        src = SCREENED_CSV
        with open(src, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f)
                    if r.get("human_decision", "").strip().lower() == "include"]
        print(f"Source: {src}  ->  {len(rows):,} human-approved articles")
    else:
        src = TAGGED_CSV
        if not src.exists():
            sys.exit(f"ERROR: {src} not found. Run organize_by_ontology.R first.")
        with open(src, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"Source: {src}  ->  {len(rows):,} articles (all tagged)")
        if not args.screened:
            print("TIP: Use --screened to import only human-approved articles.")

    if not rows:
        print("No articles to import.")
        return

    base_url = build_library_url(user_id, group_id)
    headers  = get_zotero_headers(api_key)

    print(f"Zotero library: {base_url}")
    print(f"Dry run:        {args.dry_run}")
    print(f"Batch size:     {BATCH_SIZE}")
    print()

    # ── Batch import ─────────────────────────────────────────────────────────
    log_rows = []
    n_created = n_unchanged = n_failed = 0
    batches = [rows[i:i+BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]

    for b_idx, batch in enumerate(batches):
        items = [row_to_zotero_item(r) for r in batch]
        if args.collection:
            for item in items:
                item["collections"] = [args.collection]

        results = post_batch(items, base_url, headers, args.dry_run)
        for row, res in zip(batch, results):
            status = res.get("status", "unknown")
            if status == "created":
                n_created += 1
            elif status == "unchanged":
                n_unchanged += 1
            else:
                n_failed += 1
            log_rows.append({
                "pmc_id":   row.get("pmc_id", ""),
                "title":    row.get("title", "")[:80],
                "zotero_key": res.get("key", ""),
                "status":   status,
                "error":    res.get("error", ""),
            })

        if b_idx % 5 == 0:
            print(f"  Batch {b_idx+1}/{len(batches)}  "
                  f"created={n_created} unchanged={n_unchanged} failed={n_failed}")
        time.sleep(SLEEP_S)

    # ── Write log ─────────────────────────────────────────────────────────────
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pmc_id", "title", "zotero_key", "status", "error"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\n── Zotero import complete ───────────────────────────────")
    print(f"  Created:    {n_created:,}")
    print(f"  Unchanged:  {n_unchanged:,}  (already in library)")
    print(f"  Failed:     {n_failed:,}")
    print(f"  Log:        {LOG_CSV}")

    if not args.dry_run and n_created > 0:
        print(f"\nNEXT STEPS (manual in Zotero desktop):")
        print(f"  1. Open Zotero → select all imported items")
        print(f"  2. Right-click → 'Retrieve Metadata' (fetches DOI/journal/volume)")
        print(f"  3. File → Export Library → Better BibTeX → refs/bmic-jpm.bib")


if __name__ == "__main__":
    main()
