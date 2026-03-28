"""
_zotero_add_blocked.py
Adds the blocked (not-yet-downloaded) articles to a Zotero collection via the
Web API so you can then bulk "Find Available PDF" in Zotero desktop.

Credentials are read from secrets/secrets.txt:
  zotero_api_key=YOUR_KEY
  zotero_user_id=6037399

Workflow:
  1. Run this script  → creates/reuses collection "PGx - Needs PDF"
  2. Open Zotero desktop, find the collection
  3. Select All (Ctrl+A) → right-click → "Find Available PDF"
     (Zotero will use VCU EZProxy if configured under Settings → Advanced → Proxies)

Run from: manuscript/CH_1/Literature_Review/
  python scripts/_zotero_add_blocked.py [--dry-run]
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
LOG_CSV      = Path("scripts/vcu_download_log.csv")
DOI_MAP      = Path("scripts/screened_doi_map.csv")
OA_CSV       = Path("scripts/oa_scan_results.csv")
PDF_DIR      = Path("data/scholar_pdfs")

COLLECTION_NAME = "PGx - Needs PDF"
BATCH_SIZE      = 50
SLEEP_S         = 0.3

_SKIP_KW = ["guar gum", "pomegranate", "intragastric balloon",
            "ursodeoxycholic", "gallstone", "prader-willi",
            "ldl-c", "trs2p", "bcl-xl", "protac", "berberine nanopart",
            "toosendanin", "diltiazem liver", "intranasal oxytocin",
            "post-roe", "mammography", "lipid profile"]

# ── Secrets ───────────────────────────────────────────────────────────────────

def load_secrets() -> dict:
    cfg = {}
    if not SECRETS_FILE.exists():
        return cfg
    for line in SECRETS_FILE.read_text(encoding="utf-8").splitlines():
        m = re.match(r'^\s*([^#=\s][^=]*)=(.*)$', line)
        if m:
            cfg[m.group(1).strip()] = m.group(2).strip()
    return cfg

# ── Zotero helpers ────────────────────────────────────────────────────────────

def headers(api_key: str) -> dict:
    return {"Zotero-API-Key": api_key,
            "Zotero-API-Version": "3",
            "Content-Type": "application/json"}

def base_url(user_id: str) -> str:
    return f"https://api.zotero.org/users/{user_id}"

def get_or_create_collection(base: str, hdrs: dict, name: str, dry_run: bool) -> str | None:
    """Return key of existing or newly created collection."""
    if dry_run:
        return "DRY_COLLECTION"
    r = requests.get(f"{base}/collections", headers=hdrs, timeout=15)
    for col in r.json():
        if col["data"]["name"] == name:
            print(f"  Using existing collection '{name}' ({col['key']})")
            return col["key"]
    r = requests.post(f"{base}/collections", headers=hdrs,
                      data=json.dumps([{"name": name, "parentCollection": False}]),
                      timeout=15)
    key = list(r.json().get("success", {}).values())
    if key:
        print(f"  Created collection '{name}' ({key[0]})")
        return key[0]
    print(f"  WARNING: could not create collection: {r.text[:200]}")
    return None

def post_batch(base: str, hdrs: dict, items: list, dry_run: bool) -> tuple[int, int, int]:
    if dry_run:
        return len(items), 0, 0
    r = requests.post(f"{base}/items", headers=hdrs,
                      data=json.dumps(items), timeout=30)
    if r.status_code not in (200, 201):
        print(f"  HTTP {r.status_code}: {r.text[:200]}")
        return 0, 0, len(items)
    data = r.json()
    return (len(data.get("success", {})),
            len(data.get("unchanged", {})),
            len(data.get("failed", {})))

# ── Build item from row ───────────────────────────────────────────────────────

def make_item(doi: str, title: str, collection_key: str) -> dict:
    item = {
        "itemType": "journalArticle",
        "title": title,
        "DOI": doi,
        "extra": f"DOI: {doi}",
        "tags": [{"tag": "pgx-needs-pdf"}, {"tag": "pgx-lit-review"}],
        "collections": [collection_key] if collection_key else [],
        "relations": {},
    }
    return item

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    secrets = load_secrets()
    api_key = secrets.get("zotero_api_key") or os.environ.get("ZOTERO_API_KEY", "")
    user_id = secrets.get("zotero_user_id") or os.environ.get("ZOTERO_USER_ID", "")

    if not args.dry_run and (not api_key or api_key == "YOUR_KEY_HERE"):
        sys.exit(f"ERROR: Set zotero_api_key in {SECRETS_FILE}")
    if not user_id:
        sys.exit(f"ERROR: Set zotero_user_id in {SECRETS_FILE}")

    hdrs = headers(api_key)
    burl = base_url(user_id)

    # ── Identify blocked articles ────────────────────────────────────────────
    rows = list(csv.DictReader(open(LOG_CSV, encoding="utf-8-sig")))
    latest = {}
    for r in rows:
        hid = r["hsh_id"]
        if hid not in latest or r["timestamp"] > latest[hid]["timestamp"]:
            latest[hid] = r

    doi_map = {r["screened_pmc_id"]: {"doi": r["doi"], "title": r["title"]}
               for r in csv.DictReader(open(DOI_MAP, encoding="utf-8-sig"))}

    on_disk = {p.stem for p in PDF_DIR.glob("*.pdf")}
    ok_ids  = {hid for hid, r in latest.items() if r["status"] in ("ok", "manual")}
    already = ok_ids | on_disk

    blocked = []
    for hid, info in doi_map.items():
        if hid in already:
            continue
        title = info["title"] or ""
        doi   = info["doi"] or ""
        if not doi:
            continue
        if any(k in title.lower() for k in _SKIP_KW):
            continue
        blocked.append((doi, title))

    print(f"Blocked articles to add: {len(blocked)}")
    if not blocked:
        print("Nothing to do.")
        return

    # ── Get/create collection ────────────────────────────────────────────────
    col_key = get_or_create_collection(burl, hdrs, COLLECTION_NAME, args.dry_run)

    # ── Batch upload ─────────────────────────────────────────────────────────
    items = [make_item(doi, title, col_key) for doi, title in blocked]
    batches = [items[i:i+BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]

    total_created = total_unchanged = total_failed = 0
    for i, batch in enumerate(batches):
        c, u, f = post_batch(burl, hdrs, batch, args.dry_run)
        total_created   += c
        total_unchanged += u
        total_failed    += f
        print(f"  Batch {i+1}/{len(batches)}: +{c} created, {u} unchanged, {f} failed")
        time.sleep(SLEEP_S)

    print(f"\n── Done ──────────────────────────────────────────────")
    print(f"  Created   : {total_created}")
    print(f"  Unchanged : {total_unchanged}  (already in library)")
    print(f"  Failed    : {total_failed}")
    if not args.dry_run and total_created > 0:
        print(f"\nNEXT STEPS in Zotero desktop:")
        print(f"  1. Open collection '{COLLECTION_NAME}'")
        print(f"  2. Edit → Select All (Ctrl+A)")
        print(f"  3. Right-click → Find Available PDF")
        print(f"     (routes through VCU EZProxy if configured under")
        print(f"      Settings → Advanced → Proxies)")
        print(f"  4. After Zotero fetches PDFs, run:")
        print(f"       python scripts/_import_zotero_pdfs.py")
        print(f"     to copy them into data/scholar_pdfs/ and update the log.")

if __name__ == "__main__":
    main()
