"""
_import_zotero_pdfs.py
Harvest PDFs that Zotero downloaded via "Find Available PDF" from the
"PGx - Needs PDF" collection into data/scholar_pdfs/{hsh_id}.pdf,
then update vcu_download_log.csv with status='zotero'.

Reads Zotero SQLite directly — Zotero MUST be closed when running this.

Run from: manuscript/CH_1/Literature_Review/
  python scripts/_import_zotero_pdfs.py [--dry-run]
"""

import argparse
import csv
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ZOTERO_DB  = Path(r"C:\Users\jerom\Zotero\zotero.sqlite")
ZOTERO_STORAGE = Path(r"C:\Users\jerom\Zotero\storage")
LOG_CSV    = Path("scripts/vcu_download_log.csv")
DOI_MAP    = Path("scripts/screened_doi_map.csv")
PDF_DIR    = Path("data/scholar_pdfs")

LOG_FIELDS = ["hsh_id", "title", "doi", "proxy_url", "status", "bytes", "timestamp"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_doi_map() -> dict:
    """hsh_id → {doi, title}"""
    result = {}
    for r in csv.DictReader(open(DOI_MAP, encoding="utf-8-sig")):
        result[r["screened_pmc_id"]] = {"doi": r["doi"].lower().strip(),
                                         "title": r["title"]}
    return result

def load_latest_log() -> dict:
    if not LOG_CSV.exists():
        return {}
    rows = list(csv.DictReader(open(LOG_CSV, encoding="utf-8-sig")))
    latest = {}
    for r in rows:
        hid = r["hsh_id"]
        if hid not in latest or r["timestamp"] > latest[hid]["timestamp"]:
            latest[hid] = r
    return latest

def append_log(row: dict):
    exists = LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)

# ── Query Zotero SQLite ───────────────────────────────────────────────────────

TARGET_COLLECTIONS = [
    "PGx - Needs PDF",
    "PGx Adoption",
    "Pharmacogenomic Testing",
    "PgX Implementation SD Model",
    "Literature Review",
]

def get_zotero_pdfs() -> list[dict]:
    """
    Find Zotero items in TARGET_COLLECTIONS that have PDF attachments.
    Returns list of {doi, title, storage_key}.
    Copies the SQLite first to avoid locking issues.
    """
    import tempfile
    tmp = Path(tempfile.mktemp(suffix=".sqlite"))
    shutil.copy2(ZOTERO_DB, tmp)

    con = sqlite3.connect(str(tmp))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    placeholders = ",".join("?" * len(TARGET_COLLECTIONS))
    cur.execute(f"""
        SELECT
            i.itemID,
            doi_val.value   AS doi,
            title_val.value AS title,
            att.key         AS storage_key
        FROM collectionItems ci
        JOIN collections c ON c.collectionID = ci.collectionID
        JOIN items i ON ci.itemID = i.itemID
        -- DOI field
        LEFT JOIN itemData doi_d
            ON doi_d.itemID = i.itemID
            AND doi_d.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'DOI')
        LEFT JOIN itemDataValues doi_val ON doi_val.valueID = doi_d.valueID
        -- title field
        LEFT JOIN itemData title_d
            ON title_d.itemID = i.itemID
            AND title_d.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'title')
        LEFT JOIN itemDataValues title_val ON title_val.valueID = title_d.valueID
        -- PDF attachment
        JOIN itemAttachments ia ON ia.parentItemID = i.itemID
        JOIN items att ON att.itemID = ia.itemID
        WHERE c.collectionName IN ({placeholders})
          AND ia.contentType = 'application/pdf'
          AND i.itemTypeID != (SELECT itemTypeID FROM itemTypes WHERE typeName = 'attachment')
    """, TARGET_COLLECTIONS)

    results = []
    seen = set()
    for r in cur.fetchall():
        key = r["storage_key"] or ""
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "doi":         (r["doi"] or "").lower().strip(),
            "title":       r["title"] or "",
            "storage_key": key,
        })

    con.close()
    tmp.unlink(missing_ok=True)
    return results

def find_pdf_file(storage_key: str) -> Path | None:
    """Find the PDF file under Zotero storage/{key}/"""
    if not storage_key:
        return None
    storage_dir = ZOTERO_STORAGE / storage_key
    if not storage_dir.exists():
        return None
    pdfs = list(storage_dir.glob("*.pdf"))
    return pdfs[0] if pdfs else None

# ── Main ──────────────────────────────────────────────────────────────────────

def load_title_index() -> list[tuple[str, str]]:
    """Return list of (normalised_title, article_id) from articles_screened.csv."""
    import re as _re
    def _norm(s):
        return _re.sub(r"\s+", " ", _re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()
    idx = []
    screened = Path("data/ontology/articles_screened.csv")
    if screened.exists():
        for r in csv.DictReader(open(screened, encoding="utf-8-sig")):
            aid   = (r.get("article_id") or "").strip()
            title = (r.get("title") or "").strip()
            if aid and title:
                idx.append((_norm(title), aid))
    return idx

def match_by_title(zotero_title: str, title_index: list, threshold: float = 0.82) -> str | None:
    import re as _re
    def _norm(s):
        return _re.sub(r"\s+", " ", _re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()
    def _toks(s):
        return set(s.split())
    zt = _toks(_norm(zotero_title))
    best_aid, best_score = None, 0.0
    for norm_t, aid in title_index:
        tt = _toks(norm_t)
        if not zt or not tt:
            continue
        score = len(zt & tt) / max(len(zt), len(tt))
        if score > best_score:
            best_score, best_aid = score, aid
    return best_aid if best_score >= threshold else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Print every skip/unmatched")
    args = parser.parse_args()

    # Ensure Zotero is closed
    result = subprocess.run(["tasklist", "/FI", "IMAGENAME eq zotero.exe"],
                            capture_output=True, text=True)
    if "zotero.exe" in result.stdout.lower():
        sys.exit("ERROR: Zotero is running. Close Zotero before running this script.")

    doi_map      = load_doi_map()
    on_disk      = {p.stem for p in PDF_DIR.glob("*.pdf")}
    doi_to_hid   = {v["doi"]: k for k, v in doi_map.items() if v["doi"]}
    title_index  = load_title_index()

    print(f"Querying Zotero collections: {TARGET_COLLECTIONS}")
    zotero_items = get_zotero_pdfs()
    print(f"  Found {len(zotero_items)} unique items with PDF attachments")
    print(f"  DOI index   : {len(doi_to_hid)} entries")
    print(f"  Title index : {len(title_index)} entries")
    print(f"  Already on disk: {len(on_disk)} PDFs\n")

    imported = skipped = missing = unmatched = title_matched = 0
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    for item in zotero_items:
        doi   = item["doi"]
        title = item["title"]
        label = title[:65] or doi[:65]

        # 1. DOI match
        hid = doi_to_hid.get(doi)
        if not hid and doi.startswith("doi:"):
            hid = doi_to_hid.get(doi[4:].strip())

        # 2. Title fallback
        if not hid and title:
            hid = match_by_title(title, title_index)
            if hid:
                title_matched += 1

        if not hid:
            if args.verbose:
                print(f"  ✗ UNMATCHED: {label}")
            unmatched += 1
            continue

        dest = PDF_DIR / f"{hid}.pdf"
        if dest.exists():
            if args.verbose:
                print(f"  ↩ SKIP: {label}")
            skipped += 1
            continue

        pdf_src = find_pdf_file(item["storage_key"])
        if not pdf_src:
            if args.verbose:
                print(f"  ✗ NO FILE [{item['storage_key']}]: {label}")
            missing += 1
            continue

        nbytes = pdf_src.stat().st_size
        if not args.dry_run:
            shutil.copy2(pdf_src, dest)
            append_log({
                "hsh_id":    hid,
                "title":     title[:70],
                "doi":       doi,
                "proxy_url": "zotero",
                "status":    "zotero",
                "bytes":     nbytes,
                "timestamp": datetime.utcnow().isoformat(),
            })

        print(f"  ✓ {'[DRY] ' if args.dry_run else ''}IMPORTED ({nbytes//1024} KB): {label}")
        imported += 1

    print(f"\n── Import complete {'(dry run) ' if args.dry_run else ''}─────────────────────")
    print(f"  Imported        : {imported}")
    print(f"    via DOI       : {imported - title_matched}")
    print(f"    via title     : {title_matched}")
    print(f"  Skipped (exist) : {skipped}")
    print(f"  No PDF file     : {missing}")
    print(f"  Unmatched       : {unmatched}")
    print(f"  Total on disk   : {len(list(PDF_DIR.glob('*.pdf')))}"
          f"{' (unchanged, dry-run)' if args.dry_run else ''}")

if __name__ == "__main__":
    main()
