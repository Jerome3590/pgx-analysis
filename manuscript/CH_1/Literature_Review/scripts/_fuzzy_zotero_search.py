"""
_fuzzy_zotero_search.py
For each remaining missing article, find the best-matching Zotero item by title
and import if a PDF exists in storage.

Run from: manuscript/CH_1/Literature_Review/
  python scripts/_fuzzy_zotero_search.py [--dry-run] [--threshold N]
"""
import argparse
import csv
import re
import shutil
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

ZOTERO_DB      = Path(r"C:\Users\jerom\Zotero\zotero.sqlite")
ZOTERO_STORAGE = Path(r"C:\Users\jerom\Zotero\storage")
DOI_MAP        = Path("scripts/screened_doi_map.csv")
LOG_CSV        = Path("scripts/vcu_download_log.csv")
PDF_DIR        = Path("data/scholar_pdfs")

LOG_FIELDS = ["hsh_id", "title", "doi", "proxy_url", "status", "bytes", "timestamp"]

SQL_ALL_ITEMS = """
    SELECT
        p.itemID,
        title_val.value   AS title,
        att.key           AS storage_key
    FROM items p
    JOIN itemData td
        ON td.itemID = p.itemID
        AND td.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'title')
    JOIN itemDataValues title_val
        ON title_val.valueID = td.valueID
    LEFT JOIN itemAttachments ia
        ON ia.parentItemID = p.itemID
        AND ia.contentType = 'application/pdf'
    LEFT JOIN items att
        ON att.itemID = ia.itemID
    WHERE p.itemID NOT IN (SELECT itemID FROM deletedItems)
"""

STOP = {"a","an","the","of","in","on","and","for","to","with","from","at","by",
        "is","are","its","as","or","that","this","via","using","based","among",
        "between","after","before","during","within","through"}

def tok(s: str) -> set:
    return set(re.sub(r"[^a-z0-9]", " ", s.lower()).split()) - STOP

def append_log(row: dict):
    exists = LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--threshold", type=int, default=4)
    args = parser.parse_args()

    # Load missing articles
    doi_rows  = {r["screened_pmc_id"]: r
                 for r in csv.DictReader(open(DOI_MAP, encoding="utf-8-sig"))}
    latest = {}
    for r in csv.DictReader(open(LOG_CSV, encoding="utf-8-sig")):
        hid = r["hsh_id"]
        if hid not in latest or r["timestamp"] > latest[hid]["timestamp"]:
            latest[hid] = r
    have = ({p.stem for p in PDF_DIR.glob("*.pdf")} |
            {hid for hid, r in latest.items()
             if r["status"] in ("ok","manual","zotero","oa_url","reconcile")})
    missing = {hid: r["title"]
               for hid, r in doi_rows.items()
               if hid not in have and r.get("doi")}
    print(f"Missing: {len(missing)} articles  |  threshold={args.threshold}\n")

    # Load all Zotero items with titles
    tmp = Path(tempfile.mktemp(suffix=".sqlite"))
    shutil.copy2(ZOTERO_DB, tmp)
    con = sqlite3.connect(str(tmp))
    con.row_factory = sqlite3.Row
    z_items = [(r["title"] or "", r["storage_key"] or "")
               for r in con.execute(SQL_ALL_ITEMS).fetchall()]
    con.close()
    tmp.unlink(missing_ok=True)
    print(f"Zotero items with titles: {len(z_items)}\n")

    imported = 0
    for hid, mtitle in sorted(missing.items(), key=lambda x: x[1]):
        mt = tok(mtitle)
        best_score, best_ztitle, best_key = 0, "", ""
        for ztitle, zkey in z_items:
            score = len(mt & tok(ztitle))
            if score > best_score:
                best_score, best_ztitle, best_key = score, ztitle, zkey

        # Find PDF file
        pdf_src = None
        if best_key:
            d = ZOTERO_STORAGE / best_key
            if d.exists():
                pdfs = list(d.glob("*.pdf"))
                if pdfs:
                    pdf_src = pdfs[0]

        marker = "PDF✓" if pdf_src else ("KEY " if best_key else "--- ")
        print(f"[{best_score:2d} {marker}] {mtitle[:70]}")
        if best_score >= 2:
            print(f"          → {best_ztitle[:70]}")
        if best_score < args.threshold or not pdf_src:
            continue

        dest = PDF_DIR / f"{hid}.pdf"
        print(f"  IMPORTING → {dest.name}  ({pdf_src.stat().st_size//1024} KB)")
        if not args.dry_run:
            shutil.copy2(pdf_src, dest)
            append_log({
                "hsh_id": hid, "title": mtitle[:70],
                "doi": doi_rows[hid].get("doi",""),
                "proxy_url": f"zotero_title:{best_key}",
                "status": "reconcile",
                "bytes": dest.stat().st_size,
                "timestamp": datetime.utcnow().isoformat(),
            })
        imported += 1

    print(f"\nImported: {imported}")
    print(f"On disk : {len(list(PDF_DIR.glob('*.pdf')))}/117")

if __name__ == "__main__":
    main()
