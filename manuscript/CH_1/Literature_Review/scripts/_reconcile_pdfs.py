"""
_reconcile_pdfs.py
Scans manual_review/ AND all of Zotero storage for PDFs that match
missing articles (by DOI or fuzzy title), copies them to data/scholar_pdfs/,
and updates vcu_download_log.csv.

Safe to run while Zotero is open (copies SQLite to temp file).

Run from: manuscript/CH_1/Literature_Review/
  python scripts/_reconcile_pdfs.py [--dry-run]
"""
import argparse
import csv
import re
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ZOTERO_DB      = Path(r"C:\Users\jerom\Zotero\zotero.sqlite")
ZOTERO_STORAGE = Path(r"C:\Users\jerom\Zotero\storage")
MANUAL_DIR     = Path("infrastructure_setup/manual_review")
PDF_DIR        = Path("data/scholar_pdfs")
LOG_CSV        = Path("scripts/vcu_download_log.csv")
DOI_MAP        = Path("scripts/screened_doi_map.csv")

LOG_FIELDS = ["hsh_id", "title", "doi", "proxy_url", "status", "bytes", "timestamp"]

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_doi_map():
    rows = list(csv.DictReader(open(DOI_MAP, encoding="utf-8-sig")))
    hid_by_doi   = {r["doi"].lower().strip(): r["screened_pmc_id"]
                    for r in rows if r["doi"].strip()}
    title_by_hid = {r["screened_pmc_id"]: r["title"] for r in rows}
    doi_by_hid   = {r["screened_pmc_id"]: r["doi"].lower().strip() for r in rows}
    return hid_by_doi, title_by_hid, doi_by_hid

def already_have():
    on_disk = {p.stem for p in PDF_DIR.glob("*.pdf")}
    if not LOG_CSV.exists():
        return on_disk
    latest = {}
    for r in csv.DictReader(open(LOG_CSV, encoding="utf-8-sig")):
        hid = r["hsh_id"]
        if hid not in latest or r["timestamp"] > latest[hid]["timestamp"]:
            latest[hid] = r
    ok = {hid for hid, r in latest.items()
          if r["status"] in ("ok", "manual", "zotero", "oa_url", "reconcile")}
    return on_disk | ok

def tokenize(s: str) -> set:
    return set(re.sub(r"[^a-z0-9]", " ", s.lower()).split()) - {
        "a", "an", "the", "of", "in", "on", "and", "for", "to",
        "with", "from", "at", "by", "is", "are", "its", "as"}

def title_match(pdf_name: str, hid_titles: dict, threshold=4) -> tuple[str | None, int]:
    pdf_toks = tokenize(pdf_name)
    best_hid, best_score = None, 0
    for hid, title in hid_titles.items():
        score = len(pdf_toks & tokenize(title))
        if score > best_score:
            best_score, best_hid = score, hid
    return (best_hid if best_score >= threshold else None), best_score

def append_log(row: dict):
    exists = LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)

# ── Zotero SQLite query ────────────────────────────────────────────────────────

def get_zotero_all_pdfs() -> list[dict]:
    """
    Return ALL Zotero PDF attachments with their parent item's DOI and title.
    Does NOT require a DOI — uses title for fuzzy matching.
    Copies the SQLite first to avoid locking issues.
    """
    tmp = Path(tempfile.mktemp(suffix=".sqlite"))
    shutil.copy2(ZOTERO_DB, tmp)
    con = sqlite3.connect(str(tmp))
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("""
        SELECT
            att.key          AS storage_key,
            att_title.value  AS att_filename,
            doi_val.value    AS doi,
            title_val.value  AS title
        FROM items att
        JOIN itemAttachments ia ON ia.itemID = att.itemID
        -- parent DOI
        LEFT JOIN itemData doi_d
            ON doi_d.itemID = ia.parentItemID
            AND doi_d.fieldID = (SELECT fieldID FROM fields WHERE fieldName='DOI')
        LEFT JOIN itemDataValues doi_val ON doi_val.valueID = doi_d.valueID
        -- parent title
        LEFT JOIN itemData title_d
            ON title_d.itemID = ia.parentItemID
            AND title_d.fieldID = (SELECT fieldID FROM fields WHERE fieldName='title')
        LEFT JOIN itemDataValues title_val ON title_val.valueID = title_d.valueID
        -- attachment stored filename (title field of attachment item)
        LEFT JOIN itemData att_title_d
            ON att_title_d.itemID = att.itemID
            AND att_title_d.fieldID = (SELECT fieldID FROM fields WHERE fieldName='title')
        LEFT JOIN itemDataValues att_title ON att_title.valueID = att_title_d.valueID
        WHERE ia.contentType = 'application/pdf'
          AND att.itemID NOT IN (SELECT itemID FROM deletedItems)
    """)
    results = []
    for r in cur.fetchall():
        results.append({
            "storage_key":  r["storage_key"] or "",
            "att_filename": r["att_filename"] or "",
            "doi":          (r["doi"] or "").lower().strip(),
            "title":        r["title"] or "",
        })
    con.close()
    tmp.unlink(missing_ok=True)
    return results

def find_pdf_in_storage(storage_key: str) -> Path | None:
    d = ZOTERO_STORAGE / storage_key
    if not d.exists():
        return None
    pdfs = list(d.glob("*.pdf"))
    return pdfs[0] if pdfs else None

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    hid_by_doi, title_by_hid, doi_by_hid = load_doi_map()
    have = already_have()

    # hids we still need
    need_hids   = {hid for hid in title_by_hid if hid not in have}
    need_titles = {hid: title_by_hid[hid] for hid in need_hids}
    need_dois   = {doi_by_hid[hid]: hid for hid in need_hids if doi_by_hid.get(hid)}
    print(f"Still missing: {len(need_hids)} articles\n")

    imported = 0

    # ── 1) manual_review/ ─────────────────────────────────────────────────────
    manual_pdfs = list(MANUAL_DIR.glob("*.pdf"))
    print(f"=== manual_review/ : {len(manual_pdfs)} PDFs ===")
    for pdf in sorted(manual_pdfs):
        stem = pdf.stem
        # Exact hsh_id filename
        if stem in need_hids:
            hid = stem
        else:
            # DOI slug in filename
            hid = None
            for doi, h in need_dois.items():
                doi_slug = doi.replace("/", "_").replace(".", "-")
                if doi_slug in stem or stem in doi_slug:
                    hid = h
                    break
            # Fuzzy title
            if not hid:
                hid, score = title_match(stem, need_titles)

        if not hid:
            print(f"  ✗ UNMATCHED : {pdf.name}")
            continue

        dest = PDF_DIR / f"{hid}.pdf"
        print(f"  ✓ MATCH     : {pdf.name}")
        print(f"    → {hid}  ({title_by_hid[hid][:60]})")
        if not args.dry_run:
            shutil.copy2(pdf, dest)
            append_log({"hsh_id": hid, "title": title_by_hid[hid][:70],
                        "doi": doi_by_hid.get(hid, ""), "proxy_url": "manual",
                        "status": "manual", "bytes": dest.stat().st_size,
                        "timestamp": datetime.utcnow().isoformat()})
            have.add(hid)
            need_hids.discard(hid)
            need_titles.pop(hid, None)
        imported += 1

    # ── 2) Zotero storage (all items with DOI) ────────────────────────────────
    print(f"\n=== Zotero storage : scanning {ZOTERO_STORAGE} ===")
    z_items = get_zotero_all_pdfs()
    print(f"  Found {len(z_items)} Zotero PDF items with parent DOI\n")

    for item in z_items:
        doi   = item["doi"]
        title = item["title"]
        key   = item["storage_key"]

        # Match by DOI first
        hid = need_dois.get(doi)
        # Fuzzy match on Zotero item TITLE (not filename)
        if not hid and title:
            hid, _ = title_match(title, need_titles, threshold=4)
        # Also try attachment filename as fallback
        if not hid and item["att_filename"]:
            hid, _ = title_match(item["att_filename"], need_titles, threshold=4)
        if not hid:
            continue  # not one of our missing articles

        pdf_src = find_pdf_in_storage(key)
        if not pdf_src:
            print(f"  ✗ NO FILE   [{key}]: {title[:60]}")
            continue

        dest = PDF_DIR / f"{hid}.pdf"
        print(f"  ✓ MATCH     : {title[:65]}")
        print(f"    Zotero key={key}  ({pdf_src.stat().st_size//1024} KB)  → {dest.name}")
        if not args.dry_run:
            shutil.copy2(pdf_src, dest)
            append_log({"hsh_id": hid, "title": title_by_hid[hid][:70],
                        "doi": doi_by_hid.get(hid, ""), "proxy_url": f"zotero:{key}",
                        "status": "reconcile", "bytes": dest.stat().st_size,
                        "timestamp": datetime.utcnow().isoformat()})
            have.add(hid)
            need_hids.discard(hid)
            need_titles.pop(hid, None)
        imported += 1

    # ── 3) Brute-force fuzzy scan all Zotero storage PDFs by filename ─────────
    print(f"\n=== Brute-force filename scan : {len(need_hids)} still missing ===")
    all_storage_pdfs = list(ZOTERO_STORAGE.rglob("*.pdf"))
    print(f"  Scanning {len(all_storage_pdfs)} storage PDFs by filename...\n")

    for pdf_path in sorted(all_storage_pdfs):
        if not need_titles:
            break
        # Use both the stored filename AND the parent folder name for context
        fname = pdf_path.stem
        hid, score = title_match(fname, need_titles, threshold=4)
        if not hid:
            # try just the last part after " - " (often "Author - Year - Title")
            parts = fname.split(" - ", 2)
            if len(parts) == 3:
                hid, score = title_match(parts[2], need_titles, threshold=4)
        if not hid:
            continue

        dest = PDF_DIR / f"{hid}.pdf"
        print(f"  ✓ FUZZY [{score} tok]: {fname[:65]}")
        print(f"    → {hid}  ({need_titles[hid][:60]})")
        if not args.dry_run:
            shutil.copy2(pdf_path, dest)
            append_log({"hsh_id": hid, "title": title_by_hid[hid][:70],
                        "doi": doi_by_hid.get(hid, ""), "proxy_url": f"zotero_fuzzy:{pdf_path.parent.name}",
                        "status": "reconcile", "bytes": dest.stat().st_size,
                        "timestamp": datetime.utcnow().isoformat()})
            have.add(hid)
            need_hids.discard(hid)
            need_titles.pop(hid, None)
        imported += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── Reconcile {'(dry run) ' if args.dry_run else ''}complete ─────────────────")
    print(f"  Imported    : {imported}")
    on_disk_count = len(list(PDF_DIR.glob("*.pdf"))) + (imported if args.dry_run else 0)
    print(f"  On disk now : {len(list(PDF_DIR.glob('*.pdf')))}/117")
    print(f"  Still need  : {len(need_hids)}")
    if need_hids:
        print(f"\n  Regenerate checklist: python scripts/_gen_checklist_by_publisher.py")

if __name__ == "__main__":
    main()
