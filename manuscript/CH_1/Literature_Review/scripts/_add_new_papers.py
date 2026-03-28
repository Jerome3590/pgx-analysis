"""
Add newly added Zotero papers to screened_doi_map.csv and import their PDFs.
Targets items added in last 48h that are NOT already in the doi_map.
"""
import csv, hashlib, shutil, sqlite3
from datetime import datetime
from pathlib import Path

DB      = Path(r"C:\Users\jerom\Zotero\zotero.sqlite")
STORAGE = Path(r"C:\Users\jerom\Zotero\storage")
DOI_MAP = Path("scripts/screened_doi_map.csv")
LOG_CSV = Path("scripts/vcu_download_log.csv")
PDF_DIR = Path("data/scholar_pdfs")

LOG_FIELDS = ["hsh_id", "title", "doi", "proxy_url", "status", "bytes", "timestamp"]

# ── Load existing doi_map ──────────────────────────────────────────────────────
existing = {r["doi"].lower().strip(): r
            for r in csv.DictReader(open(DOI_MAP, encoding="utf-8-sig"))
            if r.get("doi")}
existing_ids = {r["screened_pmc_id"]
                for r in csv.DictReader(open(DOI_MAP, encoding="utf-8-sig"))}

def make_hsh(doi: str) -> str:
    """Generate HSH-prefixed ID from DOI."""
    return "HSH" + hashlib.sha256(doi.lower().strip().encode()).hexdigest()[:8]

# ── Query Zotero for new items ─────────────────────────────────────────────────
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
cur = con.cursor()

cur.execute("""
    SELECT i.itemID,
           MAX(CASE WHEN f.fieldName='title' THEN iv.value END) AS title,
           MAX(CASE WHEN f.fieldName='DOI'   THEN iv.value END) AS doi,
           datetime(i.dateAdded, 'localtime') AS added
    FROM items i
    JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
    LEFT JOIN itemData  id ON i.itemID  = id.itemID
    LEFT JOIN itemDataValues iv ON id.valueID = iv.valueID
    LEFT JOIN fields f ON id.fieldID = f.fieldID
    WHERE it.typeName NOT IN ('attachment','note')
      AND datetime(i.dateAdded) >= datetime('now', '-48 hours')
    GROUP BY i.itemID
    ORDER BY i.dateAdded DESC
""")
items = cur.fetchall()

def find_pdf(item_id: int) -> Path | None:
    """Find any PDF attached to a Zotero item."""
    cur.execute("""
        SELECT i2.key
        FROM itemAttachments ia
        JOIN items i2 ON ia.itemID = i2.itemID
        WHERE ia.parentItemID = ? AND ia.contentType = 'application/pdf'
    """, (item_id,))
    for (key,) in cur.fetchall():
        folder = STORAGE / key
        if folder.exists():
            pdfs = list(folder.glob("*.pdf"))
            if pdfs:
                return pdfs[0]
    return None

# ── Identify new items with DOIs and PDFs ─────────────────────────────────────
to_add = []
seen_dois = set()

for item_id, title, doi, added in items:
    if not doi or not title:
        continue
    doi_norm = doi.lower().strip()
    if doi_norm in existing or doi_norm in seen_dois:
        continue
    seen_dois.add(doi_norm)

    hsh_id = make_hsh(doi_norm)
    # Avoid collision with existing hsh_ids
    if hsh_id in existing_ids:
        hsh_id = "HSH" + hashlib.sha256((doi_norm + "_alt").encode()).hexdigest()[:8]

    pdf_path = find_pdf(item_id)
    to_add.append({
        "item_id":  item_id,
        "hsh_id":   hsh_id,
        "title":    title,
        "doi":      doi,
        "pdf_path": pdf_path,
        "added":    added,
    })

con.close()

print(f"New items to add: {len(to_add)}")
for r in to_add:
    pdf_status = r["pdf_path"].name[:50] if r["pdf_path"] else "(no PDF found)"
    print(f"  {r['hsh_id']}  {r['doi']}")
    print(f"    {r['title'][:80]}")
    print(f"    PDF: {pdf_status}")
print()

if not to_add:
    print("Nothing to add.")
    raise SystemExit(0)

# ── Append to screened_doi_map.csv ────────────────────────────────────────────
with open(DOI_MAP, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["screened_pmc_id", "doi", "title"])
    for r in to_add:
        w.writerow({
            "screened_pmc_id": r["hsh_id"],
            "doi":             r["doi"],
            "title":           r["title"][:120],
        })
print(f"Appended {len(to_add)} rows to screened_doi_map.csv")

# ── Import PDFs ───────────────────────────────────────────────────────────────
imported = 0
no_pdf   = 0
for r in to_add:
    dest = PDF_DIR / f"{r['hsh_id']}.pdf"
    if dest.exists():
        print(f"  ↩ SKIP (exists): {dest.name}")
        continue
    if not r["pdf_path"]:
        print(f"  ✗ NO PDF: {r['hsh_id']}  {r['doi']}")
        no_pdf += 1
        continue
    shutil.copy2(r["pdf_path"], dest)
    # Append to log
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            r["hsh_id"], r["title"][:70], r["doi"],
            "zotero", "zotero",
            dest.stat().st_size, datetime.utcnow().isoformat(),
        ])
    print(f"  ✓ IMPORTED: {r['hsh_id']}  ({dest.stat().st_size // 1024} KB)  {r['title'][:60]}")
    imported += 1

print()
on_disk = len(list(PDF_DIR.glob("*.pdf")))
print(f"── Done ─────────────────────────────────")
print(f"  Added to doi_map : {len(to_add)}")
print(f"  PDFs imported    : {imported}")
print(f"  No PDF in Zotero : {no_pdf}")
print(f"  Total on disk    : {on_disk}")
