"""Check for newly added Zotero items not in screened_doi_map.csv."""
import csv, sqlite3
from pathlib import Path

DB      = Path(r"C:\Users\jerom\Zotero\zotero.sqlite")
DOI_MAP = Path("scripts/screened_doi_map.csv")
PDF_DIR = Path("data/scholar_pdfs")
STORAGE = Path(r"C:\Users\jerom\Zotero\storage")

known_dois = {r["doi"].lower().strip()
              for r in csv.DictReader(open(DOI_MAP, encoding="utf-8-sig"))
              if r.get("doi")}
on_disk    = {p.stem for p in PDF_DIR.glob("*.pdf")}

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
rows = cur.fetchall()

# Also get PDF attachments for each new item
def get_pdf(item_id):
    cur.execute("""
        SELECT i2.key
        FROM itemAttachments ia
        JOIN items i2 ON ia.itemID = i2.itemID
        WHERE ia.parentItemID = ? AND ia.contentType = 'application/pdf'
        LIMIT 1
    """, (item_id,))
    r = cur.fetchone()
    if r:
        key = r[0]
        for ext in (".pdf", f"/{key}.pdf"):
            p = STORAGE / key / f"{key}.pdf"
            if p.exists():
                return p
    return None

print(f"Zotero items added last 48h: {len(rows)}\n")
new_items = []
in_map_count = 0
for item_id, title, doi, added in rows:
    doi_norm = (doi or "").lower().strip()
    in_map   = doi_norm in known_dois if doi_norm else False
    if in_map:
        in_map_count += 1
        continue
    pdf_path = get_pdf(item_id)
    print(f"  [NEW]")
    print(f"    Added : {added}")
    print(f"    Title : {(title or '')[:90]}")
    print(f"    DOI   : {doi or '(none)'}")
    print(f"    PDF   : {pdf_path.name if pdf_path else '(none in storage)'}")
    print()
    new_items.append({"item_id": item_id, "title": title, "doi": doi, "pdf": pdf_path})

con.close()

print(f"─────────────────────────────────────────")
print(f"Total last 48h : {len(rows)}")
print(f"Already in map : {in_map_count}")
print(f"NEW (not in map): {len(new_items)}  ← need to decide: add to doi_map or ignore")
print(f"  with DOI     : {sum(1 for x in new_items if x['doi'])}")
print(f"  with PDF     : {sum(1 for x in new_items if x['pdf'])}")
