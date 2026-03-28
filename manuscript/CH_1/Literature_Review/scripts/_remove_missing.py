"""Remove articles from screened_doi_map.csv that have no PDF on disk."""
import csv
from pathlib import Path

DOI_MAP = Path("scripts/screened_doi_map.csv")
PDF_DIR = Path("data/scholar_pdfs")
LOG_CSV = Path("scripts/vcu_download_log.csv")

on_disk = {p.stem for p in PDF_DIR.glob("*.pdf")}

latest = {}
for r in csv.DictReader(open(LOG_CSV, encoding="utf-8-sig")):
    hid = r["hsh_id"]
    if hid not in latest or r["timestamp"] > latest[hid]["timestamp"]:
        latest[hid] = r
have = on_disk | {hid for hid, r in latest.items()
                  if r["status"] in ("ok", "manual", "zotero", "oa_url", "reconcile")}

rows = list(csv.DictReader(open(DOI_MAP, encoding="utf-8-sig")))
keep    = [r for r in rows if r["screened_pmc_id"] in have]
removed = [r for r in rows if r["screened_pmc_id"] not in have]

print(f"Removing {len(removed)} entries with no PDF:")
for r in removed:
    print(f"  {r['screened_pmc_id']}  {r['title'][:80]}")

print()

with open(DOI_MAP, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["screened_pmc_id", "doi", "title"])
    w.writeheader()
    w.writerows(keep)

print(f"doi_map: {len(rows)} → {len(keep)} entries")
print(f"All {len(keep)} articles have PDFs on disk.")
