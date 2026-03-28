"""
_download_oa_urls.py
Download PDFs from known open-access URLs in oa_scan_results.csv
(status == url_found_no_download).
"""
import csv
from datetime import datetime
from pathlib import Path

import requests

OA_CSV  = Path("scripts/oa_scan_results.csv")
DOI_MAP = Path("scripts/screened_doi_map.csv")
LOG_CSV = Path("scripts/vcu_download_log.csv")
PDF_DIR = Path("data/scholar_pdfs")

PDF_DIR.mkdir(parents=True, exist_ok=True)

doi_map   = {r["doi"].lower(): r["screened_pmc_id"]
             for r in csv.DictReader(open(DOI_MAP, encoding="utf-8-sig"))}
title_map = {r["doi"].lower(): r["title"]
             for r in csv.DictReader(open(DOI_MAP, encoding="utf-8-sig"))}
on_disk   = {p.stem for p in PDF_DIR.glob("*.pdf")}

urls = [r for r in csv.DictReader(open(OA_CSV))
        if r["status"] == "url_found_no_download" and r.get("pdf_url")]

print(f"OA URLs to attempt: {len(urls)}\n")
imported = 0

for r in urls:
    doi   = r["doi"].lower()
    url   = r["pdf_url"]
    title = r.get("title", "")
    hid   = doi_map.get(doi)

    if not hid:
        print(f"  NO HID : {doi[:55]}")
        continue
    if hid in on_disk:
        print(f"  SKIP   : {title[:65]}")
        continue

    print(f"  GET    : {title[:65]}")
    print(f"           {url[:90]}")
    try:
        resp = requests.get(url, timeout=30, allow_redirects=True,
                            headers={"User-Agent": "Mozilla/5.0"})
        is_pdf = resp.content[:5] == b"%PDF-" or b"%PDF-" in resp.content[:20]
        if resp.status_code == 200 and is_pdf:
            dest = PDF_DIR / f"{hid}.pdf"
            dest.write_bytes(resp.content)
            nb = len(resp.content)
            with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    hid, title[:70], doi, url, "oa_url", nb,
                    datetime.utcnow().isoformat()
                ])
            print(f"           -> OK  ({nb // 1024} KB)")
            on_disk.add(hid)
            imported += 1
        else:
            print(f"           -> FAIL  HTTP {resp.status_code}  pdf={is_pdf}")
    except Exception as e:
        print(f"           -> ERROR: {e}")

print(f"\nImported : {imported}/{len(urls)}")
print(f"On disk  : {len(on_disk)}/117")
