"""
_oa_scan_blocked.py
Check blocked articles (no_pdf / error / doi_resolve_failed) from
vcu_download_log.csv against Unpaywall and Europe PMC for free OA PDFs.
Downloads any found PDFs directly to data/scholar_pdfs/{hsh_id}.pdf.

Run from: manuscript/CH_1/Literature_Review/
  python scripts/_oa_scan_blocked.py
"""

import csv
import os
import time
from pathlib import Path
import urllib.request
import urllib.parse
import json

LOG_CSV   = Path("scripts/vcu_download_log.csv")
DOI_MAP   = Path("scripts/screened_doi_map.csv")
PDF_DIR   = Path("data/scholar_pdfs")
OUT_CSV   = Path("scripts/oa_scan_results.csv")
EMAIL     = "dixonrj@vcu.edu"
SLEEP     = 0.5

def get(url, params=None):
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": f"PGxLitReview/1.0 (mailto:{EMAIL})"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None

def unpaywall(doi):
    data = get(f"https://api.unpaywall.org/v2/{doi}", {"email": EMAIL})
    if not data:
        return None, None
    best = data.get("best_oa_location") or {}
    pdf = best.get("url_for_pdf")
    if not pdf:
        for loc in data.get("oa_locations", []):
            if loc.get("url_for_pdf"):
                pdf = loc["url_for_pdf"]
                break
    is_oa = data.get("is_oa", False)
    return pdf, is_oa

def europepmc(doi):
    data = get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
               {"query": f"DOI:{doi}", "format": "json", "resultType": "core", "pageSize": 1})
    if not data:
        return None
    results = data.get("resultList", {}).get("result", [])
    if not results:
        return None
    r = results[0]
    # Check if full text available in PMC
    if r.get("pmcid"):
        pmcid = r["pmcid"]
        return f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
    return None

def download_pdf(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": f"PGxLitReview/1.0 (mailto:{EMAIL})"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        if len(data) > 1024 and b"%PDF" in data[:16]:
            Path(dest).write_bytes(data)
            return len(data)
    except Exception:
        pass
    return 0

# ── Load download log: latest status per hsh_id ──────────────────────────────
rows = list(csv.DictReader(open(LOG_CSV, encoding="utf-8-sig")))
latest = {}
for r in rows:
    hid = r["hsh_id"]
    if hid not in latest or r["timestamp"] > latest[hid]["timestamp"]:
        latest[hid] = r

# ── Load DOI map for doi lookup ───────────────────────────────────────────────
doi_map = {}
for r in csv.DictReader(open(DOI_MAP, encoding="utf-8-sig")):
    doi_map[r["screened_pmc_id"]] = r["doi"]

# ── Already downloaded ────────────────────────────────────────────────────────
ok_ids = {hid for hid, r in latest.items() if r["status"] == "ok"}
already_files = {p.stem for p in PDF_DIR.glob("*.pdf")}

# ── Build blocked list (not already ok and no PDF on disk) ───────────────────
blocked = [r for hid, r in latest.items()
           if r["status"] != "ok" and hid not in already_files]

print(f"Total unique articles tracked : {len(latest)}")
print(f"Already downloaded (ok)       : {len(ok_ids)}")
print(f"Blocked (scanning for OA)     : {len(blocked)}")
print()

PDF_DIR.mkdir(parents=True, exist_ok=True)
results = []

for i, r in enumerate(blocked, 1):
    hid   = r["hsh_id"]
    title = r["title"][:80]
    doi   = doi_map.get(hid, "").strip()

    row = {"hsh_id": hid, "title": title, "doi": doi,
           "unpaywall_oa": "", "epmc_pdf": "", "pdf_url": "",
           "status": "", "bytes": 0}

    print(f"[{i:>3}/{len(blocked)}] {title[:70]}", end=" ", flush=True)

    if not doi:
        print("→ no_doi")
        row["status"] = "no_doi"
        results.append(row)
        continue

    time.sleep(SLEEP)

    # 1. Unpaywall
    pdf_url, is_oa = unpaywall(doi)
    row["unpaywall_oa"] = "yes" if is_oa else "no"

    # 2. Europe PMC fallback
    epmc_url = None
    if not pdf_url:
        time.sleep(SLEEP)
        epmc_url = europepmc(doi)
        row["epmc_pdf"] = epmc_url or ""

    final_url = pdf_url or epmc_url

    if final_url:
        row["pdf_url"] = final_url
        dest = PDF_DIR / f"{hid}.pdf"
        nbytes = download_pdf(final_url, dest)
        if nbytes:
            row["status"] = "downloaded"
            row["bytes"]  = nbytes
            print(f"→ OA PDF ✓  {nbytes//1024} KB")
        else:
            row["status"] = "url_found_no_download"
            print(f"→ URL found but download failed: {final_url[:60]}")
    else:
        row["status"] = "no_oa" if is_oa is False else "not_found"
        print(f"→ {'not OA' if is_oa is False else 'no OA found'}")

    results.append(row)

# ── Write results CSV ─────────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["hsh_id","title","doi","unpaywall_oa",
                                       "epmc_pdf","pdf_url","status","bytes"])
    w.writeheader()
    w.writerows(results)

# ── Summary ───────────────────────────────────────────────────────────────────
from collections import Counter
counts = Counter(r["status"] for r in results)
print()
print("── OA Scan complete ─────────────────────────────────")
print(f"  Downloaded (new PDFs)  : {counts['downloaded']}")
print(f"  URL found, no download : {counts['url_found_no_download']}")
print(f"  Not OA (need ILL)      : {counts['no_oa']}")
print(f"  No OA found            : {counts['not_found']}")
print(f"  No DOI                 : {counts['no_doi']}")
print(f"  Results CSV            : {OUT_CSV}")
print()

# ILL list
ill = [r for r in results if r["status"] in ("no_oa", "not_found", "url_found_no_download")]
if ill:
    print(f"── ILL request list ({len(ill)} articles) ─────────────────")
    for r in ill:
        print(f"  {r['doi'] or 'NO DOI':<42}  {r['title'][:65]}")
