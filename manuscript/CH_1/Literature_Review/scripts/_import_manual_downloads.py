"""
_import_manual_downloads.py
Watch C:\\Projects\\pgx-analysis\\manuscript\\infrastructure_setup\\manual_review\\
for manually downloaded PDFs and import them into data/scholar_pdfs/.

Two modes:
  --checklist   Generate TO_DOWNLOAD.csv + TO_DOWNLOAD.md in manual_review/
                (run once to get the shopping list)
  (default)     Scan manual_review/ for new PDFs, match to blocked articles,
                copy to data/scholar_pdfs/{hsh_id}.pdf, log as 'manual'.

Matching strategy (in order):
  1. Filename is exactly {hsh_id}.pdf
  2. Filename contains DOI (slashes → underscores)
  3. Fuzzy title match against blocked article list (>= 75% token overlap)

Run from: manuscript/CH_1/Literature_Review/
  python scripts/_import_manual_downloads.py --checklist
  python scripts/_import_manual_downloads.py
"""

import argparse
import csv
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

LOG_CSV      = Path("scripts/vcu_download_log.csv")
DOI_MAP      = Path("scripts/screened_doi_map.csv")
PDF_DIR      = Path("data/scholar_pdfs")
MANUAL_DIR   = Path(r"C:\Projects\pgx-analysis\manuscript\infrastructure_setup\manual_review")
OA_CSV       = Path("scripts/oa_scan_results.csv")

LOG_FIELDS = ["hsh_id", "title", "doi", "proxy_url", "status", "bytes", "timestamp"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_latest_log():
    """Return dict hsh_id → latest log row."""
    if not LOG_CSV.exists():
        return {}
    rows = list(csv.DictReader(open(LOG_CSV, encoding="utf-8-sig")))
    latest = {}
    for r in rows:
        hid = r["hsh_id"]
        if hid not in latest or r["timestamp"] > latest[hid]["timestamp"]:
            latest[hid] = r
    return latest

def load_doi_map():
    """Return dict hsh_id → {doi, title}."""
    result = {}
    for r in csv.DictReader(open(DOI_MAP, encoding="utf-8-sig")):
        result[r["screened_pmc_id"]] = {"doi": r["doi"], "title": r["title"]}
    return result

def append_log(row):
    exists = LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)

def doi_to_filestem(doi: str) -> str:
    """Normalise DOI to a filename-safe stem."""
    return doi.replace("/", "_").replace(":", "-").lower().strip()

def fuzzy_match(filename: str, candidates: list[tuple]) -> str | None:
    """
    candidates: list of (hsh_id, title)
    Returns hsh_id of best match if token overlap >= 75%, else None.
    """
    stem = Path(filename).stem.lower().replace("_", " ").replace("-", " ")
    stem_tokens = set(stem.split())
    best_hid, best_score = None, 0.0
    for hid, title in candidates:
        t_tokens = set(title.lower().split())
        if not t_tokens:
            continue
        overlap = len(stem_tokens & t_tokens) / len(t_tokens)
        if overlap > best_score:
            best_score = overlap
            best_hid = hid
    if best_score >= 0.75:
        return best_hid
    return None

# ── Publisher helper ──────────────────────────────────────────────────────────

def pub_from_url(url: str) -> str:
    u = (url or "").lower()
    for k, n in [("sciencedirect", "Elsevier"), ("sagepub", "SAGE"),
                 ("academic.oup", "OUP"), ("psycnet", "APA"),
                 ("lww", "LWW"), ("ieeexplore", "IEEE"),
                 ("muse.jhu", "MUSE"), ("thieme", "Thieme"),
                 ("wmpllc", "JOM")]:
        if k in u:
            return n
    return "Other"

# ── Relevance keywords ────────────────────────────────────────────────────────

_HIGH_KW = ["opioid", "naloxone", "buprenorphine", "pharmacogenomic",
            "pharmacogenetic", "pgx", "cyp", "drug-drug", "ddi",
            "polypharmacy", "adverse drug", "overdose", "substance use",
            "opioid use disorder", "health dispar", "racial", "telehealth",
            "moud", "suicide", "hiv", "hepatitis", "amputation",
            "prescription opioid", "injection drug", "gene", "snp"]
_SKIP_KW = ["guar gum", "pomegranate", "intragastric balloon",
            "ursodeoxycholic", "gallstone", "prader-willi",
            "ldl-c", "trs2p", "bcl-xl", "protac", "berberine nanopart",
            "toosendanin", "diltiazem liver", "intranasal oxytocin",
            "post-roe", "mammography"]

def relevance(title: str) -> str:
    t = title.lower()
    if any(k in t for k in _SKIP_KW): return "SKIP"
    if any(k in t for k in _HIGH_KW): return "HIGH"
    return "MED"

# ── Load known OA URLs ────────────────────────────────────────────────────────

def load_oa_urls() -> dict:
    """doi → pdf_url for url_found_no_download entries."""
    if not OA_CSV.exists():
        return {}
    return {r["doi"]: r["pdf_url"]
            for r in csv.DictReader(open(OA_CSV))
            if r.get("status") == "url_found_no_download" and r.get("pdf_url")}

# ── Mode: generate checklist ──────────────────────────────────────────────────

def generate_checklist():
    latest  = load_latest_log()
    doi_map = load_doi_map()
    oa_urls = load_oa_urls()

    ok_ids  = {hid for hid, r in latest.items() if r["status"] in ("ok", "manual")}
    on_disk = {p.stem for p in PDF_DIR.glob("*.pdf")}
    already = ok_ids | on_disk

    blocked = []
    for hid, info in doi_map.items():
        if hid in already:
            continue
        lr = latest.get(hid, {})
        proxy_url = lr.get("proxy_url", "")
        title = info["title"] or lr.get("title", "")
        doi   = info["doi"]
        rel   = relevance(title)
        if rel == "SKIP":
            continue
        blocked.append({
            "hsh_id":    hid,
            "doi":       doi,
            "title":     title,
            "publisher": pub_from_url(proxy_url),
            "relevance": rel,
            "url":       oa_urls.get(doi, ""),
            "status":    lr.get("status", "not_attempted"),
            "save_as":   f"{hid}.pdf",
        })

    # Sort: HIGH first, then MED; within each by publisher
    blocked.sort(key=lambda r: (0 if r["relevance"] == "HIGH" else 1, r["publisher"]))

    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = MANUAL_DIR / "TO_DOWNLOAD.csv"
    out_md  = MANUAL_DIR / "TO_DOWNLOAD.md"

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["hsh_id","doi","title","publisher",
                                           "relevance","url","status","save_as"])
        w.writeheader()
        w.writerows(blocked)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Manual Download Checklist\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n")
        f.write(f"Save PDFs to: `{MANUAL_DIR}`  \n")
        f.write("**Name each file as the `save_as` column** (= `{hsh_id}.pdf`)  \n")
        f.write("Then run: `python scripts/_import_manual_downloads.py`\n\n")

        for rel_label, rel_key in [("⭐ HIGH relevance", "HIGH"), ("🔶 MEDIUM relevance", "MED")]:
            subset = [r for r in blocked if r["relevance"] == rel_key]
            if not subset:
                continue
            f.write(f"## {rel_label} ({len(subset)} articles)\n\n")
            f.write("| # | Save as | Publisher | DOI | Title | URL |\n")
            f.write("|---|---------|-----------|-----|-------|-----|\n")
            for i, r in enumerate(subset, 1):
                url_md = f"[open]({r['url']})" if r["url"] else ""
                doi_short = r["doi"][:45] if r["doi"] else ""
                f.write(f"| {i} | `{r['save_as']}` | {r['publisher']} | "
                        f"`{doi_short}` | {r['title'][:70]} | {url_md} |\n")
            f.write("\n")

    print(f"Checklist written:")
    print(f"  CSV : {out_csv}")
    print(f"  MD  : {out_md}")
    print(f"  Articles to download: {len(blocked)}"
          f"  ({sum(1 for r in blocked if r['relevance']=='HIGH')} HIGH, "
          f"{sum(1 for r in blocked if r['relevance']=='MED')} MED)")

# ── Mode: import found PDFs ───────────────────────────────────────────────────

def import_pdfs():
    latest  = load_latest_log()
    doi_map = load_doi_map()

    ok_ids  = {hid for hid, r in latest.items() if r["status"] in ("ok", "manual")}
    on_disk = {p.stem for p in PDF_DIR.glob("*.pdf")}
    already = ok_ids | on_disk

    # Build lookup structures
    hid_set      = set(doi_map.keys())
    doi_to_hid   = {v["doi"].lower(): k for k, v in doi_map.items() if v["doi"]}
    hid_titles   = [(hid, v["title"]) for hid, v in doi_map.items()]

    pdfs = [p for p in MANUAL_DIR.glob("*.pdf")]
    if not pdfs:
        print(f"No PDFs found in {MANUAL_DIR}")
        print("Drop downloaded PDFs there and re-run.")
        return

    print(f"Found {len(pdfs)} PDF(s) in manual_review/\n")
    imported = skipped = unmatched = 0

    for pdf in sorted(pdfs):
        stem = pdf.stem
        name = pdf.name
        matched_hid = None
        match_method = ""

        # Strategy 1: exact hsh_id filename
        if stem in hid_set:
            matched_hid  = stem
            match_method = "hsh_id"

        # Strategy 2: DOI in filename (slashes → underscores)
        if not matched_hid:
            stem_norm = stem.lower()
            for doi, hid in doi_to_hid.items():
                doi_norm = doi_to_filestem(doi)
                if doi_norm in stem_norm or stem_norm in doi_norm:
                    matched_hid  = hid
                    match_method = "doi"
                    break

        # Strategy 3: fuzzy title match
        if not matched_hid:
            matched_hid = fuzzy_match(name, hid_titles)
            if matched_hid:
                match_method = "fuzzy-title"

        if not matched_hid:
            print(f"  ✗ UNMATCHED: {name}")
            print(f"    → Rename to {{hsh_id}}.pdf using hsh_id from TO_DOWNLOAD.csv")
            unmatched += 1
            continue

        info  = doi_map.get(matched_hid, {})
        title = info.get("title", "")[:70]
        doi   = info.get("doi", "")

        if matched_hid in already:
            print(f"  ↩ ALREADY DONE [{match_method}]: {title}")
            skipped += 1
            continue

        dest = PDF_DIR / f"{matched_hid}.pdf"
        shutil.copy2(pdf, dest)
        nbytes = dest.stat().st_size

        append_log({
            "hsh_id":    matched_hid,
            "title":     title,
            "doi":       doi,
            "proxy_url": "manual",
            "status":    "manual",
            "bytes":     nbytes,
            "timestamp": datetime.utcnow().isoformat(),
        })

        print(f"  ✓ IMPORTED [{match_method}]: {title}")
        print(f"    → {dest.name}  ({nbytes // 1024} KB)")
        imported += 1

    print(f"\n── Import complete ───────────────────────────────────")
    print(f"  Imported : {imported}")
    print(f"  Skipped  : {skipped} (already done)")
    print(f"  Unmatched: {unmatched} (rename to {{hsh_id}}.pdf)")
    if imported:
        # Refresh status summary
        latest2 = load_latest_log()
        total_ok = sum(1 for r in latest2.values() if r["status"] in ("ok", "manual"))
        print(f"\n  Total downloaded (auto + manual): {total_ok}/117")

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checklist", action="store_true",
                        help="Generate TO_DOWNLOAD.csv and TO_DOWNLOAD.md")
    args = parser.parse_args()

    PDF_DIR.mkdir(parents=True, exist_ok=True)

    if args.checklist:
        generate_checklist()
    else:
        import_pdfs()

if __name__ == "__main__":
    main()
