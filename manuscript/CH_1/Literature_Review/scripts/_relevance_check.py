"""
_relevance_check.py
Cross-reference blocked articles against CH_1 topic areas:
  PGx / pharmacogenomics, opioid use disorder, health disparities,
  drug-drug interactions (DDI), polypharmacy, adverse drug events.
Outputs a relevance-ranked table for manual download decisions.
"""

import csv
from pathlib import Path

LOG_CSV   = Path("scripts/vcu_download_log.csv")
OA_CSV    = Path("scripts/oa_scan_results.csv")
DOI_MAP   = Path("scripts/screened_doi_map.csv")

# ── Load latest status per article ───────────────────────────────────────────
rows = list(csv.DictReader(open(LOG_CSV, encoding="utf-8-sig")))
latest = {}
for r in rows:
    hid = r["hsh_id"]
    if hid not in latest or r["timestamp"] > latest[hid]["timestamp"]:
        latest[hid] = r

doi_map = {r["screened_pmc_id"]: r["doi"]
           for r in csv.DictReader(open(DOI_MAP, encoding="utf-8-sig"))}

oa_urls = {r["doi"]: r["pdf_url"]
           for r in csv.DictReader(open(OA_CSV))
           if r.get("pdf_url")}

# ── Relevance keywords → tier ─────────────────────────────────────────────────
HIGH = [
    "opioid", "naloxone", "buprenorphine", "methadone", "fentanyl",
    "pharmacogenomic", "pharmacogenetic", "pgx", "cyp", "gene", "snp",
    "drug-drug interaction", "ddi", "polypharmacy", "adverse drug",
    "overdose", "substance use", "opioid use disorder", "oud",
    "pain management", "prescription opioid", "opioid dispensing",
    "health dispar", "racial", "race", "ethnic", "indigenous",
    "telehealth", "telemedicine", "moud", "medication-assisted",
    "buprenorphine", "naltrexone", "ptsd", "suicide", "mental health",
    "hiv", "hepatitis", "amputation", "injection drug",
]
MEDIUM = [
    "drug reaction", "drug interaction", "drug event", "drug safety",
    "prescribing", "medication", "clinical pharmacol", "toxicol",
    "machine learning", "deep learning", "predict", "nlp",
    "natural language", "social determinant", "social factor",
    "postoperative", "cancer", "breast cancer", "lupus",
]
SKIP_KW = [
    "guar gum", "pomegranate", "intragastric balloon", "obesity interv",
    "ursodeoxycholic", "gallstone", "prader-willi", "oxytocin hyperphagia",
    "ldl-c", "trs2p", "bcl-xl", "protac", "berberine nanopart",
    "toosendanin", "diltiazem liver", "intranasal oxytocin",
    "post-roe", "mammography", "lipid profile",
]

def tier(title):
    t = title.lower()
    if any(k in t for k in SKIP_KW): return "SKIP"
    if any(k in t for k in HIGH):    return "HIGH"
    if any(k in t for k in MEDIUM):  return "MED"
    return "MED"  # when uncertain, keep

# ── Collect blocked articles ──────────────────────────────────────────────────
blocked = [r for hid, r in latest.items()
           if r["status"] != "ok"
           and Path(f"data/scholar_pdfs/{hid}.pdf").exists() is False]

# Add oa_scan URL-found ones too (manually downloadable)
oa_rows = list(csv.DictReader(open(OA_CSV)))
url_found = [r for r in oa_rows if r["status"] == "url_found_no_download"]

# ── Print by tier ─────────────────────────────────────────────────────────────
def pub(url):
    u = (url or "").lower()
    for k, n in [("sciencedirect","Elsevier"), ("sagepub","SAGE"),
                 ("academic.oup","OUP"), ("psycnet","APA"),
                 ("lww","LWW"), ("ieeexplore","IEEE"),
                 ("muse.jhu","MUSE"), ("thieme","Thieme"),
                 ("wmpllc","JOM"), ("cambridge","Cambridge")]:
        if k in u: return n
    return "Other"

print("=" * 95)
print("MANUALLY DOWNLOADABLE — URL already known (open in browser + save as PDF)")
print("=" * 95)
for r in url_found:
    t = tier(r["title"])
    print(f"  [{t:4}]  {r['title'][:70]}")
    print(f"          DOI: {r['doi']}")
    print(f"          URL: {r['pdf_url'][:90]}")
    print()

for label, statuses in [
    ("HIGH RELEVANCE — manual download / ILL recommended", ["HIGH"]),
    ("MEDIUM RELEVANCE — download if scope allows",        ["MED"]),
    ("SKIP — likely off-topic",                            ["SKIP"]),
]:
    items = [(hid, r) for hid, r in latest.items()
             if r["status"] != "ok"
             and not Path(f"data/scholar_pdfs/{hid}.pdf").exists()
             and tier(r.get("title","")) in statuses]

    # Deduplicate by title
    seen = set()
    deduped = []
    for hid, r in items:
        key = r["title"][:60]
        if key not in seen:
            seen.add(key)
            deduped.append((hid, r))

    print()
    print("=" * 95)
    print(f"{label} ({len(deduped)} articles)")
    print("=" * 95)
    for hid, r in sorted(deduped, key=lambda x: x[1].get("title","")):
        doi = doi_map.get(hid, "")
        publisher = pub(r.get("proxy_url",""))
        print(f"  [{publisher:<10}]  {r['title'][:75]}")
        if doi:
            print(f"               DOI: {doi}")
