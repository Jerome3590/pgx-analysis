"""
_gen_checklist_by_publisher.py
Regenerate TO_DOWNLOAD.md sorted by publisher (Elsevier first).
"""
import csv
from pathlib import Path

DOI_MAP  = Path("scripts/screened_doi_map.csv")
LOG_CSV  = Path("scripts/vcu_download_log.csv")
PDF_DIR  = Path("data/scholar_pdfs")
OUT_MD   = Path("infrastructure_setup/manual_review/TO_DOWNLOAD.md")
OUT_CSV  = Path("infrastructure_setup/manual_review/TO_DOWNLOAD.csv")

# ── Publisher detection from DOI prefix / journal ────────────────────────────
PUBLISHER_MAP = {
    # Elsevier DOI prefixes
    "10.1016": "Elsevier",
    "10.1053": "Elsevier",
    "10.1055": "Elsevier",
    # OUP
    "10.1093": "OUP",
    # APA / APA journals
    "10.1037": "APA",
    "10.1176": "APA",
    # Wiley
    "10.1002": "Wiley",
    "10.1111": "Wiley",
    # Springer/Nature
    "10.1007": "Springer",
    "10.1038": "Nature",
    "10.1186": "BioMed Central",
    # SAGE
    "10.1177": "SAGE",
    "10.1080": "Taylor & Francis",
    # BMJ
    "10.1136": "BMJ",
    # Lippincott / Wolters Kluwer
    "10.1097": "Lippincott",
    # JAMA Network
    "10.1001": "JAMA",
    # NLM / PMC free
    "10.1371": "PLOS",
    "10.3389": "Frontiers",
    # Misc
    "10.18553": "JMCP",
    "10.3399":  "BJGP",
    "10.1212":  "Neurology",
    "10.1161":  "AHA",
}

PUBLISHER_ORDER = [
    "Elsevier", "OUP", "APA", "Wiley", "Springer", "Lippincott",
    "SAGE", "Taylor & Francis", "JAMA", "BMJ", "Nature", "BioMed Central",
    "PLOS", "Frontiers", "Neurology", "AHA", "JMCP", "BJGP", "Other"
]

def detect_publisher(doi: str) -> str:
    prefix = "/".join(doi.split("/")[:1]) if "/" in doi else doi
    prefix = doi.split("/")[0] if "/" in doi else doi
    return PUBLISHER_MAP.get(prefix, "Other")

# ── Load data ─────────────────────────────────────────────────────────────────
doi_rows = list(csv.DictReader(open(DOI_MAP, encoding="utf-8-sig")))

log_rows = list(csv.DictReader(open(LOG_CSV, encoding="utf-8-sig")))
latest = {}
for r in log_rows:
    hid = r["hsh_id"]
    if hid not in latest or r["timestamp"] > latest[hid]["timestamp"]:
        latest[hid] = r

on_disk  = {p.stem for p in PDF_DIR.glob("*.pdf")}
ok_ids   = {hid for hid, r in latest.items()
            if r["status"] in ("ok", "manual", "zotero", "oa_url")}
have     = ok_ids | on_disk

# ── Build blocked list ────────────────────────────────────────────────────────
blocked = []
for r in doi_rows:
    hid   = r["screened_pmc_id"]
    doi   = r.get("doi", "").strip()
    title = r.get("title", "").strip()
    if hid in have or not doi:
        continue
    pub = detect_publisher(doi)
    url = f"https://doi.org/{doi}"
    blocked.append({
        "publisher": pub,
        "hsh_id":    hid,
        "doi":       doi,
        "title":     title,
        "url":       url,
        "save_as":   f"{hid}.pdf",
    })

# Sort by publisher order then title
pub_rank = {p: i for i, p in enumerate(PUBLISHER_ORDER)}
blocked.sort(key=lambda r: (pub_rank.get(r["publisher"], 99), r["title"].lower()))

# ── Write Markdown ────────────────────────────────────────────────────────────
OUT_MD.parent.mkdir(parents=True, exist_ok=True)

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(f"# Manual Download Checklist — {len(blocked)} remaining\n\n")
    f.write(f"> **Status:** 100/117 PDFs downloaded · {len(blocked)} still needed\n\n")
    f.write("## Workflow\n\n")
    f.write("### Option A — Assign PDF directly in Zotero (recommended)\n\n")
    f.write("1. Open Zotero → collection **\"PGx - Needs PDF\"** (key `GW8MHKW2`)\n")
    f.write("2. Click the **Proxy URL** link below → PDF opens in browser via VCU EZProxy\n")
    f.write("3. Save PDF anywhere (filename doesn't matter)\n")
    f.write("4. Drag the saved PDF onto the matching Zotero item to attach it\n")
    f.write("5. Repeat for all articles, then:\n")
    f.write("   - Close Zotero\n")
    f.write("   - Run: `python scripts/_import_zotero_pdfs.py`\n\n")
    f.write("### Option B — Drop file in manual_review/ folder\n\n")
    f.write("1. Save PDF to `manuscript/infrastructure_setup/manual_review/`\n")
    f.write("2. Run: `python scripts/_parse_pdf_titles.py` (matches by PDF title content)\n\n")
    f.write("---\n\n")

    current_pub = None
    for i, r in enumerate(blocked, 1):
        if r["publisher"] != current_pub:
            current_pub = r["publisher"]
            f.write(f"\n## {current_pub}\n\n")
        f.write(f"### {i}. {r['title'][:90]}\n")
        f.write(f"- **DOI:** `{r['doi']}`\n")
        f.write(f"- **Direct URL:** <{r['url']}>\n")
        f.write(f"- **Proxy URL:** <https://proxy.library.vcu.edu/login?url={r['url']}>\n")
        f.write(f"- **hsh_id:** `{r['hsh_id']}`  *(Zotero item title should match above)*\n")
        f.write(f"- [ ] PDF attached in Zotero\n\n")

# ── Write CSV ─────────────────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["publisher","hsh_id","doi","title","url","save_as"])
    w.writeheader()
    w.writerows(blocked)

# ── Summary ───────────────────────────────────────────────────────────────────
from collections import Counter
counts = Counter(r["publisher"] for r in blocked)
print(f"Blocked: {len(blocked)} articles\n")
for pub in PUBLISHER_ORDER:
    if pub in counts:
        print(f"  {pub:20s} {counts[pub]:3d}")
print(f"\nChecklist written to:\n  {OUT_MD}\n  {OUT_CSV}")
