"""
Build article_review_checklist.csv in manual_review/.

Columns: rank, article_id, title, doi, pub_year, authors,
         composite_score, pytextrank_score, combined_score, has_pdf, selected

Scope: all articles with human_decision=include from articles_screened.csv,
       published >= MIN_YEAR (default 2021 = last 5 years).
Top 85% by combined_score → selected=Y
Bottom 15% → selected='' (manual review needed)

Usage:
  python scripts/_build_review_checklist.py              # 2021+
  python scripts/_build_review_checklist.py --min-year 2020
  python scripts/_build_review_checklist.py --min-year 0  # no filter
"""
import argparse, csv, re, shutil
from pathlib import Path

import spacy
import pytextrank  # noqa: F401

SCREENED  = Path("data/ontology/articles_screened.csv")
DOI_MAP   = Path("scripts/screened_doi_map.csv")
PDF_DIR   = Path("data/scholar_pdfs")
OUT_DIR   = Path(r"C:\Projects\pgx-analysis\manuscript\infrastructure_setup\manual_review")
OUT_CSV   = OUT_DIR / "article_review_checklist.csv"
OUT_README = OUT_DIR / "REVIEW_GUIDE.md"

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--min-year", type=int, default=0,
                    help="Keep only articles published >= this year (0 = no filter, default)")
args = parser.parse_args()
MIN_YEAR = args.min_year

# ── spaCy + pytextrank ─────────────────────────────────────────────────────────
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("textrank")

TOPIC_TOKENS: dict[str, set[str]] = {
    "pgx": {
        "pharmacogenomic","pharmacogenomics","pharmacogenetic","pharmacogenetics",
        "cyp","cytochrome","cyp2d6","cyp3a4","cyp2c19","cyp2c9",
        "drug metabolism","metabolizer","genotype","allele","snp","haplotype",
        "biomarker","variant","polymorphism","gene","genetic","genomic","oprm1","comt",
    },
    "ddi": {
        "drug-drug","drug interaction","ddi","polypharmacy","drug combination",
        "interaction","comedication","coprescription","multidrug","contraindication",
    },
    "oud": {
        "opioid","opioids","naloxone","buprenorphine","methadone","fentanyl","heroin",
        "overdose","opioid use disorder","oud","substance use","addiction","dependence",
        "withdrawal","pain management","analgesic","tramadol","morphine","hydrocodone",
        "oxycodone","prescription opioid",
    },
    "pharma": {
        "adverse drug","adverse event","adr","ade","pharmacovigilance","faers",
        "spontaneous report","signal detection","drug safety","drug reaction",
        "side effect","toxicity","hepatotoxic","drug-induced",
    },
    "claims": {
        "claims","apcd","all-payer","administrative data","insurance","medicaid",
        "medicare","electronic health record","ehr","database","real-world",
        "population-based",
    },
    "ml": {
        "machine learning","deep learning","neural network","xgboost","random forest",
        "gradient boosting","classification","prediction","predictive model",
        "clinical decision support","algorithm","artificial intelligence","nlp",
        "explainab","shap",
    },
    "process": {
        "process mining","event log","workflow","bupar","ooda","petri net",
    },
}
WEIGHTS = {
    "pgx": 3.0, "ddi": 2.5, "oud": 2.5, "pharma": 2.0,
    "claims": 1.5, "ml": 1.5, "process": 1.0,
}

def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())

def ptr_score(title: str, key_phrases: str) -> float:
    combined = normalize(title + " " + key_phrases)
    s = 0.0
    try:
        doc = nlp(title[:500])
        for phrase in doc._.phrases[:15]:
            ph = normalize(phrase.text)
            for t, toks in TOPIC_TOKENS.items():
                if any(tok in ph for tok in toks):
                    s += phrase.rank * WEIGHTS[t]
                    break
    except Exception:
        pass
    for t, toks in TOPIC_TOKENS.items():
        if any(tok in combined for tok in toks):
            s += 0.05 * WEIGHTS[t]
    return round(s, 4)

# ── Load data ──────────────────────────────────────────────────────────────────
rows = list(csv.DictReader(open(SCREENED, encoding="utf-8-sig")))
included_all = [r for r in rows if r.get("human_decision") == "include"]

if MIN_YEAR > 0:
    def _keep(r):
        y = (r.get("pubdate", "") or "")[:4]
        return (not y.isdigit()) or int(y) >= MIN_YEAR  # keep unknown dates
    included = [r for r in included_all if _keep(r)]
    dropped  = len(included_all) - len(included)
    print(f"Articles with human_decision=include: {len(included_all)}")
    print(f"  Year filter ≥{MIN_YEAR}: kept {len(included)}  (dropped {dropped} pre-{MIN_YEAR})")
else:
    included = included_all
    print(f"Articles with human_decision=include: {len(included)}  (no year filter)")

doi_map = {r["screened_pmc_id"]: r for r in csv.DictReader(open(DOI_MAP, encoding="utf-8-sig"))}
pdf_ids = {p.stem for p in PDF_DIR.glob("*.pdf")}

# ── Score ──────────────────────────────────────────────────────────────────────
print("Computing pytextrank scores...")
scored = []
for i, row in enumerate(included):
    title       = row.get("title", "") or ""
    key_phrases = row.get("key_phrases", "") or ""
    composite   = float(row.get("composite_score", 0) or 0)
    ptr         = ptr_score(title, key_phrases)
    combined    = round(ptr + composite * 0.3, 4)

    # DOI: from doi_map if article_id matches
    dm = doi_map.get(row["article_id"], {})
    doi = dm.get("doi", "") if dm else ""

    scored.append({
        "article_id":        row["article_id"],
        "title":             title,
        "doi":               doi,
        "pub_year":          row.get("pubdate", "")[:4],
        "authors":           row.get("authors", "")[:80],
        "composite_score":   composite,
        "pytextrank_score":  ptr,
        "combined_score":    combined,
        "ooda_phase":        row.get("ooda_phase_primary", ""),
        "crisp_dm_phase":    row.get("crisp_dm_phase", ""),
        "ooda_crisp_label":  row.get("ooda_crisp_label", ""),
        "nih_ai_score":      row.get("nih_ai_score", ""),
        "nih_ai_tags":       row.get("nih_ai_tags", ""),
        "op_perf_tags":      row.get("op_perf_tags", ""),
        "has_pdf":           "Y" if row["article_id"] in pdf_ids else "",
        "selected":          "",
        "notes":             "",
    })
    if (i + 1) % 500 == 0:
        print(f"  ... {i+1}/{len(included)}")

# Sort by combined_score descending
scored.sort(key=lambda r: r["combined_score"], reverse=True)

# ── Carry forward prior review decisions (idempotency) ─────────────────────────
prior: dict[str, dict] = {}
if OUT_CSV.exists():
    try:
        for pr in csv.DictReader(open(OUT_CSV, encoding="utf-8-sig")):
            aid = pr.get("article_id", "").strip()
            sel = pr.get("selected", "").strip()
            notes = pr.get("notes", "").strip()
            if aid and (sel or notes):
                prior[aid] = {"selected": sel, "notes": notes}
        print(f"Prior checklist: {len(prior)} articles with existing decisions/notes")
    except Exception as e:
        print(f"  (Could not load prior checklist: {e})")

# Top 85% → selected=Y; carry over prior decisions where they exist
cutoff = int(len(scored) * 0.85)
carried  = 0
fresh_y  = 0
fresh_bl = 0
for i, row in enumerate(scored):
    row["rank"] = i + 1
    aid = row["article_id"]
    if aid in prior:
        row["selected"] = prior[aid]["selected"]
        row["notes"]    = prior[aid]["notes"]
        carried += 1
    else:
        if i < cutoff or row["has_pdf"] == "Y":
            row["selected"] = "Y"
            fresh_y += 1
        else:
            row["selected"] = "N"
            fresh_bl += 1
    # PDFs always Y — overrides even a prior N (Zotero addition beats auto-exclude)
    if row["has_pdf"] == "Y":
        row["selected"] = "Y"

y_count = sum(1 for r in scored if r["selected"] == "Y")
review_count = sum(1 for r in scored if r["selected"] == "")
print(f"\nTop 85% cutoff at rank {cutoff}")
print(f"  Carried forward  : {carried}  (prior decisions preserved)")
print(f"  Auto-selected=Y  : {fresh_y}  (new articles in top 85% or have PDF)")
print(f"  Auto-excluded=N  : {fresh_bl}  (new articles below cutoff, not in Zotero)")
print(f"  Total selected=Y : {y_count}")
print(f"  Total selected=N : {sum(1 for r in scored if r['selected'] == 'N')}")
print(f"  Blank (override) : {review_count}")

# ── Write CSV ──────────────────────────────────────────────────────────────────
FIELDS = [
    "rank", "article_id", "title", "doi", "pub_year", "authors",
    "composite_score", "pytextrank_score", "combined_score",
    "ooda_phase", "crisp_dm_phase", "ooda_crisp_label",
    "nih_ai_score", "nih_ai_tags", "op_perf_tags",
    "has_pdf", "selected", "notes",
]
OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(scored)
print(f"\nChecklist written: {OUT_CSV}")
print(f"  {len(scored)} rows · {y_count} pre-selected · {review_count} need review")

# ── Copy doi_map to manual_review ─────────────────────────────────────────────
shutil.copy2(DOI_MAP, OUT_DIR / "screened_doi_map.csv")
print(f"Copied screened_doi_map.csv → {OUT_DIR}")

# ── Write REVIEW_GUIDE.md ─────────────────────────────────────────────────────
with open(OUT_README, "w", encoding="utf-8") as f:
    f.write("# Article Review Guide\n\n")
    f.write(f"> Generated: 2026-03-26  |  Scope: {len(scored)} articles (human_decision=include, pub_year≥{MIN_YEAR})\n\n")
    f.write("## Files in this folder\n\n")
    f.write("| File | Description |\n|------|-------------|\n")
    f.write("| `article_review_checklist.csv` | **Primary review file** — upload to Google Sheets |\n")
    f.write("| `screened_doi_map.csv` | 119 articles with DOIs and PDFs on disk |\n")
    f.write("| `TO_DOWNLOAD.md` | Proxy URLs for any remaining downloads |\n\n")
    f.write("## Checklist columns\n\n")
    f.write("| Column | Description |\n|--------|-------------|\n")
    f.write("| `rank` | Sorted by combined_score (1 = most relevant) |\n")
    f.write("| `article_id` | Unique article identifier |\n")
    f.write("| `title` | Article title |\n")
    f.write("| `doi` | DOI (filled for 119 PDF articles) |\n")
    f.write("| `composite_score` | Original algorithm relevance score |\n")
    f.write("| `pytextrank_score` | PGx/OUD/DDI/FAERS topic phrase score |\n")
    f.write("| `combined_score` | pytextrank + 0.3×composite (sort key) |\n")
    f.write("| `has_pdf` | Y = PDF on disk in data/scholar_pdfs/ |\n")
    f.write(f"| `selected` | **Y** = top 85% or has PDF · **N** = below cutoff (auto-excluded). Override by changing N→Y in Google Sheets |\n")
    f.write("| `notes` | Your notes |\n\n")
    f.write("## Workflow\n\n")
    f.write("1. Upload `article_review_checklist.csv` to Google Sheets\n")
    f.write("2. Filter `selected` = N → review those rows\n")
    f.write(f"3. Set `selected` = **Y** or **N** for each of the {review_count} rows marked N\n")
    f.write("4. Export sheet back to CSV (same filename)\n")
    f.write("5. Replace `article_review_checklist.csv` in this folder\n")
    f.write("6. Run: `python scripts/_apply_checklist_decisions.py`\n\n")
    f.write("## Score thresholds used\n\n")
    f.write(f"- Publication year filter: **≥{MIN_YEAR}** ({len(included_all) - len(scored)} pre-{MIN_YEAR} articles excluded)\n")
    f.write(f"- Phase 7 pytextrank threshold: **0.20** → {len(scored)} articles passed\n")
    f.write(f"- Top 85% cutoff: combined_score ≥ **{scored[cutoff-1]['combined_score']:.4f}** "
            f"(rank {cutoff})\n")
    f.write(f"- Articles with PDFs always marked Y regardless of rank\n")

print(f"Guide written: {OUT_README}")
