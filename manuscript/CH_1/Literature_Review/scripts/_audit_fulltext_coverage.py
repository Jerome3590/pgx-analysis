"""
Audit full-text JSON coverage across all screened articles.

Outputs:
  - Console summary by decision / score tier
  - scripts/missing_fulltext.csv  — articles lacking scholar_json/, with pmc_id/doi for fetching
"""
import csv, json
from pathlib import Path
from collections import Counter, defaultdict

SCREENED    = Path("data/ontology/articles_screened.csv")
SCHOLAR_JSON = Path("data/scholar_json")
OUT_CSV     = Path("scripts/missing_fulltext.csv")

# Build index of available full-text JSONs
json_index = {p.stem for p in SCHOLAR_JSON.glob("*.json")}
print(f"scholar_json/ index: {len(json_index)} files\n")

rows = list(csv.DictReader(open(SCREENED, encoding="utf-8-sig")))

# ── Tag each row ───────────────────────────────────────────────────────────────
for row in rows:
    pmc_id     = row.get("pmc_id", "").strip()
    article_id = row.get("article_id", "").strip()
    has_json   = (pmc_id in json_index) or (f"article_{article_id}" in json_index)
    row["_has_json"]   = has_json
    row["_pmc_id"]     = pmc_id
    try:
        row["_score"] = float(row.get("composite_score", 0) or 0)
    except:
        row["_score"] = 0.0

# ── Score tiers ────────────────────────────────────────────────────────────────
def tier(score):
    if score >= 0.3:  return "high   (>=0.30)"
    if score >= 0.15: return "medium (0.15-0.30)"
    if score >= 0.05: return "low    (0.05-0.15)"
    return               "none   (<0.05)"

# ── Summary ────────────────────────────────────────────────────────────────────
total      = len(rows)
have_json  = sum(1 for r in rows if r["_has_json"])
miss_json  = total - have_json

print(f"Total screened        : {total}")
print(f"Have full-text JSON   : {have_json}  ({have_json/total*100:.1f}%)")
print(f"Missing full-text     : {miss_json}  ({miss_json/total*100:.1f}%)")
print()

# Breakdown by human_decision × has_json
print("Coverage by decision:")
for decision in ("include", "exclude", ""):
    subset = [r for r in rows if r.get("human_decision","") == decision]
    with_j  = sum(1 for r in subset if r["_has_json"])
    without = len(subset) - with_j
    label   = decision if decision else "(empty)"
    print(f"  {label:<10}  total={len(subset):5d}  with_json={with_j:5d}  missing={without:5d}")
print()

# Missing articles by score tier and decision
print("Missing full-text breakdown (decision × score tier):")
missing = [r for r in rows if not r["_has_json"]]
grid = defaultdict(int)
for r in missing:
    key = (r.get("human_decision","(empty)"), tier(r["_score"]))
    grid[key] += 1
for decision in ("include", "exclude"):
    for t in ("high   (>=0.30)", "medium (0.15-0.30)", "low    (0.05-0.15)", "none   (<0.05)"):
        n = grid.get((decision, t), 0)
        if n:
            print(f"  {decision:<10} × {t} : {n:5d}")
print()

# Missing with PMC ID (can re-download)
miss_with_pmc   = [r for r in missing if r["_pmc_id"]]
miss_no_pmc     = [r for r in missing if not r["_pmc_id"]]
print(f"Missing with PMC ID (can re-download BioC) : {len(miss_with_pmc)}")
print(f"Missing without PMC ID (need DOI/PDF)      : {len(miss_no_pmc)}")
print()

# Priority: excluded with score >= 0.10 and no full text
priority = sorted(
    [r for r in missing if r["_score"] >= 0.10],
    key=lambda r: r["_score"], reverse=True
)
print(f"Priority missing (score >= 0.10, no full text): {len(priority)}")
for r in priority[:20]:
    pmc  = r["_pmc_id"] or "(no PMC)"
    dec  = r.get("human_decision","?")
    print(f"  [{dec}] score={r['_score']:.4f}  {pmc:<14}  {(r.get('title',''))[:65]}")

# ── Write missing_fulltext.csv ─────────────────────────────────────────────────
FIELDS = ["article_id", "pmc_id", "title", "doi", "composite_score",
          "human_decision", "has_pmc", "fetch_method"]

def doi_from_row(r):
    return ""  # doi not stored in articles_screened; would come from doi_map

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    for r in missing:
        has_pmc = bool(r["_pmc_id"])
        w.writerow({
            "article_id":      r.get("article_id", ""),
            "pmc_id":          r["_pmc_id"],
            "title":           r.get("title", "")[:120],
            "doi":             "",
            "composite_score": r["_score"],
            "human_decision":  r.get("human_decision", ""),
            "has_pmc":         "Y" if has_pmc else "N",
            "fetch_method":    "pmc_bioc" if has_pmc else "unpaywall_or_manual",
        })

print(f"\nWrote {len(missing)} rows to {OUT_CSV}")
print(f"  Rows with pmc_id (pmc_bioc)       : {len(miss_with_pmc)}")
print(f"  Rows without pmc_id (unpaywall)   : {len(miss_no_pmc)}")
