"""
Derive reproducible study-characteristic counts from evidence-map outputs.

Expected inputs are not included in this public code companion. Place compatible
CSV exports under ``inputs/ch01/`` or update the paths below before running.
"""
import csv
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "inputs" / "ch01"
OUTPUT_DIR = ROOT / "outputs" / "ch01"
CHECKLIST = INPUT_DIR / "article_review_checklist.csv"
TAXONOMY = INPUT_DIR / "ooda_crisp_taxonomy.csv"
NIH_AI = INPUT_DIR / "nih_ai_checklist_tags.csv"
SNAPSHOT = OUTPUT_DIR / "study_chars_snapshot.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load sources ─────────────────────────────────────────────────────────────
with open(CHECKLIST, encoding="utf-8", errors="replace") as f:
    checklist = {r["article_id"].strip(): r for r in csv.DictReader(f)}

with open(TAXONOMY, encoding="utf-8", errors="replace") as f:
    taxonomy = list(csv.DictReader(f))

with open(NIH_AI, encoding="utf-8", errors="replace") as f:
    nih_rows = list(csv.DictReader(f))

# ── Core ML/XAI prediction subset: decide-phase included ─────────────────────
decide_inc = [
    r for r in taxonomy
    if r.get("ooda_phase", "").strip() == "decide"
    and r.get("human_decision", "").strip() == "include"
]
N = len(decide_inc)

# Join pub_year from checklist
def year_of(r):
    c = checklist.get(r.get("article_id", "").strip())
    return c.get("pub_year", "").strip() if c else ""

years = sorted(y for y in (year_of(r) for r in decide_inc) if y)

# ── Year bins ─────────────────────────────────────────────────────────────────
def bin_years(years, bins):
    """bins: list of (label, lo, hi) with lo/hi inclusive year strings."""
    result = {}
    for label, lo, hi in bins:
        n = sum(1 for y in years if lo <= y <= hi)
        pct = 100 * n // len(years) if years else 0
        result[label] = (n, pct)
    return result

YEAR_BINS = [
    ("2021–2022", "2021", "2022"),
    ("2023–2024", "2023", "2024"),
    ("2025–2026", "2025", "2026"),
]
year_counts = bin_years(years, YEAR_BINS)

# ── NIH AI score distribution ─────────────────────────────────────────────────
nih_inc = [r for r in nih_rows if r.get("human_decision", "").strip() == "include"]
score_cnt = collections.Counter(
    r.get("nih_ai_score", "").strip() for r in nih_inc
)
high_coverage = sum(v for k, v in score_cnt.items()
                    if k.isdigit() and int(k) >= 6)

# ── Print ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"TABLE 1 — Pipeline-derived statistics")
print(f"{'='*60}")
print(f"Decide-phase included studies (N): {N}")
print(f"Year range: {years[0]} – {years[-1]}")
print(f"\nPublication year bins:")
for label, (n, pct) in year_counts.items():
    bar = "#" * (n // 3)
    print(f"  {label}: {n:3d}  ({pct}%)  {bar}")

print(f"\nNIH AI included (ML/XAI breadth, N={len(nih_inc)}):")
print(f"  High-coverage (score ≥ 6): {high_coverage}")
for s in sorted(score_cnt, key=lambda x: int(x) if x.isdigit() else -1):
    print(f"  score={s}: {score_cnt[s]}")

# ── Notes on manual-coded fields ─────────────────────────────────────────────
print(f"\nNOTE: Study design, data source, sample size, and country require")
print(f"  manual extraction from full text — not auto-derived from this script.")
print(f"  Update the publication-facing study-characteristic table when manual review is refreshed.")

# ── Write snapshot CSV for diff tracking ─────────────────────────────────────
rows = [
    {"stat": "N_decide_included",  "value": N},
    {"stat": "year_min",           "value": years[0] if years else ""},
    {"stat": "year_max",           "value": years[-1] if years else ""},
]
for label, (n, pct) in year_counts.items():
    rows.append({"stat": f"year_bin_{label}", "value": f"{n} ({pct}%)"})
rows.append({"stat": "nih_ai_included", "value": len(nih_inc)})
rows.append({"stat": "high_coverage_ge6", "value": high_coverage})

with open(SNAPSHOT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["stat", "value"])
    w.writeheader()
    w.writerows(rows)

print(f"\nSnapshot written → {SNAPSHOT.name}")
