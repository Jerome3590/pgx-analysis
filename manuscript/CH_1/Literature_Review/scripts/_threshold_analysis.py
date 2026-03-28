"""
Threshold analysis — show include/exclude counts at multiple pytextrank cutoffs.
Reads pytextrank_score and combined_score from the checklist CSV (article_review_checklist.csv)
since those are the scores _phase7_review.py actually uses, not the original composite_score.

Also loads articles_screened.csv to compute full universe counts.

python scripts/_threshold_analysis.py
"""
import csv
from collections import Counter
from pathlib import Path

CHECKLIST = Path(r"C:\Projects\pgx-analysis\manuscript\infrastructure_setup\manual_review\article_review_checklist.csv")
SCREENED  = Path("data/ontology/articles_screened.csv")

# ── Load checklist (included articles with pytextrank scores) ─────────────────
ck_rows = list(csv.DictReader(open(CHECKLIST, encoding="utf-8-sig")))
print(f"Checklist columns : {list(ck_rows[0].keys())}")
print(f"Checklist rows    : {len(ck_rows)}  (human_decision=include)")

def get_score(row):
    for col in ("combined_score", "pytextrank_score", "composite_score"):
        v = row.get(col, "")
        if v not in (None, ""):
            try: return float(v)
            except: pass
    return 0.0

scores = [(r, get_score(r)) for r in ck_rows]

# ── Load all screened articles for universe context ───────────────────────────
sc_rows = list(csv.DictReader(open(SCREENED, encoding="utf-8-sig")))
total_screened = len(sc_rows)
algo_inc_ids = {r["article_id"] for r in sc_rows if r.get("include_recommended") == "include"}

print(f"\nTotal screened    : {total_screened:,}")
print(f"Algo-recommended  : {len(algo_inc_ids):,}")
print(f"Currently included (human_decision=include): {len(ck_rows):,}")

# ── Score distribution ────────────────────────────────────────────────────────
print(f"\nPytextrank combined_score distribution (4,946 included articles):")
buckets = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.01]
for lo, hi in zip(buckets, buckets[1:]):
    n = sum(1 for _, s in scores if lo <= s < hi)
    bar = "█" * (n // 50)
    print(f"  [{lo:.2f},{hi:.2f}) : {n:5d}  {bar}")

# ── Threshold sweep ───────────────────────────────────────────────────────────
print(f"\n{'Threshold':>10}  {'Include':>8}  {'Excluded from 4946':>19}  {'% of screened':>14}  {'Note'}")
print("-" * 80)
thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
for t in thresholds:
    above = [r for r, s in scores if s >= t]
    below = len(scores) - len(above)
    pct   = len(above) / total_screened * 100
    note  = ""
    if t == 0.20:
        note = "← current phase7 threshold"
    print(f"  {t:>8.2f}  {len(above):>8,}  {below:>14,} dropped  {pct:>12.1f}%  {note}")

# ── What gets dropped at each threshold (sample titles) ──────────────────────
print(f"\nSample articles that would be DROPPED at threshold 0.25 (score 0.20–0.25):")
borderline = [(r, s) for r, s in scores if 0.20 <= s < 0.25]
for r, s in sorted(borderline, key=lambda x: -x[1])[:10]:
    print(f"  [{s:.3f}]  {r.get('title','')[:75]}")

print(f"\nSample articles that would be DROPPED at threshold 0.30 (score 0.25–0.30):")
borderline2 = [(r, s) for r, s in scores if 0.25 <= s < 0.30]
for r, s in sorted(borderline2, key=lambda x: -x[1])[:10]:
    print(f"  [{s:.3f}]  {r.get('title','')[:75]}")

print(f"\nRecommendation:")
print(f"  0.20 → {sum(1 for _,s in scores if s>=0.20):,} articles  (current — comprehensive)")
print(f"  0.25 → {sum(1 for _,s in scores if s>=0.25):,} articles  (stricter)")
print(f"  0.30 → {sum(1 for _,s in scores if s>=0.30):,} articles  (tight — clearly on-topic)")
print(f"\n  More full text acquired → scores improve → raise threshold AFTER VCU downloads.")
