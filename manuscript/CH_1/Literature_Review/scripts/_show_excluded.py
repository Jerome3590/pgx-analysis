"""Show a sample of excluded articles to help calibrate the threshold."""
import csv, random
from pathlib import Path
from collections import Counter

SCREENED = Path("data/ontology/articles_screened.csv")
rows = list(csv.DictReader(open(SCREENED, encoding="utf-8-sig")))

excluded = [r for r in rows if r.get("human_decision") == "exclude"]
included = [r for r in rows if r.get("human_decision") == "include"]

print(f"Total screened : {len(rows)}")
print(f"Include        : {len(included)}")
print(f"Exclude        : {len(excluded)}")
print()

# Score distribution of excluded articles
scores = []
for r in excluded:
    try:
        scores.append(float(r.get("composite_score", 0) or 0))
    except:
        scores.append(0.0)

print(f"Excluded composite_score distribution:")
buckets = [(0,0.01),(0.01,0.05),(0.05,0.10),(0.10,0.20),(0.20,1.0)]
for lo, hi in buckets:
    n = sum(1 for s in scores if lo <= s < hi)
    print(f"  [{lo:.2f}, {hi:.2f}) : {n:5d}")
print()

# Top domain/topic breakdown
domains  = Counter(r.get("domain_primary","(none)") for r in excluded)
print("Top domains in excluded articles:")
for d, n in domains.most_common(10):
    print(f"  {d:<40} {n}")
print()

# Sample 20 random excluded articles (sorted by composite_score desc to see borderline cases)
borderline = sorted(excluded,
                    key=lambda r: float(r.get("composite_score",0) or 0),
                    reverse=True)[:30]
print(f"Top 30 excluded by composite_score (most borderline):")
print(f"{'composite':>10}  {'title'}")
print("-" * 80)
for r in borderline:
    score = r.get("composite_score","")
    title = (r.get("title","") or "")[:75]
    print(f"  {score:>8}  {title}")
