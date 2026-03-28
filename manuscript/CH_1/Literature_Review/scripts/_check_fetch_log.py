import csv
from pathlib import Path

screened = {r["article_id"]: r
            for r in csv.DictReader(open("data/ontology/articles_screened.csv", encoding="utf-8-sig"))}

rows = list(csv.DictReader(open("scripts/fulltext_fetch_log.csv", encoding="utf-8-sig")))
nf   = [r for r in rows if r["status"] == "not_found"]

real_pmc = [r for r in nf if r["pmc_id"].startswith("PMC")]
hsh_ids  = [r for r in nf if r["pmc_id"].startswith("HSH")]
other    = [r for r in nf if not r["pmc_id"].startswith("PMC") and not r["pmc_id"].startswith("HSH")]

print(f"Total not_found : {len(nf)}")
print(f"  Real PMC IDs (paywalled PMC OA)  : {len(real_pmc)}")
print(f"  HSH IDs (should use PDF extract) : {len(hsh_ids)}")
print(f"  Other                            : {len(other)}")
print()
print("Real PMC not-found (need VCU library):")
for r in real_pmc[:20]:
    sr = screened.get(r["article_id"], {})
    title = (sr.get("title", "") or "")[:65]
    dec   = sr.get("human_decision", "?")
    score = sr.get("composite_score", "?")
    print(f"  [{dec}] {r['pmc_id']}  score={score}  {title}")

