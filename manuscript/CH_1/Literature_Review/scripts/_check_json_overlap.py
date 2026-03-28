"""Understand article_id → PMC JSON mapping and which PDFs already have JSON."""
import csv, json
from pathlib import Path

data     = Path("data")
pdf_dir  = Path("data/scholar_pdfs")
doi_map  = {r["screened_pmc_id"]: r
            for r in csv.DictReader(open("scripts/screened_doi_map.csv", encoding="utf-8-sig"))}

# Which 10 PDF stems match json stems?
json_stems = {p.stem: p for p in data.rglob("*.json")}
pdf_stems  = {p.stem: p for p in pdf_dir.glob("*.pdf")}
overlap    = set(pdf_stems) & set(json_stems)
print(f"PDF stems with a matching JSON file ({len(overlap)}):")
for s in sorted(overlap):
    print(f"  {s}  →  {json_stems[s].relative_to(data)}")

# Inspect PMC JSON structure more fully
sample_pmc = next(data.rglob("*.json"))
with open(sample_pmc, encoding="utf-8") as f:
    obj = json.load(f)
doc = obj[0] if isinstance(obj, list) else obj
print(f"\nBioC JSON top-level keys: {list(doc.keys())}")
if "documents" in doc:
    d0 = doc["documents"][0] if doc["documents"] else {}
    print(f"documents[0] keys: {list(d0.keys())}")
    if "passages" in d0:
        p0 = d0["passages"][0] if d0["passages"] else {}
        print(f"passages[0] keys: {list(p0.keys())}")
        if "infons" in p0:
            print(f"passages[0].infons: {p0['infons']}")
        if "text" in p0:
            print(f"passages[0].text[:200]: {p0['text'][:200]}")

# Check articles_screened pmc_id + source_file columns
screened = list(csv.DictReader(open("data/ontology/articles_screened.csv", encoding="utf-8-sig")))
inc = [r for r in screened if r.get("human_decision") == "include"]
print(f"\nIncluded articles: {len(inc)}")
print(f"Sample pmc_ids: {[r.get('pmc_id','') for r in inc[:5]]}")
print(f"Sample source_files: {[r.get('source_file','') for r in inc[:3]]}")

# How many included have pmc_id?
with_pmc = [r for r in inc if r.get("pmc_id","").strip()]
print(f"Included with pmc_id: {len(with_pmc)}")

# Check if pmc_id JSONs exist
found = 0
for r in with_pmc[:20]:
    pmc = r["pmc_id"].strip()
    matches = [p for p in data.rglob(f"{pmc}.json")]
    if matches:
        found += 1
print(f"First 20 pmc_ids: {found} found as JSON files")
