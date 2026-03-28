import csv, json
from pathlib import Path

# ── dir structure ──────────────────────────────────────────────────────────────
data = Path("data")
for d in sorted(data.iterdir()):
    if not d.is_dir():
        continue
    all_files = list(d.rglob("*.*"))
    jsons = [f for f in all_files if f.suffix == ".json"]
    subdirs = [x for x in d.iterdir() if x.is_dir()]
    print(f"{d.name}/  files={len(all_files)}  json={len(jsons)}  subdirs={len(subdirs)}")
    for sd in subdirs[:3]:
        sj = list(sd.glob("*.json"))
        print(f"  └─ {sd.name}/  json={len(sj)}")

# ── find any JSON file and inspect structure ───────────────────────────────────
all_jsons = list(data.rglob("*.json"))
if all_jsons:
    sample = all_jsons[0]
    print(f"\nSample JSON: {sample.relative_to(data)}")
    with open(sample, encoding="utf-8") as f:
        obj = json.load(f)
    item = obj[0] if isinstance(obj, list) else obj
    if isinstance(item, dict):
        print(f"Keys: {list(item.keys())[:12]}")
        for k in list(item.keys())[:4]:
            print(f"  {k}: {str(item[k])[:100]}")

# ── doi_map hsh_ids ────────────────────────────────────────────────────────────
doi_rows = list(csv.DictReader(open("scripts/screened_doi_map.csv", encoding="utf-8-sig")))
hsh_ids = {r["screened_pmc_id"]: r for r in doi_rows}
print(f"\ndoi_map entries: {len(hsh_ids)}")
print(f"Sample hsh_ids: {list(hsh_ids.keys())[:3]}")

# ── articles_screened article_ids ─────────────────────────────────────────────
screened = list(csv.DictReader(open("data/ontology/articles_screened.csv", encoding="utf-8-sig")))
inc_ids = {r["article_id"]: r for r in screened if r.get("human_decision") == "include"}
print(f"\nIncluded article_ids: {len(inc_ids)}")
print(f"Sample: {list(inc_ids.keys())[:3]}")

# ── check what JSON files exist by article_id ─────────────────────────────────
json_stems = {p.stem for p in data.rglob("*.json")}
print(f"\nTotal JSON stems across data/: {len(json_stems)}")
overlap_inc = set(inc_ids.keys()) & json_stems
print(f"Included IDs with existing JSON : {len(overlap_inc)}")
print(f"Included IDs without JSON       : {len(set(inc_ids.keys()) - json_stems)}")

# PDFs
pdf_stems = {p.stem for p in Path("data/scholar_pdfs").glob("*.pdf")}
pdf_with_json = pdf_stems & json_stems
print(f"\nPDF articles ({len(pdf_stems)}) that already have JSON: {len(pdf_with_json)}")
print(f"PDF articles needing JSON from extraction          : {len(pdf_stems - json_stems)}")
