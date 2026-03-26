import csv
from pathlib import Path

root     = Path("data")
screened = list(csv.DictReader(open("data/ontology/articles_screened.csv", encoding="utf-8")))
includes = [r for r in screened if r.get("include_recommended") == "include"]
print(f"Pre-screened includes: {len(includes)}")

json_count = 0
no_json    = []

for row in includes:
    pmc_id   = row.get("pmc_id", "").strip()
    has_json = False
    if pmc_id and not pmc_id.startswith("HSH"):
        for p in root.rglob(f"{pmc_id}.json"):
            if p.stat().st_size > 500:
                has_json = True
                break
        if not has_json:
            for p in root.rglob(f"PMC{pmc_id}.json"):
                if p.stat().st_size > 500:
                    has_json = True
                    break
    if has_json:
        json_count += 1
    else:
        no_json.append(row)

hsh_missing = [r for r in no_json if r.get("pmc_id", "").startswith("HSH")]
pmc_missing = [r for r in no_json if not r.get("pmc_id", "").startswith("HSH")]

print(f"Have full-text JSON: {json_count}")
print(f"Missing full-text:   {len(no_json)}")
print(f"  HSH (no PMC ID):   {len(hsh_missing)}")
print(f"  PMC (download gap):{len(pmc_missing)}")

# Save the HSH-missing list for scholar lookup
with open("scripts/screened_missing_fulltext.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(includes[0].keys()))
    w.writeheader()
    w.writerows(hsh_missing)
print("Saved: scripts/screened_missing_fulltext.csv")
