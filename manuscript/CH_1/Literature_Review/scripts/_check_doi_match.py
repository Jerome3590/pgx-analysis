import csv

sm   = list(csv.DictReader(open('scripts/screened_missing_fulltext.csv', encoding='utf-8')))
miss = list(csv.DictReader(open('scripts/missing_articles_combined.csv', encoding='utf-8')))
log  = list(csv.DictReader(open('scripts/unpaywall_log.csv', encoding='utf-8')))

def norm(s):
    return s.lower().strip()[:80]

miss_title_map = {norm(r.get('title', '')): r.get('pmc_id', '') for r in miss}
doi_map        = {r['hsh_id']: r['doi'] for r in log if r.get('doi', '').strip()}

title_in_miss = sum(1 for r in sm if norm(r.get('title', '')) in miss_title_map)

matched = []
for r in sm:
    t   = norm(r.get('title', ''))
    mid = miss_title_map.get(t, '')
    doi = doi_map.get(mid, '')
    if doi:
        matched.append({
            'screened_pmc_id': mid,   # miss_pmc_id is unique per article/DOI
            'miss_pmc_id':     mid,
            'doi':             doi,
            'title':           r.get('title', '')[:80],
        })

# Deduplicate: keep first occurrence of each DOI
seen_doi = set()
deduped  = []
for m in matched:
    if m['doi'] not in seen_doi:
        seen_doi.add(m['doi'])
        deduped.append(m)
matched = deduped

print(f"screened_missing rows    : {len(sm)}")
print(f"title matched to missing : {title_in_miss}")
print(f"matched with DOI         : {len(matched)}")
if matched:
    print("sample:", matched[0])

# Save the matched DOI list for vcu_download.js
with open('scripts/screened_doi_map.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['screened_pmc_id', 'doi', 'title'])
    w.writeheader()
    for m in matched:
        w.writerow({'screened_pmc_id': m['screened_pmc_id'],
                    'doi': m['doi'], 'title': m['title']})
print("Saved: scripts/screened_doi_map.csv")
