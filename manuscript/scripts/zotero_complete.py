"""
Find Zotero items that have COMPLETE citation data (title + journal + date + doi/url)
and match chapter-relevant tag intersections. These are directly usable for bib entries.
"""
import sqlite3

db = r'C:\Users\jerom\Zotero\zotero.sqlite'
con = sqlite3.connect(db)
cur = con.cursor()

# Also fetch volume, issue, pages
cur.execute('''
SELECT
    i.key,
    MAX(CASE WHEN idf.fieldID=1  THEN idv.value END) as title,
    MAX(CASE WHEN idf.fieldID=6  THEN idv.value END) as date,
    MAX(CASE WHEN idf.fieldID=38 THEN idv.value END) as journal,
    MAX(CASE WHEN idf.fieldID=59 THEN idv.value END) as doi,
    MAX(CASE WHEN idf.fieldID=13 THEN idv.value END) as url,
    MAX(CASE WHEN idf.fieldID=10 THEN idv.value END) as volume,
    MAX(CASE WHEN idf.fieldID=11 THEN idv.value END) as issue,
    MAX(CASE WHEN idf.fieldID=14 THEN idv.value END) as pages,
    GROUP_CONCAT(DISTINCT t.name) as tags,
    GROUP_CONCAT(DISTINCT cr.lastName) as authors
FROM items i
JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
LEFT JOIN itemData idf ON i.itemID = idf.itemID
LEFT JOIN itemDataValues idv ON idf.valueID = idv.valueID
LEFT JOIN itemTags itg ON i.itemID = itg.itemID
LEFT JOIN tags t ON itg.tagID = t.tagID
LEFT JOIN itemCreators ic ON i.itemID = ic.itemID
LEFT JOIN creators cr ON ic.creatorID = cr.creatorID
WHERE it.typeName IN ("journalArticle","book","preprint","conferencePaper","report")
  AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
GROUP BY i.key
HAVING title IS NOT NULL AND journal IS NOT NULL
ORDER BY date DESC
''')
rows = cur.fetchall()
con.close()

print(f"Total items with journal name: {len(rows)}\n")

# Target tag sets per chapter section
TARGETS = {
    'CH1_XAI_leakage':        {'explainable_ai', 'target_leakage'},
    'CH1_XAI_opioid':         {'explainable_ai', 'opioid_ed'},
    'CH3_opioid_temporal':    {'opioid_ed', 'temporal_analysis'},
    'CH3_opioid_claims':      {'opioid_ed', 'claims_apcd'},
    'CH4_polyph_pharmavig':   {'polypharmacy_ed', 'pharmacovigilance'},
    'CH4_polyph_routine':     {'polypharmacy_ed', 'routine_care_utilization'},
    'CH5_xai_decide':         {'explainable_ai', 'decide'},
    'CH5_xai_ehr':            {'explainable_ai', 'ehr_fhir'},
    # Single-tag fallback for under-populated combos
    'CH1_target_leakage':     {'target_leakage'},
    'CH3_opioid_ed_complete': {'opioid_ed'},
    'CH4_polyph_complete':    {'polypharmacy_ed'},
}

seen = set()
for combo_name, req_tags in TARGETS.items():
    matches = []
    for key, title, date, journal, doi, url, vol, issue, pages, tags, authors in rows:
        if key in seen:
            continue
        tag_set = set(t.strip() for t in (tags or '').split(','))
        if req_tags.issubset(tag_set):
            yr = (date or '')[:4]
            if yr and int(yr) < 2019:
                continue
            ref = doi or url or ''
            auth_list = (authors or '').split(',')
            first_auth = auth_list[0].strip() if auth_list else '?'
            matches.append({
                'key': key, 'auth': first_auth, 'yr': yr,
                'title': title, 'journal': journal,
                'doi': doi or '', 'url': url or '',
                'vol': vol or '', 'issue': issue or '', 'pages': pages or '',
                'tags': sorted(tag_set & set().union(*TARGETS.values())),
            })

    if not matches:
        continue

    print(f"\n{'='*76}")
    print(f"  {combo_name}  ({len(matches)} complete items, 2019+)")
    print(f"{'='*76}")
    for r in sorted(matches, key=lambda x: x['yr'], reverse=True)[:10]:
        seen.add(r['key'])
        auth = r['auth']
        ref = r['doi'] or r['url']
        vi = f"vol={r['vol']} iss={r['issue']} pp={r['pages']}" if r['vol'] else ''
        print(f"  {r['yr']}  [{auth}]  {r['title'][:65]}")
        print(f"         journal: {r['journal'][:60]}")
        if vi:
            print(f"         {vi}")
        print(f"         doi/url: {ref[:65]}")
        print(f"         tags: {r['tags']}")
        print()
