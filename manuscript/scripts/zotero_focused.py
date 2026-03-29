"""
Focused Zotero query: find high-quality, directly citable items per chapter
using SPECIFIC tags only (not broad act/observe/orient/pgx-lit-review).
"""
import sqlite3
from collections import defaultdict

db = r'C:\Users\jerom\Zotero\zotero.sqlite'
con = sqlite3.connect(db)
cur = con.cursor()

cur.execute('''
SELECT
    i.key,
    MAX(CASE WHEN idf.fieldID=1  THEN idv.value END) as title,
    MAX(CASE WHEN idf.fieldID=6  THEN idv.value END) as date,
    MAX(CASE WHEN idf.fieldID=38 THEN idv.value END) as publication,
    MAX(CASE WHEN idf.fieldID=59 THEN idv.value END) as doi,
    MAX(CASE WHEN idf.fieldID=13 THEN idv.value END) as url,
    GROUP_CONCAT(DISTINCT t.name) as tags,
    MIN(cr.lastName) as first_author
FROM items i
JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
LEFT JOIN itemData idf ON i.itemID = idf.itemID
LEFT JOIN itemDataValues idv ON idf.valueID = idv.valueID
LEFT JOIN itemTags itg ON i.itemID = itg.itemID
LEFT JOIN tags t ON itg.tagID = t.tagID
LEFT JOIN itemCreators ic ON i.itemID = ic.itemID AND ic.orderIndex=0
LEFT JOIN creators cr ON ic.creatorID = cr.creatorID
WHERE it.typeName IN ("journalArticle","book","bookSection","report","preprint","conferencePaper")
  AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
GROUP BY i.key
HAVING title IS NOT NULL
ORDER BY date DESC
''')

rows = cur.fetchall()
con.close()

# Specific tags per chapter — must match AT LEAST ONE of these
SPECIFIC = {
    'CH1_SQLR':   {'explainable_ai', 'target_leakage', 'temporal_analysis',
                   'gradient_boosting', 'claims_apcd', 'scalable_analytics'},
    'CH2_arch':   {'claims_apcd', 'scalable_analytics', 'target_leakage',
                   'gradient_boosting', 'temporal_analysis'},
    'CH3_opioid': {'opioid_ed', 'temporal_analysis', 'claims_apcd',
                   'scalable_analytics', 'process_mining'},
    'CH4_polyph': {'polypharmacy_ed', 'association_rules',
                   'routine_care_utilization', 'pharmacovigilance'},
    'CH5_deploy': {'explainable_ai', 'ehr_fhir', 'gradient_boosting',
                   'decide', 'scalable_analytics'},
}

ALL_SPECIFIC = set().union(*SPECIFIC.values())

results = []
for key, title, date, pub, doi, url, tags, auth in rows:
    tag_set = set(t.strip() for t in (tags or '').split(','))
    matched_specific = tag_set & ALL_SPECIFIC
    if not matched_specific or not title:
        continue
    chapters = [ch for ch, ctags in SPECIFIC.items() if ctags & tag_set]
    yr = (date or '')[:4]
    results.append({
        'key': key, 'auth': auth or '?', 'yr': yr,
        'title': title, 'pub': pub or '',
        'ref': doi or url or '',
        'tags': sorted(matched_specific),
        'chapters': chapters,
    })

by_ch = defaultdict(list)
for r in results:
    for ch in r['chapters']:
        by_ch[ch].append(r)

already_cited = {
    'Lundberg2017','Lundberg2020','Chen2016','Prokhorenkova2018',
    'Akiba2019','Crews2021','Han2000','Becker2007','Berndt1994',
    'Wolff2019','Kapoor2023','HHS2013','Char2018','Topol2019',
    'Rajpurkar2022','FDA2021','Maher2014','Kertesz2017',
}

for ch in sorted(SPECIFIC.keys()):
    items = sorted(by_ch[ch], key=lambda x: x['yr'], reverse=True)
    print(f"\n{'='*72}")
    print(f"  {ch}  ({len(items)} specific matches)")
    print(f"{'='*72}")
    for r in items[:20]:
        print(f"  {r['yr']}  {r['auth'][:18]:18s}  {r['title'][:68]}")
        print(f"         tags={r['tags']}")
        if r['ref']:
            print(f"         {r['ref'][:65]}")
        print()

print(f"\nTotal unique specific items across all chapters: {len(results)}")
