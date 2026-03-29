"""
Query Zotero SQLite for manuscript-relevant items.
Field IDs (confirmed from schema): title=1, date=6, url=13, publicationTitle=38, DOI=59
"""
import sqlite3, sys

db = r'C:\Users\jerom\Zotero\zotero.sqlite'
con = sqlite3.connect(db)
cur = con.cursor()

# Fetch all non-deleted journal/book items with their tags + first author
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

# Chapter-relevant tag sets
CHAPTER_TAGS = {
    'CH1_SQLR':       {'explainable_ai', 'target_leakage', 'pharmacovigilance', 'pgx-lit-review', 'observe', 'orient'},
    'CH3_opioid':     {'opioid_ed', 'temporal_analysis', 'claims_apcd', 'scalable_analytics'},
    'CH4_polyph':     {'polypharmacy_ed', 'association_rules', 'routine_care_utilization'},
    'CH5_deploy':     {'act', 'ehr_fhir', 'gradient_boosting', 'decide'},
    'CH6_conclusion': {'opioid_ed', 'polypharmacy_ed', 'explainable_ai', 'act'},
}
ALL_RELEVANT = set().union(*CHAPTER_TAGS.values())

results = []
for key, title, date, pub, doi, url, tags, auth in rows:
    tag_set = set(t.strip() for t in (tags or '').split(','))
    matched_chapters = [ch for ch, ctags in CHAPTER_TAGS.items() if ctags & tag_set]
    if matched_chapters and title:
        yr = (date or '')[:4]
        results.append({
            'key': key,
            'auth': auth or '?',
            'yr': yr,
            'title': title,
            'pub': pub or '',
            'doi': doi or url or '',
            'tags': sorted(tag_set & ALL_RELEVANT),
            'chapters': matched_chapters,
        })

# Group by chapter relevance
from collections import defaultdict
by_ch = defaultdict(list)
for r in results:
    for ch in r['chapters']:
        by_ch[ch].append(r)

for ch in sorted(CHAPTER_TAGS.keys()):
    items = sorted(by_ch[ch], key=lambda x: x['yr'], reverse=True)[:15]
    print(f"\n{'='*70}")
    print(f"  {ch}  ({len(by_ch[ch])} items, showing top 15 newest)")
    print(f"{'='*70}")
    for r in items:
        print(f"  {r['yr']}  {r['auth']:20s}  {r['title'][:70]}")
        print(f"         tags={r['tags']}")
        if r['doi']:
            print(f"         doi/url={r['doi'][:60]}")
        print()

print(f"\nTotal unique relevant items: {len(results)}")

