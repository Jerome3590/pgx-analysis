"""
Curated Zotero shortlist: items with DOI/PMCID, 2018+, and HIGHEST-SPECIFICITY tags.
Priority tags (items MUST have at least one):
  explainable_ai, target_leakage, temporal_analysis, claims_apcd,
  gradient_boosting+decide combo, pharmacovigilance, routine_care_utilization
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
    MIN(cr.lastName) as last,
    MIN(cr.firstName) as first
FROM items i
JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
LEFT JOIN itemData idf ON i.itemID = idf.itemID
LEFT JOIN itemDataValues idv ON idf.valueID = idv.valueID
LEFT JOIN itemTags itg ON i.itemID = itg.itemID
LEFT JOIN tags t ON itg.tagID = t.tagID
LEFT JOIN itemCreators ic ON i.itemID = ic.itemID AND ic.orderIndex=0
LEFT JOIN creators cr ON ic.creatorID = cr.creatorID
WHERE it.typeName IN ("journalArticle","book","preprint","conferencePaper","report")
  AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
GROUP BY i.key
HAVING title IS NOT NULL
ORDER BY date DESC
''')
rows = cur.fetchall()
con.close()

# Only the most specific, directly citable tags
PRIORITY_TAGS = {
    'explainable_ai', 'target_leakage', 'temporal_analysis',
    'claims_apcd', 'routine_care_utilization', 'process_mining',
    'pharmacovigilance', 'cpt_icd_codes', 'ehr_fhir',
}

# Chapter mapping using priority tags
CH_MAP = {
    'CH1_SQLR':   {'explainable_ai', 'target_leakage', 'temporal_analysis', 'pharmacovigilance'},
    'CH2_arch':   {'claims_apcd', 'target_leakage', 'temporal_analysis', 'scalable_analytics'},
    'CH3_opioid': {'opioid_ed', 'temporal_analysis', 'claims_apcd', 'process_mining'},
    'CH4_polyph': {'polypharmacy_ed', 'pharmacovigilance', 'routine_care_utilization'},
    'CH5_deploy': {'explainable_ai', 'ehr_fhir', 'decide'},
}

already_in_bib = {
    'Lundberg2017', 'Lundberg2020', 'Chen2016', 'Prokhorenkova2018',
    'Akiba2019', 'Crews2021', 'Han2000', 'Becker2007', 'Berndt1994',
    'Wolff2019', 'Kapoor2023', 'HHS2013', 'Char2018', 'Topol2019',
    'Rajpurkar2022', 'FDA2021', 'Maher2014', 'Kertesz2017',
}

results = []
for key, title, date, pub, doi, url, tags, last, first in rows:
    tag_set = set(t.strip() for t in (tags or '').split(','))
    priority_matched = tag_set & PRIORITY_TAGS
    if not priority_matched or not title:
        continue
    yr = (date or '')[:4]
    if yr and int(yr) < 2018:
        continue  # skip older items
    ref = doi or url or ''
    if not ref:
        continue  # skip items without URL/DOI
    chapters = [ch for ch, ctags in CH_MAP.items() if ctags & tag_set]
    if not chapters:
        continue
    results.append({
        'key': key, 'last': last or '?', 'first': (first or '')[:1],
        'yr': yr, 'title': title, 'pub': pub or '',
        'doi': doi or '', 'url': url or '',
        'tags': sorted(priority_matched),
        'chapters': chapters,
    })

by_ch = defaultdict(list)
for r in results:
    for ch in r['chapters']:
        by_ch[ch].append(r)

seen = set()
for ch in sorted(CH_MAP.keys()):
    items = sorted(by_ch[ch], key=lambda x: x['yr'], reverse=True)
    print(f"\n{'='*74}")
    print(f"  {ch}  ({len(items)} targeted items with DOI/URL, 2018+)")
    print(f"{'='*74}")
    count = 0
    for r in items:
        uid = r['last'] + r['yr']
        if uid in seen:
            continue
        seen.add(uid)
        auth = f"{r['last']} {r['yr']}"
        ref = r['doi'] or r['url']
        print(f"  [{auth}]")
        print(f"    Title: {r['title'][:80]}")
        print(f"    Pub:   {r['pub'][:60]}")
        print(f"    Tags:  {r['tags']}")
        print(f"    Ref:   {ref[:70]}")
        print()
        count += 1
        if count >= 12:
            break

print(f"\nTotal curated items (2018+, with DOI, priority tags): {len(results)}")
