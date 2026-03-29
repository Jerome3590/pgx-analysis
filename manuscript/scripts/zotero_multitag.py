"""
Multi-tag intersection query: items must match 2+ specific tags simultaneously.
This filters out broad single-tag noise.
"""
import sqlite3

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
    MIN(cr.lastName) as last
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

# Multi-tag combos per chapter — item must match ALL tags in at least one combo
COMBOS = {
    'CH1 — XAI + temporal leakage':          {'explainable_ai', 'target_leakage'},
    'CH1 — XAI + opioid/polyph context':     {'explainable_ai', 'opioid_ed'},
    'CH1 — XAI + EHR/claims':                {'explainable_ai', 'claims_apcd'},
    'CH2/CH3 — claims + temporal':           {'claims_apcd', 'temporal_analysis'},
    'CH3 — opioid + trajectory/temporal':    {'opioid_ed', 'temporal_analysis'},
    'CH3 — opioid + process mining':         {'opioid_ed', 'process_mining'},
    'CH4 — polyph + pharmacovigilance':      {'polypharmacy_ed', 'pharmacovigilance'},
    'CH4 — polyph + routine care':           {'polypharmacy_ed', 'routine_care_utilization'},
    'CH4 — polyph + association rules':      {'polypharmacy_ed', 'association_rules'},
    'CH5 — XAI + decide (deployed)':         {'explainable_ai', 'decide'},
    'CH5 — XAI + EHR/FHIR':                 {'explainable_ai', 'ehr_fhir'},
    'CH5 — gradient boost + XAI':           {'gradient_boosting', 'explainable_ai'},
}

results_by_combo = {}
for combo_name, required_tags in COMBOS.items():
    matches = []
    for key, title, date, pub, doi, url, tags, last in rows:
        tag_set = set(t.strip() for t in (tags or '').split(','))
        if required_tags.issubset(tag_set):
            yr = (date or '')[:4]
            ref = doi or url or ''
            matches.append({
                'key': key, 'last': last or '?', 'yr': yr,
                'title': title or '',
                'pub': pub or '',
                'ref': ref,
                'tags': sorted(tag_set & set().union(*COMBOS.values())),
            })
    results_by_combo[combo_name] = sorted(matches, key=lambda x: x['yr'], reverse=True)

seen_keys = set()
for combo, items in results_by_combo.items():
    new_items = [r for r in items if r['key'] not in seen_keys]
    if not new_items:
        continue
    print(f"\n{'='*74}")
    print(f"  {combo}  ({len(new_items)} items)")
    print(f"{'='*74}")
    for r in new_items[:8]:
        seen_keys.add(r['key'])
        print(f"  {r['yr']}  {r['last'][:20]:20s}  {r['title'][:65]}")
        print(f"         pub={r['pub'][:50]}")
        print(f"         ref={r['ref'][:65]}")
        print()
