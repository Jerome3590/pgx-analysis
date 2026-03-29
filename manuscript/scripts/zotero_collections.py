"""
Query Zotero by collection (curated items) and also get ALL items with
full journal data (volume/issue/pages/doi) regardless of tags.
"""
import sqlite3

db = r'C:\Users\jerom\Zotero\zotero.sqlite'
con = sqlite3.connect(db)
cur = con.cursor()

# 1. Get all collections
print("=== COLLECTIONS ===")
cur.execute("""
SELECT c.key, c.collectionName,
       (SELECT COUNT(*) FROM collectionItems ci WHERE ci.collectionID=c.collectionID) as cnt
FROM collections c
ORDER BY c.collectionName
""")
for row in cur.fetchall():
    print(f"  {row[0]:10s}  {row[2]:4d}  {row[1]}")

# 2. Get all items with COMPLETE metadata (journal + doi + date)
print("\n\n=== COMPLETE ITEMS (journal + doi + year, sorted newest first) ===")
cur.execute('''
SELECT
    i.key,
    MAX(CASE WHEN idf.fieldID=1  THEN idv.value END) as title,
    MAX(CASE WHEN idf.fieldID=6  THEN idv.value END) as date,
    MAX(CASE WHEN idf.fieldID=38 THEN idv.value END) as journal,
    MAX(CASE WHEN idf.fieldID=59 THEN idv.value END) as doi,
    MAX(CASE WHEN idf.fieldID=10 THEN idv.value END) as volume,
    MAX(CASE WHEN idf.fieldID=14 THEN idv.value END) as pages,
    GROUP_CONCAT(DISTINCT cr.lastName) as authors,
    GROUP_CONCAT(DISTINCT t.name) as tags
FROM items i
JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
LEFT JOIN itemData idf ON i.itemID = idf.itemID
LEFT JOIN itemDataValues idv ON idf.valueID = idv.valueID
LEFT JOIN itemTags itg ON i.itemID = itg.itemID
LEFT JOIN tags t ON itg.tagID = t.tagID
LEFT JOIN itemCreators ic ON i.itemID = ic.itemID
LEFT JOIN creators cr ON ic.creatorID = cr.creatorID
WHERE it.typeName IN ("journalArticle","preprint","conferencePaper","report","book")
  AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
GROUP BY i.key
HAVING title IS NOT NULL AND journal IS NOT NULL AND doi IS NOT NULL
ORDER BY date DESC
''')
rows = cur.fetchall()
con.close()

# Keywords relevant to our chapters
KEYWORDS = [
    # CH1 - SQLR, XAI, leakage
    'explainable', 'shapley', 'shap', 'leakage', 'temporal validation',
    'literature review', 'systematic review',
    # CH2 - architecture, APCD, ensemble
    'all-payer', 'apcd', 'claims', 'partition', 'catboost', 'xgboost',
    'ensemble', 'imbalanced',
    # CH3 - opioid ED
    'opioid', 'opioid use disorder', 'oud', 'drug trajectory', 'dtw',
    'dynamic time warping', 'pharmacogenomic', 'cyp2d6', 'gabapentin',
    # CH4 - polypharmacy geriatric
    'polypharmacy', 'deprescribing', 'drug-drug interaction', 'beers criteria',
    'stopp', 'geriatric', 'adverse drug', 'z-code',
    # CH5 - deployment, serverless, clinical decision support
    'clinical decision support', 'serverless', 'lambda', 'dashboard',
    'precision medicine', 'federated',
]

CHAPTER_KW = {
    'CH1': ['explainable', 'shapley', 'shap', 'leakage', 'systematic review', 'literature review', 'pharmacovigilance'],
    'CH3': ['opioid', 'oud', 'gabapentin', 'cyp2d6', 'dynamic time warping', 'dtw', 'pharmacogenomic'],
    'CH4': ['polypharmacy', 'deprescribing', 'drug-drug interaction', 'beers', 'stopp', 'geriatric', 'adverse drug'],
    'CH5': ['clinical decision support', 'dashboard', 'precision medicine', 'federated', 'serverless'],
}

by_ch = {ch: [] for ch in CHAPTER_KW}
for key, title, date, journal, doi, vol, pages, authors, tags in rows:
    t_lower = (title or '').lower()
    yr = (date or '')[:4]
    auth = (authors or '').split(',')[0].strip()
    for ch, kws in CHAPTER_KW.items():
        if any(kw in t_lower for kw in kws):
            by_ch[ch].append((yr, auth, title, journal, doi, vol, pages))
            break  # only assign to first matching chapter

for ch, items in by_ch.items():
    items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
    print(f"\n{'='*76}")
    print(f"  {ch}  ({len(items_sorted)} complete citable items)")
    print(f"{'='*76}")
    for yr, auth, title, journal, doi, vol, pages in items_sorted[:15]:
        vp = f"vol={vol} pp={pages}" if vol else ''
        print(f"  {yr}  [{auth}]  {title[:65]}")
        print(f"         {journal[:55]}  {vp}")
        print(f"         doi={doi[:60]}")
        print()
