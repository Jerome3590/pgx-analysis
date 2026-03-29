"""
Extract full bib data for selected high-value Zotero items.
Target DOIs identified from previous queries.
"""
import sqlite3

db = r'C:\Users\jerom\Zotero\zotero.sqlite'
con = sqlite3.connect(db)
cur = con.cursor()

# fieldID map (confirmed from schema inspection)
FIELDS = {1: 'title', 6: 'date', 38: 'journal', 59: 'doi', 13: 'url',
          10: 'volume', 11: 'issue', 14: 'pages', 84: 'abstractNote',
          4: 'publisher', 3: 'place'}

TARGET_DOIS = [
    # CH4 - polypharmacy DDI geriatric
    '10.1016/S0140-6736(22)01841-4',   # Pirmohamed 2023 - 12-gene PGx panel, Lancet
    '10.1111/bcp.15882',               # Klopotowska 2024 - high-risk DDIs, Br J Clin Pharm
    '10.1002/cpt.2813',                # Fromm 2023 - contraindicated drug combos, CPT
    '10.1136/bmjopen-2021-055551',     # Osanlou 2022 - ADRs + polypharmacy, BMJ Open
    '10.1093/ehjcvp/pvac005',          # Tamargo 2022 - polypharmacy older patients, Eur Heart J
    '10.1016/j.semarthrit.2024.152469', # Boukhlal 2024 - polypharmacy + DDI
    # CH5 - deployment + federated
    '10.1145/3533708',                 # Joshi 2022 - federated learning healthcare, ACM THCS
    '10.1002/cpt.3153',                # Terranova 2024 - ML disease trajectory, CPT
    # CH1/CH2 - temporal validation
    '10.1136/bmj.m958',                # Mahmoudi 2020 - EHR temporal validation, BMJ
    # CH3 - opioid prediction
    '10.1016/j.addbeh.2026.108652',    # Ehmann 2026 - psychedelics + OUD
]

cur.execute('''
SELECT
    i.key,
    MAX(CASE WHEN idf.fieldID=1  THEN idv.value END) as title,
    MAX(CASE WHEN idf.fieldID=6  THEN idv.value END) as date,
    MAX(CASE WHEN idf.fieldID=38 THEN idv.value END) as journal,
    MAX(CASE WHEN idf.fieldID=59 THEN idv.value END) as doi,
    MAX(CASE WHEN idf.fieldID=10 THEN idv.value END) as volume,
    MAX(CASE WHEN idf.fieldID=11 THEN idv.value END) as issue,
    MAX(CASE WHEN idf.fieldID=14 THEN idv.value END) as pages,
    MAX(CASE WHEN idf.fieldID=13 THEN idv.value END) as url,
    GROUP_CONCAT(DISTINCT cr.lastName || ", " || cr.firstName) as authors
FROM items i
JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
LEFT JOIN itemData idf ON i.itemID = idf.itemID
LEFT JOIN itemDataValues idv ON idf.valueID = idv.valueID
LEFT JOIN itemCreators ic ON i.itemID = ic.itemID
LEFT JOIN creators cr ON ic.creatorID = cr.creatorID
WHERE it.typeName IN ("journalArticle","preprint","conferencePaper","report","book")
  AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
GROUP BY i.key
HAVING doi IS NOT NULL
''')
rows = cur.fetchall()
con.close()

# Build DOI index
doi_map = {}
for key, title, date, journal, doi, vol, issue, pages, url, authors in rows:
    if doi:
        doi_map[doi.strip().lower()] = (key, title, date, journal, doi, vol, issue, pages, url, authors)

print("Found items by DOI:\n")
for target_doi in TARGET_DOIS:
    d = target_doi.lower()
    if d in doi_map:
        key, title, date, journal, doi, vol, issue, pages, url, authors = doi_map[d]
        yr = (date or '')[:4]
        auth_list = [a.strip() for a in (authors or '').split(',')]
        # Build bib key: FirstAuthorLastNameYear
        first_auth = auth_list[0].split(',')[0].strip().replace(' ', '') if auth_list else 'Unknown'
        bib_key = f"{first_auth}{yr}"
        print(f"KEY: {bib_key}")
        print(f"  Title:   {title}")
        print(f"  Journal: {journal}")
        print(f"  Year:    {yr}   Vol: {vol}   Issue: {issue}   Pages: {pages}")
        print(f"  DOI:     {doi}")
        print(f"  Authors: {authors[:120]}")
        print()
        # Print bib entry
        auth_bib = ' and '.join(
            a.strip() for a in (authors or '').split(',') if a.strip()
        )
        print(f"  @article{{{bib_key},")
        print(f"    author  = {{{auth_bib}}},")
        print(f"    title   = {{{title}}},")
        print(f"    journal = {{{journal or ''}}},")
        print(f"    year    = {{{yr}}},")
        if vol: print(f"    volume  = {{{vol}}},")
        if issue: print(f"    number  = {{{issue}}},")
        if pages: print(f"    pages   = {{{pages}}},")
        print(f"    doi     = {{{doi}}}")
        print(f"  }}")
        print()
    else:
        print(f"NOT FOUND: {target_doi}\n")
