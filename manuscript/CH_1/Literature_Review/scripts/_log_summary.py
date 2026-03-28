import csv
from collections import defaultdict, Counter

rows = list(csv.DictReader(open('scripts/vcu_download_log.csv', encoding='utf-8-sig')))

def publisher(url):
    if not url: return 'Unknown'
    u = url.lower()
    for key, name in [
        ('sciencedirect', 'Elsevier (ScienceDirect)'),
        ('tandfonline',   'Taylor & Francis'),
        ('wiley',         'Wiley'),
        ('lww.com',       'Wolters Kluwer (LWW)'),
        ('springer',      'Springer Nature'),
        ('sagepub',       'SAGE'),
        ('academic.oup',  'Oxford (OUP)'),
        ('ahajournals',   'AHA Journals'),
        ('healthaffairs', 'Health Affairs'),
        ('jstage',        'J-STAGE'),
        ('ieeexplore',    'IEEE Xplore'),
        ('psycnet',       'APA PsycNET'),
        ('degruyter',     'De Gruyter'),
        ('muse.jhu',      'Project MUSE'),
        ('psychiatrist',  'The Psychiatrist'),
        ('wmpllc',        'Journal of Opioid Mgmt'),
        ('biorxiv',       'bioRxiv (preprint)'),
        ('cambridge',     'Cambridge UP'),
        ('thieme',        'Thieme'),
        ('rescognito',    'Rescognito'),
    ]:
        if key in u:
            return name
    return 'Other'

by_pub = defaultdict(Counter)
for r in rows:
    pub = publisher(r.get('proxy_url', '') or r.get('doi', ''))
    by_pub[pub][r['status']] += 1

total = Counter(r['status'] for r in rows)
print(f"Total processed : {len(rows)}")
print(f"OK (downloaded) : {total['ok']}")
print(f"no_pdf          : {total['no_pdf']}")
print(f"error           : {total['error']}")
print()
print(f"{'Publisher':<38} {'ok':>4} {'no_pdf':>7} {'error':>6} {'total':>6}  {'paywall?'}")
print('-' * 75)

PAYWALLED = {
    'Elsevier (ScienceDirect)', 'Wiley', 'Taylor & Francis',
    'Wolters Kluwer (LWW)', 'Springer Nature', 'SAGE',
    'Oxford (OUP)', 'AHA Journals', 'Health Affairs', 'IEEE Xplore',
    'APA PsycNET', 'De Gruyter', 'Project MUSE', 'The Psychiatrist',
    'Cambridge UP', 'Thieme', 'Journal of Opioid Mgmt',
}

for pub, c in sorted(by_pub.items(), key=lambda x: -sum(x[1].values())):
    t   = sum(c.values())
    pw  = 'paywall' if pub in PAYWALLED else 'OA/mixed'
    print(f"{pub:<38} {c['ok']:>4} {c['no_pdf']:>7} {c['error']:>6} {t:>6}  {pw}")

# no_pdf breakdown
print()
print("no_pdf articles by publisher:")
for r in rows:
    if r['status'] == 'no_pdf':
        pub = publisher(r.get('proxy_url', '') or r.get('doi', ''))
        print(f"  [{pub}] {r['title'][:70]}")
