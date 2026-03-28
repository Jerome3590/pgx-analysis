import csv
from collections import Counter

rows = list(csv.DictReader(open('scripts/oa_scan_results.csv')))
counts = Counter(r['status'] for r in rows)

print('=== OA Scan Summary ===')
for k, v in counts.most_common():
    print(f'  {k:<30} {v}')

print()
print('=== DOWNLOADED (new free PDFs) ===')
for r in rows:
    if r['status'] == 'downloaded':
        print(f"  {r['doi'][:42]:<42}  {r['title'][:65]}")

print()
print('=== URL FOUND but download blocked (try browser) ===')
for r in rows:
    if r['status'] == 'url_found_no_download':
        print(f"  {r['doi'][:42]:<42}  {r['title'][:55]}")
        print(f"    {r['pdf_url'][:90]}")
