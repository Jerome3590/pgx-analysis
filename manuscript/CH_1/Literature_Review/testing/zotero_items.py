from pyzotero import zotero

library_id = '6037399'
api_key = 'xxjsStqHkKgaSNnzb8FmG3Zb'
collection_id = 'LS75EWXU'

zot = zotero.Zotero(library_id, 'user', api_key)

# Fetch items from the library or a specific collection
if collection_id:
    items = zot.collection_items(collection_id)
else:
    items = zot.top()  # Fetch top-level items in your library

# Filter and display the JSON data for title, abstract, and extra (PMC_ID)
for item in items:
    try:
        print({
            'title': item['data'].get('title'),
            'abstract': item['data'].get('abstractNote'),
            'extra': item['data'].get('extra')
        })
    except KeyError:
        continue

