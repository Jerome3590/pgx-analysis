from pyzotero import zotero

library_id = '6037399'
api_key = 'xxjsStqHkKgaSNnzb8FmG3Zb'

zot = zotero.Zotero(library_id, 'user', api_key)

# Fetch and list all collections
collections = zot.collections()

# Print collection names and their IDs
for collection in collections:
    print(f"Collection Name: {collection['data']['name']}, Collection ID: {collection['key']}")