import requests
import os
from pyzotero import zotero

user_id = '6037399'
api_key = 'xxjsStqHkKgaSNnzb8FmG3Zb'
collection_id = 'LS75EWXU'

zot = zotero.Zotero(user_id, 'user', api_key)

# Fetch items from the library or a specific collection
if collection_id:
    items = zot.collection_items(collection_id)
else:
    items = zot.top()  # Fetch top-level items in your library

print("Fetched items: ", items)  # Print fetched items to verify

item_key = items[23]['key']  # Fetch the 19th item's key
print("Selected item key: ", item_key)  # Print the selected item key

item_json = items[23]

print(item_json)

# Setup headers for API requests
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

# API endpoint to fetch item metadata
metadata_url = f'https://api.zotero.org/users/{user_id}/items/{item_key}'
response = requests.get(metadata_url, headers=headers)
item = response.json()

print("Item metadata: ", item)  # Print item metadata

# Prepare the download directory
download_dir = 'zotero_pdf'
if not os.path.exists(download_dir):
    os.makedirs(download_dir)
    print(f"Created directory: {download_dir}")
else:
    print(f"Directory already exists: {download_dir}")

# Check if there's an attachment and download it
if 'children' in item['links']:
    attachment_url = item['links']['children']['href']
    attachment_response = requests.get(attachment_url, headers=headers)
    attachments = attachment_response.json()

    print("Attachments metadata: ", attachments)  # Print attachments metadata

    for attachment in attachments:
        if attachment['data']['itemType'] == 'attachment' and 'download' in attachment['links']:
            download_url = attachment['links']['download']['href']
            file_response = requests.get(download_url, headers=headers, stream=True)

            # Use content from the 'Extra' field to rename the file
            extra_field_content = item['data'].get('extra', 'default_filename')
            file_path = os.path.join(download_dir, f'{extra_field_content}.pdf')  # Assume PDF, adjust accordingly
            print("Downloading to: ", file_path)  # Print the intended download path

            with open(file_path, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f'File downloaded and saved as {file_path}')
else:
    print('No attachment found for this item')
