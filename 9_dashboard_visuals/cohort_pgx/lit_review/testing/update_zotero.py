import requests


user_id = '6037399'
api_key = 'xxjsStqHkKgaSNnzb8FmG3Zb'


item_key = 'item_key'  # Replace with the item key for the Zotero item you wish to update

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json',
}

# This is an example of how you might add a PMC ID
data = {
    "extra": "PMC123456"  # You can store the PMC ID in the 'extra' field or any appropriate field
}

url = f'https://api.zotero.org/users/{user_id}/items/{item_key}'

response = requests.patch(url, headers=headers, json=data)

if response.status_code == 204:
    print("Update successful!")
else:
    print("Failed to update item:", response.status_code, response.text)
