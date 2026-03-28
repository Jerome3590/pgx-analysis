import json
cookies = json.load(open('secrets/session_cookies.json'))
print(f'Total cookies: {len(cookies)}')
for c in cookies:
    domain = c.get('domain', '')
    name   = c.get('name', '')
    val    = str(c.get('value', ''))[:30]
    print(f'  {domain:45s} {name:30s} {val}')
