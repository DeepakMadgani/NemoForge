import httpx
import sys
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

r = httpx.get('http://localhost:8002/api/workflow/transparent/7bfe5bb7-17e3-4342-83b1-750a4ba0d913', timeout=10.0)
data = r.json()
steps = data.get('steps', [])

for idx, s in enumerate(steps):
    msg = s.get('message', '') or ''
    if '5 error(s)' in msg:
        print(f"Step {idx}:")
        d = s.get('data')
        print("data type:", type(d))
        print("data:", json.dumps(d, indent=2)[:2000])
        break
