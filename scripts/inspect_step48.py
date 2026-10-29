import httpx
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

r = httpx.get('http://localhost:8002/api/workflow/transparent/7bfe5bb7-17e3-4342-83b1-750a4ba0d913', timeout=10.0)
data = r.json()
steps = data.get('steps', [])
s48 = steps[48]
print(s48.get('data', {}).get('agent_output'))
