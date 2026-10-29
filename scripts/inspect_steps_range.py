import httpx
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

r = httpx.get('http://localhost:8002/api/workflow/transparent/7bfe5bb7-17e3-4342-83b1-750a4ba0d913', timeout=10.0)
data = r.json()
steps = data.get('steps', [])

for i in range(45, 80):
    if i < len(steps):
        s = steps[i]
        print(f"Step {i} [{s.get('phase')}]: {s.get('message')}")
        d = s.get('data') or {}
        if d.get('agent_output'):
            print("  Output:", str(d.get('agent_output'))[:300].replace('\n', ' '))
