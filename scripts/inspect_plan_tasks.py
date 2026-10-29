import httpx
import sys
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

r = httpx.get('http://localhost:8002/api/workflow/transparent/7bfe5bb7-17e3-4342-83b1-750a4ba0d913', timeout=10.0)
data = r.json()
steps = data.get('steps', [])

for idx, s in enumerate(steps):
    d = s.get('data') or {}
    for k, v in d.items():
        if 'task' in k.lower() or 'plan' in k.lower():
            if v:
                print(f"Step {idx} key {k}:", str(v)[:400])

for idx, s in enumerate(steps):
    msg = s.get('message', '')
    if 'tasks' in msg.lower() or 'plan' in msg.lower() or 'wave' in msg.lower():
        print(f"Step {idx} [{s.get('phase')}]: {msg}")
