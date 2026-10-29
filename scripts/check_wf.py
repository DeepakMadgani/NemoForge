import httpx
import sys
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

r = httpx.get('http://localhost:8002/api/workflow/transparent/7bfe5bb7-17e3-4342-83b1-750a4ba0d913', timeout=10.0)
print('STATUS CODE:', r.status_code)
data = r.json()
print('STATUS:', data.get('status'))
print('CURRENT NODE:', data.get('current_node') or data.get('node'))

steps = data.get('steps', [])
print('STEPS COUNT:', len(steps))

for idx, s in enumerate(steps):
    msg = s.get('message', '') or ''
    if 'failure' in msg.lower() or 'error' in msg.lower() or 'completed' in s.get('status', '').lower():
        if 'error(s)' in msg or 'failure' in msg.lower():
            print(f"\n--- STEP {idx} [{s.get('node')}] {s.get('status')} ---")
            print(msg)
            if s.get('details'):
                print("DETAILS:", json.dumps(s.get('details'), indent=2)[:500])

print("\n--- LATEST 10 STEPS ---")
for s in steps[-10:]:
    print(f"[{s.get('node')}] {s.get('status')}: {s.get('message')}")
