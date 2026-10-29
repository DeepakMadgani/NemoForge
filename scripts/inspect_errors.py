import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

path = r'C:\Users\dmadgani\Desktop\My_Aviator\workspace\CloudForge-Commerce\.aviator\tickets\7bfe5bb7-17e3-4342-83b1-750a4ba0d913.json'
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print('STATUS:', data.get('status'))
print('CURRENT NODE:', data.get('current_node') or data.get('node'))

steps = data.get('steps', [])
print(f'Total steps: {len(steps)}')

for s in steps:
    msg = s.get('message', '') or ''
    if 'error(s)' in msg or 'buildstatus.failure' in msg.lower() or 'diagnostics' in msg.lower():
        print(f"[{s.get('node')}] {msg}")
        details = s.get('details')
        if details:
            print('  Details:', str(details)[:300])

if 'state' in data:
    st = data['state']
    print('State keys:', list(st.keys()) if isinstance(st, dict) else type(st))
    if isinstance(st, dict) and 'build_output' in st:
        print('Build output:', str(st['build_output'])[:500])
    if isinstance(st, dict) and 'compiler_errors' in st:
        print('Compiler errors:', str(st['compiler_errors'])[:500])
