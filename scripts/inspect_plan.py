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
    d = s.get('data') or {}
    if 'plan' in str(d).lower() or 'task' in str(d).lower() or 'planning' in str(s.get('node', '')).lower():
        if d.get('agent_output') and ('Task 1' in d.get('agent_output') or 'tasks' in d.get('agent_output')):
            print(f"--- STEP {idx} ---")
            print(d.get('agent_output')[:1000])

print("Generated files in data:", data.get('generated_files'))
print("Candidate files in data:", data.get('candidate_files'))
