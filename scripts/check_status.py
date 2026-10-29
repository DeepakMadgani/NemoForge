import httpx
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

wf_id = sys.argv[1] if len(sys.argv) > 1 else '304e4ee9-3b05-47c9-b2b0-3a801ea6bf10'
r = httpx.get(f'http://localhost:8002/api/workflow/transparent/{wf_id}').json()
print("STATUS:", r.get("status"))
print("CURRENT PHASE:", r.get("current_phase"))
print("STEPS COUNT:", len(r.get("steps", [])))
print("--- LAST 10 STEPS ---")
for s in r.get("steps", [])[-10:]:
    phase = s.get("phase")
    status = s.get("status")
    msg = s.get("message", "")
    data = s.get("data")
    print(f"[{phase}] ({status}) {msg[:100]}")

print("\n--- TOKEN USAGE ---")
import json
print(json.dumps(r.get("token_usage", {}), indent=2))
