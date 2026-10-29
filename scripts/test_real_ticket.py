import httpx
import time
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Dynamically resolve workspace repository path
_candidates = [
    Path(__file__).resolve().parent.parent / "workspace" / "CloudForge-Commerce",
    Path(__file__).resolve().parent.parent.parent / "workspace" / "CloudForge-Commerce",
    Path(r"c:\Users\dmadgani\Desktop\My_Aviator\workspace\CloudForge-Commerce"),
]
repo_dir = next((p for p in _candidates if p.exists()), _candidates[0])

payload = {
    'project_id': '27e43581-72d7-406e-b28b-aff9e691bb4e',
    'ticket_id': 'CF-REFUND-101',
    'ticket_description': """Allow Store Operators to Refund Fulfilled Orders and Release Payment Ledger Holds:

Store operations require the capability to issue refunds on fulfilled orders directly from the management console.

Requirements:
1. Extend the order lifecycle status domain to support a REFUNDED state.
2. Implement the refund transaction: when an order is refunded, its status must become REFUNDED, and the payment reservation held in the payment ledger must be released with the reason "Order refunded".
3. Expose a REST endpoint allowing operators to submit refund requests for an order.
4. In the web management UI, add a Refund action for orders in FULFILLED status so operators can execute refunds directly from the dashboard.
5. Ensure all Java Spring Boot (mvn test) and React TypeScript (tsc) builds compile cleanly.""",
    'repo_path': str(repo_dir),
    'execution_mode': 'pipeline',
    'attachments': [],
    'writable_files': [],
    'forbidden_files': [],
    'model_mode': 'auto_stage',
    'selected_model': 'nebius/nvidia/nemotron-3-super-120b-a12b',
    'labels': ['fullstack', 'complex', 'cross-service']
}

print("Initiating workflow on real un-spoon-fed ticket CF-REFUND-101...", flush=True)
resp = httpx.post('http://localhost:8002/api/workflow/transparent/start', json=payload, timeout=20.0)
print('Start response status:', resp.status_code, flush=True)
if resp.status_code != 200:
    print('Failed:', resp.text, flush=True)
    sys.exit(1)

data = resp.json()
wf_id = data.get('workflow_id')
print(f'Workflow successfully launched! ID: {wf_id}', flush=True)

last_step_count = 0
for i in range(350):
    time.sleep(3)
    try:
        r = httpx.get(f'http://localhost:8002/api/workflow/transparent/{wf_id}', timeout=10.0)
        if r.status_code != 200:
            continue
        wf = r.json()
        status = wf.get('status')
        current_node = wf.get('current_node') or wf.get('node')
        steps = wf.get('steps', [])
        if len(steps) > last_step_count:
            for s in steps[last_step_count:]:
                phase = s.get('phase', 'STEP')
                name = s.get('name') or s.get('step_name')
                summary = s.get('summary') or s.get('status')
                message = s.get('message') or ''
                print(f"[{phase}] {name}: {summary} {message}", flush=True)
                if summary == 'awaiting_decision' or 'awaiting_decision' in str(s):
                    print("--> Submitting safe user decision 'leave'...", flush=True)
                    try:
                        httpx.post('http://localhost:8002/api/workflow/transparent/decision-response',
                                   json={'workflow_id': wf_id, 'choice': 'leave'}, timeout=5.0)
                    except Exception as de:
                        print('--> Decision post error:', de, flush=True)
            last_step_count = len(steps)

        if status in ['completed', 'failed', 'error', 'needs_approval', 'awaiting_approval']:
            print(f"\n Workflow finished with status: {status}", flush=True)
            if status in ['error', 'failed']:
                print('Error detail:', wf.get('error'), flush=True)
            break
    except Exception as e:
        print('Polling error:', e, flush=True)

