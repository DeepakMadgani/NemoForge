import json
from pathlib import Path

p = Path("C:/aviator_traces/TASK-4AFF03F1/llm_calls.jsonl")
lines = p.read_text(encoding="utf-8").splitlines()
total_cost = 0.0
total_tokens_in = 0
total_tokens_out = 0

for line in lines:
    d = json.loads(line)
    total_cost += d.get("cost_usd", 0.0) or 0.0
    total_tokens_in += d.get("tokens_in", 0) or 0
    total_tokens_out += d.get("tokens_out", 0) or 0

print(f"Total LLM calls: {len(lines)}")
print(f"Total Tokens In: {total_tokens_in:,}")
print(f"Total Tokens Out: {total_tokens_out:,}")
print(f"Total Cost USD: ${total_cost:.5f}")
print("\nSample Call 0:")
print(json.dumps(json.loads(lines[0]), indent=2))
print("\nSample Call -1:")
print(json.dumps(json.loads(lines[-1]), indent=2))

