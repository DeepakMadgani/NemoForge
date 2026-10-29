"""Print a readable table of a ticket's workflow steps from steps.jsonl.

Usage: python -m ticket_to_code.tracing.step_report <ticket_id> [trace_root]
"""

import json
import sys
from pathlib import Path
from typing import List

from ticket_to_code.tracing.paths import get_trace_root


def load_steps(ticket_id: str, trace_root: str = "") -> List[dict]:
    root = Path(trace_root) if trace_root else get_trace_root()
    path = root / ticket_id / "steps.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def format_steps(steps: List[dict]) -> str:
    lines = [f"{'#':>3}  {'node':<30} {'ok':<4} {'ms':>8}  produced / decision"]
    for s in steps:
        produced = s.get("produced") or {}
        detail = f"-> {produced.get('decision')}" if s["node"].startswith("route:") else ", ".join(
            f"{k}={json.dumps(v, default=str)[:70]}" for k, v in list(produced.items())[:4]
        )
        if s.get("error"):
            detail = f"ERROR {s['error']}"
        lines.append(f"{s['seq']:>3}  {s['node']:<30} {'yes' if s['ok'] else 'NO':<4} {s['duration_ms']:>8}  {detail}")
    total = sum(s["duration_ms"] for s in steps if not s["node"].startswith("route:"))
    lines.append(f"\n{len(steps)} entries, {total / 1000:.1f}s in nodes")
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    steps = load_steps(argv[0], argv[1] if len(argv) > 1 else "")
    print(format_steps(steps) if steps else f"no steps recorded for {argv[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
