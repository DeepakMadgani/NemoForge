"""Contract benchmark runner.

Runs canonical intake-contract tickets through contract mode and writes a report.

Usage:
    python -m ticket_to_code.benchmarks.contract_benchmark_runner \
      --workspace C:/CC4E \
      --tickets C:/path/to/ground_truth_tickets.json \
      --report C:/path/to/contract_report.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from ticket_to_code import TicketPriority, ValueEdgeTicket, run_autonomous_workflow
from ticket_to_code.runtime.intake_contract import GroundTruthTicket


def _load_tickets(path: Path) -> List[GroundTruthTicket]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["tickets"] if isinstance(payload, dict) and "tickets" in payload else payload
    if not isinstance(rows, list):
        raise ValueError("Expected JSON list or {'tickets': [...]} payload")
    return [GroundTruthTicket.model_validate(row) for row in rows]


def _to_valueedge(ticket: GroundTruthTicket) -> ValueEdgeTicket:
    priority_map = {
        "low": TicketPriority.LOW,
        "medium": TicketPriority.MEDIUM,
        "high": TicketPriority.HIGH,
    }
    return ValueEdgeTicket(
        ticket_id=ticket.ticket_id,
        title=ticket.title,
        description=ticket.description,
        acceptance_criteria=ticket.acceptance_criteria,
        priority=priority_map.get(ticket.priority.value, TicketPriority.MEDIUM),
        labels=[ticket.type.value],
    )


def run_contract_benchmark(
    workspace_path: str,
    tickets: List[GroundTruthTicket],
) -> Dict[str, Any]:
    per_ticket: List[Dict[str, Any]] = []
    solved = 0
    failed_closed = 0
    needs_retry = 0
    need_more_info = 0
    scored_rows: List[Dict[str, Any]] = []

    for gt in tickets:
        valueedge_ticket = _to_valueedge(gt)
        final_state = run_autonomous_workflow(
            ticket=valueedge_ticket,
            workspace_path=workspace_path,
            execution_mode="contract",
            contract_ticket=gt.model_dump(mode="json"),
        )
        contract_report = final_state.get("contract_report") or {}
        status = str(contract_report.get("status", "unknown"))
        if status == "solved":
            solved += 1
        elif status == "failed_closed":
            failed_closed += 1
        elif status == "needs_retry":
            needs_retry += 1
        elif status == "need_more_info":
            need_more_info += 1

        per_ticket.append(
            {
                "ticket_id": gt.ticket_id,
                "status": status,
                "contract_report": contract_report,
            }
        )

        metrics = _score_ticket(gt, contract_report)
        scored_rows.append({"ticket_id": gt.ticket_id, "metrics": metrics})

    total = len(tickets)
    summary_metrics = _aggregate_metrics(scored_rows)
    thresholds = {
        "solve_rate_min": 0.70,
        "ownership_accuracy_min": 0.85,
        "forbidden_violation_rate_max": 0.05,
        "minimality_score_min": 0.75,
    }
    threshold_eval = {
        "solve_rate_ok": summary_metrics["solve_rate"] >= thresholds["solve_rate_min"],
        "ownership_accuracy_ok": summary_metrics["avg_ownership_accuracy"] >= thresholds["ownership_accuracy_min"],
        "forbidden_violation_ok": summary_metrics["forbidden_violation_rate"] <= thresholds["forbidden_violation_rate_max"],
        "minimality_ok": summary_metrics["avg_minimality_score"] >= thresholds["minimality_score_min"],
    }

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total": total,
            "solved": solved,
            "failed_closed": failed_closed,
            "needs_retry": needs_retry,
            "need_more_info": need_more_info,
            "solve_rate": (solved / total) if total else 0.0,
        },
        "benchmark_metrics": summary_metrics,
        "phase1_thresholds": thresholds,
        "phase1_threshold_eval": threshold_eval,
        "tickets": per_ticket,
        "ticket_metrics": scored_rows,
    }


def _score_ticket(ticket: GroundTruthTicket, contract_report: Dict[str, Any]) -> Dict[str, Any]:
    changed = {
        str(item.get("path", "")).replace("\\", "/")
        for item in (contract_report.get("changed_files") or [])
        if isinstance(item, dict) and item.get("path")
    }
    expected = set(p.replace("\\", "/") for p in ticket.expected_changed_files)
    forbidden = set(p.replace("\\", "/") for p in ticket.forbidden_files)

    expected_hit = len(changed.intersection(expected))
    ownership_accuracy = expected_hit / max(1, len(expected))
    forbidden_violation = 1 if len(changed.intersection(forbidden)) > 0 else 0
    unrelated = len(changed - expected)
    minimality_score = max(0.0, 1 - (unrelated / max(1, len(changed))))

    validation = contract_report.get("validation_evidence") or []
    diagnostics_ok = any(
        isinstance(v, dict)
        and v.get("check") == "diagnostics_changed_files"
        and v.get("result") == "pass"
        for v in validation
    )
    solved = (
        contract_report.get("status") == "solved"
        and forbidden_violation == 0
        and diagnostics_ok
    )

    return {
        "ownership_accuracy": ownership_accuracy,
        "forbidden_violation": forbidden_violation,
        "minimality_score": minimality_score,
        "diagnostics_ok": diagnostics_ok,
        "solved": solved,
    }


def _aggregate_metrics(scored_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not scored_rows:
        return {
            "ticket_count": 0,
            "solve_rate": 0.0,
            "avg_ownership_accuracy": 0.0,
            "forbidden_violation_rate": 0.0,
            "avg_minimality_score": 0.0,
        }

    metrics = [r["metrics"] for r in scored_rows]
    return {
        "ticket_count": len(scored_rows),
        "solve_rate": mean(1.0 if m["solved"] else 0.0 for m in metrics),
        "avg_ownership_accuracy": mean(float(m["ownership_accuracy"]) for m in metrics),
        "forbidden_violation_rate": mean(float(m["forbidden_violation"]) for m in metrics),
        "avg_minimality_score": mean(float(m["minimality_score"]) for m in metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run intake-contract benchmark")
    parser.add_argument("--workspace", required=True, help="Workspace path to run against")
    parser.add_argument("--tickets", required=True, help="Path to canonical ground-truth ticket JSON")
    parser.add_argument("--report", required=True, help="Path to output benchmark report JSON")
    args = parser.parse_args()

    tickets = _load_tickets(Path(args.tickets))
    report = run_contract_benchmark(args.workspace, tickets)

    output_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
