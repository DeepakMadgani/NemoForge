"""Batch validation runner for real CC4E tickets.

Runs a ticket batch through the current production pipeline and emits:
- Per-ticket outcomes
- Failure taxonomy counts
- Aggregate success/latency metrics

Usage:
    python -m ticket_to_code.benchmarks.batch_validation_runner \
      --workspace C:/CC4E \
      --tickets C:/path/to/tickets.json \
      --report-dir C:/path/to/reports
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ticket_to_code import TicketPriority, ValueEdgeTicket, run_autonomous_workflow


VALID_TAXONOMY = {
    "retrieval",
    "planning",
    "localization",
    "generation",
    "validation",
    "environment",
    "unknown",
}


@dataclass
class TicketRunResult:
    ticket_id: str
    success: bool
    taxonomy: str
    latency_seconds: float
    status: str
    details: str


class BatchValidationRunner:
    def __init__(self, workspace_path: str, technology: Optional[str] = None):
        self.workspace_path = workspace_path
        self.technology = technology

    def run(self, tickets: List[Dict[str, Any]]) -> Dict[str, Any]:
        results: List[TicketRunResult] = []

        for index, row in enumerate(tickets, start=1):
            ticket = self._to_ticket(row, index)
            started = time.perf_counter()
            state: Dict[str, Any] = {}
            exc_message: Optional[str] = None

            try:
                state = run_autonomous_workflow(
                    ticket=ticket,
                    workspace_path=self.workspace_path,
                    technology=self.technology,
                    max_fix_attempts=3,
                    execution_mode="pipeline",
                )
            except Exception as exc:  # noqa: BLE001
                exc_message = str(exc)

            latency = time.perf_counter() - started
            success = self._is_success(state, exc_message)
            taxonomy, details = self._classify_failure(state, exc_message)

            result = TicketRunResult(
                ticket_id=ticket.ticket_id,
                success=success,
                taxonomy=taxonomy,
                latency_seconds=latency,
                status=str(state.get("status", "exception" if exc_message else "unknown")),
                details=details,
            )
            results.append(result)

        return self._aggregate(results)

    def _to_ticket(self, row: Dict[str, Any], index: int) -> ValueEdgeTicket:
        ticket_id = str(row.get("ticket_id") or row.get("id") or f"BATCH-{index:04d}")
        title = str(row.get("title") or row.get("ticket_title") or f"Ticket {index}")
        description = str(row.get("description") or row.get("ticket_description") or "")
        labels = row.get("labels") if isinstance(row.get("labels"), list) else []

        priority_map = {
            "critical": TicketPriority.CRITICAL,
            "high": TicketPriority.HIGH,
            "medium": TicketPriority.MEDIUM,
            "low": TicketPriority.LOW,
        }
        raw_priority = str(row.get("priority", "medium")).strip().lower()
        priority = priority_map.get(raw_priority, TicketPriority.MEDIUM)

        return ValueEdgeTicket(
            ticket_id=ticket_id,
            title=title[:140],
            description=description,
            priority=priority,
            labels=[str(x) for x in labels],
        )

    def _is_success(self, state: Dict[str, Any], exc_message: Optional[str]) -> bool:
        if exc_message:
            return False

        status = str(state.get("status", "")).lower()
        if status in {"failed", "error", "semantic_validation_failed", "candidates_invalid"}:
            return False

        generated = state.get("generated_code") or []
        errors = state.get("errors") or []

        return bool(generated) and not errors

    def _classify_failure(self, state: Dict[str, Any], exc_message: Optional[str]) -> tuple[str, str]:
        if exc_message:
            return "environment", exc_message[:300]

        status = str(state.get("status", "")).lower()
        reason = str(state.get("validation_failure_reason", "")).lower()
        errors = " ".join(str(e) for e in (state.get("errors") or [])).lower()
        combined = f"{status} {reason} {errors}"

        if not combined.strip():
            return "unknown", "No failure detail available"

        if any(k in combined for k in ["rag", "discovery", "candidate", "evidence insufficient", "no context"]):
            return "retrieval", combined[:300]

        if any(k in combined for k in ["plan", "planner", "strategy", "task selection"]):
            return "planning", combined[:300]

        if any(k in combined for k in ["localiz", "owner", "grounded", "ownership"]):
            return "localization", combined[:300]

        if any(k in combined for k in ["generate", "patch", "diff", "multi_angle_review"]):
            return "generation", combined[:300]

        if any(k in combined for k in ["build", "test", "semantic_validation", "outcome_verification", "validation"]):
            return "validation", combined[:300]

        if any(k in combined for k in ["timeout", "connection", "credential", "permission", "module", "import", "exception"]):
            return "environment", combined[:300]

        return "unknown", combined[:300]

    def _aggregate(self, results: List[TicketRunResult]) -> Dict[str, Any]:
        total = len(results)
        success_count = sum(1 for r in results if r.success)
        failed = [r for r in results if not r.success]

        taxonomy_counts: Dict[str, int] = {k: 0 for k in VALID_TAXONOMY}
        for r in failed:
            taxonomy_counts[r.taxonomy if r.taxonomy in taxonomy_counts else "unknown"] += 1

        avg_latency = sum(r.latency_seconds for r in results) / total if total else 0.0
        p95_latency = 0.0
        if results:
            sorted_lat = sorted(r.latency_seconds for r in results)
            idx = int(0.95 * (len(sorted_lat) - 1))
            p95_latency = sorted_lat[idx]

        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "summary": {
                "total_tickets": total,
                "success_count": success_count,
                "failure_count": total - success_count,
                "success_rate": (success_count / total) if total else 0.0,
                "avg_latency_seconds": avg_latency,
                "p95_latency_seconds": p95_latency,
            },
            "failure_taxonomy": taxonomy_counts,
            "tickets": [
                {
                    "ticket_id": r.ticket_id,
                    "success": r.success,
                    "taxonomy": r.taxonomy,
                    "latency_seconds": round(r.latency_seconds, 3),
                    "status": r.status,
                    "details": r.details,
                }
                for r in results
            ],
        }


def load_tickets(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("tickets"), list):
        return payload["tickets"]
    raise ValueError("Expected JSON list, JSONL, or {'tickets': [...]} payload")


def write_report(report: Dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "batch_validation_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report["summary"]
    taxonomy = report["failure_taxonomy"]
    lines = [
        "# Batch Validation Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Summary",
        "",
        f"- Total tickets: {summary['total_tickets']}",
        f"- Success count: {summary['success_count']}",
        f"- Failure count: {summary['failure_count']}",
        f"- Success rate: {summary['success_rate']:.2%}",
        f"- Avg latency: {summary['avg_latency_seconds']:.2f}s",
        f"- P95 latency: {summary['p95_latency_seconds']:.2f}s",
        "",
        "## Failure Taxonomy",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for key in sorted(taxonomy.keys()):
        lines.append(f"| {key} | {taxonomy[key]} |")

    lines.extend([
        "",
        "## Ticket Outcomes",
        "",
        "| Ticket | Success | Taxonomy | Status | Latency (s) |",
        "|---|---|---|---|---:|",
    ])

    for t in report["tickets"]:
        lines.append(
            f"| {t['ticket_id']} | {str(t['success'])} | {t['taxonomy']} | {t['status']} | {t['latency_seconds']:.3f} |"
        )

    (report_dir / "batch_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch validation and failure taxonomy.")
    parser.add_argument("--workspace", required=True, help="Workspace path to run tickets against")
    parser.add_argument("--tickets", required=True, help="Path to tickets file (.json or .jsonl)")
    parser.add_argument("--report-dir", required=True, help="Directory for output reports")
    parser.add_argument("--technology", default=None, help="Optional explicit technology mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tickets = load_tickets(Path(args.tickets))
    runner = BatchValidationRunner(args.workspace, technology=args.technology)
    report = runner.run(tickets)

    write_report(report, Path(args.report_dir))

    summary = report["summary"]
    print("Batch validation complete")
    print(f"  Total tickets : {summary['total_tickets']}")
    print(f"  Success rate  : {summary['success_rate']:.2%}")
    print(f"  Avg latency   : {summary['avg_latency_seconds']:.2f}s")
    print(f"  P95 latency   : {summary['p95_latency_seconds']:.2f}s")


if __name__ == "__main__":
    main()
