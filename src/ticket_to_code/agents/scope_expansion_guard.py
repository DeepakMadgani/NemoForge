"""
Scope Expansion Guard — keep generation inside the justified plan.

Rejects file modifications that the plan did not justify, unless there is
evidence backing the expansion. Also rejects speculative shared-model changes
(touching a shared interface/type/DTO/model the ticket does not require).

Deterministic and language-agnostic: it reasons over file paths + a small set
of evidence/justification inputs, never over syntax.
"""

from __future__ import annotations

from dataclasses import dataclass


_SHARED_MODEL_MARKERS = (
    "/models/", "/model/", "/dto/", "/dtos/", ".model.",
    "/entities/", "/types/", "saga.types",
)


def _norm(p: str) -> str:
    return (p or "").replace("\\", "/").lower()


@dataclass
class ScopeViolation:
    file: str
    kind: str            # "unjustified_new_file" | "speculative_shared_model"
    reason: str


def is_shared_model_file(file_path: str) -> bool:
    n = _norm(file_path)
    # Shared services are NOT shared domain models, even if located in a shared directory
    if "/services/" in n or n.endswith((".service.ts", ".service.js", "service.java", "service.kt")):
        return False
    return any(m in n for m in _SHARED_MODEL_MARKERS)


def evaluate_scope(
    planned_files: set[str],
    proposed_files: set[str],
    evidence_files: set[str] | None = None,
    ticket_required_files: set[str] | None = None,
) -> list[ScopeViolation]:
    """Return violations for proposed files that expand beyond the justified plan.

    Args:
        planned_files: files the plan explicitly authorized (modify/create).
        proposed_files: files generation actually wants to touch.
        evidence_files: files backed by repository evidence (justifies expansion).
        ticket_required_files: files the ticket explicitly requires (justifies
            a shared-model change).
    """
    planned = {_norm(p) for p in planned_files}
    evidence = {_norm(p) for p in (evidence_files or set())}
    required = {_norm(p) for p in (ticket_required_files or set())}

    violations: list[ScopeViolation] = []
    for raw in proposed_files:
        f = _norm(raw)
        if f in planned:
            # Planned file: still guard speculative shared-model edits.
            if is_shared_model_file(f) and f not in required and f not in evidence:
                violations.append(ScopeViolation(
                    raw, "speculative_shared_model",
                    "shared model modified but neither ticket-required nor evidence-backed",
                ))
            continue

        # Not in the plan → expansion. Must be evidence-backed.
        if f not in evidence:
            kind = "speculative_shared_model" if is_shared_model_file(f) else "unjustified_new_file"
            violations.append(ScopeViolation(
                raw, kind,
                "file not in plan and not backed by repository evidence",
            ))
    return violations


def is_expansion_allowed(
    file_path: str,
    planned_files: set[str],
    evidence_files: set[str] | None = None,
    ticket_required_files: set[str] | None = None,
) -> bool:
    """True if touching file_path is justified (planned/evidence/ticket-required)."""
    return not evaluate_scope(
        planned_files, {file_path}, evidence_files, ticket_required_files
    )
