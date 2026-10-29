"""
Generation Readiness Gate + Contract-First name enforcement.

Two responsibilities:

1. assess_generation_readiness(context) — confirm the assembled generation
   context actually contains what "generate correctly first" needs (requirements,
   existing behavior, verified capabilities, contracts, callers, reuse decisions,
   justified files, untouched files, cross-file deps, semantic constraints,
   persistence expectations). Blocks only when a CRITICAL contract/capability is
   still uncertain — never on advisory gaps.

2. check_contract_adherence(...) — when a VERIFIED contract exists, prevent the
   generator from guessing property/argument/method names: any referenced name
   that is not in the verified set but closely matches a verified name is a
   guess and is flagged.

Deterministic; no LLM; language-agnostic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum


# Context keys that "generate correctly first" expects. CRITICAL keys, when
# uncertain, block generation; the rest are advisory (reported, not blocking).
CRITICAL_KEYS = ("contracts", "verified_capabilities")
ADVISORY_KEYS = (
    "requirements", "existing_behavior", "callers", "reuse_decisions",
    "justified_files", "untouched_files", "cross_file_dependencies",
    "semantic_constraints", "persistence_expectations",
)


@dataclass
class ReadinessReport:
    missing: list[str] = field(default_factory=list)          # advisory gaps
    missing_critical: list[str] = field(default_factory=list)  # critical gaps
    uncertain_contracts: list[str] = field(default_factory=list)
    present: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.missing_critical and not self.uncertain_contracts

    def is_blocking(self) -> bool:
        """Block generation only for uncertain/missing CRITICAL contracts."""
        return bool(self.missing_critical) or bool(self.uncertain_contracts)

    def summary(self) -> str:
        parts = []
        if self.missing_critical:
            parts.append(f"CRITICAL missing: {self.missing_critical}")
        if self.uncertain_contracts:
            parts.append(f"uncertain contracts: {self.uncertain_contracts}")
        if self.missing:
            parts.append(f"advisory gaps: {self.missing}")
        return " | ".join(parts) if parts else "ready"


def _is_present(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, dict, set, tuple, str)):
        return len(value) > 0
    return True


def assess_generation_readiness(context: dict) -> ReadinessReport:
    """Assess whether the generation context is complete enough to generate."""
    report = ReadinessReport()
    for key in CRITICAL_KEYS:
        if _is_present(context.get(key)):
            report.present.append(key)
        else:
            report.missing_critical.append(key)
    for key in ADVISORY_KEYS:
        if _is_present(context.get(key)):
            report.present.append(key)
        else:
            report.missing.append(key)

    # Contracts explicitly marked uncertain/proposed block generation.
    for c in (context.get("contracts") or []):
        status = ""
        name = ""
        if isinstance(c, dict):
            status = str(c.get("status", "")).lower()
            name = str(c.get("name", c.get("symbol", "contract")))
        else:
            status = str(getattr(c, "status", "")).lower()
            name = str(getattr(c, "name", getattr(c, "symbol", "contract")))
        if status in ("uncertain", "proposed", "planned", "unverified"):
            report.uncertain_contracts.append(name)

    return report


@dataclass
class ContractViolation:
    referenced: str
    verified_closest: str
    reason: str


def _closest(name: str, candidates, threshold: float):
    best, ratio = None, 0.0
    nl = name.lower()
    for c in candidates:
        r = SequenceMatcher(None, nl, c.lower()).ratio()
        if r > ratio and r >= threshold:
            best, ratio = c, r
    return best


def check_contract_adherence(
    referenced_members: list[str],
    verified_members: list[str],
    threshold: float = 0.6,
) -> list[ContractViolation]:
    """Flag referenced names that are NOT in the verified contract but closely
    match a verified name — i.e. the generator guessed instead of using the
    verified name."""
    verified_lower = {v.lower() for v in verified_members}
    violations: list[ContractViolation] = []
    for ref in referenced_members:
        if ref.lower() in verified_lower:
            continue
        close = _closest(ref, verified_members, threshold)
        if not close:
            # containment heuristic: 'userId' vs 'otdsUserId'
            rl = ref.lower()
            for v in verified_members:
                vl = v.lower()
                if len(rl) >= 3 and (rl in vl or vl in rl):
                    close = v
                    break
        if close:
            violations.append(ContractViolation(
                ref, close,
                f"'{ref}' is not in the verified contract; closest verified name is '{close}'",
            ))
    return violations


# ── Semantic (per-requirement) readiness ────────────────────────────────────────

class RequirementStatus(str, Enum):
    UNDERSTOOD = "understood"
    PARTIALLY_UNDERSTOOD = "partially_understood"
    UNRESOLVED = "unresolved"


_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "their", "them",
    "when", "whether", "already", "should", "would", "could", "must", "have",
    "user", "users", "member", "members", "current", "other", "same", "show",
    "display", "value", "using", "based", "each", "existing", "which", "there",
}


def _key_terms(text: str) -> list[str]:
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9_]{3,}", (text or "").lower())
    seen: list[str] = []
    for t in toks:
        if t not in _STOPWORDS and t not in seen:
            seen.append(t)
    return seen[:8]


def requirement_status(requirement_text: str, grounded_facts, key_terms=None) -> str:
    """Classify how well a requirement is grounded by the collected facts."""
    terms = key_terms or _key_terms(requirement_text)
    if not terms:
        return RequirementStatus.UNDERSTOOD.value
    facts = [str(f).lower() for f in (grounded_facts or [])]
    hits = sum(1 for t in terms if any(t in gf for gf in facts))
    if hits == len(terms):
        return RequirementStatus.UNDERSTOOD.value
    if hits > 0:
        return RequirementStatus.PARTIALLY_UNDERSTOOD.value
    return RequirementStatus.UNRESOLVED.value


@dataclass
class SemanticReadinessReport:
    per_requirement: dict = field(default_factory=dict)   # name → status
    critical: set = field(default_factory=set)

    @property
    def critical_unresolved(self) -> list:
        return [
            name for name, st in self.per_requirement.items()
            if name in self.critical and st == RequirementStatus.UNRESOLVED.value
        ]

    def blocks(self) -> bool:
        """Only CRITICAL + UNRESOLVED requirements block generation."""
        return bool(self.critical_unresolved)


def evaluate_semantic_readiness(
    critical_requirements: list[str],
    optional_requirements: list[str] | None = None,
    grounded_facts=None,
) -> SemanticReadinessReport:
    """Per-requirement grounding assessment; blocks only on critical unresolved."""
    report = SemanticReadinessReport()
    for req in (critical_requirements or []):
        report.per_requirement[req] = requirement_status(req, grounded_facts)
        report.critical.add(req)
    for req in (optional_requirements or []):
        report.per_requirement[req] = requirement_status(req, grounded_facts)
    return report
