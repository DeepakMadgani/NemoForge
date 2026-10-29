"""
Semantic Requirements Extraction — cardinality & behavioral constraints.

Deterministically pulls the semantic constraints that compilation cannot see
out of the ticket text, so they can be injected into the generation context:

  * cardinality: each / every / per-user / per-row / all / multiple / single
  * read-only vs editable fields
  * conditional behavior
  * persistence / save mapping
  * duplicate prevention

Language- and project-agnostic: pure keyword/regex extraction over ticket text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_CARDINALITY_PATTERNS = {
    "each": r"\beach\b",
    "every": r"\bevery\b",
    "per_user": (
        r"\bper[-\s]?(?:selected[-\s]?)?user\b"
        r"|\bfor each (?:staged |selected )?(?:user|member)\b"
        r"|\b(?:staged|selected)\s+(?:user|member)\b"
        r"|\btheir\s+(?:organization|org|company|role|status|assignment)\b"  # distributive possessive
    ),
    "per_row": r"\bper[-\s]?row\b|\beach row\b|\brow[-\s]?(?:level|specific|by[-\s]?row)\b",
    "all": r"\ball (?:of )?(?:the )?(?:selected|staged|users|members|items)\b",
    "multiple": r"\bmultiple\b|\bmany\b",
    "single": r"\bonly one\b|\ba single\b|\bexactly one\b",
}

_READONLY_PATTERNS = r"\bread[-\s]?only\b|\bstatic text\b|\bnon[-\s]?editable\b|\bdisplay(?:ed)? as (?:read[-\s]?only|static)\b"
_EDITABLE_PATTERNS = r"\beditable\b|\bdropdown\b|\bselect(?:able)?\b|\binput field\b"
_DUPLICATE_PATTERNS = r"\bduplicate\b|\bprevent duplicate\b|\balready (?:a )?member\b|\bflag(?:ged)? as (?:a )?duplicate\b"
_PERSIST_PATTERNS = r"\bon save\b|\bpersist(?:ed|ence)?\b|\bstore(?:d)?\b|\bsave(?:d|s)?\b to\b|\bcommit(?:ted)?\b"
_CONDITION_PATTERNS = r"\bif\b|\bwhen\b|\bunless\b|\bonly if\b|\botherwise\b"


@dataclass
class SemanticConstraints:
    cardinality: list[str] = field(default_factory=list)
    read_only: bool = False
    editable: bool = False
    duplicate_prevention: bool = False
    persistence: bool = False
    has_conditions: bool = False
    representative_phrases: list[str] = field(default_factory=list)

    @property
    def is_per_item(self) -> bool:
        return any(c in self.cardinality for c in ("each", "every", "per_user", "per_row", "all"))

    def is_empty(self) -> bool:
        return not (
            self.cardinality or self.read_only or self.editable
            or self.duplicate_prevention or self.persistence or self.has_conditions
        )

    def to_prompt_block(self) -> str:
        if self.is_empty():
            return ""
        lines = ["=== SEMANTIC CONSTRAINTS (account for these BEFORE writing code) ==="]
        if self.cardinality:
            lines.append(f"- Cardinality: {', '.join(self.cardinality)}")
            if self.is_per_item:
                lines.append(
                    "  → Apply the behavior to EACH item (loop/map over the collection; "
                    "use per-row/per-item state). Do NOT special-case a single selection."
                )
        if self.read_only:
            lines.append("- Some field(s) must render READ-ONLY / static (not editable).")
        if self.editable:
            lines.append("- Some field(s) must remain EDITABLE.")
        if self.duplicate_prevention:
            lines.append("- Duplicate prevention is required (flag/reject duplicates).")
        if self.persistence:
            lines.append("- Persistence: changes must be saved/persisted as specified.")
        if self.has_conditions:
            lines.append("- Conditional behavior present — preserve all branches/conditions.")
        lines.append("=== END SEMANTIC CONSTRAINTS ===")
        return "\n".join(lines)


def extract_semantic_constraints(
    ticket_title: str = "",
    ticket_description: str = "",
    requirements_text: str = "",
) -> SemanticConstraints:
    text = "\n".join(filter(None, [ticket_title, ticket_description, requirements_text]))
    low = text.lower()
    sc = SemanticConstraints()

    for name, pat in _CARDINALITY_PATTERNS.items():
        if re.search(pat, low):
            sc.cardinality.append(name)

    sc.read_only = bool(re.search(_READONLY_PATTERNS, low))
    sc.editable = bool(re.search(_EDITABLE_PATTERNS, low))
    sc.duplicate_prevention = bool(re.search(_DUPLICATE_PATTERNS, low))
    sc.persistence = bool(re.search(_PERSIST_PATTERNS, low))
    sc.has_conditions = bool(re.search(_CONDITION_PATTERNS, low))

    # Capture a few representative sentences containing the strongest signals.
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sl = sentence.lower()
        if any(re.search(p, sl) for p in (
            _CARDINALITY_PATTERNS["each"], _CARDINALITY_PATTERNS["per_user"],
            _CARDINALITY_PATTERNS["per_row"], _READONLY_PATTERNS, _DUPLICATE_PATTERNS,
        )):
            s = sentence.strip()
            if s and s not in sc.representative_phrases:
                sc.representative_phrases.append(s[:200])
        if len(sc.representative_phrases) >= 5:
            break

    return sc
