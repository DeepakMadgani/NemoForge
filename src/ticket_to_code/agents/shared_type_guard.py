"""
Shared-Type Impact Guard — protects existing shared contracts.

When a MODIFY task rewrites a file that declares shared types
(interfaces / types / DTOs / models / enums), this compares the pre-edit
and post-edit shapes and flags DESTRUCTIVE changes that would break existing
consumers:

    - a previously declared field was REMOVED
    - an OPTIONAL field became REQUIRED (optionality narrowed)

Additive changes (new fields, new optional fields) are always allowed.

Language-agnostic: it relies only on WorkspaceSymbolScanner.scan_content(),
which already extracts property names + optionality across TS/Java/Python.
Detection is advisory input to the acceptance gate — it does not itself edit
or revert code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# member boundary + name + optional '?' + ':'  →  TS/type members
#   { contractId?: string;  |  ; fetchLastLoginInfo: boolean;
_TS_MEMBER_RE = re.compile(
    r'(?:[{;,]|^)\s*(?:readonly\s+)?([A-Za-z_$][\w$]*)\s*(\??)\s*:',
    re.MULTILINE,
)

# Java/C#/Kotlin field:  private final String contractId;  (no optionality)
_FIELD_DECL_RE = re.compile(
    r'(?:[{;]|^)\s*(?:public|private|protected|internal|final|readonly|val|var|static|\s)*'
    r'[A-Z][\w<>\[\].]*\s+([a-z_$][\w$]*)\s*[;=]',
    re.MULTILINE,
)

_TYPE_DECL_MARKERS = ("interface ", "type ", "class ", "struct ", "record ")


@dataclass
class SharedTypeRegression:
    type_or_file: str
    field: str
    kind: str            # "removed" | "optional_to_required"

    def describe(self) -> str:
        if self.kind == "removed":
            return f"field '{self.field}' removed from {self.type_or_file}"
        return f"field '{self.field}' changed optional→required in {self.type_or_file}"


def _extract_fields(content: str) -> dict:
    """Return {field_name: is_optional} for declared members of shared types."""
    if not content:
        return {}
    fields: dict = {}
    for m in _TS_MEMBER_RE.finditer(content):
        name, opt = m.group(1), m.group(2)
        is_opt = opt == "?"
        fields[name] = (fields[name] or is_opt) if name in fields else is_opt
    for m in _FIELD_DECL_RE.finditer(content):
        fields.setdefault(m.group(1), False)
    return fields


def detect_shared_type_regressions(
    old_content: str,
    new_content: str,
    file_path: str,
    scanner=None,          # kept for signature compat; extraction is self-contained
    ticket_text: str = "",
) -> list[SharedTypeRegression]:
    """Compare pre/post shapes and return destructive regressions.

    A regression is suppressed if the field name is explicitly referenced in the
    ticket text (the ticket authorizes touching it).
    """
    if not old_content or not new_content:
        return []
    # Only meaningful when the OLD file actually declared a shared type.
    if not any(marker in old_content for marker in _TYPE_DECL_MARKERS):
        return []

    old_fields = _extract_fields(old_content)
    if not old_fields:
        return []
    new_fields = _extract_fields(new_content)
    if not new_fields:
        return []

    ticket_lower = (ticket_text or "").lower()
    label = file_path.replace("\\", "/").split("/")[-1]
    regressions: list[SharedTypeRegression] = []

    for name, was_optional in old_fields.items():
        if name.lower() in ticket_lower:
            continue
        if name not in new_fields:
            regressions.append(SharedTypeRegression(label, name, "removed"))
        elif was_optional and not new_fields[name]:
            regressions.append(
                SharedTypeRegression(label, name, "optional_to_required")
            )

    return regressions
