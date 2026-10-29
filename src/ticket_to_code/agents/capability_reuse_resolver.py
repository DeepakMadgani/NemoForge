"""
Capability Reuse Resolver — REUSE / ADAPT / EXTEND / CREATE_NEW.

For each capability the planner says is needed, decide whether an existing
repository capability already satisfies it BEFORE generation creates a new one.
CREATE_NEW is only returned when no existing capability is found — and callers
must supply evidence that existing code is insufficient.

Uses the existing SymbolResolver (owner-type members) and, when available, the
WorkspaceSymbolIndex (repo-wide symbol lookup). Language-agnostic: it reasons
over symbol names, not syntax.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from typing import Optional

_FUZZY_THRESHOLD = 0.72


class ReuseDecision(str, Enum):
    REUSE_EXISTING = "reuse_existing"      # exact existing capability
    ADAPT_EXISTING = "adapt_existing"      # existing capability, different name — same intent
    EXTEND_EXISTING = "extend_existing"    # owner type exists; add the member to it
    CREATE_NEW = "create_new"              # nothing exists — evidence required


@dataclass
class ReuseResult:
    capability: str
    decision: str                 # ReuseDecision value
    existing_symbol: Optional[str]
    owner_type: Optional[str]
    evidence: str

    @property
    def requires_evidence(self) -> bool:
        """CREATE_NEW must be justified by repository evidence."""
        return self.decision == ReuseDecision.CREATE_NEW.value


class CapabilityReuseResolver:
    def __init__(self, symbol_resolver=None, symbol_index=None):
        self._resolver = symbol_resolver
        self._index = symbol_index

    def resolve(self, capability_name: str, owner_type: Optional[str] = None) -> ReuseResult:
        # 1. Owner-type scoped resolution (most precise)
        if owner_type and self._resolver is not None:
            try:
                defn = self._resolver.resolve_type(owner_type)
            except Exception:
                defn = None
            if defn is not None:
                try:
                    if defn.has_member(capability_name):
                        return ReuseResult(
                            capability_name, ReuseDecision.REUSE_EXISTING.value,
                            capability_name, owner_type,
                            f"{owner_type}.{capability_name} already exists",
                        )
                except Exception:
                    pass
                members = list(getattr(defn, "method_names", []) or []) + \
                    list(getattr(defn, "property_names", []) or [])
                close = self._closest(capability_name, members)
                if close:
                    return ReuseResult(
                        capability_name, ReuseDecision.ADAPT_EXISTING.value,
                        close, owner_type,
                        f"existing {owner_type}.{close} matches intent of '{capability_name}'",
                    )
                # Owner exists but has no equivalent → add to the existing type.
                return ReuseResult(
                    capability_name, ReuseDecision.EXTEND_EXISTING.value,
                    None, owner_type,
                    f"{owner_type} exists; extend it rather than create a new type",
                )

        # 2. Repo-wide lookup via the symbol index
        if self._index is not None:
            try:
                locs = self._index.find_symbol_references(capability_name)
            except Exception:
                locs = []
            if locs:
                return ReuseResult(
                    capability_name, ReuseDecision.REUSE_EXISTING.value,
                    capability_name, None,
                    f"symbol '{capability_name}' already exists in {locs[0]}",
                )
            close, owner = self._closest_in_index(capability_name)
            if close:
                return ReuseResult(
                    capability_name, ReuseDecision.ADAPT_EXISTING.value,
                    close, owner,
                    f"existing '{close}' matches intent of '{capability_name}'",
                )

        # 3. Nothing found — creating new requires evidence.
        return ReuseResult(
            capability_name, ReuseDecision.CREATE_NEW.value,
            None, owner_type,
            "no existing capability found — evidence of insufficiency required",
        )

    # ── internals ──────────────────────────────────────────────────────────────

    def _closest_in_index(self, name: str):
        best, best_owner, ratio = None, None, 0.0
        try:
            file_symbols = getattr(self._index, "file_symbols", {}) or {}
        except Exception:
            file_symbols = {}
        nl = name.lower()
        for _fp, fsym in file_symbols.items():
            for s in getattr(fsym, "symbols", []) or []:
                if getattr(s, "kind", "") not in ("method", "function", "property"):
                    continue
                r = SequenceMatcher(None, nl, s.name.lower()).ratio()
                if r > ratio and r >= _FUZZY_THRESHOLD:
                    best, best_owner, ratio = s.name, getattr(s, "owner_class", None), r
        return best, best_owner

    @staticmethod
    def _closest(name: str, candidates: list) -> Optional[str]:
        best, ratio = None, 0.0
        nl = name.lower()
        for c in candidates:
            r = SequenceMatcher(None, nl, c.lower()).ratio()
            if r > ratio and r >= _FUZZY_THRESHOLD:
                best, ratio = c, r
        return best
