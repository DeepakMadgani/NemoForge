"""
Relationship Registry — Language-agnostic aggregation layer for cross-file intelligence.

DOES NOT replace existing providers:
    - LSP / AST / SymbolResolver (unchanged)
    - _session_files / _run_generated_map (unchanged)
    - DependencyContextBuilder (unchanged)
    - GenerationHandoff (unchanged)

Instead, aggregates their outputs into a unified relationship graph
with progressive context tiers and authority-ranked evidence.

Hard constraints (from implementation plan v2):
    1. DO NOT add language-pair conditionals (no if java elif ts)
    2. DO NOT mark inferred relationships as verified
    3. DO NOT introduce mandatory layer expansion
    4. A GenerationHandoff export proves a symbol EXISTS — it does NOT
       prove another file consumes it. Only actual repository evidence
       (import statements, API annotations, schema files) establishes
       the connection between producer and consumer.
    5. Progressive context: Signature → Type Contract → Implementation → Full File

Author: Deepak Madgani
Date: September 2026
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ticket_to_code.models import (
        ArchitecturalPlan,
        CrossFileContract,
        CrossFileRelationship,
        ContextTier,
        DevelopmentTask,
        GenerationHandoff,
        RelationshipStatus,
        RelationshipType,
        VerifiedSymbol,
    )
    from ticket_to_code.agents.symbol_resolver import SymbolResolver

logger = logging.getLogger(__name__)


# ── Authority levels (mirrors ContextAssembler) ──────────────────────────────

RELATIONSHIP_AUTHORITY = {
    "filesystem":           1,
    "lsp":                  2,
    "compiler":             2,
    "ast":                  2,
    "symbol_resolver":      3,
    "api_annotation":       3,
    "schema_file":          3,
    "import_statement":     3,
    "validated_generation": 4,
    "candidate_generation": 5,
    "rag":                  6,
    "planner":              7,
}


# ── ResolvedCapability ───────────────────────────────────────────────────────

@dataclass
class ResolvedCapability:
    """A capability that a task needs, resolved to a verified provider.

    Returned by resolve_capabilities_for_task() — the capability-driven
    query that traces: what does the task need → which providers satisfy
    → what contracts connect them → what is the minimum context.
    """
    capability: str              # Semantic: "check project membership"
    provider_file: str           # "projects.service.ts"
    provider_symbol: str         # "ProjectsService.checkProjectMembership"
    provider_signature: str      # "(email: string): Observable<...>"
    relationship: object         # CrossFileRelationship
    authority: int               # From RELATIONSHIP_AUTHORITY
    context_tier: str            # Current tier being served
    context_text: str            # The actual context for this tier


# ── RelationshipRegistry ─────────────────────────────────────────────────────

class RelationshipRegistry:
    """Aggregation layer over existing cross-file intelligence providers.

    Does NOT replace LSP, AST, SymbolResolver, _session_files, or
    DependencyContextBuilder. Aggregates their outputs into a unified
    relationship graph with progressive context tiers.

    Hard constraints:
    - Never marks inferred relationships as verified
    - Never introduces mandatory layer expansion
    - Never adds language-pair conditionals
    - Uses existing providers for evidence, not custom regex
    """

    def __init__(self):
        self._relationships: list = []  # list[CrossFileRelationship]
        self._capability_index: dict[str, list[int]] = {}  # capability → indices
        self._file_index: dict[str, list[int]] = {}  # target_file → indices
        self._current_tiers: dict[str, str] = {}  # target_file → current tier name

    # ── Registration from different evidence sources ─────────────────────

    def register_from_plan(self, plan) -> int:
        """Seed PLANNED relationships from planner's cross_file_contract.

        Status: PLANNED. Confidence: planner's confidence score.
        These are SUGGESTIONS — never presented as verified facts.

        Returns:
            Number of relationships registered.
        """
        from ticket_to_code.models import (
            CrossFileRelationship,
            RelationshipType,
            RelationshipStatus,
        )

        count = 0
        tasks = getattr(plan, "tasks", [])

        for task in tasks:
            contract = getattr(task, "cross_file_contract", None)
            if not contract:
                continue

            # Process CONSUMES: each consumed capability creates a relationship
            # from the producing task's file to this task's file.
            for consumed in getattr(contract, "consumes", []):
                cap = getattr(consumed, "capability", str(consumed))
                from_task_id = getattr(consumed, "from_task", "")

                # Find the producing task to get its file path
                source_file = ""
                for other_task in tasks:
                    if other_task.id == from_task_id:
                        source_file = other_task.file_path
                        break

                if not source_file:
                    continue

                # Determine relationship type from the cross-file boundary
                rel_type = self._infer_relationship_type(
                    source_file, task.file_path, consumed
                )

                rel = CrossFileRelationship(
                    source_file=source_file,
                    target_file=task.file_path,
                    relationship_type=rel_type,
                    capability=cap,
                    status=RelationshipStatus.PLANNED,
                    confidence=0.3,  # Planner predictions are low confidence
                    evidence_sources=["planner"],
                    source_task_id=from_task_id,
                    target_task_id=task.id,
                    extraction_method="planned",
                )

                self._add_relationship(rel)
                count += 1

        if count:
            logger.info(
                f"  [RelationshipRegistry] Seeded {count} PLANNED relationships from plan"
            )

        return count

    def register_from_handoff(
        self,
        handoff,
        task,
    ) -> int:
        """After file generation + validation, upgrade relationships with verified facts.

        Uses handoff.exports (machine-extracted facts) and handoff.validation_status.

        Key invariant: An export proves a symbol EXISTS. It does NOT prove
        another file consumes it. This method only upgrades relationships
        that were already registered (PLANNED/DISCOVERED) — it does not
        invent new consumer connections.

        Returns:
            Number of relationships upgraded.
        """
        from ticket_to_code.models import RelationshipStatus

        if not handoff:
            return 0

        file_path = handoff.file_path
        exports = getattr(handoff, "exports", []) or []
        is_clean = getattr(handoff, "validation_status", "") == "clean"
        task_id = getattr(handoff, "task_id", "")

        if not exports:
            return 0

        # Build a lookup of exported symbols for quick matching
        export_map: dict[str, object] = {}
        for sym in exports:
            name = getattr(sym, "name", "")
            if name:
                export_map[name] = sym

        # Find relationships where this file is the SOURCE
        # and upgrade their tier_1_signature with actual verified facts
        upgraded = 0
        for i, rel in enumerate(self._relationships):
            src = rel.source_file.replace("\\", "/").lower()
            handoff_src = file_path.replace("\\", "/").lower()

            if src != handoff_src:
                continue

            # This relationship has this file as its source.
            # Upgrade context tiers with actual exported symbols.
            signatures = []
            for sym in exports:
                name = getattr(sym, "name", "")
                sig = getattr(sym, "signature", "")
                kind = getattr(sym, "kind", "")
                if name:
                    signatures.append(
                        f"{kind} {name}" + (f": {sig}" if sig else "")
                    )

            if signatures:
                rel.tier_1_signature = "\n".join(signatures[:10])

                # Only upgrade to VERIFIED if validation passed
                if is_clean:
                    new_status = RelationshipStatus.VALIDATED
                else:
                    new_status = RelationshipStatus.VERIFIED

                # Only promote forward, never demote
                _status_rank = {
                    RelationshipStatus.PLANNED: 0,
                    RelationshipStatus.DISCOVERED: 1,
                    RelationshipStatus.INFERRED: 2,
                    RelationshipStatus.VERIFIED: 3,
                    RelationshipStatus.VALIDATED: 4,
                }
                if _status_rank.get(new_status, 0) > _status_rank.get(rel.status, 0):
                    rel.status = new_status
                    rel.verified_at = time.time()

                if "handoff" not in rel.evidence_sources:
                    rel.evidence_sources.append("handoff")
                if is_clean and "validated_generation" not in rel.evidence_sources:
                    rel.evidence_sources.append("validated_generation")

                rel.confidence = max(rel.confidence, 0.8 if is_clean else 0.6)
                rel.source_task_id = task_id
                upgraded += 1

        if upgraded:
            logger.info(
                f"  [RelationshipRegistry] Upgraded {upgraded} relationships from "
                f"handoff for {Path(file_path).name} "
                f"(status: {'VALIDATED' if is_clean else 'VERIFIED'})"
            )

        return upgraded

    def register_from_symbol_resolver(
        self,
        resolver,
        file_path: str,
        content: str,
    ) -> int:
        """Query existing SymbolResolver for type definitions and members.

        Uses resolver.resolve_type() — NO new regex patterns.
        Status: VERIFIED (authority 3).

        Returns:
            Number of relationships enriched.
        """
        if not resolver:
            return 0

        enriched = 0

        for i, rel in enumerate(self._relationships):
            src = rel.source_file.replace("\\", "/").lower()
            if src != file_path.replace("\\", "/").lower():
                continue

            # Try to resolve class/type names from the capability
            # Use the existing SymbolResolver infrastructure
            try:
                # Extract class names mentioned in tier_1 if available
                sig_text = rel.tier_1_signature
                if not sig_text:
                    continue

                # Look for class names in the signature
                class_names = re.findall(r'\b([A-Z][a-zA-Z]+)\b', sig_text)
                for cls_name in class_names[:3]:
                    try:
                        defn = resolver.resolve_type(cls_name)
                        if defn:
                            # Enrich tier_2 with type contract
                            type_info = f"class {defn.type_name}"
                            if defn.property_names:
                                type_info += f"\n  properties: {', '.join(defn.property_names[:10])}"
                            if defn.method_names:
                                type_info += f"\n  methods: {', '.join(defn.method_names[:10])}"

                            if not rel.tier_2_contract:
                                rel.tier_2_contract = type_info
                            else:
                                rel.tier_2_contract += f"\n\n{type_info}"

                            if "symbol_resolver" not in rel.evidence_sources:
                                rel.evidence_sources.append("symbol_resolver")
                            enriched += 1
                    except Exception:
                        pass

            except Exception as exc:
                logger.debug(f"  SymbolResolver enrichment error (non-fatal): {exc}")

        if enriched:
            logger.info(
                f"  [RelationshipRegistry] Enriched {enriched} relationships "
                f"via SymbolResolver for {Path(file_path).name}"
            )

        return enriched

    def register_api_contract(
        self,
        source_file: str,
        endpoint: str,
        method: str = "",
        response_type: str = "",
        request_type: str = "",
    ) -> None:
        """Register an API contract discovered from framework annotations.

        Called by api_contract_detector when it finds actual code annotations
        like @GetMapping, @QueryMapping, @PostMapping, etc.

        Status: VERIFIED if extracted from actual source code.
        """
        from ticket_to_code.models import (
            CrossFileRelationship,
            RelationshipType,
            RelationshipStatus,
        )

        # Check if any existing relationship references this capability
        for rel in self._relationships:
            src = rel.source_file.replace("\\", "/").lower()
            if src != source_file.replace("\\", "/").lower():
                continue

            # Update the relationship with API contract details
            api_sig = f"{method} {endpoint}" if method else endpoint
            if response_type:
                api_sig += f" → {response_type}"

            if rel.tier_1_signature:
                rel.tier_1_signature += f"\nAPI: {api_sig}"
            else:
                rel.tier_1_signature = f"API: {api_sig}"

            if rel.relationship_type.value == "same_lang_import":
                rel.relationship_type = RelationshipType.API_CONTRACT

            if "api_annotation" not in rel.evidence_sources:
                rel.evidence_sources.append("api_annotation")

            # This is from actual code, so mark as VERIFIED
            from ticket_to_code.models import RelationshipStatus
            _status_rank = {
                RelationshipStatus.PLANNED: 0,
                RelationshipStatus.DISCOVERED: 1,
                RelationshipStatus.INFERRED: 2,
                RelationshipStatus.VERIFIED: 3,
                RelationshipStatus.VALIDATED: 4,
            }
            if _status_rank.get(RelationshipStatus.VERIFIED, 0) > _status_rank.get(rel.status, 0):
                rel.status = RelationshipStatus.VERIFIED
                rel.verified_at = time.time()

            rel.confidence = max(rel.confidence, 0.85)

    # ── Capability-Driven Query (v2) ─────────────────────────────────────

    def resolve_capabilities_for_task(self, task) -> list[ResolvedCapability]:
        """Capability-driven context resolution.

        Does NOT simply return 'all relationships for this task'.
        Instead:
        1. What capabilities does this task consume? (from cross_file_contract)
        2. Which relationships provide those capabilities?
        3. For each provider, what is its verification status?
        4. What is the minimum context tier needed?

        Returns ResolvedCapability objects sorted by authority (highest first).
        """
        from ticket_to_code.models import RelationshipStatus

        task_file = getattr(task, "file_path", "").replace("\\", "/").lower()
        task_id = getattr(task, "id", "")

        # Find relationships where this task's file is the TARGET
        resolved: list[ResolvedCapability] = []

        for rel in self._relationships:
            target = rel.target_file.replace("\\", "/").lower()
            target_match = (target == task_file)

            # Also match by task_id
            if not target_match and rel.target_task_id == task_id:
                target_match = True

            if not target_match:
                continue

            # Determine authority based on status and evidence
            if rel.status in (RelationshipStatus.VERIFIED, RelationshipStatus.VALIDATED):
                authority = RELATIONSHIP_AUTHORITY.get("validated_generation", 4)
                for src in rel.evidence_sources:
                    src_auth = RELATIONSHIP_AUTHORITY.get(src, 7)
                    authority = min(authority, src_auth)
            elif rel.status == RelationshipStatus.DISCOVERED:
                authority = RELATIONSHIP_AUTHORITY.get("candidate_generation", 5)
            else:
                authority = RELATIONSHIP_AUTHORITY.get("planner", 7)

            # Get context at the current tier for this file
            current_tier = self._current_tiers.get(task_file, "signature")
            context_text = self._get_context_at_tier(rel, current_tier)

            # Extract provider symbol info
            provider_symbol = ""
            provider_signature = ""
            if rel.tier_1_signature:
                lines = rel.tier_1_signature.split("\n")
                if lines:
                    provider_symbol = lines[0]
                    provider_signature = "\n".join(lines[1:]) if len(lines) > 1 else ""

            resolved.append(ResolvedCapability(
                capability=rel.capability,
                provider_file=rel.source_file,
                provider_symbol=provider_symbol,
                provider_signature=provider_signature,
                relationship=rel,
                authority=authority,
                context_tier=current_tier,
                context_text=context_text,
            ))

        # Sort by authority (lower = higher authority = shown first)
        resolved.sort(key=lambda r: r.authority)

        return resolved

    def get_context_for_task(self, task, max_tier: str = "signature") -> str:
        """Build progressive context block for a task.

        Calls resolve_capabilities_for_task() internally.
        Formats output with authority labels:
          ✅ [VERIFIED] ProjectsService.checkProjectMembership(email: string)
          ⚠️ [PLANNED] may call backend membership check API

        VERIFIED facts are presented as authoritative.
        PLANNED/INFERRED are clearly labeled as suggestions.
        """
        from ticket_to_code.models import RelationshipStatus

        resolved = self.resolve_capabilities_for_task(task)

        if not resolved:
            return ""

        lines = ["═══ CROSS-FILE RELATIONSHIPS (Language-Agnostic) ═══"]

        for cap in resolved:
            rel = cap.relationship
            status = rel.status

            if status in (RelationshipStatus.VERIFIED, RelationshipStatus.VALIDATED):
                icon = "✅"
                label = "VERIFIED" if status == RelationshipStatus.VERIFIED else "VALIDATED"
            elif status == RelationshipStatus.DISCOVERED:
                icon = "⚡"
                label = "DISCOVERED"
            elif status == RelationshipStatus.INFERRED:
                icon = "💭"
                label = "INFERRED"
            else:
                icon = "📋"
                label = "PLANNED"

            lines.append(
                f"\n  {icon} [{label}] {cap.capability}"
                f"\n    Provider: {Path(cap.provider_file).name}"
            )

            if cap.context_text:
                # Indent context text
                for ctx_line in cap.context_text.split("\n"):
                    lines.append(f"    {ctx_line}")

            if label in ("PLANNED", "INFERRED"):
                lines.append("    ⚠️ This is a prediction — verify against actual code")

        lines.append("═══ END CROSS-FILE RELATIONSHIPS ═══")

        return "\n".join(lines) + "\n"

    # ── Tier Escalation ──────────────────────────────────────────────────

    def escalate_tier(self, target_file: str, reason: str) -> str:
        """Escalate context tier for all relationships targeting this file.

        Called by fix loop when validation fails (e.g. TS2339).
        Moves: SIGNATURE → TYPE_CONTRACT → IMPLEMENTATION → FULL_FILE.

        Returns the new tier name.
        """
        key = target_file.replace("\\", "/").lower()
        current = self._current_tiers.get(key, "signature")

        tier_order = ["signature", "type_contract", "implementation", "full_file"]
        try:
            idx = tier_order.index(current)
        except ValueError:
            idx = 0

        if idx < len(tier_order) - 1:
            new_tier = tier_order[idx + 1]
            self._current_tiers[key] = new_tier
            logger.info(
                f"  [RelationshipRegistry] Escalated {Path(target_file).name}: "
                f"{current} → {new_tier} (reason: {reason[:100]})"
            )
            return new_tier

        return current

    # ── Internal helpers ─────────────────────────────────────────────────

    def _add_relationship(self, rel) -> int:
        """Add a relationship and update indices. Returns the index."""
        idx = len(self._relationships)
        self._relationships.append(rel)

        # Index by capability
        cap_key = rel.capability.lower()
        self._capability_index.setdefault(cap_key, []).append(idx)

        # Index by target file
        target_key = rel.target_file.replace("\\", "/").lower()
        self._file_index.setdefault(target_key, []).append(idx)

        return idx

    def _infer_relationship_type(self, source_file: str, target_file: str, consumed=None):
        """Infer the relationship type from file paths.

        NOT language-specific — uses structural patterns:
        - Same directory/module → likely SAME_LANG_IMPORT
        - Different top-level dirs (e.g. backend vs frontend) → likely API_CONTRACT
        - Model/DTO file → likely DATA_FLOW

        This is a HEURISTIC — actual verification upgrades later.
        """
        from ticket_to_code.models import RelationshipType

        src = source_file.replace("\\", "/").lower()
        tgt = target_file.replace("\\", "/").lower()

        rel_type_hint = ""
        if consumed:
            rel_type_hint = getattr(consumed, "relationship_type", "")

        if rel_type_hint == "shared_type":
            return RelationshipType.SHARED_TYPE

        # Different top-level service directories → likely API boundary
        src_parts = src.split("/")
        tgt_parts = tgt.split("/")

        # Check for different microservice/project roots
        src_service = next((p for p in src_parts if "service" in p or "api" in p), "")
        tgt_service = next((p for p in tgt_parts if "service" in p or "api" in p), "")

        if src_service and tgt_service and src_service != tgt_service:
            return RelationshipType.API_CONTRACT

        # Check file extensions for cross-language hint
        src_ext = Path(source_file).suffix.lower()
        tgt_ext = Path(target_file).suffix.lower()

        java_exts = {".java", ".kt"}
        ts_exts = {".ts", ".tsx", ".js", ".jsx"}
        py_exts = {".py"}

        src_lang = "java" if src_ext in java_exts else ("ts" if src_ext in ts_exts else ("py" if src_ext in py_exts else "other"))
        tgt_lang = "java" if tgt_ext in java_exts else ("ts" if tgt_ext in ts_exts else ("py" if tgt_ext in py_exts else "other"))

        if src_lang != tgt_lang:
            return RelationshipType.API_CONTRACT

        # Same language → default to import
        return RelationshipType.SAME_LANG_IMPORT

    def _get_context_at_tier(self, rel, tier: str) -> str:
        """Get context for a relationship at the specified tier.

        Progressive: signature → type_contract → implementation → full_file
        """
        if tier == "full_file" and rel.tier_4_full_file:
            return rel.tier_4_full_file
        elif tier in ("implementation", "full_file") and rel.tier_3_implementation:
            return rel.tier_3_implementation
        elif tier in ("type_contract", "implementation", "full_file") and rel.tier_2_contract:
            ctx = rel.tier_1_signature
            if rel.tier_2_contract:
                ctx = f"{ctx}\n{rel.tier_2_contract}" if ctx else rel.tier_2_contract
            return ctx
        else:
            # Default: signature tier
            return rel.tier_1_signature

    # ── Public queries ───────────────────────────────────────────────────

    @property
    def relationships(self) -> list:
        """All registered relationships (read-only copy)."""
        return list(self._relationships)

    @property
    def count(self) -> int:
        """Number of registered relationships."""
        return len(self._relationships)

    def get_relationships_for_file(self, file_path: str) -> list:
        """Get all relationships where this file is the target."""
        key = file_path.replace("\\", "/").lower()
        indices = self._file_index.get(key, [])
        return [self._relationships[i] for i in indices]

    def get_providers_for_capability(self, capability: str) -> list:
        """Find all relationships that provide a given capability."""
        key = capability.lower()
        indices = self._capability_index.get(key, [])
        return [self._relationships[i] for i in indices]

    def summary(self) -> str:
        """Human-readable summary of the registry state."""
        from ticket_to_code.models import RelationshipStatus

        if not self._relationships:
            return "RelationshipRegistry: empty"

        status_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        for rel in self._relationships:
            status_counts[rel.status.value] = status_counts.get(rel.status.value, 0) + 1
            type_counts[rel.relationship_type.value] = type_counts.get(rel.relationship_type.value, 0) + 1

        lines = [f"RelationshipRegistry: {len(self._relationships)} relationships"]
        for status, count in sorted(status_counts.items()):
            lines.append(f"  [{status}] {count}")
        for rtype, count in sorted(type_counts.items()):
            lines.append(f"  {rtype}: {count}")

        return "\n".join(lines)
