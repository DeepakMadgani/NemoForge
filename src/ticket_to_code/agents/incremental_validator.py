"""
Incremental Validator — Dependency-aware post-generation validation.

After each file is generated, validates:
1. Blueprint contract satisfaction (did the generator produce required symbols?)
2. Dependency health (did this change break consumers?)
3. Semantic contract verification (are all contract items still satisfied?)

Results feed back into ImplementationState for downstream tasks.

Author: Deepak Madgani
Date: August 2026
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ticket_to_code.agents.implementation_state import (
        ImplementationState,
        ValidationResult,
    )
    from ticket_to_code.agents.semantic_contract import (
        SemanticContract,
        SemanticContractBuilder,
    )
    from ticket_to_code.agents.symbol_resolver import SymbolResolver

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class IncrementalValidationResult:
    """Result of incremental validation for a single generated file.

    Extends the base ValidationResult with blueprint-aware diagnostics.
    """
    file_path: str
    is_clean: bool
    blueprint_satisfied: list[str] = field(default_factory=list)   # symbols confirmed present
    blueprint_missing: list[str] = field(default_factory=list)     # symbols NOT found in output
    dependency_issues: list[str] = field(default_factory=list)     # downstream breakage detected
    contract_broken: list[str] = field(default_factory=list)       # semantic contract violations
    architecture_violations: list[str] = field(default_factory=list)  # domain ownership violations
    unresolved_references: list[str] = field(default_factory=list)  # outbound calls to non-existent members
    shared_type_violations: list[str] = field(default_factory=list)  # destructive shared-type changes
    validated_scope: str = "file"                                   # "file", "module", "dependency_chain"

    @property
    def summary(self) -> str:
        parts = []
        if self.blueprint_satisfied:
            parts.append(f"✓ {len(self.blueprint_satisfied)} blueprints satisfied")
        if self.blueprint_missing:
            parts.append(f"✗ {len(self.blueprint_missing)} blueprints missing")
        if self.dependency_issues:
            parts.append(f"⚠ {len(self.dependency_issues)} dependency issues")
        if self.unresolved_references:
            parts.append(f"✗ {len(self.unresolved_references)} unresolved references")
        if self.shared_type_violations:
            parts.append(f"✗ {len(self.shared_type_violations)} shared-type violations")
        if self.contract_broken:
            parts.append(f"✗ {len(self.contract_broken)} contract violations")
        if self.architecture_violations:
            parts.append(f"🏗️ {len(self.architecture_violations)} architecture violations")
        return " | ".join(parts) if parts else "clean"


# ── Validator ─────────────────────────────────────────────────────────────────

class IncrementalValidator:
    """Dependency-aware post-generation validator.

    Called after each file is generated to check:
    1. Did the generator produce the symbols it was supposed to? (blueprints)
    2. Did this change break anything downstream? (dependency health)
    3. Is the semantic contract still satisfied? (contract verification)

    Usage:
        validator = IncrementalValidator(
            impl_state=impl_state,
            symbol_resolver=resolver,
        )
        result = validator.validate_generated(
            task_id="task_3",
            file_path="src/services/member.service.ts",
        )
    """

    def __init__(
        self,
        impl_state: "ImplementationState",
        symbol_resolver: Optional["SymbolResolver"] = None,
        workspace_path: Optional[Path] = None,
    ):
        self._state = impl_state
        self._resolver = symbol_resolver
        self._workspace = workspace_path

    def validate_generated(
        self,
        task_id: str,
        file_path: str,
        contract: Optional["SemanticContract"] = None,
    ) -> IncrementalValidationResult:
        """Validate a freshly generated file.

        Args:
            task_id: The task that just generated this file.
            file_path: Path to the generated file.
            contract: Optional pre-generation SemanticContract to verify.

        Returns:
            IncrementalValidationResult with detailed diagnostics.
        """
        content = self._state.get_generated_content(file_path)
        if content is None:
            # File wasn't recorded in state — nothing to validate
            return IncrementalValidationResult(
                file_path=file_path,
                is_clean=True,
                validated_scope="none",
            )

        result = IncrementalValidationResult(file_path=file_path, is_clean=True)

        # ── Phase 1: Blueprint satisfaction ───────────────────────────────
        self._check_blueprints(task_id, content, result)

        # ── Phase 2: Semantic contract verification ──────────────────────
        if contract:
            self._check_contract(contract, result)

        # ── Phase 3: Dependency health ───────────────────────────────────
        self._check_dependency_health(file_path, content, result)

        # ── Phase 3.5: Generated outbound-reference validation ───────────
        # Catches hallucinated calls (e.g. repo.findByUserIdAndProjectId)
        # where the receiver type is known but the member does not exist.
        self._check_generated_references(content, result)

        # ── Phase 4: Architecture ownership ──────────────────────────────
        self._check_architecture(file_path, result)

        # ── Record in ImplementationState ────────────────────────────────
        self._record_result(file_path, result)

        # ── Transition blueprints on success ─────────────────────────────
        if result.is_clean and result.blueprint_satisfied:
            self._transition_to_validated(task_id)

        status = "✓ CLEAN" if result.is_clean else "✗ ISSUES"
        logger.info(
            f"  [IncrementalValidator] {file_path}: {status} — {result.summary}"
        )
        return result

    # ── Phase implementations ─────────────────────────────────────────────

    def _check_blueprints(
        self,
        task_id: str,
        content: str,
        result: IncrementalValidationResult,
    ) -> None:
        """Check that all produced blueprints appear in the generated content."""
        produces, _ = self._state.get_blueprints_for_task(task_id)
        if not produces:
            return

        for bp in produces:
            # Check if the symbol exists in the generated content
            pattern = r'\b' + re.escape(bp.symbol_name) + r'\b'
            if re.search(pattern, content):
                result.blueprint_satisfied.append(
                    f"{bp.owner_class}.{bp.symbol_name}" if bp.owner_class
                    else bp.symbol_name
                )
            else:
                result.blueprint_missing.append(
                    f"{bp.owner_class}.{bp.symbol_name}" if bp.owner_class
                    else bp.symbol_name
                )
                result.is_clean = False

    def _check_contract(
        self,
        contract: "SemanticContract",
        result: IncrementalValidationResult,
    ) -> None:
        """Verify all contract items that apply to this file."""
        for item in contract.items:
            if item.is_satisfied is False:
                result.contract_broken.append(str(item))
                result.is_clean = False

    def _check_dependency_health(
        self,
        file_path: str,
        content: str,
        result: IncrementalValidationResult,
    ) -> None:
        """Check if this file's change breaks downstream consumers.

        For each blueprint that this file produces (consumed by other tasks),
        verify the symbol is still present in the generated content.
        """
        result.validated_scope = "dependency_chain"

        # Find all blueprints where this file is the source
        for bp in self._state.signature_blueprints:
            if bp.file_path.replace("\\", "/").lower() != file_path.replace("\\", "/").lower():
                continue
            if not bp.consumed_by_tasks:
                continue

            # This blueprint is consumed by other tasks — the symbol must exist
            pattern = r'\b' + re.escape(bp.symbol_name) + r'\b'
            if not re.search(pattern, content):
                consumers = ", ".join(bp.consumed_by_tasks[:3])
                result.dependency_issues.append(
                    f"Symbol '{bp.symbol_name}' expected by [{consumers}] "
                    f"not found in generated {Path(file_path).name}"
                )
                result.is_clean = False

    def _check_generated_references(
        self,
        content: str,
        result: IncrementalValidationResult,
    ) -> None:
        """Verify every outbound member reference resolves to a real symbol.

        Uses SymbolResolver (repository truth) + ImplementationState
        (symbols generated+verified this run). Unresolved references block
        handoff but do NOT roll back the file — the file stays on disk so the
        (attribution-aware) error resolver can still act on it.
        """
        if not self._resolver:
            return
        try:
            from ticket_to_code.agents.generated_reference_validator import (
                GeneratedReferenceValidator,
            )
            ref_result = GeneratedReferenceValidator(
                symbol_resolver=self._resolver,
                impl_state=self._state,
            ).validate(content)
        except Exception as exc:
            logger.debug(f"  [IncrementalValidator] reference check skipped: {exc}")
            return

        for c in ref_result.unresolved:
            hint = f" — did you mean '{c.close_match}'?" if c.close_match else ""
            result.unresolved_references.append(
                f"{c.resolved_type}.{c.member}(){hint}"
            )
        if ref_result.unresolved:
            result.is_clean = False

    def _record_result(
        self,
        file_path: str,
        result: IncrementalValidationResult,
    ) -> None:
        """Record the validation result in ImplementationState."""
        from ticket_to_code.agents.implementation_state import ValidationResult

        vr = ValidationResult(
            file_path=file_path,
            is_clean=result.is_clean,
            diagnostics=[],  # Blueprint issues aren't StructuredDiagnostics
            validation_scope=result.validated_scope,
        )
        self._state.record_validation(file_path, vr)

    def _transition_to_validated(self, task_id: str) -> None:
        """Transition IMPLEMENTED_CANDIDATE blueprints to IMPLEMENTED.

        Key invariant: Only validation can promote IMPLEMENTED_CANDIDATE.
        The code generator sets IMPLEMENTED_CANDIDATE, NOT IMPLEMENTED.
        This method provides the promotion that makes the symbol authoritative.
        """
        from ticket_to_code.models import BlueprintStatus

        for bp in self._state.signature_blueprints:
            if bp.created_by_task == task_id and bp.status == BlueprintStatus.IMPLEMENTED_CANDIDATE:
                bp.status = BlueprintStatus.IMPLEMENTED
                logger.debug(
                    f"  [IncrementalValidator] Blueprint "
                    f"{bp.owner_class}.{bp.symbol_name} → IMPLEMENTED (validated)"
                )

    # ── Phase 4: Architecture ownership check ─────────────────────────────

    def _check_architecture(
        self,
        file_path: str,
        result: IncrementalValidationResult,
    ) -> None:
        """Post-generation check: did we accidentally generate code in a wrong-domain file?

        This is the last line of defense. Even if the planner and gating missed it,
        this check catches any generated file that belongs to the wrong bounded context.
        Only produces hard failures for DECLARATIVE/VERIFIED ownership facts.
        """
        if not self._workspace:
            return

        try:
            from ticket_to_code.agents.architecture_model import ArchitectureModel

            # Load architecture model
            _arch_model = None
            for _yaml_loc in [
                self._workspace / "brain" / "knowledge" / "architecture_model.yaml",
                Path(__file__).parent.parent / "config" / "architecture_model.yaml",
            ]:
                if _yaml_loc.exists():
                    _arch_model = ArchitectureModel.load_from_yaml(str(_yaml_loc))
                    break

            if not _arch_model or not _arch_model.services:
                return

            # Check if this file belongs to a known service
            svc = _arch_model.get_service_for_file(file_path)
            if not svc:
                return  # Unknown service — no opinion

            # Check if the file's service is explicitly listed as reference_only
            # for ANY capability — this catches files that slipped through.
            for cap in _arch_model.get_all_capabilities():
                if svc.name in cap.reference_only and cap.is_hard_authority:
                    violation = (
                        f"Architecture violation: {file_path} belongs to '{svc.name}' "
                        f"which is REFERENCE_ONLY for capability '{cap.name}' "
                        f"(owned by {', '.join(cap.primary_owners)})"
                    )
                    result.architecture_violations.append(violation)
                    result.is_clean = False
                    logger.warning(f"  [IncrementalValidator] 🏗️ {violation}")

        except Exception as e:
            logger.debug(f"  [IncrementalValidator] Architecture check skipped: {e}")
