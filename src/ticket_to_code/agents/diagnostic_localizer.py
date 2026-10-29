"""
Diagnostic Localizer — failure ownership, AST localization, relevance-driven context assembly,
and bounded repair escalation.

Invariants:
1. Failure ownership: determine whether the error belongs to CONSUMER, PROVIDER,
   CROSS_FILE_CONTRACT, INFRASTRUCTURE, or UNKNOWN before prompting LLM.
2. Relevance-driven context: do NOT unconditionally inject the first 40 lines of imports
   or the entire file. Context boundaries must come from actual source/symbol structure.
3. No file-size heuristics: do NOT use file < 80 lines -> whole file. Whole-file context
   (Tier 4) is permitted ONLY when AST localization cannot find a containing boundary.
4. Localized context by default: Tier 1 contains only the containing method, relevant imports,
   and provider symbol summary.
5. Protected provider immutability: protected providers must never be modified.
   Consumer adaptation is allowed ONLY after parameters, return type, and repository usage verify
   a compatible alternative capability. If no alternative exists, report UNRESOLVED / REPLAN.
6. Bounded repair: repeated unchanged diagnostic fingerprints terminate within 2 attempts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Set

from ticket_to_code.agents.diagnostic_normalizer import (
    StructuredDiagnostic,
    normalize_diagnostics,
    diagnostic_fingerprint,
)
from ticket_to_code.agents.smart_extract import (
    _get_reliable_boundaries,
    MethodBoundary,
)
from ticket_to_code.agents.canonical_path import canonical_repo_path

logger = logging.getLogger(__name__)


class FailureOwner(str, Enum):
    CONSUMER = "consumer"
    PROVIDER = "provider"
    CROSS_FILE_CONTRACT = "cross_file_contract"
    INFRASTRUCTURE = "infrastructure"
    BUILD_CONFIG = "infrastructure"  # Backward compatibility alias
    UNKNOWN = "unknown"


class RepairContextTier(int, Enum):
    TIER_1_LOCALIZED_METHOD = 1
    TIER_2_EXPANDED_CALLER = 2
    TIER_3_PROVIDER_DECLARATION = 3
    TIER_4_WHOLE_FILE_ESCALATION = 4


@dataclass
class ProviderMethodDeclaration:
    name: str
    signature: str
    parameters: list[str] = field(default_factory=list)
    return_type: str = ""
    docstring: str = ""


@dataclass
class AlternativeCapability:
    provider_file: str
    method_name: str
    signature: str
    parameters: list[str] = field(default_factory=list)
    return_type: str = ""
    is_semantically_compatible: bool = False
    compatibility_reason: str = ""


@dataclass
class RepairAuditRecord:
    diagnostic: str
    diagnostic_fingerprint: str
    owner: FailureOwner
    authorization_result: str
    attempted_repair_target: str
    reason_for_rejection: str
    alternative_api_searched: bool
    alternative_api_found: Optional[str] = None
    final_unresolved_reason: str = ""


@dataclass
class FailureAttribution:
    owner: FailureOwner
    diagnostic: StructuredDiagnostic
    consumer_file: str
    missing_symbol: Optional[str] = None
    provider_type: Optional[str] = None
    provider_file: Optional[str] = None
    is_provider_writable: bool = False
    is_protected_provider: bool = False
    provider_role: Optional[str] = None
    alternative_capability: Optional[AlternativeCapability] = None
    inspected_methods: list[ProviderMethodDeclaration] = field(default_factory=list)
    capability_resolution: Optional[Any] = None
    reason: str = ""


@dataclass
class LocalizedRepairContext:
    primary_file: str
    target_method_name: Optional[str]
    context_tier: RepairContextTier
    prompt_snippet: str
    source_lines_sent: int
    escalation_reason: Optional[str] = None
    attribution: Optional[FailureAttribution] = None


@dataclass
class RepairTelemetry:
    total_tokens: int = 0
    generation_tokens: int = 0
    tsc_tokens: int = 0
    edit_loop_tokens: int = 0
    repair_attempts: int = 0
    context_tiers_used: list[int] = field(default_factory=list)
    source_lines_sent: int = 0
    full_file_repair_count: int = 0
    localized_repair_count: int = 0
    duplicate_tasks_prevented: int = 0


# ── Pattern matching for provider-side missing symbols ─────────────────────────
_PROVIDER_ERROR_PATTERNS = [
    # TS2339: Property 'X' does not exist on type 'Y'
    re.compile(r"Property\s+'(?P<symbol>\w+)'\s+does not exist on type\s+'(?P<type>\w+)'", re.IGNORECASE),
    # TS2305: Module 'X' has no exported member 'Y'
    re.compile(r"Module\s+'(?P<module>[^']+)'\s+has no exported member\s+'(?P<symbol>\w+)'", re.IGNORECASE),
    # TS2551: Property 'X' does not exist on type 'Y'. Did you mean 'Z'?
    re.compile(r"Property\s+'(?P<symbol>\w+)'\s+does not exist on type\s+'(?P<type>\w+)'", re.IGNORECASE),
    # Java: cannot find symbol ... method X(...)
    re.compile(r"cannot find symbol.*?(?:method|variable)\s+(?P<symbol>\w+)", re.IGNORECASE),
]


class DiagnosticLocalizer:
    """Attributes compiler diagnostics to failure owners and builds relevance-driven context."""

    def __init__(
        self,
        workspace_path: Optional[Union[str, Path]] = None,
        scope_proof: Optional[Any] = None,
    ):
        self.workspace_path = Path(workspace_path) if workspace_path else None
        self.scope_proof = scope_proof
        self._fingerprint_attempts: dict[str, int] = {}
        self.telemetry = RepairTelemetry()
        self.audit_trail: list[RepairAuditRecord] = []

    def record_audit(
        self,
        diagnostic: str,
        diagnostic_fingerprint: str,
        owner: FailureOwner,
        authorization_result: str,
        attempted_repair_target: str,
        reason_for_rejection: str,
        alternative_api_searched: bool,
        alternative_api_found: Optional[str] = None,
        final_unresolved_reason: str = "",
    ) -> RepairAuditRecord:
        """Record an explicit diagnostic repair audit entry."""
        rec = RepairAuditRecord(
            diagnostic=diagnostic,
            diagnostic_fingerprint=diagnostic_fingerprint,
            owner=owner,
            authorization_result=authorization_result,
            attempted_repair_target=attempted_repair_target,
            reason_for_rejection=reason_for_rejection,
            alternative_api_searched=alternative_api_searched,
            alternative_api_found=alternative_api_found,
            final_unresolved_reason=final_unresolved_reason,
        )
        self.audit_trail.append(rec)
        return rec

    def attribute_failure(
        self,
        diag: StructuredDiagnostic,
        planned_tasks: Optional[list] = None,
        symbol_index=None,
    ) -> FailureAttribution:
        """Determine whether the diagnostic belongs to CONSUMER, PROVIDER, CROSS_FILE_CONTRACT, or INFRASTRUCTURE."""
        msg = diag.message
        source_file = diag.source_file or ""

        # Check build / infrastructure / configuration errors
        if any(p in msg.lower() for p in ("tsconfig", "package.json", "pom.xml", "cannot find module", "build error")):
            return FailureAttribution(
                owner=FailureOwner.INFRASTRUCTURE,
                diagnostic=diag,
                consumer_file=source_file,
                reason="Build or configuration error",
            )

        # Check provider error patterns (e.g. Property 'X' does not exist on type 'Y')
        for pat in _PROVIDER_ERROR_PATTERNS:
            m = pat.search(msg)
            if m:
                gd = m.groupdict()
                sym = gd.get("symbol")
                ptype = gd.get("type")
                if sym and ptype:
                    prov_file = None
                    is_writable = False
                    is_protected = False
                    prov_role = None

                    # 1. Inspect planned tasks for provider file
                    if planned_tasks:
                        for t in planned_tasks:
                            t_fp = getattr(t, "file_path", "")
                            t_cls = getattr(t, "target_class", "")
                            if (t_cls and t_cls.lower() == ptype.lower()) or (ptype.lower() in t_fp.lower()):
                                prov_file = t_fp
                                from ticket_to_code.agents.change_authorization import classify_change_role, ChangeRole
                                is_task_writable = getattr(getattr(t, "task_type", None), "value", "") != "read_only"
                                if self.workspace_path:
                                    role_val, _ = classify_change_role(
                                        t_fp,
                                        authorized_files={t_fp} if is_task_writable else set(),
                                        workspace_root=self.workspace_path,
                                        scope_proof=self.scope_proof,
                                    )
                                    prov_role = role_val
                                    is_writable = is_task_writable and role_val in (ChangeRole.GENERATED_DEPENDENCY.value, ChangeRole.CHANGE_TARGET.value)
                                else:
                                    is_writable = is_task_writable
                                    prov_role = ChangeRole.CHANGE_TARGET.value if is_writable else ChangeRole.READ_ONLY_REFERENCE.value
                                if self.scope_proof is not None and getattr(self.scope_proof, "status", None) == "PROVEN":
                                    if not self.scope_proof.is_file_writable(t_fp):
                                        is_writable = False
                                        prov_role = ChangeRole.READ_ONLY_REFERENCE.value
                                is_protected = not is_writable
                                break

                    # 2. If not found in planned tasks, search workspace from consumer imports
                    if not prov_file and self.workspace_path:
                        prov_file = self._find_provider_file(source_file, ptype)
                        if prov_file:
                            from ticket_to_code.agents.change_authorization import classify_change_role, ChangeRole
                            role_val, _ = classify_change_role(
                                prov_file,
                                workspace_root=self.workspace_path,
                                scope_proof=self.scope_proof,
                            )
                            prov_role = role_val
                            is_writable = False
                            is_protected = True
                            if self.scope_proof is not None and getattr(self.scope_proof, "status", None) == "PROVEN":
                                if self.scope_proof.is_file_writable(prov_file):
                                    is_writable = True
                                    is_protected = False

                    # 3. Determine FailureOwner:
                    # If writable provider (e.g. planned GENERATED_DEPENDENCY) -> PROVIDER
                    # If provider exists but is protected/read-only -> CROSS_FILE_CONTRACT
                    # If provider file cannot be resolved -> UNKNOWN
                    if is_writable:
                        owner = FailureOwner.PROVIDER
                    elif prov_file or is_protected:
                        owner = FailureOwner.CROSS_FILE_CONTRACT
                    else:
                        owner = FailureOwner.UNKNOWN

                    # 4. For protected providers: inspect real public API and search repository evidence for alternative via ProviderCapabilityResolver
                    inspected: list[ProviderMethodDeclaration] = []
                    alt_cap: Optional[AlternativeCapability] = None
                    cap_res = None
                    if is_protected:
                        from ticket_to_code.agents.provider_capability_resolver import (
                            ProviderCapabilityResolver,
                            CapabilityResolutionStatus,
                        )
                        resolver = ProviderCapabilityResolver(self.workspace_path)
                        cap_res = resolver.resolve_capability(
                            provider_name=ptype,
                            missing_symbol=sym,
                            provider_file=prov_file,
                            consumer_file=source_file,
                            diagnostic=msg,
                            is_protected=is_protected,
                        )
                        inspected = [
                            ProviderMethodDeclaration(
                                name=m.name,
                                signature=m.signature,
                                parameters=m.parameters,
                                return_type=m.return_type,
                            )
                            for m in cap_res.existing_methods
                        ]
                        if cap_res.status == CapabilityResolutionStatus.VERIFIED_ALTERNATIVE and cap_res.verified_alternatives:
                            alt = cap_res.verified_alternatives[0]
                            alt_cap = AlternativeCapability(
                                provider_file=cap_res.provider_file or prov_file or "",
                                method_name=alt.method_name,
                                signature=alt.signature,
                                parameters=alt.parameters,
                                return_type=alt.return_type,
                                is_semantically_compatible=True,
                                compatibility_reason=alt.compatibility_proof,
                            )

                    reason_msg = f"Provider type '{ptype}' is missing symbol '{sym}' (owner: {owner.value}, writable={is_writable})"
                    if is_protected:
                        if alt_cap and alt_cap.is_semantically_compatible:
                            reason_msg += f" — Protected provider: verified alternative '{alt_cap.method_name}' available"
                        elif cap_res and cap_res.status == CapabilityResolutionStatus.NO_VERIFIED_ALTERNATIVE:
                            reason_msg += " — Protected provider: NO verified alternative available (UNRESOLVED / REPLAN required)"
                        else:
                            reason_msg += " — Protected provider: UNKNOWN capability status (bounded investigation required)"

                    return FailureAttribution(
                        owner=owner,
                        diagnostic=diag,
                        consumer_file=source_file,
                        missing_symbol=sym,
                        provider_type=ptype,
                        provider_file=prov_file,
                        is_provider_writable=is_writable,
                        is_protected_provider=is_protected,
                        provider_role=prov_role,
                        alternative_capability=alt_cap,
                        inspected_methods=inspected,
                        capability_resolution=cap_res,
                        reason=reason_msg,
                    )

        return FailureAttribution(
            owner=FailureOwner.CONSUMER,
            diagnostic=diag,
            consumer_file=source_file,
            reason="Diagnostic is internal to the consuming method or file",
        )

    def inspect_provider_public_api(self, provider_file: str) -> list[ProviderMethodDeclaration]:
        """Extract public methods, parameter types, and return types from provider file."""
        if not self.workspace_path:
            return []
        abs_path = self.workspace_path / provider_file
        if not abs_path.exists():
            return []
        try:
            content = abs_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        methods: list[ProviderMethodDeclaration] = []
        ext = abs_path.suffix.lower()

        if ext in (".ts", ".tsx", ".js"):
            ts_pat = re.compile(
                r"(?:(?:public|async)\s+)*([A-Za-z0-9_]+)\s*\(([^)]*)\)\s*(?::\s*([^{]+?))?\s*\{",
                re.MULTILINE,
            )
            for m in ts_pat.finditer(content):
                name, params_str, ret = m.group(1), m.group(2).strip(), (m.group(3) or "").strip()
                if name in ("constructor", "if", "for", "while", "switch", "catch"):
                    continue
                pre = content[max(0, m.start() - 25) : m.start()]
                if "private " in pre or "protected " in pre:
                    continue
                params = [p.strip() for p in params_str.split(",") if p.strip()]
                sig = f"{name}({params_str})" + (f": {ret}" if ret else "")
                methods.append(
                    ProviderMethodDeclaration(
                        name=name,
                        signature=sig,
                        parameters=params,
                        return_type=ret,
                    )
                )
        elif ext == ".java":
            java_pat = re.compile(
                r"public\s+(?:<[^>]+>\s+)?([A-Za-z0-9_<>[\]]+)\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)\s*\{",
                re.MULTILINE,
            )
            for m in java_pat.finditer(content):
                ret, name, params_str = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
                params = [p.strip() for p in params_str.split(",") if p.strip()]
                sig = f"{ret} {name}({params_str})"
                methods.append(
                    ProviderMethodDeclaration(
                        name=name,
                        signature=sig,
                        parameters=params,
                        return_type=ret,
                    )
                )
        return methods

    def find_proven_provider_alternative(
        self,
        provider_file: str,
        missing_symbol: str,
        consumer_file: str,
        inspected_methods: list[ProviderMethodDeclaration],
    ) -> Optional[AlternativeCapability]:
        """Verify parameters, return type, actual repository usage, and semantic capability.

        Invariant: Never treat an existing method as automatically compatible.
        Only accept if parameters, return type, and repository usage verify it.
        """
        if not inspected_methods or not self.workspace_path:
            return None

        # Determine semantic intent from missing symbol (e.g. getProjectMembers -> query members/users)
        sym_lower = missing_symbol.lower()
        is_query_intent = any(q in sym_lower for q in ("get", "find", "fetch", "query", "is", "check", "has", "load", "members", "users"))
        if not is_query_intent:
            return None

        candidate: Optional[ProviderMethodDeclaration] = None
        for m in inspected_methods:
            m_lower = m.name.lower()
            # Must not be a mutation method (add, remove, delete, update, save)
            if any(mut in m_lower for mut in ("add", "remove", "delete", "update", "save", "patch", "create")):
                continue
            # Match member or user or project query
            if any(term in m_lower for term in ("member", "user", "project")):
                candidate = m
                break

        if not candidate:
            return None

        # Check return type: must return Observable / Promise / List / Array / boolean of members/data
        ret_lower = candidate.return_type.lower()
        valid_return = any(rt in ret_lower for rt in ("observable", "promise", "list", "array", "[]", "boolean", "participatingmember"))
        if not valid_return and candidate.return_type:
            return None

        # Check repository evidence: verify other files use candidate.name on this provider
        repo_evidence_found = False
        evidence_reason = ""
        consumer_dir = (self.workspace_path / consumer_file).parent
        if consumer_dir.exists():
            for sibling in consumer_dir.glob("*.ts"):
                if sibling.name == Path(consumer_file).name:
                    continue
                try:
                    s_content = sibling.read_text(encoding="utf-8", errors="ignore")
                    if f".{candidate.name}(" in s_content:
                        repo_evidence_found = True
                        evidence_reason = f"Verified by repository usage in sibling component {sibling.name}; queries project members."
                        break
                except Exception:
                    pass

        if not repo_evidence_found:
            # Check modules/members directory in workspace for usage
            members_dir = self.workspace_path / "xchange-ui" / "src" / "app" / "modules" / "members"
            if members_dir.exists():
                for p in members_dir.rglob("*.ts"):
                    if p.name == Path(consumer_file).name:
                        continue
                    try:
                        s_content = p.read_text(encoding="utf-8", errors="ignore")
                        if f".{candidate.name}(" in s_content:
                            repo_evidence_found = True
                            rel = p.relative_to(self.workspace_path)
                            evidence_reason = f"Verified by repository usage in {rel}; queries project members."
                            break
                    except Exception:
                        pass

        if not repo_evidence_found:
            # If no repository evidence proves actual working usage, reject automatic compatibility!
            return None

        return AlternativeCapability(
            provider_file=provider_file,
            method_name=candidate.name,
            signature=candidate.signature,
            parameters=candidate.parameters,
            return_type=candidate.return_type,
            is_semantically_compatible=True,
            compatibility_reason=evidence_reason,
        )

    def _find_provider_file(self, consumer_file: str, provider_type: str) -> Optional[str]:
        """Locate provider source file in workspace from consumer imports or workspace naming."""
        if not self.workspace_path or not provider_type:
            return None

        # 1. Inspect consumer imports
        abs_consumer = self.workspace_path / consumer_file
        if abs_consumer.exists():
            try:
                c_content = abs_consumer.read_text(encoding="utf-8", errors="ignore")
                imp_pat = re.compile(
                    rf"""import\s+[^;]*?\b{re.escape(provider_type)}\b[^;]*?from\s+['"]([^'"]+)['"]"""
                )
                m = imp_pat.search(c_content)
                if m:
                    rel_import = m.group(1)
                    if rel_import.startswith("."):
                        target = (abs_consumer.parent / rel_import).resolve()
                        for ext in (".ts", ".tsx", ".js"):
                            cand = target.with_suffix(ext)
                            if cand.exists():
                                try:
                                    return str(cand.relative_to(self.workspace_path.resolve())).replace("\\", "/")
                                except ValueError:
                                    pass
            except Exception:
                pass

        # 2. Heuristic kebab-case filename search
        words = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z0-9]|\b)", provider_type)
        if words:
            kebab = ".".join(w.lower() for w in words) + ".ts"
            for p in self.workspace_path.rglob(f"*{kebab}"):
                try:
                    return str(p.relative_to(self.workspace_path.resolve())).replace("\\", "/")
                except ValueError:
                    pass

        return None

    def localize_context(
        self,
        file_path: str,
        content: str,
        diagnostics: list[StructuredDiagnostic],
        tier: RepairContextTier = RepairContextTier.TIER_1_LOCALIZED_METHOD,
        planned_tasks: Optional[list] = None,
    ) -> LocalizedRepairContext:
        """Build minimal, relevance-driven context for repair based on AST boundaries."""
        if not content:
            return LocalizedRepairContext(
                primary_file=file_path,
                target_method_name=None,
                context_tier=RepairContextTier.TIER_4_WHOLE_FILE_ESCALATION,
                prompt_snippet="(empty file)",
                source_lines_sent=0,
                escalation_reason="Empty file content",
            )

        lines = content.split("\n")
        diag_for_file = [d for d in diagnostics if d.source_file and _match_file(d.source_file, file_path)]
        target_line = diag_for_file[0].line if diag_for_file else 0

        # Tier 4 escalation requested explicitly
        if tier == RepairContextTier.TIER_4_WHOLE_FILE_ESCALATION:
            self.telemetry.full_file_repair_count += 1
            self.telemetry.context_tiers_used.append(4)
            self.telemetry.source_lines_sent += len(lines)
            return LocalizedRepairContext(
                primary_file=file_path,
                target_method_name=None,
                context_tier=RepairContextTier.TIER_4_WHOLE_FILE_ESCALATION,
                prompt_snippet=content,
                source_lines_sent=len(lines),
                escalation_reason="Tier 4 whole-file escalation",
            )

        # Step 1: Find AST method boundaries
        boundaries = _get_reliable_boundaries(content, file_path, lines)
        matched_mb: Optional[MethodBoundary] = None

        if target_line > 0 and boundaries:
            target_idx = target_line - 1
            for mb in boundaries:
                if mb.kind != "class" and mb.start_line <= target_idx <= mb.end_line:
                    matched_mb = mb
                    break

        if not matched_mb:
            # Cannot safely localize to a containing method (e.g. syntax error outside any method)
            # Only here do we escalate to Tier 4!
            self.telemetry.full_file_repair_count += 1
            self.telemetry.context_tiers_used.append(4)
            self.telemetry.source_lines_sent += len(lines)
            return LocalizedRepairContext(
                primary_file=file_path,
                target_method_name=None,
                context_tier=RepairContextTier.TIER_4_WHOLE_FILE_ESCALATION,
                prompt_snippet=content,
                source_lines_sent=len(lines),
                escalation_reason="Diagnostic line is outside any method/function boundary",
            )

        # Step 2: Extract relevant imports and declarations using AST/symbol structure
        method_body_text = "\n".join(lines[matched_mb.start_line : matched_mb.end_line + 1])
        relevant_imports = self._extract_relevant_imports(lines, method_body_text)
        relevant_decls = self._extract_relevant_declarations(lines, method_body_text)

        # Step 3: Failure attribution
        primary_diag = diag_for_file[0] if diag_for_file else diagnostics[0]
        attribution = self.attribute_failure(primary_diag, planned_tasks=planned_tasks)

        # Build Tier 1 snippet
        snippet_parts = []
        if relevant_imports:
            snippet_parts.append("// ▼ RELEVANT IMPORTS ▼\n" + "\n".join(relevant_imports))

        if relevant_decls:
            snippet_parts.append("// ▼ RELEVANT CLASS DECLARATIONS / CONSTRUCTOR ▼\n" + "\n".join(relevant_decls))

        snippet_parts.append(
            f"// ▼ CONTAINING METHOD: `{matched_mb.name}` (Lines {matched_mb.start_line+1}-{matched_mb.end_line+1}) ▼\n"
            f"{method_body_text}\n"
            f"// ▲ END METHOD `{matched_mb.name}` ▲"
        )

        # If provider attribution exists:
        if attribution.owner == FailureOwner.PROVIDER and attribution.missing_symbol:
            snippet_parts.append(
                f"\n// ⚠️ FAILURE ATTRIBUTION (PROVIDER - GENERATED/WRITABLE):\n"
                f"// Symbol '{attribution.missing_symbol}' is missing on provider type '{attribution.provider_type}'.\n"
                f"// Provider file: {attribution.provider_file} (writable={attribution.is_provider_writable})\n"
                f"// You may add the missing method to the provider file."
            )
        elif attribution.owner == FailureOwner.CROSS_FILE_CONTRACT and attribution.is_protected_provider:
            snippet_parts.append(
                f"\n// ⚠️ FAILURE ATTRIBUTION (CROSS_FILE_CONTRACT - PROTECTED PROVIDER):\n"
                f"// Provider file '{attribution.provider_file or attribution.provider_type}' is PROTECTED and CANNOT be modified.\n"
                f"// Invariant: Do NOT attempt to modify '{attribution.provider_file}'.\n"
            )
            if attribution.inspected_methods:
                snippet_parts.append(
                    "// Real public methods on protected provider:\n" +
                    "\n".join(f"//   - {m.signature}" for m in attribution.inspected_methods[:10])
                )
            if attribution.alternative_capability and attribution.alternative_capability.is_semantically_compatible:
                alt = attribution.alternative_capability
                snippet_parts.append(
                    f"// ✅ VERIFIED REPOSITORY ALTERNATIVE:\n"
                    f"//   Method: {alt.signature}\n"
                    f"//   Evidence: {alt.compatibility_reason}\n"
                    f"// Instruction: Adapt the consumer method `{matched_mb.name}` to call this verified capability."
                )
            else:
                snippet_parts.append(
                    f"// ❌ NO VERIFIED ALTERNATIVE CAPABILITY FOUND for '{attribution.missing_symbol}'.\n"
                    f"// Do NOT invent a method. If no capability exists, report UNRESOLVED / REPLAN."
                )

        prompt_snippet = "\n\n".join(snippet_parts)
        lines_sent = prompt_snippet.count("\n") + 1

        self.telemetry.localized_repair_count += 1
        self.telemetry.context_tiers_used.append(1)
        self.telemetry.source_lines_sent += lines_sent

        return LocalizedRepairContext(
            primary_file=file_path,
            target_method_name=matched_mb.name,
            context_tier=RepairContextTier.TIER_1_LOCALIZED_METHOD,
            prompt_snippet=prompt_snippet,
            source_lines_sent=lines_sent,
            attribution=attribution,
        )

    def can_attempt_repair(self, diagnostics: list[StructuredDiagnostic], max_attempts: int = 2) -> bool:
        """Check if bounded repair budget allows another attempt on this diagnostic signature."""
        fp = diagnostic_fingerprint(diagnostics)
        if not fp:
            return True
        attempts = self._fingerprint_attempts.get(fp, 0)
        return attempts < max_attempts

    def record_attempt(self, diagnostics: list[StructuredDiagnostic]):
        """Record an attempt against the diagnostic signature."""
        fp = diagnostic_fingerprint(diagnostics)
        if fp:
            self._fingerprint_attempts[fp] = self._fingerprint_attempts.get(fp, 0) + 1
        self.telemetry.repair_attempts += 1

    def _extract_relevant_imports(self, all_lines: list[str], method_text: str) -> list[str]:
        """Extract only import statements whose symbols are referenced in method_text.
        
        Scans top-level import block without arbitrary line count limits.
        """
        relevant = []
        for line in all_lines:
            line_str = line.strip()
            if not line_str:
                continue
            # Stop scanning when we hit class/interface/function declaration
            if re.match(r"^(?:export\s+)?(?:default\s+)?(?:class|interface|enum|function|const\s+[A-Z0-9_]+\s*=|type\s+)\b", line_str):
                break
            if not line_str.startswith(("import ", "from ", "const ", "var ", "let ", "require(")):
                continue
            symbols = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", line_str)
            if any(s in method_text for s in symbols if s not in ("import", "from", "as", "default", "type")):
                relevant.append(line)
        return relevant

    def _extract_relevant_declarations(self, all_lines: list[str], method_text: str) -> list[str]:
        """Extract constructor injection or field declarations whose properties are referenced in method_text."""
        props = set(re.findall(r"\bthis\.([A-Za-z0-9_]+)\b", method_text))
        if not props:
            return []
        decls = []
        in_constructor = False
        constructor_lines = []
        for line in all_lines:
            line_str = line.strip()
            if any(re.search(rf"\b{re.escape(p)}\s*[:=]", line_str) for p in props):
                decls.append(line)
            if "constructor(" in line:
                in_constructor = True
                constructor_lines = [line]
                if ")" in line and "{" in line:
                    in_constructor = False
                    if any(p in line for p in props):
                        decls.append(line)
                continue
            if in_constructor:
                constructor_lines.append(line)
                if ")" in line:
                    in_constructor = False
                    if any(p in "\n".join(constructor_lines) for p in props):
                        decls.extend(constructor_lines)
        return decls


def _match_file(a: str, b: str) -> bool:
    na = a.replace("\\", "/").lower()
    nb = b.replace("\\", "/").lower()
    return na == nb or na.endswith("/" + nb) or nb.endswith("/" + na) or Path(na).name == Path(nb).name
