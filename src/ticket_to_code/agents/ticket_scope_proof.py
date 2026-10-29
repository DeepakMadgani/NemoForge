"""
Ticket Scope Proof & Narrow Authorization Architecture.

Core Principles:
1. "Search can be broad. Write authorization must be narrow."
2. "Semantic RAG can discover files but NEVER grants write access."
3. "RAG answers 'what might be relevant?' — evidence and ownership answer 'what are we allowed to change?'"
4. SEARCH_SCOPE ⊇ REASONING_SCOPE ⊇ CHANGE_SCOPE
5. Never allow: SEMANTIC_RELEVANCE → WRITE_AUTHORIZATION

Defines:
- EvidenceRole: TICKET_ANCHOR, STRUCTURAL_COMPANION, DEPENDENCY, PROVIDER, RELATED_CONTEXT, UNVERIFIED
- WriteAuthorization: WRITE_ALLOWED, READ_ONLY (strictly distinct from EvidenceRole)
- ChangeIntent: Explicit contract for every planned file modification, system-authorized
- ScopeGraph: Structural graph of ticket anchor, companions, dependencies, providers, and related context
- ScopeGraphValidator: Deterministic contradiction detector (RELATED_CONTEXT ∩ WRITABLE = ∅, etc.)
- TicketScopeProof: Immutable proof object validating which files are authorized for writing
- OwnershipResolver: Concrete repository evidence reasoning (symbols, selectors, templates, AST)
- Post-generation patch verification and safe reversal
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


from ticket_to_code.agents.canonical_path import (
    canonical_component_base,
    canonical_repo_path,
    COMPONENT_EXTENSIONS,
)


class EvidenceRole(str, Enum):
    """Explicit role of an evidence item in relation to the ticket."""
    TICKET_ANCHOR = "ticket_anchor"               # Primary implementation file directly implementing ticket feature
    STRUCTURAL_COMPANION = "structural_companion" # Deterministic peer file (.html, .scss, .spec.ts)
    DEPENDENCY = "dependency"                     # Proven consumer-required contract / newly required artifact (READ_ONLY by default)
    PROVIDER = "provider"                         # Existing capability supplier (strictly read-only by default)
    RELATED_CONTEXT = "related_context"           # Discovered via semantic search / broad text (strictly read-only reference)
    UNVERIFIED = "unverified"                     # Discovered but ungrounded / uninspected (strictly read-only)


class WriteAuthorization(str, Enum):
    """Explicit write authorization status. Never conflate with EvidenceRole."""
    WRITE_ALLOWED = "write_allowed"
    READ_ONLY = "read_only"


@dataclass
class ChangeIntent:
    """Explicit declaration of intent for a planned task."""
    file_path: str
    action: str                                   # "modify" | "create" | "read_only"
    reason: str
    role: EvidenceRole
    write_authorization: WriteAuthorization = WriteAuthorization.READ_ONLY
    evidence_symbols: List[str] = field(default_factory=list)
    modification_proof: Optional[str] = None      # Explicit justification required if role is DEPENDENCY and requesting WRITE_ALLOWED

    @property
    def write_allowed(self) -> bool:
        return self.write_authorization == WriteAuthorization.WRITE_ALLOWED

    @write_allowed.setter
    def write_allowed(self, val: bool) -> None:
        self.write_authorization = WriteAuthorization.WRITE_ALLOWED if val else WriteAuthorization.READ_ONLY


@dataclass
class ScopeGraph:
    """Structural scope graph consumed by planner and gates."""
    ticket_id: str
    primary_anchor: Optional[str] = None
    primary_anchor_class: Optional[str] = None
    companions: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)           # Proven required dependencies - READ_ONLY by default!
    writable_dependencies: Set[str] = field(default_factory=set)  # Explicitly authorized dependencies proven to require modification
    providers: Set[str] = field(default_factory=set)              # Existing capability suppliers (strictly read-only)
    related_context: Set[str] = field(default_factory=set)        # Semantic search / broad context (strictly read-only)
    unverified: Set[str] = field(default_factory=set)             # Unverified files (strictly read-only)

    def all_writable_files(self) -> Set[str]:
        """
        Return the exact set of files authorized for modification.
        Invariant:
        Only primary_anchor, companions, and explicitly verified writable_dependencies may be writable.
        Read-only dependencies, providers, related_context, and unverified are strictly EXCLUDED.
        """
        res: Set[str] = set()
        if self.primary_anchor:
            res.add(self.primary_anchor)
        res.update(self.companions)
        res.update(self.writable_dependencies)
        return res

    def all_readonly_files(self) -> Set[str]:
        """Return the set of files that may only be read as context."""
        res: Set[str] = set()
        res.update(self.providers)
        res.update(self.related_context)
        res.update(self.unverified)
        for d in self.dependencies:
            if d not in self.writable_dependencies:
                res.add(d)
        return res


class ScopeGraphValidator:
    """
    Deterministic validator enforcing scope invariants:
    1. RELATED_CONTEXT ∩ WRITABLE = ∅
    2. PROVIDER ∩ WRITABLE = ∅
    3. UNVERIFIED ∩ WRITABLE = ∅
    4. WRITABLE ⊆ PROVEN_SCOPE
    5. DEPENDENCY without explicit modification proof is READ_ONLY
    6. Contradiction detection: no file can have multiple conflicting roles
    """
    @staticmethod
    def validate(graph: ScopeGraph) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        def _norm(p: str) -> str:
            return p.replace("\\", "/").lower().strip()
        
        writable = {_norm(p) for p in graph.all_writable_files() if p}
        providers = {_norm(p) for p in graph.providers if p}
        related = {_norm(p) for p in graph.related_context if p}
        unverified = {_norm(p) for p in graph.unverified if p}
        deps = {_norm(p) for p in graph.dependencies if p}
        writable_deps = {_norm(p) for p in graph.writable_dependencies if p}
        anchor = {_norm(graph.primary_anchor)} if graph.primary_anchor else set()
        comps = {_norm(p) for p in graph.companions if p}

        # Invariant 1: RELATED_CONTEXT ∩ WRITABLE = ∅
        rel_conflict = related.intersection(writable)
        if rel_conflict:
            errors.append(f"CONTRADICTION: Related context files marked writable: {sorted(rel_conflict)}")

        # Invariant 2: PROVIDER ∩ WRITABLE = ∅
        prov_conflict = providers.intersection(writable)
        if prov_conflict:
            errors.append(f"CONTRADICTION: Provider files marked writable: {sorted(prov_conflict)}")

        # Invariant 3: UNVERIFIED ∩ WRITABLE = ∅
        unv_conflict = unverified.intersection(writable)
        if unv_conflict:
            errors.append(f"CONTRADICTION: Unverified files marked writable: {sorted(unv_conflict)}")

        # Invariant 4: No dual role between PROVIDER and TICKET_ANCHOR
        prov_anchor_conflict = providers.intersection(anchor)
        if prov_anchor_conflict:
            errors.append(f"CONTRADICTION: Same file appears as both PROVIDER and TICKET_ANCHOR: {sorted(prov_anchor_conflict)}")

        # Invariant 5: No dual role between RELATED_CONTEXT and TICKET_ANCHOR
        rel_anchor_conflict = related.intersection(anchor)
        if rel_anchor_conflict:
            errors.append(f"CONTRADICTION: Same file appears as both RELATED_CONTEXT and TICKET_ANCHOR: {sorted(rel_anchor_conflict)}")

        # Invariant 6: Writable dependencies must be a subset of proven dependencies
        orphan_writable_deps = writable_deps - deps
        if orphan_writable_deps:
            errors.append(f"INVARIANT_VIOLATION: Writable dependency not in proven dependencies list: {sorted(orphan_writable_deps)}")

        # Invariant 7: WRITABLE ⊆ PROVEN_SCOPE
        proven_scope = anchor.union(comps).union(writable_deps)
        unproven_writable = writable - proven_scope
        if unproven_writable:
            errors.append(f"INVARIANT_VIOLATION: Writable files exceed proven scope: {sorted(unproven_writable)}")

        return len(errors) == 0, errors


def validate_scope_graph(graph: ScopeGraph) -> Tuple[bool, List[str]]:
    """Convenience functional interface for ScopeGraphValidator."""
    return ScopeGraphValidator.validate(graph)


@dataclass
class TicketScopeProof:
    """
    Authoritative, immutable proof that a ticket's scope is grounded in repository evidence.
    
    Invariant:
    EvidenceRole.TICKET_ANCHOR alone does not bypass change authorization.
    Only files verified in `scope_graph.all_writable_files()` may have writable tasks.
    """
    ticket_id: str
    scope_graph: ScopeGraph
    is_scope_proven: bool = False
    proof_reasoning: str = ""
    unresolved_anchors: List[str] = field(default_factory=list)
    proof_evidence: Dict[str, str] = field(default_factory=dict)

    @property
    def status(self) -> str:
        """Compatibility status property for change_authorization and gating layers."""
        if self.is_scope_proven:
            return "PROVEN"
        if self.unresolved_anchors:
            return "AMBIGUOUS" if "ambiguous" in self.proof_reasoning.lower() else "INSUFFICIENT_EVIDENCE"
        return "ZERO_RESULTS" if "zero" in self.proof_reasoning.lower() else "INSUFFICIENT_EVIDENCE"

    @property
    def approved_writable_files(self) -> Set[str]:
        """Set of all writable files validated in scope graph."""
        if self.scope_graph:
            return self.scope_graph.all_writable_files()
        return set()

    @property
    def anchor_file(self) -> Optional[str]:
        """Primary anchor file in the proven scope graph, if any."""
        if self.scope_graph:
            return self.scope_graph.primary_anchor
        return None

    def is_file_writable(self, file_path: str, workspace_root: Optional[Union[str, Path]] = None) -> bool:
        """Check whether a file path matches an authorized writable file in the scope graph."""
        if not self.is_scope_proven:
            return False
        
        norm_target = file_path.replace("\\", "/").lower().strip()
        c_target = canonical_repo_path(file_path, workspace_root) if workspace_root else None

        for w in self.scope_graph.all_writable_files():
            norm_w = w.replace("\\", "/").lower().strip()
            if norm_target == norm_w or norm_target.endswith("/" + norm_w) or norm_w.endswith("/" + norm_target):
                return True
            if c_target:
                c_w = canonical_repo_path(w, workspace_root)
                if c_w and c_w == c_target:
                    return True

        return False

    def get_file_role(self, file_path: str, workspace_root: Optional[Union[str, Path]] = None) -> EvidenceRole:
        """Derive the explicit role for a given file path."""
        norm_target = file_path.replace("\\", "/").lower().strip()
        c_target = canonical_repo_path(file_path, workspace_root) if workspace_root else None

        def _matches(set_paths: Set[str]) -> bool:
            for p in set_paths:
                norm_p = p.replace("\\", "/").lower().strip()
                if norm_target == norm_p or norm_target.endswith("/" + norm_p) or norm_p.endswith("/" + norm_target):
                    return True
                if c_target:
                    c_p = canonical_repo_path(p, workspace_root)
                    if c_p and c_p == c_target:
                        return True
            return False

        if self.scope_graph.primary_anchor and _matches({self.scope_graph.primary_anchor}):
            return EvidenceRole.TICKET_ANCHOR
        if _matches(self.scope_graph.companions):
            return EvidenceRole.STRUCTURAL_COMPANION
        if _matches(self.scope_graph.writable_dependencies) or _matches(self.scope_graph.dependencies):
            return EvidenceRole.DEPENDENCY
        if _matches(self.scope_graph.providers):
            return EvidenceRole.PROVIDER
        if _matches(self.scope_graph.related_context):
            return EvidenceRole.RELATED_CONTEXT
        return EvidenceRole.UNVERIFIED


def compute_change_intent_authorization(
    intent: ChangeIntent,
    scope_proof: TicketScopeProof,
    workspace_root: Optional[Union[str, Path]] = None,
) -> ChangeIntent:
    """
    System-computed write authorization.
    NEVER trusts an LLM boolean or ungrounded claim.
    
    Rules:
    - RELATED_CONTEXT, PROVIDER, UNVERIFIED: ALWAYS READ_ONLY.
    - DEPENDENCY: READ_ONLY unless explicitly in writable_dependencies with modification_proof.
    - TICKET_ANCHOR, STRUCTURAL_COMPANION: WRITE_ALLOWED if validated by scope_proof.
    """
    computed = ChangeIntent(
        file_path=intent.file_path,
        action=intent.action,
        reason=intent.reason,
        role=intent.role,
        write_authorization=WriteAuthorization.READ_ONLY,
        evidence_symbols=list(intent.evidence_symbols),
        modification_proof=intent.modification_proof,
    )

    if not scope_proof.is_scope_proven:
        computed.write_authorization = WriteAuthorization.READ_ONLY
        computed.reason = f"BLOCKED: Scope proof is unproven for ticket {scope_proof.ticket_id}."
        return computed

    grounded_role = scope_proof.get_file_role(intent.file_path, workspace_root=workspace_root)
    # Always ground to proven role
    if grounded_role != EvidenceRole.UNVERIFIED:
        computed.role = grounded_role

    # Security boundary: RELATED_CONTEXT, PROVIDER, UNVERIFIED are strictly READ_ONLY
    if computed.role in (EvidenceRole.RELATED_CONTEXT, EvidenceRole.PROVIDER, EvidenceRole.UNVERIFIED):
        computed.write_authorization = WriteAuthorization.READ_ONLY
        if intent.write_authorization == WriteAuthorization.WRITE_ALLOWED or getattr(intent, "write_allowed", False):
            computed.reason = f"SYSTEM_OVERRIDE: Role '{computed.role.value}' is strictly READ_ONLY. Write permission denied."
        return computed

    # DEPENDENCY is READ_ONLY by default unless explicitly authorized in writable_dependencies
    if computed.role == EvidenceRole.DEPENDENCY:
        is_writable_dep = False
        norm_p = intent.file_path.replace("\\", "/").lower().strip()
        for wd in scope_proof.scope_graph.writable_dependencies:
            norm_wd = wd.replace("\\", "/").lower().strip()
            if norm_p == norm_wd or norm_p.endswith("/" + norm_wd) or norm_wd.endswith("/" + norm_p):
                is_writable_dep = True
                break
        
        if is_writable_dep:
            computed.write_authorization = WriteAuthorization.WRITE_ALLOWED
            computed.reason = f"AUTHORIZED_DEPENDENCY: Proven required modification: {intent.modification_proof or 'Plan architecture dependency'}"
        else:
            computed.write_authorization = WriteAuthorization.READ_ONLY
            computed.reason = "SYSTEM_OVERRIDE: DEPENDENCY is READ_ONLY by default. Requires explicit modification proof."
        return computed

    # TICKET_ANCHOR or STRUCTURAL_COMPANION
    if computed.role in (EvidenceRole.TICKET_ANCHOR, EvidenceRole.STRUCTURAL_COMPANION):
        if scope_proof.is_file_writable(intent.file_path, workspace_root=workspace_root):
            computed.write_authorization = WriteAuthorization.WRITE_ALLOWED
        else:
            computed.write_authorization = WriteAuthorization.READ_ONLY
            computed.reason = f"BLOCKED: File '{intent.file_path}' is not in proven writable set."
        return computed

    computed.write_authorization = WriteAuthorization.READ_ONLY
    return computed


class OwnershipResolver:
    """
    Reasoning layer to determine which repository file OWNS the requested behavior.
    
    Prevents semantic proximity (e.g. 'edit-member' sharing the word 'member')
    from masquerading as ownership of the 'Add Members' feature.
    
    Requires concrete repository evidence:
    1. Exact / decomposed anchor token alignment
    2. Component / Class name matching
    3. Angular UI selector alignment
    4. Template & Form ownership
    5. Exclusion of pre-existing shared capability providers
    6. Negative distinction checking (e.g. 'edit' vs 'add')
    """

    def __init__(self, workspace_path: Optional[Union[str, Path]] = None) -> None:
        self.workspace_path = Path(workspace_path) if workspace_path else None
        self.last_evidence_trail: List[str] = []

    def resolve_owner(
        self,
        ticket_title: str,
        ticket_description: str,
        candidate_files: List[str],
        inspections: Optional[Dict[str, Any]] = None,
        anchor_tokens: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], float, str]:
        """
        Evaluate candidate files and return (best_owner_file, confidence, reasoning).
        
        Returns (None, 0.0, reasoning) if ownership is ambiguous or unproven.
        Maintains an explicit evidence trail in self.last_evidence_trail.
        """
        self.last_evidence_trail = []
        if not candidate_files:
            return None, 0.0, "No candidate files provided for ownership resolution."

        if isinstance(inspections, list) and anchor_tokens is None:
            anchor_tokens = inspections
            inspections = {}
        elif not isinstance(inspections, dict):
            inspections = {}
        tokens = [t.lower() for t in (anchor_tokens or []) if t]
        
        # Derive key ticket action phrases generically
        text_corpus = f"{ticket_title} {ticket_description}".lower()
        feature_phrases = []
        try:
            from ticket_to_code.agents.localization_agent import extract_feature_phrase
            extracted_tokens = extract_feature_phrase(ticket_title)
            if not extracted_tokens:
                extracted_tokens = extract_feature_phrase(ticket_description)
            if extracted_tokens:
                combined_dash = "-".join(extracted_tokens)
                combined_under = "_".join(extracted_tokens)
                combined_flat = "".join(extracted_tokens)
                feature_phrases.extend([combined_dash, combined_under, combined_flat] + extracted_tokens)
        except Exception:
            pass

        # Extract n-gram phrases and significant entity words from title and description
        clean_words = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', text_corpus) if w not in ("the", "and", "for", "with", "from", "that", "this", "are", "was")]
        for i in range(len(clean_words) - 1):
            w1, w2 = clean_words[i], clean_words[i+1]
            feature_phrases.extend([f"{w1}-{w2}", f"{w1}_{w2}", f"{w1}{w2}"])
            if i < len(clean_words) - 2:
                w3 = clean_words[i+2]
                feature_phrases.extend([f"{w1}-{w2}-{w3}", f"{w1}_{w2}_{w3}", f"{w1}{w2}{w3}"])
        feature_phrases.extend([w for w in clean_words if len(w) >= 4])
        feature_phrases = list(dict.fromkeys(p for p in feature_phrases if p and len(p) >= 3))
        
        # Check if this ticket is specifically a UI modal/component ticket
        is_ui_ticket = any(kw in text_corpus for kw in ("modal", "dialog", "view", "component", "ui", "css", "html", "scss"))
        has_ui_candidates = any(c.lower().endswith((".component.ts", ".component.html")) for c in candidate_files)
        prefer_ui_owner = is_ui_ticket and has_ui_candidates

        scored_candidates: List[Tuple[str, float, str, List[str]]] = []

        for candidate in candidate_files:
            norm_c = candidate.replace("\\", "/").lower()
            base_name = norm_c.split("/")[-1]
            stem = base_name.split(".")[0]
            candidate_trail: List[str] = []
            
            # Providers / Services cannot OWN UI modal features ONLY IF UI component candidates exist for a UI ticket!
            is_service_or_repo = norm_c.endswith(".service.ts") or norm_c.endswith("service.java") or "service" in stem or "repository" in stem
            if is_service_or_repo and prefer_ui_owner:
                trail_item = f"DEPRIORITIZED: {candidate} is a Service/Provider (UI component preferred for UI ticket)"
                candidate_trail.append(trail_item)
                scored_candidates.append((candidate, 0.2, "Service/Provider: supplies capability, does not own UI modal feature", candidate_trail))
                continue

            # Check explicit feature phrase alignment
            has_feature_stem = any(phrase in norm_c for phrase in feature_phrases)
            
            # Check token alignment (e.g. 'contract' AND 'member' or 'participating' AND 'member')
            significant_tokens = [t for t in tokens if t not in ("modal", "dialog", "component", "service", "controller")]
            if not significant_tokens:
                significant_tokens = list(tokens)
            token_matches = [t for t in significant_tokens if t in norm_c]
            token_ratio = len(token_matches) / len(significant_tokens) if significant_tokens else 0.0
            has_all_key_tokens = token_ratio >= 0.5 or (len(token_matches) >= 2) or (len(token_matches) == 1 and len(significant_tokens) == 1)

            # Check for negative distinction (e.g. 'edit-member' when ticket is 'add members')
            is_conflicting_action = False
            if "add" in text_corpus and "edit" in stem and "edit" not in text_corpus:
                is_conflicting_action = True

            score = 0.0
            reason_parts = []
            if has_feature_stem:
                score += 0.60
                reason_parts.append("Matches ticket feature phrase exactly")
                candidate_trail.append(f"Concrete evidence: Matches ticket feature phrase '{feature_phrases[0]}'")
            if has_all_key_tokens:
                score += 0.30
                reason_parts.append("Contains all decomposed anchor tokens")
                candidate_trail.append(f"Concrete evidence: Contains all key tokens {tokens}")
            if is_conflicting_action:
                score -= 0.50
                reason_parts.append("Conflicting action verb ('edit' vs ticket 'add')")
                candidate_trail.append("Negative evidence: Conflicting action verb ('edit' vs ticket 'add')")

            # AST / Selector / Inspection check
            insp = inspections.get(candidate)
            if insp:
                rel_methods = getattr(insp, "relevant_methods", []) or []
                if rel_methods:
                    score += 0.10
                    reason_parts.append(f"{len(rel_methods)} relevant inspected methods")
                    candidate_trail.append(f"AST evidence: {len(rel_methods)} relevant methods {rel_methods[:3]}")

            # Angular selector / template check if file exists
            if self.workspace_path:
                try:
                    full_p = self.workspace_path / candidate
                    if full_p.is_file():
                        content = full_p.read_text(encoding="utf-8", errors="ignore")
                        # Selector match
                        sel_m = re.search(r"selector\s*:\s*['\"]([^'\"]+)['\"]", content)
                        if sel_m:
                            selector = sel_m.group(1).lower()
                            if any(p in selector for p in feature_phrases):
                                score += 0.15
                                reason_parts.append(f"Selector '{selector}' aligns with ticket")
                                candidate_trail.append(f"Selector evidence: Component selector '{selector}'")
                except Exception:
                    pass

            # File kind priority: primary code files (.ts, .java, .py) get slight tie-breaking priority over templates/styles
            is_main_code = norm_c.endswith((".ts", ".java", ".py", ".go")) and not norm_c.endswith((".spec.ts", ".test.ts", "test.java", "tests.java"))
            is_template_or_style = norm_c.endswith((".html", ".scss", ".css"))
            if is_main_code:
                score += 0.05
            elif is_template_or_style:
                score -= 0.02

            final_score = max(0.0, min(1.0, score))
            reason = "; ".join(reason_parts) if reason_parts else "No specific evidence"
            scored_candidates.append((candidate, final_score, reason, candidate_trail))

        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        self.scored_candidates = scored_candidates
        best_file, best_score, best_reason, best_trail = scored_candidates[0]
        self.last_evidence_trail = best_trail

        # Differentiate ties among top candidates if needed
        if len(scored_candidates) > 1 and scored_candidates[0][1] == scored_candidates[1][1]:
            c0, s0, r0, t0 = scored_candidates[0]
            c1, s1, r1, t1 = scored_candidates[1]
            t0_count = sum(1 for tok in tokens if tok in c0.lower())
            t1_count = sum(1 for tok in tokens if tok in c1.lower())
            if t1_count > t0_count:
                scored_candidates[0], scored_candidates[1] = scored_candidates[1], scored_candidates[0]
                best_file, best_score, best_reason, best_trail = scored_candidates[0]
            elif t0_count == t1_count:
                self.last_evidence_trail = [f"Ambiguity: Multiple candidates ({c0}, {c1}) share identical score {best_score:.2f} without differentiating evidence."]
                return None, 0.0, f"Ambiguous ownership: Multiple top candidates share identical score {best_score:.2f} ({c0} vs {c1})."

        if best_score >= 0.70:
            return best_file, best_score, f"Proven Owner: {best_reason}"
        else:
            return None, best_score, f"Insufficient ownership proof (top score {best_score:.2f} < 0.70): {best_reason}"


def resolve_structural_companions(
    anchor_file: str,
    workspace_root: Optional[Union[str, Path]] = None,
) -> Set[str]:
    """
    Deterministically resolve physical companion files for an Angular/TypeScript component anchor
    or backend Java / Python source/test anchor.
    """
    if not anchor_file:
        return set()

    norm_anchor = anchor_file.replace("\\", "/")
    companions: Set[str] = set()

    # 1. Java structural companions (src/main/java <-> src/test/java, and Impl)
    if norm_anchor.endswith(".java"):
        if re.search(r'(^|/)src/main/java/', norm_anchor):
            test_path = re.sub(r'(^|/)src/main/java/', r'\1src/test/java/', norm_anchor)
            test_path = re.sub(r'\.java$', 'Test.java', test_path)
            tests_path = re.sub(r'(^|/)src/main/java/', r'\1src/test/java/', norm_anchor)
            tests_path = re.sub(r'\.java$', 'Tests.java', tests_path)
            cands = [test_path, tests_path]
            if norm_anchor.endswith("Impl.java"):
                cands.append(re.sub(r'Impl\.java$', '.java', norm_anchor))
            else:
                cands.append(re.sub(r'\.java$', 'Impl.java', norm_anchor))
            for cand in cands:
                if cand.lower() != norm_anchor.lower():
                    if workspace_root:
                        if (Path(workspace_root) / cand).is_file():
                            companions.add(cand)
                    else:
                        companions.add(cand)
        elif re.search(r'(^|/)src/test/java/', norm_anchor):
            main_path = re.sub(r'(^|/)src/test/java/', r'\1src/main/java/', norm_anchor)
            main_path = re.sub(r'Tests?\.java$', '.java', main_path)
            if main_path.lower() != norm_anchor.lower():
                if workspace_root:
                    if (Path(workspace_root) / main_path).is_file():
                        companions.add(main_path)
                else:
                    companions.add(main_path)
        return companions

    # 2. TypeScript / JavaScript service, pipe, directive, controller, util companions (*.spec.ts, *.test.ts)
    if norm_anchor.endswith((".ts", ".tsx", ".js", ".jsx")) and not norm_anchor.endswith((".spec.ts", ".test.ts", ".spec.js", ".test.js")):
        spec_cand = re.sub(r'\.(ts|tsx|js|jsx)$', r'.spec.\1', norm_anchor)
        test_cand = re.sub(r'\.(ts|tsx|js|jsx)$', r'.test.\1', norm_anchor)
        for cand in [spec_cand, test_cand]:
            if cand.lower() != norm_anchor.lower():
                if workspace_root:
                    if (Path(workspace_root) / cand).is_file():
                        companions.add(cand)
                else:
                    companions.add(cand)

    # 3. Angular component companions (.html, .scss, .css, .spec.ts)
    base = canonical_component_base(anchor_file, workspace_root)
    if not base:
        parts = norm_anchor.split("/")
        directory = "/".join(parts[:-1])
        filename = parts[-1]
        for ext in COMPONENT_EXTENSIONS:
            if filename.endswith(ext):
                stem = filename[:-len(ext)]
                base = f"{directory}/{stem}" if directory else stem
                break

    if not base:
        return companions

    candidate_extensions = [".component.html", ".component.scss", ".component.css", ".component.spec.ts"]
    for ext in candidate_extensions:
        comp_path = f"{base}{ext}"
        if comp_path.lower() == norm_anchor.lower():
            continue
        
        if workspace_root:
            full_path = Path(workspace_root) / comp_path
            if full_path.is_file():
                companions.add(comp_path)
        else:
            companions.add(comp_path)

    return companions


def capture_pre_run_git_status(workspace_path: Union[str, Path]) -> Set[str]:
    """Capture relative paths of all currently dirty or untracked files across git repos.
    Used as baseline so pre-existing user edits are NEVER treated as unauthorized scope violations."""
    ws = Path(workspace_path)
    dirty_files: Set[str] = set()
    git_repos = _find_git_repos(ws)
    for repo_path, rel_prefix in git_repos:
        try:
            res = subprocess.run(
                ["git", "status", "--porcelain", "-uall"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=15,
            )
            if res.returncode == 0:
                for line in (res.stdout or "").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) >= 2:
                        raw_path = parts[1].strip()
                        if "->" in raw_path:
                            p2 = raw_path.split("->")[1].strip().strip('"')
                            p = f"{rel_prefix}/{p2}" if rel_prefix else p2
                        else:
                            p2 = raw_path.strip('"')
                            p = f"{rel_prefix}/{p2}" if rel_prefix else p2
                        dirty_files.add(p.replace("\\", "/").lower().strip())
        except Exception:
            pass
    return dirty_files


def _find_git_repos(ws: Path) -> List[Tuple[Path, str]]:
    """Return list of (repo_path, rel_prefix) for git repos at ws or up to 2 levels
    of subdirectories (bounded scan so poly-repo parents are always recognized)."""
    repos: List[Tuple[Path, str]] = []
    if (ws / ".git").exists():
        repos.append((ws, ""))
        return repos
    max_scanned = 800
    scanned = 0
    try:
        level1 = [c for c in ws.iterdir() if c.is_dir() and not c.name.startswith(".")]
    except Exception:
        return repos
    for child in level1:
        if scanned >= max_scanned:
            break
        scanned += 1
        if (child / ".git").exists():
            repos.append((child, child.name))
            continue
        try:
            for gc in child.iterdir():
                if scanned >= max_scanned:
                    break
                if gc.is_dir() and not gc.name.startswith(".") and (gc / ".git").exists():
                    scanned += 1
                    repos.append((gc, f"{child.name}/{gc.name}"))
        except Exception:
            pass
    return repos


from ticket_to_code.utils.file_filters import STANDARD_SKIP_DIRS as _MANIFEST_SKIP_DIRS, is_ignored_path


def build_workspace_manifest(
    workspace_path: Union[str, Path],
    max_files: int = 200000,
) -> Dict[str, int]:
    """
    Snapshot the workspace as {relative_path: file_size} BEFORE the workflow
    touches anything. Used by the post-generation scope gate so that non-git
    (or git-detection-failed) workspaces can be diffed against a real pre-run
    state instead of guessing which files are 'new'.
    """
    ws = Path(workspace_path)
    manifest: Dict[str, int] = {}
    if not ws.exists():
        return manifest

    ignored_dir_names = {
        "node_modules", ".git", "dist", "build", "target", "bin", "obj",
        ".angular", ".cache", "venv", ".venv", "__pycache__", ".idea", ".vscode"
    }

    for root, dirs, files in os.walk(str(ws)):
        dirs[:] = [d for d in dirs if d not in ignored_dir_names and not d.startswith(".")]
        for fname in files:
            if len(manifest) >= max_files:
                return manifest
            fpath = Path(root) / fname
            if is_ignored_path(fpath, workspace_path=ws):
                continue
            try:
                rel = fpath.relative_to(ws)
                manifest[str(rel).replace("\\", "/")] = fpath.stat().st_size
            except Exception:
                continue
    return manifest


def verify_post_generation_scope(
    workspace_path: Union[str, Path],
    approved_writable_files: Set[str],
    original_contents: Optional[Dict[str, str]] = None,
    pre_run_manifest: Optional[Dict[str, int]] = None,
    pre_run_git_dirty_files: Optional[Set[str]] = None,
    session_written_files: Optional[Set[str]] = None,
) -> Tuple[bool, List[str], Set[str]]:
    """
    Inspect the actual filesystem/git status after code generation.
    Enforces the invariant:
        ACTUAL_MODIFIED_FILES ⊆ APPROVED_WRITABLE_FILES

    Detects:
    - Modified existing files
    - Newly created files
    - Deleted files
    - Renamed files

    Args:
        pre_run_manifest: {relpath: size} snapshot from BEFORE the run
            (build_workspace_manifest). When git detection fails, the gate
            diffs against this real pre-run state instead of assuming every
            file absent from original_contents is "new".
        pre_run_git_dirty_files: Set of relative paths that were ALREADY dirty/untracked
            before this workflow run started. These are developer baseline and will
            NEVER be flagged as unauthorized changes or violations.
        session_written_files: Set of files actually written or touched during
            this workflow run. Pre-existing files that the session never touched
            are never flagged as unauthorized session writes.

    Returns:
        (is_clean, violations_list, unauthorized_files_set)
    """
    ws = Path(workspace_path)
    if not ws.exists():
        return (
            False,
            [f"SCOPE_PROOF_ERROR: workspace path does not exist: {ws}"],
            set(),
        )

    approved_norm: Set[str] = {f.replace("\\", "/").lower().strip() for f in approved_writable_files if f}
    pre_dirty_norm: Set[str] = {p.replace("\\", "/").lower().strip() for p in (pre_run_git_dirty_files or set()) if p}
    session_written_norm: Optional[Set[str]] = (
        {sw.replace("\\", "/").lower().strip() for sw in session_written_files if sw}
        if session_written_files is not None else None
    )

    def _is_approved(path_str: str) -> bool:
        np = path_str.replace("\\", "/").lower().strip()
        for a in approved_norm:
            if np == a or np.endswith("/" + a) or a.endswith("/" + np):
                return True
        return False

    def _is_pre_dirty(path_str: str) -> bool:
        np = path_str.replace("\\", "/").lower().strip()
        if np in pre_dirty_norm:
            return True
        for d in pre_dirty_norm:
            if np == d or np.endswith("/" + d) or d.endswith("/" + np):
                return True
        return False

    def _was_session_touched(path_str: str) -> bool:
        if session_written_norm is None:
            return True
        np = path_str.replace("\\", "/").lower().strip()
        for sw in session_written_norm:
            if np == sw or np.endswith("/" + sw) or sw.endswith("/" + np):
                return True
        return False

    actual_modified: Set[str] = set()
    violations: List[str] = []
    git_repos = _find_git_repos(ws)
    git_detected = len(git_repos) > 0

    if git_detected:
        git_failures = 0
        for repo_path, rel_prefix in git_repos:
            try:
                res = subprocess.run(
                    ["git", "status", "--porcelain", "-uall"],
                    cwd=str(repo_path),
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if res.returncode == 0:
                    for line in (res.stdout or "").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(maxsplit=1)
                        if len(parts) >= 2:
                            raw_path = parts[1].strip()
                            if "->" in raw_path:
                                rename_parts = raw_path.split("->")
                                p1 = rename_parts[0].strip().strip('"')
                                p2 = rename_parts[1].strip().strip('"')
                                if rel_prefix:
                                    p1 = f"{rel_prefix}/{p1}"
                                    p2 = f"{rel_prefix}/{p2}"
                                if not is_ignored_path(ws / p1, workspace_path=ws) and not _is_pre_dirty(p1):
                                    if _was_session_touched(p1):
                                        actual_modified.add(p1)
                                if not is_ignored_path(ws / p2, workspace_path=ws) and not _is_pre_dirty(p2):
                                    if _was_session_touched(p2):
                                        actual_modified.add(p2)
                            else:
                                p = raw_path.strip('"')
                                if rel_prefix:
                                    p = f"{rel_prefix}/{p}"
                                if not is_ignored_path(ws / p, workspace_path=ws) and not _is_pre_dirty(p):
                                    if _was_session_touched(p):
                                        actual_modified.add(p)
                else:
                    git_failures += 1
            except Exception:
                git_failures += 1
        if git_failures:
            # Fail-closed: a safety gate must never pass silently because git
            # could not be invoked. Flag it so callers cannot mistake an
            # unverifiable workspace for a clean one.
            violations.append(
                f"SCOPE_PROOF_ERROR: git status failed for {git_failures}/{len(git_repos)} "
                "repo(s) — scope could not be verified (treat run as unverified)"
            )
            actual_modified.add("__scope_proof_git_status_failed__")
    elif pre_run_manifest is not None:
        # Non-git workspace WITH a pre-run manifest: diff against the real
        # pre-run snapshot (safe for production poly-repos).
        manifest_norm = {k.replace("\\", "/").lower(): k for k in pre_run_manifest}
        orig_keys = {k.replace("\\", "/").lower() for k in (original_contents or {})}
        for rel_key, size in pre_run_manifest.items():
            p = ws / rel_key
            if not p.exists():
                if _was_session_touched(rel_key):
                    actual_modified.add(rel_key)  # Deleted!
                continue
            try:
                if p.stat().st_size != size:
                    if rel_key.lower() in orig_keys:
                        if p.read_text(encoding="utf-8", errors="ignore") != original_contents[
                            manifest_norm.get(rel_key.lower(), rel_key)
                        ]:
                            if _was_session_touched(rel_key):
                                actual_modified.add(rel_key)  # Modified!
                    else:
                        if _was_session_touched(rel_key):
                            actual_modified.add(rel_key)  # Size changed, unknown content
            except Exception:
                pass
        for p in ws.rglob("*"):
            if p.is_file():
                if is_ignored_path(p, workspace_path=ws):
                    continue
                rel_p = str(p.relative_to(ws)).replace("\\", "/")
                if rel_p.lower() not in manifest_norm and rel_p.lower() not in orig_keys:
                    if _was_session_touched(rel_p):
                        actual_modified.add(rel_p)  # Created!
    elif original_contents is not None:
        # Non-git environment (e.g. pytest tmp_path)
        for orig_f, orig_text in original_contents.items():
            p = ws / orig_f
            if not p.exists():
                if _was_session_touched(orig_f):
                    actual_modified.add(orig_f)  # Deleted!
            else:
                try:
                    curr_text = p.read_text(encoding="utf-8", errors="ignore")
                    if curr_text != orig_text:
                        if _was_session_touched(orig_f):
                            actual_modified.add(orig_f)  # Modified!
                except Exception:
                    pass

        for p in ws.rglob("*"):
            if p.is_file():
                if is_ignored_path(p, workspace_path=ws):
                    continue
                rel_p = str(p.relative_to(ws)).replace("\\", "/")
                norm_rel = rel_p.lower()
                if not any(k.replace("\\", "/").lower() == norm_rel for k in original_contents.keys()):
                    if _was_session_touched(rel_p):
                        actual_modified.add(rel_p)

    unauthorized: Set[str] = set()
    for f in actual_modified:
        if not _is_approved(f):
            unauthorized.add(f)
            violations.append(f"SCOPE_PROOF_VIOLATION: Unapproved modification/creation of '{f}' detected on disk")

    is_clean = len(unauthorized) == 0
    return is_clean, violations, unauthorized


def revert_unauthorized_changes(
    workspace_path: Union[str, Path],
    unauthorized_files: Set[str],
    original_contents: Optional[Dict[str, str]] = None,
    pre_run_manifest: Optional[Dict[str, int]] = None,
    protected_files: Optional[Set[str]] = None,
    session_written_files: Optional[Set[str]] = None,
) -> List[str]:
    """
    Safely reverts unauthorized disk mutations.
    - NEVER touches any file in protected_files under any circumstances.
    - NEVER touches any file outside session_written_files if provided (protects untouched developer work).
    - If file was pre-existing: checkout from git or restore from original_contents.
    - If file was newly created untracked: delete from disk.
    - NEVER delete files present in pre_run_manifest unless original content is available.
    """
    ws = Path(workspace_path)
    reverted: List[str] = []
    original_contents = original_contents or {}
    pre_run_manifest = pre_run_manifest or {}
    manifest_norm = {k.replace("\\", "/").lower() for k in pre_run_manifest}
    protected_norm = {p.replace("\\", "/").lower() for p in (protected_files or set())}
    session_norm = (
        {sw.replace("\\", "/").lower().strip() for sw in session_written_files if sw}
        if session_written_files is not None else None
    )
    git_repos = _find_git_repos(ws)

    for f in unauthorized_files:
        norm_f = f.replace("\\", "/").lower().strip()
        if norm_f in protected_norm:
            reverted.append(f"PRESERVED protected intentional file: {f}")
            continue

        if session_norm is not None:
            if not any(norm_f == sn or norm_f.endswith("/" + sn) or sn.endswith("/" + norm_f) for sn in session_norm):
                reverted.append(f"PRESERVED untouched pre-existing file (not modified by this workflow session): {f}")
                continue

        full_path = ws / f

        # 1. Try restoring from original_contents if available
        matched_orig = None
        for k, v in original_contents.items():
            if k.replace("\\", "/").lower() == norm_f:
                matched_orig = v
                break

        if matched_orig is not None:
            try:
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(matched_orig, encoding="utf-8")
                reverted.append(f"Restored original content: {f}")
                continue
            except Exception:
                pass

        # 2. Check if file belongs to a detected git repository
        target_repo = None
        rel_in_repo = f
        for repo_path, rel_prefix in git_repos:
            if rel_prefix and (f == rel_prefix or f.startswith(rel_prefix + "/") or f.startswith(rel_prefix + "\\")):
                target_repo = repo_path
                rel_in_repo = f[len(rel_prefix) + 1:].replace("\\", "/")
                break
            elif not rel_prefix and repo_path == ws:
                target_repo = ws
                rel_in_repo = f.replace("\\", "/")
                break

        if target_repo is not None:
            res_tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", rel_in_repo],
                cwd=str(target_repo),
                capture_output=True,
                text=True,
            )
            is_tracked = (res_tracked.returncode == 0)
            if is_tracked:
                subprocess.run(
                    ["git", "checkout", "--", rel_in_repo],
                    cwd=str(target_repo),
                    capture_output=True,
                    text=True,
                )
                reverted.append(f"Restored tracked file via git checkout: {f}")
                continue
            else:
                if full_path.is_file():
                    try:
                        full_path.unlink()
                        reverted.append(f"Deleted unauthorized untracked file: {f}")
                    except Exception:
                        pass
                continue

        # 3. No git repo: only delete if absent from pre_run_manifest
        if pre_run_manifest and norm_f in manifest_norm:
            reverted.append(
                f"SKIPPED deletion of pre-existing file (no original content captured): {f}"
            )
            continue
        if full_path.is_file():
            try:
                full_path.unlink()
                reverted.append(f"Deleted unauthorized untracked file: {f}")
            except Exception:
                pass

    return reverted


