"""
Behavioral Understanding — synthesize the evidence into a plan-ready understanding.

This does NOT add a new gate, agent, or search system. It COMPOSES what the
existing pipeline already produces (artifact inspections, relationships, API
contracts, unresolved questions) with the existing helper modules
(semantic_requirements, capability_reuse_resolver, change_authorization) into a
single ordered understanding the planner can consume directly:

    TICKET → CURRENT BEHAVIOR → REQUESTED BEHAVIOR → BEHAVIORAL DELTA →
    EXISTING CAPABILITIES → REUSE DECISIONS → RELATIONSHIPS → RELEVANT ARTIFACTS →
    CHANGE CANDIDATES (vs REFERENCE) → UNRESOLVED CRITICAL → SEMANTIC CONSTRAINTS

The goal is UNDERSTAND-FIRST: the planner should reuse verified existing
capabilities and change only what the behavioral delta requires, instead of
reconstructing the architecture from raw evidence text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from ticket_to_code.agents.semantic_requirements import extract_semantic_constraints
from ticket_to_code.agents.change_authorization import (
    classify_change_role, ChangeRole, is_config_file, is_build_config, is_companion,
)


def _norm(p: str) -> str:
    return (p or "").replace("\\", "/").lower()


def _basename(p: str) -> str:
    return _norm(p).split("/")[-1]


# Generic action verbs used to derive "required capabilities" from ticket text.
# Language/project-agnostic — no framework or filename assumptions.
_VERB_RE = re.compile(
    r"\b(check|verify|determine|get|fetch|find|load|retrieve|list|show|display|"
    r"save|persist|store|update|create|add|remove|delete|prevent|validate|"
    r"filter|assign|resolve)\b",
    re.IGNORECASE,
)

# Non-capability tokens to skip when picking the requirement noun after a verb.
_STOP_NOUNS = {
    "whether", "that", "this", "these", "those", "there", "here", "already",
    "been", "being", "their", "them", "then", "they", "when", "will", "would",
    "should", "could", "with", "from", "into", "your", "ours", "its", "the",
    "and", "but", "for", "not", "are", "was", "were", "have", "has", "read",
    "only", "static", "text", "current", "other", "same", "user", "users",
}

_REUSE_STRONG = 0.82
_REUSE_PARTIAL = 0.55


@dataclass
class ExistingCapability:
    name: str
    owner: str = ""
    signature: str = ""
    contract: str = ""
    file: str = ""
    evidence_text: str = ""      # signature + return type + contract + facts (repo behavior)
    provenance: str = "relationship_grounded"   # relationship_grounded | rag
    grounded: bool = True        # True = verified repository evidence; False = RAG (supplementary)
    # ── Structured capability fields (planner-facing handoff) ──
    return_type: str = ""        # e.g. Observable<ParticipatingMember[]>
    params: str = ""             # parameter list from the signature
    behavior: str = ""           # short behavior summary (from code region / facts)
    relationship: str = ""       # relationship that led to this provider
    inspection_status: str = ""  # inspection depth (symbols | region | contracts …)
    relevant: bool = False       # flagged relevant-to-ticket by inspection
    sufficiency: str = ""        # sufficient | adaptable | insufficient | unknown
    reuse_decision: str = ""     # reuse | adapt | extend | create_new (back-filled)


@dataclass
class ReuseDecisionRecord:
    requirement: str
    decision: str            # reuse | adapt | extend | create_new
    existing_symbol: str = ""
    owner: str = ""
    reason: str = ""


@dataclass
class BehavioralUnderstanding:
    current_behavior: list = field(default_factory=list)
    requested_behavior: list = field(default_factory=list)
    behavioral_delta: list = field(default_factory=list)
    existing_capabilities: list = field(default_factory=list)   # ExistingCapability
    reuse_decisions: list = field(default_factory=list)         # ReuseDecisionRecord
    relevant_artifacts: list = field(default_factory=list)
    relevant_relationships: list = field(default_factory=list)
    change_candidates: list = field(default_factory=list)
    reference_artifacts: list = field(default_factory=list)
    unresolved_critical: list = field(default_factory=list)
    semantic: dict = field(default_factory=dict)
    confidence: float = 0.0

    def to_planning_block(self) -> str:
        L: list[str] = ["=== BEHAVIORAL UNDERSTANDING (authoritative — plan from this) ==="]
        if self.current_behavior:
            L.append("CURRENT BEHAVIOR:")
            L += [f"  - {c}" for c in self.current_behavior[:8]]
        if self.requested_behavior:
            L.append("REQUESTED BEHAVIOR:")
            L += [f"  - {r}" for r in self.requested_behavior[:10]]
        if self.behavioral_delta:
            L.append("BEHAVIORAL DELTA (implement exactly this):")
            L += [f"  - {d}" for d in self.behavioral_delta[:10]]
        if self.existing_capabilities:
            L.append("EXISTING CAPABILITIES (verified — reuse these, do NOT reinvent):")
            # Always surface reuse-linked / relevant capabilities; fill the rest.
            _must = [c for c in self.existing_capabilities if c.reuse_decision or c.relevant]
            _rest = [c for c in self.existing_capabilities if c not in _must]
            for c in (_must + _rest)[:12]:
                own = f"{c.owner}." if c.owner else ""
                sig = f"({c.signature})" if c.signature else "()"
                ret = f" -> {c.return_type}" if c.return_type else ""
                prov = "grounded" if getattr(c, "grounded", True) else "rag/supplementary"
                L.append(f"  - {own}{c.name}{sig}{ret}")
                meta = [f"provider: {c.file or '?'}", f"provenance: {prov}"]
                if c.inspection_status:
                    meta.append(f"inspected: {c.inspection_status}")
                if c.relevant:
                    meta.append("relevant: yes")
                L.append(f"      {' | '.join(meta)}")
                if c.reuse_decision:
                    suff = f" ({c.sufficiency})" if c.sufficiency else ""
                    L.append(f"      reuse: {c.reuse_decision.upper()}{suff}")
                if c.relationship:
                    L.append(f"      via: {c.relationship}")
                if c.contract:
                    L.append(f"      contract: {c.contract}")
                if c.behavior:
                    L.append(f"      behavior: {c.behavior}")
        if self.reuse_decisions:
            L.append("REUSE DECISIONS (AUTHORITATIVE — do NOT CREATE_NEW where REUSE/ADAPT is stated):")
            for d in self.reuse_decisions[:12]:
                sym = f" → {d.existing_symbol}" if d.existing_symbol else ""
                L.append(f"  - [{d.decision.upper()}] {d.requirement}{sym}: {d.reason}")
        if self.relevant_relationships:
            L.append("RELATIONSHIPS:")
            L += [f"  - {r}" for r in self.relevant_relationships[:12]]
        if self.relevant_artifacts:
            L.append(f"RELEVANT ARTIFACTS: {', '.join(self.relevant_artifacts[:15])}")
        if self.change_candidates:
            L.append("CHANGE CANDIDATES (only these should become tasks):")
            L += [f"  - {c}" for c in self.change_candidates[:12]]
        if self.reference_artifacts:
            L.append(f"READ-ONLY REFERENCES (understand, do NOT modify): "
                     f"{', '.join(self.reference_artifacts[:15])}")
        if self.unresolved_critical:
            L.append("UNRESOLVED CRITICAL QUESTIONS:")
            L += [f"  - {q}" for q in self.unresolved_critical[:8]]
        if self.semantic:
            flags = [k for k, v in self.semantic.items() if v]
            if flags:
                L.append(f"SEMANTIC CONSTRAINTS: {', '.join(flags)}")
        L.append(f"CONFIDENCE: {self.confidence:.2f}")
        L.append("=== END BEHAVIORAL UNDERSTANDING ===")
        return "\n".join(L)


# ── Builder ─────────────────────────────────────────────────────────────────

def _capabilities_from_inspections(inspections: dict, relationships: list | None = None) -> list:
    caps: list[ExistingCapability] = []
    seen = set()
    # Map provider file/basename → a relationship string that led to it.
    rel_by_file: dict[str, str] = {}
    for rel in (relationships or []):
        if isinstance(rel, dict):
            src = rel.get("source") or rel.get("src") or ""
            tgt = rel.get("target") or rel.get("dst") or ""
            rtype = rel.get("type") or rel.get("relationship_type") or "relates_to"
            if src and tgt:
                rel_by_file.setdefault(_basename(tgt), f"{src} --{rtype}--> {tgt}")
    for fp, insp in (inspections or {}).items():
        owner = getattr(insp, "class_name", "") or ""
        contracts = getattr(insp, "api_contracts", []) or []
        contract = contracts[0] if contracts else ""
        facts_list = [getattr(f, "fact", "") or "" for f in (getattr(insp, "facts", []) or [])]
        facts_text = " ".join(facts_list)
        code_regions = getattr(insp, "relevant_code_regions", {}) or {}
        # Actual code regions carry return shapes/fields — authoritative behavior.
        regions_text = " ".join((r.get("code", "") or "")[:300] for r in code_regions.values())
        inspection_status = getattr(insp, "inspection_depth", "") or ""
        relationship = rel_by_file.get(_basename(fp), "")
        relevant_methods = list(getattr(insp, "relevant_methods", []) or [])
        relevant_names = {getattr(m, "name", "") for m in relevant_methods}
        methods = relevant_methods or list(getattr(insp, "methods", []) or [])
        for m in methods:
            name = getattr(m, "name", "") or ""
            if not name or (owner, name) in seen:
                continue
            seen.add((owner, name))
            sig = getattr(m, "signature", "") or ""
            ret = getattr(m, "return_type", "") or ""
            # Behavior: prefer the method's own code region, else a fact naming it.
            behavior = ""
            region = code_regions.get(name)
            if isinstance(region, dict) and region.get("code"):
                behavior = " ".join(region["code"].split())[:180]
            if not behavior:
                for fact in facts_list:
                    if name in fact:
                        behavior = fact[:180]
                        break
            caps.append(ExistingCapability(
                name=name, owner=owner, signature=sig, contract=contract, file=fp,
                evidence_text=" ".join([name, sig, ret, contract, facts_text, regions_text]).lower(),
                return_type=ret, params=sig, behavior=behavior,
                relationship=relationship, inspection_status=inspection_status,
                relevant=name in relevant_names,
            ))
    return caps


def _derive_required_capabilities(text: str) -> list:
    reqs: list[str] = []
    low = text or ""
    for m in _VERB_RE.finditer(low):
        verb = m.group(1).lower()
        window = low[m.end(): m.end() + 60]
        nouns = [
            w.lower() for w in re.findall(r"[a-zA-Z][a-zA-Z0-9]{3,}", window)
            if w.lower() not in _STOP_NOUNS
        ][:3]
        for n in nouns:
            phrase = f"{verb} {n}"
            if phrase not in reqs:
                reqs.append(phrase)
    return reqs[:16]


def _best_capability_match(required_phrase: str, capabilities: list):
    """Evidence-first match: repository behavior is authoritative, name is a hint.

    Returns (capability, score). Score is boosted when the requirement's key noun
    appears in the capability's inspected evidence (signature/return/contract/facts/
    code) — so an existing capability under a different name is still matched.
    """
    noun = required_phrase.split()[-1].lower() if required_phrase.split() else ""
    target = re.sub(r"[^a-z0-9]", "", required_phrase.lower())
    best, score = None, 0.0
    for c in capabilities:
        cname = re.sub(r"[^a-z0-9]", "", c.name.lower())
        name_ratio = SequenceMatcher(None, target, cname).ratio() if cname else 0.0
        # Evidence signal: the requirement's noun is present in verified behavior.
        evidence_signal = 0.0
        if noun and len(noun) >= 4:
            if noun in (c.evidence_text or ""):
                evidence_signal = 0.9
            elif noun in c.name.lower() or c.name.lower() in noun:
                evidence_signal = 0.85
        s = max(name_ratio, evidence_signal)
        if s > score:
            best, score = c, s
    return best, score


def _feature_name_match(feature_tokens: list, path: str) -> float:
    """0..1 filename-stem match to the ticket feature phrase (a surfacing signal,
    NOT ownership). Contiguous phrase → 0.85; single distinctive token → 0.5."""
    toks = [t.lower() for t in (feature_tokens or []) if t and len(str(t)) >= 3]
    if not toks:
        return 0.0
    stem = [s for s in re.split(r"[^a-z0-9]+", _basename(path).split(".")[0].lower()) if s]
    if not stem:
        return 0.0
    joined = "".join(stem)
    if any("".join(toks[i:j]) in joined
           for i in range(len(toks)) for j in range(i + 2, len(toks) + 1)):
        return 0.85
    if len(stem) <= 2 and any(s in set(toks) for s in stem):
        return 0.5
    return 0.0


def derive_primary_targets(
    inspections: dict,
    feature_tokens: list | None = None,
    reuse_provider_files: set | None = None,
    semantic_scores: dict | None = None,
    anchor_confidence: float = 0.50,
    strong_confidence: float = 0.70,
) -> set:
    """Prove which INSPECTED artifacts directly implement the requested behavior
    → PRIMARY_FEATURE_TARGET. Evidence-based; composes existing signals only:

      * inspection floor — the artifact must have relevant methods (behavior found)
      * a reused capability PROVIDER is never primary (its capability is reused)
      * behavioral proof — the semantic verifier's confidence that the file
        IMPLEMENTS the requested behavior. A file NAMED after the feature clears a
        lower confidence bar (``anchor_confidence``); an unrelated backend/service
        must show STRONG behavioral proof (``strong_confidence``). With no verifier
        verdict, a filename-feature match + inspection is the degraded fallback.

    RAG / keyword / embedding / filename score ALONE never promotes: inspection is
    required, a reused provider is excluded, and filename only lowers the required
    behavioral confidence — it never grants ownership by itself. Language/
    framework/repository agnostic.
    """
    reuse_norm = {_norm(p) for p in (reuse_provider_files or set()) if p}
    semantic_scores = semantic_scores or {}
    feat = [t for t in (feature_tokens or []) if t]
    primary: set = set()
    for fp, insp in (inspections or {}).items():
        if not fp or is_config_file(fp) or is_build_config(fp):
            continue
        nfp = _norm(fp)
        if any(nfp == p or nfp.endswith(p) or p.endswith(nfp) for p in reuse_norm):
            continue  # reused provider → READ_ONLY, never primary
        rel_methods = list(getattr(insp, "relevant_methods", []) or [])
        if not rel_methods:
            continue  # no inspected behavior → not proven
        anchor = _feature_name_match(feat, fp) if feat else 0.0
        # Invariant: Semantic RAG can discover files but NEVER grants write access / primary target status.
        # A file MUST possess anchor alignment (anchor >= 0.50) to even be considered a primary target.
        if anchor < 0.50:
            continue

        sem = semantic_scores.get(fp) or semantic_scores.get(nfp) or {}
        if sem:
            decision = str(sem.get("decision", "include")).lower()
            if decision in ("exclude", "reject"):
                continue
            score = float(sem.get("score", sem.get("semantic_relevance_score", 0.0)) or 0.0)
            if score >= anchor_confidence:
                primary.add(fp)
            continue
        # No verifier verdict → filename-feature match + inspection (never name alone).
        if anchor >= 0.5:
            primary.add(fp)
    return primary


def build_behavioral_understanding(
    ticket_title: str,
    ticket_description: str,
    requirements_text: str = "",
    inspections: dict | None = None,
    relationships: list | None = None,
    api_contracts: list | None = None,
    unresolved: list | None = None,
    evidence_files: list | None = None,
    authorized_files: set | None = None,
    scope_declared: bool = False,
    ticket_targets_config: bool = False,
    rag_capabilities: list | None = None,
    primary_targets: set | None = None,
    feature_tokens: list | None = None,
    semantic_scores: dict | None = None,
) -> BehavioralUnderstanding:
    bu = BehavioralUnderstanding()
    inspections = inspections or {}
    relationships = relationships or []
    evidence_files = list(evidence_files or list(inspections.keys()))

    # 1. Semantic constraints (cardinality / read-only / duplicate / persistence).
    sc = extract_semantic_constraints(ticket_title, ticket_description, requirements_text)
    bu.semantic = {
        "cardinality": sc.cardinality,
        "per_item": sc.is_per_item,
        "read_only": sc.read_only,
        "editable": sc.editable,
        "duplicate_prevention": sc.duplicate_prevention,
        "persistence": sc.persistence,
        "conditions": sc.has_conditions,
    }

    # 2. Requested behavior (from requirements + salient ticket sentences).
    requested: list[str] = []
    for line in (requirements_text or "").splitlines():
        s = line.strip("-* \t")
        if len(s) > 8:
            requested.append(s[:160])
    requested += [p for p in sc.representative_phrases if p not in requested]
    bu.requested_behavior = requested[:10]

    # 3. Existing capabilities (from inspections).
    bu.existing_capabilities = _capabilities_from_inspections(inspections, relationships)
    # RAG capabilities are SUPPLEMENTARY discovery only (never authoritative).
    rag_caps: list[ExistingCapability] = []
    for rc in (rag_capabilities or []):
        if isinstance(rc, dict):
            nm = rc.get("name") or rc.get("symbol") or ""
            if not nm:
                continue
            rag_caps.append(ExistingCapability(
                name=nm, owner=rc.get("owner", ""), signature=rc.get("signature", ""),
                contract=rc.get("contract", ""), file=rc.get("file", ""),
                evidence_text=" ".join(str(v) for v in rc.values()).lower(),
                provenance="rag", grounded=False,
            ))
    # Grounded capabilities are listed first (authoritative); RAG appended.
    bu.existing_capabilities = bu.existing_capabilities + rag_caps

    # 4. Current behavior (from relationships + inspected facts).
    for rel in relationships[:10]:
        if isinstance(rel, dict):
            src = rel.get("source") or rel.get("src") or ""
            tgt = rel.get("target") or rel.get("dst") or ""
            rtype = rel.get("type") or rel.get("relationship_type") or "relates_to"
            if src and tgt:
                bu.relevant_relationships.append(f"{src} --{rtype}--> {tgt}")
    for insp in inspections.values():
        for f in (getattr(insp, "facts", []) or [])[:2]:
            fact = getattr(f, "fact", "") or ""
            if fact:
                bu.current_behavior.append(fact)
    bu.current_behavior = bu.current_behavior[:8]

    # 5. Reuse decisions — UNDERSTAND before CREATE_NEW.
    # Grounded (relationship-verified) capabilities are authoritative and are
    # evaluated FIRST. RAG capabilities are supplementary — they may confirm a
    # reuse but can NEVER be the reason to CREATE_NEW when grounding is absent.
    grounded_caps = [c for c in bu.existing_capabilities if c.grounded]
    rag_only = [c for c in bu.existing_capabilities if not c.grounded]
    required = _derive_required_capabilities(f"{ticket_title}\n{ticket_description}")
    for req in required:
        cap, ratio = _best_capability_match(req, grounded_caps)
        if cap and ratio >= _REUSE_STRONG:
            decision, reason, sym, owner = "reuse", f"verified capability '{cap.name}' satisfies this", cap.name, cap.owner
        elif cap and ratio >= _REUSE_PARTIAL:
            decision, reason, sym, owner = "adapt", f"verified '{cap.name}' partially satisfies — adapt it", cap.name, cap.owner
        else:
            # Grounded evidence has nothing — consult RAG as a discovery hint only.
            rcap, rratio = _best_capability_match(req, rag_only)
            if rcap and rratio >= _REUSE_STRONG:
                decision, reason, sym, owner = "adapt", (
                    f"RAG-suggested '{rcap.name}' — VERIFY before use (supplementary)"), rcap.name, rcap.owner
            else:
                decision, reason, sym, owner = "create_new", "no existing capability found — requires evidence", "", ""
        bu.reuse_decisions.append(ReuseDecisionRecord(
            requirement=req, decision=decision,
            existing_symbol=sym, owner=owner, reason=reason,
        ))

    # 5b. Back-fill each capability's reuse decision + sufficiency, then order so
    #     reuse-linked and relevant grounded capabilities are surfaced FIRST and
    #     never dropped by the planner-block display cap. This is what keeps
    #     members() visible to the planner as THE reusable capability.
    _suff = {"reuse": "sufficient", "adapt": "adaptable", "extend": "adaptable"}
    _decision_by_symbol: dict[str, str] = {}
    for d in bu.reuse_decisions:
        if d.existing_symbol and d.decision in _suff:
            _decision_by_symbol.setdefault(d.existing_symbol, d.decision)
    for c in bu.existing_capabilities:
        dec = _decision_by_symbol.get(c.name, "")
        if dec:
            c.reuse_decision = dec
            c.sufficiency = _suff[dec]
    bu.existing_capabilities.sort(
        key=lambda c: (
            0 if (c.grounded and c.reuse_decision) else
            1 if (c.grounded and c.relevant) else
            2 if c.grounded else 3
        )
    )

    # 5c. Prove PRIMARY_FEATURE_TARGETs by INSPECTION (Phase B). A reused
    #     capability provider is excluded (its capability is reused, so it is a
    #     read-only reference). The feature component's OWN methods are not a
    #     "provider" — so a feature-anchored file is never treated as a provider.
    _reuse_provider_files = {
        c.file for c in bu.existing_capabilities
        if c.file and c.reuse_decision in ("reuse", "adapt", "extend")
        and _feature_name_match(feature_tokens or [], c.file) < 0.5
    }
    _derived_primary = derive_primary_targets(
        inspections, feature_tokens, _reuse_provider_files, semantic_scores,
    )

    # 6. Behavioral delta (requested behavior not evidently already provided).
    delta: list[str] = []
    if sc.is_per_item:
        delta.append("Introduce PER-ITEM (per selected user/row) state — no global state.")
    if sc.read_only and sc.editable:
        delta.append("Conditionally render read-only vs editable control per item.")
    if sc.duplicate_prevention:
        delta.append("Preserve existing duplicate-prevention behavior.")
    if sc.persistence:
        delta.append("Preserve existing save/persistence behavior.")
    for req in requested[:4]:
        delta.append(f"Satisfy requirement: {req}")
    bu.behavioral_delta = delta[:10]

    # 7. Change candidates vs read-only references — EVIDENCE-PROVEN only.
    # Invariant: DISCOVERED != CHANGE_TARGET. Discovery / relationship / RAG /
    # reuse-provider status grant permission to INVESTIGATE, never to modify. A
    # file becomes a change candidate ONLY when the ticket authorized it (scope)
    # OR evidence proved it directly implements the requested behavior
    # (primary_targets). Everything else — including consumers that merely call a
    # reused capability, providers, and RAG hits — stays a read-only reference.
    bu.relevant_artifacts = evidence_files[:20]

    authorized_norm = {_norm(a) for a in (authorized_files or set())}
    proven_norm = {_norm(p) for p in (primary_targets or set())} | \
        {_norm(p) for p in _derived_primary}

    def _in(nfp: str, s: set) -> bool:
        return any(nfp == a or nfp.endswith(a) or a.endswith(nfp) for a in s if a)

    for fp in evidence_files:
        nfp = _norm(fp)
        if (is_config_file(fp) or is_build_config(fp)) and not ticket_targets_config:
            bu.reference_artifacts.append(fp)
            continue
        if scope_declared and _in(nfp, authorized_norm):
            bu.change_candidates.append(fp)
            continue
        if _in(nfp, proven_norm):
            bu.change_candidates.append(fp)            # evidence-proven feature target
            continue
        bu.reference_artifacts.append(fp)              # discovered != writable

    # Companions of change candidates (e.g. .html/.scss beside a .ts) follow it.
    _promote = [fp for fp in bu.reference_artifacts if is_companion(fp, set(bu.change_candidates))]
    for fp in _promote:
        bu.reference_artifacts.remove(fp)
        bu.change_candidates.append(fp)

    # 8. Unresolved critical + confidence.
    bu.unresolved_critical = list(unresolved or [])[:8]
    signals = sum(bool(x) for x in (
        bu.existing_capabilities, bu.current_behavior, bu.requested_behavior,
        bu.behavioral_delta, bu.change_candidates,
    ))
    bu.confidence = round(min(1.0, 0.2 * signals - 0.15 * bool(bu.unresolved_critical)), 2)
    return bu
