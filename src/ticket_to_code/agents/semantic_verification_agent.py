"""
Semantic Verification Agent — LLM-based, citation-grounded relevance gate

Reads the actual content of each candidate file and uses the LLM to classify
it against the ticket:

  MODIFY    — the ticket's target behavior is implemented here; the change
              must happen in this file.
  REFERENCE — this file will NOT change, but the planner needs it as
              knowledge: callers, contracts, type definitions, consumed
              config/styles, data-flow neighbors.
  EXCLUDE   — keyword coincidence only; no behavior, no planning value.

Design invariants:
  • Grounded verdicts. Every decision MUST cite concrete evidence — line
    numbers or symbol names copied from the provided snippet.  Citations are
    mechanically verified against the snippet; an uncited verdict is
    downgraded (the model cannot pattern-match on file names alone).
  • Decision vocabulary compatibility.  `decision` remains "include" /
    "exclude" for the 4 downstream consumers in workflow.py; MODIFY and
    REFERENCE both map to decision="include" (REFERENCE keeps ~85% of its
    score) so knowledge files are RETAINED, not demoted into oblivion.
  • Fail-open is reserved for single-file (modify-candidate) verification;
    the batch gate retries once, then fails CLOSED while preserving the
    top-ranked few (flagged `verification_status`).
  • Close the loop.  Every verdict is recorded to
    `<workspace>/.aviator/verifier_audit.json`; reconcile_verifier_audit()
    compares verdicts against the files actually modified after generation
    and reports false_include / false_exclude misjudgments.

How top AI IDEs do it:
  - Cursor: "Context Pruning" — reads file content, asks LLM to classify
  - Windsurf: "Relevance Verification" — batch LLM call on file snippets
  - Copilot Workspace: "File Relevance" — semantic check before planning

Author: Deepak Madgani
Date: August 2026 (three-way citation-grounded rebuild: September 2026)
"""

import logging
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Any, Dict, Optional

logger = logging.getLogger(__name__)

# Max characters of file content to send to LLM per candidate
_MAX_SNIPPET_CHARS = 2000
# Max candidates to verify via LLM (supports large multi-file tickets touching 10-20+ files)
_MAX_LLM_VERIFIED = 30
# Preserve-list size when the batch gate fails closed after retries
_FAIL_CLOSED_PRESERVE_TOP = 5
# Audit file cap (entries) to keep the JSON bounded
_AUDIT_MAX_ENTRIES = 500

_VALID_CATEGORIES = ("modify", "reference", "exclude")


@dataclass
class SemanticVerificationResult:
    """Result of semantic verification for a single candidate file."""
    file_path: str
    ranking_score: float
    semantic_relevance_score: float
    decision: str  # "include" | "exclude"  (downstream contract — do not change)
    reason: str
    # Three-way classification: "modify" | "reference" | "exclude".
    # modify/exclude map 1:1 onto decision; reference maps to decision="include".
    category: str = "modify"
    # Concrete evidence the LLM cited for its verdict (line refs / symbols)
    cites: List[str] = field(default_factory=list)
    # "completed" | "not_completed_retry_exhausted" | "not_completed_fail_closed"
    verification_status: str = "completed"


# ── Shared prompt fragments ──────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a precise code reviewer. You classify whether a file must be "
    "MODIFIED for a ticket, is needed as REFERENCE knowledge, or is irrelevant.\n\n"
    "HARD RULES:\n"
    "1. Ground every verdict in the provided content. Cite line numbers and/or\n"
    "   symbol names that literally appear in the snippet you were given.\n"
    "2. Never decide from the file path alone — path names are hints, not proof.\n"
    "3. Respond with JSON ONLY. No markdown fences."
)

# Change-type → expected file roles.  A single include rarely satisfies the
# ticket; the planner needs the SET of roles the change type implies.
_CHANGE_TYPE_GUIDE = """CHANGE-TYPE GUIDE (match the ticket's change type, expect the whole role-set):
- Validation / business-rule change  → MODIFY the validator/service logic; REFERENCE the DTO it returns and the controller exposing it.
- UI text / label change             → MODIFY the template or the i18n source; REFERENCE the component that binds it. NOT every component that renders the string.
- New field / column                 → MODIFY entity + DTO + mapper + API; REFERENCE consumers of the shape.
- Bug in a caller                    → MODIFY the caller; REFERENCE the callee contract.
- Config / environment behavior      → MODIFY the config consumer code; REFERENCE the config file. (A config file alone is rarely the fix.)
- Version / dependency bump          → MODIFY the manifest; EXCLUDE source files that merely mention the version."""

_COUNTER_EVIDENCE = """COUNTER-EVIDENCE CHECKLIST (any YES → lean EXCLUDE or REFERENCE):
- Does the file only CONTAIN the ticket keyword without the behavior it describes?
- Is it a test file, generated code, or a constants/DTO file with no changeable logic?
- Is it a sibling that merely imports or renders the real owner?
- Would the ticket be fixed if this file were never touched?"""


class SemanticVerificationAgent:
    """
    Verifies candidate files against ticket requirements using LLM-based,
    citation-grounded content analysis.
    """

    def __init__(self, workspace_path: Optional[str] = None):
        self._workspace_path = workspace_path
        self._llm = None
        self._audit_records: List[dict] = []

    # ── Infrastructure ────────────────────────────────────────────────────

    def _get_llm(self):
        """Lazy-load LLM to avoid import errors at module level."""
        if self._llm is None:
            try:
                from ticket_to_code.runtime.llm_factory import LLMRegistry
                self._llm = LLMRegistry.get_llm()
            except Exception as e:
                logger.warning(f"[SemanticVerification] Could not load LLM: {e}")
        return self._llm

    def _read_file_snippet(self, file_path: str, evidence_lines: Optional[List[int]] = None) -> str:
        """Read the RELEVANT sections of a file for LLM verification.

        Strategy (AST/symbol-first resolution with bounded fallback):
        1. File signature (first 30 lines) — imports, class/function declarations
        2. AST/symbol-first target resolution — containing symbol/method for each evidence hit
           (bounded line window strictly as fallback when AST resolution fails)
        3. If no evidence, read the next structural section
        """
        if not self._workspace_path:
            return ""
        try:
            full_path = Path(self._workspace_path) / file_path
            if not full_path.exists() or not full_path.is_file():
                return ""
            content = full_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()

            parts: List[str] = []
            used_chars = 0

            # Part 1: File signature (imports, class/function declarations)
            sig_end = min(30, len(lines))
            sig = "\n".join(lines[:sig_end])
            parts.append(f"[FILE SIGNATURE lines 1-{sig_end}]\n{sig}")
            used_chars += len(sig)

            # Part 2: AST/symbol-first target resolution (with bounded fallback)
            if evidence_lines:
                from ticket_to_code.agents.smart_extract import _get_reliable_boundaries, detect_language
                lang = detect_language(file_path)
                boundaries = _get_reliable_boundaries(content, lang) if lang else []

                for eline in sorted(set(evidence_lines))[:5]:
                    if used_chars >= _MAX_SNIPPET_CHARS:
                        break

                    matched_boundary = None
                    for b in boundaries:
                        if b.start_line <= eline <= b.end_line:
                            matched_boundary = b
                            break

                    if matched_boundary:
                        start = max(0, matched_boundary.start_line - 1)
                        end = min(len(lines), matched_boundary.end_line)
                        if start < sig_end:
                            start = sig_end
                        if start < end:
                            region = "\n".join(lines[start:end])
                            parts.append(f"[TARGET SYMBOL: {matched_boundary.name} lines {start+1}-{end}]\n{region}")
                            used_chars += len(region)
                    else:
                        # Fallback to bounded window only when AST/symbol resolution fails
                        start = max(0, eline - 15)
                        end = min(len(lines), eline + 15)
                        if start < sig_end:
                            start = sig_end
                        if start < end:
                            region = "\n".join(lines[start:end])
                            parts.append(f"[FALLBACK CONTEXT lines {start+1}-{end}]\n{region}")
                            used_chars += len(region)
            elif len(lines) > sig_end:
                # Part 3: No evidence lines — read middle structural section
                mid_start = sig_end
                mid_end = min(len(lines), sig_end + 50)
                remaining_budget = _MAX_SNIPPET_CHARS - used_chars
                if remaining_budget > 200:
                    mid_text = "\n".join(lines[mid_start:mid_end])[:remaining_budget]
                    parts.append(f"[BODY lines {mid_start+1}-{mid_end}]\n{mid_text}")

            return "\n\n".join(parts)[:_MAX_SNIPPET_CHARS]
        except Exception as e:
            logger.debug(f"Could not read {file_path}: {e}")
        return ""

    # ── Citation verification (mechanical, no LLM) ────────────────────────

    @staticmethod
    def _verify_citations(snippet: str, cites: List[str]) -> int:
        """Count citations that are mechanically verifiable in the snippet.

        A citation is verified when it is:
          • a number that falls inside the snippet's labeled line ranges
            (e.g. "L84", "line 97", "84-97" against [TARGET SYMBOL … lines
            80-110] or [FILE SIGNATURE lines 1-30]), or
          • a non-numeric string that appears verbatim in the snippet text
            (a symbol name, a code fragment).
        Verdicts with zero verified citations are treated as ungrounded.
        """
        if not cites:
            return 0
        snippet_text = snippet or ""
        # Collect line ranges declared in the snippet's section labels
        ranges: List[tuple] = []
        for m in re.finditer(r"lines\s+(\d+)\s*-\s*(\d+)", snippet_text):
            ranges.append((int(m.group(1)), int(m.group(2))))
        verified = 0
        for cite in cites:
            cite = str(cite).strip()
            if not cite:
                continue
            nums = [int(n) for n in re.findall(r"\d+", cite)]
            if nums and re.fullmatch(r"[Ll]?(line\s*)?\d+(\s*[-–]\s*\d+)?", cite):
                # Numeric citation: valid if any number sits in a labeled range
                if any(lo <= n <= hi for n in nums for (lo, hi) in ranges):
                    verified += 1
                elif not ranges:
                    verified += 1  # snippet had no labeled ranges — accept
            else:
                if cite in snippet_text:
                    verified += 1
        return verified

    @staticmethod
    def _normalize_category(raw: Any, fallback: str = "exclude") -> str:
        cat = str(raw or "").strip().lower()
        if cat in _VALID_CATEGORIES:
            return cat
        return fallback

    @staticmethod
    def _score_for_category(category: str, base_score: float) -> float:
        if category == "modify":
            return float(base_score)
        if category == "reference":
            return float(base_score) * 0.85
        return float(base_score) * 0.3  # exclude

    @staticmethod
    def _decision_for_category(category: str) -> str:
        # MODIFY and REFERENCE are both retained; only EXCLUDE drops out.
        return "include" if category in ("modify", "reference") else "exclude"

    # ── Batch verification (post-ranking gate) ────────────────────────────

    def verify_candidates(
        self,
        ticket: Any,
        candidates: List[Any],
        evidence_items: Optional[List[Any]] = None,
        top_n: int = 20,
    ) -> List[SemanticVerificationResult]:
        """
        Verify a list of ranked candidate files against the ticket.

        Uses LLM to read each file's content and classify it as MODIFY /
        REFERENCE / EXCLUDE with mandatory citations.  Retries once on LLM
        failure, then fails CLOSED (only the top-ranked few survive, flagged).
        Falls back to pass-through only when no LLM/workspace exists at all.
        """
        results: List[SemanticVerificationResult] = []
        llm = self._get_llm()

        if not llm or not self._workspace_path:
            logger.warning(
                "[SemanticVerification] LLM or workspace not available — "
                "falling back to pass-through mode"
            )
            passthrough = self._passthrough(candidates, top_n)
            self.record_verdicts(ticket, passthrough)
            self.flush_audit()
            return passthrough

        ticket_title = getattr(ticket, "title", "") or ""
        ticket_desc = getattr(ticket, "description", "") or ""

        # Collect evidence context per file — snippets AND line numbers
        evidence_by_file: Dict[str, List[str]] = {}
        evidence_lines_by_file: Dict[str, List[int]] = {}
        for e in (evidence_items or []):
            fp = getattr(e, "file_path", "")
            if fp:
                snippet = getattr(e, "content_snippet", "")[:200]
                evidence_by_file.setdefault(fp, []).append(snippet)
                line_num = getattr(e, "line_start", None) or getattr(e, "line_number", None)
                if not line_num:
                    m = re.search(r'line\s+(\d+)', snippet)
                    if m:
                        line_num = int(m.group(1))
                if line_num and isinstance(line_num, int):
                    evidence_lines_by_file.setdefault(fp, []).append(line_num)

        eval_candidates = candidates[:min(top_n, _MAX_LLM_VERIFIED)]

        file_entries: List[dict] = []
        for idx, candidate in enumerate(eval_candidates):
            fp = getattr(candidate, "file_path", None) or getattr(candidate, "path", "unknown")
            score = getattr(candidate, "final_score", 0.0) or getattr(candidate, "ranking_score", 0.0)
            ev_lines = evidence_lines_by_file.get(fp, [])
            snippet = self._read_file_snippet(fp, evidence_lines=ev_lines)
            ev_snippets = evidence_by_file.get(fp, [])
            ev_text = "; ".join(ev_snippets[:3]) if ev_snippets else "No direct evidence"

            file_entries.append({
                "index": idx,
                "file_path": fp,
                "score": float(score),
                "snippet": snippet if snippet else "(file not readable)",
                "evidence": ev_text[:300],
            })

        # ── LLM call with ONE retry; then fail closed ─────────────────────
        llm_results: Optional[Dict[int, dict]] = None
        last_error: Optional[Exception] = None
        for attempt in (1, 2):
            try:
                llm_results = self._batch_verify(llm, ticket_title, ticket_desc, file_entries)
                break
            except Exception as e:
                last_error = e
                logger.warning(
                    f"[SemanticVerification] LLM batch verify attempt {attempt} failed: {e}"
                )

        if llm_results is None:
            # Fail closed: keep only the top-ranked few, flag everything.
            logger.warning(
                "[SemanticVerification] Batch verification failed twice — "
                f"failing CLOSED (preserving top {_FAIL_CLOSED_PRESERVE_TOP} by rank, "
                "excluding the rest; do not treat these as LLM judgments)"
            )
            for idx, candidate in enumerate(eval_candidates):
                fp = getattr(candidate, "file_path", None) or getattr(candidate, "path", "unknown")
                score = getattr(candidate, "final_score", 0.0) or getattr(candidate, "ranking_score", 0.0)
                keep = idx < _FAIL_CLOSED_PRESERVE_TOP
                results.append(SemanticVerificationResult(
                    file_path=fp,
                    ranking_score=float(score),
                    semantic_relevance_score=float(score) if keep else float(score) * 0.3,
                    decision="include" if keep else "exclude",
                    reason=(
                        f"Verification infrastructure failed twice ({last_error}) — "
                        + ("preserved by rank, NOT LLM-verified" if keep else "failed closed")
                    ),
                    category="modify" if keep else "exclude",
                    verification_status="not_completed_fail_closed",
                ))
            self.record_verdicts(ticket, results)
            self.flush_audit()
            return results

        # ── Map results with citation grounding ───────────────────────────
        for idx, candidate in enumerate(eval_candidates):
            fp = getattr(candidate, "file_path", None) or getattr(candidate, "path", "unknown")
            score = getattr(candidate, "final_score", 0.0) or getattr(candidate, "ranking_score", 0.0)
            snippet = file_entries[idx]["snippet"]

            llm_dec = llm_results.get(idx, {})
            category = self._normalize_category(llm_dec.get("category"), fallback="exclude")
            reason = llm_dec.get("reason", "LLM verification") or "LLM verification"
            cites = [str(c) for c in (llm_dec.get("cites") or []) if str(c).strip()][:5]

            # Mechanical grounding check — an uncited verdict is downgraded,
            # never silently trusted.
            n_verified = self._verify_citations(snippet, cites)
            if n_verified == 0 and category != "exclude":
                logger.info(
                    f"  [SemanticVerification] ⚠️ Uncited {category} verdict for {fp} — "
                    "downgrading to reference-weight"
                )
                reason = f"{reason} [uncited — downgraded]"
                sem_score = float(score) * 0.6
            else:
                sem_score = self._score_for_category(category, score)

            decision = self._decision_for_category(category)
            results.append(SemanticVerificationResult(
                file_path=fp,
                ranking_score=float(score),
                semantic_relevance_score=sem_score,
                decision=decision,
                reason=reason,
                cites=cites,
            ))

        # ── Domain Ownership Override ─────────────────────────────────────
        # Architectural guardrail trumps the LLM — but only for files the LLM
        # wants to MODIFY.  REFERENCE files are read-only by definition, so
        # cross-service contracts remain available to the planner.
        try:
            from ticket_to_code.agents.architecture_model import ArchitectureModel
            from ticket_to_code.agents.domain_resolver import DomainResolver
            from ticket_to_code.agents.ownership_resolver import OwnershipResolver

            _arch_model = None
            if self._workspace_path:
                for _yaml_loc in [
                    Path(self._workspace_path) / "brain" / "knowledge" / "architecture_model.yaml",
                    Path(__file__).parent.parent / "config" / "architecture_model.yaml",
                ]:
                    if _yaml_loc.exists():
                        _arch_model = ArchitectureModel.load_from_yaml(str(_yaml_loc))
                        break

            if _arch_model and _arch_model.services:
                _domain_resolver = DomainResolver(_arch_model)
                _domain_resolution = _domain_resolver.resolve(ticket_title, ticket_desc)

                if _domain_resolution.has_resolution:
                    _ownership_resolver = OwnershipResolver(_arch_model)
                    _all_domains = _domain_resolution.primary_domains + _domain_resolution.secondary_domains
                    _overridden = 0

                    for r in results:
                        if r.decision != "include" or r.category == "reference":
                            continue  # excluded already, or read-only reference — skip
                        _ownership = _ownership_resolver.classify(r.file_path, _all_domains)
                        if _ownership.policy.value in ("DO_NOT_MODIFY", "REFERENCE_ONLY"):
                            r.semantic_relevance_score = 0.0
                            r.decision = "exclude"
                            r.category = "exclude"
                            r.reason = (
                                f"Domain ownership: {r.file_path} belongs to "
                                f"'{_ownership.service_name}', but capability "
                                f"'{_ownership.matched_capability}' is owned by "
                                f"'{_ownership.expected_service}' — REFERENCE_ONLY"
                            )
                            _overridden += 1
                            logger.info(
                                f"  [SemanticVerification] 🏗️ Domain override: "
                                f"{r.file_path} → excluded (was included by LLM)"
                            )

                    if _overridden:
                        logger.info(
                            f"  [SemanticVerification] 🏗️ {_overridden} file(s) "
                            f"overridden by domain ownership"
                        )
        except Exception as _domain_err:
            logger.debug(
                f"[SemanticVerification] Domain ownership check skipped: {_domain_err}"
            )

        included = len([r for r in results if r.decision == "include"])
        excluded = len([r for r in results if r.decision != "include"])
        logger.info(
            f"[SemanticVerification] Verified {len(results)} candidates: "
            f"{included} included "
            f"(modify={sum(1 for r in results if r.category == 'modify')}, "
            f"reference={sum(1 for r in results if r.category == 'reference')}), "
            f"{excluded} excluded"
        )
        for r in results:
            logger.info(
                f"  [{r.category:9s}] score={r.ranking_score:.2f} "
                f"sem={r.semantic_relevance_score:.2f} {r.file_path} — {r.reason[:80]}"
            )

        self.record_verdicts(ticket, results)
        self.flush_audit()
        return results

    def _batch_verify(
        self,
        llm: Any,
        ticket_title: str,
        ticket_desc: str,
        file_entries: List[dict],
    ) -> Dict[int, dict]:
        """
        Single LLM call to classify all candidate files at once.
        Returns dict mapping file index → {"category", "decision", "reason", "cites"}.
        Raises on parse failure (caller retries, then fails closed).
        """
        from langchain_core.messages import SystemMessage, HumanMessage
        from ticket_to_code.llm_utils import llm_invoke

        rules = (
            "For EACH file decide ONE category:\n"
            '  "modify"    — the ticket\'s target behavior is implemented in this file; '
            "the required change must happen here.\n"
            '  "reference" — this file will NOT change, but the planner NEEDS it as '
            "knowledge (caller, contract, type, consumed config/style, data-flow neighbor).\n"
            '  "exclude"   — keyword coincidence only; no behavior and no planning value.\n\n'
            + _CHANGE_TYPE_GUIDE + "\n\n" + _COUNTER_EVIDENCE + "\n\n"
            "CITATION RULE: every non-exclude verdict MUST include 1-3 citations — "
            "line numbers and/or symbol names copied from the file snippet above. "
            "Uncited verdicts are treated as ungrounded and downgraded.\n\n"
            "RESPOND with JSON ONLY. Format:\n"
            '[\n'
            '  {"index": 0, "category": "modify", "cites": ["L84", "validateDuplicate()"],\n'
            '   "reason": "Contains the duplicate-email validation the ticket changes"},\n'
            '  {"index": 1, "category": "reference", "cites": ["MemberDto"],\n'
            '   "reason": "Defines the DTO shape this ticket extends — read-only"},\n'
            '  {"index": 2, "category": "exclude", "cites": [],\n'
            '   "reason": "Keyword match only; unrelated module"}\n'
            ']\n'
        )

        files_text = ""
        for entry in file_entries:
            files_text += (
                f"\n--- File {entry['index']}: {entry['file_path']} "
                f"(score={entry['score']:.2f}) ---\n"
                f"Evidence: {entry['evidence']}\n"
                f"Content:\n{entry['snippet']}\n"
            )

        user_prompt = (
            f"TICKET: {ticket_title}\n"
            f"DESCRIPTION: {ticket_desc[:500]}\n\n"
            f"CANDIDATE FILES TO VERIFY:\n{files_text}\n\n"
            "Classify each file: modify / reference / exclude, with citations."
        )

        response = llm_invoke(llm, [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        # Parse — raise on failure so the caller's retry/fail-closed path runs
        from ticket_to_code.json_utils import parse_llm_json
        text = response.content if hasattr(response, "content") else str(response)
        decisions = parse_llm_json(text, expect_list=True)

        result_map: Dict[int, dict] = {}
        if isinstance(decisions, list):
            for d in decisions:
                if isinstance(d, dict) and "index" in d:
                    result_map[d["index"]] = {
                        "category": d.get("category", ""),
                        "reason": d.get("reason", ""),
                        "cites": d.get("cites", []) or [],
                    }
        if not result_map:
            raise ValueError("LLM batch verification returned no per-file decisions")
        return result_map

    # ── Single-file verification (in-loop, agentic) ───────────────────────

    def verify_single(
        self,
        ticket: Any,
        file_item: Any,
        existing_evidence: Dict[str, dict],
    ) -> "SemanticVerificationResult":
        """
        Verify a single file for relevance to the ticket (agentic loop path).

        Three-way classification with citations; ONE retry on infrastructure
        error; final fallback stays fail-open (a candidate about to be planned
        should reach the planner flagged rather than vanish), but the flag is
        recorded in verification_status so downstream gates can react.
        """
        from langchain_core.messages import SystemMessage, HumanMessage
        from ticket_to_code.llm_utils import llm_invoke

        fp = getattr(file_item, "file_path", "") or ""
        score = getattr(file_item, "relevance_score", 0.0) or 0.0
        snippet = getattr(file_item, "content_snippet", "") or ""
        relationship = getattr(file_item, "relationship", "") or ""
        details = getattr(file_item, "details", "") or ""

        from ticket_to_code.utils.file_filters import is_ignored_path
        if fp and is_ignored_path(fp, workspace_path=getattr(self, "workspace_path", None)):
            logger.info(f"  [verify_single] [exclude  ] {fp} - Auto-excluded build artifact/ignored path (0 tokens)")
            return SemanticVerificationResult(
                file_path=fp,
                ranking_score=score,
                semantic_relevance_score=0.0,
                decision="exclude",
                reason="Build artifact, compiled chunk, or ignored path",
                category="exclude",
                cites=[],
            )

        file_content = self._read_file_snippet(fp)

        # Data-flow / graph context: what we already know about this file's
        # relationships — the LLM judges position, not just content.
        context_files = []
        for efp, ev in list(existing_evidence.items())[:5]:
            if ev.get("relevant"):
                context_files.append(f"  - {efp} (relevant: {ev.get('reason', '')[:60]})")
        context_str = "\n".join(context_files) if context_files else "  (no files verified yet)"
        rel_str = f"\n  Discovery relationship: {relationship}" if relationship else ""
        det_str = f"\n  Discovery details: {str(details)[:150]}" if details else ""

        ticket_title = getattr(ticket, "title", "") or ""
        ticket_desc = getattr(ticket, "description", "") or ""

        llm = self._get_llm()
        if not llm:
            return SemanticVerificationResult(
                file_path=fp,
                ranking_score=float(score),
                semantic_relevance_score=float(score),
                decision="include",
                reason="No LLM available for verification - included by default",
                category="modify",
                verification_status="not_completed_retry_exhausted",
            )

        prompt = (
            f"TICKET:\n"
            f"  Title: {ticket_title}\n"
            f"  Description: {ticket_desc[:400]}\n\n"
            f"FILE TO VERIFY:\n"
            f"  Path: {fp}"
            f"{rel_str}{det_str}\n"
            f"  Evidence snippet: {snippet[:200]}\n\n"
            f"FILE CONTENT (first sections):\n"
            f"{file_content[:2000]}\n\n"
            f"ALREADY VERIFIED FILES:\n{context_str}\n\n"
            f"Classify this file as ONE of:\n"
            f'  "modify"    — the ticket\'s behavior is implemented here; the change must happen in this file\n'
            f'  "reference" — will NOT change, but the planner needs it (caller / contract / type / consumed config)\n'
            f'  "exclude"   — keyword coincidence only\n\n'
            + _CHANGE_TYPE_GUIDE + "\n\n" + _COUNTER_EVIDENCE + "\n\n"
            f"CITATION RULE: for modify/reference, `cites` MUST list 1-3 line numbers "
            f"or symbol names that appear verbatim in the FILE CONTENT above.\n\n"
            f"Answer with a JSON object:\n"
            f'{{"category": "modify"|"reference"|"exclude", '
            f'"cites": ["L84", "methodName"], "reason": "..."}}\n'
            f"Return ONLY the JSON."
        )

        last_exc: Optional[Exception] = None
        for attempt in (1, 2):
            try:
                from ticket_to_code.json_utils import parse_llm_json

                messages = [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
                response = llm_invoke(llm, messages)
                text = response.content if hasattr(response, "content") else str(response)
                result = parse_llm_json(text)

                if not isinstance(result, dict):
                    raise ValueError(f"Non-dict LLM verification response: {str(result)[:100]}")

                category = self._normalize_category(result.get("category"), fallback="modify")
                reason = result.get("reason", "LLM verification") or "LLM verification"
                cites = [str(c) for c in (result.get("cites") or []) if str(c).strip()][:5]

                # Mechanical citation check with one corrective re-ask
                n_verified = self._verify_citations(file_content, cites)
                if n_verified == 0 and category != "exclude" and attempt == 1:
                    logger.info(
                        f"  [verify_single] ⚠️ Uncited {category} verdict for {fp} — re-asking with grounding requirement"
                    )
                    prompt = (
                        f"YOUR PREVIOUS ANSWER LACKED VERIFIABLE CITATIONS.\n"
                        f"Re-answer for file {fp}. `cites` must contain line numbers and/or "
                        f"symbol names that appear verbatim in the FILE CONTENT above. "
                        f"If you cannot cite anything, the category is \"exclude\".\n\n"
                        + prompt
                    )
                    continue

                if n_verified == 0 and category != "exclude":
                    reason = f"{reason} [uncited — downgraded]"
                    sem_score = float(score) * 0.6
                else:
                    sem_score = self._score_for_category(category, score)

                decision = self._decision_for_category(category)
                logger.info(
                    f"  [verify_single] [{category:9s}] {fp} - {reason[:80]} "
                    f"(cites verified: {n_verified}/{len(cites)})"
                )

                out = SemanticVerificationResult(
                    file_path=fp,
                    ranking_score=float(score),
                    semantic_relevance_score=sem_score,
                    decision=decision,
                    reason=reason,
                    category=category,
                    cites=cites,
                )
                self.record_verdicts(ticket, [out])
                self.flush_audit()
                return out

            except Exception as e:
                last_exc = e
                logger.warning(f"[verify_single] Attempt {attempt} failed for {fp}: {e}")

        # Both attempts failed — fail open (modify-candidate philosophy), flagged
        out = SemanticVerificationResult(
            file_path=fp,
            ranking_score=float(score),
            semantic_relevance_score=float(score),
            decision="include",
            reason=f"Verification error after retry ({last_exc}) - included by default, NOT LLM-verified",
            category="modify",
            verification_status="not_completed_retry_exhausted",
        )
        self.record_verdicts(ticket, [out])
        self.flush_audit()
        return out

    # ── Audit loop: measure the verifier's own precision ──────────────────

    def record_verdicts(self, ticket: Any, results: List[SemanticVerificationResult]) -> None:
        """Buffer verdicts for the audit trail (flushed to .aviator/verifier_audit.json)."""
        ticket_id = getattr(ticket, "ticket_id", "") or getattr(ticket, "id", "") or ""
        for r in results:
            self._audit_records.append({
                "ticket_id": str(ticket_id),
                "file_path": r.file_path,
                "category": r.category,
                "decision": r.decision,
                "reason": r.reason[:300],
                "cites": r.cites,
                "verification_status": r.verification_status,
                "ts": datetime.now(timezone.utc).isoformat(),
            })

    def flush_audit(self) -> None:
        """Append buffered verdicts to <workspace>/.aviator/verifier_audit.json."""
        if not self._audit_records or not self._workspace_path:
            return
        try:
            audit_path = Path(self._workspace_path) / ".aviator" / "verifier_audit.json"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            existing: List[dict] = []
            if audit_path.exists():
                try:
                    existing = json.loads(audit_path.read_text(encoding="utf-8"))
                    if not isinstance(existing, list):
                        existing = []
                except Exception:
                    existing = []
            existing.extend(asdict_rec for asdict_rec in self._audit_records)
            existing = existing[-_AUDIT_MAX_ENTRIES:]
            audit_path.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.debug(
                f"[SemanticVerification] Audit flushed: {len(self._audit_records)} verdict(s) → {audit_path}"
            )
            self._audit_records.clear()
        except Exception as exc:
            logger.debug(f"[SemanticVerification] Audit flush failed (non-fatal): {exc}")

    # ── Fallback ──────────────────────────────────────────────────────────

    def _passthrough(
        self,
        candidates: List[Any],
        top_n: int,
    ) -> List[SemanticVerificationResult]:
        """Fallback: include all candidates without LLM verification."""
        results = []
        for candidate in candidates[:top_n]:
            fp = getattr(candidate, "file_path", None) or getattr(candidate, "path", "unknown")
            score = getattr(candidate, "final_score", 0.0) or getattr(candidate, "ranking_score", 0.0)
            results.append(SemanticVerificationResult(
                file_path=fp,
                ranking_score=float(score),
                semantic_relevance_score=float(score),
                decision="include",
                reason="Pass-through (LLM unavailable)",
                category="modify",
                verification_status="not_completed_retry_exhausted",
            ))
        logger.info(f"[SemanticVerification] Pass-through: {len(results)} candidates included")
        return results


# ── Module-level reconciliation (close the loop after generation) ────────────

def reconcile_verifier_audit(
    workspace_path: str,
    modified_files: List[str],
    ticket_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare recorded verdicts against the files ACTUALLY modified after
    generation and report the verifier's misjudgments:

      false_exclude — the verifier excluded a file that had to be changed
                      (the worst failure: evidence gate blocked real work)
      false_include — the verifier included (as modify) a file that was
                      never touched (planner noise)

    Writes <workspace>/.aviator/verifier_audit_report.json and returns the
    report dict.  Call this after the patch/apply phase with the list of
    files the run actually wrote.
    """
    audit_path = Path(workspace_path) / ".aviator" / "verifier_audit.json"
    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticket_id": ticket_id,
        "modified_files": list(modified_files),
        "false_exclude": [],
        "false_include": [],
        "correct_modify": 0,
        "correct_exclude": 0,
        "precision_note": "",
    }
    if not audit_path.exists():
        report["precision_note"] = "No audit trail found — nothing to reconcile."
        return report

    try:
        entries = json.loads(audit_path.read_text(encoding="utf-8"))
    except Exception as exc:
        report["precision_note"] = f"Audit file unreadable: {exc}"
        return report

    modified_norm = {str(p).replace("\\", "/").lower() for p in (modified_files or [])}

    for entry in entries:
        if ticket_id and str(entry.get("ticket_id", "")) != str(ticket_id):
            continue
        fp = str(entry.get("file_path", "")).replace("\\", "/").lower()
        category = entry.get("category", "")
        was_modified = fp in modified_norm
        if category == "modify" and not was_modified and entry.get("verification_status") == "completed":
            report["false_include"].append({"file": fp, "reason": entry.get("reason", "")[:200]})
        elif category == "exclude" and was_modified:
            report["false_exclude"].append({"file": fp, "reason": entry.get("reason", "")[:200]})
        elif category == "modify" and was_modified:
            report["correct_modify"] += 1
        elif category == "exclude" and not was_modified:
            report["correct_exclude"] += 1

    n_false_ex = len(report["false_exclude"])
    n_false_in = len(report["false_include"])
    report["precision_note"] = (
        f"correct_modify={report['correct_modify']}, correct_exclude={report['correct_exclude']}, "
        f"false_include={n_false_in}, false_exclude={n_false_ex}"
    )
    if n_false_ex or n_false_in:
        logger.warning(
            f"[VerifierAudit] ⚠️ {report['precision_note']} — "
            f"false_exclude entries should tune ranking weights / prompts"
        )
    else:
        logger.info(f"[VerifierAudit] ✅ {report['precision_note']}")

    try:
        report_path = Path(workspace_path) / ".aviator" / "verifier_audit_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:
        logger.debug(f"[VerifierAudit] Report write failed (non-fatal): {exc}")

    return report
