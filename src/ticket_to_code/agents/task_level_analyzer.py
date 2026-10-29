"""
Semantic Capability Resolver

Answers the RIGHT question for every planned task:

    "Does existing code already SOLVE this task — regardless of what it is named —
     so we should REUSE it, or does nothing solve it, so we must CREATE it?"

This is CAPABILITY / INTENT based, NOT name based. We never look up a fixed
symbol name like `getAllUsers` and check if it exists. Instead we describe the
INTENT of the task ("list every user in the current subscription"), semantically
retrieve real code that might fulfil that intent, and let an LLM judge whether any
candidate actually accomplishes it — exactly how a human (or Claude) would find an
existing method that solves a task even when its name is completely different.

Pipeline per task:
  1. Gather candidates   — semantic (RAG) + literal (repository search) retrieval
  2. Judge candidates    — LLM decides: does any candidate SOLVE the intent?
  3. Return resolution   — REUSE_EXISTING (with the real symbol) / EXTEND / CREATE_NEW

Degrades gracefully: works with only RAG, only repo-search, or a pure heuristic
fallback when no LLM/retrieval is available.

Backwards-compat: the previous name-based ``TaskLevelAnalyzer`` API is preserved
as a thin adapter so any existing imports keep working, but it now delegates to
the capability resolver instead of matching fixed names.

Author: Deepak Madgani
Date: August 2026
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ResolutionDecision(str, Enum):
    """What to do about a task, decided by capability (not name) matching."""
    REUSE_EXISTING = "reuse_existing"    # Existing code already fulfils the intent
    EXTEND_EXISTING = "extend_existing"  # Existing code partially fulfils it; extend
    CREATE_NEW = "create_new"            # Nothing fulfils it; create new code
    UNKNOWN = "unknown"                  # Not enough signal to decide


@dataclass
class CapabilityCandidate:
    """A real piece of existing code that MIGHT fulfil a task's intent."""
    file_path: str
    symbol_name: str            # the ACTUAL name in the code, whatever it is
    symbol_kind: str            # method | function | class | property | unknown
    snippet: str                # code excerpt used for judgement
    similarity: float = 0.0     # retrieval score (semantic or lexical)
    source: str = "semantic"    # semantic | repository | graph

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "symbol_name": self.symbol_name,
            "symbol_kind": self.symbol_kind,
            "similarity": round(self.similarity, 4),
            "source": self.source,
        }


@dataclass
class CapabilityResolution:
    """The decision for one task: reuse existing capability or create new."""
    task_intent: str
    decision: ResolutionDecision
    target_file: Optional[str] = None      # real file to reuse/extend (None => create)
    target_symbol: Optional[str] = None    # real existing symbol to reuse/extend
    reasoning: str = ""
    confidence: float = 0.0
    candidates_considered: List[CapabilityCandidate] = field(default_factory=list)

    @property
    def should_reuse(self) -> bool:
        return self.decision in (
            ResolutionDecision.REUSE_EXISTING,
            ResolutionDecision.EXTEND_EXISTING,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_intent": self.task_intent,
            "decision": self.decision.value,
            "target_file": self.target_file,
            "target_symbol": self.target_symbol,
            "reasoning": self.reasoning,
            "confidence": round(self.confidence, 4),
            "candidates_considered": [c.to_dict() for c in self.candidates_considered],
        }


class SemanticCapabilityResolver:
    """
    Resolve tasks by CAPABILITY, not by name.

    Usage:
        resolver = SemanticCapabilityResolver(
            rag_engine=agents.rag_engine,
            repo_search_engine=repo_search,
            llm=LLMRegistry.get_llm(assistant=True),
            workspace_path=workspace_path,
        )

        resolution = resolver.resolve_task(
            intent="Filter the member list by the selected organization",
            title="Add organization filter to members list",
            hints={"file_path": "add-members.component.ts", "language": "typescript"},
        )

        if resolution.should_reuse:
            # Point the generator at resolution.target_file / resolution.target_symbol
            # (the REAL existing symbol, whatever its name) instead of creating a duplicate.
            ...
        else:
            # Nothing solves it — create new code.
            ...
    """

    def __init__(
        self,
        rag_engine: Any = None,
        repo_search_engine: Any = None,
        llm: Any = None,
        workspace_path: Optional[str] = None,
        max_candidates: int = 8,
    ):
        self.rag_engine = rag_engine
        self.repo_search_engine = repo_search_engine
        self.llm = llm
        self.workspace_path = workspace_path
        self.max_candidates = max_candidates

    # ── Public API ────────────────────────────────────────────────────────────

    def resolve_task(
        self,
        intent: str,
        title: str = "",
        hints: Optional[Dict[str, Any]] = None,
    ) -> CapabilityResolution:
        """
        Decide whether existing code already solves `intent` (reuse) or not (create).

        Args:
            intent: Natural-language description of WHAT the task must accomplish.
            title:  Short task title (optional, adds signal).
            hints:  Optional {file_path, language, keywords: [...]} to focus retrieval.
        """
        hints = hints or {}
        query = self._build_query(intent, title, hints)
        logger.info(f"[capability] resolving intent: {query[:90]}")

        candidates = self._gather_candidates(query, hints)
        if not candidates:
            # No existing code retrieved for this intent → safe to create new.
            return CapabilityResolution(
                task_intent=intent,
                decision=ResolutionDecision.CREATE_NEW,
                reasoning="No existing code fulfils this intent (no candidates retrieved).",
                confidence=0.55,
            )

        # Judge whether any candidate ACTUALLY solves the intent (capability-based).
        resolution = self._judge_candidates(intent, title, candidates)
        resolution.candidates_considered = candidates
        logger.info(
            f"[capability] decision={resolution.decision.value} "
            f"target={resolution.target_symbol or '-'} conf={resolution.confidence:.2f}"
        )
        return resolution

    def resolve_batch(self, tasks: List[Dict[str, Any]]) -> List[CapabilityResolution]:
        """Resolve many tasks. Each task: {intent, title?, hints?}."""
        return [
            self.resolve_task(
                intent=t.get("intent", ""),
                title=t.get("title", ""),
                hints=t.get("hints"),
            )
            for t in tasks
        ]

    # ── Candidate gathering (semantic + literal) ──────────────────────────────

    def _build_query(self, intent: str, title: str, hints: Dict[str, Any]) -> str:
        parts = [p for p in (title, intent) if p]
        kw = hints.get("keywords") or []
        if kw:
            parts.append(" ".join(str(k) for k in kw))
        return " ".join(parts).strip() or intent

    def _gather_candidates(self, query: str, hints: Dict[str, Any]) -> List[CapabilityCandidate]:
        candidates: List[CapabilityCandidate] = []

        # 1) Semantic retrieval via RAG (embedding similarity — intent based)
        if self.rag_engine is not None:
            try:
                chunks = self.rag_engine.retrieve_context(
                    query=query,
                    max_results=self.max_candidates,
                    schema="code",
                ) or []
                for ch in chunks:
                    fp = ch.get("file_path", "Unknown")
                    content = ch.get("content", "") or ""
                    for cand in self._extract_symbols_from_chunk(fp, content, source="semantic"):
                        candidates.append(cand)
            except Exception as e:
                logger.debug(f"[capability] RAG retrieval failed: {e}")

        # 2) Literal / filename retrieval for files not in the vector store
        if self.repo_search_engine is not None:
            try:
                for term in self._literal_terms(query, hints):
                    results = self.repo_search_engine.search_literal(term) or []
                    for r in results[: self.max_candidates]:
                        fp = getattr(r, "file_path", None) or (r.get("file_path") if isinstance(r, dict) else None)
                        snippet = getattr(r, "snippet", "") or (r.get("snippet", "") if isinstance(r, dict) else "")
                        if not fp:
                            continue
                        candidates.append(
                            CapabilityCandidate(
                                file_path=fp,
                                symbol_name=self._guess_symbol(snippet) or "(file)",
                                symbol_kind="unknown",
                                snippet=snippet[:400],
                                similarity=float(getattr(r, "confidence", 0.4) or 0.4),
                                source="repository",
                            )
                        )
            except Exception as e:
                logger.debug(f"[capability] repository search failed: {e}")

        return self._dedupe_candidates(candidates)[: self.max_candidates]

    def _literal_terms(self, query: str, hints: Dict[str, Any]) -> List[str]:
        terms = list(hints.get("keywords") or [])
        # Pull the most content-bearing words from the intent as literal probes.
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", query) if not _is_stopword(w)]
        terms.extend(words[:4])
        seen, out = set(), []
        for t in terms:
            k = t.lower()
            if k not in seen:
                seen.add(k)
                out.append(t)
        return out[:5]

    def _extract_symbols_from_chunk(self, file_path: str, content: str, source: str) -> List[CapabilityCandidate]:
        """Pull method/function/class declarations out of a retrieved code chunk."""
        out: List[CapabilityCandidate] = []
        for kind, name in _iter_declarations(content):
            out.append(
                CapabilityCandidate(
                    file_path=file_path,
                    symbol_name=name,
                    symbol_kind=kind,
                    snippet=content[:500],
                    similarity=0.6,
                    source=source,
                )
            )
        if not out:
            # Chunk with no clear declaration — still record the file as a weak candidate.
            out.append(
                CapabilityCandidate(
                    file_path=file_path,
                    symbol_name="(file)",
                    symbol_kind="unknown",
                    snippet=content[:500],
                    similarity=0.4,
                    source=source,
                )
            )
        return out

    def _guess_symbol(self, snippet: str) -> Optional[str]:
        for _, name in _iter_declarations(snippet):
            return name
        return None

    def _dedupe_candidates(self, candidates: List[CapabilityCandidate]) -> List[CapabilityCandidate]:
        best: Dict[str, CapabilityCandidate] = {}
        for c in candidates:
            key = f"{c.file_path}::{c.symbol_name}"
            if key not in best or c.similarity > best[key].similarity:
                best[key] = c
        return sorted(best.values(), key=lambda c: c.similarity, reverse=True)

    # ── Judgement (capability, not name) ──────────────────────────────────────

    def _judge_candidates(
        self,
        intent: str,
        title: str,
        candidates: List[CapabilityCandidate],
    ) -> CapabilityResolution:
        if self.llm is not None:
            try:
                return self._judge_with_llm(intent, title, candidates)
            except Exception as e:
                logger.debug(f"[capability] LLM judge failed, falling back: {e}")
        return self._judge_heuristic(intent, candidates)

    def _judge_with_llm(
        self,
        intent: str,
        title: str,
        candidates: List[CapabilityCandidate],
    ) -> CapabilityResolution:
        from langchain_core.messages import SystemMessage, HumanMessage
        from ticket_to_code.llm_utils import llm_invoke

        listing = []
        for i, c in enumerate(candidates, 1):
            listing.append(
                f"[{i}] file={c.file_path} symbol={c.symbol_name} kind={c.symbol_kind}\n"
                f"    code:\n{_indent(c.snippet, 6)}"
            )
        candidates_block = "\n".join(listing)

        system = (
            "You are a senior engineer deciding whether an existing codebase ALREADY solves a task. "
            "Judge by CAPABILITY and INTENT, never by name. A method named completely differently "
            "still counts as solving the task if its behaviour accomplishes the intent. "
            "Only choose CREATE_NEW when NONE of the candidates accomplish the intent.\n\n"
            "Return STRICT JSON only:\n"
            "{\n"
            '  "decision": "reuse_existing" | "extend_existing" | "create_new",\n'
            '  "target_file": string | null,   // real file of the chosen candidate\n'
            '  "target_symbol": string | null, // real existing symbol name to reuse/extend\n'
            '  "confidence": number,           // 0.0 - 1.0\n'
            '  "reasoning": string             // why, referencing behaviour not names\n'
            "}"
        )
        human = (
            f"TASK TITLE: {title or '(none)'}\n"
            f"TASK INTENT (what must be accomplished):\n{intent}\n\n"
            f"EXISTING CODE CANDIDATES:\n{candidates_block}\n\n"
            "Decide: does any candidate already accomplish the intent (reuse/extend), "
            "or must new code be created? Return JSON only."
        )

        response = llm_invoke(self.llm, [SystemMessage(content=system), HumanMessage(content=human)])
        data = _parse_json(getattr(response, "content", "") or "")
        if not data:
            return self._judge_heuristic(intent, candidates)

        decision = _coerce_decision(data.get("decision"))
        return CapabilityResolution(
            task_intent=intent,
            decision=decision,
            target_file=data.get("target_file") if decision != ResolutionDecision.CREATE_NEW else None,
            target_symbol=data.get("target_symbol") if decision != ResolutionDecision.CREATE_NEW else None,
            reasoning=str(data.get("reasoning", ""))[:600],
            confidence=_clamp(float(data.get("confidence", 0.6) or 0.6)),
        )

    def _judge_heuristic(self, intent: str, candidates: List[CapabilityCandidate]) -> CapabilityResolution:
        """No-LLM fallback: overlap between intent tokens and candidate code tokens."""
        intent_tokens = _tokens(intent)
        best: Optional[CapabilityCandidate] = None
        best_overlap = 0.0
        for c in candidates:
            cand_tokens = _tokens(c.symbol_name + " " + c.snippet)
            if not cand_tokens:
                continue
            overlap = len(intent_tokens & cand_tokens) / max(1, len(intent_tokens))
            score = overlap * (0.5 + 0.5 * c.similarity)
            if score > best_overlap:
                best_overlap, best = score, c

        if best and best_overlap >= 0.45:
            return CapabilityResolution(
                task_intent=intent,
                decision=ResolutionDecision.REUSE_EXISTING,
                target_file=best.file_path,
                target_symbol=best.symbol_name if best.symbol_name != "(file)" else None,
                reasoning=f"Heuristic overlap {best_overlap:.2f} with existing '{best.symbol_name}'.",
                confidence=_clamp(0.4 + best_overlap * 0.4),
            )
        if best and best_overlap >= 0.25:
            return CapabilityResolution(
                task_intent=intent,
                decision=ResolutionDecision.EXTEND_EXISTING,
                target_file=best.file_path,
                target_symbol=best.symbol_name if best.symbol_name != "(file)" else None,
                reasoning=f"Partial heuristic overlap {best_overlap:.2f}; extend existing code.",
                confidence=_clamp(0.35 + best_overlap * 0.3),
            )
        return CapabilityResolution(
            task_intent=intent,
            decision=ResolutionDecision.CREATE_NEW,
            reasoning="No candidate meaningfully overlaps the intent; create new code.",
            confidence=0.5,
        )


# ── Backwards-compatible adapter ──────────────────────────────────────────────
# The previous name-based API delegated to fixed-name lookups. It now forwards to
# the capability resolver so old imports keep working with the correct behaviour.

class TaskLevelAnalyzer:
    """Deprecated shim. Use ``SemanticCapabilityResolver`` directly.

    Kept so existing imports do not break. ``analyze_task`` maps a task to an
    intent and returns a :class:`CapabilityResolution`.
    """

    def __init__(self, symbol_index: Any = None, workspace_path: Optional[str] = None, **kwargs):
        self._resolver = SemanticCapabilityResolver(
            rag_engine=kwargs.get("rag_engine"),
            repo_search_engine=kwargs.get("repo_search_engine"),
            llm=kwargs.get("llm"),
            workspace_path=workspace_path,
        )

    def analyze_task(
        self,
        intent: str = "",
        title: str = "",
        file_path: Optional[str] = None,
        **kwargs,
    ) -> CapabilityResolution:
        hints: Dict[str, Any] = {}
        if file_path:
            hints["file_path"] = file_path
        return self._resolver.resolve_task(intent=intent or title, title=title, hints=hints)

    def analyze_batch(self, tasks: List[Dict[str, Any]]) -> List[CapabilityResolution]:
        norm = [
            {
                "intent": t.get("intent") or t.get("reason") or t.get("title", ""),
                "title": t.get("title", ""),
                "hints": {"file_path": t.get("file_path")} if t.get("file_path") else None,
            }
            for t in tasks
        ]
        return self._resolver.resolve_batch(norm)


# ── Module-level helpers ──────────────────────────────────────────────────────

_STOPWORDS = {
    "this", "that", "with", "from", "into", "must", "should", "would", "when",
    "then", "will", "have", "does", "the", "and", "for", "add", "make", "code",
    "task", "file", "class", "method", "function", "component", "service",
}


def _is_stopword(word: str) -> bool:
    return word.lower() in _STOPWORDS


def _tokens(text: str) -> set:
    return {
        w.lower()
        for w in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text or "")
        if not _is_stopword(w)
    }


# Declaration patterns across the languages this repo generates (TS/JS/Java/Python/C#).
_DECL_PATTERNS = [
    ("class", re.compile(r"\b(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_]\w*)")),
    ("interface", re.compile(r"\b(?:export\s+)?interface\s+([A-Za-z_]\w*)")),
    ("function", re.compile(r"\b(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_]\w*)")),
    ("function", re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE)),
    # TS/Java/C# method: name( ... ) { — keyword false-positives filtered below.
    ("method", re.compile(
        r"^\s*(?:public|private|protected|internal|static|async|override|final|\s)*"
        r"([A-Za-z_]\w*)\s*\([^;{]*\)\s*(?::[^;{]+)?\s*\{",
        re.MULTILINE,
    )),
]

_METHOD_KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "function", "constructor"}


def _iter_declarations(content: str):
    """Yield (kind, name) declarations found in a code chunk."""
    seen = set()
    for kind, pat in _DECL_PATTERNS:
        for m in pat.finditer(content or ""):
            name = m.group(1)
            if kind == "method" and name.lower() in _METHOD_KEYWORDS:
                continue
            key = (kind, name)
            if key in seen:
                continue
            seen.add(key)
            yield kind, name


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in (text or "").splitlines()[:20])


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    # Strip code fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    blob = fenced.group(1) if fenced else raw
    start, end = blob.find("{"), blob.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(blob[start : end + 1])
    except Exception:
        return None


def _coerce_decision(value: Any) -> ResolutionDecision:
    try:
        return ResolutionDecision(str(value).strip().lower())
    except Exception:
        return ResolutionDecision.CREATE_NEW


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
