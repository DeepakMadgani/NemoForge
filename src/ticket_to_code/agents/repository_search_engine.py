"""
Phase 3A — Live Repository Search Engine

Filesystem-based, index-independent search across ALL repository file types.

Discovers files that are absent from SQLite, Neo4j, and semantic embeddings
because they were never crawled (e.g. .sh, .bat, .ps1, .json, .xml, .env).

Search modes:
  1. Literal   — exact string in file content  (grep-style, per-line)
  2. Filename  — fragment in repo-relative path (name + directory segments)
  3. Regex     — compiled regex match in file content

Rules:
  • Read-only — never writes to the repository
  • No dependency on SQLite, Neo4j, or embeddings
  • Deduplicates by file path per query
  • Writes a JSON trace: <workspace>/.aviator/repository_search.json

Author: Deepak Madgani
Date: June 2026
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

from ticket_to_code.agents.query_expansion import QueryExpansionEngine

logger = logging.getLogger(__name__)


# ── Supported file extensions ─────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".java", ".ts", ".tsx", ".js", ".jsx", ".py",
    ".html", ".scss", ".css", ".less",
    ".json", ".yml", ".yaml", ".properties", ".env",
    ".sh", ".bat", ".ps1", ".xml",
    ".md", ".sql", ".proto", ".graphql",
})

# ── Directories pruned from every walk — never contain indexable code ────────
from ticket_to_code.utils.file_filters import (
    STANDARD_SKIP_DIRS as _SKIP_DIRS,
    is_ignored_path,
    prune_walk_dirs,
)

# ── Safety limits ─────────────────────────────────────────────────────────────
_MAX_FILE_BYTES = 2 * 1024 * 1024   # 2 MB — skip files larger than this
_MIN_LITERAL_LEN = 4                 # Allows "26.2", version numbers

# ── Confidence by file type (content match) ───────────────────────────────────
_SCORE_MAP: dict[str, float] = {
    ".java":       0.85,
    ".ts":         0.85,
    ".tsx":        0.85,
    ".py":         0.85,
    ".js":         0.80,
    ".jsx":        0.80,
    ".sh":         0.82,
    ".bat":        0.82,
    ".ps1":        0.82,
    ".yml":        0.80,
    ".yaml":       0.80,
    ".properties": 0.80,
    ".env":        0.80,
    ".sql":        0.85,
    ".proto":      0.80,
    ".graphql":    0.80,
    ".xml":        0.78,
    ".json":       0.75,
    ".html":       0.75,
    ".scss":       0.75,
    ".css":        0.75,
    ".less":       0.75,
    ".md":         0.55,
}

# ── Real ripgrep binary fast path ─────────────────────────────────────────────
# When the `rg` binary is on PATH it is 10-100x faster than the pure-Python
# grep fallback.  Availability is checked once; every call stays read-only and
# falls back to the Python walker on any rg failure (missing binary, timeout).
import shutil as _shutil
import subprocess as _subprocess
import concurrent.futures as _futures
import time as _time

_RG_TIMEOUT_SECONDS = 30      # whole-search timeout for the rg subprocess
_REGEX_FILE_TIMEOUT_SECONDS = 2.0   # per-file regex eval timeout (catastrophic backtracking guard)
_REGEX_TOTAL_DEADLINE_SECONDS = 10.0  # whole-regex-search deadline

# Sentinel returned by _regex_search_with_timeout when a file's regex eval
# exceeded the per-file timeout (catastrophic backtracking suspected).
_REGEX_PATHOLOGICAL = object()


def _rg_binary() -> Optional[str]:
    """Path to the ripgrep binary, or None when unavailable (checked once)."""
    global _RG_PATH_CACHE
    if _RG_PATH_CACHE is _RG_UNSET:
        found = _shutil.which("rg")
        if not found:
            candidates = [
                os.path.expanduser(r"~\AppData\Local\Programs\Antigravity IDE\resources\app\node_modules\@vscode\ripgrep\bin\rg.exe"),
                os.path.expanduser(r"~\AppData\Local\Programs\Microsoft VS Code\resources\app\node_modules\@vscode\ripgrep\bin\rg.exe"),
                r"C:\Program Files\Git\usr\bin\rg.exe",
            ]
            for c in candidates:
                if os.path.isfile(c):
                    found = c
                    break
        _RG_PATH_CACHE = found
    return _RG_PATH_CACHE


_RG_UNSET = object()
_RG_PATH_CACHE: object = _RG_UNSET

# Comment-only lines are skipped when choosing the representative match line,
# so a trivial comment hit on line 1 cannot hide the real match on line 400.
_COMMENT_LINE_RE = re.compile(r'^\s*(?://|/\*|\*|\#|<!--|\'|[;\"])')

# Filename (path) match confidence — always strong signal
_FILENAME_SCORE = 0.88

# ── Hit classification ────────────────────────────────────────────────────────
# Path shapes that mark a hit as test/mock scaffolding rather than product code
_TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|__tests__|spec|e2e|fixtures?|mocks?)(/|$)"
    r"|\.(spec|test|stories|mock|fixture)\.[a-z]+$"
    r"|(^|/)(test_[\w.]+\.py|[\w.]+_test\.py)$"
    r"|(^|/)(Test\w+|\w+Tests?|\w+IT)\.java$",
    re.IGNORECASE,
)


def _classify_hit(rel_path: str, line_text: str, match_offset: int = -1) -> str:
    """Classify a raw match: code | comment | string | test_file | doc.

    Path shape decides test_file/doc; the matched line decides comment vs
    string vs code.  ``match_offset`` (position of the match inside the
    line, -1 when unknown) lets string detection check quote parity before
    the match instead of mis-tagging lines that merely start with a quote.
    """
    lower_path = rel_path.lower()
    if lower_path.endswith((".md", ".rst", ".adoc")):
        return "doc"
    if _TEST_PATH_RE.search(lower_path):
        return "test_file"
    stripped = (line_text or "").lstrip()
    if not stripped:
        return "code"
    if stripped.startswith(("//", "/*", "*", "#", "<!--")):
        return "comment"
    if match_offset >= 0:
        prefix = line_text[:match_offset]
        if prefix.count('"') % 2 == 1 or prefix.count("'") % 2 == 1:
            # Inside a string literal — but a string ARGUMENT to a call
            # (`init('paymentGateway')`, even mid-line after `function f() {`)
            # is executable usage, not a data hit: an unclosed `name(` in the
            # prefix means the match sits inside a call's argument list.
            if re.search(r"[\w.$]+\s*\([^)]*$", prefix):
                return "code"
            return "string"
    return "code"


def _bounded_levenshtein(a: str, b: str, max_dist: int) -> int:
    """Levenshtein distance with early exit once it exceeds max_dist."""
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            val = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            cur.append(val)
            row_min = min(row_min, val)
        if row_min > max_dist:
            return max_dist + 1
        prev = cur
    return prev[-1]


@dataclass
class SearchResult:
    """One match from the live repository search layer."""

    query: str          # the exact term that matched (may be an expanded form)
    query_type: str     # "literal" | "filename" | "regex"
    file_path: str      # repo-relative POSIX path
    line_number: int    # 1-based line in file; 0 for filename matches
    matched_text: str   # the matching line or path
    confidence: float   # 0.0 – 1.0
    original_literal: str = ""  # the hypothesis literal that triggered expansion
    original_query: str = ""    # original input query before fallback or normalization
    resolved_query: str = ""    # normalized or stripped query that actually matched
    search_mode: str = "exact"  # "exact" | "token_stem_fallback" | "ui_container_stripped" | "regex_error"
    error: Optional[str] = None # explicit error details (e.g. invalid regex syntax)
    is_ambiguous: bool = False  # true if multiple candidates matched during fallback
    classification: str = "code"  # code | comment | string | test_file | doc


class RepositorySearchEngine:
    """
    Phase 3A: Live, index-independent search across all repository file types.

    Complements indexed knowledge (SQLite / Neo4j / pgvector) by discovering
    files that exist in the repository but were never crawled:

      • Shell scripts        (.sh, .bat, .ps1)
      • JSON data            (.json)
      • XML config           (.xml)
      • Environment files    (.env)
      • All YAML             (.yml, .yaml)
      • Any other supported extension

    Usage::

        engine = RepositorySearchEngine(workspace_path)
        hits = engine.search_literal("26.2")
        hits += engine.search_filename("run-job")
        engine.flush_trace()   # writes .aviator/repository_search.json
    """

    def __init__(
        self,
        workspace_path: Path,
        trace_output: Optional[Path] = None,
        query_expansion_trace: Optional[Path] = None,
    ) -> None:
        self.workspace = Path(workspace_path)
        self.trace_output = trace_output or (
            self.workspace / ".aviator" / "repository_search.json"
        )
        self.query_expansion_trace = query_expansion_trace or (
            self.workspace / ".aviator" / "query_expansion.json"
        )
        self._trace_records: List[dict] = []
        self._query_expansion_records: List[dict] = []
        self.expansion_engine = QueryExpansionEngine()
        # Full provenance map: original_literal → set of expanded forms
        # Populated as search_literal() calls are made; consumed by ranking.
        self._expansion_map: dict = {}   # Dict[str, Set[str]]
        # ── Cached file index (Performance optimization, 2026-09-16) ─────
        # Built lazily on first _walk() call. Subsequent calls iterate the
        # cached list instead of re-walking os.walk() on every search.
        # Call invalidate_index() if repository contents change mid-run.
        self._file_index: Optional[List[Tuple[Path, str]]] = None
        self._file_index_built: bool = False

    # =========================================================================
    # PUBLIC SEARCH API
    # =========================================================================

    def search_literal(
        self,
        literal: str,
        *,
        extensions: Optional[Set[str]] = None,
        case_sensitive: bool = False,
        max_results: int = 30,
    ) -> List[SearchResult]:
        """
        Search every repository file for an exact string literal,
        automatically applying query expansion for versions and naming conventions.

        Args:
            literal:        String to search for in file content.
            extensions:     Restrict to these extensions; None → all supported.
            case_sensitive: Default False (version strings vary in casing).
            max_results:    Cap per query to avoid flooding evidence list.

        Returns:
            One SearchResult per matching file (first match per file).
        """
        stripped = literal.strip()
        if len(stripped) < _MIN_LITERAL_LEN:
            logger.debug(
                f"[RepoSearch] Skipping '{literal}' — shorter than {_MIN_LITERAL_LEN} chars"
            )
            return []

        expanded_terms = self.expansion_engine.expand(stripped)

        # Persist provenance: original → all expansions
        self._expansion_map[stripped] = expanded_terms

        self._query_expansion_records.append({
            "original": stripped,
            "expanded": sorted(expanded_terms),
        })

        exts   = extensions or SUPPORTED_EXTENSIONS
        results: List[SearchResult] = []
        seen:    Set[str] = set()

        # Dynamic Specificity Threshold (Agentic Search Abort)
        # We calculate entropy to avoid falsely aborting on legitimate broad changes like version bumps.
        if re.search(r'\d+\.\d+', stripped):
            specificity_threshold = max(100, max_results)  # Version numbers are high entropy, allow broad sweep
        elif len(stripped) > 15:
            specificity_threshold = max(50, max_results)   # Long phrases are highly specific
        else:
            specificity_threshold = max_results            # Generic short words get strict abort

        # ── Fast path: real ripgrep binary when available ─────────────────
        rg_path = _rg_binary()
        if rg_path:
            raw_hits = self._rg_first_match_per_file(expanded_terms, exts, case_sensitive)
        else:
            raw_hits = None

        if raw_hits is not None:
            # rg path: raw_hits is [(term, rel_path, line_no, line_text), ...]
            for term, rel_path, line_no, line_text in raw_hits:
                if len(results) > specificity_threshold:
                    break
                if rel_path in seen:
                    continue
                seen.add(rel_path)
                _needle = term if case_sensitive else term.lower()
                _hay = line_text if case_sensitive else line_text.lower()
                results.append(SearchResult(
                    query=term,
                    query_type="literal",
                    file_path=rel_path,
                    line_number=line_no,
                    matched_text=line_text[:200],
                    confidence=_SCORE_MAP.get(Path(rel_path).suffix.lower(), 0.75),
                    original_literal=stripped,
                    original_query=stripped,
                    resolved_query=term,
                    search_mode="exact",
                    classification=_classify_hit(rel_path, line_text, _hay.find(_needle)),
                ))
                self._append_trace(results[-1])
                self._query_expansion_records.append({
                    "query": term,
                    "original_literal": stripped,
                    "matched_file": rel_path,
                    "source": "query_expansion",
                })
        else:
            # Python walker fallback (no rg binary, or rg invocation failed)
            for term in expanded_terms:
                target = term if case_sensitive else term.lower()
                for abs_path, rel_path in self._walk(exts):
                    if len(results) > specificity_threshold:
                        break
                    if rel_path in seen:
                        continue
                    line_no, line_text = self._grep_first(abs_path, target, case_sensitive)
                    if line_no == 0:
                        continue
                    seen.add(rel_path)
                    _hay = line_text if case_sensitive else line_text.lower()
                    results.append(SearchResult(
                        query=term,
                        query_type="literal",
                        file_path=rel_path,
                        line_number=line_no,
                        matched_text=line_text[:200],
                        confidence=_SCORE_MAP.get(abs_path.suffix.lower(), 0.75),
                        original_literal=stripped,   # provenance tag
                        original_query=stripped,
                        resolved_query=term,
                        search_mode="exact",
                        classification=_classify_hit(rel_path, line_text, _hay.find(target)),
                    ))
                    self._append_trace(results[-1])
                    self._query_expansion_records.append({
                        "query": term,
                        "original_literal": stripped,
                        "matched_file": rel_path,
                        "source": "query_expansion",
                    })

        if len(results) > specificity_threshold:
            # Rank-and-keep instead of discarding everything. Returning []
            # silently erased legitimate evidence — keep the highest-confidence
            # hits and let the downstream EvidenceRankingEngine de-noise.
            results.sort(key=lambda r: r.confidence, reverse=True)
            kept = results[:max_results]
            logger.warning(
                f"[RepoSearch] ⚖️  CAP: Literal '{stripped}' is broad "
                f"(exceeded {specificity_threshold} matches). "
                f"Keeping top {len(kept)} by confidence; ranking engine will de-noise."
            )
            return kept

        if results:
            logger.debug(f"[RepoSearch] literal '{literal}' → {len(results)} file(s)")
            return results

        # ── Tier 2: UI Container Term Stripping (fallback only on 0 results) ──
        from ticket_to_code.agents.query_expansion import strip_ui_container_term
        stripped_container = strip_ui_container_term(stripped)
        if stripped_container and stripped_container.lower() != stripped.lower():
            logger.info(
                f"[RepoSearch] 🔄 UI container fallback: '{stripped}' → '{stripped_container}'"
            )
            fallback_hits = self.search_literal(
                stripped_container,
                extensions=extensions,
                case_sensitive=case_sensitive,
                max_results=max_results,
            )
            for hit in fallback_hits:
                hit.original_query = stripped
                hit.resolved_query = stripped_container
                hit.search_mode = "ui_container_stripped"
            return fallback_hits

        logger.debug(f"[RepoSearch] literal '{literal}' → 0 file(s)")
        return []

    def search_filename(
        self,
        name_fragment: str,
        *,
        extensions: Optional[Set[str]] = None,
        max_results: int = 20,
    ) -> List[SearchResult]:
        """
        Find files whose repo-relative POSIX path contains `name_fragment`.

        Exact substring match is the authoritative first attempt.
        If zero exact matches are found, falls back to token / singular-plural
        normalization. If multiple candidates remain, returns all as ambiguous
        rather than guessing.

        Args:
            name_fragment:  Substring to match anywhere in the path.
            extensions:     Restrict to these extensions; None → all supported.
            max_results:    Cap.

        Returns:
            One SearchResult per matching file (line_number=0).
        """
        if not name_fragment.strip():
            return []
        fragment_lower = name_fragment.strip().lower()
        exts = extensions or SUPPORTED_EXTENSIONS
        results: List[SearchResult] = []

        # ── Tier 1: Authoritative Exact Substring Matching ───────────────
        # A fragment matching at a token boundary inside the filename STEM is
        # the precise signal; matches that only continue mid-word ("member"
        # inside /memberships/) or live solely in a directory segment are kept
        # only when no stem-boundary match exists, and are demoted so the
        # ranking engine can order them below boundary-quality evidence.
        def _stem_boundary_hit(rel_path: str) -> bool:
            stem = Path(rel_path).stem.lower()
            start = 0
            while True:
                i = stem.find(fragment_lower, start)
                if i < 0:
                    return False
                before_ok = i == 0 or stem[i - 1] in "-_."
                j = i + len(fragment_lower)
                # Allow a trailing plural ('s' + boundary) so "member" still
                # matches "members.component"; reject "member" in "memberships".
                after_ok = j == len(stem) or stem[j] in "-_." or (
                    stem[j] == "s" and (j + 1 == len(stem) or stem[j + 1] in "-_.")
                )
                if before_ok and after_ok:
                    return True
                start = i + 1

        exact_hits: List[str] = []
        boundary_hits: List[str] = []
        for _, rel_path in self._walk(exts):
            if fragment_lower in rel_path.lower():
                exact_hits.append(rel_path)
                if len(boundary_hits) < max_results and _stem_boundary_hit(rel_path):
                    boundary_hits.append(rel_path)

        if exact_hits:
            # Boundary matches are the real answers; mid-word / dir-segment
            # matches are surfaced (never silently dropped) but demoted.
            if boundary_hits:
                chosen = [(p, _FILENAME_SCORE) for p in boundary_hits[:max_results]]
            else:
                chosen = [(p, _FILENAME_SCORE * 0.75) for p in exact_hits[:max_results]]
            for rel_path, score in chosen:
                r = SearchResult(
                    query=name_fragment,
                    query_type="filename",
                    file_path=rel_path,
                    line_number=0,
                    matched_text=rel_path,
                    confidence=score,
                    original_query=name_fragment,
                    resolved_query=name_fragment,
                    search_mode="exact",
                )
                results.append(r)
                self._append_trace(r)
            logger.debug(
                f"[RepoSearch] filename '{name_fragment}' (exact, "
                f"{'stem-boundary' if boundary_hits else 'substring-only'}) → {len(chosen)} file(s)"
            )
            return results

        # ── Tier 2: PascalCase Decomposition & Token/Inflection Fallback ───────
        # Reached ONLY when Tier 1 produced 0 results.
        import re as _re
        from ticket_to_code.agents.query_expansion import _UI_CONTAINER_TERMS

        def _decompose_identifier(s: str) -> str:
            # Deterministic word boundary decomposition (e.g. "AddMembersModal" -> "Add Members Modal")
            s1 = _re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', s)
            return _re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s1)

        decomposed = _decompose_identifier(name_fragment.strip())
        raw_tokens = [t for t in _re.split(r'[-_./\s]+', decomposed.lower()) if t]
        if not raw_tokens:
            return []

        def _norm_singular(t: str) -> str:
            # Deterministic singular normalization: strip trailing 's' if len > 3 and not ending in 'ss'
            if len(t) > 3 and t.endswith('s') and not t.endswith('ss'):
                return t[:-1]
            return t

        # Check if UI container terms should be stripped (e.g. 'modal', 'dialog', 'page')
        has_container = any(t in _UI_CONTAINER_TERMS for t in raw_tokens)
        tokens_no_container = [t for t in raw_tokens if t not in _UI_CONTAINER_TERMS]

        token_sets_to_try = []
        if has_container and tokens_no_container:
            token_sets_to_try.append((tokens_no_container, "ui_container_stripped"))
        token_sets_to_try.append((raw_tokens, "token_stem_fallback"))

        candidate_paths: List[str] = []
        used_mode = "token_stem_fallback"
        effective_tokens = raw_tokens

        for tok_list, mode in token_sets_to_try:
            norm_query_tokens = [_norm_singular(t) for t in tok_list]
            matches: List[str] = []
            for _, rel_path in self._walk(exts):
                p_tokens = [t for t in _re.split(r'[-_./\s\\]+', rel_path.lower()) if t]
                norm_p_tokens = set(_norm_singular(t) for t in p_tokens)
                if all(qt in norm_p_tokens for qt in norm_query_tokens):
                    matches.append(rel_path)
            if matches:
                candidate_paths = matches
                used_mode = mode
                effective_tokens = tok_list
                break

        if not candidate_paths:
            # ── Tier 3: bounded fuzzy match (typo tolerance) ─────────────
            # Only when exact AND token matching produced nothing: compare
            # the fragment against each filename stem with a cheap bounded
            # Levenshtein distance (≤2 for long fragments, ≤1 for short).
            # Multiple matches are surfaced as ambiguous, never auto-picked.
            max_dist = 2 if len(fragment_lower) >= 8 else 1
            fuzzy_hits: List[str] = []
            for _, rel_path in self._walk(exts):
                stem = Path(rel_path).stem.lower()
                if abs(len(stem) - len(fragment_lower)) > max_dist:
                    continue
                if _bounded_levenshtein(fragment_lower, stem, max_dist) <= max_dist:
                    fuzzy_hits.append(rel_path)
                    if len(fuzzy_hits) > 10:
                        break
            if fuzzy_hits:
                is_ambig = len(fuzzy_hits) > 1
                confidence = _FILENAME_SCORE * 0.5 if is_ambig else _FILENAME_SCORE * 0.6
                for c_path in fuzzy_hits[:max_results]:
                    results.append(SearchResult(
                        query=name_fragment,
                        query_type="filename",
                        file_path=c_path,
                        line_number=0,
                        matched_text=c_path,
                        confidence=confidence,
                        original_query=name_fragment,
                        resolved_query=fuzzy_hits[0] if not is_ambig else name_fragment,
                        search_mode="fuzzy_typo",
                        is_ambiguous=is_ambig,
                    ))
                logger.info(
                    f"[RepoSearch] 🔤 Fuzzy filename match for '{name_fragment}' → "
                    f"{len(fuzzy_hits)} candidate(s) (maxdist={max_dist})"
                )
                return results
            return []

        # Ambiguity check: if multiple candidates remain, surface all as ambiguous (never guess)
        is_ambiguous = len(candidate_paths) > 1
        confidence = _FILENAME_SCORE * 0.7 if is_ambiguous else _FILENAME_SCORE * 0.9

        resolved_q = " ".join([_norm_singular(t) for t in effective_tokens])
        for c_path in candidate_paths[:max_results]:
            r = SearchResult(
                query=name_fragment,
                query_type="filename",
                file_path=c_path,
                line_number=0,
                matched_text=c_path,
                confidence=confidence,
                original_query=name_fragment,
                resolved_query=resolved_q,
                search_mode=used_mode,
                is_ambiguous=is_ambiguous,
            )
            results.append(r)
            self._append_trace(r)

        if is_ambiguous:
            logger.info(
                f"[RepoSearch] ⚠️ Ambiguous fallback for filename '{name_fragment}': "
                f"matched {len(candidate_paths)} files (no automatic selection)"
            )
        else:
            logger.info(
                f"[RepoSearch] 🔄 Token/singular fallback for filename '{name_fragment}' → '{candidate_paths[0]}'"
            )

        return results

    def search_regex(
        self,
        pattern: str,
        *,
        extensions: Optional[Set[str]] = None,
        flags: int = re.IGNORECASE,
        max_results: int = 20,
    ) -> List[SearchResult]:
        """
        Search repository files with a compiled regex.

        If the regex is syntactically invalid, returns an explicit error SearchResult
        with the compilation error message rather than silently returning 0 results.

        Args:
            pattern:    Regex string.
            extensions: Restrict; None → all supported.
            flags:      re module flags.
            max_results: Cap.

        Returns:
            One SearchResult per matching file (first match).
        """
        try:
            compiled = re.compile(pattern, flags)
        except re.error as exc:
            logger.warning(f"[RepoSearch] Invalid regex '{pattern}': {exc}")
            err_res = SearchResult(
                query=pattern,
                query_type="regex",
                file_path="",
                line_number=0,
                matched_text=f"Invalid regex: {exc}",
                confidence=0.0,
                original_query=pattern,
                resolved_query=pattern,
                search_mode="regex_error",
                error=f"Invalid regex: {exc}",
            )
            self._append_trace(err_res)
            return [err_res]

        exts = extensions or SUPPORTED_EXTENSIONS
        results: List[SearchResult] = []
        seen: Set[str] = set()
        deadline = _time.monotonic() + _REGEX_TOTAL_DEADLINE_SECONDS
        pattern_timed_out = False

        for abs_path, rel_path in self._walk(exts):
            if len(results) >= max_results:
                break
            if rel_path in seen:
                continue
            if _time.monotonic() > deadline:
                logger.warning(
                    f"[RepoSearch] ⏱️ Regex search hit {int(_REGEX_TOTAL_DEADLINE_SECONDS)}s deadline "
                    f"({len(results)} result(s) kept)"
                )
                break
            try:
                if abs_path.stat().st_size > _MAX_FILE_BYTES:
                    continue
                text = abs_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            m = self._regex_search_with_timeout(compiled, text)
            if m is _REGEX_PATHOLOGICAL:
                pattern_timed_out = True
                continue
            if not m:
                continue
            line_no = text[: m.start()].count("\n") + 1
            lines = text.splitlines()
            line_text = lines[line_no - 1].strip() if line_no <= len(lines) else m.group(0)
            _ls = text.rfind("\n", 0, m.start()) + 1
            _le = text.find("\n", _ls)
            _line_raw = text[_ls:_le if _le != -1 else len(text)]
            score = _SCORE_MAP.get(abs_path.suffix.lower(), 0.75)
            r = SearchResult(
                query=pattern,
                query_type="regex",
                file_path=rel_path,
                line_number=line_no,
                matched_text=line_text[:200],
                confidence=score,
                original_query=pattern,
                resolved_query=pattern,
                search_mode="exact",
                classification=_classify_hit(rel_path, _line_raw, m.start() - _ls),
            )
            results.append(r)
            seen.add(rel_path)
            self._append_trace(r)

        if pattern_timed_out:
            logger.warning(
                f"[RepoSearch] ⚠️ Regex '{pattern[:60]}' triggered backtracking timeouts on "
                f"some files — results are partial; consider a simpler pattern"
            )

        logger.debug(f"[RepoSearch] regex '{pattern}' → {len(results)} file(s)")
        return results

    @staticmethod
    def _regex_search_with_timeout(compiled: "re.Pattern", text: str):
        """Evaluate `compiled.search(text)` under a hard per-file timeout.

        LLM-authored regexes can backtrack catastrophically; an eval that
        exceeds _REGEX_FILE_TIMEOUT_SECONDS is abandoned (the worker thread
        keeps running but its result is discarded) and the file is skipped.
        Returns the match object, None for no-match, or _REGEX_PATHOLOGICAL.
        """
        executor = _futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(compiled.search, text)
            try:
                return future.result(timeout=_REGEX_FILE_TIMEOUT_SECONDS)
            except _futures.TimeoutError:
                return _REGEX_PATHOLOGICAL
        finally:
            # shutdown(wait=False) lets the abandoned worker die in background
            executor.shutdown(wait=False)


    @property
    def expansion_map(self) -> dict:
        """
        Return the accumulated expansion provenance map.

        Dict[str, Set[str]]: maps each original hypothesis literal to the full
        set of expanded forms that were used for search.  Call after all
        search_literal() calls have been made (before flush_trace()).

        This is passed to EvidenceRankingEngine so it can credit files
        discovered via expansion with the same literal-match boost as
        files discovered via the original term.
        """
        return dict(self._expansion_map)

    def flush_trace(self) -> None:
        """
        Write accumulated trace records to `trace_output` (JSON array).
        Also writes query expansion trace.

        Overwrites the file with the current batch so each run reflects only
        the most recent search, preventing unbounded historical accumulation.
        """
        if self._trace_records:
            self.trace_output.parent.mkdir(parents=True, exist_ok=True)
            self.trace_output.write_text(
                json.dumps(self._trace_records, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(
                f"[RepoSearch] trace flushed → {self.trace_output} "
                f"({len(self._trace_records)} records)"
            )
            self._trace_records.clear()

        if self._query_expansion_records:
            self.query_expansion_trace.parent.mkdir(parents=True, exist_ok=True)
            self.query_expansion_trace.write_text(
                json.dumps(self._query_expansion_records, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(
                f"[RepoSearch] query expansion trace flushed → {self.query_expansion_trace} "
                f"({len(self._query_expansion_records)} records)"
            )
            self._query_expansion_records.clear()

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _ensure_file_index(self) -> None:
        """Build the file-path index once on first use.

        Walks os.walk() exactly once per engine lifetime and caches the
        (abs_path, rel_path, extension) tuples.  Subsequent _walk() calls
        iterate the cached list — no filesystem traversal.

        Same search semantics for a stable repository.  Call
        invalidate_index() when repository contents change or when
        required by existing behavior (e.g. after code generation).
        """
        if self._file_index_built:
            return
        index: List[Tuple[Path, str]] = []
        for root, dirs, files in os.walk(str(self.workspace)):
            # Prune in-place so os.walk skips them entirely
            prune_walk_dirs(dirs, root, self.workspace)
            root_path = Path(root)
            for fname in sorted(files):           # sorted → deterministic order
                fpath = root_path / fname
                if is_ignored_path(fpath, workspace_path=self.workspace):
                    continue
                try:
                    rel = str(fpath.relative_to(self.workspace)).replace("\\", "/")
                except ValueError:
                    continue
                index.append((fpath, rel))
        self._file_index = index
        self._file_index_built = True
        logger.debug(f"[RepoSearch] File index built: {len(index)} files")

    def invalidate_index(self) -> None:
        """Invalidate the cached file index.

        Call this when repository contents change during a ticket run
        (e.g. after code generation creates new files) so that subsequent
        searches discover newly created or deleted files.
        """
        self._file_index = None
        self._file_index_built = False
        logger.debug("[RepoSearch] File index invalidated")

    def _walk(self, extensions: Set[str]):
        """Yield (abs_path, rel_path) for all matching files under workspace.

        Uses the cached file index (built lazily on first call) instead of
        re-walking the filesystem on every search query.
        """
        self._ensure_file_index()
        for abs_path, rel_path in self._file_index:
            if self._scope_prefix and not rel_path.lower().startswith(self._scope_prefix + "/"):
                continue
            if is_ignored_path(abs_path, workspace_path=self.workspace):
                continue
            if abs_path.suffix.lower() in extensions:
                yield abs_path, rel_path

    def _rg_first_match_per_file(
        self,
        terms: List[str],
        extensions: Set[str],
        case_sensitive: bool,
    ) -> Optional[List[Tuple[str, str, int, str]]]:
        """Run the real ripgrep binary for the first match per file across terms.

        Returns a list of (term, rel_path, line_no, line_text) tuples, or None
        when rg is unusable (missing binary, invocation error, timeout) so the
        caller falls back to the Python walker.  `-m 1` gives the first match
        per file, matching the legacy per-file semantics.
        """
        rg_path = _rg_binary()
        if not rg_path:
            return None
        terms = sorted(set(terms))   # expand() yields a set — deterministic order
        cmd = [rg_path, "--no-heading", "--line-number", "--max-filesize", "2M", "-m", "1"]
        # NOTE: the sub-workspace scope must be folded into the EXTENSION
        # globs below — multiple -g globs are OR'd in rg, so a standalone
        # scope glob would be defeated by the per-extension globs.
        if not case_sensitive:
            cmd.append("--ignore-case")
        cmd.append("--fixed-strings")
        for ext in sorted(extensions):
            if self._scope_prefix:
                cmd.extend(["-g", f"{self._scope_prefix}/**/*{ext}"])
            else:
                cmd.extend(["-g", f"*{ext}"])
        for term in terms:
            cmd.extend(["-e", term])
        cmd.append(".")
        try:
            proc = _subprocess.run(
                cmd,
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_RG_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning(f"[RepoSearch] rg fast path unavailable ({exc.__class__.__name__}) — using Python walker")
            return None
        if proc.returncode not in (0, 1):   # 1 = no matches, also fine
            logger.debug(f"[RepoSearch] rg exited {proc.returncode}: {proc.stderr[:200]}")
            return None
        hits: List[Tuple[str, str, int, str]] = []
        for line in (proc.stdout or "").splitlines():
            # rg --no-heading -n format: path:line:text
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            rel_path, line_no_raw, text = parts[0], parts[1], parts[2]
            if not line_no_raw.isdigit():
                continue
            matched_term = next(
                (t for t in terms
                 if (t if case_sensitive else t.lower())
                 in (text if case_sensitive else text.lower())),
                terms[0],
            )
            # Normalize to the same clean relative form the Python walker
            # emits: rg prefixes paths with "./" because "." is the search
            # root, and string-equality consumers (visited sets, manifest
            # keys, SQLite paths) compare raw path strings across providers.
            rel_path = rel_path.replace("\\", "/")
            if rel_path.startswith("./"):
                rel_path = rel_path[2:]
            if is_ignored_path(rel_path, workspace_path=self.workspace):
                continue
            hits.append((matched_term, rel_path, int(line_no_raw), text.strip()))
        return hits

    # ── Sub-workspace scope (polyglot monorepo routing) ──────────────────
    _scope_prefix: Optional[str] = None

    def set_scope(self, prefix: Optional[str]) -> None:
        """Restrict every walk/grep to a repo-relative sub-project prefix
        (e.g. 'xchange-ui').  None clears the scope."""
        cleaned = (prefix or "").replace("\\", "/").strip("/").lower()
        self._scope_prefix = cleaned or None
        self.invalidate_index()

    def search_cooccurrence(
        self,
        terms: List[str],
        *,
        extensions: Optional[Set[str]] = None,
        max_results: int = 20,
    ) -> List[SearchResult]:
        """Files containing ALL of the given terms (case-insensitive AND).

        The highest-precision content search: one term alone can match
        hundreds of files; two or three specific terms together usually
        match only the files that genuinely implement the behavior.
        Results are ranked by total occurrence count of all terms, and each
        carries the anchor line of the RAREST term (the best single-line
        provenance available).  Needs >= 2 distinct terms of length >= 3.
        """
        cleaned: List[str] = []
        for t in terms or []:
            t = (t or "").strip()
            if len(t) >= 3 and t.lower() not in cleaned:
                cleaned.append(t.lower())
        if len(cleaned) < 2:
            return []
        cleaned = cleaned[:6]
        exts = extensions or SUPPORTED_EXTENSIONS
        hits: List[Tuple[str, str, int, str, int]] = []
        for abs_path, rel_path in self._walk(exts):
            try:
                if abs_path.stat().st_size > _MAX_FILE_BYTES:
                    continue
                content = abs_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            content_lower = content.lower()
            if not all(t in content_lower for t in cleaned):
                continue
            total = sum(content_lower.count(t) for t in cleaned)
            rarest = min(cleaned, key=lambda t: content_lower.count(t))
            line_no = 0
            anchor = ""
            for i, line in enumerate(content.splitlines(), start=1):
                if rarest in line.lower():
                    line_no, anchor = i, line.strip()
                    break
            hits.append((rel_path, anchor, line_no, rarest, total))
        hits.sort(key=lambda h: -h[4])
        results: List[SearchResult] = []
        query_str = " AND ".join(cleaned)
        for rel_path, anchor, line_no, _rarest, _total in hits[:max_results]:
            r = SearchResult(
                query=query_str,
                query_type="literal",
                file_path=rel_path,
                line_number=line_no,
                matched_text=(anchor or rel_path)[:200],
                confidence=min(_SCORE_MAP.get(Path(rel_path).suffix.lower(), 0.75) + 0.05, 1.0),
                original_query=" AND ".join(terms),
                resolved_query=query_str,
                search_mode="cooccurrence",
                classification=_classify_hit(rel_path, anchor),
            )
            results.append(r)
            self._append_trace(r)
        logger.debug(f"[RepoSearch] cooccurrence {cleaned} -> {len(results)} file(s)")
        return results

    def _grep_first(
        self,
        abs_path: Path,
        target: str,        # already lowercased if case_sensitive=False
        case_sensitive: bool,
    ) -> tuple[int, str]:
        """
        Return (line_number, line_text) of the FIRST line containing `target`.

        Comment-only lines are deprioritised: the first *code* match wins when
        one exists, so a trivial comment hit near the top of the file cannot
        hide the real match deeper down.  Returns (0, '') if not found or file
        unreadable.
        """
        try:
            if abs_path.stat().st_size > _MAX_FILE_BYTES:
                return 0, ""
            text = abs_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return 0, ""
        first_any: tuple[int, str] = (0, "")
        for i, line in enumerate(text.splitlines(), start=1):
            haystack = line if case_sensitive else line.lower()
            if target in haystack:
                if not first_any[0]:
                    first_any = (i, line.strip())
                if not _COMMENT_LINE_RE.match(line):
                    return i, line.strip()
        return first_any

    def _append_trace(self, r: SearchResult) -> None:
        """Buffer one trace record (written to disk on flush_trace())."""
        self._trace_records.append(asdict(r))
