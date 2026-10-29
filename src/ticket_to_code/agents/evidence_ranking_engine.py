"""
Phase 3B: Evidence Ranking Engine

Deterministic post-collection ranking that rescores EvidenceItems so the
files most likely to need modification appear at the top of every list the
Grounded Understanding node consumes.

Ranking formula (all additive, final score clamped to [0.10, 1.0]):

  base_score             = max(item.relevance_score) across all items for file
  + BOOST_LITERAL        = +0.35  exact hypothesis literal found in file content
  + BOOST_EXPANDED_LIT   = +0.30  file matched via query-expanded literal
                                   (e.g. 260200 is an expansion of 26.2)
  + BOOST_SYMBOL         = +0.20  SQLite symbol match (sqlite_symbol source)
  + BOOST_FTS            = +0.08  SQLite FTS content match (sqlite_fts source)
  + BOOST_MULTI_SOURCE   = +0.06 per extra unique source beyond first (max ×3)
  + BOOST_CHAIN          = +0.12  dependency chain evidence (neo4j/ts/java/css)
  + BOOST_LOCALIZED      = +0.15  file is already a localized plan task
  - PENALTY_FNAME_ONLY   = -0.25  ALL repository_search items are filename-only
  - PENALTY_INFRA        = -0.20  Kubernetes/Helm infrastructure file (hpa, pdb…)
  - PENALTY_WEAK         = -0.10  single source, not localized, no literal/symbol
                                  (SUPPRESSED when expansion match present)
  - PENALTY_BROAD_REPO   = -0.12  only repository_search, no content match
                                  (SUPPRESSED when expansion match present)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LITERAL_PREFIX    = "[literal:"
_EXPANDED_PREFIX   = "[expanded:"
_FILENAME_PREFIX   = "[filename:"

# Source families
_CHAIN_SOURCES = frozenset({"neo4j", "ts_chain", "java_chain", "css_chain", "layout_chain"})
_INDEX_SOURCES = frozenset({
    "sqlite_fts", "sqlite_symbol", "semantic",
    "neo4j", "ts_chain", "java_chain", "css_chain", "layout_chain",
})

# Kubernetes / Helm infra file stems that should never dominate ranking
_INFRA_STEMS = frozenset({
    "hpa", "poddisruptionbudget", "configmap", "ingress",
    "networkpolicy", "pvc", "pdb", "servicemonitor",
    "podmonitor", "clusterrole", "clusterrolebinding",
    "serviceaccount", "secret", "certificate",
    "chart",
    "virtualservice",
    "destinationrule",
})

# Architecture-aware file-role classification
# Deployment / automation scripts
_DEPLOY_EXTENSIONS = frozenset({".sh", ".bat", ".ps1"})
# Configuration/infrastructure files
_CONFIG_EXTENSIONS = frozenset({".yml", ".yaml", ".properties", ".env", ".xml", ".toml", ".conf", ".ini"})
# Application root patterns — matched against the POSIX rel-path
_ROOT_COMPONENT_PATTERNS = [
    r"(?:^|/)app\.component\.[tj]s$",       # Angular root component
    r"(?:^|/)main\.[tj]sx?$",               # React / Angular entry point
    r"(?:^|/)app\.[tj]sx?$",               # Generic app entry
    r"(?:^|/)application\.(java|kt)$",      # Spring Boot Application
    r"(?:^|/)startup\.(cs|vb)$",           # .NET startup
    r"(?:^|/)program\.(cs|vb)$",           # .NET Program.cs
    r"(?:^|/)index\.[tj]sx?$",             # SPA index entry
]
_ROOT_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in _ROOT_COMPONENT_PATTERNS]

# Scoring weights
_BOOST_LITERAL       = 0.35
_BOOST_EXPANDED_LIT  = 0.30   # slightly lower than direct literal — but still strong
_BOOST_SYMBOL        = 0.20
_BOOST_FTS           = 0.08
_BOOST_PER_SOURCE    = 0.06   # × min(extra_sources, 3)
_BOOST_CHAIN         = 0.12
_BOOST_LOCALIZED     = 0.15
_BOOST_ANCHOR_PATH   = 0.16   # dynamic: fires when hypothesis-derived anchor tokens
                              # overlap file path tokens. Fully repository-agnostic —
                              # no hardcoded extensions, framework roles, or domain words.
_BOOST_COMPONENT_GROUP = 0.20 # file is a sibling member of a component group
                              # (e.g., .html paired with a ranked .ts file).
                              # Strong enough to survive ranking thresholds but
                              # weaker than direct literal/symbol matches.
_BOOST_MACRO_ARCH    = 0.30   # file resides within a hypothesis-targeted macro
                              # directory (architecture-aware routing).
_PENALTY_OUTSIDE_MACRO = 0.20 # file is outside ALL hypothesis macro directories
                              # (cross-module contamination prevention).

_PENALTY_FNAME_ONLY  = 0.25
_PENALTY_INFRA       = 0.20
_PENALTY_WEAK        = 0.10
_PENALTY_BROAD_REPO  = 0.12

# Generic file-noise penalties (non-ticket-specific)
_PENALTY_TEST_FILE   = 0.22
_PENALTY_GENERATED   = 0.30
_PENALTY_DOCS        = 0.10

# Repository-boundary penalty — applied when a candidate file is not
# inside any recognized repository (e.g. Git repo).  Consistent with
# _PENALTY_INFRA / _PENALTY_OUTSIDE_MACRO in magnitude: meaningful
# enough to break ties but not strong enough to override genuinely
# superior evidence.  Repository identity is a signal, not a filter.
_PENALTY_ORPHAN_PATH = 0.20

_MIN_SCORE = 0.10
_MAX_SCORE = 1.00


# ---------------------------------------------------------------------------
# Output dataclass (trace-friendly)
# ---------------------------------------------------------------------------

@dataclass
class RankedFile:
    """Ranking result for a single unique file path."""
    file_path:          str
    final_score:        float
    original_score:     float
    rank:               int = 0

    matched_literals:    List[str] = field(default_factory=list)
    matched_expanded:    List[str] = field(default_factory=list)   # expansion matches
    matched_symbols:     List[str] = field(default_factory=list)
    supporting_sources:  List[str] = field(default_factory=list)
    ranking_reasons:     List[str] = field(default_factory=list)
    penalties:           List[str] = field(default_factory=list)
    arch_role:           str = ""   # "deploy_script" | "config_file" | "root_component" | ""
    repo_root:           str = ""   # containing repository root prefix, or "" if orphan

    def to_dict(self) -> dict:
        return {
            "rank":               self.rank,
            "file_path":          self.file_path,
            "final_score":        round(self.final_score, 4),
            "original_score":     round(self.original_score, 4),
            "arch_role":          self.arch_role,
            "repo_root":          self.repo_root,
            "matched_literals":   self.matched_literals,
            "matched_expanded":   self.matched_expanded,
            "matched_symbols":    self.matched_symbols,
            "supporting_sources": self.supporting_sources,
            "ranking_reasons":    self.ranking_reasons,
            "penalties":          self.penalties,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_infra_config(file_path: str) -> bool:
    """Return True for Kubernetes / Helm infrastructure YAML files."""
    stem = Path(file_path).stem.lower()
    return stem in _INFRA_STEMS


def _arch_role(file_path: str) -> str:
    """
    Classify a file by its architectural role.

    Returns:
        "deploy_script"   — shell/batch/PowerShell scripts
        "config_file"     — YAML, properties, env, XML, TOML, INI
        "root_component"  — application entry point / root component
        ""                — no special classification
    """
    suffix = Path(file_path).suffix.lower()
    if suffix in _DEPLOY_EXTENSIONS:
        return "deploy_script"
    if suffix in _CONFIG_EXTENSIONS:
        return "config_file"
    posix = file_path.replace("\\", "/")
    for pattern in _ROOT_PATTERNS_COMPILED:
        if pattern.search(posix):
            return "root_component"
    return ""




def _get_tiebreaker_priority(file_path: str) -> int:
    """
    Returns an integer priority (lower is better) for tie-breaking final scores.
    1. DEPLOYMENT
    2. ROOT_COMPONENT
    3. CONFIG
    4. VERSION_SOURCE
    5. VERSION_DISPLAY
    6. UNKNOWN
    7. TEST
    8. LOCK_FILE
    9. GENERATED
    """
    p = file_path.lower().replace("\\", "/")
    
    # 9. GENERATED
    if "/dist/" in p or "/node_modules/" in p or "/target/" in p or "/build/" in p:
        return 9
    # 8. LOCK_FILE
    if p.endswith(("package-lock.json", "yarn.lock", "shrinkwrap.json", "gradle.lockfile")):
        return 8
    # 7. TEST
    if (".spec." in p or ".test." in p or p.endswith("test.java") or 
        p.endswith("tests.java") or p.endswith("spec.java") or 
        p.endswith("_test.go") or p.endswith("_test.py") or 
        "/test/" in p or "/tests/" in p or "/spec/" in p or "__tests__" in p):
        return 7
    # 1. DEPLOYMENT
    if p.endswith((".sh", ".bat", ".ps1")) or "/helm/" in p or "/deploy/" in p or "/k8s/" in p or "/kubernetes/" in p or p.endswith("dockerfile") or ("manifest" in p and p.endswith((".yml", ".yaml"))) or ("deployment" in p and p.endswith((".yml", ".yaml"))):
        return 1
    # 2. ROOT_COMPONENT
    if p.endswith(("app.component.ts", "app.module.ts", "app.component.html", "app.component.scss", "main.ts", "main.tsx", "app.tsx", "app.jsx")) or "/app.component." in p or "/app.module." in p:
        return 2
    # 4. VERSION_SOURCE
    name_segment = p.rsplit("/", 1)[-1]
    if "version" in name_segment or "constant" in name_segment or "constants" in name_segment or "buildconfig" in name_segment:
        return 4
    # 3. CONFIG
    if p.endswith((".json", ".yaml", ".yml", ".env", ".properties", ".xml", ".toml", ".ini", ".cfg")):
        return 3
    
    # 6. UNKNOWN (default)
    return 6


def _is_test_file(file_path: str) -> bool:
    p = file_path.lower().replace("\\", "/")
    return (
        ".spec." in p
        or ".test." in p
        or p.endswith(("test.java", "tests.java", "spec.java", "_test.go", "_test.py"))
        or "/test/" in p
        or "/tests/" in p
        or "/spec/" in p
        or "__tests__" in p
    )


def _is_generated_or_vendor(file_path: str) -> bool:
    p = file_path.lower().replace("\\", "/")
    return any(seg in p for seg in (
        "/dist/", "/node_modules/", "/target/", "/build/", "/out/", "/generated/"
    ))


def _is_docs_file(file_path: str) -> bool:
    p = file_path.lower().replace("\\", "/")
    return p.endswith((".md", ".rst", ".txt")) or "/docs/" in p


def _is_ui_surface_file(file_path: str) -> bool:
    p = file_path.lower().replace("\\", "/")
    suffix = Path(file_path).suffix.lower()
    return (
        suffix in {".html", ".scss", ".css"}
        or "/assets/i18n/" in p
        or "/viewer/" in p
        or "/components/" in p
    )


def _is_backend_surface_file(file_path: str) -> bool:
    p = file_path.lower().replace("\\", "/")
    return (
        "/src/main/java/" in p
        or "/src/main/kotlin/" in p
        or "/controller" in p
        or "/service" in p
        or "/repository" in p
        or "/entity" in p
    )


def _derive_intent_tokens(hypotheses: list) -> Set[str]:
    """Extract lightweight intent tokens from investigation hypotheses."""
    tokens: Set[str] = set()
    for hyp in hypotheses:
        parts = [hyp.hypothesis] + list(hyp.queries or []) + list(hyp.literals or []) + list(hyp.symbols or [])
        for part in parts:
            for tok in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]*\b", (part or "").lower()):
                if len(tok) >= 3:
                    tokens.add(tok)
    return tokens


def _literal_from_snippet(snippet: str) -> Optional[str]:
    """Extract the literal value from a '[literal:...]' content_snippet."""
    if not snippet.startswith(_LITERAL_PREFIX):
        return None
    end = snippet.find("]", len(_LITERAL_PREFIX))
    if end < 0:
        return None
    return snippet[len(_LITERAL_PREFIX):end].strip("'\"")


def _expanded_from_snippet(snippet: str) -> Optional[str]:
    """
    Extract the original hypothesis literal from an expansion-provenance snippet.

    Expansion snippets have the form:
      [literal:'26.2'][expanded:'260200'] line N: ...

    Returns the VALUE of the [literal:...] tag — the original hypothesis term.
    """
    if not snippet.startswith(_LITERAL_PREFIX):
        return None
    # Must also contain [expanded:...]
    if _EXPANDED_PREFIX not in snippet:
        return None
    end = snippet.find("]", len(_LITERAL_PREFIX))
    if end < 0:
        return None
    return snippet[len(_LITERAL_PREFIX):end].strip("'\"")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class EvidenceRankingEngine:
    """
    Deterministic evidence ranker with query-expansion awareness.

    Usage::

        engine = EvidenceRankingEngine()
        ranked_items, ranked_files = engine.rank(
            evidence_items, hypotheses, localized_tasks,
            expansion_map=repo_search_engine.expansion_map,   # optional
        )
    """

    def rank(
        self,
        evidence_items: list,          # List[EvidenceItem]
        hypotheses: list,              # List[InvestigationHypothesis]
        localized_tasks: list,         # List[DevelopmentTask]
        expansion_map: Optional[Dict[str, Set[str]]] = None,
        repo_root_map: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[list, List[RankedFile]]:
        """
        Re-score all EvidenceItems and return them sorted by final score (desc).
        Also returns per-file RankedFile objects for tracing.

        Args:
            evidence_items:  All collected EvidenceItems.
            hypotheses:      Investigation hypotheses (supply literal/symbol sets).
            localized_tasks: Already-localized DevelopmentTask list.
            expansion_map:   Optional Dict[original_literal, Set[expanded_terms]]
                             from RepositorySearchEngine.expansion_map.
                             When supplied, files matched via expanded terms
                             receive the same literal-match credit as direct hits.
        """
        if not evidence_items:
            return evidence_items, []

        # Build all_literals from original hypothesis literals (lowercase)
        all_literals: Set[str] = set()
        all_symbols:  Set[str] = set()
        # Collect macro_directories from hypotheses for architecture-aware routing
        macro_directories: Set[str] = set()
        for hyp in hypotheses:
            all_literals.update(lit.lower() for lit in (hyp.literals or []))
            all_symbols.update(sym.lower() for sym in (hyp.symbols or []))
            for md in getattr(hyp, 'macro_directories', []) or []:
                if md:
                    # Normalize to forward slashes, strip trailing slash
                    macro_directories.add(md.replace('\\', '/').rstrip('/'))
        if macro_directories:
            logger.info(f"  [EvidenceRanking] Macro directories: {macro_directories}")

        intent_tokens = _derive_intent_tokens(hypotheses)

        # Dynamic domain anchors — purely derived from hypothesis literals, symbols,
        # and queries. No hardcoded words, no file-extension assumptions.
        # High-precision sources first: literals and symbols (explicitly extracted by
        # Investigation). Longer tokens from query text add breadth but only at ≥5 chars
        # to filter stopwords. These anchor tokens are what identify which specific
        # component/page/feature the ticket is about (e.g. "reviewer", "deliverable").
        anchor_tokens: Set[str] = set()
        for hyp in hypotheses:
            for term in list(hyp.literals or []) + list(hyp.symbols or []):
                for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", (term or "").lower()):
                    if len(tok) >= 4:
                        anchor_tokens.add(tok)
            for query in (hyp.queries or []):
                for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", (query or "").lower()):
                    if len(tok) >= 5:  # stricter for free-text queries
                        anchor_tokens.add(tok)

        # Build a reverse map: expanded_term → original_literal
        # e.g. "260200" → "26.2"
        # This lets ranking engine recognise expansion matches as literal matches.
        expanded_to_original: Dict[str, str] = {}
        if expansion_map:
            for original, expanded_set in expansion_map.items():
                for expanded in expanded_set:
                    expanded_to_original[expanded.lower()] = original.lower()

        # Localized file set for quick lookup
        localized_paths: Set[str] = {t.file_path for t in localized_tasks}

        # Group evidence items by file_path
        by_file: Dict[str, list] = {}
        for item in evidence_items:
            by_file.setdefault(item.file_path, []).append(item)

        # Score each unique file
        ranked_files: List[RankedFile] = []
        score_map:    Dict[str, float] = {}

        for fp, items in by_file.items():
            # Resolve repository identity for this candidate
            candidate_repo_root = (
                repo_root_map.get(fp.replace('\\', '/'))
                if repo_root_map else None
            )
            rf = self._score_file(
                fp, items, all_literals, all_symbols, localized_paths,
                expanded_to_original, anchor_tokens,
                repo_root=candidate_repo_root,
                has_repo_map=bool(repo_root_map),
            )
            # Architecture-aware macro_directories boost/penalty
            if macro_directories:
                fp_norm = fp.replace('\\', '/')
                in_macro = any(fp_norm.startswith(md) or fp_norm.startswith(md + '/') for md in macro_directories)
                if in_macro:
                    rf.final_score += _BOOST_MACRO_ARCH
                    rf.ranking_reasons.append(f"macro_arch_boost:+{_BOOST_MACRO_ARCH}")
                else:
                    # Suppress outside_macro penalty if file path matches anchor tokens
                    fp_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9]*", fp.lower()))
                    if not (anchor_tokens and (anchor_tokens & fp_tokens)):
                        rf.final_score -= _PENALTY_OUTSIDE_MACRO
                        rf.penalties.append(f"outside_macro:-{_PENALTY_OUTSIDE_MACRO}")
                rf.final_score = max(_MIN_SCORE, min(_MAX_SCORE, rf.final_score))
            ranked_files.append(rf)
            score_map[fp] = rf.final_score

        # Sort ranked files deterministic: 1. Score (desc), 2. Priority (asc), 3. Path (asc)
        ranked_files.sort(
            key=lambda r: (
                -r.final_score,
                _get_tiebreaker_priority(r.file_path),
                r.file_path.lower()
            )
        )
        for rank, rf in enumerate(ranked_files, 1):
            rf.rank = rank

        # Log top-10
        logger.info(f"  [EvidenceRanking] {len(ranked_files)} unique files scored:")
        for rf in ranked_files[:10]:
            logger.info(
                f"    #{rf.rank:2d}  {rf.final_score:.3f}  {rf.file_path}"
                + (f"  literals={rf.matched_literals}" if rf.matched_literals else "")
                + (f"  expanded={rf.matched_expanded}" if rf.matched_expanded else "")
                + (f"  [{rf.arch_role}]" if rf.arch_role else "")
            )

        for rf in ranked_files:
            logger.info(
                f"RANK_DEBUG file={rf.file_path} "
                f"final={rf.final_score:.4f}"
            )

        # Update relevance_score on EvidenceItems and sort by score deterministic
        for item in evidence_items:
            item.relevance_score = score_map.get(item.file_path, item.relevance_score)
        evidence_items.sort(
            key=lambda e: (
                -e.relevance_score,
                _get_tiebreaker_priority(e.file_path),
                e.file_path.lower()
            )
        )

        return evidence_items, ranked_files

    # ------------------------------------------------------------------
    # Per-file scorer
    # ------------------------------------------------------------------

    def _score_file(
        self,
        fp: str,
        items: list,
        all_literals:       Set[str],
        all_symbols:        Set[str],
        localized_paths:    Set[str],
        expanded_to_original: Dict[str, str],
        anchor_tokens: Set[str] = frozenset(),
        repo_root: Optional[str] = None,
        has_repo_map: bool = False,
    ) -> RankedFile:
        base = max(item.relevance_score for item in items)
        sources:   Set[str] = {item.source for item in items}
        reasons:   List[str] = []
        penalties: List[str] = []

        # ── Extract matched literals and expansion matches ──────────────
        matched_literals: List[str] = []   # direct hypothesis literal hits
        matched_expanded: List[str] = []   # hits via expansion (e.g. 260200 from 26.2)

        for item in items:
            if item.source != "repository_search":
                continue
            snippet = item.content_snippet or ""

            if _EXPANDED_PREFIX in snippet:
                # Expansion-provenance snippet: [literal:'26.2'][expanded:'260200']...
                orig_lit = _expanded_from_snippet(snippet)
                if orig_lit and orig_lit.lower() in all_literals:
                    if orig_lit not in matched_expanded:
                        matched_expanded.append(orig_lit)
            else:
                # Direct literal snippet: [literal:'26.2']...
                lit = _literal_from_snippet(snippet)
                if lit and lit.lower() in all_literals and lit not in matched_literals:
                    matched_literals.append(lit)
                elif lit:
                    # The matched term may itself be an expanded form
                    resolved = expanded_to_original.get(lit.lower())
                    if resolved and resolved in all_literals and resolved not in matched_expanded:
                        matched_expanded.append(resolved)

            # FTS items
            if item.source == "sqlite_fts":
                snip = snippet.lower()
                for lit in all_literals:
                    if lit and lit in snip and lit not in matched_literals:
                        matched_literals.append(lit)

        # ── SQLite symbol matches ────────────────────────────────────────
        matched_symbols: List[str] = []
        for item in items:
            if item.source == "sqlite_symbol" and item.symbol_name:
                if item.symbol_name.lower() in all_symbols:
                    if item.symbol_name not in matched_symbols:
                        matched_symbols.append(item.symbol_name)

        # ── Filename-only detection ──────────────────────────────────────
        repo_items = [i for i in items if i.source == "repository_search"]
        all_fname_only = bool(repo_items) and all(
            (i.content_snippet or "").startswith(_FILENAME_PREFIX)
            for i in repo_items
        )
        has_repo_content = any(
            (i.content_snippet or "").startswith(_LITERAL_PREFIX)
            or _EXPANDED_PREFIX in (i.content_snippet or "")
            for i in repo_items
        )

        # ── Architecture-aware role ──────────────────────────────────────
        role = _arch_role(fp)
        is_test = _is_test_file(fp)
        is_generated = _is_generated_or_vendor(fp)
        is_docs = _is_docs_file(fp)
        is_ui_surface = _is_ui_surface_file(fp)
        is_backend_surface = _is_backend_surface_file(fp)

        # ── Infra config detection ───────────────────────────────────────
        infra = _is_infra_config(fp)

        # ── Aggregate content signal ─────────────────────────────────────
        has_any_content = (
            bool(matched_literals)
            or bool(matched_expanded)
            or bool(matched_symbols)
            or "sqlite_fts" in sources
        )

        # ── Compute final score ──────────────────────────────────────────
        score = base

        # Direct literal match boost
        if matched_literals:
            score += _BOOST_LITERAL
            reasons.append(f"literal_match:{matched_literals}")

        # Expansion-match boost (slightly lower than direct, still strong)
        if matched_expanded:
            score += _BOOST_EXPANDED_LIT
            reasons.append(f"expansion_match:{matched_expanded}")

        # Symbol boost
        if matched_symbols:
            score += _BOOST_SYMBOL
            reasons.append(f"symbol_match:{matched_symbols}")

        # FTS boost
        if "sqlite_fts" in sources:
            score += _BOOST_FTS
            reasons.append("sqlite_fts_content")

        # Multi-source boost
        extra_sources = max(0, len(sources & (_INDEX_SOURCES | {"repository_search"})) - 1)
        if extra_sources > 0:
            ms_boost = _BOOST_PER_SOURCE * min(extra_sources, 3)
            score += ms_boost
            reasons.append(f"multi_source:{sorted(sources)}+{ms_boost:.2f}")

        # Chain boost
        if sources & _CHAIN_SOURCES:
            score += _BOOST_CHAIN
            reasons.append(f"chain:{sorted(sources & _CHAIN_SOURCES)}")

        # Localization boost
        if fp in localized_paths:
            score += _BOOST_LOCALIZED
            reasons.append("localized_plan_file")

        # Dynamic anchor-path boost — repository-agnostic, zero hardcoded extensions.
        # Fires when hypothesis-derived anchor tokens (the specific page/component/
        # business nouns in the ticket, e.g. "reviewer", "deliverable", "invoice")
        # overlap with the word tokens of this file's path. Works equally for UI,
        # backend, version, i18n, and any other ticket type — the anchor is always
        # dynamic, never hardcoded.
        if anchor_tokens:
            path_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9]*", fp.lower()))
            matched_anchors = anchor_tokens & path_tokens
            if matched_anchors:
                score += _BOOST_ANCHOR_PATH
                reasons.append(f"anchor_path:{sorted(matched_anchors)[:3]}")

        # Component group member boost — when a file was discovered as a sibling
        # of a ranked component file (e.g., .html sibling of .ts), it gets a
        # moderate boost to ensure it survives ranking. Does NOT fire for
        # standalone files or files with direct evidence.
        is_component_group_member = any(
            i.source == "component_group" for i in items
        )
        if is_component_group_member:
            score += _BOOST_COMPONENT_GROUP
            reasons.append("component_group_member")

        # ── Penalties ───────────────────────────────────────────────────
        if is_generated:
            score -= _PENALTY_GENERATED
            penalties.append("generated_or_vendor")

        if is_docs:
            score -= _PENALTY_DOCS
            penalties.append("documentation_file")

        if is_test:
            score -= _PENALTY_TEST_FILE
            penalties.append("test_file_non_test_intent")

        if all_fname_only and not has_repo_content:
            # Suppress penalty if file matches hypothesis anchor tokens
            has_anchor = bool(anchor_tokens and (anchor_tokens & set(re.findall(r"[a-zA-Z][a-zA-Z0-9]*", fp.lower()))))
            if not has_anchor:
                score -= _PENALTY_FNAME_ONLY
                penalties.append("filename_only_match")

        if infra:
            score -= _PENALTY_INFRA
            penalties.append("infra_config_file")

        # broad_repo_only penalty — SUPPRESSED when expansion provided evidence
        only_repo = sources == {"repository_search"}
        if only_repo and not has_any_content:
            score -= _PENALTY_BROAD_REPO
            penalties.append("broad_repo_only")

        # weak_single_source penalty — SUPPRESSED when expansion provided evidence
        if len(sources) == 1 and fp not in localized_paths and not has_any_content:
            score -= _PENALTY_WEAK
            penalties.append("weak_single_source")

        # ── Repository boundary signal ────────────────────────────────
        # Applied only when a repo_root_map was provided (repo discovery
        # ran successfully).  Files not inside any recognized repository
        # receive a penalty — a signal, not a filter.  A genuinely strong
        # orphan can still rank above a weak repository candidate.
        resolved_repo = repo_root if repo_root is not None else ""
        if has_repo_map and repo_root is None:
            score -= _PENALTY_ORPHAN_PATH
            penalties.append("orphan_path")

        final_pre_clamp = score
        final = round(min(_MAX_SCORE, max(_MIN_SCORE, final_pre_clamp)), 4)

        return RankedFile(
            file_path=fp,
            final_score=final,
            original_score=round(base, 4),
            matched_literals=matched_literals,
            matched_expanded=matched_expanded,
            matched_symbols=matched_symbols,
            supporting_sources=sorted(sources),
            ranking_reasons=reasons,
            penalties=penalties,
            arch_role=role,
            repo_root=resolved_repo,
        )
