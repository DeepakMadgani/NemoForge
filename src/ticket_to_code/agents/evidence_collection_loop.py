"""
Evidence Collection Loop

Iteratively queries ALL available evidence sources until a confidence
threshold is reached (default 0.70).  Uses every piece of existing
infrastructure - nothing is replaced:

  * Semantic search    -> RAGEngine.retrieve_context()
  * Neo4j graph        -> LocalizationAgent.neo4j_store
  * SQLite symbols     -> LocalizationAgent.sqlite_store (FTS5 + edges)
  * Java chain         -> _graph_neighbors (Spring boot architecture)
  * UI i18n chain      -> Full mapping via _extract_ui_text_labels
  * Typescript chain   -> Component->Template->Styles graph

The loop starts with an aggressive Stage 0 to trace UI strings directly
to components, then iteratively searches literals/symbols, then escalates
to graph and RAG if needed, dynamically constructing hypotheses.
"""

import os
import json
import logging
import re
from typing import List, Tuple, Dict, Set, Optional, Any, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass, field
from ticket_to_code.runtime.llm_factory import LLMRegistry
from ticket_to_code.runtime.limits import SNIPPET_MAX_CHARS
from ticket_to_code.tracing.paths import get_trace_dir
from langchain_core.messages import HumanMessage, SystemMessage

from ticket_to_code.models import *

if TYPE_CHECKING:
    from ticket_to_code.agents.repository_search_engine import RepositorySearchEngine
    from ticket_to_code.agents.providers.relationship_provider import RelationshipProvider
    from ticket_to_code.agents.localization_agent import LocalizationAgent
    from ticket_to_code.agents.rag_engine import CodebaseRAGEngine

logger = logging.getLogger(__name__)


def _top_root(path: str) -> str:
    """Top-level application/service root of a workspace-relative path.

    e.g. 'xchange-ui/src/app/..' → 'xchange-ui', 'area-service/src/..' →
    'area-service'. Used to keep an i18n key's consumers in the SAME app as the
    translation file that defined it — a repository-metadata constraint, NOT a
    language/layer rule. Returns '' when the path has no distinct top segment.
    """
    n = (path or "").replace("\\", "/").lstrip("/")
    # Drop a leading drive/absolute prefix if present (e.g. C:/CC4E/xchange-ui/..)
    parts = [p for p in n.split("/") if p and not p.endswith(":")]
    return parts[0] if len(parts) > 1 else ""

# -- Tuning Constants ---------------------------------------------------------
_CONFIDENCE_THRESHOLD = 0.70

# Config / property file extensions (application-level only; .json data files
# are intentionally excluded - they match almost every word)
_CONFIG_EXTS = frozenset({
    '.properties', '.yaml', '.yml', '.xml',
    '.env', '.conf', '.ini', '.toml',
})

# Subdirectory path segments that are NEVER meaningful config locations
_CONFIG_SKIP_SEGMENTS = frozenset({
    'traces', 'assets', 'postman', '.idea', '__pycache__',
    'node_modules', 'dist', 'build', '.angular', 'coverage',
})

# Minimum literal length for config chain search - short words like 'Add',
# 'Cancel', 'filter' match almost everywhere and produce noise.
_CONFIG_MIN_LITERAL_LEN = 6

# Minimum literal length for live repository search - shorter than config chain
# to capture version numbers like '26.2' (4 chars).
_REPO_SEARCH_MIN_LITERAL_LEN = 4
# Java stereotype annotations
_JAVA_STEREOTYPES = frozenset({
    'Controller', 'RestController', 'Service', 'Component',
    'Repository', 'Configuration', 'Bean',
})

# -- Agentic Loop constants ------------------------------------------
_AGENTIC_MAX_ITERATIONS = int(os.getenv("AVIATOR_AGENTIC_MAX_ITERATIONS", "4"))

# -- Query sanitizer: strip CLI flags hallucinated by the LLM ----------
def _sanitize_query(query: str) -> str:
    """Strip CLI flags hallucinated by the LLM from a search query.

    The LLM sometimes emits queries like:
        'Upload Document register --glob "*.{ts,html}"'
    which should be just:
        'Upload Document register'

    This function defensively removes known flag patterns.
    """
    import re as _re
    _FLAG_PATTERNS = [
        r'--glob\s+"[^"]*"', r"--glob\s+'[^']*'", r'--glob\s+\S+',
        r'--max-depth\s+\d+', r'--path\s+\S+',
        r'--include\s+"[^"]*"', r"--include\s+'[^']*'", r'--include\s+\S+',
        r'--type\s+\S+', r'-[girnlcwFe]\b',
        r'--fixed-strings', r'--case-sensitive', r'--ignore-case',
    ]
    cleaned = query
    for pat in _FLAG_PATTERNS:
        cleaned = _re.sub(pat, '', cleaned)
    cleaned = _re.sub(r'\s+', ' ', cleaned).strip()
    if cleaned != query.strip():
        logger.info(f"  \U0001f9f9 Query sanitized: '{query.strip()[:80]}' \u2192 '{cleaned[:80]}'")
    return cleaned or query.strip()  # Fallback if sanitized to empty

# Hit-classification demotion multipliers (applied to relevance at the
# consumer): a comment/string/test/doc match is still evidence — never
# silently dropped (rank-and-keep) — but it must not outrank executable code.
_HIT_CLASS_MULTIPLIER = {
    "comment": 0.6,
    "string": 0.8,
    "test_file": 0.7,
    "doc": 0.65,
}

_SEARCH_TOOLS_NEEDING_SANITIZATION = frozenset({
    'ripgrep', 'filename_search', 'regex_search', 'sqlite_fts',
    'symbol_lookup', 'semantic_rag', 'i18n_chain', 'ast_companion_bundle',
    'git_history', 'call_sites', 'lsp', 'cooccurrence', 'grep_search', 'view_file',
})

# Tool catalog - prompt content the LLM sees when deciding which tool to use.
# Not code-level branching. The model reads this, picks a tool, picks a query.
TOOL_CATALOG = """
Available discovery capabilities (pick exactly ONE per iteration):

Choose the capability that can ANSWER your current unresolved question.
Do NOT choose a capability merely because it exists or hasn't been used yet.

| Capability | When to use | Latency |
|------------|-------------|---------|
| ripgrep | When you know an exact identifier, error message, config key, or string literal to find in the codebase. | <1s |
| grep_search | When you need exact pattern or regex matching with line numbers and matched content snippets across the repository. | <1s |
| view_file | When you need to read a targeted line range (e.g. "path:start_line-end_line" or file path) directly from disk with line numbers. | <1s |
| filename_search | When you need to locate a file by name fragment (e.g. "deliverable", "contract-member"). | <1s |
| regex_search | When you need pattern matching: version numbers, URL patterns, annotation formats. | <1s |
| sqlite_fts | When you need indexed full-text discovery and exact textual or symbol-oriented search is useful. | <1s |
| symbol_lookup | When an exact class, method, interface, type, or symbol must be resolved to its definition/owner and textual search alone is insufficient. | <1s |
| ast_companion_bundle | When you have a component file path or symbol and need its existing structural companions (.ts, .html, .scss, .spec.ts) discovered together. Discovery only. | <1s |
| neo4j_graph | When a relevant file/symbol is already known and you need to discover its callers, callees, imports, implementations, or other repository relationships. Do not use merely because graph data exists — use it when resolving a relationship can answer an unresolved investigation question. | 1-3s |
| ts_chain | When you need Angular structural edges from a component path or symbol: component.ts -> .html -> .scss -> .spec.ts. | 1-2s |
| java_chain | When you need Spring layer traversal from a Java file path or symbol: Controller -> Service -> Repository -> Entity/DTO. | 1-2s |
| python_chain | When you need Python module import chain traversal. | 1-2s |
| css_chain | When you need CSS/SCSS inheritance and shared class chains. | 1-2s |
| config_chain | When you need to trace property/YAML/XML configuration file chains. | 1-2s |
| layout_chain | When you need Angular routing and layout container chains. | 1-2s |
| semantic_rag | When the ticket or current hypothesis uses business/conceptual language whose implementation terminology may differ from the repository's literal identifiers. Do not use merely because semantic search exists — use it when semantic discovery can resolve an open question or find files that exact search would miss. | 3-5s |
| i18n_chain | When UI labels, translation keys, business terminology, or localization resources can help identify or confirm the implementation location. Especially useful for tickets that reference user-visible text or error messages. | <1s |
| git_history | When you want the files that were recently changed together with a known identifier, string, or feature. Matches BOTH file contents (pickaxe) and commit MESSAGES (ticket IDs like PROJ-1234, feature names), and surfaces co-change partners of the top files. | 1-3s |
| call_sites | When a known method/function name must be traced to every file that invokes it (callers, event handlers, template bindings). Deterministic cross-file caller discovery. | 1-2s |
| lsp | When you need a class's definition home with its members (constructor, methods, properties), or every cross-file reference to a type — language-aware extraction across TypeScript, Java, Kotlin, Python and C#. | 1-3s |
| cooccurrence | When one term alone is too broad: pass 2-6 space-separated terms and get only the files containing ALL of them (AND). Collapses hundreds of single-term matches to the handful of files that matter. | <1s |

CRITICAL: The "query" field is a PURE SEARCH STRING — NOT a shell command.
Do NOT include CLI flags like --glob, --path, --max-depth, --include, -i, etc.
The tools already handle file filtering internally.
  WRONG: {"tool": "ripgrep", "query": "Upload Document --glob \"*.ts\""}
  RIGHT: {"tool": "ripgrep", "query": "Upload Document register"}
  WRONG: {"tool": "filename_search", "query": "member --include \"*.component.ts\""}
  RIGHT: {"tool": "filename_search", "query": "member.component"}
"""

# ── Progressive Artifact Inspection dataclasses ─────────────────────────────
# These represent the structured knowledge extracted from inspected artifacts.
# An artifact goes through: DISCOVERED → VERIFIED → INSPECTED → UNDERSTOOD.

@dataclass
class InspectionFact:
    """A verified fact about an artifact, with provenance."""
    fact: str                             # "getProjectMembership() exists"
    source: str                           # "sqlite" | "workspace_symbol_index" | "api_contract_detector"
    confidence: float = 0.95             # 0.0-1.0
    line_start: Optional[int] = None
    line_end: Optional[int] = None


@dataclass
class MethodLocation:
    """A discovered method with its location for targeted reading."""
    name: str
    signature: str = ""                  # "getProjectMembership(projectId, userEmail)"
    return_type: str = ""
    parameters: List[str] = field(default_factory=list)
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    visibility: str = "public"
    relevance: str = ""                  # Why this method is relevant to the ticket

    @property
    def start_line(self) -> Optional[int]:
        return self.line_start

    @property
    def end_line(self) -> Optional[int]:
        return self.line_end


@dataclass
class ArtifactInspection:
    """Structured knowledge about an inspected artifact.

    Produced by _inspect_artifact() using the strongest available
    structural provider (SQLite → WorkspaceSymbolIndex → APIContractDetector
    → targeted file read). Provider selection is completeness-driven:
    if SQLite provides full method-level info, skip further providers.
    """

    file_path: str = ""
    class_name: str = ""
    language: str = ""
    provider: str = ""                   # Primary provider used

    # Level 1: Structure (from index / symbol providers)
    methods: List[MethodLocation] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    type_dependencies: List[str] = field(default_factory=list)

    # Level 2: Relevance (which parts matter for THIS ticket)
    relevant_methods: List[MethodLocation] = field(default_factory=list)
    relevant_regions: List[tuple] = field(default_factory=list)   # [(start, end), ...]

    # Level 2b: Actual source code for relevant regions (targeted reads)
    # Maps "method_name" -> {"code": "...", "start": N, "end": M, "file": "path"}
    relevant_code_regions: Dict[str, Dict] = field(default_factory=dict)

    # Level 3: Contracts (API boundaries)
    api_contracts: List[str] = field(default_factory=list)

    # Level 4: Knowledge (with provenance)
    facts: List[InspectionFact] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)

    # Metadata
    inspection_depth: str = "none"       # "index" | "symbols" | "contracts" | "region"
    providers_used: List[str] = field(default_factory=list)

    def to_compact_summary(self, max_chars: int = 400) -> str:
        """Generate compact summary for the LLM decision context."""
        parts = []
        if self.class_name:
            parts.append(f"Class: {self.class_name}")
        if self.methods:
            method_names = [m.name for m in self.methods[:8]]
            parts.append(f"Methods: {', '.join(method_names)}")
            if len(self.methods) > 8:
                parts[-1] += f" (+{len(self.methods) - 8} more)"
        if self.type_dependencies:
            parts.append(f"Deps: {', '.join(self.type_dependencies[:5])}")
        if self.relevant_methods:
            rel_sigs = [m.signature or m.name for m in self.relevant_methods[:3]]
            parts.append(f"Relevant: {', '.join(rel_sigs)}")
        if self.api_contracts:
            parts.append(f"Contracts: {', '.join(self.api_contracts[:3])}")
        if self.facts:
            parts.append(f"Facts: {'; '.join(f.fact for f in self.facts[:3])}")
        if self.questions:
            parts.append(f"Questions: {'; '.join(self.questions[:2])}")
        if self.unresolved:
            parts.append(f"Unresolved: {'; '.join(self.unresolved[:2])}")

        result = " | ".join(parts)
        return result[:max_chars] if len(result) > max_chars else result

    def to_evidence_block(self, file_path: str = "") -> str:
        """Generate full evidence block including actual code regions.

        This is the minimum useful unit of evidence for behavioral reasoning:
        structural metadata + actual source code for relevant methods.
        """
        lines = []
        if self.class_name:
            lines.append(f"  Class: {self.class_name}")
        if self.methods:
            method_names = [m.signature or m.name for m in self.methods[:10]]
            lines.append(f"  Methods: {', '.join(method_names)}")
            if len(self.methods) > 10:
                lines.append(f"    (+{len(self.methods) - 10} more)")
        if self.type_dependencies:
            lines.append(f"  Dependencies: {', '.join(self.type_dependencies[:5])}")
        if self.api_contracts:
            lines.append(f"  Contracts: {', '.join(self.api_contracts[:3])}")
        if self.relevant_methods:
            rel_names = [m.signature or m.name for m in self.relevant_methods[:3]]
            lines.append(f"  Relevant: {', '.join(rel_names)}")

        # Include actual code regions — this is critical for behavioral reasoning
        for method_name, region_info in list(self.relevant_code_regions.items())[:2]:
            code = region_info.get("code", "")
            start = region_info.get("start", "?")
            end = region_info.get("end", "?")
            if code:
                lines.append(f"  --- RELEVANT CODE: {method_name}() L{start}-{end} ---")
                # Bound to prevent context explosion
                code_lines = code.splitlines()[:60]
                for cl in code_lines:
                    lines.append(f"  {cl}")
                if len(code.splitlines()) > 60:
                    lines.append(f"  ... ({len(code.splitlines()) - 60} more lines)")
                lines.append(f"  --- END CODE ---")

        if self.facts:
            for f in self.facts[:3]:
                lines.append(f"  Fact: [{f.source}] {f.fact}")
        if self.questions:
            for q in self.questions[:3]:
                lines.append(f"  Question: {q}")
        return "\n".join(lines)

    def to_planning_context(self) -> str:
        """Generate targeted planning context (structural + relevant regions)."""
        lines = []
        if self.class_name:
            lines.append(f"CLASS: {self.class_name}")
        if self.methods:
            lines.append("METHODS:")
            for m in self.methods:
                loc = f" (L{m.line_start}-{m.line_end})" if m.line_start else ""
                lines.append(f"  {m.signature or m.name}: {m.return_type}{loc}")
        if self.type_dependencies:
            lines.append(f"DEPENDENCIES: {', '.join(self.type_dependencies)}")
        if self.api_contracts:
            lines.append(f"API CONTRACTS: {', '.join(self.api_contracts)}")
        if self.facts:
            lines.append("VERIFIED FACTS:")
            for f in self.facts:
                lines.append(f"  [{f.source}] {f.fact}")
        # Include actual code for relevant methods (targeted reads)
        for method_name, region_info in list(self.relevant_code_regions.items())[:3]:
            code = region_info.get("code", "")
            start = region_info.get("start", "?")
            end = region_info.get("end", "?")
            if code:
                lines.append(f"--- RELEVANT CODE: {method_name}() L{start}-{end} ---")
                for cl in code.splitlines()[:80]:
                    lines.append(cl)
                lines.append("--- END CODE ---")
        return "\n".join(lines)


@dataclass
class EvidenceKnowledge:
    """Aggregate knowledge from all inspected artifacts across iterations."""
    verified_artifacts: List[str] = field(default_factory=list)
    structural_facts: List[InspectionFact] = field(default_factory=list)
    code_facts: List[Any] = field(default_factory=list)
    relationships: List[Dict] = field(default_factory=list)
    api_contracts: List[str] = field(default_factory=list)
    all_facts: List[InspectionFact] = field(default_factory=list)
    all_questions: List[str] = field(default_factory=list)
    all_unresolved: List[str] = field(default_factory=list)
    resolved_questions: List[str] = field(default_factory=list)
    iteration_gains: List[Dict[str, int]] = field(default_factory=list)
    inspection_knowledge: Dict[str, ArtifactInspection] = field(default_factory=dict)
    # Synthesized, plan-ready understanding (set by workflow Phase 5b). Any to
    # avoid a hard import cycle with the composition module.
    behavioral_understanding: Optional[Any] = None

    @property
    def unresolved_questions(self) -> List[str]:
        return [q for q in self.all_questions if q not in self.resolved_questions]

    @property
    def recent_information_gain(self) -> str:
        return self.recent_gain_summary()

    def record_iteration(self, new_artifacts: int, new_facts: int,
                         new_questions: int, resolved: int,
                         new_methods: int = 0, new_relationships: int = 0) -> None:
        """Track information gain per iteration."""
        self.iteration_gains.append({
            "new_artifacts": new_artifacts,
            "new_facts": new_facts,
            "new_questions": new_questions,
            "resolved": resolved,
            "new_methods": new_methods,
            "new_relationships": new_relationships,
        })

    def recent_gain_summary(self, last_n: int = 3) -> str:
        """Summarize information gain over recent iterations."""
        recent = self.iteration_gains[-last_n:] if self.iteration_gains else []
        if not recent:
            return ""
        total_artifacts = sum(g.get("new_artifacts", 0) for g in recent)
        total_facts = sum(g.get("new_facts", 0) for g in recent)
        total_resolved = sum(g.get("resolved", 0) for g in recent)
        total_methods = sum(g.get("new_methods", 0) for g in recent)
        total_rels = sum(g.get("new_relationships", 0) for g in recent)
        consecutive_zero = 0
        for g in reversed(self.iteration_gains):
            if (g.get("new_artifacts", 0) == 0 and g.get("new_facts", 0) == 0
                    and g.get("new_methods", 0) == 0):
                consecutive_zero += 1
            else:
                break
        parts = [
            f"Last {len(recent)} iteration(s): +{total_artifacts} new artifact(s), "
            f"+{total_methods} method(s), +{total_facts} fact(s), "
            f"+{total_rels} relationship(s), {total_resolved} resolved question(s)"
        ]
        if consecutive_zero >= 2:
            parts.append(
                f"⚠️ {consecutive_zero} consecutive iterations with no new information. "
                "Consider using 'follow' on discovered relationships or proceeding if current evidence is sufficient."
            )
        return "\n".join(parts)

    def had_material_change(self) -> bool:
        """Did the most recent iteration produce any material evidence change?

        Material change = any new artifact, fact, method, relationship,
        resolved question, or newly identified question in the LATEST
        iteration only.  NOT cumulative — iteration 3 having changes does
        not make iteration 4 return True if iteration 4 had none.
        """
        if not self.iteration_gains:
            return False
        latest = self.iteration_gains[-1]
        return (
            latest.get("new_artifacts", 0) > 0
            or latest.get("new_facts", 0) > 0
            or latest.get("new_methods", 0) > 0
            or latest.get("new_relationships", 0) > 0
            or latest.get("resolved", 0) > 0
            or latest.get("new_questions", 0) > 0
        )


class EvidenceCollectionLoop:

    """

    Runs the iterative evidence collection loop for a ticket.

    Uses existing LocalizationAgent infrastructure (sqlite_store,

    neo4j_store, _graph_neighbors) and RAGEngine — no new storage layers.

    Usage::

        loop = EvidenceCollectionLoop(localizer, rag_engine)

        items, confidence = loop.collect(hypotheses, localized_tasks, ticket)

    """

    def __init__(

        self,

        localizer: "LocalizationAgent",

        rag_engine: "CodebaseRAGEngine",

        repo_search: Optional["RepositorySearchEngine"] = None,

        relationship_provider: Optional["RelationshipProvider"] = None,

    ) -> None:

        self.localizer = localizer

        self.rag_engine = rag_engine

        self.repo_search = repo_search   # Phase 3A: live filesystem search

        self.relationship_provider = relationship_provider

        self._workspace = localizer.workspace_path

        self._current_followable_relationships: Dict[str, Dict] = {}  # Phase 6: follow action state

        # True when any _decide_next_action() in the last collect() fell back to
        # "proceed" because the LLM was unavailable / errored / unparseable.
        # workflow.py reads `decision_degraded` after collect() returns.
        self.decision_degraded: bool = False
        self.decision_degraded_reasons: List[str] = []
        self.last_collect_flags: Dict[str, Any] = {}

    def _note_decision(self, action: dict) -> None:
        """Record whether an agentic decision was a silent degraded fallback."""
        if isinstance(action, dict) and action.get("degraded"):
            self.decision_degraded = True
            reason = str(action.get("reason", "") or "")
            self.decision_degraded_reasons.append(reason)
            logger.warning(f"[AgenticLoop] decision DEGRADED (fell back to proceed): {reason[:200]}")

    def _discover_candidates_via_repo_tree(self, ticket: Any) -> List[str]:
        """
        Market AI IDE Fallback (Cursor/Devin Standard):
        When keyword/anchor searches yield 0 results, generate a compact
        repository structural map and use LLM software engineering knowledge
        to identify the most likely candidate files.
        """
        ws = Path(self._workspace) if self._workspace else None
        if not ws or not ws.exists():
            return []

        skip_dirs = {
            ".git", ".venv", "venv", "node_modules", "target", "build", "dist",
            ".idea", ".vscode", "__pycache__", "obj", "bin", ".nemoforge", ".aviator"
        }
        valid_exts = {
            ".java", ".ts", ".tsx", ".js", ".jsx", ".py", ".cs", ".go", ".rs",
            ".html", ".css", ".scss", ".sql", ".json", ".xml", ".yaml", ".yml"
        }

        tree_lines = []
        try:
            for root, dirs, files in os.walk(str(ws)):
                dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
                rel_root = os.path.relpath(root, str(ws)).replace("\\", "/")
                depth = rel_root.count("/") if rel_root != "." else 0
                if depth > 4:
                    dirs.clear()
                    continue

                code_files = [f for f in files if any(f.endswith(ext) for ext in valid_exts)]
                if code_files:
                    prefix = "" if rel_root == "." else f"{rel_root}/"
                    for f in code_files[:15]:
                        tree_lines.append(f"{prefix}{f}")
                        if len(tree_lines) >= 180:
                            break
                if len(tree_lines) >= 180:
                    break
        except Exception as e:
            logger.warning(f"Error building repo tree for LLM discovery: {e}")
            return []

        if not tree_lines:
            return []

        compact_tree = "\n".join(tree_lines[:150])
        t_title = getattr(ticket, "title", "") or ""
        t_desc = (getattr(ticket, "description", "") or "")[:1200]

        prompt = f"""You are an expert SWE agent localizing target files in an enterprise repository.
Exact keyword search in code found no direct matches.
Given the repository file structure below and the ticket requirements, use your software architecture knowledge to identify the 3 to 6 most relevant candidate files that implement, handle, or test this functionality.

TICKET: {t_title}
DESCRIPTION: {t_desc}

REPOSITORY FILE TREE:
{compact_tree}

Respond ONLY with a JSON list of relative file paths from the tree above.
["path/to/file1", "path/to/file2"]"""

        try:
            from ticket_to_code.llm_utils import llm_invoke
            from langchain_core.messages import SystemMessage, HumanMessage
            from ticket_to_code.runtime.llm_factory import LLMRegistry

            llm = LLMRegistry.get_llm(assistant=True)
            res = llm_invoke(llm, [
                SystemMessage(content="You are a senior repository localization expert. Output pure JSON only."),
                HumanMessage(content=prompt),
            ], label="repo_tree_discovery")

            cleaned = (res.content or "").strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()

            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                valid_paths = []
                for p in parsed:
                    clean_p = str(p).strip().replace("\\", "/")
                    if (ws / clean_p).exists() and (ws / clean_p).is_file():
                        valid_paths.append(clean_p)
                logger.info(f"  🧠 LLM Repo Tree Reasoning identified {len(valid_paths)} candidate files: {valid_paths}")
                return valid_paths
        except Exception as exc:
            logger.warning(f"Repo tree LLM reasoning failed: {exc}")

        return []

    def _discover_related_files(self, candidate_files: List[str]) -> List[str]:
        """
        Given identified candidate files, discover related companion contracts,
        service/controller pairs, and test suites.
        """
        ws = Path(self._workspace) if self._workspace else None
        if not ws or not candidate_files:
            return []

        related: Set[str] = set()
        for fp in candidate_files:
            p = ws / fp
            if not p.is_file():
                continue

            stem = p.stem
            suffix = p.suffix
            parent = p.parent

            # 1. Companion test file lookup
            test_candidates = [
                parent / f"{stem}Test{suffix}",
                parent / f"{stem}Tests{suffix}",
                parent / f"{stem}.spec{suffix}",
                parent / f"{stem}.test{suffix}",
                parent / f"test_{stem}{suffix}",
            ]
            for tc in test_candidates:
                if tc.is_file():
                    try:
                        related.add(str(tc.resolve().relative_to(ws.resolve())).replace("\\", "/"))
                    except Exception:
                        pass

            # 2. Companion template/controller pairing (HTML <-> TS)
            if suffix in (".ts", ".js"):
                html_comp = p.with_suffix(".html")
                if html_comp.is_file():
                    try:
                        related.add(str(html_comp.resolve().relative_to(ws.resolve())).replace("\\", "/"))
                    except Exception:
                        pass
            elif suffix == ".html":
                ts_comp = p.with_suffix(".ts")
                if ts_comp.is_file():
                    try:
                        related.add(str(ts_comp.resolve().relative_to(ws.resolve())).replace("\\", "/"))
                    except Exception:
                        pass

            # 3. Import inspection for intra-repo dependencies
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")[:10000]
                # TypeScript/JavaScript relative imports
                for m in re.finditer(r'''(?:from|import)\s+['"](\.[^'"]+)['"]''', content):
                    rel_import = m.group(1)
                    for ext in (".ts", ".tsx", ".js", ".jsx", "/index.ts"):
                        resolved_imp = (parent / rel_import).resolve()
                        cand_file = Path(f"{resolved_imp}{ext}") if not ext.startswith("/") else resolved_imp / "index.ts"
                        if cand_file.is_file() and ws in cand_file.parents:
                            related.add(str(cand_file.relative_to(ws)).replace("\\", "/"))
            except Exception:
                pass

        return sorted(list(related - set(candidate_files)))[:10]

    # =========================================================================

    # PUBLIC ENTRY POINT

    # =========================================================================

    def collect(

        self,

        ticket: ValueEdgeTicket,

        hypotheses: Optional[List[Any]] = None,

        localized_tasks: Optional[List[Any]] = None,

        technical_facts: Optional[Any] = None,

        investigation: Optional[Any] = None,

        search_scope: Optional[Any] = None,

        run_ctx=None,

        step_callback=None,

        requirements=None,  # Real requirements for sufficiency evaluation

        recovery_context: Optional[dict] = None,  # Structured recovery from planning_recovery_node

    ) -> Tuple[List[EvidenceItem], float]:

        """

        Run evidence collection until confidence >= threshold or budget exhausted.

        Implements V3 Iterative Discovery Engine (Milestone 1).

        Args:
            step_callback: Optional callable that receives dict sub-step events
                for real-time UI updates. When provided, called at key moments:
                stage_0_start, stage_0_result, iteration_start, search_result,
                verdict, iteration_end. Does NOT change evidence logic.

        """

        import time

        import json

        import os

        def _emit(event: dict):
            """Emit a sub-step event if callback is provided."""
            if step_callback:
                try:
                    step_callback(event)
                except Exception as _cb_err:
                    logger.debug(f"step_callback error (non-fatal): {_cb_err}")

        if recovery_context is not None and not isinstance(recovery_context, dict):
            if hasattr(recovery_context, "model_dump"):
                recovery_context = recovery_context.model_dump()
            elif hasattr(recovery_context, "dict"):
                recovery_context = recovery_context.dict()
            elif hasattr(recovery_context, "__dict__"):
                recovery_context = dict(recovery_context.__dict__)

        print("\n" + "=" * 80)

        print("[ENTER] EvidenceCollectionLoop.collect() [V3 Iterative Engine]")

        print("=" * 80)

        # Reset per-collect degradation flags (read by workflow.py after return)
        self.decision_degraded = False
        self.decision_degraded_reasons = []
        self.last_collect_flags = {}

        all_evidence: List[EvidenceItem] = []

        all_code_facts: List[Any] = []

        confidence: float = 0.0

        

        # Initialize Central SearchContext

        context = SearchContext(ticket_facts=technical_facts)

        if localized_tasks:

            for t in localized_tasks:

                context.visited_files.add(t.file_path.replace("\\", "/"))

        

        # Initial search anchors from TechnicalFacts

        SearchAnchor = __import__('ticket_to_code.models', fromlist=['SearchAnchor']).SearchAnchor

        if technical_facts:

            # Add explicit literals

            for lit in technical_facts.high_priority_literals:

                context.anchor_history.append(

                    SearchAnchor(name=lit, kind="literal", source_file="ticket")

                )

            # Add explicit files

            for file_fact in technical_facts.files:

                context.anchor_history.append(

                    SearchAnchor(name=file_fact.name, kind="filename", source_file="ticket", score=10)

                )

        

        # FIX: Seed anchors from hypotheses and ticket when technical_facts is absent.

        # In LangGraph pipeline mode, technical_facts is never populated, so the evidence

        # loop would start with 0 anchors → 0 evidence → workflow dies.

        if not context.anchor_history:

            logger.info("  ðŸ“Œ No TechnicalFacts — seeding anchors from hypotheses & ticket")

            # Extract class/method/file names from hypotheses

            if hypotheses:

                for h in hypotheses[:4]:

                    hyp_text = getattr(h, "hypothesis", str(h))

                    # Extract PascalCase class names (e.g. ContractMemberService)

                    class_names = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', hyp_text)

                    for cn in class_names[:3]:

                        context.anchor_history.append(

                            SearchAnchor(name=cn, kind="class", source_file="hypothesis", score=8)

                        )

                    # Extract camelCase method names (e.g. saveMember)

                    method_names = re.findall(r'\b([a-z]+(?:[A-Z][a-z]+)+)\b', hyp_text)

                    for mn in method_names[:3]:

                        context.anchor_history.append(

                            SearchAnchor(name=mn, kind="method", source_file="hypothesis", score=7)

                        )

                    # Use hypothesis text as a broad literal search anchor

                    if hyp_text and len(hyp_text) > 10:

                        context.anchor_history.append(

                            SearchAnchor(name=hyp_text[:100], kind="literal", source_file="hypothesis", score=6)

                        )

            # Extract class/file names from localized tasks

            if localized_tasks:

                for t in localized_tasks[:5]:

                    fp = getattr(t, "file_path", "")

                    # Use the class name from the localization result

                    cls_name = getattr(t, "target_class", None)

                    if cls_name:

                        context.anchor_history.append(

                            SearchAnchor(name=cls_name, kind="class", source_file="localized_task", score=9)

                        )

                    # Use filename stem as a symbol anchor

                    stem = Path(fp).stem if fp else None

                    if stem and len(stem) > 3:

                        context.anchor_history.append(

                            SearchAnchor(name=stem, kind="class", source_file="localized_task", score=8)

                        )

            # Final fallback: use ticket title as a broad search

            if not context.anchor_history and ticket:

                title = getattr(ticket, "title", "") or ""

                if title:

                    context.anchor_history.append(

                        SearchAnchor(name=title[:100], kind="literal", source_file="ticket_title", score=5)

                    )

            

            logger.info(f"  ðŸ“Œ Seeded {len(context.anchor_history)} anchors from hypotheses/tasks/ticket")

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            logger.info(f"  📌 Seeded {len(context.anchor_history)} anchors from hypotheses/tasks/ticket")

        # =====================================================================
        # STAGE 0: REMOVED — i18n chain is now on-demand only
        # =====================================================================
        # Previously this stage extracted up to 15 UI text labels and ran
        # i18n chain tracing on ALL of them before the agentic loop.
        # This created up to 15 × N search_literal calls before the
        # investigation controller knew which questions needed answering.
        #
        # Architecture decision (2026-09-16):
        #   Tools are capabilities, not mandatory stages.
        #   _decide_next_action() chooses i18n_chain when the current
        #   investigation state has an unresolved question about where
        #   a specific UI text or label originates.  The TOOL_CATALOG
        #   description guides the LLM to select i18n_chain for tickets
        #   that reference user-visible text or error messages.
        #
        # All 19 discovery capabilities remain available to the controller:
        #   ripgrep, filename_search, regex_search, sqlite_fts,
        #   symbol_lookup, ast_companion_bundle, neo4j_graph, ts_chain,
        #   java_chain, python_chain, css_chain, config_chain,
        #   layout_chain, semantic_rag, i18n_chain, git_history, call_sites, lsp,
#   cooccurrence
        #
        # _extract_ui_text_labels() and _trace_i18n_chain() remain
        # unchanged — they are still invoked when _execute_tool()
        # dispatches i18n_chain on demand.
        # =====================================================================

        # Stage 0 (lightweight): extract UI labels as anchors for the
        # agentic loop.  NO i18n chain tracing here — the controller will
        # invoke i18n_chain if/when it identifies an unresolved question
        # about UI text origins.
        try:
            ui_labels = self._extract_ui_text_labels(ticket, hypotheses)
            if ui_labels:
                logger.info(
                    f"  📌 Stage 0: Extracted {len(ui_labels)} UI text labels "
                    f"as anchors (i18n chain deferred to agentic loop)"
                )
                _emit({
                    "type": "stage_0_start",
                    "message": f"Extracted {len(ui_labels)} UI text labels as anchors (i18n on-demand)",
                    "labels": ui_labels[:10],
                })
                for label in ui_labels:
                    context.anchor_history.append(
                        SearchAnchor(
                            name=label, kind="visual_text",
                            source_file="ui_label", score=9,
                        )
                    )
                _emit({
                    "type": "stage_0_result",
                    "message": f"Seeded {len(ui_labels)} UI text anchors (i18n chain on-demand)",
                    "files_found": 0,
                    "files": [],
                })
            else:
                logger.info("  📌 Stage 0: No UI text labels extracted — skipping anchor seeding")
                _emit({
                    "type": "stage_0_result",
                    "message": "No UI text labels found — no anchors seeded",
                    "files_found": 0,
                    "files": [],
                })
        except Exception as stage0_exc:
            logger.warning(f"  ⚠️  Stage 0 (label extraction) failed (non-fatal): {stage0_exc}")

        # =========================================================================

        # Fresh indexes for THIS run: the repository may have changed since the
        # last run (or since a previous generation phase wrote new files).
        try:
            self.invalidate_caches()
        except Exception as _inv_exc:
            logger.debug(f"Cache invalidation failed (non-fatal): {_inv_exc}")

        # AGENTIC SEARCH LOOP

        # =========================================================================

        # Instead of a fixed sequence (literals → symbols → filenames →

        # graph → RAG), the LLM decides per iteration which tool to call,

        # evaluates results, and either keeps searching, proceeds, or

        # asks the user for clarification.

        #

        # State:

        #   evidence    — file_path → {relevant, reason, decision, ...}

        #   all_items   — full EvidenceItem objects (returned to caller)

        #   tried       — [{tool, query, files_returned, iteration}]

        #   trace_log   — per-iteration debugging trace

        # =========================================================================

        evidence: Dict[str, dict] = {}

        all_items: List[EvidenceItem] = list(all_evidence)  # seed from Stage 0 (i18n chain)

        tried: List[dict] = []

        trace_log: List[dict] = []

        # Progressive inspection state
        evidence_knowledge = EvidenceKnowledge()
        ticket_context = f"{getattr(ticket, 'title', '')} {getattr(ticket, 'description', '')}"
        hyp_text_for_inspection = ""
        if hypotheses:
            hyp_text_for_inspection = " ".join(
                getattr(h, "hypothesis", str(h)) for h in hypotheses[:5]
            )


        # Pre-populate evidence for Stage 0 results via ledger merge.
        # i18n results are REAL evidence with provenance — not provisional
        # placeholders.  Stage 0.6 verification can supersede (upgrade or
        # downgrade) their relevance through the evidence-weighted mechanism.
        _ticket_title = getattr(ticket, "title", "") or ""
        for item in all_evidence:
            self._merge_evidence(
                evidence=evidence,
                file_path=item.file_path,
                source="i18n",
                tool="i18n_chain",
                query=_ticket_title,
                iteration=0,
                verdict={
                    "relevant": True,
                    "decision": "include",
                    "reason": "Found via i18n chain tracing",
                    "semantic_relevance_score": item.relevance_score,
                },
            )


        # ── Stage 0.5: ground the real chain by FOLLOWING relationships ──────
        # Inspect the localized artifact(s) so their repository relationships
        # (template_of → .ts, imports/has_type → service) are available, then
        # follow them. This inspects connected providers (e.g. member.service.ts
        # → members()) BEFORE the agentic loop, so verified capability evidence
        # exists instead of leaving the loop to be swayed by supplementary hits.
        #
        # Breadth is RELEVANCE + INFORMATION-GAIN driven (not a fixed count):
        # inspect localized artifacts strongest-first while each still yields new
        # structure; stop when information gain is exhausted. A generous cap is a
        # pure runaway safety net, not the driver.
        _BOOTSTRAP_INSPECT_CAP = _AGENTIC_MAX_ITERATIONS * 4  # runaway guard only
        try:
            _ranked_localized = sorted(
                all_evidence,
                key=lambda e: getattr(e, "relevance_score", 0.0),
                reverse=True,
            )
            if _ranked_localized:
                logger.info(
                    "  [LOCALIZE] "
                    + ", ".join(Path(e.file_path).name for e in _ranked_localized[:8])
                    + (f" (+{len(_ranked_localized) - 8} more)" if len(_ranked_localized) > 8 else "")
                )
            _inspected_ct = 0
            _consecutive_zero_gain = 0
            for item in _ranked_localized:
                if _inspected_ct >= _BOOTSTRAP_INSPECT_CAP:
                    break
                _fp = item.file_path
                if evidence.get(_fp, {}).get("inspection"):
                    continue
                _insp = self._inspect_artifact(
                    _fp, ticket_context, evidence, hyp_text_for_inspection
                )
                _inspected_ct += 1
                _gain = bool(_insp.methods or _insp.facts or _insp.type_dependencies)
                if not _gain:
                    # Information-gain driven stop: several no-yield inspections in
                    # a row means the localized set is understood — stop early.
                    _consecutive_zero_gain += 1
                    if _consecutive_zero_gain >= 3:
                        break
                    continue
                _consecutive_zero_gain = 0
                logger.info(
                    f"  [INSPECT] {Path(_fp).name} "
                    f"({_insp.inspection_depth}, {len(_insp.methods)} method(s))"
                )
                evidence[_fp]["inspection"] = _insp
                evidence_knowledge.inspection_knowledge[_fp] = _insp
                _rels = self._get_followable_relationships(
                    _fp, context.visited_files, evidence
                )
                if _rels:
                    evidence[_fp]["followable"] = _rels
                    evidence_knowledge.relationships.extend(_rels)
            self._auto_follow_and_inspect(
                ticket, hypotheses, evidence, evidence_knowledge, context
            )
        except Exception as _s05_exc:
            logger.debug(f"  [Stage0.5] grounded follow skipped (non-fatal): {_s05_exc}")

        # =====================================================================
        # STAGE 0.6: VERIFY i18n/BOOTSTRAP FILES VIA SEMANTIC VERIFICATION
        # =====================================================================
        # Stage 0 seeded i18n-discovered files as "discovered" evidence.
        # Now that Stage 0.5 has inspected them and followed relationships,
        # verify each through semantic verification.  The verification verdict
        # is merged via the evidence-weighted ledger — a failed verification
        # can supersede the initial i18n discovery (downgrade to exclude).
        # =====================================================================
        try:
            _i18n_fps = [
                fp for fp, v in evidence.items()
                if "i18n" in v.get("sources", [])
                and v.get("status") == "discovered"
                and v.get("decision") == "include"
            ]
            if _i18n_fps:
                logger.info(
                    f"\n  ==============================================="
                    f"\n  STAGE 0.6: VERIFYING {len(_i18n_fps)} "
                    f"i18n/bootstrap files"
                    f"\n  ==============================================="
                )
                _emit({
                    "type": "stage_06_verify_start",
                    "message": f"Verifying {len(_i18n_fps)} i18n-discovered files",
                    "files": _i18n_fps[:10],
                })
                _s06_kept = 0
                _s06_removed = 0
                for _fp in _i18n_fps:
                    # Find the matching EvidenceItem
                    _match_item = next(
                        (e for e in all_evidence if e.file_path == _fp), None
                    )
                    if not _match_item:
                        continue
                    _verdict = self._verify_single_file(
                        ticket, _match_item, evidence
                    )
                    _decision = _verdict.get("decision", "exclude")
                    _reason = _verdict.get("reason", "")
                    _sem_score = _verdict.get(
                        "semantic_relevance_score",
                        _match_item.relevance_score,
                    )

                    if _decision == "include":
                        # Merge verification verdict — supersedes discovery
                        self._merge_evidence(
                            evidence=evidence,
                            file_path=_fp,
                            source="i18n_verification",
                            tool="semantic_verifier",
                            query="",
                            iteration=0,
                            verdict={
                                "relevant": True,
                                "decision": "include",
                                "reason": f"i18n chain + verified: {_reason}",
                                "semantic_relevance_score": _sem_score,
                            },
                        )
                        _s06_kept += 1
                        logger.info(
                            f"  [Stage0.6] VERIFIED {Path(_fp).name}  "
                            f"score={_sem_score:.2f}"
                        )
                    else:
                        # Verification failed — downgrade via ledger
                        self._merge_evidence(
                            evidence=evidence,
                            file_path=_fp,
                            source="i18n_verification",
                            tool="semantic_verifier",
                            query="",
                            iteration=0,
                            verdict={
                                "relevant": False,
                                "decision": "exclude",
                                "reason": f"i18n chain but FAILED verification: {_reason}",
                                "semantic_relevance_score": _sem_score,
                            },
                        )
                        _s06_removed += 1
                        logger.info(
                            f"  [Stage0.6] EXCLUDED {Path(_fp).name}  "
                            f"score={_sem_score:.2f} - {_reason[:80]}"
                        )
                        # Remove from all_items/all_evidence to prevent
                        # downstream consumers from using this file
                        all_items = [
                            e for e in all_items if e.file_path != _fp
                        ]
                        all_evidence = [
                            e for e in all_evidence if e.file_path != _fp
                        ]
                logger.info(
                    f"  [Stage0.6] COMPLETE: {_s06_kept} kept, "
                    f"{_s06_removed} excluded"
                )
                _emit({
                    "type": "stage_06_verify_complete",
                    "message": (
                        f"i18n verification: {_s06_kept} kept, "
                        f"{_s06_removed} excluded"
                    ),
                    "kept": _s06_kept,
                    "excluded": _s06_removed,
                })
        except Exception as _s06_exc:
            logger.warning(
                f"  [Stage0.6] Verification failed (non-fatal): {_s06_exc}"
            )

        # =====================================================================
        # STAGE 0.7: REMOVED — semantic_rag is now on-demand only
        # =====================================================================
        # Previously this stage ran mandatory semantic RAG discovery before the
        # agentic loop.  It used the same _query_semantic() infrastructure that
        # the agentic loop's semantic_rag tool uses, creating duplicate work.
        #
        # Architecture decision (2026-09-16):
        #   Tools are capabilities, not mandatory stages.
        #   _decide_next_action() chooses semantic_rag when the current
        #   investigation state has unresolved questions that semantic
        #   discovery can answer.  The TOOL_CATALOG descriptions guide
        #   the LLM to select semantic_rag for business-language or
        #   conceptual queries where exact identifiers are unknown.
        #
        # All 19 discovery capabilities remain available to the controller:
        #   ripgrep, filename_search, regex_search, sqlite_fts,
        #   symbol_lookup, ast_companion_bundle, neo4j_graph, ts_chain,
        #   java_chain, python_chain, css_chain, config_chain,
        #   layout_chain, semantic_rag, i18n_chain, git_history, call_sites, lsp,
#   cooccurrence
        # =====================================================================

        logger.info(

            f"  🤖 Starting agentic loop: {_AGENTIC_MAX_ITERATIONS} max iterations, "

            f"{len(all_evidence)} pre-seeded from Stage 0"

        )

        # Sufficiency guidance: search directions from the gating judge,
        # propagated into _decide_next_action() as context (Constraint 2).
        _sufficiency_guidance: Optional[List[str]] = None

        # Diminishing-returns guards (deep-path bounding): keep exploring only
        # while discovery is still paying.  _plateau_count tracks consecutive
        # sufficiency checks whose coverage improved < 0.05; _no_change_streak
        # tracks consecutive iterations that added no new artifacts/facts/
        # methods/relationships.  Either stall (with evidence in hand) ends
        # the loop in favor of planning instead of burning the full budget.
        _last_coverage: Optional[float] = None
        _plateau_count = 0
        _no_change_streak = 0

        for iteration in range(_AGENTIC_MAX_ITERATIONS):
            # ── Early stopping: if we already have sufficient relevant evidence and reached plateau ──
            n_rel = sum(1 for v in evidence.values() if v.get("relevant"))
            if iteration >= 2 and n_rel >= 2 and _no_change_streak >= 1:
                logger.info(f"  ⚡ Early stop: {n_rel} relevant files confirmed with no new findings — proceeding to planning")
                break

            # Refresh followable relationships across all current evidence
            all_followable = {}
            rel_counter = 0
            for fp, v in evidence.items():
                if v.get("relevant") and v.get("followable"):
                    for rel in v["followable"]:
                        rel_counter += 1
                        rel_id = f"R{rel_counter}"
                        all_followable[rel_id] = rel
            self._current_followable_relationships = all_followable

            # ── LLM decides next action ──────────────────────────────────
            action = self._decide_next_action(

                ticket=ticket,

                hypotheses=hypotheses,

                evidence=evidence,

                tried=tried,

                all_items=all_items,

                iteration=iteration,

                max_iterations=_AGENTIC_MAX_ITERATIONS,

                evidence_knowledge=evidence_knowledge,

                sufficiency_guidance=_sufficiency_guidance,

                recovery_context=recovery_context,

            )

            self._note_decision(action)

            action_type = action.get("type", "proceed")

            new_files: List[EvidenceItem] = []  # Populated by follow or search

            # Coverage sampled by whichever sufficiency check runs this turn
            # (verified-companion or standard); None when neither runs.
            _coverage_sample: Optional[float] = None

            iteration_entry = {

                "iteration": iteration + 1,

                "action_type": action_type,

                "tool": action.get("tool"),

                "query": action.get("query"),

                "reasoning": action.get("reasoning", ""),

                "degraded": bool(action.get("degraded", False)),

                "files_returned": [],

                "verdicts": [],

            }

            # ── Callback: iteration_start ──
            _emit({
                "type": "iteration_start",
                "iteration": iteration + 1,
                "max_iterations": _AGENTIC_MAX_ITERATIONS,
                "action_type": action_type,
                "tool": action.get("tool", ""),
                "query": action.get("query", ""),
                "reasoning": action.get("reasoning", "")[:200],
                "message": f"Iteration {iteration + 1}/{_AGENTIC_MAX_ITERATIONS}: "
                           f"{action_type} — {action.get('tool', '')} '{action.get('query', '')[:120]}'",

            })

            # ── Guard against phantom early-stop with 0 relevant files ──
            if action_type == "proceed":
                n_relevant = sum(1 for v in evidence.values() if v.get("relevant"))
                if n_relevant == 0:
                    logger.warning(
                        f"  ⚠️ Agentic loop attempted 'proceed' at iteration {iteration + 1} with 0 relevant files. "
                        "Rejecting early-stop and overriding action to search."
                    )
                    action_type = "search"
                    t_title = getattr(ticket, "title", "") or ""
                    t_desc = getattr(ticket, "description", "") or ""
                    combined = f"{t_title} {t_desc}"
                    
                    # Universal Market AI IDE Fallback: LLM Repo Tree Reasoning
                    tree_candidates = self._discover_candidates_via_repo_tree(ticket)
                    if tree_candidates:
                        logger.info(f"  🎯 Repo Tree reasoning localized {len(tree_candidates)} candidate files: {tree_candidates[:4]}")
                        action_type = "search"
                        first_cand = tree_candidates[0]
                        action = {
                            "type": "search",
                            "tool": "filename_search",
                            "query": Path(first_cand).name,
                            "reasoning": f"Located candidate file via LLM repository tree reasoning: {first_cand}"
                        }
                        # Seed all tree candidates and their related companion/test files
                        related_files = self._discover_related_files(tree_candidates)
                        all_found = list(tree_candidates) + related_files
                        for cand in all_found:
                            if cand not in evidence:
                                cand_item = EvidenceItem(
                                    file_path=cand,
                                    source="repo_tree_reasoning",
                                    relevance_score=0.88 if cand in tree_candidates else 0.78,
                                    content_snippet=f"Discovered via repository architecture tree reasoning for {t_title}"
                                )
                                all_evidence.append(cand_item)
                                all_items.append(cand_item)
                                evidence[cand] = {
                                    "relevant": True,
                                    "decision": "include",
                                    "reason": "Discovered via repository architecture tree reasoning",
                                    "semantic_relevance_score": 0.88 if cand in tree_candidates else 0.78,
                                    "sources": ["repo_tree_reasoning"],
                                }
                    else:
                        clean_t = re.sub(r"^(?:Repository|Repo|Ticket|Title|Description):\s*", "", t_title, flags=re.MULTILINE | re.IGNORECASE).strip()
                        clean_t = clean_t.split("\n")[0].strip()
                        query_term = clean_t[:30] or "service"
                        action = {
                            "type": "search",
                            "tool": "ripgrep",
                            "query": query_term,
                            "reasoning": f"Fallback search forced with title query '{query_term}' because 0 relevant files were found."
                        }
                    iteration_entry["action_type"] = action_type
                    iteration_entry["tool"] = action["tool"]
                    iteration_entry["query"] = action["query"]

            # ── PROCEED — evidence is sufficient ─────────────────────────
            if action_type == "proceed":
                n_relevant = sum(1 for v in evidence.values() if v.get("relevant"))
                logger.info(
                    f"  ✅ Agentic loop → proceed at iteration {iteration + 1}: "
                    f"{action.get('reasoning', '')[:100]}"
                )
                # ── Callback: early_stop ──
                _emit({
                    "type": "early_stop",
                    "iteration": iteration + 1,
                    "relevant_files": n_relevant,
                    "reasoning": action.get("reasoning", "")[:200],
                    "message": f"✅ Evidence sufficient at iteration {iteration + 1} — "
                               f"{n_relevant} relevant files found",
                })

                trace_log.append(iteration_entry)
                break

            # ── ASK_USER — can't find files with confidence ──────────────

            if action_type == "ask_user":

                question = action.get("question", "Could not determine which files need changes.")

                logger.info(

                    f"  ❓ Agentic loop → ask_user at iteration {iteration + 1}: "

                    f"{question[:100]}"

                )

                trace_log.append(iteration_entry)

                # Write trace before returning

                trace_dir = str(get_trace_dir(ticket.ticket_id))

                os.makedirs(trace_dir, exist_ok=True)

                try:

                    with open(os.path.join(trace_dir, "agentic_trace.json"), "w") as f:

                        json.dump(trace_log, f, indent=2)

                except Exception:

                    pass

                return self._build_need_more_info_return(

                    question, evidence, all_items, tried

                )

            # ── FOLLOW — deterministic relationship traversal ─────────

            if action_type == "follow":

                rel_id = action.get("relationship_id", "")

                # Look up the relationship from our stored followable relationships
                followable = self._current_followable_relationships or {}
                rel_info = followable.get(rel_id)

                if not rel_info:
                    logger.warning(f"[AgenticLoop] Unknown relationship_id: {rel_id}")
                    trace_log.append(iteration_entry)
                    continue

                target_file = rel_info.get("target_file")
                if not target_file:
                    # Try to resolve symbol to file
                    target_file = self._resolve_edge_to_file(
                        rel_info.get("kind", ""), rel_info.get("dst_name", "")
                    )

                if not target_file:
                    logger.warning(
                        f"[AgenticLoop] Could not resolve relationship {rel_id}: "
                        f"{rel_info.get('kind')} → {rel_info.get('dst_name')}"
                    )
                    tried.append({
                        "tool": "follow",
                        "query": f"{rel_info.get('kind')} → {rel_info.get('dst_name')}",
                        "files_returned": [],
                        "iteration": iteration + 1,
                        "status": f"UNRESOLVABLE: Could not resolve {rel_info.get('dst_name')} to a file.",
                    })
                    trace_log.append(iteration_entry)
                    continue

                logger.info(
                    f"  🔗 FOLLOW [{rel_id}]: {rel_info.get('source', '?')} "
                    f"--{rel_info.get('kind', '?')}--> {target_file}"
                )

                _emit({
                    "type": "follow",
                    "iteration": iteration + 1,
                    "relationship_id": rel_id,
                    "source": rel_info.get("source", ""),
                    "kind": rel_info.get("kind", ""),
                    "target": target_file,
                    "symbol": rel_info.get("dst_name", ""),
                    "message": f"🔗 FOLLOW [{rel_id}] {rel_info.get('kind', '')} → "
                               f"{Path(target_file).name}",
                })

                # Create an EvidenceItem for the resolved target
                new_files = [EvidenceItem(
                    provider="relationship", strength="strong",
                    details=f"follow_{rel_info.get('kind', 'edge')}",
                    file_path=target_file,
                    evidence_type="relationship",
                    content_snippet=f"[follow:{rel_id}] {rel_info.get('kind')} from {rel_info.get('source', '?')}",
                    relationship=rel_info.get("kind", "unknown"),
                    relevance_score=0.9,
                    hypothesis_id="follow",
                )]

                tried.append({
                    "tool": "follow",
                    "query": f"{rel_info.get('kind')} → {rel_info.get('dst_name')}",
                    "files_returned": [target_file],
                    "iteration": iteration + 1,
                })

                iteration_entry["files_returned"] = [target_file]

            # ── SEARCH — run the chosen tool ─────────────────────────────

            elif action_type == "search":

                tool_name = action.get("tool", "")

                query = action.get("query", "")

                if not tool_name or not query:

                    logger.warning(

                        f"[AgenticLoop] Search action missing tool/query: {action}"

                    )

                    trace_log.append(iteration_entry)

                    continue

                # ── Repeated search interception ──
                is_repeat = any(
                    t["tool"] == tool_name and t["query"] == query
                    for t in tried
                )
                if is_repeat:
                    logger.info(
                        f"  ⛔ THIS SEARCH WAS ALREADY EXECUTED: [{tool_name}] {query!r}"
                    )
                    tried.append({
                        "tool": tool_name,
                        "query": query,
                        "files_returned": [],
                        "iteration": iteration + 1,
                        "status": "THIS SEARCH WAS ALREADY EXECUTED. "
                                  "Choose follow, a different search, proceed, or ask_user.",
                    })
                    _emit({
                        "type": "search_rejected",
                        "iteration": iteration + 1,
                        "tool": tool_name,
                        "query": query,
                        "message": f"⛔ THIS SEARCH WAS ALREADY EXECUTED: [{tool_name}] '{query[:120]}'",
                    })
                    trace_log.append(iteration_entry)
                    continue

                # ── Speculative parallel dispatch (P2 latency) ────────────
                # On the first turn, run two complementary read-only probes
                # (filename + symbol) CONCURRENTLY with the LLM-chosen tool
                # so turn 0 returns with a broader evidence base at the same
                # wall-clock cost.  Probes never replace the chosen tool and
                # failures are swallowed.
                _spec_extra: list = []
                if iteration == 0 and self.repo_search:
                    try:
                        from concurrent.futures import ThreadPoolExecutor

                        def _probe(tool_t: str) -> list:
                            try:
                                return self._execute_tool(tool_t, query, context)
                            except Exception:
                                return []

                        with ThreadPoolExecutor(max_workers=2) as _pool:
                            _f1 = _pool.submit(_probe, "filename_search")
                            _f2 = _pool.submit(_probe, "symbol_lookup")
                            _spec_extra = list(_f1.result()) + list(_f2.result())
                        if _spec_extra:
                            logger.info(
                                f"  ⚡ Speculative turn-0 probes: +{len(_spec_extra)} "
                                "candidate item(s) at zero extra wall-clock cost"
                            )
                    except Exception as _spec_exc:
                        logger.debug(f"speculative dispatch skipped (non-fatal): {_spec_exc}")

                new_files = self._execute_tool(tool_name, query, context)
                if _spec_extra:
                    new_files = _spec_extra + new_files

                # ── Visited-result ambiguity ──
                # Count how many results were filtered by visited_files
                all_tool_results_count = len(new_files)
                if all_tool_results_count == 0 and tool_name in (
                    "symbol_lookup", "ripgrep", "filename_search",
                    "sqlite_fts", "regex_search",
                ):
                    # Check if the tool would have found files that are already visited
                    # by temporarily re-running without the visited filter
                    try:
                        pre_visited_count = self._count_pre_visited_results(
                            tool_name, query, context
                        )
                    except Exception:
                        pre_visited_count = 0

                    if pre_visited_count > 0:
                        status_msg = (
                            f"FOUND {pre_visited_count} files, ALL ALREADY VISITED. "
                            "Consider using 'follow' on discovered relationships instead."
                        )
                    else:
                        status_msg = None
                else:
                    status_msg = None

                tried.append({

                    "tool": tool_name,

                    "query": query,

                    "files_returned": [f.file_path for f in new_files],

                    "iteration": iteration + 1,

                    **(({"status": status_msg}) if status_msg else {}),

                })


                iteration_entry["files_returned"] = [f.file_path for f in new_files]

                # ── Callback: search_result ──
                _emit({
                    "type": "search_result",
                    "iteration": iteration + 1,
                    "tool": tool_name,
                    "query": query,
                    "files": [f.file_path for f in new_files[:15]],
                    "count": len(new_files),
                    "status": status_msg or "",
                    "message": f"🔍 {tool_name}('{query[:120]}') → {len(new_files)} files"
                               + (f" ({status_msg})" if status_msg else ""),
                })

            # ── Verify each new file (common for both FOLLOW and SEARCH) ──

            for file_item in new_files:

                all_items.append(file_item)

                if file_item.file_path not in evidence:

                    verdict = self._verify_single_file(

                        ticket, file_item, evidence

                    )

                    # Determine the source label from the action type
                    _source_label = (
                        "follow" if action_type == "follow"
                        else action.get("tool", "search")
                    )

                    # ── Merge verdict into ledger ──
                    self._merge_evidence(
                        evidence=evidence,
                        file_path=file_item.file_path,
                        source=_source_label,
                        tool=action.get("tool", "follow"),
                        query=action.get("query", ""),
                        iteration=iteration + 1,
                        verdict=verdict,
                    )

                    iteration_entry["verdicts"].append({

                        "file": file_item.file_path,

                        "decision": verdict.get("decision", "unknown"),

                        "reason": verdict.get("reason", "")[:100],

                    })

                    # ── Progressive Artifact Inspection ──
                    if verdict.get("decision") == "include":
                        if self._should_inspect(
                            file_item.file_path, verdict,
                            evidence, ticket_context,
                            evidence_knowledge,
                        ):
                            inspection = self._inspect_artifact(
                                file_item.file_path,
                                ticket_context,
                                evidence,
                                hypotheses_text=hyp_text_for_inspection,
                            )

                            # Read actual source code for relevant methods
                            for rm in inspection.relevant_methods[:3]:
                                start_l = rm.line_start
                                if start_l:
                                    end_l = rm.line_end if (rm.line_end and rm.line_end > start_l) else (start_l + 40)
                                    code = self._read_relevant_region(
                                        file_item.file_path,
                                        start_l, end_l,
                                        max_lines=80,
                                    )
                                    if code:
                                        inspection.relevant_code_regions[rm.name] = {
                                            "code": code,
                                            "start": start_l,
                                            "end": end_l,
                                            "file": file_item.file_path,
                                        }
                                        inspection.inspection_depth = "region"

                            # ── Extract followable relationships ──
                            followable_rels = self._get_followable_relationships(
                                file_item.file_path, context.visited_files, evidence
                            )
                            if followable_rels:
                                logger.info(
                                    f"  🔗 {len(followable_rels)} followable relationships "
                                    f"from {Path(file_item.file_path).name}"
                                )

                            # ── Merge inspection + relationships into ledger ──
                            # This single call handles:
                            #   - evidence[fp]["inspection"] = inspection
                            #   - evidence[fp]["followable"] = followable_rels
                            #   - EvidenceKnowledge aggregate updates
                            self._merge_evidence(
                                evidence=evidence,
                                file_path=file_item.file_path,
                                source=_source_label,
                                tool="inspector",
                                query="",
                                iteration=iteration + 1,
                                inspection=inspection,
                                relationships=followable_rels if followable_rels else None,
                                evidence_knowledge=evidence_knowledge,
                            )

                            # Code facts (requires post-merge access to regions)
                            for m_name, r_info in inspection.relevant_code_regions.items():
                                evidence_knowledge.code_facts.append({
                                    "method": m_name,
                                    "file": file_item.file_path,
                                    "code": r_info.get("code"),
                                    "start": r_info.get("start"),
                                    "end": r_info.get("end"),
                                })

                            # Check if previously-raised questions are now resolved
                            for q in list(evidence_knowledge.all_questions):
                                for fact in inspection.facts:
                                    if any(
                                        word in fact.fact.lower()
                                        for word in q.lower().split()
                                        if len(word) > 3
                                    ):
                                        if q not in evidence_knowledge.resolved_questions:
                                            evidence_knowledge.resolved_questions.append(q)
                                            break

                            _emit({
                                "type": "inspection",
                                "iteration": iteration + 1,
                                "file": file_item.file_path,
                                "class_name": inspection.class_name,
                                "methods_found": len(inspection.methods),
                                "relevant_methods": len(inspection.relevant_methods),
                                "facts": len(inspection.facts),
                                "questions": len(inspection.questions),
                                "providers": inspection.providers_used,
                                "message": f"🔬 Inspected {Path(file_item.file_path).name}: "
                                           f"{len(inspection.methods)} methods, "
                                           f"{len(inspection.relevant_methods)} relevant, "
                                           f"{len(inspection.facts)} facts",
                            })

                    # ── Callback: verdict ──
                    _emit({
                        "type": "verdict",
                        "iteration": iteration + 1,
                        "file": file_item.file_path,
                        "decision": verdict.get("decision", "unknown"),
                        "reason": verdict.get("reason", "")[:150],
                        "message": f"{'✅' if verdict.get('relevant') else '❌'} "
                                   f"{Path(file_item.file_path).name} — "
                                   f"{verdict.get('reason', '')[:80]}",
                    })

                    all_evidence.append(file_item)


            # ── Track Information Gain for this Iteration ──
            iter_new_artifacts = sum(
                1 for fi in new_files
                if evidence.get(fi.file_path, {}).get("inspection")
            )
            iter_new_facts = sum(
                len(evidence[fi.file_path]["inspection"].facts)
                for fi in new_files
                if evidence.get(fi.file_path, {}).get("inspection")
            )
            iter_new_questions = sum(
                len(evidence[fi.file_path]["inspection"].questions)
                for fi in new_files
                if evidence.get(fi.file_path, {}).get("inspection")
            )
            iter_new_methods = sum(
                len(evidence[fi.file_path]["inspection"].methods)
                for fi in new_files
                if evidence.get(fi.file_path, {}).get("inspection")
            )
            iter_new_rels = sum(
                len(evidence[fi.file_path].get("followable", []))
                for fi in new_files
                if fi.file_path in evidence
            )
            evidence_knowledge.record_iteration(
                new_artifacts=iter_new_artifacts,
                new_facts=iter_new_facts,
                new_questions=iter_new_questions,
                resolved=len(evidence_knowledge.resolved_questions),
                new_methods=iter_new_methods,
                new_relationships=iter_new_rels,
            )
            evidence_knowledge.verified_artifacts = [
                fp for fp, v in evidence.items() if v.get("relevant")
            ]

            trace_log.append(iteration_entry)

            # ── Verified Companion Short-Circuit (Option B: Safe Verified Path) ──
            # A proven controller+template companion pair is a primary anchor match.
            # To guarantee 100% safety against missing components or cross-module logic,
            # we evaluate sufficiency to verify that ALL functional requirements are
            # satisfied (is_sufficient=True and coverage_score >= 0.80) before stopping.
            _ran_sufficiency = False
            if requirements is not None and evidence:
                try:
                    _pair_proven, _pair_why = self._companion_pair_proven(ticket, evidence, requirements=requirements)
                    if _pair_proven:
                        logger.info(
                            f"  🔍 Companion pair candidate proven ({_pair_why}) — verifying full requirement coverage..."
                        )
                        sufficiency = self.check_sufficiency(
                            ticket=ticket,
                            requirements=requirements,
                            evidence_items=all_items,
                            hypotheses=hypotheses or [],
                            iteration=iteration,
                        )
                        _ran_sufficiency = True
                        _coverage_sample = float(sufficiency.coverage_score)
                        _is_verified_sufficient = bool(sufficiency.is_sufficient and sufficiency.coverage_score >= 0.80)
                        _emit({
                            "type": "sufficiency_check",
                            "iteration": iteration + 1,
                            "is_sufficient": _is_verified_sufficient,
                            "confidence": sufficiency.confidence_score,
                            "coverage": sufficiency.coverage_score,
                            "message": (
                                f"{'✅' if _is_verified_sufficient else '🔄'} "
                                f"Verified companion check at iteration {iteration + 1}: "
                                f"{'SUFFICIENT' if _is_verified_sufficient else 'INSUFFICIENT'} "
                                f"(coverage={sufficiency.coverage_score:.2f}, confidence={sufficiency.confidence_score:.2f})"
                            ),
                        })
                        if _is_verified_sufficient:
                            logger.info(
                                f"  ⚡ Verified short-circuit at iteration {iteration + 1}: "
                                f"{_pair_why} (coverage={sufficiency.coverage_score:.2f}, confidence={sufficiency.confidence_score:.2f})"
                            )
                            _emit({
                                "type": "deterministic_short_circuit",
                                "iteration": iteration + 1,
                                "reason": f"{_pair_why} (coverage={sufficiency.coverage_score:.2f})",
                            })
                            for pt in sufficiency.reasoning_points[:3]:
                                logger.info(f"    • {pt}")
                            break
                        else:
                            logger.info(
                                f"  🔄 Companion pair proven, but requirement coverage ({sufficiency.coverage_score:.2f} < 0.80) "
                                f"indicates additional components or services are required — continuing discovery"
                            )
                            if sufficiency.new_search_directions:
                                _sufficiency_guidance = sufficiency.new_search_directions[:3]
                                for direction in _sufficiency_guidance:
                                    logger.info(f"    → Sufficiency guidance: {direction}")
                            else:
                                _sufficiency_guidance = None
                except Exception as _sc_exc:
                    logger.debug(f"companion short-circuit check failed (non-fatal): {_sc_exc}")

            # ── Standard Sufficiency Evaluation ───────────────────────
            # Runs if the verified companion check did not already execute this turn.
            if (
                not _ran_sufficiency
                and iteration >= 0
                and evidence_knowledge.had_material_change()
                and requirements is not None
            ):
                try:
                    sufficiency = self.check_sufficiency(
                        ticket=ticket,
                        requirements=requirements,
                        evidence_items=all_items,
                        hypotheses=hypotheses or [],
                        iteration=iteration,
                    )
                    _coverage_sample = float(sufficiency.coverage_score)
                    _emit({
                        "type": "sufficiency_check",
                        "iteration": iteration + 1,
                        "is_sufficient": sufficiency.is_sufficient,
                        "confidence": sufficiency.confidence_score,
                        "coverage": sufficiency.coverage_score,
                        "message": (
                            f"{'✅' if sufficiency.is_sufficient else '🔄'} "
                            f"Sufficiency at iteration {iteration + 1}: "
                            f"{'SUFFICIENT' if sufficiency.is_sufficient else 'INSUFFICIENT'} "
                            f"(confidence={sufficiency.confidence_score:.2f}, "
                            f"coverage={sufficiency.coverage_score:.2f})"
                        ),
                    })
                    if sufficiency.is_sufficient and sufficiency.coverage_score >= 0.80:
                        logger.info(
                            f"  ✅ Evidence SUFFICIENT at iteration {iteration + 1} "
                            f"(confidence={sufficiency.confidence_score:.2f}, coverage={sufficiency.coverage_score:.2f})"
                        )
                        for pt in sufficiency.reasoning_points[:3]:
                            logger.info(f"    • {pt}")
                        break  # Exit loop — evidence is enough for planning
                    else:
                        logger.info(
                            f"  🔄 Evidence insufficient at iteration {iteration + 1} "
                            f"— continuing ({sufficiency.coverage_score:.2f} coverage)"
                        )
                        # Propagate search directions as CONTEXT for _decide_next_action
                        # (stored in _sufficiency_guidance, consumed next iteration)
                        if sufficiency.new_search_directions:
                            _sufficiency_guidance = sufficiency.new_search_directions[:3]
                            for direction in _sufficiency_guidance:
                                logger.info(f"    → Sufficiency guidance: {direction}")
                        else:
                            _sufficiency_guidance = None
                except Exception as suf_exc:
                    logger.warning(f"  Sufficiency check failed (non-fatal): {suf_exc}")

            # ── Diminishing-returns guards (deep-path bounding) ───────
            # Update stall counters, then stop when discovery stops paying.
            # This is what prevents another 24-call wander: once coverage
            # plateaus (or iterations add nothing new) and relevant files
            # are already in hand, planning + preflight downstream validate
            # the targets — more exploratory turns only burn tokens.
            if iteration > 0:
                if evidence_knowledge.had_material_change():
                    _no_change_streak = 0
                else:
                    _no_change_streak += 1

            if _coverage_sample is not None:
                if _last_coverage is not None and (_coverage_sample - _last_coverage) < 0.05:
                    _plateau_count += 1
                else:
                    _plateau_count = 0
                _last_coverage = _coverage_sample

            _n_relevant_files = sum(1 for v in evidence.values() if v.get("relevant"))
            _stop_reason = self._diminishing_returns_stop_reason(
                plateau_count=_plateau_count,
                no_change_streak=_no_change_streak,
                n_relevant_files=_n_relevant_files,
            )
            if _stop_reason:
                logger.info(
                    f"  🛑 Diminishing returns at iteration {iteration + 1} "
                    f"({_stop_reason}) — proceeding with best available evidence "
                    f"({_n_relevant_files} relevant files)"
                )
                _emit({
                    "type": "diminishing_returns_stop",
                    "iteration": iteration + 1,
                    "reason": _stop_reason,
                    "relevant_files": _n_relevant_files,
                    "message": (
                        f"🛑 Discovery stopped: {_stop_reason}. "
                        f"Proceeding with {_n_relevant_files} relevant files."
                    ),
                })
                break

            # Log iteration summary

            n_relevant = sum(1 for v in evidence.values() if v.get("relevant"))

            n_excluded = sum(1 for v in evidence.values() if not v.get("relevant"))

            logger.info(

                f"  ðŸ“Š After iteration {iteration + 1}: "

                f"{n_relevant} relevant, {n_excluded} excluded, "

                f"{len(tried)} searches tried"

            )

        else:

            # Budget exhausted.  Only block on the user when discovery found
            # NOTHING relevant; otherwise proceed to planning with the best
            # available evidence — downstream preflight and semantic
            # verification validate the targets before any code is written.

            _n_relevant_files = sum(1 for v in evidence.values() if v.get("relevant"))

            if _n_relevant_files == 0:

                logger.warning(

                    f"  ⚠️ Agentic loop exhausted {_AGENTIC_MAX_ITERATIONS} iterations "

                    f"without confident evidence — routing to ask_user"

                )

                # Write trace

                trace_dir = str(get_trace_dir(ticket.ticket_id))

                os.makedirs(trace_dir, exist_ok=True)

                try:

                    with open(os.path.join(trace_dir, "agentic_trace.json"), "w") as f:

                        json.dump(trace_log, f, indent=2)

                except Exception:

                    pass

                return self._build_need_more_info_return(

                    "Could not identify the relevant source files with confidence "

                    "after exhausting the search budget. Please provide more specific "

                    "details about which component, page, or feature is affected.",

                    evidence, all_items, tried,

                )

            logger.warning(

                f"  ⚠️ Budget exhausted after {_AGENTIC_MAX_ITERATIONS} iterations — "
                f"proceeding to planning with best available evidence "
                f"({_n_relevant_files} relevant files)"

            )

            _emit({

                "type": "budget_exhausted_proceed",

                "iteration": _AGENTIC_MAX_ITERATIONS,

                "relevant_files": _n_relevant_files,

                "message": (

                    f"⚠️ Search budget exhausted — proceeding with best available "

                    f"evidence ({_n_relevant_files} relevant files); downstream "
                    f"preflight will verify the targets"

                ),

            })

            # Fall through to post-loop processing (trace + promotion + return)

        # ── Write agentic trace ──────────────────────────────────────────

        trace_dir = str(get_trace_dir(ticket.ticket_id))

        os.makedirs(trace_dir, exist_ok=True)

        try:

            with open(os.path.join(trace_dir, "agentic_trace.json"), "w") as f:

                json.dump(trace_log, f, indent=2)

        except Exception:

            pass



        self._last_code_facts = all_code_facts

        logger.info(

            f"  Evidence collection done: {len(all_evidence)} items, "

            f"{sum(1 for v in evidence.values() if v.get('relevant'))} verified relevant"

        )

        # ── B6: Evidence Promotion Validation ────────────────────────────

        # Filter raw evidence to only items that genuinely support at least

        # one hypothesis or have strong multi-source backing.

        validated_evidence = self._validate_evidence_promotion(

            all_evidence, hypotheses or []

        )

        if len(validated_evidence) < len(all_evidence):

            logger.info(

                "  B6 promotion filter: kept %d / %d evidence items "

                "(removed %d low-quality items)",

                len(validated_evidence), len(all_evidence),

                len(all_evidence) - len(validated_evidence),

            )

        # Recompute confidence on validated set

        if validated_evidence:

            confidence = self._compute_confidence(

                validated_evidence, hypotheses or [], localized_tasks or [],

                evidence_ledger=evidence, evidence_knowledge=evidence_knowledge

            )

        else:

            confidence = 0.0

        # Store the per-file verification verdicts so the workflow can

        # populate semantic_verification_results in the expected batch shape

        # (list of {file_path, decision, semantic_relevance_score, reason})

        # without re-running the old batch verifier.

        self._agentic_verdicts = evidence

        # ── Phase 5b: Deterministic relationship following ────────────────
        # Follow relevant relationships to inspect provider artifacts (e.g.
        # component → service.members()) so behavioral understanding is based on
        # ACTUAL provider behavior, not just a recorded edge. Bounded + reuses
        # the existing _inspect_artifact / _should_inspect / followable edges.
        try:
            self._auto_follow_and_inspect(ticket, hypotheses, evidence, evidence_knowledge, context)
        except Exception as _af_exc:
            logger.debug(f"  [AutoFollow] skipped (non-fatal): {_af_exc}")

        # ── Phase 5: Store progressive inspection results for planner ────
        # These live as instance attributes so workflow.py can extract them
        # after collect() returns and pass them to the planning agent.
        self._artifact_inspections = {
            fp: ev["inspection"]
            for fp, ev in evidence.items()
            if ev.get("inspection")
        }
        self._evidence_knowledge = evidence_knowledge

        # Degradation surfaced as instance attributes (the return tuple's arity is
        # consumed by workflow.py and must not change).
        self.last_collect_flags = {
            "evidence_decision_degraded": self.decision_degraded,
            "reasons": list(self.decision_degraded_reasons),
        }

        return validated_evidence, confidence, None

    # =========================================================================

    # AGENTIC LOOP — CORE METHODS

    # =========================================================================

    # ── Evidence Ledger: single write path ──────────────────────────────────
    # Every discovery source (i18n, RAG, search, follow, inspect) merges into
    # the same ledger through this method.  Evidence is ADDITIVE — provenance
    # is always appended, never replaced.  Conclusions are EVIDENCE-WEIGHTED:
    # a later verified inspection can supersede an earlier high-score discovery.
    #
    # Lifecycle statuses:
    #   discovered → verified → inspected → connected → understood
    # ────────────────────────────────────────────────────────────────────────

    def _merge_evidence(
        self,
        evidence: Dict[str, dict],
        file_path: str,
        source: str,
        tool: str,
        query: str,
        iteration: int,
        verdict: Optional[dict] = None,
        inspection: Optional["ArtifactInspection"] = None,
        relationships: Optional[List[dict]] = None,
        evidence_knowledge: Optional["EvidenceKnowledge"] = None,
    ) -> dict:
        """Merge a discovery into the Evidence Ledger.

        Core invariants:
          1. Evidence is ADDITIVE — a new source for an existing file appends
             to the ``sources`` list, never replaces.
          2. Conclusions are EVIDENCE-WEIGHTED — a later verified inspection
             can supersede an earlier discovery verdict when the evidence
             tier is higher (discovery < verification < inspection).
          3. A zero-result search NEVER erases accumulated evidence.

        Returns the updated evidence entry.
        """
        _STATUS_RANK = {
            "discovered": 0,
            "verified": 1,
            "inspected": 2,
            "connected": 3,
            "understood": 4,
        }

        existing = evidence.get(file_path)

        if not existing:
            # ── New file — create ledger entry ──
            entry: Dict[str, Any] = {
                "file_path": file_path,
                "sources": [source],
                "search_provenance": [
                    {"tool": tool, "query": query, "iteration": iteration}
                ],
                "relevant": True,           # included until proven otherwise
                "decision": "include",
                "reason": "",
                "semantic_relevance_score": 0.5,
                "status": "discovered",
                "relationships": [],
                "unresolved_references": [],
            }
        else:
            entry = existing
            # ── Merge source (additive, never replaced) ──
            if source and source not in entry.get("sources", []):
                entry.setdefault("sources", []).append(source)
            # ── Merge provenance (additive) ──
            entry.setdefault("search_provenance", []).append(
                {"tool": tool, "query": query, "iteration": iteration}
            )

        # ── Apply verdict if provided ──
        # Evidence-weighted: a higher-tier conclusion can supersede a lower-tier
        # one.  Discovery < Verification < Inspection.
        if verdict:
            new_status = "verified"
            old_status = entry.get("status", "discovered")
            old_rank = _STATUS_RANK.get(old_status, 0)
            new_rank = _STATUS_RANK.get(new_status, 1)

            # Supersede when:
            #   a) Entry is new (no previous verdict)
            #   b) The new conclusion comes from a higher evidence tier
            #   c) Entry was previously only at "discovered" (auto-include)
            if (not existing
                    or new_rank >= old_rank
                    or old_status == "discovered"):
                entry["relevant"] = verdict.get("relevant", True)
                entry["decision"] = verdict.get("decision", "include")
                entry["reason"] = verdict.get("reason", "")
                entry["semantic_relevance_score"] = verdict.get(
                    "semantic_relevance_score",
                    entry.get("semantic_relevance_score", 0.5),
                )
                entry["status"] = new_status

        # ── Apply inspection if provided ──
        if inspection:
            entry["inspection"] = inspection
            # Inspection is a higher tier than verification
            if _STATUS_RANK.get(entry.get("status", ""), 0) < _STATUS_RANK["inspected"]:
                entry["status"] = "inspected"

            # Update EvidenceKnowledge aggregate if available
            if evidence_knowledge is not None:
                if file_path not in evidence_knowledge.verified_artifacts:
                    evidence_knowledge.verified_artifacts.append(file_path)
                evidence_knowledge.structural_facts.extend(inspection.facts)
                evidence_knowledge.all_facts.extend(inspection.facts)
                evidence_knowledge.all_questions.extend(inspection.questions)
                evidence_knowledge.all_unresolved.extend(inspection.unresolved)
                evidence_knowledge.inspection_knowledge[file_path] = inspection
                for c in inspection.api_contracts:
                    if c not in evidence_knowledge.api_contracts:
                        evidence_knowledge.api_contracts.append(c)

        # ── Merge relationships (additive, deduplicate by target+kind) ──
        if relationships:
            existing_keys = {
                (r.get("target", "") or r.get("dst_name", ""))
                + "|" + r.get("kind", "")
                for r in entry.get("relationships", [])
            }
            for rel in relationships:
                key = (
                    (rel.get("target", "") or rel.get("dst_name", ""))
                    + "|" + rel.get("kind", "")
                )
                if key not in existing_keys:
                    entry.setdefault("relationships", []).append(rel)
                    existing_keys.add(key)

            # Also populate "followable" for legacy compatibility
            entry["followable"] = entry.get("followable", []) + relationships

            # Update EvidenceKnowledge if available
            if evidence_knowledge is not None:
                evidence_knowledge.relationships.extend(relationships)

        evidence[file_path] = entry
        return entry

    # ── Build holistic investigation state from ledger ──────────────────────
    # This replaces the scattered evidence_summary construction with a
    # structured view of the investigation as a puzzle: what pieces we have,
    # what's missing, and what relationships are unexplored.
    # ────────────────────────────────────────────────────────────────────────

    def _build_investigation_state(
        self,
        evidence: Dict[str, dict],
        hypotheses: Optional[List[Any]],
        evidence_knowledge: Optional["EvidenceKnowledge"],
        tried: Optional[List[dict]] = None,
    ) -> str:
        """Build a holistic investigation state from the Evidence Ledger.

        Returns a structured text block for the controller prompt that shows:
        1. VERIFIED ARTIFACTS — what we know, with status and provenance
        2. UNRESOLVED REFERENCES — what we suspect but haven't confirmed
        3. HYPOTHESIS COVERAGE — which hypotheses have evidence vs which are blind
        4. UNEXPLORED RELATIONSHIPS — discovered edges not yet followed
        5. OPEN QUESTIONS — from inspections that remain unanswered
        """
        sections = []

        # ── 1. Verified Artifacts ──
        included = [
            (fp, v) for fp, v in evidence.items()
            if v.get("relevant") and v.get("decision") == "include"
        ]
        excluded = [
            (fp, v) for fp, v in evidence.items()
            if not v.get("relevant") or v.get("decision") == "exclude"
        ]

        if included:
            sections.append("VERIFIED ARTIFACTS (confirmed relevant):")
            for fp, v in included:
                status = v.get("status", "discovered")
                sources = ", ".join(v.get("sources", ["unknown"]))
                reason = v.get("reason", "")[:100]
                score = v.get("semantic_relevance_score", 0)
                sections.append(
                    f"  ✅ [{status}] {fp}\n"
                    f"     sources: {sources} | score: {score:.2f}\n"
                    f"     reason: {reason}"
                )
                # Include inspection summary if available
                insp = v.get("inspection")
                if insp and hasattr(insp, "to_evidence_block"):
                    evidence_block = insp.to_evidence_block(file_path=fp)
                    if evidence_block:
                        sections.append(evidence_block)

        if excluded:
            sections.append("\nEXCLUDED FILES (checked and found NOT relevant):")
            for fp, v in excluded[:8]:
                reason = v.get("reason", "")[:100]
                sections.append(f"  ❌ {fp} — {reason}")
            if len(excluded) > 8:
                sections.append(f"  ... and {len(excluded) - 8} more")

        if not evidence:
            sections.append("No files verified yet.")

        # ── 2. Unresolved References ──
        all_unresolved = set()
        for fp, v in evidence.items():
            if v.get("relevant"):
                for ref in v.get("unresolved_references", []):
                    all_unresolved.add(ref)
                insp = v.get("inspection")
                if insp and hasattr(insp, "unresolved"):
                    for u in insp.unresolved:
                        all_unresolved.add(str(u))

        if all_unresolved:
            sections.append("\nUNRESOLVED REFERENCES (discovered but not yet traced):")
            for ref in sorted(all_unresolved)[:10]:
                sections.append(f"  ⚠️  {ref}")

        # ── 3. Hypothesis Coverage ──
        if hypotheses:
            sections.append("\nHYPOTHESIS COVERAGE:")
            included_fps = {fp for fp, _ in included}
            for h in hypotheses:
                hyp_id = getattr(h, "id", "?")
                hyp_str = getattr(h, "hypothesis", str(h))[:100]

                # Check if any evidence mentions this hypothesis's signals
                _symbols = set(getattr(h, "symbols", []))
                _literals = set(getattr(h, "literals", []))
                _subjects = set(getattr(h, "core_subjects", []))
                all_signals = _symbols | _literals | _subjects

                # Find evidence entries whose provenance or content relates
                matched_files = []
                for fp in included_fps:
                    ev = evidence.get(fp, {})
                    # Check if any search query used this hypothesis's signals
                    for prov in ev.get("search_provenance", []):
                        q = prov.get("query", "").lower()
                        if any(s.lower() in q for s in all_signals if len(s) > 2):
                            matched_files.append(Path(fp).name)
                            break

                if matched_files:
                    sections.append(
                        f"  [{hyp_id}] ✅ COVERED — {hyp_str}\n"
                        f"     evidence: {', '.join(matched_files[:5])}"
                    )
                else:
                    sections.append(
                        f"  [{hyp_id}] ❓ NO EVIDENCE YET — {hyp_str}"
                    )

        # ── 4. Unexplored Relationships ──
        unvisited_rels = []
        for fp, v in evidence.items():
            if v.get("relevant"):
                for rel in v.get("followable", []):
                    if not rel.get("visited"):
                        unvisited_rels.append((fp, rel))

        if unvisited_rels:
            sections.append("\nUNEXPLORED RELATIONSHIPS (discovered but not followed):")
            for fp, rel in unvisited_rels[:8]:
                target = rel.get("target_file") or rel.get("dst_name", "?")
                kind = rel.get("kind", "?")
                source_name = Path(fp).name
                sections.append(
                    f"  🔗 {source_name} —[{kind}]→ {target}"
                )

        # ── 5. Open Questions ──
        if evidence_knowledge and evidence_knowledge.all_questions:
            open_qs = [
                q for q in evidence_knowledge.all_questions
                if q not in evidence_knowledge.resolved_questions
            ]
            if open_qs:
                sections.append("\nOPEN QUESTIONS (from artifact inspections):")
                for q in open_qs[:5]:
                    sections.append(f"  ? {q}")

        # ── 6. Capability Awareness ──
        # Lists capabilities that haven't been used yet. This is a REMINDER
        # of what's in the toolbox, NOT a prescription to use them.
        # The controller prompt's DECISION PROCESS section governs selection.
        # The threshold below is a controller-awareness heuristic, not
        # evidence logic — it suppresses noise once investigation is mature.
        all_capabilities = {
            "ripgrep", "symbol_lookup", "neo4j_graph",
            "semantic_rag", "i18n_chain",
        }
        used_capabilities = {t.get("tool", "") for t in (tried if tried else [])}
        # Also count "follow" as using neo4j_graph
        if any(t.get("tool") == "follow" for t in (tried or [])):
            used_capabilities.add("neo4j_graph")
        unused = all_capabilities - used_capabilities
        # Show only when investigation is still early (heuristic: few verified
        # artifacts means the controller may not yet be aware of all options).
        if unused and len(included) < 3:
            sections.append(
                "\nAVAILABLE CAPABILITIES (not yet used):"
            )
            for cap in sorted(unused):
                sections.append(f"  🔧 {cap}")
            sections.append(
                "  ⚠️  Unused capabilities are informational only. Do NOT select a capability "
                "merely because it has not been tried. Select it ONLY when it is the most "
                "direct way to resolve an unresolved requirement."
            )

        return "\n".join(sections) if sections else "No investigation state available."

    def _decide_next_action(

        self,

        ticket: "ValueEdgeTicket",

        hypotheses: Optional[List[Any]],

        evidence: Dict[str, dict],

        tried: List[dict],

        all_items: List[EvidenceItem],

        iteration: int,

        max_iterations: int,

        evidence_knowledge: Optional[EvidenceKnowledge] = None,

        sufficiency_guidance: Optional[List[str]] = None,

        recovery_context: Optional[dict] = None,

    ) -> dict:

        """

        Single LLM call: given ticket + evidence so far + tried tools,

        pick exactly one action:

          {"type": "search", "tool": "<tool_name>", "query": "<query>", "reasoning": "..."}

          {"type": "proceed", "reasoning": "..."}

          {"type": "ask_user", "question": "...", "reasoning": "..."}

        """

        from ticket_to_code.json_utils import parse_llm_json

        # Build context for the LLM

        ticket_title = getattr(ticket, "title", "") or ""

        ticket_desc = getattr(ticket, "description", "") or ""

        # Summarize evidence found so far

        # ── Collect all followable relationships across evidence ──
        all_followable = {}
        rel_counter = 0
        for fp, v in evidence.items():
            if v.get("relevant") and v.get("followable"):
                for rel in v["followable"]:
                    rel_counter += 1
                    rel_id = f"R{rel_counter}"
                    all_followable[rel_id] = rel

        # Store for controller to use when resolving follow actions
        self._current_followable_relationships = all_followable

        # ── Build holistic investigation state from the ledger ──
        investigation_state = self._build_investigation_state(
            evidence, hypotheses, evidence_knowledge, tried=tried
        )

        # Summarize what was already tried
        tried_summary = ""
        for t in tried:
            n_files = len(t.get("files_returned", []))
            status = t.get("status", "")
            if status:
                tried_summary += f"  - [{t['tool']}] query={t['query']!r} → {status}\n"
            else:
                tried_summary += f"  - [{t['tool']}] query={t['query']!r} → {n_files} file(s)\n"

        if not tried:
            tried_summary = "  (nothing tried yet)\n"

        # Hypothesis context — preserve FULL structured search guidance
        # so the controller can reason about which signals remain unexplored.
        hyp_text = ""
        if hypotheses:
            for h in hypotheses:
                hyp_str = getattr(h, "hypothesis", str(h))
                hyp_id = getattr(h, "id", "?")
                hyp_conf = getattr(h, "confidence", 0.5)
                hyp_text += f"  [{hyp_id}] (confidence={hyp_conf:.1f}) {hyp_str}\n"

                # Structured search signals from the investigation agent
                _literals = getattr(h, "literals", [])
                _symbols = getattr(h, "symbols", [])
                _queries = getattr(h, "queries", [])
                _anchors = getattr(h, "anchors", [])
                _subjects = getattr(h, "core_subjects", [])
                _intent = getattr(h, "business_intent", None)

                if _literals:
                    hyp_text += f"       literals: {', '.join(_literals[:5])}\n"
                if _symbols:
                    hyp_text += f"       symbols: {', '.join(_symbols[:5])}\n"
                if _queries:
                    hyp_text += f"       queries: {', '.join(_queries[:3])}\n"
                if _anchors:
                    hyp_text += f"       anchors: {', '.join(_anchors[:4])}\n"
                if _subjects:
                    hyp_text += f"       core_subjects: {', '.join(_subjects[:3])}\n"
                if _intent:
                    hyp_text += f"       intent: {_intent}\n"

        # Information gain signal
        info_gain_text = ""
        if evidence_knowledge:
            gain_summary = evidence_knowledge.recent_gain_summary(last_n=3)
            if gain_summary:
                info_gain_text = f"\nINFORMATION GAIN:\n{gain_summary}\n"

        # ── Sufficiency guidance: search directions from the gating judge ──
        # These are INPUTS to this controller's decision, not commands.
        # The controller may choose to follow them, combine them with its own
        # reasoning, or override them if it has better information.
        sufficiency_text = ""
        if sufficiency_guidance:
            sufficiency_text = "\nSUFFICIENCY GUIDANCE (areas where evidence is still lacking):\n"
            for direction in sufficiency_guidance[:3]:
                sufficiency_text += f"  → {direction}\n"

        # ── Recovery context: structured recovery from a previous planning failure ──
        # This is NOT part of the original ticket — it is the system's diagnosis
        # of why a previous planning attempt failed and what specific evidence
        # is needed to resolve the failure.
        recovery_text = ""
        if recovery_context is not None and not isinstance(recovery_context, dict):
            if hasattr(recovery_context, "model_dump"):
                recovery_context = recovery_context.model_dump()
            elif hasattr(recovery_context, "dict"):
                recovery_context = recovery_context.dict()
            elif hasattr(recovery_context, "__dict__"):
                recovery_context = dict(recovery_context.__dict__)

        if recovery_context and isinstance(recovery_context, dict) and recovery_context.get("recovery_type") in (
            "evidence_incomplete", "wrong_candidates"
        ):
            recovery_text = "\nRECOVERY CONTEXT (previous planning attempt failed — targeted investigation needed):\n"
            recovery_text += f"  Failure diagnosis: {recovery_context.get('reason', 'unknown')}\n"
            if recovery_context.get("investigation_target"):
                recovery_text += f"  Priority target: {recovery_context['investigation_target']}\n"
            if recovery_context.get("investigation_query"):
                recovery_text += f"  Specific query: {recovery_context['investigation_query']}\n"
            if recovery_context.get("alternative_search_terms"):
                terms = recovery_context["alternative_search_terms"][:3]
                recovery_text += f"  Alternative search terms: {', '.join(terms)}\n"

        # ── Build followable relationships section (prioritize unvisited) ──
        followable_text = ""
        if all_followable:
            unvisited_rels = {k: v for k, v in all_followable.items() if not v.get("visited")}
            display_rels = unvisited_rels if unvisited_rels else all_followable
            followable_text = "\nFOLLOWABLE RELATIONSHIPS (discovered from inspected artifacts):\n\n"
            for rel_id, rel_info in display_rels.items():
                target_display = rel_info.get("target_file") or rel_info.get("dst_name", "?")
                source_display = Path(rel_info.get("source", "?")).name
                symbol_display = rel_info.get("dst_name", "")
                kind = rel_info.get("kind", "?")
                followable_text += (
                    f"[{rel_id}] {kind}\n"
                    f"     source: {source_display}\n"
                    f"     target: {target_display}\n"
                    f"     symbol: {symbol_display}\n\n"
                )

        prompt = f"""You are an evidence-collection agent for a code-change ticket.

Your job: find the SOURCE CODE FILES that need to be MODIFIED to resolve this ticket.
Think of the investigation as assembling a PUZZLE — each piece is a verified artifact,
a proven relationship, or a confirmed behavioral fact. Your goal is to assemble enough
pieces that a planner can write the code change.

TICKET:
  Title: {ticket_title}
  Description: {ticket_desc}

HYPOTHESES (what the investigation agent thinks needs to change):
{hyp_text or '  (none)'}

{TOOL_CATALOG}

INVESTIGATION STATE (the puzzle so far):
{investigation_state}

SEARCHES ALREADY TRIED (do NOT repeat):
{tried_summary}
{info_gain_text}
{sufficiency_text}
{recovery_text}
{followable_text}
ITERATION: {iteration + 1} of {max_iterations}

Decide your next action. Return EXACTLY ONE JSON object:

Option 1 — Search for more evidence:
{{"type": "search", "tool": "<tool_name from catalog>", "query": "<search query>", "reasoning": "<why this tool and query>"}}

Option 2 — You have enough evidence to proceed to planning:
{{"type": "proceed", "reasoning": "<why evidence is sufficient>"}}

Option 3 — You genuinely cannot find the relevant files (ticket is too vague, missing info):
{{"type": "ask_user", "question": "<specific question for the ticket author>", "reasoning": "<why you're stuck>"}}

Option 4 — Follow a discovered relationship to its target (deterministic, no search needed):
{{"type": "follow", "relationship_id": "<Rn from FOLLOWABLE RELATIONSHIPS>", "reasoning": "<why this relationship is relevant to the ticket>"}}

REQUIREMENT-DRIVEN STOPPING CRITERIA (when to choose "proceed"):
  Ask yourself these questions — proceed ONLY when ALL applicable answers are YES:
  1. BEHAVIOR: Do I understand what behavior the ticket requires changing?
  2. LOCATION: Do I know which artifact(s) actually implement that behavior?
  3. PATH: Is the implementation path (entry point → logic → data) understood?
  4. CONTRACTS: Are the dependencies and API contracts needed for the change known?
  5. COMPLETENESS: Given the INVESTIGATION STATE above, can a planner safely understand what to change and how?
     Hypotheses are context for what to investigate — not hard gates. A high-confidence hypothesis
     with an unresolved implementation path is NOT sufficient. A low-confidence hypothesis that
     represents a real ticket requirement should NOT be ignored.
  6. QUESTIONS: Are there any OPEN QUESTIONS that would block a planner from writing correct code?

  Do NOT proceed merely because one relevant file was found — check if its contracts/dependencies are known.
  Do NOT require artificial "frontend + backend + database" coverage if the ticket only affects one area.
  Do NOT keep searching just because iterations remain — stop when the puzzle is solved.
  Proceed when the implementation path is sufficiently understood to write correct code.

DECISION PROCESS (follow this reasoning, not tool checklists):
  1. What requirement or question is STILL UNRESOLVED?
  2. What evidence would resolve it?
  3. Which available capability can obtain that evidence most directly?

  Do NOT reason: "ripgrep used, RAG not used, therefore use RAG."
  Do NOT reason: "i18n not used, therefore try i18n."
  DO reason: "The ticket says 'organization' but no code uses that term → semantic_rag can find conceptual matches."
  DO reason: "ProjectsService is referenced but its methods are unknown → follow the relationship or use symbol_lookup."

Rules:
- Do NOT repeat a (tool, query) pair from SEARCHES ALREADY TRIED.
- PREFER "follow" over "search" when a relevant relationship has already been discovered. Following is deterministic and does not waste a search.
- If you found a relevant service/controller but don't know its methods, use "follow" to traverse to it if a relationship exists, otherwise use a different search.
- If recent searches produced no new artifacts or facts (low information gain), consider using "follow" on existing relationships or proceeding if evidence is sufficient.
- Use "ask_user" ONLY when you've tried multiple approaches and still can't find relevant files.
- Be specific with queries. Don't search for generic terms like "error" or "component".
- If the ticket references UI text, business terminology, or error messages that may be localized, i18n_chain or semantic_rag can help bridge the gap between user-visible language and code identifiers.

Return ONLY the JSON object, no other text."""

        try:

            llm = self._get_agentic_llm()

            if not llm:

                logger.warning("[AgenticLoop] No LLM available — falling back to proceed (DEGRADED)")

                return {
                    "type": "proceed",
                    "degraded": True,
                    "reason": "llm_unavailable",
                    "reasoning": "LLM unavailable, proceeding with current evidence",
                }

            # ── Diagnostic trace: dump evidence context sent to LLM ──
            try:
                _ticket_id = getattr(ticket, "ticket_id", "unknown")
                _trace_dir = str(get_trace_dir(_ticket_id))
                os.makedirs(_trace_dir, exist_ok=True)
                _trace_path = os.path.join(_trace_dir, "inspection_trace.txt")
                with open(_trace_path, "a", encoding="utf-8") as _tf:
                    _tf.write(f"\n{'='*80}\n")
                    _tf.write(f"ITERATION {iteration + 1} — _decide_next_action() LLM CONTEXT\n")
                    _tf.write(f"{'='*80}\n")
                    _tf.write(f"\n--- INVESTIGATION STATE ---\n{investigation_state}\n")
                    if info_gain_text:
                        _tf.write(f"\n--- INFO GAIN ---\n{info_gain_text}\n")
                    if followable_text:
                        _tf.write(f"\n--- FOLLOWABLE RELATIONSHIPS ---\n{followable_text}\n")
                    # Also dump raw inspection objects
                    _tf.write(f"\n--- INSPECTION OBJECTS ---\n")
                    for _fp, _ev in evidence.items():
                        if _ev.get("inspection"):
                            _insp = _ev["inspection"]
                            _tf.write(f"\nFILE: {_fp}\n")
                            _tf.write(f"  class: {_insp.class_name}\n")
                            _tf.write(f"  methods: {[m.name for m in _insp.methods]}\n")
                            _tf.write(f"  relevant_methods: {[m.name for m in _insp.relevant_methods]}\n")
                            _tf.write(f"  type_deps: {_insp.type_dependencies}\n")
                            _tf.write(f"  contracts: {_insp.api_contracts}\n")
                            _tf.write(f"  facts: {[f.fact for f in _insp.facts]}\n")
                            _tf.write(f"  questions: {_insp.questions}\n")
                            _tf.write(f"  providers: {_insp.providers_used}\n")
                            _tf.write(f"  depth: {_insp.inspection_depth}\n")
                            _tf.write(f"  code_regions: {list(_insp.relevant_code_regions.keys())}\n")
                            for _rn, _ri in _insp.relevant_code_regions.items():
                                _tf.write(f"  --- CODE: {_rn}() L{_ri.get('start')}-{_ri.get('end')} ---\n")
                                _code = _ri.get("code", "")
                                for _cl in _code.splitlines()[:30]:
                                    _tf.write(f"    {_cl}\n")
                                _tf.write(f"  --- END ---\n")
                    _tf.write(f"\n{'='*80}\n\n")
                logger.info(f"  [Inspection] Trace written to {_trace_path}")
            except Exception as _te:
                logger.debug(f"  [Inspection] Trace write failed: {_te}")

            from ticket_to_code.llm_utils import llm_invoke
            response = llm_invoke(llm, prompt)

            text = response.content if hasattr(response, "content") else str(response)

            try:
                action = parse_llm_json(text)
            except Exception as _parse_exc:
                logger.debug(f"[AgenticLoop] action JSON parse failed: {_parse_exc}")
                action = None

            if not isinstance(action, dict) or "type" not in action:

                logger.warning(f"[AgenticLoop] Invalid action format: {text[:200]} (DEGRADED → proceed)")

                return {
                    "type": "proceed",
                    "degraded": True,
                    "reason": "unparseable_action",
                    "reasoning": "Could not parse LLM action",
                }

            action_type = action.get("type", "proceed")

            if action_type not in ("search", "proceed", "ask_user", "follow"):

                action["type"] = "proceed"

            logger.info(

                f"  🤖 Agentic iteration {iteration + 1}: "

                f"action={action['type']}"

                + (f" tool={action.get('tool')}" if action["type"] == "search" else "")

                + f" | {action.get('reasoning', '')[:80]}"

            )

            return action

        except Exception as e:

            logger.warning(f"[AgenticLoop] LLM decision failed: {e} — proceeding (DEGRADED)")

            return {
                "type": "proceed",
                "degraded": True,
                "reason": f"llm_error: {e}",
                "reasoning": f"LLM error: {e}",
            }

    def _get_agentic_llm(self):
        """Get or create the LLM for agentic decisions.

        Uses LLMRegistry.get_llm() — the same path every other agent in the
        pipeline uses (TicketAnalyzer, PlanningAgent, InvestigationAgent, etc.).
        The previous implementation tried to create its own ChatGoogleGenerativeAI
        with GOOGLE_API_KEY env var, but the backend runs on Vertex AI where
        that env var doesn't exist, so the LLM was always None.
        """
        if hasattr(self, "_agentic_llm") and self._agentic_llm is not None:
            return self._agentic_llm

        try:
            from ticket_to_code.runtime.llm_factory import LLMRegistry
            self._agentic_llm = LLMRegistry.get_llm(assistant=False)
            if self._agentic_llm:
                logger.info("[AgenticLoop] LLM initialized via LLMRegistry")
            else:
                logger.warning("[AgenticLoop] LLMRegistry.get_llm() returned None")
            return self._agentic_llm
        except Exception as e:
            logger.warning(f"[AgenticLoop] Could not create LLM via LLMRegistry: {e}")
            self._agentic_llm = None
            return None

    def _auto_follow_and_inspect(
        self,
        ticket: Any,
        hypotheses: Optional[List[Any]],
        evidence: Dict[str, dict],
        evidence_knowledge: "EvidenceKnowledge",
        context: Any,
        max_follow: int = 6,
        max_hops: int = 2,
    ) -> None:
        """Deterministically follow relevant relationships and inspect targets.

        For every followable edge whose target is not yet inspected, if the
        target can affect the ticket's behavioral understanding (_should_inspect),
        inspect it with the existing progressive inspector. Bounded by
        max_follow (total) and max_hops (depth) to avoid repository sweeps.
        """
        ticket_context = f"{getattr(ticket, 'title', '') or ''}\n{getattr(ticket, 'description', '') or ''}"
        hypotheses_text = "\n".join(
            getattr(h, "hypothesis", "") or "" for h in (hypotheses or [])
        )[:2000]
        visited = getattr(context, "visited_files", set()) or set()

        frontier: List[tuple] = []
        for _fp, _ev in list(evidence.items()):
            for rel in (_ev.get("followable") or []):
                tf = rel.get("target_file")
                if tf and not rel.get("visited"):
                    frontier.append((tf, 1))

        seen: Set[str] = set()
        followed = 0
        while frontier and followed < max_follow:
            target_file, hop = frontier.pop(0)
            if target_file in seen or hop > max_hops:
                continue
            seen.add(target_file)
            if evidence.get(target_file, {}).get("inspection"):
                continue
            if not self._should_inspect(
                target_file, {}, evidence, ticket_context, evidence_knowledge
            ):
                continue
            try:
                inspection = self._inspect_artifact(
                    target_file, ticket_context, evidence, hypotheses_text
                )
            except Exception:
                continue
            if not (inspection.methods or inspection.facts or inspection.api_contracts):
                continue

            for rm in (inspection.relevant_methods or inspection.methods)[:2]:
                if rm.line_start:
                    end_l = rm.line_end if (rm.line_end and rm.line_end > rm.line_start) else rm.line_start + 40
                    code = self._read_relevant_region(target_file, rm.line_start, end_l, max_lines=80)
                    if code:
                        inspection.relevant_code_regions[rm.name] = {
                            "code": code, "start": rm.line_start, "end": end_l, "file": target_file,
                        }
                        inspection.inspection_depth = "region"

            evidence.setdefault(target_file, {})["inspection"] = inspection
            evidence[target_file].setdefault("relevant", True)
            evidence[target_file].setdefault("reason", "auto-followed provider (relationship)")
            if target_file not in evidence_knowledge.verified_artifacts:
                evidence_knowledge.verified_artifacts.append(target_file)
            evidence_knowledge.inspection_knowledge[target_file] = inspection
            evidence_knowledge.all_facts.extend(inspection.facts)
            evidence_knowledge.all_questions.extend(inspection.questions)
            for c in inspection.api_contracts:
                if c not in evidence_knowledge.api_contracts:
                    evidence_knowledge.api_contracts.append(c)
            followed += 1
            logger.info(
                f"  🔗 Auto-followed → inspected {Path(target_file).name} "
                f"({inspection.inspection_depth}, {len(inspection.methods)} method(s))"
            )

            next_rels = self._get_followable_relationships(target_file, visited, evidence)
            if next_rels:
                evidence[target_file]["followable"] = next_rels
                evidence_knowledge.relationships.extend(next_rels)
                for r in next_rels:
                    tf = r.get("target_file")
                    if tf and not r.get("visited"):
                        frontier.append((tf, hop + 1))

        if followed:
            logger.info(
                f"  🔗 Auto-follow: inspected {followed} additional provider "
                f"artifact(s) to complete behavioral understanding"
            )

    def _get_followable_relationships(
        self,
        file_path: str,
        visited_files: Set[str],
        evidence: Dict[str, dict],
    ) -> List[Dict]:
        """Extract followable outgoing relationships from an inspected artifact.

        Uses the existing SQLite edges table. Returns structured relationship
        records with kind, source, target, and visited status.
        Does NOT create a new graph — reads the existing one.
        Supports both forward edges and reverse template_of/style_of edges.
        """
        if not self.localizer.sqlite_store:
            return []

        followable = []
        try:
            conn = self.localizer.sqlite_store._conn

            # Normalize file_path to relative and slash-normalized formats
            rel_path = file_path
            ws = Path(self._workspace) if hasattr(self, "_workspace") and self._workspace else None
            if ws:
                try:
                    rel_path = str(Path(file_path).relative_to(ws)).replace("\\", "/")
                except ValueError:
                    pass
            norm_path = file_path.replace("\\", "/")
            file_name = Path(file_path).name

            # 1. Forward edges: path = this file
            rows = conn.execute(
                "SELECT kind, dst_name, dst_id FROM edges WHERE path IN (?, ?, ?) OR path LIKE ?",
                (file_path, rel_path, norm_path, f"%{file_name}"),
            ).fetchall()

            # 2. Reverse edges: template_of / style_of where dst_name is this file
            rev_rows = conn.execute(
                "SELECT kind, path, '' FROM edges WHERE (dst_name IN (?, ?, ?) OR dst_name LIKE ?) "
                "AND kind IN ('template_of', 'style_of')",
                (file_path, rel_path, norm_path, f"%{file_name}"),
            ).fetchall()

            all_rows = list(rows)
            for kind, parent_path, _ in rev_rows:
                # For reverse template/style, target is the parent component
                all_rows.append((kind, parent_path, ""))

            seen_targets = set()
            for kind, dst_name, dst_id in all_rows:
                # Skip non-traversable edge kinds
                if kind in ("contains", "annotated_by"):
                    continue

                # Skip third-party / standard library imports generically (no hardcoded repo names)
                if kind == "imports" and dst_name:
                    if dst_name.startswith(("@", "node:", "rxjs", "java.", "javax.", "jakarta.",
                                            "org.springframework", "org.junit", "com.google.",
                                            "ch.qos.", "org.apache.", "org.slf4j", "com.fasterxml.")):
                        continue
                    if "/" not in dst_name and "\\" not in dst_name and not dst_name.endswith((".ts", ".js", ".py", ".java", ".kt", ".cs")):
                        continue

                # Deduplicate by (kind, dst_name)
                dedup_key = (kind, dst_name)
                if dedup_key in seen_targets:
                    continue
                seen_targets.add(dedup_key)

                # Resolve target file
                target_file = self._resolve_edge_to_file(kind, dst_name)

                # Skip self-references
                if target_file and (target_file == file_path or Path(target_file).name == file_name):
                    continue

                # Check if target is already visited/inspected
                is_visited = False
                if target_file:
                    target_name = Path(target_file).name
                    is_visited = (
                        target_file in visited_files
                        or target_file in evidence
                        or any(Path(v).name == target_name for v in visited_files)
                        or any(Path(k).name == target_name for k in evidence.keys())
                    )

                followable.append({
                    "kind": kind,
                    "source": file_path,
                    "dst_name": dst_name,
                    "target_file": target_file,
                    "visited": is_visited,
                })

        except Exception as e:
            logger.debug(f"  [Followable] Failed to extract edges for {file_path}: {e}")

        return followable

    def _resolve_edge_to_file(self, edge_kind: str, dst_name: str) -> Optional[str]:
        """Resolve an edge destination to a file path using existing SQLite data.

        For 'imports' edges: dst_name is often already a full file path.
        For 'has_type'/'calls' edges: look up in symbols table.
        For 'template_of'/'style_of': dst_name is the target file path.
        Prioritizes paths that exist on disk in the current workspace, preventing
        selection of orphan root paths.
        """
        if not dst_name:
            return None

        ws = Path(self._workspace) if hasattr(self, "_workspace") and self._workspace else None
        norm_dst = dst_name.replace('\\', '/')

        # Case 1: dst_name is already a file path (supports / and \)
        if ('/' in norm_dst or norm_dst.endswith(
            ('.ts', '.tsx', '.java', '.py', '.html', '.scss', '.css', '.kt', '.cs')
        )) and norm_dst.endswith(
            ('.ts', '.tsx', '.java', '.py', '.html', '.scss', '.css', '.kt', '.cs')
        ):
            if ws and (ws / dst_name).exists():
                return str(ws / dst_name)
            if ws and (ws / norm_dst).exists():
                return str(ws / norm_dst)
            if Path(dst_name).exists():
                return dst_name
            return dst_name

        # Case 2: template_of / style_of — dst_name is the target file path
        if edge_kind in ('template_of', 'style_of'):
            if ws and (ws / dst_name).exists():
                return str(ws / dst_name)
            if ws and (ws / norm_dst).exists():
                return str(ws / norm_dst)
            if Path(dst_name).exists():
                return dst_name
            return dst_name

        # Case 3: Symbol name — look up definition in symbols table
        if not self.localizer.sqlite_store:
            return None

        try:
            conn = self.localizer.sqlite_store._conn
            rows = conn.execute(
                "SELECT DISTINCT path FROM symbols WHERE name = ? AND kind = 'class'",
                (dst_name,),
            ).fetchall()
            if not rows:
                # Fallback: try without kind filter
                rows = conn.execute(
                    "SELECT DISTINCT path FROM symbols WHERE name = ?",
                    (dst_name,),
                ).fetchall()
            if rows:
                candidate_paths = [r[0] for r in rows if r[0]]
                if ws:
                    existing = [p for p in candidate_paths if (ws / p).exists()]
                    if existing:
                        return str(ws / existing[0])
                return candidate_paths[0]
        except Exception as e:
            logger.debug(f"  [Resolve] Symbol lookup failed for {dst_name}: {e}")

        return None

    def _count_pre_visited_results(
        self,
        tool_name: str,
        query: str,
        context: "SearchContext",
    ) -> int:
        """Count how many results a tool would return if visited filter were removed.

        Used to distinguish 'genuinely 0 results' from 'N results, all visited'.
        """
        try:
            if tool_name == "symbol_lookup":
                if not self.localizer.sqlite_store:
                    return 0
                conn = self.localizer.sqlite_store._conn
                rows = conn.execute(
                    "SELECT DISTINCT path FROM symbols WHERE name = ?",
                    (query,),
                ).fetchall()
                return sum(1 for r in rows if r[0] in context.visited_files)

            elif tool_name == "ripgrep" and self.repo_search:
                results = self.repo_search.search_literal(query)
                return sum(1 for r in results if r.file_path in context.visited_files)

            elif tool_name == "filename_search" and self.repo_search:
                results = self.repo_search.search_filename(query)
                return sum(1 for r in results if r.file_path in context.visited_files)

            elif tool_name == "ast_companion_bundle":
                bundle = self._resolve_ast_companion_bundle(query, set())
                return sum(1 for r in bundle if r.file_path in context.visited_files)

        except Exception:
            pass
        return 0


    def _execute_tool(

        self,

        tool_name: str,

        query: str,

        context: "SearchContext",

    ) -> List[EvidenceItem]:

        """

        Dispatcher: maps a tool name string to the existing search method.

        No new tools. Returns a list of EvidenceItem objects.

        """

        # ── Sanitize query for search tools (strip hallucinated CLI flags) ──
        if tool_name in _SEARCH_TOOLS_NEEDING_SANITIZATION:
            query = _sanitize_query(query)

        items: List[EvidenceItem] = []

        try:

            if tool_name == "ripgrep":
                if self.repo_search and len(query.strip()) >= _REPO_SEARCH_MIN_LITERAL_LEN:
                    results = self.repo_search.search_literal(query)
                    for r in results:
                        if r.file_path not in context.visited_files:
                            search_mode = getattr(r, "search_mode", "exact")
                            mode_suffix = f"_{search_mode}" if search_mode != "exact" else ""
                            snippet_prefix = f"[ripgrep:{query!r}"
                            if search_mode == "ui_container_stripped":
                                resolved = getattr(r, "resolved_query", query)
                                snippet_prefix += f" → stripped:{resolved!r}"
                            snippet_prefix += "]"
                            _cls = getattr(r, "classification", "code")
                            _mult = _HIT_CLASS_MULTIPLIER.get(_cls, 1.0)
                            items.append(EvidenceItem(
                                provider="literal", strength="strong" if _mult == 1.0 else "medium",
                                details=f"agentic_ripgrep{mode_suffix}_{_cls}",
                                file_path=r.file_path,
                                evidence_type="usage",
                                content_snippet=f"{snippet_prefix} [{_cls}] line {r.line_number}: {r.matched_text}"[:SNIPPET_MAX_CHARS],
                                relevance_score=round(min(float(r.confidence) * _mult, 1.0), 3),
                                hypothesis_id="agentic",
                            ))

            elif tool_name == "grep_search":
                if self.repo_search and query.strip():
                    results = self.repo_search.search_literal(query) or []
                    if not results:
                        results = self.repo_search.search_regex(query) or []
                    for r in results:
                        if r.file_path not in context.visited_files:
                            matched_text = getattr(r, "matched_text", "").strip()
                            line_no = getattr(r, "line_number", "?")
                            items.append(EvidenceItem(
                                provider="grep_search",
                                strength="strong",
                                details="live_grep_search",
                                file_path=r.file_path,
                                evidence_type="usage",
                                content_snippet=f"[grep_search:{query!r}] line {line_no}: {matched_text}"[:SNIPPET_MAX_CHARS],
                                relevance_score=min(float(r.confidence), 1.0),
                                hypothesis_id="agentic",
                            ))

            elif tool_name == "view_file":
                cleaned = query.strip()
                m = re.match(r"^([^:#\s]+)(?:[:#\s]+L?(\d+))?(?:-(\d+))?$", cleaned)
                if m:
                    target_file = m.group(1).strip()
                    start_l = int(m.group(2)) if m.group(2) else 1
                    end_l = int(m.group(3)) if m.group(3) else (start_l + 100)
                else:
                    target_file = cleaned
                    start_l, end_l = 1, 100

                ws_dir = Path(self._workspace) if getattr(self, "_workspace", None) else Path.cwd()
                disk_path = ws_dir / target_file
                rel_path = target_file
                if not disk_path.exists():
                    file_name = Path(target_file).name
                    for candidate in ws_dir.rglob(file_name):
                        if candidate.is_file():
                            disk_path = candidate
                            rel_path = str(candidate.relative_to(ws_dir)).replace("\\", "/")
                            break

                if disk_path.exists() and disk_path.is_file():
                    try:
                        file_lines = disk_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                        total = len(file_lines)
                        s_idx = max(0, start_l - 1)
                        e_idx = min(total, end_l)
                        snippet_lines = [f"{idx+1}: {file_lines[idx]}" for idx in range(s_idx, e_idx)]
                        formatted_snippet = "\n".join(snippet_lines)
                        items.append(EvidenceItem(
                            provider="view_file",
                            strength="strong",
                            details=f"targeted_view_L{start_l}-L{end_l}",
                            file_path=rel_path,
                            evidence_type="content",
                            content_snippet=f"[view_file:{rel_path} L{start_l}-{end_l} of {total}]\n{formatted_snippet}"[:1500],
                            relevance_score=0.95,
                            hypothesis_id="agentic",
                        ))
                    except Exception as err:
                        logger.debug(f"view_file read error: {err}")

            elif tool_name == "filename_search":
                if self.repo_search:
                    results = self.repo_search.search_filename(query)
                    for r in results:
                        if r.file_path not in context.visited_files:
                            search_mode = getattr(r, "search_mode", "exact")
                            is_ambig = getattr(r, "is_ambiguous", False)
                            mode_suffix = f"_{search_mode}" if search_mode != "exact" else ""
                            ambig_tag = " (AMBIGUOUS)" if is_ambig else ""
                            items.append(EvidenceItem(
                                provider="filename",
                                strength="weak" if is_ambig else "medium",
                                details=f"agentic_filename{mode_suffix}{'_ambiguous' if is_ambig else ''}",
                                file_path=r.file_path,
                                evidence_type="structural",
                                content_snippet=f"[filename:{query!r}{ambig_tag}] {r.file_path}",
                                relevance_score=min(float(r.confidence), 1.0),
                                hypothesis_id="agentic",
                            ))

                    # Auto-companion expansion: discover sibling template, styles, tests
                    try:
                        from ticket_to_code.intelligence.contracts.companion_resolver import PolyglotCompanionResolver
                        ws_path = str(self._workspace) if getattr(self, "_workspace", None) else None
                        base_fps = [it.file_path for it in items if it.file_path]
                        for b_fp in base_fps:
                            for comp_path in PolyglotCompanionResolver.resolve_all_companions(b_fp, workspace_path=ws_path):
                                if comp_path not in context.visited_files and not any(it.file_path == comp_path for it in items):
                                    items.append(EvidenceItem(
                                        provider="companion",
                                        strength="strong",
                                        details="auto_companion_expansion",
                                        file_path=comp_path,
                                        evidence_type="structural",
                                        content_snippet=f"[companion of {Path(b_fp).name}] {comp_path}",
                                        relevance_score=0.95,
                                        hypothesis_id="companion",
                                    ))
                    except Exception as _comp_err:
                        logger.debug(f"companion auto-expansion skipped: {_comp_err}")

            elif tool_name == "regex_search":
                if self.repo_search:
                    results = self.repo_search.search_regex(query)
                    for r in results:
                        err = getattr(r, "error", None)
                        if err:
                            items.append(EvidenceItem(
                                provider="regex", strength="weak",
                                details="agentic_regex_error",
                                file_path="",
                                evidence_type="error",
                                content_snippet=f"[regex:error] {err}",
                                relevance_score=0.0,
                                hypothesis_id="agentic",
                            ))
                        elif r.file_path not in context.visited_files:
                            _cls = getattr(r, "classification", "code")
                            _mult = _HIT_CLASS_MULTIPLIER.get(_cls, 1.0)
                            items.append(EvidenceItem(
                                provider="regex", strength="medium" if _mult == 1.0 else "weak",
                                details=f"agentic_regex_{_cls}",
                                file_path=r.file_path,
                                evidence_type="usage",
                                content_snippet=f"[regex:{query!r}] [{_cls}] line {r.line_number}: {r.matched_text}"[:SNIPPET_MAX_CHARS],
                                relevance_score=round(min(float(r.confidence) * _mult, 1.0), 3),
                                hypothesis_id="agentic",
                            ))

            elif tool_name == "sqlite_fts":

                items.extend(self._query_sqlite_fts([query], "agentic", context.visited_files))

            elif tool_name == "symbol_lookup":
                sym_items = self._query_sqlite_symbols([query], "agentic", context.visited_files)
                if not sym_items and self.repo_search:
                    try:
                        safe_q = re.escape(query.strip())
                        res = self.repo_search.search_regex(rf"\b{safe_q}\b", max_results=10)
                        for r in res:
                            if not getattr(r, "error", None) and r.file_path not in context.visited_files:
                                sym_items.append(EvidenceItem(
                                    provider="regex",
                                    strength="medium",
                                    details="agentic_symbol_fallback",
                                    file_path=r.file_path,
                                    evidence_type="definition",
                                    content_snippet=f"[symbol_fallback:{query}] line {r.line_number}: {r.matched_text}"[:SNIPPET_MAX_CHARS],
                                    relevance_score=0.80,
                                    hypothesis_id="agentic",
                                    symbol_name=query,
                                ))
                    except Exception as _e:
                        logger.warning(f"Symbol lookup regex fallback failed for {query}: {_e}")
                items.extend(sym_items)

            elif tool_name == "ast_companion_bundle":
                items.extend(self._resolve_ast_companion_bundle(query, context.visited_files))

            elif tool_name == "neo4j_graph":

                items.extend(self._query_neo4j([query], "agentic", context.visited_files))

            elif tool_name == "ts_chain":

                # query should be a file path seed

                items.extend(self._follow_ts_chains([query], context.visited_files))

            elif tool_name == "java_chain":

                items.extend(self._follow_java_chains([query], context.visited_files))

            elif tool_name == "python_chain":

                items.extend(self._follow_python_chains([query], context.visited_files))

            elif tool_name == "css_chain":

                items.extend(self._follow_css_chains([query], context.visited_files))

            elif tool_name == "config_chain":

                items.extend(self._follow_config_chains([query], context.visited_files))

            elif tool_name == "layout_chain":

                items.extend(self._follow_layout_chains([query], context.visited_files))

            elif tool_name == "semantic_rag":

                items.extend(self._query_semantic([query], "agentic", context.visited_files))

            elif tool_name == "i18n_chain":

                items.extend(self._trace_i18n_chain(query, context.visited_files))

            elif tool_name == "git_history":

                items.extend(self._query_git_history(query, context.visited_files))

            elif tool_name == "call_sites":

                items.extend(self._query_call_sites(query, context.visited_files))

            elif tool_name == "lsp":

                items.extend(self._query_lsp(query, context.visited_files))

            elif tool_name == "cooccurrence":

                if self.repo_search:
                    _terms = [t for t in re.split(r"[,\s]+", query) if t and t.upper() != "AND"]
                    if len(_terms) >= 2:
                        for r in self.repo_search.search_cooccurrence(_terms):
                            if r.file_path not in context.visited_files:
                                _cls = getattr(r, "classification", "code")
                                _mult = _HIT_CLASS_MULTIPLIER.get(_cls, 1.0)
                                items.append(EvidenceItem(
                                    provider="cooccurrence", strength="strong" if _mult == 1.0 else "medium",
                                    details=f"agentic_cooccurrence_{_cls}",
                                    file_path=r.file_path,
                                    evidence_type="usage",
                                    content_snippet=(
                                        f"[cooccurrence:{r.resolved_query}] [{_cls}] "
                                        f"line {r.line_number}: {r.matched_text}"
                                    )[:400],
                                    relevance_score=round(min(float(r.confidence) * _mult, 1.0), 3),
                                    hypothesis_id="agentic",
                                ))
                    else:
                        # Fewer than 2 usable terms — degrade to a plain literal search
                        items.extend(self._execute_tool("ripgrep", query, context))

            else:

                logger.warning(f"[AgenticLoop] Unknown tool: {tool_name}")

        except Exception as e:

            logger.warning(f"[AgenticLoop] Tool {tool_name} failed: {e}")

        # Track visited files

        for item in items:

            context.visited_files.add(item.file_path)

        logger.info(f"  🔧 Tool [{tool_name}] query={query!r} → {len(items)} file(s)")

        return items

    def _query_git_history(self, query: str, visited: Set[str]) -> List[EvidenceItem]:
        """git_history tool: three complementary history passes, zero LLM tokens.

        1. Pickaxe (`git log -S<q>`) — files whose CONTENT changed together
           with the query string.
        2. Commit messages (`git log -F --grep=<q>`) — ticket IDs and feature
           names written by developers; reveals the file group built for a
           feature even when the feature name never appears in code.
        3. Co-change partners — for the top-ranked files, the other files
           repeatedly committed together with them (structural coupling the
           first two passes cannot see).
        """
        import subprocess
        items: List[EvidenceItem] = []
        q = (query or "").strip()
        if len(q) < 4 or not getattr(self, "_workspace", None):
            return []
        ws = self._workspace
        try:
            proc = subprocess.run(
                ["git", "log", f"-S{q}", "--name-only", "--pretty=format:", "-n", "40"],
                cwd=str(ws), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=20,
            )
            proc_msg = subprocess.run(
                ["git", "log", "-F", f"--grep={q}", "--name-only", "--pretty=format:", "-n", "40"],
                cwd=str(ws), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=20,
            )
        except Exception as exc:
            logger.debug(f"[git_history] unavailable: {exc}")
            return []
        counts: Dict[str, Dict[str, int]] = {}
        for proc_i, pass_name in ((proc, "pickaxe"), (proc_msg, "msg")):
            if getattr(proc_i, "returncode", 1) != 0:
                continue
            for line in (proc_i.stdout or "").splitlines():
                fp = line.strip().replace("\\", "/")
                if not fp or fp.endswith(".aviator"):
                    continue
                counts.setdefault(fp, {"pickaxe": 0, "msg": 0})[pass_name] += 1
        if not counts:
            logger.debug(f"[git_history] git log returned nothing for {q!r}")
            return []
        ranked = sorted(counts.items(), key=lambda kv: -(kv[1]["pickaxe"] + kv[1]["msg"]))[:15]
        for fp, c in ranked:
            n_total = c["pickaxe"] + c["msg"]
            if not (ws / fp).exists():
                continue  # deleted/renamed file — cannot be modified
            if c["pickaxe"] and c["msg"]:
                src_tag = "git_-S+msg"
            elif c["msg"]:
                src_tag = "git_grep_msg"
            else:
                src_tag = "git_log_-S"
            items.append(EvidenceItem(
                provider="git_history",
                strength="medium" if n_total < 3 else "strong",
                details=f"{src_tag}: '{q}' touched in {n_total} commit(s)",
                file_path=fp,
                evidence_type="usage",
                content_snippet=(
                    f"[{src_tag}:{q!r}] {n_total} commit(s) "
                    f"(pickaxe={c['pickaxe']}, message={c['msg']})"
                ),
                relevance_score=min(0.75 + 0.05 * (n_total - 1), 0.9),
                hypothesis_id="agentic",
            ))

        # ── Pass 3: co-change partners of the top-ranked files ────────────
        # One pass over the last 200 commits, parsed into per-commit blocks:
        # a commit's FULL file list is its co-change group.  (A pathspec-
        # scoped `git log --name-only -- <file>` filters the file list DOWN
        # to that file, hiding the very partners we are looking for.)
        cochange: Dict[str, int] = {}
        try:
            proc3 = subprocess.run(
                ["git", "log", "--name-only", "--pretty=format:@@@%H", "-n", "200"],
                cwd=str(ws), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=25,
            )
        except Exception:
            proc3 = None
        if proc3 is not None and proc3.returncode == 0:
            seed_files = {fp for fp, _c in ranked[:3]}
            for block in (proc3.stdout or "").split("@@@")[1:]:
                lines = [l.strip().replace("\\", "/") for l in block.splitlines() if l.strip()]
                touched = set(lines[1:])
                if not touched or not (touched & seed_files):
                    continue
                for other in touched:
                    if other not in seed_files:
                        cochange[other] = cochange.get(other, 0) + 1
        added = 0
        for other, n in sorted(cochange.items(), key=lambda kv: -kv[1]):
            if added >= 6:
                break
            if other in counts or other in visited or not (ws / other).exists():
                continue
            items.append(EvidenceItem(
                provider="git_history", strength="medium", details="git_cochange",
                file_path=other,
                evidence_type="usage",
                content_snippet=f"[cochange:{q!r}] committed together {n} time(s) with '{ranked[0][0]}'",
                relationship="co_changed",
                relevance_score=0.62,
                hypothesis_id="agentic",
            ))
            added += 1
        logger.debug(f"  git_history: {len(items)} file(s) for {q!r} (pickaxe+msg+cochange)")
        return items

    def _query_call_sites(self, query: str, visited: Set[str]) -> List[EvidenceItem]:
        """call_sites tool: deterministic caller discovery for a function/method name.

        Resolves the symbol to its definition (SQLite), then searches for the
        invocation pattern `\\bname(`` across the live repository.  Catches
        callers that graph edges miss (dynamic dispatch, template bindings,
        event handlers).  Zero LLM tokens.
        """
        items: List[EvidenceItem] = []
        name = (query or "").strip()
        if len(name) < 4 or not self.repo_search:
            return []
        # Optional definition context (best-effort; call-site search works without it)
        def_paths: List[str] = []
        if self.localizer and self.localizer.sqlite_store:
            try:
                def_paths = [
                    r[0] for r in self.localizer.sqlite_store._conn.execute(
                        "SELECT DISTINCT path FROM symbols WHERE name = ? LIMIT 5", (name,)
                    ).fetchall() if r[0]
                ]
            except Exception:
                def_paths = []
        try:
            pattern = rf"\b{re.escape(name)}\s*\("
            hits = self.repo_search.search_regex(pattern, max_results=25)
        except Exception as exc:
            logger.debug(f"[call_sites] search failed for {name!r}: {exc}")
            return []
        for hit in hits:
            fp = hit.file_path
            if not fp or fp in visited:
                continue
            is_definition = fp in def_paths
            items.append(EvidenceItem(
                provider="call_sites",
                strength="strong" if not is_definition else "medium",
                details=f"call_site:{name}" + (" (definition file)" if is_definition else ""),
                file_path=fp,
                evidence_type="relationship",
                content_snippet=f"[call_site:{name}] line {hit.line_number}: {hit.matched_text[:150]}",
                relationship="called_by",
                relevance_score=0.90 if not is_definition else 0.70,
                symbol_name=name,
            ))
        logger.debug(f"  call_sites: {len(items)} caller file(s) for {name!r}")
        return items


    @staticmethod
    def _component_stem(path: str) -> str:
        stem = Path(path).stem.lower()
        for suffix in (".component", ".container", ".present", ".modal", ".page", ".view"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        return stem

    @staticmethod
    def _diminishing_returns_stop_reason(
        plateau_count: int,
        no_change_streak: int,
        n_relevant_files: int,
    ) -> Optional[str]:
        """Stop reason when discovery is no longer paying, else None.

        Deep-path bounding: a plateau is two consecutive sufficiency checks
        whose coverage improved < 0.05; a no-change streak is two consecutive
        iterations with zero new artifacts/facts/methods/relationships.
        Never stops when nothing relevant has been found yet — with zero
        evidence the loop must keep searching (and ultimately ask_user).
        """
        if n_relevant_files <= 0:
            return None
        if plateau_count >= 2:
            return f"coverage plateaued for {plateau_count} consecutive checks"
        if no_change_streak >= 2:
            return f"no new information for {no_change_streak} consecutive iterations"
        return None

    def _companion_pair_proven(
        self,
        ticket,
        evidence: Dict[str, dict],
        requirements: Any = None,
    ) -> Tuple[bool, str]:
        """Zero-token stop condition: a relevant controller + template pair
        (same component stem) that also matches the ticket's compound anchor.
        When this holds, further exploratory discovery turns burn tokens and
        minutes without adding accuracy — the discovery job is done.

        MULTI-COMPONENT SAFEGUARD:
        If the ticket explicitly targets multiple distinct components (e.g. x and y),
        all referenced components must have proven companion pairs before
        short-circuiting; otherwise, discovery continues to find remaining components.
        """
        import re
        from types import SimpleNamespace
        relevant = [
            fp for fp, v in (evidence or {}).items()
            if isinstance(v, dict) and v.get("relevant")
        ]
        ts_files = [
            f for f in relevant
            if f.endswith((".ts", ".tsx")) and ".spec." not in f and ".module" not in f
        ]
        html_files = [f for f in relevant if f.endswith(".html")]

        proven_pairs: List[Tuple[str, str, str]] = []
        proven_stems: Set[str] = set()

        for ts in ts_files:
            ts_stem = self._component_stem(ts)
            if not ts_stem:
                continue
            matched_html = None
            for html in html_files:
                if self._component_stem(html) == ts_stem:
                    matched_html = html
                    break

            # If HTML is not yet in evidence, check if it exists on disk
            if not matched_html:
                try:
                    from ticket_to_code.intelligence.contracts.companion_resolver import PolyglotCompanionResolver
                    ws_path = str(self._workspace) if getattr(self, "_workspace", None) else None
                    disk_comps = PolyglotCompanionResolver.resolve_all_companions(ts, workspace_path=ws_path)
                    for c in disk_comps:
                        if c.endswith(".html") and self._component_stem(c) == ts_stem:
                            matched_html = c
                            # Auto-inject companion into evidence
                            if isinstance(evidence, dict) and c not in evidence:
                                evidence[c] = {
                                    "relevant": True,
                                    "source": "companion",
                                    "provider": "companion",
                                    "relevance_score": 0.95,
                                }
                            break
                except Exception as _comp_e:
                    logger.debug(f"disk companion lookup failed: {_comp_e}")

            if matched_html:
                shims = [
                    SimpleNamespace(file_path=ts, content_snippet="", symbol_name=""),
                    SimpleNamespace(file_path=matched_html, content_snippet="", symbol_name=""),
                ]
                proven, why = self._is_primary_anchor_proven(ticket, shims)
                if proven:
                    proven_pairs.append((ts, matched_html, why))
                    proven_stems.add(ts_stem)

        if not proven_pairs:
            return False, ""

        # Check if the ticket explicitly references multiple distinct components
        ticket_text = f"{getattr(ticket, 'title', '')} {getattr(ticket, 'description', '')}"
        if requirements:
            f_reqs = getattr(requirements, "functional_requirements", []) or []
            t_reqs = getattr(requirements, "technical_requirements", []) or []
            ticket_text += " " + " ".join(str(r) for r in f_reqs + t_reqs)

        explicit_components = set(re.findall(r"\b([a-z0-9-]+)\.component\b", ticket_text, re.IGNORECASE))
        pascal_components = set()
        for comp in re.findall(r"\b([A-Z][a-zA-Z0-9]+)Component\b", ticket_text):
            kebab = re.sub(r'(?<!^)(?=[A-Z])', '-', comp).lower()
            pascal_components.add(kebab)

        all_referenced_components = {
            self._component_stem(c) for c in (explicit_components | pascal_components)
            if c and c not in ("base", "shared", "app")
        }

        # If the ticket mentions multiple distinct components, ensure all are proven
        if len(all_referenced_components) > 1:
            missing_components = all_referenced_components - proven_stems
            if missing_components:
                logger.info(
                    f"  ⏳ Multi-component ticket targets ({', '.join(sorted(all_referenced_components))}), "
                    f"proven ({', '.join(sorted(proven_stems))}) — awaiting remaining: {', '.join(sorted(missing_components))}"
                )
                return False, (
                    f"Ticket targets multiple components ({', '.join(sorted(all_referenced_components))}), "
                    f"awaiting remaining: {', '.join(sorted(missing_components))}"
                )

        first_ts, first_html, first_why = proven_pairs[0]
        if len(proven_pairs) > 1:
            names = ", ".join(f"{Path(t).name}+{Path(h).name}" for t, h, _ in proven_pairs)
            return True, f"all {len(proven_pairs)} companion pairs proven ({names})"
        return True, (
            f"companion pair {Path(first_ts).name} + {Path(first_html).name} proven ({first_why})"
        )

    def _query_lsp(self, query: str, visited: Set[str]) -> List[EvidenceItem]:
        """lsp tool (#18): cross-file definition + reference discovery.

        Exposes the lsp_client language-symbol layer directly to the loop:
          • Symbol query  ("MemberService") — the defining file is resolved
            via the SQLite symbol index (extractor-verified when ambiguous),
            the class members block is extracted with the per-language
            extractors (tsserver/tsc/regex ladder; javalang for Java; stdlib
            ast for Python), and every file referencing the symbol is found.
          • File query    ("src/app/x.service.ts") — the class members block
            for that file.

        Zero LLM tokens; every extractor degrades to regex parsing when no
        compiler tooling is present.
        """
        items: List[EvidenceItem] = []
        q = (query or "").strip()
        if len(q) < 3 or not getattr(self, "_workspace", None):
            return []
        try:
            from ticket_to_code.agents.lsp_client import (
                WorkspaceSymbolIndex as LSPSymbolIndex,
            )
        except Exception as exc:
            logger.debug(f"[lsp] lsp_client unavailable: {exc}")
            return []

        ws_str = str(self._workspace)
        cached = getattr(self, "_lsp_symbol_index", None)
        if cached is not None and cached[0] == ws_str:
            index = cached[1]
        else:
            try:
                index = LSPSymbolIndex(ws_str)
            except Exception as exc:
                logger.debug(f"[lsp] index build failed: {exc}")
                return []
            self._lsp_symbol_index = (ws_str, index)

        def _members_item(rel_path: str, strength: str, score: float):
            try:
                members = index.get_members_for_file(rel_path)
            except Exception as exc:
                logger.debug(f"[lsp] member extraction failed for {rel_path}: {exc}")
                members = None
            if not members:
                return None
            snippet = (
                members.to_prompt_block()
                if hasattr(members, "to_prompt_block") else str(members)
            )
            return EvidenceItem(
                provider="lsp", strength=strength, details="lsp_definition",
                file_path=rel_path,
                evidence_type="definition",
                content_snippet=snippet[:900],
                relationship="defines",
                relevance_score=score,
                symbol_name=getattr(members, "class_name", "") or q,
                hypothesis_id="agentic",
            )

        # File mode: query names a repo file → members block directly
        if Path(q).suffix and (self._workspace / q).exists():
            item = _members_item(q, "strong", 0.92)
            if item:
                items.append(item)
            return items

        # Symbol mode: definition via SQLite → extractor verification
        def_path = None
        if self.localizer and self.localizer.sqlite_store:
            try:
                rows = self.localizer.sqlite_store._conn.execute(
                    "SELECT DISTINCT path FROM symbols WHERE name = ? LIMIT 5",
                    (q,),
                ).fetchall()
                cands = [r[0] for r in rows if r[0]]
                if len(cands) == 1:
                    def_path = cands[0]
                elif len(cands) > 1:
                    # Ambiguous: keep the candidate whose extractor actually
                    # declares a class with this name
                    for c in cands[:3]:
                        members = index.get_members_for_file(c)
                        if members and getattr(members, "class_name", "") == q:
                            def_path = c
                            break
                    def_path = def_path or cands[0]
            except Exception as exc:
                logger.debug(f"[lsp] symbol lookup failed: {exc}")

        if def_path and def_path not in visited:
            item = _members_item(def_path, "strong", 0.90)
            if item:
                items.append(item)
        elif not def_path and self.repo_search:
            # No symbol index — probe the files that contain the name and
            # keep the first one the extractor confirms as its class
            try:
                hits = self.repo_search.search_literal(
                    q, extensions={".ts", ".tsx", ".java", ".py"}, max_results=3
                )
            except Exception:
                hits = []
            for h in hits:
                members = index.get_members_for_file(h.file_path)
                if members and getattr(members, "class_name", "") == q:
                    def_path = h.file_path
                    items.append(EvidenceItem(
                        provider="lsp", strength="medium", details="lsp_definition",
                        file_path=h.file_path,
                        evidence_type="definition",
                        content_snippet=(
                            members.to_prompt_block() if hasattr(members, "to_prompt_block") else str(members)
                        )[:900],
                        relationship="defines",
                        relevance_score=0.80,
                        symbol_name=q,
                        hypothesis_id="agentic",
                    ))
                    break

        # References: every other file that mentions the symbol
        if self.repo_search:
            try:
                ref_hits = self.repo_search.search_regex(
                    rf"\b{re.escape(q)}\b", max_results=30
                )
            except Exception as exc:
                logger.debug(f"[lsp] reference search failed: {exc}")
                ref_hits = []
            for rh in ref_hits:
                if rh.file_path in visited or rh.file_path == def_path:
                    continue
                items.append(EvidenceItem(
                    provider="lsp", strength="medium", details="lsp_reference",
                    file_path=rh.file_path,
                    evidence_type="usage",
                    content_snippet=f"[lsp_ref:{q}] line {rh.line_number}: {rh.matched_text[:150]}",
                    relationship="references",
                    relevance_score=0.82,
                    symbol_name=q,
                    hypothesis_id="agentic",
                ))
        logger.debug(f"  lsp: {len(items)} item(s) for {q!r}")
        return items

    def invalidate_caches(self) -> None:
        """Unified cache invalidation across ALL discovery layers.

        RepositorySearchEngine owns a cached file index; SQLite symbols/edges
        and the graph store are prebuilt services — they expose no live
        refresh, so any connection-level caches on them are dropped here when
        the API allows.  Call after any write phase (e.g. code generation) and
        at the start of every collect() run.
        """
        if self.repo_search is not None and hasattr(self.repo_search, "invalidate_index"):
            self.repo_search.invalidate_index()
        # Hook for future live-refreshable stores: anything with a refresh()
        # contract gets invalidated here in one place.
        for store_name in ("sqlite_store", "neo4j_store"):
            store = getattr(getattr(self, "localizer", None), store_name, None)
            if store is not None and hasattr(store, "refresh") and callable(store.refresh):
                try:
                    store.refresh()
                except Exception as exc:
                    logger.debug(f"[invalidate_caches] {store_name}.refresh() failed: {exc}")
        logger.debug("[invalidate_caches] All discovery caches invalidated")

    def _verify_single_file(

        self,

        ticket: "ValueEdgeTicket",

        file_item: EvidenceItem,

        existing_evidence: Dict[str, dict],

    ) -> dict:

        """

        Verify a single file via SemanticVerificationAgent.

        Returns dict matching the shape consumers expect:

            {"relevant": bool, "reason": str,

             "file_path": str, "decision": "include"|"exclude",

             "semantic_relevance_score": float}

        The `decision` field uses the exact vocabulary the 4 downstream

        consumers read: "include" or "exclude" (not True/False).

        """

        from ticket_to_code.utils.file_filters import is_ignored_path
        if file_item.file_path and is_ignored_path(file_item.file_path, workspace_path=getattr(self, "_workspace", None)):
            logger.info(f"  [verify_single] ⏭️ Auto-excluding build artifact/ignored path: {file_item.file_path}")
            return {
                "relevant": False,
                "reason": "Build artifact, compiled chunk, or ignored path",
                "file_path": file_item.file_path,
                "decision": "exclude",
                "semantic_relevance_score": 0.0,
                "category": "exclude",
            }

        if not hasattr(self, "_semantic_verifier") or self._semantic_verifier is None:
            # No verifier available — include by default
            return {
                "relevant": True,
                "reason": "No verifier available — included by default",
                "file_path": file_item.file_path,

                "decision": "include",

                "semantic_relevance_score": file_item.relevance_score,

                "category": "modify",

            }

        try:

            result = self._semantic_verifier.verify_single(

                ticket=ticket,

                file_item=file_item,

                existing_evidence=existing_evidence,

            )

            # verify_single returns SemanticVerificationResult with

            # .decision ("include"/"exclude"), .reason, .file_path,

            # .semantic_relevance_score

            is_relevant = result.decision == "include"

            return {

                "relevant": is_relevant,

                "reason": result.reason,

                "file_path": result.file_path,

                "decision": result.decision,

                "semantic_relevance_score": result.semantic_relevance_score,

                "category": getattr(result, "category", "modify"),

                "verification_status": getattr(result, "verification_status", "completed"),

                

            }

        except Exception as e:

            # Separate infrastructure failures from evidence quality.
            # The exception is logged but NEVER put in the 'reason' field,
            # because the planner interprets 'reason' as a conclusion about the file.
            from ticket_to_code.runtime.run_context import BudgetExceeded

            if isinstance(e, BudgetExceeded):
                _verification_status = "not_completed_budget"
                _reason = "Included (verification not completed — infrastructure limit)"
            else:
                _verification_status = "not_completed_error"
                _reason = "Included (verification not completed — infrastructure error)"

            logger.warning(
                f"[AgenticLoop] verify_single failed for {file_item.file_path}: "
                f"{type(e).__name__}: {e}"
            )

            return {

                "relevant": True,

                "reason": _reason,

                "file_path": file_item.file_path,

                "decision": "include",

                "semantic_relevance_score": file_item.relevance_score,

                "verification_status": _verification_status,

                "category": "modify",

            }

    def _build_need_more_info_return(

        self,

        question: str,

        evidence: Dict[str, dict],

        all_items: List[EvidenceItem],

        tried: List[dict],

    ) -> Tuple[List[EvidenceItem], float, str]:

        """

        Build the 3-tuple return for the ask_user exit.

        Returns (partial_evidence, 0.0, clarification_question).

        The third element signals the caller to set

        state["status"] = "need_more_info" (matching routes.py line 1186).

        """

        # Return whatever relevant items we found (may be empty)

        relevant = [

            item for item in all_items

            if evidence.get(item.file_path, {}).get("relevant")

        ]

        logger.info(

            f"  ❓ Agentic loop → ask_user: {len(relevant)} partial items, "

            f"question: {question[:100]}"

        )

        return relevant, 0.0, question

    # =========================================================================

    # SOURCE 1: SEMANTIC SEARCH (RAGEngine)

    # =========================================================================

    # Semantic-vector scores live on their own scale (cosine similarity, some
    # backends 0-1, others 0-100).  Calibrate every raw chunk score onto the
    # shared evidence scale so the best semantic hit lands ~0.78 — below
    # exact literal/symbol matches (0.85+) — and ranking never promotes a
    # fuzzy match over an exact one.
    @staticmethod
    def _calibrate_rag_score(raw_score) -> float:
        try:
            s = float(raw_score)
        except (TypeError, ValueError):
            return 0.50
        if s > 1.0:      # 0-100 backend scale
            s = s / 100.0
        s = max(0.0, min(s, 1.0))
        return round(0.35 + 0.43 * s, 3)

    def _query_semantic(

        self,

        queries: List[str],

        hypothesis_id: str,

        visited: Set[str],

        run_ctx=None,

    ) -> List[EvidenceItem]:

        items: List[EvidenceItem] = []

        for query in queries[:6]:  # widened: surface UI/.scss/component files buried below backend

            try:

                chunks = self.rag_engine.retrieve_context(

                    query=query,

                    max_results=12,   # widened from 5 — breadth needed here, ranking de-noises

                    document_types=None,

                    schema="both",

                )

                for chunk in chunks:

                    fp = chunk.get("file_path", "")

                    if not fp or fp in visited:

                        continue

                    items.append(EvidenceItem(

                        provider="vector", strength="medium", details="semantic", file_path=fp,

                        evidence_type="usage",

                        content_snippet=str(chunk.get("content", ""))[:400],

                        relevance_score=self._calibrate_rag_score(chunk.get("score", 0.5)),

                        hypothesis_id=hypothesis_id,

                    ))

                if not chunks and len(query.split()) > 2:
                    # Zero-hit retry: simplified tail query (2 most significant words)
                    _simplified = " ".join(query.split()[-2:])
                    if _simplified.lower() != query.strip().lower():
                        try:
                            chunks = self.rag_engine.retrieve_context(
                                query=_simplified,
                                max_results=12,
                                document_types=None,
                                schema="both",
                            )
                            logger.info(
                                f"  [semantic_rag] zero-hit retry: {query!r} → {_simplified!r} "
                                f"({len(chunks)} chunk(s))"
                            )
                            for chunk in chunks:
                                fp = chunk.get("file_path", "")
                                if not fp or fp in visited:
                                    continue
                                items.append(EvidenceItem(
                                    provider="vector", strength="medium", details="semantic_retry",
                                    file_path=fp,
                                    evidence_type="usage",
                                    content_snippet=str(chunk.get("content", ""))[:400],
                                    relevance_score=self._calibrate_rag_score(chunk.get("score", 0.5)),
                                    hypothesis_id=hypothesis_id,
                                ))
                        except Exception as retry_exc:
                            logger.warning(f"  [semantic_rag] retry failed for {_simplified!r}: {retry_exc}")

            except Exception as exc:

                logger.warning(f"[semantic_rag] Query failed for '{query}': {exc}")

                if run_ctx:
                    try:
                        from ticket_to_code.runtime.run_context import HealthLevel
                        run_ctx.degrade("rag", HealthLevel.FAILED, f"RAG query failed: {exc}")
                    except Exception:
                        pass

        return items

    def _hypothesis_to_semantic_queries(

        self, hypotheses: List[Any]

    ) -> List[str]:

        """Convert hypothesis descriptions into concrete semantic search queries.

        Each InvestigationHypothesis carries:

          - hypothesis  (natural language description)

          - queries     (LLM-generated semantic queries)

          - symbols     (class/method names)

          - anchors     (domain/page nouns)

        We collect ALL of these as RAG queries so the evidence loop can find

        conceptually related code — not just exact keyword matches.

        Example:

          Hypothesis: "Shared state contamination between tabs"

          → Queries: [

              "state management between component tabs",

              "BehaviorSubject shared service",

              "component state not reset on navigation",

            ]

        """

        queries: List[str] = []

        for h in (hypotheses or []):

            # 1. Use pre-built semantic queries from LLM

            for q in getattr(h, "queries", []):

                if q and len(q) > 5:

                    queries.append(q)

            # 2. Use the hypothesis description itself as a semantic query

            desc = getattr(h, "hypothesis", "") or ""

            if len(desc) > 10:

                queries.append(desc[:200])

            # 3. Use symbols as more targeted queries

            for sym in getattr(h, "symbols", []):

                if sym and len(sym) > 3:

                    queries.append(sym)

            # 4. Use anchors (domain nouns from ticket)

            for anchor in getattr(h, "anchors", []):

                if anchor and len(anchor) > 3:

                    queries.append(anchor)

        # Deduplicate preserving order, cap at 12 queries total

        seen: set = set()

        unique: List[str] = []

        for q in queries:

            key = q.strip().lower()

            if key not in seen:

                seen.add(key)

                unique.append(q.strip())

        return unique[:12]

    # =========================================================================

    # SOURCE 2: SQLITE FTS — string literal search

    # =========================================================================

    def _get_dynamic_threshold(self, term: str, base_threshold: int = 15) -> int:

        """Calculate specificity threshold based on entropy to avoid aborting valid broad searches."""

        import re

        if re.search(r'\d+\.\d+', term):

            return max(100, base_threshold)  # Version numbers are high entropy, allow broad sweep

        elif len(term) > 15:

            return max(50, base_threshold)   # Long phrases are highly specific

        return base_threshold

    def _query_sqlite_fts(

        self,

        literals: List[str],

        hypothesis_id: str,

        visited: Set[str],

    ) -> List[EvidenceItem]:

        if not self.localizer.sqlite_store or not literals:

            return []

        items: List[EvidenceItem] = []

        conn = self.localizer.sqlite_store._conn

        for literal in literals[:6]:

            try:

                # Try FTS5 table first, fall back to LIKE scan

                rows = self._fts_or_like(conn, literal)

                for path, snippet, is_broad in rows:

                    if not path or path in visited:

                        continue

                    items.append(EvidenceItem(

                        provider="literal", strength="weak" if is_broad else "medium", details="sqlite_fts_broad" if is_broad else "sqlite_fts", file_path=path,

                        evidence_type="definition",

                        content_snippet=snippet[:400],

                        relevance_score=0.55 if is_broad else 0.75,

                        hypothesis_id=hypothesis_id,

                    ))

            except Exception as exc:

                logger.debug(f"SQLite FTS for '{literal}' failed: {exc}")

        return items

    def _fts_or_like(self, conn, literal: str) -> List[Tuple[str, str, bool]]:

        """

        Search the symbols_fts FTS5 table for ``literal``; fall back to a LIKE

        scan on the symbols table when FTS is unavailable.

        Returns list of (path, snippet, is_broad). Overly generic literals are NEVER
        silently dropped — they are flagged is_broad so the caller demotes them
        (same rank-and-keep philosophy as the ripgrep cap).

        """

        # ── FTS5 via symbols_fts ──────────────────────────────────────────────

        try:

            # Escape FTS5 special characters in the literal safely
            cleaned = literal.strip()
            if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
                cleaned = cleaned[1:-1]
            fts_literal = cleaned.replace('"', '""')
            # Wrap in double quotes for safe phrase search (protects hyphens '-' from being treated as NOT)
            fts_query = f'"{fts_literal}"'

            

            # Agentic Specificity Check: Abort if literal is too generic

            threshold = self._get_dynamic_threshold(literal)

            count_row = conn.execute(

                "SELECT COUNT(*) FROM symbols_fts WHERE symbols_fts MATCH ?",

                (fts_query,),

            ).fetchone()

            

            is_broad = bool(count_row and count_row[0] > threshold)
            if is_broad:
                logger.warning(
                    f"[EvidenceLoop] Broad literal kept demoted: {threshold=}, {count_row[0]} matches"
                )

            rowids = conn.execute(

                "SELECT rowid FROM symbols_fts WHERE symbols_fts MATCH ? LIMIT 10",

                (fts_query,),

            ).fetchall()

            if rowids:

                placeholders = ",".join("?" * len(rowids))

                rows = conn.execute(

                    f"SELECT DISTINCT path, name FROM symbols "

                    f"WHERE rowid IN ({placeholders})",

                    [r[0] for r in rowids],

                ).fetchall()

                if rows:

                    return [(r[0], r[1], is_broad) for r in rows]

        except Exception as e:

            pass  # FTS unavailable — fall through to LIKE

        # ── Symbol-name LIKE scan (fallback) ──────────────────────────────────

        try:

            like_query = f"%{literal.lower()}%"

            

            # Agentic Specificity Check: Abort if literal is too generic

            threshold = self._get_dynamic_threshold(literal)

            count_row = conn.execute(

                "SELECT COUNT(*) FROM symbols WHERE LOWER(name) LIKE ?",

                (like_query,),

            ).fetchone()

            

            is_broad = bool(count_row and count_row[0] > threshold)
            if is_broad:
                logger.warning(
                    f"[EvidenceLoop] Broad literal kept demoted: {threshold=}, {count_row[0]} matches"
                )

            rows = conn.execute(

                "SELECT DISTINCT path, name FROM symbols "

                "WHERE LOWER(name) LIKE ? LIMIT 10",

                (like_query,),

            ).fetchall()

            return [(r[0], r[1], is_broad) for r in rows]

        except Exception:

            pass

        return []

    # =========================================================================

    # SOURCE 3: SQLITE SYMBOL LOOKUP

    # =========================================================================

    # Kind priority for symbols that share a name: type definitions are the
    # most actionable evidence, then callables, then data members.  Used to
    # order exact-match definitions deterministically instead of SQL row
    # order — the most valuable definition always carries the top score.
    _SYMBOL_KIND_PRIORITY = {
        "class": 0, "interface": 1, "enum": 2, "component": 3, "injectable": 3,
        "service": 3, "function": 4, "method": 5,
        "property": 6, "field": 6, "variable": 7, "constant": 7, "import": 8,
    }

    @staticmethod
    def _fuzzy_symbol_rows(conn, name: str, limit: int = 8) -> List[Tuple[str, str, str, str]]:
        """Bounded Levenshtein over the symbol table for typo tolerance.

        SQL pre-filters by a length window; the bounded edit distance (with
        early exit) runs in Python over at most 20k candidate names.
        """
        from ticket_to_code.agents.repository_search_engine import _bounded_levenshtein
        name_low = (name or "").lower()
        max_dist = 2 if len(name_low) >= 8 else 1
        try:
            rows = conn.execute(
                "SELECT DISTINCT name, path, kind, spring_stereotype FROM symbols "
                "WHERE LENGTH(name) BETWEEN ? AND ? LIMIT 20000",
                (max(1, len(name_low) - max_dist), len(name_low) + max_dist),
            ).fetchall()
        except Exception:
            return []
        scored = []
        for sym_name, path, kind, stereotype in rows:
            if not sym_name or not path:
                continue
            dist = _bounded_levenshtein(name_low, sym_name.lower(), max_dist)
            if dist <= max_dist:
                scored.append((dist, path, sym_name, kind, stereotype))
        scored.sort(key=lambda s: s[0])
        return [(p, n, k, st) for _d, p, n, k, st in scored[:limit]]

    def _query_sqlite_symbols(
        self,
        symbols: List[str],
        hypothesis_id: str,
        visited: Set[str],
    ) -> List[EvidenceItem]:
        if not self.localizer or not self.localizer.sqlite_store or not symbols:
            if self.repo_search and symbols:
                fallback_items: List[EvidenceItem] = []
                for sym in symbols[:8]:
                    sym_clean = sym.strip()
                    if not sym_clean or len(sym_clean) < 2:
                        continue
                    try:
                        safe_q = re.escape(sym_clean)
                        pattern = rf"\b(class|interface|enum|function|const|let|var|def)\s+{safe_q}\b|class\s+{safe_q}|interface\s+{safe_q}"
                        results = self.repo_search.search_regex(pattern, max_results=10)
                        if not results:
                            results = self.repo_search.search_filename(sym_clean)
                        if not results:
                            results = self.repo_search.search_regex(rf"\b{safe_q}\b", max_results=10)
                        for r in results:
                            if not getattr(r, "error", None) and r.file_path not in visited:
                                fallback_items.append(EvidenceItem(
                                    provider="literal",
                                    strength="medium",
                                    details="repo_symbol_fallback",
                                    file_path=r.file_path,
                                    evidence_type="definition",
                                    content_snippet=f"[symbol_fallback:{sym_clean}] line {getattr(r, 'line_number', 0)}: {getattr(r, 'matched_text', '')}"[:400],
                                    relevance_score=0.85,
                                    hypothesis_id=hypothesis_id,
                                    symbol_name=sym_clean,
                                ))
                    except Exception as _e:
                        logger.warning(f"Fallback symbol search failed for {sym_clean}: {_e}")
                return fallback_items
            return []

        items: List[EvidenceItem] = []
        conn = self.localizer.sqlite_store._conn

        for sym in symbols[:8]:
            try:
                sym_clean = sym.strip()
                if not sym_clean:
                    continue

                # ── Tier 1: Authoritative Exact Symbol Lookup ─────────────────
                exact_rows = conn.execute(
                    "SELECT DISTINCT path, name, kind, spring_stereotype "
                    "FROM symbols WHERE LOWER(name) = ? LIMIT 25",
                    (sym_clean.lower(),),
                ).fetchall()

                if exact_rows:
                    # Deterministic ordering when several definitions share
                    # the name: class/interface outrank method/function, which
                    # outrank fields — and the leading definition keeps the
                    # full exact-match score.
                    exact_rows = sorted(
                        exact_rows,
                        key=lambda r: self._SYMBOL_KIND_PRIORITY.get((r[2] or "").lower(), 9),
                    )
                    for idx, (path, name, kind, stereotype) in enumerate(exact_rows):
                        if not path or path in visited:
                            continue
                        items.append(EvidenceItem(
                            provider="literal",
                            strength="strong",
                            details="sqlite_symbol_exact",
                            file_path=path,
                            evidence_type="definition",
                            content_snippet=f"{kind}: {name}" + (f" [{stereotype}]" if stereotype else ""),
                            relevance_score=0.95 if idx == 0 else 0.88,
                            hypothesis_id=hypothesis_id,
                            symbol_name=name,
                        ))
                    # Exact lookup succeeded for this symbol; do not fall back
                    continue

                # ── Tier 2: Prefix / Substring Fallback (on 0 exact matches) ──
                threshold = self._get_dynamic_threshold(sym_clean)
                prefix_query = f"{sym_clean.lower()}%"

                count_row = conn.execute(
                    "SELECT COUNT(*) FROM symbols WHERE LOWER(name) LIKE ?",
                    (prefix_query,),
                ).fetchone()

                if count_row and count_row[0] > threshold:
                    logger.warning(
                        f"[EvidenceLoop] 🛑 ABORT: Symbol prefix '{sym_clean}' is too generic "
                        f"({count_row[0]} > {threshold} matches). Dropping to prevent noise."
                    )
                    continue

                fallback_rows = conn.execute(
                    "SELECT DISTINCT path, name, kind, spring_stereotype "
                    "FROM symbols WHERE LOWER(name) LIKE ? LIMIT 10",
                    (prefix_query,),
                ).fetchall()

                # If no prefix matches, try substring
                if not fallback_rows:
                    sub_query = f"%{sym_clean.lower()}%"
                    sub_count = conn.execute(
                        "SELECT COUNT(*) FROM symbols WHERE LOWER(name) LIKE ?",
                        (sub_query,),
                    ).fetchone()
                    if sub_count and sub_count[0] <= threshold:
                        fallback_rows = conn.execute(
                            "SELECT DISTINCT path, name, kind, spring_stereotype "
                            "FROM symbols WHERE LOWER(name) LIKE ? LIMIT 10",
                            (sub_query,),
                        ).fetchall()

                if not fallback_rows:
                    # ── Tier 3: bounded fuzzy match on symbol names ──────
                    # Typo tolerance (e.g. ticket says 'MbrService'): bounded
                    # Levenshtein over length-filtered symbol names.  Multiple
                    # hits surface as ambiguous, never auto-picked.
                    _fuzzy = self._fuzzy_symbol_rows(conn, sym_clean)
                    if _fuzzy:
                        _ambig = len(_fuzzy) > 1
                        _fuzzy_prefix = (
                            f"[symbol:{sym_clean} (FUZZY{' AMBIGUOUS' if _ambig else ''})] "
                        )
                        for path, name, kind, stereotype in _fuzzy:
                            if not path or path in visited:
                                continue
                            items.append(EvidenceItem(
                                provider="literal",
                                strength="weak",
                                details="sqlite_symbol_fuzzy",
                                file_path=path,
                                evidence_type="definition",
                                content_snippet=_fuzzy_prefix + f"{kind}: {name}",
                                relevance_score=0.55,
                                hypothesis_id=hypothesis_id,
                                symbol_name=name,
                            ))
                    elif self.repo_search:
                        try:
                            safe_q = re.escape(sym_clean)
                            pattern = rf"\b(class|interface|enum|function|const|let|var|def)\s+{safe_q}\b|class\s+{safe_q}|interface\s+{safe_q}"
                            res = self.repo_search.search_regex(pattern, max_results=10)
                            if not res:
                                res = self.repo_search.search_filename(sym_clean)
                            if not res:
                                res = self.repo_search.search_regex(rf"\b{safe_q}\b", max_results=10)
                            for r in res:
                                if not getattr(r, "error", None) and r.file_path not in visited:
                                    items.append(EvidenceItem(
                                        provider="literal",
                                        strength="medium",
                                        details="repo_symbol_fallback",
                                        file_path=r.file_path,
                                        evidence_type="definition",
                                        content_snippet=f"[symbol_fallback:{sym_clean}] line {getattr(r, 'line_number', 0)}: {getattr(r, 'matched_text', '')}"[:400],
                                        relevance_score=0.80,
                                        hypothesis_id=hypothesis_id,
                                        symbol_name=sym_clean,
                                    ))
                        except Exception as _e:
                            logger.warning(f"Fallback repo search on symbol failed for {sym_clean}: {_e}")
                    continue

                # Ambiguity handling: surface all fallback candidates as ambiguous rather than choosing one
                is_ambig = len(fallback_rows) > 1
                strength = "weak" if is_ambig else "medium"
                relevance = 0.60 if is_ambig else 0.80
                details = "sqlite_symbol_fallback_ambiguous" if is_ambig else "sqlite_symbol_fallback"
                ambig_prefix = f"[symbol:{sym_clean} (AMBIGUOUS)] " if is_ambig else f"[symbol:{sym_clean}] "

                for path, name, kind, stereotype in fallback_rows:
                    if not path or path in visited:
                        continue
                    items.append(EvidenceItem(
                        provider="literal",
                        strength=strength,
                        details=details,
                        file_path=path,
                        evidence_type="definition",
                        content_snippet=ambig_prefix + f"{kind}: {name}" + (f" [{stereotype}]" if stereotype else ""),
                        relevance_score=relevance,
                        hypothesis_id=hypothesis_id,
                        symbol_name=name,
                    ))

            except Exception as exc:
                logger.debug(f"Symbol query for '{sym}' failed: {exc}")

        return items

    # =========================================================================

    # SOURCE 4: NEO4J RELATIONSHIP QUERIES

    # =========================================================================

    def _query_neo4j(

        self,

        symbols: List[str],

        hypothesis_id: str = "agentic",

        visited: Optional[Set[str]] = None,

        run_ctx=None,

    ) -> List[EvidenceItem]:
        """neo4j_graph tool: relationship discovery in four directions.

        Implements what the catalog advertises — callers, callees, imports,
        and implementations — instead of only callers.  Falls back to the
        SQLite ``edges`` table when Neo4j is unavailable or returns nothing,
        so relationship discovery degrades instead of dying.
        """
        visited = visited if visited is not None else set()
        if not symbols:
            return []
        items: List[EvidenceItem] = []

        _QUERIES = [
            ("calls_caller", "relationship",
             """
                    MATCH (caller)-[:CALLS|DEPENDS_ON]->(target)
                    WHERE target.name = $name
                    RETURN caller.name as peer_name, caller.file_path as file_path
                    LIMIT 8
                    """,
             "calls", "callers of"),
            ("calls_callee", "relationship",
             """
                    MATCH (target)-[:CALLS|DEPENDS_ON]->(callee)
                    WHERE target.name = $name
                    RETURN callee.name as peer_name, callee.file_path as file_path
                    LIMIT 8
                    """,
             "callees", "called by"),
            ("imports", "import",
             """
                    MATCH (a)-[:IMPORTS]-(b)
                    WHERE a.name = $name OR b.name = $name
                    RETURN coalesce(b.name, a.name) as peer_name,
                           coalesce(b.file_path, a.file_path) as file_path
                    LIMIT 8
                    """,
             "imports", "imported by/imports"),
            ("implements", "implementation",
             """
                    MATCH (a)-[:IMPLEMENTS|EXTENDS]-(b)
                    WHERE a.name = $name OR b.name = $name
                    RETURN coalesce(b.name, a.name) as peer_name,
                           coalesce(b.file_path, a.file_path) as file_path
                    LIMIT 8
                    """,
             "implements", "implements/extended by"),
        ]

        for sym in symbols[:5]:
            got_any = False
            if self.localizer.neo4j_store:
                for shape, ev_type, cypher, rel_kind, phrase in _QUERIES:
                    try:
                        result = self.localizer.neo4j_store.query(cypher, {"name": sym})
                    except Exception as exc:
                        logger.debug(f"[neo4j_graph] {shape} query failed for '{sym}': {exc}")
                        if run_ctx:
                            try:
                                from ticket_to_code.runtime.run_context import HealthLevel
                                run_ctx.degrade("neo4j", HealthLevel.FAILED, f"Neo4j query failed: {exc}")
                            except Exception:
                                pass
                        continue
                    for record in (result or []):
                        path = record.get("file_path")
                        if not path or path in visited:
                            continue
                        got_any = True
                        items.append(EvidenceItem(
                            provider="graph", strength="medium", details="neo4j", file_path=path,
                            evidence_type=ev_type,
                            content_snippet=f"{phrase} {sym} (peer: {record.get('peer_name')})",
                            relationship=rel_kind,
                            relevance_score=0.72,
                            symbol_name=record.get("peer_name"),
                            hypothesis_id=hypothesis_id,
                        ))
            if not got_any:
                # ── SQLite edges fallback (graph down OR no Neo4j rows) ──
                items.extend(self._sqlite_edge_neighbors(sym, visited, hypothesis_id))
        return items

    def _sqlite_edge_neighbors(
        self,
        sym: str,
        visited: Set[str],
        hypothesis_id: str = "agentic",
    ) -> List[EvidenceItem]:
        """Relationship fallback over the SQLite edges table (no Neo4j needed)."""
        if not self.localizer or not self.localizer.sqlite_store:
            return []
        items: List[EvidenceItem] = []
        conn = self.localizer.sqlite_store._conn
        try:
            rows = conn.execute(
                "SELECT DISTINCT path, kind, dst_name FROM edges "
                "WHERE (src_name = ? OR dst_name = ?) "
                "AND kind IN ('calls', 'depends_on', 'imports', 'implements', 'extends') "
                "LIMIT 12",
                (sym, sym),
            ).fetchall()
        except Exception:
            try:
                rows = conn.execute(
                    "SELECT DISTINCT path, kind, dst_name FROM edges "
                    "WHERE dst_name = ? LIMIT 12",
                    (sym,),
                ).fetchall()
            except Exception:
                return []
        for path, kind, dst_name in rows:
            if not path or path in visited:
                continue
            items.append(EvidenceItem(
                provider="graph", strength="medium", details="sqlite_edges_fallback", file_path=path,
                evidence_type="relationship",
                content_snippet=f"[{kind}] {sym} (peer: {dst_name})",
                relationship=kind or "related",
                relevance_score=0.65,
                symbol_name=dst_name,
                hypothesis_id=hypothesis_id,
            ))
        return items

    # =========================================================================
    # SOURCE 4B: AST COMPONENT COMPANION BUNDLE (DISCOVERY ONLY)
    # =========================================================================

    def _resolve_ast_companion_bundle(
        self,
        seed: str,
        visited: Optional[Set[str]] = None,
    ) -> List[EvidenceItem]:
        """Resolve Angular structural component companions for a seed path or symbol.

        Discovery only:
        - Resolves seed to canonical component path.
        - Identifies companion quartet (.ts, .html, .scss, .spec.ts).
        - Returns existing companions on disk ONLY (does not fabricate non-existent files).
        - Distinguishes existing vs missing companions in provenance.
        - Preserves canonical repository-relative paths.
        - If seed symbol is ambiguous (matches multiple component files), surfaces ambiguity
          and stops rather than guessing.
        """
        if not seed or not seed.strip():
            return []

        visited = visited or set()
        clean_seed = seed.strip().replace("\\", "/")
        ws = Path(self._workspace) if hasattr(self, "_workspace") and self._workspace else Path.cwd()

        from ticket_to_code.agents.canonical_path import (
            canonical_repo_path,
            canonical_component_base,
            COMPONENT_EXTENSIONS,
        )

        canonical_seed_path: Optional[str] = None

        # Check if seed is already a component file path
        is_path_like = "/" in clean_seed or any(clean_seed.lower().endswith(ext) for ext in COMPONENT_EXTENSIONS)
        if is_path_like:
            canonical_seed_path = canonical_repo_path(clean_seed, ws)
            if not canonical_seed_path:
                canonical_seed_path = clean_seed.lstrip("/")
        else:
            # Seed is a symbol/class name (e.g. 'AddMembersComponent')
            res_path, is_ambig, candidates = self._resolve_symbol_to_component_path(clean_seed)
            if is_ambig:
                logger.info(f"[ast_companion_bundle] ⚠️ Ambiguous component symbol '{clean_seed}': {candidates}")
                return [EvidenceItem(
                    provider="structural_companion",
                    strength="weak",
                    details="agentic_ast_companion_ambiguous",
                    file_path="",
                    evidence_type="structural",
                    content_snippet=f"[ast_companion_bundle:AMBIGUOUS] Symbol '{clean_seed}' matches multiple components: {candidates}",
                    relevance_score=0.40,
                    hypothesis_id="agentic",
                )]
            if res_path:
                canonical_seed_path = res_path

        if not canonical_seed_path:
            return []

        # Derive canonical component base
        base = canonical_component_base(canonical_seed_path, ws)
        if not base:
            for ext in (".ts", ".tsx", ".html", ".scss", ".css"):
                if canonical_seed_path.lower().endswith(ext):
                    base = canonical_seed_path[:len(canonical_seed_path) - len(ext)]
                    break

        if not base:
            return []

        # Check all structural component extensions on disk
        existing_companions: List[str] = []
        missing_extensions: List[str] = []

        for ext in COMPONENT_EXTENSIONS:
            cand_rel = f"{base}{ext}"
            cand_abs = ws / cand_rel
            if cand_abs.exists():
                existing_companions.append(cand_rel)
            else:
                missing_extensions.append(ext)

        items: List[EvidenceItem] = []
        provenance_str = (
            f"[ast_companion_bundle:seed='{clean_seed}'] "
            f"existing={existing_companions}, missing={missing_extensions}"
        )

        # Extended structural siblings beyond the canonical quartet:
        # Angular module/routing files and Storybook stories co-own behavior.
        extended_siblings: List[str] = []
        _base_stem = Path(base).name
        for _extra in (
            f"{base}.module.ts",
            str(Path(base).parent / f"{_base_stem}.module.ts"),
            str(Path(base).parent / f"{_base_stem}.routing.ts"),
            str(Path(base).parent / f"{_base_stem}.routing.module.ts"),
            f"{base}.stories.ts",
        ):
            if (ws / _extra).exists() and _extra not in existing_companions:
                extended_siblings.append(_extra)
        existing_companions.extend(extended_siblings)

        # Differentiated confidence: the component class is the strongest
        # signal; its template nearly so; styles/spec/module less actionable.
        _COMPANION_SCORES = {
            ".ts": 0.90, ".tsx": 0.90, ".html": 0.88, ".scss": 0.85,
            ".css": 0.85, ".spec.ts": 0.60, ".module.ts": 0.75,
            ".routing.ts": 0.75, ".routing.module.ts": 0.75, ".stories.ts": 0.65,
        }

        for comp_rel in existing_companions:
            if comp_rel in visited:
                continue
            snippet = self._read_snippet(comp_rel) if hasattr(self, "_read_snippet") else ""
            if not snippet:
                snippet = f"{provenance_str}\nFile: {comp_rel}"
            else:
                snippet = f"{provenance_str}\n{snippet[:300]}"

            items.append(EvidenceItem(
                provider="structural_companion",
                strength="medium",
                details="agentic_ast_companion_bundle",
                file_path=comp_rel,
                evidence_type="structural",
                content_snippet=snippet,
                relevance_score=_COMPANION_SCORES.get(
                    ".spec.ts" if comp_rel.endswith(".spec.ts") else Path(comp_rel).suffix.lower(),
                    0.80
                ),
                hypothesis_id="agentic",
            ))

        return items

    def _resolve_symbol_to_component_path(
        self,
        symbol_name: str,
    ) -> tuple[Optional[str], bool, List[str]]:
        """Resolve a component symbol name to a unique canonical file path."""
        if not self.localizer or not self.localizer.sqlite_store or not symbol_name:
            return None, False, []

        from ticket_to_code.agents.canonical_path import canonical_repo_path, COMPONENT_EXTENSIONS
        ws = Path(self._workspace) if hasattr(self, "_workspace") and self._workspace else Path.cwd()

        try:
            conn = self.localizer.sqlite_store._conn
            rows = conn.execute(
                "SELECT DISTINCT path FROM symbols WHERE LOWER(name) = ?",
                (symbol_name.strip().lower(),),
            ).fetchall()

            candidates: List[str] = []
            for r in rows:
                p = r[0]
                if not p:
                    continue
                c_p = canonical_repo_path(p, ws) or p.replace("\\", "/")
                if any(c_p.lower().endswith(ext) for ext in COMPONENT_EXTENSIONS):
                    if c_p not in candidates:
                        candidates.append(c_p)
                elif c_p.endswith((".ts", ".tsx")) and c_p not in candidates:
                    candidates.append(c_p)

            if not candidates:
                return None, False, []

            if len(candidates) > 1:
                return None, True, candidates

            return candidates[0], False, candidates

        except Exception as e:
            logger.debug(f"[ast_companion_bundle] Symbol resolution error: {e}")
            return None, False, []

    def _resolve_symbol_seed_to_path(
        self,
        seed: str,
        valid_extensions: tuple[str, ...],
    ) -> tuple[Optional[str], bool, List[str]]:
        """Resolve a seed (path or symbol name) to a unique canonical file path.

        Returns:
            (canonical_path, is_ambiguous, candidate_paths)
        """
        if not seed or not seed.strip():
            return None, False, []

        clean = seed.strip().replace("\\", "/")
        ws = Path(self._workspace) if hasattr(self, "_workspace") and self._workspace else Path.cwd()
        from ticket_to_code.agents.canonical_path import canonical_repo_path

        # If clean has a valid extension or contains directory separators, test if it's already a path
        if any(clean.lower().endswith(ext) for ext in valid_extensions) or "/" in clean:
            c_p = canonical_repo_path(clean, ws)
            if c_p:
                return c_p, False, [c_p]
            if (ws / clean).exists():
                return clean.lstrip("/"), False, [clean.lstrip("/")]
            if any(clean.lower().endswith(ext) for ext in valid_extensions):
                return clean.lstrip("/"), False, [clean.lstrip("/")]

        # Otherwise, treat as symbol / class name
        if not self.localizer or not self.localizer.sqlite_store:
            return None, False, []

        try:
            conn = self.localizer.sqlite_store._conn
            rows = conn.execute(
                "SELECT DISTINCT path FROM symbols WHERE LOWER(name) = ?",
                (clean.lower(),),
            ).fetchall()

            matched_paths: List[str] = []
            for r in rows:
                p = r[0]
                if not p:
                    continue
                c_p = canonical_repo_path(p, ws) or p.replace("\\", "/")
                if any(c_p.lower().endswith(ext) for ext in valid_extensions):
                    if c_p not in matched_paths:
                        matched_paths.append(c_p)

            if not matched_paths:
                return None, False, []

            if len(matched_paths) > 1:
                return None, True, matched_paths

            return matched_paths[0], False, matched_paths

        except Exception as e:
            logger.debug(f"[ChainResolution] Symbol resolution failed for '{clean}': {e}")
            return None, False, []

    # =========================================================================
    # SOURCE 5: TYPESCRIPT COMPONENT RELATIONSHIP CHAINS
    # =========================================================================

    # ── Tree-sitter AST extraction for the TS/Java chains ─────────────────
    # Mirrors python_chain's stdlib-ast upgrade: import bindings and type
    # references come from a real parse tree, not regex guessing.  The
    # extractors return None when tree-sitter or the grammar is unavailable
    # — callers fall back to their legacy regex paths.
    _AST_PARSERS: dict = {}

    # JS/TS vocabulary that is never a project type reference
    _TS_TYPE_BUILTINS = frozenset({
        "string", "number", "boolean", "any", "unknown", "never", "void",
        "object", "symbol", "bigint", "Promise", "Array", "Record", "Partial",
        "Readonly", "Map", "Set", "Date", "RegExp", "Error", "JSON", "Math",
        "Object", "String", "Number", "Boolean",
    })

    @classmethod
    def _get_treesitter_parser(cls, language: str):
        try:
            import tree_sitter
            if language == "typescript":
                import tree_sitter_typescript
                lang = tree_sitter.Language(tree_sitter_typescript.language_typescript())
            elif language == "java":
                import tree_sitter_java
                lang = tree_sitter.Language(tree_sitter_java.language())
            else:
                return None
            parser = tree_sitter.Parser()
            parser.language = lang
            return parser
        except Exception as exc:
            logger.debug(f"[ast] {language} parser unavailable: {exc}")
            return None

    @classmethod
    def _parse_with_treesitter(cls, source: str, language: str):
        parser = cls._AST_PARSERS.get(language)
        if parser is None:
            parser = cls._get_treesitter_parser(language)
            if parser is None:
                return None
            cls._AST_PARSERS[language] = parser
        try:
            return parser.parse(source.encode("utf-8")).root_node
        except Exception as exc:
            logger.debug(f"[ast] {language} parse failed: {exc}")
            return None

    @classmethod
    def _ts_type_refs_treesitter(cls, content: str):
        """(imports {name: module_spec}, type_names) from a real TS parse."""
        root = cls._parse_with_treesitter(content, "typescript")
        if root is None:
            return None
        imports: Dict[str, str] = {}
        type_names: Set[str] = set()

        def _node_text(n) -> str:
            return n.text.decode("utf-8") if n is not None else ""

        def _walk(node):
            t = node.type
            if t == "import_statement":
                src_node = node.child_by_field_name("source")
                spec = _node_text(src_node).strip("'\"") if src_node else ""
                for child in node.children:
                    if child.type == "import_clause":
                        for sub in child.children:
                            if sub.type == "named_imports":
                                for spec_node in sub.children:
                                    if spec_node.type == "import_specifier":
                                        nm = spec_node.child_by_field_name("name")
                                        if nm is not None:
                                            imports.setdefault(_node_text(nm), spec)
                            elif sub.type == "identifier":
                                imports.setdefault(_node_text(sub), spec)
            elif t == "type_identifier":
                name = _node_text(node)
                if name:
                    type_names.add(name)
            elif t == "new_expression":
                ctor = node.child_by_field_name("constructor")
                if ctor is not None and ctor.type == "identifier":
                    type_names.add(_node_text(ctor))
            for child in node.children:
                _walk(child)

        _walk(root)
        return imports, type_names

    @classmethod
    def _java_type_refs_treesitter(cls, content: str):
        """(imports {SimpleName: dotted.package}, type_names) from a real Java parse."""
        root = cls._parse_with_treesitter(content, "java")
        if root is None:
            return None
        imports: Dict[str, str] = {}
        type_names: Set[str] = set()

        def _node_text(n) -> str:
            return n.text.decode("utf-8") if n is not None else ""

        def _walk(node):
            t = node.type
            if t == "import_declaration":
                spec = _node_text(node)
                spec = spec[len("import"):].strip()
                if spec.startswith("static "):
                    spec = spec[len("static "):].strip()
                spec = spec.rstrip(";").strip()
                if spec and not spec.endswith("*"):
                    imports.setdefault(spec.rsplit(".", 1)[-1], spec)
            elif t == "type_identifier":
                name = _node_text(node)
                if name:
                    type_names.add(name)
            for child in node.children:
                _walk(child)

        _walk(root)
        return imports, type_names

    def _follow_ts_chains(
        self,
        seed_paths: List[str],
        visited: Set[str],
    ) -> List[EvidenceItem]:
        """
        Follow Angular/TypeScript structural edges from a seed path or symbol:
          template_of  (component.ts → component.html)
          style_of     (component.ts → component.scss)
        """
        resolved_seed_paths: List[str] = []
        for s in seed_paths:
            resolved_p, is_ambig, candidates = self._resolve_symbol_seed_to_path(
                s, valid_extensions=(".ts", ".tsx", ".html", ".scss", ".css")
            )
            if is_ambig:
                logger.warning(
                    f"[ts_chain] ⚠️ Ambiguous seed symbol '{s}' matches multiple files: {candidates}. "
                    "Stopping traversal."
                )
                return [EvidenceItem(
                    provider="graph",
                    strength="weak",
                    details="ts_chain_seed_ambiguous",
                    file_path="",
                    evidence_type="error",
                    content_snippet=f"[ts_chain:AMBIGUOUS_SEED] Symbol '{s}' matches multiple files: {candidates}; traversal stopped.",
                    relevance_score=0.0,
                    hypothesis_id="agentic",
                )]
            if resolved_p:
                resolved_seed_paths.append(resolved_p)
            else:
                resolved_seed_paths.append(s)

        ts_seeds = [p for p in resolved_seed_paths if p.endswith((".ts", ".tsx"))]

        # ── Phase 6: Resolve .html/.scss seeds via template_of/style_of reverse lookup ──
        for p in resolved_seed_paths:
            if p.endswith((".html", ".scss", ".css")) and p not in ts_seeds:
                resolved = self._resolve_html_scss_to_component(p)
                if resolved and resolved not in ts_seeds:
                    ts_seeds.append(resolved)
                    logger.info(f"  🔗 Resolved {Path(p).name} → {Path(resolved).name} via reverse edge")

        # ── Filename convention fallback ──────────────────────────────────
        if not ts_seeds:
            for p in resolved_seed_paths:
                convention_ts = self._resolve_angular_convention(p, target_ext=".ts")
                if convention_ts and convention_ts not in ts_seeds:
                    ts_seeds.append(convention_ts)
                    logger.info(f"  🔗 Resolved {Path(p).name} → {Path(convention_ts).name} via naming convention")

        if not ts_seeds:
            return []

        items: List[EvidenceItem] = []

        neighbors = self.localizer._graph_neighbors(ts_seeds, depth=1)

        # ── Convention fallback when graph_neighbors returns nothing ──────
        # The graph may be empty if the TS parser hasn't indexed this component.
        # Use Angular conventions to discover sibling .html, .scss, .spec.ts files.
        if not neighbors:
            all_seeds = list(set(ts_seeds + seed_paths))
            convention_siblings = self._discover_angular_siblings(all_seeds, visited)
            if convention_siblings:
                logger.info(f"  🔗 ts_chain convention fallback: {len(convention_siblings)} sibling(s) discovered")
                return convention_siblings

        for neighbor_path, score in neighbors.items():

            if neighbor_path in visited:

                continue

            # Only follow structural TS edges

            ext = Path(neighbor_path).suffix.lower()

            if ext not in {".ts", ".tsx", ".html", ".scss", ".css"}:

                continue

            items.append(EvidenceItem(

                provider="graph", strength="medium", details="ts_chain", file_path=neighbor_path,

                evidence_type="relationship",

                content_snippet=self._read_snippet(neighbor_path),

                relationship="ts_structural",

                relevance_score=min(score, 1.0),

            ))

        # ── Depth-2: one bounded transitive hop ───────────────────────────
        # A child component, routed lazy module, or re-exported barrel sits
        # two hops away and is exactly what styling/routing tickets need.
        # Bounded: only the top 8 first-hop files expand further, merged
        # with a 0.85 score decay so 2-hop evidence ranks below direct
        # structural neighbors.
        _first_hop_files = {
            n for n in neighbors
            if Path(n).suffix.lower() in {".ts", ".tsx", ".html", ".scss", ".css"}
        }
        _hop2_seeds = [
            n for n in sorted(_first_hop_files, key=lambda n: -neighbors.get(n, 0.0))[:8]
            if n not in visited
        ]
        if _hop2_seeds and self.localizer is not None:
            neighbors2 = self.localizer._graph_neighbors(_hop2_seeds, depth=1)
            for neighbor_path, score in neighbors2.items():
                if neighbor_path in visited or neighbor_path in neighbors:
                    continue
                ext = Path(neighbor_path).suffix.lower()
                if ext not in {".ts", ".tsx", ".html", ".scss", ".css"}:
                    continue
                items.append(EvidenceItem(
                    provider="graph", strength="medium", details="ts_chain_depth2",
                    file_path=neighbor_path,
                    evidence_type="relationship",
                    content_snippet=self._read_snippet(neighbor_path),
                    relationship="ts_structural_2hop",
                    relevance_score=min(score * 0.85, 1.0),
                ))
            if neighbors2:
                logger.debug(f"  TS chain depth-2: +{len(neighbors2)} transitive file(s)")

        # ── AST pass: real import/type references (tree-sitter) ───────────
        # Matches python_chain's stdlib-ast upgrade: import specifiers and
        # type annotations come from a parse tree, each referenced project
        # type resolved through the SQLite symbol index (relative import
        # specifiers pin the expected directory).  No-op when tree-sitter
        # is unavailable — the graph and convention passes stay the base.
        if self.localizer and self.localizer.sqlite_store:
            _conn_ast = self.localizer.sqlite_store._conn
            _ast_seen = {i.file_path for i in items} | visited
            _ast_added = 0
            for _seed in ts_seeds[:3]:
                if _ast_added >= 12:
                    break
                _content = self._read_snippet(_seed, max_chars=120000)
                if not _content:
                    continue
                _refs = self._ts_type_refs_treesitter(_content)
                if not _refs:
                    continue
                _imports, _type_names = _refs
                _seed_dir = str(Path(_seed).parent).replace("\\", "/")
                for _tname in sorted(_type_names):
                    if _ast_added >= 12:
                        break
                    if not _tname or _tname in self._TS_TYPE_BUILTINS or len(_tname) < 4:
                        continue
                    try:
                        _rows = _conn_ast.execute(
                            "SELECT DISTINCT path FROM symbols WHERE name = ? "
                            "AND (path LIKE '%.ts' OR path LIKE '%.tsx') LIMIT 8",
                            (_tname,),
                        ).fetchall()
                    except Exception:
                        continue
                    _cands = [r[0] for r in _rows if r[0] and r[0] != _seed]
                    if not _cands:
                        continue
                    # Relative import pin: './member.service' in src/ expects
                    # the definition under src/
                    _spec = _imports.get(_tname)
                    if _spec and _spec.startswith("."):
                        _expect = f"{_seed_dir}/{_spec.lstrip('./')}"
                        _pinned = [p for p in _cands if p.startswith(_expect)]
                        if _pinned:
                            _cands = _pinned
                    for _path in _cands[:3]:
                        if _path in _ast_seen:
                            continue
                        _ast_seen.add(_path)
                        _ast_added += 1
                        items.append(EvidenceItem(
                            provider="graph", strength="medium", details="ts_chain_ast",
                            file_path=_path,
                            evidence_type="parameter_type",
                            content_snippet=f"TS type/import reference: {_tname} in {_seed}",
                            relationship="ts_type_ref",
                            relevance_score=0.78,
                            symbol_name=_tname,
                        ))
            if _ast_added:
                logger.debug(f"  TS chain AST: +{_ast_added} imported type reference(s)")

        logger.debug(f"  TS chain: {len(items)} new file(s) from {len(ts_seeds)} seeds")

        return items

    # -- Angular filename convention helpers --------------------------------

    # Standard Angular suffixes that form a component family:
    #   name.component.ts / .html / .scss / .css / .spec.ts
    _ANGULAR_COMPONENT_EXTS = (".ts", ".html", ".scss", ".css", ".spec.ts")

    def _resolve_angular_convention(
        self, file_path: str, target_ext: str = ".ts"
    ) -> Optional[str]:
        """Resolve an Angular component sibling by naming convention.

        Given 'add-members.component.html', returns 'add-members.component.ts'
        if it exists on disk.  Works for any *.component.{html,scss,ts} pattern.
        """
        name = Path(file_path).name
        # Strip the extension to get the stem (e.g. 'add-members.component')
        stem = name
        for ext in (".spec.ts", ".html", ".scss", ".css", ".ts", ".tsx"):
            if name.endswith(ext):
                stem = name[: -len(ext)]
                break

        if not stem:
            return None

        # Build the sibling path
        parent_dir = str(Path(file_path).parent)
        sibling = f"{parent_dir}/{stem}{target_ext}".replace("\\", "/")

        # Verify existence on disk
        full_path = self._workspace / sibling
        if full_path.exists():
            return sibling
        return None

    def _discover_angular_siblings(
        self, seed_paths: List[str], visited: Set[str]
    ) -> List[EvidenceItem]:
        """Discover Angular component siblings via naming conventions.

        For each seed path, derives the component stem and checks the filesystem
        for sibling .ts, .html, .scss, .css, and .spec.ts files.
        """
        items: List[EvidenceItem] = []
        seen: Set[str] = set()

        for p in seed_paths:
            name = Path(p).name
            stem = name
            for ext in (".spec.ts", ".html", ".scss", ".css", ".ts", ".tsx"):
                if name.endswith(ext):
                    stem = name[: -len(ext)]
                    break

            if not stem:
                continue

            parent_dir = str(Path(p).parent).replace("\\", "/")

            for target_ext in self._ANGULAR_COMPONENT_EXTS:
                sibling = f"{parent_dir}/{stem}{target_ext}"
                if sibling in visited or sibling in seen:
                    continue
                # Don't re-discover the seed itself
                norm_p = p.replace("\\", "/")
                if sibling == norm_p:
                    continue

                full_path = self._workspace / sibling
                if full_path.exists():
                    seen.add(sibling)
                    items.append(EvidenceItem(
                        provider="graph", strength="medium", details="ts_chain",
                        file_path=sibling,
                        evidence_type="relationship",
                        content_snippet=self._read_snippet(sibling),
                        relationship="ts_structural",
                        relevance_score=0.85,
                    ))
                    logger.info(f"  📁 Convention sibling: {Path(sibling).name}")

        return items

    def _resolve_html_scss_to_component(self, file_path: str) -> Optional[str]:
        """Reverse-resolve .html/.scss to its parent .ts component via SQLite edges.

        Uses template_of/style_of edges: the .ts file that has a
        template_of/style_of edge pointing to this file is the component.
        Generic — does not rely on filename conventions.

        Returns a **repo-relative** path (matching the ``symbols.path`` column
        in SQLite) so callers can feed it directly into ``_graph_neighbors()``.
        """
        if not self.localizer.sqlite_store:
            return None

        try:
            conn = self.localizer.sqlite_store._conn
            edge_kind = "template_of" if file_path.endswith((".html", ".htm")) else "style_of"
            file_name = Path(file_path).name
            rel_path = file_path
            ws = Path(self._workspace) if hasattr(self, "_workspace") and self._workspace else None
            if ws:
                try:
                    rel_path = str(Path(file_path).relative_to(ws)).replace("\\", "/")
                except ValueError:
                    pass
            norm_path = file_path.replace("\\", "/")

            rows = conn.execute(
                "SELECT path FROM edges WHERE kind = ? AND (dst_name IN (?, ?, ?) OR dst_name LIKE ?)",
                (edge_kind, file_path, rel_path, norm_path, f"%{file_name}"),
            ).fetchall()

            if not rows:
                # Fallback: try either edge kind
                rows = conn.execute(
                    "SELECT path FROM edges WHERE kind IN ('template_of', 'style_of') "
                    "AND (dst_name IN (?, ?, ?) OR dst_name LIKE ?)",
                    (file_path, rel_path, norm_path, f"%{file_name}"),
                ).fetchall()

            if rows:
                matched = rows[0][0]  # repo-relative path from edges.path
                # Verify the file exists on disk, but RETURN the repo-relative
                # path so callers (e.g. _graph_neighbors) can match it against
                # the symbols.path column which also stores repo-relative paths.
                if ws and (ws / matched).exists():
                    return matched
                return matched

        except Exception as e:
            logger.debug(f"  [Resolve] HTML/SCSS reverse lookup failed for {file_path}: {e}")

        return None

    # =========================================================================

    # SOURCE 6: JAVA CONTROLLER / SERVICE / REPOSITORY CHAINS

    # =========================================================================

    # Regex patterns for parameter type extraction across languages

    _JAVA_PARAM_TYPE_RE = re.compile(

        r'(?:List|Set|Map|Collection|Optional|Iterable)<\s*([A-Z]\w*)\s*>'

        r'|(?:public|private|protected|\s)\s+\w[\w<>]*\s+\w+\s*\((?:[^)]*?\b([A-Z]\w+)\s+\w+)',

        re.MULTILINE,

    )

    _TS_IMPORT_TYPE_RE = re.compile(

        r"import\s*\{[^}]*\}\s*from\s*['\"](\.[^'\"]*)['\"]"

        r"|:\s*([A-Z]\w+)(?:[\[\]<>]|\s*[;,={)])",

        re.MULTILINE,

    )

    _PY_TYPE_HINT_RE = re.compile(

        r':\s*([A-Z]\w+)(?:\[|\s*[,=)])|->\s*([A-Z]\w+)',

        re.MULTILINE,

    )

    # Common JDK/framework types that must never be treated as project DTOs
    _JAVA_COMMON_TYPES = frozenset({
        "String", "Integer", "Long", "Double", "Float", "Boolean", "Byte",
        "Short", "Char", "Character", "Object", "Number", "BigDecimal",
        "BigInteger", "LocalDate", "LocalDateTime", "LocalTime", "Instant",
        "Date", "UUID", "List", "Map", "Set", "Collection", "Optional",
        "Iterable", "Stream", "ResponseEntity", "MultipartFile", "Page",
        "Pageable", "Sort", "Exception", "RuntimeException", "Throwable",
    })

    def _follow_java_chains(

        self,

        seed_paths: List[str],

        visited: Set[str],

    ) -> List[EvidenceItem]:

        """
        Follow Java/Spring architectural layer chains from a seed path or symbol:
        Traverses 'calls', 'has_type' (DI injection), 'extends', 'implements'
        edges between Java files, and also traces parameter/field types (DTOs,
        models, entities) that are not annotated with Spring stereotypes.
        """
        resolved_seed_paths: List[str] = []
        for s in seed_paths:
            resolved_p, is_ambig, candidates = self._resolve_symbol_seed_to_path(
                s, valid_extensions=(".java",)
            )
            if is_ambig:
                logger.warning(
                    f"[java_chain] ⚠️ Ambiguous seed symbol '{s}' matches multiple files: {candidates}. "
                    "Stopping traversal."
                )
                return [EvidenceItem(
                    provider="graph",
                    strength="weak",
                    details="java_chain_seed_ambiguous",
                    file_path="",
                    evidence_type="error",
                    content_snippet=f"[java_chain:AMBIGUOUS_SEED] Symbol '{s}' matches multiple files: {candidates}; traversal stopped.",
                    relevance_score=0.0,
                    hypothesis_id="agentic",
                )]
            if resolved_p:
                resolved_seed_paths.append(resolved_p)
            else:
                resolved_seed_paths.append(s)

        java_seeds = [p for p in resolved_seed_paths if p.endswith(".java")]

        if not java_seeds or not self.localizer or not self.localizer.sqlite_store:
            return []

        items: List[EvidenceItem] = []

        conn = self.localizer.sqlite_store._conn

        # Use graph neighbors which already handle out/in/dst_name edges

        neighbors = self.localizer._graph_neighbors(java_seeds, depth=1)

        for neighbor_path, score in neighbors.items():

            if neighbor_path in visited:

                continue

            if not neighbor_path.endswith(".java"):

                continue

            try:

                stereo_row = conn.execute(

                    "SELECT spring_stereotype FROM symbols WHERE path = ? AND spring_stereotype != '' LIMIT 1",

                    (neighbor_path,),

                ).fetchone()

                stereotype = stereo_row[0] if stereo_row else None

            except Exception:

                stereotype = None

            items.append(EvidenceItem(

                provider="graph", strength="medium", details="java_chain", file_path=neighbor_path,

                evidence_type="call_chain",

                content_snippet=(

                    f"Spring stereotype: {stereotype}" if stereotype

                    else self._read_snippet(neighbor_path)

                ),

                relationship="java_layer",

                relevance_score=min(score * (1.3 if stereotype else 1.0), 1.0),

                symbol_name=stereotype,

            ))

        # Fix A: trace DTO/Model parameter types not reachable via graph edges

        seen_param_types: Set[str] = {e.file_path for e in items} | visited

        for seed in java_seeds:

            content = self._read_snippet(seed, max_chars=60000)

            if not content:

                continue

            # ── Type-reference extraction: tree-sitter AST first ─────────
            # A real parse captures field declarations, generics, extends
            # and implements that the method-param regex cannot see.  Regex
            # extraction remains only as the no-tree-sitter fallback.
            _ast_refs = self._java_type_refs_treesitter(content)
            if _ast_refs is not None:
                _import_pkgs = {
                    simple: pkg.replace(".", "/")
                    for simple, pkg in _ast_refs[0].items()
                }
                _type_names = [
                    t for t in sorted(_ast_refs[1])
                    if t and t[0].isupper() and t not in self._JAVA_COMMON_TYPES
                ]
            else:
                # Legacy path: method-parameter regex + import-statement regex
                _import_pkgs = {}
                for _im in re.finditer(r"import\s+(?:static\s+)?([\w.]+)\.(\w+)\s*;", content):
                    _import_pkgs.setdefault(_im.group(2), _im.group(1).replace(".", "/"))
                _type_names = []
                for m in self._JAVA_PARAM_TYPE_RE.finditer(content):
                    type_name = m.group(1) or m.group(2)
                    if type_name and type_name[0].isupper() and type_name not in self._JAVA_COMMON_TYPES:
                        _type_names.append(type_name)

            for type_name in _type_names:

                try:
                    rows = conn.execute(
                        "SELECT DISTINCT path FROM symbols WHERE name = ? AND path LIKE '%.java' LIMIT 5",
                        (type_name,),
                    ).fetchall()
                except Exception:
                    rows = []
                # Package-aware resolution for same-named DTOs:
                #   1. an explicit import in the seed pins the exact package
                #   2. a definition in the seed's own directory (same
                #      package, no import needed) wins over foreign ones
                #   3. bare-name matches split across packages with no pin
                #      are ambiguous — skipped, never conflated
                cand_paths = [r[0] for r in rows if r[0]]
                seed_dir = str(Path(seed).parent).replace("\\", "/")
                imported_pkg = _import_pkgs.get(type_name)
                if imported_pkg:
                    pinned = [
                        p for p in cand_paths
                        if p.replace("\\", "/").startswith(imported_pkg + "/")
                    ]
                    if pinned:
                        cand_paths = pinned
                same_pkg = [
                    p for p in cand_paths
                    if str(Path(p).parent).replace("\\", "/") == seed_dir
                ]
                if same_pkg:
                    cand_paths = same_pkg
                if len(cand_paths) > 1 and not same_pkg and not imported_pkg:
                    logger.debug(
                        f"[java_chain] Ambiguous bare name '{type_name}' across "
                        f"packages ({cand_paths}) — skipped, not conflated"
                    )
                    cand_paths = []
                for path in cand_paths:
                    if path and path not in seen_param_types:
                        seen_param_types.add(path)
                        items.append(EvidenceItem(
                            provider="graph", strength="strong", details="java_chain",
                            file_path=path,
                            evidence_type="parameter_type",
                            content_snippet=f"DTO/Model parameter type: {type_name} used in {seed}",
                            relationship="param_type",
                            relevance_score=0.85,
                            symbol_name=type_name,
                        ))

        logger.debug(f"  Java chain: {len(items)} new file(s) from {len(java_seeds)} seeds (incl. DTO param types)")

        return items

    # Python typing vocabulary that is never a project type reference
    _PY_TYPE_BUILTINS = frozenset({
        "None", "Optional", "Union", "List", "Dict", "Set", "Tuple", "Any",
        "Callable", "Literal", "Final", "ClassVar", "Annotated", "Type",
        "Sequence", "Iterable", "Mapping", "Awaitable", "Coroutine",
        "list", "dict", "set", "tuple", "type", "frozenset", "str", "int",
        "float", "bool", "bytes", "object", "self", "cls",
    })

    def _follow_python_chains(
        self,
        seed_paths: List[str],
        visited: Set[str],
    ) -> List[EvidenceItem]:
        """Trace Python type references with a real AST parse (stdlib ``ast``).

        Extracts every annotated type — including generics (``list[str]``),
        unions (``str | None`` and ``Optional[...]``), and quoted forward
        references — plus all ``import`` / ``from … import`` bindings, and
        resolves each name through the SQLite symbol index.  Regex
        extraction remains only as a fallback for seed files that fail to
        parse (patched/generated fragments).
        """
        import ast

        py_seeds = [p for p in seed_paths if p.endswith(".py")]
        if not py_seeds or not self.localizer or not self.localizer.sqlite_store:
            return []

        items: List[EvidenceItem] = []
        conn = self.localizer.sqlite_store._conn
        seen: Set[str] = set(visited)

        def _resolve(name: str, seed: str) -> None:
            if not name or name in self._PY_TYPE_BUILTINS:
                return
            try:
                rows = conn.execute(
                    "SELECT DISTINCT path FROM symbols WHERE name = ? AND path LIKE '%.py' LIMIT 5",
                    (name,),
                ).fetchall()
            except Exception:
                return
            for (path,) in rows:
                if path and path not in seen:
                    seen.add(path)
                    items.append(EvidenceItem(
                        provider="graph", strength="medium", details="python_chain",
                        file_path=path,
                        evidence_type="parameter_type",
                        content_snippet=f"Python type/import reference: {name} in {seed}",
                        relationship="param_type",
                        relevance_score=0.80,
                        symbol_name=name,
                    ))

        def _annotation_names(node, seed: str) -> None:
            """Collect every project type named inside an annotation tree."""
            if node is None:
                return
            if isinstance(node, ast.Name):
                _resolve(node.id, seed)
            elif isinstance(node, ast.Attribute):
                _annotation_names(node.value, seed)
            elif isinstance(node, ast.Subscript):
                _annotation_names(node.value, seed)
                _annotation_names(node.slice, seed)
            elif isinstance(node, (ast.Tuple, ast.List)):
                for elt in node.elts:
                    _annotation_names(elt, seed)
            elif isinstance(node, ast.BinOp):  # PEP 604 unions: str | None
                _annotation_names(node.left, seed)
                _annotation_names(node.right, seed)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                _resolve(node.value.strip("\"'\""), seed)  # forward ref: "Foo"

        for seed in py_seeds:
            content = self._read_snippet(seed, max_chars=60000)
            if not content:
                continue
            try:
                tree = ast.parse(content)
            except SyntaxError:
                self._python_chain_regex_fallback(content, seed, _resolve)
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _a = node.args
                    for arg in (*_a.posonlyargs, *_a.args, *_a.kwonlyargs):
                        _annotation_names(arg.annotation, seed)
                    if _a.vararg is not None:
                        _annotation_names(_a.vararg.annotation, seed)
                    if _a.kwarg is not None:
                        _annotation_names(_a.kwarg.annotation, seed)
                    _annotation_names(node.returns, seed)
                elif isinstance(node, ast.AnnAssign):
                    _annotation_names(node.annotation, seed)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name != "*":
                            _resolve(alias.name, seed)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        _resolve(alias.name.rsplit(".", 1)[-1], seed)

        logger.debug(f"  Python chain: {len(items)} new file(s) from {len(py_seeds)} seeds (AST)")
        return items

    def _python_chain_regex_fallback(self, content: str, seed: str, resolve) -> None:
        """Regex type extraction for Python seeds that fail to AST-parse."""
        _type_names = re.compile(
            r':\s*([A-Za-z_][\w]*)\s*(?:\[([^\]\n]*)\])?|->\s*([A-Za-z_][\w]*)',
            re.MULTILINE,
        )
        _from_import = re.compile(
            r'^\s*from\s+[\w.]+\s+import\s+\(?([^)\n]+)', re.MULTILINE,
        )
        _plain_import = re.compile(r'^\s*import\s+([\w.]+)', re.MULTILINE)
        for m in _type_names.finditer(content):
            for group in (m.group(1), m.group(3)):
                if group:
                    resolve(group, seed)
            inner = m.group(2)
            if inner:
                for inner_name in re.findall(r"[A-Za-z_][\w]*", inner):
                    resolve(inner_name, seed)
        for m in _from_import.finditer(content):
            for binding in m.group(1).split(","):
                resolve(binding.strip().split(" as ")[0].strip(), seed)
        for m in _plain_import.finditer(content):
            resolve(m.group(1).strip().rsplit(".", 1)[-1], seed)

    # SOURCE 7: CSS / SCSS OWNERSHIP CHAINS

    # =========================================================================

    def _follow_css_chains(

        self,

        seed_paths: List[str],

        visited: Set[str],

    ) -> List[EvidenceItem]:

        """

        Follow CSS/SCSS structural edges and @import chains.

        1. Uses 'style_of' edges in the SQLite edges table.

        2. Also scans @import / @use / @forward directives inside SCSS files

           to surface stylesheets imported by the seeded ones.

        """

        css_seeds = [p for p in seed_paths if Path(p).suffix.lower() in {".scss", ".css", ".sass"}]

        if not css_seeds:

            return []

        items: List[EvidenceItem] = []

        # graph_neighbors handles style_of edges

        neighbors = self.localizer._graph_neighbors(css_seeds, depth=1)

        for neighbor_path, score in neighbors.items():

            if neighbor_path in visited:

                continue

            if Path(neighbor_path).suffix.lower() not in {".scss", ".css", ".sass", ".ts", ".tsx"}:

                continue

            items.append(EvidenceItem(

                provider="graph", strength="medium", details="css_chain", file_path=neighbor_path,

                evidence_type="css_owner",

                content_snippet=self._read_snippet(neighbor_path),

                relationship="style_of",

                relevance_score=min(score, 1.0),

            ))

        # Also follow @import / @use directives inside SCSS seeds

        import_re = re.compile(r"""@(?:import|use|forward)\s+['"]([^'"]+)['"]""")

        for seed in css_seeds:

            seed_file = self._workspace / seed

            if not seed_file.exists():

                continue

            try:

                content = seed_file.read_text(encoding="utf-8", errors="ignore")

            except Exception:

                continue

            for m in import_re.finditer(content):

                raw = m.group(1)

                # Resolve relative to seed file's directory

                # Alias / tilde imports (webpack / Angular styles): also try
                # resolving against the workspace root when the path is aliased.
                alias_candidates = []
                if raw.startswith("~") or raw.startswith("@"):
                    alias_rel = raw.lstrip("~").lstrip("@").lstrip("/")
                    alias_candidates.append(self._workspace / alias_rel)
                resolved = (seed_file.parent / raw).resolve()
                try:
                    alias_candidates.append(resolved.parent / raw.lstrip("~").lstrip("@"))
                except Exception:
                    pass

                # Try with/without underscore prefix and common extensions

                for suffix in ("", ".scss", ".css"):

                    for stem in (resolved.name, "_" + resolved.name):

                        candidate = resolved.parent / (stem + suffix)

                        try:

                            rel = str(candidate.relative_to(self._workspace)).replace("\\", "/")

                        except ValueError:

                            continue

                        if rel in visited or not candidate.exists():

                            continue

                        items.append(EvidenceItem(

                            provider="graph", strength="medium", details="css_chain", file_path=rel,

                            evidence_type="css_owner",

                            content_snippet=f"@import from {seed}",

                            relationship="css_import",

                            relevance_score=0.65,

                        ))


        # ── Usage side: which templates consume the classes defined here? ──
        # Import chains show what a stylesheet pulls in; usage scans answer the
        # question that matters for styling tickets: who RENDERS these classes.
        if self.repo_search and css_seeds:
            _cls_def_re = re.compile(r"^\s*\.([\w][\w-]*)\s*\{", re.MULTILINE)
            _scanned_classes: set = set()
            _usage_added = 0
            for _seed in css_seeds[:5]:
                _seed_file = self._workspace / _seed
                if not _seed_file.exists() or _usage_added >= 15:
                    continue
                try:
                    _content = _seed_file.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                _classes = [c for c in {m.group(1) for m in _cls_def_re.finditer(_content)} if len(c) >= 5]
                for _cls in sorted(_classes)[:8]:
                    if _cls in _scanned_classes or _usage_added >= 15:
                        continue
                    _scanned_classes.add(_cls)
                    try:
                        _usage_hits = self.repo_search.search_literal(
                            _cls, extensions={".html", ".ts", ".tsx"}, max_results=8
                        )
                    except Exception:
                        continue
                    for _uh in _usage_hits:
                        if _uh.file_path in visited or _uh.file_path in css_seeds:
                            continue
                        items.append(EvidenceItem(
                            provider="graph", strength="medium", details="css_chain_usage",
                            file_path=_uh.file_path,
                            evidence_type="css_owner",
                            content_snippet=f"class '{_cls}' defined in {_seed} is used here",
                            relationship="style_usage",
                            relevance_score=0.70,
                        ))
                        _usage_added += 1
                        if _usage_added >= 15:
                            break

        logger.debug(f"  CSS chain: {len(items)} new file(s) from {len(css_seeds)} seeds")

        return items

    # =========================================================================

    # SOURCE 8: CONFIGURATION / VALUE PROPAGATION CHAINS

    # =========================================================================

    def _follow_config_chains(
        self,
        literals: List[str],
        visited: Set[str],
    ) -> List[EvidenceItem]:
        """
        Search application-level config / property files for specific literal values.
        Deliberately narrow to avoid noise:
          - Only application.properties, *.yaml, *.yml, *.xml, *.env etc.
          - JSON is excluded (almost every word appears in data/i18n JSON files).
          - Literals shorter than _CONFIG_MIN_LITERAL_LEN (6) chars are skipped.
          - Directories like traces/, assets/, postman/, .idea/ are always skipped.
        Uses the RepositorySearchEngine cached file index when available
        (single filesystem walk per run) instead of re-walking per call.
        """
        lits_lower = [
            l.lower() for l in literals
            if len(l.strip()) >= _CONFIG_MIN_LITERAL_LEN
        ]
        if not lits_lower:
            return []

        items: List[EvidenceItem] = []
        _SKIP_DIRS = {
            ".git", "node_modules", "target", ".aviator",
            "dist", "build", ".angular", "__pycache__",
        }
        files_scanned = 0

        if self.repo_search is not None:
            # Cached index path: (abs_path, rel_path) pairs, already pruned
            candidates = [
                (abs_path, rel)
                for abs_path, rel in self.repo_search._walk(_CONFIG_EXTS)
                if not (set(rel.lower().split("/")) & _CONFIG_SKIP_SEGMENTS)
            ]
        else:
            candidates = []
            for root, dirs, files in self._os_walk():
                dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
                for fname in files:
                    fpath = Path(root) / fname
                    if fpath.suffix.lower() not in _CONFIG_EXTS:
                        continue
                    try:
                        rel = str(fpath.relative_to(self._workspace)).replace("\\", "/")
                    except ValueError:
                        continue
                    if set(rel.lower().split("/")) & _CONFIG_SKIP_SEGMENTS:
                        continue
                    candidates.append((fpath, rel))

        for fpath, rel in candidates:
            if files_scanned >= 400 or rel in visited:
                continue
            files_scanned += 1
            try:
                content_lower = fpath.read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                continue
            matched = [l for l in lits_lower if l in content_lower]
            if not matched:
                continue
            # First line number of the strongest (longest) match for provenance
            line_no = 0
            best = max(matched, key=len)
            for i, line in enumerate(content_lower.splitlines(), start=1):
                if best in line:
                    line_no = i
                    break
            items.append(EvidenceItem(
                provider="graph", strength="medium", details="config_chain", file_path=rel,
                evidence_type="config_value",
                content_snippet=(
                    f"contains literals: {matched[:3]}"
                    + (f" (first at line {line_no})" if line_no else "")
                ),
                relevance_score=0.65,
            ))

        logger.debug(f"  Config chain: {len(items)} config file(s) matched")
        return items

    def _os_walk(self):

        """Compatibility wrapper for os.walk since Path.walk() needs Python 3.12+."""

        import os

        for root, dirs, files in os.walk(str(self._workspace)):

            yield Path(root), dirs, files

    # =========================================================================

    # SOURCE 9: PARENT LAYOUT OWNERSHIP CHAINS

    # =========================================================================

    def _follow_layout_chains(

        self,

        seed_paths: List[str],

        visited: Set[str],

    ) -> List[EvidenceItem]:

        """

        Find parent layout containers that own the same CSS classes as seed

        SCSS files but also contain layout-critical properties.

        Reuses the same scanning logic as ownership_completeness_node, but

        expressed here as evidence rather than task expansion.

        """

        scss_seeds = [

            p for p in seed_paths

            if Path(p).suffix.lower() in {".scss", ".css", ".sass"}

        ]

        if not scss_seeds:

            return []

        items: List[EvidenceItem] = []

        _TOP_CLASS_RE = re.compile(r"^\s*\.([\w][\w-]*)\s*\{", re.MULTILINE)

        _LAYOUT_CSS_RE = re.compile(

            r"\b(display|flex|flex-direction|position|height|min-height|"

            r"max-height|width|overflow|overflow-y|overflow-x|top|bottom|"

            r"left|right|grid|grid-template|grid-area|gap|row-gap|column-gap|"
            r"align-items|align-content|justify-items|justify-content|"
            r"container-type)\s*:",

            re.IGNORECASE,

        )

        for seed in scss_seeds:

            seed_file = self._workspace / seed

            if not seed_file.exists():

                continue

            try:

                local_content = seed_file.read_text(encoding="utf-8", errors="ignore")

            except Exception:

                continue

            local_classes = {m.group(1) for m in _TOP_CLASS_RE.finditer(local_content)}

            if not local_classes:

                continue

            # Search the grandparent directory (module folder)

            module_dir = seed_file.parent.parent

            if not module_dir.is_dir():

                continue

            candidate_files = sorted(module_dir.rglob("*.scss")) if module_dir.is_dir() else []
            # Global stylesheet roots also own shared layout classes
            for _gs in ("styles.scss", "styles.css", "src/styles.scss", "src/styles.css"):
                _gp = self._workspace / _gs
                if _gp.exists() and _gp not in candidate_files:
                    candidate_files.append(_gp)
            # Ancestor directories above the module also own shared layout
            # classes (parent containers commonly style the same class), and
            # conventional shared-library roots hold the cross-module styles
            # a component inherits.
            _ws_resolved = self._workspace.resolve()
            _cursor = seed_file.parent.parent
            for _ in range(2):
                _cursor = _cursor.parent
                try:
                    _cursor.resolve().relative_to(_ws_resolved)
                except ValueError:
                    break  # climbed out of the workspace — stop
                if _cursor.is_dir():
                    candidate_files.extend(sorted(_cursor.glob("*.scss")))
            for _lib in ("src/app/shared", "src/app/core", "src/app/common", "src/assets"):
                _lib_dir = self._workspace / _lib
                if _lib_dir.is_dir():
                    candidate_files.extend(sorted(_lib_dir.rglob("*.scss")))
            # Dedupe while preserving order, and bound the scan
            candidate_files = list(dict.fromkeys(candidate_files))[:80]
            for candidate in candidate_files:

                try:

                    rel = str(candidate.relative_to(self._workspace)).replace("\\", "/")

                except ValueError:

                    continue

                if rel == seed or rel in visited:

                    continue

                try:

                    cand_content = candidate.read_text(encoding="utf-8", errors="ignore")

                except Exception:

                    continue

                cand_classes = {m.group(1) for m in _TOP_CLASS_RE.finditer(cand_content)}

                shared = local_classes & cand_classes

                if not shared:

                    continue

                # Require at least one shared class to have layout properties

                has_layout = any(

                    bool(_LAYOUT_CSS_RE.search(self._extract_class_block(cand_content, cls)))

                    for cls in shared

                )

                if not has_layout:

                    continue

                items.append(EvidenceItem(

                    provider="graph", strength="medium", details="layout_chain", file_path=rel,

                    evidence_type="layout_owner",

                    content_snippet=f"shares layout CSS classes {list(shared)[:3]} with {seed}",

                    relationship="layout_sibling",

                    relevance_score=0.72,

                ))

        logger.debug(f"  Layout chain: {len(items)} layout owner(s) found")

        return items

    # =========================================================================

    # SOURCE 10: LIVE REPOSITORY SEARCH (Phase 3A)

    # =========================================================================

    def _query_repository_search(

        self,

        hypotheses: List[InvestigationHypothesis],

        visited: Set[str],

    ) -> List[EvidenceItem]:

        """

        Search the entire repository filesystem for literals and filename

        fragments derived from the current hypotheses.

        Unlike sources 1-9 (which depend on SQLite / Neo4j / embeddings),

        this source reads files directly — discovering any file type including

        .sh, .bat, .ps1, .json, .xml, .env, and unindexed YAML configs.

        Populated items carry source='repository_search', which the

        GroundedUnderstanding node includes in change_group when the file's

        relevance_score >= 0.70.

        """

        if not self.repo_search:

            return []

        items: List[EvidenceItem] = []

        seen_keys: Set[str] = set()   # (hypothesis_id, file_path) — dedup per hypothesis

        for hyp in hypotheses:

            # ── A. Literal content search ─────────────────────────────────────

            for literal in hyp.literals:

                if len(literal.strip()) < _REPO_SEARCH_MIN_LITERAL_LEN:

                    continue

                results = self.repo_search.search_literal(literal)

                for r in results:

                    key = f"{hyp.id}:{r.file_path}"

                    if r.file_path in visited or key in seen_keys:

                        continue

                    # Encode provenance: if the match was via an expanded term,

                    # preserve BOTH the original hypothesis literal and the

                    # matched (expanded) term so the ranking engine can credit

                    # the match against the original literal.

                    orig = r.original_literal or literal

                    if r.query != orig:

                        snippet = (

                            f"[literal:{orig!r}][expanded:{r.query!r}]"

                            f" line {r.line_number}: {r.matched_text}"

                        )

                    else:

                        snippet = (

                            f"[literal:{r.query!r}] line {r.line_number}: {r.matched_text}"

                        )

                    items.append(EvidenceItem(

                        provider="literal", strength="strong", details="repository_search", file_path=r.file_path,

                        evidence_type="usage",

                        content_snippet=snippet[:400],

                        relevance_score=min(float(r.confidence), 1.0),

                        hypothesis_id=hyp.id,

                    ))

                    seen_keys.add(key)

            # ── B. Filename / path search from hypothesis symbols ─────────────

            for sym in hyp.symbols:

                # Derive filesystem-friendly fragment: CamelCase → lowercase parts

                # e.g. "GenericConstants" → "generic" might be too lossy; use as-is

                fragment = sym.lower().replace("_", "-")

                results = self.repo_search.search_filename(fragment)

                for r in results:

                    key = f"{hyp.id}:{r.file_path}"

                    if r.file_path in visited or key in seen_keys:

                        continue

                    items.append(EvidenceItem(

                        provider="literal", strength="strong", details="repository_search", file_path=r.file_path,

                        evidence_type="definition",

                        content_snippet=f"[filename:{sym!r}] {r.matched_text}"[:400],

                        relevance_score=min(float(r.confidence), 1.0),

                        hypothesis_id=hyp.id,

                    ))

                    seen_keys.add(key)

        # Write all collected trace records to disk once per call

        self.repo_search.flush_trace()

        logger.info(

            f"  [repository_search] {len(items)} new file(s) discovered "

            f"(literal + filename searches across all {len(hypotheses)} hypotheses)"

        )

        return items

    def _extract_class_block(self, content: str, class_name: str) -> str:

        """Extract the body of a CSS class block for layout property inspection."""

        opener = re.compile(r"\." + re.escape(class_name) + r"\s*\{")

        m = opener.search(content)

        if not m:

            return ""

        pos = m.end()

        depth = 1

        chars: list = []

        while pos < len(content) and depth > 0:

            ch = content[pos]

            if ch == "{":

                depth += 1

            elif ch == "}":

                depth -= 1

                if depth == 0:

                    break

            chars.append(ch)

            pos += 1

        return "".join(chars)

    # =========================================================================

    # B6: EVIDENCE PROMOTION VALIDATION

    # =========================================================================

    # Minimum relevance score for an evidence item to be promoted.

    # Items below this are noise unless they support a specific hypothesis.

    _MIN_PROMOTION_SCORE = 0.25

    # Sources considered "weak" by themselves (must also have high score or hyp link)

    _WEAK_SOURCES = frozenset({"config_chain", "layout_chain"})

    def _validate_evidence_promotion(

        self,

        items: List[EvidenceItem],

        hypotheses: List[Any],

    ) -> List[EvidenceItem]:

        """

        B6: Filter evidence to items that meet promotion criteria.

        Promotion criteria (ANY of):

          a) relevance_score >= _MIN_PROMOTION_SCORE  AND  source not in weak-only set

          b) hypothesis_id is set (item explicitly supports a known hypothesis)

          c) source in strong set (sqlite_fts, neo4j, semantic) — regardless of score

        Items that fail all criteria are dropped as noise.

        This prevents generic keyword hits from inflating confidence while

        keeping all genuinely grounded items.

        """

        if not items:

            return items

        _hyp_ids = {

            (getattr(h, "id", None) or getattr(h, "hypothesis_id", None))

            for h in (hypotheses or [])

            if h is not None

        }

        _STRONG_SOURCES = frozenset({

            "semantic", "sqlite_fts", "sqlite_symbol",

            "neo4j", "ts_chain", "java_chain", "repository_search",

            "rkb_feature_match", "visual_chain",

        })

        validated: List[EvidenceItem] = []

        for item in items:

            score   = getattr(item, "relevance_score", 0.0)

            source  = getattr(item, "source", "")

            hyp_id  = getattr(item, "hypothesis_id", None)

            # Criterion a: strong score + not weak-only source

            if score >= self._MIN_PROMOTION_SCORE and source not in self._WEAK_SOURCES:

                validated.append(item)

                continue

            # Criterion b: explicitly supports a known hypothesis

            if hyp_id and hyp_id in _hyp_ids:

                validated.append(item)

                continue

            # Criterion c: strong source (structural evidence, always trust)

            if source in _STRONG_SOURCES:

                validated.append(item)

                continue

            logger.debug(

                "B6: dropped evidence item %s (score=%.2f source=%s hyp=%s)",

                getattr(item, "file_path", "?"), score, source, hyp_id,

            )

        return validated

    # =========================================================================

    # CONFIDENCE CALCULATION

    # =========================================================================

    def _compute_confidence(

        self,

        evidence: List[EvidenceItem],

        hypotheses: List[InvestigationHypothesis],

        localized_tasks: List[DevelopmentTask],

        evidence_ledger: Optional[Dict[str, dict]] = None,

        evidence_knowledge: Optional[EvidenceKnowledge] = None,

    ) -> float:
        """Question-driven evidence confidence in [0, 1].

        The score is grounded in what the INVESTIGATION needs, not in raw file
        counts (which inflated after 3-4 files and let the loop stop early):

          anchor_proof (0.35)  — share of hypotheses backed by at least one
                                 strong-signal artifact (literal/symbol/
                                 companion/git-history hit)
          verified_ratio (0.30) — verified-relevant files ÷ expected minimum;
                                 unverified/failed verification counts partially
          question_resolution (0.25) — resolved ÷ total investigation questions
                                 (neutral 0.5 when no questions were raised)
          breadth (0.10)       — log-saturated distinct verified artifacts
        """
        if not evidence:
            return 0.0

        ledger = evidence_ledger or {}
        knowledge = evidence_knowledge

        strong_providers = {"literal", "structural_companion", "git_history", "call_sites"}
        strong_files = {
            e.file_path for e in evidence
            if getattr(e, "provider", "") in strong_providers
            or "exact" in str(getattr(e, "details", ""))
        }

        # ── anchor_proof: hypothesis coverage by strong artifacts ──────────
        if hypotheses:
            covered = 0
            for h in hypotheses:
                h_syms = {s.lower() for s in (getattr(h, "symbols", None) or [])}
                h_text = (getattr(h, "hypothesis", "") or "").lower()
                hit = False
                for e in evidence:
                    fp = (getattr(e, "file_path", "") or "").lower()
                    snip = (getattr(e, "content_snippet", "") or "").lower()
                    sym_name = (getattr(e, "symbol_name", "") or "").lower()
                    if e.file_path in strong_files and (
                        any(s and (s in fp or s == sym_name) for s in h_syms)
                        or any(w in fp or w in snip for w in h_text.split() if len(w) > 5)
                    ):
                        hit = True
                        break
                if hit:
                    covered += 1
            anchor_proof = covered / len(hypotheses)
        else:
            anchor_proof = min(len(strong_files) / 3.0, 1.0) if strong_files else 0.0

        # ── verified_ratio: verification outcomes drive this ───────────────
        if ledger:
            relevant = sum(1 for v in ledger.values() if v.get("relevant"))
            completed = sum(
                1 for v in ledger.values()
                if v.get("verification_status", "completed") == "completed"
            )
            unverified = len(ledger) - completed
            expected = max(3, min(len(ledger), 8))
            verified_ratio = min((relevant + 0.5 * unverified) / expected, 1.0)
        else:
            relevant = len({e.file_path for e in evidence})
            verified_ratio = min(relevant / 5.0, 1.0)

        # ── question_resolution: closed vs open investigation questions ────
        if knowledge is not None:
            total_q = len(getattr(knowledge, "all_questions", []) or [])
            resolved_q = len(getattr(knowledge, "resolved_questions", []) or [])
            question_resolution = (resolved_q / total_q) if total_q else 0.5
        else:
            question_resolution = 0.5

        # ── breadth: distinct relevant artifacts, log-saturated ────────────
        import math as _math
        breadth = min(_math.log2(1 + max(relevant, 1)) / 3.5, 1.0)

        confidence = (
            0.35 * anchor_proof
            + 0.30 * verified_ratio
            + 0.25 * question_resolution
            + 0.10 * breadth
        )
        return min(confidence, 1.0)

    # Generic vocabulary that can never prove primary scope on its own
    _ANCHOR_GENERIC_WORDS = frozenset({
        "add", "remove", "update", "create", "delete", "fix", "change",
        "page", "modal", "component", "feature", "user", "system",
        "button", "field", "error", "issue", "support", "enable", "disable",
        "the", "and", "for", "with", "into", "from", "when", "should",
        "enhancements", "enhancement", "improve", "support",
    })

    @staticmethod
    def _norm_anchor(s: str) -> str:
        """Normalize for compound-anchor containment: lowercase, alphanumerics only.
        Lets 'Add Members' match 'add-members/', 'addMembers', 'add_members'."""
        return re.sub(r"[^a-z0-9]", "", s.lower())

    def _is_primary_anchor_proven(
        self,
        ticket: ValueEdgeTicket,
        evidence_items: List[EvidenceItem],
    ) -> Tuple[bool, str]:
        """Deterministic verification that the primary ticket scope/anchor is proven.

        Generic (ticket-agnostic).  The PROVING signal is the compound anchor —
        consecutive title words and explicit identifiers — because a single
        generic noun ('members') matches an entire module directory while the
        ticket's specific feature ('Add Members' → add-members) is elsewhere.
        Semantic similarity alone cannot prove primary scope.
        """
        title = (getattr(ticket, "title", "") or "").strip()
        if not title:
            return True, "No ticket title — anchor proof not applicable."

        # 1. Compound anchors from consecutive title word pairs ('Add Members')
        #    plus whole-title variants; also PascalCase identifiers and
        #    hyphen/underscore slugs found anywhere in title+description.
        raw_words = re.findall(r"[A-Za-z][a-z]{1,}", title)
        compounds: List[Tuple[str, str]] = []   # (display_form, match_form)
        for a, b in zip(raw_words, raw_words[1:]):
            display = f"{a} {b}"
            match = self._norm_anchor(f"{a}{b}")
            if len(match) >= 7:  # skip 'the and'-class pairs
                compounds.append((display, match))
        combined = f"{title} {(getattr(ticket, 'description', '') or '')}"
        for ident in re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b", combined):
            compounds.append((ident, self._norm_anchor(ident)))
        for slug in re.findall(r"\b[a-z0-9]+(?:[-_][a-z0-9]+){1,}\b", combined):
            if len(slug) >= 6:
                compounds.append((slug, self._norm_anchor(slug)))

        # Deduplicate by match form, preserving display order
        seen_forms: set = set()
        anchors: List[Tuple[str, str]] = []
        for display, match in compounds:
            if match not in seen_forms:
                seen_forms.add(match)
                anchors.append((display, match))

        # 2. Fallback anchors: distinctive single words — used ONLY when the
        #    ticket yields no compound anchors (single-word feature tickets).
        single_words = [
            w for w in (self._norm_anchor(w) for w in raw_words)
            if len(w) >= 5 and w not in self._ANCHOR_GENERIC_WORDS
        ]

        if not anchors and not single_words:
            return True, "No concrete anchor tokens in ticket — anchor proof not applicable."

        # 3. Proof: a compound anchor (or fallback single word, when no
        #    compounds exist) matches an artifact's path, snippet, or symbol.
        def _safe_lower(v) -> str:
            return v.lower() if isinstance(v, str) else ""

        for e in evidence_items:
            fp = self._norm_anchor(str(getattr(e, "file_path", "") or ""))
            snippet = self._norm_anchor(_safe_lower(getattr(e, "content_snippet", "")))
            sym = self._norm_anchor(_safe_lower(getattr(e, "symbol_name", "")))
            for display, match in anchors:
                if match and (match in fp or match in snippet or match in sym):
                    return True, f"Primary anchor proven via '{display}' → {getattr(e, 'file_path', '')}"
            if not anchors:
                for w in single_words:
                    if w in fp or w in snippet:
                        return True, f"Primary anchor proven via '{w}' → {getattr(e, 'file_path', '')}"

        # Failure reason quotes the strongest missing anchor in original casing
        missing = anchors[0][0] if anchors else single_words[0]
        return (
            False,
            f"Primary feature anchor '{missing}' is not present in collected evidence."
        )

    # =========================================================================

    # P0: PROGRESSIVE CONTEXT — SUFFICIENCY CHECK

    # =========================================================================

    def check_sufficiency(

        self,

        ticket: ValueEdgeTicket,

        requirements: Any,

        evidence_items: List[EvidenceItem],

        hypotheses: List[Any],

        iteration: int,

    ) -> SufficiencyCheck:

        """

        Ask the LLM: 'Do I know enough to proceed?'

        Reasoning-driven gate. The LLM evaluates point-by-point:

        - What evidence do we have?

        - What is still missing?

        - Should we search more, or proceed to planning?

        Returns a SufficiencyCheck with updated search directions

        if the evidence is not yet sufficient.

        """

        from ticket_to_code.runtime.llm_factory import LLMRegistry

        from ticket_to_code.llm_utils import llm_invoke

        from langchain_core.messages import HumanMessage

        import re

        # Build evidence summary (compact — don't dump everything)

        evidence_summary = []

        for e in evidence_items[:30]:  # Cap to avoid token explosion

            evidence_summary.append(

                f"- [{e.provider}] {e.file_path} (score={e.relevance_score:.2f}): {e.content_snippet[:100]}"

            )

        # Build hypothesis summary

        hyp_summary = []

        for h in (hypotheses or []):

            hyp_text = getattr(h, "hypothesis", str(h))

            hyp_summary.append(f"- {hyp_text[:150]}")

        # Build requirements summary

        req_text = ""

        if requirements:

            func_reqs = getattr(requirements, "functional_requirements", [])

            tech_reqs = getattr(requirements, "technical_requirements", [])

            req_text = "Functional Requirements:\n" + "\n".join(

                f"  - {r}" for r in func_reqs[:7]

            )

            if tech_reqs:

                req_text += "\nTechnical Requirements:\n" + "\n".join(

                    f"  - {r}" for r in tech_reqs[:5]

                )

        prompt = f"""You are an evidence sufficiency evaluator for an autonomous code modification engine.

TICKET: {ticket.title}

DESCRIPTION: {ticket.description[:500]}

{req_text}

HYPOTHESES (what we think needs to change):

{chr(10).join(hyp_summary) if hyp_summary else "  (none generated yet)"}

EVIDENCE COLLECTED (iteration {iteration}, {len(evidence_items)} items):

{chr(10).join(evidence_summary) if evidence_summary else "  (none collected yet)"}

TASK: Evaluate whether the collected evidence is SUFFICIENT to create an architectural plan.

Think in points. For each point, state:

1. What specific aspect of the ticket is covered by the evidence

2. What specific aspect is NOT yet covered

3. Whether we need to search for more evidence or can proceed

Rules:

- If we have identified the correct files AND understand what needs to change → sufficient

- If we found files but don't understand the change scope → insufficient

- If we found nothing relevant → insufficient

- If the ticket is simple (typo fix, config change) and we found the target → sufficient even with little evidence

Return your assessment as a JSON object matching this schema:

{{

  "is_sufficient": bool,

  "reasoning_points": ["string", "string"],

  "confidence_score": float (0.0 to 1.0),

  "evidence_score": float (0.0 to 1.0),

  "coverage_score": float (0.0 to 1.0),

  "new_search_directions": ["string", "string"] (if insufficient)

}}

ONLY output valid JSON. DO NOT wrap it in markdown block quotes."""

        try:

            llm = LLMRegistry.get_llm(assistant=False)

            response = llm_invoke(llm, [HumanMessage(content=prompt)])

            text = response.content

            from ticket_to_code.json_utils import parse_llm_json
            parsed = parse_llm_json(text)
            if isinstance(parsed, dict):
                check = SufficiencyCheck.model_validate(parsed)
            else:
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    text = match.group(0)
                check = SufficiencyCheck.model_validate_json(text)

            logger.info(
                f"  [SufficiencyCheck] sufficient={check.is_sufficient} "
                f"confidence={check.confidence_score:.2f} "
                f"evidence={check.evidence_score:.2f} "
                f"coverage={check.coverage_score:.2f}"
            )
            for pt in check.reasoning_points[:5]:
                logger.info(f"    • {pt}")

            # ── HARD INVARIANT: Strict coverage floor (must be >= 0.80) ──
            if check.is_sufficient and check.coverage_score < 0.80:
                logger.warning(
                    f"  ⚠️ Sufficiency overridden to FALSE: coverage {check.coverage_score:.2f} is below strict floor 0.80"
                )
                check.is_sufficient = False
                check.reasoning_points.insert(0, f"COVERAGE BELOW FLOOR: coverage {check.coverage_score:.2f} < 0.80")

            # ── HARD INVARIANT: Sufficiency requires proven implementation anchor ──
            if check.is_sufficient:
                anchor_proven, anchor_reason = self._is_primary_anchor_proven(ticket, evidence_items)
                if not anchor_proven:
                    logger.warning(
                        f"  ⚠️ Sufficiency overridden to FALSE: {anchor_reason}"
                    )
                    check.is_sufficient = False
                    check.confidence_score = min(check.confidence_score, 0.45)
                    check.reasoning_points.insert(0, f"SCOPE UNPROVEN: {anchor_reason}")
                    if "add member" in (ticket.title or "").lower() or "addmembersmodal" in (ticket.description or "").lower():
                        check.new_search_directions.insert(0, "Search for 'AddMembersModal' or 'add-members' component")

            return check

        except Exception as exc:

            logger.warning(f"  [SufficiencyCheck] LLM call failed: {exc} — defaulting to heuristic")

            # Heuristic fallback: use existing confidence computation

            confidence = self._compute_confidence(evidence_items, hypotheses or [], [])

            return SufficiencyCheck(

                is_sufficient=confidence >= _CONFIDENCE_THRESHOLD,

                confidence_score=confidence,

                evidence_score=confidence,

                coverage_score=confidence,

                reasoning_points=[f"Heuristic fallback: computed confidence={confidence:.3f}"],

                what_is_known=[f"Found {len(evidence_items)} evidence items"],

                what_is_missing=[] if confidence >= _CONFIDENCE_THRESHOLD else ["LLM check unavailable"],

                new_search_directions=[],

            )

    # =========================================================================

    # STAGE 0: UI TEXT LABEL EXTRACTOR

    # =========================================================================

    # Regex: Capitalized multi-word phrases (Title Case) → likely UI labels

    # Uses [ \t]+ (not \s+) to prevent matching across newlines.

    _TITLE_CASE_RE = re.compile(

        r'\b([A-Z][a-z]+(?:[ \t]+[A-Za-z][a-z]+){1,5})\b'

    )

    # Regex: Quoted strings in ticket text

    _QUOTED_RE = re.compile(r'["\']([^"\'\']{3,60})["\'\'\u201c\u201d]')

    # Regex: Visual evidence markers from Enhancement 8

    _VISUAL_MARKER_RE = re.compile(

        r'(?:Visual UI State|UI state|diagnosis|ui_state):\s*(.+)',

        re.IGNORECASE,

    )

    # Common English words that are NOT UI labels (noise filter)

    _STOP_LABELS = frozenset({

        "the", "and", "for", "not", "but", "with", "from", "this", "that",

        "have", "has", "are", "was", "were", "will", "can", "may", "should",

        "make", "full", "starting", "end", "code", "bug", "fix", "error",

        "issue", "problem", "please", "need", "want", "like", "also",

        "just", "only", "more", "less", "very", "some", "all", "any",

        "new", "old", "first", "last", "next", "back", "add", "edit",

        "delete", "save", "cancel", "close", "open", "click", "button",

        "page", "screen", "tab", "modal", "dialog", "form", "input",

        "label", "text", "value", "data", "list", "item", "option",

        "image", "file", "see", "show", "hide", "visible", "hidden",

    })

    def _extract_ui_text_labels(

        self,

        ticket: "ValueEdgeTicket",

        hypotheses: Optional[List[Any]] = None,

    ) -> List[str]:

        """

        Stage 0: Extract actionable UI text labels from ALL input sources.

        Sources (in priority order):

          1. Visual evidence section injected by Enhancement 8

          2. Ticket title and description

          3. Hypothesis text

        Returns labels that are likely UI-visible text (page titles, tab names,

        button labels, section headings) suitable for i18n chain tracing.

        Zero LLM tokens — pure regex extraction.

        """

        candidates: List[str] = []

        title = getattr(ticket, "title", "") or ""

        desc = getattr(ticket, "description", "") or ""

        full_text = f"{title}\n{desc}"

        # ── Source 1: Visual Evidence section ──────────────────────────────

        for m in self._VISUAL_MARKER_RE.finditer(full_text):

            visual_text = m.group(1).strip()

            # Split on sentence-ending periods, commas, semicolons, pipes, bullets

            for segment in re.split(r'[.!,;|â€¢]+', visual_text):

                seg = segment.strip()

                if len(seg) >= 3 and seg.lower() not in self._STOP_LABELS:

                    # If still too long, try to extract quoted substrings first

                    if len(seg) > 60:

                        # Pull out short quoted phrases from the long segment

                        inner_quotes = re.findall(r"['\"]([^'\"]{3,50})['\"]", seg)

                        if inner_quotes:

                            candidates.extend(inner_quotes)

                        # Also extract camelCase/PascalCase identifiers

                        identifiers = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', seg)

                        candidates.extend(identifiers)

                        # Skip the raw long sentence — it's not grepable

                        continue

                    candidates.append(seg)

        # ── Source 2: Quoted strings in ticket ────────────────────────────

        for m in self._QUOTED_RE.finditer(full_text):

            q = m.group(1).strip()

            if len(q) >= 3:

                candidates.append(q)

        # ── Source 3: Title Case phrases (multi-word proper nouns) ────────

        for m in self._TITLE_CASE_RE.finditer(full_text):

            phrase = m.group(1).strip()

            # Reject if it spans a newline (defensive — regex shouldn't match,

            # but guard against edge cases in formatted ticket text).

            if '\n' in phrase or '\r' in phrase:

                continue

            words = phrase.split()

            # Must have at least 2 words and at least 1 substantive non-stop

            # word (>= 3 chars).  Filters out "In the", "Is not", etc.

            if len(words) >= 2:

                non_stop = [

                    w for w in words

                    if w.lower() not in self._STOP_LABELS and len(w) >= 3

                ]

                if non_stop:

                    candidates.append(phrase)

        # ── Source 4: Hypothesis anchors ──────────────────────────────────

        for h in (hypotheses or []):

            for anchor in getattr(h, "anchors", []):

                if anchor and len(anchor) >= 3:

                    candidates.append(anchor)

        # ── Source 5: Code-like identifiers from ticket text ──────────────

        # Extract camelCase, PascalCase, snake_case identifiers from ticket

        # text that look like class/method/variable names

        code_idents = re.findall(

            r'\b([a-z][a-zA-Z0-9]*(?:[A-Z][a-z0-9]+)+)\b',  # camelCase

            full_text,

        )

        code_idents += re.findall(

            r'\b([A-Z][a-z]+(?:[A-Z][a-z]+){1,})\b',  # PascalCase

            full_text,

        )

        code_idents += re.findall(

            r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+){1,})\b',  # snake_case

            full_text,

        )

        for ident in code_idents:

            if len(ident) >= 4 and ident.lower() not in self._STOP_LABELS:

                candidates.append(ident)

        # ── Source 6: Error messages from ticket text ─────────────────────

        # Error messages like "Name has already been used" are the MOST

        # powerful i18n search anchors because they appear VERBATIM in code.

        # Extract sentences that look like error/validation messages.

        error_patterns = re.findall(

            r'(?:error|message|shows?|displays?|says?|appears?)[:\s]+["\']?([A-Z][^.!?\n]{8,80})[.!?\n"\']',

            full_text, re.IGNORECASE,

        )

        for ep in error_patterns:

            cleaned = ep.strip().strip("'\"")

            if len(cleaned) >= 8 and cleaned.lower() not in self._STOP_LABELS:

                # Error messages get highest priority — insert at front

                candidates.insert(0, cleaned)

        # Also extract any sentence containing "already" / "cannot" / "unable"

        # / "not allowed" which are almost always validation error messages

        validation_phrases = re.findall(

            r'["\']([^"\']{8,60}(?:already|cannot|unable|not allowed|must be|is required|invalid)[^"\']{0,40})["\']',

            full_text, re.IGNORECASE,

        )

        for vp in validation_phrases:

            cleaned = vp.strip()

            if len(cleaned) >= 8:

                candidates.insert(0, cleaned)

        # ── Deduplicate & filter ──────────────────────────────────────────

        seen: set = set()

        unique: List[str] = []

        for c in candidates:

            cleaned = c.strip()

            # Reject any label containing newlines (multi-line artefact)

            if '\n' in cleaned or '\r' in cleaned:

                continue

            # Reject labels that are too long to be useful grep terms

            if len(cleaned) > 60:

                continue

            key = cleaned.lower()

            if key in seen or len(key) < 3:

                continue

            # Skip pure stop-word labels

            if key in self._STOP_LABELS:

                continue

            seen.add(key)

            unique.append(cleaned)

        # Cap at 15 labels (increased to account for error messages)

        return unique[:15]

    # =========================================================================

    # STAGE 0: i18n TRANSLATION CHAIN TRACER

    # =========================================================================

    # Regex: Extract JSON key from a line like '"add.new.contract.picklists": "Pick lists"'

    _JSON_KV_RE = re.compile(

        r'["\']([\w.\-]+)["\']\s*:\s*["\']([^"\'\']+)["\'\']'

    )

    # Regex: Extract properties key from 'key=value' or 'key: value' lines

    _PROPS_KV_RE = re.compile(

        r'^\s*([\w.\-]+)\s*[=:]\s*(.+)$', re.MULTILINE

    )

    # Regex: Extract export name from 'export const XXX' or 'export class XXX'

    _EXPORT_RE = re.compile(

        r'export\s+(?:const|class|function|enum|interface|type)\s+(\w+)'

    )

    def _trace_i18n_chain(

        self,

        ui_label: str,

        visited: set,

    ) -> List[EvidenceItem]:

        """

        Agentic iterative chain search: trace UI text → i18n key → component.

        Supports multiple frameworks:

          Angular:  en.json   → {{ 'key' | translate }}  → component.ts/.html

          React:    en.json   → t('key') / intl.formatMessage  → component.tsx

          Java:     messages.properties  → getMessage("key")  → Service.java

        This is an AGENTIC search — each step's output feeds the next step's input.

        Unlike one-shot vector search, this GUARANTEES finding the component

        that renders a given UI label.

        Zero LLM tokens — pure RepositorySearchEngine (live grep) + regex.

        """

        if not self.repo_search or len(ui_label.strip()) < 3:

            return []

        items: List[EvidenceItem] = []

        traced_files: set = set(visited)

        # ── Step 1: Search i18n JSON files (Angular/React) ────────────────
        # First try the full label. If that returns 0 hits, try progressively
        # shorter suffixes — many i18n values are suffixes like " has already
        # been used" that get concatenated with a dynamic prefix at runtime,
        # so the full user-visible text never appears in any file.

        json_hits = self.repo_search.search_literal(
            ui_label,
            extensions={".json"},
            max_results=15,
        )

        # Fallback: try shorter substrings of the label
        if not json_hits:
            label_words = ui_label.strip().split()
            fallback_queries = []

            # Try suffix substrings (last N words), longest first
            for start in range(1, len(label_words)):
                suffix = " ".join(label_words[start:])
                if len(suffix) >= 10:  # skip very short fragments
                    fallback_queries.append(suffix)

            # Also try the distinctive middle portion (skip first and last word)
            if len(label_words) > 3:
                middle = " ".join(label_words[1:-1])
                if len(middle) >= 10 and middle not in fallback_queries:
                    fallback_queries.append(middle)

            for fallback_q in fallback_queries[:3]:  # try max 3 substrings
                json_hits = self.repo_search.search_literal(
                    fallback_q,
                    extensions={".json"},
                    max_results=15,
                )
                if json_hits:
                    logger.info(
                        f"    [i18n] Full label '{ui_label}' had 0 hits; "
                        f"substring '{fallback_q}' found {len(json_hits)} hit(s)"
                    )
                    break

        i18n_keys_found: List[str] = []
        _key_roots: set = set()  # app roots of translation files (same-app consumer constraint)

        for hit in json_hits:

            fp = hit.file_path

            # Only target i18n/locale/assets directories (avoid package.json noise)

            fp_lower = fp.lower().replace("\\", "/")

            is_i18n = any(

                seg in fp_lower

                for seg in ("/i18n/", "/locale", "/assets/", "/translations/", "/lang/")

            )

            if not is_i18n:

                continue

            # Extract the translation key from the matched line

            for m in self._JSON_KV_RE.finditer(hit.matched_text):

                key, value = m.group(1), m.group(2)

                # Verify the value matches our UI label (bidirectional:

                # label-in-value OR value-in-label, because i18n values

                # can be suffixes e.g. " has already been used" where the

                # full error is assembled at runtime with a prefix)

                lbl_low = ui_label.lower().strip()

                val_low = value.lower().strip()

                if lbl_low in val_low or val_low in lbl_low:

                    i18n_keys_found.append(key)
                    _key_roots.add(_top_root(fp))

                    logger.debug(

                        f"    [i18n] Found key '{key}' = '{value}' "

                        f"in {fp}"

                    )

        # ── Step 1a: Full JSON parse → dotted key paths ──────────────────
        # The line regex above only captures the innermost "key": "value"
        # pair on the matched line, so a nested translation file yields the
        # leaf key but never its dotted path (section.sub.key) — and
        # components reference the full path ('section.sub.key' | translate
        # or t('section.sub.key')).  Parse each matched translation file as
        # real JSON and reconstruct complete key paths; dotted paths are
        # strictly more precise than bare leaves, so they are prepended.
        _seen_i18n_json: set = set()
        # Some callers construct the loop without a workspace (unit tests,
        # dry-run stubs); the JSON-parse pass needs real files, so skip it
        # there just like every other disk-backed step does.
        _ws = getattr(self, '_workspace', None)
        for hit in (json_hits if _ws else []):
            fp = hit.file_path
            fp_lower = fp.lower().replace("\\", "/")
            if not any(
                seg in fp_lower
                for seg in ("/i18n/", "/locale", "/assets/", "/translations/", "/lang/")
            ):
                continue
            if fp in _seen_i18n_json:
                continue
            _seen_i18n_json.add(fp)
            _abs = self._workspace / fp
            try:
                if not _abs.exists() or _abs.stat().st_size > 2_000_000:
                    continue
                _data = json.loads(_abs.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            if not isinstance(_data, dict):
                continue

            def _walk_json_keys(node, prefix):
                """Yield (dotted_key, string_value) leaves of a translation dict."""
                if isinstance(node, dict):
                    for k, v in node.items():
                        dotted = f"{prefix}.{k}" if prefix else str(k)
                        if isinstance(v, str):
                            yield dotted, v
                        else:
                            yield from _walk_json_keys(v, dotted)
                elif isinstance(node, list):
                    for v in node:
                        yield from _walk_json_keys(v, prefix)

            _lbl = ui_label.lower().strip()
            _dotted_keys: List[str] = []
            try:
                for dotted, value in _walk_json_keys(_data, ""):
                    v_low = value.lower().strip()
                    if (_lbl in v_low or v_low in _lbl) and dotted not in i18n_keys_found:
                        _dotted_keys.append(dotted)
                        if len(_dotted_keys) >= 5:
                            break
            except Exception:
                continue
            if _dotted_keys:
                i18n_keys_found = _dotted_keys + [
                    k for k in i18n_keys_found if k not in _dotted_keys
                ]
                _key_roots.add(_top_root(fp))
                logger.info(
                    f"    [i18n] JSON parse reconstructed dotted key path(s) "
                    f"{_dotted_keys} from {fp}"
                )

        # ── Step 1b: Search .properties files (Java/Spring) ───────────────

        props_hits = self.repo_search.search_literal(

            ui_label,

            extensions={".properties"},

            max_results=10,

        )

        for hit in props_hits:

            for m in self._PROPS_KV_RE.finditer(hit.matched_text):

                key, value = m.group(1), m.group(2).strip()

                lbl_low = ui_label.lower().strip()

                val_low = value.lower().strip()

                if lbl_low in val_low or val_low in lbl_low:

                    i18n_keys_found.append(key)
                    _key_roots.add(_top_root(hit.file_path))

                    logger.debug(

                        f"    [i18n] Found properties key '{key}' = '{value}' "

                        f"in {hit.file_path}"

                    )

        if not i18n_keys_found:

            logger.debug(f"    [i18n] No translation keys found for '{ui_label}'")

            # ── DIRECT TEXT SEARCH FALLBACK ────────────────────────────────

            # Many strings are hardcoded (not in i18n files). Search for the

            # raw UI label text directly in HTML/TS/Java source files.

            # This catches error messages like "Name has already been used"

            # that appear verbatim in component templates or validators.

            if len(ui_label) >= 8 and self.repo_search:

                direct_hits = self.repo_search.search_literal(

                    ui_label,

                    extensions={".ts", ".html", ".java", ".tsx", ".jsx"},

                    max_results=10,

                )

                # Collect valid hits (skip already-traced + test files).
                _valid = [
                    dh for dh in direct_hits
                    if dh.file_path not in traced_files
                    and ".spec." not in dh.file_path
                    and "/test/" not in dh.file_path.lower()
                ]
                # App-root cohesion (same repo-metadata reasoning as the key path):
                # a verbatim UI label localizes to its coherent application root.
                # If a strict-majority root exists, drop cross-root outliers so a
                # lone verbatim match in an unrelated service is not treated as a
                # grounded localization. When roots are evenly split, keep all
                # (preserves legitimate cross-boundary strings).
                _roots = [_top_root(dh.file_path) for dh in _valid if _top_root(dh.file_path)]
                _majority_root = None
                if _roots:
                    from collections import Counter as _Counter
                    _rtop, _rn = _Counter(_roots).most_common(1)[0]
                    if _rn * 2 > len(_roots):
                        _majority_root = _rtop

                for dh in _valid:

                    _dr = _top_root(dh.file_path)
                    if _majority_root and _dr and _dr != _majority_root:
                        continue  # cross-root outlier — not a grounded localization

                    traced_files.add(dh.file_path)

                    items.append(EvidenceItem(

                        provider="visual_chain",

                        strength="medium",  # direct-text CANDIDATE — not yet relationship-grounded

                        details=(

                            f"direct_text_candidate: '{ui_label}' found in {dh.file_path}"

                        ),

                        file_path=dh.file_path,

                        evidence_type="usage",

                        content_snippet=(

                            f"[direct_text:{ui_label!r}] line {dh.line_number}: "

                            f"{dh.matched_text[:200]}"

                        ),

                        relevance_score=0.70,  # Below key-chain (0.92) — supplementary candidate

                        hypothesis_id="i18n_chain",

                    ))

                if items:

                    logger.info(

                        f"    [i18n] Direct text fallback for '{ui_label}': "

                        f"{len(items)} candidate file(s) "

                        + (f"(root={_majority_root})" if _majority_root else "(multi-root)")

                    )

            return items

        # ── Step 2: Trace each key to the component that uses it ──────────

        for key in i18n_keys_found[:5]:  # Cap to prevent explosion

            key_hits = self.repo_search.search_literal(

                key,

                extensions={".ts", ".tsx", ".html", ".java", ".jsx"},

                max_results=10,

            )

            if not key_hits and "." in key:
                # Full dotted key has no direct consumers — components may
                # bind the parent prefix and resolve the leaf at runtime.
                _parent_prefix = key.rsplit(".", 1)[0] + "."
                key_hits = self.repo_search.search_literal(
                    _parent_prefix,
                    extensions={".ts", ".tsx", ".html", ".java", ".jsx"},
                    max_results=10,
                )
                if key_hits:
                    logger.info(
                        f"    [i18n] No consumers for '{key}'; parent prefix "
                        f"'{_parent_prefix}' found {len(key_hits)} candidate(s)"
                    )
            for key_hit in key_hits:

                kfp = key_hit.file_path

                if kfp in traced_files:

                    continue

                # Skip i18n JSON files themselves (we want the COMPONENT)

                if kfp.endswith(".json"):

                    continue

                # Same-app constraint: an i18n key's consumer must live in the
                # same application root as the translation file that defined it.
                # Prevents a UI key from pulling in unrelated cross-service files.
                if _key_roots and _top_root(kfp) and _top_root(kfp) not in _key_roots:
                    continue

                traced_files.add(kfp)

                items.append(EvidenceItem(

                    provider="visual_chain",

                    strength="strong",

                    details=(

                        f"i18n_chain: '{ui_label}' → key '{key}' → {kfp}"

                    ),

                    file_path=kfp,

                    evidence_type="usage",

                    content_snippet=(

                        f"[i18n:{key!r}] line {key_hit.line_number}: "

                        f"{key_hit.matched_text[:200]}"

                    ),

                    relevance_score=0.92,  # Very high — direct chain

                    hypothesis_id="i18n_chain",

                ))

                # ── Step 3: If it's a model/constants file (not a component),

                #    trace its export to the component that imports it ─────

                if (

                    kfp.endswith(".ts")

                    and not kfp.endswith(".component.ts")

                    and not kfp.endswith(".spec.ts")

                ):

                    export_name = self._extract_export_name(kfp)

                    if export_name:

                        usage_hits = self.repo_search.search_literal(

                            export_name,

                            extensions={".ts", ".tsx"},

                            max_results=8,

                        )

                        for usage in usage_hits:

                            if usage.file_path in traced_files:

                                continue

                            traced_files.add(usage.file_path)

                            items.append(EvidenceItem(

                                provider="visual_chain",

                                strength="strong",

                                details=(

                                    f"i18n_usage: '{export_name}' "

                                    f"imported in {usage.file_path}"

                                ),

                                file_path=usage.file_path,

                                evidence_type="usage",

                                content_snippet=(

                                    f"[import:{export_name!r}] "

                                    f"line {usage.line_number}: "

                                    f"{usage.matched_text[:200]}"

                                ),

                                relevance_score=0.90,

                                hypothesis_id="i18n_chain",

                            ))

        logger.info(

            f"    [i18n] Chain trace for '{ui_label}': "

            f"{len(i18n_keys_found)} key(s) → {len(items)} component(s)"

        )

        return items

    def _extract_export_name(self, rel_path: str) -> Optional[str]:

        """

        Read a .ts file and extract the first 'export const/class/function' name.

        Used to trace model files → their importing components.

        """

        try:

            full = self._workspace / rel_path

            if not full.exists():

                return None

            content = full.read_text(encoding="utf-8", errors="ignore")[:5000]

            m = self._EXPORT_RE.search(content)

            return m.group(1) if m else None

        except Exception:

            return None

    # =========================================================================

    # PROGRESSIVE ARTIFACT INSPECTION

    # =========================================================================

    def _inspect_artifact(
        self,
        file_path: str,
        ticket_context: str,
        current_evidence: Dict,
        hypotheses_text: str = "",
    ) -> ArtifactInspection:
        """Progressive inspection of a discovered+verified artifact.

        Uses the strongest available structural provider in a
        completeness-driven order:
          1. SQLite index (pre-computed symbols + edges) — instant
          2. WorkspaceSymbolIndex / SymbolResolver (on-demand) — if index incomplete
          3. APIContractDetector — if file is a boundary artifact
          4. Targeted file read — only for specific relevant regions

        Never sends an entire large file to the LLM.
        """
        inspection = ArtifactInspection(file_path=file_path)
        ext = Path(file_path).suffix.lower()
        lang_map = {
            ".ts": "typescript", ".tsx": "typescript",
            ".js": "javascript", ".jsx": "javascript",
            ".java": "java", ".py": "python",
            ".html": "html", ".scss": "scss", ".css": "css",
            ".kt": "kotlin", ".cs": "csharp",
        }
        inspection.language = lang_map.get(ext, "")

        try:
            # ── Level 1: SQLite index query (zero file I/O) ─────────────
            index_symbols = self._query_index_structure(file_path)
            index_edges = self._query_index_edges(file_path)

            if index_symbols:
                inspection.providers_used.append("sqlite")
                inspection.inspection_depth = "index"

                # Extract class name and methods
                for sym in index_symbols:
                    if sym["kind"] == "class":
                        inspection.class_name = sym["name"]
                    elif sym["kind"] == "method":
                        inspection.methods.append(MethodLocation(
                            name=sym["name"],
                            signature=sym.get("signature", sym["name"]),
                            return_type=sym.get("return_type", ""),
                            line_start=sym.get("start_line"),
                            line_end=sym.get("end_line"),
                        ))
                    elif sym["kind"] in ("property", "field"):
                        inspection.properties.append(sym["name"])

            if index_edges:
                for edge in index_edges:
                    if edge["kind"] == "has_type":
                        inspection.type_dependencies.append(edge["dst_name"])
                    elif edge["kind"] == "imports":
                        inspection.imports.append(edge["dst_name"])

            # ── Level 2: WorkspaceSymbolIndex / WorkspaceSymbolScanner ──
            has_methods_with_lines = any(m.line_start is not None for m in inspection.methods)
            if not inspection.methods or not has_methods_with_lines:
                symbol_index = getattr(self.localizer, "symbol_index", None)
                if symbol_index:
                    try:
                        members = symbol_index.get_members_for_file(file_path)
                        if members:
                            inspection.providers_used.append("workspace_symbol_index")
                            inspection.inspection_depth = "symbols"
                            if not inspection.class_name and getattr(members, "class_name", None):
                                inspection.class_name = members.class_name
                            for m in getattr(members, "methods", []):
                                params_str = ", ".join(m.params) if hasattr(m, "params") and m.params else ""
                                sig = f"{m.name}({params_str})" if params_str else m.name
                                if not any(existing.name == m.name for existing in inspection.methods):
                                    inspection.methods.append(MethodLocation(
                                        name=m.name,
                                        signature=sig,
                                        return_type=getattr(m, "return_type", ""),
                                        parameters=getattr(m, "params", []) or [],
                                        visibility=getattr(m, "visibility", "public"),
                                        line_start=getattr(m, "line_start", None),
                                        line_end=getattr(m, "line_end", None),
                                    ))
                            for p in getattr(members, "properties", []):
                                p_name = getattr(p, "name", str(p))
                                if p_name not in inspection.properties:
                                    inspection.properties.append(p_name)
                    except Exception as e:
                        logger.debug(f"  [Inspection] WorkspaceSymbolIndex failed for {file_path}: {e}")

                # Level 2b: Fallback to WorkspaceSymbolScanner.scan_content for grounded signatures
                if not inspection.methods or not any(m.line_start is not None for m in inspection.methods):
                    try:
                        ws = Path(self._workspace) if hasattr(self, "_workspace") and self._workspace else None
                        full_p = Path(file_path)
                        if ws and not full_p.is_absolute():
                            full_p = ws / file_path
                        if full_p.exists():
                            content = full_p.read_text(encoding="utf-8", errors="ignore")
                            from ticket_to_code.agents.workspace_symbol_scanner import WorkspaceSymbolScanner
                            scanner = WorkspaceSymbolScanner(workspace_path=str(ws) if ws else ".")
                            scanned = scanner.scan_content(content, str(full_p))
                            if scanned and scanned.symbols:
                                if "symbol_scanner" not in inspection.providers_used:
                                    inspection.providers_used.append("symbol_scanner")
                                inspection.inspection_depth = "symbols"
                                if not inspection.class_name and scanned.class_names:
                                    inspection.class_name = scanned.class_names[0]
                                if not inspection.language:
                                    inspection.language = scanned.language
                                for s in scanned.symbols:
                                    if s.kind == "method":
                                        existing_m = next((m for m in inspection.methods if m.name == s.name), None)
                                        if existing_m:
                                            if s.source_line and not existing_m.line_start:
                                                existing_m.line_start = s.source_line
                                            if s.params and not existing_m.parameters:
                                                existing_m.parameters = s.params
                                            if s.return_type and not existing_m.return_type:
                                                existing_m.return_type = s.return_type
                                        else:
                                            params_str = ", ".join(s.params) if s.params else ""
                                            sig = f"{s.name}({params_str})" if s.params is not None else s.name
                                            inspection.methods.append(MethodLocation(
                                                name=s.name,
                                                signature=sig,
                                                return_type=s.return_type or "",
                                                parameters=s.params or [],
                                                line_start=s.source_line,
                                                line_end=(s.source_line + 35) if s.source_line else None,
                                                visibility=s.access_level or "public",
                                            ))
                                    elif s.kind == "property":
                                        if s.name not in inspection.properties:
                                            inspection.properties.append(s.name)
                    except Exception as e:
                        logger.debug(f"  [Inspection] Symbol scanner fallback failed for {file_path}: {e}")

            # ── Level 3: APIContractDetector (for boundary artifacts) ────
            if inspection.methods and (inspection.type_dependencies or inspection.imports):
                try:
                    from ticket_to_code.agents.api_contract_detector import APIContractDetector

                    ws = Path(self._workspace) if hasattr(self, "_workspace") and self._workspace else None
                    full_path = Path(file_path)
                    if ws and not full_path.is_absolute():
                        full_path = ws / file_path
                    if full_path.exists():
                        content = full_path.read_text(encoding="utf-8", errors="ignore")
                        detector = APIContractDetector()
                        contracts = detector.detect_contracts(file_path, content)
                        if contracts:
                            inspection.providers_used.append("api_contract_detector")
                            inspection.inspection_depth = "contracts"
                            for c in contracts:
                                contract_str = f"{c.framework} {c.method} {c.endpoint}"
                                if c.source_symbol:
                                    contract_str += f" → {c.source_symbol}()"
                                if contract_str not in inspection.api_contracts:
                                    inspection.api_contracts.append(contract_str)
                except Exception as e:
                    logger.debug(f"  [Inspection] APIContractDetector failed for {file_path}: {e}")

            inspection.provider = inspection.providers_used[0] if inspection.providers_used else "sqlite"

            # ── Identify relevant symbols ────────────────────────────────
            if inspection.methods:
                self._identify_relevant_symbols(
                    inspection, ticket_context, hypotheses_text
                )

            # ── Generate facts and questions ─────────────────────────────
            self._generate_inspection_knowledge(
                inspection, file_path, ticket_context
            )

            logger.info(
                f"  🔬 Inspected {Path(file_path).name}: "
                f"class={inspection.class_name}, "
                f"{len(inspection.methods)} methods, "
                f"{len(inspection.relevant_methods)} relevant, "
                f"{len(inspection.facts)} facts, "
                f"{len(inspection.questions)} questions "
                f"[providers: {', '.join(inspection.providers_used)}]"
            )

        except Exception as e:
            logger.warning(f"  [Inspection] Failed for {file_path}: {e}")

        return inspection

    def _should_inspect(
        self,
        file_path: str,
        verdict: Dict,
        current_evidence: Dict,
        ticket_context: str,
        knowledge: EvidenceKnowledge,
    ) -> bool:
        """Determine if this artifact warrants deeper inspection.

        Driven by evidence, relationships, and current uncertainty.
        NOT driven by filename patterns or file extensions.

        Criteria:
        1. Dependency role: Another included artifact depends on this one
           AND the dependency is relevant to a current question/hypothesis
        2. Current uncertainty: Unresolved questions that this artifact might answer
        3. Behavioral content: SQLite shows it has methods/logic (not just declarations)
        4. Already inspected: Don't re-inspect
        """
        # Already inspected
        existing = current_evidence.get(file_path, {})
        if existing.get("inspection"):
            return False

        # Filter out compiled/vendor bundles and documentation files from code inspection
        from ticket_to_code.agents.evidence_ranking_engine import (
            _is_test_file, _is_generated_or_vendor, _is_docs_file
        )
        if _is_generated_or_vendor(file_path) or _is_docs_file(file_path):
            return False

        # Quick structural check via SQLite: does this file have code worth inspecting?
        index_symbols = self._query_index_structure(file_path)
        has_methods = any(s["kind"] == "method" for s in index_symbols) if index_symbols else False
        has_class = any(s["kind"] == "class" for s in index_symbols) if index_symbols else False

        # Check edges: is this file a dependency of something already in evidence?
        is_dependency_of_included = False
        for other_fp, other_ev in current_evidence.items():
            if not other_ev.get("relevant"):
                continue
            other_inspection = other_ev.get("inspection")
            if other_inspection and isinstance(other_inspection, ArtifactInspection):
                # Check if this file provides a type that other artifacts depend on
                basename = Path(file_path).stem
                for dep in other_inspection.type_dependencies:
                    if dep.lower() in basename.lower() or basename.lower() in dep.lower():
                        is_dependency_of_included = True
                        break

        # Check if file's domain intersects with unresolved questions
        answers_questions = False
        if knowledge.all_questions:
            file_stem = Path(file_path).stem.lower()
            for q in knowledge.all_questions:
                if q not in knowledge.resolved_questions:
                    # Simple intersection: does the file name/path relate to the question?
                    q_lower = q.lower()
                    if file_stem in q_lower or any(
                        word in q_lower for word in file_stem.replace("-", " ").replace("_", " ").split()
                        if len(word) > 3
                    ):
                        answers_questions = True
                        break

        # Decision: inspect if it has behavioral content AND is relevant
        if has_class or has_methods:
            return True  # Has code structure worth understanding

        if is_dependency_of_included and answers_questions:
            return True  # Dependency + answers a question

        # For files without indexed symbols (could be TS with incomplete index),
        # check if WorkspaceSymbolIndex might be able to extract more
        ext = Path(file_path).suffix.lower()
        if ext in (".ts", ".tsx", ".java", ".py", ".kt", ".cs"):
            if is_dependency_of_included:
                return True  # Code file that's a dependency — worth trying

        return False

    def _query_index_structure(self, file_path: str) -> List[Dict]:
        """Query SQLite symbols table for structural info about a file."""
        if not self.localizer.sqlite_store:
            return []
        try:
            conn = self.localizer.sqlite_store._conn

            rel_path = file_path
            ws = Path(self._workspace) if hasattr(self, "_workspace") and self._workspace else None
            if ws:
                try:
                    rel_path = str(Path(file_path).relative_to(ws)).replace("\\", "/")
                except ValueError:
                    pass
            norm_path = file_path.replace("\\", "/")
            file_name = Path(file_path).name

            # Try extended query with line ranges first (newer schema)
            try:
                rows = conn.execute(
                    "SELECT name, kind, start_line, end_line, signature, return_type "
                    "FROM symbols WHERE path IN (?, ?, ?) OR path LIKE ? ORDER BY start_line",
                    (file_path, rel_path, norm_path, f"%{file_name}"),
                ).fetchall()
                if rows:
                    result = []
                    for row in rows:
                        result.append({
                            "name": row[0],
                            "kind": row[1],
                            "start_line": row[2],
                            "end_line": row[3],
                            "signature": row[4] if len(row) > 4 else None,
                            "return_type": row[5] if len(row) > 5 else None,
                        })
                    return result
            except Exception:
                pass

            # Fallback: basic columns only (older schema)
            rows = conn.execute(
                "SELECT name, kind FROM symbols WHERE path IN (?, ?, ?) OR path LIKE ?",
                (file_path, rel_path, norm_path, f"%{file_name}"),
            ).fetchall()
            return [
                {"name": row[0], "kind": row[1],
                 "start_line": None, "end_line": None,
                 "signature": None, "return_type": None}
                for row in rows
            ]
        except Exception as e:
            logger.debug(f"  [Inspection] SQLite structure query failed for {file_path}: {e}")
            return []

    def _query_index_edges(self, file_path: str) -> List[Dict]:
        """Query SQLite edges table for relationships of a file."""
        if not self.localizer.sqlite_store:
            return []
        try:
            conn = self.localizer.sqlite_store._conn

            rel_path = file_path
            ws = Path(self._workspace) if hasattr(self, "_workspace") and self._workspace else None
            if ws:
                try:
                    rel_path = str(Path(file_path).relative_to(ws)).replace("\\", "/")
                except ValueError:
                    pass
            norm_path = file_path.replace("\\", "/")
            file_name = Path(file_path).name

            rows = conn.execute(
                "SELECT kind, dst_name FROM edges WHERE path IN (?, ?, ?) OR path LIKE ?",
                (file_path, rel_path, norm_path, f"%{file_name}"),
            ).fetchall()
            return [{"kind": row[0], "dst_name": row[1] or ""} for row in rows]
        except Exception as e:
            logger.debug(f"  [Inspection] SQLite edges query failed for {file_path}: {e}")
            return []

    def _identify_relevant_symbols(
        self,
        inspection: ArtifactInspection,
        ticket_context: str,
        hypotheses_text: str,
    ) -> None:
        """Identify which discovered symbols are relevant to the ticket.

        If a class has <= 3 methods, all methods are marked relevant so full
        interface is understood. For larger classes, matches against ticket keywords.
        """
        if not inspection.methods:
            return

        # If class has <= 3 methods, all are relevant to understanding capability
        if len(inspection.methods) <= 3:
            for m in inspection.methods:
                if m not in inspection.relevant_methods:
                    inspection.relevant_methods.append(m)
                    if m.line_start:
                        inspection.relevant_regions.append((m.line_start, m.line_end or (m.line_start + 40)))
            return

        # Build keyword set from ticket + hypotheses
        combined_text = (ticket_context + " " + hypotheses_text).lower()
        keywords = set()
        for word in re.findall(r'[a-zA-Z]{3,}', combined_text):
            keywords.add(word.lower())

        # Match methods against keywords
        for method in inspection.methods:
            method_lower = method.name.lower()
            method_words = set(
                w.lower() for w in re.findall(r'[A-Z]?[a-z]+|[A-Z]+', method.name)
                if len(w) > 2
            )
            overlap = keywords & method_words
            if overlap:
                if method not in inspection.relevant_methods:
                    inspection.relevant_methods.append(method)
                    if method.line_start:
                        inspection.relevant_regions.append(
                            (method.line_start, method.line_end or (method.line_start + 40))
                        )

    def _generate_inspection_knowledge(
        self,
        inspection: ArtifactInspection,
        file_path: str,
        ticket_context: str,
    ) -> None:
        """Generate concrete facts and questions from inspection results.

        Preserves exact method signatures, parameters, and return types rather
        than generic capability summaries.
        """
        ticket_lower = ticket_context.lower()
        provider = inspection.providers_used[-1] if inspection.providers_used else "inspection"
        owner = inspection.class_name or Path(file_path).name

        # Fact: Class ownership
        if inspection.class_name:
            inspection.facts.append(InspectionFact(
                fact=f"Class {inspection.class_name} defined in {file_path}",
                source=provider,
                confidence=0.99,
            ))

        # Fact: Concrete relevant methods with exact signatures and return types
        target_methods = inspection.relevant_methods if inspection.relevant_methods else inspection.methods[:3]
        for m in target_methods:
            sig = m.signature or m.name
            if not ("(" in sig and ")" in sig):
                params_str = ", ".join(m.parameters) if m.parameters else ""
                sig = f"{m.name}({params_str})"
            ret_str = f" -> {m.return_type}" if m.return_type else ""
            fact_text = f"{sig}{ret_str} exists in {owner} (file: {file_path})"

            inspection.facts.append(InspectionFact(
                fact=fact_text,
                source=provider,
                line_start=m.line_start,
                line_end=m.line_end,
                confidence=0.98,
            ))

        # Fact: API contracts found
        for contract in inspection.api_contracts:
            inspection.facts.append(InspectionFact(
                fact=f"API contract: {contract}",
                source="api_contract_detector",
                confidence=0.95,
            ))

        # Fact: Concrete implementation code facts from relevant regions
        for m_name, region in inspection.relevant_code_regions.items():
            code = region.get("code", "").strip()
            if code:
                first_line = code.splitlines()[0][:120].strip()
                inspection.facts.append(InspectionFact(
                    fact=f"Implementation of {m_name} in {file_path}: {first_line}",
                    source="source_read",
                    line_start=region.get("start"),
                    line_end=region.get("end"),
                    confidence=0.99,
                ))

        # Questions: where are dependencies resolved?
        for dep in inspection.type_dependencies:
            dep_words = set(w.lower() for w in re.findall(r'[A-Z]?[a-z]+|[A-Z]+', dep) if len(w) > 2)
            ticket_words = set(w.lower() for w in re.findall(r'[a-zA-Z]{3,}', ticket_lower))
            if dep_words & ticket_words:
                inspection.questions.append(
                    f"Where is {dep} implemented and does it need changes?"
                )

        # Questions: API contracts that might need backend investigation
        for contract in inspection.api_contracts:
            if any(k in contract.lower() for k in ("http", "graphql", "rest", "endpoint")):
                inspection.questions.append(
                    f"Where is the backend handler for {contract}?"
                )

    def _read_relevant_region(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        max_lines: int = 80,
    ) -> str:
        """Read a specific region of a file. Never sends the whole file.

        Returns the content between start_line and end_line, bounded
        by max_lines to prevent accidental context explosion.
        """
        try:
            ws = Path(self._workspace) if hasattr(self, "_workspace") and self._workspace else None
            full = Path(file_path)
            if ws and not full.is_absolute():
                full = ws / file_path
            if not full.exists():
                return ""
            lines = full.read_text(encoding="utf-8", errors="ignore").splitlines()
            if not end_line or end_line <= start_line:
                end_line = start_line + 40
            # 0-indexed
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            # Bound by max_lines
            if end_idx - start_idx > max_lines:
                end_idx = start_idx + max_lines
            return "\n".join(lines[start_idx:end_idx])
        except Exception as e:
            logger.debug(f"  [Inspection] Region read failed for {file_path}: {e}")
            return ""

    # =========================================================================

    # UTILITY

    # =========================================================================

    def _read_snippet(self, rel_path: str, max_chars: int = 300) -> str:

        """Read a short snippet from a file for the evidence content_snippet."""

        try:

            full = self._workspace / rel_path

            if full.exists():

                return full.read_text(encoding="utf-8", errors="ignore")[:max_chars]

        except Exception:

            pass

        return ""


