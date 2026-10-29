"""
Localization Agent - Exact File and Symbol Discovery

This is THE MOST CRITICAL agent in autonomous engineering.

RESPONSIBILITY:
Find EXACT targets for code modifications using structural reasoning (NOT semantic search).

Uses:
- SQLite AST for symbol lookups
- Neo4j for dependency graphs
- Spring intelligence for architecture understanding
- Execution path analysis

Does NOT use:
- Large RAG queries (only architectural guidelines)
- Semantic similarity for localization (unreliable)

Author: Deepak Madgani
Date: May 27, 2026
"""

import json
import logging
import os
import re
from collections import Counter
from typing import List, Optional, Set, Dict
from pathlib import Path

from ticket_to_code.runtime.llm_factory import LLMRegistry
from ticket_to_code.llm_utils import llm_invoke
from langchain_core.messages import SystemMessage, HumanMessage

from ticket_to_code.brain.brain_storage import RepositoryBrainStorage
from ticket_to_code.plan_gating import _extract_json_payload
from ticket_to_code.utils.file_filters import is_ignored_path, prune_walk_dirs, STANDARD_SKIP_DIRS
from ticket_to_code.models import (
    StructuredRequirements,
    LocalizationResult,
    TargetFile,
    TargetMethod,
    ExecutionPath,
    ImpactAnalysis,
    StageThought,
    ThinkingChain,
)

logger = logging.getLogger(__name__)


def _episodic_confidence(verification_level) -> float:
    """Confidence granted to a file recalled from episodic memory, by how the
    source episode was verified: test-verified fixes are trusted most, compile-only
    less, and unknown/unverified episodes are only a weak hint."""
    if verification_level == "test_verified":
        return 0.90
    if verification_level == "compile_only":
        return 0.75
    return 0.5


# ── Ticket-relevance ownership (pure, testable, language/framework agnostic) ──
# The ticket names a FEATURE ("Add Members modal", "Export widget page"). The
# file that IMPLEMENTS that feature is the owner — NOT whichever file happens to
# define/calculate/persist the most symbols. Structural busyness is participation,
# never ownership. A file becomes a primary target only when it matches the
# ticket's feature (a "feature anchor") or the ticket declares it.

_UI_ANCHOR_NOUNS = (
    "modal", "dialog", "page", "screen", "view", "form", "panel", "component",
    "button", "menu", "tab", "list", "table", "card", "section", "widget",
    "popup", "drawer", "sidebar", "header", "footer", "banner", "toolbar", "grid",
)
_FEATURE_STOP = {
    "the", "a", "an", "in", "on", "of", "to", "for", "and", "or", "is", "are",
    "when", "that", "this", "with", "from", "into", "new", "add", "existing",
}
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _split_identifier(s: str) -> list:
    """Split a filename stem into lowercase tokens (camelCase + separators)."""
    s = _CAMEL_RE.sub(" ", s or "")
    return [w.lower() for w in re.split(r"[^A-Za-z0-9]+", s) if w]


def extract_ticket_namespace(text: str, workspace_path: Optional[Path] = None) -> Optional[str]:
    """Extract ticket title namespace prefix like 'Jato:' or 'Payments:'.
    
    If the prefix corresponds to a real directory in the workspace (e.g. src/jato or jato),
    returns the canonical lowercase namespace (e.g. 'jato').
    """
    if not text:
        return None
    m = re.match(r"^(?:\[([A-Za-z0-9_-]+)\]|([A-Za-z0-9_-]+)\s*:)\s*", text.strip())
    if not m:
        return None
    raw_ns = (m.group(1) or m.group(2) or "").lower()
    if raw_ns in _FEATURE_STOP or len(raw_ns) < 2:
        return None
    if workspace_path:
        wp = Path(workspace_path)
        candidates = [
            wp / "src" / raw_ns,
            wp / raw_ns,
            wp / f"{raw_ns}-ui",
            wp / "src" / "app" / raw_ns,
        ]
        if any(c.exists() and c.is_dir() for c in candidates):
            return raw_ns
    return raw_ns


def extract_feature_phrase(text: str) -> list:
    """Extract the ticket's primary feature tokens (generic, no hardcoding).

    Prefers target scope tokens from 'in/on/for <Scope>' (e.g. 'in All Projects' -> ['projects']),
    or 1-2 content words immediately before a UI-anchor noun, filtering out layout defect nouns
    (footer, scroll, overflow) when describing a bug.
    """
    clean_text = re.sub(r"^(?:\[[A-Za-z0-9_-]+\]|[A-Za-z0-9_-]+\s*:)\s*", "", text or "").strip()

    # 1. Target scope check: e.g. "in All Projects", "for Deliverables", "on Contract Details"
    target_match = re.search(r"\b(?:in|on|for|under)\s+(?:all\s+)?([A-Za-z][A-Za-z0-9_-]+)", clean_text, re.IGNORECASE)
    if target_match:
        target_token = target_match.group(1).lower()
        if target_token not in _FEATURE_STOP and len(target_token) >= 3:
            return [target_token]

    # 2. Defect layout words should not be treated as feature anchor nouns when describing a defect
    _DEFECT_NOUNS = {"footer", "header", "scroll", "overflow", "scrollbar", "bar"}

    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9]+", clean_text)]
    for i, w in enumerate(words):
        if w in _UI_ANCHOR_NOUNS and i >= 1 and w not in _DEFECT_NOUNS:
            prev = [x for x in words[max(0, i - 3): i] if x not in _FEATURE_STOP and len(x) >= 3]
            if prev:
                return prev[-2:]
    content = [w for w in words if w not in _FEATURE_STOP and len(w) >= 3 and w not in _DEFECT_NOUNS]
    return content[:2] if content else [w for w in words if w not in _FEATURE_STOP and len(w) >= 3][:2]


def feature_anchor_score(feature_tokens: list, path: str, namespace: Optional[str] = None) -> float:
    """0.0-1.0 measure that a file's NAME matches the ticket's feature phrase.

    High only when the feature phrase appears (contiguously) in the filename stem,
    so a keyword-dense backend service does NOT anchor to a UI feature ticket.
    Supports sub-stem prefix stripping (e.g. 'jprojects' -> 'projects' for Jato).
    """
    toks = [t.lower() for t in (feature_tokens or []) if t and len(str(t)) >= 3]
    if not toks:
        return 0.0
    base = str(path).replace("\\", "/").split("/")[-1]
    stem = base.split(".")[0]
    stem_tokens = _split_identifier(stem)
    if not stem_tokens:
        return 0.0

    # Sub-stem handling: if stem starts with 'j' (e.g. jprojects, jtable) or namespace prefix
    # expand tokens to include the stripped version
    if len(stem) > 3 and stem.startswith("j") and stem[1].isalpha():
        sub_tokens = _split_identifier(stem[1:])
        stem_tokens.extend(sub_tokens)

    stem_join = "".join(stem_tokens)
    # Contiguous 2+ token feature phrase present in the stem → strong anchor.
    contiguous = any(
        "".join(toks[i:j]) in stem_join
        for i in range(len(toks)) for j in range(i + 2, len(toks) + 1)
    )
    if contiguous:
        return 0.85
    # Single distinctive feature token that IS the stem (short, focused filename).
    tokset = set(toks)
    if any(w in tokset for w in stem_tokens):
        matched = sum(1 for w in stem_tokens if w in tokset)
        return round(min(0.6, 0.3 * matched + 0.2), 3)
    return 0.0


_OWNERSHIP_CONFIG_EXTS = {
    ".yml", ".yaml", ".json", ".toml", ".ini", ".properties",
    ".conf", ".cfg", ".xml", ".sh", ".bat", ".ps1",
}


def verify_ownership(path: str, investigation: dict, features: dict) -> tuple:
    """Discovery-time ownership ROLE (advisory, pure, testable).

    NEVER grants PRIMARY from structural counts, semantic similarity, or the
    feature anchor. Primary ownership is proven later by inspection-backed
    behavioral evidence (Phase B), not by discovery. A keyword/symbol/vector-
    dense backend service therefore lands at SUPPORTING, never PRIMARY.

    Roles:
      DISPLAY_OWNER — template/style companion of a feature component
      SUPPORTING    — participates (structural/semantic/config) → inspect to prove
      READ_ONLY     — references only
    """
    investigation = investigation or {}
    features = features or {}
    d = investigation.get("defines", 0)
    di = investigation.get("displays", 0)
    c = investigation.get("calculates", 0)
    p = investigation.get("persists", 0)
    ext = Path(path).suffix.lower()
    s_vec = features.get("s_vec", 0.0)

    # Template/style files → DISPLAY_OWNER (companions of a feature component).
    if ext in {".html", ".scss", ".css", ".sass"}:
        detail = f"{di} display binding(s)" if di else "template/style file"
        return ("DISPLAY_OWNER", detail)

    # Config/infra: an editable candidate, but change-authorization gates it
    # downstream (config is READ_ONLY unless the ticket targets config).
    p_norm = path.replace("\\", "/").lower()
    name = Path(path).name.lower()
    _is_config = (
        ext in _OWNERSHIP_CONFIG_EXTS or name.startswith(".env")
        or name in {"dockerfile", "makefile"}
    )
    _is_locked = (
        name in {
            "package-lock.json", "yarn.lock", "shrinkwrap.json",
            "gradle.lockfile", "poetry.lock", "pipfile.lock", "composer.lock",
        }
        or name.endswith(".lock")
        or "/dist/" in p_norm or "/node_modules/" in p_norm
        or "/target/" in p_norm or "/build/" in p_norm
        or "/.venv/" in p_norm or "/venv/" in p_norm
        or "brain/knowledge/" in p_norm
    )
    if _is_config and not _is_locked:
        return ("SUPPORTING", "config/infrastructure candidate — verify by inspection")

    # Repository, DAO, or Model/Entity interfaces/classes → SUPPORTING (participates in persistence layer)
    _is_data_access = (
        "repository" in name or "dao" in name or "entity" in name
        or "model" in name or "schema" in name
    )
    if _is_data_access and not _is_locked:
        return ("SUPPORTING", "data access / persistence component — verify by inspection")

    # Semantic match OR structural participation → SUPPORTING (candidate to
    # INSPECT). Neither proves ticket ownership on its own.
    if s_vec >= 0.50:
        return ("SUPPORTING", f"semantic match (verify by inspection): s_vec={s_vec:.2f}")
    if d >= 1 or c >= 1 or p >= 1:
        return ("SUPPORTING",
                f"participates structurally (defines={d}, calculates={c}, persists={p}) — verify by inspection")

    return ("READ_ONLY", f"references only (defines={d}, calculates={c}, displays={di})")


# Surfacing boost weight. Feature-anchor only re-orders candidates so the correct
# component is INSPECTED — it never grants ownership or write authority.
_FEATURE_ANCHOR_WEIGHT = 0.5


def apply_feature_anchor_boost(candidates: list, feature_tokens: list,
                               weight: float = _FEATURE_ANCHOR_WEIGHT,
                               namespace: Optional[str] = None) -> list:
    """Raise the rank of candidates whose FILENAME matches the ticket feature
    phrase so the right component surfaces for inspection.

    Also applies namespace routing boost/penalty if ticket specifies a namespace prefix.
    Discovery signal ONLY: it changes ordering and records ``feature_anchor`` in
    features; it never sets ownership_type or authorizes a write. Files with no
    filename match are untouched (semantic/inspection must surface them). Returns
    the list re-sorted by ``unified_score`` (desc).
    """
    toks = [t for t in (feature_tokens or []) if t]
    ns_lower = namespace.lower() if namespace else None

    for cand in candidates:
        p_norm = cand.get("path", "").replace("\\", "/").lower()
        anchor = feature_anchor_score(toks, cand.get("path", ""), namespace=ns_lower) if toks else 0.0
        feats = cand.setdefault("features", {})
        feats["feature_anchor"] = anchor
        base = float(cand.get("unified_score", 0.0) or 0.0)
        sigs = list(cand.get("signals", []))

        # 1. Feature Anchor Boost
        if anchor > 0.0:
            base = round(min(2.0, base + weight * anchor), 6)
            sigs.append(f"feature_anchor:{anchor:.2f}")

        # 2. Namespace Boost / Penalty
        if ns_lower:
            in_ns = (f"/{ns_lower}/" in p_norm or p_norm.startswith(f"src/{ns_lower}/") or p_norm.startswith(f"{ns_lower}/"))
            if in_ns:
                base = round(min(2.0, base + 0.80), 6)
                sigs.append(f"namespace_match:{ns_lower}")
            elif any(comp_seg in p_norm for comp_seg in ["/modules/home/", "/modules/project/"]):
                # Conflicting module when ticket explicitly targeted namespace
                base = round(max(0.0, base - 0.60), 6)
                sigs.append(f"namespace_penalty:{ns_lower}")

        cand["unified_score"] = base
        cand["confidence"] = round(min(0.95, base), 3)
        cand["signals"] = list(dict.fromkeys(sigs))

    candidates.sort(key=lambda x: -float(x.get("unified_score", 0.0) or 0.0))
    return candidates


class LocalizationAgent:
    """
    Localization Agent - Exact Target Discovery
    
    THE MOST IMPORTANT AGENT for autonomous engineering.
    
    Finds EXACT files, classes, and methods to modify using:
    - SQLite AST queries (symbol search)
    - Neo4j graph traversal (dependencies)
    - Spring intelligence (architecture)
    - Execution path analysis (full impact)
    
    Does NOT rely on semantic search for localization.
    
    This agent MUST succeed for the workflow to work correctly.
    """
    
    def __init__(self, workspace_path: str):
        """
        Initialize Localization Agent.
        
        Args:
            workspace_path: Path to project workspace
        """
        self.workspace_path = Path(workspace_path)
        self.llm = LLMRegistry.get_llm()
        self.workflow_discovery = None
        self.enable_consensus_confidence = os.getenv(
            "AVIATOR_ENABLE_CONSENSUS_CONFIDENCE", "0"
        ) == "1"
        self.enable_workflow_precision_gate = os.getenv(
            "AVIATOR_ENABLE_WORKFLOW_PRECISION_GATE", "0"
        ) == "1"
        self.consensus_boost_weight = float(
            os.getenv("AVIATOR_CONSENSUS_BOOST_WEIGHT", "0.08")
        )
        self.workflow_gate_min_direct_sources = int(
            os.getenv("AVIATOR_WORKFLOW_GATE_MIN_DIRECT_SOURCES", "1")
        )
        self.repository_brain = RepositoryBrainStorage()
        self.repository_brain_loaded = False
        
        # Initialize storage connections
        self._init_storage()
        self._maybe_bootstrap_repository_brain()
        self._load_repository_brain()
        
        logger.info("🎯 Localization Agent initialized")
    
    def _maybe_bootstrap_repository_brain(self):
        """Ensure the directory brain exists for this project before localization runs.

        New (or removed-then-re-added) projects have no brain on disk, so this generates it
        once with LLM-authored 'why' descriptions. Already-indexed projects are left as-is;
        freshness is maintained after successful tickets by the memory-update phase.
        """
        if os.getenv("AVIATOR_ENABLE_BRAIN_BOOTSTRAP", "1") != "1":
            return
        try:
            from ticket_to_code.brain.generate_repository_brain import ensure_repository_brain
            summary = ensure_repository_brain(str(self.workspace_path), llm=self.llm)
            logger.info(
                "🧠 Brain bootstrap: %s (dirs=%s)",
                summary.get("bootstrap", summary.get("status")),
                summary.get("directories", "-"),
            )
        except Exception as exc:
            logger.warning("Repository brain bootstrap failed (non-fatal): %s", exc)

    def _init_storage(self):
        """Initialize SQLite and Neo4j connections (each independently)."""
        # ── SQLite ────────────────────────────────────────────────────────────
        try:
            from ticket_to_code.runtime.storage import SqliteStore
            index_path = self.workspace_path / ".aviator" / "index.db"
            if index_path.exists():
                self.sqlite_store = SqliteStore(str(index_path))
                logger.info(f"✅ Connected to SQLite index: {index_path}")
            else:
                logger.warning(f"⚠️ SQLite index not found: {index_path}")
                self.sqlite_store = None
        except Exception as e:
            logger.error(f"SQLite initialization failed: {e}", exc_info=True)
            self.sqlite_store = None

        # ── Neo4j (optional) ─────────────────────────────────────────────────
        try:
            import os, logging as _logging
            if os.getenv("NEO4J_PASSWORD"):
                # Suppress Neo4j driver notification spam (GqlStatusObject warnings
                # for missing labels/properties on a fresh/empty graph).
                _logging.getLogger("neo4j").setLevel(_logging.ERROR)
                _logging.getLogger("neo4j.notifications").setLevel(_logging.ERROR)

                from ticket_to_code.runtime.storage import Neo4jStore
                self.neo4j_store = Neo4jStore(workspace_path=str(self.workspace_path))
                # Monkey-patch a generic query() method if missing.
                # workflow_discovery.py calls neo4j_store.query(cypher, params)
                # but aviator_core's Neo4jStore only exposes high-level methods.
                # (Monkey-patch removed: Neo4jStore now natively provides a secure, scoped query() method)
                logger.info("✅ Connected to Neo4j graph database")
            else:
                logger.warning("⚠️ Neo4j not configured")
                self.neo4j_store = None
        except Exception as e:
            logger.warning(f"Neo4j initialization failed (non-fatal): {e}")
            self.neo4j_store = None

        # ── Vector Store (Semantic Code Search) ─────────────────────────────────────────────────
        # Disabled per architecture requirement: No New Vector Database Yet
        self.vector_store = None

    def _load_repository_brain(self):
        """Load Repository Brain JSON once at startup for Brain-first localization."""
        try:
            semantic_files = self.repository_brain.load_default(str(self.workspace_path))
            self.repository_brain_loaded = True
            
            if not semantic_files:
                logger.warning(
                    "⚠️ Loaded 0 semantic ownership files. Localization will rely on deterministic fallback signals (Neo4j/SQLite/Grep)."
                )
            else:
                logger.info(
                    "✅ Loaded Repository Brain: %s semantic ownership files from %s",
                    len(semantic_files),
                    self.repository_brain.loaded_directory,
                )
        except Exception as exc:
            self.repository_brain_loaded = False
            logger.error(f"Repository Brain initialization failed: {exc}", exc_info=True)

    def localize_targets(
        self,
        requirements: StructuredRequirements,
        investigation_context: Optional[Dict] = None
    ) -> LocalizationResult:
        """
        Find EXACT files and methods to modify.
        
        This is the CORE function of autonomous engineering.
        
        Strategy:
        1. Query SQLite for exact symbol matches
        2. Query Neo4j for dependency analysis
        3. Analyze execution paths
        4. Determine CREATE vs MODIFY decisions
        5. Identify impacted components
        
        Args:
            requirements: Structured requirements from analysis
            investigation_context: Optional context from investigation
            
        Returns:
            LocalizationResult with exact targets and high confidence
        """
        logger.info("🎯 Starting LOCALIZATION (exact target discovery)")
        
        print("\n" + "="*80)
        print("[ENTER] LocalizationAgent.localize_targets()")
        print("   Purpose: Find EXACT files/methods to modify using AST + Graph")
        print(f"   Affected Components: {requirements.affected_components}")
        print("="*80)
        
        if not self.sqlite_store:
            logger.error("❌ SQLite not available - cannot localize")
            return self._fallback_localization(requirements)
        
        # STEP 0: Consult Repository Knowledge Base (RKB)
        logger.info("🧠 STEP 0: Consulting Repository Knowledge Base (RKB)...")
        rkb_files = self._consult_rkb(requirements)
        print(f"BRAIN_ARTIFACTS={len(rkb_files)}")
        if rkb_files:
            logger.info(f"   RKB identified {len(rkb_files)} related files from High-Level Features")
        
        # STEP 1: Find exact files from SQLite
        logger.info("📊 STEP 1: Querying SQLite AST for exact files...")
        target_files = self._find_exact_files(requirements)
        print(f"TARGET_FILES_AFTER_EXACT_MATCH={len(target_files)}")
        
        # Merge RKB files
        existing_paths = {f.file_path for f in target_files}
        for f in rkb_files:
            if f.file_path not in existing_paths:
                target_files.append(f)
                
        # ── Global Isolation Fix: Filter out files that don't exist in the current workspace ──
        # This prevents global graph databases (like Neo4j) or RKB from leaking external repository files
        # into other workspaces.
        valid_target_files = []
        for tf in target_files:
            full_path = os.path.join(self.workspace_path, tf.file_path)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                valid_target_files.append(tf)
        target_files = valid_target_files
        print(f"TARGET_FILES_AFTER_ISOLATION_FILTER={len(target_files)}")
                
        logger.info(f"   Found {len(target_files)} target files (AST + RKB)")
        
        # STEP 2: Find exact methods from SQLite
        logger.info("🔍 STEP 2: Querying SQLite for exact methods...")
        target_methods = self._find_exact_methods(requirements, target_files)
        logger.info(f"   Found {len(target_methods)} target methods")
        
        # STEP 3: Analyze dependencies with Neo4j
        logger.info("🌐 STEP 3: Analyzing dependencies with Neo4j...")
        dependencies = self._analyze_dependencies(target_files)
        
        # ── Global Isolation Fix: Filter out dependencies that don't exist ──
        valid_dependencies = []
        for d in dependencies:
            full_path = os.path.join(self.workspace_path, d.file_path)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                valid_dependencies.append(d)
        dependencies = valid_dependencies
        print(f"TARGET_FILES_AFTER_DEPENDENCIES={len(target_files)}, DEPS={len(dependencies)}")
        
        logger.info(f"   Found {len(dependencies)} dependent components")
        
        # STEP 4: Analyze execution paths
        logger.info("🛤️ STEP 4: Analyzing execution paths...")
        execution_paths = self._analyze_execution_paths(target_files)
        logger.info(f"   Found {len(execution_paths)} execution paths")
        
        # STEP 5: Impact analysis
        logger.info("💥 STEP 5: Performing impact analysis...")
        impact = self._perform_impact_analysis(
            target_files, 
            dependencies, 
            execution_paths
        )
        logger.info(f"   Impact: {len(impact.affected_apis)} APIs, "
                   f"{len(impact.affected_services)} services, "
                   f"{len(impact.affected_controllers)} controllers")
        
        # STEP 6: Calculate confidence
        confidence = self._calculate_localization_confidence(
            target_files,
            target_methods,
            dependencies,
            requirements
        )
        logger.info(f"   Localization confidence: {confidence:.2f}")
        
        result = LocalizationResult(
            target_files=target_files,
            target_methods=target_methods,
            dependencies=dependencies,
            execution_paths=execution_paths,
            impact_analysis=impact,
            confidence=confidence,
            requires_human_review=confidence < 0.75  # Flag low confidence
        )
        
        logger.info(
            f"✅ Localization complete:\n"
            f"   Files: {len(target_files)}\n"
            f"   Methods: {len(target_methods)}\n"
            f"   Dependencies: {len(dependencies)}\n"
            f"   Confidence: {confidence:.2f}\n"
            f"   Needs Review: {result.requires_human_review}"
        )
        
        return result
    
    def _consult_rkb(self, requirements: StructuredRequirements) -> List[TargetFile]:
        """Consult Repository Knowledge Base for feature boundaries."""
        brain_files = self._consult_repository_brain(requirements)
        if brain_files:
            return brain_files

        rkb_files = []
        if not getattr(self, 'vector_store', None) or not getattr(self, 'neo4j_store', None):
            return rkb_files
            
        search_terms = requirements.functional_requirements + requirements.affected_components
        if not search_terms:
            return rkb_files
            
        for req in search_terms:
            try:
                # Semantic search against Features
                if hasattr(self.vector_store, 'search'):
                    results = self.vector_store.search(query=req, limit=1)
                else:
                    continue
                    
                for res in results:
                    feature_name = getattr(res, 'metadata', {}).get("feature_name")
                    if feature_name:
                        # Pull subgraph from Neo4j - Feature nodes are not implemented in Neo4j
                        logger.debug(f"Feature '{feature_name}' found in VectorDB, but Neo4j Feature graph is not implemented.")
            except Exception as e:
                logger.error(f"RKB query failed: {e}")
                
        # deduplicate
        return list({f.file_path: f for f in rkb_files}.values())

    def _consult_repository_brain(self, requirements: StructuredRequirements) -> List[TargetFile]:
        """Query Repository Brain first and convert primary artifacts into TargetFile candidates."""
        if not self.repository_brain_loaded:
            print("BRAIN_REPOSITORY_LOADED=False")
            return []

        search_terms: List[str] = []
        search_terms.extend(requirements.affected_components)
        search_terms.extend(requirements.functional_requirements)
        search_terms.extend(requirements.technical_requirements[:3])
        search_terms = [term.strip() for term in search_terms if term and term.strip()]
        if not search_terms:
            return []

        combined_query = " ".join(search_terms)
        print(f"BRAIN_COMBINED_QUERY_LOCALIZER={combined_query}")
        workflow_matches = self.repository_brain.find_workflows(combined_query)
        print(f"BRAIN_WORKFLOW_MATCHES_LOCALIZER={len(workflow_matches)}")
        if not workflow_matches:
            return []

        workflow_names = [str(match["workflow"]) for match in workflow_matches[:3]]
        artifact_candidates = self.repository_brain.get_primary_artifacts(
            combined_query,
            workflow_matches,
        )
        print(f"BRAIN_ARTIFACT_CANDIDATES={len(artifact_candidates)}")
        if not artifact_candidates:
            return []

        target_files: List[TargetFile] = []
        for artifact in artifact_candidates:
            resolved = self._resolve_repository_brain_artifact(artifact)
            if not resolved:
                continue

            relative_path, absolute_path, language = resolved
            reason = "Repository Brain primary artifact"
            if workflow_names:
                reason += f" for workflows: {', '.join(workflow_names)}"

            target_files.append(TargetFile(
                file_path=relative_path,
                absolute_path=absolute_path,
                language=language,
                task_type="modify",
                confidence=0.82,
                reason=reason,
            ))

        deduped = list({target.file_path: target for target in target_files}.values())
        print(f"BRAIN_RESOLVED_TARGET_FILES={len(deduped)}")
        if deduped:
            logger.info(
                "   Repository Brain identified %s target files across %s domains",
                len(deduped),
                len(self.repository_brain.get_domains()),
            )
        return deduped

    def _resolve_repository_brain_artifact(self, artifact: str) -> Optional[tuple[str, str, str]]:
        """Resolve a Brain artifact string to relative path, absolute path, and language."""
        normalized_artifact = artifact.strip().replace("\\", "/").lstrip("./")
        if not normalized_artifact:
            return None

        direct_match = self.workspace_path / normalized_artifact
        if direct_match.exists() and direct_match.is_file():
            return (
                normalized_artifact,
                str(direct_match),
                self._infer_language_from_path(normalized_artifact),
            )

        basename = Path(normalized_artifact).name
        sqlite_match = self._resolve_artifact_with_sqlite(normalized_artifact, basename)
        if sqlite_match:
            return sqlite_match

        filesystem_match = self._resolve_artifact_with_filesystem(basename)
        if filesystem_match:
            return filesystem_match

        logger.debug(f"Repository Brain artifact could not be resolved: {artifact}")
        return None

    def _resolve_artifact_with_sqlite(
        self,
        artifact_path: str,
        basename: str,
    ) -> Optional[tuple[str, str, str]]:
        """Resolve artifact using the SQLite files table when available."""
        if not self.sqlite_store or not hasattr(self.sqlite_store, "_conn"):
            return None

        try:
            rows = self.sqlite_store._conn.execute(
                """
                SELECT path, language
                FROM files
                WHERE path = ? OR path LIKE ? OR path LIKE ?
                ORDER BY CASE WHEN path = ? THEN 0 ELSE 1 END, LENGTH(path)
                LIMIT 10
                """,
                (
                    artifact_path,
                    f"%/{basename}",
                    f"%{artifact_path}",
                    artifact_path,
                ),
            ).fetchall()
        except Exception as exc:
            logger.debug(f"SQLite artifact resolution failed for '{artifact_path}': {exc}")
            return None

        for path_value, language in rows:
            relative_path = str(path_value).replace("\\", "/")
            absolute_path = self.workspace_path / relative_path
            if absolute_path.exists() and absolute_path.is_file():
                return (
                    relative_path,
                    str(absolute_path),
                    language or self._infer_language_from_path(relative_path),
                )

        return None

    def _resolve_artifact_with_filesystem(self, basename: str) -> Optional[tuple[str, str, str]]:
        """Resolve artifact by basename search as a last fallback."""
        if not basename:
            return None

        try:
            import os
            for root, dirs, files in os.walk(str(self.workspace_path)):
                prune_walk_dirs(dirs, root, self.workspace_path)
                dirs.sort()
                files.sort()
                for file in files:
                    if file == basename:
                        match = Path(root) / file
                        if is_ignored_path(match, workspace_path=self.workspace_path):
                            continue
                        relative_path = match.relative_to(self.workspace_path).as_posix()
                        return (
                            relative_path,
                            str(match),
                            self._infer_language_from_path(relative_path),
                        )
        except Exception as exc:
            logger.debug(f"Filesystem artifact resolution failed for '{basename}': {exc}")

        return None

    def _infer_language_from_path(self, path_value: str) -> str:
        """Infer language from file extension for Repository Brain candidates."""
        suffix = Path(path_value).suffix.lower()
        return {
            ".java": "java",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".py": "python",
            ".cs": "csharp",
            ".go": "go",
            ".html": "html",
            ".scss": "scss",
            ".css": "css",
            ".json": "json",
            ".xml": "xml",
            ".yml": "yaml",
            ".yaml": "yaml",
        }.get(suffix, "unknown")

    def _consult_workflow_paths(self, keywords: List[str]) -> List[str]:
        """Consult workflow discovery for structurally connected file paths."""
        discovery = getattr(self, "workflow_discovery", None)
        if not discovery or not keywords:
            return []

        discovered: List[str] = []
        seen: Set[str] = set()
        for kw in keywords[:5]:
            try:
                for rel_path in discovery.get_workflow_files(kw):
                    normalized = rel_path.replace("\\", "/")
                    if normalized in seen:
                        continue
                    seen.add(normalized)
                    discovered.append(normalized)
            except Exception as exc:
                logger.debug(f"Workflow discovery lookup failed for '{kw}': {exc}")

        return discovered

    def _find_exact_files(
        self, 
        requirements: StructuredRequirements
    ) -> List[TargetFile]:
        """
        Find exact files using SQLite AST queries.
        
        NO semantic search - only exact structural queries.
        """
        target_files = []
        
        # Query 1: Search by component names
        for component in requirements.affected_components:
            logger.info(f"   Searching for: {component}")
            
            # Query SQLite for exact matches — two paths:
            #   1. Files WITH symbols (Java, TS, Python) matched by symbol name or path
            #   2. Files WITHOUT symbols (HTML, SCSS, CSS) matched by path only
            # The UNION ensures template/style files are discoverable even though
            # the indexer registers them as bare FileRecord rows (no symbols).
            query = """
            SELECT DISTINCT f.path, f.language
            FROM files f
            JOIN symbols s ON f.path = s.path
            WHERE s.name LIKE ? OR f.path LIKE ?
            UNION
            SELECT DISTINCT f.path, f.language
            FROM files f
            WHERE f.path LIKE ?
              AND f.language IN ('html', 'scss', 'css')
            """
            
            try:
                # Use parameterized query to prevent SQL injection/syntax errors
                pattern = f"%{component}%"
                results = self.sqlite_store._conn.execute(query, (pattern, pattern, pattern)).fetchall()
                
                for row in results:
                    relative_path, language = row
                    file_path = relative_path
                    
                    # Check if file exists
                    full_path = self.workspace_path / relative_path
                    if not full_path.exists():
                        logger.warning(f"   ⚠️ File not found: {relative_path}")
                        continue
                    
                    # Determine CREATE vs MODIFY
                    task_type = "modify"  # File exists in SQLite
                    
                    target_file = TargetFile(
                        file_path=relative_path,
                        absolute_path=str(full_path),
                        language=language,
                        task_type=task_type,
                        confidence=0.95,  # High confidence - exact match from AST
                        reason=f"Exact match for component: {component}"
                    )
                    
                    target_files.append(target_file)
                    logger.info(f"   ✅ Found: {relative_path} ({task_type})")
                    
            except Exception as e:
                logger.error(f"   ❌ Query failed for {component}: {e}")
                continue
        
        # Query 2: Check for missing files (CREATE scenarios)
        missing_files = self._identify_missing_files(requirements, target_files)
        target_files.extend(missing_files)
        
        return target_files
    
    def _find_exact_methods(
        self,
        requirements: StructuredRequirements,
        target_files: List[TargetFile]
    ) -> List[TargetMethod]:
        """
        Find exact methods to modify within target files.
        
        Uses SQLite symbol table for precise method lookup.
        """
        target_methods = []
        
        for file in target_files:
            if file.task_type == "create":
                # New files don't have existing methods
                continue
            
            logger.info(f"   Analyzing methods in: {file.file_path}")
            
            # Query SQLite for methods in this file
            query = """
            SELECT 
                s.name,
                s.fully_qualified_name,
                s.kind,
                s.start_line,
                s.end_line,
                s.signature
            FROM symbols s
            WHERE s.path = ?
              AND s.kind IN ('method', 'function')
            ORDER BY s.start_line
            """
            
            try:
                results = self.sqlite_store._conn.execute(query, (file.file_path,)).fetchall()
                
                for row in results:
                    name, fqn, kind, start_line, end_line, signature = row
                    
                    # Determine if this method is relevant
                    is_relevant = self._is_method_relevant(
                        name, 
                        requirements
                    )
                    
                    if is_relevant:
                        # Extract the body if possible
                        body_content = None
                        try:
                            full_path = self.workspace_path / file.file_path
                            if full_path.exists():
                                lines = full_path.read_text(encoding="utf-8").splitlines()
                                if 1 <= start_line <= len(lines) and 1 <= end_line <= len(lines):
                                    body_content = "\n".join(lines[start_line-1:end_line])
                        except Exception as e:
                            logger.debug(f"Could not read body for {fqn}: {e}")

                        method = TargetMethod(
                            name=name,
                            fully_qualified_name=fqn,
                            file_path=file.file_path,
                            start_line=start_line,
                            end_line=end_line,
                            signature=signature,
                            body=body_content,
                            operation="modify",
                            confidence=0.85
                        )
                        
                        target_methods.append(method)
                        logger.info(f"   ✅ Target method: {name} (lines {start_line}-{end_line})")
                        
            except Exception as e:
                logger.error(f"   ❌ Method query failed: {e}")
                continue
        
        return target_methods
    
    def _analyze_dependencies(
        self,
        target_files: List[TargetFile]
    ) -> List[str]:
        """
        Analyze dependencies using Neo4j graph.
        
        Finds all components that depend on target files.
        """
        if not self.neo4j_store:
            logger.warning("Neo4j not available - skipping dependency analysis")
            return []
        
        dependencies = []
        
        for file in target_files:
            # Extract class name from file path
            class_name = Path(file.file_path).stem
            
            logger.info(f"   Analyzing dependencies for: {class_name}")
            
            # Query Neo4j for dependents
            query = """
            MATCH (caller {workspace: $workspace})-[:CALLS|DEPENDS_ON]->(target:Class {workspace: $workspace})
            WHERE target.name = $class_name
            RETURN DISTINCT caller.name AS dependent
            """
            
            try:
                results = self.neo4j_store.query(query, {"class_name": class_name, "workspace": str(self.workspace_path)})
                
                for record in results:
                    dependent = record["dependent"]
                    if dependent and dependent not in dependencies:
                        dependencies.append(dependent)
                        logger.info(f"   ✅ Dependent: {dependent}")
                        
            except Exception as e:
                logger.error(f"   ❌ Neo4j query failed: {e}")
                continue
        
        return dependencies
    
    def _analyze_execution_paths(
        self,
        target_files: List[TargetFile]
    ) -> List[ExecutionPath]:
        """
        Analyze execution paths using Neo4j graph traversal.
        
        Example: UI → Controller → Service → Repository → DB
        """
        if not self.neo4j_store:
            return []
        
        execution_paths = []
        
        for file in target_files:
            class_name = Path(file.file_path).stem
            
            logger.info(f"   Analyzing execution path for: {class_name}")
            
            # Query Neo4j for call paths
            query = """
            MATCH path = (start {workspace: $workspace})-[:CALLS*1..5]->(target:Class {workspace: $workspace})
            WHERE target.name = $class_name
            RETURN path
            LIMIT 10
            """
            
            try:
                results = self.neo4j_store.query(query, {"class_name": class_name, "workspace": str(self.workspace_path)})
                
                for record in results:
                    # Extract path nodes
                    path_nodes = []
                    # TODO: Parse Neo4j path object
                    
                    exec_path = ExecutionPath(
                        start_component=path_nodes[0] if path_nodes else "Unknown",
                        end_component=class_name,
                        intermediate_components=path_nodes[1:-1] if len(path_nodes) > 2 else [],
                        path_length=len(path_nodes)
                    )
                    
                    execution_paths.append(exec_path)
                    
                # Query Neo4j for UI-to-API paths including State Management
                ui_query = """
                MATCH path = (c:UIComponent {workspace: $workspace})-[:USES_STATE|USES|CALLS*1..4]->(api:ApiEndpoint {workspace: $workspace})
                WHERE c.name = $class_name OR api.url CONTAINS $class_name
                RETURN path
                LIMIT 10
                """
                ui_results = self.neo4j_store.query(ui_query, {"class_name": class_name, "workspace": str(self.workspace_path)})
                for record in ui_results:
                    path_nodes = ["UIComponent", "Intermediate", "ApiEndpoint"]
                    exec_path = ExecutionPath(
                        start_component="UIComponent",
                        end_component="ApiEndpoint",
                        intermediate_components=["UIStore/AngularService"],
                        path_length=3
                    )
                    execution_paths.append(exec_path)
                    
            except Exception as e:
                logger.error(f"   ❌ Execution path query failed: {e}")
                continue
        
        return execution_paths
    
    def _perform_impact_analysis(
        self,
        target_files: List[TargetFile],
        dependencies: List[str],
        execution_paths: List[ExecutionPath]
    ) -> ImpactAnalysis:
        """
        Perform comprehensive impact analysis.
        
        Identifies:
        - Affected APIs
        - Affected services
        - Affected controllers
        - Affected database tables
        - Migration requirements
        """
        impact = ImpactAnalysis()
        
        # Analyze file types to determine impact
        for file in target_files:
            file_name_lower = file.file_path.lower()
            
            if "controller" in file_name_lower:
                impact.affected_controllers.append(file.file_path)
                impact.api_changes_required = True
                
            elif "service" in file_name_lower:
                if ".ts" in file_name_lower:
                    impact.affected_ui_services.append(file.file_path)
                else:
                    impact.affected_services.append(file.file_path)
                
            elif "repository" in file_name_lower:
                impact.affected_repositories.append(file.file_path)
                impact.database_changes_required = True
                
            elif "dto" in file_name_lower or "model" in file_name_lower:
                impact.affected_dtos.append(file.file_path)
                
            elif "component" in file_name_lower and ".ts" in file_name_lower:
                impact.affected_ui_components.append(file.file_path)
                
            elif ("store" in file_name_lower or "state" in file_name_lower) and ".ts" in file_name_lower:
                impact.affected_ui_stores.append(file.file_path)
        
        # Check if API contracts affected
        if impact.affected_controllers:
            impact.affected_apis = [
                f"API endpoints in {controller}" 
                for controller in impact.affected_controllers
            ]
        
        # Determine migration needs
        if impact.database_changes_required:
            impact.requires_migration = True
            impact.migration_type = "schema_change"
        
        logger.info(
            f"   Impact summary:\n"
            f"   - APIs: {len(impact.affected_apis)}\n"
            f"   - Services: {len(impact.affected_services)}\n"
            f"   - Controllers: {len(impact.affected_controllers)}\n"
            f"   - Repositories: {len(impact.affected_repositories)}\n"
            f"   - Migration required: {impact.requires_migration}"
        )
        
        return impact
    
    def _identify_missing_files(
        self,
        requirements: StructuredRequirements,
        existing_files: List[TargetFile]
    ) -> List[TargetFile]:
        """
        Identify files that need to be CREATED (not in SQLite).
        
        Example: Test files that don't exist yet.
        """
        missing_files = []
        
        # Check for missing test files
        for file in existing_files:
            if file.task_type == "modify":
                # Generate test file path
                test_path = self._generate_test_path(file.file_path)
                
                # Check if test file exists in SQLite
                query = """
                SELECT COUNT(*) FROM files 
                WHERE path = ?
                """
                
                try:
                    result = self.sqlite_store._conn.execute(query, (test_path,)).fetchall()
                    count = result[0][0] if result else 0
                    
                    if count == 0:
                        # Test file doesn't exist - need to CREATE
                        test_file = TargetFile(
                            file_path=test_path,
                            absolute_path=str(self.workspace_path / test_path),
                            language=file.language,
                            task_type="create",
                            confidence=0.90,
                            reason=f"Test file for {file.file_path}"
                        )
                        
                        missing_files.append(test_file)
                        logger.info(f"   ✅ Missing test file: {test_path} (will CREATE)")
                        
                except Exception as e:
                    logger.error(f"   ❌ Test file check failed: {e}")
                    continue
        
        return missing_files
    
    def _generate_test_path(self, source_path: str) -> str:
        """Generate test file path from source file path"""
        # Example: src/services/SupplierService.java → src/tests/SupplierServiceTest.java
        path = Path(source_path)
        test_name = f"{path.stem}Test{path.suffix}"
        return str(Path("src/tests") / test_name)
    
    def _is_method_relevant(
        self,
        method_name: str,
        requirements: StructuredRequirements
    ) -> bool:
        """
        Determine if a method is relevant to requirements.
        
        Uses simple keyword matching (can be enhanced with LLM).
        """
        # Check if method name appears in requirements
        all_requirements = (
            requirements.functional_requirements +
            requirements.technical_requirements
        )
        
        for req in all_requirements:
            if method_name.lower() in req.lower():
                return True
        
        return False
    
    def _calculate_localization_confidence(
        self,
        target_files: List[TargetFile],
        target_methods: List[TargetMethod],
        dependencies: List[str],
        requirements: StructuredRequirements
    ) -> float:
        """
        Calculate confidence score for localization.
        
        Based on:
        - Number of exact matches found
        - Coverage of affected components
        - Quality of dependency analysis
        
        Returns: 0.0 to 1.0
        """
        score = 0.0
        
        # Factor 1: Found target files (40% weight)
        if target_files:
            found_count = len(target_files)
            expected_count = len(requirements.affected_components)
            score += 0.4 * min(found_count / max(expected_count, 1), 1.0)
        
        # Factor 2: Found target methods (30% weight)
        if target_methods:
            score += 0.3
        
        # Factor 3: Dependency analysis completed (20% weight)
        if dependencies:
            score += 0.2
        
        # Factor 4: All affected components covered (10% weight)
        if all(
            any(comp.lower() in f.file_path.lower() for f in target_files)
            for comp in requirements.affected_components
        ):
            score += 0.1
        
        return min(score, 1.0)
    
    def _fallback_localization(
        self,
        requirements: StructuredRequirements
    ) -> LocalizationResult:
        """
        Fallback when SQLite/Neo4j unavailable.
        
        Uses LLM to guess targets (low confidence).
        """
        logger.warning("⚠️ Using fallback localization (low confidence)")
        
        # Basic fallback based on component names
        target_files = []
        
        for component in requirements.affected_components:
            # Guess file path
            guessed_path = f"src/services/{component}.java"
            
            target_file = TargetFile(
                file_path=guessed_path,
                absolute_path=str(self.workspace_path / guessed_path),
                language="java",
                task_type="modify",  # Assume modify
                confidence=0.3,  # LOW confidence
                reason=f"Fallback guess for {component}"
            )
            
            target_files.append(target_file)
        
        return LocalizationResult(
            target_files=target_files,
            target_methods=[],
            dependencies=[],
            execution_paths=[],
            impact_analysis=ImpactAnalysis(),
            confidence=0.3,  # LOW confidence
            requires_human_review=True  # MUST review
        )
    
    # =========================================================================
    # REPOSITORY INTELLIGENCE — GRAPH, OWNERSHIP, UNIFIED RANKING
    # =========================================================================

    # Edge-kind weights for graph-neighbor scoring.
    # Larger weight = stronger evidence of co-ownership.
    _EDGE_WEIGHTS: dict[str, float] = {
        "template_of":  0.80,   # TS class → HTML template (very strong)
        "style_of":     0.60,   # TS class → SCSS/CSS style
        "calls":        0.70,   # method → method
        "has_type":     0.55,   # constructor injection (Service A uses Service B)
        "implements":   0.55,   # class → interface
        "extends":      0.50,   # class → superclass
        "annotated_by": 0.40,   # symbol → decorator / annotation
        "references":   0.30,   # type reference
        "imports":      0.25,   # import (broad, noisy)
        "contains":     0.10,   # structural containment (low signal)
    }

    # Unified-score signal weights (must sum to 1.0).
    # w_sym is language-agnostic: fires for any language whose annotations/stereotype
    # match ticket keywords.  Replaces the old separate w_ja + w_ts pair.
    # w_path scores direct path-token overlap — the strongest ownership signal
    # when a file's path NAMES the concept being changed.
    _W_FS      = 0.10   # filesystem keyword density
    _W_SQL     = 0.15   # SQLite symbol hits (domain-normalized)
    _W_SYM     = 0.15   # annotation/stereotype match (any language)
    _W_PATH    = 0.15   # path-token overlap with ticket keywords
    _W_GR      = 0.15   # graph proximity
    _W_VEC     = 0.15   # vector cosine (0 when not provided)
    _W_CONTENT = 0.15   # FTS5 content search hits
    _W_OWN_MULT = 1.0  # ownership multiplier coefficient (was 0.5)

    # Kept for back-compat; individual language weights are replaced by _W_SYM.
    _W_JA  = 0.0
    _W_TS  = 0.0

    # ── Phase 2: Relationship boost + cluster constants ──────────────────────
    _W_REL_BOOST      = 0.08   # additive boost per companion (max 0.08×0.90×1.0=0.072)
    _MIN_BASE_FOR_REL = 0.05   # companion needs own merit before s_rel applies

    # Structural edges → form ownership clusters + companion lists.
    # Behavioral edges (calls, imports, annotated_by, contains) → feed s_gr ONLY.
    _STRUCTURAL_EDGES: frozenset = frozenset({
        "template_of",   # component.ts → component.html  (Angular templateUrl)
        "style_of",      # component.ts → component.scss  (Angular styleUrls)
        "has_type",      # constructor DI: Class A injects Class B
        "implements",    # Class A implements Interface B
        "extends",       # Class A extends Class B
    })

    def _graph_neighbors(
        self,
        paths: list[str],
        depth: int = 1,
    ) -> dict[str, float]:
        """Return neighboring file paths with their aggregated proximity score.

        Uses the SQLite ``edges`` table which is populated by both the Java
        parser and (after re-indexing) the TypeScript parser.  No Neo4j
        dependency required.

        Args:
            paths:  List of repo-relative file paths whose neighborhoods to expand.
            depth:  Graph traversal depth (currently only depth=1 is used to
                    bound expansion; recursive calls are straightforward if needed).

        Returns:
            Dict mapping ``neighbor_path → proximity_score`` where score is
            the maximum edge weight across all edges connecting the neighbor
            to any seed path.  Seed paths themselves are excluded.
        """
        if not self.sqlite_store or not paths:
            return {}

        conn = self.sqlite_store._conn
        seed_set = set(paths)
        neighbors: dict[str, float] = {}

        placeholders = ",".join("?" * len(paths))

        # ── Out-edges: symbols in seed files point to symbols in other files ──
        try:
            out_rows = conn.execute(
                f"""
                SELECT e.kind, s_dst.path AS neighbor_path
                FROM symbols s_src
                JOIN edges e ON e.src_id = s_src.id
                JOIN symbols s_dst ON s_dst.id = e.dst_id
                WHERE s_src.path IN ({placeholders})
                  AND s_dst.path NOT IN ({placeholders})
                  AND e.dst_id IS NOT NULL
                """,
                paths + paths,
            ).fetchall()
        except Exception as exc:
            logger.debug(f"_graph_neighbors out-edges query failed: {exc}")
            out_rows = []

        for kind, neighbor_path in out_rows:
            w = self._EDGE_WEIGHTS.get(kind, 0.10)
            neighbors[neighbor_path] = max(neighbors.get(neighbor_path, 0.0), w)

        # ── In-edges: other files' symbols point into seed files ──────────────
        try:
            in_rows = conn.execute(
                f"""
                SELECT e.kind, s_src.path AS neighbor_path
                FROM symbols s_dst
                JOIN edges e ON e.dst_id = s_dst.id
                JOIN symbols s_src ON s_src.id = e.src_id
                WHERE s_dst.path IN ({placeholders})
                  AND s_src.path NOT IN ({placeholders})
                """,
                paths + paths,
            ).fetchall()
        except Exception as exc:
            logger.debug(f"_graph_neighbors in-edges query failed: {exc}")
            in_rows = []

        for kind, neighbor_path in in_rows:
            w = self._EDGE_WEIGHTS.get(kind, 0.10)
            neighbors[neighbor_path] = max(neighbors.get(neighbor_path, 0.0), w)

        # ── Unresolved dst_name edges (TEMPLATE_OF / STYLE_OF / IMPORTS) ─────
        # These edges have dst_id=NULL but dst_name = a repo-relative file path.
        try:
            dn_rows = conn.execute(
                f"""
                SELECT e.kind, e.dst_name
                FROM symbols s_src
                JOIN edges e ON e.src_id = s_src.id
                WHERE s_src.path IN ({placeholders})
                  AND e.dst_id IS NULL
                  AND e.dst_name IS NOT NULL
                  AND e.kind IN ('template_of', 'style_of', 'imports')
                """,
                paths,
            ).fetchall()
        except Exception as exc:
            logger.debug(f"_graph_neighbors dst_name query failed: {exc}")
            dn_rows = []

        for kind, dst_name in dn_rows:
            if dst_name and dst_name not in seed_set:
                w = self._EDGE_WEIGHTS.get(kind, 0.10)
                neighbors[dst_name] = max(neighbors.get(dst_name, 0.0), w)

        return neighbors

    def _path_ownership_score(self, file_path: str, keywords: list[str]) -> float:
        """Score how directly the file's path names the concepts in the ticket.

        Normalises by the number of MEANINGFUL path tokens (not by keyword
        count) so long ticket descriptions don't dilute the signal.

        Algorithm:
        1. Tokenise path on separators; discard generic tokens (< 4 chars or
           structural dir names like src/main/java).
        2. For each meaningful path token, check if it appears in the ticket
           keyword set OR if any ticket keyword appears as a substring of it
           (e.g. "header" ⊂ "jheader").
        3. Return  matched_path_tokens / total_meaningful_path_tokens.

        Example:
          path  = "xchange-ui/src/jato/header/jheader.component.ts"
          kws   = {"jato", "header", "version", "dashboard", ...}
          tokens= [xchange, jato, header, jheader, component]
          matched = jato(✓) + header(✓) + jheader(∋header ✓) = 3
          score = 3/5 = 0.60

        GenericControllerHandler.java scores 0 for the same keywords because
        none of its path tokens overlap with {jato, header, version, …}.
        """
        if not keywords:
            return 0.0

        path_lower = file_path.lower()
        # Tokenise path
        raw_tokens = re.split(r"[/\\\-_.]", path_lower)

        # Filter out generic structural tokens
        _GENERIC = {"src", "main", "java", "test", "com", "org", "net",
                    "impl", "api", "lib", "util", "utils", "config"}
        path_tokens = [t for t in raw_tokens if len(t) >= 4 and t not in _GENERIC]
        if not path_tokens:
            return 0.0

        kw_set = {k.lower() for k in keywords if len(k) >= 4}
        if not kw_set:
            return 0.0

        # Count path tokens that contain (or exactly equal) a ticket keyword
        matched = sum(
            1 for tok in path_tokens
            if tok in kw_set or any(kw in tok for kw in kw_set)
        )

        # Normalise by path token count — independent of ticket length
        return min(1.0, matched / len(path_tokens))

    def _compute_ownership(
        self,
        path: str,
        keywords: list[str],
        content: str = "",
    ) -> float:
        """Compute a generic ownership score for a candidate file.

        Ownership means "this file *defines* the behavior" rather than merely
        referencing it.  Four independent signals are combined:

        D — Definition density
            Fraction of the file's symbols whose name matches a ticket keyword.
            High for a constants / config file; low for a controller that only
            calls getVersion().

        L — Literal density
            Fraction of the file's *non-blank* lines that contain a string
            literal (quoted text) matching a ticket keyword.  Declaration files
            have the literal; reference files only have identifiers.

        S — Stereotype match
            1.0 if any symbol's spring_stereotype or annotations[] intersects
            the keyword set.  Separates @Configuration from @RestController.

        G — Graph centrality
            Ratio of in-edges to total edges.  Files referenced by many others
            (constants, interfaces) are more likely to be the source of truth.

        Returns a float in [0, 1].
        """
        D = 0.0
        L = 0.0
        S = 0.0
        G = 0.0

        kw_lower = [k.lower() for k in keywords]

        # ── D: definition density from SQLite ─────────────────────────────────
        if self.sqlite_store:
            try:
                conn = self.sqlite_store._conn
                total_syms = conn.execute(
                    "SELECT COUNT(*) FROM symbols WHERE path = ?", (path,)
                ).fetchone()[0]
                if total_syms > 0:
                    matched = conn.execute(
                        "SELECT COUNT(*) FROM symbols WHERE path = ? AND ("
                        + " OR ".join("LOWER(name) LIKE ?" for _ in kw_lower)
                        + ")",
                        [path] + [f"%{k}%" for k in kw_lower],
                    ).fetchone()[0]
                    D = matched / total_syms

                # ── S: stereotype / annotation match ──────────────────────────
                strow = conn.execute(
                    "SELECT spring_stereotype, annotations FROM symbols WHERE path = ?",
                    (path,),
                ).fetchall()
                for stereotype, ann_json in strow:
                    st_text = (stereotype or "").lower()
                    if any(k in st_text for k in kw_lower):
                        S = 1.0
                        break
                    if ann_json:
                        try:
                            anns = json.loads(ann_json) if isinstance(ann_json, str) else ann_json
                            if any(k in a.lower() for k in kw_lower for a in anns if isinstance(a, str)):
                                S = 1.0
                                break
                        except Exception:
                            pass

                # ── G: graph centrality (in-edges vs all edges) ───────────────
                conn2 = self.sqlite_store._conn
                in_deg = conn2.execute(
                    """
                    SELECT COUNT(*) FROM edges e
                    JOIN symbols s ON s.id = e.dst_id
                    WHERE s.path = ?
                    """,
                    (path,),
                ).fetchone()[0]
                out_deg = conn2.execute(
                    """
                    SELECT COUNT(*) FROM edges e
                    JOIN symbols s ON s.id = e.src_id
                    WHERE s.path = ?
                    """,
                    (path,),
                ).fetchone()[0]
                total_deg = in_deg + out_deg
                if total_deg > 0:
                    G = in_deg / total_deg

            except Exception as exc:
                logger.debug(f"_compute_ownership SQLite query failed for {path}: {exc}")

        # ── L: literal density from file content ──────────────────────────────
        if content:
            lines = [l for l in content.splitlines() if l.strip()]
            if lines:
                literal_re = re.compile(r'["\']([^"\']{2,})["\']')
                literal_lines = sum(
                    1 for line in lines
                    if any(k in m.group(1).lower() for k in kw_lower
                           for m in literal_re.finditer(line))
                )
                L = literal_lines / len(lines)

        return min(1.0, 0.35 * D + 0.30 * L + 0.20 * S + 0.15 * G)

    # =========================================================================
    # PHASE 2 — OWNERSHIP CLUSTERS, INVESTIGATION, VERIFICATION
    # =========================================================================

    def _graph_expand_candidates(self, candidates: list[dict]) -> list[dict]:
        """Pull structural companion files into the candidate pool.

        Uses ONLY structural edges (template_of, style_of, has_type, implements,
        extends). Adds files reachable from the top-20 seeds that are NOT yet
        in the pool — e.g. a .scss file that scored zero on the path walk
        because its content contained no ticket keywords.

        Returns the augmented candidates list (original + new entries).
        """
        if not self.sqlite_store:
            return candidates

        existing_paths = {c["path"] for c in candidates}
        conn = self.sqlite_store._conn
        seed_paths = [c["path"] for c in candidates[:20]]
        if not seed_paths:
            return candidates

        ph = ",".join("?" * len(seed_paths))
        struct_kinds_sql = ",".join(f"'{k}'" for k in sorted(self._STRUCTURAL_EDGES))

        new_entries: dict[str, dict] = {}
        try:
            rows = conn.execute(
                f"""
                SELECT e.kind, e.dst_name, s_src.path AS seed_path
                FROM symbols s_src
                JOIN edges e ON e.src_id = s_src.id
                WHERE s_src.path IN ({ph})
                  AND e.kind IN ({struct_kinds_sql})
                  AND e.dst_id IS NULL
                  AND e.dst_name IS NOT NULL
                """,
                seed_paths,
            ).fetchall()
        except Exception as exc:
            logger.debug(f"_graph_expand_candidates query failed: {exc}")
            rows = []

        for kind, dst_name, seed_path in rows:
            if not dst_name or dst_name in existing_paths or dst_name in new_entries:
                continue
            full = self.workspace_path / dst_name
            if not full.exists():
                continue
            try:
                content = full.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                content = ""
            # Pre-seed s_gr so the file is not zero-scored before Phase 2 boost
            new_entries[dst_name] = {
                "path":               dst_name,
                "confidence":         0.0,
                "signals":            ["graph_expanded", f"via:{kind}"],
                "raw_score":          0.0,
                "_content":           content,
                "_graph_expand_kind": kind,
                "_graph_expand_seed": seed_path,
                "_preseed_s_gr":      self._EDGE_WEIGHTS.get(kind, 0.10),
            }

        if new_entries:
            logger.debug(
                f"  Graph-expanded {len(new_entries)} structural companion(s): "
                + ", ".join(Path(p).name for p in new_entries)
            )
            return candidates + list(new_entries.values())
        return candidates

    def _build_ownership_clusters(
        self,
        enriched: list[dict],
        s_own_map: dict[str, float],
    ) -> list[dict]:
        """Group scored candidates into ownership clusters via structural edges.

        Seeding: candidates with s_own >= 0.10, processed in score-descending order.
        Each seed queries structural edges to find:
          companions  — template_of, style_of, implements, extends  (cap: 5)
          supporting  — has_type (constructor DI)                   (cap: 5)

        A file can belong to only one cluster (highest-ownership seed wins).
        Behavioral edges (calls, imports, etc.) are NEVER used here.

        Returns list of cluster dicts.
        """
        if not self.sqlite_store:
            return []

        conn = self.sqlite_store._conn
        assigned: set[str] = set()
        clusters: list[dict] = []
        score_map = {c["path"]: c.get("unified_score", 0.0) for c in enriched}

        seeds = sorted(
            [c for c in enriched if s_own_map.get(c["path"], 0.0) >= 0.10],
            key=lambda c: -s_own_map.get(c["path"], 0.0),
        )

        for seed_c in seeds:
            seed_path = seed_c["path"]
            if seed_path in assigned:
                continue

            companions: list[str] = []
            supporting: list[str] = []
            evidence:   list[str] = []
            seen_dst:   set[str]  = set()

            # Query 1: dst_id=NULL edges — catches template_of and style_of
            try:
                rows1 = conn.execute(
                    """
                    SELECT e.kind, e.dst_name
                    FROM symbols s JOIN edges e ON e.src_id = s.id
                    WHERE s.path = ?
                      AND e.kind IN ('template_of','style_of','has_type',
                                     'implements','extends')
                      AND e.dst_id IS NULL
                      AND e.dst_name IS NOT NULL
                    """,
                    (seed_path,),
                ).fetchall()
            except Exception:
                rows1 = []

            # Query 2: resolved dst_id edges — catches has_type, implements, extends
            try:
                rows2 = conn.execute(
                    """
                    SELECT e.kind, s_dst.path
                    FROM symbols s_src
                    JOIN edges e ON e.src_id = s_src.id
                    JOIN symbols s_dst ON s_dst.id = e.dst_id
                    WHERE s_src.path = ?
                      AND e.kind IN ('template_of','style_of','has_type',
                                     'implements','extends')
                      AND e.dst_id IS NOT NULL
                    """,
                    (seed_path,),
                ).fetchall()
            except Exception:
                rows2 = []

            for kind, dst_path in list(rows1) + list(rows2):
                if not dst_path or dst_path in seen_dst or dst_path == seed_path:
                    continue
                seen_dst.add(dst_path)
                if dst_path in assigned:
                    continue

                # Declared structural relationships are always valid companions;
                # others require minimum own merit.
                is_declared = kind in {"template_of", "style_of"}
                base_score = score_map.get(dst_path, 0.0)
                if not is_declared and base_score < self._MIN_BASE_FOR_REL:
                    continue

                if kind in {"template_of", "style_of", "implements", "extends"}:
                    companions.append(dst_path)
                    evidence.append(f"{kind}:{Path(dst_path).name}")
                elif kind == "has_type":
                    supporting.append(dst_path)
                    evidence.append(f"has_type:{Path(dst_path).name}")

            # Sort by score, cap at 5 each
            companions = sorted(
                companions, key=lambda p: score_map.get(p, 0.0), reverse=True)[:5]
            supporting = sorted(
                supporting, key=lambda p: score_map.get(p, 0.0), reverse=True)[:5]

            assigned.add(seed_path)
            for p in companions + supporting:
                assigned.add(p)

            n_members = len(companions) + len(supporting)
            cluster_score = s_own_map.get(seed_path, 0.0) * (
                1.0 + 0.10 * min(5, n_members))

            clusters.append({
                "primary":       seed_path,
                "companions":    companions,
                "supporting":    supporting,
                "cluster_score": cluster_score,
                "evidence":      evidence,
                "ownership_type":  None,   # filled after _verify_ownership
                "investigation":   None,   # filled after _investigate_candidate
            })

        return clusters

    def _relationship_score(self, edge_kind: str) -> float:
        """Relationship weight for a structural edge kind.

        Determines s_rel for a companion file.
        Higher = stronger structural declaration of co-ownership.
        """
        return {
            "template_of":  0.90,   # explicit Angular templateUrl
            "style_of":     0.80,   # explicit Angular styleUrls
            "implements":   0.55,   # interface contract
            "has_type":     0.55,   # constructor injection
            "extends":      0.45,   # inheritance
        }.get(edge_kind, 0.0)

    def _apply_relationship_boost(
        self,
        enriched: list[dict],
        clusters: list[dict],
    ) -> list[dict]:
        """Write s_rel and cluster-membership metadata into each candidate.

        Only structural companions receive s_rel > 0.  The boost is applied
        inside _unified_score only when base >= _MIN_BASE_FOR_REL.
        """
        # Build lookup maps
        path_to_info: dict[str, tuple] = {}     # path → (cluster, role, edge_kind)
        primary_scores: dict[str, float] = {}
        score_map = {e["path"]: e.get("unified_score", 0.0) for e in enriched}

        for cl in clusters:
            prim = cl["primary"]
            primary_scores[prim] = score_map.get(prim, 0.5)
            path_to_info[prim] = (cl, "primary", "")

            # Resolve companion → edge_kind from the evidence list
            evidence_map: dict[str, str] = {}
            for ev in cl.get("evidence", []):
                if ":" in ev:
                    kind, fname = ev.split(":", 1)
                    for p in cl["companions"] + cl["supporting"]:
                        if Path(p).name == fname:
                            evidence_map[p] = kind

            for comp in cl["companions"]:
                path_to_info[comp] = (cl, "companion",
                                      evidence_map.get(comp, "template_of"))
            for sup in cl["supporting"]:
                path_to_info[sup] = (cl, "supporting", "has_type")

        result = []
        for entry in enriched:
            path = entry["path"]
            feat = dict(entry.get("features", {}))

            if path in path_to_info:
                cl, role, edge_kind = path_to_info[path]
                prim_path  = cl["primary"]
                prim_score = primary_scores.get(prim_path, 0.5)

                s_rel = (0.0 if role == "primary"
                         else self._relationship_score(edge_kind))
                feat["s_rel"]          = s_rel
                feat["cluster_id"]     = prim_path
                feat["cluster_role"]   = role
                feat["cluster_score"]  = cl["cluster_score"]
                feat["_primary_score"] = prim_score
                feat["rel_evidence"]   = cl.get("evidence", [])

                sigs = list(entry.get("signals", []))
                if role == "primary" and cl.get("companions"):
                    sigs.append("cluster:primary")
                elif role == "companion":
                    sigs.append(f"cluster:companion({edge_kind})")
                elif role == "supporting":
                    sigs.append("cluster:supporting")
                entry = {**entry, "features": feat,
                         "signals": list(dict.fromkeys(sigs))}
            else:
                feat["s_rel"]          = 0.0
                feat["cluster_id"]     = None
                feat["cluster_role"]   = "isolated"
                feat["cluster_score"]  = 0.0
                feat["_primary_score"] = 0.0
                feat["rel_evidence"]   = []
                entry = {**entry, "features": feat}

            result.append(entry)

        return result

    def _consensus_score(self, top_candidates: list[dict]) -> dict[str, float]:
        """Pairwise signal-axis consensus for top candidates.

        For each candidate, counts how many signal axes it leads on relative to
        the others.  A file that leads consistently across multiple independent
        axes is a more credible owner than one that spikes on a single axis.

        Returns dict mapping path → consensus_score [0, 1].
        """
        axes = ["s_own", "s_path", "s_sym", "s_sql", "s_gr"]
        scores: dict[str, float] = {}
        if not top_candidates:
            return scores

        all_vals = {
            ax: [c.get("features", {}).get(ax, 0.0) for c in top_candidates]
            for ax in axes
        }

        for i, c in enumerate(top_candidates):
            path = c["path"]
            axes_won = 0
            for ax in axes:
                my_val = c.get("features", {}).get(ax, 0.0)
                others = [v for j, v in enumerate(all_vals[ax]) if j != i]
                if others and my_val >= max(others) * 0.85:  # within 15% of best
                    axes_won += 1

            cluster_bonus = (
                0.10 if c.get("features", {}).get("cluster_role") == "primary"
                else 0.0
            )
            scores[path] = min(1.0, (axes_won / max(1, len(axes))) + cluster_bonus)

        return scores

    def _workflow_precision_gate(self, candidate: dict) -> tuple[bool, str]:
        """Decide whether workflow evidence is strong enough to boost a candidate.

        Workflow paths may add evidence, but they should not dominate candidates
        that lack direct supporting signals. When the feature flag is off, the
        legacy workflow boost behavior is preserved.
        """
        if not self.enable_workflow_precision_gate:
            return True, "disabled"

        signals = [str(sig) for sig in candidate.get("signals", [])]
        if not signals:
            return False, "no_signals"

        direct_markers = (
            "content_match",
            "semantic_catalog_match",
            "localization_asset",
            "rkb_feature_match",
            "ticket_history_match",
            "sqlite_symbol",
            "sym_match",
        )
        direct_support = [
            sig for sig in signals
            if any(marker in sig for marker in direct_markers)
        ]
        if len(direct_support) < self.workflow_gate_min_direct_sources:
            return False, "insufficient_direct_evidence"

        if float(candidate.get("raw_score", 0.0)) < 0.05:
            return False, "weak_base_score"

        return True, "ok"

    def _investigate_candidate(
        self,
        path: str,
        content: str,
        keywords: list[str],
    ) -> dict:
        """LLM-free, pattern-based code analysis.

        Determines HOW the file relates to the ticket concept.  Four independent
        ownership signals — all derived from regex matching on actual file content:

          defines    — symbol DEFINITIONS (class/function/const) matching keywords
          displays   — template bindings, interpolations, view-layer output
          calculates — assignments and computations of keyword-related values
          persists   — writes to storage (repository, store, localStorage, etc.)

        Returns dict with integer counts for each signal.
        """
        kw_lower = {k.lower() for k in keywords if len(k) >= 4}
        ext = Path(path).suffix.lower()

        # ── DEFINES: symbol definitions whose name contains a keyword ────────
        define_re = re.compile(
            r"(?:^|\s)(?:class|function|def|const|let|var|interface|enum"
            r"|type|public|private|protected)\s+(\w+)",
            re.I | re.MULTILINE,
        )
        defines = sum(
            1 for m in define_re.finditer(content)
            if any(kw in m.group(1).lower() for kw in kw_lower)
        )

        # ── DISPLAYS: template interpolations / DOM bindings / view output ───
        displays = 0
        for kw in kw_lower:
            # Angular {{ keyword }}
            if re.search(
                r"\{\{[^}]*" + re.escape(kw) + r"[^}]*\}\}", content, re.I):
                displays += 1
            # Angular [attr]="expr"
            if re.search(
                r'\[[\w\-]+\]\s*=\s*["\'][^"\']*' + re.escape(kw), content, re.I):
                displays += 1
            # HTML text content
            if re.search(
                r">\s*[^<]*" + re.escape(kw) + r"[^<]*\s*<", content, re.I):
                displays += 1
        # Template/style files always carry at least 1 display signal
        if ext in {".html", ".scss", ".css", ".sass"} and content.strip():
            displays = max(displays, 1)

        # ── CALCULATES: assignment / computation of keyword-related values ───
        calculates = 0
        for kw in kw_lower:
            if re.search(
                r"(?:this\.)?\b\w*" + re.escape(kw) + r"\w*\s*[+\-]?=(?!=)",
                content, re.I,
            ):
                calculates += 1
            if re.search(r"\breturn\b[^;{]*" + re.escape(kw), content, re.I):
                calculates += 1

        # ── PERSISTS: write-to-storage operations ────────────────────────────
        persist_patterns = [
            r"\.save\s*\(",
            r"\.persist\s*\(",
            r"repository\.\w+\s*\(",
            r"store\.dispatch\s*\(",
            r"localStorage\.setItem\s*\(",
            r"sessionStorage\.setItem\s*\(",
            r"this\.\w+\$\.next\s*\(",        # RxJS BehaviorSubject.next()
            r"patchState\s*\(",               # NgRx component store
            r"\.update\s*\([^)]*=>\s*\(",     # NgRx signal store update
        ]
        persists = sum(
            1 for pat in persist_patterns if re.search(pat, content, re.I))

        return {
            "defines":    defines,
            "displays":   displays,
            "calculates": calculates,
            "persists":   persists,
        }

    def _verify_ownership(
        self,
        path: str,
        investigation: dict,
        features: dict,
    ) -> tuple[str, str]:
        """Discovery-time ownership role (advisory). Delegates to the pure
        ``verify_ownership``: structure/similarity/feature-anchor never grant
        PRIMARY — primary ownership is proven by inspection (Phase B)."""
        return verify_ownership(path, investigation, features)

    def _format_cluster_trace(self, cluster: dict) -> dict:
        """Serialize a cluster to a trace/ranking-friendly dict."""
        return {
            "primary":       cluster["primary"],
            "companions":    cluster["companions"],
            "supporting":    cluster["supporting"],
            "cluster_score": round(cluster["cluster_score"], 3),
            "evidence":      cluster["evidence"],
            "ownership_type":  cluster.get("ownership_type"),
            "investigation":   cluster.get("investigation"),
        }

    def _unified_score(self, features: dict) -> float:
        """Compute the final unified candidate score from pre-extracted features.

        Phase 1 formula — language-agnostic, repository-aware:

        base = W_FS  × s_fs        (filesystem keyword density)
             + W_SQL × s_sql       (SQLite symbol hits, domain-normalised)
             + W_SYM × s_sym       (annotation/stereotype match, any language)
             + W_PATH× s_path      (path-token overlap with ticket keywords)
             + W_GR  × s_gr        (graph proximity)
             + W_VEC × s_vec       (vector cosine, 0 when unavailable)

        score = base × (1 + W_OWN_MULT × s_own) × ownership_gate

        ``ownership_gate`` is 0.15 when ALL of these are true:
          - s_path == 0 (no path-keyword overlap)
          - s_sym  == 0 (no annotation/stereotype match)
          - s_own  < 0.05 (near-zero ownership)
        Files with no structural connection to the ticket are soft-gated
        before the graph + filesystem signals can inflate their score.

        Args:
            features: dict with keys:
                s_fs   — normalized filesystem keyword density  [0, 1]
                s_sql  — domain-normalized SQLite symbol hits   [0, 1]
                s_sym  — annotation/stereotype match (any lang) {0, 1}
                s_path — path-token overlap with keywords       [0, 1]
                s_gr   — graph proximity score                  [0, 1]
                s_vec  — vector cosine similarity               [0, 1]
                s_own  — ownership score                        [0, 1]

        Returns float in [0, ~2.0] (ownership can boost up to 2×).
        """
        w_sym = getattr(self, "current_weights", {}).get("sym", self._W_SYM)
        w_path = getattr(self, "current_weights", {}).get("path", self._W_PATH)
        w_content = getattr(self, "current_weights", {}).get("content", self._W_CONTENT)

        # Per-candidate available weight normalization:
        # A candidate is only evaluated against optional dynamic subsystems (Graph, Vector)
        # if that subsystem actually produced a signal (s_gr > 0, s_vec > 0) for this candidate.
        # This prevents unpopulated or noisy external graph/vector systems from globally diluting
        # high-precision filesystem and SQLite symbol hits.
        graph_ok = features.get("s_gr", 0.0) > 0.0
        vec_ok = features.get("s_vec", 0.0) > 0.0
        sql_ok = getattr(self, "sqlite_store", None) is not None

        # Calculate live available weight denominator for this candidate
        available = self._W_FS + w_path
        if sql_ok:
            available += self._W_SQL + w_sym + w_content
        if graph_ok:
            available += self._W_GR
        if vec_ok:
            available += self._W_VEC

        weighted_sum = (
            self._W_FS       * features.get("s_fs",   0.0)
            + (self._W_SQL    * features.get("s_sql",  0.0) if sql_ok else 0.0)
            + (w_sym          * features.get("s_sym",  0.0) if sql_ok else 0.0)
            + w_path         * features.get("s_path", 0.0)
            + (self._W_GR     * features.get("s_gr",   0.0) if graph_ok else 0.0)
            + (self._W_VEC    * features.get("s_vec",  0.0) if vec_ok else 0.0)
            + (w_content      * features.get("s_content", 0.0) if sql_ok else 0.0)
        )

        base = weighted_sum / max(available, 0.20)
        base += (
            0.50 * features.get("s_loc",      0.0)  # Explicit boost for translation assets
            + 0.50 * features.get("s_semantic", 0.0)  # Semantic Catalog Evidence
            + 0.40 * features.get("s_workflow", 0.0)  # Workflow Memory Evidence
            + 0.45 * features.get("s_history",  0.0)  # Historical Ticket Memory
            + 0.30 * features.get("s_rkb",      0.0)  # RKB Feature Evidence
            + 0.60 * features.get("s_dataflow", 0.0)  # Bounded Backward DataFlow Reachability
        )
        s_own      = features.get("s_own",      0.0)
        s_path     = features.get("s_path",     0.0)
        s_sym      = features.get("s_sym",      0.0)
        s_content  = features.get("s_content",  0.0)
        s_loc      = features.get("s_loc",      0.0)
        s_semantic = features.get("s_semantic", 0.0)
        s_workflow = features.get("s_workflow", 0.0)
        s_history  = features.get("s_history",  0.0)
        s_rkb      = features.get("s_rkb",      0.0)
        s_dataflow = features.get("s_dataflow", 0.0)

        # Ownership gate: soft-penalise files that have no path/annotation/content
        # connection to the ticket AND negligible ownership evidence.
        gate = 0.15 if (s_path == 0.0 and s_sym == 0.0 and s_content == 0.0 and s_loc == 0.0 and s_semantic == 0.0 and s_workflow == 0.0 and s_history == 0.0 and s_rkb == 0.0 and s_dataflow == 0.0 and s_own < 0.05) else 1.0

        score = base * (1.0 + self._W_OWN_MULT * s_own) * gate

        # Phase 2: additive relationship boost for structural companions.
        # Guard: companion must have non-trivial own merit (base >= threshold)
        # before s_rel applies — prevents zero-base files being lifted by a
        # relationship edge alone.
        s_rel = features.get("s_rel", 0.0)
        if s_rel > 0.0 and base >= self._MIN_BASE_FOR_REL:
            primary_score = features.get("_primary_score", 1.0)
            score += self._W_REL_BOOST * s_rel * min(1.0, primary_score)

        return score

    def analyze_candidates(
        self,
        raw_candidates: list[dict],
        ticket_text: str,
        *,
        vector_scores: dict[str, float] | None = None,
        focused_keywords: list[str] | None = None,
    ) -> list[dict]:
        """Compute rich per-candidate features and re-rank by unified score.

        Called at the end of ``discover_repository_candidates`` so all
        downstream consumers receive an already-scored, ownership-aware list.

        Args:
            raw_candidates:   Output of the filesystem walk phase — list of dicts
                ``{path, confidence, signals, raw_score, _content}``.
            ticket_text:      Full ticket text used for keyword extraction when
                ``focused_keywords`` is not provided.
            vector_scores:    Optional dict mapping path → cosine similarity [0,1]
                from the RAG engine.  If None, ``s_vec=0`` for all.
            focused_keywords: Optional pre-filtered keyword list.  When provided,
                these are used in place of extracting from ``ticket_text``.  This
                allows the caller (``discover_repository_candidates``) to pass
                concept-specific keywords that exclude noise from ticket
                descriptions (steps-to-reproduce, URLs, credentials, etc.).

        Returns:
            Same list enriched with ``features`` dict and re-sorted by
            ``unified_score`` descending.  ``confidence`` is replaced with the
            unified score (clamped to [0, 0.95]).
        """
        if not raw_candidates:
            return []

        # Use caller-supplied focused keywords when available; otherwise extract
        # from the full ticket text.  Focused keywords exclude noise terms that
        # appear in ticket descriptions but not in the actual change concept
        # (e.g. org names embedded in Java package paths, UI navigation words).
        keywords = (
            focused_keywords
            if focused_keywords is not None
            else self._extract_keywords(ticket_text)
        )
        max_fs = max((c.get("raw_score", 0.0) for c in raw_candidates), default=1.0) or 1.0

        # ── SQL hit counts per file (one query for all candidates) ────────────
        sql_hit_map: dict[str, int] = {}
        if self.sqlite_store and keywords:
            conn = self.sqlite_store._conn
            all_paths = [c["path"] for c in raw_candidates]
            ph = ",".join("?" * len(all_paths))
            for kw in keywords[:6]:
                try:
                    rows = conn.execute(
                        f"SELECT path, COUNT(*) FROM symbols WHERE path IN ({ph})"
                        " AND (LOWER(name) LIKE ? OR LOWER(qualified_name) LIKE ?)"
                        " GROUP BY path",
                        all_paths + [f"%{kw.lower()}%", f"%{kw.lower()}%"],
                    ).fetchall()
                    for p, cnt in rows:
                        sql_hit_map[p] = sql_hit_map.get(p, 0) + cnt
                except Exception:
                    pass
        max_sql = max(sql_hit_map.values(), default=1) or 1

        # ── Domain-normalised SQL: compute max hits per language so Java's
        #    larger index does not suppress TypeScript scores.  Repository-aware:
        #    we look at what languages the candidates ACTUALLY belong to.
        lang_max_sql: dict[str, int] = {}

        # ── Graph neighbors of the top-15 candidates by fs score ─────────────
        top_paths = [c["path"] for c in raw_candidates[:15]]
        neighbor_map = self._graph_neighbors(top_paths)

        # ── Stereotype / decorator per file (one query) ───────────────────────
        stereotype_map: dict[str, tuple[str, list]] = {}
        if self.sqlite_store:
            try:
                conn = self.sqlite_store._conn
                all_paths = [c["path"] for c in raw_candidates]
                ph = ",".join("?" * len(all_paths))
                rows = conn.execute(
                    f"SELECT path, spring_stereotype, annotations, language "
                    f"FROM symbols JOIN files USING(path) WHERE path IN ({ph})",
                    all_paths,
                ).fetchall()
                for p, st, ann, lang in rows:
                    stereotype_map.setdefault(p, (lang or "", []))
                    existing_lang, existing_ann = stereotype_map[p]
                    new_ann = existing_ann[:]
                    if st:
                        new_ann.append(st)
                    if ann:
                        try:
                            new_ann.extend(json.loads(ann) if isinstance(ann, str) else ann)
                        except Exception:
                            pass
                    stereotype_map[p] = (lang or existing_lang, new_ann)
            except Exception as exc:
                logger.debug(f"analyze_candidates stereotype query failed: {exc}")

        # Build per-language max SQL hit count for domain-normalised scoring.
        for path, cnt in sql_hit_map.items():
            lang = stereotype_map.get(path, ("", []))[0] or "unknown"
            if cnt > lang_max_sql.get(lang, 0):
                lang_max_sql[lang] = cnt

        kw_lower = [k.lower() for k in keywords]
        kw_set = set(kw_lower)
        test_intent = any(k in kw_set for k in {
            "test", "tests", "spec", "e2e", "integration", "regression"
        })

        # ── Layer Intent Analysis ─────────────────────────────────────────────
        # Derive layer intent from ticket text markers without repo-specific assumptions.
        # HTTP status codes (400, 401, 404, 500) and backend keywords indicate server/API defects.
        has_http_status = bool(re.search(r"\b(?:400|401|403|404|409|500|502|503)\b", ticket_text))
        has_backend_kw = bool(re.search(r"\b(?:server[- ]?side|backend|api\b|apis\b|endpoint|database|sql\b|payload|template|bulk upload template|downloaded? document|spreadsheet|export)\b", ticket_text, re.I))
        backend_intent = has_http_status or has_backend_kw

        # Visual layout, DOM, CSS, component terms indicate frontend UI defects.
        has_frontend_kw = bool(re.search(r"\b(?:ui\b|frontend\b|web\b|dashboard\b|console\b|view\b|views\b|page\b|tab\b|dialog\b|modal\b|button\b|action\b|checkbox\b|dropdown\b|rendering\b|render\b|html\b|styles?\b|grid\b|css\b|flexbox\b|scrollbar\b|overflow\b|react\b|angular\b|vue\b|component\b)\b", ticket_text, re.I))
        frontend_intent = has_frontend_kw

        # ── Style-presentation intent ────────────────────────────────────────
        # A ticket about colour/spacing/overflow describes a VISUAL defect, so the
        # change belongs in a stylesheet — not the sibling component class. Without
        # this, "option label color" ranks contract-option.component.ts above
        # contract-option.component.scss: the right component, the wrong file.
        # Purely lexical, no repo-specific assumptions.
        has_style_kw = bool(re.search(
            r"\b(?:colou?rs?\b|backgrounds?\b|borders?\b|paddings?\b|margins?\b|spacings?\b"
            r"|alignments?\b|fonts?\b|typography\b|opacity\b|hover\b|icons? size\b|shadows?\b"
            r"|rounded?\b|highlight(?:ed|ing)?\b|styling\b|style fix\b|css\b|scss\b|sass\b"
            r"|overflow\b|scrollbars?\b|flexbox\b|layout\b|responsive\b|theme\b|truncat\w*\b)\b",
            ticket_text, re.I,
        ))
        styling_intent = has_style_kw

        def _is_test_path(p: str) -> bool:
            pl = p.lower().replace("\\", "/")
            return (
                ".spec." in pl
                or ".test." in pl
                or pl.endswith(("test.java", "tests.java", "spec.java", "_test.go", "_test.py"))
                or "/test/" in pl
                or "/tests/" in pl
                or "/spec/" in pl
                or "__tests__" in pl
            )

        def _is_generated_path(p: str) -> bool:
            pl = p.lower().replace("\\", "/")
            return any(seg in pl for seg in (
                "/dist/", "/node_modules/", "/target/", "/build/", "/out/", "/generated/"
            ))

        enriched = []
        for c in raw_candidates:
            path = c["path"]
            raw_score = float(c.get("raw_score", 0.0))
            content = c.get("_content", "")

            # Normalized filesystem score
            s_fs = raw_score / max_fs

            # Domain-normalised SQLite symbol-hit score.
            # Divide by the max hits for files of the SAME language so a Java file
            # with 200 symbol hits doesn't make a TypeScript file with 10 hits
            # look insignificant.
            raw_sql = sql_hit_map.get(path, 0)
            file_lang = stereotype_map.get(path, ("", []))[0] or "unknown"
            lang_max = lang_max_sql.get(file_lang, 1) or 1
            s_sql = min(1.0, raw_sql / lang_max)

            # Unified annotation/stereotype match — language-agnostic.
            # Fires for any language whose stored annotations or spring_stereotype
            # contains a ticket keyword.  For TypeScript, this now includes
            # enriched annotations like "selector:cc-jato-header", "seltok:header".
            lang, anns = stereotype_map.get(path, ("", []))
            ann_texts = " ".join(str(a).lower() for a in anns)
            s_sym = 1.0 if any(k in ann_texts for k in kw_lower) else 0.0

            # Path-token overlap (Phase 1 primary ownership signal)
            s_path = self._path_ownership_score(path, keywords)

            # Graph proximity: this file is a neighbor of a top candidate
            s_gr = neighbor_map.get(path, 0.0)

            # Vector cosine (external input, optional)
            s_vec = (vector_scores or {}).get(path, 0.0)

            # Ownership
            s_own = self._compute_ownership(path, keywords, content)
            
            # Content Match
            signals = list(c.get("signals", []))
            s_content = 1.0 if "content_match" in signals else 0.0
            s_loc = 1.0 if "localization_asset" in signals else 0.0
            s_semantic = 1.0 if "semantic_catalog_match" in signals else 0.0
            s_workflow = 1.0 if "workflow_path" in signals else 0.0
            s_history = 1.0 if "ticket_history_match" in signals else 0.0
            s_rkb = 1.0 if "rkb_feature_match" in signals else 0.0
            s_dataflow = float(c.get("features", {}).get("s_dataflow", 0.0))
            if s_dataflow == 0.0 and any("dataflow_reachability" in s for s in signals):
                s_dataflow = 1.0

            features = {
                "s_fs": s_fs, "s_sql": s_sql,
                "s_sym": s_sym, "s_path": s_path,
                "s_gr": s_gr, "s_vec": s_vec,
                "s_own": s_own, "s_content": s_content,
                "s_loc": s_loc, "s_semantic": s_semantic,
                "s_workflow": s_workflow, "s_history": s_history, "s_rkb": s_rkb,
                "s_dataflow": s_dataflow,
                # Keep legacy keys so callers that set s_ja / s_ts explicitly still work
                "s_ja": 0.0, "s_ts": 0.0,
            }
            unified = self._unified_score(features)

            # Generic reranking adjustments (non ticket-specific):
            # suppress test/generated noise unless ticket intent requests tests.
            path_is_test = _is_test_path(path)
            path_is_generated = _is_generated_path(path)

            penalty = 0.0
            if path_is_generated:
                penalty += 0.25
            if path_is_test and not test_intent:
                penalty += 0.20

            # Layer-intent contradiction penalties:
            p_lower = path.lower().replace("\\", "/")
            if backend_intent and not frontend_intent:
                # Strong backend intent: penalize pure frontend UI presentation components & client services
                is_frontend = (
                    p_lower.endswith((".html", ".scss", ".css", ".sass", ".less"))
                    or "component." in p_lower
                    or "/src/app/" in p_lower
                    or "-ui/" in p_lower
                    or p_lower.startswith("xchange-ui/")
                    or "/views/" in p_lower
                ) and not any(k in p_lower for k in ("-service/", "-apis/", "/backend/"))
                if is_frontend:
                    penalty += 0.40
                    signals.append("penalty:layer_contradiction(frontend)")
            elif frontend_intent and not backend_intent:
                # Strong frontend intent: penalize pure backend services/repositories
                is_backend = (
                    p_lower.endswith((".java", ".cs", ".go", ".sql"))
                    or any(seg in p_lower for seg in ("/services/", "/service/", "/controllers/", "/controller/", "/repository/", "/repositories/"))
                ) and not ("-ui/" in p_lower or "/src/app/" in p_lower)
                if is_backend:
                    penalty += 0.35
                    signals.append("penalty:layer_contradiction(backend)")

            # ── Style-presentation intent: prefer the stylesheet over the class ──
            # Applied AFTER the layer check and only on its own, so a backend
            # ticket that merely mentions "template" is unaffected. Modest
            # penalty rather than a boost: boosts are added post-normalization
            # and saturate the 0.95 clamp, so penalties are the safe direction.
            if styling_intent:
                is_stylesheet = p_lower.endswith((".scss", ".css", ".sass", ".less"))
                is_logic_or_template = p_lower.endswith((".ts", ".tsx", ".html", ".java", ".py", ".cs", ".kt"))
                if is_logic_or_template and not is_stylesheet:
                    penalty += 0.30
                    signals.append("penalty:style_contradiction(prefer_stylesheet)")

            unified = max(0.0, unified - penalty)

            # Build signals list for planner/prompt display.
            # Merge rather than overwrite: the penalty annotations appended above
            # are appended to the local list, so reassigning from c["signals"]
            # here silently discarded them and the diagnostics never reached the
            # planner. Preserve both.
            signals = list(dict.fromkeys(list(c.get("signals", [])) + signals))
            if s_sql > 0:    signals.append("sqlite_symbol")
            if s_sym > 0:    signals.append("sym_match")
            if s_path > 0:   signals.append(f"path_own:{s_path:.2f}")
            if s_gr > 0:     signals.append(f"graph:{s_gr:.2f}")
            if s_own > 0.30: signals.append(f"ownership:{s_own:.2f}")
            if s_loc > 0:    signals.append(f"loc_asset")
            if s_semantic > 0: signals.append("semantic_catalog")
            if s_workflow > 0: signals.append("workflow_memory")
            if s_history > 0:  signals.append("ticket_history")
            if s_rkb > 0:      signals.append("rkb_evidence")
            if path_is_test and not test_intent:
                signals.append("penalty:test_noise")
            if path_is_generated:
                signals.append("penalty:generated")
            if penalty > 0:
                signals.append(f"penalty:{penalty:.2f}")

            enriched.append({
                **c,
                "confidence": round(min(0.95, unified), 3),
                "unified_score": unified,
                "features": features,
                "signals": list(dict.fromkeys(signals)),  # deduplicate, preserve order
            })

        enriched.sort(key=lambda x: (-x["unified_score"], x["path"]))

        # ── Phase 2: graph expansion → clusters → investigation → verification ──

        # Step 1: Graph-expand — pull companion files (template/style/injected)
        # into the pool if they were absent from the path walk.
        expanded = self._graph_expand_candidates(enriched)
        existing_paths_set = {e["path"] for e in enriched}
        new_raws = [c for c in expanded if c["path"] not in existing_paths_set]
        if new_raws:
            for c in new_raws:
                path = c["path"]
                content = c.get("_content", "")
                raw_sql = sql_hit_map.get(path, 0)
                file_lang = stereotype_map.get(path, ("", []))[0] or "unknown"
                lang_max = lang_max_sql.get(file_lang, 1) or 1
                s_sql_e = min(1.0, raw_sql / lang_max)
                lang_e, anns_e = stereotype_map.get(path, ("", []))
                ann_texts_e = " ".join(str(a).lower() for a in anns_e)
                s_sym_e = 1.0 if any(k in ann_texts_e for k in kw_lower) else 0.0
                s_path_e = self._path_ownership_score(path, keywords)
                s_gr_e = max(
                    neighbor_map.get(path, 0.0),
                    c.get("_preseed_s_gr", 0.0),
                )
                s_vec_e = (vector_scores or {}).get(path, 0.0)
                s_own_e = self._compute_ownership(path, keywords, content)
                feat_e = {
                    "s_fs": 0.0, "s_sql": s_sql_e,
                    "s_sym": s_sym_e, "s_path": s_path_e,
                    "s_gr": s_gr_e, "s_vec": s_vec_e, "s_own": s_own_e,
                    "s_ja": 0.0, "s_ts": 0.0,
                }
                unified_e = self._unified_score(feat_e)
                sigs_e = list(c.get("signals", []))
                if s_sql_e > 0:   sigs_e.append("sqlite_symbol")
                if s_sym_e > 0:   sigs_e.append("sym_match")
                if s_path_e > 0:  sigs_e.append(f"path_own:{s_path_e:.2f}")
                if s_own_e > 0.3: sigs_e.append(f"ownership:{s_own_e:.2f}")
                enriched.append({
                    **c,
                    "confidence":    round(min(0.95, unified_e), 3),
                    "unified_score": unified_e,
                    "features":      feat_e,
                    "signals":       list(dict.fromkeys(sigs_e)),
                })
            enriched.sort(key=lambda x: -x["unified_score"])

        # Step 2: Build ownership clusters (structural edges only).
        s_own_map = {
            e["path"]: e.get("features", {}).get("s_own", 0.0) for e in enriched
        }
        clusters = self._build_ownership_clusters(enriched, s_own_map)

        # Step 3: Write s_rel + cluster metadata into every candidate.
        enriched = self._apply_relationship_boost(enriched, clusters)

        # Step 4: Re-score companions that received s_rel > 0; re-sort.
        for entry in enriched:
            if entry.get("features", {}).get("s_rel", 0.0) > 0.0:
                entry["unified_score"] = self._unified_score(entry["features"])
                entry["confidence"] = round(min(0.95, entry["unified_score"]), 3)
                s_r = entry["features"]["s_rel"] * self._W_REL_BOOST
                sigs = list(entry.get("signals", []))
                sigs.append(f"rel_boost:{s_r:.3f}")
                entry["signals"] = list(dict.fromkeys(sigs))
        enriched.sort(key=lambda x: -x["unified_score"])

        # Step 5: Consensus scoring on top-10.
        consensus_map = self._consensus_score(enriched[:10])
        for entry in enriched:
            consensus_score = round(consensus_map.get(entry["path"], 0.0), 3)
            features = entry.get("features", {})
            features["consensus_score"] = consensus_score
            entry["consensus_confidence"] = consensus_score

        if self.enable_consensus_confidence:
            for entry in enriched:
                consensus_score = float(entry.get("consensus_confidence", 0.0))
                if consensus_score <= 0.0:
                    continue
                entry["unified_score"] = min(
                    2.0,
                    entry["unified_score"] * (1.0 + self.consensus_boost_weight * consensus_score),
                )
                entry["confidence"] = round(min(0.95, entry["unified_score"]), 3)
                sigs = list(entry.get("signals", []))
                sigs.append(f"consensus:{consensus_score:.2f}")
                entry["signals"] = list(dict.fromkeys(sigs))
            enriched.sort(key=lambda x: -x["unified_score"])

        # Step 5.5: Feature-anchor surfacing — raise the correct component into
        # the inspected set. Discovery signal ONLY: it re-orders candidates and
        # records feature_anchor; it never sets ownership_type or write authority.
        _ns = extract_ticket_namespace(ticket_text, self.workspace_path)
        _feature_tokens = extract_feature_phrase(ticket_text)
        apply_feature_anchor_boost(enriched, _feature_tokens, namespace=_ns)

        # Step 6: Investigate top-20 — assigns an advisory ownership ROLE
        # (primary ownership is proven later by inspection, not here).
        for entry in enriched[:20]:
            path    = entry["path"]
            content = entry.get("_content", "")
            if not content:
                try:
                    content = (self.workspace_path / path).read_text(
                        encoding="utf-8", errors="ignore")
                except Exception:
                    content = ""
            inv = self._investigate_candidate(path, content, keywords)
            own_type, own_reason = self._verify_ownership(
                path, inv, entry.get("features", {}))
            feat = entry.get("features", {})
            feat["ownership_type"]   = own_type
            feat["ownership_reason"] = own_reason

            # Classify specific architectural component role
            name_lower = Path(path).name.lower()
            if "controller" in name_lower or "resource" in name_lower:
                feat["component_role"] = "CONTROLLER"
            elif "service" in name_lower:
                feat["component_role"] = "SERVICE"
            elif "repository" in name_lower or "dao" in name_lower:
                feat["component_role"] = "REPOSITORY"
            elif "handler" in name_lower:
                feat["component_role"] = "HANDLER"
            elif "component" in name_lower:
                feat["component_role"] = "COMPONENT"

            feat["inv_defines"]      = inv["defines"]
            feat["inv_displays"]     = inv["displays"]
            feat["inv_calculates"]   = inv["calculates"]
            feat["inv_persists"]     = inv["persists"]
            sigs = list(entry.get("signals", []))
            sigs.append(f"verified:{own_type}")
            entry["signals"] = list(dict.fromkeys(sigs))

        # Step 7: Attach cluster trace to cluster-primary candidates.
        cluster_by_primary = {cl["primary"]: cl for cl in clusters}
        for entry in enriched:
            path = entry["path"]
            if path in cluster_by_primary:
                cl = cluster_by_primary[path]
                feat = entry.get("features", {})
                cl["ownership_type"] = feat.get("ownership_type", "UNKNOWN")
                cl["investigation"]  = {
                    k: feat.get(f"inv_{k}", 0)
                    for k in ("defines", "displays", "calculates", "persists")
                }
                entry["cluster"] = self._format_cluster_trace(cl)

        return enriched

    # =========================================================================
    # REPOSITORY DISCOVERY (used by workflow discovery_node — runs BEFORE planning)
    # =========================================================================

    def _merge_or_add_candidate(self, candidates: list, existing_paths: set, new_candidate: dict):
        path = new_candidate["path"].replace("\\", "/")
        path_lc = path.lower()
        _JAVA_SERVICES = (
            "area-service", "contract-service", "issues-service", "project-service",
            "se-connector-apis", "se-xchange-service", "user-service", "workflow-service",
        )
        if getattr(self, "_current_is_isolated_ui", False):
            if any(path_lc.startswith(f"{svc}/") or f"/{svc}/" in path_lc for svc in _JAVA_SERVICES) or path_lc.endswith(".java"):
                return
        if getattr(self, "_current_is_isolated_backend", False):
            if path_lc.startswith("xchange-ui/") or "/xchange-ui/" in path_lc or any(path_lc.endswith(ext) for ext in (".ts", ".html", ".scss", ".css")):
                return

        if path in existing_paths:
            for c in candidates:
                if c["path"] == path:
                    for sig in new_candidate.get("signals", []):
                        if sig not in c["signals"]:
                            c["signals"].append(sig)
                    c["raw_score"] = max(c.get("raw_score", 0), new_candidate.get("raw_score", 0))
                    if new_candidate.get("promotion_reason") and "promotion_reason" not in c:
                        c["promotion_reason"] = new_candidate["promotion_reason"]
                    break
        else:
            candidates.append(new_candidate)
            existing_paths.add(path)

    def _expand_reverse_dataflow_from_seeds(
        self,
        raw_candidates: list[dict],
        existing_paths: set[str],
        max_seeds: int = 8,
        max_hops: int = 3,
    ) -> None:
        """
        Bounded backward dataflow expansion from symptom seeds.
        Walks 1-3 hops backwards (callers, importers, data suppliers, controllers)
        from top seed files to discover root-cause defect sites that lack direct
        lexical overlap with the ticket description (e.g. status overwrite loops).
        """
        if not getattr(self, "sqlite_store", None) or not hasattr(self.sqlite_store, "_conn"):
            return

        try:
            sorted_seeds = sorted(
                raw_candidates,
                key=lambda c: float(c.get("raw_score", 0.0) or 0.0),
                reverse=True,
            )[:min(max_seeds, 8)]

            logger.info("   DataFlow seeds (%d): %s", len(sorted_seeds), [s.get('path') for s in sorted_seeds])

            current_frontier = [(s["path"], 1) for s in sorted_seeds if s.get("path")]
            visited_paths = {p for p, _ in current_frontier}
            conn = self.sqlite_store._conn

            while current_frontier:
                target_path, hop = current_frontier.pop(0)
                if hop > min(max_hops, 2):
                    continue

                stem = Path(target_path).stem
                alt_stem = stem.replace("-", "").replace("_", "")

                # Fetch symbol names defined in target_path for indexed exact match
                sym_rows = conn.execute(
                    "SELECT name FROM symbols WHERE path = ? AND length(name) > 3 LIMIT 8",
                    (target_path,),
                ).fetchall()
                sym_names = [r[0] for r in sym_rows]

                caller_files: dict[str, str] = {}

                # 1. Exact match on symbol names (uses idx_edges_dst_name, sub-millisecond)
                if sym_names:
                    ph = ",".join("?" * len(sym_names))
                    sym_callers = conn.execute(
                        f"""
                        SELECT DISTINCT s_src.path, e.kind
                        FROM edges e
                        JOIN symbols s_src ON e.src_id = s_src.id
                        WHERE e.kind IN ('imports', 'calls', 'template_of')
                          AND e.dst_name IN ({ph}) AND s_src.path != ?
                        LIMIT 15
                        """,
                        (*sym_names, target_path),
                    ).fetchall()
                    for r in sym_callers:
                        caller_files[r[0]] = r[1]

                # 2. Path stem import match (single bounded query)
                import_callers = conn.execute(
                    """
                    SELECT DISTINCT s_src.path, e.kind
                    FROM edges e
                    JOIN symbols s_src ON e.src_id = s_src.id
                    WHERE e.kind IN ('imports', 'calls', 'template_of')
                      AND (e.dst_name LIKE ? OR e.dst_name LIKE ?)
                      AND s_src.path != ?
                    LIMIT 15
                    """,
                    (f"%{stem}%", f"%{alt_stem}%", target_path),
                ).fetchall()
                for r in import_callers:
                    caller_files[r[0]] = r[1]

                reach_weight = round(1.0 / hop, 3)

                for caller_path, edge_kind in caller_files.items():
                    full_p = self.workspace_path / caller_path
                    if not full_p.is_file():
                        continue
                    if is_ignored_path(full_p, workspace_path=self.workspace_path):
                        continue

                    if caller_path in existing_paths:
                        for c in raw_candidates:
                            if c["path"] == caller_path:
                                c.setdefault("signals", [])
                                sig = f"dataflow_reachability:hop_{hop}:{edge_kind}"
                                if sig not in c["signals"]:
                                    c["signals"].append(sig)
                                feat = c.setdefault("features", {})
                                feat["s_dataflow"] = max(feat.get("s_dataflow", 0.0), reach_weight)
                                c["raw_score"] = max(c.get("raw_score", 0.0), 4.0 * reach_weight)
                                break
                    else:
                        content = ""
                        try:
                            content = full_p.read_text(encoding="utf-8", errors="ignore")
                        except Exception:
                            pass

                        new_cand = {
                            "path": caller_path,
                            "confidence": 0.0,
                            "signals": [
                                "dataflow_expansion",
                                f"dataflow_reachability:hop_{hop}:{edge_kind}",
                                f"dataflow_source:{Path(target_path).name}",
                            ],
                            "raw_score": 4.0 * reach_weight,
                            "_content": content,
                            "promotion_reason": f"Backward DataFlow {edge_kind} (Hop {hop} from {Path(target_path).name})",
                            "features": {
                                "s_dataflow": reach_weight,
                                "dataflow_depth": hop,
                            },
                        }
                        self._merge_or_add_candidate(raw_candidates, existing_paths, new_cand)

                    if caller_path not in visited_paths and hop < max_hops:
                        visited_paths.add(caller_path)
                        current_frontier.append((caller_path, hop + 1))

            logger.info(
                f"   Backward DataFlow expansion completed: evaluated {len(sorted_seeds)} seeds, "
                f"total raw candidates now {len(raw_candidates)}"
            )
        except Exception as e:
            logger.warning(f"Backward DataFlow expansion failed (non-fatal): {e}", exc_info=True)

    def discover_repository_candidates(self, ticket_text: str, top_n: int = 50, evidence_items: list = None, hypotheses: list = None, rag_engine=None, run_ctx=None, extra_search_terms: list = None) -> list:
        """
        Pre-planning discovery: find real files from the repository that are
        relevant to this ticket, before the planner runs.

        Returns a list of enriched candidate dicts sorted by unified score:
          [{path, confidence, unified_score, features, signals, raw_score}]

        ``top_n`` defaults to 30 (raised from 12) to provide the multi-candidate
        analysis pipeline with enough options.  The planner receives only the
        top-5 after analysis.

        ``extra_search_terms`` — optional list of additional keywords injected by
        the planning recovery system to widen the search after a planning failure.
        """
        from ticket_to_code.models import LocalizationCandidate

        SOURCE_EXTS = {
            ".ts", ".tsx", ".js", ".jsx", ".html", ".scss", ".css", ".vue",
            ".java", ".cs", ".py", ".go", ".sql", ".xml", ".json",
        }

        keywords = self._extract_keywords(ticket_text)
        if not keywords:
            return []

        # ── Concept keyword extraction ────────────────────────────────────────
        _summary = ticket_text.split('\n\n')[0].strip()
        concept_keywords: list[str] = self._extract_keywords(_summary)
        _summary_was_sufficient = len(concept_keywords) >= 3
        if not _summary_was_sufficient:
            # Short or vague title — extend to first 400 chars
            concept_keywords = self._extract_keywords(ticket_text[:400])
        if not concept_keywords:
            concept_keywords = keywords  # failsafe: never discard everything

        # Title namespace extraction & prefix expansion (e.g. Jato: -> jato)
        _ticket_ns = extract_ticket_namespace(ticket_text, self.workspace_path)
        if _ticket_ns:
            if _ticket_ns not in concept_keywords:
                concept_keywords.insert(0, _ticket_ns)
            if _ticket_ns not in keywords:
                keywords.insert(0, _ticket_ns)
            if _ticket_ns in ("jato", "j"):
                expanded_j = []
                for kw in list(concept_keywords):
                    if not kw.startswith("j") and len(kw) >= 3:
                        expanded_j.append(f"j{kw}")
                for jkw in expanded_j:
                    if jkw not in concept_keywords:
                        concept_keywords.append(jkw)
                    if jkw not in keywords:
                        keywords.append(jkw)
                logger.info(f"   🎯 Namespace '{_ticket_ns}' active: expanded component keywords with 'j' prefix: {expanded_j}")
            
        # Smart Anchor Routing (Priority 3): Only inject core_subjects into ranking
        if hypotheses:
            for h in hypotheses:
                if hasattr(h, "core_subjects") and h.core_subjects:
                    for core in h.core_subjects:
                        if core and core not in concept_keywords:
                            concept_keywords.append(core)
                            logger.info(f"   ⚓ Added core subject anchor: {core}")

        # ── Planning Recovery: inject extra search terms from recovery context ─
        if extra_search_terms:
            for term in extra_search_terms:
                if term and term.lower() not in [k.lower() for k in concept_keywords]:
                    concept_keywords.append(term)
                    logger.info(f"   🔄 Added recovery search term: {term}")

        ticket_lc = ticket_text.lower()
        # ── Domain Keyword Expansion (Saga Handlers / Data Cleanup) ───────
        if any(w in ticket_lc for w in ("ghost", "orphan", "coordinator", "removing", "stale", "contract")):
            for extra_kw in ["handler", "saga", "user", "coordinator", "member", "contract", "repository", "delete"]:
                if extra_kw not in concept_keywords:
                    concept_keywords.append(extra_kw)
                if extra_kw not in keywords:
                    keywords.append(extra_kw)

        # ── Phase 1: path-keyword walk (all keywords for maximum recall) ──────
        # We walk with ALL keywords so that any potentially-relevant file is
        # included for the content-boost phase.  Scoring and graph-seeding use
        # concept_keywords only (computed after the walk).
        # ── Monorepo Subsystem / Domain Isolation ─────────────────────────────
        # If the workspace contains multiple services (e.g. monorepo with UI + Java microservices),
        # prevent cross-domain contamination so UI tickets never pull in Java backend services,
        # and Java backend tickets never pull in Angular UI components.
        _UI_INDICATORS = {
            "ui", "html", "css", "scss", "component", "dialog", "modal", "button",
            "dropdown", "tab", "navbar", "view", "frontend", "inline error", "template",
            "form", "overlay", "table header", "viewport", "scroll", "scrollbar", "icon",
            "banner", "click", "display", "truncate", "ellipsis", "toast", "tooltip",
            "invalid file types", "upload document register", "register document reload",
            "color", "theme", "spacing", "flexbox", "grid", "checkbox", "radio",
        }
        _BACKEND_INDICATORS = {
            "controller", "serviceimpl", "repository", "endpoint", "sql", "database",
            "migration", "kafka", "rabbitmq", "flyway", "liquibase", "jpa", "entity",
            "dao", "maven", "pom.xml", "backend", "grpc", "graphql", "mutation",
        }
        _JAVA_SERVICES = {
            "area-service", "contract-service", "issues-service", "project-service",
            "se-connector-apis", "se-xchange-service", "user-service", "workflow-service",
        }
        target_java_service = next(
            (svc for svc in _JAVA_SERVICES if svc in ticket_lc or svc.replace("-", " ") in ticket_lc),
            None
        )
        has_ui_affinity = any(re.search(r'\b' + re.escape(w) + r'\b', ticket_lc) for w in _UI_INDICATORS)
        has_backend_affinity = any(re.search(r'\b' + re.escape(w) + r'\b', ticket_lc) for w in _BACKEND_INDICATORS)
        explicit_ui_target = "xchange-ui" in ticket_lc or ("xchange" in ticket_lc and "ui" in ticket_lc)
        explicit_be_service = target_java_service is not None or any(svc in ticket_lc or svc.replace("-", " ") in ticket_lc for svc in _JAVA_SERVICES)

        is_isolated_ui = (has_ui_affinity and not has_backend_affinity) or explicit_ui_target
        is_isolated_backend = (has_backend_affinity and not has_ui_affinity) or explicit_be_service
        self._current_is_isolated_ui = is_isolated_ui
        self._current_is_isolated_backend = is_isolated_backend

        path_hits_all: list[tuple[int, Path, str]] = []
        for root, dirs, files in os.walk(self.workspace_path):
            prune_walk_dirs(dirs, root, self.workspace_path)
            dirs.sort()
            files.sort()
            if target_java_service:
                dirs[:] = [d for d in dirs if d == target_java_service or not (d in _JAVA_SERVICES or d == "xchange-ui")]
            elif is_isolated_ui:
                dirs[:] = [d for d in dirs if d not in _JAVA_SERVICES]
            elif is_isolated_backend:
                dirs[:] = [d for d in dirs if d != "xchange-ui"]

            rel_root = str(Path(root).relative_to(self.workspace_path)).replace("\\", "/")
            if target_java_service and any(rel_root == svc or rel_root.startswith(f"{svc}/") for svc in _JAVA_SERVICES if svc != target_java_service):
                continue
            if is_isolated_ui and any(rel_root == svc or rel_root.startswith(f"{svc}/") for svc in _JAVA_SERVICES):
                continue
            if is_isolated_backend and (rel_root == "xchange-ui" or rel_root.startswith("xchange-ui/")):
                continue

            for fname in files:
                full = Path(root) / fname
                if full.suffix.lower() not in SOURCE_EXTS:
                    continue
                if is_isolated_ui and full.suffix.lower() == ".java":
                    continue
                if is_isolated_backend and full.suffix.lower() in {".ts", ".html", ".scss", ".css"} and "xchange-ui" in str(full):
                    continue
                if is_ignored_path(full, workspace_path=self.workspace_path):
                    continue
                rel = str(full.relative_to(self.workspace_path)).replace("\\", "/")
                rel_lower = rel.lower()
                score = sum(1 for kw in keywords if kw.lower() in rel_lower)
                if score > 0:
                    path_hits_all.append((score, full, rel))

        if not path_hits_all:
            return []

        # ── Keyword-frequency guard (applied only when summary was insufficient)
        # When the title was too short and we fell back to the first 400 chars,
        # the extended text may still contain infra terms.  Remove any concept
        # keyword that appears in > 25 % of the candidate files' paths.
        if not _summary_was_sufficient and len(concept_keywords) > 1:
            _kw_path_counts: dict[str, int] = {}
            for _, _, _rel in path_hits_all:
                _rl = _rel.lower()
                for _kw in concept_keywords:
                    if _kw.lower() in _rl:
                        _kw_path_counts[_kw] = _kw_path_counts.get(_kw, 0) + 1
            _n_hits = len(path_hits_all)
            _filtered_kws = [
                _kw for _kw in concept_keywords
                if _kw_path_counts.get(_kw, 0) <= _n_hits * 0.25
            ]
            if _filtered_kws:  # only narrow if at least one keyword survives
                concept_keywords = _filtered_kws

        # ── Phase 2: rescore by concept keywords + content boost ──────────────
        # Re-rank path_hits by concept-keyword path score so that files whose
        # paths name the ticket concept (high specificity) dominate the top-100
        # fed to the unified scoring pipeline.  Files included via noise keywords
        # only (concept_score = 0) are kept for content matching but sorted last.
        ranking_concept_kws = list(concept_keywords)
        if target_java_service:
            _svc_tokens = set(target_java_service.lower().replace("-", " ").split())
            _pruned = [k for k in ranking_concept_kws if k.lower() not in _svc_tokens]
            if _pruned:
                ranking_concept_kws = _pruned

        path_hits: list[tuple[float, Path, str]] = []
        for _, full, rel in path_hits_all:
            rel_lc = rel.lower()
            concept_path_score = float(
                sum(1 for kw in ranking_concept_kws if kw.lower() in rel_lc)
            )
            # Domain Intent Boost: In Java microservices, Saga step handlers and repositories
            # directly control business command flows and entity persistence queries.
            if "sagas/handlers" in rel_lc and any(w in ticket_lc for w in ("coordinator", "user", "member", "role")):
                concept_path_score += 3.0
            if "repository" in rel_lc and any(w in ticket_lc for w in ("contract", "member", "count", "ghost", "orphan")):
                concept_path_score += 2.0

            path_hits.append((concept_path_score, full, rel))
        path_hits.sort(key=lambda x: (-x[0], x[2]))

        top_kws = ranking_concept_kws[:15]
        raw_candidates: list[dict] = []
        for path_score, full, rel in path_hits[:350]:
            signals = ["keyword_path"] if path_score > 0 else []
            content = ""
            try:
                content = full.read_text(encoding="utf-8", errors="ignore")
                content_score = sum(1 for kw in top_kws if kw.lower() in content.lower())
                if content_score > 0:
                    signals.append("keyword_content")
                    if not signals or signals[0] != "keyword_path":
                        signals.insert(0, "keyword_path")  # ensure keyword_path present
                total_score = path_score + content_score
            except Exception:
                total_score = float(path_score)
            if total_score > 0:
                raw_candidates.append({
                    "path":      rel,
                    "confidence": 0.0,          # placeholder; unified scoring fills this
                    "signals":   signals,
                    "raw_score": float(total_score),
                    "_content":  content,        # carried for ownership computation; stripped before return
                })

        # ── Phase 1.1: Content Search (SQLite FTS5) ─────────────────────────
        # Search exact contents of files (HTML, SCSS, JSON, TS, Java) for keywords
        # ensuring that files without the keyword in their path are still retrieved.
        existing_raw_paths = {c["path"] for c in raw_candidates}
        if getattr(self, "sqlite_store", None):
            # Pass 1: Individual keywords for dense HTML templates
            for kw in top_kws[:3]:
                try:
                    content_hits = self.sqlite_store.search_content(kw, limit=30)
                    for row in content_hits:
                        rel_path = row["path"]
                        if rel_path in existing_raw_paths:
                            continue
                        
                        full_path = self.workspace_path / rel_path
                        if not full_path.exists() or is_ignored_path(full_path, workspace_path=self.workspace_path):
                            continue
                            
                        content = ""
                        try:
                            content = full_path.read_text(encoding="utf-8", errors="ignore")
                        except Exception:
                            pass
                            
                        self._merge_or_add_candidate(
                            raw_candidates,
                            existing_raw_paths,
                            {
                                "path": rel_path,
                                "confidence": 0.0,
                                "signals": ["content_match"],
                                "raw_score": 1.0,  # FTS match is strong evidence
                                "_content": content,
                                "promotion_reason": f"Content FTS Match: '{kw}'"
                            }
                        )
                except Exception as e:
                    logger.debug(f"Content FTS search failed for '{kw}': {e}")
            
            # Pass 2: Dedicated translations and constants retrieval
            try:
                from ticket_to_code.agents.localization_asset_classifier import LocalizationAssetClassifier
                # We use more keywords here because constants might have broader terminology
                asset_sql = LocalizationAssetClassifier.get_sql_like_clauses()
                trans_hits = self.sqlite_store.search_localization_assets(top_kws[:5], asset_sql, limit=15)
                current_max_fs = max([c.get("raw_score", 0.0) for c in raw_candidates] + [1.0])
                for row in trans_hits:
                    rel_path = row["path"]
                    full_path = self.workspace_path / rel_path
                    if not full_path.exists() or is_ignored_path(full_path, workspace_path=self.workspace_path):
                        continue
                        
                    content = ""
                    try:
                        content = full_path.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        pass
                        
                    self._merge_or_add_candidate(
                        raw_candidates,
                        existing_raw_paths,
                        {
                            "path": rel_path,
                            "confidence": 0.0,
                            "signals": ["content_match", "localization_asset"],
                            "raw_score": 1.0,  # FTS match is strong evidence
                            "_content": content,
                            "promotion_reason": f"Translation FTS Match"
                        }
                    )
            except Exception as e:
                logger.debug(f"Translation FTS search failed: {e}")

        # ── Phase 1.2: Semantic Concept Layer (Graph RAG in SQLite & Neo4j) ──
        try:
            from ticket_to_code.indexing.semantic_concept_indexer import SemanticConceptIndexer
            concept_indexer = SemanticConceptIndexer(str(self.workspace_path))
            concept_hits = concept_indexer.query_concept_candidates(concept_keywords, top_n=10)
            for hit in concept_hits:
                rel_path = hit.get("file_path")
                if not rel_path:
                    continue
                full_path = self.workspace_path / rel_path
                if not full_path.exists() or is_ignored_path(full_path, workspace_path=self.workspace_path):
                    continue
                content = ""
                try:
                    content = full_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    pass
                self._merge_or_add_candidate(
                    raw_candidates,
                    existing_raw_paths,
                    {
                        "path": rel_path,
                        "confidence": 0.85,
                        "signals": [f"concept_graph:{hit.get('concept_name')}"],
                        "raw_score": 3.0 * float(hit.get("relevance", 1.0)),
                        "_content": content,
                        "promotion_reason": f"Semantic Concept Graph ({hit.get('concept_name')})",
                        "features": {
                            "s_graph": float(hit.get("relevance", 1.0)),
                        },
                    }
                )
        except Exception as e:
            logger.debug(f"SQLite Semantic Concept search failed: {e}")

        if getattr(self, "neo4j_store", None):
            try:
                concept_query = """
                MATCH (s:SemanticConcept {workspace: $workspace})
                WHERE any(alias IN s.aliases WHERE toLower(alias) IN $keywords) 
                   OR toLower(s.name) IN $keywords
                MATCH (s)-[r:RELATES_TO]->(n {workspace: $workspace})
                RETURN s.name as concept, n.name as artifact_name, n.file_path as file_path, r.confidence as confidence, r.source as source
                """
                results = self.neo4j_store.query(concept_query, {"keywords": [k.lower() for k in concept_keywords]})
                for row in results:
                    rel_path = row.get("file_path")
                    if not rel_path:
                        continue
                    full_path = self.workspace_path / rel_path
                    if not full_path.exists() or is_ignored_path(full_path, workspace_path=self.workspace_path):
                        continue
                    content = ""
                    try:
                        content = full_path.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        pass
                    self._merge_or_add_candidate(
                        raw_candidates,
                        existing_raw_paths,
                        {
                            "path": rel_path,
                            "confidence": float(row.get("confidence", 0.9)),
                            "signals": ["semantic_catalog_match"],
                            "raw_score": float(row.get("confidence", 0.9)),
                            "_content": content,
                            "promotion_reason": f"Semantic Catalog Match ({row.get('concept')} - {row.get('source', 'llm')})"
                        }
                    )
            except Exception as e:
                logger.debug(f"Neo4j Semantic Catalog Search failed: {e}")

        # ── Phase 1.3: Episodic Memory Recall (Cross-Run Learning) ────────────
        try:
            from ticket_to_code.memory.episodic_memory import EpisodicMemoryStore
            ep_store = EpisodicMemoryStore(str(self.workspace_path))
            episodes = ep_store.recall_similar_episodes(ticket_text, top_k=2)
            for ep, sim_score in episodes:
                if sim_score >= 0.35:
                    for r_file in ep.resolved_files:
                        full_path = self.workspace_path / r_file
                        if not full_path.exists() or is_ignored_path(full_path, workspace_path=self.workspace_path):
                            continue
                        content = ""
                        try:
                            content = full_path.read_text(encoding="utf-8", errors="ignore")
                        except Exception:
                            pass
                        vl = getattr(ep, "verification_level", None)
                        self._merge_or_add_candidate(
                            raw_candidates,
                            existing_raw_paths,
                            {
                                "path": r_file,
                                "confidence": _episodic_confidence(vl),
                                "signals": [
                                    f"episodic_recall:{ep.episode_id}",
                                    f"similarity:{sim_score:.2f}",
                                    f"verification:{vl or 'unknown'}",
                                ],
                                "raw_score": 4.0 * sim_score,
                                "_content": content,
                                "promotion_reason": f"Episodic Memory Match ({ep.ticket_id}, sim={sim_score:.2f})",
                                "features": {
                                    "s_episodic": sim_score,
                                },
                            }
                        )
        except Exception as e:
            logger.debug(f"Episodic Memory recall failed: {e}")

        # ── Phase 1.5: Evidence Promotion ────────────────────────────
        # Bypass path-keyword filtering for files explicitly promoted by upstream evidence.
        # IMPORTANT: LangGraph state serialization often converts Pydantic objects to raw dicts.
        # We must support BOTH object attribute access AND dict key access to prevent files like
        # run-job.sh from being silently dropped when ev is a dict instead of an EvidenceItem object.
        evidence_items = evidence_items or []
        evidence_candidates_by_path = {}
        
        for ev in evidence_items:
            # Support both Pydantic object and raw dict (LangGraph state serialization)
            if isinstance(ev, dict):
                path = ev.get("file_path") or ev.get("path")
                score = float(ev.get("relevance_score", ev.get("score", 0.0)))
                source = ev.get("source", "unknown_source")
                content_snippet = ev.get("content_snippet", ev.get("snippet", ""))
                hypothesis_id = ev.get("hypothesis_id", "None")
            else:
                path = getattr(ev, "file_path", None)
                score = float(getattr(ev, "relevance_score", 0.0))
                source = getattr(ev, "source", "unknown_source")
                content_snippet = getattr(ev, "content_snippet", "")
                hypothesis_id = getattr(ev, "hypothesis_id", "None")
            
            if not path:
                continue
            
            if path not in evidence_candidates_by_path or score > evidence_candidates_by_path[path]["confidence"]:
                evidence_candidates_by_path[path] = {
                    "confidence": score,
                    "query_type": source,
                    "matched_text": content_snippet,
                    "hypothesis_id": hypothesis_id,
                }
                
        current_max_fs = max([c.get("raw_score", 0.0) for c in raw_candidates] + [1.0])
        existing_raw_paths = {c["path"] for c in raw_candidates}
        for rel_path, info in evidence_candidates_by_path.items():
            if rel_path in existing_raw_paths:
                continue
            
            full_path = self.workspace_path / rel_path
            if not full_path.exists() or is_ignored_path(full_path, workspace_path=self.workspace_path):
                continue

            content = ""
            try:
                content = full_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
                
            raw_candidates.append({
                "path": rel_path,
                "confidence": 0.0,
                "signals": ["keyword_path", f"evidence:{info['query_type']}"], 
                "raw_score": current_max_fs * float(info["confidence"]),
                "_content": content,
                "evidence_score": info["confidence"],
                "matched_literals": info["matched_text"],
                "hypothesis_id": info["hypothesis_id"],
                "promotion_reason": f"Promoted via Upstream Evidence ({info['query_type']})"
            })

        # ── Phase 1.6: RKB Feature Discovery ────────────────────────────
        # Semantically search the Vector DB for cross-cutting business features
        # and pull their Neo4j graph implementation bounds.
        try:
            if getattr(self, 'vector_store', None) and getattr(self, 'neo4j_store', None):
                for kw in concept_keywords[:3]:  # Top 3 concepts
                    if hasattr(self.vector_store, 'search'):
                        rkb_results = self.vector_store.search(query=kw, limit=1)
                        for res in rkb_results:
                            feature_name = getattr(res, 'metadata', {}).get("feature_name")
                            if feature_name:
                                logger.debug(f"Phase 1.6: Feature '{feature_name}' found in VectorDB, but Neo4j Feature bounds are not implemented.")
        except Exception as e:
            logger.warning(f"RKB Feature Discovery failed: {e}")

        # ── Phase 1.7: Workflow Discovery ────────────────────────────────
        # Reuse the dedicated workflow graph to surface complete Route →
        # Component → Store/Service → API → Controller → BackendService paths.
        # Workflow paths are NOT standalone evidence. They only boost existing candidates.
        workflow_paths = self._consult_workflow_paths(concept_keywords)
        if workflow_paths:
            logger.info(f"   Workflow discovery surfaced {len(workflow_paths)} file path(s)")
            workflow_paths_set = set(workflow_paths)
            for cand in raw_candidates:
                if cand["path"] in workflow_paths_set:
                    if "workflow_path" not in cand["signals"]:
                        cand["signals"].append("workflow_path")
                    gate_ok, gate_reason = self._workflow_precision_gate(cand)
                    if gate_ok:
                        cand["raw_score"] *= 1.2  # legacy boost preserved when gate passes
                        cand["signals"].append("workflow_gate_pass")
                        logger.info(f"   🛤️ Boosted existing candidate via Workflow: {cand['path']}")
                    else:
                        cand["signals"].append(f"workflow_gate:{gate_reason}")
                        logger.info(
                            f"   ⏸️ Workflow boost gated for {cand['path']} "
                            f"({gate_reason})"
                        )

        if not raw_candidates:
            return []

        # ── Phase 1.8: Bounded Backward DataFlow Expansion from Symptom Seeds ───
        # When defect causes are separated from symptom terms (e.g. status overwritten
        # in a parent component/controller while the ticket mentions UI buttons/constants),
        # lexical search misses the defect site.
        # We trace backward callers, importers, and data suppliers 1-3 hops from top seeds.
        self._expand_reverse_dataflow_from_seeds(raw_candidates, existing_raw_paths)

        # ── B1: Vector similarity scoring via RAG engine ─────────────────────────
        # Compute pgvector cosine-similarity scores for each candidate so that
        # _analyze_candidates can populate the s_vec signal.  When the RAG
        # engine is unavailable or returns no results, vector_scores stays None
        # and s_vec defaults to 0.0 (backward-compatible, no exception).
        vector_scores: dict[str, float] | None = None
        if rag_engine is not None and getattr(rag_engine, "is_initialized", False):
            try:
                # Use the ticket text as the semantic query
                _vs = getattr(rag_engine, "vector_store", None)
                if _vs is not None and hasattr(_vs, "similarity_search_with_score"):
                    _results = _vs.similarity_search_with_score(ticket_text[:512], k=50)
                    _vec_map: dict[str, float] = {}
                    for doc, score in (_results or []):
                        fp = doc.metadata.get("file_path", "")
                        if fp and fp not in _vec_map:
                            _vec_map[fp] = max(0.0, 1.0 - float(score))
                    if _vec_map:
                        vector_scores = _vec_map
                        logger.info(
                            "B1 vector scoring: %d candidates scored via pgvector",
                            len(_vec_map),
                        )
            except Exception as _ve:
                logger.debug("B1 vector scoring failed (non-fatal): %s", _ve)

        # ── Phase 3: unified ranking (graph + ownership + SQLite + annotations)
        # Pass concept_keywords so that annotation/stereotype matching and
        # graph-seed selection use the same noise-free keyword set.
        enriched = self.analyze_candidates(
            raw_candidates, ticket_text,
            focused_keywords=concept_keywords,
            vector_scores=vector_scores,  # B1: real s_vec scores
        )

        # ── Phase 4: Candidate Package Builder ──────────────────────────────────
        # Instead of passing raw file strings to the Planner/Ownership Agent, we
        # build structured JSON packages.  Each package captures:
        #   - candidate_origin: WHY this file reached the pipeline (for debugging)
        #   - ownership_signals: WHAT signals support its candidacy (grouped by type)
        #   - matched_snippets: actual code excerpts so LLM can verify symbol claims
        #
        # Budget-based cap (not a fixed count):
        #   max_primary_candidates=15, max_graph_neighbors_per_owner=5, max_total=50
        CANDIDATE_BUDGET = 50

        result = []
        for c in enriched[:CANDIDATE_BUDGET]:
            content  = c.get("_content", "")
            signals  = c.get("signals", [])
            path     = c.get("path", "")

            # ── Build candidate_origin ─────────────────────────────────────────
            # Reconstructs from signals so it is entirely data-driven, not LLM.
            candidate_origin = {
                "literal_match":   any("keyword" in s or "content_match" in s or "fts" in s for s in signals),
                "graph_expansion": any("graph" in s or "rel_boost" in s for s in signals),
                "brain_match":     any("semantic_catalog" in s or "rkb_feature" in s for s in signals),
                "multi_source":    len({
                    "fs"    if any("keyword_path" in s or "keyword_content" in s for s in signals) else None,
                    "graph" if any("graph" in s for s in signals) else None,
                    "sqlite" if any("sqlite" in s for s in signals) else None,
                    "evidence" if any("evidence:" in s for s in signals) else None,
                } - {None}) >= 2,
                "cluster_id":     next((s.split("cluster:")[-1] for s in signals if "cluster:" in s), None),
            }

            # ── Build ownership_signals ────────────────────────────────────────
            ownership_signals: dict = {
                "literal_matches":  c.get("features", {}).get("literal_matches", []),
                "symbol_matches":   c.get("features", {}).get("sqlite_symbols", []),
                "routing_matches":  c.get("features", {}).get("routing_matches", []),
                "graph_edges":      [],
                "brain_matches":    [],
            }
            # Populate graph_edges from graph relationships stored in features
            gf = c.get("features", {})
            if gf.get("graph_neighbors"):
                for nb in gf["graph_neighbors"][:5]:
                    ownership_signals["graph_edges"].append(
                        f"neighbor:{nb}" if isinstance(nb, str) else str(nb)
                    )
            # Fallback: extract from signals strings like "consumed_by:X"
            for sig in signals:
                if ":" in sig and not any(prefix in sig for prefix in ["keyword", "sqlite", "graph:", "path_own", "cluster", "verified", "evidence", "rel_boost", "workflow"]):
                    ownership_signals["graph_edges"].append(sig)

            # ── Build matched_snippets ─────────────────────────────────────────
            # Pull actual code lines around matched literals so the Ownership Agent
            # can verify that a symbol truly *defines* a value rather than just
            # mentioning it.
            matched_snippets: list = []
            if content and concept_keywords:
                lines = content.splitlines()
                seen_lines: set = set()
                for kw in concept_keywords[:6]:
                    kw_lc = kw.lower()
                    for i, line in enumerate(lines):
                        if kw_lc in line.lower() and i not in seen_lines:
                            seen_lines.add(i)
                            # Classify the snippet type heuristically
                            stripped = line.strip()
                            if any(tok in stripped for tok in ("const ", "readonly ", "= '", '= "', "export const", "static ")):
                                snip_type = "constant"
                            elif any(tok in stripped for tok in ("def ", "function ", "() {", "): ")):
                                snip_type = "method"
                            elif stripped.endswith("{") or stripped.endswith(","):
                                snip_type = "config"
                            else:
                                snip_type = "literal"
                            matched_snippets.append({
                                "type":    snip_type,
                                "text":    stripped[:200],
                                "line_no": i + 1,
                            })
                            if len(matched_snippets) >= 5:
                                break
                    if len(matched_snippets) >= 5:
                        break

            # ── Assemble final package ─────────────────────────────────────────
            c_pkg = {k: v for k, v in c.items() if k not in ("_content",)}
            c_pkg["candidate_origin"]   = candidate_origin
            c_pkg["ownership_signals"]  = ownership_signals
            c_pkg["matched_snippets"]   = matched_snippets
            result.append(c_pkg)

        logger.info(
            f"  🔭 Repository Discovery: {len(result)} candidate(s) found (budget cap={CANDIDATE_BUDGET})\n"
            + "\n".join(
                f"    {c['confidence']:.3f}  {c['path']}"
                f"  [{', '.join(c.get('signals', [])[:3])}]"
                + (f" origin={c.get('candidate_origin', {})}" if logger.isEnabledFor(logging.DEBUG) else "")
                for c in result[:8]
            )
        )

        # ── Phase 5: LLM-based relevance re-ranking ──────────────────────────
        # Instead of passing ALL keyword-matched files to the planner, run a
        # fast LLM call that scores each candidate's actual relevance to the
        # ticket's *intent* (not just keyword overlap).  This prevents the
        # planner from receiving irrelevant files (e.g. Java DTOs for a CSS
        # scrollbar bug) and hallucinating unnecessary modify tasks.
        #
        # The re-ranker splits candidates into 3 tiers:
        # ── Domain Isolation Enforcement on Final Candidates ─────────────────
        _JAVA_SERVICES_TUPLE = (
            "area-service", "contract-service", "issues-service", "project-service",
            "se-connector-apis", "se-xchange-service", "user-service", "workflow-service",
        )
        if is_isolated_ui:
            result = [
                c for c in result
                if not (any(c["path"].replace("\\", "/").lower().startswith(f"{svc}/") or f"/{svc}/" in c["path"].replace("\\", "/").lower() for svc in _JAVA_SERVICES_TUPLE) or c["path"].lower().endswith(".java"))
            ]
        if is_isolated_backend:
            result = [
                c for c in result
                if not (c["path"].replace("\\", "/").lower().startswith("xchange-ui/") or "/xchange-ui/" in c["path"].replace("\\", "/").lower() or any(c["path"].lower().endswith(ext) for ext in (".ts", ".html", ".scss", ".css")))
            ]

        # ── Companion Expansion and Intent-Based Boosting ─────────────────────
        result = self._expand_companions_with_intent(result, ticket_text)

        tier1, tier2, _tier3 = self._rerank_by_ticket_intent(result, ticket_text)

        logger.info(
            f"  🎯 Re-Ranker: Tier1={len(tier1)}, Tier2={len(tier2)}, Dropped={len(_tier3)}"
        )
        for c in tier1[:5]:
            logger.info(
                f"    ✅ T1  {c.get('rerank_score', '?'):>2}/10  {c['path']}"
                f"  — {c.get('rerank_reason', '')[:60]}"
            )
        for c in tier2[:3]:
            logger.info(
                f"    📦 T2  {c.get('rerank_score', '?'):>2}/10  {c['path']}"
                f"  — {c.get('rerank_reason', '')[:60]}"
            )

        # Store tier2 candidates on a special key so the caller (workflow
        # discovery_node) can stash them in state for fallback expansion.
        # The primary return list is tier1 only.
        if tier1:
            for c in tier1:
                c["_tier"] = "primary"
            for c in tier2:
                c["_tier"] = "backup"
            # Attach tier2 as metadata on the first candidate (workflow extracts it)
            tier1[0]["_tier2_backup"] = tier2
            return tier1
        else:
            # Re-ranker filtered everything — fall back to the original unfiltered list
            logger.warning("  ⚠️ Re-Ranker filtered all candidates — falling back to unranked list")
            return result

    def _resolve_companion_group(self, file_path: str) -> list[str]:
        """Resolve sibling companion files (.scss, .html, .ts) for a component group."""
        skill_comps = [file_path]
        try:
            from ticket_to_code.skills.language_skill_registry import LanguageSkillRegistry
            skill_comps = LanguageSkillRegistry.get_instance().resolve_companions(file_path, str(self.workspace_path))
        except Exception as exc:
            logger.debug(f"[_resolve_companion_group] Skill resolver error: {exc}")

        try:
            from ticket_to_code.intelligence.contracts.companion_resolver import PolyglotCompanionResolver
            companions = PolyglotCompanionResolver.resolve_all_companions(file_path, str(self.workspace_path))
        except Exception as exc:
            logger.debug(f"[_resolve_companion_group] PolyglotCompanionResolver error: {exc}")
            companions = [file_path]

        group = list(dict.fromkeys([file_path] + [c for c in skill_comps if c != file_path] + [c for c in companions if c != file_path]))
        return group

    def _expand_companions_with_intent(self, candidates: list[dict], ticket_text: str) -> list[dict]:
        """Expand Angular/UI components to their companion files (.scss, .html).

        Applies intent-based boosts:
        - If ticket has style/visual intent, boost .scss/.css by +2.0 raw / +0.15 confidence
        - If ticket has template/DOM intent, boost .html by +1.5 raw / +0.10 confidence
        - Ensures deterministic tie-breaking (-confidence, -raw_score, path).
        """
        if not candidates:
            return []

        t_lc = ticket_text.lower()
        is_style = any(re.search(r'\b' + re.escape(w) + r'\b', t_lc) for w in [
            "css", "scss", "style", "styles", "styling", "color", "colors", "theme",
            "height", "width", "scrollbar", "scroll", "overflow", "background",
            "iconsize", "uxbreaks", "space", "spacing", "padding", "margin",
            "font", "fontsize", "alignment", "border", "shadow"
        ])
        is_template = any(re.search(r'\b' + re.escape(w) + r'\b', t_lc) for w in [
            "html", "template", "templates", "display", "window", "button", "label",
            "text", "icon", "dialog", "section", "table", "column", "row", "view",
            "message", "count", "action", "modal", "tooltip", "checkbox", "radio",
            "dropdown", "select", "banner", "header", "footer"
        ])

        new_cands = []
        seen = set()

        for c in candidates:
            p = c.get("path", "")
            if not p:
                continue
            ext = Path(p).suffix.lower()
            c_copy = dict(c)
            if p not in seen:
                seen.add(p)
                # In-place intent boost if file already matches intent
                if is_style and ext in {".scss", ".css"}:
                    c_copy["raw_score"] = c.get("raw_score", 1.0) + 2.0
                    c_copy["unified_score"] = c.get("unified_score", 0.8) + 0.15
                    c_copy["confidence"] = min(0.95, round(c_copy["unified_score"], 3))
                    c_copy["signals"] = list(c.get("signals", [])) + ["style_intent_boost"]
                elif is_template and ext == ".html":
                    c_copy["raw_score"] = c.get("raw_score", 1.0) + 1.5
                    c_copy["unified_score"] = c.get("unified_score", 0.8) + 0.10
                    c_copy["confidence"] = min(0.95, round(c_copy["unified_score"], 3))
                    c_copy["signals"] = list(c.get("signals", [])) + ["template_intent_boost"]
                new_cands.append(c_copy)

            # Expand companion files if this is an Angular/UI component
            if "xchange-ui" in p or ext in {".ts", ".html", ".scss"}:
                group = self._resolve_companion_group(p)
                for comp in group:
                    if comp not in seen:
                        seen.add(comp)
                        comp_ext = Path(comp).suffix.lower()
                        base_raw = c.get("raw_score", 1.0)
                        base_unif = c.get("unified_score", 0.8)
                        signals = list(c.get("signals", [])) + ["companion_expansion"]

                        if is_style and comp_ext in {".scss", ".css"}:
                            raw_sc = base_raw + 2.0
                            unif_sc = base_unif + 0.15
                            signals.append("style_intent_boost")
                        elif is_template and comp_ext == ".html":
                            raw_sc = base_raw + 1.5
                            unif_sc = base_unif + 0.10
                            signals.append("template_intent_boost")
                        else:
                            raw_sc = base_raw * 0.90
                            unif_sc = base_unif * 0.90

                        conf = min(0.95, round(unif_sc, 3))
                        comp_dict = {
                            "path": comp,
                            "confidence": conf,
                            "unified_score": unif_sc,
                            "raw_score": raw_sc,
                            "signals": signals,
                            "candidate_origin": dict(c.get("candidate_origin", {})),
                            "ownership_signals": dict(c.get("ownership_signals", {})),
                            "matched_snippets": [],
                        }
                        new_cands.append(comp_dict)

        # Deterministic tie-breaking: -confidence, -raw_score, path
        new_cands.sort(key=lambda x: (-x.get("confidence", 0.0), -x.get("raw_score", 0.0), x["path"]))
        return new_cands

    # ── LLM-based Relevance Re-Ranker ─────────────────────────────────────────

    def _build_file_summary(self, file_path: str, content: str) -> str:
        """Build a compact summary of a file for the re-ranker LLM.

        Extracts key structural information (class names, CSS selectors,
        HTML elements, method signatures) without sending the full file
        content — keeping the re-ranker call fast and token-efficient.
        """
        ext = Path(file_path).suffix.lower()
        parts = [file_path]

        if ext in (".scss", ".css", ".less", ".sass"):
            classes = re.findall(r"\.([a-zA-Z][\w-]*)\s*\{", content)
            if classes:
                parts.append(f"CSS classes: {', '.join(classes[:12])}")
            has_props = []
            for prop in ("overflow", "height", "max-height", "scrollbar",
                         "position", "display", "flex", "z-index"):
                if prop in content.lower():
                    has_props.append(prop)
            if has_props:
                parts.append(f"Properties: {', '.join(has_props)}")

        elif ext in (".html", ".htm"):
            tags = re.findall(r"<([\w-]+)", content)
            unique = list(dict.fromkeys(tags))[:15]
            if unique:
                parts.append(f"Elements: {', '.join(unique)}")
            directives = re.findall(r"\*ng\w+", content)
            if directives:
                parts.append(f"Directives: {', '.join(list(set(directives))[:6])}")

        elif ext in (".ts", ".tsx", ".js", ".jsx"):
            cls = re.findall(r"class\s+(\w+)", content)
            if cls:
                parts.append(f"Class: {cls[0]}")
            if ".component.ts" in file_path:
                parts.append("Type: Angular Component")
            elif ".service.ts" in file_path:
                parts.append("Type: Angular Service")
            elif ".module.ts" in file_path:
                parts.append("Type: Angular Module")
            methods = re.findall(
                r"(?:public\s+|private\s+|protected\s+|async\s+)*(\w+)\s*\(",
                content,
            )
            # Filter out common noise
            methods = [
                m for m in methods
                if m not in ("if", "for", "while", "switch", "catch",
                             "return", "import", "require", "constructor")
            ][:8]
            if methods:
                parts.append(f"Methods: {', '.join(methods)}")

        elif ext == ".java":
            cls = re.findall(r"class\s+(\w+)", content)
            if cls:
                parts.append(f"Class: {cls[0]}")
            annotations = re.findall(r"@(\w+)", content)
            key_annots = [a for a in set(annotations)
                          if a in ("Entity", "RestController", "Service",
                                   "Repository", "Component", "Controller",
                                   "RequestMapping", "Table")]
            if key_annots:
                parts.append(f"Annotations: {', '.join(key_annots)}")
            if "DTO" in file_path or "Dto" in file_path:
                parts.append("Type: Data Transfer Object")
            # Extract method names
            methods = re.findall(
                r"(?:public\s+|private\s+|protected\s+|static\s+)*[\w<>\[\]]+\s+(\w+)\s*\(",
                content,
            )
            methods = [
                m for m in methods
                if m not in ("if", "for", "while", "switch", "catch", "return")
            ][:8]
            if methods:
                parts.append(f"Methods: {', '.join(methods)}")

        return " | ".join(parts)

    def _rerank_by_ticket_intent(
        self,
        candidates: list,
        ticket_text: str,
    ) -> "tuple[list, list, list]":
        """Score each candidate's relevance to the ticket's actual intent.

        Uses a single fast LLM call with compact file summaries.
        Returns (tier1, tier2, tier3) where:
          tier1: score 7-10 — primary targets for the planner
          tier2: score 4-6 — backup candidates for fallback expansion
          tier3: score 0-3 — irrelevant, dropped

        Falls back gracefully: if the LLM call fails, all candidates
        stay in tier1 (i.e. the re-ranker becomes a no-op).
        """
        if not candidates or len(candidates) <= 3:
            # Too few candidates to bother re-ranking
            return candidates, [], []

        # Build compact summaries for each candidate
        summaries = []
        # We re-rank up to 50 candidates (full candidate budget)
        rerank_pool = candidates[:50]
        overflow = candidates[50:]  # These go straight to tier2

        for c in rerank_pool:
            content = c.get("_content", "")
            if not content:
                # Try reading from disk if content was stripped earlier
                try:
                    fp = self.workspace_path / c["path"]
                    if fp.exists():
                        content = fp.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    content = ""
            summary = self._build_file_summary(c["path"], content)
            summaries.append(summary)

        # Load architecture knowledge from brain folder (dynamic — works for ANY project)
        arch_context = ""
        try:
            import os as _os, glob as _glob
            _ws = getattr(self, 'workspace_path', None) or str(self._workspace) if hasattr(self, '_workspace') else None
            if _ws:
                _brain_dir = _os.path.join(str(_ws), "brain", "knowledge")
                if _os.path.isdir(_brain_dir):
                    for _md in sorted(_glob.glob(_os.path.join(_brain_dir, "*.md"))):
                        try:
                            with open(_md, "r", encoding="utf-8") as _f:
                                _content = _f.read().strip()
                            if _content and len(_content) < 5000:
                                arch_context += f"\n{_content}\n"
                        except Exception:
                            pass
                    if arch_context:
                        arch_context = f"\nPROJECT ARCHITECTURE CONTEXT:\n{arch_context}"
        except Exception:
            pass

        prompt = f"""You are a code relevance expert. Given a ticket and candidate files, score each file's RELEVANCE to implementing the fix.

TICKET:
{ticket_text[:600]}
{arch_context}
CANDIDATE FILES:
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(summaries))}

SCORING RULES:
- 7-10: This file MUST be modified or directly involved in implementing the fix (directly executes the behavior, saga step handler, repository query method, or data entity)
- 4-6: This file MIGHT be needed (related component, callers, uncertain if changes required)
- 0-3: This file is NOT relevant to this specific ticket (different layer/concern, unrelated domain entities)

Think step by step:
1. What is the ticket asking? (e.g. updating a user role/coordinator to member, checking contract counts, deleting ghost participating members).
2. Microservice Saga Handlers & Step Handlers: In Event-Driven Java Microservices, updates to users/roles or domain commands are executed by Saga Step Handlers (*Handler.java in sagas/handlers/). If the ticket updates a user or project entity, the corresponding Saga Handler (e.g. UpdateProjectUserHandler) directly implements the update logic and MUST be scored 8-10!
3. Repositories: If checking counts or existence (e.g. 0 contracts), Spring Data Repositories (*Repository.java) that provide or need the query method (e.g. ContractMemberRepository) MUST be scored 7-10!
4. Which files are just keyword matches but from an unrelated feature or service? Score 0-3.

Output a JSON array ONLY, no markdown:
[{{ "index": 1, "score": 10, "reason": "Controls the described behavior via XYZ" }}]"""

        try:
            response = llm_invoke(self.llm, prompt)
            raw = response.content.strip()

            # Clean markdown fences if present
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            scores = []
            try:
                # Try standard JSON parsing first
                scores = json.loads(raw)
                if not isinstance(scores, list):
                    raise ValueError("Expected JSON array")
            except Exception as e:
                # REGEX FALLBACK: The LLM output was likely truncated (unterminated string).
                # Extract all fully completed {"index": X, "score": Y, ...} objects we can find.
                logger.warning(f"  ⚠️ Standard JSON parse failed ({e}). Attempting regex recovery on truncated string.")
                import re
                
                # Match dicts containing "index" and "score"
                matches = re.findall(r'\{\s*"index"\s*:\s*\d+\s*,\s*"score"\s*:\s*\d+[^}]*\}', raw)
                for match in matches:
                    try:
                        scores.append(json.loads(match))
                    except json.JSONDecodeError:
                        continue
                
                if not scores:
                    raise ValueError("Regex recovery failed to extract any valid scores.")
                logger.info(f"  ✅ Regex recovery salvaged {len(scores)} scores.")

            # Build index→score map
            score_map: dict[int, dict] = {}
            for entry in scores:
                idx = int(entry.get("index", 0)) - 1  # 1-indexed → 0-indexed
                if 0 <= idx < len(rerank_pool):
                    score_map[idx] = {
                        "score": min(10, max(0, int(entry.get("score", 5)))),
                        "reason": str(entry.get("reason", ""))[:120],
                    }

            # Split into tiers
            tier1, tier2, tier3 = [], [], []
            for i, c in enumerate(rerank_pool):
                info = score_map.get(i, {"score": 5, "reason": "not scored"})
                c["rerank_score"] = info["score"]
                c["rerank_reason"] = info["reason"]

                if info["score"] >= 7:
                    tier1.append(c)
                elif info["score"] >= 4:
                    tier2.append(c)
                else:
                    tier3.append(c)

            # Sort tiers by score descending
            tier1.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
            tier2.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)

            # Overflow candidates (beyond top 30) go to tier2 as backup
            for c in overflow:
                c["rerank_score"] = 4
                c["rerank_reason"] = "overflow — not re-ranked"
                tier2.append(c)

            logger.info(
                f"  Re-Ranker LLM scored {len(score_map)}/{len(rerank_pool)} "
                f"candidates: T1={len(tier1)}, T2={len(tier2)}, T3={len(tier3)}"
            )
            return tier1, tier2, tier3

        except Exception as exc:
            # Graceful fallback: if re-ranker fails, treat ALL as tier1
            logger.warning(f"  ⚠️ Re-Ranker failed ({exc}) — all candidates stay in tier1")
            for c in candidates:
                c["rerank_score"] = 5
                c["rerank_reason"] = "reranker_fallback"
            return candidates, [], []


    # =========================================================================
    # PLANNED TASK LOCALIZATION (used by workflow localize_node)
    # =========================================================================

    def localize_planned_tasks(
        self,
        tasks: "List[DevelopmentTask]",
        ticket_context: str = "",
        thinking_chain: Optional[ThinkingChain] = None,
    ) -> "tuple[List[DevelopmentTask], ThinkingChain]":
        """
        Resolve each planner-suggested task to an ACTUAL repository location.
        Returns (localized_tasks, updated_thinking_chain).
        """
        from ticket_to_code.models import TaskType  # avoid circular at module level

        localized = []
        for task in tasks:
            try:
                localized.append(self._localize_via_sqlite(task, TaskType, ticket_context))
            except Exception as exc:
                logger.warning(
                    f"Localization failed for {task.id} ({task.title}): {exc}. "
                    "Keeping planner path."
                )
                task.localization_reason = f"Localization error — keeping planner path: {exc}"
                localized.append(task)

        # ── Emit StageThought ─────────────────────────────────────────────────
        exact_hits = [t for t in localized if t.localization_confidence and t.localization_confidence >= 0.90]
        thought = StageThought(
            stage="localization",
            summary=(
                f"Localized {len(localized)} task(s); {len(exact_hits)} with high confidence (≥0.90). "
                f"Files resolved: {', '.join(t.file_path for t in localized[:4])}."
            ),
            key_decisions=[
                f"[{t.id}] {t.task_type.value}: {t.file_path}  conf={t.localization_confidence:.2f}  — {t.localization_reason[:80] if t.localization_reason else ''}"
                for t in localized[:6]
            ],
            signals_noted=[
                f"exact_path_hits={len(exact_hits)}",
                f"total_tasks={len(localized)}",
            ],
            confidence=0.9 if exact_hits else 0.6,
        )
        updated_chain = (thinking_chain or ThinkingChain()).append(thought)
        return localized, updated_chain

    def _localize_via_sqlite(self, task, TaskType, ticket_context: str = ""):
        """Resolve a planner task to an actual repository file.

        Priority (no special-casing for ticket type or technology):
          1. Planner's exact path exists on disk              → use it directly
          2. Planner's filename found anywhere on disk        → use that file
          3. Planner suggested a backend file (.java/.cs)     → SQLite symbol search
          4. Keyword-based filesystem search                  → universal fallback
          5. Infer CREATE location / keep planner path
        """
        from pathlib import Path as _Path

        logger.info(
            f"\n{'='*60}\n"
            f"🔍 LOCALIZING [{task.id}]: {task.title}\n"
            f"   Planner suggested: {task.file_path} ({task.task_type.value})\n"
            f"{'='*60}"
        )

        # ── STEP 0: Explicit CREATE tasks bypass search ───────────────────────────
        if task.task_type == TaskType.CREATE:
            inferred = self._infer_create_location(task)
            if inferred:
                old_path = task.file_path
                task.file_path = inferred
                task.localization_reason = (
                    f"CREATE task: using inferred repository location: {inferred} "
                    f"(planner had: {old_path})"
                )
                logger.info(f"  📂 CREATE → {inferred} (inferred from repo patterns)")
            else:
                task.localization_reason = f"CREATE task: keeping planner path: {task.file_path}"
                logger.info(f"  📝 CREATE → {task.file_path} (planner path kept)")
            
            task.localization_confidence = 1.0 # High confidence because we are respecting the CREATE intent
            self._resolve_method_targets(task, TaskType)
            return task

        # ── STEP 1: Planner's exact path exists on disk ───────────────────────
        if task.file_path:
            exact = self.workspace_path / task.file_path
            if exact.exists() and exact.is_file():
                logger.info(f"  ✅ EXACT PATH → {task.file_path}  (planner path exists on disk)")
                task.task_type = TaskType.MODIFY
                task.localization_confidence = 0.95
                task.localization_reason = f"Planner path exists on disk: {task.file_path}"
                self._resolve_method_targets(task, TaskType)
                return task

        # ── STEP 2: Search for planner's filename anywhere on disk ───────────
        if task.file_path:
            planner_filename = _Path(task.file_path).name  # e.g. "library-delete.component.scss"
            if planner_filename:
                found = self._find_by_exact_filename(planner_filename)
                if found:
                    old_path = task.file_path
                    task.file_path = found
                    task.task_type = TaskType.MODIFY
                    task.localization_confidence = 0.75  # filename match: correct file, but path was invented
                    task.localization_reason = (
                        f"Planner filename '{planner_filename}' matched on disk: {found} "
                        f"(planner had: {old_path})"
                    )
                    logger.info(f"  ✅ FILENAME MATCH → {found}  (filename: {planner_filename}, confidence: 0.75)")
                    self._resolve_method_targets(task, TaskType)
                    return task

        # ── STEP 3: SQLite symbol index — any indexed language ────────────────
        # NOTE: The _BACKEND_EXTS whitelist has been removed.  When the TS
        # parser is available, STEP 3 also returns TypeScript symbols.
        # Until then, it returns Java/C# as before, but now scored via the
        # unified formula rather than a flat 0.85 constant.
        sqlite_candidates: list[tuple[float, str, str]] = []  # (score, path, class_name)
        if self.sqlite_store:
            keywords = self._extract_keywords(task.title + " " + task.description)
            logger.info(f"  🔑 SQLite symbol search. Keywords: {keywords}")

            for keyword in keywords[:6]:
                try:
                    rows = self.sqlite_store.search_symbols(
                        keyword, kinds=["class", "interface"], limit=10
                    )
                except Exception as exc:
                    logger.debug(f"  search_symbols failed for '{keyword}': {exc}")
                    continue
                for row in rows:
                    row_path = row["path"]
                    if not (self.workspace_path / row_path).exists():
                        continue
                    # Score via unified model (no flat constant)
                    try:
                        content = (self.workspace_path / row_path).read_text(
                            encoding="utf-8", errors="ignore"
                        )
                    except Exception:
                        content = ""
                    keywords_all = self._extract_keywords(task.title + " " + task.description + " " + ticket_context)
                    s_own  = self._compute_ownership(row_path, keywords_all, content)
                    s_path = self._path_ownership_score(row_path, keywords_all)
                    feat = {
                        "s_fs": 0.0, "s_sql": 1.0,
                        "s_sym": 1.0, "s_path": s_path,
                        "s_gr": 0.0, "s_vec": 0.0, "s_own": s_own,
                        # legacy keys
                        "s_ja": 0.0, "s_ts": 0.0,
                    }
                    score = self._unified_score(feat)
                    sqlite_candidates.append((score, row_path, row["name"]))

        # ── STEP 4: Keyword filesystem search — universal fallback ────────────
        _planner_hint = task.file_path or ""
        fs_candidates = self._find_by_filesystem(task, extra_text=ticket_context,
                                                 planner_path_hint=_planner_hint)

        # ── STEP 4.5: Workflow Discovery fallback ────────────────────────────
        # If the planner path and filesystem search miss, reuse workflow graph
        # paths to resolve domain workflows such as Library / Deliverables /
        # Transmittals without introducing a second search system.
        workflow_keywords = self._extract_keywords(f"{task.title} {task.description} {ticket_context}")
        workflow_paths = self._consult_workflow_paths(workflow_keywords)
        if workflow_paths:
            logger.info(f"  🧭 Workflow discovery fallback candidates: {workflow_paths[:5]}")
            for wf_path in workflow_paths:
                if not (self.workspace_path / wf_path).exists():
                    continue
                old_path = task.file_path
                task.file_path = wf_path
                task.task_type = TaskType.MODIFY
                task.localization_confidence = 0.78
                task.localization_reason = (
                    f"Workflow discovery matched {wf_path} (planner had: {old_path})"
                )
                logger.info(f"  ✅ WORKFLOW DISCOVERY → {wf_path}")
                self._resolve_method_targets(task, TaskType)
                return task

        # ── Pick best across STEPs 3 and 4 ───────────────────────────────────
        # Both contribute to the same pool; highest unified score wins.
        # This removes the structural Java advantage (was: SQL=0.85, FS≤0.60).
        all_scored: list[tuple[float, str, str, str]] = []  # (score, path, name, source)
        for score, path, name in sqlite_candidates:
            all_scored.append((score, path, name, "sqlite"))
        if fs_candidates:
            for fc in fs_candidates:
                all_scored.append((fc.confidence, fc.path, "", "filesystem"))

        if all_scored:
            all_scored.sort(key=lambda x: -x[0])

            # ── STEP 4.9: LLM Confirmation ────────────────────────────────────
            # Instead of silently picking the best fuzzy match, ask the LLM to
            # confirm which candidate (if any) is the file it intended to modify.
            old_path = task.file_path
            top_candidates = all_scored[:5]  # send up to 5 candidates
            llm_result = self._llm_confirm_localization(
                task, old_path, top_candidates, ticket_context
            )

            if llm_result and llm_result.get("action") == "USE":
                selected = llm_result.get("selected_file", "")
                llm_reasoning = llm_result.get("reasoning", "")
                # Verify the selected file is actually in our candidate list
                valid_paths = [p for _, p, _, _ in top_candidates]
                if selected in valid_paths:
                    task.file_path = selected
                    task.task_type = TaskType.MODIFY
                    best_score = next((s for s, p, _, _ in top_candidates if p == selected), 0.0)
                    task.localization_confidence = round(min(0.92, best_score), 3)
                    task.localization_reason = (
                        f"LLM confirmed: MODIFY {selected} (planner had: {old_path}). "
                        f"Reasoning: {llm_reasoning}"
                    )
                    logger.info(f"  ✅ LLM CONFIRMED → {selected}  (score: {best_score:.3f}, reason: {llm_reasoning[:80]})")

                    # Store ranked alternatives for retry loop
                    task.file_candidates = [
                        type("FC", (), {"path": p, "confidence": s, "signals": [src]})()
                        for s, p, _, src in all_scored
                    ] if all_scored else []
                    if len(all_scored) > 1:
                        logger.info(
                            f"  📋 {len(all_scored)-1} alternative(s): "
                            + ", ".join(f"{p} ({s:.3f})" for s, p, _, _ in all_scored[1:4])
                        )
                    self._resolve_method_targets(task, TaskType)
                    return task
                else:
                    logger.warning(
                        f"  ⚠️ LLM selected '{selected}' which is not in candidate list. Treating as SKIP."
                    )

            # LLM said SKIP or returned invalid — log and fall through to STEP 5
            skip_reason = (llm_result or {}).get("reasoning", "LLM did not confirm any candidate")
            logger.info(
                f"  🚫 LLM SKIPPED localization for '{old_path}': {skip_reason}"
            )

        # ── STEP 5: No file found — infer CREATE location or keep planner path ─
        inferred = self._infer_create_location(task)
        if inferred:
            old_path = task.file_path
            task.file_path = inferred
            task.localization_reason = (
                f"No existing file found. CREATE at inferred location: {inferred} "
                f"(planner had: {old_path})"
            )
            logger.info(f"  📂 CREATE → {inferred} (inferred from repo patterns)")
        else:
            task.localization_reason = (
                f"No existing file found. CREATE at planner-suggested path: {task.file_path}"
            )
            logger.info(f"  📝 CREATE → {task.file_path} (planner path kept)")
        task.localization_confidence = 0.0
        self._resolve_method_targets(task, TaskType)
        return task


    def _llm_confirm_localization(self, task, planner_path: str,
                                   candidates: list, ticket_context: str) -> dict:
        """Ask the LLM to confirm which candidate file (if any) matches the
        planner's intended MODIFY target.

        Returns a dict with:
          {"action": "USE", "selected_file": "...", "reasoning": "..."}
        or
          {"action": "SKIP", "reasoning": "..."}
        or None on error.
        """
        candidates_str = "\n".join(
            f"  {i+1}. {path}  (score: {score:.3f}, source: {src})"
            for i, (score, path, _name, src) in enumerate(candidates)
        )

        prompt = f"""You are a localization verification agent.

The planner proposed to MODIFY the file: "{planner_path}"
But this file does NOT exist in the repository.

TASK CONTEXT:
  Title: {task.title}
  Description: {task.description}

TICKET CONTEXT:
{ticket_context[:1500] if ticket_context else '(none)'}

The following candidate files were found via fuzzy search:
{candidates_str}

YOUR JOB:
Determine if any of these candidates is the file the planner actually intended to modify.
- If YES, select the correct file.
- If NO candidate is a reasonable match (the planner may have hallucinated or the task requires a NEW file), respond with SKIP.

Output JSON only:
{{
  "action": "USE" | "SKIP",
  "selected_file": "exact path from the candidate list (only if action=USE)",
  "reasoning": "Brief explanation of why this file is or isn't the intended target"
}}"""

        try:
            from ticket_to_code.plan_gating import _extract_json_payload
            response_text = _extract_json_payload(llm_invoke(self.llm, prompt).content)
            result = json.loads(response_text)
            logger.info(
                f"  🤖 LLM Localization Confirmation: action={result.get('action')}, "
                f"file={result.get('selected_file', 'N/A')}, "
                f"reasoning={result.get('reasoning', '')[:100]}"
            )
            return result
        except Exception as exc:
            logger.warning(f"  ⚠️ LLM localization confirmation failed: {exc}. Treating as SKIP.")
            return {"action": "SKIP", "reasoning": f"LLM call failed: {exc}"}

    def _resolve_method_targets(self, task, TaskType) -> None:
        """
        After file localization, scan the resolved file to find the exact
        method(s) to modify.  Populates:
          task.target_class     — dominant class name in the file
          task.target_method    — best matching method name
          task.allowed_methods  — all methods the generator is permitted to touch
          task.new_file_creation_allowed — False for modify, True only for create

        Uses regex AST (no external parser dependency) — works for Java, TypeScript,
        Python, C#, JavaScript/Angular.
        """
        # CREATE tasks: no method targeting needed — generator writes the whole new file
        if task.task_type == TaskType.CREATE:
            task.new_file_creation_allowed = True
            return

        task.new_file_creation_allowed = False  # hard lock for modify tasks

        file_path = self.workspace_path / task.file_path
        if not file_path.exists():
            logger.warning(f"  ⚠️  Cannot resolve methods — file not found: {file_path}")
            return

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.warning(f"  ⚠️  Cannot read file for method scan: {exc}")
            return

        # ── Extract class name ────────────────────────────────────────────
        # Supports Java/C# (class/interface Foo), TypeScript (class Foo / export class Foo),
        # Python (class Foo)
        class_match = re.search(
            r"(?:export\s+)?(?:abstract\s+)?class\s+(\w+)|interface\s+(\w+)",
            content
        )
        if class_match:
            task.target_class = class_match.group(1) or class_match.group(2)

        # ── Extract all method signatures in the file ─────────────────────
        # Pattern covers: Java/C# public/private/protected methods, TypeScript methods,
        # Angular component methods, Python defs
        method_pattern = re.compile(
            r"(?:(?:public|private|protected|static|async|override|abstract)\s+)*"
            r"(?:[\w<>\[\]]+\s+)?"          # return type (Java/C#) — optional
            r"(\w+)\s*\("                    # method name + open paren
            r"[^)]*\)\s*(?::\s*[\w<>\[\]|]+)?"  # params + optional TS return type
            r"\s*(?:\{|=>|\{)",              # opening brace or arrow
            re.MULTILINE
        )
        python_def_pattern = re.compile(r"^\s*def\s+(\w+)\s*\(", re.MULTILINE)

        found_methods: list[str] = []
        for m in method_pattern.finditer(content):
            name = m.group(1)
            # Filter out language keywords that match the pattern
            if name not in {"if", "for", "while", "switch", "catch", "try", "else", "return",
                            "new", "class", "interface", "constructor", "function"}:
                found_methods.append(name)
        for m in python_def_pattern.finditer(content):
            name = m.group(1)
            if name not in found_methods:
                found_methods.append(name)

        if not found_methods:
            logger.info(f"  ℹ️  No method signatures found in {task.file_path}")
            return

        logger.info(f"  🔬 Methods found in file ({len(found_methods)}): {found_methods[:10]}")

        # ── Match best method(s) against task keywords ───────────────────
        keywords = self._extract_keywords(task.title + " " + task.description)
        kw_lower = {k.lower() for k in keywords}

        scored: list[tuple[int, str]] = []
        for method in found_methods:
            # Score: count how many keywords appear in the method name (camel-split)
            method_tokens = {t.lower() for t in re.findall(r"[A-Z][a-z]+|[a-z]+", method)}
            score = len(method_tokens & kw_lower)
            if score > 0:
                scored.append((score, method))

        scored.sort(key=lambda x: -x[0])

        # Also honour planner-supplied allowed_methods (including new methods to be added)
        planner_allowed = list(task.allowed_methods or [])

        if scored:
            # If planner specified allowed_methods, prioritize planner's targets
            if planner_allowed:
                best_method = planner_allowed[0]
                task.target_method = best_method
                # Combine planner methods with top keyword matches, deduplicating
                combined = list(dict.fromkeys(planner_allowed + [m for _, m in scored[:3]]))
                task.allowed_methods = combined
            else:
                best_method = scored[0][1]
                task.target_method = best_method
                task.allowed_methods = [m for _, m in scored[:3]]
            logger.info(
                f"  🎯 Method targets resolved: target={best_method}, "
                f"allowed={task.allowed_methods}"
            )
        elif planner_allowed:
            # Trust planner's allowed_methods if keyword search found nothing
            task.target_method = planner_allowed[0]
            task.allowed_methods = planner_allowed
            logger.info(f"  🎯 Using planner method targets: {task.allowed_methods}")
        else:
            # No match — generator must figure it out from context; leave allowed_methods empty
            # (PatchValidator will be lenient when allowed_methods is empty)
            logger.info(f"  ℹ️  No keyword-matched methods — generator has full file scope")

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful tokens from ticket text.

        Extracts whole word-boundary tokens and constituent CamelCase/PascalCase
        sub-tokens to resolve concatenated identifiers (e.g. BackButtonAddProject
        -> back, button, project).
        """
        stop_words = {
            "the", "a", "an", "in", "of", "for", "to", "and", "or", "is",
            "are", "was", "be", "file", "add", "update", "create", "modify",
            "new", "with", "from", "that", "this", "should", "must", "code",
            "src", "test", "impl", "have", "view", "message",
            "when", "then", "step", "steps", "seen", "result", "results",
            "able", "verify",
            "observed", "expected", "criteria", "acceptance",
            "git", "commit", "merge", "branch",
        }
        raw_tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*\b", text)
        expanded_tokens = []
        for t in raw_tokens:
            expanded_tokens.append(t)
            s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", t)
            s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
            parts = s.split()
            if len(parts) > 1:
                expanded_tokens.extend(parts)

        seen, result = set(), []
        for t in expanded_tokens:
            tl = t.lower()
            if len(tl) >= 3 and tl not in stop_words and tl not in seen:
                seen.add(tl)
                result.append(tl)
        return result

    def _find_by_exact_filename(self, filename: str) -> Optional[str]:
        """Walk the workspace and return the relative path of the first file
        whose name matches exactly. Skips non-source directories."""
        for root, dirs, files in os.walk(str(self.workspace_path)):
            prune_walk_dirs(dirs, root, self.workspace_path)
            dirs.sort()
            files.sort()
            if filename in files:
                full = Path(root) / filename
                if is_ignored_path(full, workspace_path=self.workspace_path):
                    continue
                try:
                    rel = full.relative_to(self.workspace_path)
                    return str(rel).replace("\\", "/")
                except ValueError:
                    pass
        return None

    def _find_by_filesystem(self, task, extra_text: str = "", ts_only: bool = False,
                             planner_path_hint: str = "",
                             top_n: int = 3) -> list:
        """
        Filesystem fallback for languages not indexed by SQLite.

        Returns a ranked list of up to `top_n` LocalizationCandidate objects
        instead of a single winner — callers pick [0] as primary and can retry
        with subsequent candidates if validation fails.

        Scoring signals (all generic, no file/module names hardcoded):
        - keyword_path:     ticket keywords found in file path
        - planner_segments: path tokens from planner hint found in file path
        - extension_match:  file extension matches planner-suggested extension
        - keyword_content:  ticket keywords found in file content

        extra_text: additional keyword source (ticket title+description).
        planner_path_hint: planner-suggested path; segments used for bonus scoring
            and extension-locking.
        ts_only: when True, exclude Java files (legacy param, kept for compatibility).
        top_n: number of candidates to return.
        """
        from ticket_to_code.models import LocalizationCandidate
        SKIP_DIRS = {".git", "node_modules", "target", ".aviator", "dist", "build", ".idea", "__pycache__"}
        if ts_only:
            SOURCE_EXTS = {".ts", ".tsx", ".html", ".scss", ".js", ".jsx", ".vue"}
        else:
            SOURCE_EXTS = {".ts", ".js", ".jsx", ".tsx", ".vue", ".java", ".cs", ".py", ".html", ".scss"}

        # If the planner's path has a recognizable extension, restrict the search
        # to that extension only — prevents e.g. a .ts file winning over an .scss task.
        if planner_path_hint:
            _hint_ext = Path(planner_path_hint.replace("\\", "/")).suffix.lower()
            if _hint_ext in SOURCE_EXTS:
                SOURCE_EXTS = {_hint_ext}
                logger.debug(f"  Extension-locked search to '{_hint_ext}' (from planner hint)")

        # Combine task keywords + ticket keywords; ticket keywords are most specific
        combined_text = task.title + " " + task.description + " " + extra_text
        keywords = self._extract_keywords(combined_text)
        if not keywords:
            return None

        # Planner path segments used for module-aware bonus scoring
        # e.g. planner said "modules/library/components/..." → tokens = ["library", "components"]
        _planner_segments: list[str] = []
        if planner_path_hint:
            import posixpath as _pp
            _hint_lower = planner_path_hint.replace("\\", "/").lower()
            _planner_segments = [p for p in _hint_lower.split("/") if len(p) > 2]

        # Phase 1: collect all files with any path-keyword match
        # (score_path, full_path, rel_path)
        path_candidates: list[tuple[int, Path, str]] = []
        for root, dirs, files in os.walk(self.workspace_path):
            prune_walk_dirs(dirs, root, self.workspace_path)
            dirs.sort()
            files.sort()
            for fname in files:
                full = Path(root) / fname
                if full.suffix.lower() not in SOURCE_EXTS:
                    continue
                if is_ignored_path(full, workspace_path=self.workspace_path):
                    continue
                rel = str(full.relative_to(self.workspace_path)).replace("\\", "/")
                rel_lower = rel.lower()
                score = sum(1 for kw in keywords if kw.lower() in rel_lower)
                # Bonus: shared path segments with planner's intended path
                if _planner_segments:
                    score += sum(1 for seg in _planner_segments if seg in rel_lower)
                if score > 0:
                    path_candidates.append((score, full, rel))

        if path_candidates:
            # Phase 2: boost score with content matches on top candidates
            # Only scan top-20 by path score to keep it fast
            path_candidates.sort(key=lambda x: (-x[0], x[2]))
            top_candidates = path_candidates[:20]
            top_kws = keywords[:6]  # use up to 6 keywords for content scoring

            combined: list[tuple[int, str, list[str]]] = []  # (score, rel_path, signals)
            for path_score, full, rel in top_candidates:
                signals = ["keyword_path"]
                rel_lower = rel.lower()
                if _planner_segments and any(seg in rel_lower for seg in _planner_segments):
                    signals.append("planner_segments")
                if planner_path_hint and Path(planner_path_hint.replace("\\", "/")).suffix.lower() == full.suffix.lower():
                    signals.append("extension_match")
                try:
                    text = full.read_text(encoding="utf-8", errors="ignore")
                    content_score = sum(1 for kw in top_kws if kw.lower() in text.lower())
                    if content_score > 0:
                        signals.append("keyword_content")
                    combined.append((path_score + content_score, rel, signals))
                except Exception:
                    combined.append((path_score, rel, signals))

            combined.sort(key=lambda x: -x[0])
            max_score = combined[0][0] if combined else 1

            candidates = []
            keywords_all = self._extract_keywords(combined_text)
            for score, rel, signals in combined[:top_n]:
                # Unified score for filesystem candidates (no SQLite/graph/vec)
                s_fs = score / max(max_score, 1)
                try:
                    content = (self.workspace_path / rel).read_text(
                        encoding="utf-8", errors="ignore"
                    )
                except Exception:
                    content = ""
                s_own = self._compute_ownership(rel, keywords_all, content)
                feat = {
                    "s_fs": s_fs, "s_sql": 0.0,
                    "s_ja": 0.0,  "s_ts": 0.0,
                    "s_gr": 0.0,  "s_vec": 0.0, "s_own": s_own,
                }
                unified = round(min(0.95, self._unified_score(feat)), 3)
                candidates.append(LocalizationCandidate(
                    path=rel,
                    confidence=unified,
                    signals=signals,
                    raw_score=float(score),
                ))

            if candidates:
                logger.info(
                    f"  🗂️  Filesystem match: {candidates[0].path} "
                    f"(combined score={candidates[0].raw_score})"
                )
            return candidates

        # Phase 3: No path matches at all — full content scan
        logger.info("  🔎 No path match; scanning file contents (top keywords)...")
        top_kws = keywords[:4]
        content_matches: list[tuple[int, str]] = []
        for root, dirs, files in os.walk(self.workspace_path):
            prune_walk_dirs(dirs, root, self.workspace_path)
            dirs.sort()
            files.sort()
            for fname in files:
                full = Path(root) / fname
                if full.suffix.lower() not in SOURCE_EXTS:
                    continue
                if is_ignored_path(full, workspace_path=self.workspace_path):
                    continue
                try:
                    text = full.read_text(encoding="utf-8", errors="ignore")
                    score = sum(1 for kw in top_kws if kw.lower() in text.lower())
                    if score >= 2:  # require at least 2 keyword matches for content-only
                        rel = str(full.relative_to(self.workspace_path)).replace("\\", "/")
                        content_matches.append((score, rel))
                except Exception:
                    continue

        if content_matches:
            content_matches.sort(key=lambda x: -x[0])
            max_score = content_matches[0][0]
            candidates = []
            keywords_all = self._extract_keywords(
                task.title + " " + task.description + " " + extra_text
            )
            for score, rel in content_matches[:top_n]:
                s_fs = score / max(max_score, 1)
                try:
                    content = (self.workspace_path / rel).read_text(
                        encoding="utf-8", errors="ignore"
                    )
                except Exception:
                    content = ""
                s_own = self._compute_ownership(rel, keywords_all, content)
                feat = {
                    "s_fs": s_fs, "s_sql": 0.0,
                    "s_ja": 0.0,  "s_ts": 0.0,
                    "s_gr": 0.0,  "s_vec": 0.0, "s_own": s_own,
                }
                unified = round(min(0.95, self._unified_score(feat)), 3)
                candidates.append(LocalizationCandidate(
                    path=rel, confidence=unified,
                    signals=["keyword_content"], raw_score=float(score)
                ))
            logger.info(f"  🗂️  Filesystem content-only match: {candidates[0].path} (score={content_matches[0][0]})")
            return candidates

        return []

    def _infer_create_location(self, task) -> Optional[str]:
        """
        Infer the best CREATE location using Repository Brain directory knowledge.

        Returns a repository-relative path (for example: src/app/service/NewFile.py).
        Returns None when localization cannot be trusted so planner path is preserved.
        """
        primary_brain_path = (
            self.workspace_path
            / "brain"
            / "knowledge"
            / "generated_directory_brain.json"
        )
        legacy_brain_path = self.workspace_path / ".agents" / "repository_brain.json"

        if not primary_brain_path.exists():
            try:
                from ticket_to_code.brain.generate_repository_brain import ensure_repository_brain
                ensure_repository_brain(str(self.workspace_path), llm=self.llm)
            except Exception as exc:
                logger.warning("Repository Brain bootstrap failed during CREATE localization (%s)", exc)

        brain_path = primary_brain_path if primary_brain_path.exists() else legacy_brain_path
        if not brain_path.exists():
            logger.warning(
                "Repository Brain file not found at %s or %s; preserving planner path for %s",
                primary_brain_path,
                legacy_brain_path,
                getattr(task, "file_path", "(unknown)"),
            )
            return None

        try:
            with brain_path.open("r", encoding="utf-8") as handle:
                brain_entries = json.load(handle)
        except Exception as exc:
            logger.warning(
                "Failed to load Repository Brain from %s; preserving planner path for %s (%s)",
                brain_path,
                getattr(task, "file_path", "(unknown)"),
                exc,
            )
            return None

        if not isinstance(brain_entries, list) or not brain_entries:
            logger.warning(
                "Repository Brain file %s did not contain usable directory entries; preserving planner path for %s",
                brain_path,
                getattr(task, "file_path", "(unknown)"),
            )
            return None

        directory_lines: list[str] = []
        for entry in brain_entries:
            if not isinstance(entry, dict):
                continue
            directory_path = str(entry.get("directory_path", "")).strip()
            if not directory_path:
                continue
            business_domain = str(entry.get("business_domain", "")).strip() or "(unknown)"
            technical_role = str(entry.get("technical_role", "")).strip() or "(unknown)"
            directory_lines.append(
                f"- directory_path: {directory_path} | business_domain: {business_domain} | technical_role: {technical_role}"
            )

        if not directory_lines:
            logger.warning(
                "Repository Brain file %s had no valid directory entries; preserving planner path for %s",
                brain_path,
                getattr(task, "file_path", "(unknown)"),
            )
            return None

        proposed_file = Path(str(getattr(task, "file_path", ""))).name.strip()
        if not proposed_file:
            logger.warning(
                "CREATE task has no resolvable filename in file_path=%s; preserving planner path",
                getattr(task, "file_path", "(unknown)"),
            )
            return None

        task_title = getattr(task, "title", "") or "(none)"
        task_description = getattr(task, "description", "") or "(none)"
        planner_justification = getattr(task, "explicit_planner_justification", None) or "(none)"
        original_path = str(getattr(task, "file_path", "") or "").replace("\\", "/")

        workspace_norm = str(self.workspace_path).replace("\\", "/").rstrip("/")

        def _sanitize_selected_directory(raw_directory: str) -> Optional[str]:
            normalized = str(raw_directory or "").replace("\\", "/").strip().strip('"').strip("'")
            if not normalized:
                return None

            if normalized.lower() in {
                "/",
                ".",
                "root",
                "repo-root",
                "repository-root",
                "repository root",
            }:
                return "/"

            if re.match(r"^[A-Za-z]:/", normalized):
                low_norm = normalized.lower()
                low_workspace = workspace_norm.lower()
                if low_norm == low_workspace:
                    return "/"
                if low_norm.startswith(low_workspace + "/"):
                    normalized = normalized[len(workspace_norm):].lstrip("/")
                else:
                    return None
            elif normalized.startswith("/"):
                normalized = normalized.lstrip("/")

            while normalized.startswith("./"):
                normalized = normalized[2:]

            parts = [part for part in normalized.split("/") if part and part != "."]
            if any(part == ".." for part in parts):
                return None

            if not parts:
                return "/"

            return "/".join(parts)

        prompt = f"""You are localizing a CREATE task in an enterprise codebase.

TASK TITLE:
{task_title}

TASK DESCRIPTION:
{task_description}

PLANNER JUSTIFICATION:
{planner_justification}

PROPOSED FILENAME:
{proposed_file}

REPOSITORY BRAIN DIRECTORY CANDIDATES:
{os.linesep.join(directory_lines)}

RULES:
- Infrastructure and configuration files MUST stay at the repository root unless the Repository Brain explicitly indicates otherwise.
- Examples of root-level infrastructure/config files include Dockerfile, docker-compose.yml, *.json, *.yml, *.yaml, *.toml, *.xml, *.lock, .env, and similar build/deployment artifacts.
- Source code files should be placed according to the Repository Brain's architectural ownership.
- Choose the single best directory path for the proposed file.

Return JSON only with exactly one key:
{{
  "selected_directory": "<directory path or />"
}}

If the best location is the repository root, return "/".
If the planner proposed a pure infrastructure/config root artifact (for example docker-compose.yml, package.json, pom.xml, *.yml, *.yaml, *.json, *.toml, *.xml, *.lock, .env), prefer "/".
Do not include markdown, code fences, or extra keys.
"""

        try:
            response = llm_invoke(self.llm, prompt)
            response_text = _extract_json_payload(getattr(response, "content", str(response)))
            payload = json.loads(response_text)
        except Exception as exc:
            logger.warning(
                "Create-location LLM selection failed for %s; preserving planner path (%s)",
                getattr(task, "file_path", "(unknown)"),
                exc,
            )
            return None

        if not isinstance(payload, dict) or set(payload.keys()) != {"selected_directory"}:
            logger.warning(
                "Create-location LLM returned invalid JSON schema for %s: keys=%s",
                getattr(task, "file_path", "(unknown)"),
                list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__,
            )
            return None

        selected_directory = str(payload.get("selected_directory", "")).strip()
        if not selected_directory:
            logger.warning(
                "Create-location LLM returned no selected_directory for %s; preserving planner path",
                getattr(task, "file_path", "(unknown)"),
            )
            return None

        normalized_directory = _sanitize_selected_directory(selected_directory)
        if normalized_directory is None:
            logger.warning(
                "Create-location LLM returned unsafe/invalid directory '%s' for %s; preserving planner path",
                selected_directory,
                getattr(task, "file_path", "(unknown)"),
            )
            return None

        if normalized_directory == "/":
            # Keep root-level placement deterministic for infrastructure/config artifacts.
            return proposed_file

        # If LLM selected the planner's existing parent directory, keep original path shape.
        original_parent = Path(original_path).parent.as_posix() if original_path else ""
        if original_parent in {"", "."}:
            original_parent = ""
        if original_parent and normalized_directory == original_parent:
            return original_path

        return f"{normalized_directory}/{proposed_file}"

    def close(self):
        """Close storage connections"""
        if self.sqlite_store:
            self.sqlite_store.close()
        if self.neo4j_store:
            self.neo4j_store.close()


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def localize_targets(
    requirements: StructuredRequirements,
    workspace_path: str,
    investigation_context: Optional[Dict] = None
) -> LocalizationResult:
    """
    Convenience function to localize targets.
    
    Args:
        requirements: Structured requirements
        workspace_path: Project workspace path
        investigation_context: Optional investigation context
        
    Returns:
        Localization result with exact targets
    """
    agent = LocalizationAgent(workspace_path)
    try:
        return agent.localize_targets(requirements, investigation_context)
    finally:
        agent.close()
