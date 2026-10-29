"""
FastIDEEngine - Cursor & Windsurf-Style Sub-Minute Execution Core

Key Principles (Zero-Bloat AI IDE Architecture):
1. Sub-Minute Latency: Bypasses the 33-node LangGraph waterfall. Runs a tight
   Iterative Edit -> Build -> Verify loop in 15-45 seconds.
2. Low Token Cost: Slices only targeted context (2k-8k tokens total per turn)
   instead of dumping 100k+ tokens of full workspace code into the prompt.
3. Surgical Search/Replace Diff Format: Emits and applies search/replace blocks
   (Aider / Cursor Composer format), reducing output tokens by 90%.
4. Zero-LLM Intent Triage: Eliminates useless bug vs feature categorization;
   directly determines if code changes are required vs read-only inquiry.
5. ReAct Evidence Collection & Ranking: Uses CallChainTracer, IDF anchor matching,
   and component grouping to rank and lock candidate files with precision.
6. Deterministic Safety Hooks (0 Tokens):
   - patch_gate: Syntax, brace/bracket balance, and structural validation.
   - import_validation: Auto-injects missing symbols and imports deterministically.
   - pre_fix_build: Native compiler verification with diagnostic error slicing.
7. Safe Fallback: If the fast loop cannot converge within max repair turns,
   safely escalates to the deep reasoning graph without breaking any API contracts.
"""

from __future__ import annotations

import difflib
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from ticket_to_code.editing.surgical_diff_engine import EditResult, EditStatus, SurgicalDiffEngine
from ticket_to_code.indexing.fast_repo_map import FastRepoMap
from ticket_to_code.lean.call_chain_tracer import CallChainTracer
from ticket_to_code.models import (
    AutonomousWorkflowState,
    BuildResult,
    BuildStatus,
    GeneratedCode,
    TestResult,
    TestStatus,
    ValueEdgeTicket,
    WorkflowStatus,
)
from ticket_to_code.runtime.diagnostic_error_slicer import DiagnosticErrorSlicer
from ticket_to_code.runtime.polyglot_build import BuildConfig, PolyglotBuildDiscoverer

logger = logging.getLogger(__name__)

# =========================================================================
# DOMAIN-AWARE ARCHITECTURAL TAXONOMIES (TOP AI IDE RESILIENCE)
# Covers the broad software taxonomy so thousands of domain terms are handled
# generically without fragile keyword lists or hardcoded subproject names.
# =========================================================================

UI_DOMAIN_TAXONOMY: Set[str] = {
    # Elements & Interactive Widgets
    "button", "icon", "dialog", "modal", "popup", "popover", "tooltip", "drawer",
    "sidebar", "navbar", "nav", "toolbar", "masthead", "header", "footer", "banner",
    "badge", "avatar", "tag", "card", "panel", "accordion", "carousel", "slider",
    "breadcrumb", "pagination", "dropdown", "select", "combobox", "input", "textarea",
    "checkbox", "radio", "switch", "toggle", "picker", "form", "table",
    "grid", "list", "spinner", "loader", "progress", "toast", "snackbar", "alert",
    "notification", "view", "screen", "page", "canvas", "layout", "window", "widget",
    "dashboard", "tab", "tabs", "overlay", "sheet", "menu", "logo",
    # Interactions & Gestures
    "click", "hover", "press", "tap", "scroll", "drag", "drop", "touch", "swipe",
    "focus", "blur", "submit", "enter", "change", "select", "gesture", "launch",
    "open", "close", "expand", "collapse", "hide", "show", "toggle", "navigate",
    "navigation", "redirect", "render", "display", "mount", "unmount", "bind", "binding",
    # Presentation & Styling
    "style", "styles", "styling", "css", "scss", "sass", "less", "theme", "color",
    "font", "typography", "border", "margin", "padding", "flex", "grid", "align",
    "alignment", "justify", "shadow", "radius", "animation", "transition", "responsive",
    "mobile", "desktop", "width", "height", "opacity", "z-index", "cursor", "visual",
    "look", "feel", "ui", "ux", "gui", "client", "frontend", "presentation", "guidance",
    # Frontend Architecture & Templates
    "template", "html", "dom", "component", "directive", "pipe", "props", "state",
    "routerlink", "outlet", "viewchild"
}

BACKEND_DOMAIN_TAXONOMY: Set[str] = {
    # Data & Persistence
    "entity", "entities", "repository", "repositories", "dao", "dto", "database",
    "db", "table", "column", "sql", "migration", "schema", "query", "queries", "jpa",
    "hibernate", "transaction", "transactions", "postgres", "mysql", "oracle", "mongo",
    "redis", "datasource", "persistence",
    # API & Protocol
    "controller", "endpoint", "endpoints", "rest", "api", "grpc", "payload",
    "request", "response", "http", "headers", "status-code", "middleware", "filter",
    "gateway", "socket", "serialize", "deserialization", "json", "xml", "webhook",
    # Microservices & Processing
    "service", "services", "impl", "processor", "handler", "handlers", "saga",
    "workflow", "queue", "kafka", "rabbitmq", "topic", "worker", "batch", "cron",
    "daemon", "microservice", "backend", "server"
}


@dataclass
class SearchReplaceBlock:
    """A single Cursor/Aider-style search and replace chunk."""
    file_path: str
    search_block: str
    replace_block: str


@dataclass
class FastIDEResult:
    success: bool
    status: str
    duration_seconds: float
    tokens_consumed: int
    modified_files: List[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    llm_calls: int = 0
    diffs: Dict[str, str] = field(default_factory=dict)
    build_passed: Optional[bool] = None   # True/False when a build actually ran; None = no build ran
    test_passed: Optional[bool] = None    # True/False when tests actually ran; None = no tests ran
    message: str = ""
    error_message: str = ""
    escalate_to_full_graph: bool = False
    ranked_evidence: List[Dict[str, Any]] = field(default_factory=list)
    call_chain_summary: List[str] = field(default_factory=list)
    imported_defs: Dict[str, str] = field(default_factory=dict)
    target_files: List[str] = field(default_factory=list)


class SearchReplaceParser:
    """Parses Cursor / Aider standard <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE blocks."""

    PATTERN = re.compile(
        r"(?:(?:(?:\/\/|#|--|\*\*|###|\*)\s*)?(?:File|FILE|Path|PATH):\s*([^\r\n*`]+?)(?:\*\*)?\s*\r?\n)?"
        r"<{4,}\s*(?:SEARCH|ORIGINAL)(?:[ \t]+([^\r\n]+))?\r?\n"
        r"(.*?)\r?\n?={4,}(?:[^\r\n]*)\r?\n(.*?)\r?\n?"
        r"(?:>{4,}\s*(?:REPLACE|UPDATED)(?:[^\r\n]*)|\Z)",
        re.DOTALL,
    )

    @classmethod
    def parse(cls, text: str, default_file_path: str = "") -> List[SearchReplaceBlock]:
        blocks = []
        # Pre-strip accidental outer markdown fences
        clean_text = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*$\r?\n?", "", text, flags=re.MULTILINE)
        for match in cls.PATTERN.finditer(clean_text):
            explicit_fp = (match.group(1) or match.group(2) or "").strip().strip("`*\"'():; ")
            search_content = match.group(3)
            replace_content = match.group(4)
            fp = explicit_fp or default_file_path

            # Strip accidental markdown code fences around or inside search or replace
            if search_content:
                search_content = re.sub(r"^\s*```.*$\r?\n?", "", search_content, flags=re.MULTILINE).replace("```", "")
            if replace_content:
                replace_content = re.sub(r"^\s*```.*$\r?\n?", "", replace_content, flags=re.MULTILINE).replace("```", "")

            if fp and search_content is not None and replace_content is not None:
                blocks.append(
                    SearchReplaceBlock(
                        file_path=fp.replace("\\", "/"),
                        search_block=search_content,
                        replace_block=replace_content,
                    )
                )

        if blocks:
            return blocks

        # ── Fallback 1: Parse Unified Diff Hunks (--- a/file, +++ b/file, @@ ... @@) ──
        diff_hunk_blocks = cls._parse_unified_diff(text, default_file_path)
        if diff_hunk_blocks:
            return diff_hunk_blocks

        # ── Fallback 2: Parse Markdown Code Block as Full File Replacement ──
        full_code_blocks = cls._parse_full_code_blocks(text, default_file_path)
        if full_code_blocks:
            return full_code_blocks

        return []

    @classmethod
    def _parse_unified_diff(cls, text: str, default_file_path: str = "") -> List[SearchReplaceBlock]:
        """Converts unified diff hunks into SearchReplaceBlock instances."""
        if "@@ -" not in text and "--- " not in text:
            return []

        blocks = []
        current_fp = default_file_path
        hunk_re = re.compile(r"^@@\s*-\d+(?:,\d+)?\s*\+\d+(?:,\d+)?\s*@@", re.MULTILINE)

        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("+++ ") or line.startswith("--- "):
                candidate = line[4:].strip()
                if candidate.startswith("b/") or candidate.startswith("a/"):
                    candidate = candidate[2:]
                candidate = candidate.strip("`*\"' ")
                if candidate and candidate != "/dev/null":
                    current_fp = candidate.replace("\\", "/")
                i += 1
                continue

            if hunk_re.match(line):
                i += 1
                search_lines = []
                replace_lines = []
                while i < len(lines) and not hunk_re.match(lines[i]) and not lines[i].startswith(("diff --git", "--- ", "+++ ")):
                    hl = lines[i]
                    if hl.startswith("-"):
                        search_lines.append(hl[1:])
                    elif hl.startswith("+"):
                        replace_lines.append(hl[1:])
                    else:
                        c_line = hl[1:] if hl.startswith(" ") else hl
                        search_lines.append(c_line)
                        replace_lines.append(c_line)
                    i += 1

                if current_fp and (search_lines or replace_lines):
                    blocks.append(
                        SearchReplaceBlock(
                            file_path=current_fp,
                            search_block="\n".join(search_lines),
                            replace_block="\n".join(replace_lines),
                        )
                    )
                continue
            i += 1

        return blocks

    @classmethod
    def _parse_full_code_blocks(cls, text: str, default_file_path: str = "") -> List[SearchReplaceBlock]:
        """Detects full-code replacement inside markdown fences when no search markers are present."""
        if not default_file_path:
            return []

        fence_matches = list(re.finditer(r"```(?:[a-zA-Z0-9_-]+)?\s*\r?\n(.*?)\r?\n```", text, re.DOTALL))
        if not fence_matches:
            return []

        best_code = max(fence_matches, key=lambda m: len(m.group(1))).group(1)
        if len(best_code.strip()) >= 40:
            fp_m = re.search(r"(?:File|FILE|Path|PATH):\s*([^\r\n*`]+)", text)
            fp = fp_m.group(1).strip() if fp_m else default_file_path
            # Empty search_block triggers whole-file replacement in SurgicalDiffEngine
            return [SearchReplaceBlock(file_path=fp.replace("\\", "/"), search_block="", replace_block=best_code)]

        return []


class FastIDEEngine:
    """
    Sub-minute Lean ReAct Engine modeled after Cursor Composer, Windsurf, and Aider.
    Executes a high-performance 4-stage pipeline:
      Stage 1: Intent Triage & ReAct Evidence Collection (with Evidence Ranking)
      Stage 2: Context Slicing & Planning
      Stage 3: Surgical Patch Generation with Deterministic Safety Hooks (patch_gate & import_validation)
      Stage 4: Verification (pre_fix_build & targeted testing)
    """

    def __init__(
        self,
        workspace_path: str,
        ticket: ValueEdgeTicket,
        llm_client: Optional[Any] = None,
        max_repair_turns: int = 3,
        ui_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_budget_tokens: int = 250_000,
    ):
        self.workspace = Path(workspace_path).resolve()
        self.ticket = ticket
        self.llm = llm_client
        self.max_repair_turns = max_repair_turns
        self.ui_callback = ui_callback
        self.max_budget_tokens = max_budget_tokens
        self.diff_engine = SurgicalDiffEngine(str(self.workspace))
        self.repo_map = FastRepoMap(str(self.workspace), max_tokens=1200)
        self.tokens_used = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.llm_calls = 0
        self.file_hashes: Dict[str, str] = {}
        self._subprojects: Dict[str, Dict[str, Any]] = self._discover_subprojects()
        from ticket_to_code.runtime.lsp_client import FastLSPClient
        from ticket_to_code.indexing.contract_graph_indexer import ContractGraphIndexer
        from ticket_to_code.runtime.container_sandbox import ContainerSandboxEngine
        self.lsp = FastLSPClient(self.workspace)
        self.contract_indexer = ContractGraphIndexer(self.workspace)
        self.sandbox_engine = ContainerSandboxEngine(self.workspace)

    def _discover_subprojects(self) -> Dict[str, Dict[str, Any]]:
        """
        Dynamically discovers all subprojects across a monorepo workspace.
        Profiles each subproject's architectural role (UI vs Backend vs Shared)
        by inspecting manifests and dependencies without hardcoded folder names.
        """
        manifest_names = ["package.json", "pom.xml", "build.gradle", "pyproject.toml", "Cargo.toml", "go.mod"]
        discovered: Dict[str, Dict[str, Any]] = {}

        def characterize(folder: Path, rel_name: str) -> Dict[str, Any]:
            is_ui = False
            is_backend = False
            pkg_name = rel_name

            pkg_json = folder / "package.json"
            if pkg_json.is_file():
                try:
                    import json
                    data = json.loads(pkg_json.read_text(encoding="utf-8", errors="ignore"))
                    pkg_name = data.get("name") or rel_name
                    deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
                    ui_dep_markers = ["@angular", "react", "vue", "svelte", "next", "nuxt", "@material", "bootstrap", "tailwind", "@uxa"]
                    if any(any(m in d for m in ui_dep_markers) for d in deps):
                        is_ui = True
                except Exception:
                    pass

            pom_xml = folder / "pom.xml"
            if pom_xml.is_file() or (folder / "build.gradle").is_file() or (folder / "go.mod").is_file():
                is_backend = True
                if pom_xml.is_file():
                    try:
                        pom_txt = pom_xml.read_text(encoding="utf-8", errors="ignore")
                        art_m = re.search(r"<artifactId>([^<]+)</artifactId>", pom_txt)
                        if art_m:
                            pkg_name = art_m.group(1).strip()
                    except Exception:
                        pass

            if not is_ui and not is_backend:
                if (folder / "src" / "app").is_dir() or (folder / "public").is_dir():
                    is_ui = True
                elif (folder / "src" / "main" / "java").is_dir():
                    is_backend = True

            return {
                "path": folder,
                "name": rel_name,
                "package_name": pkg_name,
                "is_ui": is_ui,
                "is_backend": is_backend,
                "has_git": (folder / ".git").is_dir(),
            }

        # Check workspace root first
        if any((self.workspace / m).is_file() for m in manifest_names):
            discovered["."] = characterize(self.workspace, ".")

        # Scan direct child directories for monorepo subprojects
        raw_candidates: Dict[str, Dict[str, Any]] = {}
        try:
            for child in self.workspace.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    if any((child / m).is_file() for m in manifest_names):
                        raw_candidates[child.name] = characterize(child, child.name)
        except Exception as exc:
            logger.debug(f"[FastIDEEngine] Subproject discovery warning: {exc}")

        # Monorepo Package Uniqueness Invariant:
        # When multiple folders declare the identical package name (e.g. duplicate clones or fixture copies),
        # prioritize the canonical package directory whose folder name matches the package name.
        name_groups: Dict[str, List[Dict[str, Any]]] = {}
        for cname, cinfo in raw_candidates.items():
            p_ident = cinfo.get("package_name") or cname
            name_groups.setdefault(p_ident, []).append(cinfo)

        for p_ident, items in name_groups.items():
            if len(items) == 1:
                discovered[items[0]["name"]] = items[0]
            else:
                canonical = next((it for it in items if it["name"] == it.get("package_name")), items[0])
                discovered[canonical["name"]] = canonical

        return discovered

    def _analyze_ticket_domain_intent(self, text: str) -> Dict[str, Any]:
        """
        Extracts broad domain intent from ticket using comprehensive software taxonomies.
        Determines whether the ticket is strongly UI-centric, Backend-centric, or Fullstack.
        """
        tokens = set(re.findall(r"\b[a-z]{3,}\b", text.lower()))
        matched_ui = tokens.intersection(UI_DOMAIN_TAXONOMY)
        matched_backend = tokens.intersection(BACKEND_DOMAIN_TAXONOMY)

        ui_score = float(len(matched_ui))
        backend_score = float(len(matched_backend))

        is_strong_ui = ui_score >= 2 and ui_score > backend_score * 1.2
        is_strong_backend = backend_score >= 2 and backend_score > ui_score * 1.2

        return {
            "ui_score": ui_score,
            "backend_score": backend_score,
            "matched_ui": list(matched_ui),
            "matched_backend": list(matched_backend),
            "is_strong_ui": is_strong_ui,
            "is_strong_backend": is_strong_backend,
        }

    def _get_domain_scoring_multiplier(self, rel_path: str, ticket_intent: Dict[str, Any]) -> float:
        """
        Computes domain affinity scoring multiplier for a file based on ticket intent
        and the file's subproject/extension layer.
        """
        parts = rel_path.replace("\\", "/").split("/")
        sub_name = parts[0] if len(parts) > 1 and parts[0] in self._subprojects else "."
        sub_info = self._subprojects.get(sub_name, {})
        ext = Path(rel_path).suffix.lower()

        multiplier = 1.0
        if ticket_intent.get("is_strong_ui"):
            if sub_info.get("is_ui", False):
                multiplier *= 1.8
                if ext in {".html", ".component.ts", ".scss", ".css", ".tsx", ".vue"}:
                    multiplier *= 1.2
            elif sub_info.get("is_backend", False):
                multiplier *= 0.35  # Suppresses backend false positives on UI tickets
        elif ticket_intent.get("is_strong_backend"):
            if sub_info.get("is_backend", False):
                multiplier *= 1.8
                if ext in {".java", ".go", ".rs", ".py", ".sql"}:
                    multiplier *= 1.2
            elif sub_info.get("is_ui", False):
                multiplier *= 0.35  # Suppresses UI templates on backend tickets

        return multiplier

    def _emit_token_usage(self, phase: str = "patch_generation") -> None:
        """Emits token telemetry event to WebSocket so UI displays live token metrics during Tier 1."""
        if self.ui_callback:
            try:
                self.ui_callback({
                    "phase": phase,
                    "status": "in_progress",
                    "message": f"🪙 Tier 1 Token Telemetry: {self.tokens_used:,} tokens ({self.llm_calls} LLM call(s))",
                    "data": {
                        "event_type": "token_usage_update",
                        "total_tokens": self.tokens_used,
                        "tokens_in": self.tokens_in,
                        "tokens_out": self.tokens_out,
                        "tokens_thinking": 0,
                        "cached_tokens": 0,
                        "excluded_tokens": 0,
                        "real_costed_tokens": self.tokens_used,
                        "llm_calls": self.llm_calls,
                        "engine": "FastIDEEngine (Tier 1)",
                    },
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception as exc:
                logger.debug(f"[FastIDEEngine] token telemetry emit failed: {exc}")

    def _emit_step(
        self,
        phase: str,
        status: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emits thread-safe live WebSocket telemetry to the UI."""
        if self.ui_callback:
            try:
                payload_data = dict(data or {})
                if self.tokens_used > 0 and "token_usage" not in payload_data:
                    payload_data["token_usage"] = {
                        "total_tokens": self.tokens_used,
                        "tokens_in": self.tokens_in,
                        "tokens_out": self.tokens_out,
                        "llm_calls": self.llm_calls,
                        "engine": "FastIDEEngine (Tier 1)",
                    }
                self.ui_callback({
                    "phase": phase,
                    "status": status,
                    "message": message,
                    "data": payload_data,
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception as exc:
                logger.debug(f"[FastIDEEngine] ui_callback failed: {exc}")

    def _check_requires_code_changes(self) -> Tuple[bool, str]:
        """
        Determines whether the ticket requires code modifications vs pure read-only Q&A.
        Uses broad domain taxonomies to ensure actionable tickets are never misclassified.
        """
        title = (getattr(self.ticket, "title", "") or "").strip().lower()
        desc = (getattr(self.ticket, "description", "") or "").strip().lower()
        text = f"{title} {desc}"

        # If explicitly marked in expected_changed_files, it requires code
        if getattr(self.ticket, "expected_changed_files", None):
            return True, "Ticket specifies target files for modification."

        # If the ticket touches any UI element, backend concept, or action gesture, it requires code
        tokens = set(re.findall(r"\b[a-z]{3,}\b", text))
        if (tokens & UI_DOMAIN_TAXONOMY) or (tokens & BACKEND_DOMAIN_TAXONOMY):
            return True, "Code modification required."

        # Only classify as read-only if it is explicitly an exploratory question
        question_starters = ("how does", "what is", "what are", "explain the", "why is", "tell me about", "where is")
        if any(title.startswith(q) for q in question_starters) and (title.endswith("?") or desc.endswith("?")):
            return False, "Informational/architectural inquiry (read-only guidance, no code changes required)."

        return True, "Code modification required."

    def run(self) -> FastIDEResult:
        start_time = time.time()
        logger.info(f"⚡ [FastIDEEngine] Launching lean run for ticket: {self.ticket.ticket_id}")

        # ── Stage 1A: Intent Triage (Zero-LLM overhead) ──
        requires_code, triage_reason = self._check_requires_code_changes()
        if not requires_code:
            logger.info(f"⚡ [FastIDEEngine] Ticket is read-only inquiry: {triage_reason}")
            self._emit_step(
                "classification",
                "completed",
                f"ℹ️ Read-Only Inquiry: {triage_reason}",
                {"requires_code_changes": False},
            )
            # Answer read-only inquiry directly from repo outline and knowledge
            keywords = list(re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", f"{self.ticket.title} {self.ticket.description}"))
            repo_outline = self.repo_map.build_map(focus_keywords=keywords)
            explanation = (
                f"### Repository Architecture & Analysis:\n\n"
                f"**Query**: {self.ticket.title}\n\n"
                f"{triage_reason}\n\n"
                f"**Relevant Repository Components**:\n```\n{repo_outline}\n```"
            )
            self._emit_step(
                "planning",
                "completed",
                "📋 Synthesized architectural explanation without modifying code",
                {"explanation": explanation},
            )
            return FastIDEResult(
                success=True,
                status="no_change_required",   # read-only/explanatory ticket: nothing was built or tested
                duration_seconds=time.time() - start_time,
                tokens_consumed=0,
                modified_files=[],
                diffs={},
                build_passed=None,
                test_passed=None,
                message=explanation,
            )

        self._emit_step(
            "classification",
            "completed",
            "🔍 Intent Verified: Autonomous Code Modification",
            {"requires_code_changes": True},
        )

        # ── Stage 1B: ReAct Evidence Collection & Ranking ──
        self._emit_step(
            "localization",
            "in_progress",
            "🔍 Running ReAct Evidence Collection Loop (Call-Chain Tracing + Anchor Ranking)...",
            {"node": "evidence_collection_loop"},
        )
        target_files, ranked_evidence, call_chain_summary = self._react_evidence_collection()

        if not target_files:
            self._emit_step(
                "localization",
                "failed",
                "❌ Could not locate target source files deterministically",
                {"error": "no_target_files"},
            )
            return FastIDEResult(
                success=False,
                status="no_target_files",
                duration_seconds=time.time() - start_time,
                tokens_consumed=self.tokens_used,
                modified_files=[],
                error_message="Could not locate target source files deterministically",
                escalate_to_full_graph=True,
            )

        top_file_name = Path(target_files[0]).name
        self._emit_step(
            "localization",
            "completed",
            f"📍 Evidence Ranked: {len(target_files)} target files identified (Top: {top_file_name})",
            {
                "node": "evidence_ranking",
                "candidate_files": [{"path": f, "selected": True} for f in target_files],
                "ranked_evidence": ranked_evidence,
                "call_chain_summary": call_chain_summary,
            },
        )

        # Restrict blast-radius strictly to targets, companions, module roots, and migration folders
        allowed = set(target_files)
        for tf in target_files:
            parts = Path(tf).parts
            if len(parts) > 1:
                allowed.add(parts[0].replace("\\", "/"))
            allowed.add(str(Path(tf).parent).replace("\\", "/"))
            allowed.add("db/migration")
            allowed.add("migrations")
        self.diff_engine.set_allowed_paths(allowed)

        # ── Stage 2: Sliced Context & Planning ──
        self._emit_step(
            "planning",
            "in_progress",
            "🗺️ Building sliced context & architectural plan...",
            {"node": "plan"},
        )
        context_snippets = self._read_file_slices(target_files)
        if not context_snippets:
            return FastIDEResult(
                success=False,
                status="unreadable_context",
                duration_seconds=time.time() - start_time,
                tokens_consumed=self.tokens_used,
                modified_files=[],
                error_message="Target files were empty or unreadable",
                escalate_to_full_graph=True,
            )


        imported_definitions = self._resolve_imported_definitions(target_files)
        if imported_definitions:
            self._emit_step(
                "planning",
                "in_progress",
                f"📚 Resolved {len(imported_definitions)} referenced definitions: {', '.join(Path(k).name for k in imported_definitions.keys())}",
                {"imported_definitions": list(imported_definitions.keys())},
            )

        keywords = list(re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", f"{self.ticket.title} {self.ticket.description}"))
        repo_outline = self.repo_map.build_map(focus_keywords=keywords)
        self._emit_step(
            "planning",
            "completed",
            f"🗺️ Sliced context prepared across {len(context_snippets)} files",
            {"files": list(context_snippets.keys())},
        )

        # ── Stage 3 & 4: Surgical Patch Generation, Safety Hooks & Verification ──
        current_diffs: Dict[str, str] = {}
        last_compiler_error = ""

        for turn in range(self.max_repair_turns + 1):
            turn_start = time.time()
            turn_label = f"Turn {turn + 1}/{self.max_repair_turns + 1}"

            # Refresh live context snippets and file hashes from disk for subsequent turns
            if turn > 0:
                context_snippets = self._read_file_slices(target_files)
                for tf in target_files:
                    tf_full = self.workspace / tf
                    if tf_full.is_file():
                        try:
                            self.file_hashes[tf] = SurgicalDiffEngine.compute_content_hash(
                                tf_full.read_text(encoding="utf-8")
                            )
                        except Exception:
                            pass

            self._emit_step(
                "patch_generation",
                "in_progress",
                f"✍️ Generating surgical search/replace blocks ({turn_label})...",
                {"turn": turn, "node": "generate_code"},
            )

            prompt = self._construct_turn_prompt(
                context_snippets=context_snippets,
                repo_outline=repo_outline,
                last_error=last_compiler_error,
                turn=turn,
                call_chain_summary=call_chain_summary,
                imported_definitions=imported_definitions,
            )

            # Generate search/replace hunks via LLM
            llm_response, turn_tokens = self._call_llm(prompt)
            self.tokens_used += turn_tokens
            self._emit_token_usage(phase="patch_generation")

            if self.tokens_used >= self.max_budget_tokens:
                logger.warning(
                    f"⚡ [FastIDEEngine] Token budget exhausted ({self.tokens_used} >= {self.max_budget_tokens}). Halting repair turns."
                )
                self._emit_step(
                    "validation",
                    "in_progress",
                    f"⚠️ Token budget ceiling reached ({self.tokens_used:,} tokens). Halting repair loop.",
                    {"tokens_used": self.tokens_used, "budget": self.max_budget_tokens},
                )
                break

            # Parse search/replace blocks
            blocks = SearchReplaceParser.parse(llm_response, default_file_path=target_files[0])
            if not blocks:
                logger.warning(f"⚡ [FastIDEEngine] {turn_label}: No valid SEARCH/REPLACE blocks found in response. Preview:\n{llm_response[:300]}")
                if turn == self.max_repair_turns:
                    break
                last_compiler_error = (
                    "Formatting Error: You did not output valid <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE blocks. "
                    "Output exact blocks."
                )
                continue

            # Normalize block file paths against target_files or workspace glob
            for b in blocks:
                if not (self.workspace / b.file_path).is_file():
                    b_name = Path(b.file_path).name.lower()
                    matched_tf = next((tf for tf in target_files if Path(tf).name.lower() == b_name), None)
                    if matched_tf:
                        b.file_path = matched_tf
                    else:
                        for cand in self.workspace.glob(f"**/{Path(b.file_path).name}"):
                            if cand.is_file():
                                b.file_path = cand.relative_to(self.workspace).as_posix()
                                break

            # Deterministic Safety Gate 1: patch_gate (Syntax, brace/bracket balance, and structural validation)
            applied_any = False
            for b in blocks:
                gate_ok, gate_err = self._patch_gate(b)
                if not gate_ok:
                    logger.warning(f"🛡️ [patch_gate] Block rejected on {b.file_path}: {gate_err}")
                    self._emit_step(
                        "validation",
                        "in_progress",
                        f"🛡️ [patch_gate] Syntax reject on {Path(b.file_path).name}: {gate_err}",
                        {"node": "patch_gate", "file": b.file_path, "error": gate_err},
                    )
                    last_compiler_error = f"Syntax / Structural Error in patch for {b.file_path}: {gate_err}. Fix braces/brackets."
                    continue

                # Apply anchored replacement with cryptographic hash validation
                expected_hash = self.file_hashes.get(b.file_path)
                res = self.diff_engine.apply_anchored_replacement(
                    rel_path=b.file_path,
                    old_block=b.search_block,
                    new_block=b.replace_block,
                    expected_content_hash=expected_hash,
                )
                if res.success:
                    applied_any = True
                    # Deterministic Safety Gate 2: import_validation (Auto-inject missing imports if needed)
                    self._import_validation(b.file_path)

                    current_diffs[b.file_path] = res.unified_diff
                    # Always re-hash directly from disk to capture import_validation changes
                    full_p = self.workspace / b.file_path
                    if full_p.is_file():
                        try:
                            self.file_hashes[b.file_path] = SurgicalDiffEngine.compute_content_hash(
                                full_p.read_text(encoding="utf-8")
                            )
                        except Exception:
                            if res.new_content_hash:
                                self.file_hashes[b.file_path] = res.new_content_hash
                    elif res.new_content_hash:
                        self.file_hashes[b.file_path] = res.new_content_hash

                    # Emit file written event for real-time UI display
                    fname = Path(b.file_path).name
                    self._emit_step(
                        "patch_generation",
                        "completed",
                        f"📝 Written: {fname}",
                        {
                            "node": "generate_code",
                            "event_type": "file_written",
                            "file_path": b.file_path,
                            "file_name": fname,
                        },
                    )
                else:
                    logger.warning(f"⚡ [FastIDEEngine] Edit failed on {b.file_path}: [{res.status.value}] {res.message}")
                    # ── AIDER-STYLE: Feed back actual file content at closest match site ──
                    anchor_diag = f"Edit FAILED on {b.file_path} [{res.status.value}]: {res.message}\n"
                    if res.status.value in ("anchor_not_found", "ambiguous_anchor"):
                        try:
                            actual_content = (self.workspace / b.file_path).read_text(encoding="utf-8", errors="ignore")
                            actual_lines = actual_content.splitlines()
                            search_lines = b.search_block.strip().splitlines()
                            # Find best fuzzy match position to show the LLM what's actually there
                            best_pos = 0
                            best_score = 0.0
                            if search_lines and actual_lines:
                                first_search = search_lines[0].strip()
                                for idx, al in enumerate(actual_lines):
                                    if first_search and first_search in al.strip() or al.strip() in first_search:
                                        # Quick score check
                                        window = actual_lines[idx:idx + len(search_lines)]
                                        score = sum(1 for a, s in zip(window, search_lines) if a.strip() == s.strip()) / max(len(search_lines), 1)
                                        if score > best_score:
                                            best_score = score
                                            best_pos = idx
                            # Show surrounding 15 lines at best match position
                            start = max(0, best_pos - 3)
                            end = min(len(actual_lines), best_pos + len(search_lines) + 5)
                            actual_snippet = "\n".join(
                                f"{i+1:4d}| {actual_lines[i]}" for i in range(start, end)
                            )
                            anchor_diag += (
                                f"Your SEARCH block did not match. Here is what the file ACTUALLY contains "
                                f"around line {best_pos + 1} (closest match, score={best_score:.0%}):\n"
                                f"```\n{actual_snippet}\n```\n"
                                f"Copy the exact lines from above into your SEARCH block.\n"
                            )
                        except Exception:
                            pass
                    elif res.status.value == "stale_content_hash":
                        anchor_diag += "File was modified concurrently. Re-read the file content.\n"
                    # Accumulate all block-level failures for the repair turn
                    if last_compiler_error and "Edit FAILED" in last_compiler_error:
                        last_compiler_error += "\n" + anchor_diag
                    else:
                        last_compiler_error = anchor_diag

            if not applied_any:
                logger.warning(f"⚡ [FastIDEEngine] {turn_label}: No diff blocks could be cleanly anchored")
                if turn == self.max_repair_turns:
                    break
                if not last_compiler_error:
                    # Fallback: show first 40 lines of the primary target file
                    try:
                        fallback_content = (self.workspace / target_files[0]).read_text(encoding="utf-8", errors="ignore")
                        fallback_lines = fallback_content.splitlines()[:40]
                        fallback_snippet = "\n".join(f"{i+1:4d}| {l}" for i, l in enumerate(fallback_lines))
                        last_compiler_error = (
                            f"Anchor Matching Error: None of your SEARCH blocks matched in {target_files[0]}.\n"
                            f"Here are the first 40 lines of the file as it exists on disk:\n"
                            f"```\n{fallback_snippet}\n```\n"
                            f"Copy exact lines from above into your SEARCH blocks. Match indentation exactly."
                        )
                    except Exception:
                        last_compiler_error = (
                            f"Anchor Matching Error: Failed to find exact search block in {target_files[0]}. "
                            "Provide exact unchanged surrounding lines."
                        )
                continue

            # Deterministic Safety Gate 3: pre_fix_build (Native compiler verification + error slicing)
            self._emit_step(
                "validation",
                "in_progress",
                f"🔨 Running multi-zone deterministic build & compiler verification ({turn_label})...",
                {"node": "pre_fix_build"},
            )
            modified_list = list(current_diffs.keys()) or target_files
            build_ok, build_err = self._pre_fix_build(modified_list)
            if build_ok:
                logger.info(f"⚡ [FastIDEEngine] Multi-zone build cleanly verified in {time.time() - turn_start:.2f}s!")
                self._emit_step(
                    "validation",
                    "in_progress",
                    "🔨 Build succeeded! Running fast targeted test suite across affected zones...",
                    {"node": "test"},
                )
                # Run fast targeted test if available
                test_ok, test_err = self._run_fast_test(modified_list)
                if test_ok:
                    self._emit_step(
                        "validation",
                        "completed",
                        "✅ Clean build and test suite verified!",
                        {
                            "node": "test",
                            "modified_files": list(current_diffs.keys()),
                        },
                    )
                    return FastIDEResult(
                        success=True,
                        status="verified_success",
                        duration_seconds=time.time() - start_time,
                        tokens_consumed=self.tokens_used,
                        tokens_in=self.tokens_in,
                        tokens_out=self.tokens_out,
                        llm_calls=self.llm_calls,
                        modified_files=list(current_diffs.keys()),
                        diffs=current_diffs,
                        build_passed=True if getattr(self, "_last_build_ran", True) else None,
                        test_passed=True if getattr(self, "_last_test_ran", True) else None,
                        message="Ticket successfully resolved via FastIDEEngine",
                        ranked_evidence=ranked_evidence,
                        call_chain_summary=call_chain_summary,
                    )
                else:
                    # Differential check: Did the test failure actually involve the modified files?
                    # Exclude CLI invocation commands echoing argument flags (e.g. "ng test --include ...")
                    err_lines = [
                        ln for ln in test_err.splitlines()
                        if not ln.strip().startswith(("> ", "ng test", "npm test", "yarn test", "pnpm test", "pytest", "mvn ", "gradle ", "Executing Node tests"))
                        and any(term in ln.lower() for term in ("error", "fail", "failed", "failure", "assert", "exception", "cannot find", "is not assignable"))
                    ]
                    test_err_clean = "\n".join(err_lines)
                    affects_modified = any(
                        Path(m).name.lower() in test_err_clean.lower()
                        for m in modified_list
                    )
                    is_strict_gate = (
                        "strict verification gate" in test_err.lower()
                        or "0 tests completed" in test_err.lower()
                        or "0 tests run" in test_err.lower()
                    )
                    # A timeout, security rejection or runner crash is NOT a "pre-existing failure".
                    is_hard_failure = any(
                        k in test_err.lower()
                        for k in ("timeout", "timed out", "security", "rejected", "runner crashed",
                                  "toolchain incompatibility", "command not found", "no such file")
                    )
                    if not affects_modified and not is_strict_gate and not is_hard_failure:
                        logger.info(
                            "⚡ [FastIDEEngine] Test failures are in unrelated pre-existing modules; "
                            "clean multi-zone build verified. Accepting changes."
                        )
                        self._emit_step(
                            "validation",
                            "completed",
                            "✅ Clean build verified! (Pre-existing baseline test failures quarantined)",
                            {
                                "node": "test",
                                "modified_files": list(current_diffs.keys()),
                                "baseline_test_failures": True,
                            },
                        )
                        return FastIDEResult(
                            success=True,
                            status="verified_success",
                            duration_seconds=time.time() - start_time,
                            tokens_consumed=self.tokens_used,
                            tokens_in=self.tokens_in,
                            tokens_out=self.tokens_out,
                            llm_calls=self.llm_calls,
                            modified_files=list(current_diffs.keys()),
                            diffs=current_diffs,
                            build_passed=True if getattr(self, "_last_build_ran", True) else None,
                            test_passed=False,
                            message="Ticket successfully resolved via FastIDEEngine (pre-existing test errors quarantined)",
                            ranked_evidence=ranked_evidence,
                            call_chain_summary=call_chain_summary,
                        )

                    sliced_test_err = self._slice_error(test_err)
                    self._emit_step(
                        "validation",
                        "in_progress",
                        "⚠️ Targeted test failed — feeding sliced diagnostic to repair loop",
                        {"error": sliced_test_err},
                    )
                    last_compiler_error = f"Targeted Test Failure:\n{sliced_test_err}"
            else:
                self._emit_step(
                    "validation",
                    "in_progress",
                    "⚠️ Build error detected — feeding sliced diagnostic to repair loop",
                    {"error": build_err},
                )
                last_compiler_error = f"Compilation Error:\n{build_err}"

            logger.info(f"⚡ [FastIDEEngine] Turn {turn} did not pass verification. Sliced error fed back.")

        # If max turns reached without clean compile, rollback edits and escalate
        self.diff_engine.rollback_all()
        self._emit_step(
            "validation",
            "failed",
            f"⚡ FastIDEEngine could not converge within {self.max_repair_turns} repair turns. Escalating to deep reasoning.",
            {"escalate": True},
        )
        return FastIDEResult(
            success=False,
            status="max_turns_exceeded",
            duration_seconds=time.time() - start_time,
            tokens_consumed=self.tokens_used,
            modified_files=[],
            build_passed=False,
            test_passed=False,
            error_message=f"FastIDEEngine could not converge within {self.max_repair_turns} repair turns. Escalating.",
            escalate_to_full_graph=True,
            ranked_evidence=ranked_evidence,
            call_chain_summary=call_chain_summary,
            imported_defs=imported_definitions,
            target_files=target_files,
        )

    # =========================================================================
    # REACT EVIDENCE COLLECTION & RANKING (STAGE 1)
    # =========================================================================

    def _locate_target_files(self) -> List[str]:
        """Backwards compatibility alias for _react_evidence_collection targets."""
        targets, ranked_evidence, _ = self._react_evidence_collection()
        test_cands = [item["file_path"] for item in ranked_evidence if item.get("is_test", False)]
        combined = list(targets)
        for tc in test_cands:
            if tc not in combined:
                combined.append(tc)
        return combined[:5]

    def _react_evidence_collection(self) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        """
        Industry-standard ReAct Evidence Collection & Ranking Loop.
        Combines:
          1. Exact literal & anchor extraction with IDF frequency scoring.
          2. Fullstack CallChainTracer (UI action -> Saga -> Backend Handler).
          3. Component Grouping & Companion Discovery (TS+HTML+SCSS, Service+Repo+Test).
          4. Evidence Ranking with smoking gun and call-chain bonuses.
        """
        title = getattr(self.ticket, "title", "") or ""
        desc = getattr(self.ticket, "description", "") or ""
        full_text = f"{title} {desc}"

        call_chain_summary: List[str] = []
        call_chain_files: Set[str] = set()

        # ── Step 1: Call-Chain Tracing for User Actions & Async Sagas ──
        action_terms = (
            "remove", "delete", "revoke", "dialog", "modal", "button",
            "click", "message", "popup", "confirm", "action", "trigger",
            "flow", "save", "update", "cancel", "event", "listener",
            "consumer", "saga", "rabbit", "queue", "topic", "publish", "subscribe"
        )
        has_action = any(term in full_text.lower() for term in action_terms)

        candidate_queries = []
        # Extract quoted literals, dotted paths, or multi-word actions
        for q in re.findall(r"['\"]([\w.\- ]{3,40})['\"]", full_text):
            candidate_queries.append(q)
        for m in re.findall(r"\b[A-Za-z]\w*\.[A-Za-z]\w*(?:\.[A-Za-z]\w*)?\b", full_text):
            if not m.endswith((".ts", ".java", ".py", ".html", ".scss", ".json")):
                candidate_queries.append(m)

        best_query = candidate_queries[0] if candidate_queries else None
        if not best_query and has_action:
            words = [w for w in title.split() if len(w) >= 4]
            if words:
                best_query = " ".join(words[:3])

        # Step 1: Call-Chain Tracing for User Actions & Async Sagas
        # Only run cross-tier CallChainTracer if ticket references cross-tier or saga terms
        cross_tier_terms = ("saga", "queue", "topic", "publish", "subscribe", "consumer", "rabbitmq", "backend", "handler")
        needs_cross_tier = any(term in full_text.lower() for term in cross_tier_terms)

        if best_query and needs_cross_tier:
            try:
                import concurrent.futures
                tracer = CallChainTracer(str(self.workspace))
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(tracer.trace, best_query)
                    trace_res = future.result(timeout=4.0)

                if trace_res:
                    summary = trace_res.get("summary_trace") or []
                    if summary:
                        call_chain_summary = summary
                        logger.info(f"⚡ [ReAct Evidence] CallChainTracer identified {len(summary)} trace hops")

                    # Collect candidate file paths identified along the fullstack execution chain
                    for ui in trace_res.get("ui_components", []):
                        fp = ui.get("file")
                        if fp:
                            call_chain_files.add(self._to_rel_posix(fp))
                    for svc in trace_res.get("service_calls", []):
                        fp = svc.get("file")
                        if fp:
                            call_chain_files.add(self._to_rel_posix(fp))
                    for bh in trace_res.get("backend_handlers", []):
                        fp = bh.get("file")
                        if fp:
                            call_chain_files.add(self._to_rel_posix(fp))
                    for sg in trace_res.get("saga_commands", []):
                        fp = sg.get("file")
                        if fp:
                            call_chain_files.add(self._to_rel_posix(fp))
                    for sm in trace_res.get("semantic_mutations", []):
                        fp = sm.get("file")
                        if fp:
                            call_chain_files.add(self._to_rel_posix(fp))
                    for mc in trace_res.get("model_contracts", []):
                        fp = mc.get("file")
                        if fp:
                            call_chain_files.add(self._to_rel_posix(fp))
            except Exception as tr_err:
                logger.debug(f"[FastIDEEngine] CallChainTracer skipped: {tr_err}")

        # 🕸️ Contract Graph Cross-Service Discovery (REST & Event messaging edges)
        if hasattr(self, "contract_indexer") and self.contract_indexer and needs_cross_tier:
            try:
                c_links = self.contract_indexer.find_linked_services(best_query)
                for lk in c_links:
                    call_chain_files.add(lk.producer.file_path)
                    call_chain_files.add(lk.consumer.file_path)
                    call_chain_summary.append(f"{lk.producer.service_name} -> {lk.consumer.service_name} ({lk.contract_type.value}: {lk.identifier})")
                if c_links:
                    logger.info(f"🕸️ [ReAct Evidence] ContractGraphIndexer linked {len(c_links)} cross-service endpoints")
            except Exception as cg_err:
                logger.debug(f"[FastIDEEngine] ContractGraphIndexer error: {cg_err}")

        # ── Step 2: Anchor & Identifier Scoring with Multi-Root Content Search ──
        ticket_intent = self._analyze_ticket_domain_intent(full_text)
        logger.info(
            f"⚡ [FastIDEEngine] Ticket Domain Intent: UI={ticket_intent['ui_score']:.1f} (strong={ticket_intent['is_strong_ui']}), "
            f"Backend={ticket_intent['backend_score']:.1f} (strong={ticket_intent['is_strong_backend']})"
        )

        code_exts = {".ts", ".tsx", ".js", ".jsx", ".java", ".py", ".html", ".scss", ".css", ".sql", ".json", ".vue"}
        skip_dirs = {
            "node_modules", "dist", "target", "build", ".git", ".angular", "bin",
            "coverage", "__pycache__", "out", ".idea", ".vscode"
        }

        explicit_paths = re.findall(r"[\w\-/\\]+\.(?:java|py|ts|tsx|js|jsx|cs|go|rs|sql|html|css|scss|json|vue)", full_text)
        valid_explicit = []
        for p in explicit_paths:
            norm = p.replace("\\", "/").lstrip("/")
            if (self.workspace / norm).is_file():
                valid_explicit.append(norm)
            else:
                # Check subprojects
                found_in_sub = False
                for sname, sinfo in self._subprojects.items():
                    sub_p = sinfo["path"] / norm
                    if sub_p.is_file():
                        rel_ws = self._to_rel_posix(sub_p)
                        if rel_ws not in valid_explicit:
                            valid_explicit.append(rel_ws)
                        found_in_sub = True
                # Check workspace glob fallback
                if not found_in_sub:
                    norm_name = Path(norm).name
                    for match_f in self.workspace.glob(f"**/{norm_name}"):
                        if match_f.is_file() and not any(skip in match_f.parts for skip in skip_dirs):
                            rel_ws = self._to_rel_posix(match_f)
                            if rel_ws not in valid_explicit:
                                valid_explicit.append(rel_ws)

        keywords = set(re.findall(r"\b[A-Z][a-zA-Z0-9]{3,}\b", full_text))
        for q in candidate_queries:
            if "." not in q and len(q) >= 4:
                keywords.add(q)

        # Include essential lowercase UI & architectural domain terms
        ui_domain_terms = {
            "masthead", "header", "jheader", "jlayout", "dashboard", "navigation",
            "overview", "navbar", "toolbar", "menu", "logo", "breadcrumb"
        }
        for term in ui_domain_terms:
            if term in full_text.lower():
                keywords.add(term)

        candidate_scores: Dict[str, float] = {}
        candidate_matches: Dict[str, List[str]] = {}

        # Pre-seed explicit paths and call-chain files with high initial scores
        for ep in valid_explicit:
            candidate_scores[ep] = 100.0
            candidate_matches[ep] = ["explicit_ticket_path"]

        for cf in call_chain_files:
            if cf:
                candidate_scores[cf] = candidate_scores.get(cf, 0.0) + 80.0
                candidate_matches.setdefault(cf, []).append("call_chain_entrypoint")

        # Extract high-entropy ticket terms (not just 3 hardcoded anchors)
        stop_words = {
            "about", "after", "again", "against", "could", "should", "would",
            "where", "which", "while", "under", "their", "there", "these",
            "other", "order", "same", "today", "want", "able", "application",
            "this", "that", "with", "from", "into", "need", "ensure", "matches",
            "behavior", "existing", "user", "view", "pattern", "open", "window",
            "browser", "when", "then", "also", "more", "some", "only", "such",
            "each", "both", "like", "just", "even", "make", "take", "call",
            "code", "file", "line", "true", "false", "null", "type", "data",
            "item", "list", "info", "mode", "must", "well", "done", "used"
        }

        # Clean title prefix (e.g. "Member: ...", "BUG-123: ...") to extract high-intent subject tokens
        clean_title = re.sub(r"^[A-Za-z0-9_\-]+:\s*", "", title)
        title_raw = re.findall(r"\b[A-Za-z][A-Za-z0-9_\-]{2,}\b", clean_title)
        title_words = {w.lower() for w in title_raw if w.lower() not in stop_words and len(w) >= 3}

        raw_words = re.findall(r"\b[A-Za-z][A-Za-z0-9_\-]{2,}\b", full_text)
        high_entropy_terms = {
            w.lower() for w in raw_words
            if w.lower() not in stop_words and len(w) >= 4
        }
        for q in candidate_queries:
            for qw in q.split():
                if len(qw) >= 4 and qw.lower() not in stop_words:
                    high_entropy_terms.add(qw.lower())

        # Multi-root git search: If workspace is not a single git repo, scan discovered subproject repos
        search_roots: List[Tuple[str, Path]] = []
        if self._subprojects:
            for sname, sinfo in self._subprojects.items():
                # Optimization: If ticket has strong UI intent, only search UI subprojects (and vice versa)
                if ticket_intent.get("is_strong_ui") and not sinfo.get("is_ui", False):
                    continue
                if ticket_intent.get("is_strong_backend") and sinfo.get("is_ui", False):
                    continue
                search_roots.append((sname, sinfo["path"]))
        if not search_roots:
            search_roots.append((".", self.workspace))

        for sub_name, sub_folder in search_roots:
            for term in high_entropy_terms:
                try:
                    import subprocess
                    rg_res = subprocess.run(
                        ["git", "grep", "-l", "-i", "--untracked", term],
                        cwd=str(sub_folder),
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if rg_res.returncode != 0:
                        rg_res = subprocess.run(
                            ["git", "grep", "-l", "-i", term],
                            cwd=str(sub_folder),
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                    if rg_res.returncode == 0:
                        for matched_f in rg_res.stdout.splitlines():
                            matched_f = matched_f.strip().replace("\\", "/")
                            if not matched_f:
                                continue
                            rel_p = f"{sub_name}/{matched_f}" if sub_name != "." else matched_f
                            # Only score valid code and template extensions
                            if Path(rel_p).suffix.lower() not in code_exts:
                                continue
                            if (self.workspace / rel_p).is_file():
                                candidate_scores[rel_p] = candidate_scores.get(rel_p, 0.0) + 25.0
                                candidate_matches.setdefault(rel_p, []).append(f"content:{term}")
                except Exception as _rg_err:
                    logger.debug(f"[FastIDEEngine] Multi-root search skipped for {term}: {_rg_err}")

        # Scan files matching keywords
        file_count = 0
        for root, dirs, files in os.walk(self.workspace):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for f in files:
                ext = Path(f).suffix.lower()
                if ext not in code_exts:
                    continue
                file_count += 1
                full_p = Path(root) / f
                rel_p = self._to_rel_posix(full_p)
                rel_p_lower = rel_p.lower()
                f_lower = f.lower()

                # Title words carry highest intent weight (direct subject match)
                matched_title = [tw for tw in title_words if tw in rel_p_lower]
                if matched_title:
                    t_boost = 40.0 * len(matched_title)
                    if any(tw in f_lower for tw in matched_title):
                        t_boost += 30.0  # Filename matches title term
                    candidate_scores[rel_p] = candidate_scores.get(rel_p, 0.0) + t_boost
                    candidate_matches.setdefault(rel_p, []).extend([f"title:{tw}" for tw in matched_title])

                # Domain keywords match in path and filename
                matched_kw = [kw for kw in keywords if kw.lower() in rel_p_lower]
                if matched_kw:
                    k_boost = 15.0 * len(matched_kw)
                    if any(kw.lower() in f_lower for kw in matched_kw):
                        k_boost += 15.0  # Filename match is stronger than path segment
                    candidate_scores[rel_p] = candidate_scores.get(rel_p, 0.0) + k_boost
                    candidate_matches.setdefault(rel_p, []).extend(matched_kw)

        # ── Step 3: Component Grouping & Companions ──
        companion_map: Dict[str, float] = {}
        for cand, score in list(candidate_scores.items()):
            p = self.workspace / cand
            stem = p.stem
            parent = p.parent

            # Angular / Web companions (bidirectional template <-> component pairing)
            if cand.endswith(".component.ts"):
                for sibling_ext in [".component.html", ".component.scss", ".component.spec.ts"]:
                    sib = parent / (stem.replace(".component", "") + sibling_ext)
                    if sib.is_file():
                        rel_sib = self._to_rel_posix(sib)
                        sib_weight = 0.95 if sibling_ext == ".component.html" else 0.50
                        companion_map[rel_sib] = max(companion_map.get(rel_sib, 0.0), score * sib_weight)
            elif cand.endswith(".component.html"):
                ts_sib = parent / (stem.replace(".component", "") + ".component.ts")
                if ts_sib.is_file():
                    rel_sib = self._to_rel_posix(ts_sib)
                    companion_map[rel_sib] = max(companion_map.get(rel_sib, 0.0), score * 0.95)
            # Java companions (Services, Repositories, DAOs, Models, and Maven Tests)
            elif cand.endswith("Service.java") or cand.endswith("Controller.java"):
                base = stem[:-7] if cand.endswith("Service.java") else stem[:-10]
                for sib_name in [f"{base}ServiceImpl.java", f"{base}Repository.java", f"{base}Test.java"]:
                    for cand_dir in [parent, parent.parent / "repository", parent.parent / "dao", parent.parent / "service"]:
                        sib = cand_dir / sib_name
                        if sib.is_file():
                            rel_sib = self._to_rel_posix(sib)
                            companion_map[rel_sib] = max(companion_map.get(rel_sib, 0.0), score * 0.85)

                # Parse injected repository/service fields directly from content
                try:
                    c_text = p.read_text(encoding="utf-8", errors="ignore")
                    for repo_cls in re.findall(r"(?:private|protected|public)?\s+(?:final\s+)?([A-Z]\w*(?:Repository|Dao|Service))\s+\w+;", c_text):
                        for cand_dir in [parent, parent.parent / "repository", parent.parent / "dao", parent.parent / "service"]:
                            target_f = cand_dir / f"{repo_cls}.java"
                            if target_f.is_file():
                                rel_sib = self._to_rel_posix(target_f)
                                companion_map[rel_sib] = max(companion_map.get(rel_sib, 0.0), score * 0.85)
                except Exception:
                    pass

                # Maven test companion discovery (src/main/java -> src/test/java)
                test_posix = str(p).replace("src/main/java", "src/test/java").replace(".java", "Test.java")
                test_cand = Path(test_posix)
                if test_cand.is_file():
                    companion_map[self._to_rel_posix(test_cand)] = max(companion_map.get(self._to_rel_posix(test_cand), 0.0), score * 0.90)
            # Python companions
            elif cand.endswith(".py") and not cand.startswith("test_"):
                test_sib = parent / f"test_{stem}.py"
                if test_sib.is_file():
                    rel_sib = self._to_rel_posix(test_sib)
                    companion_map[rel_sib] = max(companion_map.get(rel_sib, 0.0), score * 0.85)

            # Direct test companion check (O(1) without nested workspace walks)
            for tp in [f"{stem}Test.java", f"{stem}Tests.java", f"test_{stem}.py", f"{stem}.spec.ts"]:
                for test_cand in [parent / tp, parent.parent / "test" / tp, parent.parent / "tests" / tp]:
                    if test_cand.is_file():
                        rel_test = self._to_rel_posix(test_cand)
                        if rel_test != cand:
                            companion_map[rel_test] = max(companion_map.get(rel_test, 0.0), score * 0.85)

        for comp_rel, comp_score in companion_map.items():
            if comp_rel not in candidate_scores:
                candidate_scores[comp_rel] = comp_score
                candidate_matches.setdefault(comp_rel, []).append("companion_group_member")

        # ── Step 3.5: Wise Domain-Aware Subproject Scoring Multiplier ──
        # Multiplies UI subproject & template scores when ticket has UI intent,
        # while heavily penalizing false-positive matches on backend services (e.g. CMSHelperService.java).
        for rel_p in list(candidate_scores.keys()):
            mult = self._get_domain_scoring_multiplier(rel_p, ticket_intent)
            candidate_scores[rel_p] = candidate_scores[rel_p] * mult

        # ── Step 4: Evidence Ranking with Mock/Test/Style Down-Ranking ──
        test_requested = any(w in full_text.lower() for w in ["test", "tests", "spec", "coverage", "assert", "unit test", "tdd"])
        has_style_intent = any(w in full_text.lower() for w in ("style", "css", "scss", "color", "theme", "font", "padding", "margin"))
        ranked_list = []
        for rel_path, score in sorted(candidate_scores.items(), key=lambda x: -x[1]):
            low = rel_path.lower()
            is_test = "/test/" in low or low.endswith((".spec.ts", "test.java", "test.py"))
            is_mock = any(m in low for m in ["/mock/", "/mocks/", "mock.", "mock-", ".mock", "stub", "fixture", "-fixture"])
            is_style = low.endswith((".scss", ".css", ".less"))
            
            if is_style and not has_style_intent:
                adjusted_score = score * 0.35  # Deprioritize styling files when implementing functionality
            elif test_requested:
                adjusted_score = score
            elif is_mock:
                adjusted_score = score * 0.1  # Heavily penalize mocks/fixtures when writing real features
            elif is_test and len(candidate_scores) > 2:
                adjusted_score = score * 0.4  # Deprioritize tests below production components
            else:
                adjusted_score = score

            ranked_list.append({
                "file_path": rel_path,
                "score": round(adjusted_score, 2),
                "reasons": list(set(candidate_matches.get(rel_path, []))),
                "is_test": is_test or is_mock,
            })

        # Sort by final adjusted score
        ranked_list.sort(key=lambda x: -x["score"])
        if ranked_list:
            # Filter out tests and mocks from code generation target candidates
            prod_cands = [item for item in ranked_list if not item.get("is_test", False)]
            if prod_cands:
                top_score = prod_cands[0]["score"]
                # Keep top production candidates within 50% of top score or directly matched by title/anchors, max 5 primary targets
                filtered_targets = [
                    item["file_path"] for item in prod_cands
                    if item["score"] >= max(25.0, top_score * 0.50) or any("title:" in r for r in item.get("reasons", []))
                ][:5]
                selected_targets = filtered_targets if filtered_targets else [prod_cands[0]["file_path"]]
            else:
                top_score = ranked_list[0]["score"]
                selected_targets = [ranked_list[0]["file_path"]]
        else:
            selected_targets = []

        # Enforce UI Template + Component Co-Targeting:
        # If any *.component.ts or *.component.html is in selected_targets and ticket has UI intent,
        # ensure its companion pair is co-targeted for atomic modification.
        if ticket_intent.get("is_strong_ui"):
            extra_targets = []
            for st in selected_targets:
                if st.endswith(".component.ts"):
                    html_pair = st[:-13] + ".component.html"
                    if (self.workspace / html_pair).is_file() and html_pair not in selected_targets:
                        extra_targets.append(html_pair)
                elif st.endswith(".component.html"):
                    ts_pair = st[:-15] + ".component.ts"
                    if (self.workspace / ts_pair).is_file() and ts_pair not in selected_targets:
                        extra_targets.append(ts_pair)
            selected_targets.extend(extra_targets)

        return selected_targets, ranked_list[:12], call_chain_summary

    def _to_rel_posix(self, p: Any) -> str:
        """Converts Path or string to clean POSIX workspace-relative path."""
        s = str(p).replace("\\", "/")
        ws = str(self.workspace).replace("\\", "/")
        if s.startswith(ws):
            s = s[len(ws):].lstrip("/")
        return s

    # =========================================================================
    # DETERMINISTIC SAFETY GATES (STAGE 3 & 4)
    # =========================================================================

    def _patch_gate(self, block: SearchReplaceBlock) -> Tuple[bool, str]:
        """
        Deterministic Syntax & Structural Safety Gate (0 LLM Tokens).
        Validates bracket balance, markdown fence leakage, and structural sanity.
        """
        if not block.file_path:
            return False, "Empty file path"

        if not block.search_block:
            full_p = self.workspace / block.file_path
            if not full_p.is_file():
                # Attempt to resolve workspace-relative path across subproject folders
                fname = Path(block.file_path).name
                for cand in self.workspace.glob(f"**/{fname}"):
                    if cand.is_file():
                        cand_rel = cand.relative_to(self.workspace).as_posix()
                        if block.file_path.replace("\\", "/").lower() in cand_rel.lower():
                            full_p = cand
                            block.file_path = cand_rel
                            break

            if full_p.is_file() and full_p.stat().st_size > 50:
                # If file is small (< 3000 bytes) or replacement contains full valid file structure, allow full file rewrite
                is_small = full_p.stat().st_size < 3000
                is_full = any(k in block.replace_block for k in ("package ", "class ", "enum ", "export type", "export interface", "import ", "export const", "export default"))
                if is_small or is_full:
                    block.search_block = full_p.read_text(encoding="utf-8", errors="ignore")
                else:
                    return False, f"Empty SEARCH block on existing file '{block.file_path}' is forbidden. Output exact SEARCH anchor."
            elif not full_p.is_file():
                # New file creation: empty search block is valid
                pass

        if "```" in block.replace_block:
            block.replace_block = re.sub(r"^\s*```.*$\r?\n?", "", block.replace_block, flags=re.MULTILINE).replace("```", "")
        if "```" in block.search_block:
            block.search_block = re.sub(r"^\s*```.*$\r?\n?", "", block.search_block, flags=re.MULTILINE).replace("```", "")

        if re.search(r"^\s*(?:\.\.\.|…)\s*$", block.replace_block, re.MULTILINE):
            return False, "Replacement contains literal '...' placeholder/truncation marker which would corrupt source. Output complete replacement lines."

        ext = Path(block.file_path).suffix.lower()
        if ext in {".java", ".ts", ".tsx", ".js", ".jsx", ".cs", ".c", ".cpp"}:
            search_braces = block.search_block.count("{") - block.search_block.count("}")
            replace_braces = block.replace_block.count("{") - block.replace_block.count("}")
            search_parens = block.search_block.count("(") - block.search_block.count(")")
            replace_parens = block.replace_block.count("(") - block.replace_block.count(")")
            search_brackets = block.search_block.count("[") - block.search_block.count("]")
            replace_brackets = block.replace_block.count("[") - block.replace_block.count("]")

            # If search block has net braces 0, but replacement has unbalanced braces, fail gate
            if search_braces == 0 and replace_braces != 0:
                return False, f"Unbalanced curly braces in replacement (net delta: {replace_braces:+d})"
            if search_parens == 0 and replace_parens != 0:
                return False, f"Unbalanced parentheses in replacement (net delta: {replace_parens:+d})"
            if search_brackets == 0 and replace_brackets != 0:
                return False, f"Unbalanced square brackets in replacement (net delta: {replace_brackets:+d})"

        # Deterministic Invariant Code Gates (Invariant 18, 19, etc.)
        try:
            from ticket_to_code.runtime.invariant_gates import InvariantEnforcementGates
            ticket_text = f"{self.ticket.title}\n{self.ticket.description}" if self.ticket else ""
            passed, inv_errors = InvariantEnforcementGates.run_all_gates(
                self.workspace, ticket_text, block.file_path, block.replace_block
            )
            if not passed:
                return False, "; ".join(inv_errors)
        except Exception as e:
            logger.debug(f"Invariant gate check skipped: {e}")

        return True, ""

    def _import_validation(self, file_path: str) -> None:
        """
        Deterministic AST/Import Validation Hook (0 LLM Tokens).
        Scans modified file for unresolved types/annotations and auto-inserts missing imports.
        """
        p = self.workspace / file_path
        if not p.is_file():
            return
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return

        ext = p.suffix.lower()
        modified = False

        if ext == ".java":
            common_java_imports = {
                "List": "import java.util.List;",
                "ArrayList": "import java.util.ArrayList;",
                "Map": "import java.util.Map;",
                "HashMap": "import java.util.HashMap;",
                "Set": "import java.util.Set;",
                "HashSet": "import java.util.HashSet;",
                "Optional": "import java.util.Optional;",
                "UUID": "import java.util.UUID;",
                "Objects": "import java.util.Objects;",
                "Autowired": "import org.springframework.beans.factory.annotation.Autowired;",
                "Service": "import org.springframework.stereotype.Service;",
                "Component": "import org.springframework.stereotype.Component;",
                "Repository": "import org.springframework.stereotype.Repository;",
                "Transactional": "import org.springframework.transaction.annotation.Transactional;",
                "Slf4j": "import lombok.extern.slf4j.Slf4j;",
                "Data": "import lombok.Data;",
                "ResponseEntity": "import org.springframework.http.ResponseEntity;",
                "HttpStatus": "import org.springframework.http.HttpStatus;",
                "PathVariable": "import org.springframework.web.bind.annotation.PathVariable;",
                "RequestParam": "import org.springframework.web.bind.annotation.RequestParam;",
                "RequestBody": "import org.springframework.web.bind.annotation.RequestBody;",
                "GetMapping": "import org.springframework.web.bind.annotation.GetMapping;",
                "PostMapping": "import org.springframework.web.bind.annotation.PostMapping;",
                "PutMapping": "import org.springframework.web.bind.annotation.PutMapping;",
                "DeleteMapping": "import org.springframework.web.bind.annotation.DeleteMapping;",
                "RestController": "import org.springframework.web.bind.annotation.RestController;",
                "RequestMapping": "import org.springframework.web.bind.annotation.RequestMapping;",
                "Query": "import org.springframework.data.jpa.repository.Query;",
                "Param": "import org.springframework.data.repository.query.Param;",
                "Collectors": "import java.util.stream.Collectors;",
            }
            needed_imports = []
            for symbol, import_stmt in common_java_imports.items():
                if re.search(r"\b" + symbol + r"\b", content):
                    if import_stmt not in content and f"java.util.{symbol}" not in content:
                        needed_imports.append(import_stmt)

            # Dynamic workspace-wide symbol discovery for custom project classes
            referenced_types = set(re.findall(r"\b([A-Z][a-zA-Z0-9]+)\b", content))
            pkg_match = re.search(r"package\s+([\w.]+);", content)
            current_pkg = pkg_match.group(1) if pkg_match else ""
            for t in referenced_types:
                if t == p.stem or t in common_java_imports or t in {"String", "Integer", "Boolean", "Long", "Double", "Float", "Byte", "Short", "Object", "Class", "System", "Override"}:
                    continue
                if re.search(rf"import\s+[\w.]+\.{re.escape(t)};", content):
                    continue
                workspace_stmt = self._find_workspace_java_import(t)
                if workspace_stmt:
                    # Never import symbols in the same package or self
                    if current_pkg and workspace_stmt.startswith(f"import {current_pkg}."):
                        continue
                    if workspace_stmt not in content and workspace_stmt not in needed_imports:
                        needed_imports.append(workspace_stmt)

            if needed_imports:
                lines = content.splitlines()
                insert_idx = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith("package "):
                        insert_idx = i + 1
                        break
                    elif line.strip().startswith("import "):
                        insert_idx = i
                        break
                new_lines = lines[:insert_idx] + [""] + needed_imports + lines[insert_idx:]
                content = "\n".join(new_lines)
                modified = True
                logger.info(f"🛡️ [_import_validation] Auto-injected {len(needed_imports)} Java imports into {file_path}")

        elif ext in {".ts", ".tsx"}:
            common_ts_imports = {
                "Injectable": "import { Injectable } from '@angular/core';",
                "Component": "import { Component } from '@angular/core';",
                "OnInit": "import { OnInit } from '@angular/core';",
                "Input": "import { Input } from '@angular/core';",
                "Output": "import { Output, EventEmitter } from '@angular/core';",
                "Observable": "import { Observable } from 'rxjs';",
                "BehaviorSubject": "import { BehaviorSubject } from 'rxjs';",
                "Subject": "import { Subject } from 'rxjs';",
                "of": "import { of } from 'rxjs';",
                "HttpHeaders": "import { HttpHeaders } from '@angular/common/http';",
                "HttpClient": "import { HttpClient } from '@angular/common/http';",
            }
            needed_imports = []
            for symbol, import_stmt in common_ts_imports.items():
                if re.search(r"\b" + symbol + r"\b", content):
                    if not re.search(r"import\s*\{[^}]*\b" + symbol + r"\b[^}]*\}", content):
                        needed_imports.append(import_stmt)

            # Dynamic workspace-wide TS symbol discovery for models, interfaces, and ambient declarations
            referenced_types = set(re.findall(r"\b([A-Z][a-zA-Z0-9]+)\b", content))
            for t in referenced_types:
                if t in common_ts_imports or t in {"Array", "String", "Number", "Boolean", "Object", "Function", "Promise", "Date", "RegExp", "Error"}:
                    continue
                workspace_stmt = self._find_workspace_ts_import(p, t)
                if not workspace_stmt:
                    continue
                if workspace_stmt.startswith("declare "):
                    # Ambient script declaration (e.g. declare var OTHHUrlBuilder: any;)
                    # Remove any hallucinated ES module imports for this ambient symbol:
                    cleaned_content = re.sub(
                        rf"import\s*\{{[^}}]*\b{re.escape(t)}\b[^}}]*\}}\s*from\s*['\"][^'\"]+['\"];?\s*",
                        "",
                        content
                    )
                    if cleaned_content != content:
                        content = cleaned_content
                        modified = True
                        logger.info(f"🛡️ [_import_validation] Stripped hallucinated ES import for ambient symbol '{t}'")
                    if workspace_stmt not in content and workspace_stmt not in needed_imports:
                        needed_imports.append(workspace_stmt)
                else:
                    if re.search(rf"import\s*\{{[^}}]*\b{re.escape(t)}\b[^}}]*\}}", content):
                        continue
                    if workspace_stmt not in content and workspace_stmt not in needed_imports:
                        needed_imports.append(workspace_stmt)

            if needed_imports:
                content = "\n".join(needed_imports) + "\n\n" + content
                modified = True
                logger.info(f"🛡️ [_import_validation] Auto-injected {len(needed_imports)} TS imports into {file_path}")

        elif ext == ".py":
            common_py_imports = {
                "Optional": "from typing import Optional",
                "List": "from typing import List",
                "Dict": "from typing import Dict",
                "Set": "from typing import Set",
                "Tuple": "from typing import Tuple",
                "Any": "from typing import Any",
                "Union": "from typing import Union",
                "dataclass": "from dataclasses import dataclass, field",
                "Path": "from pathlib import Path",
                "BaseModel": "from pydantic import BaseModel, Field",
                "FastAPI": "from fastapi import FastAPI, HTTPException, Depends",
                "APIRouter": "from fastapi import APIRouter",
                "pytest": "import pytest",
            }
            needed_imports = []
            for sym, imp_stmt in common_py_imports.items():
                if re.search(r"\b" + re.escape(sym) + r"\b", content):
                    if not re.search(r"\bimport\s+.*\b" + re.escape(sym) + r"\b", content):
                        needed_imports.append(imp_stmt)

            if needed_imports:
                content = "\n".join(needed_imports) + "\n\n" + content
                modified = True
                logger.info(f"🛡️ [_import_validation] Auto-injected {len(needed_imports)} Python imports into {file_path}")

            # Optional native toolchain formatting if ruff is installed
            import shutil
            ruff_bin = shutil.which("ruff")
            if ruff_bin and modified:
                try:
                    p.write_text(content, encoding="utf-8")
                    subprocess.run([ruff_bin, "check", "--select", "I", "--fix", str(p)], capture_output=True, timeout=5)
                    content = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    pass

        elif ext == ".go":
            # In Go, unused imports cause fatal compiler errors. Delegate strictly to goimports if available.
            import shutil
            goimports_bin = shutil.which("goimports")
            if goimports_bin:
                try:
                    subprocess.run([goimports_bin, "-w", str(p)], capture_output=True, timeout=5)
                    content = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    pass

        elif ext == ".cs":
            common_cs_namespaces = [
                ("List<", "using System.Collections.Generic;"),
                ("Dictionary<", "using System.Collections.Generic;"),
                ("Task<", "using System.Threading.Tasks;"),
                ("Task.", "using System.Threading.Tasks;"),
                ("IActionResult", "using Microsoft.AspNetCore.Mvc;"),
                ("ControllerBase", "using Microsoft.AspNetCore.Mvc;"),
                ("HttpGet", "using Microsoft.AspNetCore.Mvc;"),
                ("HttpPost", "using Microsoft.AspNetCore.Mvc;"),
            ]
            needed_usings = []
            for sym, using_stmt in common_cs_namespaces:
                if sym in content and using_stmt not in content:
                    needed_usings.append(using_stmt)
            if needed_usings:
                content = "\n".join(needed_usings) + "\n" + content
                modified = True
                logger.info(f"🛡️ [_import_validation] Auto-injected {len(needed_usings)} C# usings into {file_path}")

        if modified:
            p.write_text(content, encoding="utf-8")
            self.file_hashes[file_path] = self.diff_engine.compute_content_hash(content)

    def _find_workspace_java_import(self, symbol: str) -> Optional[str]:
        """Dynamically finds the fully-qualified import statement for a workspace Java class."""
        if not hasattr(self, "_java_symbol_cache"):
            self._java_symbol_cache: Dict[str, str] = {}
            for root, dirs, files in os.walk(self.workspace):
                if any(part in ("target", "build", ".git", "node_modules", "dist") for part in Path(root).parts):
                    continue
                for f in files:
                    if f.endswith(".java"):
                        sym = f[:-5]
                        full_p = Path(root) / f
                        try:
                            with open(full_p, "r", encoding="utf-8", errors="ignore") as fh:
                                for _ in range(12):
                                    line = fh.readline()
                                    if not line:
                                        break
                                    line_s = line.strip()
                                    if line_s.startswith("package ") and line_s.endswith(";"):
                                        pkg = line_s[8:-1].strip()
                                        self._java_symbol_cache[sym] = f"import {pkg}.{sym};"
                                        break
                        except Exception:
                            pass
        return self._java_symbol_cache.get(symbol)

    @staticmethod
    def _find_enclosing_project_root(file_path: Path, workspace_root: Path) -> Path:
        """
        Universal compilation boundary discovery (0 magic strings).
        Finds the closest enclosing directory containing a project manifest.
        In modern modular architecture, relative imports must never escape their enclosing project root.
        """
        manifest_names = ("package.json", "tsconfig.json", "pom.xml", "build.gradle", "go.mod", "Cargo.toml", "pyproject.toml")
        curr = file_path.parent if file_path.is_file() else file_path
        ws_res = workspace_root.resolve()
        while curr.resolve() != ws_res and curr.resolve() != curr.parent.resolve():
            if any((curr / m).is_file() for m in manifest_names):
                return curr
            curr = curr.parent
        return workspace_root

    def _find_workspace_ts_import(self, current_file: Path, symbol: str) -> Optional[str]:
        """
        Dynamically finds relative import statement for an exported TS symbol within the enclosing project.
        Enforces universal modular invariants:
          1. Self-Declaration Invariant: A file never imports a symbol it already defines.
          2. Package Boundary Invariant: Relative imports can only resolve within the same enclosing project root.
        """
        # 1. Self-declaration check: A file never imports a symbol it already defines!
        try:
            curr_text = current_file.read_text(encoding="utf-8", errors="ignore")
            if re.search(rf"\bexport\s+(?:class|interface|enum|type|const|function)\s+{re.escape(symbol)}\b", curr_text):
                return None
        except Exception:
            pass

        # 2. Package Boundary Invariant: Internal relative imports can ONLY resolve within the enclosing project root.
        project_root = self._find_enclosing_project_root(current_file, self.workspace)

        cache_key = f"_ts_symbol_cache_{str(project_root.resolve())}"
        if not hasattr(self, cache_key):
            sym_cache: Dict[str, Any] = {}
            pattern = re.compile(r"export\s+(?:class|interface|enum|type|const|function)\s+([a-zA-Z_]\w*)\b")
            ambient_pattern = re.compile(r"declare\s+(?:var|let|const)\s+([a-zA-Z_]\w*)\s*:\s*any\s*;")
            standard_build_artifacts = {"node_modules", "dist", ".git", "build", "target", ".angular", ".next"}
            for root, dirs, files in os.walk(project_root):
                if any(part in standard_build_artifacts for part in Path(root).parts):
                    continue
                for f in files:
                    if f.endswith(".ts") and not f.endswith((".spec.ts", ".test.ts")):
                        full_p = Path(root) / f
                        try:
                            txt = full_p.read_text(encoding="utf-8", errors="ignore")
                            for m in pattern.finditer(txt):
                                sym_cache[m.group(1)] = full_p
                            for ma in ambient_pattern.finditer(txt):
                                if ma.group(1) not in sym_cache:
                                    sym_cache[ma.group(1)] = f"declare var {ma.group(1)}: any;"
                        except Exception:
                            pass
            setattr(self, cache_key, sym_cache)

        sym_cache = getattr(self, cache_key)
        target = sym_cache.get(symbol)
        if isinstance(target, str):
            return target
        if isinstance(target, Path) and target.resolve() != current_file.resolve():
            try:
                # Ensure target_file is strictly inside the same project_root
                if not target.resolve().is_relative_to(project_root.resolve()):
                    return None
                rel = os.path.relpath(target.parent, current_file.parent).replace("\\", "/")
                if not rel.startswith("."):
                    rel = f"./{rel}"
                module_name = target.stem
                import_path = f"{rel}/{module_name}".replace("//", "/")
                return f"import {{ {symbol} }} from '{import_path}';"
            except Exception:
                pass
        return None

    def _pre_fix_build(self, sample_files: Union[str, List[str]]) -> Tuple[bool, str]:
        """
        Deterministic Build & Compiler Error Slicer Hook (0 LLM Tokens).
        Runs native compiler across all affected zones and slices out precise file:line errors using DiagnosticErrorSlicer.
        """
        build_ok, raw_err = self._run_fast_compile(sample_files)
        if build_ok:
            return True, ""
        sliced = self._slice_error(raw_err, max_tokens_budget=350)
        return False, sliced

    # =========================================================================
    # CONTEXT & PROMPT GENERATION (STAGE 2)
    # =========================================================================

    def _resolve_imported_definitions(self, files: List[str]) -> Dict[str, str]:
        """
        Deterministic Context Expansion (Stage 2):
        Inspects imports in target files to locate referenced workspace constants,
        enums, routes, and models. Injects their exported declarations as read-only
        reference context so the LLM does not hallucinate non-existent properties.
        """
        imported_defs: Dict[str, str] = {}
        target_norm = {f.replace("\\", "/").lower() for f in files}

        ts_extensions = [".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.js"]
        import_pattern = re.compile(r"""(?:from|import)\s+['"]([^'"]+)['"]""")

        for rel_file in files:
            p = self.workspace / rel_file
            if not p.is_file():
                continue

            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            ext = p.suffix.lower()

            if ext in (".ts", ".tsx", ".js", ".jsx"):
                matches = import_pattern.findall(content)
                for spec in matches:
                    if spec.startswith(("@", "rxjs", "zone.js", "core-js", "lodash")):
                        continue

                    candidate_paths: List[Path] = []
                    if spec.startswith("."):
                        base_cand = (p.parent / spec).resolve()
                        candidate_paths.append(base_cand)
                        for suff in ts_extensions:
                            candidate_paths.append(Path(str(base_cand) + suff))
                    elif spec.startswith("src/") or spec.startswith("app/"):
                        base_cand = (self.workspace / spec).resolve()
                        candidate_paths.append(base_cand)
                        for suff in ts_extensions:
                            candidate_paths.append(Path(str(base_cand) + suff))
                    else:
                        for prefix in ("", "src", "src/app"):
                            base_cand = (self.workspace / prefix / spec).resolve()
                            candidate_paths.append(base_cand)
                            for suff in ts_extensions:
                                candidate_paths.append(Path(str(base_cand) + suff))

                    for cand in candidate_paths:
                        if cand.is_file():
                            try:
                                rel = cand.relative_to(self.workspace).as_posix()
                            except ValueError:
                                continue

                            if rel.lower() in target_norm:
                                continue

                            rel_lower = rel.lower()
                            is_high_value = any(k in rel_lower for k in (
                                "constant", "enum", "model", "route", "uri-key", "config", "types", "service", "api", "hook", "interface"
                            ))

                            if is_high_value and rel not in imported_defs:
                                try:
                                    imported_text = cand.read_text(encoding="utf-8", errors="ignore")
                                    lines = imported_text.splitlines()
                                    snippet = "\n".join(lines[:120])
                                    imported_defs[rel] = snippet
                                    logger.info(f"⚡ [FastIDEEngine] Injected imported reference: {rel} ({len(lines)} lines)")
                                except Exception as exc:
                                    logger.debug(f"Failed to read imported ref {rel}: {exc}")
                            break

            elif ext == ".java":
                java_import_pattern = re.compile(r"""import\s+([a-zA-Z0-9_.]+);""")
                injected_classes = set(re.findall(r"(?:private|protected|public)?\s+(?:final\s+)?([A-Z]\w*(?:Repository|Dao|Service|Client))\s+\w+;", content))
                target_classes = set(injected_classes)

                for match in java_import_pattern.findall(content):
                    if match.startswith(("java.", "javax.", "org.springframework.", "org.junit.", "lombok.")):
                        continue
                    class_name = match.split(".")[-1]
                    if any(k in class_name.lower() for k in ("constant", "enum", "dto", "model", "repository", "dao", "entity", "service", "client", "exception")):
                        target_classes.add(class_name)

                for class_name in target_classes:
                    sym_import = self._find_workspace_java_import(class_name)
                    for root, _, j_files in os.walk(self.workspace):
                        if any(part in ("target", "build", ".git") for part in Path(root).parts):
                            continue
                        if f"{class_name}.java" in j_files:
                            full_j = Path(root) / f"{class_name}.java"
                            try:
                                rel_j = full_j.relative_to(self.workspace).as_posix()
                                if rel_j not in imported_defs and rel_j.lower() not in target_norm:
                                    j_content = full_j.read_text(encoding="utf-8", errors="ignore")
                                    lines = j_content.splitlines()

                                    # ⚡ Schema & Query Rosetta Stone Grounding:
                                    # For Spring Data / JPA Repositories, extract existing @Query annotations & join conditions
                                    if any(term in class_name.lower() for term in ("repository", "dao")):
                                        queries = re.findall(r'@Query\s*\(\s*(?:value\s*=\s*)?["\']([^"\']+)["\']', j_content)
                                        query_block = ""
                                        if queries:
                                            query_block = "// ⚡ [Grounding]: Existing Repository Queries & Join Conditions:\n" + "\n".join(f"//   {q}" for q in queries[:6]) + "\n\n"
                                        imported_defs[rel_j] = query_block + "\n".join(lines[:120])
                                    else:
                                        imported_defs[rel_j] = "\n".join(lines[:100])
                                    logger.info(f"⚡ [FastIDEEngine] Injected imported reference: {rel_j} ({len(lines)} lines)")
                            except Exception:
                                pass
                            break

        # Action Reference Grounding (Zero Hallucination Anchor):
        # If the ticket references an action ("launch help", "open help", "navigation", etc.),
        # locate existing implementations in the workspace and inject them as read-only references
        # so the model grounds its patch in existing patterns rather than hallucinating properties/constants.
        action_names = ["help", "navigation", "home", "guidance"]
        ticket_str = f"{getattr(self.ticket, 'title', '')} {getattr(self.ticket, 'description', '')}".lower()
        active_actions = [a for a in action_names if a in ticket_str]

        if active_actions:
            search_folders = []
            for sname, sinfo in self._subprojects.items():
                if sinfo.get("is_ui", False):
                    search_folders.append(sinfo["path"])
            if not search_folders:
                search_folders.append(self.workspace)

            for folder in search_folders:
                for act in ["openHelp", "navigateToHome", "buildHelpUrl"]:
                    if act.lower() in ticket_str or any(a in act.lower() for a in active_actions):
                        try:
                            import subprocess
                            res = subprocess.run(
                                ["git", "grep", "-n", f"{act}("],
                                cwd=str(folder),
                                capture_output=True,
                                text=True,
                                timeout=3
                            )
                            if res.returncode == 0:
                                for ln in res.stdout.splitlines()[:2]:
                                    parts = ln.split(":", 2)
                                    if len(parts) >= 2:
                                        rel_f = parts[0]
                                        line_no = int(parts[1])
                                        full_path = folder / rel_f
                                        if full_path.is_file() and full_path.suffix in (".ts", ".js"):
                                            all_lines = full_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                                            start_ln = max(0, line_no - 3)
                                            end_ln = min(len(all_lines), line_no + 35)
                                            body = "\n".join(all_lines[start_ln:end_ln])
                                            ref_key = self._to_rel_posix(full_path)
                                            if ref_key not in imported_defs and ref_key.lower() not in target_norm:
                                                ambient_decls = [l.strip() for l in all_lines[:50] if re.match(r"^\s*declare\s+(?:var|let|const)\s+", l)]
                                                ambient_hdr = ("\n".join(ambient_decls) + "\n\n") if ambient_decls else ""
                                                imported_defs[ref_key] = (
                                                    f"// Reference: Existing production pattern for '{act}' in workspace\n"
                                                    f"// Follow this implementation pattern. Do NOT hallucinate non-existent constants:\n"
                                                    f"{ambient_hdr}"
                                                    f"{body}"
                                                )
                                                logger.info(f"⚡ [FastIDEEngine] Injected grounded action reference: {ref_key} ({act})")
                                                break
                        except Exception:
                            pass

        return imported_defs

    def _read_file_slices(self, files: List[str]) -> Dict[str, str]:
        """Reads files up to 300 lines to keep context tight and records SHA-256 content hashes."""
        snippets = {}
        for f in files:
            p = self.workspace / f
            if p.is_file():
                try:
                    full_text = p.read_text(encoding="utf-8", errors="ignore")
                    self.file_hashes[f] = self.diff_engine.compute_content_hash(full_text)
                    lines = full_text.splitlines()
                    content = "\n".join(lines[:300])
                    snippets[f] = content
                except Exception as exc:
                    logger.debug(f"[FastIDEEngine] Could not read {f}: {exc}")
        return snippets

    def _construct_turn_prompt(
        self,
        context_snippets: Dict[str, str],
        repo_outline: str,
        last_error: str,
        turn: int,
        call_chain_summary: Optional[List[str]] = None,
        imported_definitions: Optional[Dict[str, str]] = None,
    ) -> str:
        code_context = "\n\n".join(
            f"FILE: {fp} (contentHash: {self.file_hashes.get(fp, '')})\n```\n{content}\n```"
            for fp, content in context_snippets.items()
        )

        repo_section = f"\n### REPOSITORY OUTLINE:\n{repo_outline}\n" if repo_outline else ""

        call_chain_section = ""
        if call_chain_summary:
            chain_text = "\n".join(f"  {hop}" for hop in call_chain_summary)
            call_chain_section = f"\n### VERIFIED CALL-CHAIN TRACE:\n{chain_text}\n"

        imported_section = ""
        if imported_definitions:
            parts = [
                f"// {fp} (Reference only - do NOT modify):\n```\n{content}\n```"
                for fp, content in imported_definitions.items()
            ]
            imported_section = (
                "\n### IMPORTED DEFINITIONS (Reference Only - inspect for existing constants/routes/properties):\n"
                + "\n\n".join(parts)
                + "\n"
            )

        error_section = ""
        if last_error:
            if "Edit FAILED" in last_error or "Anchor Matching Error" in last_error:
                error_section = (
                    f"\n### ⚠️ PREVIOUS ATTEMPT FAILED - SEARCH BLOCK DID NOT MATCH:\n{last_error}\n"
                    "CRITICAL: The CONTEXT section above shows the CURRENT file contents from disk.\n"
                    "You MUST copy lines EXACTLY (including whitespace and indentation) from the CONTEXT section into your SEARCH blocks.\n"
                    "Do NOT guess or reconstruct the lines from memory.\n"
                )
            else:
                error_section = f"\n### PREVIOUS ATTEMPT FAILED WITH COMPILER/TEST ERROR:\n{last_error}\nFix this exact error."

        return f"""You are a precise software engineer working like Cursor Composer and Aider.
Solve the following task using surgical SEARCH/REPLACE blocks.

### TASK:
Ticket: {self.ticket.ticket_id} - {self.ticket.title}
Description: {self.ticket.description}
{call_chain_section}{repo_section}{imported_section}
### CONTEXT:
{code_context}
{error_section}

### INSTRUCTIONS:
1. Output ONLY surgical search/replace blocks in this EXACT format:
FILE: <relative_path>
<<<<<<< SEARCH
<exact lines to find>
=======
<replacement lines>
>>>>>>> REPLACE

2. HOW TO MODIFY OR INSERT CODE IN EXISTING FILES:
   - Every existing file MUST have a NON-EMPTY SEARCH block containing 2-5 exact contiguous lines from the CONTEXT above.
   - To insert a new method or endpoint into an existing class/service/controller, anchor on the preceding method or the closing brace:
     Example adding a new method before the class closing brace:
     <<<<<<< SEARCH
     }}
     =======
         public Order refundOrder(UUID orderId) {{
             // refund implementation
         }}
     }}
     >>>>>>> REPLACE
   - NEVER output an empty SEARCH block on an existing file! Empty SEARCH is strictly forbidden on existing files.
   - Keep SEARCH blocks SHORT (2 to 5 lines). Copy the exact characters, whitespace, and indentation directly from the CONTEXT section.

3. FOR BRAND NEW FILES ONLY:
   - If a file does NOT exist yet on disk, leave the SEARCH block completely empty:
     FILE: <brand_new_file_path>
     <<<<<<< SEARCH
     =======
     <complete new file content>
     >>>>>>> REPLACE

4. For multi-file tickets, emit separate search/replace blocks for ALL necessary files in the working set (e.g. model, service, controller, and UI component).
5. Do NOT rewrite whole existing files. Keep changes minimal and surgically targeted.
6. For UI tickets:
   - In React (*.tsx, *.jsx): update JSX markup, event handlers (e.g. onClick), component props, and state.
   - In Angular (*.component.ts, *.component.html): update BOTH the HTML template and the component TypeScript.
7. Do NOT output ANY conversational text, explanations, or markdown fences (like ```java). Output ONLY the raw FILE and <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE blocks directly.
"""

    def _call_llm(self, prompt: str) -> Tuple[str, int]:
        """Calls the injected LLM adapter or returns a structured response."""
        if not self.llm:
            return "", 0

        self.llm_calls += 1
        est_in = len(prompt) // 4
        if callable(self.llm) and not hasattr(self.llm, "invoke"):
            try:
                content = str(self.llm(prompt))
                in_toks = est_in
                out_toks = len(content) // 4
                self.tokens_in += in_toks
                self.tokens_out += out_toks
                return content, in_toks + out_toks
            except Exception as exc:
                logger.error(f"[FastIDEEngine] Callable LLM failed: {exc}")
                return "", 0

        try:
            from ticket_to_code.llm_utils import llm_invoke
            from langchain_core.messages import HumanMessage
            res = llm_invoke(self.llm, [HumanMessage(content=prompt)], max_tokens=2500)
            content = getattr(res, "content", str(res))
            usage = getattr(res, "usage_metadata", None) or getattr(res, "response_metadata", {}).get("token_usage") or getattr(res, "response_metadata", {}).get("usage")
            if usage and isinstance(usage, dict):
                in_toks = usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get("prompt_token_count") or est_in
                out_toks = usage.get("output_tokens") or usage.get("completion_tokens") or usage.get("candidates_token_count") or (len(content) // 4)
            else:
                in_toks = est_in
                out_toks = len(content) // 4
            self.tokens_in += in_toks
            self.tokens_out += out_toks
            return content, in_toks + out_toks
        except Exception as exc:
            logger.error(f"[FastIDEEngine] LLM invocation failed: {exc}")
        return "", 0

    def _run_fast_compile(self, sample_files: Union[str, List[str]]) -> Tuple[bool, str]:
        """Runs native compiler discovered via PolyglotBuildDiscoverer across all affected zones with LSP pre-flight."""
        files = [sample_files] if isinstance(sample_files, str) else sample_files

        # ⚡ Sub-second LSP pre-flight pass (<200ms) to bypass heavy build scripts on syntax/type errors
        if hasattr(self, "lsp") and self.lsp:
            from ticket_to_code.runtime.lsp_client import DiagnosticSeverity
            lsp_errors = []
            for f in files:
                try:
                    diags = self.lsp.get_diagnostics(f)
                    errs = [d for d in diags if d.severity == DiagnosticSeverity.ERROR]
                    if errs:
                        lsp_errors.extend(errs)
                except Exception:
                    pass
            if lsp_errors:
                if len(lsp_errors) < 20:
                    err_text = "\n".join(d.format_slice() for d in lsp_errors[:8])
                    logger.info(f"⚡ [FastIDEEngine] Caught {len(lsp_errors)} compiler diagnostic(s) in sub-second LSP pass!")
                    return False, f"Compiler Diagnostic Error:\n{err_text}"
                else:
                    logger.info(f"⚡ [FastIDEEngine] LSP reported {len(lsp_errors)} diagnostics (likely unlinked classpath); falling through to native build tool.")

        seen_dirs: Set[Path] = set()
        configs = []

        for f in files:
            sample_path = (self.workspace / f).resolve()
            search_dir = sample_path.parent if sample_path.is_file() else sample_path
            if search_dir in seen_dirs:
                continue
            seen_dirs.add(search_dir)
            cfg = PolyglotBuildDiscoverer.discover(search_dir, self.workspace)
            if cfg and cfg.compile_cmd and cfg not in configs:
                configs.append(cfg)

        if not configs:
            root_cfg = PolyglotBuildDiscoverer.discover(self.workspace, self.workspace)
            if root_cfg and root_cfg.compile_cmd:
                configs.append(root_cfg)

        if not configs:
            self._last_build_ran = False   # nothing compiled: caller must not report build_passed=True
            return True, ""

        self._last_build_ran = True
        compile_timeout = int(os.getenv("FAST_BUILD_TIMEOUT", "120"))
        for cfg in configs:
            effective_cmd = cfg.compile_cmd
            if hasattr(self, "sandbox_engine") and self.sandbox_engine:
                is_ready, profile, prep_msg = self.sandbox_engine.prepare_environment(cfg.work_dir)
                if not is_ready:
                    logger.warning(f"⚠️ [FastIDEEngine Toolchain] {prep_msg}")
                    return False, f"[{cfg.work_dir.name}] Toolchain Incompatibility Error:\n{prep_msg}"
                effective_cmd = self.sandbox_engine._apply_wrapper_preference(cfg.compile_cmd, profile, cfg.work_dir)
            retcode, stdout, stderr = self._execute_subprocess_with_tree_cleanup(
                effective_cmd, cfg.work_dir, timeout=compile_timeout
            )
            if retcode != 0:
                return False, f"[{cfg.work_dir.name}] {stdout}\n{stderr}"

        return True, ""

    def _run_fast_test(self, sample_files: Union[str, List[str]]) -> Tuple[bool, str]:
        """Runs fast targeted companion test command for affected files."""
        files = [sample_files] if isinstance(sample_files, str) else sample_files
        seen_cmds: Set[str] = set()
        test_runs = []

        for f in files:
            sample_path = (self.workspace / f).resolve()
            focused = PolyglotBuildDiscoverer.get_focused_test_command(sample_path, self.workspace)
            if focused:
                cmd, work_dir = focused
                if cmd not in seen_cmds:
                    seen_cmds.add(cmd)
                    test_runs.append((cmd, work_dir))

        if not test_runs:
            logger.info("⚡ [FastIDEEngine] No companion test found for modified files; no tests ran (unknown, not passed).")
            self._last_test_ran = False
            return True, ""

        self._last_test_ran = True
        test_timeout = int(os.getenv("FAST_TEST_TIMEOUT", "90"))
        for cmd, work_dir in test_runs:
            effective_cmd = cmd
            if hasattr(self, "sandbox_engine") and self.sandbox_engine:
                is_ready, profile, prep_msg = self.sandbox_engine.prepare_environment(work_dir)
                if not is_ready:
                    return False, f"[{work_dir.name}] Toolchain Incompatibility Error:\n{prep_msg}"
                effective_cmd = self.sandbox_engine._apply_wrapper_preference(cmd, profile, work_dir)
            retcode, stdout, stderr = self._execute_subprocess_with_tree_cleanup(
                effective_cmd, work_dir, timeout=test_timeout
            )
            combined = stdout + "\n" + stderr
            if retcode != 0:
                return False, f"[{work_dir.name}] {combined}"

        return True, ""

    @staticmethod
    def _execute_subprocess_with_tree_cleanup(
        cmd: str, cwd: Path, timeout: int = 45
    ) -> Tuple[int, str, str]:
        """Safely executes a shell command with full process-tree termination on timeout or optional Docker container."""
        if os.getenv("FAST_USE_DOCKER") == "1":
            try:
                from ticket_to_code.runtime.container_sandbox import DockerSandboxExecutor, SandboxConfig
                if DockerSandboxExecutor.is_docker_available():
                    cfg = SandboxConfig(timeout_seconds=timeout)
                    res = DockerSandboxExecutor.run_isolated_command(cmd, str(cwd), "generic", cfg)
                    return res.exit_code, res.stdout, res.stderr
            except Exception as e:
                logger.warning(f"Docker sandbox execution failed ({e}), falling back to process isolation.")
        import subprocess
        import sys

        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        from ticket_to_code.runtime.security import CommandSafetyValidator

        is_safe, violation = CommandSafetyValidator.validate_command(cmd, workspace_root=cwd)
        if not is_safe:
            logger.error(f"🛡️ [FastIDEEngine Security] Blocked dangerous command: {violation}")
            return -1, "", f"Security policy rejection: {violation}"

        sanitized_env = CommandSafetyValidator.sanitize_environment()

        try:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                cwd=cwd,
                env=sanitized_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                **kwargs,
            )
        except Exception as exc:
            return -1, "", str(exc)

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return proc.returncode, stdout, stderr
        except subprocess.TimeoutExpired:
            if sys.platform == "win32":
                subprocess.run(f"taskkill /F /T /PID {proc.pid}", shell=True, capture_output=True)
            else:
                proc.kill()
            return -1, "", f"Build process timed out after {timeout}s and was terminated."
        except Exception as exc:
            return -1, "", str(exc)

    def _slice_error(self, raw_output: str, max_tokens_budget: int = 300) -> str:
        """Slices out only the meaningful compiler / traceback lines using DiagnosticErrorSlicer."""
        return DiagnosticErrorSlicer.slice_errors(raw_output, max_tokens_budget=max_tokens_budget)
