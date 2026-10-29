import os
import re
import json
import logging
from typing import Tuple, List, Any, Optional
from pathlib import Path
from ticket_to_code.models import TaskType
from ticket_to_code.runtime.llm_factory import LLMRegistry

logger = logging.getLogger(__name__)
# ----------------------------------------------------------------------
# ARCHITECTURE NOTE: Plan Gating Pipeline
# We replaced mathematical gates with Contextual LLM Judgment + Mechanical Grounding Post-Checks:
# Pass 1.5: Module Boundary Enforcement (Before LLM). Rejects tasks that violate architecture dependency rules.
# Pass 2/3: Contextual LLM Judgment + Mechanical Grounding Post-Checks:
# 1. Text Match: Ensures if the LLM says "this file has a hardcoded string X", we normalize and verify it exists.
# 2. Graph Match: Ensures if the LLM cites architectural relevance, the graph provider actually found that edge.
# 3. Implicit Structural: Ensures new files with an architectural need (no string to replace) are linked to a real consumer with a mechanically verified structural marker (e.g. "@RestController").
# 4. CREATE Consumer Check: Ensures proposed new files are structurally demanded by the content of accepted consumers.
# 5. Rationale Coherence: Ensures files are judged against the Planner's Stated Strategy, preventing hallucinations.
# Pass 2.5: Deep architectural scan for forbidden imports/references based on module_graph boundaries.
#
# Known Limitations:
# - Reject-Dependency Check (_task_references_path): Because we cannot inspect the *proposed* code of a MODIFY
#   task before it is generated, we must rely on regex word-boundary searches over the planner's `description`
#   and `explicit_planner_justification` to detect if an accepted task references a rejected CREATE task.
#   This means the check will ONLY fire if the LLM happened to explicitly name the exact filename in its prose.
#   If it writes generically ("this file needs updating"), the dependency will be silently missed. This is an
#   accepted, bounded gap for now.
# - DELETE/RENAME tasks are currently routed through the same LLM judgment prompt as MODIFY, which asks about
#   "modifying" the file. This is semantically imprecise for deletions and is a known follow-up item, not
#   fixed here.
# ----------------------------------------------------------------------


def _is_blocked_path(path: str) -> bool:
    low = path.lower().replace("\\", "/")
    if "package-lock.json" in low or "yarn.lock" in low or "pnpm-lock.yaml" in low: return True
    if "/dist/" in low or "/build/" in low or "/out/" in low: return True
    if "/node_modules/" in low or "/.git/" in low: return True
    if low.endswith(".svg") or low.endswith(".png") or low.endswith(".jpg"): return True
    return False

def passes_mechanical_prefilter(task, ranked_files, discovered_files, workspace_path: str = None) -> Tuple[bool, dict]:
    task_type = getattr(task.task_type, "value", str(task.task_type)).lower()
    if task_type == "create":
        return True, {"score": 1.0}

    norm = task.file_path.replace("\\", "/").lower()
    ranked = next((r for r in ranked_files if getattr(r, "file_path", "").replace("\\", "/").lower() == norm), None)

    score = 0.0
    if ranked:
        score = float(getattr(ranked, "final_score", 0.0))
    else:
        cand = next((d for d in discovered_files if d.get("path", "").replace("\\", "/").lower() == norm), None)
        if cand:
            score = float(cand.get("confidence", 0.0))

    if score < 0.05:
        if workspace_path:
            workspace_root = Path(workspace_path).resolve()
            candidate = Path(task.file_path)
            full_path = candidate if candidate.is_absolute() else (workspace_root / candidate)
            try:
                resolved = full_path.resolve()
                if resolved.is_file() and (resolved == workspace_root or workspace_root in resolved.parents):
                    return True, {"score": 1.0, "reason": "File exists on disk inside workspace (100% grounded)"}
            except Exception:
                pass
        return False, {"score": score, "reason": "File is completely ungrounded (score < 0.05)"}

    return True, {"score": score}

def normalize_text(text: str) -> str:
    return re.sub(r'[\s\'"`]', '', text).lower()

def _extract_json_payload(text: str) -> str:
    """Strip Markdown code fences (```json ... ``` or plain ``` ... ```) around a JSON
    payload without corrupting the payload itself.

    NOTE: do NOT use str.strip("```json") for this — str.strip(chars) removes any of the
    *individual characters* in the argument from each end of the string, not the literal
    substring "```json". That silently eats legitimate trailing characters (e.g. a payload
    ending in the letter 'n' or 'o') and is not safe for arbitrary LLM JSON output.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r'^```[a-zA-Z]*\s*', '', stripped)
        stripped = re.sub(r'```\s*$', '', stripped)
    return stripped.strip()

def verify_grounding_fact_text(fact: str, file_path: str, workspace_path: str, symbols: Optional[List[str]] = None) -> bool:
    try:
        full_path = os.path.join(workspace_path, file_path)
        if not os.path.exists(full_path):
            return False
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        norm_content = normalize_text(content)
        if normalize_text(fact) in norm_content:
            return True
        # Symbol-level grounding backstop:
        # If fact was slightly paraphrased, check if any referenced symbol exists in file
        if symbols:
            generic_symbols = {
                "class", "interface", "service", "controller", "component",
                "function", "method", "string", "list", "object", "model",
            }
            for sym in symbols:
                clean_sym = sym.split("::")[-1].split(".")[-1].strip()
                if not clean_sym:
                    continue
                low_sym = clean_sym.lower()
                if len(clean_sym) < 4 or low_sym in generic_symbols:
                    continue
                # Require a token-level hit to avoid loose substring acceptance.
                if re.search(rf'(?<![A-Za-z0-9_]){re.escape(clean_sym)}(?![A-Za-z0-9_])', content, re.IGNORECASE):
                    return True
        # File stem / class name backstop (e.g. AddMembersComponent in add-members.component.ts)
        # Keep this strict so it only applies to component-style files with class markers.
        low_path = file_path.replace("\\", "/").lower()
        is_component_file = ".component." in low_path
        has_class_marker = ("export class " in content) or ("class " in content)
        stem = Path(file_path).stem.replace("-", "").replace("_", "").lower()
        if is_component_file and has_class_marker and stem and len(stem) > 5 and stem in norm_content:
            return True
        return False
    except Exception as e:
        logger.error(f"Error reading file {file_path} for text verification: {e}")
        return False

def _task_is_additive(task) -> bool:
    """True when the task is adding new code to an existing file (not editing existing code)."""
    desc = (getattr(task, "description", "") or "").lower()
    return any(kw in desc for kw in ("add ", "implement ", "create ", "introduce ", "expose ", "insert ", "add new"))


def _primary_type_in_file(file_path: str, workspace_path: str) -> bool:
    """True when the primary type name derived from the file path stem exists as a token in the file.
    e.g. ParticipatingMembersRepository.java → looks for token 'ParticipatingMembersRepository'.
    This is the correct grounding proof for additive tasks: the class/interface that will
    house the new method must already exist in this file."""
    try:
        stem = Path(file_path).stem  # e.g. ParticipatingMembersRepository or add-members.component
        # Convert kebab/snake to PascalCase: add-members.component → AddMembersComponent
        type_name = ''.join(w.capitalize() for w in re.split(r'[-_.]', stem))
        if len(type_name) < 4:
            return False
        full_path = os.path.join(workspace_path, file_path) if workspace_path else file_path
        if not os.path.exists(full_path):
            return False
        content = Path(full_path).read_text(encoding='utf-8', errors='ignore')
        return bool(re.search(rf'(?<![A-Za-z0-9_]){re.escape(type_name)}(?![A-Za-z0-9_])', content))
    except Exception:
        return False


def verify_grounding_fact_graph(fact: str, file_path: str, evidence_items) -> bool:
    norm_fact = normalize_text(fact)
    norm_file = file_path.replace("\\", "/").lower()

    for item in evidence_items:
        item_path = getattr(item, "file_path", "").replace("\\", "/").lower()
        if item_path == norm_file or norm_file in item_path:
            snippet = getattr(item, "content_snippet", "")
            details = getattr(item, "details", "")
            combined = snippet + " " + details
            if combined and norm_fact in normalize_text(combined):
                return True
    return False

def verify_grounding_fact_structural(fact: str, file_path: str, workspace_path: str) -> bool:
    """Verifies an 'implicit_structural' grounding fact: a real structural marker that
    must genuinely exist in the consumer file. Used when a CREATE task's need is
    architectural (e.g. a new endpoint on an existing controller) rather than replacing
    an existing literal."""
    try:
        full_path = os.path.join(workspace_path, file_path)
        if not os.path.exists(full_path):
            return False
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        fact_clean = fact.strip()
        if not fact_clean or fact_clean not in content:
            return False

        valid_prefixes = (
            "@", "class ", "interface ", "extends ", "implements ",
            "export ", "export class", "function ", "public class", "public interface",
        )
        is_structural = any(fact_clean.startswith(prefix) for prefix in valid_prefixes)

        if not is_structural:
            return False

        return True
    except Exception as e:
        logger.error(f"Error reading file {file_path} for structural verification: {e}")
        return False

# Note: Legacy contextual_llm_judgment_modify and contextual_llm_judgment_create were removed
# because run_gating_logic enforces AI-IDE style mechanical gating (file existence,
# module boundaries, blocked paths) rather than burning redundant LLM calls.


def _task_references_path(task, referenced_task, workspace_path: str = None) -> bool:
    ref_path = referenced_task.file_path.replace("\\", "/")
    ref_basename = os.path.basename(ref_path)

    text_to_search = ""
    if hasattr(task, "description") and task.description:
        text_to_search += task.description + " "
    if hasattr(task, "explicit_planner_justification") and task.explicit_planner_justification:
        text_to_search += task.explicit_planner_justification

    if not text_to_search:
        return False

    text_to_search = text_to_search.lower()

    # 1. Exact full path match (no word boundary needed if path is specific)
    if ref_path.lower() in text_to_search:
        return True

    # 2. Exact basename match with regex word boundary
    # This prevents matching 'version' inside 'conversion' while allowing 'version.ts'
    pattern_basename = r'\b' + re.escape(ref_basename.lower()) + r'\b'
    if re.search(pattern_basename, text_to_search):
        return True

    return False

def validate_plan_consistency(accepted_tasks, rejected_tasks, plan, workspace_path: str = None) -> Tuple[bool, str, List[Any]]:
    if not accepted_tasks:
        return False, "All proposed tasks were rejected by gating logic.", rejected_tasks

    # Fail closed: dropping any write-intent task makes the plan incomplete.
    rejected_write_tasks = []
    for task in rejected_tasks:
        t_type = getattr(task.task_type, "value", str(task.task_type)).lower()
        if t_type in {"modify", "delete", "refactor"}:
            rejected_write_tasks.append(task)

    if rejected_write_tasks:
        return (
            False,
            "Plan is inconsistent: write-intent tasks were rejected by gating; forcing replanning.",
            rejected_write_tasks,
        )

    for task in accepted_tasks:
        for r_task in rejected_tasks:
            if _task_references_path(task, r_task, workspace_path):
                reason = f"Task {task.file_path} depends on rejected task {r_task.file_path}. "
                if getattr(r_task.task_type, "value", str(r_task.task_type)) == "create":
                    reason += f"If {r_task.file_path} is legitimately required, you must explicitly name the consumer {task.file_path} and extract the grounding fact from it."
                return False, reason, [r_task]

    deleted_paths = {t.file_path for t in accepted_tasks if getattr(t.task_type, "value", str(t.task_type)) == "delete"}
    for task in accepted_tasks:
        for d_path in deleted_paths:
            if task.file_path == d_path:
                continue  # don't check a delete task against its own deletion
            if _task_references_path(task, next((t for t in accepted_tasks if t.file_path == d_path), task), workspace_path):
                return False, f"Task {task.file_path} references a file being deleted ({d_path}).", [task]

    return True, "", []

_workspace_java_cache: dict = {}

def _get_workspace_java_classes(workspace_path: str) -> dict:
    global _workspace_java_cache
    key = str(Path(workspace_path).resolve())
    if key in _workspace_java_cache:
        return _workspace_java_cache[key]

    cache = {}
    for path in Path(workspace_path).rglob("*"):
        if path.is_file() and path.suffix in (".java", ".ts"):
            if "node_modules" in path.parts or "target" in path.parts or ".git" in path.parts:
                continue
            basename = path.stem
            try:
                rel_path = path.relative_to(workspace_path)
            except ValueError:
                rel_path = path
            cache.setdefault(basename, []).append(str(rel_path).replace("\\", "/"))

    _workspace_java_cache[key] = cache
    return cache

def pass_2_5_boundary_check(task, referenced_symbols: list, g_fact: str, reason: str, module_graph, workspace_path: str) -> tuple[bool, str]:
    if not module_graph:
        return True, ""

    java_classes = _get_workspace_java_classes(workspace_path)

    # 1. Start with structured LLM symbols
    symbols_to_check = set()
    for sym in referenced_symbols:
        sym_base = sym.replace(".java", "").replace(".ts", "").split(".")[-1]
        symbols_to_check.add(sym_base)

    # 2. Add prose-scan backstop from g_fact, reason, and task description
    text_to_search = (str(g_fact) + " " + str(reason) + " " + str(task.description)).replace("\\", " ")
    prose_words = re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b', text_to_search)
    for word in prose_words:
        if word in java_classes:
            symbols_to_check.add(word)

    for symbol_base in symbols_to_check:
        if symbol_base in java_classes:
            candidates = java_classes[symbol_base]

            allowed_candidate_exists = False
            forbidden_candidates = []

            for target_path in candidates:
                if module_graph.can_import(task.file_path, target_path):
                    allowed_candidate_exists = True
                else:
                    target_module = module_graph.get_module_for_file(target_path)
                    forbidden_candidates.append(target_module)

            if len(candidates) > 0:
                if not allowed_candidate_exists:
                    # ALL candidates for this symbol are in forbidden modules
                    # It's an unambiguous violation
                    tmod = forbidden_candidates[0] if forbidden_candidates else "unknown"
                    return False, f"REJECTED: Pass 2.5 Architectural violation (Task references '{symbol_base}' which belongs to forbidden module '{tmod}')"
                else:
                    # Ambiguous case: some candidates are allowed, some forbidden.
                    # We only reject if the text explicitly disambiguates to a forbidden module's package path.
                    for target_path in candidates:
                        if not module_graph.can_import(task.file_path, target_path):
                            target_module = module_graph.get_module_for_file(target_path)
                            target_path_norm = target_path.replace("\\", "/")
                            if "src/main/java/" in target_path_norm:
                                pkg_path = target_path_norm.split("src/main/java/")[1].replace("/", ".")
                                pkg_path = pkg_path.replace(".java", "")
                                if pkg_path in text_to_search:
                                    return False, f"REJECTED: Pass 2.5 Architectural violation (Task explicitly references forbidden symbol '{pkg_path}' from module '{target_module}')"

                    logger.info(f"Pass 2.5: Ambiguous symbol '{symbol_base}' referenced, treated as inconclusive.")

    return True, ""

def run_gating_logic(
    plan,
    discovered_files,
    ranked_files,
    evidence_items,
    workspace_path,
    ticket_text,
    scope_enforcer=None,
) -> Tuple[List[Any], List[Any], bool, str, List[dict]]:
    """
    AI-IDE style gating (matches Cursor / Claude Code / Copilot Workspace behaviour):

    The planner already read the actual file contents when building the plan.
    That reading IS the grounding. We do NOT re-validate with a second LLM call.

    Rules (same as every production AI IDE):
      MODIFY / DELETE / RENAME  →  accept if file exists on disk.
                                   If not on disk, accept if discovery score ≥ 0.3.
                                   Otherwise reject (wrong file path in plan).
      CREATE                    →  always accept (new file, nothing to verify yet).
      Blocked paths             →  always reject (dist/, node_modules/, lock files, …).
      Scope Policy              →  reject if outside allowed_write_roots or in denied_write_roots.
      Architectural boundary    →  reject if module graph explicitly forbids the import.

    Compilation / tests are the real correctness gate — applied after generation.
    """
    rationale = getattr(plan, "rationale", "No rationale provided.")

    accepted = []
    rejected = []
    blocked_tasks = []
    task_logs = []

    # Load module graph once (used only for hard architectural boundary violations).
    try:
        from ticket_to_code.utils.dependency_graph import get_dependency_graph
        module_graph = get_dependency_graph(workspace_path)
    except Exception:
        module_graph = None

    boundary_rejected: set = set()

    # ── Pass 1: blocked paths ─────────────────────────────────────────────────
    active_tasks = []
    for task in plan.tasks:
        t_type = getattr(task.task_type, "value", str(task.task_type))
        if t_type == "read_only":
            continue
        if _is_blocked_path(task.file_path):
            blocked_tasks.append(task)
            task_logs.append({"task": task.file_path, "type": t_type, "status": "rejected", "reason": "blocked_path"})
        else:
            active_tasks.append(task)

    # ── Pass 1.2: Scope Policy Blast-Radius Enforcement (Machine-Enforced) ─────
    if scope_enforcer:
        surviving_active = []
        for task in active_tasks:
            if not scope_enforcer.can_write(task.file_path):
                rejected.append(task)
                task_logs.append({
                    "task": task.file_path,
                    "type": getattr(task.task_type, "value", str(task.task_type)),
                    "status": "rejected",
                    "reason": f"ScopePolicyViolation: Path '{task.file_path}' is outside allowed write boundary or inside denied_write_roots.",
                    "grounding_type": "none",
                    "grounding_fact": "none",
                })
            else:
                surviving_active.append(task)
        active_tasks = surviving_active

    # ── Pass 1.5: module boundary (hard architectural violation only) ──────────
    if module_graph:
        for task in active_tasks:
            for other_task in plan.tasks:
                if task.id == other_task.id:
                    continue
                depends = other_task.id in (getattr(task, "dependencies", None) or [])
                if not depends:
                    depends = _task_references_path(task, other_task, workspace_path)
                if depends and not module_graph.can_import(task.file_path, other_task.file_path):
                    boundary_rejected.add(task.id)
                    rejected.append(task)
                    task_logs.append({
                        "task": task.file_path,
                        "type": getattr(task.task_type, "value", str(task.task_type)),
                        "status": "rejected",
                        "reason": f"Architectural boundary violation: {task.file_path} cannot depend on {other_task.file_path}",
                        "grounding_type": "none", "grounding_fact": "none",
                    })
                    break

    # ── Pass 1.6: Domain Ownership (advisory — log only) ────────────────────────
    # Architecture model provides CONTEXT about which service owns which files.
    # This is advisory intelligence: we log a warning if a task targets files
    # outside the expected bounded context, but we do NOT hard-block the task.
    # Reason: architecture ownership is approximation, not absolute truth.
    # A developer (or AI) may legitimately need to modify cross-service files.
    _domain_warned: set = set()
    try:
        from ticket_to_code.agents.architecture_model import ArchitectureModel
        from ticket_to_code.agents.domain_resolver import DomainResolver
        from ticket_to_code.agents.ownership_resolver import OwnershipResolver

        # Load architecture model
        _arch_model = None
        for _yaml_loc in [
            Path(workspace_path) / "brain" / "knowledge" / "architecture_model.yaml",
            Path(__file__).parent / "config" / "architecture_model.yaml",
        ]:
            if _yaml_loc.exists():
                _arch_model = ArchitectureModel.load_from_yaml(str(_yaml_loc))
                break

        if _arch_model and _arch_model.services:
            # Extract ticket text → domains
            _domain_resolver = DomainResolver(_arch_model)
            # ticket_text is the combined title + description already
            _domain_resolution = _domain_resolver.resolve(ticket_text, "")

            if _domain_resolution.has_resolution:
                _ownership_resolver = OwnershipResolver(_arch_model)
                _all_domains = _domain_resolution.primary_domains + _domain_resolution.secondary_domains

                for task in active_tasks:
                    if task.id in boundary_rejected:
                        continue
                    _ownership = _ownership_resolver.classify(task.file_path, _all_domains)

                    # Advisory only: log the concern but let the task proceed
                    if _ownership.is_hard and _ownership.policy.value == "DO_NOT_MODIFY":
                        _domain_warned.add(task.id)
                        task_logs.append({
                            "task": task.file_path,
                            "type": getattr(task.task_type, "value", str(task.task_type)),
                            "status": "advisory_warning",
                            "reason": (
                                f"Domain ownership advisory: {task.file_path} belongs to "
                                f"'{_ownership.service_name}' but capability '{_ownership.matched_capability}' "
                                f"is owned by '{_ownership.expected_service}'. "
                                f"Task is proceeding — verify this cross-service modification is intentional."
                            ),
                            "grounding_type": "domain_ownership_advisory",
                            "grounding_fact": "; ".join(_ownership.evidence),
                        })
                        logger.warning(
                            f"  🏗️ DOMAIN ADVISORY: {task.file_path} — "
                            f"service={_ownership.service_name}, expected={_ownership.expected_service} "
                            f"(task allowed to proceed)"
                        )
    except Exception as _domain_err:
        logger.warning(f"  🏗️ Domain ownership check failed (non-fatal): {_domain_err}")

    # ── Pass 2: file-existence gate (AI-IDE style) ────────────────────────────
    for task in active_tasks:
        if task.id in boundary_rejected:
            continue


        t_type = getattr(task.task_type, "value", str(task.task_type))

        if t_type == "create":
            # New file — nothing to verify yet; code generation + compilation is the gate.
            accepted.append(task)
            task_logs.append({"task": task.file_path, "type": t_type, "status": "accepted",
                               "reason": "create task — accepted unconditionally", "grounding_type": "create"})
            continue

        # MODIFY / DELETE / RENAME: file must exist on disk (or have strong discovery score).
        full_path = os.path.join(workspace_path, task.file_path) if workspace_path else task.file_path
        file_on_disk = os.path.exists(full_path)

        if file_on_disk:
            accepted.append(task)
            task_logs.append({"task": task.file_path, "type": t_type, "status": "accepted",
                               "reason": "file exists on disk", "grounding_type": "file_exists"})
        else:
            # Fall back to discovery score so high-confidence RAG hits still pass.
            norm = task.file_path.replace("\\", "/").lower()
            ranked = next((r for r in ranked_files if getattr(r, "file_path", "").replace("\\", "/").lower() == norm), None)
            score = float(getattr(ranked, "final_score", 0.0)) if ranked else 0.0
            if score < 0.3:
                cand = next((d for d in discovered_files if d.get("path", "").replace("\\", "/").lower() == norm), None)
                score = float(cand.get("confidence", 0.0)) if cand else 0.0

            if score >= 0.3:
                accepted.append(task)
                task_logs.append({"task": task.file_path, "type": t_type, "status": "accepted",
                                   "reason": f"file not on disk but discovery score {score:.2f} ≥ 0.3",
                                   "grounding_type": "discovery_score"})
            else:
                rejected.append(task)
                task_logs.append({"task": task.file_path, "type": t_type, "status": "rejected",
                                   "reason": f"file not found on disk and discovery score {score:.2f} < 0.3 — likely wrong path"})

    all_rejected = rejected + blocked_tasks
    is_consistent, failure_reason, _ = validate_plan_consistency(accepted, all_rejected, plan, workspace_path)
    return accepted, all_rejected, is_consistent, failure_reason, task_logs
