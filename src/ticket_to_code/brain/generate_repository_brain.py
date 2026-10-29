import os
import sys
import json
import re
import hashlib
import logging
import argparse
import concurrent.futures
import time
from pathlib import Path
from collections import defaultdict

# ── Enrichment budget: brain "why" descriptions are optional gloss. A slow or
# hung LLM endpoint must never stall workflow startup (seen 2026-09-18: one
# folder-description call blocked graph creation for 10+ minutes), so every
# call is time-boxed and the whole enrichment phase has a hard budget.
ENRICH_CALL_TIMEOUT_SECONDS = 45
ENRICH_MAX_CALLS = 25
ENRICH_DEADLINE_SECONDS = 180

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# NOTE: brain_storage / brain_builder are imported lazily inside the heavy full-build path
# (generate_repository_brain) so the lightweight scan/refresh path stays dependency-light and
# unit-testable without the full aviator stack.

logger = logging.getLogger(__name__)

# Sidecar cache that remembers the LLM-authored "why" (business_domain / technical_role /
# core_responsibilities) for each directory, keyed by a content signature of that directory.
# It lets the self-updating brain reuse expensive semantic descriptions across runs and only
# re-ask the LLM for folders whose structure actually changed.
SEMANTIC_CACHE_FILENAME = "brain_semantic_cache.json"
_WHY_FIELDS = ("business_domain", "technical_role", "core_responsibilities")

BRAIN_DIR_NAME = "brain"
BRAIN_KNOWLEDGE_DIR = "knowledge"
BRAIN_FILE_NAME = "generated_directory_brain.json"
LEGACY_BRAIN_DIR = ".agents"
LEGACY_BRAIN_FILE_NAME = "repository_brain.json"


from ticket_to_code.utils.file_filters import STANDARD_SKIP_DIRS as _SKIP_DIRS, is_ignored_path, prune_walk_dirs

_SOURCE_EXTS = {
    ".ts", ".tsx", ".js", ".jsx", ".java", ".cs", ".py", ".go",
    ".html", ".scss", ".css", ".vue", ".svelte",
}
_SYMBOL_PATTERNS = [
    re.compile(r"\bclass\s+([A-Z][A-Za-z0-9_]+)"),
    re.compile(r"\b(?:export\s+)?(?:default\s+)?function\s+([A-Za-z_][A-Za-z0-9_]+)"),
    re.compile(r"\bconst\s+([A-Za-z_][A-Za-z0-9_]+)\s*=\s*(?:\([^)]*\)|[A-Za-z_])"),
    re.compile(r"@Component\(|@Injectable\(|@Directive\("),
    re.compile(r"\bexport\s+(?:const|class|function|interface|type)\s+([A-Za-z_][A-Za-z0-9_]+)"),
]
_SELECTOR_PATTERN = re.compile(r"selector:\s*['\"]([^'\"]+)['\"]")


def _resolve_brain_paths(workspace: Path) -> dict:
    """Return canonical and legacy brain paths for a workspace."""
    primary_dir = workspace / BRAIN_DIR_NAME / BRAIN_KNOWLEDGE_DIR
    primary_file = primary_dir / BRAIN_FILE_NAME
    legacy_file = workspace / LEGACY_BRAIN_DIR / LEGACY_BRAIN_FILE_NAME
    return {
        "primary_dir": primary_dir,
        "primary_file": primary_file,
        "legacy_file": legacy_file,
    }


def _humanize_dir(directory_path: str) -> str:
    parts = [p for p in re.split(r"[\\/]", directory_path) if p and p != "."]
    if not parts:
        return "Root"
    return " / ".join(w.replace("-", " ").replace("_", " ").title() for w in parts[-2:])


def _extract_symbols_from_text(text: str, limit: int = 12) -> list[str]:
    found: list[str] = []
    for pattern in _SYMBOL_PATTERNS:
        for match in pattern.findall(text):
            name = match if isinstance(match, str) else (match[0] if match else "")
            name = (name or "").strip()
            if name and name not in found:
                found.append(name)
            if len(found) >= limit:
                return found
    return found


def scan_source_directory_brain(workspace_path: str) -> list[dict]:
    """Build a directory brain grounded in the ACTUAL source tree.

    Walks the workspace, groups source files by their containing directory, and
    records real repo-relative file paths plus lightweight symbol/intent terms
    (component names, exported symbols, CSS selectors, filename stems). This is
    repository-agnostic and does not depend on any pre-authored markdown.
    """
    workspace = Path(workspace_path)
    if not workspace.exists():
        return []

    by_dir: dict[str, dict] = {}

    for root, dirs, files in os.walk(workspace):
        prune_walk_dirs(dirs, root, workspace)
        root_path = Path(root)
        source_files = [
            f for f in files
            if Path(f).suffix.lower() in _SOURCE_EXTS
            and not is_ignored_path(root_path / f, workspace_path=workspace)
        ]
        if not source_files:
            continue

        try:
            rel_dir = root_path.relative_to(workspace).as_posix()
        except ValueError:
            rel_dir = "."
        rel_dir = rel_dir or "."

        entry = by_dir.setdefault(
            rel_dir,
            {
                "directory_path": rel_dir,
                "business_domain": _humanize_dir(rel_dir),
                "technical_role": "Source Module",
                "core_responsibilities": [],
                "intent_terms": [],
                "key_artifacts": [],
            },
        )

        for fname in sorted(source_files):
            fpath = root_path / fname
            rel_file = fpath.relative_to(workspace).as_posix()
            entry["key_artifacts"].append(rel_file)

            stem = Path(fname).stem
            for token in re.split(r"[.\-_]", stem):
                token = token.strip()
                if token and token.lower() not in {"index", "app", "main", "styles", "style"} and token not in entry["intent_terms"]:
                    entry["intent_terms"].append(token)

            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = ""
            if text:
                for sym in _extract_symbols_from_text(text):
                    if sym not in entry["intent_terms"]:
                        entry["intent_terms"].append(sym)
                sel = _SELECTOR_PATTERN.search(text)
                if sel and sel.group(1) not in entry["intent_terms"]:
                    entry["intent_terms"].append(sel.group(1))

    result: list[dict] = []
    for entry in by_dir.values():
        result.append(
            {
                "directory_path": entry["directory_path"],
                "business_domain": entry["business_domain"],
                "technical_role": entry["technical_role"],
                "core_responsibilities": f"Source files under {entry['directory_path']}",
                "intent_terms": entry["intent_terms"][:24],
                "key_artifacts": entry["key_artifacts"][:60],
            }
        )
    return sorted(result, key=lambda x: x["directory_path"])


# ──────────────────────────────────────────────────────────────────────────────
# Two-layer self-updating brain: mechanical scan (the "what") + cached LLM "why"
# ──────────────────────────────────────────────────────────────────────────────

def _folder_signature(entry: dict) -> str:
    """Deterministic content fingerprint for a directory entry.

    The signature changes only when the *structure* of a folder changes (files added
    or removed, or symbols/intent terms changing). Pure comment/whitespace edits that
    do not alter filenames or extracted symbols keep the same signature, so the cached
    "why" can be safely reused. Editing back to a previous structure reproduces the
    previous signature, which is what makes undo/revert self-healing.
    """
    basis = {
        "artifacts": sorted(str(a) for a in (entry.get("key_artifacts") or [])),
        "intent": sorted(str(t) for t in (entry.get("intent_terms") or [])),
    }
    payload = json.dumps(basis, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_semantic_cache(cache_path: Path) -> dict:
    """Load the semantic "why" cache, tolerating a missing or corrupt file."""
    if not cache_path.exists():
        return {"version": 1, "folders": {}}
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        logger.warning("Semantic cache %s is unreadable/corrupt; starting fresh (%s)", cache_path, exc)
        return {"version": 1, "folders": {}}

    folders = data.get("folders") if isinstance(data, dict) else None
    if not isinstance(folders, dict):
        return {"version": 1, "folders": {}}
    return {"version": 1, "folders": folders}


def _atomic_write_json(path: Path, data) -> None:
    """Write JSON atomically so a crash mid-write can never corrupt the brain/cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path)


def _cache_entry_has_why(entry) -> bool:
    """True when a cached description record carries at least one non-empty field."""
    if not isinstance(entry, dict):
        return False
    return any(str(entry.get(field, "")).strip() for field in _WHY_FIELDS)


def _bound_descriptions(descriptions: dict, current_signature: str, max_keep: int = 8) -> dict:
    """Keep the signature history per directory bounded so the cache cannot grow forever.

    Retains the most-recently-inserted signatures and always preserves the current one.
    Relies on dict insertion order (Python 3.7+).
    """
    if len(descriptions) <= max_keep:
        return descriptions
    items = list(descriptions.items())
    trimmed = items[-max_keep:]
    if current_signature not in {k for k, _ in trimmed} and current_signature in descriptions:
        trimmed = trimmed[1:] + [(current_signature, descriptions[current_signature])]
    return dict(trimmed)


def _apply_why(entry: dict, why: dict) -> None:
    """Overlay LLM/cached description fields onto a mechanical entry (never blanks them)."""
    for field in _WHY_FIELDS:
        value = str(why.get(field, "")).strip()
        if value:
            entry[field] = value


def _strip_json_fences(text: str) -> str:
    """Strip Markdown ```json ... ``` fences around a JSON payload without corrupting it.

    Kept local (instead of importing plan_gating._extract_json_payload) so the brain
    scan/refresh path has no dependency on the aviator LLM stack.
    """
    stripped = str(text).strip()
    if stripped.startswith("```"):
        stripped = re.sub(r'^```[a-zA-Z]*\s*', '', stripped)
        stripped = re.sub(r'```\s*$', '', stripped)
    return stripped.strip()


def _describe_folder_with_llm(entry: dict, llm) -> "dict | None":
    """Ask the LLM to explain a single directory's purpose. Returns None on any failure.

    Output contract is strict JSON with exactly the three "why" fields. Anything else
    (network error, malformed JSON, empty description) degrades gracefully to None so the
    caller keeps the mechanical defaults.
    """
    if llm is None or not hasattr(llm, "invoke"):
        return None

    directory_path = str(entry.get("directory_path", ".")) or "."
    artifacts = [str(a) for a in (entry.get("key_artifacts") or [])][:20]
    intent = [str(t) for t in (entry.get("intent_terms") or [])][:24]

    prompt = f"""You are documenting ONE directory of a software repository so an automated
localization agent can later decide where new files belong.

DIRECTORY PATH:
{directory_path}

FILES IN THIS DIRECTORY:
{json.dumps(artifacts, ensure_ascii=False)}

SYMBOLS / INTENT TERMS EXTRACTED FROM THE CODE:
{json.dumps(intent, ensure_ascii=False)}

Infer the directory's purpose from the file names and symbols. Be concise and factual.
Return JSON only, with EXACTLY these three keys and no others:
{{
  "business_domain": "<short business/functional area, e.g. 'Payments' or 'Authentication'>",
  "technical_role": "<architectural role, e.g. 'REST Controllers' or 'Data Access Layer'>",
  "core_responsibilities": "<one sentence describing what lives here and why>"
}}
Do not include markdown, code fences, or any extra keys."""

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
            _future = _pool.submit(llm.invoke, prompt)
            try:
                response = _future.result(timeout=ENRICH_CALL_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"brain enrichment LLM call exceeded {ENRICH_CALL_TIMEOUT_SECONDS}s"
                )
        content = getattr(response, "content", response)
        payload = json.loads(_strip_json_fences(str(content)))
    except Exception as exc:
        logger.warning("LLM enrichment failed for directory '%s'; keeping mechanical defaults (%s)", directory_path, exc)
        return None

    if not isinstance(payload, dict):
        return None

    why = {field: str(payload.get(field, "")).strip() for field in _WHY_FIELDS}
    if not any(why.values()):
        return None
    return why


def refresh_repository_brain(
    workspace_path: str,
    llm=None,
    enrich: bool = True,
    prune: bool = True,
) -> dict:
    """Drift-aware, self-updating regeneration of the directory brain.

    Design invariants (why undo/deletion is always correct):
    1. The brain is ALWAYS a full mechanical scan of the current tree — never a patched
       diff — so deleted/undone folders simply disappear and can never linger.
    2. The LLM "why" is cached per directory keyed by a content signature. Unchanged
       folders reuse their cached description for free; only new/changed folders cost an
       LLM call. Reverting a folder to a prior structure reproduces its old signature and
       reuses the old, correct description.
    3. The cache is rebuilt from the current scan each run, so orphaned entries for
       deleted folders are pruned automatically and cannot grow unbounded.

    Safe to call repeatedly. Never raises for environmental problems; returns a summary
    dict describing what happened.
    """
    workspace = Path(workspace_path)
    if not workspace.exists():
        logger.warning("Refuse to refresh brain: workspace does not exist: %s", workspace)
        return {"status": "skipped_no_workspace", "directories": 0}

    paths = _resolve_brain_paths(workspace)

    source_entries = scan_source_directory_brain(str(workspace))
    if not source_entries:
        # Never overwrite a previously good brain with an empty scan (e.g. transient
        # filesystem state or a workspace with no recognized source files).
        logger.warning("Brain refresh produced 0 source directories for %s; leaving existing brain untouched", workspace)
        return {"status": "skipped_empty_scan", "directories": 0}

    brain_dir = paths["primary_dir"]
    cache_path = brain_dir / SEMANTIC_CACHE_FILENAME
    old_cache = _load_semantic_cache(cache_path)
    old_folders = old_cache.get("folders", {})

    if enrich and llm is None:
        try:
            from ticket_to_code.runtime.llm_factory import LLMRegistry
            llm = LLMRegistry.get_llm()
        except Exception as exc:
            logger.info("No LLM available for brain enrichment; using mechanical + cached descriptions only (%s)", exc)
            llm = None

    new_folders: dict = {}
    reused = 0
    enriched = 0
    unenriched = 0
    enrich_calls = 0
    enrich_deadline = time.monotonic() + ENRICH_DEADLINE_SECONDS

    for entry in source_entries:
        dir_path = entry["directory_path"]
        signature = _folder_signature(entry)

        old_folder = old_folders.get(dir_path)
        old_descriptions = old_folder.get("descriptions") if isinstance(old_folder, dict) else None
        if not isinstance(old_descriptions, dict):
            old_descriptions = {}

        # Carry the bounded signature history forward so a revert to a prior structure
        # reuses its old description for free instead of paying for the LLM again.
        descriptions = dict(old_descriptions)
        cached_why = descriptions.get(signature)

        if _cache_entry_has_why(cached_why):
            # Case 1 — structure matches a known signature: reuse for free.
            _apply_why(entry, cached_why)
            reused += 1
        else:
            # Case 2 — new or changed structure: try to (re)learn "why" from the
            # LLM, bounded by a per-call timeout and a global enrichment budget
            # (calls + wall clock). Past the budget, folders keep mechanical
            # defaults and a later refresh can enrich them.
            why = None
            if (
                enrich
                and llm is not None
                and enrich_calls < ENRICH_MAX_CALLS
                and time.monotonic() < enrich_deadline
            ):
                enrich_calls += 1
                why = _describe_folder_with_llm(entry, llm)
            if why:
                _apply_why(entry, why)
                descriptions[signature] = why
                enriched += 1
            else:
                # No description available yet; keep mechanical defaults so a future run
                # with an LLM can enrich it.
                unenriched += 1

        if descriptions:
            new_folders[dir_path] = {
                "current_signature": signature,
                "descriptions": _bound_descriptions(descriptions, signature),
            }

    pruned = 0
    if prune:
        pruned = sum(1 for key in old_folders if key not in new_folders)

    brain_path = paths["primary_file"]
    try:
        _atomic_write_json(brain_path, source_entries)
        _atomic_write_json(cache_path, {"version": 1, "folders": new_folders})
    except Exception as exc:
        logger.warning("Failed to persist refreshed brain to %s (%s)", brain_path, exc)
        return {"status": "write_failed", "directories": len(source_entries), "error": str(exc)}

    logger.info(
        "🧠 Brain refreshed: %s dirs (reused=%s, enriched=%s, unenriched=%s, pruned=%s) → %s",
        len(source_entries), reused, enriched, unenriched, pruned, brain_path,
    )
    return {
        "status": "refreshed",
        "directories": len(source_entries),
        "reused": reused,
        "enriched": enriched,
        "unenriched": unenriched,
        "enrich_calls": enrich_calls,
        "enrich_budget_exhausted": bool(
            enrich_calls >= ENRICH_MAX_CALLS or time.monotonic() >= enrich_deadline
        ),
        "pruned": pruned,
        "brain_path": str(brain_path),
        "cache_path": str(cache_path),
    }


def ensure_repository_brain(workspace_path: str, llm=None, enrich: bool = True) -> dict:
    """Guarantee a project has a usable directory brain before localization runs.

    Bootstrap policy:
    - New project (no brain on disk yet): generate it now and enrich each directory's
      "why" with the LLM, so brain-driven CREATE placement works from the very first ticket.
    - Project we've already indexed: reuse the existing brain as-is (cheap, no scan/LLM).
      Ongoing freshness is handled after successful tickets by ``refresh_repository_brain``.

    Removing a project and adding it again simply removes ``brain/knowledge`` with it, so the
    next bootstrap sees no brain and regenerates with the LLM — exactly the "check whether it
    has knowledge or not" behavior. Safe to call repeatedly; never raises for env problems.
    """
    workspace = Path(workspace_path)
    paths = _resolve_brain_paths(workspace)
    brain_path = paths["primary_file"]
    legacy_path = paths["legacy_file"]

    if brain_path.exists():
        logger.info("🧠 Repository brain already present at %s; reusing (no bootstrap needed).", brain_path)
        return {"status": "exists", "bootstrap": "reused", "brain_path": str(brain_path)}

    if legacy_path.exists():
        logger.info("🧠 Legacy repository brain found at %s; migrating to %s", legacy_path, brain_path)
        try:
            brain_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_data = json.loads(legacy_path.read_text(encoding="utf-8"))
            _atomic_write_json(brain_path, legacy_data)
            return {
                "status": "exists",
                "bootstrap": "migrated_legacy",
                "brain_path": str(brain_path),
                "legacy_brain_path": str(legacy_path),
            }
        except Exception as exc:
            logger.warning("Legacy brain migration failed from %s to %s (%s)", legacy_path, brain_path, exc)

    logger.info("🧠 No repository brain for %s — bootstrapping now with LLM enrichment.", workspace)
    result = refresh_repository_brain(workspace_path, llm=llm, enrich=enrich, prune=True)
    result["bootstrap"] = "created" if result.get("status") == "refreshed" else result.get("status")
    return result


def _domains_to_directory_brain(domains) -> list[dict]:
    """Convert markdown-derived domain/workflow/capability output into loadable directory brain items."""
    directory_map: dict[str, dict] = {}

    for domain in domains:
        domain_name = getattr(domain, "name", "Unknown") or "Unknown"
        workflows = getattr(domain, "workflows", []) or []

        for workflow in workflows:
            workflow_name = getattr(workflow, "name", "Core Workflow") or "Core Workflow"
            capabilities = getattr(workflow, "capabilities", []) or []

            for capability in capabilities:
                purpose = getattr(capability, "purpose", "") or ""
                capability_name = getattr(capability, "name", "Capability") or "Capability"
                artifacts = []
                artifacts.extend(getattr(capability, "primary_artifacts", []) or [])
                artifacts.extend(getattr(capability, "supporting_artifacts", []) or [])

                for artifact in artifacts:
                    artifact_path = str(artifact or "").strip().replace("\\", "/")
                    if not artifact_path:
                        continue
                    directory_path = artifact_path.rsplit("/", 1)[0] if "/" in artifact_path else "."

                    item = directory_map.setdefault(
                        directory_path,
                        {
                            "directory_path": directory_path,
                            "business_domain": domain_name,
                            "technical_role": workflow_name,
                            "core_responsibilities": [],
                            "intent_terms": [],
                            "key_artifacts": [],
                        },
                    )
                    item["business_domain"] = item.get("business_domain") or domain_name
                    item["technical_role"] = item.get("technical_role") or workflow_name

                    if purpose and purpose not in item["core_responsibilities"]:
                        item["core_responsibilities"].append(purpose)
                    if capability_name and capability_name not in item["intent_terms"]:
                        item["intent_terms"].append(capability_name)
                    if artifact_path not in item["key_artifacts"]:
                        item["key_artifacts"].append(artifact_path)

    result = []
    for item in directory_map.values():
        result.append(
            {
                "directory_path": item["directory_path"],
                "business_domain": item["business_domain"],
                "technical_role": item["technical_role"],
                "core_responsibilities": "; ".join(item["core_responsibilities"][:6]),
                "intent_terms": item["intent_terms"][:12],
                "key_artifacts": item["key_artifacts"][:20],
            }
        )

    return sorted(result, key=lambda x: x["directory_path"])

def generate_repository_brain(workspace_path: str, knowledge_path: str = None) -> bool:
    """
    Auto-generates the repository brain from workspace markdown files and IDE knowledge base.
    """
    print(f"Generating repository brain for workspace: {workspace_path}")
    workspace = Path(workspace_path)
    
    # Files to look for in workspace
    files_to_read = [
        "PRODUCT_BRAIN.md",
        "REPOSITORY_BRAIN.md",
        "REPOSITORY_BRAIN_V2.md",
        "REPOSITORY_UNDERSTANDING_AUDIT.md",
        "WORKFLOW_RECONSTRUCTION_REPORT.md",
        "ARCHITECTURE.md"
    ]
    
    contents = {}
    
    # 1. Look in workspace root
    for f in files_to_read:
        path = workspace / f
        if path.exists():
            print(f"Found in workspace: {f}")
            contents[f] = path.read_text(encoding="utf-8")
            
    # 2. Look in IDE knowledge base if provided
    if knowledge_path and Path(knowledge_path).exists():
        kb_path = Path(knowledge_path)
        print(f"Scanning knowledge base: {kb_path}")
        for md_file in kb_path.rglob("*.md"):
            name = md_file.name
            if name not in contents:
                print(f"Found in knowledge base: {name}")
                contents[name] = md_file.read_text(encoding="utf-8")
                
    if not contents:
        print("Warning: No architecture/brain markdown files found. Using empty templates.")
        contents["PRODUCT_BRAIN.md"] = "# Default Domain\n\nNo architecture context provided."
        contents["REPOSITORY_BRAIN.md"] = ""
        
    try:
        from ticket_to_code.brain.brain_builder import RepositoryBrainBuilder
        from ticket_to_code.brain.brain_storage import RepositoryBrainStorage
    except Exception as exc:
        print(f"RepositoryBrainBuilder/Storage unavailable ({exc}); falling back to source-only brain refresh.")
        return refresh_repository_brain(workspace_path, enrich=False).get("status") == "refreshed"

    builder = RepositoryBrainBuilder(llm_enabled=False)
    
    # Try to find primary files
    product_md = contents.get("PRODUCT_BRAIN.md", "")
    if not product_md and contents:
        # Fallback to the largest markdown file as product brain
        product_md = max(contents.values(), key=len)
        
    repo_md = contents.get("REPOSITORY_BRAIN_V2.md") or contents.get("REPOSITORY_BRAIN.md", "")
    
    other_audits = [v for k, v in contents.items() if k not in ["PRODUCT_BRAIN.md", "REPOSITORY_BRAIN_V2.md", "REPOSITORY_BRAIN.md"]]
    
    try:
        domains = builder.build_from_markdown(product_md, repo_md, other_audits)

        storage = RepositoryBrainStorage()

        # Source-grounded directory brain: scan the ACTUAL repo tree so the brain
        # references real files (primary grounding, repository-agnostic).
        source_brain = scan_source_directory_brain(workspace_path)
        # Markdown-derived directory brain: supplementary business-domain context.
        markdown_brain = _domains_to_directory_brain(domains)

        # Merge: source entries win on directory_path; markdown-only dirs are kept.
        merged: dict[str, dict] = {}
        for item in markdown_brain:
            merged[item["directory_path"]] = item
        for item in source_brain:
            merged[item["directory_path"]] = item  # overwrite with grounded data
        directory_brain = [merged[k] for k in sorted(merged.keys())]

        # Primary runtime location used by LocalizationAgent / RepositoryBrainStorage
        brain_dir = workspace / "brain" / "knowledge"
        brain_dir.mkdir(parents=True, exist_ok=True)
        output_path = brain_dir / "generated_directory_brain.json"
        output_path.write_text(json.dumps(directory_brain, indent=2), encoding="utf-8")

        # Keep a legacy copy under .agents for older tooling.
        agents_dir = workspace / ".agents"
        agents_dir.mkdir(exist_ok=True)
        legacy_output = agents_dir / "repository_brain.json"
        legacy_output.write_text(json.dumps(directory_brain, indent=2), encoding="utf-8")

        # Prime the loader to validate the output shape.
        storage.load_directory(brain_dir)
        print(f"Successfully generated repository brain: {output_path}")
        print(f"Legacy copy written: {legacy_output}")
        print(
            f"Stats: {len(domains)} markdown domains, {len(source_brain)} source dirs, "
            f"{len(directory_brain)} total directory entries."
        )
        return True
    except Exception as e:
        print(f"Error generating repository brain: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-generate Repository Brain")
    parser.add_argument("--workspace", required=True, help="Path to workspace root")
    parser.add_argument("--knowledge-path", help="Path to IDE knowledge base (e.g. .gemini/antigravity-ide/knowledge)")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Drift-aware self-update: mechanical scan + cached LLM 'why' (fast, idempotent).",
    )
    parser.add_argument(
        "--ensure",
        action="store_true",
        help="Bootstrap: generate the brain with LLM 'why' only if it does not already exist.",
    )
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="With --refresh/--ensure, skip LLM enrichment (mechanical + cached descriptions only).",
    )
    args = parser.parse_args()

    if args.ensure:
        summary = ensure_repository_brain(args.workspace, enrich=not args.no_enrich)
        print(json.dumps(summary, indent=2))
        sys.exit(0 if summary.get("status") in {"refreshed", "exists", "skipped_empty_scan"} else 1)

    if args.refresh:
        summary = refresh_repository_brain(args.workspace, enrich=not args.no_enrich)
        print(json.dumps(summary, indent=2))
        sys.exit(0 if summary.get("status") in {"refreshed", "skipped_empty_scan"} else 1)

    success = generate_repository_brain(args.workspace, args.knowledge_path)
    sys.exit(0 if success else 1)
