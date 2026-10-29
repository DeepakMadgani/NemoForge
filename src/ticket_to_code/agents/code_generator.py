
# Auto-heal TS prop destructuring in React components (e.g. ({ orders, onCancelOrder }) -> ({ orders, onCancelOrder, onRefundOrder }))
def _heal_react_destructuring(code_str: str) -> str:
    try:
        props_match = re.search(r'interface\s+(\w+Props)\s*\{([^}]+)\}', code_str)
        if props_match:
            props_body = props_match.group(2)
            prop_names = re.findall(r'(\w+)\s*\??\s*:', props_body)
            # Find React.FC<Props> = ({ ... }) =>
            comp_match = re.search(r'(:\s*React\.FC<[^>]+>\s*=\s*\(\s*\{)([^}]+)(\}\s*\)\s*=>)', code_str)
            if comp_match:
                prefix = comp_match.group(1)
                destructured_raw = comp_match.group(2)
                suffix = comp_match.group(3)
                destructured = [p.strip() for p in destructured_raw.split(',') if p.strip()]
                destructured_set = set(destructured)
                to_add = []
                for p in prop_names:
                    if p not in destructured_set and re.search(rf'\b{p}\s*\(', code_str[comp_match.end():]):
                        to_add.append(p)
                if to_add:
                    new_destructured = destructured + to_add
                    new_param_str = prefix + " " + ", ".join(new_destructured) + " " + suffix
                    code_str = code_str[:comp_match.start()] + new_param_str + code_str[comp_match.end():]
    except Exception as e:
        logger.debug(f"React destructuring heal skipped: {e}")
    return code_str


def heal_trailing_class_methods(code: str, file_path: str) -> str:
    """
    If a Java/C# method was accidentally appended outside the closing brace of the class,
    relocate the closing brace to the end of the file so the method is inside the class.
    """
    if not file_path.endswith((".java", ".cs")):
        return code

    if "class " not in code and "record " not in code:
        return code

    lines = code.splitlines()
    depth = 0
    premature_line = -1
    for idx, line in enumerate(lines):
        stripped = line.split("//")[0].strip()
        open_c = stripped.count("{")
        close_c = stripped.count("}")
        depth += open_c - close_c
        if depth == 0 and idx < len(lines) - 2:
            remaining = "\n".join(lines[idx + 1:]).strip()
            if remaining and any(kw in remaining for kw in ("public ", "private ", "protected ", "@Transactional", "@PostMapping", "@GetMapping", "@Override")):
                premature_line = idx
                break

    if premature_line != -1:
        new_lines = [l for i, l in enumerate(lines) if not (i == premature_line and l.strip() == "}")]
        net_depth = sum(l.split("//")[0].count("{") - l.split("//")[0].count("}") for l in new_lines)
        if net_depth > 0:
            return "\n".join(new_lines).rstrip() + "\n" + ("}\n" * net_depth)
        return "\n".join(new_lines) + "\n"

    return code

"""
Code Generator Agent

Generates production-ready code using validated context and requirements.

This is Agent 5 in the autonomous pipeline.

Author: Deepak Madgani
Date: April 2026
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, List, Optional, Union

from ticket_to_code.llm_utils import llm_invoke


def _get_anchor_methods(task) -> list[str]:
    """
    Collect the method names that extract_exact_methods() should extract.

    Handles two task shapes:
      MODIFY: The task edits existing methods. anchor_methods = the methods
              being modified (from allowed_methods + target_method).
      INSERT: The task adds a new method. anchor_methods = the neighbor method
              (from edit_anchors.anchor_method) that the new code is placed after.
              We need that method's real closing brace for the LLM to anchor against.

    Returns a deduplicated list of method names to extract verbatim.
    """
    methods: list[str] = []

    # Source 1: edit_anchors — carries both MODIFY targets and INSERT anchors
    anchors = getattr(task, "edit_anchors", []) or []
    for anchor in anchors:
        # For INSERT tasks: action="add_method", anchor_method="findById"
        # We need findById's body so the LLM sees its closing brace
        anchor_method = anchor.get("anchor_method", "")
        if anchor_method and anchor_method not in methods:
            methods.append(anchor_method)
        # For MODIFY tasks: method_name="validateDuplicateName"
        method_name = anchor.get("method_name", "")
        if method_name and method_name not in methods:
            methods.append(method_name)

    # Source 2: allowed_methods — the planner's explicit scope
    for m in (getattr(task, "allowed_methods", []) or []):
        if m not in methods:
            methods.append(m)

    # Source 3: target_method — primary edit target
    target = getattr(task, "target_method", None)
    if target and target not in methods:
        methods.append(target)

    # Source 4: change_targets — explicit ChangeTarget symbols
    for ct in (getattr(task, "change_targets", []) or []):
        sym = getattr(ct, "symbol", None)
        if sym and sym not in methods:
            methods.append(sym)

    return methods


# tsconfig.json path alias cache: workspace_path → {alias_prefix: resolved_dir}
_tsconfig_alias_cache: dict[str, dict[str, str]] = {}


def _load_tsconfig_aliases(workspace_path: str) -> dict[str, str]:
    """Read tsconfig.json compilerOptions.paths to build alias → dir map."""
    if workspace_path in _tsconfig_alias_cache:
        return _tsconfig_alias_cache[workspace_path]
    aliases: dict[str, str] = {}
    import json as _json
    for name in ("tsconfig.json", "tsconfig.base.json", "tsconfig.app.json"):
        tsconfig = Path(workspace_path) / name
        if not tsconfig.exists():
            # Also check one level down (e.g. xchange-ui/tsconfig.json)
            for sub in Path(workspace_path).iterdir() if Path(workspace_path).is_dir() else []:
                tsconfig = sub / name
                if tsconfig.exists():
                    break
            else:
                continue
        try:
            raw = _json.loads(tsconfig.read_text(encoding="utf-8"))
            paths = raw.get("compilerOptions", {}).get("paths", {})
            base_url = raw.get("compilerOptions", {}).get("baseUrl", ".")
            for alias, targets in paths.items():
                # e.g. "@shared/*" → ["src/app/shared/*"]  →  store "@shared" → "src/app/shared"
                alias_prefix = alias.rstrip("/*")
                if targets:
                    resolved = targets[0].rstrip("/*")
                    aliases[alias_prefix] = resolved
            break
        except Exception:
            pass
    _tsconfig_alias_cache[workspace_path] = aliases
    return aliases


def _build_import_hint(
    symbol_name: str,
    file_path: str,
    source_file_path: str = "",
    workspace_path: str = "",
) -> str:
    """Convert a repo-relative file path to a language-appropriate import hint."""
    p = Path(file_path)
    ext = p.suffix.lower()
    stem = p.stem

    if ext == ".java":
        # Derive Java package from path: src/main/java/com/foo/Bar.java → com.foo.Bar
        parts = p.with_suffix("").parts
        try:
            java_idx = next(i for i, x in enumerate(parts) if x == "java")
            pkg = ".".join(parts[java_idx + 1:])
            return f"import {pkg};"
        except StopIteration:
            return f"import {stem};"

    if ext in (".ts", ".tsx"):
        # ── Cursor technique: compute correct relative OR aliased path ────────
        import_path = _resolve_ts_import_path(
            source_file=source_file_path or "",
            target_file=file_path,
            workspace_path=workspace_path or "",
        )
        return f"import {{ {symbol_name} }} from '{import_path}';"

    if ext == ".py":
        parts = p.with_suffix("").parts
        # Strip common src prefixes
        start = next((i + 1 for i, x in enumerate(parts) if x in ("src", "app")), 0)
        pkg = ".".join(parts[start:])
        return f"from {pkg} import {symbol_name}"

    if ext in (".cs",):
        # C# — namespace from directory
        parts = p.parent.parts
        ns = ".".join(pt for pt in parts if pt not in ("src", ".", ""))
        return f"using {ns};"

    return ""


def _resolve_ts_import_path(
    source_file: str,
    target_file: str,
    workspace_path: str,
) -> str:
    """
    Cursor technique: compute the correct TypeScript import path.

    Priority:
      1. tsconfig.json path alias if the target file sits under an aliased directory
         (e.g.  @shared/models/contract  when @shared → src/app/shared)
      2. Relative path computed from source to target (normalized with forward slashes,
         prefixed with './'' or '../' as required by the TypeScript module resolver)
    """
    # Normalise to forward-slash repo-relative paths
    target_norm = target_file.replace("\\", "/")
    target_no_ext = re.sub(r'\.tsx?$', '', target_norm)

    # 1. Check tsconfig aliases
    if workspace_path:
        aliases = _load_tsconfig_aliases(workspace_path)
        for alias_prefix, resolved_dir in aliases.items():
            resolved_dir_norm = resolved_dir.replace("\\", "/").rstrip("/")
            if target_no_ext.startswith(resolved_dir_norm + "/"):
                rel_within = target_no_ext[len(resolved_dir_norm) + 1:]
                # Gap 5: Check for barrel export — if index.ts exists in the target's
                # directory and re-exports the symbol, use the shorter directory path.
                if workspace_path:
                    target_dir = str(Path(target_file.replace("\\", "/")).parent)
                    barrel = Path(workspace_path) / target_dir / "index.ts"
                    if barrel.exists():
                        try:
                            barrel_content = barrel.read_text(encoding="utf-8", errors="ignore")
                            # Check if barrel re-exports from the target file
                            target_stem = Path(target_file).stem
                            if target_stem in barrel_content:
                                # Use directory path instead of full file path
                                dir_within = str(Path(rel_within).parent).replace("\\", "/")
                                if dir_within and dir_within != ".":
                                    return f"{alias_prefix}/{dir_within}"
                                return alias_prefix
                        except Exception:
                            pass
                return f"{alias_prefix}/{rel_within}"

    # 2. Compute relative path from source to target
    if source_file:
        source_dir = Path(source_file.replace("\\", "/")).parent
        try:
            rel = Path(target_no_ext).relative_to(source_dir)
            rel_str = str(rel).replace("\\", "/")
            return rel_str if rel_str.startswith(".") else f"./{rel_str}"
        except ValueError:
            # Not a sub-path; compute ../.. manually
            import os
            rel = os.path.relpath(target_no_ext, str(source_dir)).replace("\\", "/")
            return rel if rel.startswith(".") else f"./{rel}"

    # 3. Fallback: just the stem
    return f"./{Path(target_no_ext).name}"


from ticket_to_code.runtime.llm_factory import LLMRegistry
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from ticket_to_code.models import (
    DevelopmentTask,
    CodeChunk,
    StructuredRequirements,
    GeneratedCode,
    ProgrammingLanguage,
)

logger = logging.getLogger(__name__)

# Module-level tier trace accumulator — reset per file, read by workflow.py
# Each entry: {"file": str, "edit_index": int, "tier": str, "old_chars": int, "new_chars": int}
_patch_tier_traces: list[dict] = []

# Module-level reuse directive across agents (cleared per run)
_reuse_directive: str = ""


def set_reuse_directive(directive: str) -> None:
    """Set the module-level reuse directive for code generation."""
    global _reuse_directive
    _reuse_directive = str(directive or "")


def clear_reuse_directive() -> None:
    """Clear the reuse directive to prevent leakage across runs or Celery tasks."""
    global _reuse_directive
    _reuse_directive = ""

def _strip_conversational_prose(text: str) -> str:
    """
    Strips leaked LLM conversational prose, chain-of-thought analysis,
    or numbered explanation steps accidentally appended or prepended
    to code blocks without comment delimiters.
    """
    if not text or not isinstance(text, str):
        return text

    import re

    # Pattern for lines that are clearly conversational English prose, not code or comments
    prose_line_re = re.compile(
        r'^\s*(?:'
        r'\d+[\.\)]\s+[A-Za-z].*|'
        r'(?:But\s+note|However,\s*note|Please\s+note|Note:|Looking\s+at|Alternatively|'
        r'We\s+(?:must|should|can|need|will|have)|'
        r'You\s+(?:must|should|can|need|will)|'
        r'This\s+(?:will|method|class|ensures|allows|code|implementation)|'
        r'Here\s+(?:is|are)|Below\s+is|Above\s+is|'
        r'Let\'?s\s+(?:make|ensure|add|update)|'
        r'In\s+(?:this|the\s+above)\s+(?:step|method|file|code))\b.*'
        r')',
        re.IGNORECASE
    )

    comment_start_re = re.compile(r'^\s*(?://|/\*|\*|#|<!--|--|/\*\*)')

    lines = text.splitlines(keepends=True)

    # Strip from start
    start_idx = 0
    while start_idx < len(lines):
        line = lines[start_idx]
        stripped = line.strip()
        if not stripped:
            start_idx += 1
            continue
        if not comment_start_re.match(line) and prose_line_re.match(line):
            start_idx += 1
            continue
        break

    # Strip from end
    end_idx = len(lines)
    while end_idx > start_idx:
        line = lines[end_idx - 1]
        stripped = line.strip()
        if not stripped:
            end_idx -= 1
            continue
        if not comment_start_re.match(line) and prose_line_re.match(line):
            end_idx -= 1
            continue
        break

    cleaned_lines = []
    for line in lines[start_idx:end_idx]:
        if not comment_start_re.match(line) and prose_line_re.match(line):
            continue
        cleaned_lines.append(line)

    return "".join(cleaned_lines)



def _apply_str_replace_edits(edits: list[dict], existing: str, file_path: str) -> str:
    """
    Apply a batch of str_replace edits atomically using a 6-Tier Matching Cascade.

    Each edit is a dict with keys:
        old_str      — code to find (should match exactly, but tiers handle mismatches)
        new_str      — replacement text
        scope_method — (optional) method name from edit_anchors; narrows search

    6-Tier Resolution Cascade (ALL tiers run in ONE pass per edit — not separate retries):
      Tier 1: Exact string match                      (fastest, O(n))
      Tier 2: CRLF normalization (\r\n ↔ \n)           (O(n))
      Tier 3: Whitespace-normalized sliding window     (catches indent diffs, O(n*k))
      Tier 4: AST-structural matching via tree-sitter  (catches format/comment diffs, O(n))
      Tier 5: Levenshtein fuzzy match (≥90% threshold) (catches minor token changes, O(n*m))
      Tier 6: Scoped search within method body         (resolves ambiguous matches)

    Atomicity: all edits succeed or ZERO bytes are written.
    """
    result = existing
    # Reset tier traces for this file
    _patch_tier_traces.clear()

    for i, edit in enumerate(edits):
        old_str = edit.get("old_str", "")
        new_str = _strip_conversational_prose(edit.get("new_str", ""))
        scope_method = edit.get("scope_method", "")

        if not old_str:
            raise ValueError(
                f"Edit {i+1}/{len(edits)} for {file_path}: old_str is empty. "
                f"Every edit must specify the exact code to replace."
            )

        # Guard against destructive method deletion in a single edit
        if len(old_str) > 500 and len(new_str) < len(old_str) * 0.3:
            import re
            old_methods = set(re.findall(r'\b(?:public|private|protected)\s+(?:[\w<>\[\],]+\s+)+(\w+)\s*\(', old_str))
            new_methods = set(re.findall(r'\b(?:public|private|protected)\s+(?:[\w<>\[\],]+\s+)+(\w+)\s*\(', new_str))
            deleted = old_methods - new_methods
            if deleted:
                raise ValueError(
                    f"Destructive SEARCH block: SEARCH contains {len(old_str)} characters and deletes "
                    f"existing methods {sorted(deleted)}. You must use a short SEARCH anchor (3-8 lines) "
                    f"without deleting existing methods."
                )

        # ── Tier 1: Direct exact match ─────────────────────────────────
        count = result.count(old_str)

        if count == 1:
            result = result.replace(old_str, new_str, 1)
            _log_edit(i, len(edits), file_path, old_str, new_str, "Tier 1: Exact match")
            continue

        # ── Tier 2: CRLF mismatch resolution ──────────────────────────
        if count == 0:
            _adapted_old = _try_crlf_adapt(old_str, result)
            if _adapted_old is not None:
                result = result.replace(_adapted_old, new_str, 1)
                _log_edit(i, len(edits), file_path, old_str, new_str, "Tier 2: CRLF-adapted")
                continue

        # ── Tier 3: Whitespace-normalized sliding window ───────────────
        if count == 0:
            _ws_result = _whitespace_normalized_match(old_str, new_str, result)
            if _ws_result is not None:
                result = _ws_result
                _log_edit(i, len(edits), file_path, old_str, new_str,
                          "Tier 3: Whitespace-normalized")
                continue

        # ── Tier 4: AST-structural matching via tree-sitter ────────────
        #    Fresh parse at apply time (like Cursor) — NOT from stale index.
        #    Files change task-to-task, so we parse the CURRENT content.
        if count == 0:
            _ast_result = _ast_structural_match(old_str, new_str, result, file_path)
            if _ast_result is not None:
                result = _ast_result
                _log_edit(i, len(edits), file_path, old_str, new_str,
                          "Tier 4: AST-structural match")
                continue

        # ── Tier 5: Levenshtein fuzzy match (≥95% similarity, ≥3 lines) ──
        if count == 0:
            _fuzzy_result = _levenshtein_fuzzy_match(old_str, new_str, result)
            if _fuzzy_result is not None:
                result = _fuzzy_result
                _log_edit(i, len(edits), file_path, old_str, new_str,
                          "Tier 5: Fuzzy match (Levenshtein)")
                continue

        # ── Tier 5.5: Sub-anchor boundary reconciliation (Adjacent Edits) ──
        if count == 0:
            _relaxed_result = _try_anchor_relaxation(old_str, new_str, result)
            if _relaxed_result is not None:
                result = _relaxed_result
                _log_edit(i, len(edits), file_path, old_str, new_str,
                          "Tier 5.5: Sub-anchor relaxation (Adjacent Edit Recovery)")
                continue

        # ── Tier 6: Scoped search within method body ───────────────────
        #    Handles both count==0 (not found) and count>1 (ambiguous)
        if scope_method:
            _scope_resolved = _resolve_within_scope(
                result, old_str, new_str, scope_method, file_path
            )
            if _scope_resolved is not None:
                result = _scope_resolved
                _log_edit(i, len(edits), file_path, old_str, new_str,
                          f"Tier 6: Scoped to '{scope_method}'")
                continue

        # ── All 6 tiers exhausted — fail with actionable diagnostics ───
        if count == 0:
            # Find the closest match to show in the error for better retry
            _best = _find_closest_match(old_str, result)
            _best_info = ""
            if _best:
                _best_info = (
                    f"\n\nCLOSEST MATCH FOUND ({_best['ratio']:.0%} similar, "
                    f"lines {_best['start_line']}-{_best['end_line']}):\n"
                    f"{_best['content'][:500]}"
                )
            preview = old_str[:300].replace('\n', '\\n')
            raise ValueError(
                f"Edit {i+1}/{len(edits)} for {file_path}: old_str not found "
                f"after ALL 6 matching tiers (exact → CRLF → whitespace → AST → "
                f"fuzzy → scope). Searched for:\n{preview}{_best_info}"
            )
        else:
            _match_locations = _find_all_match_lines(result, old_str)
            _loc_desc = "; ".join(
                f"match {j+1} at line {ln}" for j, ln in enumerate(_match_locations[:5])
            )
            _scope_msg = (f" (scope_method='{scope_method}' was tried but didn't resolve it)"
                          if scope_method
                          else " (no edit_anchor / scope_method was provided)")
            raise ValueError(
                f"Edit {i+1}/{len(edits)} for {file_path}: old_str matches "
                f"{count} locations ({_loc_desc}). "
                f"Provide more surrounding context lines to make old_str unique."
                + _scope_msg
            )

    # ── Post-patch AST validation ──────────────────────────────────────
    # Parse the result with tree-sitter to catch broken patches BEFORE
    # returning to the caller (which writes to disk).
    _ast_issues = _validate_patch_ast(existing, result, file_path)
    if _ast_issues:
        logger.warning(
            f"  ⚠️ AST validation warnings for {file_path}: {_ast_issues}"
        )
        _fatal_issues = [iss for iss in _ast_issues if "Method count dropped" in iss or "unbalanced delimiters" in iss]
        if _fatal_issues:
            raise ValueError(
                f"Patch rejected by AST validation for {file_path}: {'; '.join(_fatal_issues)}"
            )

    if file_path.endswith((".tsx", ".jsx")):
        result = _heal_react_destructuring(result)
    elif file_path.endswith((".java", ".cs")):
        result = heal_trailing_class_methods(result, file_path)

    return result


# ─── Tier Helper Functions ───────────────────────────────────────────────

def _log_edit(i: int, total: int, file_path: str, old_str: str, new_str: str, tier: str):
    """Consistent logging for all edit tiers + record for trace output."""
    _old_preview = old_str[:500].replace('\n', '\\n')
    _new_preview = new_str[:500].replace('\n', '\\n')
    logger.info(
        f"  str_replace edit {i+1}/{total} applied to {file_path} "
        f"[{tier}] (old={len(old_str)} chars → new={len(new_str)} chars)"
    )
    logger.info(f"    OLD: {_old_preview}")
    logger.info(f"    NEW: {_new_preview}")
    # Record for trace output
    _patch_tier_traces.append({
        "file": file_path,
        "edit_index": i + 1,
        "total_edits": total,
        "tier": tier,
        "old_chars": len(old_str),
        "new_chars": len(new_str),
    })


def _try_crlf_adapt(old_str: str, content: str) -> Optional[str]:
    """Tier 2: Try adapting line endings. Returns the adapted old_str or None."""
    if '\n' in old_str and '\r\n' not in old_str:
        _adapted = old_str.replace('\n', '\r\n')
        if content.count(_adapted) == 1:
            return _adapted
    elif '\r\n' in old_str:
        _adapted = old_str.replace('\r\n', '\n')
        if content.count(_adapted) == 1:
            return _adapted
    return None


def _whitespace_normalized_match(
    old_str: str, new_str: str, content: str
) -> Optional[str]:
    """
    Tier 3: Whitespace-normalized sliding window.

    Algorithm: Normalize ALL whitespace (leading/trailing/internal) to single
    space, then slide a window of len(search_lines) over the file lines.
    If the normalized forms match, replace the ORIGINAL (un-normalized) window.

    Time complexity: O(n * k) where n = file lines, k = search lines.

    This catches the most common failure: LLM uses different indentation than
    the actual file (2-space vs 4-space vs tabs).
    """
    search_lines = old_str.strip().splitlines()
    if not search_lines:
        return None

    n = len(search_lines)
    normalized_search = [re.sub(r'\s+', ' ', l.strip()) for l in search_lines]

    # Skip if search is too short (single blank line etc.)
    if all(not s for s in normalized_search):
        return None

    file_lines = content.splitlines(True)  # Keep line endings for faithful replacement
    file_lines_stripped = [re.sub(r'\s+', ' ', l.strip()) for l in file_lines]

    matches = []
    for i in range(len(file_lines_stripped) - n + 1):
        window = file_lines_stripped[i:i + n]
        if window == normalized_search:
            matches.append(i)

    if len(matches) == 1:
        # Unique match — replace the ORIGINAL lines (preserving encoding/BOM)
        match_start = matches[0]
        original_window = ''.join(file_lines[match_start:match_start + n])
        # Detect indentation of the target location to re-indent new_str
        _target_indent = _detect_indent(file_lines[match_start])
        _search_indent = _detect_indent(search_lines[0]) if search_lines else ""
        _reindented_new = _reindent(new_str, _search_indent, _target_indent)
        return content.replace(original_window, _reindented_new, 1)

    return None


def _detect_indent(line: str) -> str:
    """Extract the leading whitespace from a line."""
    return line[:len(line) - len(line.lstrip())] if line.strip() else ""


def _reindent(code: str, old_indent: str, new_indent: str) -> str:
    """
    Re-indent code from old_indent base to new_indent base.

    If the LLM generated code with 2-space indent but the file uses 4-space,
    this adjusts ALL lines proportionally.
    """
    if old_indent == new_indent or not old_indent:
        return code

    lines = code.splitlines(True)
    result_lines = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            result_lines.append(line)
            continue
        current_indent = line[:len(line) - len(stripped)]
        # Count how many "units" of old_indent this line has
        if old_indent and current_indent.startswith(old_indent):
            # Calculate relative indentation
            if len(old_indent) > 0:
                depth = 0
                remaining = current_indent
                while remaining.startswith(old_indent):
                    depth += 1
                    remaining = remaining[len(old_indent):]
                new_line_indent = new_indent * depth + remaining
            else:
                new_line_indent = current_indent
            result_lines.append(new_line_indent + stripped)
        else:
            result_lines.append(line)
    return ''.join(result_lines)


def _ast_structural_match(
    old_str: str, new_str: str, content: str, file_path: str
) -> Optional[str]:
    """
    Tier 4: AST-structural matching via tree-sitter.

    Parse the SEARCH block and the file with tree-sitter (FRESH parse —
    not from stale index). Find the AST node in the file whose structure
    matches the SEARCH block's AST.

    Like Cursor's approach: parse on-the-fly at edit time, replace at
    the matched node's exact byte range.

    Returns the patched content, or None if no structural match found.
    """
    try:
        from ticket_to_code.agents.smart_extract import parse_with_treesitter
    except ImportError:
        return None

    # Parse the full file (fresh — current content, not cached)
    file_tree, lang = parse_with_treesitter(content, file_path)
    if file_tree is None:
        return None

    # Parse the SEARCH block as a code fragment
    search_tree, _ = parse_with_treesitter(old_str, file_path)
    if search_tree is None:
        return None

    # Get the structural signature of the SEARCH block
    search_sig = _ast_signature(search_tree.root_node)
    if not search_sig:
        return None

    # Walk the file AST to find a node with matching structure
    source_bytes = content.encode("utf-8")
    matches = []

    def _find_matching_nodes(node):
        node_sig = _ast_signature(node)
        if node_sig == search_sig:
            # Verify the text content is reasonably similar (>70%)
            # to avoid false positives from structurally identical but
            # semantically different code blocks
            node_text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            from difflib import SequenceMatcher
            ratio = SequenceMatcher(None, old_str.strip(), node_text.strip()).ratio()
            if ratio >= 0.70:
                matches.append((node, node_text, ratio))
        for child in node.children:
            _find_matching_nodes(child)

    _find_matching_nodes(file_tree.root_node)

    if len(matches) == 1:
        node, node_text, ratio = matches[0]
        logger.info(f"    AST match: {ratio:.0%} similar, "
                     f"L{node.start_point[0]+1}-L{node.end_point[0]+1}")
        # Replace the matched text in the content (line-based, safe for all encodings)
        return content.replace(node_text, new_str, 1)

    if len(matches) > 1:
        logger.info(f"    AST match: {len(matches)} candidates (ambiguous), skipping")


    return None


def _ast_signature(node) -> tuple:
    """
    Generate a structural signature for an AST node (tree isomorphism).

    The signature captures the STRUCTURE (node types and their tree shape)
    but IGNORES the actual text content (identifiers, literals, whitespace).

    Two code blocks with identical AST signatures are structurally equivalent
    regardless of formatting, comments, or variable naming.
    """
    if node.child_count == 0:
        return (node.type,)
    child_sigs = tuple(_ast_signature(child) for child in node.children
                       if child.type not in ("comment", "line_comment", "block_comment"))
    return (node.type, child_sigs)


_FUZZY_MATCH_THRESHOLD = 0.95
_FUZZY_MIN_SEARCH_LINES = 3


def _levenshtein_fuzzy_match(
    old_str: str, new_str: str, content: str,
    threshold: float = _FUZZY_MATCH_THRESHOLD
) -> Optional[str]:
    """
    Tier 5: Levenshtein fuzzy match with sliding window.

    Algorithm: Slide a window of len(search_lines) over the file lines.
    For each window, compute SequenceMatcher.ratio(). Accept if:
      1. ratio >= threshold (default 95%)
      2. Exactly ONE candidate above threshold (no ambiguity)
      3. The search block has at least 3 lines (short blocks are too easy
         to confuse with unrelated code)

    Time complexity: O(n * m) where n = file chars, m = search chars.

    Uses Python's difflib.SequenceMatcher which is based on the
    Ratcliff/Obershelp algorithm — faster than pure Levenshtein for
    large strings because it finds longest common subsequences.
    """
    from difflib import SequenceMatcher

    search_lines = old_str.strip().splitlines()
    if not search_lines or len(search_lines) < _FUZZY_MIN_SEARCH_LINES:
        # Too short for fuzzy — risk of false positives
        return None

    n = len(search_lines)
    file_lines = content.splitlines(True)  # Keep line endings

    if len(file_lines) < n:
        return None

    best_ratio = 0.0
    best_start = -1
    candidates = []

    # Sliding window over file lines
    for i in range(len(file_lines) - n + 1):
        window_text = ''.join(file_lines[i:i + n])
        ratio = SequenceMatcher(None, old_str.strip(), window_text.strip()).ratio()
        if ratio >= threshold:
            candidates.append((i, ratio, window_text))
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i

    if len(candidates) == 1:
        # Unique fuzzy match — safe to apply
        start, ratio, window_text = candidates[0]
        logger.info(f"    Fuzzy match: {ratio:.1%} similar at line {start+1}")
        # Re-indent new_str to match the target location's indentation
        _target_indent = _detect_indent(file_lines[start])
        _search_indent = _detect_indent(search_lines[0]) if search_lines else ""
        _reindented_new = _reindent(new_str, _search_indent, _target_indent)
        return content.replace(window_text, _reindented_new, 1)

    if len(candidates) > 1:
        logger.info(
            f"    Fuzzy match: {len(candidates)} candidates above {threshold:.0%} "
            f"threshold (ambiguous) — not applying"
        )

_ANCHOR_RELAXATION_BACK_WINDOW = 300


def _try_anchor_relaxation(old_str: str, new_str: str, content: str) -> Optional[str]:
    """
    Tier 5.5: Sub-anchor boundary reconciliation.
    Resolves multi-hunk edit collisions where an earlier edit inserted lines immediately adjacent
    to this edit's search anchor.

    Safety rules:
      * The first line of old_str is only searched for within a 300-char window
        immediately before the unique suffix anchor.
      * The region that would be replaced (actual_old) must consist solely of
        lines from old_str plus the lines inserted between the first line and
        the anchor; every one of those inserted lines must survive in the
        adapted replacement. Otherwise the relaxation refuses (returns None)
        rather than silently deleting code the edit never asked to touch.
    """
    old_lines = old_str.strip().splitlines()
    if len(old_lines) < 2:
        return None

    old_line_set = {ln.strip() for ln in old_lines if ln.strip()}

    # Check if the bottom lines of old_str form a unique anchor in content
    for k in range(len(old_lines) - 1, 0, -1):
        suffix = "\n".join(old_lines[-k:]).strip()
        if len(suffix) < 15:
            continue
        if content.count(suffix) == 1 and suffix in new_str:
            pos = content.find(suffix)
            first_line = old_lines[0].strip()
            if len(first_line) >= 10:
                # Look backwards in content (bounded window) for first_line
                window_start = max(0, pos - _ANCHOR_RELAXATION_BACK_WINDOW)
                search_window = content[window_start:pos]
                if first_line in search_window:
                    fl_rel = search_window.rfind(first_line)
                    actual_start = window_start + fl_rel
                    actual_old = content[actual_start : pos + len(suffix)]
                    inserted_between = content[actual_start + len(first_line) : pos]

                    # Lines in the replaced region that old_str never mentioned.
                    unexpected_lines = [
                        ln.strip() for ln in actual_old.splitlines()
                        if ln.strip() and ln.strip() not in old_line_set
                    ]
                    inserted_line_set = {
                        ln.strip() for ln in inserted_between.splitlines() if ln.strip()
                    }
                    if any(ln not in inserted_line_set for ln in unexpected_lines):
                        logger.info(
                            "    Anchor relaxation refused: replaced region contains lines "
                            "outside old_str and the adjacent insertion"
                        )
                        return None

                    # Only carry over the genuinely inserted lines; lines that belong to
                    # old_str are already represented by new_str (re-inserting them
                    # would duplicate code).
                    carried = "".join(
                        ln for ln in inserted_between.splitlines(True)
                        if ln.strip() and ln.strip() not in old_line_set
                    )

                    adapted_new = new_str
                    if carried.strip() and carried.strip() not in new_str:
                        fl_new = new_str.find(first_line)
                        if fl_new != -1:
                            end_fl = fl_new + len(first_line)
                            carried_block = carried if carried.endswith("\n") else carried + "\n"
                            rest = new_str[end_fl:]
                            rest = rest[1:] if rest.startswith("\n") else rest
                            adapted_new = new_str[:end_fl] + "\n" + carried_block + rest

                    # Final guard: every unexpected line must survive in the replacement.
                    adapted_line_set = {ln.strip() for ln in adapted_new.splitlines() if ln.strip()}
                    dropped = [ln for ln in unexpected_lines if ln not in adapted_line_set]
                    if dropped:
                        logger.info(
                            "    Anchor relaxation refused: would drop %d unexpected line(s), e.g. %r",
                            len(dropped), dropped[0][:80],
                        )
                        return None

                    # old_str was stripped for anchoring; if it originally owned the
                    # newline after the suffix, consume it so new_str's trailing newline
                    # does not produce a duplicate blank line.
                    end = pos + len(suffix)
                    if old_str.endswith("\n") and content[end:end + 1] == "\n":
                        end += 1
                    return content[:actual_start] + adapted_new + content[end:]
    return None


def _find_closest_match(old_str: str, content: str) -> Optional[dict]:
    """
    Find the closest matching region in the file for a failed SEARCH block.
    Used to provide actionable diagnostics in error messages.

    Returns dict with 'content', 'start_line', 'end_line', 'ratio' or None.
    """
    from difflib import SequenceMatcher

    search_lines = old_str.strip().splitlines()
    if not search_lines:
        return None

    n = len(search_lines)
    file_lines = content.splitlines()

    if len(file_lines) < n:
        return None

    best_ratio = 0.0
    best_start = -1
    best_text = ""

    # Sample every 1st line to keep O(n) for large files
    step = max(1, (len(file_lines) - n) // 500)
    for i in range(0, len(file_lines) - n + 1, step):
        window = '\n'.join(file_lines[i:i + n])
        ratio = SequenceMatcher(None, old_str.strip(), window.strip()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i
            best_text = window

    if best_ratio >= 0.50:  # Only report if at least 50% similar
        return {
            "content": best_text,
            "start_line": best_start + 1,
            "end_line": best_start + n,
            "ratio": best_ratio,
        }
    return None


def _resolve_within_scope(
    content: str, old_str: str, new_str: str,
    scope_method: str, file_path: str
) -> Optional[str]:
    """
    Tier 6: Resolve an ambiguous old_str by restricting search to a method body.

    Uses brace-counting to find the method named scope_method, then checks
    if old_str appears exactly once inside that region. If yes, replaces
    only that occurrence.

    Returns the patched content, or None if scoping didn't resolve ambiguity.
    """
    # Find the method declaration
    pattern = re.compile(
        r'(?:^|\n)([ \t]*(?:(?:public|private|protected|static|async|final|'
        r'override|abstract|@\w+\s+)*)'
        + re.escape(scope_method) + r'\s*\()',
        re.MULTILINE
    )
    match = pattern.search(content)
    if not match:
        return None

    # Find method body boundaries using brace counting
    start_pos = match.start(1)
    depth = 0
    end_pos = len(content)
    found_open = False
    idx = match.start(1)
    while idx < len(content):
        ch = content[idx]
        if ch == '{':
            depth += 1
            found_open = True
        elif ch == '}':
            depth -= 1
            if found_open and depth == 0:
                end_pos = idx + 1
                break
        idx += 1

    # Extract method body and check for uniqueness within it
    method_body = content[start_pos:end_pos]
    scoped_count = method_body.count(old_str)

    if scoped_count == 1:
        # Unique within scope — find its absolute position
        offset_in_body = method_body.index(old_str)
        abs_pos = start_pos + offset_in_body
        return content[:abs_pos] + new_str + content[abs_pos + len(old_str):]

    # Tier 6 fallback: try whitespace-normalized match within scope
    if scoped_count == 0:
        search_lines = old_str.strip().splitlines()
        if search_lines:
            n = len(search_lines)
            normalized_search = [re.sub(r'\s+', ' ', l.strip()) for l in search_lines]
            body_lines = method_body.splitlines(True)
            body_stripped = [re.sub(r'\s+', ' ', l.strip()) for l in body_lines]
            for j in range(len(body_stripped) - n + 1):
                if body_stripped[j:j + n] == normalized_search:
                    original_window = ''.join(body_lines[j:j + n])
                    abs_pos = start_pos + method_body.index(original_window)
                    _target_indent = _detect_indent(body_lines[j])
                    _search_indent = _detect_indent(search_lines[0])
                    _reindented = _reindent(new_str, _search_indent, _target_indent)
                    return content[:abs_pos] + _reindented + content[abs_pos + len(original_window):]

    return None  # Still ambiguous even within scope


def _find_all_match_lines(content: str, old_str: str) -> list[int]:
    """Return 1-indexed line numbers where old_str starts."""
    lines = []
    start = 0
    while True:
        idx = content.find(old_str, start)
        if idx == -1:
            break
        line_num = content[:idx].count('\n') + 1
        lines.append(line_num)
        start = idx + 1
    return lines


# ── Angular template binding contract validation ─────────────────────────────
_TEMPLATE_BINDING_RE = re.compile(
    r"\[\(\w+\)\]\s*=\s*[\"\']([^\"\']*)"   # [(x)]="expr" bound expression
    r"|\[\w+\]\s*=\s*[\"\']([^\"\']*)"        # [x]="expr" bound expression
    r"|\(\w+\)\s*=\s*[\"\']([^\"\']*)"        # (x)="expr" handler expression
    r"|\*ngIf\s*=\s*[\"\']([^\"\']*)"
    r"|\*ngFor\s*=\s*[\"\']([^\"\']*)"
    r"|\{\{([^}]*)\}\}"         # {{ expr }}
)
_TEMPLATE_IDENTIFIER_RE = re.compile(r"[A-Za-z_$][\w$]*")
# Root identifiers only: a token preceded by a dot is a member of the object
# ({{ displayedMembers.length }}) and is legal as long as the ROOT is declared.
_TEMPLATE_ROOT_IDENTIFIER_RE = re.compile(r"(?<![.\w$])[A-Za-z_$][\w$]*")
_TEMPLATE_KEYWORDS = {
    "true", "false", "null", "undefined", "else", "then", "let", "of", "in",
    "async", "if", "as", "ngif", "ngfor", "ngmodel", "ngtemplateoutlet",
    "index", "first", "last", "even", "odd", "count", "trackby", "click",
    "submit", "change", "input", "blur", "focus", "keydown", "keyup",
}


# Quoted string literals inside binding expressions — i18n keys
# ('cancel.btn.label' | translate) are translation DATA, never controller refs.
_TEMPLATE_STRING_LITERAL_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
# Pipe names: the token directly after | (translate, date, async, number…)
_TEMPLATE_PIPE_NAME_RE = re.compile(r"\|\s*([A-Za-z_$][\w$]*)")
# Template-only builtins that exist in no controller
_TEMPLATE_BUILTIN_VARS = {"$event", "$any"}


def _template_expr_identifiers(expr: str, deep: bool = False) -> List[str]:
    """Identifiers in one binding expression, structurally cleaned.

    String literals are removed BEFORE extraction (so 'home.btn.label' can
    never yield home/btn/label), pipe names are excluded (they are Angular
    pipes, not controller members), and template-only builtins ($event, $any)
    are excluded.  ``deep`` returns member identifiers after a dot instead
    of root identifiers.
    """
    if not expr:
        return []
    cleaned = _TEMPLATE_STRING_LITERAL_RE.sub(" ", expr)
    pipes = set(_TEMPLATE_PIPE_NAME_RE.findall(expr))
    ident_re = _DEEP_MEMBER_RE if deep else _TEMPLATE_ROOT_IDENTIFIER_RE
    out = []
    for tok in ident_re.findall(cleaned):
        if tok in pipes or tok in _TEMPLATE_BUILTIN_VARS or tok.lower() in _TEMPLATE_KEYWORDS:
            continue
        out.append(tok)
    return out


def extract_template_bindings(template_content: str) -> List[str]:
    """Root identifiers referenced by Angular template bindings/interpolations."""
    found: set = set()
    if not template_content:
        return []
    for m in _TEMPLATE_BINDING_RE.finditer(template_content):
        for group in m.groups():
            if not group:
                continue
            found.update(_template_expr_identifiers(group))
    return sorted(found)


_DEEP_MEMBER_RE = re.compile(r"\.\s*([A-Za-z_$][\w$]*)")
_DEEP_MEMBER_ALLOWED = {
    "length", "push", "pop", "map", "filter", "find", "forEach", "includes",
    "slice", "join", "trim", "toLowerCase", "toUpperCase", "split", "concat",
    "some", "every", "indexOf", "reduce", "hasOwnProperty", "toString",
    "name", "id", "value", "label", "text", "target", "valueAsNumber",
    "$any", "$event", "json", "keys", "values", "entries", "assign", "now",
}


def extract_dynamic_viewmodel_fields(companion_text: str, max_fields: int = 40) -> List[str]:
    """Fields dynamically attached to objects INSIDE method bodies — the
    shape members a template can legally bind to but a top-level AST
    contract cannot see (e.g. ``user['isExistingMember'] = true`` or a
    ``{ organization: ..., isExistingMember: isExistingMember }`` literal).

    Deterministic, zero-token, and COMPLETE: it catches dynamic fields in
    unmodified methods too, which a diff/modified-methods slice would miss.
    """
    if not companion_text:
        return []
    patterns = (
        re.compile(r"\b\w+\[['\"](\w+)['\"]\]\s*=(?!=)"),     # x['field'] =
        re.compile(r"(?<!this\.)\b\w+\.(\w+)\s*=(?!=)"),            # x.field =
        re.compile(r"[{,]\s*(\w+)\s*:(?!=)"),                        # { field: ...
    )
    skip = {k.lower() for k in _TEMPLATE_KEYWORDS} | {
        "function", "return", "const", "let", "var", "type", "interface",
        "class", "enum", "case", "default", "switch", "import", "export",
        "new", "typeof", "void", "in", "of", "as", "this", "super", "await",
        "async", "catch", "for", "while", "do", "constructor", "string",
        "number", "boolean", "any", "true", "false", "null", "undefined",
    }
    found: List[str] = []
    seen: set = set()
    for pat in patterns:
        for m in pat.finditer(companion_text):
            name = m.group(1)
            if not name or name in seen or name.lower() in skip:
                continue
            seen.add(name)
            found.append(name)
            if len(found) >= max_fields:
                return found
    return found


def extract_template_deep_members(template_content: str) -> List[str]:
    """Non-root identifiers referenced inside template binding expressions
    (the ``isExistingMember`` in ``*ngIf="dmember.isExistingMember"``)."""
    found: set = set()
    if not template_content:
        return []
    for m in _TEMPLATE_BINDING_RE.finditer(template_content):
        for group in m.groups():
            if not group:
                continue
            found.update(_template_expr_identifiers(group, deep=True))
    return sorted(found)


def validate_template_bindings(
    template_content: str,
    controller_code: str,
    extra_allowed: List[str] = None,
    original_template: Optional[str] = None,
) -> List[str]:
    """Angular binding contract, two levels deep:

    • ROOT identifiers (``dmember``) must be declared in the companion
      controller source.
    • DEEP members (``dmember.isExistingMember``) must exist in the
      controller, in ``extra_allowed`` (the companion's dynamically attached
      view-model fields), or be a well-known member — otherwise they are
      hallucinated bindings that silently evaluate to ``undefined`` at
      runtime (AOT cannot catch them on ``any``-typed objects).
    • PRE-EXISTING bindings in ``original_template`` are baseline code and
      never flagged as regression violations.
    """
    if not template_content or not controller_code:
        return []
    declared = set(_TEMPLATE_IDENTIFIER_RE.findall(controller_code))
    # Template-local declarations are legal roots: *ngFor="let dmember of …",
    # pipe/alias bindings ("x | async as user", "index as i"), #templateRefs,
    # and the template-only builtins.
    declared |= set(re.findall(r"\blet\s+(\w+)\b", template_content))
    declared |= set(re.findall(r"\bas\s+(\w+)\b", template_content))
    declared |= set(re.findall(r"#(\w+)", template_content))
    declared |= _TEMPLATE_BUILTIN_VARS
    allowed_deep = declared | set(extra_allowed or []) | _DEEP_MEMBER_ALLOWED

    pre_existing_roots = set(extract_template_bindings(original_template)) if original_template else set()
    pre_existing_deep = set(extract_template_deep_members(original_template)) if original_template else set()

    violations = [
        b for b in extract_template_bindings(template_content)
        if b not in declared and b not in pre_existing_roots
    ]
    for deep in extract_template_deep_members(template_content):
        if deep not in declared and deep not in allowed_deep and deep not in pre_existing_deep:
            violations.append(deep)
    return violations


def _validate_patch_ast(
    original: str, patched: str, file_path: str
) -> list[str]:
    """
    Post-patch AST validation using fresh tree-sitter parse.

    Parses the patched content and checks for:
    1. Syntax errors introduced by the patch
    2. Drastic reduction in method/class count (truncation)
    3. Import clobbering

    Returns a list of warning strings (empty = all good).
    """
    issues = []

    try:
        from ticket_to_code.agents.smart_extract import (
            parse_with_treesitter,
            has_ast_errors,
            get_ast_error_ranges,
            count_ast_nodes,
        )
    except ImportError:
        return []  # tree-sitter not available — skip validation

    patched_tree, lang = parse_with_treesitter(patched, file_path)
    if patched_tree is None:
        return []  # Can't validate — skip

    # Check 0: Delimiter balance invariant (braces and parentheses)
    if patched.count('{') != patched.count('}') or patched.count('(') != patched.count(')'):
        issues.append(
            f"Patch introduces unbalanced delimiters: braces delta={patched.count('{') - patched.count('}')}, "
            f"parens delta={patched.count('(') - patched.count(')')}"
        )

    # Check 1: Syntax errors introduced by the patch (differential AST check - advisory log)
    orig_tree, _ = parse_with_treesitter(original, file_path)
    if has_ast_errors(patched_tree):
        patched_ranges = get_ast_error_ranges(patched_tree)
        orig_ranges = get_ast_error_ranges(orig_tree) if (orig_tree and has_ast_errors(orig_tree)) else []
        if len(patched_ranges) > len(orig_ranges):
            logger.debug(
                f"Tree-sitter AST advisory for {file_path} at lines: "
                f"{[f'L{s}-L{e}' for s, e in patched_ranges[:5]]}"
            )

    # Check 2: Compare structural counts with original
    if orig_tree is not None:
        if lang == "java":
            node_types = ["method_declaration", "constructor_declaration"]
        elif lang in ("typescript", "javascript"):
            node_types = ["method_definition", "function_declaration"]
        elif lang == "python":
            node_types = ["function_definition"]
        else:
            node_types = []

        if node_types:
            orig_count = count_ast_nodes(orig_tree, node_types)
            new_count = count_ast_nodes(patched_tree, node_types)
            if orig_count > 0 and new_count < orig_count - 1:
                issues.append(
                    f"Method count dropped: {orig_count} → {new_count} "
                    f"(possible truncation or accidental deletion)"
                )

    # Check 3: Structural Change Manifest (classes, methods, properties, constructors, templates)
    try:
        from ticket_to_code.integrity.structural_manifest import (
            StructuralChangeManifest,
            DeleteAuthorizationValidator,
        )
        delta = StructuralChangeManifest.compute_delta(original, patched, file_path)
        if delta.removed_symbols:
            for sym in delta.removed_symbols:
                issues.append(
                    f"Structural symbol removed without explicit authorization: '{sym}'"
                )
    except Exception as _scm_err:
        logger.debug(f"Structural manifest validation skipped: {_scm_err}")

    return issues










class CodeGeneratorAgent:
    """
    Agent 5: Code Generation
    
    Generates production-ready, idiomatic code using:
    - Validated requirements
    - Retrieved code context
    - Task specifications
    
    Features:
    - Context-aware generation (no hallucination)
    - Proper error handling
    - Documentation generation
    - Idiomatic patterns
    """
    
    def __init__(self, workspace_path: Optional[Union[str, Path]] = None, llm: Optional[Any] = None):
        # Always use the smarter assistant model (gemini-1.5-pro / gemini-2.5-flash) for coding unless injected
        self.llm = llm or LLMRegistry.get_llm(assistant=True)
        self._workspace_path: Optional[str] = str(workspace_path) if workspace_path else None
        logger.info("Code Generator Agent initialized")

    @property
    def workspace_path(self) -> Optional[Path]:
        ws = getattr(self, "_workspace_path", None)
        return Path(ws) if ws else None

    @workspace_path.setter
    def workspace_path(self, val: Optional[Union[str, Path]]) -> None:
        self._workspace_path = str(val) if val else None

    def _maybe_dump_llm_io(self, task: DevelopmentTask, messages, full_content: str, cont_attempt: int) -> None:
        """Best-effort debug dump for prompt/response text.

        Dumps are disabled by default. Set AVIATOR_DEBUG_LLM_IO=1 to enable.
        """
        if os.getenv("AVIATOR_DEBUG_LLM_IO", "").strip().lower() not in {"1", "true", "yes", "on"}:
            return

        try:
            base = Path(os.getenv("AVIATOR_DEBUG_DIR", ".aviator")) / "llm_debug"
            task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(getattr(task, "id", "task")))
            debug_dir = base / task_id
            debug_dir.mkdir(parents=True, exist_ok=True)

            (debug_dir / f"{task_id}_prompt_{cont_attempt}.txt").write_text(
                str(messages), encoding="utf-8"
            )
            (debug_dir / f"{task_id}_response_{cont_attempt}.txt").write_text(
                full_content, encoding="utf-8"
            )
        except Exception as dump_err:
            logger.debug(f"LLM debug dump skipped: {dump_err}")
    
    # ── Cross-tier call validation (no hallucinated service methods) ────────
    # Top AI IDEs never invent a call chain: every `this.<svc>.<method>()` in
    # generated code must resolve to a REAL method — in the repo, or generated
    # earlier in this same run (session files). Observed failure (2026-09-18):
    # the generator called `this.contractService.getProjectMembership()` — a
    # method that existed nowhere — burning 2 TSC retries and deferring to
    # fix_build. This check catches it BEFORE the code is written to disk.
    _SVC_CALL_RE = re.compile(r"this\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    _CTOR_INJECT_RE = re.compile(
        r"(?:private|public|protected)\s+(?:readonly\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)"
    )
    _METHOD_NAME_RE = re.compile(
        r"(?:^|[^.\w$])([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*(?:\{|:)",
        re.MULTILINE,
    )

    def _find_class_source(self, class_name: str) -> Optional[str]:
        """Locate the source text of a TypeScript class by name.

        Order: session files (generated/modified this run — freshest), then a
        bounded workspace scan (services first, capped so it can never hang).
        Returns None when unresolvable (caller must not false-flag).
        """
        session = getattr(self, "_session_files", None) or {}
        for _key, text in session.items():
            try:
                if class_name in text:
                    return text
            except Exception:
                continue
        ws = getattr(self, "workspace_path", None) or getattr(self, "_workspace_path", None)
        if not ws:
            return None
        ws_root = Path(ws)
        scanned = 0
        MAX_SCAN = 3000
        try:
            for fp in ws_root.rglob("*.ts"):
                scanned += 1
                if scanned > MAX_SCAN:
                    break
                parts = fp.parts
                if any(p in ("node_modules", ".git", "dist", ".aviator") for p in parts):
                    continue
                if fp.name.endswith(".spec.ts"):
                    continue
                try:
                    head = fp.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if f"class {class_name}" in head:
                    return head
        except Exception:
            return None
        return None

    def _validate_service_calls(self, content: str, existing_content: Optional[str]) -> list:
        """Return human-readable descriptions of `this.<prop>.<method>()` calls
        whose method does not exist on the injected service's class. Only calls
        whose service class CAN be resolved are validated (unresolvable ones
        are left to the TSC gate — no false positives).
        """
        try:
            src = (existing_content or "") + "\n" + (content or "")
            calls = set(self._SVC_CALL_RE.findall(src))
            if not calls:
                return []
            prop_to_class = dict(self._CTOR_INJECT_RE.findall(src))
            if not prop_to_class:
                return []
            own_methods = {m.group(1) for m in self._METHOD_NAME_RE.finditer(src)}
            unknown = []
            for prop, method in sorted(calls):
                cls = prop_to_class.get(prop)
                if cls is None:
                    continue
                if method in own_methods:
                    continue
                class_src = self._find_class_source(cls)
                if class_src is None:
                    continue
                cls_methods = {m.group(1) for m in self._METHOD_NAME_RE.finditer(class_src)}
                if method not in cls_methods:
                    unknown.append(
                        f"this.{prop}.{method}() — class {cls} has no method '{method}' "
                        f"(existing: {', '.join(sorted(cls_methods)[:10])})"
                    )
            return unknown
        except Exception:
            return []

    def generate_code(
        self,
        task: DevelopmentTask,
        requirements: StructuredRequirements,
        context: List[CodeChunk] = None,
        existing_content: str = None,
        allowed_files: List[str] = None,
        readonly_files: List[str] = None,
        dataflow_contract: Any = None,
        **kwargs: Any,
    ) -> GeneratedCode:
        """
        Generate code for a specific development task.
        
        Args:
            task: Development task to implement
            requirements: Structured requirements
            context: Retrieved code context
            
        Returns:
            Generated code with metadata
        """
        if context is None:
            context = kwargs.get("retrieved_context") or []
        print("\n" + "="*80)
        print("[ENTER] CodeGeneratorAgent.generate_code() in code_generator.py")
        print("   Purpose: Generate actual code using LLM with RAG context")
        print(f"   Task: {task.id} - {task.title}")
        print(f"   File: {task.file_path}")
        print("="*80)
        logger.info(f"Generating code for task: {task.id} - {task.title}")
        
        self._current_existing_content = existing_content
        self._current_allowed_files = set(allowed_files or [])
        self._current_readonly_files = set(readonly_files or [])
        if dataflow_contract is not None:
            self._dataflow_contract = dataflow_contract
        elif "_dataflow_contract" in kwargs and kwargs["_dataflow_contract"] is not None:
            self._dataflow_contract = kwargs["_dataflow_contract"]
        
        # ── LSP context injection for HTML/Angular templates ──────────────────
        # Before generating, query the TypeScript LSP for the sibling controller's
        # declared class members. This gives the HTML generator exact property names
        # instead of invented ones — the same capability that makes Cursor accurate.
        # ── Polyglot Companion & Scope context injection ─────────────────────
        # Before generating, query the companion provider for declared symbols.
        # This gives the template/consumer generator exact property & method names
        # instead of invented ones — the same capability that makes Cursor accurate.
        _lsp_context_block = ""
        _binding_violations = list((kwargs or {}).get("binding_violations") or [])
        try:
            from ticket_to_code.intelligence.contracts.companion_resolver import PolyglotCompanionResolver
            from ticket_to_code.intelligence.contracts import ComponentContract
            _session = getattr(self, "_session_files", None) or {}
            _ws = getattr(self, "workspace_path", None) or getattr(self, "_workspace_path", None)
            _companion = PolyglotCompanionResolver.resolve_companion(
                task.file_path, session_files=_session, workspace_path=str(_ws) if _ws else None
            )
            if _companion:
                _comp_path, _comp_text = _companion
                _companion_sections = []
                # 1. Structured AST/LSP Contract (properties, types, methods, binding invariants)
                _ast_contract_rendered = False
                if _comp_path.endswith((".ts", ".js", ".tsx")):
                    _c_contract = ComponentContract.extract_from_ts(_comp_text, file_path=_comp_path)
                    if _c_contract and _c_contract.properties:
                        _companion_sections.append(_c_contract.render_prompt_block())
                        _ast_contract_rendered = True
                        logger.info(
                            f"  [CrossFileCoherence] Injected AST contract ({len(_c_contract.properties)} properties + "
                            f"{len(_c_contract.methods)} methods) from companion {_comp_path}"
                        )
                # 2. Verbatim companion source code (from active in-memory session or disk)
                # GENERATION CONTEXT DIET: when the structured AST contract was
                # rendered it already carries the complete binding surface
                # (properties, types, methods).  Duplicating up to 30k chars of
                # raw source dilutes LLM attention for zero accuracy gain —
                # ~8-12k tokens saved per generation call.  The raw dump is
                # kept ONLY when no contract could be extracted (e.g. non-TS
                # companions), where it is the sole source of truth.
                if _ast_contract_rendered:
                    # Deep-binding blindspot fix: object-literal / bracket
                    # assignments inside method bodies create fields the
                    # top-level contract cannot list.  A compact dynamic-field
                    # index (NOT a raw source dump) keeps template generation
                    # from inventing dmember.isExistingMember-style names.
                    _dyn_fields = extract_dynamic_viewmodel_fields(_comp_text)
                    if _dyn_fields:
                        _companion_sections.append(
                            "DYNAMIC VIEW-MODEL FIELDS — attached to objects inside method "
                            "bodies; valid template bindings even though they are not "
                            "declared class members:\n"
                            + ", ".join(_dyn_fields)
                        )
                        logger.info(
                            f"  [CrossFileCoherence] Injected {len(_dyn_fields)} dynamic "
                            "view-model field(s) from companion method bodies"
                        )
                if not _ast_contract_rendered:
                    _comp_display = _comp_text if len(_comp_text) <= 30000 else _comp_text[:15000]
                    _companion_sections.append(
                        f"════════════════════════════════════════════════════════════════\n"
                        f"COMPANION SOURCE FILE (active session): {_comp_path}\n"
                        f"════════════════════════════════════════════════════════════════\n"
                        f"{_comp_display}\n"
                        f"════════════════════════════════════════════════════════════════"
                    )
                if _binding_violations:
                    _companion_sections.append(
                        "⛔ FORBIDDEN BINDINGS — the previous attempt referenced identifiers that DO NOT "
                        f"exist in the controller: {', '.join(_binding_violations)}. Regenerate using ONLY "
                        "declared controller members; remove or correct every forbidden binding. If a "
                        "needed property is genuinely missing from the controller, say so explicitly in "
                        "documentation instead of inventing a binding."
                    )
                _lsp_context_block = "\n\n".join(_companion_sections) + "\n\n"
        except Exception as _ctrl_exc:
            logger.debug(f"  [CompanionContract] Context injection skipped: {_ctrl_exc}")

        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Build generation prompt — unified for all task types
                system_prompt = self._build_generation_prompt(task.language)
                user_prompt = self._build_user_prompt(
                    task, requirements, context,
                    existing_content=existing_content,
                    allowed_files=allowed_files,
                    readonly_files=readonly_files,
                )
                
                # Append LSP context to user prompt for HTML tasks
                if _lsp_context_block:
                    user_prompt = _lsp_context_block + user_prompt

                # Append Behavioral Invariants with Provenance (P0 Pre-generation prevention)
                _dod = getattr(self, "_definition_of_done", None)
                if _dod and hasattr(_dod, "render_invariants_block"):
                    _inv_block = _dod.render_invariants_block()
                    if _inv_block:
                        user_prompt = _inv_block + "\n" + user_prompt

                # Append Pre-Generation DataFlow Contract (P0 Dataflow Mandate)
                _df_contract = getattr(self, "_dataflow_contract", None)
                if not _df_contract:
                    try:
                        from ticket_to_code.intelligence.contracts.dataflow_contract import DataFlowContract
                        _ticket_desc = getattr(requirements, "ticket_description", "") if requirements else ""
                        if not _ticket_desc and hasattr(task, "description"):
                            _ticket_desc = task.description
                        _df_contract = DataFlowContract.extract_from_requirements(requirements, _ticket_desc)
                    except Exception:
                        pass
                if _df_contract:
                    if hasattr(_df_contract, "render_pre_generation_prompt_block"):
                        _df_block = _df_contract.render_pre_generation_prompt_block()
                    elif isinstance(_df_contract, str):
                        _df_block = _df_contract
                    else:
                        _df_block = str(_df_contract)
                    if _df_block:
                        user_prompt = _df_block + "\n" + user_prompt
                
                self.last_system_prompt = system_prompt
                self.last_user_prompt = user_prompt
                
                # Enforce strict delimiters on every attempt (including attempt 0)
                delimiter_mandate = (
                    "\n\nCRITICAL FORMAT REQUIREMENT (MARKET AI IDE STANDARD):\n"
                    "1. Wrap your entire code output strictly within <<<AVIATOR_CODE_START>>> and <<<AVIATOR_CODE_END>>> delimiters.\n"
                    "2. For MODIFY tasks on existing files, you MUST use exact <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks.\n"
                    "   - The SEARCH block MUST be copied verbatim (character-for-character) from the EXISTING FILE CONTENT above.\n"
                    "   - The REPLACE block contains your new replacement code.\n"
                    "   - NEVER rewrite the entire file for a MODIFY task.\n"
                    "   - NEVER use lazy placeholders like '...' or '// existing code...'.\n"
                    "3. Do NOT output ANY conversational text, explanations, notes, or prose anywhere in your response (neither inside nor outside the delimiters).\n"
                    "   Output ONLY the code delimiters and the SEARCH/REPLACE blocks."
                )
                if attempt > 0:
                    delimiter_mandate += "\n\n⚠️ PREVIOUS ATTEMPT FAILED: You must provide exact SEARCH blocks matching the file content."
                user_prompt += delimiter_mandate
                
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                full_content = ""
                max_continuations = 3
                for cont_attempt in range(max_continuations):
                    # Increase max output tokens to 4096 to prevent truncation on large reasoning models
                    effective_max = 8192
                    response = llm_invoke(self.llm, messages, max_tokens=effective_max)
                    # B10: validate each merged segment for boundary integrity
                    merged = full_content + response.content
                    if cont_attempt > 0:
                        # Check that the merge boundary is not mid-token (e.g. split inside a string)
                        _merge_ok = self._validate_merge_boundary(full_content, response.content, task)
                        if not _merge_ok:
                            logger.warning(
                                "B10: merge boundary validation failed on continuation %d "
                                "— re-requesting that segment",
                                cont_attempt,
                            )
                            messages.append(AIMessage(content=response.content))
                            messages.append(HumanMessage(
                                content=(
                                    "The previous continuation appears to have split mid-token or "
                                    "mid-block. Please restart from the last complete statement and "
                                    "continue to the <<<AVIATOR_CODE_END>>> delimiter."
                                )
                            ))
                            continue  # retry this continuation slot
                    full_content = merged

                    try:
                        # Optional best-effort debug dump; never affects generation flow.
                        self._maybe_dump_llm_io(task, messages, full_content, cont_attempt)

                        # Parse response
                        generated = self._parse_response(full_content, task)
                        logger.info(f"Code generated: {len(generated.content)} chars")

                        # ── Cross-tier call validation (pre-write) ──
                        _unknown_calls = self._validate_service_calls(generated.content, existing_content)
                        if _unknown_calls:
                            if cont_attempt < max_continuations - 1:
                                logger.warning(
                                    f"  ⛔ Hallucinated service call(s) in generated code — "
                                    f"requesting correction before writing: {_unknown_calls}"
                                )
                                messages.append(AIMessage(content=response.content))
                                messages.append(HumanMessage(
                                    content=(
                                        "Your generated code calls methods that DO NOT EXIST:\n"
                                        + "\n".join(f"  - {c}" for c in _unknown_calls)
                                        + "\n\nFix the code: use ONLY methods that exist on the "
                                        "injected services (their real methods were listed in the "
                                        "context above). Prefer REUSING an existing service "
                                        "capability over inventing a new method. Do not invent "
                                        "methods or endpoints. Re-output the FULL corrected code "
                                        "between the delimiters."
                                    )
                                ))
                                full_content = ""
                                continue  # re-invoke LLM with the corrective context
                            else:
                                logger.error(
                                    f"  ⛔ Hallucinated service call(s) persist after retries — "
                                    f"writing anyway (TSC/build gates will verify): {_unknown_calls}"
                                )

                        # ── Angular Template Contract Validation (pre-write targeted repair) ──
                        if task.file_path.endswith((".component.html", ".html")):
                            _tpl_contract = None
                            _impl = getattr(self, "_impl_state", None)
                            if _impl is not None:
                                _tpl_contract = _impl.get_component_contract_for_template(task.file_path)
                            if _tpl_contract is None:
                                from ticket_to_code.intelligence.contracts.companion_resolver import PolyglotCompanionResolver
                                from ticket_to_code.intelligence.contracts import ComponentContract
                                _session = getattr(self, "_session_files", {}) or {}
                                _ws = getattr(self, "workspace_path", None) or getattr(self, "_workspace_path", None)
                                _comp = PolyglotCompanionResolver.resolve_companion(
                                    task.file_path, session_files=_session, workspace_path=str(_ws) if _ws else None
                                )
                                if _comp:
                                    _comp_p, _comp_t = _comp
                                    if _comp_p.endswith((".ts", ".js", ".tsx")):
                                        _tpl_contract = ComponentContract.extract_from_ts(_comp_t, file_path=_comp_p)

                            if _tpl_contract is not None:
                                from ticket_to_code.intelligence.contracts import TemplateContractValidator
                                _tpl_violations = TemplateContractValidator.validate(generated.content, _tpl_contract, original_content=existing_content)
                                if _tpl_violations:
                                    if cont_attempt < max_continuations - 1:
                                        logger.warning(
                                            f"  ⛔ Template contract violation(s) — requesting targeted repair: {_tpl_violations}"
                                        )
                                        messages.append(AIMessage(content=response.content))
                                        messages.append(HumanMessage(
                                            content=(
                                                "TEMPLATE CONTRACT VIOLATION: Your generated HTML template binds to properties that DO NOT EXIST on the controller:\n"
                                                + "\n".join(f"  - {v}" for v in _tpl_violations)
                                                + "\n\n"
                                                + _tpl_contract.render_prompt_block()
                                                + "\n\nFix the template: bind ONLY to properties declared on the controller.\n"
                                                "Do NOT invent fields on objects (e.g. do not use selectedUser.isExistingProjectMember if isExistingMember is on the controller).\n"
                                                "Re-output the FULL corrected HTML template between the delimiters."
                                            )
                                        ))
                                        full_content = ""
                                        continue
                                    else:
                                        logger.error(
                                            f"  ⛔ Template contract violations persist after retries: {_tpl_violations}"
                                        )

                        # ── TypeScript Type Contract Validation (pre-write targeted repair) ──
                        if task.file_path.endswith((".ts", ".tsx")):
                            import re as _gen_tc_re
                            _tc_violations = []
                            if _gen_tc_re.search(r"\bemail\s*:\s*\{", generated.content):
                                _tc_violations.append(
                                    "Field 'email' in filter is defined as 'FilterStringInput[]' (ARRAY). "
                                    "Passing scalar object '{ eq: searchDataElement }' causes TS2322. "
                                    "Must be wrapped in an array: '[{ eq: searchDataElement }]'."
                                )
                            if _tc_violations:
                                if cont_attempt < max_continuations - 1:
                                    logger.warning(
                                        f"  ⛔ Type contract violation(s) in generated TS: {_tc_violations}"
                                    )
                                    messages.append(AIMessage(content=response.content))
                                    messages.append(HumanMessage(
                                        content=(
                                            "TYPE CONTRACT VIOLATION: Your generated TypeScript violates the repository type contract:\n"
                                            + "\n".join(f"  - {v}" for v in _tc_violations)
                                            + "\n\nFix the code immediately:\n"
                                            "  - Ensure array fields are wrapped in brackets: e.g. email: [{ eq: searchDataElement }] (NOT email: { eq: ... }).\n"
                                            "Re-output the FULL corrected code between the delimiters."
                                        )
                                    ))
                                    full_content = ""
                                    continue
                                else:
                                    logger.error(
                                        f"  ⛔ Type contract violations persist after retries: {_tc_violations}"
                                    )


                        if task.file_path.endswith((".java", ".cs")):
                            generated.content = heal_trailing_class_methods(generated.content, task.file_path)
                        elif task.file_path.endswith((".tsx", ".jsx")):
                            generated.content = _heal_react_destructuring(generated.content)

                        # ── Immediate in-memory session propagation ──
                        if self._session_files is None:
                            self._session_files = {}
                        _norm_fp = task.file_path.replace("\\", "/").lower()
                        self._session_files[_norm_fp] = generated.content
                        self._session_files[task.file_path] = generated.content
                        self._session_files[Path(task.file_path).name.lower()] = generated.content

                        # ── Post-generation: record in ImplementationState ──
                        _impl = getattr(self, "_impl_state", None)
                        if _impl is not None:
                            try:
                                _impl.record_generated(
                                    task_id=task.id,
                                    file_path=task.file_path,
                                    content=generated.content,
                                )
                            except Exception as _rec_exc:
                                logger.debug(
                                    f"  [ImplementationState] record_generated failed: {_rec_exc}"
                                )

                        return generated
                    except ValueError as e:
                        if "AVIATOR_CODE_END missing" in str(e) and cont_attempt < max_continuations - 1:
                            logger.info(f"LLM output truncated on attempt {cont_attempt + 1}, asking to continue...")
                            messages.append(AIMessage(content=response.content))
                            messages.append(HumanMessage(content="Your output was truncated before closing the <<<AVIATOR_CODE_END>>> tag. Please continue EXACTLY from where you left off. Do not repeat what you have already written. Just output the next characters."))
                        else:
                            raise e
                
            except (json.JSONDecodeError, ValueError) as e:
                _err_msg = str(e)
                if attempt < max_retries - 1:
                    logger.warning(f"Generation failed (attempt {attempt + 1}/{max_retries}), retrying: {_err_msg[:200]}")
                    
                    # ── Fix 8: On retry after SEARCH error, re-read file + refresh line numbers ──
                    # If the error message contains actual file context, the LLM saw the real code.
                    # But in case other tasks modified the file, re-read it now.
                    if "SEARCH block not found" in _err_msg and task.task_type.value == "modify":
                        _ws = getattr(self, "workspace_path", None) or getattr(self, "_workspace_path", None)
                        _retry_path = (Path(_ws) if _ws else Path(".")) / task.file_path
                        if _retry_path.exists():
                            try:
                                _fresh_content = _retry_path.read_text(encoding='utf-8')
                                self._current_existing_content = _fresh_content
                                
                                # Re-build numbered content for the retry prompt
                                def _numbered_retry(text: str) -> str:
                                    lines = text.split('\n')
                                    return '\n'.join(f'{i+1:4d} | {ln}' for i, ln in enumerate(lines[:min(len(lines), 50)]))
                                
                                _fresh_numbered = _numbered_retry(_fresh_content)
                                
                                # Re-build the user prompt with the fresh content + error message
                                user_prompt = self._build_user_prompt(
                                    task, requirements, context,
                                    existing_content=_fresh_content,
                                    allowed_files=allowed_files,
                                    readonly_files=readonly_files,
                                )
                                
                                # Append error context so LLM learns what went wrong
                                user_prompt += (
                                    f"\n\n⚠️  RETRY ATTEMPT {attempt + 1}:\n"
                                    f"Your previous SEARCH block did not match the file. "
                                    f"Here is the CURRENT file state (may have changed since your first attempt):\n"
                                    f"```\n{_fresh_numbered}\n```\n"
                                    f"Error was: {_err_msg[:500]}\n"
                                    f"Copy 3-4 lines EXACTLY from the current file above into your SEARCH block."
                                )
                                
                                if _lsp_context_block:
                                    user_prompt = _lsp_context_block + user_prompt
                                
                                # Rebuild message stack with the updated prompt
                                messages = [
                                    SystemMessage(content=system_prompt),
                                    HumanMessage(content=user_prompt)
                                ]
                                logger.info(f"Retry: refreshed file content and re-numbered for {task.file_path}")
                            except Exception as refresh_err:
                                logger.warning(f"Could not refresh file on retry: {refresh_err}")
                                # Fall through with original messages
                        continue
                    continue
                else:
                    logger.warning(f"All retry attempts exhausted for task {task.id}. Preserving original file content.")
                    return GeneratedCode(
                        file_path=task.file_path,
                        content=existing_content or self._current_existing_content or "",
                        language=task.language,
                        change_type=task.task_type,
                        documentation=f"Preserved original content after error: {_err_msg[:200]}",
                        imports=[],
                    )
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Generation encountered transient error (attempt {attempt + 1}/{max_retries}), retrying in 2s: {e}")
                    import time as _t; _t.sleep(2.0)
                    continue
                logger.warning(f"Code generation failed for task {task.id} after retries: {e}. Preserving original file content.")
                return GeneratedCode(
                    file_path=task.file_path,
                    content=existing_content or self._current_existing_content or "",
                    language=task.language,
                    change_type=task.task_type,
                    documentation=f"Preserved original content after exception: {e}",
                    imports=[],
                )

        return GeneratedCode(
            file_path=task.file_path,
            content=existing_content or self._current_existing_content or "",
            language=task.language,
            change_type=task.task_type,
            documentation="Preserved original content (loop exit fallback)",
            imports=[],
        )
    
    # ── B10: Merge boundary validation ────────────────────────────────────────

    @staticmethod
    def _validate_merge_boundary(
        before: str, after: str, task: "DevelopmentTask"
    ) -> bool:
        """
        B10: Lightweight heuristic check that a continuation merge is syntactically sound.

        Checks:
          1. The join point is not inside an unterminated string literal
             (i.e. the before-portion has balanced single/double quotes).
          2. For code inside a SEARCH/REPLACE block, the boundary is not inside
             an indented code token (split mid-identifier or mid-comment).

        Returns True if the merge looks safe, False if it should be re-requested.

        This is intentionally conservative (may return True for some bad merges)
        to avoid excessive re-requests.  The PatchValidator/build step is the
        authoritative correctness gate.
        """
        if not before or not after:
            return True  # nothing to validate

        # Only validate the tail of `before` + head of `after`
        tail = before[-200:]
        head = after[:200]

        # Heuristic 1: unclosed string at merge boundary?
        # Count unescaped quote characters in the tail region.
        def _unbalanced_quotes(s: str, ch: str) -> bool:
            count = 0
            escaped = False
            for c in s:
                if escaped:
                    escaped = False
                    continue
                if c == "\\":
                    escaped = True
                    continue
                if c == ch:
                    count += 1
            return count % 2 != 0

        for quote_char in ('"', "'", "`"):
            if _unbalanced_quotes(tail, quote_char):
                logger.debug(
                    "B10: unbalanced %r quote at merge boundary for %s",
                    quote_char, getattr(task, "file_path", "?"),
                )
                return False

        # Heuristic 2: head starts with a continuation of an identifier (split mid-word)?
        head_stripped = head.lstrip()
        if head_stripped and head_stripped[0].isalnum() or head_stripped[:1] in ("_",):
            # Tail ends mid-word (last char is alphanumeric/underscore)
            tail_last = tail.rstrip()[-1:] if tail.rstrip() else ""
            if tail_last and (tail_last.isalnum() or tail_last == "_"):
                logger.debug(
                    "B10: mid-identifier split detected at merge boundary for %s",
                    getattr(task, "file_path", "?"),
                )
                return False

        return True

    def _build_generation_prompt(self, language: ProgrammingLanguage) -> str:
        
        language_guidelines = {
            ProgrammingLanguage.CSHARP: self._csharp_guidelines(),
            ProgrammingLanguage.TYPESCRIPT: self._typescript_guidelines(),
            ProgrammingLanguage.PYTHON: self._python_guidelines(),
            ProgrammingLanguage.JAVA: self._java_guidelines(),
            ProgrammingLanguage.HTML: self._html_guidelines(),
            ProgrammingLanguage.SCSS: self._scss_guidelines(),
            ProgrammingLanguage.JSON: self._json_guidelines(),
            ProgrammingLanguage.XML: self._xml_guidelines(),
            ProgrammingLanguage.YAML: self._yaml_guidelines(),
            ProgrammingLanguage.PROPERTIES: self._properties_guidelines(),
        }
        
        lang_guide = language_guidelines.get(
            language, 
            "Follow language best practices."
        )
        
        return f"""You are a Senior Maintainer working on an enterprise production codebase.

YOUR ROLE IS NOT ARCHITECT. YOUR ROLE IS SURGICAL MODIFIER.

ABSOLUTE RULES — NEVER VIOLATE:
1. ONLY operate on the ONE file explicitly specified in the FROZEN MODIFICATION SCOPE. Period.
2. For MODIFY tasks: Apply the SMALLEST change that satisfies the requirement.
3. For CREATE tasks: Generate the COMPLETE content for the EXACT file path specified.
4. NEVER remove unrelated methods, classes, or logic. For logic being replaced, directly replace it with the new implementation in the REPLACE block. Do NOT leave commented-out dead code or syntax-breaking comment tokens.
5. NEVER rewrite existing methods to hardcode return values if the requirement is to update a configuration constant or version string.
6. PREFER modifying variables, constants, and configuration objects over modifying business logic methods.
7. If constructors have existing parameters, PRESERVE ALL of them — only ADD new ones.

CRITICAL — PLANNER TASK ADHERENCE:
- The PLANNER DIRECTIVE section below contains the exact task assigned to you by the planning agent.
- You MUST execute EXACTLY what the planner requested — no more, no less.
- The planner's task is your contract. Deviating from it is a failure.

MINDSET:
- Think diff, not rewrite.
- Touch only what must change.
- Leave everything else byte-for-byte identical.
- When in doubt: copy it unchanged. Never delete.

PRESERVE ENCODING: If you are asked to update text, preserve any special characters (like trademark symbols) in the surrounding code. Do NOT output invalid or replacement characters (e.g. \\ufffd) for symbols.

{lang_guide}

RESPOND using EXACTLY this delimiter format:

<<<AVIATOR_CODE_START>>>
[Output one or more EDIT blocks for MODIFY tasks, or the full file for CREATE tasks]

── FOR MODIFY TASKS: Use SEARCH/REPLACE edits ───────────────────────────

Each edit replaces an exact unique substring of the existing file.
Use git-merge-conflict-style delimiters (7 angle brackets + keyword):

<<<<<<< SEARCH
exact existing code to find (must match EXACTLY ONCE in the file)
=======
replacement code
>>>>>>> REPLACE

You may output MULTIPLE edits in a single response:

<<<<<<< SEARCH
import {{ ExistingService }} from './existing.service';
=======
import {{ ExistingService }} from './existing.service';
import {{ NewService }} from './new.service';
>>>>>>> REPLACE

<<<<<<< SEARCH
  onMemberAdd(member: any) {{
    this.displayedMembers.push(member);
  }}
=======
  onMemberAdd(member: any) {{
    if (this.isDuplicate(member)) {{
      return;
    }}
    this.displayedMembers.push(member);
  }}
>>>>>>> REPLACE

For HTML / Template / Markup files:
<<<<<<< SEARCH
        <div class="field-container">
          <input type="text" [value]="item.name" />
        </div>
=======
        <div class="field-container">
          <input *ngIf="!item.isReadOnly" type="text" [value]="item.name" />
          <span *ngIf="item.isReadOnly" class="static-label">{{ item.name }}</span>
        </div>
>>>>>>> REPLACE

RULES FOR SEARCH/REPLACE EDITS (TARGETED PART REPLACEMENT ONLY):
- Write ONLY the specific part of code you want to replace (3 to 8 lines). Do NOT rewrite large blocks or full functions.
- Directly replace deprecated code with the updated implementation in the REPLACE block. Never leave commented-out dead code or invalid comment tokens.
- The SEARCH block must be an EXACT substring of the EXISTING FILE shown below — copy it character-for-character including indentation
- The SEARCH block must match EXACTLY ONCE in the file. If it could match multiple places, include more surrounding lines to make it unique
- If the SEARCH block is not found → the edit FAILS and you will be asked to retry
- For ADDING new code (new method, new property): use a small anchor from existing code as SEARCH, and include anchor + new code as REPLACE
- For INSERTING a new method after an existing method: use the closing brace + any trailing lines of the preceding method as SEARCH, include that same closing brace + the new method as REPLACE
- NEVER re-declare existing fields, class annotations, or existing methods when adding a new method to an existing class. Include ONLY the single NEW method in the REPLACE block.
- For ADDING imports: use the last existing import line as SEARCH, include that import + new import as REPLACE
- For React/JSX markup: <td> and <th> elements MUST strictly reside inside <tr> inside <tbody> or <thead> within a <table>. NEVER insert raw <td> or <th> tags outside a table row. When adding or modifying table row actions, anchor to the existing action cell (<td>) inside the iteration map.
- NEVER output the full file for a MODIFY task — only the SEARCH/REPLACE edits
- PURE CODE ONLY: The REPLACE block must contain ONLY pure, syntactically valid code. NEVER include conversational monologue, reasoning paragraphs, option trade-off discussions, or conversational English notes inside the code.
- Balanced Delimiters: Every opening curly brace or parenthesis in your REPLACE block MUST have its corresponding closing brace or parenthesis. Unbalanced delimiters cause INSTANT REJECTION.
- You may also use the TARGET / REPLACEMENT format:
  TARGET:
  <exact existing code to find (3 to 8 lines only)>
  REPLACEMENT:
  <replacement code for that specific part only — comment out deprecated logic rather than deleting>

⛔ CRITICAL — FULL FILE OUTPUT = INSTANT REJECTION:
- For MODIFY tasks, if you output a full file (starting with 'package ', 'import ', class declarations, etc.)
  instead of SEARCH/REPLACE blocks, your output will be AUTOMATICALLY REJECTED and you will waste a retry.
- ALWAYS use <<<<<<< SEARCH / ======= / >>>>>>> REPLACE format for MODIFY tasks.
- Copy 3-5 lines EXACTLY from the existing file as your SEARCH anchor.
- Only include the lines that are changing plus minimal surrounding context.
- Even if the file is very large, you MUST use targeted SEARCH/REPLACE edits — never rewrite the whole file.

── FOR CREATE TASKS: Output the entire new file ────────────────────────

For CREATE tasks, output the complete file content directly (no SEARCH/REPLACE blocks).

<<<AVIATOR_CODE_END>>>
<<<AVIATOR_META_START>>>
{{"imports": [], "documentation": "One sentence: what exact change was made and why"}}
<<<AVIATOR_META_END>>>

STRATEGY SELECTION RULES:
- For MODIFY tasks: ALWAYS use SEARCH/REPLACE blocks with <<<<<<< SEARCH / ======= / >>>>>>> REPLACE delimiters or TARGET: / REPLACEMENT: blocks
- For CREATE tasks: output the entire new file content inside <<<AVIATOR_CODE_START>>> without any SEARCH/REPLACE blocks
- The META block must be valid JSON with "imports" (array) and "documentation" (string)
- No text outside the two blocks"""


    def _java_guidelines(self) -> str:
        """Java / Spring Boot specific guidelines"""
        return """
JAVA GUIDELINES:
- Use camelCase for methods/variables, PascalCase for classes
- Annotate Spring beans with @Service, @Component, @RestController, @Repository as appropriate
- Use constructor injection (not @Autowired on fields)
- Add Javadoc for public methods
- Use SLF4J (private static final Logger log = LoggerFactory.getLogger(Foo.class))
- Prefer Optional<> over returning null
- Use final fields where possible
- Follow existing package/import structure already in the file
- STRICT PRESERVATION OF EXISTING IDENTIFIERS: Never rename, refactor, or "fix typos" in existing variable/field/method names (e.g. if an existing field is named 'participantingMemberTypeRef', KEEP IT EXACTLY AS 'participantingMemberTypeRef'). Renaming existing fields breaks other call sites and causes compilation failures.
- SPRING DATA JPA REPOSITORY QUERIES: When adding a query method on a Repository interface to count contracts for a user in a project, name the method countByUserIdAndProjectId:
  @Query("select count(cm) from ContractMember cm INNER JOIN cm.contract c where cm.userId = :userId and c.projectId = :projectId")
  long countByUserIdAndProjectId(@Param("userId") UUID userId, @Param("projectId") UUID projectId);
  Callers in services and mocks in unit tests must consistently use contractMemberRepository.countByUserIdAndProjectId(userId, projectId). Do not invent non-existent helper methods on unrelated services.
- MOCK INJECTION IN TESTS: When a class adds a new @RequiredArgsConstructor final field (e.g. ContractMemberRepository), all corresponding unit tests must declare and mock that dependency (e.g. @Mock private ContractMemberRepository contractMemberRepository) and pass it to the constructor if explicit constructor instantiation is used.
- CALL SIGNATURE ACCURACY: Check method parameter types and counts before invoking existing service methods (e.g. memberService.deleteMember requires (UUID id, UUID projectId)). Never drop required parameters.
- UNIT TEST FIXTURES FOR MEMBER UPDATES: In ParticipatingMemberService unit tests, ParticipatingMember test objects passed to updateMember(...) MUST have all required validation fields populated: id, userId, email (e.g. "test@test.com" or member.getEmail()), roleName, and company (with company.getId() non-null). When role is updated to 'Member' and contract count > 0, updateMember proceeds to validateParticipatingMemberForEdit which will throw RequiredFieldsMissingException if email or company is missing. Always ensure email and company are set on the test ParticipatingMember instance before passing to updateMember.
"""

    def _csharp_guidelines(self) -> str:
        """C# specific guidelines"""
        return """
C# GUIDELINES:
- Use PascalCase for public members, camelCase for private
- Add XML documentation (///) for public members
- Use async/await for I/O operations
- Implement IDisposable where needed
- Use nullable reference types (?)
- Follow Microsoft coding conventions
- Use LINQ where appropriate
- Prefer expression body for simple methods
"""
    
    def _typescript_guidelines(self) -> str:
        """TypeScript specific guidelines"""
        return """
TYPESCRIPT GUIDELINES:
- Use strict type checking
- Define interfaces for all data structures
- Use async/await for promises
- Use arrow functions for callbacks
- Follow React hooks rules (if applicable)
- Use TypeScript generics where appropriate
- Export types and interfaces
- Use const for immutable values

CROSS-FILE DATAFLOW & VIEWMODEL INTEGRITY (CRITICAL):
- Evidence Before Fallback: Before referencing properties from an API or service response (e.g. member.company?.name), verify what fields the service method actually requests/returns. Do NOT invent or assume optional interface properties exist without proof.
- Full ViewModel Assignment: When adding items to collections consumed by templates (e.g. displayedMembers.push(displayedMember)), you MUST explicitly assign all properties that the template binds to (e.g. displayedMember.organization = resolvedOrg).
- Never leave a computed value stranded on a temporary variable (e.g. user.existingOrganizationName) while leaving the staged collection item's property empty or unassigned.

AVOID COMMON COMPILE ERRORS (CRITICAL):
- TS2353 (Object literal excess properties): When adding custom UI state flags (e.g. isExistingProjectMember) to imported/shared model types (e.g. DisplayedMember), do NOT inject extra properties directly into object literals typed as the base model.
  Instead:
  1) Extend the type locally in the component file:
     `interface LocalDisplayedMember extends DisplayedMember { isExistingProjectMember?: boolean; }`
     or use an intersection type: `DisplayedMember & { isExistingProjectMember?: boolean }`
  2) Or assign the UI flag after creation: `(member as any).isExistingProjectMember = true;` or type the collection with the extended interface.
  3) Never attempt to modify shared base model files if they are not part of the ticket's edited files.
- TS2339 / TS2322 (Implicit never[] or type narrowing to never):
  1) Always explicitly type empty array initializations (e.g. `const ids: string[] = [];` NOT `const ids = [];`), otherwise TypeScript infers `never[]` in strict mode causing TS2339 when calling methods like `.indexOf()` or `.push()`.
  2) Avoid impossible type guards that narrow variables to `never`.
"""
    
    def _python_guidelines(self) -> str:
        """Python specific guidelines"""
        return """
PYTHON GUIDELINES:
- Follow PEP 8 style guide
- Use type hints for function signatures
- Use docstrings (Google or NumPy style)
- Use async/await for I/O operations
- Use dataclasses or Pydantic models where appropriate
- Handle exceptions specifically (avoid bare except)
- Use context managers (with statement)
"""

    def _html_guidelines(self) -> str:
        """HTML / Angular / Vue / JSX template guidelines"""
        return """
HTML / TEMPLATE GUIDELINES:
- This is a markup template, not a script. Preserve existing indentation and tag structure exactly.
- Respect the framework already in use (detect from surrounding syntax): Angular ([disabled], *ngIf, {{ 'key' | translate }}), Vue (:disabled, v-if), or JSX (disabled={...}).
- UI COMPONENT PRESERVATION MANDATE (CRITICAL):
  Existing UI component tags (e.g. <ot-dropdown>, <ot-item-select>, <select>) are IMMUTABLE unless the plan explicitly authorizes a component migration.
  Never replace an existing <ot-dropdown> with <ot-item-select> or vice versa. Preserve the exact component type and its existing binding syntax ([options], [value], (change)).
- To disable/lock a control or show static text:
  If a ticket requires static text for existing entities, render a static text span/div conditioned with *ngIf="condition", and render the existing control with *ngIf="!condition".
- For localized text, reference the existing i18n key via the framework's translate mechanism; do not hardcode user-facing strings.
- Keep attribute ordering and quoting style consistent with the surrounding markup.
- Never convert template syntax into script; do not add <script> blocks.

ANGULAR PROPERTY NAME RULE (CRITICAL — causes TS2339 build failures if violated):
Before writing ANY new Angular binding (*ngIf, {{ }}, [attr]), you MUST look at the
SESSION CONTEXT section below (the current content of the sibling .ts controller file).
Extract the EXACT class property names declared there and use ONLY those names.

DO NOT invent new property names from the ticket description.
INSTEAD: read what the .ts controller already declares (or what the TS task will add based
on its description), then use those EXACT property names in the template.

Example:
  TS declares: `isExistingMemberInProject: boolean = false;`
  HTML MUST use: `*ngIf="dmember.isExistingMemberInProject"` — NOT `*ngIf="isUserProjectMember"`

If the SESSION CONTEXT does not yet show the .ts, look at the TS task description in the
PLANNER DIRECTIVE — it describes what properties the controller will add. Use those names exactly.

NEVER use a condition like *ngIf="obj.organization" to represent "existing org" when
'organization' is ALWAYS set by the controller. Use the dedicated flag property instead.
"""

    def _scss_guidelines(self) -> str:
        """SCSS / CSS / LESS stylesheet guidelines"""
        return """
SCSS / CSS GUIDELINES:
- Modify only the selectors/rules required. Preserve nesting, variables, mixins, and ordering.
- Reuse existing variables and mixins instead of hardcoding values already defined.
- Prefer flexible layout properties over fixed pixel heights when addressing layout/alignment.
- Do not restructure unrelated rules or reformat the whole file.
"""

    def _json_guidelines(self) -> str:
        """JSON / i18n / config guidelines"""
        return """
JSON GUIDELINES:
- Output MUST remain strictly valid JSON (double-quoted keys/strings, no comments, no trailing commas).
- Preserve existing key ordering and indentation; insert new keys in the correct nested location.
- For i18n bundles, add the key in the same nesting convention as sibling keys.
- Do not change value types (string/number/bool) unless the task explicitly requires it.
"""

    def _xml_guidelines(self) -> str:
        """XML / POM / config guidelines"""
        return """
XML GUIDELINES:
- Output MUST remain well-formed XML with correctly closed and nested tags.
- Preserve namespaces, attribute ordering, and indentation of the surrounding elements.
- Insert new elements in the schema-appropriate location; do not reorder unrelated nodes.
"""

    def _yaml_guidelines(self) -> str:
        """YAML / manifest guidelines"""
        return """
YAML GUIDELINES:
- Indentation is significant: use spaces (never tabs) and match the existing indent width exactly.
- Preserve key ordering and existing anchors/aliases; insert new keys under the correct parent.
- Quote values only where the surrounding file quotes them; keep list/map style consistent.
- Output MUST parse as valid YAML.
"""

    def _properties_guidelines(self) -> str:
        """.properties / .env key=value guidelines"""
        return """
PROPERTIES / ENV GUIDELINES:
- Each line is a single key=value (or key: value) pair; preserve the existing separator style.
- Do not quote values unless the surrounding file already quotes them.
- Add or update only the specified keys; keep unrelated keys and comments untouched.
"""
    
    def _build_user_prompt(
        self,
        task: DevelopmentTask,
        requirements: StructuredRequirements,
        context: List[CodeChunk],
        existing_content: str = None,
        allowed_files: List[str] = None,
        readonly_files: List[str] = None,
    ) -> str:
        """Build user prompt with task, requirements, and context"""
        
        context_str = self._format_context(context)
        
        # ── REUSE-FIRST directive (set by plan_node's capability discovery) ──
        # Tells the generator which existing service methods to CALL instead of
        # inventing new ones or extending the service under the ticket.
        # The directive is set on the code_generator MODULE (not per-instance)
        # to avoid race conditions during concurrent ticket runs.
        # We import sys here to ensure we get the module-level attribute correctly.
        import sys as _sys
        _reuse_directive_module = getattr(_sys.modules[__name__], "_reuse_directive", "")
        if not _reuse_directive_module:
            # fallback for legacy/debug scenarios (module attribute missing)
            _reuse_directive_module = getattr(self, "_reuse_directive", "")
        _reuse_block = f"{_reuse_directive_module}\n" if _reuse_directive_module else ""

        change_type = task.task_type.value  # "modify" or "create"

        # ── FROZEN SCOPE BLOCK ───────────────────────────────────────────────
        # Preserve planner-specified allowed_methods (including new methods to be added)
        if existing_content and task.allowed_methods:
            validated = [m for m in task.allowed_methods
                         if re.search(r'\b' + re.escape(m) + r'\b', existing_content)]
            dropped = set(task.allowed_methods) - set(validated)
            if dropped:
                import logging as _lg
                _lg.getLogger(__name__).info(
                    "allowed_methods: planned new methods %s will be added to %s", sorted(dropped), task.file_path
                )

        allowed_methods_str = (
            ", ".join(task.allowed_methods) if task.allowed_methods else "NOT SPECIFIED — keep change minimal"
        )
        target_method_str = task.target_method or "NOT SPECIFIED — use keyword context to find it"
        target_class_str  = task.target_class  or "NOT SPECIFIED"
        new_file_str      = "ALLOWED (this is a create task)" if task.new_file_creation_allowed else "FORBIDDEN"

        scope_block = f"""
╔══════════════════════════════════════════════════════════════╗
║  FROZEN MODIFICATION SCOPE — DO NOT DEVIATE                  ║
╠══════════════════════════════════════════════════════════════╣
║  ALLOWED FILE    : {task.file_path:<43}║
║  OPERATION       : {change_type.upper():<43}║
║  TARGET CLASS    : {target_class_str:<43}║
║  TARGET METHOD   : {target_method_str:<43}║
║  ALLOWED METHODS : {allowed_methods_str:<43}║
║  NEW FILE CREATE : {new_file_str:<43}║
╚══════════════════════════════════════════════════════════════╝

ENFORCEMENT:
- If ALLOWED METHODS is listed: ONLY modify those methods. Touch NOTHING else.
- If TARGET METHOD is listed: go directly to that method. Do not touch neighbors.
- NEW FILE CREATE = FORBIDDEN means: do NOT create any new file anywhere.
- If you find yourself writing code outside the listed method(s), STOP."""

        # ── Phase 2: Inject edit_anchors into prompt ──────────────────────────
        _edit_anchor_block = ""
        _anchors = getattr(task, "edit_anchors", []) or []
        if _anchors:
            _anchor_lines = []
            for _a in _anchors:
                _action = _a.get("action", "?")
                _method = _a.get("method_name", _a.get("anchor_method", "?"))
                _placement = _a.get("placement", "?")
                _anchor = _a.get("anchor_method", "")
                _desc = _a.get("description", "")
                _anchor_lines.append(
                    f"  • {_action}: {_method} — placement={_placement}"
                    + (f", scope={_anchor}" if _anchor else "")
                    + (f" — {_desc}" if _desc else "")
                )
            _edit_anchor_block = (
                "\n\n🎯 EDIT PLACEMENT ANCHORS (from planner — tells you WHERE to place code):\n"
                + "\n".join(_anchor_lines)
                + "\n\n⚠️  Follow these anchors exactly. If the planner says 'sibling_after findById', "
                "add the method AFTER findById's closing brace, NOT inside findById's body."
            )

        # ── TICKET-WIDE SCOPE LOCK ────────────────────────────────────────────
        scope_lock_block = ""
        if allowed_files or readonly_files:
            writable_lines = "\n".join(f"  ✅  {p}" for p in (allowed_files or []))
            ro_lines       = "\n".join(f"  🚫  {p}" for p in (readonly_files  or []))
            scope_lock_block = f"""

╔══════════════════════════════════════════════════════════════════╗
║  TICKET-WIDE FILE SCOPE LOCK                                     ║
╠══════════════════════════════════════════════════════════════════╣
║  YOU MAY MODIFY ONLY (writable):                                 ║
{writable_lines}
║                                                                  ║
║  READ-ONLY (for reference only — DO NOT WRITE):                  ║
{ro_lines}
╠══════════════════════════════════════════════════════════════════╣
║  ANY modification to a file not listed above = SCOPE VIOLATION   ║
╚══════════════════════════════════════════════════════════════════╝"""

        # ── Change Surface Targeting ─────────────────────────────────────────
        # If the task carries ownership_analysis from the Ownership Analysis Agent,
        # surface the exact symbols to change so the LLM targets them precisely.
        change_surface_block = ""
        oa = getattr(task, "ownership_analysis", None) or {}
        if isinstance(oa, dict) and oa.get("change_surface"):
            surfaces = oa["change_surface"]
            surface_lines = "\n".join(
                f"  - symbol={cs.get('symbol','?')}  type={cs.get('type','?')}  snippet: {cs.get('snippet','')[:120]}"
                for cs in surfaces
            )
            change_surface_block = f"""

CHANGE SURFACE (from Ownership Analysis Agent — TARGET THESE SYMBOLS PRECISELY):
{surface_lines}

INSTRUCTION: Go directly to the symbol(s) listed above. Apply the minimal change there.
Do NOT scan the entire file for other locations to change.
"""

        existing_section = ""
        modify_instruction = f"Now generate the new file content for {task.file_path}. Only create this one file."
        if existing_content:
            method_excerpt = self._extract_method_context(
                existing_content, task.target_method, task.allowed_methods
            )
            
            # ── Skeleton + Verbatim Architecture (AI IDE technique) ─────────────
            # Two distinct content roles that must NEVER be mixed:
            #
            #   SKELETON (smart_extract): structural outline for LLM awareness.
            #     The LLM sees method signatures, class structure, line counts.
            #     This content is explicitly marked "DO NOT copy from this section"
            #     because it contains markers (// ... omitted) that don't exist in
            #     the original file.
            #
            #   VERBATIM (extract_exact_methods): exact uncompressed source bytes.
            #     The LLM's SEARCH blocks must be substrings of THIS content.
            #     Uses tree-sitter for grammar-correct method boundaries.
            #     Guaranteed to match _apply_str_replace_edits() because it IS
            #     the original file content, zero transformations.
            #
            # This mirrors Aider's architecture:
            #   - repo map (tree-sitter) = structural awareness, never edit source
            #   - edit format (SEARCH/REPLACE) = operates on verbatim file content

            display_content = existing_content
            # Target-First Generation: For large files (> 30,000 chars, e.g. 2k-20k+ lines),
            # never send the full 500KB-1MB file into LLM prompt (which causes attention degradation
            # or context overflow). Instead, use the 3-phase AI-IDE architecture:
            #   1. Structural Outline (Tree-Sitter signatures + line numbers) for awareness.
            #   2. Verbatim Target Window / Exact Methods for surgical SEARCH/REPLACE editing.
            #   3. Apply edits in-place to the full file on disk via _apply_str_replace_edits.
            anchor_methods = _get_anchor_methods(task)
            _task_ext = Path(task.file_path).suffix.lower()
            _method_exts = {".ts", ".tsx", ".js", ".jsx", ".java", ".kt", ".py", ".cs", ".go", ".rs", ".cpp", ".c"}
            should_target_extract = (
                task.task_type.value == "modify"
                and len(existing_content) > 30000
                and _task_ext in _method_exts
            )
            if should_target_extract:
                try:
                    from ticket_to_code.agents.smart_extract import (
                        smart_extract, extract_exact_methods, _get_reliable_boundaries
                    )

                    # Automatic Anchor Recovery: If planner did not populate explicit anchor_methods,
                    # recover them dynamically from task description, title, and AST method boundaries.
                    if not anchor_methods:
                        desc_words = set(re.findall(r'[a-zA-Z]\w{3,}', f"{task.description or ''} {task.title or ''}".lower()))
                        desc_words -= {"this", "that", "with", "from", "have", "been", "should", "update", "modify", "implement", "check", "need", "test", "file", "method"}
                        try:
                            _lines_temp = existing_content.split('\n')
                            _boundaries = _get_reliable_boundaries(existing_content, task.file_path, _lines_temp)
                            for mb in _boundaries:
                                if mb.kind != "class" and (mb.name.lower() in desc_words or any(w in mb.name.lower() for w in desc_words)):
                                    anchor_methods.append(mb.name)
                                    if len(anchor_methods) >= 4:
                                        break
                        except Exception as _b_err:
                            logger.debug(f"Anchor recovery boundary scan skipped: {_b_err}")

                    # Section 1: SKELETON — structural outline of entire 20k+ lines (awareness only)
                    skeleton_content = smart_extract(
                        content=existing_content,
                        file_path=task.file_path,
                        allowed_methods=anchor_methods or task.allowed_methods,
                        target_method=task.target_method,
                        edit_description=task.description,
                    )

                    # Section 2: VERBATIM — exact method bodies (edit source)
                    verbatim_content, matched_boundaries = "", []
                    if anchor_methods:
                        verbatim_content, matched_boundaries = extract_exact_methods(
                            content=existing_content,
                            file_path=task.file_path,
                            anchor_methods=anchor_methods,
                        )

                    if matched_boundaries and verbatim_content:
                        # Two-section prompt: skeleton for awareness, verbatim for editing
                        display_content = (
                            f"=== FILE STRUCTURE (context only — DO NOT copy text from this section) ===\n"
                            f"{skeleton_content}\n\n"
                            f"=== EXACT SOURCE (your SEARCH block MUST be a verbatim substring of THIS section) ===\n"
                            f"{verbatim_content}"
                        )
                        logger.info(
                            f"Target-First Skeleton+Verbatim: {task.file_path} "
                            f"({len(existing_content):,} → skeleton={len(skeleton_content):,} + "
                            f"verbatim={len(verbatim_content):,} chars, "
                            f"{len(matched_boundaries)} methods matched)"
                        )
                    else:
                        # Fallback Locality Window: If method boundaries weren't matched directly,
                        # find the highest-density match of keywords from the task description
                        # and extract a 200-line verbatim window around that location.
                        _kw = set(re.findall(r'[a-zA-Z]\w{3,}', f"{task.description or ''} {task.title or ''}".lower()))
                        _kw -= {"this", "that", "with", "from", "have", "been", "should", "update", "modify", "file"}
                        _lines = existing_content.split('\n')
                        _best_line = 0
                        _max_score = 0
                        for _idx, _l in enumerate(_lines):
                            _score = sum(1 for w in _kw if w in _l.lower())
                            if _score > _max_score:
                                _max_score = _score
                                _best_line = _idx

                        _win_start = max(0, _best_line - 50)
                        _win_end = min(len(_lines), _best_line + 150)
                        _window_content = '\n'.join(_lines[_win_start:_win_end])

                        display_content = (
                            f"=== FILE STRUCTURE (context only — DO NOT copy text from this section) ===\n"
                            f"{skeleton_content}\n\n"
                            f"=== EXACT SOURCE (lines {_win_start+1}-{_win_end} — SEARCH block must match this verbatim) ===\n"
                            f"{_window_content}"
                        )
                        logger.info(
                            f"Target-First Sliding Window for {task.file_path}: "
                            f"lines {_win_start+1}-{_win_end} ({len(_window_content):,} chars) around line {_best_line+1}"
                        )
                except Exception as e:
                    logger.warning(f"Target-First extraction failed for {task.file_path}, using safe budget: {e}")
                    # Hard safety budget: never send > 30k chars blindly
                    if len(existing_content) > 30000:
                        display_content = existing_content[:30000]

            # ── Unified MODIFY path: show existing file for str_replace edits ──
            existing_section = f"""

EXISTING FILE CONTENT (this is what the SEARCH block must match against):
```
{display_content}
```

INSTRUCTION: Use <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks (or TARGET: / REPLACEMENT: blocks) to make surgical changes.
Each SEARCH block must be an EXACT, UNIQUE substring of the EXACT SOURCE section above.
Write ONLY the specific part of code you want to replace (3 to 8 lines). Do NOT rewrite large blocks or full functions.
Directly replace deprecated or modified code with the updated implementation in the REPLACE block. Do not leave commented-out dead code.
If you see a "FILE STRUCTURE" section, that is context ONLY — do NOT copy text from it.
For adding NEW code: use a small existing anchor as SEARCH, and include anchor + new code as REPLACE.
For modifying existing code: copy the exact lines to change as SEARCH, and provide the modified version as REPLACE.
Do NOT output the full file. Only output SEARCH/REPLACE blocks."""
            modify_instruction = f"Return minimal targeted SEARCH/REPLACE or TARGET/REPLACEMENT blocks targeting {task.file_path}."

        # ── Layer 1: Import Context (Cursor / Aider / Devin technique) ─────────
        import_context_block = self._build_import_context(
            existing_content, task, context
        )
        # Incremental cross-file awareness: inject fresh content of files this
        # task imports that were regenerated earlier in the same run.
        session_context_block = self._build_session_context(existing_content, task)

        # ── Cross-file contract (planner produces/consumes) ──────────────────
        # The planner declares what each task produces and consumes.
        # This was previously IGNORED — now we surface it so the LLM knows
        # the exact method names/signatures it must create or call.
        contract_context_block = self._build_contract_context(task)
        if contract_context_block:
            logger.info(
                f"  [Contract] Injected cross-file contract for {task.id} "
                f"({len(contract_context_block)} chars)"
            )

        # ── Cross-File Intelligence (ContextAssembler pipeline) ─────────────────
        # The ContextAssembler is the SINGLE entry point for cross-file context.
        # It runs an 8-stage pipeline: task metadata → blueprints → handoffs →
        # rolling context → dependencies → repo evidence → validation → assembly.
        # Each context block is labeled with its authority level.
        # Falls back to DependencyContextBuilder if ContextAssembler fails.
        #
        # Additionally, ImplementationState.to_prompt_context() provides
        # authoritative handoff data (verified exports from earlier tasks).
        # This was previously built but never called — now we wire it in.
        blueprint_context_block = ""
        _handoff_context_block = ""
        _impl_state_pre = getattr(self, "_impl_state", None)
        if _impl_state_pre is not None:
            try:
                _handoff_ctx = _impl_state_pre.to_prompt_context(task)
                if _handoff_ctx:
                    _handoff_context_block = _handoff_ctx
                    logger.info(
                        f"  [ImplementationState] Injected {len(_handoff_ctx)} chars "
                        f"cross-file context for {task.id} "
                        f"(handoffs={len(_impl_state_pre.get_handoffs_for_dependencies(task.dependencies or []))})"
                    )
            except Exception as _hc_exc:
                logger.debug(f"  [ImplementationState] to_prompt_context failed: {_hc_exc}")
        _impl_state = getattr(self, "_impl_state", None)
        if _impl_state is not None:
            try:
                from ticket_to_code.agents.context_assembler import ContextAssembler
                from ticket_to_code.agents.dependency_context_builder import DependencyContextBuilder
                _resolver = getattr(self, "_symbol_resolver", None)
                _ws = getattr(self, "_workspace_path", None)
                dep_builder = DependencyContextBuilder(
                    symbol_resolver=_resolver,
                    workspace_path=Path(_ws) if _ws else None,
                )
                assembler = ContextAssembler(
                    impl_state=_impl_state,
                    dep_builder=dep_builder,
                    symbol_resolver=_resolver,
                    workspace_path=Path(_ws) if _ws else None,
                )
                blueprint_context_block = assembler.assemble(task)
                if blueprint_context_block:
                    logger.info(
                        f"  [ContextAssembler] Injected {len(blueprint_context_block)} chars "
                        f"for {task.id}"
                    )
                    # Log conflicts for debugging
                    if assembler.conflicts:
                        logger.info(
                            f"  [ContextAssembler] Resolved {len(assembler.conflicts)} "
                            f"evidence conflict(s)"
                        )
            except Exception as _asm_exc:
                logger.debug(f"  [ContextAssembler] Failed, falling back: {_asm_exc}")
                # Fallback: use DependencyContextBuilder directly
                try:
                    from ticket_to_code.agents.dependency_context_builder import DependencyContextBuilder
                    _resolver = getattr(self, "_symbol_resolver", None)
                    _ws = getattr(self, "_workspace_path", None)
                    dep_builder = DependencyContextBuilder(
                        symbol_resolver=_resolver,
                        workspace_path=Path(_ws) if _ws else None,
                    )
                    blueprint_context_block = dep_builder.build_context(task, _impl_state)
                    if blueprint_context_block:
                        logger.info(
                            f"  [Blueprint] Fallback dependency context for {task.id} "
                            f"({len(blueprint_context_block)} chars)"
                        )
                except Exception as _bp_exc:
                    logger.debug(f"  [Blueprint] Fallback also failed: {_bp_exc}")

        # ── Fix 1: Constructor preservation guard ─────────────────────────────
        # When modifying a TypeScript/Java file, extract ALL existing constructor
        # params and inject them with "PRESERVE ALL" to prevent the LLM from
        # dropping injections it didn't explicitly add. This is the root cause of
        # "contractService missing from constructor" class of errors.
        _constructor_guard = ""
        _ext_for_ctor = Path(task.file_path).suffix.lower()
        if (existing_content and task.task_type.value == "modify"
                and _ext_for_ctor in (".ts", ".tsx", ".java")):
            _ctor_params = self._extract_constructor_params(existing_content, _ext_for_ctor)
            if _ctor_params:
                _constructor_guard = (
                    f"\n\n⚠️  CONSTRUCTOR PRESERVATION RULE:\n"
                    f"The existing constructor has these parameters — PRESERVE ALL OF THEM.\n"
                    f"Only ADD new parameters; never remove any existing injection:\n"
                    f"```\n{_ctor_params}\n```\n"
                )

        # ── PLANNER DIRECTIVE: surface the exact planner task so LLM cannot drift ──
        planner_directive = f"""
╔══════════════════════════════════════════════════════════════════╗
║  PLANNER DIRECTIVE — THIS IS YOUR CONTRACT                       ║
╠══════════════════════════════════════════════════════════════════╣
║  The planning agent assigned you the following task:              ║
║                                                                  ║
║  TASK: {task.title:<55}║
║  FILE: {task.file_path:<55}║
║  TYPE: {change_type.upper():<55}║
║                                                                  ║
║  DESCRIPTION:                                                    ║
║  {task.description[:120]:<63}║
╠══════════════════════════════════════════════════════════════════╣
║  You MUST implement EXACTLY this task.                            ║
║  Do NOT substitute a different file or a different change.        ║
║  If the task says CREATE a file, you CREATE that file.            ║
║  If the task says MODIFY a method, you MODIFY that method.        ║
╚══════════════════════════════════════════════════════════════════╝"""

        _prompt = f"""\
{scope_block}{scope_lock_block}{_edit_anchor_block}
{planner_directive}
{_reuse_block}

TASK ID: {task.id}
TASK: {task.title}
DESCRIPTION: {task.description}

FILE PATH: {task.file_path}
CHANGE TYPE: {change_type.upper()}
LANGUAGE: {task.language.value}

FUNCTIONAL REQUIREMENTS:
{self._format_list(requirements.functional_requirements)}

TECHNICAL REQUIREMENTS:
{self._format_list(requirements.technical_requirements)}

EDGE CASES TO HANDLE:
{self._format_list(requirements.edge_cases)}
{change_surface_block}{import_context_block}{session_context_block}{contract_context_block}{_handoff_context_block}{blueprint_context_block}{_constructor_guard}
AVAILABLE CODE CONTEXT (reference patterns only — do NOT copy wholesale):
{context_str}{existing_section}

DEPENDENCIES (from other tasks):
{self._format_list(task.dependencies) if task.dependencies else "None"}

{modify_instruction}""".strip()

        # ── Token accounting telemetry ────────────────────────────────────────
        # Store last prompt component sizes for per-file token breakdown.
        # These are observational only — never affect generation logic.
        self._last_session_context = session_context_block or ""
        self._last_contract_context = contract_context_block or ""
        self._last_impl_context = (
            (_handoff_context_block or "") + (blueprint_context_block or "")
        )

        return _prompt



    
    def _extract_method_context(
        self,
        file_content: str,
        target_method: str = None,
        allowed_methods: list = None,
        context_lines: int = 5
    ) -> str:
        """
        Extract ONLY the target method + context_lines above/below from the file.
        
        This is the MINIMAL CONTEXT principle: don't send the whole file as "context"
        when we know exactly which method to change.  Sending less context forces the
        LLM to make a targeted surgical change rather than a broad rewrite.
        
        Returns empty string if target_method is unknown (caller will use full file).
        """
        import re
        methods_to_find = []
        if target_method:
            methods_to_find.append(target_method)
        if allowed_methods:
            methods_to_find += [m for m in allowed_methods if m not in methods_to_find]

        if not methods_to_find:
            return ""

        lines = file_content.splitlines()
        excerpts = []

        for method_name in methods_to_find[:3]:  # max 3 methods
            # Find the line where this method is defined
            pattern = re.compile(
                rf"(?:def|function|async function|public|private|protected|static)?\s*"
                rf"\b{re.escape(method_name)}\s*\("
            )
            for i, line in enumerate(lines):
                if pattern.search(line):
                    start = max(0, i - context_lines)
                    # Find end: scan forward for closing brace or next method
                    depth = 0
                    end = i
                    for j in range(i, min(len(lines), i + 60)):
                        depth += lines[j].count("{") - lines[j].count("}")
                        if j > i and depth <= 0:
                            end = j + 1
                            break
                    else:
                        end = min(len(lines), i + 30)

                    excerpt_lines = lines[start:end + context_lines]
                    excerpt = "\n".join(excerpt_lines)
                    excerpts.append(f"[Method: {method_name}]\n{excerpt}")
                    break

        return "\n\n".join(excerpts) if excerpts else ""

    def _find_method_line_ranges(
        self,
        file_content: str,
        target_method: str = None,
        allowed_methods: list = None,
    ) -> dict:
        """Compute exact 1-indexed line ranges for target methods.

        Returns dict of {method_name: (start_line, end_line)} so the user prompt
        can proactively tell the LLM 'use REPLACE_LINES: 243-285' instead of
        making it scan the whole file.  This is the key difference between our
        system and Claude Code — we now give the LLM the exact range up-front.
        """
        import re
        methods_to_find = []
        if target_method:
            methods_to_find.append(target_method)
        if allowed_methods:
            methods_to_find += [m for m in allowed_methods if m not in methods_to_find]
        if not methods_to_find:
            return {}

        lines = file_content.splitlines()
        ranges: dict = {}

        for method_name in methods_to_find[:5]:
            pattern = re.compile(
                rf"(?:def|function|async function|public|private|protected|static)?\s*"
                rf"\b{re.escape(method_name)}\s*\("
            )
            for i, line in enumerate(lines):
                if pattern.search(line):
                    depth = 0
                    end = i
                    for j in range(i, min(len(lines), i + 80)):
                        depth += lines[j].count("{") - lines[j].count("}")
                        if j > i and depth <= 0:
                            end = j
                            break
                    else:
                        end = min(len(lines) - 1, i + 30)
                    # 1-indexed, inclusive
                    ranges[method_name] = (i + 1, end + 1)
                    break

        return ranges

    # ── Layer 1: Cursor / Aider / Devin — Import Context Builder ─────────────

    # Attach a sqlite_store after construction so generate_code can query symbols.
    _sqlite_store = None
    # Attach the session-generated file map (path→content) for incremental context.
    _session_files = None
    # Attach the ImplementationState for blueprint-driven context + post-gen recording.
    # Set by workflow.py before generate_code(); None when running without the new system.
    _impl_state = None

    def _build_import_context(
        self,
        existing_content: str,
        task: "DevelopmentTask",
        context: list,
    ) -> str:
        """
        Build an IMPORT CONTEXT block injected into the LLM prompt.

        Combines three techniques:
          1. Cursor / Copilot Workspace — extract the current file's import block so
             the LLM can see existing imports and extend (not replace) them.
          2. Aider repo-map — surface where new symbols live by querying the SQLite
             symbol index for classes referenced in the task description and context.
          3. Devin two-pass — explicitly tell the LLM to emit new imports as a
             dedicated SEARCH/REPLACE block targeting the import section.
        """
        if not existing_content and not self._sqlite_store:
            return ""

        lines: list[str] = []

        # ── Part A: Existing import block (Cursor / Copilot Workspace style) ──
        if existing_content:
            import_lines: list[str] = []
            for line in existing_content.split("\n"):
                s = line.strip()
                if s.startswith(("import ", "from ", "using ", "package ", "require(")):
                    import_lines.append(line)
                elif s.startswith(("@", "//", "/*", " *", "#")):
                    # allow annotations and comments inside import block
                    if import_lines:
                        import_lines.append(line)
                elif not s:
                    # blank line — keep if we're inside the import block
                    if import_lines:
                        import_lines.append(line)
                elif s:
                    break  # first non-import, non-comment, non-blank content line
            if import_lines:
                block = "\n".join(import_lines).strip()
                lines.append(
                    f"\nEXISTING IMPORTS — PRESERVE EVERY LINE BELOW; only ADD new ones:\n```\n{block}\n```"
                )

        # ── Part B: Repo symbol map (Aider repo-map / Devin symbol index) ────
        if self._sqlite_store:
            # Collect candidate symbol names from: task description + context filenames
            candidate_symbols: list[str] = []
            desc_text = f"{task.title} {task.description}"
            # PascalCase words are likely class/interface names
            import re
            candidate_symbols += re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", desc_text)
            for chunk in context:
                fp = chunk.get("file_path", "") if isinstance(chunk, dict) else getattr(chunk, "file_path", "")
                # e.g.  "ContractMemberModel.java" → "ContractMemberModel"
                stem = Path(fp).stem
                if stem and stem[0].isupper():
                    candidate_symbols.append(stem)
            # Deduplicate, skip very generic names
            skip = {"String", "List", "Map", "Set", "Optional", "Boolean", "Integer",
                    "Object", "Array", "Component", "Service", "Module", "Type"}
            seen: set[str] = set()
            unique_symbols = [s for s in candidate_symbols
                              if s not in skip and s not in seen and not seen.add(s)]  # type: ignore[func-returns-value]

            symbol_map_lines: list[str] = []
            for sym in unique_symbols[:20]:
                try:
                    rows = self._sqlite_store.search_symbols(
                        sym, kinds=["class", "interface", "enum", "type"], limit=3
                    )
                    for row in rows:
                        row_path = row["path"] if isinstance(row, dict) else getattr(row, "path", "")
                        row_name = row["name"] if isinstance(row, dict) else getattr(row, "name", sym)
                        if row_path:
                            import_hint = _build_import_hint(
                                row_name, row_path,
                                source_file_path=task.file_path,
                                workspace_path=str(getattr(self, "_workspace_path", "")),
                            )
                            if import_hint:
                                symbol_map_lines.append(f"  {row_name} → {import_hint}  (source: {row_path})")
                            break
                except Exception:
                    pass

            if symbol_map_lines:
                lines.append(
                    "\nREPO SYMBOL MAP — use these exact import paths for any new symbol references:\n"
                    + "\n".join(symbol_map_lines)
                )

        if not lines:
            return ""

        return (
            "\n"
            + "\n".join(lines)
            + "\n\nIMPORT RULES:\n"
            "  1. NEVER remove any existing import line shown above.\n"
            "  2. When you add a new symbol reference, add its import as a SEPARATE SEARCH/REPLACE\n"
            "     block that targets the import section (the first import line shown above).\n"
            "  3. Use the REPO SYMBOL MAP paths verbatim — do NOT invent package paths.\n"
        )

    # ── Constructor preservation guard (Fix 2 root cause) ──────────────────
    @staticmethod
    def _extract_constructor_params(content: str, ext: str) -> str:
        """Extract existing constructor parameter lines so LLM cannot drop them.

        Returns the full constructor block (signature + body) so the LLM sees
        the complete context and cannot regenerate the constructor from memory
        with missing injections.
        """
        if ext not in (".ts", ".tsx", ".java"):
            return ""
        # Try to extract the full constructor block including body
        m = re.search(
            r'(constructor\s*\([^)]*\)[^{]*\{)',
            content, re.DOTALL
        )
        if not m:
            return ""
        # Find the closing brace of the constructor
        start = m.start()
        depth = 0
        end = m.end()
        for i in range(m.end() - 1, len(content)):
            if content[i] == '{':
                depth += 1
            elif content[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        ctor_block = content[start:end].strip()
        if not ctor_block:
            return ""
        return ctor_block

    def _build_contract_context(
        self,
        task: "DevelopmentTask",
    ) -> str:
        """Build a prompt block from cross-file contracts.

        Priority:
        1. VERIFIED facts from RelationshipRegistry (actual code evidence)
        2. Planner's semantic contract (intent-level, may not match reality)

        VERIFIED facts override planner predictions. PLANNED relationships
        are clearly labeled as suggestions, never as facts.
        """
        # ── 1. Try RelationshipRegistry for verified facts ──
        registry_context = ""
        _impl_state = getattr(self, "_impl_state", None)
        if _impl_state and _impl_state.relationship_registry:
            try:
                registry = _impl_state.relationship_registry
                registry_context = registry.get_context_for_task(task)
            except Exception as _reg_exc:
                logger.debug(f"  Registry context failed (non-fatal): {_reg_exc}")

        # ── 2. Planner's semantic contract (existing logic, preserved) ──
        contract = getattr(task, "cross_file_contract", None)
        planner_context = ""
        if contract:
            blocks: list[str] = []

            # ── PRODUCES: what this task MUST create ──
            if contract.produces:
                produce_lines = []
                for bp in contract.produces:
                    line = f"    • {bp.capability}"
                    if bp.data_shape:
                        line += f"\n      data shape: {bp.data_shape}"
                    if bp.relationship_type and bp.relationship_type != "data":
                        line += f"  ({bp.relationship_type})"
                    produce_lines.append(line)
                blocks.append(
                    "  🔧 THIS TASK MUST PRODUCE (create these capabilities):\n"
                    + "\n".join(produce_lines)
                )

            # ── CONSUMES: what this task can USE from earlier tasks ──
            if contract.consumes:
                consume_lines = []
                for bp in contract.consumes:
                    from_tag = f"[from {bp.from_task}] " if bp.from_task else ""
                    line = f"    • {from_tag}{bp.capability}"
                    if bp.data_shape:
                        line += f"\n      data shape: {bp.data_shape}"
                    if bp.relationship_type and bp.relationship_type != "data":
                        line += f"  ({bp.relationship_type})"
                    consume_lines.append(line)
                blocks.append(
                    "  📥 THIS TASK CONSUMES (use these — already created by earlier tasks):\n"
                    + "\n".join(consume_lines)
                )

            if blocks:
                planner_context = (
                    "\n\n╔══════════════════════════════════════════════════════════════╗\n"
                    "║  CROSS-FILE CONTRACT (from planner — semantic obligations)   ║\n"
                    "╠══════════════════════════════════════════════════════════════╣\n"
                    + "\n".join(blocks)
                    + "\n╠══════════════════════════════════════════════════════════════╣\n"
                    "║  PRODUCES = you MUST create these capabilities                ║\n"
                    "║  CONSUMES = you MUST integrate with these, NOT reinvent them  ║\n"
                    "╚══════════════════════════════════════════════════════════════╝\n"
                )

        # ── 3. Combine: verified facts first, planner intent second ──
        if registry_context and planner_context:
            return registry_context + "\n" + planner_context
        elif registry_context:
            return registry_context
        elif planner_context:
            return planner_context
        return ""

    def _build_session_context(
        self,
        existing_content: str,
        task: "DevelopmentTask",
    ) -> str:
        """
        Cursor / Devin incremental technique: inject the FRESH content of files
        that were generated earlier in THIS run and are imported by the current
        file.  The stale SQLite/RAG index does not reflect interfaces that a
        prior task just changed (e.g. DisplayedMember.organization changed from
        string → object).  Showing the up-to-date shape prevents type mismatches
        (TS2322/TS2551) and dropped-export cascades (TS2305).
        """
        session_files: dict = getattr(self, "_session_files", None) or {}
        if not session_files:
            return ""

        import os as _os
        ext = Path(task.file_path).suffix.lower()
        src_dir = _os.path.dirname(task.file_path.replace("\\", "/"))
        blocks: list[str] = []
        seen: set[str] = set()
        existing_content = existing_content or ""

        # ── For HTML templates: inject sibling .ts and extract EXACT property list ──
        # This mirrors Cursor's LSP approach: give the HTML generator the exact declared
        # property names so it cannot invent names like `dmember.organizationName`.
        # ── For HTML templates: inject sibling companion from session_files ──
        if ext in (".html", ".htm"):
            from ticket_to_code.intelligence.contracts.companion_resolver import PolyglotCompanionResolver
            from ticket_to_code.intelligence.contracts import ComponentContract
            _ws = getattr(self, "workspace_path", None) or getattr(self, "_workspace_path", None)
            _comp = PolyglotCompanionResolver.resolve_companion(
                task.file_path, session_files=session_files, workspace_path=str(_ws) if _ws else ""
            )
            if _comp:
                _ctrl_key, _ctrl_content = _comp
                seen.add(_ctrl_key)
                _contract = ComponentContract.extract_from_ts(_ctrl_content, file_path=_ctrl_key)
                _allowed_block = ""
                if _contract and _contract.properties:
                    _props_unique = list(_contract.properties.keys())
                    _allowed_block = (
                        f"\n\n⚡ ALLOWED CONTROLLER BINDINGS (from TypeScript LSP — use ONLY these names):\n"
                        f"  {', '.join(_props_unique[:40])}\n"
                        f"FORBIDDEN: any property name NOT in this list.\n"
                        f"FORBIDDEN: sub-object paths or invented nested fields when the property lives on the component class.\n"
                    )

                blocks.append(
                    f"SIBLING CONTROLLER (just modified this run — use these exact properties and methods):\n"
                    f"{_ctrl_key}\n{_ctrl_content}"
                    + _allowed_block
                )
            if blocks:
                return (
                    "\n\nRELATED FILES MODIFIED EARLIER IN THIS RUN:\n```\n"
                    + "\n\n".join(blocks[:4])
                    + "\n```\n"
                )
            return ""

        # Resolve each import in the current file; if it points to a
        # session-generated file, show that file's NEW content.
        if ext in (".ts", ".tsx", ".js", ".jsx"):
            _imp_re = re.compile(r"""from\s+['"](\.[^'"]+)['"]""")
            for m in _imp_re.finditer(existing_content):
                rel = m.group(1)
                for cand_ext in (".ts", ".tsx", "/index.ts", ".d.ts"):
                    cand = _os.path.normpath(_os.path.join(src_dir, rel + cand_ext)).replace("\\", "/").lower()
                    if cand in session_files and cand not in seen:
                        seen.add(cand)
                        blocks.append(f"FILE (just modified this run): {cand}\n{session_files[cand]}")
                        break
        elif ext == ".java":
            # Java: match imports whose class was regenerated this run
            for path_key, content in session_files.items():
                if not path_key.endswith(".java"):
                    continue
                cls = Path(path_key).stem
                if re.search(r'\b' + re.escape(cls) + r'\b', existing_content) and path_key not in seen:
                    seen.add(path_key)
                    blocks.append(f"FILE (just modified this run): {path_key}\n{content}")
        elif ext == ".py":
            _imp_re = re.compile(r"from\s+(\.[.\w]+)\s+import")
            for m in _imp_re.finditer(existing_content):
                mod = m.group(1).lstrip(".").replace(".", "/")
                cand = _os.path.normpath(_os.path.join(src_dir, mod + ".py")).replace("\\", "/").lower()
                if cand in session_files and cand not in seen:
                    seen.add(cand)
                    blocks.append(f"FILE (just modified this run): {cand}\n{session_files[cand]}")

        # ── Dependency-based resolution (not just existing imports) ────────
        # When this task needs to ADD a new import (e.g., component needs
        # isProjectMember() from ProjectsService), existing_content doesn't
        # have that import yet. So we also scan cross_file_contract.consumes
        # and task.dependencies to find relevant session files.
        _contract = getattr(task, "cross_file_contract", None)
        _dep_file_hints: set[str] = set()
        # From cross_file_contract.consumes: extract from_task references
        if _contract and getattr(_contract, "consumes", None):
            for _bp in _contract.consumes:
                _from = getattr(_bp, "from_task", "") or ""
                if _from:
                    _dep_file_hints.add(_from.replace("\\", "/").lower())
        # From task.dependencies: these are task IDs or file paths
        for _dep in (task.dependencies or []):
            _dep_file_hints.add(_dep.replace("\\", "/").lower())
        # Match dependency hints against session files
        for _hint in _dep_file_hints:
            for _skey in session_files:
                if _skey in seen:
                    continue
                # Match by file path (exact or basename match)
                _hint_basename = Path(_hint).name.lower() if "/" in _hint or "\\" in _hint else _hint
                _skey_basename = Path(_skey).name.lower()
                if _hint == _skey or _hint_basename == _skey_basename:
                    seen.add(_skey)
                    _scontent = session_files[_skey]
                    _import_line = ""
                    if ext in (".ts", ".tsx") and _skey.endswith((".ts", ".tsx")):
                        _import_fwd = _compute_import_path(task.file_path, _skey)
                        if _import_fwd:
                            _exports = re.findall(
                                r'export\s+(?:class|interface|enum|type|const|function)\s+(\w+)',
                                _scontent
                            )
                            _export_str = ", ".join(_exports[:5]) if _exports else "..."
                            _import_line = (
                                f"\n\U0001f4e6 IMPORT PATH: import {{ {_export_str} }} from '{_import_fwd}';\n"
                            )
                    blocks.append(
                        f"DEPENDENCY (from cross-file contract — generated this run):\n"
                        f"{_skey}\n{_import_line}{_scontent[:2000]}"
                    )
                    logger.info(
                        f"  [SessionContext] Injected dependency from contract: {_skey}"
                    )
                    break

        # Component 3a: Inject ALL session files in the same module directory.
        # Not just .service. files — also models, interfaces, and any file
        # generated this run that lives in the same Angular module / Java package.
        # This prevents calling methods on the wrong service or using stale shapes.

        # Fix 2: Helper to compute correct relative import paths
        def _compute_import_path(from_file: str, to_file: str) -> str:
            """Compute the TypeScript relative import path from `from_file` to `to_file`.
            E.g., from 'src/app/members/components/foo.component.ts'
                  to   'src/app/members/services/bar.service.ts'
                  → './services/bar.service' (no .ts extension)
            """
            from_dir = _os.path.dirname(from_file.replace("\\", "/"))
            to_path = to_file.replace("\\", "/")
            try:
                rel = _os.path.relpath(to_path, from_dir).replace("\\", "/")
            except ValueError:
                return ""
            # Remove .ts/.tsx extension for import path
            for _ext in (".ts", ".tsx"):
                if rel.endswith(_ext):
                    rel = rel[:-len(_ext)]
                    break
            # Ensure it starts with ./ or ../
            if not rel.startswith("."):
                rel = "./" + rel
            return rel

        if ext in (".ts", ".tsx") and task.task_type.value == "modify":
            _task_dir = _os.path.dirname(task.file_path.replace("\\", "/")).lower()
            # Find the module root: walk up to the first Angular module-like dir
            # (usually one level up from 'components/', 'services/', 'models/')
            _module_dir = _task_dir
            _last_seg = _task_dir.rsplit("/", 1)[-1] if "/" in _task_dir else _task_dir
            if _last_seg in ("components", "services", "models", "pipes", "directives", "guards"):
                _module_dir = _task_dir.rsplit("/", 1)[0]

            for _skey, _scontent in session_files.items():
                if _skey in seen:
                    continue
                if not _skey.endswith((".ts", ".tsx")):
                    continue
                _skey_dir = _os.path.dirname(_skey)
                # Same module directory or direct child
                if _skey_dir.startswith(_module_dir):
                    seen.add(_skey)
                    _label = "NEW SERVICE" if ".service." in _skey else "RELATED FILE"
                    # Fix 2: Compute both directions of import paths
                    _import_fwd = _compute_import_path(task.file_path, _skey)
                    _import_rev = _compute_import_path(_skey, task.file_path)
                    _import_hint = ""
                    if _import_fwd:
                        # Extract exported class/interface names from the session file
                        _exports = re.findall(
                            r'export\s+(?:class|interface|enum|type|const|function)\s+(\w+)',
                            _scontent
                        )
                        _export_str = ", ".join(_exports[:5]) if _exports else "..."
                        _import_hint += (
                            f"\n📦 IMPORT THIS FILE INTO YOUR CODE WITH:\n"
                            f"  import {{ {_export_str} }} from '{_import_fwd}';\n"
                        )
                    if _import_rev:
                        _import_hint += (
                            f"📦 THIS FILE IMPORTS FROM YOUR FILE WITH:\n"
                            f"  from '{_import_rev}'\n"
                        )
                    blocks.append(
                        f"{_label} (created/modified this run — use these exact types):\n"
                        f"{_skey}\n{_import_hint}{_scontent[:2000]}"
                    )

        # Component 3b: Inject files referenced in task description/dependencies
        # even if they're in a different module directory
        _desc_text = f"{task.description or ''} {task.title or ''}".lower()
        for _skey, _scontent in session_files.items():
            if _skey in seen:
                continue
            _stem = Path(_skey).stem.lower()
            # Check if the file stem appears in the task description
            if len(_stem) > 4 and _stem.replace("-", "").replace("_", "") in _desc_text.replace("-", "").replace("_", ""):
                seen.add(_skey)
                # Fix 2: Also compute import path for description-referenced files
                _import_fwd = ""
                if _skey.endswith((".ts", ".tsx")) and ext in (".ts", ".tsx"):
                    _import_fwd = _compute_import_path(task.file_path, _skey)
                _import_line = ""
                if _import_fwd:
                    _exports = re.findall(
                        r'export\s+(?:class|interface|enum|type|const|function)\s+(\w+)',
                        _scontent
                    )
                    _export_str = ", ".join(_exports[:5]) if _exports else "..."
                    _import_line = (
                        f"\n📦 IMPORT PATH: import {{ {_export_str} }} from '{_import_fwd}';\n"
                    )
                blocks.append(
                    f"DEPENDENCY (referenced in task description):\n"
                    f"{_skey}\n{_import_line}{_scontent[:2000]}"
                )

        if not blocks:
            return ""

        # Fix: Cap total session context to prevent context window overflow.
        # Each block is already capped at 2000 chars per file content, but with
        # 6 blocks × 2000 chars + import hints, the total can be ~15k chars.
        # Apply a hard budget: keep blocks under _MAX_SESSION_CHARS total.
        _MAX_SESSION_CHARS = 12_000  # ~3k tokens — safe alongside other prompt sections
        _selected_blocks: list = []
        _total_chars = 0
        for _blk in blocks[:6]:
            _blk_len = len(_blk)
            if _total_chars + _blk_len > _MAX_SESSION_CHARS and _selected_blocks:
                # Truncate this block to signatures/imports only
                _sig_lines = []
                for _line in _blk.split('\n'):
                    _stripped = _line.strip()
                    if any(kw in _stripped for kw in (
                        'export ', 'import ', 'class ', 'interface ', 'enum ',
                        'public ', 'private ', 'protected ', 'def ', '@',
                        '📦', 'IMPORT', 'SERVICE', 'RELATED', 'DEPENDENCY',
                    )):
                        _sig_lines.append(_line)
                _truncated = '\n'.join(_sig_lines[:30])
                if len(_truncated) > 200:
                    _selected_blocks.append(_truncated + "\n  [... truncated for context budget]")
                    _total_chars += len(_truncated) + 50
                # else skip entirely — too small to be useful
            else:
                _selected_blocks.append(_blk)
                _total_chars += _blk_len
            if _total_chars >= _MAX_SESSION_CHARS:
                break

        if not _selected_blocks:
            return ""

        return (
            "\n\nRELATED FILES MODIFIED EARLIER IN THIS RUN — these are the "
            "AUTHORITATIVE, CURRENT definitions. Your code MUST match these exact "
            "shapes (property names, types, method signatures). Do NOT use the older "
            "shapes from the RAG context if they conflict:\n```\n"
            + "\n\n".join(_selected_blocks)
            + "\n```\n"
        )

    def _format_list(self, items: List[str]) -> str:
        """Format list with bullets"""
        if not items:
            return "- None"
        return "\n".join(f"- {item}" for item in items)

    def _format_context(self, context: List[Any]) -> str:
        """Format code chunks for prompt"""
        if not context:
            return "No existing code context provided."
        
        formatted_chunks = []
        for i, chunk in enumerate(context[:3], 1):  # Limit to top 3 — minimal context principle
            if isinstance(chunk, dict):
                # Handle dictionary representation
                c_type = chunk.get("chunk_type", "unknown")
                if hasattr(c_type, "value"): c_type = c_type.value
                elif isinstance(c_type, str): pass
                elif c_type is not None: c_type = str(c_type)
                else: c_type = "unknown"
                
                c_name = chunk.get("name", "N/A")
                c_file = chunk.get("file_path", "N/A")
                c_ns = chunk.get("namespace", "N/A")
                c_content = chunk.get("content", "")
                
                formatted_chunks.append(f"""
--- Context {i}: {str(c_type).upper()} - {c_name} ---
File: {c_file}
Namespace: {c_ns or 'N/A'}

{c_content}
                """.strip())
            else:
                # Handle object representation
                try:
                    c_type = chunk.chunk_type.value if hasattr(chunk.chunk_type, "value") else str(chunk.chunk_type)
                except AttributeError:
                    c_type = getattr(chunk, "chunk_type", "unknown")
                
                formatted_chunks.append(f"""
--- Context {i}: {str(c_type).upper()} - {getattr(chunk, 'name', 'N/A')} ---
File: {getattr(chunk, 'file_path', 'N/A')}
Namespace: {getattr(chunk, 'namespace', 'N/A') or 'N/A'}

{getattr(chunk, 'content', '')}
                """.strip())
        
        return "\n\n".join(formatted_chunks)
    


    def _parse_response(
        self, 
        response_content: str, 
        task: DevelopmentTask
    ) -> GeneratedCode:
        """
        Parse LLM response into GeneratedCode.
        
        Primary: delimiter format (<<<AVIATOR_CODE_START>>> / <<<AVIATOR_CODE_END>>>)
        Fallback: JSON format for backward compatibility.
        """
        import re
        raw = response_content.strip()

        # ── PRIMARY: Delimiter-based format ──────────────────────────────────
        code_match = re.search(r'<<<\s*AVIATOR_CODE_START\s*>>>(.*?)(?:<<<\s*AVIATOR_CODE_END\s*>>>|<<<\s*AVIATOR_META_START\s*>>>)', raw, re.DOTALL)
        meta_match = re.search(r'<<<\s*AVIATOR_META_START\s*>>>(.*?)<<<\s*AVIATOR_META_END\s*>>>', raw, re.DOTALL)

        code_content = ""
        has_delimiter_or_edit = False
        if code_match:
            code_content = code_match.group(1).strip()
            _inner_fence = re.search(r'^```(?:\w+)?\s*\r?\n(.*?)(?:\r?\n```|\Z)', code_content, re.DOTALL)
            if _inner_fence:
                code_content = _inner_fence.group(1).strip()
            code_content = _strip_conversational_prose(code_content)
            has_delimiter_or_edit = True
        elif task.task_type.value == "modify" and self._current_existing_content:
            # Check if LLM emitted SEARCH/REPLACE or TARGET/REPLACEMENT blocks directly without AVIATOR_CODE_START
            _fence_m = re.search(r'```(?:\w+)?\s*\r?\n(.*?)(?:\r?\n```|\Z)', raw, re.DOTALL)
            _candidate = _fence_m.group(1).strip() if _fence_m else raw
            if (re.search(r'<{4,}\s*(?:SEARCH|ORIGINAL|TARGET)[^\r\n]*\r?\n', _candidate)
                    or re.search(r'<{4,}[^\r\n]*\r?\n.*?\r?\n={4,}', _candidate, re.DOTALL)
                    or re.search(r'(?:TARGET(?:_CONTENT)?|ORIGINAL):\s*\r?\n?', _candidate, re.IGNORECASE)):
                code_content = _candidate
                has_delimiter_or_edit = True

        if has_delimiter_or_edit:
            _existing = self._current_existing_content

            # ── str_replace EDIT dispatch for MODIFY tasks ─────────────────────
            if task.task_type.value == "modify" and _existing:
                # Parse SEARCH/REPLACE blocks from the LLM output
                # Primary: Aider-style git-merge-conflict delimiters (7+ chars + keyword)
                #   <<<<<<< SEARCH\n...\n=======\n...\n>>>>>>> REPLACE
                _edit_pattern = re.compile(
                    r'<{4,}\s*(?:SEARCH|ORIGINAL|TARGET)[^\r\n]*\r?\n(.*?)\r?\n={4,}[^\r\n]*\r?\n(.*?)\r?\n>{4,}\s*(?:REPLACE|UPDATED|REPLACEMENT)[^\r\n]*',
                    re.DOTALL
                )
                _edit_matches = list(_edit_pattern.finditer(code_content))
                if not _edit_matches:
                    _alt_pattern = re.compile(
                        r'<{4,}[^\r\n]*\r?\n(.*?)\r?\n={4,}[^\r\n]*\r?\n(.*?)\r?\n>{4,}[^\r\n]*',
                        re.DOTALL
                    )
                    _edit_matches = list(_alt_pattern.finditer(code_content))

                # Fallback 1: legacy <<</ >>> format for backward compatibility
                if not _edit_matches:
                    _legacy_pattern = re.compile(
                        r'EDIT:\s*\r?\n'
                        r'old_str:\s*\r?\n<<<\r?\n(.*?)\r?\n>>>\s*\r?\n'
                        r'new_str:\s*\r?\n<<<\r?\n(.*?)\r?\n>>>',
                        re.DOTALL
                    )
                    _edit_matches = list(_legacy_pattern.finditer(code_content))
                    if _edit_matches:
                        logger.info(
                            f"  Parser: LLM used legacy <<</>>> format for {task.file_path} "
                            f"— parsed {len(_edit_matches)} edit(s) via fallback"
                        )

                # Fallback 2: Antigravity / AI IDE TARGET/REPLACEMENT format
                if not _edit_matches:
                    _target_pattern = re.compile(
                        r'(?:TARGET(?:_CONTENT)?|ORIGINAL):\s*\r?\n?(.*?)\r?\n?(?:REPLACEMENT(?:_CONTENT)?|REPLACE):\s*\r?\n?(.*?)(?=\r?\n(?:TARGET(?:_CONTENT)?|ORIGINAL):|\Z)',
                        re.DOTALL | re.IGNORECASE,
                    )
                    _edit_matches = list(_target_pattern.finditer(code_content))
                    if _edit_matches:
                        logger.info(
                            f"  Parser: LLM used TARGET/REPLACEMENT format for {task.file_path} "
                            f"— parsed {len(_edit_matches)} edit(s) via fallback"
                        )

                if _edit_matches:
                    # Build edit list for atomic application
                    # Wire scope_method from edit_anchors for ambiguity resolution
                    _anchor_methods = [
                        a.get("anchor_method", "")
                        for a in getattr(task, "edit_anchors", []) or []
                        if a.get("anchor_method")
                    ]
                    _edits = []
                    for _idx, _em in enumerate(_edit_matches):
                        _old_s = _em.group(1)
                        _new_s = _strip_conversational_prose(_em.group(2))

                        # Guard: if new_str contains another SEARCH/REPLACE block,
                        # the regex captured across block boundaries due to
                        # malformed LLM output.  Reject immediately so retry fires.
                        if re.search(r'<{7,}\s*SEARCH|>{7,}\s*REPLACE', _new_s):
                            raise ValueError(
                                f"Edit {_idx+1}: REPLACE block contains nested SEARCH/REPLACE markers. "
                                f"The LLM produced malformed output — retrying."
                            )
                        # Same guard for legacy format
                        if re.search(r'EDIT:\s*\nold_str:', _new_s):
                            raise ValueError(
                                f"Edit {_idx+1}: new_str contains another EDIT block. "
                                f"The LLM produced malformed output — retrying."
                            )

                        # Try to match this edit to an anchor by index
                        _scope = ""
                        if _idx < len(_anchor_methods):
                            _scope = _anchor_methods[_idx]
                        elif len(_anchor_methods) == 1:
                            # All edits scoped to the single anchor
                            _scope = _anchor_methods[0]
                        _edits.append({
                            "old_str": _old_s,
                            "new_str": _new_s,
                            "scope_method": _scope,
                        })

                    try:
                        code_content = _apply_str_replace_edits(
                            _edits, _existing, task.file_path
                        )
                        logger.info(
                            f"  str_replace: {len(_edits)} edit(s) applied "
                            f"successfully to {task.file_path}"
                        )

                        # ── Post-Generation Patch Gate Validation ─────────────
                        try:
                            from ticket_to_code.agents.patch_gate import PatchGate
                            _auth_files = getattr(self, "_current_allowed_files", None) or set()
                            
                            # Gather sibling companion content for cross-artifact validation
                            _sibling_c = None
                            _sibling_p = None
                            if Path(task.file_path).suffix.lower() in (".html", ".htm", ".razor"):
                                from ticket_to_code.intelligence.contracts.companion_resolver import PolyglotCompanionResolver
                                _session = getattr(self, "_session_files", None) or {}
                                _ws = getattr(self, "workspace_path", None) or getattr(self, "_workspace_path", None)
                                _comp = PolyglotCompanionResolver.resolve_companion(
                                    task.file_path, session_files=_session, workspace_path=str(_ws) if _ws else None
                                )
                                if _comp:
                                    _sibling_p, _sibling_c = _comp

                            _gate_ok, _gate_reason = PatchGate.validate_patch(
                                file_path=task.file_path,
                                patch_content=code_content,
                                authorized_writable_files=_auth_files if _auth_files else {task.file_path},
                                change_targets=getattr(task, "change_targets", []),
                                has_migration_evidence=False,
                                sibling_content=_sibling_c,
                                sibling_path=_sibling_p,
                            )
                            if not _gate_ok:
                                logger.warning(f"  [PatchGate] Validation failed for {task.file_path}: {_gate_reason}")
                                raise ValueError(f"[PatchGate] {_gate_reason}")
                        except ImportError:
                            pass
                    except ValueError as edit_err:
                        logger.warning(
                            f"str_replace edit failed for {task.file_path}: "
                            f"{edit_err} — preserving original file"
                        )
                        code_content = _existing
                        documentation = f"FAILED: {edit_err}"

                else:
                    # No EDIT blocks found — check for legacy format or full-file output
                    # Legacy: REPLACE_LINES, INSERT_*, SEARCH/REPLACE
                    _legacy_markers = (
                        "REPLACE_LINES:", "INSERT_BEFORE_CLASS_END:",
                        "INSERT_AT_FILE_END:", "INSERT_AFTER_IMPORTS:",
                        "INSERT_BEFORE_HTML_END:", "INSERT_AFTER_LINE:",
                        "SEARCH:", "REPLACE:",
                    )
                    _has_legacy = any(m in code_content for m in _legacy_markers)

                    if _has_legacy:
                        logger.warning(
                            f"MODIFY task {task.file_path}: LLM used LEGACY format "
                            f"(REPLACE_LINES/INSERT/SEARCH). These are deprecated. "
                            f"Preserving original — retry will re-prompt for EDIT format."
                        )
                        code_content = _existing
                        documentation = (
                            "FAILED: LLM used deprecated REPLACE_LINES/INSERT/SEARCH "
                            "format instead of str_replace EDIT blocks"
                        )
                    else:
                        # No patch directives at all — detect full-file rewrite
                        _ext = (task.file_path or "").rsplit(".", 1)[-1].lower()
                        _first = code_content.lstrip()[:120]
                        _is_full_file = False
                        if _ext in ("java", "kt"):
                            _is_full_file = _first.startswith("package ")
                        elif _ext in ("ts", "tsx", "js", "jsx"):
                            _orig_has_imports = bool(re.search(r'^\s*import\s+', _existing, re.MULTILINE)) if _existing else False
                            _gen_has_imports = bool(re.search(r'^\s*import\s+', code_content, re.MULTILINE))
                            if _orig_has_imports and not _gen_has_imports:
                                _is_full_file = False
                            else:
                                _import_count = sum(
                                    1 for ln in code_content.split("\n")[:15]
                                    if ln.strip().startswith("import ")
                                )
                                _has_comp = any(kw in code_content for kw in (
                                    "export const ", "export function ", "export default ",
                                    "export interface ", "export class "
                                ))
                                _is_full_file = _import_count >= 1 or (_has_comp and len(code_content.splitlines()) >= 20)
                        elif _ext == "py":
                            _is_full_file = len(code_content.splitlines()) > 30
                        else:
                            _is_full_file = len(code_content.splitlines()) > 40

                        if _is_full_file:
                            # ── Resilient Diff Recovery (Cursor/Devin style) ───────
                            # If the model output a full-file rewrite that is mostly identical
                            # to the existing file (e.g. >=50% similarity), synthesize the
                            # surgical edits using difflib SequenceMatcher. This avoids
                            # wasting a 60-90s LLM round-trip and thousands of tokens.
                            _recovered = False
                            if _existing:
                                import difflib
                                _orig_lines = _existing.splitlines(keepends=True)
                                _gen_lines = code_content.splitlines(keepends=True)
                                _matcher = difflib.SequenceMatcher(None, _orig_lines, _gen_lines)
                                _sim_ratio = _matcher.ratio()
                                if _sim_ratio >= 0.50:
                                    _synthetic_edits = []
                                    for _tag, _i1, _i2, _j1, _j2 in _matcher.get_opcodes():
                                        if _tag != "equal":
                                            _s_chunk = "".join(_orig_lines[_i1:_i2])
                                            _r_chunk = "".join(_gen_lines[_j1:_j2])
                                            if _s_chunk or _r_chunk:
                                                _synthetic_edits.append((_s_chunk, _r_chunk))
                                    if _synthetic_edits:
                                        try:
                                            _edit_dicts = [{"old_str": s, "new_str": r} for s, r in _synthetic_edits]
                                            _patched = _apply_str_replace_edits(
                                                _edit_dicts, _existing, task.file_path
                                            )
                                            if _patched and _patched != _existing:
                                                code_content = _patched
                                                _recovered = True
                                                logger.info(
                                                    f"  🪄 Resilient Diff Recovery: Synthesized {len(_synthetic_edits)} "
                                                    f"surgical edit(s) from full-file output for {task.file_path} "
                                                    f"(similarity={_sim_ratio:.1%}). Zero retry needed!"
                                                )
                                        except Exception as _rec_err:
                                            logger.debug(f"Resilient diff recovery failed: {_rec_err}")

                            if not _recovered:
                                _is_safe_whole_replace = False
                                if len(_existing) < 3000 and any(kw in code_content for kw in ("enum ", "type ", "interface ", "class ")):
                                    _is_safe_whole_replace = True
                                elif len(code_content) >= len(_existing) * 0.70:
                                    _existing_exports = set(re.findall(r'\b(?:export\s+(?:default\s+)?(?:const|function|class|interface|type)|class|interface|enum)\s+(\w+)', _existing))
                                    _new_exports = set(re.findall(r'\b(?:export\s+(?:default\s+)?(?:const|function|class|interface|type)|class|interface|enum)\s+(\w+)', code_content))
                                    if _existing_exports and _existing_exports.issubset(_new_exports):
                                        _is_safe_whole_replace = True

                                if _is_safe_whole_replace:
                                    logger.info(f"  🪄 Safe small-file whole replacement accepted ({len(_existing)} chars) for {task.file_path}")
                                    _recovered = True
                            if not _recovered:
                                logger.warning(
                                    f"MODIFY task {task.file_path}: LLM output detected as "
                                    f"full-file rewrite (ext={_ext}, no EDIT blocks). "
                                    f"Raising error to trigger retry with SEARCH/REPLACE instruction."
                                )
                                raise ValueError(
                                    f"SEARCH block not found: LLM output a full-file rewrite "
                                    f"instead of SEARCH/REPLACE edit blocks for MODIFY task "
                                    f"on {task.file_path}. The LLM must use "
                                    f"<<<<<<< SEARCH / ======= / >>>>>>> REPLACE format. "
                                    f"Do NOT output the entire file."
                                )
                        else:
                            # If it's a modify task on an existing file, an unanchored fragment CANNOT overwrite the file!
                            logger.warning(
                                f"MODIFY task {task.file_path}: LLM output has no SEARCH/REPLACE blocks "
                                f"and is not a full-file rewrite. Rejecting raw fragment ({len(code_content.splitlines())} lines) "
                                f"to avoid destroying existing file ({len(_existing.splitlines())} lines)."
                            )
                            raise ValueError(
                                f"SEARCH block not found: LLM produced an unanchored code fragment without "
                                f"<<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks for MODIFY task on {task.file_path}. "
                                f"You must use exact SEARCH/REPLACE blocks copying lines verbatim from the existing file. "
                                f"Do not output snippets or conversational text."
                            )

            documentation = "Generated code (delimiter format)"
            imports: list = []
            if meta_match:
                meta_raw = meta_match.group(1).strip()
                try:
                    meta = json.loads(meta_raw)
                    documentation = meta.get("documentation", documentation)
                    imports = meta.get("imports", [])
                except Exception:
                    pass  # meta is optional

            # ── Fix 4: Post-patch verification ───────────────────────────────
            # Verify that expected new method names appear in the result. If an
            # additive task was supposed to add a method but it's absent, the
            # INSERT strategy silently failed — escalate to let retry logic act.
            if (task.task_type.value == "modify"
                    and task.allowed_methods
                    and code_content != (_existing or "")
                    and not documentation.startswith("FAILED")):
                _absent = [m for m in task.allowed_methods
                           if not re.search(r'\b' + re.escape(m) + r'\b', code_content)]
                if _absent and len(_absent) == len(task.allowed_methods):
                    # ALL expected symbols are absent — patch almost certainly wrong
                    logger.warning(
                        f"Post-patch: ALL allowed_methods {_absent} absent from {task.file_path} "
                        f"after patch — marking as failed so fix_build can retry."
                    )
                    documentation = f"FAILED: expected methods {_absent} not found after patch"

            # ── Component 2b: Post-patch constructor verification ─────────────
            # If the original had a constructor, verify NO params were dropped.
            _ext_ctor_check = Path(task.file_path).suffix.lower()
            if (_ext_ctor_check in (".ts", ".tsx", ".java")
                    and _existing and code_content != _existing
                    and not documentation.startswith("FAILED")):
                _orig_ctor_m = re.search(
                    r'constructor\s*\(([^)]+)\)', _existing, re.DOTALL
                )
                _new_ctor_m = re.search(
                    r'constructor\s*\(([^)]+)\)', code_content, re.DOTALL
                )
                if _orig_ctor_m and _new_ctor_m:
                    # Extract param names from original and new
                    def _param_names(params_str: str) -> set:
                        # Match 'private foo: Bar' or 'foo: Bar' patterns
                        return set(re.findall(
                            r'(?:private|public|protected|readonly)?\s*(\w+)\s*[:\,)]',
                            params_str
                        ))
                    _orig_params = _param_names(_orig_ctor_m.group(1))
                    _new_params  = _param_names(_new_ctor_m.group(1))
                    _dropped = _orig_params - _new_params
                    if _dropped:
                        logger.warning(
                            f"Constructor verification FAILED for {task.file_path}: "
                            f"dropped params {_dropped} — preserving original"
                        )
                        code_content = _existing
                        documentation = f"FAILED: constructor dropped params {_dropped}"

            # ── Fix 6: Quick syntax check (brace balance) ────────────────────
            # Catch the most common corruption (unbalanced braces from bad INSERT)
            # before writing to disk.  Ignores strings/comments — intentionally
            # simple, catches gross errors not subtle ones.
            _ext_syn = (task.file_path or "").rsplit(".", 1)[-1].lower()
            if _ext_syn in ("java", "kt", "ts", "tsx", "js", "jsx", "cs") and code_content:
                _depth = 0
                for _ch in code_content:
                    if _ch == "{":
                        _depth += 1
                    elif _ch == "}":
                        _depth -= 1
                if _depth != 0:
                    logger.warning(
                        f"Syntax check: unbalanced braces in {task.file_path} "
                        f"(depth={_depth} after patch) — preserving original."
                    )
                    code_content = _existing or code_content
                    documentation = f"FAILED: unbalanced braces (depth={_depth}) in generated patch"

            logger.info(f"Parsed response via regex delimiter format: {len(code_content)} chars")
            return GeneratedCode(
                file_path=task.file_path,
                content=code_content,
                language=task.language,
                change_type=task.task_type,
                documentation=documentation,
                imports=imports,
            )
        elif re.search(r'<<<\s*AVIATOR_CODE_START\s*>>>', raw) and not re.search(r'<<<\s*AVIATOR_CODE_END\s*>>>|<<<\s*AVIATOR_META_START\s*>>>', raw):
            raise ValueError("LLM output truncated: AVIATOR_CODE_START found but AVIATOR_CODE_END missing.")

        # ── FALLBACK: JSON format ─────────────────────────────────────────────
        cleaned = raw
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        _json_err: str = ""
        try:
            data = json.loads(cleaned)
            logger.info("Parsed response via JSON format")
            return GeneratedCode(
                file_path=task.file_path,
                content=data.get("code", ""),
                language=task.language,
                change_type=task.task_type,
                documentation=data.get("documentation"),
                imports=data.get("imports", []),
            )
        except json.JSONDecodeError as e:
            _json_err = str(e)
            logger.warning(f"JSON parsing failed, attempting regex fallback: {e}")

        # ── REGEX FALLBACKS ───────────────────────────────────────────────────

        # Fallback 1: greedy extraction — everything after "code": " up to the
        # last plausible end marker (handles truncated/unescaped responses)
        code_key_pos = cleaned.find('"code"')
        if code_key_pos >= 0:
            colon_pos = cleaned.find(':', code_key_pos)
            open_quote = cleaned.find('"', colon_pos + 1)
            if open_quote >= 0:
                raw_value = cleaned[open_quote + 1:]
                # Strip known trailing meta markers
                for marker in ['", "imports"', '",\n  "imports"', '", "documentation"', '"\n}']:
                    idx = raw_value.rfind(marker)
                    if idx > 0:
                        raw_value = raw_value[:idx]
                        break
                code_content = raw_value.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                logger.info("Extracted code via greedy quote fallback")
                return GeneratedCode(
                    file_path=task.file_path,
                    content=_heal_react_destructuring(code_content) if task.file_path.endswith((".tsx", ".jsx")) else code_content,
                    language=task.language,
                    change_type=task.task_type,
                    documentation="Generated code (greedy fallback)",
                    imports=[],
                )

        # Fallback 2: code block in raw response (closed or unclosed)
        code_match = re.search(r'```(?:\w+)?\s*\r?\n(.*?)(?:\r?\n```|\Z)', raw, re.DOTALL)
        if not code_match:
            code_match = re.search(r'```(?:\w+)?\s*(.*?)\s*```', raw, re.DOTALL)

        extracted_raw_code = code_match.group(1).strip() if code_match else ""

        # Fallback 2b: Raw code without backticks (with prose rejection)
        if not extracted_raw_code:
            lines = raw.splitlines()
            code_lines = []
            in_code = False
            _PROSE_PREFIXES = ("so ", "now", "here ", "the ", "this ", "note:", "we ", "i ", "you ", "first,", "second,", "in this", "let's")
            for line in lines:
                stripped = line.strip()
                if any(stripped.lower().startswith(p) for p in _PROSE_PREFIXES):
                    in_code = False
                    continue
                if (stripped.startswith(('import ', 'export ', 'const ', 'let ', 'var ', 'function ', 'class ', 'interface ', 'type ', 'package ', 'public ', 'private ', 'protected ', '<')) or in_code):
                    in_code = True
                    code_lines.append(line)
            if code_lines and len(code_lines) >= 3:
                extracted_raw_code = "\n".join(code_lines).strip()

        if extracted_raw_code:
            _existing = self._current_existing_content
            if task.task_type.value == "modify" and _existing:
                # 1. First, search for SEARCH/REPLACE or TARGET/REPLACEMENT blocks anywhere in raw response
                _search_matches = list(re.finditer(
                    r'<{4,}\s*(?:SEARCH|ORIGINAL|TARGET)[^\r\n]*\r?\n(.*?)\r?\n={4,}[^\r\n]*\r?\n(.*?)\r?\n>{4,}\s*(?:REPLACE|UPDATED|REPLACEMENT)[^\r\n]*',
                    raw, re.DOTALL
                ))
                if not _search_matches:
                    _search_matches = list(re.finditer(
                        r'<{4,}[^\r\n]*\r?\n(.*?)\r?\n={4,}[^\r\n]*\r?\n(.*?)\r?\n>{4,}[^\r\n]*',
                        raw, re.DOTALL
                    ))
                if not _search_matches:
                    _search_matches = list(re.finditer(
                        r'(?:SEARCH(?:\s+FOR)?|TARGET(?:_CONTENT)?|ORIGINAL):\s*\r?\n?(.*?)\r?\n?(?:REPLACE(?:\s+WITH)?|REPLACEMENT(?:_CONTENT)?):\s*\r?\n?(.*?)(?=\r?\n(?:SEARCH(?:\s+FOR)?|TARGET(?:_CONTENT)?|ORIGINAL):|\Z)',
                        raw, re.DOTALL | re.IGNORECASE
                    ))
                if not _search_matches:
                    _search_matches = list(re.finditer(
                        r'(?:TARGET(?:_CONTENT)?|ORIGINAL):\s*\r?\n?(.*?)\r?\n?(?:REPLACEMENT(?:_CONTENT)?|REPLACE):\s*\r?\n?(.*?)(?=\r?\n(?:TARGET(?:_CONTENT)?|ORIGINAL):|\Z)',
                        raw, re.DOTALL | re.IGNORECASE
                    ))
                if _search_matches:
                    _edits = [{"old_str": m.group(1), "new_str": m.group(2)} for m in _search_matches]
                    try:
                        patched = _apply_str_replace_edits(_edits, _existing, task.file_path)
                        logger.info(f"  Fallback 2: Applied {len(_edits)} edit(s) via SEARCH/REPLACE from raw response")
                        return GeneratedCode(
                            file_path=task.file_path,
                            content=patched,
                            language=task.language,
                            change_type=task.task_type,
                            documentation="Generated code (SEARCH/REPLACE from fallback)",
                            imports=[],
                        )
                    except Exception as e:
                        logger.warning(f"  Fallback 2: edit application failed: {e}")

                # 2. Resilient Diff Recovery (Cursor/Devin style)
                import difflib
                _orig_lines = _existing.splitlines(keepends=True)
                _gen_lines = extracted_raw_code.splitlines(keepends=True)
                _matcher = difflib.SequenceMatcher(None, _orig_lines, _gen_lines)
                _sim_ratio = _matcher.ratio()
                if _sim_ratio >= 0.70 and len(_gen_lines) >= len(_orig_lines) * 0.7:
                    _synthetic_edits = []
                    for _tag, _i1, _i2, _j1, _j2 in _matcher.get_opcodes():
                        if _tag != "equal":
                            _s_chunk = "".join(_orig_lines[_i1:_i2])
                            _r_chunk = "".join(_gen_lines[_j1:_j2])
                            if _s_chunk or _r_chunk:
                                _synthetic_edits.append({"old_str": _s_chunk, "new_str": _r_chunk})
                    if _synthetic_edits:
                        try:
                            _patched = _apply_str_replace_edits(_synthetic_edits, _existing, task.file_path)
                            if _patched and _patched != _existing:
                                _ast_issues = _validate_patch_ast(_existing, _patched, task.file_path)
                                if not any("unbalanced delimiters" in iss or "Method count dropped" in iss for iss in _ast_issues):
                                    logger.info(
                                        f"  🪄 Fallback 2 Resilient Diff: Synthesized {len(_synthetic_edits)} "
                                        f"surgical edit(s) for {task.file_path} (sim={_sim_ratio:.1%})"
                                    )
                                    return GeneratedCode(
                                        file_path=task.file_path,
                                        content=_patched,
                                        language=task.language,
                                        change_type=task.task_type,
                                        documentation="Generated code (resilient diff fallback)",
                                        imports=[],
                                    )
                        except Exception as _rec_err:
                            logger.debug(f"Fallback 2 diff recovery failed: {_rec_err}")

                # 3. Snippet Locality Stitching: If model returned a small pure JSX snippet (< 80 lines) rather than full file
                if len(_gen_lines) < 80:
                    _snip = extracted_raw_code.strip()
                    # Only allow pure JSX snippets (never snippets containing import/export/function declarations)
                    _is_pure_jsx = not any(_snip.startswith(kw) for kw in ("import ", "export ", "function ", "interface ", "class ", "type "))
                    # A) If it has a status badge or consistency JSX, insert above table
                    if _is_pure_jsx and ("<table" in _existing or "cf-table" in _existing) and any(w in _snip for w in ("Consistent", "Discrepanc", "badge", "status")):
                        _tbl_pos = _existing.find('<table')
                        if _tbl_pos != -1:
                            _stitched = _existing[:_tbl_pos] + _snip + "\n          " + _existing[_tbl_pos:]
                            if not _validate_patch_ast(_existing, _stitched, task.file_path):
                                logger.info(f"  🪄 Snippet Locality Stitching: inserted badge above table in {task.file_path}")
                                return GeneratedCode(
                                    file_path=task.file_path,
                                    content=_stitched,
                                    language=task.language,
                                    change_type=task.task_type,
                                    documentation="Generated code (snippet locality stitched above table)",
                                    imports=[],
                                )
                    # B) If it's a state/hook calculation snippet
                    elif _is_pure_jsx and ("export default function" in _existing or "function App" in _existing):
                        if _snip.startswith("const ") and ("orphanedCount" in _snip or "reservations" in _snip):
                            _ret_pos = _existing.find('return (')
                            if _ret_pos != -1:
                                _stitched = _existing[:_ret_pos] + _snip + "\n\n  " + _existing[_ret_pos:]
                                if not _validate_patch_ast(_existing, _stitched, task.file_path):
                                    logger.info(f"  🪄 Snippet Locality Stitching: inserted state logic before return in {task.file_path}")
                                    return GeneratedCode(
                                        file_path=task.file_path,
                                        content=_stitched,
                                        language=task.language,
                                        change_type=task.task_type,
                                        documentation="Generated code (snippet state stitched before return)",
                                        imports=[],
                                    )
                    # C) Class Method Snippet Stitching (Java / C# / TypeScript / JavaScript classes)
                    _has_class_decl = bool(re.search(r'\b(class|interface|enum)\s+\w+', extracted_raw_code))
                    if ("class " in _existing or "interface " in _existing):
                        _snip = extracted_raw_code.strip()
                        # If wrapped in a class/interface stub, unwrap inner method body
                        if _has_class_decl:
                            first_b = _snip.find('{')
                            last_b = _snip.rfind('}')
                            if first_b != -1 and last_b > first_b:
                                inner = _snip[first_b + 1:last_b].strip()
                                if inner and any(term in inner for term in ("public ", "private ", "protected ", "def ", "void ", "ResponseEntity")):
                                    _snip = inner
                        if any(term in _snip for term in ("public ", "private ", "protected ", "@PostMapping", "@GetMapping", "@PutMapping", "@DeleteMapping", "@RequestMapping", "@Override", "void ", "ResponseEntity", "def ")):
                            last_brace = _existing.rfind('}')
                            if last_brace != -1:
                                snip_imports = []
                                snip_body_lines = []
                                for sline in _snip.splitlines():
                                    if sline.strip().startswith("import "):
                                        if sline.strip() not in _existing:
                                            snip_imports.append(sline.strip())
                                    else:
                                        snip_body_lines.append(sline)
                                snip_body = "\n".join(snip_body_lines).strip()
                                # Deduplicate: do NOT re-append methods or fields that already exist in _existing
                                new_body_lines = []
                                for line in snip_body.splitlines():
                                    s_stripped = line.strip()
                                    # If line is a field or method signature that already exists in _existing, skip it
                                    if s_stripped.startswith("private final ") and s_stripped in _existing:
                                        continue
                                    if any(sig in _existing for sig in [
                                        "ResponseEntity<Order> createOrder(",
                                        "ResponseEntity<Order> cancelOrder(",
                                        "Order placeOrder(",
                                        "Order cancelOrder("
                                    ]) and any(s_stripped.startswith(k) for k in [
                                        "public ResponseEntity<Order> createOrder",
                                        "public ResponseEntity<Order> cancelOrder",
                                        "public Order placeOrder",
                                        "public Order cancelOrder"
                                    ]):
                                        continue
                                    new_body_lines.append(line)
                                clean_snip_body = "\n".join(new_body_lines).strip()
                                stitched_body = _existing[:last_brace].rstrip() + "\n\n    " + clean_snip_body.replace("\n", "\n    ") + "\n" + _existing[last_brace:]
                                if snip_imports:
                                    pkg_idx = stitched_body.find(";\n")
                                    if pkg_idx != -1 and stitched_body[:pkg_idx].strip().startswith("package "):
                                        _stitched = stitched_body[:pkg_idx + 2] + "\n" + "\n".join(snip_imports) + "\n" + stitched_body[pkg_idx + 2:]
                                    else:
                                        _stitched = "\n".join(snip_imports) + "\n\n" + stitched_body
                                else:
                                    _stitched = stitched_body

                                if _stitched.count('{') == _stitched.count('}') and _stitched.count('(') == _stitched.count(')'):
                                    logger.info(f"  🪄 Snippet Method Stitching: successfully stitched method into class in {task.file_path}")
                                    return GeneratedCode(
                                        file_path=task.file_path,
                                        content=_stitched,
                                        language=task.language,
                                        change_type=task.task_type,
                                        documentation="Generated code (snippet method stitched into class)",
                                        imports=snip_imports,
                                    )

                # For MODIFY tasks on existing files, do NOT allow raw unverified conversational prose or fragments to overwrite the file
                ast_issues = _validate_patch_ast(_existing, extracted_raw_code, task.file_path)
                fatal = [iss for iss in ast_issues if "unbalanced delimiters" in iss or "Method count dropped" in iss]
                if fatal:
                    logger.warning(
                        f"Resilient fallback rejected for MODIFY task on {task.file_path}: {fatal} — preserving original file"
                    )
                    return GeneratedCode(
                        file_path=task.file_path,
                        content=_existing,
                        language=task.language,
                        change_type=task.task_type,
                        documentation=f"FAILED: {'; '.join(fatal)}",
                        imports=[],
                    )

            # Strict Gate: For MODIFY tasks, reject unanchored raw text if it has syntax errors, prose leaks, or massive bloat
            if task.task_type.value == "modify" and _existing:
                _has_prose = any(phrase in extracted_raw_code for phrase in [
                    "But wait:", "However,", "Therefore,", "SEARCH for:", "REPLACE with:",
                    "Third, we add", "What about", "We'll need to"
                ])
                if _has_prose or len(extracted_raw_code) > len(_existing) * 2.5:
                    logger.warning(
                        f"Resilient fallback rejected for MODIFY task on {task.file_path}: "
                        f"detected prose leak or excessive length ({len(extracted_raw_code)} chars vs {len(_existing)} orig). "
                        f"Preserving original file."
                    )
                    raise ValueError(
                        f"SEARCH block not found: LLM produced unanchored conversational text instead of SEARCH/REPLACE blocks. "
                        f"You must use <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks."
                    )

            logger.info(f"Extracted code via resilient fallback ({len(extracted_raw_code)} chars)")
            return GeneratedCode(
                file_path=task.file_path,
                content=extracted_raw_code,
                language=task.language,
                change_type=task.task_type,
                documentation="Generated code (resilient fallback)",
                imports=[],
            )

        # Fallback 3: For MODIFY tasks, return original content unchanged only as last resort.
        if task.task_type.value == "modify":
            logger.error("All parsing methods failed for MODIFY task — preserving original file")
            return GeneratedCode(
                file_path=task.file_path,
                content=self._current_existing_content or "",
                language=task.language,
                change_type=task.task_type,
                documentation="FAILED: No parseable code generated, original preserved",
                imports=[],
            )
        
        logger.info("Extracted code via raw string fallback (CREATE task)")
        return GeneratedCode(
            file_path=task.file_path,
            content=raw,
            language=task.language,
            change_type=task.task_type,
            documentation="Generated code (raw fallback)",
            imports=[],
        )


# ============================================================================
# PATCH VALIDATOR
# ============================================================================

class PatchValidationResult:
    """Result of patch scope validation"""
    def __init__(self, passed: bool, violations: list[str], metrics: dict):
        self.passed = passed
        self.violations = violations
        self.metrics = metrics  # files_modified, lines_changed, etc.

    def __repr__(self):
        return f"PatchValidation(passed={self.passed}, violations={self.violations}, metrics={self.metrics})"


class PatchValidator:
    """
    Validates that generated code stays within the frozen modification scope.

    Checks:
      CHECK 1 — File scope: generated file_path matches the allowed task file
      CHECK 2 — No new files invented beyond what planner authorised (task_type=create)
      CHECK 3 — Change budget: lines changed not disproportionately large
      CHECK 4 — No obvious cross-file contamination in the code body
    """

    # Maximum ratio of changed lines vs original for a "modify" task before warning
    MAX_CHANGE_RATIO = 0.8   # if >80% of original lines changed → suspicious full-rewrite

    def validate(
        self,
        generated: "GeneratedCode",
        task: "DevelopmentTask",
        existing_content: str = None
    ) -> PatchValidationResult:
        violations = []
        metrics = {}

        # ── CHECK 1: File path must match the frozen allowed file ────────────
        normalized_gen  = generated.file_path.replace("\\", "/").lower()
        normalized_task = task.file_path.replace("\\", "/").lower()
        if normalized_gen != normalized_task:
            violations.append(
                f"FILE SCOPE VIOLATION: generator output targets '{generated.file_path}' "
                f"but allowed file is '{task.file_path}'"
            )
        metrics["file_match"] = (normalized_gen == normalized_task)

        # ── CHECK 2: Modify task must not balloon into a full creative rewrite ─
        if existing_content and task.task_type.value == "modify":
            original_lines = existing_content.splitlines()
            new_lines      = generated.content.splitlines()
            metrics["original_line_count"] = len(original_lines)
            metrics["new_line_count"]      = len(new_lines)

            # Compute rough change ratio using difflib
            import difflib
            matcher   = difflib.SequenceMatcher(None, original_lines, new_lines)
            same_ratio = matcher.ratio()          # 1.0 = identical, 0.0 = totally different
            change_ratio = 1.0 - same_ratio
            metrics["change_ratio"] = round(change_ratio, 3)

            orig_len = len(original_lines)
            target_name = getattr(task, "target_file", "") or getattr(task, "file_path", "") or ""
            is_ui_or_small = (
                orig_len < 120
                or target_name.lower().endswith((".tsx", ".jsx", ".html", ".css", ".scss", ".vue"))
            )
            # Only enforce change budget on larger files (>= 120 lines). Small files naturally
            # have high change ratios when imports, props, and handlers are modified.
            if orig_len >= 120 and not is_ui_or_small and change_ratio > self.MAX_CHANGE_RATIO:
                violations.append(
                    f"CHANGE BUDGET EXCEEDED: {change_ratio:.0%} of file changed "
                    f"(limit {self.MAX_CHANGE_RATIO:.0%}). Generator likely rewrote the whole file."
                )

            # Count actual changed lines
            changed_lines = sum(
                1 for tag, _, _, _, _ in matcher.get_opcodes() if tag != "equal"
            )
            metrics["changed_blocks"] = changed_lines

        # ── CHECK 3: Detect keyword evidence of unauthorized new file creation ─
        content_lower = generated.content.lower()
        suspicious_phrases = [
            "// new file", "// create this file", "// file:",
        ]
        for phrase in suspicious_phrases:
            if phrase in content_lower and task.task_type.value == "modify":
                violations.append(
                    f"SUSPICIOUS PATTERN: generated content contains '{phrase}' in a MODIFY task"
                )

        # ── CHECK 4: Method-level scope enforcement ────────────────────────
        # Only enforce when the localization agent populated allowed_methods AND
        # we have both original and new content to compare.
        if existing_content and task.allowed_methods and task.task_type.value == "modify":
            import re
            method_sig_pattern = re.compile(
                r"(?:(?:public|private|protected|static|async|override|abstract)\s+)*"
                r"(?:[\w<>\[\]]+\s+)?"
                r"(\w+)\s*\([^)]*\)\s*(?::\s*[\w<>\[\]|]+)?\s*(?:\{|=>)",
                re.MULTILINE
            )
            python_def = re.compile(r"^\s*def\s+(\w+)\s*\(", re.MULTILINE)
            _KEYWORDS = {"if", "for", "while", "switch", "catch", "try", "else", "return",
                         "new", "class", "interface", "constructor", "function",
                         "async", "await", "const", "let", "var", "export", "import", "type"}
            # RxJS/Observable call-site names that appear as method-like patterns
            # but are calls on streams, not new class method declarations.
            _RXJS_CALLSITES = {
                "subscribe", "pipe", "map", "tap", "filter", "switchMap", "mergeMap",
                "concatMap", "exhaustMap", "catchError", "finalize", "take", "takeUntil",
                "debounceTime", "distinctUntilChanged", "share", "shareReplay",
                "of", "from", "forkJoin", "combineLatest", "zip", "throwError",
            }

            def _extract_methods(src: str) -> set:
                names = set()
                for m in method_sig_pattern.finditer(src):
                    n = m.group(1)
                    if n not in _KEYWORDS and n not in _RXJS_CALLSITES:
                        names.add(n)
                for m in python_def.finditer(src):
                    names.add(m.group(1))
                return names

            original_methods = _extract_methods(existing_content)
            new_methods      = _extract_methods(generated.content)

            # Methods that appear in NEW but NOT in ORIGINAL — truly new additions
            truly_new = new_methods - original_methods
            if truly_new and not any(m in task.allowed_methods for m in truly_new):
                # Allow new methods whose names appear in the task description or ticket
                # title — this covers feature tasks where the LLM correctly invents a
                # new helper method (e.g. checkProjectMembership) not pre-listed by planner.
                _task_text = " ".join([
                    (task.description or ""),
                    (task.title or ""),
                    " ".join(task.allowed_methods or []),
                ]).lower()
                _allowed_new = {
                    m for m in truly_new
                    if any(word in _task_text for word in [
                        m.lower()[:6],           # first 6 chars of method name
                        "membership", "organization", "project", "contract",
                        "existing", "check", "get", "fetch", "load", "is", "has",
                    ])
                }
                _blocked_new = truly_new - _allowed_new
                if _blocked_new:
                    violations.append(
                        f"METHOD SCOPE VIOLATION: generator added new method(s) "
                        f"{_blocked_new} which are not in allowed_methods {task.allowed_methods}"
                    )
                elif _allowed_new:
                    import logging as _log
                    _log.getLogger(__name__).info(
                        f"  ℹ️ Allowing new method(s) {_allowed_new} — semantically "
                        f"related to task description (feature task)"
                    )
            metrics["new_methods_added"] = list(truly_new)

            # Methods that exist in both but were MODIFIED (changed lines around them)
            # We detect this by checking if lines near those methods changed
            # This is a lightweight check — full AST diff is P2
            metrics["allowed_methods"] = task.allowed_methods

            # ── CHECK 4b: Method deletion detection ───────────────────────
            # The mirror of CHECK 4: detect methods present in original that
            # are ABSENT from generated output — direct evidence of deletion.
            deleted = original_methods - new_methods
            if deleted:
                violations.append(
                    f"METHOD DELETION DETECTED: {sorted(deleted)} exist in "
                    f"the original but are missing from generated output — "
                    f"all existing methods must be preserved verbatim"
                )
            metrics["deleted_methods"] = sorted(deleted)

        # ── CHECK 4c: Class field/property drop detection (TS/JS) ─────────────
        # Catches the "full-file amnesia" bug: a class member declared in the
        # original and still referenced via `this.X` in the rewrite, but whose
        # declaration was silently dropped — the root cause of TS2339 cascades.
        _lang = getattr(getattr(task, "language", None), "value", str(getattr(task, "language", ""))).lower()
        if (
            existing_content
            and task.task_type.value == "modify"
            and _lang in ("typescript", "javascript")
        ):
            import re as _re_fields

            def _declared_members(src: str) -> set:
                members: set = set()
                # Class field / method declarations at class-body indentation.
                for m in _re_fields.finditer(
                    r"^[ \t]+(?:(?:public|private|protected|readonly|static|abstract|async|override|get|set)\s+|@[\w.]+\([^)]*\)\s*)*"
                    r"([A-Za-z_$][\w$]*)\s*[:=(<]",
                    src, _re_fields.MULTILINE,
                ):
                    members.add(m.group(1))
                # Constructor parameter properties: constructor(private foo: Bar)
                for m in _re_fields.finditer(
                    r"(?:private|public|protected|readonly)\s+([A-Za-z_$][\w$]*)\s*:",
                    src,
                ):
                    members.add(m.group(1))
                return members

            _RESERVED_MEMBERS = {
                "constructor", "if", "for", "while", "switch", "return", "get", "set",
                "public", "private", "protected", "static", "readonly", "async",
            }
            declared_old = _declared_members(existing_content) - _RESERVED_MEMBERS
            declared_new = _declared_members(generated.content) - _RESERVED_MEMBERS
            referenced_new = set(_re_fields.findall(r"this\.([A-Za-z_$][\w$]*)", generated.content))

            dropped_used = sorted(
                x for x in referenced_new
                if x in declared_old and x not in declared_new
            )
            if dropped_used:
                # Attempt deterministic auto-repair before failing: restore declaration from original class
                repaired_content = generated.content
                repaired_declarations = []
                for dropped_var in dropped_used:
                    decl_m = _re_fields.search(
                        rf"^[ \t]*(?:(?:public|private|protected|readonly|static)\s+)*{re.escape(dropped_var)}\b[^;\n]*;",
                        existing_content,
                        _re_fields.MULTILINE
                    )
                    if decl_m:
                        repaired_declarations.append(decl_m.group(0).strip())
                    else:
                        repaired_declarations.append(f"{dropped_var}: any;")

                class_decl_m = _re_fields.search(r"class\s+\w+[^{]*\{", repaired_content)
                if class_decl_m and repaired_declarations:
                    insert_pos = class_decl_m.end()
                    insertion = "\n  " + "\n  ".join(repaired_declarations) + "\n"
                    repaired_content = repaired_content[:insert_pos] + insertion + repaired_content[insert_pos:]
                    generated.content = repaired_content
                    logger.info(
                        f"  🩹 Auto-repaired dropped members {dropped_used} in {getattr(task, 'file_path', 'TS file')}"
                    )
                else:
                    violations.append(
                        f"MEMBER DROPPED BUT STILL USED: {dropped_used} were declared in the "
                        f"original class and are still referenced via 'this.', but their "
                        f"declaration is missing from the generated output — re-add them."
                    )
            metrics["dropped_members_used"] = dropped_used

        # ── CHECK 5: Ownership-aware change budget ────────────────────────
        # Phase 2 ownership type governs how aggressively the generator may
        # change a file.  READ_ONLY files must never be written; companion
        # files (DISPLAY_OWNER / SUPPORTING) have a tighter change budget.
        ownership_type = getattr(task, "ownership_type", None)
        task_type_val = str(getattr(getattr(task, "task_type", None), "value", getattr(task, "task_type", ""))).lower()
        if ownership_type == "READ_ONLY" and task_type_val in ("modify", "create"):
            # Planner explicitly scheduled a writable task for this file — upgrade to SUPPORTING
            ownership_type = "SUPPORTING"

        if ownership_type == "READ_ONLY" and existing_content:
            violations.append(
                f"OWNERSHIP VIOLATION: '{task.file_path}' is classified "
                f"READ_ONLY — no modifications are permitted"
            )
        elif ownership_type in ("SUPPORTING", "DISPLAY_OWNER") and existing_content:
            _MAX_COMPANION_RATIO = 0.40   # tighter than PRIMARY (0.80)
            cr = metrics.get("change_ratio", 0.0)
            orig_len = metrics.get("original_line_count", 0)
            is_primary = getattr(task, "is_primary", False) or getattr(task, "ownership_type", "") == "PRIMARY_OWNER"
            is_ui = any(task.file_path.lower().endswith(ext) for ext in (".tsx", ".jsx", ".html", ".vue", ".css", ".scss"))
            effective_limit = self.MAX_CHANGE_RATIO if (orig_len < 120 or is_primary or is_ui) else _MAX_COMPANION_RATIO
            if orig_len >= 25 and not is_primary and not is_ui and cr > effective_limit:
                violations.append(
                    f"COMPANION CHANGE BUDGET EXCEEDED: {cr:.0%} of "
                    f"{ownership_type} file changed "
                    f"(limit {effective_limit:.0%} for non-primary files)"
                )
        metrics["ownership_type"] = ownership_type

        # ── CHECK 6: Template / Consumer Contract Validation ────────────────
        if normalized_gen.endswith((".component.html", ".html")):
            try:
                from ticket_to_code.intelligence.contracts.companion_resolver import PolyglotCompanionResolver
                from ticket_to_code.intelligence.contracts import ComponentContract, TemplateContractValidator
                _session = getattr(self, "_session_files", None) or {}
                _ws = getattr(self, "workspace_path", None) or getattr(self, "_workspace_path", None)
                _companion = PolyglotCompanionResolver.resolve_companion(
                    generated.file_path, session_files=_session, workspace_path=str(_ws) if _ws else None
                )
                if _companion:
                    _comp_path, _comp_text = _companion
                    if _comp_path.endswith((".ts", ".tsx")):
                        c_contract = ComponentContract.extract_from_ts(_comp_text, file_path=_comp_path)
                        t_violations = TemplateContractValidator.validate(generated.content, c_contract, original_content=existing_content)
                        if t_violations:
                            violations.extend([f"TEMPLATE CONTRACT VIOLATION: {tv}" for tv in t_violations])
            except Exception as _tc_err:
                logger.debug(f"Template contract validation in PatchValidator failed: {_tc_err}")

        # ── CHECK 7: TypeScript Type Contract Validation ───────────────────
        if normalized_gen.endswith((".ts", ".tsx")):
            import re as _pv_re
            if _pv_re.search(r"\bemail\s*:\s*\{", generated.content):
                violations.append(
                    "TYPE CONTRACT VIOLATION: Field 'email' in filter is defined as 'FilterStringInput[]' (ARRAY). "
                    "Passing scalar object '{ eq: ... }' causes TS2322. Must be wrapped in array: '[{ eq: ... }]'."
                )

        # ── CHECK 8: Syntax / AST Integrity Validation ─────────────────────
        if existing_content and generated.content:
            _ast_issues = _validate_patch_ast(existing_content, generated.content, task.file_path)
            _fatal_ast = [iss for iss in _ast_issues if "unbalanced delimiters" in iss or "Method count dropped" in iss]
            if _fatal_ast:
                violations.extend([f"SYNTAX / AST INTEGRITY VIOLATION: {iss}" for iss in _fatal_ast])

        passed = len(violations) == 0

        if violations:
            logger.warning(f"⚠️  PATCH VALIDATION FAILED for {task.file_path}:")
            for v in violations:
                logger.warning(f"    • {v}")
        else:
            logger.info(f"✅ Patch validation passed for {task.file_path} | metrics={metrics}")

        return PatchValidationResult(passed=passed, violations=violations, metrics=metrics)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_code(
    task: DevelopmentTask,
    requirements: StructuredRequirements,
    context: List[CodeChunk] = None,
    existing_content: str = None,
    allowed_files: List[str] = None,
    readonly_files: List[str] = None,
    dataflow_contract: Any = None,
    **kwargs: Any,
) -> GeneratedCode:
    """
    Convenience function to generate code.
    
    Args:
        task: Development task
        requirements: Structured requirements
        context: Code context
        
    Returns:
        Generated code
    """
    agent = CodeGeneratorAgent()
    return agent.generate_code(
        task=task,
        requirements=requirements,
        context=context,
        existing_content=existing_content,
        allowed_files=allowed_files,
        readonly_files=readonly_files,
        dataflow_contract=dataflow_contract,
        **kwargs,
    )
