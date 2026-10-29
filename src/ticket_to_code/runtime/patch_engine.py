"""
Standalone Patch Engine & Applicator for Ticket-to-Code Chatbot.

Replaces aviator_core.parsers.enhanced_patch_engine and
aviator_core.patch_applicator with lightweight implementations.

Zero dependency on aviator_core.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Transformation model — same interface as aviator_core's Transformation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Transformation:
    """Describes a single code transformation to apply."""
    file_path: str = ""
    description: str = ""
    original_code: str = ""
    new_code: str = ""
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    method_name: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# EnhancedPatchEngine — replaces aviator_core.parsers.enhanced_patch_engine
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedPatchEngine:
    """
    Standalone patch engine that can generate and apply code transformations.

    Drop-in replacement for aviator_core.parsers.enhanced_patch_engine.EnhancedPatchEngine.
    """

    def __init__(self, repo_path: str | Path = "", **kwargs):
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()

    def generate_diff(
        self,
        file_path: str,
        original: str,
        modified: str,
    ) -> str:
        """Generate a unified diff between original and modified content."""
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        )
        return "".join(diff)

    def parse(self, diff_text: str) -> List[Dict[str, Any]]:
        """Parse unified diff text into structured hunks."""
        hunks = []
        current_hunk = None
        for line in diff_text.splitlines():
            if line.startswith("@@"):
                if current_hunk:
                    hunks.append(current_hunk)
                current_hunk = {"header": line, "lines": []}
            elif current_hunk is not None:
                current_hunk["lines"].append(line)
        if current_hunk:
            hunks.append(current_hunk)
        return hunks if hunks else [{"header": "@@", "lines": diff_text.splitlines()}]

    def apply_transformation(
        self,
        transformation: Transformation,
    ) -> Tuple[bool, str]:
        """
        Apply a single transformation to a file.

        Returns:
            (success, message) tuple.
        """
        target = self.repo_path / transformation.file_path
        if not target.exists():
            return False, f"File not found: {target}"

        try:
            content = target.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines(keepends=True)

            if transformation.original_code and transformation.new_code:
                # Search-and-replace mode
                if transformation.original_code in content:
                    new_content = content.replace(
                        transformation.original_code,
                        transformation.new_code,
                        1,  # Replace only first occurrence
                    )
                    target.write_text(new_content, encoding="utf-8")
                    return True, f"Applied transformation to {transformation.file_path}"
                else:
                    return False, f"Original code not found in {transformation.file_path}"

            elif transformation.start_line and transformation.end_line:
                # Line-range replacement mode
                start = max(0, transformation.start_line - 1)
                end = min(len(lines), transformation.end_line)
                new_lines = transformation.new_code.splitlines(keepends=True)
                lines[start:end] = new_lines
                target.write_text("".join(lines), encoding="utf-8")
                return True, f"Applied line-range transformation to {transformation.file_path}"

            else:
                return False, "Transformation has neither original_code nor line range"

        except Exception as exc:
            return False, f"Error applying transformation: {exc}"

    def apply_transformations(
        self,
        transformations: List[Transformation],
    ) -> Dict[str, Any]:
        """Apply a batch of transformations and return summary."""
        results = {"success": 0, "failed": 0, "errors": []}
        for t in transformations:
            ok, msg = self.apply_transformation(t)
            if ok:
                results["success"] += 1
            else:
                results["failed"] += 1
                results["errors"].append(msg)
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# ContextExpander — replaces aviator_core.context_expander
# ═══════════════════════════════════════════════════════════════════════════════

class ContextExpander:
    """
    Standalone context expander for enriching file context.

    Reads file contents, dependencies, and knowledge base documents
    to build a rich context package for LLM consumption.
    """

    def __init__(
        self,
        repo_path: str | Path,
        knowledge_base_path: Optional[str] = None,
        **kwargs,
    ):
        self.repo_path = Path(repo_path)
        self.knowledge_base_path = Path(knowledge_base_path) if knowledge_base_path else None

    def expand_context(
        self,
        file_paths: List[str],
        method_names: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Build expanded context for the given files.

        Returns a dict with file contents, import dependencies, and
        relevant knowledge base documents.
        """
        context = {
            "files": {},
            "dependencies": [],
            "knowledge": [],
        }

        for fp in file_paths:
            full_path = self.repo_path / fp if not Path(fp).is_absolute() else Path(fp)
            if full_path.exists():
                try:
                    content = full_path.read_text(encoding="utf-8", errors="replace")
                    context["files"][fp] = {
                        "content": content,
                        "lines": len(content.splitlines()),
                    }
                    # Extract imports
                    deps = self._extract_imports(content, fp)
                    context["dependencies"].extend(deps)
                except Exception as exc:
                    logger.warning("Cannot read %s: %s", fp, exc)

        # Load knowledge base docs if available
        if self.knowledge_base_path and self.knowledge_base_path.exists():
            for kb_file in self.knowledge_base_path.rglob("*.md"):
                try:
                    kb_content = kb_file.read_text(encoding="utf-8", errors="replace")
                    context["knowledge"].append({
                        "path": str(kb_file),
                        "content": kb_content[:2000],  # Truncate to save tokens
                    })
                except Exception:
                    pass

        return context

    @staticmethod
    def _extract_imports(content: str, file_path: str) -> List[str]:
        """Extract import statements from code."""
        imports = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                imports.append(stripped)
            elif stripped.startswith("using "):
                imports.append(stripped)
            elif re.match(r'^import\s+', stripped):
                imports.append(stripped)
        return imports


# ═══════════════════════════════════════════════════════════════════════════════
# PatchApplicator — replaces aviator_core.patch_applicator
# ═══════════════════════════════════════════════════════════════════════════════

class PatchApplicator:
    """
    Standalone patch applicator for applying code patches to files.

    Drop-in replacement for aviator_core.patch_applicator.PatchApplicator.
    """

    def __init__(self, repo_path: str | Path, **kwargs):
        self.repo_path = Path(repo_path)

    def apply_patch(
        self,
        file_path: str,
        patch_content: Optional[str] = None,
        method_name: Optional[str] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Apply a patch to a file. Supports unified diff string or (file_path, content).
        """
        if patch_content is None:
            # Single argument: diff text
            diff_text = file_path
            match = re.search(r'^\+\+\+ [b/]*(.*?)$', diff_text, re.MULTILINE)
            if not match:
                match = re.search(r'^--- [a/]*(.*?)$', diff_text, re.MULTILINE)
            if not match:
                return False, "Could not identify target file from diff"
            resolved_path = match.group(1).strip()
            if resolved_path.startswith("b/") or resolved_path.startswith("a/"):
                resolved_path = resolved_path[2:]
            target = self.repo_path / resolved_path
            patch_content = diff_text
            file_path = resolved_path
        else:
            target = self.repo_path / file_path

        if not target.exists():
            # New file
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(patch_content, encoding="utf-8")
            return True, f"Created new file: {file_path}"

        try:
            original = target.read_text(encoding="utf-8", errors="replace")

            # If patch looks like a unified diff, try to apply it
            if patch_content.startswith("---") or patch_content.startswith("diff --git"):
                return self._apply_unified_diff(target, original, patch_content)

            # Line-range replacement
            if start_line and end_line:
                lines = original.splitlines(keepends=True)
                new_lines = patch_content.splitlines(keepends=True)
                start = max(0, start_line - 1)
                end = min(len(lines), end_line)
                lines[start:end] = new_lines
                target.write_text("".join(lines), encoding="utf-8")
                return True, f"Applied line-range patch to {file_path}"

            # Full content replacement
            target.write_text(patch_content, encoding="utf-8")
            return True, f"Applied full patch to {file_path}"

        except Exception as exc:
            return False, f"Patch application failed: {exc}"

    def _apply_unified_diff(
        self, target: Path, original: str, diff_text: str
    ) -> Tuple[bool, str]:
        """Attempt to apply a unified diff."""
        try:
            # Simple line-by-line diff application
            orig_lines = original.splitlines(keepends=True)
            result_lines = list(orig_lines)
            offset = 0

            for line in diff_text.splitlines():
                if line.startswith("@@"):
                    # Parse hunk header: @@ -start,count +start,count @@
                    match = re.match(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
                    if match:
                        offset = int(match.group(1)) - 1

            # If we couldn't parse the diff properly, fall back to full replacement
            # Extract just the '+' lines as new content
            new_lines = []
            in_diff = False
            for line in diff_text.splitlines(keepends=True):
                if line.startswith("+++") or line.startswith("---"):
                    in_diff = True
                    continue
                if in_diff:
                    if line.startswith("+"):
                        new_lines.append(line[1:])
                    elif line.startswith(" "):
                        new_lines.append(line[1:])
                    # Skip '-' lines (removed)

            if new_lines:
                target.write_text("".join(new_lines), encoding="utf-8")
                return True, f"Applied unified diff to {target.name}"

            return False, "Could not parse unified diff"

        except Exception as exc:
            return False, f"Diff application error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════════
# Compatibility Aliases
# ═══════════════════════════════════════════════════════════════════════════════

EnhancedPatchApplicator = PatchApplicator
from ticket_to_code.runtime.localizer import FileCandidate
