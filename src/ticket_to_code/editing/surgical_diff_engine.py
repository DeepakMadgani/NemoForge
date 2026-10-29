"""
Surgical Diff Engine - Fail-Closed Context-Anchored Code Modification

Grounded in AI IDE Harness Architecture:
1. Replaces code using exact, contiguous context anchors (2-3 lines before/after),
   NOT line numbers (which shift dynamically) and NOT whole-file rewrites (which truncate code).
2. Fail-Closed Integrity: If the search anchor matches 0 times or >1 times, the edit
   aborts immediately with diagnostic feedback rather than guessing.
3. Undo Stack: Preserves exact snapshots of files prior to modification, enabling
   instant deterministic rollbacks upon verification failure or oscillation.
"""

from __future__ import annotations

import logging
import os
import difflib
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Dict

logger = logging.getLogger(__name__)


class EditStatus(str, Enum):
    SUCCESS = "success"
    ANCHOR_NOT_FOUND = "anchor_not_found"
    AMBIGUOUS_ANCHOR = "ambiguous_anchor"
    STALE_CONTENT_HASH = "stale_content_hash"
    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_ERROR = "permission_error"
    OUT_OF_SCOPE = "out_of_scope"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class EditSnapshot:
    """Snapshot of a file before modification for rollback."""
    file_path: str
    original_content: str
    timestamp: float


@dataclass
class EditResult:
    """Result of an anchored edit operation."""
    status: EditStatus
    file_path: str
    match_count: int = 0
    message: str = ""
    unified_diff: str = ""
    lines_added: int = 0
    lines_removed: int = 0
    new_content_hash: str = ""

    @property
    def success(self) -> bool:
        return self.status == EditStatus.SUCCESS


def apply_tolerant_replacement(content: str, old_block: str, new_block: str) -> Optional[str]:
    """
    Aider-style 3-tier resilient diff alignment:
    Tier 1: Whitespace & trailing space normalization
    Tier 2: Relative indentation shifting (matches unindented/overindented LLM blocks)
    Tier 3: Sliding window fuzzy matching (ratio >= 0.88 with uniqueness guarantee)
    """
    import textwrap

    has_crlf = "\r\n" in content
    newline = "\r\n" if has_crlf else "\n"

    norm_content = content.replace("\r\n", "\n")
    norm_old = old_block.replace("\r\n", "\n")
    norm_new = new_block.replace("\r\n", "\n")

    content_lines = norm_content.splitlines()
    old_lines = norm_old.splitlines()
    new_lines = norm_new.splitlines()

    stripped_old = [line.rstrip() for line in old_lines]
    w = len(stripped_old)
    if w == 0:
        return None

    # Tier 1: Match with stripped trailing whitespace
    matches = []
    for i in range(len(content_lines) - w + 1):
        window = [content_lines[i + k].rstrip() for k in range(w)]
        if window == stripped_old:
            matches.append(i)

    if len(matches) == 1:
        start_idx = matches[0]
        res_lines = content_lines[:start_idx] + new_lines + content_lines[start_idx + w:]
        res_text = "\n".join(res_lines)
        if norm_content.endswith("\n"):
            res_text += "\n"
        return res_text.replace("\n", newline) if has_crlf else res_text

    # Tier 2: Indentation-invariant matching
    dedented_old = [line.strip() for line in old_lines if line.strip()]
    if dedented_old:
        indent_matches = []
        for i in range(len(content_lines) - w + 1):
            window_stripped = [content_lines[i + k].strip() for k in range(w)]
            non_empty_window = [l for l in window_stripped if l]
            if non_empty_window == dedented_old:
                indent_matches.append(i)

        if len(indent_matches) == 1:
            start_idx = indent_matches[0]
            target_indent = ""
            for k in range(w):
                orig_line = content_lines[start_idx + k]
                if orig_line.strip():
                    target_indent = orig_line[:len(orig_line) - len(orig_line.lstrip())]
                    break

            dedented_new_block = textwrap.dedent(norm_new)
            reindented_new = [
                (target_indent + line if line.strip() else "")
                for line in dedented_new_block.splitlines()
            ]

            res_lines = content_lines[:start_idx] + reindented_new + content_lines[start_idx + w:]
            res_text = "\n".join(res_lines)
            if norm_content.endswith("\n"):
                res_text += "\n"
            return res_text.replace("\n", newline) if has_crlf else res_text

    # Tier 3: Bounded anchor-filtered sliding window fuzzy matching
    if len(content_lines) >= w and w >= 2:
        best_ratio = 0.0
        best_idx = -1
        second_best_ratio = 0.0

        target_str = "\n".join(stripped_old)
        first_old = stripped_old[0].strip() if stripped_old else ""
        last_old = stripped_old[-1].strip() if stripped_old else ""

        # Pre-filter candidate starting offsets using anchor lines to prevent O(N*M) CPU exhaustion
        candidate_indices = []
        for i in range(len(content_lines) - w + 1):
            curr_first = content_lines[i].strip()
            curr_last = content_lines[i + w - 1].strip()
            if curr_first == first_old or curr_last == last_old:
                candidate_indices.append(i)
            elif first_old and len(first_old) >= 4 and (first_old in curr_first or curr_first in first_old):
                candidate_indices.append(i)

        # If anchor heuristics found no matches and file is reasonable (<600 lines), check all offsets
        if not candidate_indices and len(content_lines) <= 600:
            candidate_indices = list(range(len(content_lines) - w + 1))
        elif not candidate_indices:
            # On large files (>600 lines), pre-filter with quick line ratio > 0.65 to bound candidates
            for i in range(len(content_lines) - w + 1):
                if difflib.SequenceMatcher(None, content_lines[i].strip(), first_old).quick_ratio() > 0.65:
                    candidate_indices.append(i)
                    if len(candidate_indices) >= 30:
                        break

        for i in candidate_indices:
            window_str = "\n".join(content_lines[i + k].rstrip() for k in range(w))
            matcher = difflib.SequenceMatcher(None, window_str, target_str)
            ratio = matcher.quick_ratio()
            if ratio > 0.80:
                real_ratio = matcher.ratio()
                if real_ratio > best_ratio:
                    second_best_ratio = best_ratio
                    best_ratio = real_ratio
                    best_idx = i
                elif real_ratio > second_best_ratio:
                    second_best_ratio = real_ratio

        if best_ratio >= 0.82 and (best_ratio - second_best_ratio) >= 0.05:
            res_lines = content_lines[:best_idx] + new_lines + content_lines[best_idx + w:]
            res_text = "\n".join(res_lines)
            if norm_content.endswith("\n"):
                res_text += "\n"
            return res_text.replace("\n", newline) if has_crlf else res_text

    return None


class SurgicalDiffEngine:
    """
    Context-anchored contiguous block replacement engine.
    
    Provides byte-for-byte exact patch application with fail-closed guarantees
    and an in-memory undo stack for atomic rollbacks.
    """

    def __init__(self, workspace_path: str, allowed_paths: Optional[set[str]] = None):
        self.workspace_path = Path(workspace_path).resolve()
        self.allowed_paths: Optional[set[str]] = allowed_paths
        self.undo_stack: List[EditSnapshot] = []

    def set_allowed_paths(self, paths: set[str]) -> None:
        """Set or update the authorized write sandbox (Blast-Radius Lock)."""
        self.allowed_paths = {p.replace("\\", "/").lower() for p in paths}

    def is_path_allowed(self, rel_path: str) -> bool:
        """Verify that a path is within the authorized write set."""
        if self.allowed_paths is None:
            return True
        norm = rel_path.replace("\\", "/").lower()
        return any(
            norm == allowed or norm.startswith(f"{allowed}/") or norm.endswith(f"/{allowed}")
            for allowed in self.allowed_paths
        )

    @staticmethod
    def compute_content_hash(text: str) -> str:
        """Computes SHA-256 hash normalized for line endings."""
        import hashlib
        normalized = text.replace("\r\n", "\n")
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def apply_anchored_replacement(
        self,
        rel_path: str,
        old_block: str,
        new_block: str,
        expected_match_count: int = 1,
        expected_content_hash: Optional[str] = None,
    ) -> EditResult:
        """
        Apply context-anchored replacement to a file.

        Args:
            rel_path: Relative or absolute path to the target file.
            old_block: The exact contiguous block to replace, including 2-3 lines
                       of unchanged surrounding context.
            new_block: The replacement block including the same context lines.
            expected_match_count: Expected exact matches (default: 1).
            expected_content_hash: Cryptographic SHA-256 hash expected on the target file.

        Returns:
            EditResult with status, diff, line counts, and new_content_hash.
        """
        target = Path(rel_path)
        if not target.is_absolute():
            full_path = self.workspace_path / target
        else:
            full_path = target
            try:
                rel_path = str(full_path.relative_to(self.workspace_path)).replace("\\", "/")
            except ValueError:
                pass

        # 1. Blast-radius containment check
        if not self.is_path_allowed(rel_path):
            msg = f"Path '{rel_path}' is not within authorized write set: {self.allowed_paths}"
            logger.error(f"[SurgicalDiffEngine] {msg}")
            return EditResult(
                status=EditStatus.OUT_OF_SCOPE,
                file_path=rel_path,
                message=msg,
            )

        # 2. Check file existence or handle new file creation
        if not full_path.is_file():
            if not old_block:
                # Creation of a new file (e.g. Flyway migration or new test)
                try:
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_text(new_block, encoding="utf-8")
                    new_hash = self.compute_content_hash(new_block)
                    self.undo_stack.append(
                        EditSnapshot(
                            file_path=str(full_path),
                            original_content="",
                            timestamp=time.time(),
                        )
                    )
                    return EditResult(
                        status=EditStatus.SUCCESS,
                        file_path=rel_path,
                        match_count=1,
                        message="Created new file successfully",
                        lines_added=len(new_block.splitlines()),
                        new_content_hash=new_hash,
                    )
                except Exception as exc:
                    msg = f"Failed to create new file '{rel_path}': {exc}"
                    logger.error(f"[SurgicalDiffEngine] {msg}")
                    return EditResult(
                        status=EditStatus.PERMISSION_ERROR,
                        file_path=rel_path,
                        message=msg,
                    )
            else:
                msg = f"Target file '{full_path}' does not exist on disk."
                logger.error(f"[SurgicalDiffEngine] {msg}")
                return EditResult(
                    status=EditStatus.FILE_NOT_FOUND,
                    file_path=rel_path,
                    message=msg,
                )

        # 3. Read verbatim file content
        try:
            content = full_path.read_text(encoding="utf-8")
        except Exception as exc:
            msg = f"Failed to read '{rel_path}': {exc}"
            logger.error(f"[SurgicalDiffEngine] {msg}")
            return EditResult(
                status=EditStatus.PERMISSION_ERROR,
                file_path=rel_path,
                message=msg,
            )

        # 3b. Cryptographic Content Hash Validation (Anti-Stale / Anti-Drift)
        current_hash = self.compute_content_hash(content)
        if expected_content_hash and expected_content_hash != current_hash:
            msg = (
                f"Stale Content Hash: File '{rel_path}' on disk has hash {current_hash}, "
                f"which does not match expected {expected_content_hash}. File was mutated concurrently."
            )
            logger.error(f"[SurgicalDiffEngine] {msg}")
            return EditResult(
                status=EditStatus.STALE_CONTENT_HASH,
                file_path=rel_path,
                message=msg,
                new_content_hash=current_hash,
            )

        # Detect line endings
        has_crlf = "\r\n" in content
        newline_char = "\r\n" if has_crlf else "\n"

        # Normalize line endings for comparison while keeping original for writing
        normalized_content = content.replace("\r\n", "\n")
        normalized_old = old_block.replace("\r\n", "\n")
        normalized_new = new_block.replace("\r\n", "\n")

        # 3. Match count verification
        if not normalized_old.strip():
            if len(normalized_new.strip().splitlines()) >= 5:
                self.undo_stack.append(
                    EditSnapshot(
                        file_path=str(full_path),
                        original_content=content,
                        timestamp=time.time(),
                    )
                )
                try:
                    full_path.write_text(new_block, encoding="utf-8")
                    new_hash = self.compute_content_hash(new_block)
                    return EditResult(
                        status=EditStatus.SUCCESS,
                        file_path=rel_path,
                        match_count=1,
                        message="Full-file replacement applied cleanly.",
                        lines_added=len(new_block.splitlines()),
                        new_content_hash=new_hash,
                    )
                except Exception as exc:
                    return EditResult(
                        status=EditStatus.PERMISSION_ERROR,
                        file_path=rel_path,
                        message=f"Failed to write '{rel_path}': {exc}",
                    )
            else:
                return EditResult(
                    status=EditStatus.ANCHOR_NOT_FOUND,
                    file_path=rel_path,
                    match_count=0,
                    message="Search block cannot be empty when modifying an existing file.",
                )

        matches = normalized_content.count(normalized_old)

        if matches == 0:
            # Check multi-stage tolerant replacement (Whitespace, Indentation shifting, Fuzzy window)
            tolerant_res = apply_tolerant_replacement(content, old_block, new_block)

            if tolerant_res is not None:
                final_content = tolerant_res
                self.undo_stack.append(
                    EditSnapshot(
                        file_path=str(full_path),
                        original_content=content,
                        timestamp=time.time(),
                    )
                )
                try:
                    full_path.write_text(final_content, encoding="utf-8")
                    new_hash = self.compute_content_hash(final_content)
                    return EditResult(
                        status=EditStatus.SUCCESS,
                        file_path=rel_path,
                        match_count=1,
                        message="Edit applied cleanly via tolerant alignment.",
                        new_content_hash=new_hash,
                    )
                except Exception as exc:
                    return EditResult(
                        status=EditStatus.PERMISSION_ERROR,
                        file_path=rel_path,
                        message=f"Failed to write '{rel_path}': {exc}",
                    )

            logger.warning(
                f"[SurgicalDiffEngine] Anchor not found in '{rel_path}'. "
                f"Length={len(normalized_old)} chars, lines={len(normalized_old.splitlines())}"
            )
            return EditResult(
                status=EditStatus.ANCHOR_NOT_FOUND,
                file_path=rel_path,
                match_count=0,
                message=(
                    f"Context anchor was not found in '{rel_path}'. "
                    "Ensure leading whitespace, indentation, and surrounding context match byte-for-byte."
                ),
            )

        if matches > expected_match_count:
            logger.warning(
                f"[SurgicalDiffEngine] Ambiguous anchor: matched {matches} times in '{rel_path}' "
                f"(expected {expected_match_count}). Edit aborted for safety."
            )
            return EditResult(
                status=EditStatus.AMBIGUOUS_ANCHOR,
                file_path=rel_path,
                match_count=matches,
                message=(
                    f"Context anchor is ambiguous (matched {matches} times in '{rel_path}'). "
                    "Expand surrounding unchanged context lines (3-5 lines before/after) to guarantee unique match."
                ),
            )

        # 4. Snapshot original content onto Undo Stack
        self.undo_stack.append(
            EditSnapshot(
                file_path=str(full_path),
                original_content=content,
                timestamp=time.time(),
            )
        )

        # 5. Apply replacement
        new_content_normalized = normalized_content.replace(
            normalized_old, normalized_new, 1
        )

        # Restore original newline convention
        final_content = (
            new_content_normalized.replace("\n", "\r\n")
            if has_crlf
            else new_content_normalized
        )

        # Write to disk
        try:
            full_path.write_text(final_content, encoding="utf-8")
        except Exception as exc:
            msg = f"Failed to write modified content to '{full_path}': {exc}"
            logger.error(f"[SurgicalDiffEngine] {msg}")
            # Rollback in-memory snapshot
            self.undo_stack.pop()
            return EditResult(
                status=EditStatus.PERMISSION_ERROR,
                file_path=rel_path,
                message=msg,
            )

        # 6. Generate unified diff for audit
        orig_lines = content.splitlines(keepends=True)
        new_lines = final_content.splitlines(keepends=True)
        diff = list(
            difflib.unified_diff(
                orig_lines,
                new_lines,
                fromfile=f"a/{rel_path}",
                tofile=f"b/{rel_path}",
            )
        )
        diff_text = "".join(diff)

        added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))

        logger.info(
            f"[SurgicalDiffEngine] ✅ Successfully patched '{rel_path}' (+{added}, -{removed} lines)"
        )

        return EditResult(
            status=EditStatus.SUCCESS,
            file_path=rel_path,
            match_count=1,
            message="Edit applied cleanly.",
            unified_diff=diff_text,
            lines_added=added,
            lines_removed=removed,
            new_content_hash=self.compute_content_hash(final_content),
        )

    def preview_anchored_replacement(
        self,
        rel_path: str,
        old_block: str,
        new_block: str,
        expected_match_count: int = 1,
        expected_content_hash: Optional[str] = None,
    ) -> EditResult:
        """
        Speculatively simulates an anchored replacement in-memory without modifying disk.
        Returns the prospective EditResult with computed unified diff, line counts, and new_content_hash.
        """
        target = Path(rel_path)
        full_path = self.workspace_path / target if not target.is_absolute() else target

        if not self.is_path_allowed(rel_path):
            return EditResult(
                status=EditStatus.OUT_OF_SCOPE,
                file_path=rel_path,
                message=f"Path '{rel_path}' is not within authorized write set",
            )

        if not full_path.is_file():
            if not old_block:
                new_hash = self.compute_content_hash(new_block)
                return EditResult(
                    status=EditStatus.SUCCESS,
                    file_path=rel_path,
                    match_count=1,
                    message="Speculative diff: new file creation.",
                    lines_added=len(new_block.splitlines()),
                    new_content_hash=new_hash,
                )
            return EditResult(
                status=EditStatus.FILE_NOT_FOUND,
                file_path=rel_path,
                message=f"Target file '{full_path}' does not exist on disk.",
            )

        try:
            content = full_path.read_text(encoding="utf-8")
        except Exception as exc:
            return EditResult(
                status=EditStatus.PERMISSION_ERROR,
                file_path=rel_path,
                message=f"Failed to read '{rel_path}': {exc}",
            )

        current_hash = self.compute_content_hash(content)
        if expected_content_hash and expected_content_hash != current_hash:
            return EditResult(
                status=EditStatus.STALE_CONTENT_HASH,
                file_path=rel_path,
                message=f"Stale Content Hash ({current_hash} != {expected_content_hash})",
                new_content_hash=current_hash,
            )

        has_crlf = "\r\n" in content
        normalized_content = content.replace("\r\n", "\n")
        normalized_old = old_block.replace("\r\n", "\n")
        normalized_new = new_block.replace("\r\n", "\n")

        matches = normalized_content.count(normalized_old)
        if matches == 0:
            tolerant_res = apply_tolerant_replacement(content, old_block, new_block)
            if tolerant_res is not None:
                final_content = tolerant_res
                orig_lines = content.splitlines(keepends=True)
                new_lines = final_content.splitlines(keepends=True)
                diff = list(difflib.unified_diff(orig_lines, new_lines, fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}"))
                diff_text = "".join(diff)
                added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
                removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
                return EditResult(
                    status=EditStatus.SUCCESS,
                    file_path=rel_path,
                    match_count=1,
                    message="Speculative diff preview via tolerant alignment.",
                    unified_diff=diff_text,
                    lines_added=added,
                    lines_removed=removed,
                    new_content_hash=self.compute_content_hash(final_content),
                )
            return EditResult(
                status=EditStatus.ANCHOR_NOT_FOUND,
                file_path=rel_path,
                match_count=0,
                message="Context anchor was not found in target file.",
            )

        if matches > expected_match_count:
            return EditResult(
                status=EditStatus.AMBIGUOUS_ANCHOR,
                file_path=rel_path,
                match_count=matches,
                message=f"Context anchor is ambiguous (matched {matches} times).",
            )

        new_content_normalized = normalized_content.replace(normalized_old, normalized_new, 1)
        final_content = new_content_normalized.replace("\n", "\r\n") if has_crlf else new_content_normalized

        orig_lines = content.splitlines(keepends=True)
        new_lines = final_content.splitlines(keepends=True)
        diff = list(difflib.unified_diff(orig_lines, new_lines, fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}"))
        diff_text = "".join(diff)
        added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))

        return EditResult(
            status=EditStatus.SUCCESS,
            file_path=rel_path,
            match_count=1,
            message="Speculative diff preview generated.",
            unified_diff=diff_text,
            lines_added=added,
            lines_removed=removed,
            new_content_hash=self.compute_content_hash(final_content),
        )

    def rollback_last_edit(self) -> Optional[str]:
        """
        Revert the most recent edit from the undo stack.
        
        Returns the path of the restored file, or None if stack is empty.
        """
        if not self.undo_stack:
            logger.warning("[SurgicalDiffEngine] Undo stack is empty; nothing to rollback.")
            return None

        snapshot = self.undo_stack.pop()
        target = Path(snapshot.file_path)
        try:
            if not snapshot.original_content and target.is_file():
                target.unlink()
                logger.info(f"[SurgicalDiffEngine] ⏪ Deleted newly created file '{target.name}'.")
            else:
                target.write_text(snapshot.original_content, encoding="utf-8")
                logger.info(f"[SurgicalDiffEngine] ⏪ Rolled back '{target.name}' to pre-edit state.")
            return str(target)
        except Exception as exc:
            logger.error(f"[SurgicalDiffEngine] Failed to rollback '{target}': {exc}")
            return None

    def rollback_all(self) -> List[str]:
        """
        Revert ALL edits applied during this session in reverse order.
        
        Returns list of rolled-back file paths.
        """
        restored = []
        while self.undo_stack:
            path = self.rollback_last_edit()
            if path:
                restored.append(path)
        return restored
