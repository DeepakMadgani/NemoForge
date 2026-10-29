"""
Unified Hunk-Level Diff & Selective Review Engine.

Provides Cursor Composer-style selective diff inspection:
- Decomposes unified diffs into discrete, verifiable hunks.
- Enables per-hunk acceptance, rejection, or partial staging.
- Computes deterministic patch application and rollback without full-file clobbering.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple


@dataclass
class DiffHunk:
    hunk_index: int
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    header: str
    diff_lines: List[str]
    old_lines: List[str] = field(default_factory=list)
    new_lines: List[str] = field(default_factory=list)
    status: str = "pending"  # 'pending', 'accepted', 'rejected'


@dataclass
class FileDiff:
    file_path: str
    original_content: str
    modified_content: str
    hunks: List[DiffHunk] = field(default_factory=list)
    raw_diff: str = ""
    is_noop: bool = False


class DiffReviewEngine:
    """Computes unified diffs and manages hunk-level acceptance or rejection."""

    HUNK_HEADER_RE = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$')

    @classmethod
    def compute_file_diff(
        cls,
        file_path: str,
        original_content: str,
        modified_content: str,
    ) -> FileDiff:
        """Generates a structured FileDiff decomposed into individual hunks."""
        orig_lines = original_content.splitlines(keepends=True)
        mod_lines = modified_content.splitlines(keepends=True)

        if original_content == modified_content:
            return FileDiff(
                file_path=file_path,
                original_content=original_content,
                modified_content=modified_content,
                is_noop=True,
            )

        raw_diff_lines = list(
            difflib.unified_diff(
                orig_lines,
                mod_lines,
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                n=3,
            )
        )
        raw_diff_str = "".join(raw_diff_lines)

        # Parse raw unified diff into discrete hunks
        hunks: List[DiffHunk] = []
        curr_hunk_index = 0
        curr_hunk_lines: List[str] = []
        curr_old_lines: List[str] = []
        curr_new_lines: List[str] = []
        curr_header = ""
        old_start, old_count = 0, 0
        new_start, new_count = 0, 0

        for line in raw_diff_lines:
            header_match = cls.HUNK_HEADER_RE.match(line)
            if header_match:
                if curr_hunk_lines:
                    hunks.append(
                        DiffHunk(
                            hunk_index=curr_hunk_index,
                            old_start=old_start,
                            old_count=old_count,
                            new_start=new_start,
                            new_count=new_count,
                            header=curr_header,
                            diff_lines=list(curr_hunk_lines),
                            old_lines=list(curr_old_lines),
                            new_lines=list(curr_new_lines),
                        )
                    )
                    curr_hunk_index += 1
                    curr_hunk_lines.clear()
                    curr_old_lines.clear()
                    curr_new_lines.clear()

                curr_header = line.strip()
                old_start = int(header_match.group(1))
                old_count = int(header_match.group(2)) if header_match.group(2) else 1
                new_start = int(header_match.group(3))
                new_count = int(header_match.group(4)) if header_match.group(4) else 1
                curr_hunk_lines.append(line)
            elif curr_hunk_lines:
                curr_hunk_lines.append(line)
                if line.startswith("-"):
                    curr_old_lines.append(line[1:])
                elif line.startswith("+"):
                    curr_new_lines.append(line[1:])
                elif line.startswith(" "):
                    curr_old_lines.append(line[1:])
                    curr_new_lines.append(line[1:])

        if curr_hunk_lines:
            hunks.append(
                DiffHunk(
                    hunk_index=curr_hunk_index,
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    header=curr_header,
                    diff_lines=list(curr_hunk_lines),
                    old_lines=list(curr_old_lines),
                    new_lines=list(curr_new_lines),
                )
            )

        return FileDiff(
            file_path=file_path,
            original_content=original_content,
            modified_content=modified_content,
            hunks=hunks,
            raw_diff=raw_diff_str,
            is_noop=len(hunks) == 0,
        )

    @classmethod
    def apply_selected_hunks(
        cls,
        original_content: str,
        hunks: List[DiffHunk],
        accepted_indices: Set[int],
    ) -> str:
        """Selectively applies only accepted hunks to the original text, leaving rejected hunks untouched."""
        if not hunks or not accepted_indices:
            return original_content

        orig_lines = original_content.splitlines(keepends=True)
        # Sort hunks in reverse line order so modifications don't shift upcoming line offsets
        sorted_hunks = sorted(
            [h for h in hunks if h.hunk_index in accepted_indices],
            key=lambda h: h.old_start,
            reverse=True,
        )

        result_lines = list(orig_lines)
        for h in sorted_hunks:
            # 1-indexed to 0-indexed line slice
            start_idx = max(0, h.old_start - 1)
            end_idx = start_idx + h.old_count

            # Extract the new lines from this hunk
            new_chunk_lines = []
            for l in h.diff_lines[1:]:  # skip @@ header
                if l.startswith("+"):
                    new_chunk_lines.append(l[1:])
                elif l.startswith(" "):
                    new_chunk_lines.append(l[1:])

            result_lines[start_idx:end_idx] = new_chunk_lines

        return "".join(result_lines)
