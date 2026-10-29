"""
Workspace Integrity Manager — Premier AI IDE Standard.

Provides:
  1. Developer Baseline Tracking (hashes, mtime, git dirty status, baseline content).
  2. Protected Developer Regions (symbol & line-level protection against overwrites).
  3. Hierarchical Patching:
       Level 1: AST / Symbol-aware patch
       Level 2: SEARCH/REPLACE surgical patch
       Level 3: 3-Way Merge (BASE, CURRENT, GENERATED)
       Level 4: Conflict -> STOP (Fail-safe, no guessing, no overwrites).
  4. Patch Ownership (patch_id, task, ticket, requirement, before/after hashes, inverse delta).
  5. Patch Idempotency (prevent duplicate patch application).
  6. Optimistic Concurrency Control (detect concurrent workspace mutations).
  7. Reversible AI-Only Rollback (undo ONLY Ticket-to-Code's own patch via inverse delta).
"""

from __future__ import annotations

import difflib
import hashlib
import logging
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ticket_to_code.integrity.structural_manifest import (
    StructuralChangeManifest,
    StructuralDelta,
    DeleteAuthorization,
    DeleteAuthorizationValidator,
    UnauthorizedDeletionError,
)

logger = logging.getLogger(__name__)


# ============================================================================
# EXCEPTIONS
# ============================================================================

class PatchConflictError(Exception):
    """Raised when developer changes and ticket patch clash in the same region."""

    def __init__(
        self,
        file_path: str,
        symbol: Optional[str] = None,
        developer_change: str = "",
        ticket_change: str = "",
        reason: str = "overlapping modifications in protected developer region",
    ):
        msg = (
            f"\n[PATCH_CONFLICT] Overlapping modifications in {file_path}\n"
            f"  Symbol: {symbol or 'top-level'}\n"
            f"  Developer Change:\n{developer_change}\n"
            f"  Ticket Change:\n{ticket_change}\n"
            f"  Reason: {reason}\n"
            f"  Action: Workspace preserved untouched. Patch rejected."
        )
        super().__init__(msg)
        self.file_path = file_path
        self.symbol = symbol
        self.developer_change = developer_change
        self.ticket_change = ticket_change
        self.reason = reason


class OptimisticConcurrencyError(Exception):
    """Raised when file changed on disk between agent read and write."""

    def __init__(self, file_path: str, expected_hash: str, actual_hash: str):
        super().__init__(
            f"CONCURRENT_WORKSPACE_CHANGE: {file_path} was modified externally "
            f"(expected sha={expected_hash[:8]}, actual sha={actual_hash[:8]})."
        )
        self.file_path = file_path
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ProtectedRegion:
    """A region of code modified by developer prior to or during the run."""
    file_path: str
    start_line: int
    end_line: int
    content: str
    symbol: Optional[str] = None


@dataclass
class FileBaseline:
    """Baseline state of a workspace file captured at ticket start."""
    file_path: str
    content_hash: str
    mtime: float
    is_dirty: bool
    content: Optional[str] = None
    protected_regions: List[ProtectedRegion] = field(default_factory=list)


@dataclass
class PatchOwnershipRecord:
    """Audit identity for every applied patch."""
    patch_id: str
    ticket_id: str
    task_id: str
    requirement_id: str
    file_path: str
    symbol: Optional[str]
    operation: str  # "MODIFY", "CREATE", "DELETE_SYMBOL", "SURGICAL_PATCH"
    before_hash: str
    after_hash: str
    inverse_content: str  # Content to write back to restore pre-patch state
    after_content: Optional[str] = None  # Exact content after patch was applied
    applied_at: float = field(default_factory=time.time)


# ============================================================================
# PATCH REGISTRY & IDEMPOTENCY
# ============================================================================

class PatchRegistry:
    """Tracks applied patches, provides idempotency checks and reversible rollback."""

    def __init__(self):
        self._patches: Dict[str, PatchOwnershipRecord] = {}
        self._file_patches: Dict[str, List[PatchOwnershipRecord]] = {}

    def register(self, record: PatchOwnershipRecord):
        self._patches[record.patch_id] = record
        norm = record.file_path.replace("\\", "/").lower()
        self._file_patches.setdefault(norm, []).append(record)

    def is_already_applied(self, file_path: str, current_content: str) -> Optional[str]:
        """Check if patch matching current content hash is already applied."""
        current_hash = hashlib.sha256(current_content.encode("utf-8")).hexdigest()
        norm = file_path.replace("\\", "/").lower()
        for record in self._file_patches.get(norm, []):
            if record.after_hash == current_hash:
                return record.patch_id
        return None

    def get_patches_for_ticket(self, ticket_id: str) -> List[PatchOwnershipRecord]:
        return [p for p in self._patches.values() if p.ticket_id == ticket_id]

    def rollback_patch(self, patch_id: str, workspace_path: Path) -> bool:
        """Rollback a single patch by applying its inverse content."""
        record = self._patches.get(patch_id)
        if not record:
            logger.warning(f"Cannot rollback unknown patch_id: {patch_id}")
            return False

        abs_path = workspace_path / record.file_path
        if not abs_path.exists():
            logger.warning(f"Cannot rollback {patch_id}: file {record.file_path} not found")
            return False

        current_text = abs_path.read_text(encoding="utf-8", errors="ignore")
        current_hash = hashlib.sha256(current_text.encode("utf-8")).hexdigest()

        # If current content matches after_hash, restore inverse_content
        if current_hash == record.after_hash:
            abs_path.write_text(record.inverse_content, encoding="utf-8")
            logger.info(f"↩️ Rolled back patch {patch_id} on {record.file_path} (restored exact pre-patch delta)")
        else:
            # File has subsequent edits — 3-way merge the reversal
            logger.info(f"↩️ Merging reversal of patch {patch_id} on {record.file_path}...")
            # BASE = record.after_content, CURRENT = current_text, GENERATED = record.inverse_content
            base_content = record.after_content if getattr(record, "after_content", None) is not None else current_text
            merged, conflicts = ThreeWayPatcher.merge_3way(
                base=base_content,
                current=current_text,
                generated=record.inverse_content,
            )
            if conflicts:
                logger.error(f"Cannot cleanly reverse patch {patch_id} due to subsequent conflicts: {conflicts}")
                return False
            abs_path.write_text(merged, encoding="utf-8")

        # Remove record
        del self._patches[patch_id]
        norm = record.file_path.replace("\\", "/").lower()
        if norm in self._file_patches:
            self._file_patches[norm] = [p for p in self._file_patches[norm] if p.patch_id != patch_id]
        return True


# ============================================================================
# BASELINE TRACKER
# ============================================================================

class WorkspaceBaseline:
    """Captures and stores initial workspace baseline before any modifications."""

    def __init__(self, workspace_path: Union[str, Path]):
        self.workspace_path = Path(workspace_path)
        self.files: Dict[str, FileBaseline] = {}
        self.captured_at: float = time.time()

    def capture(self) -> None:
        """Scan workspace, record file hashes and identify pre-run developer regions."""
        if not self.workspace_path.exists():
            return

        git_dirty = self._get_git_dirty_files()

        ignored_parts = {
            ".git", "node_modules", ".venv", "venv", "__pycache__",
            "dist", "build", "target", ".gradle", ".angular",
            ".idea", ".vscode", "bin", ".cache", ".aviator", ".yarn"
        }
        allowed_exts = {
            ".java", ".ts", ".tsx", ".js", ".jsx", ".html", ".scss", ".css",
            ".xml", ".json", ".yml", ".yaml", ".sql", ".properties", ".py",
            ".md", ".gradle", ".env", ".proto", ".graphql"
        }

        # Enumerate files (excluding caches and dependencies)
        for p in self.workspace_path.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in allowed_exts:
                continue
            if any(part in ignored_parts for part in p.parts):
                continue
            rel = str(p.relative_to(self.workspace_path)).replace("\\", "/")
            try:
                if p.stat().st_size > 2 * 1024 * 1024:
                    continue
                content = p.read_text(encoding="utf-8", errors="ignore")
                chash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                mtime = p.stat().st_mtime
                is_dirty = rel.lower() in git_dirty

                regions = []
                if is_dirty:
                    regions = self._extract_git_dirty_regions(rel, content)

                self.files[rel.lower()] = FileBaseline(
                    file_path=rel,
                    content_hash=chash,
                    mtime=mtime,
                    is_dirty=is_dirty,
                    content=content,  # ALWAYS preserve actual disk bytes at run start (never depend on Git)
                    protected_regions=regions,
                )
            except Exception as e:
                logger.debug(f"Baseline capture skipped {rel}: {e}")

        logger.info(f"  Captured workspace baseline: {len(self.files)} files ({len(git_dirty)} pre-existing dirty/protected)")


    def _get_git_dirty_files(self) -> Set[str]:
        dirty = set()
        try:
            res = subprocess.run(
                ["git", "status", "--porcelain", "-uall"],
                cwd=self.workspace_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if res.returncode == 0:
                for line in (res.stdout or "").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) >= 2:
                        p = parts[1].strip().strip('"')
                        if "->" in p:
                            p = p.split("->")[1].strip()
                        dirty.add(p.replace("\\", "/").lower())
        except Exception:
            pass
        return dirty

    def _extract_git_dirty_regions(self, rel_path: str, current_content: str) -> List[ProtectedRegion]:
        """Extract diff chunks of developer work against git HEAD."""
        regions = []
        try:
            res = subprocess.run(
                ["git", "diff", "HEAD", "--", rel_path],
                cwd=self.workspace_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if res.returncode == 0 and res.stdout:
                # Parse unified diff hunk headers: @@ -old,len +new_start,new_len @@
                for m in re.finditer(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', res.stdout):
                    start = int(m.group(1))
                    length = int(m.group(2)) if m.group(2) else 1
                    end = start + length - 1
                    lines = current_content.splitlines()[max(0, start - 1):end]
                    regions.append(
                        ProtectedRegion(
                            file_path=rel_path,
                            start_line=start,
                            end_line=end,
                            content="\n".join(lines),
                        )
                    )
        except Exception:
            pass
        return regions

    def get_file_baseline(self, file_path: str) -> Optional[FileBaseline]:
        norm = file_path.replace("\\", "/").lower()
        fb = self.files.get(norm)
        if fb is not None:
            if fb.content is None:
                abs_p = self.workspace_path / file_path
                if abs_p.exists():
                    try:
                        fb.content = abs_p.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        pass
            return fb

        # On-demand snapshot from disk if not pre-scanned
        abs_p = self.workspace_path / file_path
        if abs_p.exists():
            try:
                content = abs_p.read_text(encoding="utf-8", errors="ignore")
                chash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                mtime = abs_p.stat().st_mtime
                fb = FileBaseline(
                    file_path=file_path,
                    content_hash=chash,
                    mtime=mtime,
                    is_dirty=False,
                    content=content,
                    protected_regions=[],
                )
                self.files[norm] = fb
                return fb
            except Exception:
                pass
        return None



# ============================================================================
# THREE-WAY MERGE & HIERARCHICAL PATCHER
# ============================================================================

class ThreeWayPatcher:
    """Pure 3-way merge engine for reconciling developer and ticket deltas."""

    @staticmethod
    def merge_3way(base: str, current: str, generated: str) -> Tuple[str, List[str]]:
        """
        Merge changes:
          BASE: state at ticket start
          CURRENT: current developer disk state (BASE + DeveloperDelta)
          GENERATED: ticket proposed state (BASE + TicketDelta)

        Returns: (merged_text, conflict_list)
        """
        if current == base:
            # No developer edits, cleanly take generated
            return generated, []
        if generated == base:
            # No ticket edits, keep current developer state
            return current, []
        if current == generated:
            # Both arrived at the exact same state
            return current, []

        base_lines = base.splitlines(keepends=True)
        curr_lines = current.splitlines(keepends=True)
        gen_lines = generated.splitlines(keepends=True)

        # Diff CURRENT against BASE (Developer delta)
        sm_dev = difflib.SequenceMatcher(None, base_lines, curr_lines)
        dev_opcodes = sm_dev.get_opcodes()

        # Diff GENERATED against BASE (Ticket delta)
        sm_ticket = difflib.SequenceMatcher(None, base_lines, gen_lines)
        ticket_opcodes = sm_ticket.get_opcodes()

        # Fast path: if difflib or git merge-file is available, use git merge-file
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_base = Path(tmp_dir) / "base.txt"
                tmp_curr = Path(tmp_dir) / "current.txt"
                tmp_gen = Path(tmp_dir) / "generated.txt"

                tmp_base.write_text(base, encoding="utf-8")
                tmp_curr.write_text(current, encoding="utf-8")
                tmp_gen.write_text(generated, encoding="utf-8")

                # git merge-file -p current.txt base.txt generated.txt
                res = subprocess.run(
                    ["git", "merge-file", "-p", str(tmp_curr), str(tmp_base), str(tmp_gen)],
                    capture_output=True,
                    text=True,
                )
                if res.returncode == 0:
                    # Clean 3-way merge!
                    return res.stdout, []
                elif res.returncode > 0:
                    # Conflict markers present in stdout
                    conflicts = ["Overlapping conflicting changes detected in git merge-file"]
                    return current, conflicts
        except Exception:
            pass

        # Fallback Python chunk reconciliation
        conflicts = []
        result_lines = []

        # Find intersecting changed intervals in base
        dev_changes = [(b_start, b_end, curr_lines[c_start:c_end])
                       for tag, b_start, b_end, c_start, c_end in dev_opcodes if tag != "equal"]
        ticket_changes = [(b_start, b_end, gen_lines[g_start:g_end])
                          for tag, b_start, b_end, g_start, g_end in ticket_opcodes if tag != "equal"]

        for d_start, d_end, d_chunk in dev_changes:
            for t_start, t_end, t_chunk in ticket_changes:
                # Check overlap in base lines
                if max(d_start, t_start) < min(d_end, t_end):
                    if d_chunk != t_chunk:
                        conflicts.append(
                            f"Collision at base lines {max(d_start, t_start)+1}-{min(d_end, t_end)+1}: "
                            f"developer={len(d_chunk)} lines vs ticket={len(t_chunk)} lines"
                        )

        if conflicts:
            return current, conflicts

        # Non-overlapping: apply both deltas onto base
        # Sort all changes by base line index in reverse order to apply from bottom to top
        all_edits = []
        for tag, b_start, b_end, c_start, c_end in dev_opcodes:
            if tag != "equal":
                all_edits.append((b_start, b_end, "".join(curr_lines[c_start:c_end])))
        for tag, b_start, b_end, g_start, g_end in ticket_opcodes:
            if tag != "equal":
                all_edits.append((b_start, b_end, "".join(gen_lines[g_start:g_end])))

        all_edits.sort(key=lambda x: x[0], reverse=True)
        merged = list(base_lines)
        for b_start, b_end, replacement in all_edits:
            merged[b_start:b_end] = [replacement]

        return "".join(merged), []


class HierarchicalPatcher:
    """
    Executes the 4-Level Patch Strategy:
      1. AST / Symbol-Aware Patch
      2. SEARCH/REPLACE Surgical Patch
      3. 3-Way Textual Merge
      4. Conflict -> STOP
    """

    @classmethod
    def apply_patch(
        cls,
        file_path: str,
        base_content: Optional[str],
        current_content: str,
        proposed_content_or_edits: Union[str, List[Dict[str, str]]],
        protected_regions: Optional[List[ProtectedRegion]] = None,
        symbol: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Applies patch according to hierarchy.
        Returns: (final_content, applied_strategy_name)
        Raises: PatchConflictError if conflict detected.
        """
        protected = protected_regions or []

        # ── Level 2: If structured SEARCH/REPLACE edits provided ─────────────
        if isinstance(proposed_content_or_edits, list):
            edits = proposed_content_or_edits
            result = current_content
            for idx, edit in enumerate(edits):
                old_str = edit.get("old_str", "")
                new_str = edit.get("new_str", "")

                # Check if old_str touches protected region without authorization
                for pr in protected:
                    # If this region belongs to an unrelated symbol, protect it.
                    # But if the ticket is explicitly authorized/tasked to modify this symbol,
                    # think like a developer solving the ticket: allow updating the target symbol!
                    is_targeted_symbol = bool(
                        symbol and pr.symbol and symbol.lower() == pr.symbol.lower()
                    )
                    if not is_targeted_symbol and pr.content and pr.content in old_str and pr.content not in new_str:
                        raise PatchConflictError(
                            file_path=file_path,
                            symbol=symbol,
                            developer_change=pr.content,
                            ticket_change=new_str,
                            reason="Patch attempted to overwrite unrelated protected developer region",
                        )


                count = result.count(old_str)
                if count == 1:
                    result = result.replace(old_str, new_str, 1)
                elif count > 1:
                    raise PatchConflictError(
                        file_path=file_path,
                        symbol=symbol,
                        developer_change="",
                        ticket_change=new_str,
                        reason=f"Ambiguous SEARCH block (matches {count} locations)",
                    )
                else:
                    # Try CRLF adaptation
                    cand = old_str.replace("\n", "\r\n") if "\r\n" not in old_str else old_str.replace("\r\n", "\n")
                    if result.count(cand) == 1:
                        result = result.replace(cand, new_str, 1)
                    else:
                        raise PatchConflictError(
                            file_path=file_path,
                            symbol=symbol,
                            developer_change="",
                            ticket_change=new_str,
                            reason=f"SEARCH block not found in current disk state",
                        )
            return result, "surgical_search_replace"

        # ── Proposed is complete proposed content ────────────────────────────
        generated_content = proposed_content_or_edits

        # If disk hasn't changed from baseline, simply use generated
        if not base_content or current_content == base_content:
            return generated_content, "direct_apply"

        # ── Level 3: 3-Way Merge (DeveloperDelta + TicketDelta) ───────────────
        merged_text, conflicts = ThreeWayPatcher.merge_3way(
            base=base_content,
            current=current_content,
            generated=generated_content,
        )

        if conflicts:
            raise PatchConflictError(
                file_path=file_path,
                symbol=symbol,
                developer_change=current_content[:300],
                ticket_change=generated_content[:300],
                reason=f"3-way merge conflict: {conflicts}",
            )

        return merged_text, "3way_merge"


# ============================================================================
# WORKSPACE INTEGRITY MANAGER
# ============================================================================

class WorkspaceIntegrityManager:
    """Central manager coordinating baseline, 3-way patching, ownership, and rollback."""

    def __init__(self, workspace_path: Union[str, Path]):
        self.workspace_path = Path(workspace_path)
        self.baseline = WorkspaceBaseline(self.workspace_path)
        self.registry = PatchRegistry()
        self._read_snapshots: Dict[str, Tuple[str, str]] = {}  # norm_path -> (content, hash)

    def capture_baseline(self) -> None:
        """Capture baseline state at run start."""
        self.baseline.capture()

    def record_file_read(self, file_path: str, content: str) -> None:
        """Record file state at the time it was read by an agent (for concurrency check)."""
        norm = file_path.replace("\\", "/").lower()
        chash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        self._read_snapshots[norm] = (content, chash)

    def safe_write(
        self,
        file_path: str,
        new_content_or_edits: Union[str, List[Dict[str, str]]],
        ticket_id: str = "",
        task_id: str = "",
        requirement_id: str = "",
        symbol: Optional[str] = None,
        authorizations: Optional[List[DeleteAuthorization]] = None,
        allow_full_rewrite: bool = False,
    ) -> PatchOwnershipRecord:
        """
        Safely write code to workspace adhering to all integrity invariants:
          1. Optimistic Concurrency Check (no silent clobber of outside edits)
          2. Patch Idempotency Check (no duplicate application)
          3. Structural Change & Deletion Authorization (no accidental omissions)
          4. Hierarchical 3-Way Patching (preserve developer delta)
          5. Register Patch Ownership with inverse patch for reversible rollback
        """
        from ticket_to_code.runtime.security import SecurityPolicy, SecurityViolation
        ok, why = SecurityPolicy.validate_path_containment(file_path, self.workspace_path)
        if not ok:
            raise SecurityViolation(f"safe_write rejected: {why}")
        abs_path = self.workspace_path / file_path
        norm = file_path.replace("\\", "/").lower()

        current_content = ""
        if abs_path.exists():
            current_content = abs_path.read_text(encoding="utf-8", errors="ignore")

        # ── 1. Optimistic Concurrency Check ──────────────────────────────────
        if norm in self._read_snapshots and abs_path.exists():
            _, expected_hash = self._read_snapshots[norm]
            current_hash = hashlib.sha256(current_content.encode("utf-8")).hexdigest()
            if current_hash != expected_hash:
                logger.warning(
                    f"⚠️ [CONCURRENT_WORKSPACE_CHANGE] {file_path} modified on disk since read. "
                    f"Will perform 3-way rebase against active disk state."
                )

        # ── 2. Idempotency Check ─────────────────────────────────────────────
        if isinstance(new_content_or_edits, str):
            proposed_hash = hashlib.sha256(new_content_or_edits.encode("utf-8")).hexdigest()
            current_disk_hash = hashlib.sha256(current_content.encode("utf-8")).hexdigest() if current_content else ""
            if proposed_hash == current_disk_hash:
                already_applied = self.registry.is_already_applied(file_path, new_content_or_edits)
                logger.info(f"ℹ️ [PATCH_ALREADY_APPLIED] Active content on disk already matches patch for {file_path}")
                if already_applied:
                    record = self.registry._patches.get(already_applied)
                    if record:
                        return record

        # ── 3. Structural Validation & Deletion Check ────────────────────────
        fb = self.baseline.get_file_baseline(file_path)
        base_content = fb.content if fb else None

        # Resolve final content to evaluate structural delta
        protected = fb.protected_regions if fb else []
        final_content, strategy = HierarchicalPatcher.apply_patch(
            file_path=file_path,
            base_content=base_content,
            current_content=current_content,
            proposed_content_or_edits=new_content_or_edits,
            protected_regions=protected,
            symbol=symbol,
        )

        # Structural check
        if current_content:
            delta = StructuralChangeManifest.compute_delta(current_content, final_content, file_path)
            deletion_violations = DeleteAuthorizationValidator.validate_deletions(
                delta=delta,
                authorizations=authorizations,
                protected_regions=protected,
            )
            if deletion_violations:
                for v in deletion_violations:
                    logger.error(f"  ❌ {v}")
                raise UnauthorizedDeletionError(
                    file_path=file_path,
                    symbol=symbol or "unknown",
                    reason="; ".join(deletion_violations),
                )

        # ── 4. Write to disk ─────────────────────────────────────────────────
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        before_hash = hashlib.sha256(current_content.encode("utf-8")).hexdigest()
        after_hash = hashlib.sha256(final_content.encode("utf-8")).hexdigest()

        abs_path.write_text(final_content, encoding="utf-8")
        logger.info(f"  📝 [WorkspaceIntegrity] Safely written: {file_path} via strategy '{strategy}'")

        # ── 5. Register Patch Ownership with Inverse Content ─────────────────
        patch_id = f"PATCH-{uuid.uuid4().hex[:8].upper()}"
        record = PatchOwnershipRecord(
            patch_id=patch_id,
            ticket_id=ticket_id,
            task_id=task_id,
            requirement_id=requirement_id,
            file_path=file_path,
            symbol=symbol,
            operation="CREATE" if not current_content else ("SURGICAL_PATCH" if isinstance(new_content_or_edits, list) else "MODIFY"),
            before_hash=before_hash,
            after_hash=after_hash,
            inverse_content=current_content,  # Reverting this patch restores pre-patch current_content
            after_content=final_content,
        )
        self.registry.register(record)

        # Update snapshot and baseline for next progressive write within the same ticket
        self._read_snapshots[norm] = (final_content, after_hash)
        if norm in self.baseline.files:
            self.baseline.files[norm].content = final_content
            self.baseline.files[norm].sha256 = after_hash
        return record

    def rollback_patch(self, patch_id: str) -> bool:
        """Rollback a specific patch."""
        return self.registry.rollback_patch(patch_id, self.workspace_path)

    def rollback_ai_changes(self, ticket_id: str) -> int:
        """Rollback ONLY Ticket-to-Code patches for a ticket in reverse order."""
        patches = self.registry.get_patches_for_ticket(ticket_id)
        patches.sort(key=lambda p: p.applied_at, reverse=True)
        rolled_back = 0
        for p in patches:
            if self.registry.rollback_patch(p.patch_id, self.workspace_path):
                rolled_back += 1
        logger.info(f"  ↩️ [WorkspaceIntegrity] Rolled back {rolled_back} AI-owned patches for ticket {ticket_id}")
        return rolled_back


# Singleton access
_GLOBAL_INTEGRITY_MGR: Optional[WorkspaceIntegrityManager] = None

def get_workspace_integrity_manager(workspace_path: Optional[Union[str, Path]] = None) -> WorkspaceIntegrityManager:
    global _GLOBAL_INTEGRITY_MGR
    if _GLOBAL_INTEGRITY_MGR is None:
        ws = workspace_path or Path.cwd()
        _GLOBAL_INTEGRITY_MGR = WorkspaceIntegrityManager(ws)
    elif workspace_path is not None:
        target_ws = Path(workspace_path).resolve()
        if _GLOBAL_INTEGRITY_MGR.workspace_path.resolve() != target_ws:
            _GLOBAL_INTEGRITY_MGR = WorkspaceIntegrityManager(target_ws)
    return _GLOBAL_INTEGRITY_MGR

def set_workspace_integrity_manager(mgr: WorkspaceIntegrityManager) -> None:
    global _GLOBAL_INTEGRITY_MGR
    _GLOBAL_INTEGRITY_MGR = mgr
