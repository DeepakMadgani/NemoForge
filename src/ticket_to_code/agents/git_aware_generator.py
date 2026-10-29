"""
Git-Aware Generator — Track Changes and Show Diffs

Integrates with git to:
  - Show diffs of changes for every edit
  - Track what files were modified
  - Provide rollback capability
  - Generate meaningful commit messages

Author: Deepak Madgani
Date: August 2026
"""

import logging
import subprocess
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class ChangeType(str, Enum):
    """Type of change made."""
    ADDED = "added"  # New file or new content
    MODIFIED = "modified"  # File modified
    DELETED = "deleted"  # File or content removed
    RENAMED = "renamed"  # File renamed
    UNKNOWN = "unknown"


@dataclass
class FileDiff:
    """Difference for a single file."""
    file_path: str
    change_type: ChangeType
    lines_added: int = 0
    lines_removed: int = 0
    diff_content: str = ""  # Unified diff format
    before_content: Optional[str] = None  # Full content before
    after_content: Optional[str] = None  # Full content after

    def to_dict(self):
        return {
            "file_path": self.file_path,
            "change_type": self.change_type.value,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "diff_content": self.diff_content,
        }

    def get_summary(self) -> str:
        """Get one-line summary of change."""
        return f"{self.change_type.value}: {self.file_path} (+{self.lines_added} -{self.lines_removed})"


@dataclass
class GenerationSession:
    """Session tracking all changes made during code generation."""
    session_id: str
    ticket_id: str
    workspace_path: str
    files_changed: List[FileDiff]
    commit_message: str = ""
    created_at: str = ""

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "ticket_id": self.ticket_id,
            "workspace_path": self.workspace_path,
            "files_changed": [f.to_dict() for f in self.files_changed],
            "commit_message": self.commit_message,
            "created_at": self.created_at,
        }


class GitAwareGenerator:
    """
    Tracks file changes and integrates with git for diffing and rollback.

    Usage:
        gen = GitAwareGenerator("/path/to/workspace")
        
        # Start tracking a session
        session = gen.start_session(ticket_id="TICKET-123")
        
        # After generating a file
        gen.track_change(
            file_path="add-members.component.ts",
            new_content=generated_code,
        )
        
        # Show diff
        diff = gen.get_diff("add-members.component.ts")
        print(diff.diff_content)
        
        # Commit changes with meaningful message
        gen.commit_session(
            session=session,
            message="TICKET-123: Add member selection and org filtering",
        )
        
        # Or rollback
        gen.rollback_session(session)
    """

    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.current_session: Optional[GenerationSession] = None
        self._file_snapshots: Dict[str, str] = {}  # file_path → original content
        self._changes: List[FileDiff] = []

    def start_session(self, session_id: str, ticket_id: str) -> GenerationSession:
        """Start a new generation session."""
        from datetime import datetime

        session = GenerationSession(
            session_id=session_id,
            ticket_id=ticket_id,
            workspace_path=str(self.workspace_path),
            files_changed=[],
            created_at=datetime.now().isoformat(),
        )

        self.current_session = session
        self._changes = []
        self._file_snapshots = {}

        logger.info(f"📝 Started generation session: {session_id} for {ticket_id}")

        return session

    def track_change(
        self,
        file_path: str,
        new_content: str,
        change_type: ChangeType = ChangeType.MODIFIED,
    ) -> FileDiff:
        """
        Track a file change.

        Args:
            file_path: Relative path from workspace
            new_content: New file content
            change_type: Type of change (ADDED, MODIFIED, DELETED)
        """
        full_path = self.workspace_path / file_path

        # Get before content
        before_content = None
        if full_path.exists():
            before_content = full_path.read_text(encoding="utf-8", errors="ignore")
        else:
            before_content = ""
            if change_type == ChangeType.MODIFIED:
                change_type = ChangeType.ADDED

        # Store snapshot if first change to this file
        if file_path not in self._file_snapshots:
            self._file_snapshots[file_path] = before_content

        # Calculate diff
        diff_content = self._compute_diff(file_path, before_content, new_content)
        lines_added, lines_removed = self._count_diff_lines(diff_content)

        # Create FileDiff object
        file_diff = FileDiff(
            file_path=file_path,
            change_type=change_type,
            lines_added=lines_added,
            lines_removed=lines_removed,
            diff_content=diff_content,
            before_content=before_content,
            after_content=new_content,
        )

        self._changes.append(file_diff)

        logger.info(f"  {file_diff.get_summary()}")

        return file_diff

    def get_diff(self, file_path: str) -> Optional[FileDiff]:
        """Get tracked diff for a file."""
        for change in self._changes:
            if change.file_path == file_path:
                return change

        return None

    def get_all_diffs(self) -> List[FileDiff]:
        """Get all tracked changes."""
        return self._changes

    def get_session_summary(self) -> Dict:
        """Get summary of all changes in session."""
        if not self.current_session:
            return {}

        total_added = sum(c.lines_added for c in self._changes)
        total_removed = sum(c.lines_removed for c in self._changes)

        by_type = {}
        for change in self._changes:
            type_name = change.change_type.value
            if type_name not in by_type:
                by_type[type_name] = []
            by_type[type_name].append(change.file_path)

        return {
            "session_id": self.current_session.session_id,
            "ticket_id": self.current_session.ticket_id,
            "total_files_changed": len(self._changes),
            "total_lines_added": total_added,
            "total_lines_removed": total_removed,
            "by_type": by_type,
            "files": [c.to_dict() for c in self._changes],
        }

    def commit_session(
        self,
        message: str,
        auto_push: bool = False,
    ) -> Optional[str]:
        """
        Commit all tracked changes to git.

        Args:
            message: Commit message
            auto_push: Whether to push to remote

        Returns: Commit hash, or None if git not available
        """
        if not self.current_session:
            logger.error("No active session to commit")
            return None

        try:
            # Stage all changes
            subprocess.run(
                ["git", "add", "."],
                cwd=self.workspace_path,
                capture_output=True,
                check=True,
            )

            # Commit
            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.workspace_path,
                capture_output=True,
                text=True,
                check=True,
            )

            commit_hash = result.stdout.split()[2]  # Extract hash from output

            logger.info(f"✅ Committed: {commit_hash}")

            if auto_push:
                subprocess.run(
                    ["git", "push"],
                    cwd=self.workspace_path,
                    capture_output=True,
                    check=True,
                )
                logger.info(f"✅ Pushed to remote")

            return commit_hash

        except subprocess.CalledProcessError as e:
            logger.warning(f"Git commit failed: {e.stderr}")
            return None

    def rollback_session(self) -> bool:
        """
        Rollback all changes made in this session.

        Returns: True if rollback successful, False otherwise
        """
        if not self.current_session:
            logger.error("No active session to rollback")
            return False

        try:
            # Restore original content
            for file_path, original_content in self._file_snapshots.items():
                full_path = self.workspace_path / file_path

                if original_content == "":
                    # File didn't exist originally, delete it
                    if full_path.exists():
                        full_path.unlink()
                        logger.info(f"  Deleted: {file_path}")
                else:
                    # Restore original content
                    full_path.write_text(original_content, encoding="utf-8")
                    logger.info(f"  Restored: {file_path}")

            logger.info(f"✅ Session rolled back")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def generate_commit_message(
        self,
        ticket_id: str,
        ticket_description: Optional[str] = None,
    ) -> str:
        """
        Generate a meaningful commit message based on changes.

        Example output:
            "TICKET-123: Add member selection and org filtering
            
            - Create AddMembersComponent with form controls
            - Update contract member service with new query
            - Add organization dropdown to template"
        """
        summary = self.get_session_summary()

        lines = [f"{ticket_id}: {ticket_description or 'Code generation'}\n"]

        # Group by file type
        if "added" in summary.get("by_type", {}):
            lines.append("Added files:")
            for file_path in summary["by_type"]["added"][:3]:
                lines.append(f"  - {file_path}")

        if "modified" in summary.get("by_type", {}):
            lines.append("Modified files:")
            for file_path in summary["by_type"]["modified"][:3]:
                lines.append(f"  - {file_path}")

        # Add stats
        lines.append(
            f"\nStats: +{summary['total_lines_added']} -{summary['total_lines_removed']} lines"
        )

        return "\n".join(lines)

    def show_diff_in_ui(self, file_path: str) -> str:
        """
        Format diff for UI display (colorized).

        Returns: Formatted diff with ANSI color codes
        """
        diff = self.get_diff(file_path)
        if not diff:
            return f"No changes for {file_path}"

        lines = diff.diff_content.split("\n")
        colored_lines = []

        for line in lines:
            if line.startswith("+") and not line.startswith("+++"):
                colored_lines.append(f"\033[92m{line}\033[0m")  # Green
            elif line.startswith("-") and not line.startswith("---"):
                colored_lines.append(f"\033[91m{line}\033[0m")  # Red
            elif line.startswith("@@"):
                colored_lines.append(f"\033[94m{line}\033[0m")  # Blue
            else:
                colored_lines.append(line)

        return "\n".join(colored_lines)

    def _compute_diff(
        self,
        file_path: str,
        before_content: str,
        after_content: str,
    ) -> str:
        """Compute unified diff between before/after."""
        import difflib

        before_lines = before_content.split("\n")
        after_lines = after_content.split("\n")

        diff = difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        )

        return "\n".join(diff)

    def _count_diff_lines(self, diff_content: str) -> Tuple[int, int]:
        """Count added and removed lines in diff."""
        added = 0
        removed = 0

        for line in diff_content.split("\n"):
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1

        return added, removed
