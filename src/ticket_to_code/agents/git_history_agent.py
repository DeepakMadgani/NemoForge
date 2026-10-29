"""
Git History & Change Tracking Agent — Enhancement 9

Provides git-based evidence for the evidence collection loop:
- Recently changed files
- Blame for specific lines
- Commits mentioning a ticket ID

Safety: READ-ONLY. Only runs `git log`, `git blame`, `git diff`.
Never modifies the repository.

Author: Deepak Madgani
Date: July 2026
"""

import logging
import subprocess
import re
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# MODELS
# ============================================================================

class RecentChange(BaseModel):
    """A file that was recently modified in git."""
    file_path: str = Field(..., description="Path to changed file")
    last_author: str = Field("", description="Who last modified it")
    last_date: str = Field("", description="When it was last modified")
    change_count: int = Field(0, description="Number of changes in period")
    commit_hash: str = Field("", description="Last commit hash")


class BlameLine(BaseModel):
    """Blame info for a specific line."""
    file_path: str = Field(...)
    line_number: int = Field(...)
    author: str = Field("")
    date: str = Field("")
    commit_hash: str = Field("")
    commit_message: str = Field("")


class RelatedCommit(BaseModel):
    """A commit related to a ticket."""
    commit_hash: str = Field(...)
    author: str = Field("")
    date: str = Field("")
    message: str = Field("")
    files_changed: List[str] = Field(default_factory=list)


class GitEvidence(BaseModel):
    """Collected git evidence for the evidence pipeline."""
    has_git_data: bool = Field(False)
    recent_changes: List[RecentChange] = Field(default_factory=list)
    related_commits: List[RelatedCommit] = Field(default_factory=list)
    blame_results: List[BlameLine] = Field(default_factory=list)
    hotspot_files: List[str] = Field(
        default_factory=list,
        description="Files changed most frequently — likely sources of bugs"
    )


class GitHistoryAgent:
    """
    Provides git-based evidence for the investigation pipeline.

    All operations are read-only git commands (log, blame, diff).
    """

    def __init__(self, workspace_path: str):
        self.workspace = Path(workspace_path)
        self._git_available: Optional[bool] = None

    def _is_git_repo(self) -> bool:
        """Check if workspace is a git repository."""
        if self._git_available is not None:
            return self._git_available
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=str(self.workspace),
                capture_output=True, text=True, timeout=5
            )
            self._git_available = result.returncode == 0
        except Exception:
            self._git_available = False
        return self._git_available

    def _run_git(self, args: List[str], timeout: int = 15) -> Optional[str]:
        """Run a git command and return stdout, or None on failure."""
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=str(self.workspace),
                capture_output=True, text=True,
                timeout=timeout, encoding="utf-8", errors="replace"
            )
            if result.returncode == 0:
                return result.stdout
            logger.debug(f"git {' '.join(args)} failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.debug(f"git {' '.join(args)} timed out")
        except Exception as exc:
            logger.debug(f"git {' '.join(args)} error: {exc}")
        return None

    # ------------------------------------------------------------------
    # 1. Recently changed files
    # ------------------------------------------------------------------

    def recent_changes(self, days: int = 7, max_files: int = 30) -> List[RecentChange]:
        """
        Find files changed in the last N days.

        Returns list of RecentChange sorted by change count (descending).
        """
        if not self._is_git_repo():
            return []

        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        output = self._run_git([
            "log", f"--since={since}",
            "--name-only", "--pretty=format:%H|%an|%ad|%s",
            "--date=short"
        ])

        if not output:
            return []

        file_stats: Dict[str, Dict] = {}
        current_commit = {}

        for line in output.strip().splitlines():
            if "|" in line and len(line.split("|")) >= 3:
                parts = line.split("|", 3)
                current_commit = {
                    "hash": parts[0][:8],
                    "author": parts[1],
                    "date": parts[2],
                }
            elif line.strip() and current_commit:
                fp = line.strip()
                if fp not in file_stats:
                    file_stats[fp] = {
                        "last_author": current_commit["author"],
                        "last_date": current_commit["date"],
                        "commit_hash": current_commit["hash"],
                        "count": 0,
                    }
                file_stats[fp]["count"] += 1

        changes = [
            RecentChange(
                file_path=fp,
                last_author=info["last_author"],
                last_date=info["last_date"],
                change_count=info["count"],
                commit_hash=info["commit_hash"],
            )
            for fp, info in file_stats.items()
        ]

        # Sort by change frequency (hotspots first)
        changes.sort(key=lambda c: c.change_count, reverse=True)
        return changes[:max_files]

    # ------------------------------------------------------------------
    # 2. Blame a specific line
    # ------------------------------------------------------------------

    def blame_line(self, file_path: str, line_number: int) -> Optional[BlameLine]:
        """Get blame info for a specific file and line."""
        if not self._is_git_repo():
            return None

        output = self._run_git([
            "blame", "-L", f"{line_number},{line_number}",
            "--porcelain", file_path
        ])

        if not output:
            return None

        commit_hash = ""
        author = ""
        date = ""
        message = ""

        for line in output.strip().splitlines():
            if line.startswith("author "):
                author = line[7:]
            elif line.startswith("author-time "):
                try:
                    ts = int(line[12:])
                    date = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                except ValueError:
                    pass
            elif line.startswith("summary "):
                message = line[8:]
            elif re.match(r'^[0-9a-f]{40}', line):
                commit_hash = line.split()[0][:8]

        return BlameLine(
            file_path=file_path,
            line_number=line_number,
            author=author,
            date=date,
            commit_hash=commit_hash,
            commit_message=message,
        )

    # ------------------------------------------------------------------
    # 3. Find commits mentioning a ticket ID
    # ------------------------------------------------------------------

    def find_related_commits(
        self, ticket_id: str, max_commits: int = 10
    ) -> List[RelatedCommit]:
        """Find commits whose messages mention the ticket ID."""
        if not self._is_git_repo() or not ticket_id:
            return []

        output = self._run_git([
            "log", f"--grep={ticket_id}", "--all",
            f"-{max_commits}",
            "--pretty=format:%H|%an|%ad|%s",
            "--date=short", "--name-only"
        ])

        if not output:
            return []

        commits: List[RelatedCommit] = []
        current: Optional[Dict] = None

        for line in output.strip().splitlines():
            if "|" in line and len(line.split("|")) >= 3:
                if current:
                    commits.append(RelatedCommit(**current))
                parts = line.split("|", 3)
                current = {
                    "commit_hash": parts[0][:8],
                    "author": parts[1],
                    "date": parts[2],
                    "message": parts[3] if len(parts) > 3 else "",
                    "files_changed": [],
                }
            elif line.strip() and current:
                current["files_changed"].append(line.strip())

        if current:
            commits.append(RelatedCommit(**current))

        return commits

    # ------------------------------------------------------------------
    # 4. Main evidence gathering
    # ------------------------------------------------------------------

    def gather_evidence(
        self, ticket_id: str = "", days: int = 7
    ) -> GitEvidence:
        """
        Main entry point: gather all git-based evidence.

        Args:
            ticket_id: Ticket ID to search in commit messages.
            days: How many days of history to search.

        Returns:
            GitEvidence with all findings.
        """
        if not self._is_git_repo():
            logger.info("GitHistoryAgent: not a git repo, skipping")
            return GitEvidence(has_git_data=False)

        recent = self.recent_changes(days=days)
        related = self.find_related_commits(ticket_id)

        # Identify hotspot files (changed 3+ times in the period)
        hotspots = [c.file_path for c in recent if c.change_count >= 3]

        result = GitEvidence(
            has_git_data=bool(recent or related),
            recent_changes=recent,
            related_commits=related,
            hotspot_files=hotspots,
        )

        logger.info(
            f"GitHistoryAgent: {len(recent)} recent changes, "
            f"{len(related)} related commits, {len(hotspots)} hotspots"
        )
        return result
