"""
Git Worktree & Shadow Workspace Isolation Sandbox.

Guarantees zero working-copy pollution by executing autonomous code mutations
and verification inside isolated git worktrees or shadow scratch spaces.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SandboxEnvironment:
    original_repo: Path
    sandbox_path: Path
    ticket_id: str
    is_git_worktree: bool
    branch_name: Optional[str] = None


class GitWorktreeSandbox:
    """Manages ephemeral, isolated workspace environments for autonomous agent mutation."""

    @classmethod
    def create_sandbox(
        cls,
        base_repo_path: str | Path,
        ticket_id: str,
        sandbox_parent_dir: Optional[str | Path] = None,
    ) -> SandboxEnvironment:
        repo = Path(base_repo_path).resolve()
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in ticket_id)
        default_parent = repo.parent / ".ticket_to_code_sandboxes" if not (repo.parent / ".aviator_sandboxes").exists() else (repo.parent / ".aviator_sandboxes")
        parent = Path(sandbox_parent_dir).resolve() if sandbox_parent_dir else default_parent
        parent.mkdir(parents=True, exist_ok=True)
        sandbox_dir = parent / f"sandbox_{safe_id}"

        # Clean old directory if exists
        if sandbox_dir.exists():
            cls.cleanup_sandbox(sandbox_dir)

        # Check if git is available and repo has .git
        has_git = (repo / ".git").exists()
        branch_name = f"ticket_to_code/sandbox/{safe_id}"

        if has_git:
            try:
                # Attempt to create git worktree
                proc = subprocess.run(
                    ["git", "worktree", "add", "-b", branch_name, str(sandbox_dir), "HEAD"],
                    cwd=str(repo),
                    capture_output=True,
                    text=True,
                    timeout=30,
                    shell=(sys.platform == "win32"),
                )
                if proc.returncode == 0:
                    return SandboxEnvironment(
                        original_repo=repo,
                        sandbox_path=sandbox_dir,
                        ticket_id=ticket_id,
                        is_git_worktree=True,
                        branch_name=branch_name,
                    )
            except Exception:
                pass

        # Fallback: lightweight copy of source files (excluding heavy build dirs)
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        for item in repo.iterdir():
            if item.name in [".git", "node_modules", ".gradle", "target", "build", "dist", ".ticket_to_code", ".ticket_to_code_sandboxes", ".aviator", ".aviator_sandboxes", "__pycache__", ".venv"]:
                continue
            dest = sandbox_dir / item.name
            try:
                if item.is_dir():
                    shutil.copytree(item, dest, ignore=shutil.ignore_patterns("node_modules", "target", "build", ".gradle", "__pycache__"))
                else:
                    shutil.copy2(item, dest)
            except Exception:
                pass

        return SandboxEnvironment(
            original_repo=repo,
            sandbox_path=sandbox_dir,
            ticket_id=ticket_id,
            is_git_worktree=False,
            branch_name=None,
        )

    @classmethod
    def cleanup_sandbox(cls, sandbox_env: SandboxEnvironment | Path) -> bool:
        path = sandbox_env.sandbox_path if isinstance(sandbox_env, SandboxEnvironment) else Path(sandbox_env)
        if not path.exists():
            return True

        orig_repo = sandbox_env.original_repo if isinstance(sandbox_env, SandboxEnvironment) else path.parent.parent
        is_worktree = sandbox_env.is_git_worktree if isinstance(sandbox_env, SandboxEnvironment) else False

        if is_worktree and orig_repo and (orig_repo / ".git").exists():
            try:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(path)],
                    cwd=str(orig_repo),
                    capture_output=True,
                    timeout=15,
                    shell=(sys.platform == "win32"),
                )
                if isinstance(sandbox_env, SandboxEnvironment) and sandbox_env.branch_name:
                    subprocess.run(
                        ["git", "branch", "-D", sandbox_env.branch_name],
                        cwd=str(orig_repo),
                        capture_output=True,
                        timeout=15,
                        shell=(sys.platform == "win32"),
                    )
            except Exception:
                pass

        try:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
            return True
        except Exception:
            return False

    PROTECTED_CONFIG_PATTERNS = (
        "application*.yml", "application*.yaml", "application*.properties",
        ".env*", "appsettings*.json", "*secret*", "*credential*", "*local*.json"
    )

    @classmethod
    def apply_sandbox_to_original(
        cls,
        sandbox_env: SandboxEnvironment,
        changed_rel_paths: list[str],
        allow_protected_configs: bool = False
    ) -> bool:
        """Copies verified changed files back from sandbox to the original repository.
        Protects local developer configuration/secret files from unintended collateral overwrite.
        """
        if not sandbox_env.sandbox_path.exists() or not sandbox_env.original_repo.exists():
            return False

        import fnmatch
        for rel in changed_rel_paths:
            norm_name = Path(rel).name.lower()
            if not allow_protected_configs and any(fnmatch.fnmatch(norm_name, p) for p in cls.PROTECTED_CONFIG_PATTERNS):
                # Guard against overwriting local credentials/debug configs
                continue

            src = sandbox_env.sandbox_path / rel.lstrip("/\\")
            dst = sandbox_env.original_repo / rel.lstrip("/\\")
            if src.exists() and src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        return True
