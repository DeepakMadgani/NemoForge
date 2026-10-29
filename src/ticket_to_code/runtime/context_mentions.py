"""
Context Mention Resolver: Zero-Token, Sub-Millisecond Context Injection.

Provides Cursor-style '@' context expansion:
- @file:<path> or @<filename.ext> (e.g. @ContractMemberRepository.java)
- @symbol:<identifier> (e.g. @UpdateProjectUserHandler)
- @git:staged, @git:diff, @git:status
- @dir:<folder>

Resolves context deterministically from the local filesystem, AST indices,
and git CLI before model generation (0 LLM tokens, <5ms execution).
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class ResolvedMention:
    mention_type: str  # 'file', 'symbol', 'git', 'dir'
    tag: str           # e.g. '@ContractMemberRepository.java'
    target: str        # e.g. 'src/.../ContractMemberRepository.java'
    content: str       # extracted content
    token_estimate: int = 0
    resolved: bool = True
    error_message: Optional[str] = None


@dataclass
class MentionExpansionResult:
    expanded_text: str
    resolved_mentions: List[ResolvedMention] = field(default_factory=list)
    injected_context_block: str = ""
    total_tokens_estimated: int = 0


class ContextMentionResolver:
    """Parses and resolves @mentions from developer prompt messages."""

    # Matches @file:path, @symbol:name, @git:status, @git:diff, @git:staged, or @filename.ext
    MENTION_RE = re.compile(
        r'@(?:(file|symbol|git|dir):([A-Za-z0-9_./\\-]+)|([A-Za-z0-9_./\\-]+\.[a-zA-Z0-9]+))'
    )

    def __init__(self, workspace_path: Path):
        self.workspace_path = Path(workspace_path).resolve()
        self._basename_index: Optional[Dict[str, List[Path]]] = None

    def _ensure_basename_index(self) -> Dict[str, List[Path]]:
        """Lazily indexes file basenames in the workspace for fast lookup."""
        if self._basename_index is not None:
            return self._basename_index

        index: Dict[str, List[Path]] = {}
        ignore_dirs = {".git", ".gradle", "node_modules", "target", "build", "dist", ".idea", ".vscode", "__pycache__"}

        for root, dirs, files in os.walk(self.workspace_path):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for f in files:
                f_path = Path(root) / f
                index.setdefault(f.lower(), []).append(f_path)

        self._basename_index = index
        return index

    def resolve_mentions(self, prompt: str, max_file_bytes: int = 64_000) -> MentionExpansionResult:
        """Parses all @mentions in the prompt and builds ground-truth context blocks."""
        if not prompt or "@" not in prompt:
            return MentionExpansionResult(expanded_text=prompt)

        matches = list(self.MENTION_RE.finditer(prompt))
        if not matches:
            return MentionExpansionResult(expanded_text=prompt)

        resolved_mentions: List[ResolvedMention] = []
        seen_tags: Set[str] = set()

        for m in matches:
            full_match = m.group(0)
            if full_match in seen_tags:
                continue
            seen_tags.add(full_match)

            m_type = m.group(1)
            m_target = m.group(2)
            implicit_file = m.group(3)

            if implicit_file:
                # Explicit filename like @ContractMemberRepository.java
                resolved = self._resolve_file_mention(full_match, implicit_file, max_file_bytes)
            elif m_type == "file":
                resolved = self._resolve_file_mention(full_match, m_target, max_file_bytes)
            elif m_type == "git":
                resolved = self._resolve_git_mention(full_match, m_target)
            elif m_type == "symbol":
                resolved = self._resolve_symbol_mention(full_match, m_target)
            elif m_type == "dir":
                resolved = self._resolve_dir_mention(full_match, m_target)
            else:
                continue

            if resolved:
                resolved_mentions.append(resolved)

        if not resolved_mentions:
            return MentionExpansionResult(expanded_text=prompt)

        # Build structured context block
        context_blocks = ["\n<!-- DETERMINISTIC CONTEXT (RESOLVED VIA @MENTIONS) -->"]
        total_tokens = 0

        for rm in resolved_mentions:
            if rm.resolved and rm.content:
                tokens = max(1, len(rm.content) // 4)
                rm.token_estimate = tokens
                total_tokens += tokens
                context_blocks.append(
                    f"\n<context_mention tag=\"{rm.tag}\" type=\"{rm.mention_type}\" target=\"{rm.target}\">\n"
                    f"{rm.content}\n"
                    f"</context_mention>"
                )
            elif rm.error_message:
                context_blocks.append(
                    f"\n<context_mention tag=\"{rm.tag}\" error=\"{rm.error_message}\" />"
                )

        context_blocks.append("<!-- END CONTEXT MENTIONS -->\n")
        injected_block = "\n".join(context_blocks)

        return MentionExpansionResult(
            expanded_text=prompt,
            resolved_mentions=resolved_mentions,
            injected_context_block=injected_block,
            total_tokens_estimated=total_tokens,
        )

    def _resolve_file_mention(self, tag: str, file_target: str, max_bytes: int) -> ResolvedMention:
        """Finds and reads a file mentioned via @file:... or @filename.ext."""
        # Check direct path relative to workspace
        target_path = (self.workspace_path / file_target).resolve()
        if not target_path.exists() or not target_path.is_file():
            # Check basename index
            index = self._ensure_basename_index()
            basename = Path(file_target).name.lower()
            candidates = index.get(basename, [])
            if candidates:
                target_path = candidates[0]
            else:
                return ResolvedMention(
                    mention_type="file",
                    tag=tag,
                    target=file_target,
                    content="",
                    resolved=False,
                    error_message=f"File not found in workspace: '{file_target}'",
                )

        try:
            raw_bytes = target_path.read_bytes()
            if len(raw_bytes) > max_bytes:
                content = raw_bytes[:max_bytes].decode("utf-8", errors="replace") + "\n... [TRUNCATED - EXCEEDS SIZE LIMIT]"
            else:
                content = raw_bytes.decode("utf-8", errors="replace")

            rel_path = str(target_path.relative_to(self.workspace_path)).replace("\\", "/")
            return ResolvedMention(
                mention_type="file",
                tag=tag,
                target=rel_path,
                content=content,
                resolved=True,
            )
        except Exception as e:
            return ResolvedMention(
                mention_type="file",
                tag=tag,
                target=file_target,
                content="",
                resolved=False,
                error_message=f"Error reading file: {e}",
            )

    def _resolve_git_mention(self, tag: str, git_cmd: str) -> ResolvedMention:
        """Extracts git status, diff, or staged changes."""
        valid_cmds = {
            "status": ["git", "status", "--short"],
            "diff": ["git", "diff"],
            "staged": ["git", "diff", "--staged"],
            "log": ["git", "log", "-n", "5", "--oneline"],
        }
        cmd = valid_cmds.get(git_cmd.lower())
        if not cmd:
            return ResolvedMention(
                mention_type="git",
                tag=tag,
                target=git_cmd,
                content="",
                resolved=False,
                error_message=f"Unsupported git mention: '{git_cmd}'. Supported: status, diff, staged, log",
            )

        if not (self.workspace_path / ".git").exists():
            return ResolvedMention(
                mention_type="git",
                tag=tag,
                target=f"git {git_cmd}",
                content="[Not a git repository / clean directory]",
                resolved=True,
            )

        try:
            res = subprocess.run(
                cmd,
                cwd=str(self.workspace_path),
                shell=(sys.platform == "win32"),
                capture_output=True,
                text=True,
                timeout=5,
            )
            out = res.stdout.strip() or "[No git output / clean working tree]"
            return ResolvedMention(
                mention_type="git",
                tag=tag,
                target=f"git {git_cmd}",
                content=out,
                resolved=True,
            )
        except Exception as e:
            return ResolvedMention(
                mention_type="git",
                tag=tag,
                target=git_cmd,
                content="",
                resolved=False,
                error_message=f"Git command failed: {e}",
            )

    def _resolve_symbol_mention(self, tag: str, symbol_name: str) -> ResolvedMention:
        """Finds symbol definition across workspace files using ripgrep or python AST."""
        try:
            from ticket_to_code.runtime.tool_registry import UnifiedToolRegistry
            registry = UnifiedToolRegistry(self.workspace_path)
            res = registry.find_symbol_definition(symbol_name)
            return ResolvedMention(
                mention_type="symbol",
                tag=tag,
                target=symbol_name,
                content=res,
                resolved=bool(res and "not found" not in res.lower()),
                error_message=None if res and "not found" not in res.lower() else f"Symbol '{symbol_name}' not found",
            )
        except Exception as e:
            return ResolvedMention(
                mention_type="symbol",
                tag=tag,
                target=symbol_name,
                content="",
                resolved=False,
                error_message=f"Symbol resolution failed: {e}",
            )

    def _resolve_dir_mention(self, tag: str, dir_path: str) -> ResolvedMention:
        """Lists directory structure for @dir:<folder>."""
        target_dir = (self.workspace_path / dir_path).resolve()
        if not target_dir.exists() or not target_dir.is_dir():
            return ResolvedMention(
                mention_type="dir",
                tag=tag,
                target=dir_path,
                content="",
                resolved=False,
                error_message=f"Directory not found: '{dir_path}'",
            )

        entries = []
        try:
            for item in sorted(target_dir.iterdir()):
                prefix = "[DIR]  " if item.is_dir() else "[FILE] "
                entries.append(f"{prefix}{item.name}")
            return ResolvedMention(
                mention_type="dir",
                tag=tag,
                target=str(target_dir.relative_to(self.workspace_path)).replace("\\", "/"),
                content="\n".join(entries[:50]),
                resolved=True,
            )
        except Exception as e:
            return ResolvedMention(
                mention_type="dir",
                tag=tag,
                target=dir_path,
                content="",
                resolved=False,
                error_message=f"Failed to list directory: {e}",
            )
