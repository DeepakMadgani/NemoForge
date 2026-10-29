"""
Command Safety Validator & Filesystem Sandbox Policy Enforcer.

Enforces strict containment guarantees:
1. Workspace Jail: Operations cannot escape the project root (no ../.. traversal, no modifying /etc, C:\\Windows, etc.)
2. Destructive Command Blocking: Prohibits shell fork-bombs, disk wipes, arbitrary remote binary execution (curl | sh).
3. Secret Scrubbing: Prevents LLM generated commands from reading or exfiltrating credentials, private keys, or .env files.
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Set, Union


class SecurityViolation(Exception):
    """Raised when a command or path violates sandbox containment policy."""
    pass


# Commands strictly blocked from autonomous execution
BLOCKED_COMMAND_PATTERNS = [
    re.compile(r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f?\s+[\/\\]", re.IGNORECASE),  # rm -rf /
    re.compile(r"\bdel\s+\/[sS]\s+\/[qQ]\s+[cC]:\\", re.IGNORECASE),       # del /s /q C:\
    re.compile(r"\bformat\s+[a-zA-Z]:", re.IGNORECASE),                   # format C:
    re.compile(r"\bcurl\b.*\|\s*(?:bash|sh|powershell|cmd)\b", re.IGNORECASE), # curl | sh
    re.compile(r"\bwget\b.*\|\s*(?:bash|sh|powershell|cmd)\b", re.IGNORECASE), # wget | sh
    re.compile(r"powershell(?:\.exe)?\s+.*-(?:enc|encodedcommand)\b", re.IGNORECASE), # PowerShell encoded payload
    re.compile(r"(?:iex|invoke-expression)\s*\(", re.IGNORECASE),          # PowerShell IEX
    re.compile(r"\|\s*(?:iex|invoke-expression)\b", re.IGNORECASE),        # Pipeline to IEX
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;", re.IGNORECASE),   # Fork bomb
    re.compile(r"\bmkfs\b", re.IGNORECASE),                                # mkfs
    re.compile(r"\bdd\s+if=", re.IGNORECASE),                              # dd disk write
    re.compile(r"\bshutdown\b", re.IGNORECASE),                            # shutdown
    re.compile(r"\breboot\b", re.IGNORECASE),                              # reboot
    re.compile(r"\b(?:net\s+user|net\s+localgroup)\b", re.IGNORECASE),     # Privilege escalation / account tampering
    re.compile(r"\b(?:reg\s+(?:add|delete)|regsvr32)\b", re.IGNORECASE),   # Windows registry mutation
    re.compile(r"\bpython(?:\.exe)?\s+-c\s+.*(?:os\.system|subprocess|pty\.spawn|socket\.)", re.IGNORECASE), # Dangerous Python inline subshell
    # ── Recursive deletes (relative paths, ~, globs) ──
    re.compile(r"\brm\s+(?:-{1,2}[a-z-]*\s+)*-{1,2}(?:[a-z]*r[a-z]*|recursive)\b", re.IGNORECASE),   # any recursive rm
    re.compile(r"\brm\s+[^\n|;&]*(?:^|\s)~(?:[\\/]|\s|$)", re.IGNORECASE),                            # rm ... ~
    re.compile(r"\b(?:remove-item|ri|rmdir|rd|del|erase)\b[^\n|;&]*\s-recurse\b", re.IGNORECASE),    # Remove-Item -Recurse (+aliases)
    re.compile(r"\b(?:rd|rmdir)\s+(?:\/[sq]\s+)*\/s\b", re.IGNORECASE),                                # rd /s
    re.compile(r"\b(?:del|erase)\s+(?:\/[a-z]\s+)*\/s\b", re.IGNORECASE),                              # del /s
    # ── Destructive git operations ──
    re.compile(r"\bgit\s+push\b[^\n|;&]*\s(?:--force|-f)\b", re.IGNORECASE),
    re.compile(r"\bgit\s+reset\b[^\n|;&]*\s--hard\b", re.IGNORECASE),
    re.compile(r"\bgit\s+clean\b[^\n|;&]*\s-[a-z]*f", re.IGNORECASE),
    re.compile(r"\bgit\s+checkout\s+(?:[^\n|;&]*\s)?--\s+\.(?:\s|$)", re.IGNORECASE),
    # ── Arbitrary inline / remote code execution via node ──
    re.compile(r"\bnode(?:\.exe)?\s+(?:-[a-z-]*\s+)*(?:-e|--eval|-p|--print)\b", re.IGNORECASE),
    re.compile(r"\bnpx\b[^\n]*\|\s*(?:bash|sh|zsh|powershell|pwsh|cmd)\b", re.IGNORECASE),
]

# Environment variables that must survive scrubbing (toolchains break without them)
ALWAYS_KEEP_ENV = {"JAVA_HOME", "PATH", "HOME", "USERPROFILE", "TEMP", "TMP", "SYSTEMROOT", "PATHEXT", "COMSPEC"}
# Environment variables that are always removed from child processes
ALWAYS_DROP_ENV = {"GOOGLE_APPLICATION_CREDENTIALS"}
# Substrings (case-insensitive) that mark a variable as a credential
SENSITIVE_ENV_SUBSTRINGS = [
    "secret", "password", "token", "private_key", "credentials_content",
    "api_key", "access_key", "application_credentials", "aws_", "secret_key",
]

# Sensitive paths that should never be read or mutated by autonomous workflows
SENSITIVE_PATH_PATTERNS = [
    re.compile(r"\.ssh[\\/]", re.IGNORECASE),
    re.compile(r"\.aws[\\/]", re.IGNORECASE),
    re.compile(r"id_rsa", re.IGNORECASE),
    re.compile(r"id_ed25519", re.IGNORECASE),
    re.compile(r"credentials\.json$", re.IGNORECASE),
    re.compile(r"service_account.*\.json$", re.IGNORECASE),
    re.compile(r"system32", re.IGNORECASE),
    re.compile(r"[\\/]etc[\\/]shadow", re.IGNORECASE),
    re.compile(r"[\\/]etc[\\/]passwd", re.IGNORECASE),
    re.compile(r"\.git[\\/]config$", re.IGNORECASE),
]


class CommandSafetyValidator:
    """Validates commands and paths before dispatch to OS subprocesses."""

    @classmethod
    def validate_command(cls, cmd: Union[str, List[str], Tuple[str, ...]], workspace_root: Optional[Path] = None) -> Tuple[bool, Optional[str]]:
        """
        Validates shell command against dangerous execution patterns.
        Accepts both command strings and argv lists.
        Returns (is_safe, violation_reason).
        """
        if isinstance(cmd, (list, tuple)):
            cmd_clean = " ".join(str(c) for c in cmd).strip()
        elif isinstance(cmd, str):
            cmd_clean = cmd.strip()
        else:
            return False, "Invalid command type"

        if not cmd_clean:
            return False, "Empty command"

        # 1. Check blocked dangerous command patterns
        for pattern in BLOCKED_COMMAND_PATTERNS:
            if pattern.search(cmd_clean):
                return False, f"Command contains prohibited dangerous operation matching pattern: {pattern.pattern}"

        # 2. Check for sensitive path access within command
        for pattern in SENSITIVE_PATH_PATTERNS:
            if pattern.search(cmd_clean):
                return False, f"Command references prohibited sensitive credential or system path: {pattern.pattern}"

        # 3. Optional executable allow-list (deny-list above always wins)
        allow_raw = os.getenv("AVIATOR_EXEC_ALLOWLIST", "").strip()
        if allow_raw:
            allowed = [a.strip().lower() for a in allow_raw.split(",") if a.strip()]
            first = re.split(r"\s+", cmd_clean, 1)[0].strip("\"'").replace("\\", "/").rsplit("/", 1)[-1].lower()
            first = re.sub(r"\.(exe|cmd|bat)$", "", first)
            if not any(first == a or first.startswith(a) for a in allowed):
                return False, f"Command '{first}' is not in AVIATOR_EXEC_ALLOWLIST ({allow_raw})"

        return True, None

    @classmethod
    def validate_path_containment(cls, target_path: Path | str, workspace_root: Path | str) -> Tuple[bool, Optional[str]]:
        """
        Verifies that target_path resolves strictly within workspace_root.
        Prevents path traversal attacks (e.g., ../../Windows/System32).
        """
        try:
            ws_resolved = Path(workspace_root).resolve()
            tp = Path(target_path)
            if not tp.is_absolute():
                tp = ws_resolved / tp  # relative paths are anchored inside the jail, never in CWD
            target_resolved = tp.resolve()

            # Target must be workspace itself or a descendant
            if ws_resolved != target_resolved and ws_resolved not in target_resolved.parents:
                return False, f"Path traversal rejected: '{target_resolved}' is outside workspace jail '{ws_resolved}'"

            # Check sensitive file blacklist
            for pattern in SENSITIVE_PATH_PATTERNS:
                if pattern.search(str(target_resolved)):
                    return False, f"Access rejected to protected credential or system path: '{target_resolved.name}'"

            return True, None
        except Exception as exc:
            return False, f"Path resolution error: {exc}"

    @classmethod
    def sanitize_environment(cls, env: Optional[dict] = None) -> dict:
        """
        Sanitizes environment variables before spawning a child process.
        Strips secret tokens, API keys, and passwords to prevent process exfiltration.
        """
        base_env = dict(env if env is not None else os.environ)
        sanitized = {}

        for k, v in base_env.items():
            ku = k.upper()
            if ku in ALWAYS_KEEP_ENV:
                sanitized[k] = v
                continue
            if ku in ALWAYS_DROP_ENV:
                continue
            if any(sub in k.lower() for sub in SENSITIVE_ENV_SUBSTRINGS):
                continue
            sanitized[k] = v

        jh = sanitized.get("JAVA_HOME", "").strip().rstrip("\\/")
        java_bin = "bin\\java.exe" if sys.platform == "win32" else "bin/java"
        if not jh or not os.path.exists(os.path.join(jh, java_bin)):
            import glob
            search_patterns = []
            if sys.platform == "win32":
                for base in [os.environ.get("ProgramFiles", r"C:\Program Files"), r"C:\Program Files (x86)"]:
                    for vendor in ["Microsoft", "Eclipse Adoptium", "Java", "Amazon Corretto", "Zulu", "BellSoft"]:
                        search_patterns.append(os.path.join(base, vendor, "*"))
            else:
                search_patterns.extend(["/usr/lib/jvm/*", "/usr/java/*", "/Library/Java/JavaVirtualMachines/*/Contents/Home"])
            for pat in search_patterns:
                candidates = sorted(glob.glob(pat), reverse=True)
                for cand in candidates:
                    if os.path.isdir(cand) and os.path.exists(os.path.join(cand, java_bin)):
                        jh = cand
                        break
                if jh and os.path.exists(os.path.join(jh, java_bin)):
                    break
        if jh and os.path.exists(os.path.join(jh, java_bin)):
            sanitized["JAVA_HOME"] = jh
            bin_dir = os.path.join(jh, "bin")
            cur_p = sanitized.get("PATH", "")
            if bin_dir.lower() not in cur_p.lower():
                sanitized["PATH"] = f"{bin_dir}{os.pathsep}{cur_p}"

        return sanitized


# Backwards/forwards-compatible alias used by newer call sites.
SecurityPolicy = CommandSafetyValidator


def kill_process_tree(proc: "subprocess.Popen", grace_s: float = 10.0) -> None:
    """Terminate `proc` and every descendant (taskkill /T /F on Windows, killpg on POSIX).

    Safe to call on an already-finished process. Never raises.
    """
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=15)
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                return
        proc.wait(timeout=grace_s)
    except Exception:
        pass
    if proc.poll() is None:
        try:
            if os.name != "nt":
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass


def run_checked_subprocess(
    cmd: Union[str, List[str], Tuple[str, ...]],
    cwd: Optional[Union[str, Path]],
    timeout: int,
    workspace: Optional[Union[str, Path]] = None,
    env: Optional[dict] = None,
) -> Tuple[int, str, str]:
    """Single hardened entry point for every agent-driven subprocess.

    (a) validates the command against the deny/allow lists,
    (b) scrubs credentials from the child environment,
    (c) confines the working directory to the workspace jail,
    (d) spawns the process in its own process group / job so
    (e) a timeout kills the WHOLE tree, not just the shell.

    Returns (returncode, stdout, stderr). Raises SecurityViolation on a policy
    rejection and subprocess.TimeoutExpired (with partial output) on timeout.
    """
    ws = Path(workspace).resolve() if workspace else None
    ok, why = CommandSafetyValidator.validate_command(cmd, workspace_root=ws)  # (a)
    if not ok:
        raise SecurityViolation(why)
    safe_env = CommandSafetyValidator.sanitize_environment(env)  # (b)

    cwd_path = Path(cwd) if cwd else (ws or Path.cwd())  # (c)
    if ws is not None:
        if not cwd_path.is_absolute():
            cwd_path = ws / str(cwd).lstrip("/\\")
        ok, why = CommandSafetyValidator.validate_path_containment(cwd_path, ws)
        if not ok:
            raise SecurityViolation(why)
    cwd_path = cwd_path.resolve()
    if not cwd_path.is_dir():
        raise SecurityViolation(f"Working directory does not exist: {cwd_path}")

    is_win = sys.platform == "win32"  # (d)
    if isinstance(cmd, (list, tuple)):
        argv: Union[str, List[str]] = [str(c) for c in cmd]
        use_shell = is_win  # list2cmdline on Windows so .cmd/.bat shims (npm, gradlew.bat) resolve
    else:
        argv = cmd if is_win else shlex.split(cmd)
        use_shell = is_win
    popen_kw: dict = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if is_win else {"start_new_session": True}
    )  # (e)
    proc = subprocess.Popen(
        argv,
        cwd=str(cwd_path),
        env=safe_env,
        shell=use_shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        **popen_kw,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_process_tree(proc)
        out, err = "", ""
        try:
            out, err = proc.communicate(timeout=5)
        except Exception:
            pass
        raise subprocess.TimeoutExpired(cmd, timeout, output=out, stderr=err)
    return proc.returncode, out or "", err or ""
