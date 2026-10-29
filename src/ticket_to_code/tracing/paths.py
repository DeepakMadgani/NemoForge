"""Single source of truth for the trace directory.

Resolution order:
  1. TICKET_TO_CODE_TRACE_DIR environment variable.
  2. <workspace>/.nemoforge/traces when a workspace path is supplied.
  3. The legacy Windows location C:/aviator_traces when it already exists
     (keeps existing local installs working).
  4. ~/.nemoforge/traces.

All callers that previously hard-coded "C:/aviator_traces" go through here so
the code runs on Linux/macOS and inside containers.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

_LEGACY_WINDOWS_DIR = Path("C:/aviator_traces")


def get_trace_root(workspace: Optional[Union[str, Path]] = None) -> Path:
    env = os.environ.get("TICKET_TO_CODE_TRACE_DIR", "").strip()
    if env:
        return Path(env)
    if workspace:
        return Path(workspace) / ".nemoforge" / "traces"
    if os.name == "nt" and _LEGACY_WINDOWS_DIR.is_dir():
        return _LEGACY_WINDOWS_DIR
    return Path.home() / ".nemoforge" / "traces"


def get_trace_dir(ticket_id: str, workspace: Optional[Union[str, Path]] = None, create: bool = False) -> Path:
    """Trace directory for one ticket."""
    path = get_trace_root(workspace) / str(ticket_id)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
