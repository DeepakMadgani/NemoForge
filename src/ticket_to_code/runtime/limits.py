"""Single source of truth for harness truncation limits.

Every tool/observation/snippet cap in the codebase imports from here so the
limits stop drifting between the ~7 copies of each tool. All values can be
overridden with environment variables of the same name prefixed AVIATOR_.
"""
from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(f"AVIATOR_{name}", "").strip()
    if raw.isdigit():
        return int(raw)
    return default


# Max characters of a single tool result returned to the model (search, read, run_command).
TOOL_OUTPUT_MAX_CHARS: int = _env_int("TOOL_OUTPUT_MAX_CHARS", 8000)
# Max lines kept from a search / grep style tool.
SEARCH_MAX_LINES: int = _env_int("SEARCH_MAX_LINES", 100)
# Max characters of one observation kept in a ReAct step history entry (older steps).
OBSERVATION_MAX_CHARS: int = _env_int("OBSERVATION_MAX_CHARS", 2000)
# Max characters of the MOST RECENT read/view observation kept verbatim so the model can edit against it.
LATEST_READ_MAX_CHARS: int = _env_int("LATEST_READ_MAX_CHARS", 12000)
# Max characters per evidence snippet shown to the planner.
SNIPPET_MAX_CHARS: int = _env_int("SNIPPET_MAX_CHARS", 800)
# Max characters of compiler/test output fed back into a repair turn.
BUILD_OUTPUT_MAX_CHARS: int = _env_int("BUILD_OUTPUT_MAX_CHARS", 6000)
# Max output tokens for a targeted SEARCH/REPLACE edit vs a whole-file generation.
EDIT_MAX_TOKENS: int = _env_int("EDIT_MAX_TOKENS", 1024)
WHOLE_FILE_MAX_TOKENS: int = _env_int("WHOLE_FILE_MAX_TOKENS", 8192)


def truncate(text: str, limit: int, marker: str = "\n... [truncated {n} chars]") -> str:
    """Cut `text` to `limit` chars with an explicit marker so the model knows it is partial."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + marker.format(n=len(text) - limit)
