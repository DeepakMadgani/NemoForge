"""Utilities for normalizing terminal/build error output."""

from __future__ import annotations

import re
from typing import Iterable, List


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_ERROR_LOCATION_RE = re.compile(r"(:\d+(?::\d+)?|\[\d+,\d+\])")

# Noise patterns that inflate token counts without providing actionable compiler error signals
_NOISE_LINE_PATTERNS = [
    re.compile(r"^\s*Deprecation Warning.*", re.IGNORECASE),
    re.compile(r".*Sass @import rules are deprecated.*", re.IGNORECASE),
    re.compile(r".*More info and automated migrator.*", re.IGNORECASE),
    re.compile(r".*root stylesheet.*", re.IGNORECASE),
    re.compile(r".*sass-loader.*", re.IGNORECASE),
    re.compile(r"^\s*npm (?:warn|notice).*", re.IGNORECASE),
    re.compile(r".*CommonJS or AMD dependencies can cause optimization bailouts.*", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*\|\s*@import.*", re.IGNORECASE),
    re.compile(r"^[│╵╷|\s\^~-]+$"),
    re.compile(r"^\s*[│╵╷].*"),
    re.compile(r"^\s*\d+\s*[│|].*"),
    re.compile(r".*https://sass-lang.com.*", re.IGNORECASE),
]


def strip_ansi(text: str) -> str:
    """Remove ANSI terminal escape sequences from text."""
    if not text:
        return ""
    return _ANSI_ESCAPE_RE.sub("", text)


def normalize_error_lines(lines: Iterable[str]) -> List[str]:
    """
    Return non-empty, ANSI-free diagnostic lines with pure compiler/build noise stripped.
    Drastically reduces token costs during build error resolution passes.
    """
    cleaned: List[str] = []
    for line in lines:
        plain = strip_ansi(line)
        if not plain or not plain.strip():
            continue
        trimmed = plain.strip()
        # Filter out repetitive Sass deprecation warnings and general npm notices
        if any(p.search(trimmed) for p in _NOISE_LINE_PATTERNS):
            continue
        cleaned.append(trimmed)
    return cleaned


def normalize_error_identity(error_line: str) -> str:
    """Normalize one diagnostic line for location-insensitive comparison."""
    plain = strip_ansi(error_line)
    return _ERROR_LOCATION_RE.sub("", plain).strip()
