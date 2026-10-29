"""Deterministic ticket triage and the physical scope fence.

Classification is lexical (scored keyword signals), never a model call. The fence is a
filesystem gate every read/grep/glob passes through, so a UI ticket physically cannot open
a backend file — the gate raises before the file is read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

Layer = Literal["ui", "frontend_logic", "backend", "fullstack", "unknown"]

# (compiled pattern, weight). Highest total score wins; ties resolve to the narrowest layer.
_LAYER_SIGNALS: Dict[str, List[Tuple[re.Pattern, int]]] = {
    "ui": [
        (re.compile(r"\b(column|grid|table|tab|label|tooltip|icon|badge|layout|css|scss|style|"
                    r"reorder|re-order|order|align|placeholder|header)\b", re.I), 3),
        (re.compile(r"\b(button|dropdown|modal|dialog|menu|screen|page|view)\b", re.I), 1),
    ],
    "frontend_logic": [
        (re.compile(r"\b(component|directive|pipe|observable|rxjs|router|guard|form\s?control)\b", re.I), 2),
        (re.compile(r"\b(typescript|angular|\.ts\b)\b", re.I), 1),
    ],
    "backend": [
        (re.compile(r"\b(endpoint|api|controller|repository|migration|sql|schema|entity|"
                    r"service\s?impl|saga|kafka|queue|dao|jpa|hibernate)\b", re.I), 3),
        (re.compile(r"\b(java|spring|maven|gradle|\.java\b)\b", re.I), 1),
        # Java-style class identifiers named in a ticket (…Impl, …Exception, …Controller, …Service,
        # …Repository, …Entity, …DTO) are a strong backend signal that generic words like 'guard' lack.
        (re.compile(r"\b[A-Z]\w+(?:Impl|Exception|Controller|Repository|Entity|DTO|Dao)\b"), 3),
    ],
}

_NARROWNESS = {"ui": 0, "frontend_logic": 1, "backend": 2, "fullstack": 3}

# Per-layer allow/deny globs (relative to the chosen sub-project root).
_LAYER_POLICY: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "ui": {
        "allow": ("src/**", "projects/**"),
        "deny": ("**/*.java", "**/*.sql", "**/*.yml", "**/*.yaml", "**/*.properties",
                 "**/pom.xml", "**/*.gradle", "**/target/**", "**/*Service*/**/*.java"),
    },
    "frontend_logic": {
        "allow": ("src/**", "projects/**"),
        "deny": ("**/*.java", "**/*.sql", "**/target/**"),
    },
    "backend": {
        "allow": ("src/**", "**/*.java", "**/*.sql", "**/*.yml", "**/pom.xml"),
        "deny": ("**/node_modules/**", "**/dist/**"),
    },
    "fullstack": {
        "allow": ("src/**", "**/*.java", "**/*.sql", "**/*.yml", "**/*.yaml", "**/*.ts", "**/*.html", "**/*.scss", "**/pom.xml", "**/package.json"),
        "deny": ("**/node_modules/**", "**/dist/**", "**/target/**", "**/.gradle/**"),
    },
}


class ScopeViolation(RuntimeError):
    """Raised when a tool tries to touch a path outside the locked scope."""


@dataclass(frozen=True)
class ScopeContract:
    ticket_id: str
    layer: Layer
    confidence: float
    root: str                              # absolute sub-project root the run is fenced to
    allow_globs: Tuple[str, ...]
    deny_globs: Tuple[str, ...]
    anchors: Tuple[str, ...]               # exact strings pulled from the ticket
    screen_hint: str = ""
    screen_noun: str = ""                   # feature noun (e.g. 'deliverable') for path affinity
    grid_hint: bool = False                 # true only for grid/column tickets (gates the table boost)
    ticket_text: str = ""                   # raw ticket text; ranks which service methods are relevant

    def relative(self, target_path: str) -> Optional[str]:
        try:
            rel = Path(target_path).resolve().relative_to(Path(self.root).resolve())
        except (ValueError, OSError):
            return None
        return rel.as_posix()


# ── anchor extraction ────────────────────────────────────────────────────────

_QUOTED = re.compile(r"[\"'“‘]([^\"'”’]{2,60})[\"'”’]")
_TITLE_SEQ = re.compile(r"\b([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+){0,3})\b")
# High-signal code identifiers named in a ticket pin the exact owner: a camelCase method
# (deleteCoordinator) or a dotted key/filename (project.coordinator.delete.half.msg) is rare, so
# idf weighting makes it dominate over generic words shared by many files.
_IDENT = re.compile(r"\b([a-z][a-z0-9]*[A-Z][A-Za-z0-9]{2,})\b")
_DOTTED = re.compile(r"\b([a-z][\w-]*(?:\.[a-z][\w-]*){2,})\b")
_STOP = {"the", "this", "that", "grid", "tab", "page", "column", "columns", "order", "should", "must"}


# Word immediately before a UI container noun names the screen: 'Deliverables tab grid' -> deliverable.
_SCREEN_NOUN = re.compile(r"\b([A-Za-z][A-Za-z-]{2,})\s+(?:tab|grid|page|screen|table|view|list|panel|modal|dialog)\b", re.I)


def screen_noun(text: str) -> str:
    m = _SCREEN_NOUN.search(text)
    if not m:
        return ""
    noun = m.group(1).lower()
    return noun[:-1] if noun.endswith("s") else noun   # singular stem for path matching


def extract_anchors(text: str) -> Tuple[str, ...]:
    """Exact string anchors (quoted phrases + Title-Case column tokens), de-duplicated, order-stable."""
    found: List[str] = []
    seen = set()

    def add(tok: str) -> None:
        tok = tok.strip()
        key = tok.lower()
        if 2 <= len(tok) <= 60 and key not in seen and key not in _STOP:
            seen.add(key)
            found.append(tok)

    for m in _QUOTED.finditer(text):
        add(m.group(1))
    for m in _TITLE_SEQ.finditer(text):
        add(m.group(1))
    for m in _IDENT.finditer(text):
        add(m.group(1))
    for m in _DOTTED.finditer(text):
        add(m.group(1))
    return tuple(found)


# ── sub-project resolution ───────────────────────────────────────────────────

def _detect_subprojects(workspace: Path) -> List[Path]:
    """Direct children that look like deployable projects (own package.json / build file)."""
    markers = ("package.json", "angular.json", "pom.xml", "build.gradle", "pyproject.toml", "*.csproj")
    out: List[Path] = []
    if any((workspace / m).exists() for m in markers if "*" not in m):
        out.append(workspace)
    for child in sorted(p for p in workspace.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if any((child / m).exists() for m in markers if "*" not in m) or (
            "*.csproj" in markers and list(child.glob("*.csproj"))
        ):
            out.append(child)
    return out


def _pick_root(workspace: Path, layer: Layer, text: str, explicit: Optional[str]) -> Path:
    # explicit is the search ceiling, not necessarily the final root: a monorepo path still
    # resolves down to the matching sub-project so allow-globs like src/** line up.
    base = Path(explicit).resolve() if explicit else workspace
    subs = _detect_subprojects(base)
    if not subs:
        return base.resolve()
    low = text.lower()
    # Prefer a sub-project named in the ticket text.
    for s in subs:
        if s.name.lower() in low and s != base:
            return s.resolve()
    if layer in ("ui", "frontend_logic"):
        for s in subs:
            if (s / "angular.json").exists() or (s / "package.json").exists():
                return s.resolve()
    if layer == "backend":
        for s in subs:
            if (s / "pom.xml").exists() or (s / "build.gradle").exists():
                return s.resolve()
    if layer == "fullstack":
        return base.resolve()
    return (subs[0] if subs else base).resolve()


def classify_layer(text: str) -> Tuple[Layer, float]:
    scores: Dict[str, int] = {}
    for layer, signals in _LAYER_SIGNALS.items():
        scores[layer] = sum(w for pat, w in signals if pat.search(text))
    top = max(scores.values()) if scores else 0
    if top == 0:
        return "unknown", 0.0
    winners = sorted((l for l, s in scores.items() if s == top), key=lambda l: _NARROWNESS[l])
    # Backend + UI both strongly present → genuinely full-stack.
    if scores.get("backend", 0) >= 3 and scores.get("ui", 0) >= 3:
        return "fullstack", 0.5
    layer = winners[0]  # type: ignore[assignment]
    runner_up = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
    confidence = min(1.0, 0.5 + 0.1 * (top - runner_up) + 0.1 * top)
    return layer, round(confidence, 2)  # type: ignore[return-value]


def triage(ticket_id: str, title: str, description: str,
           workspace_path: str, explicit_root: Optional[str] = None,
           labels: Optional[List[str]] = None) -> ScopeContract:
    """Build the ScopeContract with zero model calls."""
    text = " ".join(filter(None, [title, description, " ".join(labels or [])]))
    layer, confidence = classify_layer(text)
    root = _pick_root(Path(workspace_path), layer, text, explicit_root)
    policy = _LAYER_POLICY.get(layer, {"allow": ("**",), "deny": ()})
    return ScopeContract(
        ticket_id=ticket_id,
        layer=layer,
        confidence=confidence,
        root=str(root),
        allow_globs=policy["allow"],
        deny_globs=policy["deny"],
        anchors=extract_anchors(text),
        screen_hint=(extract_anchors(title)[:1] or ("",))[0],
        screen_noun=screen_noun(f"{title} {description}"),
        grid_hint=bool(re.search(r"\b(grid|columns?|reorder|re-order|cdkColumnDef)\b", text, re.I)),
        ticket_text=text[:4000],
    )


# ── the physical fence ───────────────────────────────────────────────────────

@dataclass
class ScopeFence:
    scope: ScopeContract
    reads: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)

    def check(self, target_path: str) -> str:
        rel = self.scope.relative(target_path)
        if rel is None:
            self.violations.append(target_path)
            raise ScopeViolation(f"{target_path} is outside scope root {self.scope.root}")
        if any(fnmatch(rel, g) for g in self.scope.deny_globs):
            self.violations.append(target_path)
            raise ScopeViolation(f"{target_path} blocked by deny policy for layer '{self.scope.layer}'")
        if self.scope.allow_globs and not any(fnmatch(rel, g) for g in self.scope.allow_globs):
            self.violations.append(target_path)
            raise ScopeViolation(f"{target_path} not in allow policy for layer '{self.scope.layer}'")
        self.reads.append(rel)
        return target_path

    def allows(self, target_path: str) -> bool:
        try:
            self.check(target_path)
            return True
        except ScopeViolation:
            return False
