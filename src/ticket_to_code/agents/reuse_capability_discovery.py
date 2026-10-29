"""Deterministic reuse-capability discovery.

The ticket-to-code workflow repeatedly failed a class of tickets where the
correct solution is to REUSE an existing frontend service capability (e.g.
``member.service.ts`` -> ``members(filter)`` returning ``ParticipatingMember[]``)
instead of inventing a new backend method or hallucinating a service call
that exists nowhere (observed 2026-09-18: ``contractService.getProjectMembership()``
generated for a backend method with no REST endpoint and no frontend binding).

This module is pure filesystem + regex — no LLM, no index required — so it can
run as a deterministic pre-seed before planning:

    capabilities = discover_frontend_capabilities(workspace_path, frontend_root, keywords)

Each capability carries the service file, its class, and its public method
signatures, ready to be injected into planner/generator prompts with a strict
REUSE-FIRST directive.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

_FRONTEND_SERVICE_GLOB = "*.service.ts"
_SKIP_PARTS = {"node_modules", "dist", ".aviator", ".git"}

# Words that identify "backend intent" in a ticket: when present, a frontend
# ticket MAY legitimately include backend tasks (new endpoint etc.).
_BACKEND_INTENT_KEYWORDS = (
    "endpoint",
    "rest api",
    "new api",
    "api should",
    "backend",
    "controller",
    "migration",
    "database schema",
    "should return from server",
    "server should",
)

_RESERVED_WORDS = {
    "constructor", "ngOnInit", "ngOnDestroy", "if", "switch", "for", "while",
    "catch", "try", "return", "throw", "throwError", "of", "map", "filter",
    "switchMap", "pipe", "catchError", "tap", "finalize", "subscribe",
    "get", "set", "import", "export", "from", "as", "class", "interface",
}

_METHOD_RE = re.compile(
    r"""(?:^|[{;}])\s*(?:(?:public|private|protected|async|static)\s+)*([A-Za-z0-9_$]+)\s*\(([^)]*)\)\s*(?::\s*([^{;]+))?\s*\{""",
    re.MULTILINE,
)
_CLASS_RE = re.compile(r"export\s+(?:default\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)")
_CTOR_PROP_RE = re.compile(r"(?:private|public|protected)\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)")


def is_frontend_file(path: str) -> bool:
    """Heuristic: Angular/Vue/React component or service file."""
    p = path.replace("\\", "/").lower()
    return (
        p.endswith(".component.ts")
        or p.endswith(".component.html")
        or p.endswith(".component.scss")
        or p.endswith(".component.css")
        or p.endswith(".vue")
        or p.endswith(".jsx")
        or p.endswith(".tsx")
        or p.endswith(".service.ts")
        or p.endswith(".html")
        or p.endswith(".scss")
    )


def ticket_has_backend_intent(ticket_text: str) -> bool:
    t = (ticket_text or "").lower()
    return any(k in t for k in _BACKEND_INTENT_KEYWORDS)


def _service_root_for(path: str, workspace_root: Path) -> Optional[Path]:
    """Return the top-level service directory (poly-repo member) containing path."""
    p = Path(path.replace("\\", "/"))
    if p.is_absolute():
        try:
            p = p.relative_to(workspace_root)
        except ValueError:
            return None
    parts = p.parts
    return workspace_root / parts[0] if len(parts) > 1 else workspace_root


def _parse_service_file(fp: Path) -> Dict:
    try:
        text = fp.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}
    cls = _CLASS_RE.search(text)
    if not cls:
        return {}
    methods = []
    for m in _METHOD_RE.finditer(text):
        name = m.group(1)
        if name in _RESERVED_WORDS:
            continue
        methods.append({"name": name, "args": m.group(2).strip()})
    if not methods:
        return {}
    return {
        "file": str(fp),
        "class": cls.group(1),
        "methods": methods,
        "constructor_services": [c[1] for c in _CTOR_PROP_RE.findall(text)],
    }


def _keyword_score(text: str, keywords: List[str]) -> int:
    t = text.lower()
    return sum(1 for k in keywords if k and k.lower() in t)


def discover_frontend_capabilities(
    workspace_path: str,
    frontend_root: Optional[Path] = None,
    keywords: Optional[List[str]] = None,
    max_results: int = 8,
) -> List[Dict]:
    """Find existing frontend service capabilities relevant to the keywords.

    Scans ``*.service.ts`` files under frontend_root (or the whole workspace
    when not given), extracts class + public methods, and ranks by keyword
    overlap (file name, class name, method names).
    """
    ws = Path(workspace_path)
    root = Path(frontend_root) if frontend_root else ws
    if not root.exists():
        return []
    keywords = [k.strip() for k in (keywords or []) if k and k.strip()]
    scored: List[Dict] = []
    for fp in root.rglob(_FRONTEND_SERVICE_GLOB):
        if any(part in _SKIP_PARTS for part in fp.parts):
            continue
        if fp.name.endswith(".spec.ts"):
            continue
        info = _parse_service_file(fp)
        if not info:
            continue
        rel = str(fp.relative_to(ws)).replace("\\", "/")
        blob = " ".join([fp.stem, info["class"], " ".join(m["name"] for m in info["methods"])])
        score = _keyword_score(blob, keywords)
        if score <= 0:
            continue
        info["rel_path"] = rel
        info["score"] = score
        scored.append(info)
    scored.sort(key=lambda c: -c["score"])
    return scored[:max_results]


def build_reuse_directive(capabilities: List[Dict], workspace_path: Optional[str] = None) -> str:
    """Render the REUSE-FIRST prompt block for planner/generator prompts,
    hydrating parameter type contracts for discovered service methods.
    """
    if not capabilities:
        return ""
    lines = [
        "\n=== REUSE-FIRST: EXISTING CAPABILITIES (READ-ONLY) ===",
        "The ticket's data needs are already provided by these existing services.",
        "You MUST reuse them by CALLING them. You MUST NOT:",
        "  - add new methods to these service files,",
        "  - invent service methods that do not exist anywhere,",
        "  - call backend/Java services from frontend code unless a frontend",
        "    service method already wraps that call.",
        "Available capabilities:",
    ]
    for c in capabilities:
        sigs = ", ".join(f"{m['name']}({m['args']})" for m in c["methods"][:8])
        lines.append(f"  [AVAILABLE] {c['rel_path']}  class {c['class']}: {sigs}")

    # Transitive Type Hydration: Hydrate exact type contracts for discovered methods
    type_contracts = []
    try:
        from ticket_to_code.intelligence.typescript import TypeScriptTypeResolver
        for c in capabilities:
            svc_file = c.get("file")
            if not svc_file:
                continue
            for m in c.get("methods", [])[:6]:
                contract = TypeScriptTypeResolver.build_method_type_contract(
                    svc_file, m["name"], workspace_root=workspace_path
                )
                if not contract.is_empty():
                    block = contract.render_prompt_block()
                    if block and block not in type_contracts:
                        type_contracts.append(block)
    except Exception:
        pass

    if type_contracts:
        lines.append("\n" + "\n".join(type_contracts))

    lines.append("=== END REUSE-FIRST ===\n")
    return "\n".join(lines)


def frontend_anchor_root(paths: List[str], workspace_path: str) -> Optional[Path]:
    """Return the frontend service root when the ticket's anchors are frontend files.

    A ticket is 'frontend-anchored' when at least one candidate path is a
    frontend file AND that file lives under a service directory that looks like
    a frontend app (contains package.json). Backend files never match.
    """
    ws = Path(workspace_path)
    roots = set()
    for p in paths:
        if not p or not is_frontend_file(p):
            continue
        root = _service_root_for(p, ws)
        if root and (root / "package.json").exists():
            roots.add(root)
    if len(roots) == 1:
        return roots.pop()
    if roots:
        # Multiple frontend roots: prefer the one containing the most anchors.
        counts: Dict[Path, int] = {}
        for p in paths:
            if p and is_frontend_file(p):
                r = _service_root_for(p, ws)
                if r in roots:
                    counts[r] = counts.get(r, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0] if counts else None
    return None
