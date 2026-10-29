"""Lean path orchestrator: Triage → Locate → Lock → Patch → Verify ladder → Converge.

One-way doors: once files are locked, discovery/planning are unreachable. The repair loop can
only route Patch↔Verify, carries only the locked context + last diff + structured failure, and
halts on convergence (repeat/oscillating signature), not on an arbitrary call count.

The LLM patcher is injected (PatchFn) so the deterministic spine is fully testable offline.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)

# Fast in-memory cache for call-chain traces across locate / lock cycles
_TRACE_CACHE: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}

from ticket_to_code.lean.convergence import Action, ConvergenceTracker, StructuredFailure
from ticket_to_code.lean.hunks import ApplyResult, Hunk, plan_hunks, write_result
from ticket_to_code.lean.locate import Candidate, companions, locate
from ticket_to_code.lean.scope import ScopeContract, ScopeFence, triage


class ClarifyRequest(Exception):
    """Raised by the patcher when the exact API/decision needed isn't in context — the run stops and
    asks the user instead of inventing a method name that compiles but is wrong at runtime."""

    def __init__(self, question: str):
        super().__init__(question)
        self.question = question


@dataclass(frozen=True)
class LockedContext:
    """Immutable after Locate. Discovery/planning cannot reopen it."""
    scope: ScopeContract
    files: tuple                      # relative paths
    framework_facts: tuple
    baselines: Dict[str, str]         # rel -> original content
    dependency_context: str = ""      # read-only API reference: method signatures + exact import paths


@dataclass
class VerifyOutcome:
    ok: bool
    failure: Optional[StructuredFailure] = None
    rung: str = ""


class PatchFn(Protocol):
    def __call__(self, ctx: LockedContext, last_diff: str,
                 failure: Optional[StructuredFailure]) -> List[Hunk]: ...


class VerifyFn(Protocol):
    def __call__(self, ctx: LockedContext, written: List[str]) -> VerifyOutcome: ...


@dataclass
class TokenMeter:
    """Observability + backstop only. It does NOT cut the model off mid-work; convergence does."""
    tokens: int = 0
    llm_calls: int = 0
    hard_token_ceiling: int = 2_000_000   # generous; trips only on genuine runaway
    events: List[str] = field(default_factory=list)

    def add(self, tokens: int, calls: int = 1, label: str = "") -> None:
        self.tokens += tokens
        self.llm_calls += calls
        self.events.append(f"{label}:{tokens}t/{calls}c")
        if self.tokens > self.hard_token_ceiling:
            raise RuntimeError(f"token ceiling {self.hard_token_ceiling} exceeded ({label})")


@dataclass
class LeanResult:
    status: str                       # verified | unverified | out_of_scope | no_candidates | error
    scope: Optional[ScopeContract] = None
    locked_files: List[str] = field(default_factory=list)
    attempts: int = 0
    reason: str = ""
    tokens: int = 0
    llm_calls: int = 0
    elapsed_s: float = 0.0
    trace: List[str] = field(default_factory=list)


def should_use_lean(scope: ScopeContract) -> bool:
    """Lean path claims clearly-scoped single-layer tickets with anchors."""
    return (
        scope.layer in ("ui", "frontend_logic", "backend")
        and scope.confidence >= 0.5
        and len(scope.anchors) >= 1
    )


def load_framework_facts(scope: ScopeContract) -> tuple:
    """Domain rules that stop the model guessing. Discovered, not hardcoded per ticket:
      1. a repo-authored facts file (.aviator/framework_facts.md) if present, then
      2. known component adapters detected from the code (e.g. a numeric-splice table)."""
    facts: List[str] = []
    try:
        repo_facts = Path(scope.root) / ".aviator" / "framework_facts.md"
        if repo_facts.is_file():
            txt = repo_facts.read_text(encoding="utf-8", errors="ignore").strip()
            if txt:
                facts.append(txt[:1500])
    except OSError:
        pass
    # Adapter: a shared table component that positions columns by a numeric column-def spliced
    # into the column array. Detected structurally, so it only fires where that pattern exists.
    try:
        for p in Path(scope.root).rglob("*table*.component.ts"):
            code = p.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"splice\(\s*\+?\w*[Dd]ef?\.?\w*\s*,\s*0\s*,", code) or "splice(+cDef.name" in code:
                facts.append(
                    f"COLUMN ORDERING CONTRACT for the shared table '{p.name}':\n"
                    "- It builds a column array from the bound columns list (the component's displayed-columns "
                    "array, e.g. select, then status, name, list, ...), then inserts each column whose "
                    "<ng-container> has a NUMERIC id by splicing that id as an index into the array.\n"
                    "- The numeric-id columns are processed in the order their <ng-container> blocks appear in "
                    "the template, and each splice shifts the positions of everything after it.\n"
                    "- Therefore a column's final visual position depends on BOTH its numeric id AND the order of "
                    "the <ng-container> blocks. To reach a target order you must: (a) set each movable column's "
                    "numeric id to its intended final index, and (b) order the <ng-container> blocks by ASCENDING "
                    "numeric id so earlier splices settle before later ones. Static columns (those without a "
                    "numeric id, e.g. Name) occupy the remaining positions in the displayed-columns array order.\n"
                    "- Reordering only the HTML blocks, or only the numeric ids, WITHOUT keeping blocks in ascending "
                    "id order, produces a wrong visual order."
                )
                break
    except OSError:
        pass
    return tuple(facts)


def build_locked_context(scope: ScopeContract, cands: List[Candidate], fence: ScopeFence) -> LockedContext:
    """Lock the top candidate's component group, plus the support files it depends on (the i18n
    translation file and shared model it references) so a feature that spans directories — e.g. a
    component that needs a new translation key — can be completed in one locked context."""
    locked_rel: List[str] = []
    top = cands[0]
    top_dir = Path(top.path).parent
    group_paths = [top.path] + companions(top)
    for c in cands[1:4]:
        if Path(c.path).parent == top_dir:
            group_paths += [c.path] + companions(c)
    for comp in group_paths:
        rel = scope.relative(comp)
        if rel and rel not in locked_rel and fence.allows(comp):
            locked_rel.append(rel)
    # Multi-directory scoping: labels/badges live in an i18n file and types in a shared model, both
    # outside the component directory. Add the specific support files the locked component references.
    for extra in _discover_support_files(scope, locked_rel, fence):
        if extra not in locked_rel:
            locked_rel.append(extra)
    baselines = {rel: (Path(scope.root) / rel).read_text(encoding="utf-8", errors="ignore") for rel in locked_rel}
    return LockedContext(scope=scope, files=tuple(locked_rel),
                         framework_facts=load_framework_facts(scope), baselines=baselines,
                         dependency_context=build_dependency_context(scope, locked_rel, fence))


_TRANSLATE_KEY = re.compile(
    r"'([A-Za-z][\w.\-]*\.[\w.\-]+)'\s*\|\s*translate"          # 'a.b' | translate
    r"|\.instant\(\s*'([A-Za-z][\w.\-]*\.[\w.\-]+)'"            # translateService.instant('a.b')
    r"|translate\(\s*'([A-Za-z][\w.\-]*\.[\w.\-]+)'"            # translate('a.b')
)


def _discover_support_files(scope: ScopeContract, component_rel: List[str], fence: ScopeFence) -> List[str]:
    """Find the i18n translation file the locked component uses (best translation-key overlap), so a
    cross-directory edit that must add a label/badge key can be completed inside one lock. Models are
    left out deliberately: a component extends its row type inline, and locking the wrong model adds
    noise, so only the translation file — the real out-of-directory dependency — is added."""
    root = Path(scope.root)
    keys: set = set()
    for rel in component_rel:
        try:
            text = (root / rel).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in _TRANSLATE_KEY.finditer(text):
            k = m.group(1) or m.group(2) or m.group(3)
            if k:
                keys.add(k)
    i18n_root = root / "src" / "assets" / "i18n"
    if not keys or not i18n_root.is_dir():
        return []
    best, best_hits = None, 0
    for jf in i18n_root.rglob("en.json"):
        try:
            content = jf.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hits = sum(1 for k in keys if f'"{k}"' in content)
        if hits > best_hits:
            best, best_hits = jf, hits
    if best is not None and best_hits >= 1 and fence.allows(str(best)):
        rel = scope.relative(str(best))
        if rel:
            return [rel]
    return []


# ── Industry-standard AST & NLP package bindings (Tree-sitter, NLTK, rank-bm25) ──
try:
    from tree_sitter import Language as _TSLanguage, Parser as _TSParser
    import tree_sitter_typescript as _ts_mod
    _TS_LANG = _TSLanguage(_ts_mod.language_typescript())
    _TS_PARSER = _TSParser(_TS_LANG)
except Exception:
    _TS_PARSER = None

try:
    from tree_sitter import Language as _JavaLanguage, Parser as _JavaParser
    import tree_sitter_java as _java_mod
    _JAVA_LANG = _JavaLanguage(_java_mod.language())
    _JAVA_PARSER = _JavaParser(_JAVA_LANG)
except Exception:
    _JAVA_PARSER = None

try:
    from nltk.stem import PorterStemmer as _PorterStemmer
    _STEMMER = _PorterStemmer()
except Exception:
    _STEMMER = None

try:
    from nltk.corpus import stopwords as _nltk_sw
    _NLTK_STOP = set(_nltk_sw.words("english"))
except Exception:
    _NLTK_STOP = set()

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
except Exception:
    _BM25Okapi = None

_IMPORT_RE = re.compile(r"import\s*\{([^}]*)\}\s*from\s*'([^']+)'")
# A class-body method declaration: name(args) [: Return] {  — excludes control-flow keywords.
# The argument list may contain one level of `{...}` (e.g. `payload: {}` or `b: {x: number}`): those
# generic-payload methods are exactly the ones whose signatures need surfacing.
_METHOD_RE = re.compile(
    r"^[ \t]*(?:public\s+|async\s+)*"
    r"(?!(?:private|protected|constructor|ngOnInit|ngOnDestroy|ngAfterViewInit|ngOnChanges|if|for|while|switch|return|get|set|else|catch)\b)"
    r"([a-zA-Z_]\w*)\s*\(((?:[^;{}()]|\{[^{}]*\})*?)\)\s*(?::\s*([^;{]+?))?\s*\{",
    re.M,
)

# ── Java extraction (DTO getters + Spring service/interface method signatures) ──
_JAVA_IMPORT_RE = re.compile(r"import\s+(?:static\s+)?([\w.]+)\s*;")
_JAVA_METHOD_RE = re.compile(
    r"(?:public|protected)\s+(?:static\s+|final\s+|synchronized\s+|abstract\s+|default\s+|native\s+)*"
    r"([\w.$]+(?:<[^>{};=]*>)?(?:\[\])?)\s+(\w+)\s*\(([^;{)]*)\)\s*(?:throws\s+[\w., ]+)?\s*[{;]"
)
_JAVA_FIELD_RE = re.compile(
    r"(?:private|protected|public)\s+(?:static\s+|final\s+)*"
    r"([\w.$]+(?:<[^>{};=]*>)?(?:\[\])?)\s+(\w+)\s*[=;]"
)
# External packages we can't (and needn't) resolve to a repo file.
_JAVA_EXTERNAL = ("java.", "javax.", "jakarta.", "org.springframework", "org.slf4j", "lombok",
                  "com.fasterxml", "org.junit", "org.mockito", "reactor.", "org.apache", "io.",
                  "com.google", "org.hibernate", "jakarta.")


def _lang_of(rel: str) -> str:
    if rel.endswith(".java"):
        return "java"
    if rel.endswith(".ts"):
        return "ts"
    return ""


def _resolve_ts_import(root: Path, from_rel: str, mod_path: str) -> Optional[Path]:
    if mod_path.startswith("."):
        base = root / Path(from_rel).parent / mod_path
    elif mod_path.startswith("src/"):
        base = root / mod_path
    else:
        return None
    try:
        cand = Path(str(base.resolve()) + ".ts")
    except OSError:
        return None
    return cand if cand.is_file() else None


def _resolve_java_import(root: Path, fqcn: str) -> Optional[Path]:
    """A fully-qualified class name maps 1:1 to a path under src/main|test/java."""
    if fqcn.startswith(_JAVA_EXTERNAL):
        return None
    rel = fqcn.replace(".", "/") + ".java"
    for base in ("src/main/java", "src/test/java"):
        cand = root / base / rel
        if cand.is_file():
            return cand
    return None


def _java_fqcn(root: Path, path: Path) -> str:
    posix = path.as_posix()
    for marker in ("/src/main/java/", "/src/test/java/"):
        i = posix.find(marker)
        if i >= 0:
            return posix[i + len(marker):-5].replace("/", ".")
    return path.stem


def _ts_signatures(text: str, max_methods: int = 16) -> List[str]:
    if _TS_PARSER:
        try:
            tree = _TS_PARSER.parse(text.encode("utf-8", errors="ignore"))
            sigs: List[str] = []
            def walk(n):
                if n.type in ("method_definition", "method_signature"):
                    mods = [c.text.decode("utf-8") for c in n.children if c.type == "accessibility_modifier"]
                    if "private" not in mods and "protected" not in mods:
                        name_n = n.child_by_field_name("name")
                        params_n = n.child_by_field_name("parameters")
                        ret_n = n.child_by_field_name("return_type")
                        if name_n:
                            name = name_n.text.decode("utf-8")
                            if name not in ("constructor", "ngOnInit", "ngOnDestroy", "ngAfterViewInit", "ngOnChanges", "get", "set"):
                                params = " ".join(params_n.text.decode("utf-8").split()) if params_n else "()"
                                ret = (ret_n.text.decode("utf-8") if ret_n else "").strip()
                                sig = f"{name}{params}" + (f"{ret}" if ret.startswith(":") else f": {ret}" if ret else "")
                                if sig not in sigs:
                                    sigs.append(sig)
                for child in n.children:
                    walk(child)
            walk(tree.root_node)
            if sigs:
                return sigs[:max_methods]
        except Exception:
            pass
    sigs: List[str] = []
    for m in _METHOD_RE.finditer(text):
        name, args, ret = m.group(1), " ".join(m.group(2).split()), (m.group(3) or "").strip()
        sig = f"{name}({args})" + (f": {ret}" if ret else "")
        if sig not in sigs:
            sigs.append(sig)
        if len(sigs) >= max_methods:
            break
    return sigs


_TS_FIELD_RE = re.compile(r"^[ \t]*(?:readonly\s+)?([a-zA-Z_]\w*)\??\s*:\s*([^;=\n{()]+);", re.M)


def _ts_fields(text: str, max_fields: int = 22) -> List[str]:
    out, seen = [], set()
    for m in _TS_FIELD_RE.finditer(text):
        name, typ = m.group(1), " ".join(m.group(2).split())
        if name in seen or name in ("import", "from", "return"):
            continue
        seen.add(name)
        out.append(f"{name}: {typ}")
        if len(out) >= max_fields:
            break
    return out


def _java_signatures(text: str, max_methods: int = 16) -> List[str]:
    if _JAVA_PARSER:
        try:
            tree = _JAVA_PARSER.parse(text.encode("utf-8", errors="ignore"))
            sigs: List[str] = []
            def walk(n):
                if n.type == "method_declaration":
                    mods = [c.text.decode("utf-8") for c in n.children if c.type in ("modifiers",)]
                    mod_str = " ".join(mods)
                    if "private" not in mod_str and "protected" not in mod_str:
                        name_n = n.child_by_field_name("name")
                        type_n = n.child_by_field_name("type")
                        params_n = n.child_by_field_name("parameters")
                        if name_n:
                            name = name_n.text.decode("utf-8")
                            if name not in ("if", "for", "while", "switch", "catch", "new", "return"):
                                ret = type_n.text.decode("utf-8") if type_n else "void"
                                params = " ".join(params_n.text.decode("utf-8").split()) if params_n else "()"
                                sig = f"{ret} {name}{params}"
                                if sig not in sigs:
                                    sigs.append(sig)
                for child in n.children:
                    walk(child)
            walk(tree.root_node)
            if sigs:
                return sigs[:max_methods]
        except Exception:
            pass
    sigs: List[str] = []
    for m in _JAVA_METHOD_RE.finditer(text):
        ret, name, args = m.group(1), m.group(2), " ".join(m.group(3).split())
        if name in ("if", "for", "while", "switch", "catch", "new", "return"):
            continue
        sig = f"{ret} {name}({args})"
        if sig not in sigs:
            sigs.append(sig)
        if len(sigs) >= max_methods:
            break
    return sigs


def _java_fields(text: str, max_fields: int = 22) -> List[str]:
    out, seen = [], set()
    for m in _JAVA_FIELD_RE.finditer(text):
        typ, name = " ".join(m.group(1).split()), m.group(2)
        if name in seen or typ in ("return", "new"):
            continue
        seen.add(name)
        out.append(f"{typ} {name}")
        if len(out) >= max_fields:
            break
    return out


def _members_block(path: Path, header: str, lang: str, as_model: bool) -> Optional[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    if lang == "java":
        methods = _java_signatures(text)
        fields = _java_fields(text)
        lines = []
        if methods:
            lines.append("methods: " + "; ".join(methods[:14]))
        if fields and (as_model or not methods):
            lines.append("fields: " + "; ".join(fields[:18]))
        return f"// {header}\n  " + "\n  ".join(lines) if lines else None
    # ts
    if as_model:
        fields = _ts_fields(text)
        return f"// {header}; fields:\n  " + "\n  ".join(fields) if fields else None
    sigs = _ts_signatures(text)
    return f"// {header} (read-only):\n  " + "\n  ".join(sigs) if sigs else None


# ── Sibling call-site mining ─────────────────────────────────────────────────
# Many service methods take generic payloads (`{}` / `any`), so the signature does not say which
# properties are required. Real call-sites in sibling code are the ground truth for the payload shape
# and for which method actually performs the operation, so the patcher copies a proven pattern
# instead of inventing one that merely type-checks.

_GENERIC_PARAM_RE = re.compile(r":\s*(?:\{\s*\}|any\b|object\b|Object\b|Record<|\{\s*\[)")
_USAGE_STOP = {"the", "a", "an", "and", "or", "in", "on", "at", "by", "for", "is", "it", "of", "to",
               "this", "that", "with", "from", "when", "should", "must", "will", "have", "also", "into",
               "only", "after", "before", "does", "need", "keep", "than", "then", "each", "every", "other",
               "data", "value", "item", "items", "list", "service", "model",
               "user", "users", "message", "dialog", "modal", "screen", "click", "warn", "error"}


def _stems_list(text: str) -> List[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    words = re.findall(r"[A-Za-z]{3,}", spaced)
    stop = _USAGE_STOP | _NLTK_STOP
    if _STEMMER:
        return [_STEMMER.stem(w.lower()) for w in words if w.lower() not in stop]
    return [w.lower()[:5] for w in words if w.lower() not in stop]


def _stems(text: str) -> set:
    return set(_stems_list(text))


def _call_snippet(lines: List[str], idx: int, name: str, max_lines: int = 12) -> str:
    """The call `.name(...)` on lines[idx], cut at ITS OWN closing paren — so the payload is shown but
    not the chained `.subscribe(...)` callbacks that follow. A chained call (`.name(` at line start)
    keeps the previous line so the receiver is visible."""
    import textwrap
    m = re.search(r"\.\s*" + re.escape(name) + r"\s*\(", lines[idx])
    out: List[str] = []
    if idx > 0 and lines[idx].lstrip().startswith("."):
        out.append(lines[idx - 1].rstrip())
    depth = 1
    j, col = idx, (m.end() if m else 0)
    while j < min(idx + max_lines, len(lines)):
        line = lines[j]
        closed_at = None
        for k in range(col if j == idx else 0, len(line)):
            ch = line[k]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    closed_at = k
                    break
        if closed_at is not None:
            out.append(line[:closed_at + 1].rstrip())
            break
        out.append(line.rstrip())
        j, col = j + 1, 0
    snippet_text = textwrap.dedent("\n".join(out)).strip("\n")
    # A bare-identifier argument (`removeContractMembers(contractMembers)`) hides the payload shape, which
    # is built a few lines earlier. Pull that declaration in so the shape is visible.
    am = re.search(r"\(\s*([A-Za-z_]\w*)\s*(?:,|\))", snippet_text.split("\n")[-1] if snippet_text else "")
    if am and am.group(1) not in ("this", "null", "undefined", "true", "false"):
        var = am.group(1)
        decl = re.compile(r"\b(?:let|const|var)\s+" + re.escape(var) + r"\b|\b" + re.escape(var) + r"\s*[:=]\s*[\[{]")
        for b in range(idx - 1, max(idx - 40, -1), -1):
            if decl.search(lines[b]):
                block: List[str] = []
                depth = 0
                for k in range(b, min(b + 16, idx)):
                    block.append(lines[k].rstrip())
                    depth += lines[k].count("{") + lines[k].count("[") - lines[k].count("}") - lines[k].count("]")
                    if depth <= 0 and k > b - 1 and (lines[k].rstrip().endswith((";", "}", "]")) or depth == 0 and k > b):
                        break
                return textwrap.dedent("\n".join(block + out)).strip("\n")
    return snippet_text


def _service_methods(root: Path, lang: str) -> List[tuple]:
    """(method_name, args, defining_path) for every service method in the project."""
    out: List[tuple] = []
    if lang == "java":
        base = root / "src" / "main" / "java"
        files = [p for p in sorted(base.rglob("*.java")) if p.stem.endswith(("Service", "ServiceImpl", "Repository"))] if base.is_dir() else []
        for p in files:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if _JAVA_PARSER:
                try:
                    tree = _JAVA_PARSER.parse(text.encode("utf-8", errors="ignore"))
                    def walk_java(n):
                        if n.type == "method_declaration":
                            mods = [c.text.decode("utf-8") for c in n.children if c.type in ("modifiers",)]
                            mod_str = " ".join(mods)
                            if "private" not in mod_str and "protected" not in mod_str:
                                name_n = n.child_by_field_name("name")
                                params_n = n.child_by_field_name("parameters")
                                if name_n:
                                    nm = name_n.text.decode("utf-8")
                                    if nm not in ("if", "for", "while", "switch", "catch", "new", "return"):
                                        params = " ".join(params_n.text.decode("utf-8").split()) if params_n else "()"
                                        out.append((nm, params, p))
                        for child in n.children:
                            walk_java(child)
                    walk_java(tree.root_node)
                    continue
                except Exception:
                    pass
            for m in _JAVA_METHOD_RE.finditer(text):
                out.append((m.group(2), " ".join(m.group(3).split()), p))
    else:
        base = root / "src" / "app"
        files = [p for p in sorted(base.rglob("*.service.ts")) if ".spec." not in p.name] if base.is_dir() else []
        for p in files:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if _TS_PARSER:
                try:
                    tree = _TS_PARSER.parse(text.encode("utf-8", errors="ignore"))
                    def walk_ts(n):
                        if n.type in ("method_definition", "method_signature"):
                            mods = [c.text.decode("utf-8") for c in n.children if c.type == "accessibility_modifier"]
                            if "private" not in mods and "protected" not in mods:
                                name_n = n.child_by_field_name("name")
                                params_n = n.child_by_field_name("parameters")
                                if name_n:
                                    nm = name_n.text.decode("utf-8")
                                    if nm not in ("constructor", "ngOnInit", "ngOnDestroy", "ngAfterViewInit", "ngOnChanges", "get", "set"):
                                        params = " ".join(params_n.text.decode("utf-8").split()) if params_n else "()"
                                        out.append((nm, params, p))
                        for child in n.children:
                            walk_ts(child)
                    walk_ts(tree.root_node)
                    continue
                except Exception:
                    pass
            for m in _METHOD_RE.finditer(text):
                out.append((m.group(1), " ".join(m.group(2).split()), p))
    return [t for t in out if t[0] not in ("constructor", "get", "set") and len(t[0]) >= 4]


def build_usage_examples(scope: ScopeContract, locked_rel: List[str], lang: str,
                         max_methods: int = 10, sites_per_method: int = 1, max_chars: int = 4200) -> str:
    """Real call-sites (from sibling code) for the service methods most relevant to the ticket."""
    root = Path(scope.root)
    ticket = getattr(scope, "ticket_text", "") or ""
    if not ticket.strip():
        return ""
    ticket_stems = _stems(ticket)
    ticket_low = ticket.lower()
    locked_abs = {str((root / r).resolve()) for r in locked_rel}

    scored: List[tuple] = []
    seen_names: set = set()
    all_methods = _service_methods(root, lang)
    # Inverse-frequency weight per stem: 'contr'/'membe'/'remov' are specific to a few methods, while a
    # verb like 'updat' is in dozens — so a method that matches on RARE stems is the relevant one.
    stem_df: Dict[str, int] = {}
    for n, _, _ in all_methods:
        for s in _stems(n):
            stem_df[s] = stem_df.get(s, 0) + 1
    total = max(len(all_methods), 1)
    # Services the locked code already uses are the likeliest homes of the method it needs; a service
    # whose own name is a ticket concept (contract.service.ts for a 'contracts' ticket) is next.
    used_services: set = set()
    for rel in locked_rel:
        if rel.endswith((".ts", ".java")):
            try:
                body = (root / rel).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            used_services |= {m.lower() for m in re.findall(r"(\w*[Ss]ervice\w*)\s*[:,)]", body)}
            used_services |= {m.lower() for m in re.findall(r"this\.(\w+)\.", body)}
    bm_scores = None
    if _BM25Okapi and all_methods:
        try:
            corpus = [_stems_list(f"{name} {args} {path.stem}") for name, args, path in all_methods]
            bm = _BM25Okapi(corpus)
            query_tokens = _stems_list(ticket)
            bm_scores = bm.get_scores(query_tokens)
        except Exception:
            bm_scores = None

    for idx, (name, args, path) in enumerate(all_methods):
        if name in seen_names:
            continue
        shared = _stems(name) & ticket_stems
        named = any(c.isupper() for c in name[1:]) and name.lower() in ticket_low
        if bm_scores is not None:
            base_score = float(bm_scores[idx])
            if base_score <= 0.0 and len(shared) < 1 and not named:
                continue
            score = base_score
            if named:
                score += 20.0
            if _GENERIC_PARAM_RE.search(args):
                score += 15.0
            stem_of_file = path.stem.lower().replace(".service", "").replace("-", "")
            if any(stem_of_file[:6] and stem_of_file[:6] in u for u in used_services):
                score += 15.0
            if _stems(path.stem) & ticket_stems:
                score += 15.0
            if re.search(r"^(?:get|find|fetch)\w+By\w+", name, re.I) and len(shared) >= 1:
                score += 30.0
            if name.startswith(("emit", "set", "reset")) and not named:
                score -= 25.0
        else:
            if len(shared) < 2 and not named:
                continue
            score = sum(math.log((total + 1) / (stem_df.get(s, 0) + 1)) for s in shared) + (4.0 if named else 0.0)
            if _GENERIC_PARAM_RE.search(args):
                score += 1.5   # a generic payload means the signature alone can't show the shape
            stem_of_file = path.stem.lower().replace(".service", "").replace("-", "")
            if any(stem_of_file[:6] and stem_of_file[:6] in u for u in used_services):
                score += 3.0   # defined in a service the locked code already uses
            if _stems(path.stem) & ticket_stems:
                score += 2.5   # the service itself is a ticket concept (contracts.service for a contracts ticket)
            if re.search(r"^(?:get|find|fetch)\w+By\w+", name, re.I) and len(shared) >= 2:
                score += 5.0   # multi-entity query connecting ticket entities (e.g. getContractsByProjectId)
            elif len(shared) >= 2:
                score += 2.0
            if name.startswith(("emit", "set", "reset")) and not named:
                score -= 4.0
        seen_names.add(name)
        scored.append((score, name, path))
    scored.sort(key=lambda t: -t[0])
    chosen = scored[:max_methods]
    if not chosen:
        return ""

    # Collect candidate call-sites across sibling source files (read each file once).
    src_dir = root / ("src/main/java" if lang == "java" else "src/app")
    exts = ("*.java",) if lang == "java" else ("*.ts",)
    pats = {name: re.compile(r"\.\s*" + re.escape(name) + r"\s*\(") for _, name, _ in chosen}
    defs = {name: str(path.resolve()) for _, name, path in chosen}
    sites: Dict[str, List[tuple]] = {name: [] for _, name, _ in chosen}
    anchor_dir = Path(locked_abs.copy().pop()).parent.parts if locked_abs else ()
    if src_dir.is_dir():
        for ext in exts:
            for f in src_dir.rglob(ext):
                fp = str(f.resolve())
                low = f.name.lower()
                if (fp in locked_abs or ".spec." in low or low.endswith(("test.java", "tests.java"))
                        or "/test/" in f.as_posix()):
                    continue
                try:
                    lines = f.read_text(encoding="utf-8", errors="ignore").split("\n")
                except OSError:
                    continue
                for name, pat in pats.items():
                    if fp == defs[name]:
                        continue
                    for i, line in enumerate(lines):
                        if pat.search(line):
                            near = len([1 for a, b in zip(anchor_dir, f.resolve().parent.parts) if a == b])
                            snippet = _call_snippet(lines, i, name)
                            rank = near + (3 if "{" in snippet else 0)
                            sites[name].append((rank, f, i + 1, snippet))
                            break  # one site per file per method keeps examples diverse
    blocks: List[str] = []
    used = 0
    for _, name, _ in chosen:
        picks = sorted(sites[name], key=lambda t: -t[0])[:sites_per_method]
        for _, f, ln, snippet in picks:
            rel = scope.relative(str(f)) or f.name
            block = f"// {name} — real call-site in {rel}:{ln}\n{snippet}"
            if used + len(block) > max_chars:
                return "\n\n".join(blocks)
            blocks.append(block)
            used += len(block)
    return "\n\n".join(blocks)


def build_dependency_context(scope: ScopeContract, locked_rel: List[str], fence: ScopeFence,
                             max_apis: int = 10, max_services_index: int = 16) -> str:
    """Read-only, polyglot API reference for the patcher: the exact import paths plus method signatures
    (TS services / Java Spring services) and field shapes (TS models / Java DTOs & entities) of what the
    locked code imports, and a compact index of other services/DTOs — each with signatures — so an
    un-imported dependency can be adopted without guessing. This removes the self-unhealable failure
    modes 'cannot find module' (TS) and 'cannot find symbol' (Java)."""
    root = Path(scope.root)
    blocks: List[str] = []
    seen: set = set()
    langs: set = set()
    for rel in locked_rel:
        lang = _lang_of(rel)
        if not lang:
            continue
        langs.add(lang)
        try:
            text = (root / rel).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if lang == "ts":
            for m in _IMPORT_RE.finditer(text):
                names, mod = m.group(1), m.group(2)
                f = _resolve_ts_import(root, rel, mod)
                if not f or str(f) in seen:
                    continue
                posix = f.as_posix()
                is_model = "/models/" in posix
                if "/services/" not in posix and not is_model:
                    continue
                seen.add(str(f))
                blk = _members_block(f, f"{names.strip()} — import from '{mod}'", "ts", is_model)
                if blk:
                    blocks.append(blk)
                if len(blocks) >= max_apis:
                    break
        else:  # java
            for m in _JAVA_IMPORT_RE.finditer(text):
                fqcn = m.group(1)
                f = _resolve_java_import(root, fqcn)
                if not f or str(f) in seen:
                    continue
                seen.add(str(f))
                is_dto = "/dto/" in f.as_posix() or f.stem.lower().endswith(("dto", "count", "request", "response", "entity"))
                blk = _members_block(f, f"{f.stem} — import {fqcn};", "java", is_dto)
                if blk:
                    blocks.append(blk)
                if len(blocks) >= max_apis:
                    break
        if len(blocks) >= max_apis:
            break
    # Index of other services/DTOs (with signatures) so an un-imported dependency is adoptable.
    index: List[str] = []
    if "java" in langs:
        jroot = root / "src" / "main" / "java"
        if jroot.is_dir():
            for p in sorted(jroot.rglob("*.java")):
                if str(p) in seen:
                    continue
                nm, posix = p.stem, p.as_posix()
                if not (nm.endswith(("Service", "ServiceImpl", "Dto", "DTO", "Repository", "Entity", "Count")) or "/dto/" in posix):
                    continue
                blk = _members_block(p, f"{nm} — import {_java_fqcn(root, p)};", "java", "/dto/" in posix)
                if blk:
                    index.append(blk)
                if len(index) >= max_services_index:
                    break
    else:
        app_dir = root / "src" / "app"
        if app_dir.is_dir():
            for p in sorted(app_dir.rglob("*.service.ts")):
                if ".spec." in p.name or str(p) in seen:
                    continue
                posix = p.as_posix()
                idx = posix.find("src/")
                imp = posix[idx:-3] if idx >= 0 else p.stem
                txt = p.read_text(encoding="utf-8", errors="ignore")
                mcls = re.search(r"export\s+class\s+(\w+)", txt)  # the real class name, not a guess
                cls = mcls.group(1) if mcls else (
                    "".join(w.capitalize() for w in p.stem.replace(".service", "").split("-")) + "Service")
                methods = _ts_signatures(txt, max_methods=12)
                line = f"{cls} — import from '{imp}'"
                if methods:
                    line += ":\n    " + "\n    ".join(methods)
                index.append(line)
                if len(index) >= max_services_index:
                    break
    parts: List[str] = []
    if blocks:
        parts.append("APIS/MODELS THE LOCKED CODE ALREADY USES (call these; do NOT edit these files):\n" + "\n".join(blocks))
    if index:
        parts.append("OTHER AVAILABLE DEPENDENCIES (import/inject using the exact path shown):\n  "
                     + "\n  ".join(index))
    usage_lang = "java" if "java" in langs else "ts"
    try:
        usage = build_usage_examples(scope, locked_rel, usage_lang)
    except Exception:  # noqa: BLE001 — examples are an aid; never let mining break localization
        usage = ""
    if usage:
        parts.append("REAL CALL-SITES FROM SIBLING CODE (copy the method choice and the exact payload shape "
                     "from these proven examples; do NOT invent a different method or payload):\n" + usage)
    # ── Fullstack call-chain tracing (links UI actions -> Sagas -> Backend handlers) ──
    try:
        # Gate 1: Exclude grid/column-reorder tickets (pure UI layout, zero Saga/backend calls)
        # and exclude pure backend tickets (which do not start from UI dialogs/buttons).
        is_grid = getattr(scope, "grid_hint", False)
        if not is_grid and scope.layer in ("ui", "frontend_logic", "fullstack"):
            raw_text = (getattr(scope, "ticket_text", "") or "").lower()
            action_terms = (
                "remove", "delete", "revoke", "dialog", "modal", "button",
                "click", "message", "popup", "confirm", "action", "trigger",
                "flow", "save", "update", "cancel"
            )
            has_action_intent = any(term in raw_text for term in action_terms)

            if has_action_intent:
                from ticket_to_code.lean.call_chain_tracer import CallChainTracer

                candidate_queries: List[str] = []
                # Select only specific anchors: dotted i18n keys or multi-word dialog phrases
                if getattr(scope, "anchors", None):
                    for a in scope.anchors:
                        if "." in a and not a.endswith((".ts", ".html", ".js", ".json", ".java")):
                            candidate_queries.append(a)
                        elif len(a.split()) >= 3 or len(a) >= 15:
                            candidate_queries.append(a)

                # Cap at 1 high-signal query to prevent redundant repository walks
                best_query = candidate_queries[0] if candidate_queries else None

                if best_query:
                    cache_key = (str(scope.root), best_query)
                    if cache_key in _TRACE_CACHE:
                        trace_res = _TRACE_CACHE[cache_key]
                    else:
                        tracer = CallChainTracer(scope.root)
                        trace_res = tracer.trace(best_query)
                        _TRACE_CACHE[cache_key] = trace_res

                    if trace_res and trace_res.get("summary_trace"):
                        chain_summary = "\n".join(f"  {line}" for line in trace_res["summary_trace"])
                        parts.append("CANDIDATE FULLSTACK EXECUTION CHAIN (HEURISTIC / UNVERIFIED):\n" + chain_summary)
    except Exception as _trace_err:
        logger.warning(f"[LeanPath] CallChainTracer invocation failed: {_trace_err}")

    gate_context = getattr(scope, "clarified_context", "")
    if gate_context:
        parts.append(gate_context)
    return "\n\n".join(parts)


def run_lean_path(
    ticket_id: str, title: str, description: str, workspace_path: str,
    patch_fn: PatchFn, verify_fn: VerifyFn,
    explicit_root: Optional[str] = None, labels: Optional[List[str]] = None,
    hard_timeout_s: float = 600.0, meter: Optional[TokenMeter] = None,
) -> LeanResult:
    t0 = time.time()
    meter = meter or TokenMeter()
    trace: List[str] = []

    # Stage 1: Triage (0 calls)
    scope = triage(ticket_id, title, description, workspace_path, explicit_root, labels)
    trace.append(f"triage: layer={scope.layer} conf={scope.confidence} root={Path(scope.root).name} "
                 f"anchors={list(scope.anchors)}")
    if not should_use_lean(scope):
        return LeanResult("out_of_scope", scope=scope, reason=f"not a lean ticket (layer={scope.layer}, "
                          f"conf={scope.confidence})", trace=trace, elapsed_s=time.time() - t0)

    # Stage 2+3: Scope fence + deterministic locate (0 calls)
    cands, fence = locate(scope)
    trace.append(f"locate: {len(cands)} candidate(s); fence reads={len(fence.reads)} "
                 f"violations={len(fence.violations)}")
    if not cands:
        return LeanResult("no_candidates", scope=scope, reason="no file met the anchor co-occurrence bar",
                          trace=trace, elapsed_s=time.time() - t0)

    # Stage 4: Lock context (one-way door). Lock the top candidate's component group only;
    # add further candidates solely when they share the top file's directory (same feature).
    ctx = build_locked_context(scope, cands, fence)
    locked_rel = list(ctx.files)
    trace.append(f"lock: {locked_rel}; facts={len(ctx.framework_facts)}")

    # Stage 5+6: Patch → Verify ladder → Convergence (one-way, Patch↔Verify only)
    tracker = ConvergenceTracker()
    last_diff = ""
    failure: Optional[StructuredFailure] = None
    attempts = 0
    infra_strikes = 0
    max_infra_strikes = 4

    def restore() -> None:
        for rel, content in ctx.baselines.items():
            (Path(scope.root) / rel).write_text(content, encoding="utf-8")

    try:
        while True:
            if time.time() - t0 > hard_timeout_s:
                restore()
                return LeanResult("unverified", scope=scope, locked_files=locked_rel, attempts=attempts,
                                  reason=f"hard timeout {hard_timeout_s}s", tokens=meter.tokens,
                                  llm_calls=meter.llm_calls, trace=trace, elapsed_s=time.time() - t0)
            attempts += 1
            # Every attempt patches the pristine baseline so the model's search text always matches
            # disk and the cached prefix. The failure feedback (not prior edits) guides the next patch.
            if failure is not None:
                restore()
            hunks = patch_fn(ctx, last_diff, failure)
            plan = plan_hunks(scope.root, list(hunks), allowed_files=set(ctx.files))
            if not plan.ok:
                detail = " | ".join(f"[{reason}] {h.file}: search={h.search[:160]!r}"
                                    for h, reason in plan.rejected)
                failure = StructuredFailure(kind="patch_reject", file=(hunks[0].file if hunks else None),
                                            actual=[r for _, r in plan.rejected],
                                            hint=("your previous hunks were rejected. Copy each 'search' string VERBATIM "
                                                  "from the file content shown above (exact characters, including whitespace) "
                                                  "and make it match exactly once. Rejected hunks: " + detail))
                trace.append(f"attempt {attempts}: hunks rejected {[r for _, r in plan.rejected]}")
                if tracker.evaluate(failure) is not Action.CONTINUE_REPAIR:
                    restore()
                    return LeanResult("unverified", scope=scope, locked_files=locked_rel, attempts=attempts,
                                      reason=tracker.reason, tokens=meter.tokens, llm_calls=meter.llm_calls,
                                      trace=trace, elapsed_s=time.time() - t0)
                continue
            if not plan.applied:
                # An empty patch can never count as success: the model produced no edit, so verifying
                # the untouched baseline (which already passes) would be a vacuous "verified".
                failure = StructuredFailure(kind="patch_reject", file=None, actual=["no hunks produced"],
                                            hint=("you produced no edits. Implement the ticket by editing the locked "
                                                  "files with concrete search/replace hunks. If the behaviour seems to "
                                                  "exist, make the ticket's exact requirement explicit (e.g. the specific "
                                                  "error code / role) in the locked files."))
                trace.append(f"attempt {attempts}: no hunks produced")
                if tracker.evaluate(failure) is not Action.CONTINUE_REPAIR:
                    restore()
                    return LeanResult("unverified", scope=scope, locked_files=locked_rel, attempts=attempts,
                                      reason=tracker.reason, tokens=meter.tokens, llm_calls=meter.llm_calls,
                                      trace=trace, elapsed_s=time.time() - t0)
                continue
            written = write_result(scope.root, plan)
            last_diff = "; ".join(f"{h.file}: {h.search!r}->{h.replace!r}" for h in plan.applied)[:4000]
            outcome = verify_fn(ctx, written)
            if outcome.ok:
                return LeanResult("verified", scope=scope, locked_files=locked_rel, attempts=attempts,
                                  reason="all verify rungs passed", tokens=meter.tokens,
                                  llm_calls=meter.llm_calls, trace=trace + [f"attempt {attempts}: verified"],
                                  elapsed_s=time.time() - t0)
            failure = outcome.failure
            # Environment failures (backend down, login, grid not rendered) are NOT code defects and
            # must never feed the convergence tracker — otherwise a flaky verify masquerades as a loop.
            if failure is not None and failure.kind == "infra":
                infra_strikes += 1
                trace.append(f"attempt {attempts}: verify unavailable ({failure.actual}) [{infra_strikes}/{max_infra_strikes}]")
                attempts -= 1  # this attempt did not test the edit; don't count it
                if infra_strikes >= max_infra_strikes:
                    restore()
                    return LeanResult("unverified", scope=scope, locked_files=locked_rel, attempts=attempts,
                                      reason=f"live verification unavailable after {max_infra_strikes} tries "
                                             f"(environment): {failure.actual}", tokens=meter.tokens,
                                      llm_calls=meter.llm_calls, trace=trace, elapsed_s=time.time() - t0)
                restore()           # the just-written edit couldn't be tested; re-patch from baseline
                failure = None      # keep this environment failure out of the convergence tracker
                continue
            trace.append(f"attempt {attempts}: rung={outcome.rung} failure={failure.signature() if failure else '?'} "
                         f":: {str(getattr(failure, 'actual', ''))[:400]}")
            if tracker.evaluate(failure) is not Action.CONTINUE_REPAIR:
                restore()
                return LeanResult("unverified", scope=scope, locked_files=locked_rel, attempts=attempts,
                                  reason=tracker.reason, tokens=meter.tokens, llm_calls=meter.llm_calls,
                                  trace=trace, elapsed_s=time.time() - t0)
    except ClarifyRequest as cq:
        restore()
        return LeanResult("needs_clarification", scope=scope, locked_files=locked_rel, attempts=attempts,
                          reason=cq.question, tokens=meter.tokens, llm_calls=meter.llm_calls,
                          trace=trace + [f"clarify: {cq.question}"], elapsed_s=time.time() - t0)
    except Exception as exc:  # noqa: BLE001
        restore()
        return LeanResult("error", scope=scope, locked_files=locked_rel, attempts=attempts,
                          reason=f"{type(exc).__name__}: {exc}", tokens=meter.tokens,
                          llm_calls=meter.llm_calls, trace=trace, elapsed_s=time.time() - t0)
