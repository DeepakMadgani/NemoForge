"""
Workspace Symbol Scanner — Pre-Planning Analysis

Inspects ALL files in the workspace BEFORE planning and extracts exact symbols
(properties, methods, classes, interfaces) so the planner makes INFORMED decisions
instead of guessing property names.

Key insight: The planner should KNOW what exists before it plans to create it.
This prevents HTML from inventing property names that don't exist in TS.

Author: Deepak Madgani
Date: August 2026
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SymbolInfo:
    """A single symbol (property, method, class, etc.) in a file."""
    name: str
    kind: str  # "property", "method", "class", "interface", "enum", "function"
    type_hint: Optional[str] = None  # e.g., "string", "boolean", "Observable<User>"
    access_level: Optional[str] = None  # "public", "private", "protected", ""
    is_optional: bool = False  # For TS ? properties
    source_line: Optional[int] = None
    # ── Signature fields (for grounded cross-task contracts) ──
    owner_class: Optional[str] = None  # Owning class name, e.g. "ContractMemberService"
    params: Optional[List[str]] = None  # Parameter strings, e.g. ["UUID userId", "UUID projectId"]
    return_type: Optional[str] = None  # Return type, e.g. "ProjectMembershipResult"

    def to_dict(self):
        d = asdict(self)
        # Exclude None fields for backward compat
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class FileSymbols:
    """All symbols extracted from a single file."""
    file_path: str  # Relative path from workspace root
    language: str  # "typescript", "java", "python", "html"
    symbols: List[SymbolInfo]  # All symbols found
    class_names: List[str]  # Top-level class/interface/component names
    extraction_method: str  # "lsp", "ast", "regex", "manual"
    error: Optional[str] = None  # If extraction failed

    def get_properties(self) -> List[SymbolInfo]:
        """Return only property symbols."""
        return [s for s in self.symbols if s.kind == "property"]

    def get_methods(self) -> List[SymbolInfo]:
        """Return only method symbols."""
        return [s for s in self.symbols if s.kind == "method"]

    def get_classes(self) -> List[SymbolInfo]:
        """Return only class symbols."""
        return [s for s in self.symbols if s.kind == "class"]

    def to_dict(self):
        return {
            "file_path": self.file_path,
            "language": self.language,
            "symbols": [s.to_dict() for s in self.symbols],
            "class_names": self.class_names,
            "extraction_method": self.extraction_method,
            "error": self.error,
        }


class WorkspaceSymbolScanner:
    """
    Scans all files in workspace, extracts symbols using LSP/AST, and builds
    a queryable index. Prevents planner from guessing—it has EXACT data.

    Usage:
        scanner = WorkspaceSymbolScanner(workspace_path, lsp_client, include_patterns)
        index = scanner.scan_workspace()
        
        # Check: does TS file already have this property?
        if index.has_property("add-members.component.ts", "isUserProjectMember"):
            # Don't generate it in HTML—it already exists!
            pass
    """

    def __init__(
        self,
        workspace_path: str,
        lsp_client=None,
        include_patterns: Optional[List[str]] = None,
        max_files: int = 500,  # Safety limit
    ):
        self.workspace_path = Path(workspace_path)
        self.lsp_client = lsp_client
        self.include_patterns = include_patterns or [
            "**/*.ts",
            "**/*.tsx",
            "**/*.java",
            "**/*.py",
            "**/*.html",
        ]
        self.max_files = max_files
        self.file_symbols: Dict[str, FileSymbols] = {}  # path → FileSymbols

    # ── In-memory content scanning (for generated artifacts) ─────────────

    def scan_content(self, content: str, file_path: str) -> FileSymbols:
        """Extract symbols from in-memory content without reading from disk.

        Used by GenerationHandoff to extract grounded symbols/signatures from
        generated code that is still in ImplementationState.generated_files.

        Returns the same FileSymbols that scan_workspace() produces for on-disk
        files, so the output can be converted to VerifiedSymbol objects for
        handoff.exports.

        Args:
            content: The source code content (string).
            file_path: Relative or absolute path (used only for language detection).

        Returns:
            FileSymbols with extracted symbols, class_names, and signatures.
        """
        if not content or not content.strip():
            return FileSymbols(
                file_path=file_path,
                language=self._get_language(Path(file_path)),
                symbols=[],
                class_names=[],
                extraction_method="empty",
            )

        lang = self._get_language(Path(file_path))

        try:
            if lang in ("typescript", "javascript"):
                return self._extract_typescript_regex_from_content(file_path, content)
            elif lang == "java":
                return self._extract_java_regex_from_content(file_path, content)
            elif lang == "python":
                return self._extract_python_ast_from_content(file_path, content)
            elif lang in ("html", "htm"):
                return self._extract_html_bindings_from_content(file_path, content)
            elif lang == "kotlin":
                # Kotlin shares Java-like syntax for public API
                return self._extract_java_regex_from_content(file_path, content)
            else:
                return FileSymbols(
                    file_path=file_path,
                    language=lang,
                    symbols=[],
                    class_names=[],
                    extraction_method="unsupported_language",
                )
        except Exception as e:
            logger.debug(f"  scan_content failed for {file_path}: {e}")
            return FileSymbols(
                file_path=file_path,
                language=lang,
                symbols=[],
                class_names=[],
                extraction_method="scan_content_failed",
                error=str(e),
            )

    def scan_workspace(self) -> "WorkspaceSymbolIndex":
        """Scan workspace and build index."""
        logger.info(f"Scanning workspace for symbols: {self.workspace_path}")

        # Find all candidate files
        files_to_scan = self._find_files()
        logger.info(f"  Found {len(files_to_scan)} candidate files")

        if len(files_to_scan) > self.max_files:
            logger.warning(f"  ⚠️  Limiting to {self.max_files} files (found {len(files_to_scan)})")
            files_to_scan = files_to_scan[: self.max_files]

        # Extract symbols from each file
        for i, fpath in enumerate(files_to_scan, 1):
            rel_path = fpath.relative_to(self.workspace_path).as_posix()

            try:
                file_symbols = self._extract_symbols(fpath, rel_path)
                self.file_symbols[rel_path] = file_symbols

                if file_symbols.error:
                    logger.debug(f"  [{i}/{len(files_to_scan)}] {rel_path} — {file_symbols.error}")
                else:
                    sym_count = len(file_symbols.symbols)
                    logger.debug(f"  [{i}/{len(files_to_scan)}] {rel_path} — {sym_count} symbols")

            except Exception as e:
                logger.warning(f"  [{i}/{len(files_to_scan)}] {rel_path} — EXCEPTION: {e}")
                self.file_symbols[rel_path] = FileSymbols(
                    file_path=rel_path,
                    language=self._get_language(fpath),
                    symbols=[],
                    class_names=[],
                    extraction_method="failed",
                    error=str(e),
                )

        logger.info(f"Symbol extraction complete: {len(self.file_symbols)} files processed")

        # Build and return index
        return WorkspaceSymbolIndex(self.file_symbols)

    def _find_files(self) -> List[Path]:
        """Find all files matching include_patterns."""
        found = set()
        for pattern in self.include_patterns:
            found.update(self.workspace_path.glob(pattern))
        # Exclude node_modules, .venv, build output, etc.
        exclude_dirs = {
            "node_modules",
            ".venv",
            "venv",
            "build",
            "dist",
            ".git",
            "__pycache__",
            "target",
            ".angular",
        }
        filtered = [
            f
            for f in found
            if f.is_file()
            and not any(exc in f.parts for exc in exclude_dirs)
        ]
        return sorted(filtered)

    def _get_language(self, fpath: Path) -> str:
        """Determine language from file extension."""
        ext = fpath.suffix.lower()
        lang_map = {
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".java": "java",
            ".py": "python",
            ".html": "html",
            ".htm": "html",
            ".cs": "csharp",
            ".kt": "kotlin",
        }
        return lang_map.get(ext, "unknown")

    def _extract_symbols(self, fpath: Path, rel_path: str) -> FileSymbols:
        """Extract symbols from a single file using LSP if available, else AST/regex."""
        lang = self._get_language(fpath)

        # TypeScript: use LSP (tsc --declaration)
        if lang in ("typescript", "javascript"):
            if self.lsp_client:
                try:
                    class_members = self.lsp_client.get_class_members(rel_path)
                    if class_members:
                        symbols = []
                        for prop in class_members.properties:
                            symbols.append(
                                SymbolInfo(
                                    name=prop.name,
                                    kind="property",
                                    type_hint=getattr(prop, "type", None),
                                    access_level=getattr(prop, "access", "public"),
                                    source_line=getattr(prop, "line", None),
                                )
                            )
                        for method in class_members.methods:
                            symbols.append(
                                SymbolInfo(
                                    name=method.name,
                                    kind="method",
                                    type_hint=getattr(method, "return_type", None),
                                    access_level=getattr(method, "access", "public"),
                                    source_line=getattr(method, "line", None),
                                )
                            )
                        return FileSymbols(
                            file_path=rel_path,
                            language=lang,
                            symbols=symbols,
                            class_names=[class_members.class_name],
                            extraction_method="lsp",
                        )
                except Exception as e:
                    logger.debug(f"  LSP failed for {rel_path}: {e}, falling back to regex")

            # Fallback: regex extraction for TypeScript
            return self._extract_typescript_regex(rel_path, fpath)

        # Java: use javalang if available
        elif lang == "java":
            if self.lsp_client:
                try:
                    class_members = self.lsp_client.get_class_members(rel_path)
                    if class_members:
                        symbols = []
                        for prop in class_members.properties:
                            symbols.append(
                                SymbolInfo(
                                    name=prop.name,
                                    kind="property",
                                    type_hint=getattr(prop, "type", None),
                                    access_level=getattr(prop, "access", "public"),
                                    source_line=getattr(prop, "line", None),
                                )
                            )
                        for method in class_members.methods:
                            symbols.append(
                                SymbolInfo(
                                    name=method.name,
                                    kind="method",
                                    type_hint=getattr(method, "return_type", None),
                                    access_level=getattr(method, "access", "public"),
                                    source_line=getattr(method, "line", None),
                                )
                            )
                        return FileSymbols(
                            file_path=rel_path,
                            language=lang,
                            symbols=symbols,
                            class_names=[class_members.class_name],
                            extraction_method="lsp",
                        )
                except Exception as e:
                    logger.debug(f"  LSP (Java) failed for {rel_path}: {e}, falling back to regex")

            return self._extract_java_regex(rel_path, fpath)

        # Python: use ast module
        elif lang == "python":
            return self._extract_python_ast(rel_path, fpath)

        # HTML: extract template properties
        elif lang in ("html", "htm"):
            return self._extract_html_bindings(rel_path, fpath)

        # Unknown: return empty
        else:
            return FileSymbols(
                file_path=rel_path,
                language=lang,
                symbols=[],
                class_names=[],
                extraction_method="unknown",
            )

    def _extract_typescript_regex(self, rel_path: str, fpath: Path) -> FileSymbols:
        """Extract TS symbols via regex (fallback from LSP)."""
        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
            return self._extract_typescript_regex_from_content(rel_path, content)
        except Exception as e:
            return FileSymbols(
                file_path=rel_path,
                language="typescript",
                symbols=[],
                class_names=[],
                extraction_method="regex_failed",
                error=str(e),
            )

    def _extract_typescript_regex_from_content(
        self, rel_path: str, content: str
    ) -> FileSymbols:
        """Extract TS symbols from in-memory content via regex."""
        import re

        symbols = []

        # Extract class/interface name
        class_match = None
        for line in content.split("\n"):
            if "export class " in line or "export interface " in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p in ("class", "interface") and i + 1 < len(parts):
                        class_match = parts[i + 1].strip("{")
                        break
            if class_match:
                break

        # Extract properties (simplified)
        prop_pattern = r"^\s+(\w+)\s*[:=]"
        for line in content.split("\n"):
            m = re.match(prop_pattern, line)
            if m:
                prop_name = m.group(1)
                symbols.append(
                    SymbolInfo(
                        name=prop_name,
                        kind="property",
                        owner_class=class_match,
                    )
                )

        # Extract methods with params and return type
        # Match: methodName(param1: type, param2: type): ReturnType {
        method_sig_pattern = re.compile(
            r"^\s+(?:async\s+)?(?:public\s+|private\s+|protected\s+)?"
            r"(\w+)\s*\(([^)]*)\)\s*(?::\s*([\w<>\[\]|&\s,]+?))?\s*\{",
            re.MULTILINE,
        )
        for m in method_sig_pattern.finditer(content):
            method_name = m.group(1)
            raw_params = m.group(2).strip()
            return_type = (m.group(3) or "void").strip()
            if method_name in ("constructor", "ngOnInit", "ngOnDestroy", "if", "for", "while"):
                continue
            params = [p.strip() for p in raw_params.split(",")] if raw_params else []
            symbols.append(
                SymbolInfo(
                    name=method_name,
                    kind="method",
                    owner_class=class_match,
                    params=params,
                    return_type=return_type,
                )
            )

        return FileSymbols(
            file_path=rel_path,
            language="typescript",
            symbols=symbols,
            class_names=[class_match] if class_match else [],
            extraction_method="regex",
        )

    def _extract_java_regex(self, rel_path: str, fpath: Path) -> FileSymbols:
        """Extract Java symbols via regex (fallback from LSP)."""
        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
            return self._extract_java_regex_from_content(rel_path, content)
        except Exception as e:
            return FileSymbols(
                file_path=rel_path,
                language="java",
                symbols=[],
                class_names=[],
                extraction_method="regex_failed",
                error=str(e),
            )

    def _extract_java_regex_from_content(
        self, rel_path: str, content: str
    ) -> FileSymbols:
        """Extract Java symbols from in-memory content via regex.

        Enhanced to capture full method signatures: return type, parameters,
        visibility, and owner class — not just method names.
        """
        import re

        symbols = []

        # Extract class/interface/enum names
        class_names = []
        class_pattern = re.compile(
            r"(?:public\s+)?(?:abstract\s+)?(class|interface|enum)\s+(\w+)"
        )
        for m in class_pattern.finditer(content):
            kind, name = m.group(1), m.group(2)
            class_names.append(name)
            symbols.append(
                SymbolInfo(
                    name=name,
                    kind=kind,
                    access_level="public",
                )
            )

        # Determine the primary class name (first public class)
        primary_class = class_names[0] if class_names else None

        # Extract fields with type information
        field_pattern = re.compile(
            r"^\s+(private|public|protected)\s+"
            r"(?:static\s+)?(?:final\s+)?"
            r"([\w<>\[\],\s]+?)\s+(\w+)\s*[;=]",
            re.MULTILINE,
        )
        for m in field_pattern.finditer(content):
            access = m.group(1)
            field_type = m.group(2).strip()
            field_name = m.group(3)
            # Skip common false positives
            if field_name in ("class", "interface", "enum", "return", "new", "throw"):
                continue
            symbols.append(
                SymbolInfo(
                    name=field_name,
                    kind="property",
                    type_hint=field_type,
                    access_level=access,
                    owner_class=primary_class,
                )
            )

        # Extract methods with FULL signature: visibility, return type, name, params
        method_pattern = re.compile(
            r"^\s+(public|private|protected)\s+"
            r"(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?"
            r"([\w<>\[\],\s]+?)\s+"  # return type
            r"(\w+)"                  # method name
            r"\s*\(([^)]*)\)",        # parameters
            re.MULTILINE,
        )
        for m in method_pattern.finditer(content):
            access = m.group(1)
            return_type = m.group(2).strip()
            method_name = m.group(3)
            raw_params = m.group(4).strip()

            # Skip constructors and common false positives
            if method_name in (primary_class, "if", "for", "while", "switch", "catch", "new"):
                continue
            # Skip annotations captured as methods
            if return_type in ("class", "interface", "enum"):
                continue

            # Parse parameters into list
            params = []
            if raw_params:
                for param in raw_params.split(","):
                    param = param.strip()
                    if param:
                        # Remove annotations like @RequestParam, @PathVariable
                        param_clean = re.sub(r"@\w+(?:\([^)]*\))?\s*", "", param).strip()
                        params.append(param_clean)

            symbols.append(
                SymbolInfo(
                    name=method_name,
                    kind="method",
                    type_hint=return_type,
                    access_level=access,
                    owner_class=primary_class,
                    params=params,
                    return_type=return_type,
                )
            )

        return FileSymbols(
            file_path=rel_path,
            language="java",
            symbols=symbols,
            class_names=class_names,
            extraction_method="regex",
        )

    def _extract_python_ast(self, rel_path: str, fpath: Path) -> FileSymbols:
        """Extract Python symbols using ast.parse()."""
        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
            return self._extract_python_ast_from_content(rel_path, content)
        except Exception as e:
            return FileSymbols(
                file_path=rel_path,
                language="python",
                symbols=[],
                class_names=[],
                extraction_method="ast_failed",
                error=str(e),
            )

    def _extract_python_ast_from_content(
        self, rel_path: str, content: str
    ) -> FileSymbols:
        """Extract Python symbols from in-memory content using ast.parse()."""
        import ast as _ast

        tree = _ast.parse(content)
        symbols = []
        class_names = []

        for node in _ast.walk(tree):
            if isinstance(node, _ast.ClassDef):
                class_names.append(node.name)
                # Extract class members
                for item in node.body:
                    if isinstance(item, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                        # Extract parameter names + annotations
                        params = []
                        for arg in item.args.args:
                            if arg.arg == "self":
                                continue
                            ann = ""
                            if arg.annotation:
                                try:
                                    ann = _ast.unparse(arg.annotation)
                                except Exception:
                                    ann = ""
                            params.append(
                                f"{arg.arg}: {ann}" if ann else arg.arg
                            )
                        # Extract return type annotation
                        ret_type = "None"
                        if item.returns:
                            try:
                                ret_type = _ast.unparse(item.returns)
                            except Exception:
                                ret_type = "?"
                        symbols.append(
                            SymbolInfo(
                                name=item.name,
                                kind="method" if item.name != "__init__" else "constructor",
                                source_line=item.lineno,
                                owner_class=node.name,
                                params=params,
                                return_type=ret_type,
                            )
                        )
                    elif isinstance(item, _ast.Assign):
                        for target in item.targets:
                            if isinstance(target, _ast.Name):
                                symbols.append(
                                    SymbolInfo(
                                        name=target.id,
                                        kind="property",
                                        source_line=item.lineno,
                                        owner_class=node.name,
                                    )
                                )

            elif isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and not class_names:
                # Top-level function
                params = []
                for arg in node.args.args:
                    ann = ""
                    if arg.annotation:
                        try:
                            ann = _ast.unparse(arg.annotation)
                        except Exception:
                            ann = ""
                    params.append(f"{arg.arg}: {ann}" if ann else arg.arg)
                ret_type = "None"
                if node.returns:
                    try:
                        ret_type = _ast.unparse(node.returns)
                    except Exception:
                        ret_type = "?"
                symbols.append(
                    SymbolInfo(
                        name=node.name,
                        kind="function",
                        source_line=node.lineno,
                        params=params,
                        return_type=ret_type,
                    )
                )

        return FileSymbols(
            file_path=rel_path,
            language="python",
            symbols=symbols,
            class_names=class_names,
            extraction_method="ast",
        )

    def _extract_html_bindings(self, rel_path: str, fpath: Path) -> FileSymbols:
        """Extract HTML template bindings (properties used in template)."""
        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
            return self._extract_html_bindings_from_content(rel_path, content)
        except Exception as e:
            return FileSymbols(
                file_path=rel_path,
                language="html",
                symbols=[],
                class_names=[],
                extraction_method="regex_failed",
                error=str(e),
            )

    def _extract_html_bindings_from_content(
        self, rel_path: str, content: str
    ) -> FileSymbols:
        """Extract HTML template bindings from in-memory content."""
        import re

        symbols = []

        patterns = [
            r"\{\{\s*(\w+)\s*\}\}",  # {{ property }}
            r"\[(\w+)\]\s*=",  # [property]=
            r"\(\w+\)\s*=",  # (event)=
            r"\*ng\w+\s*=\s*[\"']([^\"']*)[\"']",  # *ngIf="property"
            r"ngModel\s*=\s*[\"']([^\"']*)[\"']",  # ngModel
        ]

        found_props = set()
        for pattern in patterns:
            for match in re.finditer(pattern, content):
                prop_name = match.group(1).split(".")[0].split("[")[0]
                if prop_name and not prop_name.startswith("$"):
                    found_props.add(prop_name)

        for prop_name in sorted(found_props):
            symbols.append(
                SymbolInfo(
                    name=prop_name, kind="property", access_level="template"
                )
            )

        return FileSymbols(
            file_path=rel_path,
            language="html",
            symbols=symbols,
            class_names=[],
            extraction_method="regex",
        )


class WorkspaceSymbolIndex:
    """
    Queryable index of all workspace symbols.

    Usage:
        index = WorkspaceSymbolIndex(file_symbols_dict)
        
        # Check if a property exists in a file
        if index.has_property("add-members.component.ts", "isUserProjectMember"):
            print("Property already exists, don't duplicate!")
        
        # Get all properties used in HTML
        html_props = index.get_template_properties("add-members.component.html")
        
        # Get all files that reference a symbol
        refs = index.find_symbol_references("isUserProjectMember")
    """

    def __init__(self, file_symbols: Dict[str, FileSymbols]):
        self.file_symbols = file_symbols
        self._build_reverse_index()

    def _build_reverse_index(self):
        """Build reverse index: symbol → [files that have it]."""
        self._symbol_locations: Dict[str, List[str]] = defaultdict(list)
        for file_path, file_sym in self.file_symbols.items():
            for sym in file_sym.symbols:
                self._symbol_locations[sym.name].append(file_path)

    def has_property(self, file_path: str, property_name: str) -> bool:
        """Check if a file has a specific property."""
        file_sym = self.file_symbols.get(file_path)
        if not file_sym:
            return False
        return any(s.name == property_name and s.kind == "property" for s in file_sym.symbols)

    def has_method(self, file_path: str, method_name: str) -> bool:
        """Check if a file has a specific method."""
        file_sym = self.file_symbols.get(file_path)
        if not file_sym:
            return False
        return any(s.name == method_name and s.kind == "method" for s in file_sym.symbols)

    def get_properties(self, file_path: str) -> List[SymbolInfo]:
        """Get all properties in a file."""
        file_sym = self.file_symbols.get(file_path)
        return file_sym.get_properties() if file_sym else []

    def get_methods(self, file_path: str) -> List[SymbolInfo]:
        """Get all methods in a file."""
        file_sym = self.file_symbols.get(file_path)
        return file_sym.get_methods() if file_sym else []

    def get_template_properties(self, html_file_path: str) -> List[str]:
        """Get all properties referenced in an HTML template."""
        file_sym = self.file_symbols.get(html_file_path)
        if not file_sym or file_sym.language not in ("html", "htm"):
            return []
        return [s.name for s in file_sym.symbols if s.kind == "property"]

    def find_symbol_references(self, symbol_name: str) -> List[str]:
        """Find all files that reference a symbol."""
        return self._symbol_locations.get(symbol_name, [])

    def get_file_symbols(self, file_path: str) -> Optional[FileSymbols]:
        """Get all symbols for a file."""
        return self.file_symbols.get(file_path)

    def to_json(self) -> str:
        """Serialize entire index to JSON for UI display."""
        data = {
            file_path: file_sym.to_dict()
            for file_path, file_sym in self.file_symbols.items()
        }
        return json.dumps(data, indent=2)

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        total_files = len(self.file_symbols)
        total_symbols = sum(len(fs.symbols) for fs in self.file_symbols.values())
        total_properties = sum(
            len(fs.get_properties()) for fs in self.file_symbols.values()
        )
        total_methods = sum(len(fs.get_methods()) for fs in self.file_symbols.values())
        
        by_language = defaultdict(lambda: {"files": 0, "symbols": 0})
        for file_path, file_sym in self.file_symbols.items():
            by_language[file_sym.language]["files"] += 1
            by_language[file_sym.language]["symbols"] += len(file_sym.symbols)

        return {
            "total_files": total_files,
            "total_symbols": total_symbols,
            "total_properties": total_properties,
            "total_methods": total_methods,
            "by_language": dict(by_language),
        }
