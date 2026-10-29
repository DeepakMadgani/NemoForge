"""Polyglot Companion Resolver.

Resolves companion contracts and relationships across languages and frameworks:
  - Angular: .component.html <-> .component.ts <-> .component.scss / .spec.ts
  - React / Next.js: .tsx / .jsx <-> .module.css / .css, component <-> custom hooks
  - Vue: .vue SFC or sibling script/style files
  - Blazor / .NET: .razor <-> .razor.cs
  - Python (Django/Flask/FastAPI): templates/**/*.html <-> views.py / routes.py
  - Java (Spring): templates/**/*.html <-> *Controller.java
  - Universal Unit Tests: *test*.py <-> *.py, *Test.java <-> *.java, *.spec.ts <-> *.ts

Resolves companions first from in-memory session changes (_session_files / memory),
then from workspace disk.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class PolyglotCompanionResolver:
    """Discovers companion files across frameworks, templates, styles, and tests."""

    COMPANION_PAIRS = [
        # Angular
        (".component.html", [".component.ts", ".ts", ".component.scss", ".component.css", ".component.spec.ts"]),
        (".component.ts", [".component.html", ".component.scss", ".component.css", ".component.spec.ts"]),
        (".component.scss", [".component.html", ".component.ts"]),
        # React / Next.js
        (".tsx", [".module.css", ".module.scss", ".css", ".test.tsx", ".spec.tsx"]),
        (".jsx", [".module.css", ".module.scss", ".css", ".test.jsx", ".spec.jsx"]),
        # Vue
        (".vue", [".ts", ".js", ".css", ".scss", ".spec.ts", ".test.ts"]),
        # Blazor / C#
        (".razor", [".razor.cs", ".razor.css"]),
        (".razor.cs", [".razor"]),
        # Python
        (".html", ["views.py", "routes.py", "app.py"]),
        (".j2", ["views.py", "routes.py", "app.py"]),
        ("test_", [".py"]),
        ("_test.py", [".py"]),
        # Java Spring
        (".java", ["Test.java", "Tests.java"]),
    ]

    @classmethod
    def resolve_companion(
        cls,
        file_path: str,
        session_files: Optional[Dict[str, str]] = None,
        workspace_path: Optional[str] = None,
    ) -> Optional[Tuple[str, str]]:
        """Find the primary companion file and its content.

        Returns:
            Tuple of (companion_relative_path, companion_content) or None.
        """
        if not file_path:
            return None

        norm_fp = file_path.replace("\\", "/")
        p = Path(norm_fp)
        session_files = session_files or {}
        session_norm = {k.replace("\\", "/").lower(): v for k, v in session_files.items()}
        ws = Path(workspace_path) if workspace_path else None

        # Check framework-specific rules
        candidates = cls._generate_candidate_paths(norm_fp)

        # 1. Check in-memory session files first (most authoritative for active changes)
        for cand in candidates:
            cand_norm = cand.lower()
            if cand_norm in session_norm:
                return cand, session_norm[cand_norm]
            for s_path, s_content in session_norm.items():
                if s_path.endswith("/" + cand_norm) or cand_norm.endswith("/" + s_path):
                    return s_path, s_content
            # Basename matching in session files
            cand_base = Path(cand).name.lower()
            for s_path, s_content in session_norm.items():
                if Path(s_path).name.lower() == cand_base:
                    return s_path, s_content

        # 2. Check disk (direct path or workspace relative)
        for cand in candidates:
            cand_p = Path(cand)
            if cand_p.is_file():
                try:
                    return str(cand_p).replace("\\", "/"), cand_p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    pass
            if ws and ws.exists():
                ws_cand = ws / cand
                if ws_cand.is_file():
                    try:
                        rel = str(ws_cand.relative_to(ws)).replace("\\", "/")
                        return rel, ws_cand.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        pass
                # Also check sibling in same folder on disk
                sibling = (ws / p.parent) / cand_p.name
                if sibling.is_file():
                    try:
                        rel = str(sibling.relative_to(ws)).replace("\\", "/")
                        return rel, sibling.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        pass

        return None

    @classmethod
    def resolve_all_companions(
        cls,
        file_path: str,
        workspace_path: Optional[str] = None,
    ) -> List[str]:
        """Find ALL existing companion files on disk for a given source path."""
        if not file_path:
            return []
        norm_fp = file_path.replace("\\", "/")
        p = Path(norm_fp)
        ws = Path(workspace_path) if workspace_path else None
        candidates = cls._generate_candidate_paths(norm_fp)
        found: List[str] = []
        seen = {norm_fp.lower()}

        for cand in candidates:
            cand_p = Path(cand)
            resolved_str = None
            if cand_p.is_file():
                resolved_str = str(cand_p).replace("\\", "/")
            elif ws and ws.exists():
                ws_cand = ws / cand
                if ws_cand.is_file():
                    resolved_str = str(ws_cand.relative_to(ws)).replace("\\", "/")
                else:
                    sibling = (ws / p.parent) / cand_p.name
                    if sibling.is_file():
                        resolved_str = str(sibling.relative_to(ws)).replace("\\", "/")
            elif p.parent.exists():
                sibling = p.parent / cand_p.name
                if sibling.is_file():
                    resolved_str = str(sibling).replace("\\", "/")

            if resolved_str and resolved_str.lower() not in seen:
                seen.add(resolved_str.lower())
                found.append(resolved_str)

        # Cross-Service Contract Companions (FLAW-08)
        if ws:
            cross_comps = CrossServiceContractResolver.resolve_companion_files(norm_fp, str(ws))
            for cc in cross_comps:
                if cc.lower() not in seen:
                    seen.add(cc.lower())
                    found.append(cc)

        return found

    @classmethod
    def _generate_candidate_paths(cls, file_path: str) -> List[str]:
        """Generate plausible companion paths for a given source path."""
        p = Path(file_path)
        name = p.name
        parent = str(p.parent).replace("\\", "/")
        parent_prefix = f"{parent}/" if parent and parent != "." else ""
        candidates = []

        # Angular specific
        if name.endswith(".component.html"):
            base = name[:-len(".component.html")]
            candidates.append(f"{parent_prefix}{base}.component.ts")
            candidates.append(f"{parent_prefix}{base}.ts")
        elif name.endswith(".component.ts"):
            base = name[:-len(".component.ts")]
            candidates.append(f"{parent_prefix}{base}.component.html")
            candidates.append(f"{parent_prefix}{base}.component.scss")
            candidates.append(f"{parent_prefix}{base}.component.css")
        elif name.endswith(".component.scss") or name.endswith(".component.css"):
            base = name[:-len(".component.scss")] if name.endswith(".component.scss") else name[:-len(".component.css")]
            candidates.append(f"{parent_prefix}{base}.component.html")
            candidates.append(f"{parent_prefix}{base}.component.ts")
        elif name.endswith(".html"):
            base = name[:-len(".html")]
            candidates.append(f"{parent_prefix}{base}.component.ts")
            candidates.append(f"{parent_prefix}{base}.ts")
            # Python / Django / Flask view companions
            candidates.append(f"{parent_prefix}views.py")
            candidates.append(f"{parent_prefix}routes.py")
            if parent:
                parent_p = Path(parent)
                candidates.append(f"{parent_p.parent}/views.py".replace("\\", "/"))
                candidates.append(f"{parent_p.parent}/routes.py".replace("\\", "/"))

        # Blazor specific
        elif name.endswith(".razor"):
            base = name[:-len(".razor")]
            candidates.append(f"{parent_prefix}{base}.razor.cs")
        elif name.endswith(".razor.cs"):
            base = name[:-len(".razor.cs")]
            candidates.append(f"{parent_prefix}{base}.razor")

        # React / Vue
        elif name.endswith((".tsx", ".jsx", ".vue")):
            base = p.stem
            candidates.append(f"{parent_prefix}{base}.module.css")
            candidates.append(f"{parent_prefix}{base}.module.scss")
            candidates.append(f"{parent_prefix}{base}.css")
            candidates.append(f"{parent_prefix}use{base}.ts")

        # Tests <-> Source (Bidirectional Polyglot Resolution)
        norm_fp = file_path.replace("\\", "/")
        if name.startswith("test_") and name.endswith(".py"):
            base = name[len("test_"):]
            candidates.append(f"{parent_prefix}{base}")
        elif name.endswith("_test.py"):
            base = name[:-len("_test.py")] + ".py"
            candidates.append(f"{parent_prefix}{base}")
        elif name.endswith(".py"):
            base = name[:-len(".py")]
            candidates.append(f"{parent_prefix}test_{base}.py")
            candidates.append(f"{parent_prefix}{base}_test.py")
            candidates.append(f"tests/test_{base}.py")

        elif name.endswith((".spec.ts", ".test.ts")):
            base = name.replace(".spec.ts", ".ts").replace(".test.ts", ".ts")
            candidates.append(f"{parent_prefix}{base}")
        elif name.endswith(".ts") and not name.endswith(".d.ts"):
            base = name[:-len(".ts")]
            candidates.append(f"{parent_prefix}{base}.spec.ts")
            candidates.append(f"{parent_prefix}{base}.test.ts")

        # Java Source <-> Test (Maven & Gradle standard layout)
        elif name.endswith(("Test.java", "Tests.java")):
            suffix_len = len("Tests.java") if name.endswith("Tests.java") else len("Test.java")
            base = name[:-suffix_len] + ".java"
            candidates.append(f"{parent_prefix}{base}")
            if "/src/test/java/" in norm_fp or norm_fp.startswith("src/test/java/"):
                src_fp = norm_fp.replace("/src/test/java/", "/src/main/java/")
                if src_fp.startswith("src/test/java/"):
                    src_fp = "src/main/java/" + src_fp[len("src/test/java/"):]
                src_p = Path(src_fp)
                src_parent = str(src_p.parent).replace("\\", "/")
                candidates.append(f"{src_parent}/{base}")
        elif name.endswith(".java"):
            base = name[:-len(".java")]
            candidates.append(f"{parent_prefix}{base}Test.java")
            candidates.append(f"{parent_prefix}{base}Tests.java")
            if "/src/main/java/" in norm_fp or norm_fp.startswith("src/main/java/"):
                test_fp = norm_fp.replace("/src/main/java/", "/src/test/java/")
                if test_fp.startswith("src/main/java/"):
                    test_fp = "src/test/java/" + test_fp[len("src/main/java/"):]
                test_p = Path(test_fp)
                test_parent = str(test_p.parent).replace("\\", "/")
                candidates.append(f"{test_parent}/{base}Test.java")
                candidates.append(f"{test_parent}/{base}Tests.java")

        return candidates


# ─── Cross-Service Contract Resolver (FLAW-08) ───────────────────────────────

class CrossServiceContractResolver:
    """Discovers REST API contracts, routes, and companion files across microservices:

    - Java Spring Controllers (@RequestMapping, @GetMapping)
    - Spring Cloud Gateway routes (application.yml Path predicates)
    - Angular TypeScript HTTP Services (HttpClient.get/post)
    - JPA Entities and Flyway SQL migrations
    """

    @classmethod
    def resolve_contracts(cls, file_path: str, workspace_path: str) -> Dict[str, List[str]]:
        results: Dict[str, List[str]] = {
            "controllers": [],
            "gateway_routes": [],
            "frontend_services": [],
            "data_contracts": [],
        }
        if not file_path or not workspace_path:
            return results

        import re
        ws = Path(workspace_path)
        if not ws.exists():
            return results

        fp = Path(file_path)
        content = ""
        try:
            target_p = ws / file_path if not fp.is_absolute() else fp
            if target_p.is_file():
                content = target_p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return results

        extracted_endpoints = set()

        # 1. From Java Controller
        if file_path.endswith("Controller.java"):
            base_mappings = re.findall(r'@(?:Request|Rest)?Mapping\s*\(\s*(?:value\s*=\s*)?["\']([^"\']+)["\']', content)
            method_mappings = re.findall(r'@(?:Get|Post|Put|Delete|Patch)Mapping\s*\(\s*(?:value\s*=\s*)?["\']([^"\']+)["\']', content)
            base = base_mappings[0] if base_mappings else ""
            for mm in method_mappings:
                full = f"{base.rstrip('/')}/{mm.lstrip('/')}"
                extracted_endpoints.add(full)
            if base:
                extracted_endpoints.add(base)

        # 2. From Angular Service or Component
        elif file_path.endswith((".service.ts", ".component.ts")):
            for m in re.finditer(r'["\'](/api/v\d+/[^"\']+)["\']', content):
                extracted_endpoints.add(m.group(1))

        # Scan workspace for matching endpoints
        for ep in extracted_endpoints:
            parts = [p for p in ep.split("/") if p and p not in ("api", "v1", "v2")]
            domain_token = parts[0] if parts else ""

            # Find matching Controllers
            for c_path in ws.glob("**/*Controller.java"):
                c_rel = str(c_path.relative_to(ws)).replace("\\", "/")
                if domain_token.lower() in c_rel.lower() and c_rel not in results["controllers"]:
                    results["controllers"].append(c_rel)

            # Find matching Gateway configurations
            for g_path in ws.glob("**/application.yml"):
                g_rel = str(g_path.relative_to(ws)).replace("\\", "/")
                try:
                    g_text = g_path.read_text(encoding="utf-8", errors="ignore")
                    if ep in g_text or domain_token in g_text:
                        if g_rel not in results["gateway_routes"]:
                            results["gateway_routes"].append(g_rel)
                except Exception:
                    pass

            # Find matching Angular services
            for s_path in ws.glob("**/*.service.ts"):
                s_rel = str(s_path.relative_to(ws)).replace("\\", "/")
                try:
                    s_text = s_path.read_text(encoding="utf-8", errors="ignore")
                    if ep in s_text or domain_token in s_text:
                        if s_rel not in results["frontend_services"]:
                            results["frontend_services"].append(s_rel)
                except Exception:
                    pass

        # Match Entity <-> Migration
        if "entity" in file_path.lower() or file_path.endswith(".sql"):
            stem = fp.stem.replace("V1.1__", "").replace("V1__", "").replace("Entity", "")
            for sql_p in ws.glob("**/V*__*.sql"):
                sql_rel = str(sql_p.relative_to(ws)).replace("\\", "/")
                if stem.lower() in sql_rel.lower() and sql_rel not in results["data_contracts"]:
                    results["data_contracts"].append(sql_rel)

        return results

    @classmethod
    def resolve_companion_files(cls, file_path: str, workspace_path: str) -> List[str]:
        """Returns a deduplicated list of all cross-service companion files."""
        contracts = cls.resolve_contracts(file_path, workspace_path)
        all_files = []
        norm_orig = file_path.replace("\\", "/").lower()
        for group in contracts.values():
            for f in group:
                if f.lower() != norm_orig and f not in all_files:
                    all_files.append(f)
        return all_files

