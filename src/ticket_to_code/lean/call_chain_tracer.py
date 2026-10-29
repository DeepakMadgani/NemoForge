"""
Canonical Fullstack Call-Chain Tracer for Lean Path & Chatbot.

Traces execution from UI user actions (dialog text, button clicks, i18n keys)
through Angular components and business services, Saga orchestration commands,
down to Java backend Saga handlers and persistence services.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

logger = logging.getLogger(__name__)

# Prune generated artifacts, node_modules, build output, and server distribution folders
_PRUNE_DIRS = {
    "node_modules", "build", "target", ".gradle", "bin", ".git", ".angular",
    "dist", "coverage", "__pycache__", "out", ".idea", ".vscode", "server", "public"
}

# Generic UI framework plumbing to exclude from business service traces
_UI_PLUMBING = {
    "modalService", "notifService", "translateService", "cdr", "router",
    "fb", "formBuilder", "dialogService", "alertService", "changeDetectorRef",
    "themeManager", "sessionStorage", "localStorage"
}


class CallChainTracer:
    """Traces fullstack execution paths between Angular frontend, Sagas, and Java microservices."""

    def __init__(self, workspace_path: str):
        raw_p = Path(workspace_path).resolve()

        def _is_service_dir(d: Path) -> bool:
            if not d.is_dir() or d.name.startswith((".", "node_modules", "target", "build", "dist", "out")):
                return False
            return (
                any(
                    (d / m).exists()
                    for m in (
                        "pom.xml",
                        "build.gradle",
                        "build.gradle.kts",
                        "package.json",
                        "angular.json",
                        "pyproject.toml",
                        "Cargo.toml",
                        "go.mod",
                    )
                )
                or (d / "src").is_dir()
            )

        # 1. If raw_p itself contains multiple services/subprojects, it is already the monorepo root
        try:
            children_services = [d for d in raw_p.iterdir() if _is_service_dir(d)]
        except Exception:
            children_services = []

        if len(children_services) >= 2:
            self.root = raw_p
            return

        # 2. Only if raw_p is a single leaf service, check if parent is the monorepo root
        parent = raw_p.parent
        # CRITICAL: Never climb to drive root (C:\, /, etc.)
        if parent != raw_p and parent != Path(parent.anchor) and len(parent.parts) > 1:
            try:
                sibling_builds = [d for d in parent.iterdir() if _is_service_dir(d)]
                if len(sibling_builds) >= 2:
                    self.root = parent
                    return
            except Exception:
                pass

        self.root = raw_p

    @property
    def ui_root(self) -> Path:
        """Return the directory containing frontend application sources."""
        # 1. Check child directories for dedicated frontend project (standard monorepo structure)
        try:
            for child in sorted(self.root.iterdir()):
                if child.is_dir() and not child.name.startswith((".", "node_modules", "target", "build", "dist")):
                    if (child / "angular.json").exists() or ((child / "package.json").exists() and (child / "src" / "app").exists()):
                        return child
        except Exception:
            pass
        # 2. Fall back to child directories with src/app or package.json
        try:
            for child in sorted(self.root.iterdir()):
                if child.is_dir() and not child.name.startswith((".", "node_modules", "target", "build", "dist")):
                    if (child / "src" / "app").exists() or (child / "package.json").exists():
                        return child
        except Exception:
            pass
        # 3. If root itself contains frontend sources
        if (self.root / "angular.json").exists() or (self.root / "src" / "app").exists():
            return self.root
        return self.root

    def trace(self, query: str) -> Dict[str, Any]:
        """Perform bi-directional fullstack call-chain tracing for a query.
        
        Args:
            query: UI dialog phrase, i18n key, component name, saga command, or method.
        """
        query_clean = query.strip().strip("'\"")
        result: Dict[str, Any] = {
            "query": query_clean,
            "i18n_matches": [],
            "ui_components": [],
            "service_calls": [],
            "semantic_mutations": [],
            "model_contracts": [],
            "saga_commands": [],
            "orchestrated_workflows": [],
            "backend_handlers": [],
            "rest_endpoints": [],
            "domain_entities": [],
            "summary_trace": [],
            "is_fully_resolved": False,
        }

        # 1. Look for i18n translation matches if query looks like text
        i18n_keys = self._find_i18n_keys(query_clean)
        result["i18n_matches"] = i18n_keys

        # 2. Find UI templates and components referencing the keys or query
        search_anchors = [m["key"] for m in i18n_keys]
        if len(query_clean.split()) <= 4 and len(query_clean) >= 6:
            search_anchors.append(query_clean)

        ui_files = self._find_ui_components(search_anchors)

        # 2b. Direct file or component path match
        if not ui_files and (query_clean.endswith((".ts", ".html")) or ".component" in query_clean):
            q_norm = query_clean.replace("\\", "/").rstrip("/").split("/")[-1]
            q_stem = q_norm.replace(".component.ts", "").replace(".component.html", "").replace(".ts", "").replace(".html", "")
            for root, dirs, files in os.walk(self.ui_root):
                dirs[:] = [d for d in dirs if d not in _PRUNE_DIRS]
                for f in files:
                    if f.endswith((".component.ts", ".component.html")):
                        if q_norm == f or (q_stem and q_stem in f):
                            f_path = Path(root) / f
                            ts_p = f_path if f_path.suffix == ".ts" else f_path.with_suffix(".ts")
                            html_p = f_path if f_path.suffix == ".html" else f_path.with_suffix(".html")
                            c_dict = {
                                "html_file": str(html_p.relative_to(self.root)).replace("\\", "/") if html_p.exists() else str(f_path.relative_to(self.root)).replace("\\", "/"),
                                "ts_file": str(ts_p.relative_to(self.root)).replace("\\", "/") if ts_p.exists() else None,
                                "matched_anchors": [query_clean]
                            }
                            if not any(x["html_file"] == c_dict["html_file"] for x in ui_files):
                                ui_files.append(c_dict)

        # 2c. Direct service method match (e.g. updateProjectUsers)
        if not ui_files and re.match(r'^[a-zA-Z_]\w*$', query_clean):
            app_dir = self.ui_root / "src" / "app" if (self.ui_root / "src" / "app").exists() else self.ui_root
            for root, dirs, files in os.walk(app_dir):
                dirs[:] = [d for d in dirs if d not in _PRUNE_DIRS]
                for f in files:
                    if f.endswith(".component.ts"):
                        ts_p = Path(root) / f
                        try:
                            t_txt = ts_p.read_text(encoding="utf-8", errors="ignore")
                            if f".{query_clean}(" in t_txt or f" {query_clean}(" in t_txt:
                                html_p = ts_p.with_suffix(".html")
                                c_dict = {
                                    "html_file": str(html_p.relative_to(self.root)).replace("\\", "/") if html_p.exists() else str(ts_p.relative_to(self.root)).replace("\\", "/"),
                                    "ts_file": str(ts_p.relative_to(self.root)).replace("\\", "/"),
                                    "matched_anchors": [query_clean]
                                }
                                if not any(x["html_file"] == c_dict["html_file"] for x in ui_files):
                                    ui_files.append(c_dict)
                        except Exception:
                            pass

        result["ui_components"] = ui_files

        # 3. From UI components, extract method handlers, semantic mutations, and business service calls
        target_service_calls: List[Dict[str, Any]] = []
        for comp in ui_files:
            ts_path = comp.get("ts_file")
            if ts_path:
                details = self._analyze_ts_component(ts_path)
                comp["analysis"] = details
                result["ui_component_meta"] = details
                result["service_calls"].extend(details.get("service_calls", []))
                result["semantic_mutations"].extend(details.get("semantic_mutations", []))
                result["model_contracts"].extend(details.get("model_contracts", []))
                target_service_calls.extend(details.get("service_calls", []))

        # 4. Resolve exact saga commands invoked by those specific service methods
        resolved_cmds = self._find_sagas_for_service_calls(target_service_calls)

        # 4b. Direct saga / workflow command check if query_clean is itself a saga name
        direct_wfs = self._find_workflow_definitions([query_clean])
        if direct_wfs:
            for dw in direct_wfs:
                s_name = dw.get("saga_name")
                if s_name and s_name not in resolved_cmds:
                    resolved_cmds.append(s_name)
            # Reverse-trace from saga command to UI callers if no UI files found yet
            if not result["ui_components"]:
                app_dir = self.ui_root / "src" / "app" if (self.ui_root / "src" / "app").exists() else self.ui_root
                matched_svc_methods = set()
                for root, dirs, files in os.walk(app_dir):
                    dirs[:] = [d for d in dirs if d not in _PRUNE_DIRS]
                    for f in files:
                        if f.endswith(".service.ts"):
                            s_p = Path(root) / f
                            try:
                                s_txt = s_p.read_text(encoding="utf-8", errors="ignore")
                                if f"'{query_clean}'" in s_txt or f'"{query_clean}"' in s_txt:
                                    for m_m in re.finditer(r'(?:public|private|protected)?\s*([a-zA-Z_]\w*)\s*\([^)]*\)[^{]*\{', s_txt):
                                        m_name = m_m.group(1)
                                        m_body = self._extract_method_body(s_txt, m_name)
                                        if m_body and (f"'{query_clean}'" in m_body or f'"{query_clean}"' in m_body):
                                            matched_svc_methods.add(m_name)
                            except Exception:
                                pass

                if matched_svc_methods:
                    for root, dirs, files in os.walk(app_dir):
                        dirs[:] = [d for d in dirs if d not in _PRUNE_DIRS]
                        for fc in files:
                            if fc.endswith(".component.ts"):
                                cp_p = Path(root) / fc
                                try:
                                    c_txt = cp_p.read_text(encoding="utf-8", errors="ignore")
                                    for sm in matched_svc_methods:
                                        if f".{sm}(" in c_txt:
                                            hp = cp_p.with_suffix(".html")
                                            cd = {
                                                "html_file": str(hp.relative_to(self.root)).replace("\\", "/") if hp.exists() else str(cp_p.relative_to(self.root)).replace("\\", "/"),
                                                "ts_file": str(cp_p.relative_to(self.root)).replace("\\", "/"),
                                                "matched_anchors": [query_clean, sm]
                                            }
                                            if not any(x["html_file"] == cd["html_file"] for x in result["ui_components"]):
                                                result["ui_components"].append(cd)
                                                details = self._analyze_ts_component(str(cp_p.relative_to(self.root)).replace("\\", "/"))
                                                cd["analysis"] = details
                                                if not result.get("ui_component_meta"):
                                                    result["ui_component_meta"] = details
                                                result["service_calls"].extend(details.get("service_calls", []))
                                                result["model_contracts"].extend(details.get("model_contracts", []))
                                except Exception:
                                    pass

        result["saga_commands"] = sorted(list(set(resolved_cmds)))

        # 5. Extract multi-step orchestrated workflow definitions (e.g. from YAML/JSON workflow configs)
        orchestrated_wfs = self._find_workflow_definitions(result["saga_commands"])
        result["orchestrated_workflows"] = orchestrated_wfs

        # 6. Trace Saga commands and workflow steps to Java backend handlers (SagasConfig -> Handler -> Service)
        all_commands_to_trace = list(result["saga_commands"])
        for wf in orchestrated_wfs:
            for step in wf.get("steps", []):
                s_name = step.get("step")
                if s_name and s_name not in all_commands_to_trace:
                    all_commands_to_trace.append(s_name)

        for cmd in all_commands_to_trace:
            backend_matches = self._find_backend_saga_handlers(cmd)
            result["backend_handlers"].extend(backend_matches)

        # 7. Extract domain concepts across query, translations, and component files
        domain_stems = self._extract_domain_stems(query_clean, result)

        # 8. Polyglot REST Endpoint Tracing (Python FastAPI/Flask, Go Gin/Stdlib, Spring MVC)
        result["rest_endpoints"] = self._find_rest_endpoints(query_clean, domain_stems)

        # 9. Check if full chain is resolved (supporting Saga or REST)
        has_ui = bool(result["i18n_matches"] or result["ui_components"])
        has_transport = bool(result["saga_commands"] or result["rest_endpoints"])
        has_backend = any(bh.get("role") in ("Saga Handler", "Service Implementation", "REST Controller", "HTTP Handler") for bh in result["backend_handlers"]) or bool(result["rest_endpoints"])
        result["is_fully_resolved"] = bool((has_ui or has_transport) and has_backend)

        # 10. Discover monorepo microservices (grounds architecture to prevent hallucinated services)
        result["microservices"] = self._discover_microservices()

        # 11. Discover domain entities & database table ownership across microservices
        result["domain_entities"] = self._discover_domain_entities(domain_stems)

        # 12. Discover existing sibling domain sagas & handlers (e.g. removeContractMembers)
        result["related_domain_capabilities"] = self._find_related_domain_capabilities(query_clean, domain_stems)

        # 13. Build chronological summary trace (capped to eliminate noise)
        result["summary_trace"] = self._build_summary(result)
        return result

    def _find_i18n_keys(self, text_fragment: str) -> List[Dict[str, str]]:
        """Fast search of JSON translation catalogs for text matches, preferring src/assets/i18n."""
        matches = []
        lower_frag = text_fragment.lower()
        if len(lower_frag) < 4:
            return matches

        ui_dir = self.ui_root
        if not ui_dir.exists():
            return matches

        for root, dirs, files in os.walk(ui_dir):
            dirs[:] = [d for d in dirs if d not in _PRUNE_DIRS]
            if "src" not in root or "i18n" not in root:
                continue
            for f in files:
                if f.endswith(".json"):
                    full_p = Path(root) / f
                    try:
                        data = json.loads(full_p.read_text(encoding="utf-8", errors="ignore"))
                        rel = str(full_p.relative_to(self.root)).replace("\\", "/")
                        self._recurse_json(data, "", lower_frag, rel, matches)
                    except Exception:
                        continue

        matches.sort(key=lambda m: (0 if "en.json" in m["file"] else 1, 0 if "src/" in m["file"] else 1))

        unique = {}
        for m in matches:
            if m["key"] not in unique:
                unique[m["key"]] = m
        return list(unique.values())

    def _recurse_json(self, obj: Any, prefix: str, target: str, file_rel: str, matches: List[Dict[str, str]]):
        if isinstance(obj, dict):
            for k, v in obj.items():
                cur = f"{prefix}.{k}" if prefix else k
                self._recurse_json(v, cur, target, file_rel, matches)
        elif isinstance(obj, str):
            clean_obj = obj.strip().lower()
            if (target in clean_obj) or (len(clean_obj) >= 15 and clean_obj in target) or (len(target) >= 6 and target in prefix.lower()):
                matches.append({
                    "key": prefix,
                    "text": obj,
                    "file": file_rel
                })

    def _find_ui_components(self, anchor_strings: List[str]) -> List[Dict[str, Any]]:
        """Fast search of Angular components using the specified anchors."""
        results = []
        clean_anchors = [s for s in anchor_strings if s]
        if not clean_anchors:
            return results

        app_dir = self.ui_root / "src" / "app"
        if not app_dir.exists():
            app_dir = self.ui_root

        for root, dirs, files in os.walk(app_dir):
            dirs[:] = [d for d in dirs if d not in _PRUNE_DIRS]
            for f in files:
                if f.endswith(".html"):
                    html_p = Path(root) / f
                    try:
                        content = html_p.read_text(encoding="utf-8", errors="ignore")
                        hit_anchors = [a for a in clean_anchors if a in content]
                        if hit_anchors:
                            ts_p = html_p.with_suffix(".ts")
                            results.append({
                                "html_file": str(html_p.relative_to(self.root)).replace("\\", "/"),
                                "ts_file": str(ts_p.relative_to(self.root)).replace("\\", "/") if ts_p.exists() else None,
                                "matched_anchors": hit_anchors
                            })
                    except Exception:
                        continue
        return results

    def _extract_call_args(self, text: str, start_idx: int) -> str:
        """Extract arguments inside parentheses with balanced parenthesis tracking."""
        depth = 1
        in_string: Optional[str] = None
        escaped = False
        for idx in range(start_idx, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == in_string:
                    in_string = None
                continue
            if ch in ("'", '"', "`"):
                in_string = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return text[start_idx:idx].strip()
        return text[start_idx:min(start_idx + 80, len(text))].strip()

    def _analyze_ts_component(self, ts_rel_path: str) -> Dict[str, Any]:
        """Analyze TypeScript component, resolving injected service fields, semantic mutations, and model contracts."""
        out = {
            "component_class": None,
            "component_methods": [],
            "injected_services": {},
            "component_inputs": [],
            "ui_idioms": {},
            "service_calls": [],
            "semantic_mutations": [],
            "model_contracts": []
        }
        full_path = self.root / ts_rel_path
        if not full_path.exists():
            return out

        try:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
            # 0. Extract component class name and declared action methods
            cls_m = re.search(r'export\s+class\s+(\w+)', content)
            if cls_m:
                out["component_class"] = cls_m.group(1)

            methods = []
            for mm in re.finditer(r'^\s*(?:public|private|protected)?\s*([a-zA-Z_]\w*)\s*\([^)]*\)\s*(?::\s*[^\{]+)?\s*\{', content, flags=re.MULTILINE):
                m_name = mm.group(1)
                if m_name not in ("constructor", "ngOnInit", "ngOnDestroy", "ngAfterViewInit", "ngOnChanges", "if", "for", "switch") and m_name not in methods:
                    methods.append(m_name)
            out["component_methods"] = methods

            # 1. Map constructor fields to their types: e.g. `private memberService: MemberService`
            field_types: Dict[str, str] = {}
            for param in re.findall(r"(?:private|public|protected)\s+(\w+)\s*:\s*(\w+)", content):
                field_types[param[0]] = param[1]
            out["injected_services"] = field_types

            # 1b. Component Inputs and UI Idioms
            inputs = [m[0] for m in re.findall(r'@Input\(\)\s*([a-zA-Z_]\w*)', content)]
            out["component_inputs"] = inputs

            modal_close = list(set(re.findall(r'this\.(\w+\.(?:close|dismiss))\s*\(', content)))
            notifs = list(set(re.findall(r'this\.(\w+\.(?:notifySuccess|notifyError|notify|openSnackBar|showSuccess|showError))\s*\(', content)))
            data_binding = None
            if "this.config[" in content or "this.config." in content:
                cfg_props = re.findall(r'this\.config\[[\'"](\w+)[\'"]\]', content) + re.findall(r'this\.config\.(\w+)', content)
                ex = f" (e.g. this.config['{cfg_props[0]}'])" if cfg_props else ""
                data_binding = f"Receives context via @Input() config: any{ex}; NOT this.data"
            elif inputs:
                data_binding = f"@Input() bindings: {', '.join(inputs)}"

            out["ui_idioms"] = {
                "modal_close": modal_close[:2],
                "notifications": notifs[:2],
                "data_binding": data_binding
            }

            # 2. Identify legitimate business service fields (must be injected with type ending in Service or field ending in Service/Svc)
            valid_service_fields = {
                f: cls for f, cls in field_types.items()
                if (cls.endswith("Service") or f.endswith(("Service", "Svc")))
                and f not in _UI_PLUMBING and cls not in _UI_PLUMBING
            }

            seen = set()
            for svc_field, svc_class in valid_service_fields.items():
                pattern = rf"\bthis\.{re.escape(svc_field)}\.([a-zA-Z_]\w*)\s*\("
                for m in re.finditer(pattern, content):
                    method = m.group(1)
                    if method in ("subscribe", "pipe", "forEach", "push", "unsubscribe", "pop", "splice"):
                        continue
                    key = (svc_field, method)
                    if key not in seen:
                        seen.add(key)
                        args = self._extract_call_args(content, m.end())
                        if len(args) > 80:
                            args = args[:77] + "..."
                        # Check if any other UI components also call this method
                        sibling_callers = []
                        try:
                            ui_app = self.ui_root / "src" / "app"
                            search_root = ui_app if ui_app.exists() else self.ui_root
                            curr_file_name = Path(ts_rel_path).name
                            for r_sub, d_sub, f_sub in os.walk(search_root):
                                d_sub[:] = [d for d in d_sub if d not in _PRUNE_DIRS]
                                for fn in f_sub:
                                    if fn.endswith(".component.ts") and fn != curr_file_name:
                                        p_sub = Path(r_sub) / fn
                                        try:
                                            sub_txt = p_sub.read_text(encoding="utf-8", errors="ignore")
                                            if f".{method}(" in sub_txt:
                                                cls_m = re.search(r"export\s+class\s+(\w+)", sub_txt)
                                                sibling_callers.append(cls_m.group(1) if cls_m else fn)
                                        except Exception:
                                            pass
                        except Exception:
                            pass

                        out["service_calls"].append({
                            "component": ts_rel_path,
                            "service_field": svc_field,
                            "service_class": svc_class,
                            "method": method,
                            "args": args,
                            "sibling_callers": sorted(list(set(sibling_callers)))
                        })

            # 3. Detect semantic mutations (e.g. member.roleName = RoleConstant.MEMBER_KEY)
            for m in re.finditer(r'(\w+)\.(roleName|status|role|state|accessLevel)\s*=\s*([^;\n]+)', content):
                target = m.group(1)
                prop = m.group(2)
                val = m.group(3).strip().strip("'\"")
                out["semantic_mutations"].append(f"{target}.{prop} = {val}")

            # 4. Resolve imported models and extract declared property names
            model_names = set()
            for imp_symbols in re.findall(r"import\s*\{([^}]+)\}\s*from", content):
                for sym in imp_symbols.split(","):
                    sym = sym.strip()
                    if sym and not sym.endswith(("Service", "Component", "Module", "Directive", "Pipe", "Constant")):
                        if any(term in sym.lower() for term in ("member", "user", "project", "contract", "role", "dto", "item", "entity")):
                            model_names.add(sym)

            for m_name in sorted(list(model_names)):
                m_contract = self._discover_ts_model_contract(m_name)
                if m_contract:
                    out["model_contracts"].append(m_contract)
        except Exception as e:
            logger.debug(f"Error analyzing TS component: {e}")
        return out

    def _discover_ts_model_contract(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Find TypeScript interface or class definition in UI workspace and extract property names."""
        for root, dirs, files in os.walk(self.ui_root):
            dirs[:] = [d for d in dirs if d not in _PRUNE_DIRS]
            for f in files:
                if f.endswith((".ts", ".d.ts")) and not f.endswith((".spec.ts", ".test.ts")):
                    full_p = Path(root) / f
                    try:
                        txt = full_p.read_text(encoding="utf-8", errors="ignore")
                        pattern = rf"(?:export\s+)?(?:interface|class)\s+{re.escape(model_name)}\b[^{{]*\{{([^}}]+)\}}"
                        m = re.search(pattern, txt)
                        if m:
                            body = m.group(1)
                            props = re.findall(r"([a-zA-Z_]\w*)\s*\??:", body)
                            if props:
                                rel_p = str(full_p.relative_to(self.root)).replace("\\", "/")
                                return {
                                    "model_name": model_name,
                                    "file": rel_p,
                                    "properties": sorted(list(set(props)))
                                }
                    except Exception:
                        pass
        return None

    def _find_sagas_for_service_calls(self, service_calls: List[Dict[str, Any]]) -> List[str]:
        """Resolve Saga commands exclusively within the specific service file owning each method."""
        saga_cmds = []
        app_dir = self.ui_root / "src" / "app"
        if not app_dir.exists():
            app_dir = self.ui_root

        for sc in service_calls:
            method_name = sc["method"]
            svc_class = sc.get("service_class", "")
            svc_field = sc.get("service_field", "")
            expected_stem = re.sub(r'Service$', '', svc_class, flags=re.I).lower() if svc_class else svc_field.lower().replace("service", "")

            for root, dirs, files in os.walk(app_dir):
                dirs[:] = [d for d in dirs if d not in _PRUNE_DIRS]
                for f in files:
                    if f.endswith(".service.ts"):
                        full_p = Path(root) / f
                        try:
                            text = full_p.read_text(encoding="utf-8", errors="ignore")
                            if svc_class and f"class {svc_class}" not in text:
                                if expected_stem and expected_stem not in f.lower():
                                    continue
                            body = self._extract_method_body(text, method_name)
                            if body:
                                found = re.findall(r"doCommand\(\s*['\"]([^'\"]+)['\"]", body)
                                saga_cmds.extend(found)
                        except Exception:
                            continue

        return list(set(saga_cmds))

    def _extract_method_body(self, text: str, method_name: str) -> Optional[str]:
        """Extract method implementation using balanced brace tracking."""
        m = re.search(rf"\b{re.escape(method_name)}\s*\([^)]*\)[^{{]*\{{", text)
        if not m:
            return None

        start_idx = m.end() - 1
        depth = 0
        in_string: Optional[str] = None
        escaped = False

        for idx in range(start_idx, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == in_string:
                    in_string = None
                continue

            if ch in ("'", '"', "`"):
                in_string = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start_idx:idx + 1]

        return None

    def _get_java_files_index(self) -> Dict[str, List[Path]]:
        """Index of all .java file basename -> List[Path] across workspace."""
        if not hasattr(self, "_java_files_cache"):
            idx: Dict[str, List[Path]] = {}
            for root, dirs, files in os.walk(self.root):
                dirs[:] = [d for d in dirs if d not in _PRUNE_DIRS]
                for f in files:
                    if f.endswith(".java"):
                        idx.setdefault(f, []).append(Path(root) / f)
            self._java_files_cache = idx
        return self._java_files_cache

    def _get_saga_config_files(self) -> List[Tuple[Path, str]]:
        """Lazy cached list of (Path, text) for Saga configuration classes."""
        if not hasattr(self, "_saga_config_cache"):
            cfgs: List[Tuple[Path, str]] = []
            java_idx = self._get_java_files_index()
            for fn, paths in java_idx.items():
                if "Config" in fn or "Saga" in fn:
                    for p in paths:
                        try:
                            cfgs.append((p, p.read_text(encoding="utf-8", errors="ignore")))
                        except Exception:
                            pass
            self._saga_config_cache = cfgs
        return self._saga_config_cache

    def _get_workflow_yaml_candidates(self) -> List[Tuple[Path, str]]:
        """Lazy cached list of (Path, text) for workflow YAML files."""
        if not hasattr(self, "_workflow_yaml_cache"):
            cands = []
            for root, dirs, files in os.walk(self.root):
                dirs[:] = [d for d in dirs if d not in _PRUNE_DIRS]
                parts = Path(root).parts
                if any(p in ("test", "tests") for p in parts):
                    continue
                for f in files:
                    if f.endswith((".yml", ".yaml", ".json")):
                        fp = Path(root) / f
                        try:
                            txt = fp.read_text(encoding="utf-8", errors="ignore")
                            if "steps:" in txt or '"steps"' in txt or "'steps'" in txt or "RoutingKey" in txt or "routingKey" in txt:
                                cands.append((fp, txt))
                        except Exception:
                            pass
            self._workflow_yaml_cache = cands
        return self._workflow_yaml_cache

    def _find_workflow_definitions(self, saga_commands: List[str]) -> List[Dict[str, Any]]:
        """Inspect workflow YAML/JSON files to extract all distributed saga steps across services."""
        import yaml
        results = []
        if not saga_commands:
            return results

        yaml_candidates = self._get_workflow_yaml_candidates()
        for yml_p, content in yaml_candidates:
            try:
                matched_cmds = [cmd for cmd in saga_commands if f"{cmd}:" in content or f'"{cmd}"' in content or f"'{cmd}'" in content]
                if not matched_cmds:
                    continue

                data = yaml.safe_load(content)
                if not isinstance(data, dict):
                    continue

                def _search_sagas(node):
                    out_sagas = {}
                    if isinstance(node, dict):
                        for k, v in node.items():
                            if k in matched_cmds and isinstance(v, dict) and "steps" in v and isinstance(v["steps"], dict):
                                out_sagas[k] = v
                            else:
                                out_sagas.update(_search_sagas(v))
                    return out_sagas

                found_map = _search_sagas(data)
                rel_path = str(yml_p.relative_to(self.root)).replace("\\", "/")

                for s_name, s_cfg in found_map.items():
                    completion_rk = s_cfg.get("completionRoutingKey") if isinstance(s_cfg, dict) else None
                    step_entries = []
                    for step_name, step_cfg in s_cfg.get("steps", {}).items():
                        r_key = step_cfg.get("stepRoutingKey") if isinstance(step_cfg, dict) else None
                        params = step_cfg.get("params") if isinstance(step_cfg, dict) else None
                        handlers = self._find_backend_saga_handlers(step_name)
                        h_names = [h.get("class") for h in handlers if h.get("role") == "Saga Handler"]
                        target_svc = None
                        if handlers:
                            for h in handlers:
                                if h.get("role") == "Saga Handler":
                                    parts = Path(h["file"]).parts
                                    target_svc = parts[0] if parts else None
                                    break
                        step_entries.append({
                            "step": step_name,
                            "stepRoutingKey": r_key,
                            "routing_key": r_key,
                            "params": params,
                            "handlers": h_names,
                            "target_service": target_svc or (r_key.split(".")[-1] if r_key and "." in r_key else None)
                        })

                    results.append({
                        "saga_name": s_name,
                        "defined_in": rel_path,
                        "completion_routing_key": completion_rk,
                        "steps": step_entries
                    })
            except Exception:
                continue

        unique = {}
        for r in results:
            if r["saga_name"] not in unique:
                unique[r["saga_name"]] = r
        return list(unique.values())

    def _find_backend_saga_handlers(self, saga_command: str) -> List[Dict[str, Any]]:
        """Trace Saga command through SagasConfig to the actual Java Handler and its service calls."""
        if not hasattr(self, "_backend_handlers_cache"):
            self._backend_handlers_cache = {}
        if saga_command in self._backend_handlers_cache:
            return self._backend_handlers_cache[saga_command]

        handlers = []
        resolved_handler_classes: Set[str] = set()

        config_candidates = self._get_saga_config_files()
        for full_p, text in config_candidates:
            if f'"{saga_command}"' in text or f"'{saga_command}'" in text:
                rel_p = str(full_p.relative_to(self.root)).replace("\\", "/")
                h_match = re.search(rf'addStepHandler\(\s*["\']{re.escape(saga_command)}["\']\s*,\s*(\w+)\s*\)', text)
                handler_cls = None
                if h_match:
                    var_name = h_match.group(1)
                    if var_name and var_name[0].isupper() and var_name.endswith("Handler"):
                        handler_cls = var_name
                    else:
                        t_match = re.search(rf"(\w+Handler)\s+{re.escape(var_name)}\b", text)
                        if t_match:
                            handler_cls = t_match.group(1)
                        elif var_name and var_name.endswith("Handler"):
                            handler_cls = var_name[0].upper() + var_name[1:]
                else:
                    h_cls_match = re.search(rf'handleBy\(\s*(\w+Handler)\.class\s*\)', text)
                    if h_cls_match:
                        handler_cls = h_cls_match.group(1)

                if handler_cls:
                    resolved_handler_classes.add(handler_cls)
                handlers.append({
                    "file": rel_p,
                    "class": full_p.stem,
                    "role": "Saga Configuration",
                    "saga_command": saga_command,
                    "dispatched_handler": handler_cls
                })

        java_idx = self._get_java_files_index()
        for h_cls in resolved_handler_classes:
            matched_paths = java_idx.get(f"{h_cls}.java", [])
            for h_file in matched_paths:
                try:
                    h_text = h_file.read_text(encoding="utf-8", errors="ignore")
                    rel_p = str(h_file.relative_to(self.root)).replace("\\", "/")
                    invocations = re.findall(r"(\w+Service)\.(\w+)\(([^)]*)\)", h_text)

                    ext_match = re.search(r'class\s+\w+\s+extends\s+(\w+)', h_text)
                    super_cls = ext_match.group(1) if ext_match else None

                    proc_match = re.search(r'(?:protected|public)\s+void\s+(process|execute|handle)\s*\(([^)]+)\)', h_text)
                    entry_method = f"protected void {proc_match.group(1)}({proc_match.group(2)}) throws Exception" if proc_match else None

                    state_idioms = sorted(list(set(re.findall(r'\b(getFieldAs\w+|getField|setField\w+|hasField\w+|get\w+AsUUID)\b', h_text))))
                    repos = sorted(list(set(re.findall(r'\b(\w+Repository)\b', h_text))))

                    handlers.append({
                        "file": rel_p,
                        "class": h_cls,
                        "role": "Saga Handler",
                        "saga_command": saga_command,
                        "super_class": super_cls,
                        "entry_method": entry_method,
                        "state_idioms": state_idioms[:6],
                        "repositories": repos[:5],
                        "invoked_services": [f"{s}.{m}" for s, m, _ in invocations[:5]]
                    })

                    # Polymorphic Interface Resolution & Concrete Implementation Tracing
                    for s, m, _ in invocations[:5]:
                        impls = self._get_interface_implementations(s)
                        for impl_path, impl_cls in impls:
                            impl_rel = str(impl_path.relative_to(self.root)).replace("\\", "/")
                            method_snippet = ""
                            try:
                                impl_text = impl_path.read_text(encoding="utf-8", errors="ignore")
                                mb = self._extract_method_body(impl_text, m)
                                if mb:
                                    method_snippet = mb[:300]
                            except Exception:
                                pass

                            handlers.append({
                                "file": impl_rel,
                                "class": impl_cls,
                                "role": "Service Implementation",
                                "interface": s,
                                "method": m,
                                "method_body": method_snippet,
                                "saga_command": saga_command,
                            })
                except Exception:
                    pass

        self._backend_handlers_cache[saga_command] = handlers
        return handlers

    def _extract_method_body(self, text: str, method_name: str) -> Optional[str]:
        """Extracts the balanced-brace method body for a method name."""
        pattern = rf"(?:public|protected|private|static|\s)*[\w<>\[\],\s]+\s+{re.escape(method_name)}\s*\([^)]*\)[^{{]*\{{"
        m = re.search(pattern, text)
        if not m:
            return None
        start_idx = m.end() - 1
        depth = 0
        for i in range(start_idx, len(text)):
            c = text[i]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return text[start_idx:i+1]
        return None

    def _get_interface_implementations(self, interface_name: str) -> List[Tuple[Path, str]]:
        """Resolves concrete classes implementing an interface (e.g. MemberService -> MemberServiceImpl)."""
        clean_name = interface_name[0].upper() + interface_name[1:] if interface_name else ""
        java_idx = self._get_java_files_index()
        results: List[Tuple[Path, str]] = []
        seen_paths: Set[Path] = set()

        # 1. Direct convention: MemberServiceImpl.java, MemberServiceBean.java, DefaultMemberService.java
        for cand in [f"{clean_name}Impl.java", f"{clean_name}Bean.java", f"Default{clean_name}.java"]:
            if cand in java_idx:
                for p in java_idx[cand]:
                    if p not in seen_paths:
                        results.append((p, p.stem))
                        seen_paths.add(p)

        # 2. Scanning for 'implements InterfaceName'
        pattern = re.compile(rf'\bclass\s+(\w+)\b[^{{]*\bimplements\b[^{{]*\b{re.escape(clean_name)}\b')
        for fname, paths in java_idx.items():
            for p in paths:
                if p in seen_paths:
                    continue
                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                    m = pattern.search(txt)
                    if m:
                        results.append((p, m.group(1)))
                        seen_paths.add(p)
                except Exception:
                    pass

        return results

    def _discover_microservices(self) -> List[str]:
        """Detect actual microservice directories in monorepo to ground architecture."""
        out = []
        try:
            for d in sorted(self.root.iterdir()):
                if d.is_dir() and not d.name.startswith((".", "node_modules", "target", "build", "dist", "out")):
                    if any((d / m).exists() for m in ("pom.xml", "build.gradle", "package.json", "pyproject.toml", "Cargo.toml", "go.mod")):
                        out.append(d.name)
        except Exception:
            pass
        return out

    def _extract_domain_stems(self, query: str, result: Dict[str, Any]) -> Set[str]:
        """Extract domain concepts from query, resolved translations, components, and service calls."""
        stop_words = {
            "this", "that", "with", "from", "will", "have", "would", "then", "when",
            "access", "error", "half", "msg", "popup", "button", "click", "dialog",
            "modal", "please", "table", "component", "service", "item", "data"
        }
        raw_words = list(re.findall(r"[a-zA-Z]{4,}", query))

        # Add words from translation texts
        for m in result.get("i18n_matches", []):
            raw_words.extend(re.findall(r"[a-zA-Z]{4,}", m.get("text", "")))
            raw_words.extend(re.findall(r"[a-zA-Z]{4,}", m.get("key", "")))

        # Add words from UI component and service methods
        for c in result.get("ui_components", []):
            raw_words.extend(re.findall(r"[a-zA-Z]{4,}", c.get("html_file", "")))
        for sc in result.get("service_calls", []):
            raw_words.extend(re.findall(r"[a-zA-Z]{4,}", sc.get("method", "")))
            raw_words.extend(re.findall(r"[a-zA-Z]{4,}", sc.get("service_field", "")))

        tokens = [w.lower() for w in raw_words if w.lower() not in stop_words]
        stems = {t[:-1] if t.endswith("s") else t for t in tokens} | set(tokens)
        return stems

    def _discover_domain_entities(self, stems: Set[str]) -> List[Dict[str, Any]]:
        """Discover ORM entities and database tables matching domain concepts across all microservices."""
        entities = []
        seen = set()
        if not stems:
            return entities

        for root, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in dirs if d not in _PRUNE_DIRS]
            parts = Path(root).parts
            if "src" not in parts or "main" not in parts or any(p in ("test", "tests") for p in parts):
                continue
            for f in files:
                if not f.endswith((".java", ".kt", ".ts", ".py")):
                    continue
                stem_lower = Path(f).stem.lower()
                if any(s in stem_lower for s in stems):
                    full_p = Path(root) / f
                    try:
                        text = full_p.read_text(encoding="utf-8", errors="ignore")
                        table_m = re.search(r'@Table\s*\(\s*name\s*=\s*["\']([^"\']+)["\']', text)
                        entity_m = re.search(r'@Entity\b', text)
                        if table_m or entity_m:
                            tname = table_m.group(1) if table_m else Path(f).stem.lower()
                            rel_parts = full_p.relative_to(self.root).parts
                            owning_svc = rel_parts[0] if len(rel_parts) > 1 else "root"
                            key = (tname, Path(f).stem, owning_svc)
                            if key not in seen:
                                seen.add(key)
                                repo_methods = []
                                repo_signatures = []
                                uses_uuid = False
                                repo_name = f"{Path(f).stem}Repository"
                                r_cand_paths = [
                                    full_p.parent / f"{repo_name}.java",
                                    full_p.parent.parent / "repository" / f"{repo_name}.java",
                                    full_p.parent.parent / "dao" / f"{repo_name}.java",
                                ]
                                r_found = next((p for p in r_cand_paths if p.exists()), None)
                                if not r_found:
                                    svc_dir = self.root / owning_svc if owning_svc != "root" else self.root
                                    for r_file in svc_dir.glob(f"**/{repo_name}.java"):
                                        if not any(part in _PRUNE_DIRS for part in r_file.parts):
                                            r_found = r_file
                                            break
                                if r_found:
                                    try:
                                        r_txt = r_found.read_text(encoding="utf-8", errors="ignore")
                                        if "UUID" in r_txt:
                                            uses_uuid = True
                                        for rm in re.finditer(r'([A-Za-z0-9_<>, \t]+)\s+([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*;', r_txt):
                                            ret_type = rm.group(1).strip()
                                            m_id = rm.group(2).strip()
                                            params = rm.group(3).strip()
                                            if m_id not in ("equals", "hashCode", "toString", "default") and not ret_type.startswith(("//", "/*", "*")):
                                                if m_id not in repo_methods:
                                                    repo_methods.append(m_id)
                                                clean_params = re.sub(r'@\w+(\([^)]*\))?\s*', '', params).strip()
                                                clean_params = re.sub(r'\s+', ' ', clean_params)
                                                repo_signatures.append(f"{ret_type} {m_id}({clean_params})")
                                    except Exception:
                                        pass
                                id_m = re.search(r'@Id\s+(?:private|protected|public)?\s*(\w+)\s+id;', text)
                                id_type = id_m.group(1) if id_m else ("UUID" if uses_uuid else "String")
                                entities.append({
                                    "entity": Path(f).stem,
                                    "table": tname,
                                    "service": owning_svc,
                                    "file": str(full_p.relative_to(self.root)).replace("\\", "/"),
                                    "id_type": id_type,
                                    "repo_methods": repo_methods[:6],
                                    "repo_signatures": repo_signatures[:6]
                                })
                    except Exception:
                        pass
        return entities

    def _find_related_domain_capabilities(self, query: str, domain_stems: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """Identify existing sibling Sagas and handlers dynamically from repository files."""
        caps = []
        stop_words = {"this", "that", "with", "from", "will", "have", "would", "then", "when", "access", "error", "half", "msg"}
        tokens = [w.lower() for w in re.findall(r"[a-zA-Z]{4,}", query) if w.lower() not in stop_words]
        stems = {t[:-1] if t.endswith("s") else t for t in tokens} | set(tokens)
        if domain_stems:
            stems |= domain_stems
        if not stems:
            return caps

        import yaml
        yaml_candidates = self._get_workflow_yaml_candidates()

        declared_sagas: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for yml_p, content in yaml_candidates:
            try:
                data = yaml.safe_load(content)
                if not isinstance(data, dict):
                    continue

                rel_yml = str(yml_p.relative_to(self.root)).replace("\\", "/")
                def _collect(node):
                    if isinstance(node, dict):
                        for k, v in node.items():
                            if isinstance(v, dict) and "steps" in v and isinstance(v["steps"], dict):
                                declared_sagas[k] = (rel_yml, v)
                            else:
                                _collect(v)
                _collect(data)
            except Exception:
                continue

        del_verbs = {"remove", "delete", "revoke", "drop", "cancel", "clear", "purge", "demote"}
        meta_words = {"this", "that", "with", "from", "will", "have", "would", "then", "when", "access", "error", "half", "msg", "project", "data", "app"}
        entity_stems = stems - del_verbs - meta_words
        has_action_intent = bool(stems & del_verbs)

        matched_sagas = []
        for s_name, (rel_yml, s_cfg) in declared_sagas.items():
            s_lower = s_name.lower()
            score = 0
            matched_ents = [e for e in entity_stems if e in s_lower]
            if matched_ents:
                score += 50 * len(matched_ents)
            if has_action_intent and any(v in s_lower for v in del_verbs):
                score += 30

            # If specific entities were identified, only keep sagas that match at least one domain entity
            if entity_stems:
                if matched_ents:
                    matched_sagas.append((score, s_name, rel_yml, s_cfg))
            elif score > 0:
                matched_sagas.append((score, s_name, rel_yml, s_cfg))

        matched_sagas.sort(key=lambda item: item[0], reverse=True)

        for score, s_name, rel_yml, s_cfg in matched_sagas[:4]:
            steps_dict = s_cfg.get("steps", {})
            step_details = []
            for step_name in steps_dict.keys():
                handlers = self._find_backend_saga_handlers(step_name)
                handler_names = [h.get("class") for h in handlers if h.get("role") == "Saga Handler"]
                if handler_names:
                    step_details.append(f"`{step_name}` ({', '.join(handler_names)})")
                else:
                    step_details.append(f"`{step_name}`")

            caps.append({
                "saga_name": s_name,
                "defined_in": rel_yml,
                "steps": step_details
            })

        return caps

    def _find_rest_endpoints(self, query: str, domain_stems: Set[str]) -> List[Dict[str, Any]]:
        """Polyglot REST endpoint tracing for Python (FastAPI/Flask), Go (Gin/Stdlib), Node (Express), and Spring."""
        endpoints = []
        target_words = {w.lower() for w in re.findall(r'[a-zA-Z]{3,}', query)} | domain_stems

        for root, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in dirs if d not in _PRUNE_DIRS]
            for f in files:
                ext = Path(f).suffix.lower()
                if ext not in (".py", ".go", ".ts", ".js", ".java"):
                    continue
                full_p = Path(root) / f
                try:
                    txt = full_p.read_text(encoding="utf-8", errors="ignore")
                    rel_p = str(full_p.relative_to(self.root)).replace("\\", "/")

                    # Python FastAPI / Flask: @app.get("/users") or @router.post("/items")
                    if ext == ".py":
                        for m in re.finditer(r'@(?:app|router|api)\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']', txt, re.IGNORECASE):
                            method, path = m.group(1).upper(), m.group(2)
                            if any(w in path.lower() for w in target_words):
                                endpoints.append({
                                    "file": rel_p,
                                    "framework": "FastAPI/Flask",
                                    "http_method": method,
                                    "path": path,
                                    "role": "REST Controller"
                                })

                    # Go: router.GET("/users", handler) or http.HandleFunc("/items", handler)
                    elif ext == ".go":
                        for m in re.finditer(r'\.(?:GET|POST|PUT|DELETE|Handle|HandleFunc)\s*\(\s*["\']([^"\']+)["\']\s*,\s*([a-zA-Z_]\w*)', txt):
                            path, handler = m.group(1), m.group(2)
                            if any(w in path.lower() for w in target_words):
                                endpoints.append({
                                    "file": rel_p,
                                    "framework": "Go HTTP/Gin",
                                    "path": path,
                                    "handler": handler,
                                    "role": "HTTP Handler"
                                })

                    # Spring MVC: @GetMapping("/users") or @RequestMapping(value="/items")
                    elif ext == ".java":
                        for m in re.finditer(r'@(Get|Post|Put|Delete)Mapping\s*\(\s*(?:value\s*=\s*)?["\']([^"\']+)["\']', txt):
                            method, path = m.group(1).upper(), m.group(2)
                            if any(w in path.lower() for w in target_words):
                                endpoints.append({
                                    "file": rel_p,
                                    "framework": "Spring MVC",
                                    "http_method": method,
                                    "path": path,
                                    "role": "REST Controller"
                                })
                except Exception:
                    pass

        return endpoints[:10]

    def _build_summary(self, result: Dict[str, Any]) -> List[str]:
        """Produce an authoritative step-by-step trace narrative."""
        summary = []
        if result["i18n_matches"]:
            for m in result["i18n_matches"][:2]:
                summary.append(f"1. [UI Dialog String] Key `{m['key']}` in `{m['file']}`: \"{m['text']}\"")
        if result["ui_components"]:
            for c in result["ui_components"][:2]:
                meta = c.get("analysis") or result.get("ui_component_meta") or {}
                cls_name = meta.get("component_class")
                methods = meta.get("component_methods", [])
                injected = meta.get("injected_services", {})
                idioms = meta.get("ui_idioms", {})

                comp_line = f"2. [UI Component] Template: `{c['html_file']}` | Logic: `{c.get('ts_file')}`"
                if cls_name:
                    comp_line += f" (Class: `{cls_name}`)"
                summary.append(comp_line)

                detail_lines = []
                if methods:
                    detail_lines.append(f"Declared Action Methods: {', '.join(f'`{m}()`' for m in methods[:4])}")
                if injected:
                    inj_str = ", ".join(f"`{k}: {v}`" for k, v in list(injected.items())[:6])
                    detail_lines.append(f"Injected Services: {inj_str}")
                if idioms.get("data_binding"):
                    detail_lines.append(f"Data Binding: {idioms['data_binding']}")
                if idioms.get("modal_close") or idioms.get("notifications"):
                    calls = idioms.get("modal_close", []) + idioms.get("notifications", [])
                    detail_lines.append(f"Observed Dialog APIs: {', '.join(f'`{cl}`' for cl in calls)}")

                if detail_lines:
                    summary.append(
                        "2a. [UI Component Ground Truth] " + ". ".join(detail_lines) + ". "
                        "UI CONSTRAINTS: Use the component's existing injected services and dialog APIs. "
                        "Do NOT hallucinate Angular Material (MatDialog, MatSnackBar, this.data)."
                    )
        if result.get("semantic_mutations"):
            mut_str = ", ".join(result["semantic_mutations"][:3])
            summary.append(f"2b. [UI Action Semantics] Component updates fields: `{mut_str}` (Role Demotion / state update, not an entity deletion).")
        if result["service_calls"]:
            for sc in result["service_calls"][:2]:
                s_callers = sc.get("sibling_callers", [])
                warn_shared = ""
                if s_callers:
                    warn_shared = (
                        f"\n    CRITICAL ARCHITECTURAL GUARD (SHARED METHOD): Method `{sc['method']}` is ALSO invoked by sibling component(s): {', '.join(f'`{c}`' for c in s_callers)}. "
                        f"NEVER modify the underlying `{sc['method']}` saga or mutation to add cascading deletion side-effects, as doing so will break {', '.join(f'`{c}`' for c in s_callers)} (e.g. editing user profile/details). "
                        "Always introduce a dedicated new saga command (e.g. `remove...`) instead!"
                    )
                summary.append(
                    f"3. [Frontend Service] `{sc['service_field']}.{sc['method']}({sc.get('args', '')})`. "
                    "SAGA DISPATCH PROTOCOL: Sagas are dispatched via GraphQL mutation `this.sagaService.doCommand('<sagaCommand>', payload).pipe(switchMap(...))`. "
                    f"NEVER invent REST POST endpoints like `http.post('/api/sagas')`.{warn_shared}"
                )

        # Frontend Model Contract (Interface field ground truth)
        if result.get("model_contracts"):
            for mc in result["model_contracts"][:2]:
                props_list = mc["properties"][:10]
                props_str = ", ".join(f"`{p}`" for p in props_list)
                example_props = ", ".join(f"`{p}`" for p in props_list[:3])
                summary.append(
                    f"3b. [Frontend Model Contract] Model `{mc['model_name']}` in `{mc['file']}` defines valid properties: {props_str}. "
                    f"Use exact declared properties (such as {example_props}); NEVER guess or invent non-existent property names or fictional status strings (e.g. 'DELETED')."
                )

        # Orchestrated workflows with distributed step breakdown
        if result.get("orchestrated_workflows"):
            for ow in result["orchestrated_workflows"][:2]:
                steps_lines = []
                for st in ow.get("steps", []):
                    st_target = f" -> handled by `{', '.join(st['handlers'])}` in `{st['target_service']}`" if st.get("handlers") and st.get("target_service") else (f" in `{st['target_service']}`" if st.get("target_service") else "")
                    param_str = f" (params: {st['params']})" if st.get("params") else ""
                    steps_lines.append(f"    - Step `{st['step']}` (stepRoutingKey: `{st.get('routing_key')}`){st_target}{param_str}")
                comp_rk = ow.get("completion_routing_key")
                comp_rk_str = f" `completionRoutingKey: {comp_rk}`" if comp_rk else " `completionRoutingKey: ...`"
                summary.append(
                    f"4. [Distributed Saga Workflow] Command `{ow['saga_name']}` in `{ow['defined_in']}` orchestrates {len(ow.get('steps', []))} steps across microservices:\n"
                    + "\n".join(steps_lines)
                    + f"\n    Canonical Saga YAML schema (Spring Cloud / message-driven sagas):\n"
                    + f"    {ow['saga_name']}:\n"
                    + f"      {comp_rk_str.strip()}\n"
                    + f"      steps:\n"
                    + f"        <stepName>:\n"
                    + f"          stepRoutingKey: <routingKey>\n"
                    + "    (STRICT RULE: `steps:` is a flat map of named step keys, NEVER a YAML list with '- step:'. Keys use camelCase `stepRoutingKey`, NEVER snake_case `routing_key`. DO NOT put `handler:` or Java class names in `application.yml` steps; in message-driven sagas, routing is specified via `stepRoutingKey:`, and handlers are registered in Java configuration classes via `.addStepHandler('<stepName>', handlerInstance)`). "
                    "All cross-service steps must be declared under `steps:`."
                )
        elif result["saga_commands"]:
            for cmd in result["saga_commands"][:2]:
                summary.append(f"4. [Orchestration / Saga Command] `{cmd}`")

        # Polyglot REST Endpoints
        if result.get("rest_endpoints"):
            for ep in result["rest_endpoints"][:3]:
                http_m = ep.get("http_method", "HTTP")
                summary.append(
                    f"4b. [REST/HTTP Route] `{http_m} {ep['path']}` in `{ep['file']}` ({ep.get('framework', 'REST')}) -> Role: `{ep.get('role', 'Controller')}`"
                )

        # Java handlers fallback if no workflow definition was parsed
        if not result.get("orchestrated_workflows") and result["backend_handlers"]:
            for bh in result["backend_handlers"][:4]:
                if bh.get("role") == "Saga Configuration":
                    dispatched = f" -> delegates to `{bh['dispatched_handler']}`" if bh.get("dispatched_handler") else ""
                    summary.append(f"5. [Saga Config] `{bh['class']}` in `{bh['file']}`{dispatched}")
                elif bh.get("role") == "Saga Handler":
                    invs = ", ".join(bh.get("invoked_services", []))
                    summary.append(f"5b. [Backend Java Handler] `{bh['class']}` in `{bh['file']}` -> calls `{invs or 'internal logic'}`")

        # Real Backend Handler Architectural Archetype
        real_handlers = [bh for bh in result.get("backend_handlers", []) if bh.get("role") == "Saga Handler" and bh.get("super_class")]
        if real_handlers:
            rh = real_handlers[0]
            idioms_str = ", ".join(f"`{i}`" for i in rh.get("state_idioms", []))
            repos_str = ", ".join(f"`{r}`" for r in rh.get("repositories", []))
            invs_str = ", ".join(f"`{s}`" for s in rh.get("invoked_services", []))
            dep_parts = [p for p in (repos_str, invs_str) if p]
            dep_summary = f"Observed dependencies: {', '.join(dep_parts)}. " if dep_parts else ""
            summary.append(
                f"5c. [Backend Handler Archetype] Existing handler in `{rh['file']}` extends `{rh['super_class']}` implementing `{rh.get('entry_method', 'protected void process(SagaState state)')}`. "
                f"Observed state APIs: {idioms_str or '`getField(...)`'}. {dep_summary}"
                "CALLING CONVENTION: Helper methods like `setFieldForNextStep(key, value, state)` and `getFieldAsObject(key, Class, state)` are protected methods on the handler base class taking `state` as an argument. "
                "NEVER call `state.setFieldForNextStep(...)`. "
                "REGISTRATION: Handlers are @Component classes registered in the service's configuration class via `.addStepHandler('<stepName>', handlerInstance)`. "
                "ARCHITECTURAL NOTE (DUAL-PLANE DOMAIN & SECURITY REVOCATION): In distributed enterprise architectures, access involves TWO layers: "
                "(1) Relational domain records (e.g. member/junction database tables in domain service). "
                "(2) External security & permissions realms (e.g. CMS Area security groups in permissions/area service). "
                "Any cascading removal MUST coordinate BOTH layers across services—never leave security permissions or CMS groups active while only deleting the database record."
            )

        # Inferred database entities and repository methods (heuristic candidate)
        if result.get("domain_entities"):
            ent_strs = []
            for e in result["domain_entities"][:6]:
                r_sigs = f" (repo signatures: {'; '.join(e.get('repo_signatures', []))})" if e.get("repo_signatures") else (f" (repo methods: {', '.join(e['repo_methods'])})" if e.get("repo_methods") else "")
                id_info = f" [id: {e.get('id_type', 'UUID')}]" if e.get("id_type") else ""
                ent_strs.append(f"Table `{e['table']}` (`{e['entity']}`{id_info}) owned by `{e['service']}`{r_sigs}")
            summary.append(
                f"6. [Inferred Domain Entities & Repositories (Candidate)] Inferred entities matching domain stems: {'; '.join(ent_strs)}. "
                "TYPE & SIGNATURE SAFETY: Use exact parameter types declared on repository methods (e.g. java.util.UUID vs String). "
                "Only use verified declared repository methods; do NOT invent non-existent query methods."
            )

        # Architecture grounding (purely discovered from disk)
        ms = result.get("microservices", [])
        if ms:
            services_str = ", ".join(f"`{s}`" for s in ms)
            summary.append(
                f"7. [Monorepo Architecture Context] Discovered services in repository: {services_str}. "
                "Only invoke or modify these discovered services; do not assume or invent non-existent services."
            )

        # Existing sibling domain capabilities (purely discovered from disk)
        rel_caps = result.get("related_domain_capabilities", [])
        for cap in rel_caps[:3]:
            steps_summary = ", ".join(cap.get("steps", []))
            summary.append(
                f"8. [Existing Sibling Saga] Command `{cap['saga_name']}` found in `{cap['defined_in']}` -> steps: {steps_summary}. "
                "Check existing workflows before proposing new ones."
            )

        return summary
