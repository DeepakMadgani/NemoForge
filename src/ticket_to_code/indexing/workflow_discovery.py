"""
Workflow Discovery Layer — Full Execution Path Tracing

Discovers complete execution paths across CC4E:
  Route → Component → Store → Service → API → Controller → BackendService → Entity

Reuses:
  - SQLite AST index (symbol search)
  - Neo4j graph (edge storage)
  - Repository Intelligence catalogs (api_catalog.json, domain_catalog.json)
  - UIGraphEnricher labels (UIComponent, UIStore, AngularService)

Does NOT create:
  - New databases
  - New parsers
  - Duplicate indexes

Author: Deepak Madgani
Date: June 2026
"""

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class WorkflowDiscovery:
    """
    Discovers complete execution workflows across a fullstack application.

    Operates as a graph enrichment pass:
    1. Reads existing data sources (SQLite AST, file content, catalogs)
    2. Writes explicit typed edges into the existing Neo4j graph
    3. Assembles full workflow paths and generates workflow_catalog.json
    """

    SKIP_DIRS = {".git", "node_modules", "dist", "build", ".angular",
                 "coverage", "__pycache__", ".venv", "target", ".aviator"}

    def __init__(self, sqlite_store, neo4j_store, workspace_path: str,
                 catalog_dir: Optional[str] = None):
        self.sqlite_store = sqlite_store
        self.neo4j_store = neo4j_store
        self.workspace_path = Path(workspace_path)
        
        if catalog_dir:
            self.catalog_dir = Path(catalog_dir)
        else:
            env_catalog = os.getenv("AVIATOR_CATALOG_DIR")
            if env_catalog and Path(env_catalog).exists():
                self.catalog_dir = Path(env_catalog)
            else:
                self.catalog_dir = self.workspace_path / "Repository_Intelligence" / "catalogs"

        # Loaded catalogs
        self._api_catalog: List[dict] = []
        self._domain_catalog: List[dict] = []
        self._service_catalog: List[dict] = []

        # Discovered elements
        self.routes: List[dict] = []
        self.components: List[dict] = []
        self.stores: List[dict] = []
        self.services: List[dict] = []
        self.api_endpoints: List[dict] = []
        self.controllers: List[dict] = []
        self.backend_services: List[dict] = []
        self.entities: List[dict] = []
        self.workflows: Dict[str, dict] = {}

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def discover_workflows(self) -> Dict[str, dict]:
        """
        Execute full workflow discovery pipeline.

        Returns:
            Dictionary of discovered workflows keyed by workflow name.
        """
        logger.info("🔄 START Workflow Discovery")

        if not self.neo4j_store:
            logger.error("❌ CRITICAL: Neo4j unavailable. Workflow Discovery cannot run without graph database.")
            raise ValueError("Neo4j store is required for Workflow Discovery")

        try:
            # Load existing catalogs
            self._load_catalogs()

            # Phase 1: Angular Routes
            self._discover_routes()

            # Phase 2: Component → Service/Store mapping
            self._discover_component_dependencies()

            # Routes are discovered before components, so link them again now
            # that component nodes exist in Neo4j.
            self._link_routes_to_components()

            # Phase 3: Store → Service mapping
            self._discover_store_dependencies()

            # Phase 4: Service → API endpoint mapping
            self._discover_service_api_calls()

            # Phase 5: API → Spring Controller mapping
            self._discover_api_controller_mapping()

            # Phase 6: Controller → Backend Service mapping
            self._discover_controller_dependencies()

            # Phase 7: Backend Service → Entity mapping
            self._discover_entity_mapping()

            # Phase 8: Assemble workflows
            self._assemble_workflows()

            # Phase 9: Ingest Semantic Catalog
            self._ingest_semantic_catalog()

            # Save catalog
            self._save_workflow_catalog()

            logger.info(f"Routes discovered: {len(self.routes)}")
            logger.info(f"Components discovered: {len(self.components)}")
            logger.info(f"Stores discovered: {len(self.stores)}")
            logger.info(f"Services discovered: {len(self.services)}")
            logger.info(f"API Endpoints discovered: {len(self.api_endpoints)}")
            logger.info(f"Controllers discovered: {len(self.controllers)}")
            logger.info(f"Backend Services discovered: {len(self.backend_services)}")
            logger.info(f"Entities discovered: {len(self.entities)}")

            if self.neo4j_store:
                for edge in ["RENDERS", "USES", "CALLS", "CALLS_API", "HANDLED_BY"]:
                    res = self.neo4j_store.query(f"MATCH ({{workspace: $workspace}})-[r:{edge}]->({{workspace: $workspace}}) RETURN count(r) as c", {"workspace": str(self.workspace_path)})
                    rel_count = res[0]["c"] if res else 0
                    logger.info(f"[:{edge}] nodes created: N/A, relationships created: {rel_count}")

            logger.info("✅ Workflow Discovery Complete")

            return self.workflows

        except Exception as e:
            logger.error(f"❌ Workflow Discovery failed: {e}", exc_info=True)
            return {}

    # =========================================================================
    # CATALOG LOADING
    # =========================================================================

    def _load_catalogs(self):
        """Load existing Repository Intelligence catalogs."""
        for name, attr in [
            ("api_catalog.json", "_api_catalog"),
            ("domain_catalog.json", "_domain_catalog"),
            ("service_catalog.json", "_service_catalog"),
        ]:
            path = self.catalog_dir / name
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        setattr(self, attr, json.load(f))
                    logger.info(f"  📂 Loaded {name}: {len(getattr(self, attr))} entries")
                except Exception as e:
                    logger.warning(f"  ⚠️ Failed to load {name}: {e}")

    # =========================================================================
    # PHASE 1: ANGULAR ROUTE EXTRACTION
    # =========================================================================

    def _discover_routes(self):
        """
        Parse Angular routing modules to extract Route → Component edges.

        Handles:
        - loadChildren: () => import('...').then(m => m.SomeModule)
        - loadComponent: () => import('...').then(m => m.SomeComponent)
        - component: SomeComponent (direct reference)
        """
        logger.info("  🗺️ Phase 1: Discovering Angular Routes...")

        routing_files = self._find_files("*routing*.ts")
        routing_files.extend(self._find_files("app-routing.module.ts"))
        routing_files = list(set(routing_files))  # deduplicate

        # Patterns for route extraction
        load_component_re = re.compile(
            r"path:\s*['\"]([^'\"]+)['\"].*?"
            r"loadComponent:\s*\(\)\s*=>\s*import\(['\"]([^'\"]+)['\"]\)"
            r"\.then\(\s*\w+\s*=>\s*\w+\.(\w+)\s*\)",
            re.DOTALL
        )
        load_children_re = re.compile(
            r"path:\s*['\"]([^'\"]+)['\"].*?"
            r"loadChildren:\s*\(\)\s*=>\s*import\(['\"]([^'\"]+)['\"]\)"
            r"\.then\(\s*\w+\s*=>\s*\w+\.(\w+)\s*\)",
            re.DOTALL
        )
        direct_component_re = re.compile(
            r"path:\s*['\"]([^'\"]+)['\"].*?"
            r"component:\s*(\w+)",
            re.DOTALL
        )

        for rf in routing_files:
            try:
                content = rf.read_text(encoding="utf-8", errors="ignore")
                rel_path = str(rf.relative_to(self.workspace_path)).replace("\\", "/")

                # Extract loadComponent routes (standalone components)
                for match in load_component_re.finditer(content):
                    route_path, import_path, component_name = match.groups()
                    route = {
                        "path": route_path,
                        "component": component_name,
                        "import_path": import_path,
                        "source_file": rel_path,
                        "type": "loadComponent"
                    }
                    self.routes.append(route)
                    self._create_route_node(route)

                # Extract loadChildren routes (lazy-loaded modules)
                for match in load_children_re.finditer(content):
                    route_path, import_path, module_name = match.groups()
                    route = {
                        "path": route_path,
                        "component": module_name,
                        "import_path": import_path,
                        "source_file": rel_path,
                        "type": "loadChildren"
                    }
                    self.routes.append(route)
                    self._create_route_node(route)

                # Extract direct component routes
                for match in direct_component_re.finditer(content):
                    route_path, component_name = match.groups()
                    if component_name not in ("CapabilityGuard",):
                        route = {
                            "path": route_path,
                            "component": component_name,
                            "import_path": "",
                            "source_file": rel_path,
                            "type": "direct"
                        }
                        self.routes.append(route)
                        self._create_route_node(route)

            except Exception as e:
                logger.warning(f"  ⚠️ Failed to parse routing file {rf}: {e}")

        logger.info(f"  🗺️ Discovered {len(self.routes)} routes")

    def _create_route_node(self, route: dict):
        """Create Route node and RENDERS edge in Neo4j."""
        if not self.neo4j_store:
            return
        self.neo4j_store.query(
            "MERGE (r:Route {workspace: $workspace, path: $path}) "
            "SET r.source_file = $source_file, r.type = $type",
            {"path": route["path"], "source_file": route["source_file"],
             "type": route["type"], "workspace": str(self.workspace_path)}
        )

        # Link Route to Component (search by UIComponent label first, then Class)
        self.neo4j_store.query("""
            MATCH (r:Route {workspace: $workspace, path: $path})
            OPTIONAL MATCH (c:UIComponent {workspace: $workspace, name: $component})
            OPTIONAL MATCH (c2:Class {workspace: $workspace, name: $component})
            WITH r, COALESCE(c, c2) AS target
            WHERE target IS NOT NULL
            MERGE (r)-[:RENDERS]->(target)
        """, {"path": route["path"], "component": route["component"], "workspace": str(self.workspace_path)})

    def _link_routes_to_components(self):
        """Re-link Route nodes after component nodes have been created."""
        if not self.neo4j_store:
            return

        for route in self.routes:
            self.neo4j_store.query("""
                MATCH (r:Route {workspace: $workspace, path: $path})
                OPTIONAL MATCH (c:UIComponent {workspace: $workspace, name: $component})
                OPTIONAL MATCH (c2:Class {workspace: $workspace, name: $component})
                WITH r, COALESCE(c, c2) AS target
                WHERE target IS NOT NULL
                MERGE (r)-[:RENDERS]->(target)
            """, {"path": route["path"], "component": route["component"], "workspace": str(self.workspace_path)})

    # =========================================================================
    # PHASE 2: COMPONENT → SERVICE/STORE MAPPING
    # =========================================================================

    def _discover_component_dependencies(self):
        """
        For each Angular component, extract constructor injections and
        create USES edges to Services and Stores.
        """
        logger.info("  🧩 Phase 2: Discovering Component Dependencies...")

        component_files = self._find_files("*.component.ts")
        known_services = self._get_known_class_names("Service")
        known_stores = self._get_known_class_names("Store") | self._get_known_class_names("State")

        for cf in component_files:
            try:
                content = cf.read_text(encoding="utf-8", errors="ignore")
                rel_path = str(cf.relative_to(self.workspace_path)).replace("\\", "/")

                # Extract class name
                class_match = re.search(
                    r"export\s+class\s+(\w+Component)", content
                )
                if not class_match:
                    continue
                component_name = class_match.group(1)
                self.components.append({
                    "name": component_name,
                    "file_path": rel_path
                })

                if self.neo4j_store:
                    self.neo4j_store.query(
                        "MERGE (c:Class {workspace: $workspace, name: $name}) SET c:UIComponent, c.file_path = $file_path",
                        {"name": component_name, "file_path": rel_path, "workspace": str(self.workspace_path)}
                    )

                # Extract constructor injections
                injections = self._extract_constructor_injections(content)

                for inj_type, _inj_name in injections:
                    if inj_type in known_services or inj_type.endswith("Service"):
                        if self.neo4j_store:
                            self.neo4j_store.query(
                                "MERGE (s:Class {workspace: $workspace, name: $svc}) SET s:AngularService",
                                {"svc": inj_type, "workspace": str(self.workspace_path)}
                            )
                            self.neo4j_store.query("""
                                MATCH (c {workspace: $workspace, name: $comp})
                                WHERE c:UIComponent OR c:Class
                                MATCH (s {workspace: $workspace, name: $svc})
                                WHERE s:AngularService OR s:Class
                                MERGE (c)-[:USES]->(s)
                            """, {"comp": component_name, "svc": inj_type, "workspace": str(self.workspace_path)})

                    elif inj_type in known_stores or inj_type.endswith("Store") or inj_type.endswith("State"):
                        if self.neo4j_store:
                            self.neo4j_store.query(
                                "MERGE (s:Class {workspace: $workspace, name: $store}) SET s:UIStore",
                                {"store": inj_type, "workspace": str(self.workspace_path)}
                            )
                            self.neo4j_store.query("""
                                MATCH (c {workspace: $workspace, name: $comp})
                                WHERE c:UIComponent OR c:Class
                                MATCH (s {workspace: $workspace, name: $store})
                                WHERE s:UIStore OR s:Class
                                MERGE (c)-[:USES]->(s)
                            """, {"comp": component_name, "store": inj_type, "workspace": str(self.workspace_path)})

            except Exception as e:
                logger.debug(f"  Failed to process component {cf.name}: {e}")

        logger.info(f"  🧩 Processed {len(self.components)} components")

    # =========================================================================
    # PHASE 3: STORE → SERVICE MAPPING
    # =========================================================================

    def _discover_store_dependencies(self):
        """
        For each Store/State class, find injected services and create CALLS edges.
        """
        logger.info("  🏪 Phase 3: Discovering Store → Service Dependencies...")

        store_files = self._find_files("*.state.ts")
        store_files.extend(self._find_files("*.store.ts"))
        store_files = list(set(store_files))
        known_services = self._get_known_class_names("Service")
        count = 0

        for sf in store_files:
            try:
                content = sf.read_text(encoding="utf-8", errors="ignore")

                # Extract store/state class name
                class_match = re.search(
                    r"export\s+class\s+(\w+(?:Store|State))", content
                )
                if not class_match:
                    continue
                store_name = class_match.group(1)
                rel_path = str(sf.relative_to(self.workspace_path)).replace("\\", "/")
                self.stores.append({"name": store_name, "file_path": rel_path, "workspace": str(self.workspace_path)})

                if self.neo4j_store:
                    self.neo4j_store.query(
                        "MERGE (s:Class {workspace: $workspace, name: $name}) SET s:UIStore, s.file_path = $file_path",
                        {"name": store_name, "file_path": rel_path, "workspace": str(self.workspace_path)}
                    )

                # Extract constructor injections
                injections = self._extract_constructor_injections(content)

                for inj_type, _inj_name in injections:
                    if inj_type in known_services or inj_type.endswith("Service"):
                        if self.neo4j_store:
                            self.neo4j_store.query(
                                "MERGE (s:Class {workspace: $workspace, name: $svc}) SET s:AngularService",
                                {"svc": inj_type, "workspace": str(self.workspace_path)}
                            )
                            self.neo4j_store.query("""
                                MATCH (st {workspace: $workspace, name: $store})
                                WHERE st:UIStore OR st:Class
                                MATCH (svc {workspace: $workspace, name: $svc})
                                WHERE svc:AngularService OR svc:Class
                                MERGE (st)-[:CALLS]->(svc)
                            """, {"store": store_name, "svc": inj_type, "workspace": str(self.workspace_path)})
                        count += 1

            except Exception as e:
                logger.debug(f"  Failed to process store {sf.name}: {e}")

        logger.info(f"  🏪 Created {count} Store→Service edges")

    # =========================================================================
    # PHASE 4: SERVICE → API ENDPOINT MAPPING
    # =========================================================================

    def _discover_service_api_calls(self):
        """
        For each Angular service, extract HTTP/GraphQL/Saga API calls
        and create CALLS_API edges to ApiEndpoint nodes.
        """
        logger.info("  🌐 Phase 4: Discovering Service → API Calls...")

        service_files = self._find_files("*.service.ts")
        count = 0

        # Patterns for API call extraction
        http_call_re = re.compile(
            r"this\.http\.(get|post|put|delete|patch)\s*[<(]\s*"
            r"(?:[^,)]+,\s*)?[`'\"]([^`'\"]+)[`'\"]",
            re.IGNORECASE
        )
        graphql_re = re.compile(
            r"this\.gqlService\s*\.\s*graphql\s*\(\s*"
            r"[^,]+,\s*`[^`]*?(query|mutation)\s+(\w+)",
            re.DOTALL
        )
        saga_re = re.compile(
            r"this\.saga(?:s)?Service\.doCommand(?:Sync)?\s*\(\s*['\"](\w+)['\"]",
        )
        apiurl_re = re.compile(
            r"this\.apiUrl\.getSourceURL\(\s*URIKeyConstant\.(\w+)\s*\)",
        )

        for sf in service_files:
            try:
                content = sf.read_text(encoding="utf-8", errors="ignore")
                rel_path = str(sf.relative_to(self.workspace_path)).replace("\\", "/")

                # Extract service class name
                class_match = re.search(
                    r"export\s+class\s+(\w+Service)", content
                )
                if not class_match:
                    continue
                service_name = class_match.group(1)
                self.services.append({"name": service_name, "file_path": rel_path, "workspace": str(self.workspace_path)})

                if self.neo4j_store:
                    self.neo4j_store.query(
                        "MERGE (s:Class {workspace: $workspace, name: $name}) SET s:AngularService, s.file_path = $file_path",
                        {"name": service_name, "file_path": rel_path, "workspace": str(self.workspace_path)}
                    )

                # Determine backend target from URIKeyConstant
                backend_targets = set()
                for m in apiurl_re.finditer(content):
                    backend_targets.add(m.group(1).lower())  # e.g. "AREA" → "area"

                # Extract HTTP calls
                for m in http_call_re.finditer(content):
                    method, url = m.groups()
                    endpoint_name = f"{method.upper()} {url}"
                    self._create_api_edge(service_name, endpoint_name, "http", backend_targets)
                    count += 1

                # Extract GraphQL calls
                for m in graphql_re.finditer(content):
                    op_type, op_name = m.groups()
                    endpoint_name = f"graphql:{op_type}:{op_name}"
                    self._create_api_edge(service_name, endpoint_name, "graphql", backend_targets)
                    count += 1

                # Extract Saga commands
                for m in saga_re.finditer(content):
                    command_name = m.group(1)
                    endpoint_name = f"saga:{command_name}"
                    self._create_api_edge(service_name, endpoint_name, "saga", backend_targets)
                    count += 1

            except Exception as e:
                logger.debug(f"  Failed to process service {sf.name}: {e}")

        logger.info(f"  🌐 Created {count} Service→API edges")

    def _create_api_edge(self, service_name: str, endpoint_name: str,
                         api_type: str, backend_targets: Set[str]):
        """Create ApiEndpoint node and CALLS_API edge."""
        # Determine project from backend targets
        project = ""
        if backend_targets:
            target = next(iter(backend_targets))
            project_map = {
                "area": "area-service",
                "project": "project-service",
                "issues": "issues-service",
                "load": "ene-load-service",
                "transmittals": "area-service",
                "library": "area-service",
                "deliverables": "area-service"
            }
            project = project_map.get(target, f"{target}-service")

        if self.neo4j_store:
            self.neo4j_store.query(
                "MERGE (a:ApiEndpoint {workspace: $workspace, name: $name}) "
                "SET a.type = $type, a.project = $project",
                {"name": endpoint_name, "type": api_type, "project": project, "workspace": str(self.workspace_path)}
            )
            self.neo4j_store.query("""
                MATCH (svc {workspace: $workspace, name: $svc})
                WHERE svc:AngularService OR svc:Class
                MATCH (a:ApiEndpoint {workspace: $workspace, name: $endpoint})
                MERGE (svc)-[:CALLS_API]->(a)
            """, {"svc": service_name, "endpoint": endpoint_name, "workspace": str(self.workspace_path)})

        self.api_endpoints.append({
            "name": endpoint_name, "type": api_type,
            "project": project, "service": service_name
        })

    # =========================================================================
    # PHASE 5: API → SPRING CONTROLLER MAPPING
    # =========================================================================

    def _discover_api_controller_mapping(self):
        """
        Map API endpoints to Spring controllers using api_catalog.json.
        """
        logger.info("  🎮 Phase 5: Mapping API → Controllers...")
        count = 0

        # Build controller index from api_catalog
        controller_index: Dict[str, Set[str]] = {}  # controller_class → set of projects
        for entry in self._api_catalog:
            ctrl = entry.get("class", "")
            project = entry.get("project", "")
            if ctrl:
                ctrl_name = ctrl.replace(".java", "")
                if ctrl_name not in controller_index:
                    controller_index[ctrl_name] = set()
                controller_index[ctrl_name].add(project)
                self.controllers.append({
                    "name": ctrl_name,
                    "project": project,
                    "method": entry.get("method", ""),
                    "route": entry.get("route", "")
                })

        # Inject GraphQL controllers to ensure fallback matches work
        controller_index["Query"] = {"area-service"}
        controller_index["Mutation"] = {"area-service"}

        # Ensure controller nodes exist in Neo4j with Controller label
        seen_controllers: Set[str] = set()
        for ctrl_name, projects in controller_index.items():
            if ctrl_name not in seen_controllers:
                if self.neo4j_store:
                    self.neo4j_store.query(
                        "MATCH (c:Class {workspace: $workspace, name: $name}) SET c:Controller",
                        {"name": ctrl_name, "workspace": str(self.workspace_path)}
                    )
                    # Also create if not exists
                    # We just use the first project for the node's main project property
                    self.neo4j_store.query(
                        "MERGE (c:Controller {workspace: $workspace, name: $name}) SET c.project = $project",
                        {"name": ctrl_name, "project": next(iter(projects)) if projects else "", "workspace": str(self.workspace_path)}
                    )
                seen_controllers.add(ctrl_name)

        # Fetch newly discovered API endpoints from Phase 4
        if self.neo4j_store:
            res = self.neo4j_store.query("MATCH (a:ApiEndpoint {workspace: $workspace}) RETURN a.name as name, a.type as type", {"workspace": str(self.workspace_path)})
            existing_names = {e["name"] for e in self.api_endpoints}
            for r in res:
                if r["name"] not in existing_names:
                    self.api_endpoints.append({
                        "name": r["name"],
                        "type": r.get("type", "http"),
                        "project": ""
                    })

        logger.info(f"Phase 5 running on {len(self.api_endpoints)} endpoints")

        # Link ApiEndpoints to Controllers
        for endpoint in self.api_endpoints:
            ep_project = endpoint.get("project", "")
            
            # If project is known, filter. Otherwise search all controllers.
            if ep_project:
                matching_ctrls = [
                    name for name, projs in controller_index.items()
                    if ep_project in projs
                ]
            else:
                matching_ctrls = list(controller_index.keys())

            ep_name_lower = endpoint["name"].lower()
            if "transmittal" in ep_name_lower:
                logger.info(f"Checking {endpoint['name']}, project='{ep_project}', matching_ctrls size={len(matching_ctrls)}, contains TransmittalController? {'TransmittalController' in matching_ctrls}")

            # Refine match: use endpoint name keywords
            for ctrl_name in matching_ctrls:
                # Extract domain keyword from controller name
                # e.g. "LibraryController" → "library"
                domain = ctrl_name.replace("Controller", "").lower()
                if domain and domain in ep_name_lower:
                    if "transmittal" in ep_name_lower:
                        logger.info(f"Phase 5 MATCH: {endpoint['name']} -> {ctrl_name}")
                    if self.neo4j_store:
                        self.neo4j_store.query("""
                            MATCH (a:ApiEndpoint {workspace: $workspace, name: $endpoint})
                            MATCH (c:Controller {workspace: $workspace, name: $ctrl})
                            MERGE (a)-[:HANDLED_BY]->(c)
                        """, {"endpoint": endpoint["name"], "ctrl": ctrl_name, "workspace": str(self.workspace_path)})
                    count += 1

            # Fallback: if endpoint is graphql/saga type containing a domain keyword
            if count == 0 and endpoint["type"] in ("graphql", "saga"):
                if "transmittal" in ep_name_lower:
                    logger.info(f"Phase 5 FALLBACK for {endpoint['name']}")
                for ctrl_name in matching_ctrls:
                    domain = ctrl_name.replace("Controller", "").lower()
                    # Check if any word in the endpoint name matches the domain
                    ep_words = re.findall(r"[a-zA-Z]+", endpoint["name"])
                    for word in ep_words:
                        if domain and word.lower().startswith(domain[:4]):
                            if "transmittal" in ep_name_lower:
                                logger.info(f"Phase 5 FALLBACK MATCH: {endpoint['name']} -> {ctrl_name}")
                            if self.neo4j_store:
                                self.neo4j_store.query("""
                                    MATCH (a:ApiEndpoint {workspace: $workspace, name: $endpoint})
                                    MATCH (c:Controller {workspace: $workspace, name: $ctrl})
                                    MERGE (a)-[:HANDLED_BY]->(c)
                                """, {"endpoint": endpoint["name"], "ctrl": ctrl_name, "workspace": str(self.workspace_path)})
                            count += 1
                            break

        # =========================================================================
        # PHASE 5.1: GRAPHQL & SAGA HANDLER DISCOVERY
        # =========================================================================
        logger.info("  🎮 Phase 5.1: Mapping GraphQL & Saga → Handlers...")
        # Add GraphQL Controllers
        gql_handlers = self._find_java_files("Query.java") + self._find_java_files("Mutation.java")
        for f in gql_handlers:
            content = f.read_text(encoding="utf-8", errors="ignore")
            ctrl_name = f.stem
            for m in re.finditer(r'@(?:Query|Mutation)Mapping\s*\(\s*["\'](\w+)["\']\s*\)', content):
                op = m.group(1)
                ep_name = f"graphql:query:{op}" if ctrl_name == "Query" else f"graphql:mutation:{op}"
                if self.neo4j_store:
                    self.neo4j_store.query("""
                        MERGE (c:Controller {workspace: $workspace, name: $ctrl})
                        WITH c
                        MATCH (a:ApiEndpoint {workspace: $workspace, name: $endpoint})
                        MERGE (a)-[:HANDLED_BY]->(c)
                    """, {"ctrl": ctrl_name, "endpoint": ep_name, "workspace": str(self.workspace_path)})
                count += 1

        # Add Saga Handlers
        saga_handlers = self._find_java_files("*Handler.java")
        for f in saga_handlers:
            content = f.read_text(encoding="utf-8", errors="ignore")
            ctrl_name = f.stem
            # Match class Name, e.g. EditCustomValueHandler -> editCustomValue
            # Match @Component or Saga step handler comment to verify it's a Saga handler
            if "Handler" in ctrl_name:
                base_name = ctrl_name.replace("Handler", "")
                if base_name:
                    base_name = base_name[0].lower() + base_name[1:]
                    ep_name = f"saga:{base_name}"
                    
                    if "updateCustomValue" in content or "editCustomValue" in content or "customValue" in content.lower():
                        # hard fallback for known aliases (e.g. editCustomValue handles updateCustomValue)
                        ep_name_alt = "saga:updateCustomValue"
                        if self.neo4j_store:
                            self.neo4j_store.query("""
                                MERGE (c:Controller {workspace: $workspace, name: $ctrl})
                                WITH c
                                MATCH (a:ApiEndpoint {workspace: $workspace, name: $endpoint})
                                MERGE (a)-[:HANDLED_BY]->(c)
                            """, {"ctrl": ctrl_name, "endpoint": ep_name_alt, "workspace": str(self.workspace_path)})

                    if self.neo4j_store:
                        self.neo4j_store.query("""
                            MERGE (c:Controller {workspace: $workspace, name: $ctrl})
                            WITH c
                            MATCH (a:ApiEndpoint {workspace: $workspace, name: $endpoint})
                            MERGE (a)-[:HANDLED_BY]->(c)
                        """, {"ctrl": ctrl_name, "endpoint": ep_name, "workspace": str(self.workspace_path)})
                    count += 1

        logger.info(f"  🎮 Created {count} API→Controller edges")

    # =========================================================================
    # PHASE 6: CONTROLLER → BACKEND SERVICE MAPPING
    # =========================================================================

    def _discover_controller_dependencies(self):
        """
        Scan Spring Controllers, Query.java, Mutation.java, and *Handler.java for injected BackendServices.
        """
        logger.info("  ⚙️ Phase 6: Discovering Controller → BackendService Dependencies...")
        count = 0

        controller_files = self._find_java_files("*Controller.java")
        controller_files.extend(self._find_java_files("Query.java"))
        controller_files.extend(self._find_java_files("Mutation.java"))
        controller_files.extend(self._find_java_files("*Handler.java"))

        # Pattern for Java @Autowired and constructor injection
        autowired_re = re.compile(
            r"(?:@Autowired\s+)?(?:private|protected)\s+"
            r"(?:final\s+)?(?:[\w\.]+\.)?(\w+Service\w*)\s+\w+",
            re.MULTILINE
        )

        for cf in controller_files:
            try:
                content = cf.read_text(encoding="utf-8", errors="ignore")
                ctrl_name = cf.stem  # e.g. "LibraryController"

                for m in autowired_re.finditer(content):
                    backend_svc = m.group(1)
                    self.backend_services.append({
                        "name": backend_svc,
                        "controller": ctrl_name
                    })

                    if self.neo4j_store:
                        # Ensure BackendService node
                        self.neo4j_store.query(
                            "MERGE (bs:BackendService {workspace: $workspace, name: $name})",
                            {"name": backend_svc, "workspace": str(self.workspace_path)}
                        )
                        # Also label existing Class nodes
                        self.neo4j_store.query(
                            "MATCH (c:Class {workspace: $workspace, name: $name}) SET c:BackendService",
                            {"name": backend_svc, "workspace": str(self.workspace_path)}
                        )
                        self.neo4j_store.query("""
                            MATCH (c:Controller {workspace: $workspace, name: $ctrl})
                            MATCH (bs:BackendService {workspace: $workspace, name: $svc})
                            MERGE (c)-[:USES]->(bs)
                        """, {"ctrl": ctrl_name, "svc": backend_svc, "workspace": str(self.workspace_path)})
                    count += 1

            except Exception as e:
                logger.debug(f"  Failed to process controller {cf.name}: {e}")

        logger.info(f"  ⚙️ Created {count} Controller→BackendService edges")

    # =========================================================================
    # PHASE 7: BACKEND SERVICE → ENTITY MAPPING
    # =========================================================================

    def _discover_entity_mapping(self):
        """
        Map backend services to JPA entities using domain_catalog.json
        and source file scanning for Repository/Entity references.
        """
        logger.info("  💾 Phase 7: Discovering BackendService → Entity Dependencies...")
        count = 0

        # Build entity set from domain_catalog
        entity_set: Dict[str, str] = {}  # entity_name → project
        for entry in self._domain_catalog:
            entity_name = entry.get("domain_entity", "")
            project = entry.get("project", "")
            if entity_name:
                entity_set[entity_name] = project
                if self.neo4j_store:
                    # Ensure Entity node in Neo4j
                    self.neo4j_store.query(
                        "MERGE (e:Entity {workspace: $workspace, name: $name}) SET e.project = $project",
                        {"name": entity_name, "project": project, "workspace": str(self.workspace_path)}
                    )
                    self.neo4j_store.query(
                        "MATCH (c:Class {workspace: $workspace, name: $name}) SET c:Entity",
                        {"name": entity_name, "workspace": str(self.workspace_path)}
                    )

        self.entities = [{"name": n, "project": p, "workspace": str(self.workspace_path)}
                         for n, p in entity_set.items()]

        # Scan backend service files for entity references
        service_impl_files = self._find_java_files("*ServiceImpl.java")
        service_impl_files.extend(self._find_java_files("*Service.java"))
        service_impl_files = list(set(service_impl_files))

        # Pattern for entity/repository references
        entity_ref_re = re.compile(
            r"(?:private|protected)\s+(?:final\s+)?(\w+Repository)\s+\w+",
        )
        import_entity_re = re.compile(
            r"import\s+[\w.]+\.entity\.(\w+)\s*;",
        )

        for sf in service_impl_files:
            try:
                content = sf.read_text(encoding="utf-8", errors="ignore")
                svc_name = sf.stem  # e.g. "LibraryServiceImpl"

                # Check if this is actually a backend service
                is_backend_svc = any(
                    bs["name"] in svc_name or svc_name in bs["name"]
                    for bs in self.backend_services
                )
                if not is_backend_svc and not svc_name.endswith("ServiceImpl"):
                    continue

                # Find entity references via imports
                for m in import_entity_re.finditer(content):
                    entity_name = m.group(1)
                    if entity_name in entity_set:
                        # Find the correct BackendService node name
                        # "LibraryServiceImpl" → match "LibraryService"
                        base_svc = svc_name.replace("Impl", "")
                        if self.neo4j_store:
                            self.neo4j_store.query("""
                                MATCH (bs:BackendService {workspace: $workspace})
                                WHERE bs.name = $svc OR bs.name = $base
                                MATCH (e:Entity {workspace: $workspace, name: $entity})
                                MERGE (bs)-[:USES]->(e)
                            """, {"svc": svc_name, "base": base_svc,
                                  "entity": entity_name, "workspace": str(self.workspace_path)})
                        count += 1

            except Exception as e:
                logger.debug(f"  Failed to process service file {sf.name}: {e}")

        logger.info(f"  💾 Created {count} BackendService→Entity edges")

    # =========================================================================
    # PHASE 8: WORKFLOW ASSEMBLY
    # =========================================================================

    def _assemble_workflows(self):
        """
        Query Neo4j for complete execution paths and group them into
        named workflows.
        """
        logger.info("  🔗 Phase 8: Assembling Workflows...")

        # Query for all workflow paths
        results = self.neo4j_store.query("""
            MATCH (r:Route {workspace: $workspace})-[:RENDERS]->(comp {workspace: $workspace})
            WHERE comp:UIComponent OR comp:Class
            OPTIONAL MATCH (comp {workspace: $workspace})-[:USES]->(svc_or_store {workspace: $workspace})
            OPTIONAL MATCH (svc_or_store {workspace: $workspace})-[:CALLS]->(downstream_svc {workspace: $workspace})
            OPTIONAL MATCH (svc_or_store {workspace: $workspace})-[:CALLS_API]->(api:ApiEndpoint {workspace: $workspace})
            OPTIONAL MATCH (downstream_svc {workspace: $workspace})-[:CALLS_API]->(api2:ApiEndpoint {workspace: $workspace})
            OPTIONAL MATCH (api {workspace: $workspace})-[:HANDLED_BY]->(ctrl:Controller {workspace: $workspace})
            OPTIONAL MATCH (api2 {workspace: $workspace})-[:HANDLED_BY]->(ctrl2:Controller {workspace: $workspace})
            OPTIONAL MATCH (ctrl {workspace: $workspace})-[:USES]->(bs:BackendService {workspace: $workspace})
            OPTIONAL MATCH (ctrl2 {workspace: $workspace})-[:USES]->(bs2:BackendService {workspace: $workspace})
            OPTIONAL MATCH (bs {workspace: $workspace})-[:USES]->(entity:Entity {workspace: $workspace})
            OPTIONAL MATCH (bs2 {workspace: $workspace})-[:USES]->(entity2:Entity {workspace: $workspace})
            RETURN r.path as route,
                   comp.name as component,
                   COLLECT(DISTINCT svc_or_store.name) as services_stores,
                   COLLECT(DISTINCT downstream_svc.name) as downstream_services,
                   COLLECT(DISTINCT COALESCE(api.name, api2.name)) as api_endpoints,
                   COLLECT(DISTINCT COALESCE(ctrl.name, ctrl2.name)) as controllers,
                   COLLECT(DISTINCT COALESCE(bs.name, bs2.name)) as backend_services,
                   COLLECT(DISTINCT COALESCE(entity.name, entity2.name)) as entities
        """)

        # Group by workflow domain
        workflow_map: Dict[str, dict] = {}

        for row in results:
            route = row.get("route", "")
            component = row.get("component", "")
            if not route or not component:
                continue

            # Determine workflow name from route path
            workflow_name = self._infer_workflow_name(route, component)
            if not workflow_name:
                continue

            if workflow_name not in workflow_map:
                workflow_map[workflow_name] = {
                    "routes": [],
                    "components": [],
                    "stores": [],
                    "services": [],
                    "api_endpoints": [],
                    "controllers": [],
                    "backend_services": [],
                    "entities": []
                }

            wf = workflow_map[workflow_name]
            if route not in wf["routes"]:
                wf["routes"].append(route)
            if component not in wf["components"]:
                wf["components"].append(component)

            # Collect all connected elements (filter out None/empty)
            for item in row.get("services_stores", []):
                if item and item not in wf["services"] and item not in wf["stores"]:
                    if "Store" in item or "State" in item:
                        wf["stores"].append(item)
                    else:
                        wf["services"].append(item)

            for item in row.get("downstream_services", []):
                if item and item not in wf["services"]:
                    wf["services"].append(item)

            for item in row.get("api_endpoints", []):
                if item and item not in wf["api_endpoints"]:
                    wf["api_endpoints"].append(item)

            for item in row.get("controllers", []):
                if item and item not in wf["controllers"]:
                    wf["controllers"].append(item)

            for item in row.get("backend_services", []):
                if item and item not in wf["backend_services"]:
                    wf["backend_services"].append(item)

            for item in row.get("entities", []):
                if item and item not in wf["entities"]:
                    wf["entities"].append(item)

        self.workflows = workflow_map
        logger.info(f"  🔗 Assembled {len(self.workflows)} workflows")

    def _infer_workflow_name(self, route: str, component: str) -> Optional[str]:
        """Infer workflow name from route path and component name."""
        # Extract meaningful segments from route
        path_lower = route.lower()
        comp_lower = component.lower()

        # Known CC4E domains
        domains = [
            "library", "deliverable", "transmittal", "register",
            "project", "contract", "member", "task", "report",
            "dashboard", "issue", "viewer"
        ]

        for domain in domains:
            if domain in path_lower or domain in comp_lower:
                return f"{domain}_workflow"

        # Fallback: use the last meaningful path segment
        segments = [s for s in route.split("/") if s and not s.startswith(":")]
        if segments:
            return f"{segments[-1]}_workflow"

        return None

    def _assemble_workflows_from_memory(self) -> Dict[str, dict]:
        """
        Assemble workflows from in-memory collections when Neo4j is unavailable.

        Strategy:
        - Seed workflow buckets from discovered routes (definitive Route→Component links)
        - Enrich each bucket by matching domain keyword against component/store/service/
          api/controller/backend-service/entity names and file paths
        """
        workflow_map: Dict[str, dict] = {}

        # Seed from routes — each route gives us a definitive Component link
        for route in self.routes:
            wf_name = self._infer_workflow_name(route["path"], route["component"])
            if not wf_name:
                continue
            if wf_name not in workflow_map:
                workflow_map[wf_name] = {
                    "routes": [], "components": [], "stores": [],
                    "services": [], "api_endpoints": [], "controllers": [],
                    "backend_services": [], "entities": [],
                    "source_files": [],
                }
            wf = workflow_map[wf_name]
            if route["path"] not in wf["routes"]:
                wf["routes"].append(route["path"])
            if route["component"] not in wf["components"]:
                wf["components"].append(route["component"])
            if route.get("source_file") and route["source_file"] not in wf["source_files"]:
                wf["source_files"].append(route["source_file"])

        # Also seed from domains not covered by routes
        domains = [
            "library", "deliverable", "transmittal", "register",
            "project", "contract", "member", "task", "report",
            "dashboard", "issue", "viewer",
        ]
        for domain in domains:
            wf_name = f"{domain}_workflow"
            if wf_name not in workflow_map:
                workflow_map[wf_name] = {
                    "routes": [], "components": [], "stores": [],
                    "services": [], "api_endpoints": [], "controllers": [],
                    "backend_services": [], "entities": [],
                    "source_files": [],
                }

        # Enrich each workflow bucket using domain keyword matching
        for wf_name, wf in workflow_map.items():
            domain = wf_name.replace("_workflow", "").lower()

            for c in self.components:
                if domain in c["name"].lower() or domain in c["file_path"].lower():
                    if c["name"] not in wf["components"]:
                        wf["components"].append(c["name"])
                    if c["file_path"] not in wf["source_files"]:
                        wf["source_files"].append(c["file_path"])

            for s in self.stores:
                if domain in s["name"].lower() or domain in s["file_path"].lower():
                    if s["name"] not in wf["stores"]:
                        wf["stores"].append(s["name"])
                    if s["file_path"] not in wf["source_files"]:
                        wf["source_files"].append(s["file_path"])

            for s in self.services:
                if domain in s["name"].lower() or domain in s["file_path"].lower():
                    if s["name"] not in wf["services"]:
                        wf["services"].append(s["name"])
                    if s["file_path"] not in wf["source_files"]:
                        wf["source_files"].append(s["file_path"])

            for a in self.api_endpoints:
                svc_domain = a.get("service", "").lower()
                if domain in a["name"].lower() or domain in svc_domain:
                    if a["name"] not in wf["api_endpoints"]:
                        wf["api_endpoints"].append(a["name"])

            for c in self.controllers:
                if domain in c["name"].lower() or domain in c.get("route", "").lower():
                    if c["name"] not in wf["controllers"]:
                        wf["controllers"].append(c["name"])

            for b in self.backend_services:
                if domain in b["name"].lower():
                    if b["name"] not in wf["backend_services"]:
                        wf["backend_services"].append(b["name"])

            for e in self.entities:
                if domain in e["name"].lower():
                    if e["name"] not in wf["entities"]:
                        wf["entities"].append(e["name"])

        # Drop empty buckets (no components AND no routes means the domain wasn't found)
        return {
            k: v for k, v in workflow_map.items()
            if v["routes"] or v["components"]
        }

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _extract_constructor_injections(self, content: str) -> List[Tuple[str, str]]:
        """
        Extract TypeScript constructor parameter injections.

        Returns list of (type_name, param_name) tuples.
        Handles both traditional and standalone Angular patterns,
        including @Inject, readonly, and generic types.
        """
        ctor_match = re.search(
            r"constructor\s*\((.*?)\)\s*\{",
            content,
            re.DOTALL
        )
        if not ctor_match:
            return []

        params_str = ctor_match.group(1)
        params = []
        depth = 0
        current = []
        for char in params_str:
            if char == '<' or char == '(':
                depth += 1
            elif char == '>' or char == ')':
                depth -= 1
            elif char == ',' and depth == 0:
                params.append(''.join(current).strip())
                current = []
                continue
            current.append(char)
        if current:
            params.append(''.join(current).strip())

        results = []
        for p in params:
            if not p: continue
            parts = p.split(':', 1)
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                # Remove default value if any
                type_part = right.split('=')[0].strip()
                # Remove generic portion
                type_base = type_part.split('<')[0].strip()
                
                # Extract the parameter name
                # left side could be "@Inject(TOKEN) private readonly param_name"
                left_words = re.findall(r'[\w\$]+', left)
                if left_words:
                    param_name = left_words[-1]
                    results.append((type_base, param_name))
        return results

    def _get_known_class_names(self, suffix: str) -> Set[str]:
        """
        Get known class names from SQLite AST index by suffix.
        Falls back to filesystem scan if SQLite unavailable.
        """
        names: Set[str] = set()

        if self.sqlite_store:
            try:
                if hasattr(self.sqlite_store, "conn"):
                    cursor = self.sqlite_store.conn.cursor()
                elif hasattr(self.sqlite_store, "connection"):
                    cursor = self.sqlite_store.connection.cursor()
                else:
                    return names

                cursor.execute(
                    "SELECT DISTINCT name FROM symbols "
                    "WHERE name LIKE ? AND kind = 'class'",
                    (f"%{suffix}",)
                )
                for row in cursor.fetchall():
                    names.add(row[0])

            except Exception as e:
                logger.debug(f"SQLite query for {suffix} failed: {e}")

        # Supplement from existing catalogs if available
        if suffix == "Service":
            for entry in self._service_catalog:
                svc = entry.get("service_class", "").replace(".java", "")
                if svc:
                    names.add(svc)

        return names

    def _find_files(self, pattern: str) -> List[Path]:
        """Find files matching glob pattern, skipping excluded directories."""
        results = []
        for root, dirs, files in os.walk(self.workspace_path):
            dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS]
            for fname in files:
                full = Path(root) / fname
                if full.match(pattern):
                    results.append(full)
        return results

    def _find_java_files(self, pattern: str) -> List[Path]:
        """Find Java files matching glob pattern across all backend projects."""
        results = []
        # Search in workspace and known CC4E backend project directories
        search_roots = [self.workspace_path]

        # Also search CC4E backend projects if they exist at the same level
        workspace_parent = self.workspace_path.parent
        for project in ["area-service", "issues-service", "project-service",
                        "ene-load-service", "sagas-service", "bim-gateway",
                        "se-connector-apis"]:
            proj_path = workspace_parent / project
            if proj_path.exists():
                search_roots.append(proj_path)

        for search_root in search_roots:
            for root, dirs, files in os.walk(search_root):
                dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS]
                for fname in files:
                    full = Path(root) / fname
                    if full.match(pattern):
                        results.append(full)
        return results

    def _save_workflow_catalog(self):
        """Save discovered workflows to workflow_catalog.json."""
        catalog_path = self.catalog_dir / "workflow_catalog.json"
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(catalog_path, "w", encoding="utf-8") as f:
                json.dump(self.workflows, f, indent=2, default=str)
            logger.info(f"  📄 Saved workflow catalog to {catalog_path}")
        except Exception as e:
            logger.error(f"  ❌ Failed to save workflow catalog: {e}")

    # =========================================================================
    # QUERY API (consumed by LocalizationAgent)
    # =========================================================================

    def get_workflow_files(self, keyword: str) -> List[str]:
        """
        Query Neo4j for all files in workflows matching a keyword.

        Used by LocalizationAgent to promote workflow-related files
        when a ticket mentions a specific domain (e.g. "Library").

        Args:
            keyword: Domain keyword to match (e.g. "library", "transmittal")

        Returns:
            List of file paths belonging to the matching workflow chain.
        """
        if not self.neo4j_store:
            return self._get_workflow_files_from_memory(keyword)

        try:
            # Stage 1: direct Route → Component file paths (keyword on route path or component name)
            r1 = self.neo4j_store.query("""
                MATCH (r:Route {workspace: $workspace})-[:RENDERS]->(c {workspace: $workspace})
                WHERE c.file_path IS NOT NULL
                  AND (toLower(r.path) CONTAINS toLower($keyword)
                       OR toLower(c.name) CONTAINS toLower($keyword))
                RETURN DISTINCT c.file_path AS file_path
            """, {"keyword": keyword, "workspace": str(self.workspace_path)})

            # Stage 2: services/stores linked FROM keyword-matched components
            r2 = self.neo4j_store.query("""
                MATCH (c {workspace: $workspace})-[:USES|CALLS]->(s {workspace: $workspace})
                WHERE (c:UIComponent OR c:UIStore)
                  AND s.file_path IS NOT NULL
                  AND toLower(c.name) CONTAINS toLower($keyword)
                RETURN DISTINCT s.file_path AS file_path
            """, {"keyword": keyword, "workspace": str(self.workspace_path)})

            # Stage 3: components with keyword directly in their file_path
            r3 = self.neo4j_store.query("""
                MATCH (c {workspace: $workspace})
                WHERE c.file_path IS NOT NULL
                  AND toLower(c.file_path) CONTAINS toLower($keyword)
                RETURN DISTINCT c.file_path AS file_path
            """, {"keyword": keyword, "workspace": str(self.workspace_path)})

            seen: Set[str] = set()
            paths: List[str] = []
            for row in list(r1) + list(r2) + list(r3):
                fp = row.get("file_path")
                if fp and fp not in seen:
                    seen.add(fp)
                    paths.append(fp)

            return paths

        except Exception as e:
            logger.warning(f"Workflow file query failed: {e}")
            return self._get_workflow_files_from_memory(keyword)

    def _get_workflow_files_from_memory(self, keyword: str) -> List[str]:
        """
        Return source file paths for workflows matching *keyword* using the
        in-memory catalog.  If ``self.workflows`` has not been assembled yet
        this method tries to load the on-disk catalog first.
        """
        # Try loading catalog from disk if in-memory copy is empty
        if not self.workflows:
            catalog_path = self.catalog_dir / "workflow_catalog.json"
            if catalog_path.exists():
                try:
                    with open(catalog_path, "r", encoding="utf-8") as f:
                        self.workflows = json.load(f)
                except Exception:
                    pass

        kw_lower = keyword.lower()
        file_paths: Set[str] = set()

        for wf_name, wf_data in self.workflows.items():
            # Match by workflow name
            matched = kw_lower in wf_name
            if not matched:
                # Match by component / service / store names
                all_names = (
                    wf_data.get("routes", [])
                    + wf_data.get("components", [])
                    + wf_data.get("services", [])
                    + wf_data.get("stores", [])
                )
                matched = any(kw_lower in n.lower() for n in all_names)

            if not matched:
                continue

            # Collect source_files stored directly in the catalog
            for fp in wf_data.get("source_files", []):
                if fp:
                    file_paths.add(fp)

            # Also resolve paths from in-memory element lists
            domain = wf_name.replace("_workflow", "").lower()
            for c in self.components:
                if domain in c["name"].lower() or kw_lower in c["name"].lower():
                    file_paths.add(c["file_path"])
            for s in self.services:
                if domain in s["name"].lower() or kw_lower in s["name"].lower():
                    file_paths.add(s["file_path"])
            for s in self.stores:
                if domain in s["name"].lower() or kw_lower in s["name"].lower():
                    file_paths.add(s["file_path"])

        return [p for p in file_paths if p]

    # =========================================================================
    # PHASE 9: SEMANTIC KNOWLEDGE CATALOG INGESTION
    # =========================================================================
    def _ingest_semantic_catalog(self):
        """
        Loads the LLM-generated semantic_catalog.json and injects it into Neo4j
        as SemanticConcept nodes linked to workflow artifacts.
        """
        logger.info("  🧠 Phase 9: Ingesting Semantic Knowledge Catalog...")
        if not self.neo4j_store:
            return

        catalog_path = self.catalog_dir / "semantic_catalog.json"
        if not catalog_path.exists():
            logger.warning("  ⚠️ semantic_catalog.json not found. Run semantic_catalog_builder.py first.")
            return

        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                catalog = json.load(f)

            concepts = catalog.get("concepts", [])
            for concept in concepts:
                name = concept.get("name")
                aliases = concept.get("aliases", [])
                ui_labels = concept.get("ui_labels", [])
                translations = concept.get("translations", [])
                related_workflow = concept.get("related_workflow")
                artifacts = concept.get("artifacts", [])

                # 1. Merge the SemanticConcept node
                query_concept = """
                MERGE (s:SemanticConcept {workspace: $workspace, name: $name})
                SET s.aliases = $aliases,
                    s.ui_labels = $ui_labels,
                    s.translations = $translations,
                    s.related_workflow = $related_workflow
                """
                self.neo4j_store.query(query_concept, {
                    "name": name,
                    "aliases": aliases,
                    "ui_labels": ui_labels,
                    "translations": translations,
                    "related_workflow": related_workflow
                })

                # 2. Create RELATES_TO edges to artifacts
                for artifact in artifacts:
                    a_name = artifact.get("name")
                    a_type = artifact.get("type", "File")
                    confidence = float(artifact.get("confidence", 1.0))
                    source = artifact.get("source", "workflow_catalog")
                    
                    # We link based on the artifact name regardless of specific node type (Component, Service, etc)
                    # by matching any node with that name.
                    query_edge = """
                    MATCH (s:SemanticConcept {workspace: $workspace, name: $name})
                    MATCH (n {workspace: $workspace}) WHERE n.name = $a_name
                    MERGE (s)-[r:RELATES_TO]->(n)
                    SET r.confidence = $confidence, r.source = $source
                    """
                    self.neo4j_store.query(query_edge, {
                        "name": name,
                        "a_name": a_name,
                        "confidence": confidence,
                        "source": source
                    })

            logger.info(f"  ✅ Ingested {len(concepts)} Semantic Concepts into Neo4j")
        except Exception as e:
            logger.warning(f"  ⚠️ Failed to ingest semantic catalog: {e}")
