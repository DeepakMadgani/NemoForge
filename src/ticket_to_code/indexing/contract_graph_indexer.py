"""
Cross-Repository Semantic Graph & Communication Contract Indexer.

Indexes REST endpoints, Kafka/RabbitMQ topics, and DTO schemas across
distinct microservices and frontend clients in enterprise monorepos.
Constructs semantic edges:
  (UI Client) --[:CALLS_REST]--> (API Gateway) --[:ROUTES_TO]--> (Backend Service)
  (Backend Service) --[:PUBLISHES_EVENT]--> (Message Topic) --[:CONSUMED_BY]--> (Sagas / Worker)
Enables instantaneous cross-service localization for full-stack and distributed tickets.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ticket_to_code.contract_graph")


class ContractType(str, Enum):
    REST = "rest"
    KAFKA_TOPIC = "kafka_topic"
    RABBIT_QUEUE = "rabbit_queue"
    DTO_SCHEMA = "dto_schema"


class RoleType(str, Enum):
    PRODUCER = "producer"  # or client / publisher
    CONSUMER = "consumer"  # or server / subscriber


GENERIC_CRUD_VERBS: Set[str] = {
    "get", "set", "post", "put", "delete", "init", "run", "execute",
    "findall", "findbyid", "save", "update", "load", "handle", "validate",
    "render", "create", "remove", "fetch", "check", "clear", "reset",
    "index", "show", "edit", "new", "list", "search", "destroy", "apply",
    "process", "call", "start", "stop"
}



@dataclass
class ContractNode:
    service_name: str
    contract_type: ContractType
    role: RoleType
    identifier: str  # URL route path, topic name, or DTO name
    http_method: str = ""  # GET, POST, etc.
    file_path: str = ""
    line_number: int = 1
    code_snippet: str = ""


@dataclass
class ContractLink:
    producer: ContractNode
    consumer: ContractNode
    contract_type: ContractType
    identifier: str


class ContractGraphIndexer:
    """
    Builds and queries the communication contract graph across all services in the monorepo.
    """

    def __init__(self, workspace_path: str | Path) -> None:
        self.workspace = Path(workspace_path).resolve()
        self.nodes: List[ContractNode] = []
        self.links: List[ContractLink] = []
        self._indexed: bool = False

    def build_index(self) -> None:
        """Discovers and links all REST and messaging contracts in the workspace."""
        self.nodes.clear()
        self.links.clear()

        # 1. Scan REST Server Endpoints (Java Spring Boot, Go, Python)
        self._scan_backend_rest()

        # 2. Scan REST Client Callers (Angular / TypeScript HTTP calls)
        self._scan_frontend_rest()

        # 3. Scan Messaging Topics (Kafka / RabbitMQ)
        self._scan_messaging_topics()

        # 4. Link producers with consumers
        self._link_contracts()
        self._indexed = True
        logger.info(f"🕸️ [ContractGraphIndexer] Indexed {len(self.nodes)} endpoints; assembled {len(self.links)} cross-service links")

    def _git_roots(self) -> List[Path]:
        """Returns all git sub-repositories or the root workspace."""
        roots: List[Path] = []
        if (self.workspace / ".git").exists():
            roots.append(self.workspace)
        for child in self.workspace.iterdir():
            if child.is_dir() and (child / ".git").exists():
                roots.append(child)
        return roots if roots else [self.workspace]

    def _search_pattern(self, root: Path, pattern: str, extensions: Tuple[str, ...]) -> List[Tuple[str, int, str]]:
        """
        Executes git grep if git is present in root, otherwise uses fast python regex traversal.
        Returns list of (inner_path, line_no, snippet).
        """
        results: List[Tuple[str, int, str]] = []
        if (root / ".git").exists():
            try:
                cmd = ["git", "grep", "-n", "-I", "-P", pattern]
                if extensions:
                    cmd += ["--"] + [f"*{ext}" for ext in extensions]
                res = subprocess.run(
                    cmd,
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    timeout=5,
                    shell=False,
                )
                if res.returncode == 0:
                    for line in res.stdout.splitlines():
                        parts = line.split(":", 2)
                        if len(parts) >= 3:
                            line_no = int(parts[1]) if parts[1].isdigit() else 1
                            results.append((parts[0].replace("\\", "/"), line_no, parts[2].strip()))
                    return results
                elif res.returncode == 1:
                    # Clean git grep exit: 0 matches found in repository. Fast-path return without rglob!
                    return []
            except Exception:
                pass

        # Fallback: file tree search across target extensions
        try:
            rgx = re.compile(pattern)
            for ext in extensions:
                for p in root.rglob(f"*{ext}"):
                    if any(part in p.parts for part in ("node_modules", "target", "build", "dist", ".git", ".aviator", ".venv")):
                        continue
                    try:
                        rel = str(p.relative_to(root)).replace("\\", "/")
                        for idx, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                            if rgx.search(line):
                                results.append((rel, idx, line.strip()))
                    except Exception:
                        pass
        except Exception:
            pass
        return results

    def _scan_backend_rest(self) -> None:
        """Finds Spring Boot controllers, class-level prefixes, and route methods across subprojects."""
        for root in self._git_roots():
            hits = self._search_pattern(root, r'@(Get|Post|Put|Delete|Request)Mapping', (".java", ".kt", ".scala"))
            files_content_cache: Dict[str, List[str]] = {}
            files_prefix_cache: Dict[str, str] = {}

            for inner_path, line_no, snippet in hits:
                f_rel = f"{root.name}/{inner_path}" if root != self.workspace else inner_path
                svc = root.name if root != self.workspace else (f_rel.split("/")[0] if "/" in f_rel else "root")

                full_file_path = root / inner_path
                file_lines = files_content_cache.get(str(full_file_path))
                if file_lines is None:
                    if full_file_path.is_file():
                        try:
                            raw_txt = full_file_path.read_text(encoding="utf-8", errors="ignore")
                            file_lines = raw_txt.splitlines()
                            m_class_prefix = re.search(r'@(?:RestController|Controller)[^\{]*?@RequestMapping\s*\(\s*(?:value\s*=\s*)?["\']([^"\']+)["\']', raw_txt, re.DOTALL)
                            files_prefix_cache[str(full_file_path)] = m_class_prefix.group(1).strip() if m_class_prefix else ""
                        except Exception:
                            file_lines = []
                            files_prefix_cache[str(full_file_path)] = ""
                    else:
                        file_lines = []
                        files_prefix_cache[str(full_file_path)] = ""
                    files_content_cache[str(full_file_path)] = file_lines

                class_prefix = files_prefix_cache.get(str(full_file_path), "")

                m_type = re.search(r'@(Get|Post|Put|Delete|Request)Mapping', snippet)
                m_route = re.search(r'(?:value|path)?\s*=?\s*\{?\s*["\']([^"\']+)["\']', snippet)
                method_route = m_route.group(1).strip() if m_route else ""
                if not method_route:
                    m_const = re.search(r'@(?:Get|Post|Put|Delete|Request)Mapping\s*\(\s*(?:(?:value|path)\s*=\s*)?([A-Za-z0-9_.]+)\s*\)', snippet)
                    if m_const:
                        const_name = m_const.group(1).split(".")[-1]
                        raw_doc = "\n".join(file_lines)
                        m_val = re.search(rf'(?:String|final|const)\s+{re.escape(const_name)}\s*=\s*["\']([^"\']+)["\']', raw_doc)
                        if m_val:
                            method_route = m_val.group(1).strip()
                        else:
                            proj_hits = self._search_pattern(root, rf'(?:String|final|const)\s+{re.escape(const_name)}\s*=\s*["\']([^"\']+)["\']', (".java", ".ts"))
                            if proj_hits:
                                m_proj = re.search(r'["\']([^"\']+)["\']', proj_hits[0][2])
                                if m_proj:
                                    method_route = m_proj.group(1).strip()
                        if not method_route:
                            method_route = const_name.lower().replace("_", "/")

                # If this hit is on the class-level @RequestMapping itself, skip or normalize
                if method_route == class_prefix and any("class " in l for l in file_lines[line_no - 1 : line_no + 3]):
                    continue

                # Combine class prefix + method route
                if method_route.startswith(class_prefix):
                    route = method_route
                else:
                    joined_route = f"{class_prefix}/{method_route}".replace("//", "/").strip("/")
                    route = joined_route if joined_route else "endpoint"

                method = m_type.group(1).upper() if m_type else "GET"

                self.nodes.append(ContractNode(
                    service_name=svc,
                    contract_type=ContractType.REST,
                    role=RoleType.CONSUMER,  # Server listens/consumes request
                    identifier=self._normalize_route(route),
                    http_method=method if method != "REQUEST" else "ALL",
                    file_path=f_rel,
                    line_number=line_no,
                    code_snippet=snippet,
                ))




                # Extract method symbol name from current or subsequent lines
                window = "\n".join(file_lines[line_no - 1 : min(len(file_lines), line_no + 3)])
                m_method = re.search(r'(?:public|protected|private)\s+[\w\<\>\[\],\s]+\s+([a-zA-Z0-9_]+)\s*\(', window)
                if m_method:
                    fn_name = m_method.group(1)
                    if fn_name.lower() not in GENERIC_CRUD_VERBS and len(fn_name) >= 6:
                        self.nodes.append(ContractNode(
                            service_name=svc,
                            contract_type=ContractType.DTO_SCHEMA,
                            role=RoleType.CONSUMER,
                            identifier=fn_name,
                            file_path=f_rel,
                            line_number=line_no,
                            code_snippet=snippet,
                        ))

    def _scan_frontend_rest(self) -> None:
        """Finds this.http calls, axios, fetch, and client service invocations across all subprojects."""
        for root in self._git_roots():
            # 1. Standard HTTP methods (Angular HttpClient, Axios, custom api clients, fetch)
            hits = self._search_pattern(root, r'(?:\.|\b)(get|post|put|delete|patch|fetch)\s*(?:<[^>]+>)?\s*\(\s*[`"\']([^`"\']+)[\'"`]', (".ts", ".tsx", ".js", ".jsx"))
            for inner_path, line_no, snippet in hits:
                f_rel = f"{root.name}/{inner_path}" if root != self.workspace else inner_path
                m = re.search(r'(?:\.|\b)(get|post|put|delete|patch|fetch)\s*(?:<[^>]+>)?\s*\(\s*[`"\']([^`"\']+)[\'"`]', snippet, re.IGNORECASE)
                if m:
                    method = m.group(1).upper()
                    if method == "FETCH":
                        method = "GET"
                    raw_route = m.group(2).strip()
                    if not raw_route.endswith((".png", ".svg", ".jpg", ".css", ".scss", ".json", ".ico")) and ("/" in raw_route or raw_route.startswith("api")):
                        svc = root.name if root != self.workspace else (f_rel.split("/")[0] if "/" in f_rel else "root")
                        self.nodes.append(ContractNode(
                            service_name=svc,
                            contract_type=ContractType.REST,
                            role=RoleType.PRODUCER,  # Client produces request
                            identifier=self._normalize_route(raw_route),
                            http_method=method,
                            file_path=f_rel,
                            line_number=line_no,
                            code_snippet=snippet,
                        ))

            # 2. Frontend Service Method Invocations (e.g. this.memberService.deleteMember(id))
            hits_svc = self._search_pattern(root, r'(?:this\.[a-zA-Z0-9_]+Service|[a-zA-Z0-9_]+Client)\.([a-zA-Z0-9_]+)\s*\(', (".ts", ".tsx", ".js", ".jsx"))
            for inner_path, line_no, snippet in hits_svc:
                f_rel = f"{root.name}/{inner_path}" if root != self.workspace else inner_path
                m = re.search(r'(?:this\.[a-zA-Z0-9_]+Service|[a-zA-Z0-9_]+Client)\.([a-zA-Z0-9_]+)\s*\(', snippet)
                if m:
                    fn_name = m.group(1)
                    if fn_name.lower() not in GENERIC_CRUD_VERBS and len(fn_name) >= 6:
                        svc = root.name if root != self.workspace else (f_rel.split("/")[0] if "/" in f_rel else "root")
                        self.nodes.append(ContractNode(
                            service_name=svc,
                            contract_type=ContractType.DTO_SCHEMA,
                            role=RoleType.PRODUCER,
                            identifier=fn_name,
                            file_path=f_rel,
                            line_number=line_no,
                            code_snippet=snippet,
                        ))

            # 3. React Custom Hooks (export function use... or export const use...)
            hits_hooks = self._search_pattern(root, r'export\s+(?:function|const)\s+(use[A-Z]\w+)', (".ts", ".tsx", ".js", ".jsx"))
            for inner_path, line_no, snippet in hits_hooks:
                f_rel = f"{root.name}/{inner_path}" if root != self.workspace else inner_path
                m = re.search(r'export\s+(?:function|const)\s+(use[A-Z]\w+)', snippet)
                if m:
                    hook_name = m.group(1)
                    svc = root.name if root != self.workspace else (f_rel.split("/")[0] if "/" in f_rel else "root")
                    self.nodes.append(ContractNode(
                        service_name=svc,
                        contract_type=ContractType.DTO_SCHEMA,
                        role=RoleType.CONSUMER,
                        identifier=hook_name,
                        file_path=f_rel,
                        line_number=line_no,
                        code_snippet=snippet,
                    ))

    def _scan_messaging_topics(self) -> None:
        """Finds Kafka and RabbitMQ topics across all subprojects."""
        for root in self._git_roots():
            hits = self._search_pattern(root, r'@KafkaListener\s*\(\s*topics\s*=\s*["\']([^"\']+)["\']', (".java", ".go", ".py"))
            for inner_path, line_no, snippet in hits:
                f_rel = f"{root.name}/{inner_path}" if root != self.workspace else inner_path
                m = re.search(r'@KafkaListener\s*\(\s*topics\s*=\s*["\']([^"\']+)["\']', snippet)
                if m:
                    topic = m.group(1).strip()
                    svc = root.name if root != self.workspace else (f_rel.split("/")[0] if "/" in f_rel else "root")
                    self.nodes.append(ContractNode(
                        service_name=svc,
                        contract_type=ContractType.KAFKA_TOPIC,
                        role=RoleType.CONSUMER,
                        identifier=topic,
                        file_path=f_rel,
                        line_number=line_no,
                        code_snippet=snippet,
                    ))

    def _link_contracts(self) -> None:
        """Connects producers and consumers with matching identifiers (routes or symbol names)."""
        from collections import defaultdict

        producers = [n for n in self.nodes if n.role == RoleType.PRODUCER]
        consumers = [n for n in self.nodes if n.role == RoleType.CONSUMER]

        # Index consumers for fast O(1) candidate lookup
        dto_consumers = defaultdict(list)
        kafka_consumers = defaultdict(list)
        rest_consumers_by_last_seg = defaultdict(list)
        rest_all_consumers = []

        for c in consumers:
            if c.contract_type == ContractType.DTO_SCHEMA:
                dto_consumers[c.identifier.lower()].append(c)
            elif c.contract_type == ContractType.KAFKA_TOPIC:
                kafka_consumers[c.identifier].append(c)
            elif c.contract_type == ContractType.REST:
                rest_all_consumers.append(c)
                parts = [p for p in c.identifier.split("/") if p and p != "*"]
                if parts:
                    rest_consumers_by_last_seg[parts[-1]].append(c)

        for p in producers:
            if p.contract_type == ContractType.DTO_SCHEMA:
                p_id = p.identifier.lower()
                if p_id not in GENERIC_CRUD_VERBS and len(p_id) >= 6:
                    for c in dto_consumers.get(p_id, []):
                        self.links.append(ContractLink(
                            producer=p,
                            consumer=c,
                            contract_type=p.contract_type,
                            identifier=p.identifier,
                        ))
            elif p.contract_type == ContractType.KAFKA_TOPIC:
                for c in kafka_consumers.get(p.identifier, []):
                    self.links.append(ContractLink(
                        producer=p,
                        consumer=c,
                        contract_type=p.contract_type,
                        identifier=p.identifier,
                    ))
            elif p.contract_type == ContractType.REST:
                parts = [p_part for p_part in p.identifier.split("/") if p_part and p_part != "*"]
                candidates = rest_consumers_by_last_seg.get(parts[-1], []) if parts else rest_all_consumers
                for c in candidates:
                    if self._routes_match(p.identifier, c.identifier):
                        self.links.append(ContractLink(
                            producer=p,
                            consumer=c,
                            contract_type=p.contract_type,
                            identifier=p.identifier,
                        ))


    def find_linked_services(self, file_path_or_keyword: str) -> List[ContractLink]:
        """
        Given a file or keyword, returns all related cross-service links.
        Example: passing 'member.component.ts' returns the backend Java controller it talks to!
        """
        if not self._indexed:
            self.build_index()

        target = file_path_or_keyword.lower().replace("\\", "/")
        matching_links: List[ContractLink] = []

        for link in self.links:
            if (target in link.producer.file_path.lower() or
                target in link.consumer.file_path.lower() or
                target in link.identifier.lower()):
                matching_links.append(link)

        return matching_links

    @staticmethod
    def _normalize_route(route: str) -> str:
        # Strip URL parameters like ${id} or :id or {id}
        clean = re.sub(r"[\$\{][^\}]+[\}]", "*", route)
        clean = re.sub(r":\w+", "*", clean)
        return clean.strip("/")

    @staticmethod
    def _routes_match(route1: str, route2: str) -> bool:
        if not route1 or not route2:
            return False
        if route1 == route2:
            return True
        # Check subpath matching (e.g. "api/v1/projects" vs "/projects")
        parts1 = [p for p in route1.split("/") if p and p != "*"]
        parts2 = [p for p in route2.split("/") if p and p != "*"]
        if not parts1 or not parts2:
            return False
        return parts1[-1] == parts2[-1] and (len(parts1) == 1 or len(parts2) == 1 or parts1[-2] == parts2[-2])
