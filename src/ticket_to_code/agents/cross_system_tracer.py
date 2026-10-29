"""
Cross-System Architecture Tracer — Enhancement 2

Traces how systems interact across microservice boundaries:
- HTTP API calls between services
- Shared database tables
- Message queue producers/consumers
- Shared configuration keys

Safety: READ-ONLY. Only reads source files.

Author: Deepak Madgani
Date: July 2026
"""

import re
import logging
from pathlib import Path
from typing import List, Optional, Dict, Set
from collections import defaultdict

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# MODELS
# ============================================================================

class SystemLink(BaseModel):
    """A link between two systems/services."""
    source_system: str = Field(..., description="System making the call")
    target_system: str = Field(..., description="System being called")
    link_type: str = Field(..., description="http | database | queue | config | grpc")
    source_file: str = Field("")
    target_endpoint: str = Field("", description="URL, table name, queue name, etc.")
    line_number: int = Field(0)
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class CrossSystemAnalysis(BaseModel):
    """Complete cross-system analysis result."""
    has_cross_system_data: bool = Field(False)
    links: List[SystemLink] = Field(default_factory=list)
    systems_detected: List[str] = Field(default_factory=list)
    system_graph: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="system -> [systems it calls]"
    )
    shared_resources: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Resources (tables, queues) used by multiple systems"
    )


# ============================================================================
# DETECTION PATTERNS
# ============================================================================

# HTTP client calls: HttpClient, RestTemplate, fetch, axios, requests
_HTTP_CALL = re.compile(
    r'(?:HttpClient|RestTemplate|fetch|axios|requests)\s*[\.(]\s*'
    r'(?:get|post|put|delete|patch|GetAsync|PostAsync)\s*\(\s*'
    r'[`"\']([^`"\']+)[`"\']',
    re.IGNORECASE,
)

# URL string patterns (likely API calls)
_URL_PATTERN = re.compile(
    r'["\']https?://(?:localhost|\$\{?\w+\}?)[:\d]*(/[\w/\-{}\$]+)["\']',
)

# Database table references
_DB_TABLE = re.compile(
    r'(?:FROM|INTO|UPDATE|JOIN|TABLE)\s+[\["`]?(\w+)[\]"`]?',
    re.IGNORECASE,
)

# Message queue patterns
_QUEUE = re.compile(
    r'(?:publish|subscribe|send|receive|produce|consume)\s*\(\s*["\'](\w[\w.\-]*)["\']',
    re.IGNORECASE,
)

_SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", "dist",
    "build", "target", ".angular", "vendor", "bin", "obj",
}

_SCAN_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".cs", ".go",
    ".jsx", ".tsx",
}


class CrossSystemTracer:
    """
    Traces inter-system interactions across microservice boundaries.
    """

    def __init__(self, workspace_path: str):
        self.workspace = Path(workspace_path)

    def _detect_systems(self) -> Dict[str, Path]:
        """
        Detect distinct systems/services in the workspace.
        Looks for project markers (pom.xml, package.json, .csproj, etc.)
        """
        systems: Dict[str, Path] = {}

        markers = ["pom.xml", "package.json", "*.csproj", "pyproject.toml", "go.mod"]
        for marker in markers:
            for m in self.workspace.rglob(marker):
                if any(skip in m.parts for skip in _SKIP_DIRS):
                    continue
                system_name = m.parent.name
                if system_name != self.workspace.name:
                    systems[system_name] = m.parent

        # If no sub-projects found, treat workspace as single system
        if not systems:
            systems[self.workspace.name] = self.workspace

        return systems

    def _scan_for_links(
        self, system_name: str, system_path: Path
    ) -> List[SystemLink]:
        """Scan a system's source files for cross-system links."""
        links: List[SystemLink] = []

        for file_path in system_path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in _SCAN_EXTENSIONS:
                continue
            if any(skip in file_path.parts for skip in _SKIP_DIRS):
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            rel_path = str(file_path.relative_to(self.workspace))

            # HTTP calls
            for m in _HTTP_CALL.finditer(content):
                url = m.group(1)
                line_num = content[:m.start()].count("\n") + 1
                # Try to extract target system from URL
                target = self._infer_target_from_url(url)
                links.append(SystemLink(
                    source_system=system_name,
                    target_system=target or "external",
                    link_type="http",
                    source_file=rel_path,
                    target_endpoint=url,
                    line_number=line_num,
                    confidence=0.7 if target else 0.4,
                ))

            # URL patterns
            for m in _URL_PATTERN.finditer(content):
                endpoint = m.group(1)
                line_num = content[:m.start()].count("\n") + 1
                target = self._infer_target_from_url(endpoint)
                if target and target != system_name:
                    links.append(SystemLink(
                        source_system=system_name,
                        target_system=target,
                        link_type="http",
                        source_file=rel_path,
                        target_endpoint=endpoint,
                        line_number=line_num,
                        confidence=0.5,
                    ))

            # Database tables
            for m in _DB_TABLE.finditer(content):
                table = m.group(1)
                if table.lower() in ("select", "where", "set", "values", "as"):
                    continue
                line_num = content[:m.start()].count("\n") + 1
                links.append(SystemLink(
                    source_system=system_name,
                    target_system="database",
                    link_type="database",
                    source_file=rel_path,
                    target_endpoint=table,
                    line_number=line_num,
                    confidence=0.6,
                ))

            # Message queues
            for m in _QUEUE.finditer(content):
                queue = m.group(1)
                line_num = content[:m.start()].count("\n") + 1
                links.append(SystemLink(
                    source_system=system_name,
                    target_system="queue",
                    link_type="queue",
                    source_file=rel_path,
                    target_endpoint=queue,
                    line_number=line_num,
                    confidence=0.6,
                ))

        return links

    @staticmethod
    def _infer_target_from_url(url: str) -> Optional[str]:
        """Try to infer the target system from a URL path."""
        # Common patterns: /api/members, /service-name/endpoint
        parts = url.strip("/").split("/")
        if len(parts) >= 2:
            first = parts[0].lower()
            if first in ("api", "v1", "v2"):
                return parts[1] if len(parts) > 1 else None
            return first
        return None

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def trace(self) -> CrossSystemAnalysis:
        """
        Trace all cross-system interactions in the workspace.
        """
        systems = self._detect_systems()
        if not systems:
            return CrossSystemAnalysis(has_cross_system_data=False)

        all_links: List[SystemLink] = []
        for name, path in systems.items():
            links = self._scan_for_links(name, path)
            all_links.extend(links)

        if not all_links:
            return CrossSystemAnalysis(
                has_cross_system_data=False,
                systems_detected=list(systems.keys()),
            )

        # Build system graph
        graph: Dict[str, Set[str]] = defaultdict(set)
        for link in all_links:
            if link.target_system != link.source_system:
                graph[link.source_system].add(link.target_system)

        # Find shared resources
        resource_users: Dict[str, Set[str]] = defaultdict(set)
        for link in all_links:
            if link.link_type in ("database", "queue"):
                resource_users[link.target_endpoint].add(link.source_system)

        shared = [
            {"resource": name, "type": "shared", "systems": list(users)}
            for name, users in resource_users.items()
            if len(users) > 1
        ]

        return CrossSystemAnalysis(
            has_cross_system_data=True,
            links=all_links,
            systems_detected=list(systems.keys()),
            system_graph={k: list(v) for k, v in graph.items()},
            shared_resources=shared,
        )
