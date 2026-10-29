"""
Live Service Probing — Enhancement 10

HTTP/WebSocket probing to verify service endpoints are working:
- HTTP GET/POST with expected status codes
- WebSocket connection testing
- Endpoint discovery from code (parse route decorators)

Safety: Only performs read-only HTTP requests (GET by default).
Uses timeouts to prevent hanging.

Author: Deepak Madgani
Date: July 2026
"""

import re
import logging
import json
from pathlib import Path
from typing import List, Optional, Dict
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# MODELS
# ============================================================================

class EndpointProbe(BaseModel):
    """Result of probing a single endpoint."""
    url: str = Field(...)
    method: str = Field("GET")
    expected_status: int = Field(200)
    actual_status: Optional[int] = Field(None)
    response_time_ms: int = Field(0)
    success: bool = Field(False)
    error: Optional[str] = Field(None)
    response_snippet: Optional[str] = Field(None, description="First 200 chars of response")


class DiscoveredEndpoint(BaseModel):
    """An endpoint discovered from source code."""
    path: str = Field(...)
    method: str = Field("GET")
    source_file: str = Field("")
    line_number: int = Field(0)
    handler_name: str = Field("")


class ServiceProbeResult(BaseModel):
    """Complete service probing result."""
    has_probe_data: bool = Field(False)
    discovered_endpoints: List[DiscoveredEndpoint] = Field(default_factory=list)
    probe_results: List[EndpointProbe] = Field(default_factory=list)
    healthy_count: int = Field(0)
    unhealthy_count: int = Field(0)
    base_url: str = Field("")


# ============================================================================
# ROUTE DISCOVERY PATTERNS
# ============================================================================

# Python Flask/FastAPI: @app.route("/path"), @app.get("/path"), @router.post("/path")
_PY_ROUTE = re.compile(
    r'@(?:app|router|blueprint)\s*\.\s*(get|post|put|delete|patch|route)\s*\(\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)

# Node.js Express: app.get('/path'), router.post('/path')
_NODE_ROUTE = re.compile(
    r'(?:app|router)\s*\.\s*(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)

# Java Spring: @GetMapping("/path"), @PostMapping("/path"), @RequestMapping("/path")
_JAVA_ROUTE = re.compile(
    r'@(Get|Post|Put|Delete|Patch|Request)Mapping\s*\(\s*(?:value\s*=\s*)?["\']([^"\']+)["\']',
    re.IGNORECASE,
)

# C# ASP.NET: [HttpGet("path")], [Route("path")]
_DOTNET_ROUTE = re.compile(
    r'\[Http(Get|Post|Put|Delete|Patch)\s*\(\s*["\']?([^"\')\]]*)["\']?\s*\)',
    re.IGNORECASE,
)

_SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", "dist",
    "build", "target", ".angular", "vendor",
}

_SCAN_EXTENSIONS = {".py", ".js", ".ts", ".java", ".cs"}


class ServiceProber:
    """
    Discovers and probes service endpoints to verify they're working.
    """

    DEFAULT_TIMEOUT = 5  # seconds

    def __init__(self, workspace_path: str, base_url: str = "http://localhost:8000"):
        self.workspace = Path(workspace_path)
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # 1. Discover endpoints from code
    # ------------------------------------------------------------------

    def discover_endpoints(self, max_files: int = 100) -> List[DiscoveredEndpoint]:
        """
        Scan source code for route/endpoint declarations.
        """
        endpoints: List[DiscoveredEndpoint] = []
        scanned = 0

        for file_path in self.workspace.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in _SCAN_EXTENSIONS:
                continue
            if any(skip in file_path.parts for skip in _SKIP_DIRS):
                continue
            if scanned >= max_files:
                break

            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            scanned += 1
            rel_path = str(file_path.relative_to(self.workspace))
            ext = file_path.suffix.lower()

            patterns = []
            if ext == ".py":
                patterns = [_PY_ROUTE]
            elif ext in (".js", ".ts"):
                patterns = [_NODE_ROUTE]
            elif ext == ".java":
                patterns = [_JAVA_ROUTE]
            elif ext == ".cs":
                patterns = [_DOTNET_ROUTE]

            for pattern in patterns:
                for m in pattern.finditer(content):
                    method = m.group(1).upper()
                    if method == "REQUEST":
                        method = "GET"
                    path = m.group(2)
                    if not path.startswith("/"):
                        path = "/" + path

                    line_num = content[:m.start()].count("\n") + 1
                    endpoints.append(DiscoveredEndpoint(
                        path=path,
                        method=method,
                        source_file=rel_path,
                        line_number=line_num,
                    ))

        logger.info(f"ServiceProber: discovered {len(endpoints)} endpoints from {scanned} files")
        return endpoints

    # ------------------------------------------------------------------
    # 2. Probe endpoints
    # ------------------------------------------------------------------

    def probe_endpoint(
        self, path: str, method: str = "GET", expected_status: int = 200
    ) -> EndpointProbe:
        """Probe a single endpoint."""
        import time

        url = f"{self.base_url}{path}"
        start = time.time()

        try:
            req = Request(url, method=method)
            req.add_header("User-Agent", "T2C-ServiceProber/1.0")
            req.add_header("Accept", "application/json")

            with urlopen(req, timeout=self.DEFAULT_TIMEOUT) as resp:
                status = resp.status
                body = resp.read(500).decode("utf-8", errors="replace")
                elapsed = int((time.time() - start) * 1000)

                return EndpointProbe(
                    url=url, method=method,
                    expected_status=expected_status,
                    actual_status=status,
                    response_time_ms=elapsed,
                    success=(status == expected_status),
                    response_snippet=body[:200],
                )

        except HTTPError as e:
            elapsed = int((time.time() - start) * 1000)
            return EndpointProbe(
                url=url, method=method,
                expected_status=expected_status,
                actual_status=e.code,
                response_time_ms=elapsed,
                success=(e.code == expected_status),
                error=str(e),
            )

        except (URLError, TimeoutError, ConnectionError) as e:
            elapsed = int((time.time() - start) * 1000)
            return EndpointProbe(
                url=url, method=method,
                expected_status=expected_status,
                response_time_ms=elapsed,
                success=False,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # 3. Main probe
    # ------------------------------------------------------------------

    def probe_all(
        self,
        endpoints: Optional[List[DiscoveredEndpoint]] = None,
        max_probes: int = 20,
    ) -> ServiceProbeResult:
        """
        Discover and probe service endpoints.

        Args:
            endpoints: Pre-discovered endpoints (or auto-discover if None).
            max_probes: Maximum number of endpoints to probe.

        Returns:
            ServiceProbeResult with all findings.
        """
        if endpoints is None:
            endpoints = self.discover_endpoints()

        if not endpoints:
            return ServiceProbeResult(has_probe_data=False)

        # Only probe GET endpoints by default (safe)
        safe_endpoints = [e for e in endpoints if e.method == "GET"][:max_probes]

        results: List[EndpointProbe] = []
        for ep in safe_endpoints:
            result = self.probe_endpoint(ep.path, ep.method)
            results.append(result)

        healthy = sum(1 for r in results if r.success)
        unhealthy = sum(1 for r in results if not r.success)

        probe_result = ServiceProbeResult(
            has_probe_data=True,
            discovered_endpoints=endpoints,
            probe_results=results,
            healthy_count=healthy,
            unhealthy_count=unhealthy,
            base_url=self.base_url,
        )

        logger.info(
            f"ServiceProber: {healthy}/{len(results)} endpoints healthy, "
            f"{len(endpoints)} total discovered"
        )
        return probe_result
