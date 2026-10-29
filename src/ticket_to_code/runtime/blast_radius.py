"""
Cross-Service Blast-Radius & Contract Regression Analyzer.

Statically analyzes the downstream blast radius of proposed file mutations across
microservice boundaries, API Gateways, and Frontend UI consumers before compilation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set


@dataclass
class BlastRadiusReport:
    modified_services: List[str] = field(default_factory=list)
    affected_gateways: List[str] = field(default_factory=list)
    affected_frontend_callers: List[str] = field(default_factory=list)
    affected_downstream_services: List[str] = field(default_factory=list)
    risk_level: str = "low"  # low, medium, high, critical
    risk_reasons: List[str] = field(default_factory=list)
    suggested_verification_targets: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"### 🛡️ Blast-Radius Analysis (Risk: {self.risk_level.upper()})"]
        if self.modified_services:
            lines.append(f"- **Modified Services**: {', '.join(self.modified_services)}")
        if self.affected_gateways:
            lines.append(f"- **Affected Gateway Routes**: {', '.join(self.affected_gateways)}")
        if self.affected_frontend_callers:
            lines.append(f"- **Affected UI Callers**: {', '.join(self.affected_frontend_callers)}")
        if self.affected_downstream_services:
            lines.append(f"- **Affected Microservices**: {', '.join(self.affected_downstream_services)}")
        if self.risk_reasons:
            lines.append("- **Identified Risk Factors**:")
            for r in self.risk_reasons:
                lines.append(f"  • {r}")
        if self.suggested_verification_targets:
            lines.append(f"- **Suggested Verification Scope**: {', '.join(self.suggested_verification_targets)}")
        return "\n".join(lines)


class BlastRadiusAnalyzer:
    """Analyzes the cross-service impact of file modifications."""

    # Java Spring
    SPRING_ENDPOINT_RE = re.compile(
        r'@(?:Get|Post|Put|Delete|Patch|Request)Mapping\s*\(\s*(?:value\s*=\s*)?["\']([^"\']+)["\']',
        re.IGNORECASE
    )
    # Python FastAPI / Flask
    PY_ENDPOINT_RE = re.compile(
        r'@(?:app|router|api)\.(?:get|post|put|delete|patch|route)\s*\(\s*["\']([^"\']+)["\']',
        re.IGNORECASE
    )
    # Go Gin / Echo / Chi / net/http
    GO_ENDPOINT_RE = re.compile(
        r'\.(?:GET|POST|PUT|DELETE|PATCH|Handle|HandleFunc)\s*\(\s*["\']([^"\']+)["\']',
        re.IGNORECASE
    )
    # Node Express / NestJS
    NODE_ENDPOINT_RE = re.compile(
        r'(?:\.(?:get|post|put|delete|patch)|@(?:Get|Post|Put|Delete|Patch))\s*\(\s*["\']([^"\']+)["\']',
        re.IGNORECASE
    )

    GATEWAY_PATH_RE = re.compile(r'Path\s*=\s*([^\n\r]+)', re.IGNORECASE)

    # Angular HttpClient, Axios, Fetch, apiClient
    HTTP_CALL_RE = re.compile(
        r'(?:this\.http|axios|apiClient)\.(?:get|post|put|delete)\s*(?:<[^>]+>)?\s*\(\s*[`"\']([^`"\']+)[`"\']|fetch\s*\(\s*[`"\']([^`"\']+)[`"\']',
        re.IGNORECASE
    )

    @classmethod
    def analyze(cls, workspace_root: Path | str, changed_files: List[str]) -> BlastRadiusReport:
        ws = Path(workspace_root).resolve()
        report = BlastRadiusReport()

        modified_endpoints: Set[str] = set()
        modified_services: Set[str] = set()

        for f_rel in changed_files:
            f_path = ws / f_rel.lstrip("/\\")
            if not f_path.exists() or not f_path.is_file():
                continue

            parts = Path(f_rel).parts
            if parts:
                modified_services.add(parts[0])

            try:
                content = f_path.read_text(encoding="utf-8", errors="ignore")
                # Look for REST endpoints across Java, Python, Go, Node
                for regex in (cls.SPRING_ENDPOINT_RE, cls.PY_ENDPOINT_RE, cls.GO_ENDPOINT_RE, cls.NODE_ENDPOINT_RE):
                    for m in regex.finditer(content):
                        ep = m.group(1).strip()
                        if ep:
                            modified_endpoints.add(ep)
            except Exception:
                pass

        report.modified_services = sorted(list(modified_services))

        if not modified_endpoints:
            # Low risk internal changes
            report.risk_level = "low"
            report.suggested_verification_targets = report.modified_services
            return report

        # Search for consumers of these endpoints across Gateway and UI
        risk_reasons = []
        affected_gw = set()
        affected_ui = set()

        # Gateway patterns (YAML, conf)
        gw_patterns = ("**/application*.yml", "**/application*.yaml", "**/gateway*.yml", "**/gateway*.yaml", "**/routes*.yml")
        # Frontend UI patterns (Angular, React, Vue, TS/JS API clients)
        ui_patterns = ("**/*.service.ts", "**/*api*.ts", "**/*api*.js", "**/*client*.ts", "**/*client*.js")

        for ep in modified_endpoints:
            clean_token = ep.split("{")[0].strip("/")
            if not clean_token or len(clean_token) < 4:
                continue

            # Search gateway configs
            for pat in gw_patterns:
                for yml_file in ws.glob(pat):
                    try:
                        txt = yml_file.read_text(encoding="utf-8", errors="ignore")
                        if clean_token in txt:
                            affected_gw.add(str(yml_file.relative_to(ws)).replace("\\", "/"))
                    except Exception:
                        pass

            # Search UI frontend clients & services
            for pat in ui_patterns:
                for client_file in ws.glob(pat):
                    try:
                        txt = client_file.read_text(encoding="utf-8", errors="ignore")
                        if clean_token in txt:
                            affected_ui.add(str(client_file.relative_to(ws)).replace("\\", "/"))
                    except Exception:
                        pass

        report.affected_gateways = sorted(list(affected_gw))
        report.affected_frontend_callers = sorted(list(affected_ui))

        if modified_endpoints:
            risk_reasons.append(f"Modified REST endpoint contracts: {', '.join(sorted(list(modified_endpoints))[:4])}")

        if affected_gw and affected_ui:
            report.risk_level = "critical"
            risk_reasons.append("Cross-boundary impact: Modifies endpoints routed through Spring Cloud Gateway and consumed by Angular UI.")
        elif affected_gw or affected_ui:
            report.risk_level = "high"
            risk_reasons.append("External impact: Endpoints exposed to gateway routing or frontend HTTP services.")
        else:
            report.risk_level = "medium"

        report.risk_reasons = risk_reasons
        report.suggested_verification_targets = sorted(list(
            modified_services
            | {Path(g).parts[0] for g in affected_gw if Path(g).parts}
            | {Path(u).parts[0] for u in affected_ui if Path(u).parts}
        ))

        return report
