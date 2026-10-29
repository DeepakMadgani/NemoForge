"""Fullstack Two-Stage Pipeline (Java Spring Boot Backend + Angular Frontend).

Coordinates cross-repository, multi-service fullstack tickets:
1. Stage 1 (Backend First): Discovers the target Java microservice, updates DTO / Controller / Service,
   and verifies with Gradle compilation / test gate.
2. API Contract Bridge: Extracts the updated REST endpoint path and DTO fields from the Java AST.
3. Stage 2 (Frontend Second): Injects the generated API contract into the Angular context (xchange-ui),
   updates the service / component / template, and gates with `npm run build` and behavioral verification.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ticket_to_code.lean.lean_path import (
    Candidate, LeanResult, LockedContext, TokenMeter, run_lean_path,
)
from ticket_to_code.lean.locate import locate
from ticket_to_code.lean.scope import ScopeContract, ScopeFence, triage


@dataclass
class ApiContract:
    """Cross-boundary contract between Java backend and Angular frontend."""
    endpoint: str = ""
    http_method: str = "GET"
    dto_name: str = ""
    fields: Dict[str, str] = field(default_factory=dict)     # name -> typescript_type
    raw_snippet: str = ""


@dataclass
class FullstackResult:
    status: str                                             # verified | unverified | error
    backend_result: Optional[LeanResult] = None
    frontend_result: Optional[LeanResult] = None
    contract: Optional[ApiContract] = None
    elapsed_s: float = 0.0
    trace: List[str] = field(default_factory=list)


def extract_api_contract_from_java(root: Path, written_rel: List[str]) -> ApiContract:
    """Extracts REST endpoint and DTO structure from modified Java backend files."""
    contract = ApiContract()
    for rel in written_rel:
        p = root / rel
        if not p.is_file() or not rel.endswith(".java"):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        # Check for Spring REST annotations: @GetMapping, @PostMapping, @RequestMapping
        m_endpoint = re.search(r'@(?:Get|Post|Put|Delete|Patch)Mapping\s*\(\s*(?:value\s*=\s*)?["\']([^"\']+)["\']', text)
        if m_endpoint:
            contract.endpoint = m_endpoint.group(1)
            contract.http_method = "POST" if "@PostMapping" in text else "GET"

        # Check for DTO class definitions
        if "dto" in rel.lower() or p.stem.endswith(("Dto", "DTO", "Count", "Response", "Request")):
            contract.dto_name = p.stem
            # Extract fields
            for line in text.splitlines():
                fm = re.search(r'private\s+([\w<>]+)\s+(\w+)\s*;', line)
                if fm:
                    jtype, fname = fm.group(1), fm.group(2)
                    tstype = "number" if jtype in ("int", "Integer", "long", "Long", "double", "Double") else "boolean" if jtype in ("boolean", "Boolean") else "string"
                    contract.fields[fname] = tstype

    # Format TypeScript interface snippet for Angular
    if contract.fields:
        ts_lines = [f"  {k}: {v};" for k, v in contract.fields.items()]
        contract.raw_snippet = f"export interface {contract.dto_name or 'BackendPayload'} {{\n" + "\n".join(ts_lines) + "\n}"
    return contract


def find_service_root(workspace: Path, hint_name: str) -> Optional[Path]:
    """Finds a microservice subproject directory within a monorepo workspace."""
    # Check direct child
    cand = workspace / hint_name
    if cand.is_dir():
        return cand
    for p in workspace.iterdir():
        if p.is_dir() and hint_name.lower() in p.name.lower():
            return p
    return None


def run_fullstack_pipeline(ticket_id: str, title: str, description: str,
                           workspace: str,
                           backend_patch_fn, backend_verify_fn,
                           frontend_patch_fn, frontend_verify_fn,
                           backend_service_hint: str = "issues-service",
                           frontend_service_hint: str = "xchange-ui",
                           hard_timeout_s: float = 600.0) -> FullstackResult:
    """Executes a fullstack ticket across Java backend and Angular frontend."""
    t0 = time.time()
    trace = [f"fullstack pipeline started for {ticket_id}"]
    ws = Path(workspace)

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 1: Backend Java Microservice
    # ──────────────────────────────────────────────────────────────────────────
    backend_root = find_service_root(ws, backend_service_hint) or ws
    trace.append(f"stage 1 backend root: {backend_root.name}")

    backend_meter = TokenMeter()
    backend_res = run_lean_path(
        f"{ticket_id}-BE",
        f"[Backend] {title}",
        f"Backend Java requirements:\n{description}",
        str(backend_root),
        backend_patch_fn,
        backend_verify_fn,
        explicit_root=str(backend_root),
        labels=["backend", "java", "spring-boot"],
        hard_timeout_s=hard_timeout_s / 2,
        meter=backend_meter
    )

    if backend_res.status != "verified":
        trace.append(f"stage 1 backend failed: {backend_res.reason}")
        return FullstackResult(
            status="unverified",
            backend_result=backend_res,
            elapsed_s=round(time.time() - t0, 1),
            trace=trace
        )

    trace.append(f"stage 1 backend verified in {backend_res.attempts} attempts")

    # ──────────────────────────────────────────────────────────────────────────
    # API Contract Bridge: Extract Java DTO / Endpoint -> TypeScript types
    # ──────────────────────────────────────────────────────────────────────────
    contract = extract_api_contract_from_java(backend_root, backend_res.locked_files)
    trace.append(f"api contract bridge: endpoint='{contract.endpoint}', dto='{contract.dto_name}' with {len(contract.fields)} fields")

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 2: Frontend Angular Client
    # ──────────────────────────────────────────────────────────────────────────
    frontend_root = find_service_root(ws, frontend_service_hint) or ws
    trace.append(f"stage 2 frontend root: {frontend_root.name}")

    # Inject the backend API contract into the frontend requirements
    frontend_desc = (
        f"Frontend Angular requirements:\n{description}\n\n"
        f"// BACKEND REST API CONTRACT (STAGE 1 PRODUCED):\n"
        f"// Endpoint: {contract.endpoint}\n"
        f"{contract.raw_snippet}\n"
    )

    frontend_meter = TokenMeter()
    frontend_res = run_lean_path(
        f"{ticket_id}-FE",
        f"[Frontend] {title}",
        frontend_desc,
        str(frontend_root),
        frontend_patch_fn,
        frontend_verify_fn,
        explicit_root=str(frontend_root),
        labels=["ui", "frontend", "angular"],
        hard_timeout_s=hard_timeout_s / 2,
        meter=frontend_meter
    )

    if frontend_res.status != "verified":
        trace.append(f"stage 2 frontend failed: {frontend_res.reason}")
        return FullstackResult(
            status="unverified",
            backend_result=backend_res,
            frontend_result=frontend_res,
            contract=contract,
            elapsed_s=round(time.time() - t0, 1),
            trace=trace
        )

    trace.append(f"stage 2 frontend verified in {frontend_res.attempts} attempts")
    trace.append("fullstack pipeline verified successfully across both layers")

    return FullstackResult(
        status="verified",
        backend_result=backend_res,
        frontend_result=frontend_res,
        contract=contract,
        elapsed_s=round(time.time() - t0, 1),
        trace=trace
    )
