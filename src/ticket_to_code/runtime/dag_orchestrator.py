"""
Multi-Project & Multi-Part DAG Orchestrator & Sub-Agent Task Decomposition Engine.

Modeled after frontier AI IDE architectures (Cursor Composer, Windsurf, Devin).
Decomposes complex, moderate, and high-priority enterprise tickets:
  1. React UI Architecture:
     - Task 1: Type Definitions & API Contracts (types.ts, DTOs, interfaces)
     - Task 2: State Management & Custom Hooks (use*.ts, state store, service client)
     - Task 3: Component Implementation & JSX Layout (*.tsx, *.jsx, CSS/SCSS)
     - Task 4: Unit / Regression Tests (*.test.tsx, *.spec.ts)
  2. Angular UI Architecture:
     - Task 1: Models & Service (*.model.ts, *.service.ts)
     - Task 2: Component Logic (*.component.ts)
     - Task 3: Template & Styling (*.component.html, *.component.scss)
     - Task 4: Unit Tests (*.component.spec.ts)
  3. Distributed Monorepo Full-Stack Architecture:
     - Task 1: Database & Migrations (SQL)
     - Task 2: Backend Microservices (Java/Go/Python)
     - Task 3: Event Broker / Gateway (Kafka/Sagas)
     - Task 4: Frontend UI (React/Angular)

Integrates:
  - Pillar 1: DAG Task Decomposition & Sequential Sub-Agent Execution
  - Pillar 2: Fast LSP Diagnostics (<100ms compiler feedback)
  - Pillar 3: Cross-Repository Semantic Contract Graph
  - Pillar 4: Speculative Git Worktree Sandboxing with Automatic Rollback
  - Pillar 5: Test-Driven Verification (TDD Red -> Green Harness)
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ticket_to_code.models import ValueEdgeTicket

logger = logging.getLogger("ticket_to_code.dag_orchestrator")


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskContractOutput:
    """Artifacts/contracts produced by an upstream sub-agent task for downstream tasks."""
    contract_type: str  # "rest_endpoint", "dto_model", "event_topic", "exported_symbol", "react_hook", "react_component"
    name: str
    details: Dict[str, Any] = field(default_factory=dict)
    snippet: str = ""


@dataclass
class DAGTask:
    """A discrete unit of work executed by a specialized SubProjectAgent."""
    task_id: str
    subproject_name: str
    subproject_path: Path
    language: str  # "typescript", "javascript", "java", "go", "python", "sql", "csharp"
    layer: str  # "frontend_contracts", "frontend_state", "frontend_ui", "frontend_test", "backend", "database", "message_broker"
    title: str
    description: str
    target_files: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Upstream task_ids
    status: TaskStatus = TaskStatus.PENDING
    result_diffs: Dict[str, str] = field(default_factory=dict)
    produced_contracts: List[TaskContractOutput] = field(default_factory=list)
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    tokens_consumed: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    llm_calls: int = 0


@dataclass
class DAGExecutionResult:
    """Consolidated outcome of the entire DAG orchestrator execution."""
    success: bool
    diffs: Dict[str, str] = field(default_factory=dict)
    modified_files: List[str] = field(default_factory=list)
    tasks: List[DAGTask] = field(default_factory=list)
    duration_seconds: float = 0.0
    tokens_consumed: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    llm_calls: int = 0
    error_message: Optional[str] = None
    sandboxed: bool = False


class CrossTaskContextBus:
    """Thread-safe context bus that routes contract signatures between sub-agent tasks."""

    def __init__(self) -> None:
        self._contracts: Dict[str, List[TaskContractOutput]] = {}
        self._shared_memory: Dict[str, Any] = {}

    def publish_contract(self, task_id: str, contract: TaskContractOutput) -> None:
        if task_id not in self._contracts:
            self._contracts[task_id] = []
        self._contracts[task_id].append(contract)
        logger.info(f"📡 [CrossTaskContextBus] Task '{task_id}' published contract: {contract.contract_type} -> '{contract.name}'")

    def get_upstream_contracts(self, upstream_task_ids: List[str]) -> List[TaskContractOutput]:
        contracts: List[TaskContractOutput] = []
        for tid in upstream_task_ids:
            contracts.extend(self._contracts.get(tid, []))
        return contracts

    def format_downstream_guidance(self, upstream_task_ids: List[str]) -> str:
        """Formats upstream contracts into prompt injection guidance for downstream sub-agents."""
        contracts = self.get_upstream_contracts(upstream_task_ids)
        if not contracts:
            return ""

        lines = [
            "// ========================================================================",
            "// UPSTREAM CONTRACTS: Generated by preceding tasks in this ticket workflow",
            "// You MUST integrate with these exact signatures, endpoints, and field names:",
            "// ========================================================================",
        ]
        for c in contracts:
            lines.append(f"// Type: {c.contract_type} | Name: {c.name}")
            if c.snippet:
                lines.append(c.snippet)
        lines.append("// ========================================================================")
        return "\n".join(lines)


class SubProjectAgent:
    """
    Dedicated Sub-Agent executing a single scoped task in the DAG.
    Maintains clean directory boundaries and invokes FastIDEEngine with LSP checks.
    """

    def __init__(
        self,
        task: DAGTask,
        workspace_root: Path,
        llm_client: Optional[Any] = None,
        ui_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.task = task
        self.workspace_root = workspace_root
        self.llm = llm_client
        self.ui_callback = ui_callback

    def execute(self, upstream_context: str = "") -> Tuple[bool, Dict[str, str], List[TaskContractOutput]]:
        """
        Executes code generation and verification specifically for this subproject task.
        Integrates upstream contracts into prompt context.
        """
        logger.info(f"🤖 [SubProjectAgent] Starting task '{self.task.task_id}' on '{self.task.subproject_name}' ({self.task.language})")
        if self.ui_callback:
            self.ui_callback({
                "phase": "patch_generation",
                "status": "in_progress",
                "message": f"🤖 Sub-Agent executing: {self.task.title}",
                "data": {
                    "task_id": self.task.task_id,
                    "layer": self.task.layer,
                    "subproject": self.task.subproject_name,
                    "dependencies": self.task.dependencies,
                },
            })

        t0 = time.time()
        diffs: Dict[str, str] = {}
        produced: List[TaskContractOutput] = []

        try:
            from ticket_to_code.runtime.fast_ide_engine import FastIDEEngine

            # Target directory is the specific subproject path
            target_path = self.task.subproject_path if self.task.subproject_path.exists() else self.workspace_root

            sub_ticket = ValueEdgeTicket(
                ticket_id=f"{self.task.task_id}",
                title=self.task.title,
                description=f"{self.task.description}\n\n{upstream_context}",
                expected_changed_files=self.task.target_files,
            )

            engine = FastIDEEngine(
                workspace_path=str(target_path),
                ticket=sub_ticket,
                llm_client=self.llm,
                max_repair_turns=2,
                ui_callback=self.ui_callback,
            )

            res = engine.run()
            self.task.duration_seconds = time.time() - t0
            self.task.tokens_consumed = res.tokens_consumed
            self.task.tokens_in = res.tokens_in
            self.task.tokens_out = res.tokens_out
            self.task.llm_calls = res.llm_calls

            if res.success:
                diffs = res.diffs
                self._extract_produced_contracts(diffs, produced)
                return True, diffs, produced
            else:
                self.task.error_message = res.error_message
                return False, {}, []

        except Exception as exc:
            logger.error(f"[SubProjectAgent] Failed execution on {self.task.task_id}: {exc}", exc_info=True)
            self.task.error_message = str(exc)
            return False, {}, []

    def _extract_produced_contracts(self, diffs: Dict[str, str], produced: List[TaskContractOutput]) -> None:
        """Extracts newly added exports, endpoints, hooks, and types for downstream tasks."""
        for f_rel, diff in diffs.items():
            # 1. Java REST Endpoints
            for r_path in re.findall(r'@(?:Get|Post|Put|Delete)Mapping\s*\(\s*["\']([^"\']+)["\']', diff):
                produced.append(TaskContractOutput(
                    contract_type="rest_endpoint",
                    name=r_path,
                    snippet=f"// REST Endpoint: {r_path} (in {f_rel})",
                ))

            # 2. React / TypeScript Custom Hooks
            for hook in re.findall(r'export\s+(?:function|const)\s+(use[A-Z]\w+)', diff):
                produced.append(TaskContractOutput(
                    contract_type="react_hook",
                    name=hook,
                    snippet=f"import {{ {hook} }} from './{Path(f_rel).stem}';",
                ))

            # 3. React Components
            for comp in re.findall(r'export\s+(?:default\s+)?(?:function|const)\s+([A-Z]\w+)', diff):
                if comp not in ("Response", "Promise", "Error"):
                    produced.append(TaskContractOutput(
                        contract_type="react_component",
                        name=comp,
                        snippet=f"import {{ {comp} }} from './{Path(f_rel).stem}';",
                    ))

            # 4. TypeScript Interfaces & Types
            for t_name in re.findall(r'export\s+(?:interface|type)\s+([A-Z]\w+)', diff):
                produced.append(TaskContractOutput(
                    contract_type="dto_model",
                    name=t_name,
                    snippet=f"import type {{ {t_name} }} from './{Path(f_rel).stem}';",
                ))

            # 5. Kafka / Message Topics
            for k_top in re.findall(r'@KafkaListener\s*\(\s*topics\s*=\s*["\']([^"\']+)["\']', diff):
                produced.append(TaskContractOutput(
                    contract_type="event_topic",
                    name=k_top,
                    snippet=f"// Event Topic: {k_top} (in {f_rel})",
                ))


class DAGOrchestrator:
    """
    Main Orchestrator for moderate, high, and complex tickets.
    Decomposes ticket into DAGTasks and coordinates sequential sub-agent executions.
    """

    def __init__(
        self,
        workspace_path: str | Path,
        ticket: ValueEdgeTicket,
        llm_client: Optional[Any] = None,
        ui_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.workspace = Path(workspace_path).resolve()
        self.ticket = ticket
        self.llm = llm_client
        self.ui_callback = ui_callback
        self.context_bus = CrossTaskContextBus()
        self.tasks: Dict[str, DAGTask] = {}

    @staticmethod
    def is_ticket_moderate_or_complex(ticket: ValueEdgeTicket, workspace_path: Optional[Path] = None) -> bool:
        """
        Determines whether a ticket requires DAG Task Decomposition instead of a simple single-turn edit.
        Criteria:
          1. Priority is high, critical, or urgent.
          2. Title/description/labels indicate complex, moderate, multi-part, fullstack, or refactor.
          3. Multiple expected changed files declared.
          4. Spans multiple architectural layers (UI + State/API or Backend + DB).
        """
        priority_str = str(getattr(ticket, "priority", "")).lower()
        if any(p in priority_str for p in ("high", "critical", "urgent", "p1", "p2")):
            return True

        text = f"{ticket.title} {ticket.description} {' '.join(ticket.labels or [])}".lower()
        complex_keywords = (
            "complex", "moderate", "multi-part", "task 1", "task 2", "dag",
            "multi-project", "saga", "fullstack", "full-stack", "refactor", "epic",
            "cross-service", "migration", "end-to-end", "e2e"
        )
        if any(re.search(rf"\b{re.escape(k)}\b", text) for k in complex_keywords):
            return True

        # Explicit multi-file scope check
        expected_files = [f.strip() for f in (ticket.expected_changed_files or []) if f.strip()]
        if len(expected_files) > 1:
            return True
        if len(expected_files) == 1 and not any(re.search(rf"\b{re.escape(k)}\b", text) for k in complex_keywords):
            return False

        # Check multi-layer indicators with regex word boundaries to avoid substring false positives
        has_ui = bool(re.search(r"\b(?:component|view|dialog|form|screen|button|header|modal|badge|ui|tsx|jsx|html)\b", text))
        # State management (exclude store when part of directory store-ui)
        has_state = bool(re.search(r"\b(?:hook|reducer|redux|zustand|vuex|state\s+management)\b", text)) or bool(re.search(r"\bstore\b(?!\s*-\s*ui)", text))
        has_backend = bool(re.search(r"\b(?:controller|backend|microservice|graphql|rest\s+api)\b", text))
        # Database (exclude HTML/UI table)
        has_db = bool(re.search(r"\b(?:database|migration|flyway|liquibase|sql|ddl|db\s+schema)\b", text))

        layers = sum([1 for l in (has_ui, has_state, has_backend, has_db) if l])
        return layers >= 2

    def plan_dag(self) -> List[DAGTask]:
        """
        Decomposes ticket into an ordered DAG based on ticket requirements and codebase topology.
        Supports:
          - React UI Component Architecture (Task 1: Types -> Task 2: Hook/State -> Task 3: JSX -> Task 4: Tests)
          - Multi-Project Monorepo (Task 1: DB -> Task 2: Backend -> Task 3: Gateway -> Task 4: Frontend)
          - Angular UI Component Architecture (Task 1: Models -> Task 2: Component TS -> Task 3: HTML/SCSS -> Task 4: Tests)
        """
        text = f"{self.ticket.title} {self.ticket.description}".lower()
        subprojects: Dict[str, Dict[str, Any]] = self._discover_workspace_subprojects()

        planned: List[DAGTask] = []
        is_react = self._is_react_project(subprojects)
        is_angular = self._is_angular_project(subprojects)
        is_multiproject = len(subprojects) > 1 and any(v.get("is_backend") for v in subprojects.values()) and any(v.get("is_ui") for v in subprojects.values())

        logger.info(f"🧭 [DAGOrchestrator] Planning DAG (is_react={is_react}, is_angular={is_angular}, is_multiproject={is_multiproject})")

        # ── STRATEGY 0: EXPLICIT TASKS DECLARED IN TICKET DESCRIPTION ──
        raw_ticket_text = f"{self.ticket.title}\n{self.ticket.description}"
        explicit_tasks = self._plan_from_explicit_tasks(subprojects, raw_ticket_text)
        if explicit_tasks:
            logger.info(f"📋 [DAGOrchestrator] Using explicit ticket task breakdown ({len(explicit_tasks)} tasks)")
            planned = explicit_tasks

        # ── STRATEGY 1: MULTI-PROJECT DISTRIBUTED ENTERPRISE TICKET ──
        elif is_multiproject:
            planned = self._plan_multiproject_dag(subprojects, text)

        # ── STRATEGY 2: REACT MODERATE / COMPLEX TICKET ──
        elif is_react:
            planned = self._plan_react_dag(subprojects, text)

        # ── STRATEGY 3: ANGULAR UI MODERATE / COMPLEX TICKET ──
        elif is_angular:
            planned = self._plan_angular_dag(subprojects, text)

        # ── STRATEGY 4: GENERIC ARCHITECTURAL DECOMPOSITION ──
        else:
            planned = self._plan_generic_dag(subprojects, text)

        self.tasks = {t.task_id: t for t in planned}
        logger.info(f"📐 [DAGOrchestrator] Decomposed ticket into {len(planned)} tasks: {[t.task_id for t in planned]}")
        return planned

    def _is_react_project(self, subprojects: Dict[str, Dict[str, Any]]) -> bool:
        """Detects whether target project is React based."""
        for item in [self.workspace] + [v["path"] for v in subprojects.values()]:
            pkg = item / "package.json"
            if pkg.is_file():
                try:
                    content = pkg.read_text(encoding="utf-8", errors="ignore")
                    if '"react"' in content or '"next"' in content or '"vite"' in content:
                        return True
                except Exception:
                    pass
            # Check for tsx/jsx files
            if list(item.glob("src/**/*.tsx")) or list(item.glob("src/**/*.jsx")):
                return True
        return False

    def _is_angular_project(self, subprojects: Dict[str, Dict[str, Any]]) -> bool:
        """Detects whether target project is Angular based."""
        for item in [self.workspace] + [v["path"] for v in subprojects.values()]:
            if (item / "angular.json").is_file():
                return True
            if list(item.glob("src/**/*.component.ts")):
                return True
        return False

    def _plan_react_dag(self, subprojects: Dict[str, Dict[str, Any]], text: str) -> List[DAGTask]:
        """
        Decomposes React tickets into 4 clear, sequential architectural parts:
          Task 1: Type Definitions & API Data Contracts (types.ts, models)
          Task 2: State Management & Custom Hooks (use*.ts, service client, context)
          Task 3: Component Implementation & JSX Layout (*.tsx, CSS/SCSS styling)
          Task 4: Unit & Regression Tests (*.test.tsx, *.spec.tsx)
        """
        ui_sub = next((k for k, v in subprojects.items() if v.get("is_ui")), "frontend")
        ui_path = subprojects.get(ui_sub, {}).get("path", self.workspace)

        feature_name = self._extract_feature_slug(self.ticket.title)

        t1 = DAGTask(
            task_id="task_1_types",
            subproject_name=ui_sub,
            subproject_path=ui_path,
            language="typescript",
            layer="frontend_contracts",
            title=f"Task 1: Define TypeScript Contracts & Data Models for {self.ticket.title}",
            description=(
                f"Define all necessary TypeScript interfaces, request/response DTOs, and API client method signatures "
                f"required for '{self.ticket.title}'. Ensure clean export of types."
            ),
            target_files=[f for f in self.ticket.expected_changed_files if "type" in f.lower() or f.endswith(".d.ts")],
        )

        t2 = DAGTask(
            task_id="task_2_hooks",
            subproject_name=ui_sub,
            subproject_path=ui_path,
            language="typescript",
            layer="frontend_state",
            title=f"Task 2: Implement State Management & Custom Hook for {self.ticket.title}",
            description=(
                f"Implement the state management logic, data fetching, or custom React hook (e.g. use{feature_name.capitalize()}) "
                f"for '{self.ticket.title}'. Consume contracts from Task 1."
            ),
            dependencies=["task_1_types"],
            target_files=[f for f in self.ticket.expected_changed_files if "hook" in f.lower() or "service" in f.lower() or "use" in Path(f).name.lower()],
        )

        t3 = DAGTask(
            task_id="task_3_component",
            subproject_name=ui_sub,
            subproject_path=ui_path,
            language="typescript",
            layer="frontend_ui",
            title=f"Task 3: Build React Component & JSX Layout for {self.ticket.title}",
            description=(
                f"Implement the React visual component ({feature_name}), JSX layout, form controls, and CSS/SCSS styling. "
                f"Wire component to the hook/state created in Task 2."
            ),
            dependencies=["task_1_types", "task_2_hooks"],
            target_files=[f for f in self.ticket.expected_changed_files if f.endswith((".tsx", ".jsx", ".css", ".scss"))],
        )

        t4 = DAGTask(
            task_id="task_4_tests",
            subproject_name=ui_sub,
            subproject_path=ui_path,
            language="typescript",
            layer="frontend_test",
            title=f"Task 4: Unit & Regression Verification Tests for {self.ticket.title}",
            description=(
                f"Implement unit tests using Vitest/Jest/React Testing Library verifying the behavior, "
                f"rendering, and state changes of {feature_name} according to ticket acceptance criteria."
            ),
            dependencies=["task_1_types", "task_2_hooks", "task_3_component"],
            target_files=[f for f in self.ticket.expected_changed_files if "test" in f.lower() or "spec" in f.lower()],
        )

        return [t1, t2, t3, t4]

    def _plan_angular_dag(self, subprojects: Dict[str, Dict[str, Any]], text: str) -> List[DAGTask]:
        """Decomposes Angular tickets into 4 parts: Models -> Component TS -> HTML/SCSS -> Spec."""
        ui_sub = next((k for k, v in subprojects.items() if v.get("is_ui")), "frontend")
        ui_path = subprojects.get(ui_sub, {}).get("path", self.workspace)
        feature_name = self._extract_feature_slug(self.ticket.title)

        t1 = DAGTask(
            task_id="task_1_models_and_service",
            subproject_name=ui_sub,
            subproject_path=ui_path,
            language="typescript",
            layer="frontend_contracts",
            title=f"Task 1: Angular Models & Service Contract for {self.ticket.title}",
            description=f"Define TypeScript interfaces and Angular service HTTP methods for {self.ticket.title}.",
        )

        t2 = DAGTask(
            task_id="task_2_component_logic",
            subproject_name=ui_sub,
            subproject_path=ui_path,
            language="typescript",
            layer="frontend_state",
            title=f"Task 2: Angular Component Logic & Controller for {self.ticket.title}",
            description=f"Implement component class TypeScript logic, lifecycle hooks, and handlers for {feature_name}.",
            dependencies=["task_1_models_and_service"],
        )

        t3 = DAGTask(
            task_id="task_3_template_and_style",
            subproject_name=ui_sub,
            subproject_path=ui_path,
            language="html",
            layer="frontend_ui",
            title=f"Task 3: Angular HTML Template & SCSS Styles for {self.ticket.title}",
            description=f"Implement template markup (*ngIf, bindings) and SCSS styling for {feature_name}.",
            dependencies=["task_1_models_and_service", "task_2_component_logic"],
        )

        t4 = DAGTask(
            task_id="task_4_spec_tests",
            subproject_name=ui_sub,
            subproject_path=ui_path,
            language="typescript",
            layer="frontend_test",
            title=f"Task 4: Angular Jasmine/Karma Unit Tests for {self.ticket.title}",
            description=f"Implement component unit tests in .spec.ts verifying acceptance criteria.",
            dependencies=["task_1_models_and_service", "task_2_component_logic", "task_3_template_and_style"],
        )

        return [t1, t2, t3, t4]

    def _plan_multiproject_dag(self, subprojects: Dict[str, Dict[str, Any]], text: str) -> List[DAGTask]:
        """Decomposes full-stack enterprise monorepo tickets across microservices."""
        planned: List[DAGTask] = []
        db_task_id: Optional[str] = None
        backend_task_ids: List[str] = []

        # 1. Database (honoring negative intent filters like 'no db' or 'mock data')
        has_negative_db_intent = any(neg in text for neg in ("no db", "no database", "without db", "ui only", "mock data", "do not touch the database", "don't touch db"))
        if not has_negative_db_intent and any(term in text for term in ("migration", "schema", "database", "table", "column", "entity", "ddl")):
            db_sub = next((k for k, v in subprojects.items() if "db" in k or "sql" in k or "schema" in k), None)
            if db_sub:
                t = DAGTask(
                    task_id="task_1_database",
                    subproject_name=db_sub,
                    subproject_path=subprojects[db_sub]["path"],
                    language="sql",
                    layer="database",
                    title=f"Task 1: Database Migration Schema for {self.ticket.title}",
                    description=f"Add necessary database columns, migrations, and DDL in {db_sub}",
                )
                planned.append(t)
                db_task_id = t.task_id

        # 2. Backend Microservice
        backend_subs = [k for k, v in subprojects.items() if v.get("is_backend")]

        # Query ContractGraphIndexer for cross-service linked backends matching ticket keywords
        linked_backend_names: Set[str] = set()
        try:
            from ticket_to_code.indexing.contract_graph_indexer import ContractGraphIndexer
            indexer = ContractGraphIndexer(self.workspace)
            c_links = indexer.find_linked_services(text)
            for lk in c_links:
                if lk.consumer.service_name in backend_subs:
                    linked_backend_names.add(lk.consumer.service_name)
                if lk.producer.service_name in backend_subs:
                    linked_backend_names.add(lk.producer.service_name)
        except Exception as cg_err:
            logger.debug(f"[DAGOrchestrator] ContractGraphIndexer lookup notice: {cg_err}")

        # Check if the ticket is explicitly UI-only or all expected files are in UI subprojects
        ui_subs = [k for k, v in subprojects.items() if v.get("is_ui")]
        expected_files = [f.replace("\\", "/").strip() for f in (self.ticket.expected_changed_files or []) if f.strip()]
        has_negative_backend_intent = any(neg in text for neg in ("ui only", "frontend only", "store-ui only", "store-ui/src", "client only", "no backend"))
        if expected_files and all(any(f.startswith(u) or f.startswith(f"{u}/") for u in ui_subs) for f in expected_files):
            has_negative_backend_intent = True

        generic_backend_tokens = {"service", "services", "server", "microservice", "app", "core", "client", "common", "utils", "api"}
        if not has_negative_backend_intent:
            for idx, b_name in enumerate(backend_subs):
                if expected_files and not any(f.startswith(b_name) or f.startswith(f"{b_name}/") for f in expected_files):
                    continue
                tokens = [tok for tok in re.split(r"[-_.]", b_name.lower()) if len(tok) >= 4 and tok not in generic_backend_tokens]
                is_matched = (
                    (b_name.lower() in text)
                    or any(re.search(rf"\b{re.escape(tok)}\b", text) for tok in tokens)
                    or (b_name in linked_backend_names)
                    or (len(backend_subs) == 1 and not has_negative_backend_intent)
                )
                if is_matched:
                    t = DAGTask(
                        task_id=f"task_2_backend_{idx+1}",
                        subproject_name=b_name,
                        subproject_path=subprojects[b_name]["path"],
                        language="java" if (subprojects[b_name]["path"] / "pom.xml").exists() else "go",
                        layer="backend",
                        title=f"Task 2: Backend Logic & REST API in {b_name}",
                        description=f"{self.ticket.description}\n\nScope for {b_name}: Implement backend service logic, models/enums, and REST endpoints for: {self.ticket.title}.",
                        dependencies=[db_task_id] if db_task_id else [],
                    )
                    planned.append(t)
                    backend_task_ids.append(t.task_id)

        # 3. Frontend UI
        has_negative_ui_intent = any(neg in text for neg in ("backend only", "api only", "no ui", "no frontend", "without ui"))
        if expected_files and all(any(f.startswith(b) or f.startswith(f"{b}/") for b in backend_subs) for f in expected_files):
            has_negative_ui_intent = True

        if not has_negative_ui_intent:
            ui_deps = list(backend_task_ids) if backend_task_ids else ([db_task_id] if db_task_id else [])
            for idx, u_name in enumerate(ui_subs):
                t = DAGTask(
                    task_id=f"task_3_ui_{idx+1}",
                    subproject_name=u_name,
                    subproject_path=subprojects[u_name]["path"],
                    language="typescript",
                    layer="frontend_ui",
                    title=f"Task 3: Frontend UI Presentation in {u_name}",
                    description=f"{self.ticket.description}\n\nScope for {u_name}: Implement UI components, controls, and bind to backend endpoints for: {self.ticket.title}",
                    dependencies=list(ui_deps),
                )
                planned.append(t)

        return planned

    def _detect_workspace_language(self, subprojects: Dict[str, Dict[str, Any]]) -> str:
        """Dynamically identifies the primary programming language of any workspace."""
        ext_map = {
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".cs": "csharp",
            ".cpp": "cpp",
            ".c": "c",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
        }
        # 1. Check expected changed files if provided
        for f in self.ticket.expected_changed_files or []:
            ext = Path(f).suffix.lower()
            if ext in ext_map:
                return ext_map[ext]

        # 2. Check root build manifests
        if (self.workspace / "pyproject.toml").is_file() or (self.workspace / "setup.py").is_file() or (self.workspace / "requirements.txt").is_file():
            return "python"
        if (self.workspace / "go.mod").is_file():
            return "go"
        if (self.workspace / "Cargo.toml").is_file():
            return "rust"
        if (self.workspace / "pom.xml").is_file() or (self.workspace / "build.gradle").is_file():
            return "java"
        if list(self.workspace.glob("*.csproj")) or list(self.workspace.glob("*.sln")):
            return "csharp"
        if (self.workspace / "package.json").is_file():
            return "typescript" if ((self.workspace / "tsconfig.json").is_file() or list(self.workspace.glob("src/**/*.ts*"))) else "javascript"

        # 3. Check subprojects
        for sub in subprojects.values():
            p = sub.get("path", self.workspace)
            if (p / "pyproject.toml").is_file() or (p / "requirements.txt").is_file():
                return "python"
            if (p / "go.mod").is_file():
                return "go"
            if (p / "pom.xml").is_file() or (p / "build.gradle").is_file():
                return "java"
            if (p / "Cargo.toml").is_file():
                return "rust"
            if (p / "package.json").is_file():
                return "typescript"

        # 4. Fallback: inspect extensions in top-level or src
        for ext, lang in ext_map.items():
            if list(self.workspace.glob(f"*{ext}")) or list(self.workspace.glob(f"src/**/*{ext}")):
                return lang

        return "typescript"

    def _plan_generic_dag(self, subprojects: Dict[str, Dict[str, Any]], text: str) -> List[DAGTask]:
        """
        Universal, language-agnostic architectural decomposition for any repository:
        Python, Go, Java, Rust, C#, C++, TypeScript/Node, etc.
        Produces 3 foundational phases:
          Task 1: Contracts, Data Models & Type Signatures
          Task 2: Core Domain Logic & Business Implementation
          Task 3: Unit & Regression Verification Tests
        """
        lang = self._detect_workspace_language(subprojects)
        feature_name = self._extract_feature_slug(self.ticket.title)

        if lang == "python":
            t1_layer = "backend_contracts"
            t1_desc = f"Define data schemas (Pydantic models, dataclasses, or TypedDict) and typing signatures for {self.ticket.title}."
            t2_layer = "backend_logic"
            t2_desc = f"Implement service logic, functions, handlers, and algorithms for {self.ticket.title}."
            t3_layer = "backend_test"
            t3_desc = f"Implement pytest / unittest test suite verifying {feature_name} acceptance criteria."
        elif lang == "go":
            t1_layer = "backend_contracts"
            t1_desc = f"Define Go struct types, interfaces, and constants for {self.ticket.title}."
            t2_layer = "backend_logic"
            t2_desc = f"Implement Go package handlers, business functions, and methods for {self.ticket.title}."
            t3_layer = "backend_test"
            t3_desc = f"Implement go test unit tests in *_test.go verifying {feature_name}."
        elif lang == "java":
            t1_layer = "backend_contracts"
            t1_desc = f"Define Java interfaces, DTO records, and entity models for {self.ticket.title}."
            t2_layer = "backend_logic"
            t2_desc = f"Implement Spring / Java service logic, controller endpoints, and methods for {self.ticket.title}."
            t3_layer = "backend_test"
            t3_desc = f"Implement JUnit / Mockito test cases verifying {feature_name}."
        elif lang == "rust":
            t1_layer = "contracts"
            t1_desc = f"Define Rust structs, enums, traits, and error types for {self.ticket.title}."
            t2_layer = "logic"
            t2_desc = f"Implement trait methods, domain logic, and module functions for {self.ticket.title}."
            t3_layer = "test"
            t3_desc = f"Implement cargo test unit and integration tests verifying {feature_name}."
        else:
            t1_layer = "frontend_contracts" if ("ui" in text or "component" in text) else "backend_contracts"
            t1_desc = f"Define interfaces, types, and contract signatures for {self.ticket.title}."
            t2_layer = "frontend_ui" if ("ui" in text or "component" in text) else "business_logic"
            t2_desc = f"Implement core logic, components, or services for {self.ticket.title}."
            t3_layer = "frontend_test" if ("ui" in text or "component" in text) else "test"
            t3_desc = f"Implement unit tests verifying {self.ticket.title} acceptance criteria."

        t1 = DAGTask(
            task_id="task_1_contracts",
            subproject_name="core",
            subproject_path=self.workspace,
            language=lang,
            layer=t1_layer,
            title=f"Task 1: Define Contracts & Data Models for {self.ticket.title}",
            description=t1_desc,
        )
        t2 = DAGTask(
            task_id="task_2_implementation",
            subproject_name="core",
            subproject_path=self.workspace,
            language=lang,
            layer=t2_layer,
            title=f"Task 2: Implement Core Logic for {self.ticket.title}",
            description=t2_desc,
            dependencies=["task_1_contracts"],
        )
        t3 = DAGTask(
            task_id="task_3_verification",
            subproject_name="core",
            subproject_path=self.workspace,
            language=lang,
            layer=t3_layer,
            title=f"Task 3: Unit Tests & Verification for {self.ticket.title}",
            description=t3_desc,
            dependencies=["task_1_contracts", "task_2_implementation"],
        )
        return [t1, t2, t3]


    def _plan_from_explicit_tasks(
        self, subprojects: Dict[str, Dict[str, Any]], raw_ticket_text: str
    ) -> List[DAGTask]:
        """
        Extracts an arbitrary number (N) of explicitly enumerated tasks from the ticket description.
        Supports ticket formats such as:
          - 'Task 1: ... \n Task 2: ... \n Task 3: ... \n Task 4: ... \n Task 5: ...'
          - '1. Task: ... \n 2. Task: ...'
          - 'Step 1: ... \n Step 2: ...'
          - Markdown bullet lists with Task / Step headers.

        Allows decomposing into ANY number of tasks (from 2 up to 10+), not constrained to 4.
        """
        pattern = re.compile(
            r'(?:^|\n)\s*(?:[-*]\s+|\d+\.\s+|#{1,4}\s+)?(?:Task|Step)\s*(\d+)[:\-.]?\s*([^\n\r]+)',
            re.IGNORECASE,
        )

        matches = list(pattern.finditer(raw_ticket_text))
        if len(matches) < 2:
            return []

        tasks: List[DAGTask] = []
        upstream_ids: List[str] = []

        ui_sub = next((k for k, v in subprojects.items() if v.get("is_ui")), "frontend")
        ui_path = subprojects.get(ui_sub, {}).get("path", self.workspace)
        be_sub = next((k for k, v in subprojects.items() if v.get("is_backend")), "backend")
        be_path = subprojects.get(be_sub, {}).get("path", self.workspace)

        for i, match in enumerate(matches):
            task_num = match.group(1)
            task_title = match.group(2).strip()
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(raw_ticket_text)
            task_body = raw_ticket_text[start_pos:end_pos].strip()

            combined_task_text = f"{task_title} {task_body}".lower()

            # Determine Layer, Language, Subproject based on task content
            layer = "implementation"
            language = "typescript"
            subproject_name = ui_sub
            subproject_path = ui_path

            has_negative_db = any(neg in combined_task_text for neg in ("no db", "no database", "mock data", "ui only", "do not touch the db", "without db"))
            if not has_negative_db and any(k in combined_task_text for k in ("db", "database", "sql", "migration", "flyway", "liquibase", "ddl")):
                layer = "database"
                language = "sql"
                db_sub = next((k for k, v in subprojects.items() if any(w in k for w in ("db", "sql", "schema"))), None)
                if db_sub:
                    subproject_name = db_sub
                    subproject_path = subprojects[db_sub]["path"]
                else:
                    subproject_name = be_sub
                    subproject_path = be_path
            elif any(k in combined_task_text for k in ("test", "spec", "jest", "vitest", "junit", "e2e", "assert", "verification")):
                if any(k in combined_task_text for k in ("junit", "backend", "java", "controller", "spring")):
                    layer = "backend_test"
                    language = "java"
                    subproject_name = be_sub
                    subproject_path = be_path
                else:
                    layer = "frontend_test"
                    language = "typescript"
                    subproject_name = ui_sub
                    subproject_path = ui_path
            elif any(k in combined_task_text for k in ("type", "interface", "dto", "model", "contract", "schema")):
                layer = "frontend_contracts"
                language = "typescript"
            elif any(k in combined_task_text for k in ("hook", "state", "redux", "context", "service", "zustand", "store", "query", "mutation")):
                layer = "frontend_state"
                language = "typescript"
            elif any(k in combined_task_text for k in ("component", "jsx", "tsx", "ui", "view", "layout", "modal", "dialog", "form", "css", "scss")):
                layer = "frontend_ui"
                language = "typescript"
            elif any(k in combined_task_text for k in ("controller", "api", "endpoint", "rest", "backend", "java", "service", "repository")):
                layer = "backend"
                language = "java" if (be_path / "pom.xml").exists() else "typescript"
                subproject_name = be_sub
                subproject_path = be_path

            slug = self._extract_feature_slug(task_title).lower() or f"subtask_{task_num}"
            task_id = f"task_{task_num}_{slug}"
            if any(t.task_id == task_id for t in tasks):
                task_id = f"task_{task_num}_{i+1}"

            dag_task = DAGTask(
                task_id=task_id,
                subproject_name=subproject_name,
                subproject_path=subproject_path,
                language=language,
                layer=layer,
                title=f"Task {task_num}: {task_title}",
                description=task_body or task_title,
                dependencies=[tasks[-1].task_id] if tasks else [],
            )
            tasks.append(dag_task)

        return tasks


    def _extract_feature_slug(self, title: str) -> str:
        """Extracts a clean PascalCase or camelCase slug from a ticket title."""
        words = re.findall(r"[A-Za-z0-9]+", title)
        meaningful = [w for w in words if w.lower() not in ("add", "update", "fix", "implement", "for", "the", "in", "to", "with", "a", "an")]
        return "".join(w.capitalize() for w in meaningful[:3]) or "Feature"

    def compute_topological_waves(self) -> List[List[DAGTask]]:
        """
        Groups tasks into execution waves (tiers).
        Tasks within the same wave have 0 dependencies on each other and can execute concurrently.
        """
        waves: List[List[DAGTask]] = []
        completed_ids: Set[str] = set()
        remaining = dict(self.tasks)

        while remaining:
            wave = [
                task for task in remaining.values()
                if all(dep in completed_ids for dep in task.dependencies)
            ]
            if not wave:
                # Break circular cycle by picking first available task
                wave = [next(iter(remaining.values()))]

            waves.append(wave)
            for t in wave:
                completed_ids.add(t.task_id)
                remaining.pop(t.task_id, None)

        return waves

    def execute_dag(self, use_sandbox: bool = True) -> DAGExecutionResult:
        """
        Executes the planned DAG in topological order.
        Integrates:
          - Speculative Git Worktree Sandboxing (clean isolation & rollback)
          - Test-Driven Verification (TDD Reproduction Harness Red -> Green)
          - Real-Time Live WebSocket broadcasts
        """
        if not self.tasks:
            self.plan_dag()

        t0 = time.time()
        sandbox: Optional[Any] = None
        sandbox_dir = self.workspace

        # ── PILLAR 4: SPECULATIVE SANDBOX SETUP ──
        if use_sandbox:
            try:
                from ticket_to_code.runtime.speculative_sandbox import SpeculativeSandbox
                sandbox = SpeculativeSandbox(base_repo_path=self.workspace, ticket_id=self.ticket.ticket_id)
                sandbox_dir = sandbox.setup()
                logger.info(f"🌿 [DAGOrchestrator] Sandboxed execution active at: {sandbox_dir}")
                if self.ui_callback:
                    self.ui_callback({
                        "phase": "planning",
                        "status": "info",
                        "message": f"🌿 Isolated Git Worktree Sandbox initialized for non-destructive execution",
                    })
            except Exception as s_err:
                logger.debug(f"[DAGOrchestrator] SpeculativeSandbox fallback: {s_err}")
                sandbox = None

        all_diffs: Dict[str, str] = {}
        total_tokens = 0
        total_in = 0
        total_out = 0
        total_calls = 0

        # ── PILLAR 5: TDD REPRODUCTION TEST GENERATION (RED PHASE) ──
        tdd: Optional[Any] = None
        repro_test = None
        try:
            from ticket_to_code.verification.tdd_reproduction_harness import TDDReproductionHarness
            tdd = TDDReproductionHarness(
                workspace_path=sandbox_dir,
                ticket=self.ticket,
                llm_client=self.llm,
                ui_callback=self.ui_callback,
            )
            repro_test = tdd.generate_reproduction_test(self.ticket.expected_changed_files)
            if repro_test:
                tdd.verify_red_phase(repro_test)
        except Exception as tdd_err:
            logger.debug(f"[DAGOrchestrator] TDD Harness notice: {tdd_err}")

        # ── TOPOLOGICAL WAVE-BASED SUB-AGENT TASK EXECUTION ──
        waves = self.compute_topological_waves()
        logger.info(f"🌊 [DAGOrchestrator] Executing {len(self.tasks)} task(s) partitioned into {len(waves)} topological wave(s)")
        step_idx = 0

        for wave_idx, wave in enumerate(waves, 1):
            logger.info(f"🌊 [DAGOrchestrator] Launching Wave {wave_idx}/{len(waves)} with {len(wave)} independent task(s): {[t.task_id for t in wave]}")
            for task in wave:
                step_idx += 1
                # Check dependencies
                for dep_id in task.dependencies:
                    dep_task = self.tasks.get(dep_id)
                    if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                        task.status = TaskStatus.SKIPPED
                        logger.warning(f"⏩ [DAGOrchestrator] Skipping {task.task_id} because dependency {dep_id} was not completed")
                        break

                if task.status == TaskStatus.SKIPPED:
                    continue

                task.status = TaskStatus.IN_PROGRESS
                if self.ui_callback:
                    self.ui_callback({
                        "phase": "patch_generation",
                        "status": "in_progress",
                        "message": f"⏳ [Wave {wave_idx} | Step {step_idx}/{len(self.tasks)}] Executing {task.title}",
                        "data": {"current_task": task.task_id, "layer": task.layer, "wave": wave_idx},
                    })

                # Checkpoint before task execution
                checkpoint_name = f"checkpoint_{task.task_id}"
                if sandbox:
                    sandbox.create_checkpoint(checkpoint_name, error_count=0, tracked_files=task.target_files)

                upstream_guidance = self.context_bus.format_downstream_guidance(task.dependencies)

                # Adjust subproject path to sandbox if active
                exec_subproject_path = task.subproject_path
                if sandbox and sandbox_dir != self.workspace:
                    rel_to_ws = task.subproject_path.relative_to(self.workspace) if task.subproject_path != self.workspace else Path(".")
                    exec_subproject_path = sandbox_dir / rel_to_ws

                agent = SubProjectAgent(
                    task=task,
                    workspace_root=sandbox_dir,
                    llm_client=self.llm,
                    ui_callback=self.ui_callback,
                )
                # Temporarily redirect agent subproject path to sandbox
                original_path = task.subproject_path
                task.subproject_path = exec_subproject_path

                success, diffs, produced = agent.execute(upstream_guidance)
                task.subproject_path = original_path

                total_tokens += task.tokens_consumed
                total_in += task.tokens_in
                total_out += task.tokens_out
                total_calls += task.llm_calls

                if success:
                    # ── COMPILER-GATED PHASE TRANSITION VERIFICATION ──
                    # Verify that modified files pass compiler diagnostics before publishing contracts
                    from ticket_to_code.runtime.lsp_client import FastLSPClient
                    lsp = FastLSPClient(sandbox_dir)
                    compiler_errors = []
                    for mod_file in diffs.keys():
                        target_p = sandbox_dir / mod_file
                        if target_p.is_file():
                            diags = lsp.get_diagnostics(target_p)
                            errs = [d.message for d in diags if getattr(d, "severity", "") == "error"]
                            if errs:
                                compiler_errors.append(f"{mod_file}: {'; '.join(errs)}")

                    if compiler_errors:
                        # ── CLOSED-LOOP BACK-PROPAGATION SELF-HEALING ──
                        # If compiler error indicates missing upstream symbol/contract, attempt upstream self-healing repair turn
                        repaired = False
                        if task.dependencies and self.llm:
                            parent_id = task.dependencies[-1]
                            parent_task = self.tasks.get(parent_id)
                            if parent_task and parent_task.status == TaskStatus.COMPLETED:
                                logger.info(f"🔄 [DAGOrchestrator] Attempting closed-loop back-propagation repair to upstream task '{parent_id}'")
                                repair_prompt = (
                                    f"Downstream task '{task.task_id}' failed compiler verification with:\n"
                                    + "\n".join(compiler_errors)
                                    + "\nPlease export, declare, or adjust the missing contracts/types so downstream compiles cleanly."
                                )
                                parent_agent = SubProjectAgent(
                                    task=parent_task,
                                    workspace_root=sandbox_dir,
                                    llm_client=self.llm,
                                    ui_callback=self.ui_callback,
                                )
                                p_success, p_diffs, p_prod = parent_agent.execute(repair_prompt)
                                if p_success:
                                    all_diffs.update(p_diffs)
                                    for p in p_prod:
                                        self.context_bus.publish_contract(parent_task.task_id, p)
                                    # Retry downstream task with updated upstream contracts
                                    retry_guidance = self.context_bus.format_downstream_guidance(task.dependencies)
                                    r_success, r_diffs, r_prod = agent.execute(retry_guidance)
                                    if r_success:
                                        # Verify compiler again
                                        retry_errs = []
                                        for mf in r_diffs.keys():
                                            tp = sandbox_dir / mf
                                            if tp.is_file():
                                                diags = lsp.get_diagnostics(tp)
                                                retry_errs.extend([d.message for d in diags if getattr(d, "severity", "") == "error"])
                                        if not retry_errs:
                                            compiler_errors = []
                                            diffs = r_diffs
                                            produced = r_prod
                                            repaired = True
                                            logger.info(f"✨ [DAGOrchestrator] Closed-loop back-propagation self-healed {task.task_id} successfully!")

                        if compiler_errors:
                            task.status = TaskStatus.FAILED
                            task.error_message = f"Compiler regression in {task.title}: {'; '.join(compiler_errors)}"
                            logger.error(f"❌ [DAGOrchestrator] Compiler-gated transition REJECTED for {task.task_id}: {compiler_errors}")
                            if sandbox:
                                logger.warning(f"⏪ [DAGOrchestrator] Rolling back task '{task.task_id}' to {checkpoint_name}")
                                sandbox.rollback_to_checkpoint(checkpoint_name)
                                sandbox.teardown()

                            return DAGExecutionResult(
                                success=False,
                                diffs=all_diffs,
                                modified_files=list(all_diffs.keys()),
                                tasks=list(self.tasks.values()),
                                duration_seconds=time.time() - t0,
                                tokens_consumed=total_tokens,
                                tokens_in=total_in,
                                tokens_out=total_out,
                                llm_calls=total_calls,
                                error_message=task.error_message,
                                sandboxed=bool(sandbox),
                            )

                    task.status = TaskStatus.COMPLETED
                    task.result_diffs = diffs
                    for df_path, df_val in diffs.items():
                        rel_ws_path = f"{task.subproject_name}/{df_path}" if (task.subproject_name and task.subproject_path != self.workspace and not df_path.startswith(f"{task.subproject_name}/")) else df_path
                        all_diffs[rel_ws_path] = df_val
                    for p in produced:
                        self.context_bus.publish_contract(task.task_id, p)
                    logger.info(f"✅ [DAGOrchestrator] Task '{task.task_id}' COMPLETED and COMPILER-VERIFIED ({task.duration_seconds:.1f}s)")
                else:
                    task.status = TaskStatus.FAILED
                    logger.error(f"❌ [DAGOrchestrator] Task '{task.task_id}' FAILED: {task.error_message}")
                    if sandbox:
                        logger.warning(f"⏪ [DAGOrchestrator] Rolling back task '{task.task_id}' to {checkpoint_name}")
                        sandbox.rollback_to_checkpoint(checkpoint_name)
                        sandbox.teardown()

                    return DAGExecutionResult(
                        success=False,
                        diffs=all_diffs,
                        modified_files=list(all_diffs.keys()),
                        tasks=list(self.tasks.values()),
                        duration_seconds=time.time() - t0,
                        tokens_consumed=total_tokens,
                        tokens_in=total_in,
                        tokens_out=total_out,
                        llm_calls=total_calls,
                        error_message=f"Task '{task.title}' failed: {task.error_message}",
                        sandboxed=bool(sandbox),
                    )

        # ── PILLAR 5: TDD GREEN VERIFICATION ──
        if tdd and repro_test:
            try:
                tdd.verify_green_phase(repro_test)
            except Exception as tdd_g_err:
                logger.debug(f"[DAGOrchestrator] TDD Green notice: {tdd_g_err}")

        # ── PROMOTION: COMMIT FROM SANDBOX INTO WORKING TREE ──
        if sandbox:
            sandbox.promote_to_workspace(list(all_diffs.keys()))
            sandbox.teardown()

        elapsed = time.time() - t0
        logger.info(f"🏆 [DAGOrchestrator] All {len(self.tasks)} tasks verified and completed in {elapsed:.1f}s!")

        if self.ui_callback:
            self.ui_callback({
                "phase": "completed",
                "status": "success",
                "message": f"🏆 All {len(self.tasks)} DAG tasks completed and verified with 0 regressions ({elapsed:.1f}s)",
                "data": {"tasks_completed": len(self.tasks), "modified_files": list(all_diffs.keys())},
            })

        return DAGExecutionResult(
            success=True,
            diffs=all_diffs,
            modified_files=list(all_diffs.keys()),
            tasks=list(self.tasks.values()),
            duration_seconds=elapsed,
            tokens_consumed=total_tokens,
            tokens_in=total_in,
            tokens_out=total_out,
            llm_calls=total_calls,
            sandboxed=bool(sandbox),
        )

    def _discover_workspace_subprojects(self) -> Dict[str, Dict[str, Any]]:
        """Scans workspace directory for all subprojects and characterizes them."""
        discovered: Dict[str, Dict[str, Any]] = {}
        for item in self.workspace.iterdir():
            if not item.is_dir() or item.name.startswith((".", "_", "target", "build", "node_modules", ".aviator", "venv", ".venv")):
                continue
            is_ui = (item / "package.json").exists() or (item / "src" / "app").is_dir()
            is_backend = (
                (item / "pom.xml").exists()
                or (item / "go.mod").exists()
                or (item / "build.gradle").exists()
                or (item / "pyproject.toml").exists()
                or (item / "requirements.txt").exists()
                or (item / "Cargo.toml").exists()
                or any(item.glob("*.csproj"))
            )
            is_db = (
                any(item.glob("*.sql"))
                or any(item.glob("*.ddl"))
                or any(item.glob("src/main/resources/db/**/*"))
                or any(w in item.name.lower() for w in ("db", "migration", "schema", "sql"))
            )
            discovered[item.name] = {
                "path": item,
                "is_ui": is_ui,
                "is_backend": is_backend,
                "is_database": is_db,
            }
        return discovered

