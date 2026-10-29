"""
Deductive State Blackboard (Reasoning & Hypothesis Tracker).

Maintains verified ground-truth facts, rejected assumptions, discovered contracts,
and deductive state across multi-turn ReAct tool execution cycles.
Prevents LLM hallucination by forcing every assertion to be grounded in live disk observations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


import json
from pathlib import Path


@dataclass
class ObservationRecord:
    tool_name: str
    target: str
    summary: str
    raw_snippet: str
    deductions: List[str] = field(default_factory=list)


class DeductiveBlackboard:
    """Stateful scratchpad that accumulates verified facts and eliminates hallucinations

    during multi-turn autonomous investigation.
    """

    def __init__(self, workspace_path: str = ""):
        self.workspace_path = workspace_path
        self.verified_facts: List[str] = []
        self.rejected_hypotheses: List[str] = []
        self.discovered_endpoints: List[str] = []
        self.discovered_entities: Dict[str, List[str]] = {}
        self.ui_framework_detected: Optional[str] = None
        self.history: List[ObservationRecord] = []
        self._seen_facts: Set[str] = set()
        self._seen_rejections: Set[str] = set()
        self.fact_metadata: Dict[str, Dict[str, Any]] = {}

    def record_fact(self, fact: str, file_path: str = "", entity: str = "") -> None:
        """Add a verified fact grounded in code observation, tracking associated file and entity."""
        cleaned = fact.strip()
        if cleaned and cleaned not in self._seen_facts:
            self._seen_facts.add(cleaned)
            self.verified_facts.append(cleaned)
            self.fact_metadata[cleaned] = {
                "file": file_path.replace("\\", "/").strip(),
                "entity": entity.strip(),
            }

    def evict_facts_for_path(self, file_path: str) -> int:
        """Evicts obsolete facts when a file is modified to prevent stale/contradictory assertions."""
        if not file_path:
            return 0
        norm = file_path.replace("\\", "/").strip()
        fname = Path(norm).name.lower()
        evicted = 0
        surviving_facts = []

        for fact in self.verified_facts:
            meta = self.fact_metadata.get(fact, {})
            tagged_file = meta.get("file", "").lower()
            fact_low = fact.lower()
            
            # Check if fact was explicitly tagged with this file or references its basename
            is_stale = (
                (tagged_file and (tagged_file in norm.lower() or norm.lower() in tagged_file))
                or (fname and len(fname) >= 4 and fname in fact_low)
            )
            if is_stale:
                evicted += 1
                self._seen_facts.discard(fact)
                self.fact_metadata.pop(fact, None)
            else:
                surviving_facts.append(fact)

        self.verified_facts = surviving_facts
        return evicted

    def evict_contradicted_facts(self, regex_pattern: str) -> int:
        """Evicts facts matching a regex contradiction pattern."""
        pat = re.compile(regex_pattern, re.IGNORECASE)
        evicted = 0
        surviving = []
        for fact in self.verified_facts:
            if pat.search(fact):
                evicted += 1
                self._seen_facts.discard(fact)
                self.fact_metadata.pop(fact, None)
            else:
                surviving.append(fact)
        self.verified_facts = surviving
        return evicted

    def parse_and_ingest_structured_facts(self, text: str) -> List[str]:
        """Parse structured facts from XML tags (<verified_fact file="...">...</verified_fact>)

        and JSON blobs ({'file': '...', 'fact': '...'}).
        """
        extracted: List[str] = []

        # 1. XML tag extraction
        xml_pattern = re.compile(r'<verified_fact(?:\s+file=["\']([^"\']+)["\'])?(?:\s+entity=["\']([^"\']+)["\'])?>\s*(.*?)\s*</verified_fact>', re.DOTALL | re.IGNORECASE)
        for m in xml_pattern.finditer(text):
            fpath = m.group(1) or ""
            entity = m.group(2) or ""
            fact_text = m.group(3).strip()
            if fact_text:
                self.record_fact(fact_text, file_path=fpath, entity=entity)
                extracted.append(fact_text)

        # 2. JSON fact blocks
        json_pattern = re.compile(r'```json\s*(\[\s*\{.*?\}\s*\]|\{.*?\})\s*```', re.DOTALL)
        for m in json_pattern.finditer(text):
            raw_json = m.group(1)
            try:
                data = json.loads(raw_json)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if isinstance(item, dict) and "fact" in item:
                        f_text = str(item["fact"]).strip()
                        f_path = str(item.get("file", "")).strip()
                        f_ent = str(item.get("entity", "")).strip()
                        if f_text:
                            self.record_fact(f_text, file_path=f_path, entity=f_ent)
                            extracted.append(f_text)
            except Exception:
                pass

        return extracted

    def record_rejection(self, rejection: str) -> None:
        """Record an explicit rejection of a speculative hypothesis."""
        cleaned = rejection.strip()
        if cleaned and cleaned not in self._seen_rejections:
            self._seen_rejections.add(cleaned)
            self.rejected_hypotheses.append(cleaned)

    def record_endpoint(self, endpoint: str) -> None:
        """Record a verified REST endpoint or gateway route."""
        cleaned = endpoint.strip()
        if cleaned and cleaned not in self.discovered_endpoints:
            self.discovered_endpoints.append(cleaned)

    def record_entity_field(self, entity_name: str, field_def: str) -> None:
        """Record a verified entity or database column."""
        self.discovered_entities.setdefault(entity_name, [])
        if field_def not in self.discovered_entities[entity_name]:
            self.discovered_entities[entity_name].append(field_def)

    def ingest_tool_observation(self, tool_name: str, target: str, output: str) -> List[str]:
        """Automatically parses tool outputs for structural facts, signatures, and anti-hallucination deductions."""
        deductions: List[str] = []
        snippet = output[:3000]

        # 1. Deduce Java Entity Column / Field Types (e.g. priority as String vs Enum)
        if "priority" in output.lower():
            if re.search(r'String\s+priority\b', output):
                fact = "Ground Truth: 'priority' field is a plain Java String, NOT an Enum (verified in entity definition)."
                self.record_fact(fact)
                self.record_rejection("DO NOT use 'IssuePriority' enum: It does not exist in the codebase. Use case-insensitive String matching (UPPER(i.priority)).")
                deductions.append(fact)
            elif re.search(r'enum\s+IssuePriority\b', output):
                fact = "Ground Truth: 'IssuePriority' is declared as an Enum in the codebase."
                self.record_fact(fact)
                deductions.append(fact)

        # 2. Deduce DTO Constructors & Backward Compatibility
        m_dto = re.search(r'public\s+([A-Za-z0-9_]+)\s*\(([^)]+)\)', output)
        if m_dto and ("dto" in target.lower() or "Count" in target or "Dto" in target):
            dto_name = m_dto.group(1)
            params = [p.strip() for p in m_dto.group(2).split(",") if p.strip()]
            fact = f"Ground Truth: '{dto_name}' defines existing constructor with {len(params)} parameters ({', '.join(params[:4])}...)."
            self.record_fact(fact)
            self.record_rejection(f"DO NOT delete the existing {len(params)}-argument constructor in '{dto_name}'. Existing tests and callers depend on it. Overload instead.")
            deductions.append(fact)

        # 3. Deduce UI Framework (Enterprise UX Components vs Angular Material)
        if any(tag in output for tag in ["<ot-icon-button", "<ot-button", "ot-ellipsis", "<se-table", "ee-ng-components"]):
            self.ui_framework_detected = "Enterprise UX Components (@cloudforge/components-lib)"
            fact = "Ground Truth: UI uses Enterprise UX Components components (e.g. <ot-icon-button>, <se-table>), NOT Angular Material."
            self.record_fact(fact)
            self.record_rejection("DO NOT use Angular Material components (<mat-chip>, <mat-icon>, <mat-cell>). They are not imported in this project and will cause template parse errors.")
            deductions.append(fact)

        # 4. Deduce UI Navigation vs Overlay Trigger
        if "navigateToOverlay" in output:
            fact = "Ground Truth: Issue action triggers side overlay via 'navigateToOverlay(row, ...)', NOT an Angular route navigation [routerLink]."
            self.record_fact(fact)
            self.record_rejection("DO NOT replace 'navigateToOverlay' with '[routerLink]'. Issue details open in an overlay.")
            deductions.append(fact)

        # 5. Deduce Spring Cloud Gateway Pass-Through
        if "Path=/api/v1/issues/**" in output or "Path=/api/v1/projects/**" in output:
            m_route = re.search(r'Path=([^\n]+)', output)
            route_val = m_route.group(1).strip() if m_route else "/api/v1/issues/**"
            fact = f"Ground Truth: Gateway routes '{route_val}' directly to the microservice via Spring Cloud Gateway without payload alteration."
            self.record_fact(fact)
            self.record_rejection("DO NOT modify Gateway routing: Route predicates already proxy all sub-paths transparently.")
            deductions.append(fact)

        # 6. Deduce REST URL Mappings
        m_url = re.search(r'([A-Za-z0-9_]+_URL)\s*=\s*["\']([^"\']+)["\']', output)
        if m_url:
            const_name, path_val = m_url.group(1), m_url.group(2)
            fact = f"Ground Truth: Endpoint mapping constant '{const_name}' = '{path_val}'."
            self.record_fact(fact)
            self.record_endpoint(f"{const_name}: {path_val}")
            deductions.append(fact)

        # 7. Deduce Database Columns from Flyway Migration SQL
        if ".sql" in target.lower() or "create table" in output.lower():
            m_cols = re.findall(r'([a-zA-Z0-9_]+)\s+(varchar|text|bigint|int|uuid|boolean|timestamp)', output, re.IGNORECASE)
            if m_cols:
                table_match = re.search(r'create\s+table\s+(?:if\s+not\s+exists\s+)?([a-zA-Z0-9_]+)', output, re.IGNORECASE)
                tname = table_match.group(1) if table_match else target
                for col_name, col_type in m_cols[:15]:
                    self.record_entity_field(tname, f"{col_name} ({col_type})")
                fact = f"Ground Truth: Database table '{tname}' verified with columns: {', '.join(f'{c[0]}:{c[1]}' for c in m_cols[:6])}."
                self.record_fact(fact)
                deductions.append(fact)

        rec = ObservationRecord(
            tool_name=tool_name,
            target=target,
            summary=f"{tool_name} on {target} -> {len(deductions)} deductions",
            raw_snippet=snippet,
            deductions=deductions,
        )
        self.history.append(rec)
        return deductions

    def has_verified_target(self, file_path_substring: str) -> bool:
        """Check whether the model has executed at least one inspection tool targeting a specific file."""
        sub = file_path_substring.lower().replace("\\", "/")
        return any(sub in rec.target.lower().replace("\\", "/") for rec in self.history)

    def render_prompt_xml(self) -> str:
        """Renders the stateful <deductive_blackboard> XML block to inject into the LLM context."""
        facts_block = "\n".join(f"- {f}" for f in self.verified_facts) or "- No ground truth facts verified yet."
        rejections_block = "\n".join(f"- {r}" for r in self.rejected_hypotheses) or "- None."
        endpoints_block = "\n".join(f"- {e}" for e in self.discovered_endpoints) or "- None."
        
        entities_block = ""
        if self.discovered_entities:
            ent_lines = []
            for ent, cols in self.discovered_entities.items():
                ent_lines.append(f"  * {ent}: {', '.join(cols[:8])}")
            entities_block = "\n" + "\n".join(ent_lines)
        else:
            entities_block = " None."

        ui_fw = f"- UI Framework: {self.ui_framework_detected}" if self.ui_framework_detected else "- UI Framework: Unverified."

        return (
            "<deductive_blackboard>\n"
            "# DEDUCTIVE REASONING BLACKBOARD (GROUND TRUTH FACTS ACROSS TOOL CALLS)\n\n"
            "## 1. Verified Codebase Facts (Hard Constraints):\n"
            f"{facts_block}\n\n"
            "## 2. Explicitly Rejected Hypotheses (Forbidden Inventions):\n"
            f"{rejections_block}\n\n"
            "## 3. Discovered REST Contracts & Routes:\n"
            f"{endpoints_block}\n\n"
            "## 4. Discovered Database & Entity Schema:\n"
            f"{entities_block}\n\n"
            "## 5. UI Environment:\n"
            f"{ui_fw}\n\n"
            "CRITICAL INSTRUCTION: All code changes and architectural plans MUST strictly adhere to the verified facts above. "
            "Never generate code using entities or enums in the Rejected Hypotheses list!\n"
            "</deductive_blackboard>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes current deductive state to a clean dictionary."""
        return {
            "verified_facts": list(self.verified_facts),
            "rejected_hypotheses": list(self.rejected_hypotheses),
            "discovered_endpoints": list(self.discovered_endpoints),
            "discovered_entities": {k: list(v) for k, v in self.discovered_entities.items()},
            "ui_framework_detected": self.ui_framework_detected,
            "fact_metadata": dict(self.fact_metadata),
        }

    def load_dict(self, data: Dict[str, Any]) -> None:
        """Loads deductions from a serialized dictionary."""
        if not isinstance(data, dict):
            return
        for f in data.get("verified_facts", []):
            meta = data.get("fact_metadata", {}).get(f, {})
            self.record_fact(f, file_path=meta.get("file", ""), entity=meta.get("entity", ""))
        for r in data.get("rejected_hypotheses", []):
            self.record_rejection(r)
        for e in data.get("discovered_endpoints", []):
            self.record_endpoint(e)
        for ent, fields in data.get("discovered_entities", {}).items():
            for fld in fields:
                self.record_entity_field(ent, fld)
        if data.get("ui_framework_detected"):
            self.ui_framework_detected = data["ui_framework_detected"]

    def save_to_disk(self, session_id: str, storage_dir: Optional[str] = None) -> Optional[str]:
        """Persist blackboard facts to disk so deductions survive server restarts and process crashes."""
        try:
            base_dir = Path(storage_dir) if storage_dir else (Path(self.workspace_path) / ".aviator" / "blackboard")
            base_dir.mkdir(parents=True, exist_ok=True)
            safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', session_id or "default")
            target = base_dir / f"{safe_id}.json"
            target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
            return str(target)
        except Exception:
            return None

    def load_from_disk(self, session_id: str, storage_dir: Optional[str] = None) -> bool:
        """Load persisted blackboard facts from disk if existing."""
        try:
            base_dir = Path(storage_dir) if storage_dir else (Path(self.workspace_path) / ".aviator" / "blackboard")
            safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', session_id or "default")
            target = base_dir / f"{safe_id}.json"
            if target.is_file():
                data = json.loads(target.read_text(encoding="utf-8"))
                self.load_dict(data)
                return True
        except Exception:
            pass
        return False

