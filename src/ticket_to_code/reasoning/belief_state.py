"""Belief-state primitives for the CC4E reasoning kernel.

Design intent:
- Observations are immutable raw outputs from tools.
- Evidence is an interpretation derived from observations.
- Beliefs and theories are revisable and evidence-backed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass(frozen=True)
class Observation:
    step: int
    action: str
    args: Dict[str, object]
    output: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass(frozen=True)
class EvidenceItem:
    step: int
    source_action: str
    claim_key: str
    claim_text: str
    polarity: int  # +1 supports, -1 contradicts
    source_reliability: float
    references: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class WorkingTheory:
    key: str
    statement: str
    candidates: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)
    status: str = "open"  # open | corroborated | refuted


@dataclass
class Objective:
    key: str
    text: str
    status: str = "open"  # open | satisfied | blocked
    evidence_refs: List[str] = field(default_factory=list)


@dataclass
class Contradiction:
    claim_key: str
    support_refs: List[str] = field(default_factory=list)
    contradict_refs: List[str] = field(default_factory=list)
    status: str = "open"  # open | resolved


@dataclass
class BeliefState:
    """Ranked possibilities by dimension, each tied to evidence refs."""
    dimensions: Dict[str, List[Dict[str, object]]] = field(
        default_factory=lambda: {
            "owner": [],
            "service": [],
            "api": [],
            "ui": [],
            "config": [],
            "tests": [],
        }
    )

    def upsert_candidate(self, dimension: str, candidate: str, evidence_ref: str) -> None:
        rows = self.dimensions.setdefault(dimension, [])
        for row in rows:
            if row.get("candidate") == candidate:
                refs = row.setdefault("evidence_refs", [])
                if evidence_ref not in refs:
                    refs.append(evidence_ref)
                row["support_count"] = int(row.get("support_count", 0)) + 1
                return
        rows.append(
            {
                "candidate": candidate,
                "evidence_refs": [evidence_ref],
                "support_count": 1,
            }
        )

    def ranked(self, dimension: str) -> List[Dict[str, object]]:
        rows = list(self.dimensions.get(dimension, []))
        rows.sort(key=lambda r: int(r.get("support_count", 0)), reverse=True)
        return rows


def objective_coverage(objectives: List[Objective]) -> float:
    if not objectives:
        return 0.0
    done = sum(1 for o in objectives if o.status == "satisfied")
    return float(done) / float(len(objectives))


def source_reliability(action: str) -> float:
    """Reliability of source action, not confidence in beliefs/theories."""
    table = {
        "search_code": 0.55,
        "grep": 0.60,
        "rag": 0.50,
        "read_file": 0.75,
        "find_owner": 0.70,
        "recall_memory": 0.55,
        "recall_feature": 0.65,
        "brain_query": 0.65,
        "brain_search_feature": 0.65,
        "brain_get_owner_files": 0.70,
        "brain_get_playbook": 0.55,
        "brain_get_root_causes": 0.55,
        "write_patch": 0.85,
        "run_build": 0.90,
        "verify_outcome": 0.95,
    }
    return float(table.get(action, 0.50))
