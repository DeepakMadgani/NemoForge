"""Active CC4E Brain manager for ticket-to-code reasoning.

Loads structured Brain artifacts once, builds in-memory indexes, and serves
fast deterministic queries. The Brain does not reason; it returns facts.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


class BrainManager:
    """Query layer over `src/ticket_to_code/knowledge/brain` metadata."""

    def __init__(self, brain_root: Optional[str] = None):
        base = Path(__file__).resolve().parents[1] / "knowledge" / "brain"
        self.brain_root = Path(brain_root) if brain_root else base
        self.meta_root = self.brain_root / "metadata"

        self.machine: Dict[str, Any] = {}
        self.feature_graph: Dict[str, Any] = {"nodes": [], "edges": []}
        self.ticket_patterns: List[Dict[str, Any]] = []
        self.playbooks: List[Dict[str, Any]] = []

        self._feature_by_id: Dict[str, Dict[str, Any]] = {}
        self._feature_by_name: Dict[str, Dict[str, Any]] = {}
        self._playbook_by_name: Dict[str, Dict[str, Any]] = {}
        self._pattern_by_id: Dict[str, Dict[str, Any]] = {}
        self._skill_to_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self._load()

    def _read_json(self, path: Path, default: Any) -> Any:
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("BrainManager read failed (%s): %s", path, exc)
        return default

    def _load(self) -> None:
        self.machine = self._read_json(self.brain_root / "18_Machine_Readable_Brain.json", {})
        self.feature_graph = self._read_json(self.meta_root / "feature_graph.json", {"nodes": [], "edges": []})
        self.ticket_patterns = self._read_json(self.meta_root / "ticket_patterns.json", [])
        self.playbooks = self._read_json(self.meta_root / "investigation_playbooks.json", [])

        for f in self.machine.get("features", []):
            fid = _norm(str(f.get("id", "")))
            if fid:
                self._feature_by_id[fid] = f

        for node in self.feature_graph.get("nodes", []):
            fid = _norm(str(node.get("id", "")))
            if fid:
                self._feature_by_id.setdefault(fid, node)
            fname = _norm(str(node.get("name", "")))
            if fname:
                self._feature_by_name[fname] = node

        for p in self.playbooks:
            key = _norm(str(p.get("playbook", "")))
            if key:
                self._playbook_by_name[key] = p

        for p in self.ticket_patterns:
            pid = _norm(str(p.get("pattern_id", "")))
            if pid:
                self._pattern_by_id[pid] = p
            for sk in (p.get("typical_skills") or []):
                k = _norm(str(sk))
                if not k:
                    continue
                self._skill_to_patterns.setdefault(k, []).append(p)

    def search_feature(self, query: str, limit: int = 6) -> List[Dict[str, Any]]:
        q = _norm(query)
        if not q:
            return []
        scored: List[tuple[int, Dict[str, Any]]] = []
        for node in self.feature_graph.get("nodes", []):
            name = _norm(str(node.get("name", "")))
            fid = _norm(str(node.get("id", "")))
            text = f"{name} {fid}"
            if q in text:
                score = 10 if q == name or q == fid else 5
                scored.append((score, node))
            else:
                overlap = len(set(q.split()) & set(text.split()))
                if overlap:
                    scored.append((overlap, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:limit]]

    def search_service(self, query: str, limit: int = 6) -> List[Dict[str, Any]]:
        q = _norm(query)
        out: List[Dict[str, Any]] = []
        for node in self.feature_graph.get("nodes", []):
            meta = node.get("metadata") or {}
            service = _norm(str(meta.get("owner_service", "")))
            if service and q in service:
                out.append({
                    "feature": node.get("name"),
                    "feature_id": node.get("id"),
                    "owner_service": meta.get("owner_service"),
                    "primary_files": meta.get("primary_files", []),
                })
        return out[:limit]

    def search_ticket_pattern(self, ticket_text: str, intent: str = "", limit: int = 4) -> List[Dict[str, Any]]:
        q = _norm(ticket_text)
        q_tokens = [t for t in q.split() if len(t) >= 3]
        scored: List[tuple[int, Dict[str, Any]]] = []
        for p in self.ticket_patterns:
            pid = _norm(str(p.get("pattern_id", "")))
            p_intent = _norm(str(p.get("intent", "")))
            score = 0
            if intent and intent == p_intent:
                score += 4
            haystacks = [
                pid,
                _norm(" ".join([str(x) for x in (p.get("typical_features") or [])])),
                _norm(" ".join([str(x) for x in (p.get("typical_files") or [])])),
                _norm(" ".join([str(x) for x in (p.get("typical_skills") or [])])),
                _norm(" ".join([str(x) for x in (p.get("investigation_order") or [])])),
                _norm(" ".join([str(x) for x in (p.get("verification_order") or [])])),
            ]
            for tok in q_tokens:
                if any(tok in h for h in haystacks if h):
                    score += 2
            if q and any(q in h for h in haystacks if h):
                score += 3
            if score > 0:
                scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:limit]]

    def search_skill(self, skill_name: str, limit: int = 4) -> List[Dict[str, Any]]:
        return (self._skill_to_patterns.get(_norm(skill_name), []) or [])[:limit]

    def get_playbook(self, name_or_query: str) -> Optional[Dict[str, Any]]:
        q = _norm(name_or_query)
        if q in self._playbook_by_name:
            return self._playbook_by_name[q]
        for key, pb in self._playbook_by_name.items():
            if q and q in key:
                return pb
        return None

    def get_root_causes(self, pattern_id_or_query: str) -> List[Dict[str, Any]]:
        q = _norm(pattern_id_or_query)
        pattern = self._pattern_by_id.get(q)
        if pattern is None:
            for p in self.ticket_patterns:
                if q in _norm(str(p.get("pattern_id", ""))):
                    pattern = p
                    break
        if pattern is None:
            return []
        fp = pattern.get("failure_probability") or {}
        out = [{"cause": k, "probability": float(v)} for k, v in fp.items()]
        out.sort(key=lambda x: x["probability"], reverse=True)
        return out

    def get_related_features(self, feature_id_or_name: str, limit: int = 8) -> List[Dict[str, Any]]:
        q = _norm(feature_id_or_name)
        related: List[Dict[str, Any]] = []
        for e in self.feature_graph.get("edges", []):
            src = _norm(str(e.get("source", "")))
            dst = _norm(str(e.get("target", "")))
            if q == src or q == dst:
                related.append(e)
        return related[:limit]

    def get_owner_files(self, feature_id_or_name: str) -> List[str]:
        q = _norm(feature_id_or_name)
        node = self._feature_by_id.get(q) or self._feature_by_name.get(q)
        if not node:
            matches = self.search_feature(feature_id_or_name, limit=1)
            node = matches[0] if matches else None
        if not node:
            return []
        meta = node.get("metadata") or {}
        files = meta.get("primary_files") or []
        return [str(x) for x in files]

    def get_feature(self, feature_id_or_name: str) -> Optional[Dict[str, Any]]:
        q = _norm(feature_id_or_name)
        return self._feature_by_id.get(q) or self._feature_by_name.get(q)

    def query_ticket(self, ticket_text: str, ticket_type: str = "") -> Dict[str, Any]:
        """Ticket-centric Brain query used by the kernel before RAG."""
        features = self.search_feature(ticket_text, limit=4)
        patterns = self.search_ticket_pattern(ticket_text, intent=ticket_type, limit=3)

        playbooks: List[Dict[str, Any]] = []
        for p in patterns:
            pid = _norm(str(p.get("pattern_id", "")))
            for pb in self.playbooks:
                pbk = _norm(str(pb.get("playbook", "")))
                if pbk and (pbk in pid or pid in pbk):
                    playbooks.append(pb)
                    break

        owner_files: List[str] = []
        for f in features:
            owner_files.extend(self.get_owner_files(str(f.get("id") or f.get("name") or "")))
        owner_files = list(dict.fromkeys(owner_files))[:12]

        skills: List[str] = []
        for p in patterns:
            skills.extend([str(x) for x in (p.get("typical_skills") or [])])
        skills = list(dict.fromkeys(skills))[:8]

        return {
            "features": features,
            "ticket_patterns": patterns,
            "playbooks": playbooks,
            "skills": skills,
            "owner_files": owner_files,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "brain_root": str(self.brain_root),
            "feature_nodes": len(self.feature_graph.get("nodes", [])),
            "feature_edges": len(self.feature_graph.get("edges", [])),
            "ticket_patterns": len(self.ticket_patterns),
            "playbooks": len(self.playbooks),
            "machine_features": len(self.machine.get("features", [])),
        }
