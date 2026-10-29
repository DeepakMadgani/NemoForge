import json
import logging
import os
import re
import datetime
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

# We use the new models from models.py for parsing the stored brain
from ticket_to_code.models import CapabilityGraphReport, CapabilityGraphNode, FileParticipant

logger = logging.getLogger(__name__)

class CapabilityVersionHistory(BaseModel):
    """Wrapper to store the history of a capability"""
    capability_id: str
    versions: List[CapabilityGraphNode] = Field(default_factory=list)
    latest_version_num: int = 0

class RepositoryBrainStorage:
    """
    Handles loading of the structured Semantic Capability Knowledge JSONs.
    Now uses the V2 Architecture: Domain -> Feature -> Capability graph.
    Supports Memory Evolution via Versioning.
    """
    def __init__(self):
        self.semantic_nodes: List[CapabilityGraphNode] = []
        self.history_map: Dict[str, CapabilityVersionHistory] = {}
        self.loaded_directory: Optional[Path] = None

    def record_successful_ticket(self, file_path: str, verification_level: str = "test_verified") -> None:
        """
        Record a successful ticket run against a specific file path, including verification level.

        This persists lightweight runtime learning metadata in a sidecar state
        file so workflow completion does not depend on future V2 brain-schema
        evolution work.
        """
        if not file_path:
            return

        if not self.loaded_directory:
            logger.warning(
                "Repository brain loaded_directory is unavailable; cannot persist successful ticket for %s",
                file_path,
            )
            return

        state_path = self.loaded_directory / "repository_brain_runtime.state"
        normalized_path = file_path.replace("\\", "/").strip()
        now = datetime.datetime.utcnow().isoformat() + "Z"

        state: Dict[str, object] = {"files": {}}
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
            except Exception as exc:
                logger.warning("Failed to read repository brain runtime state %s: %s", state_path, exc)

        files = state.setdefault("files", {})
        if not isinstance(files, dict):
            files = {}
            state["files"] = files

        entry = files.setdefault(
            normalized_path,
            {
                "original_path": file_path,
                "success_count": 0,
                "first_success_at": now,
                "last_success_at": now,
                "last_verification_level": verification_level,
                "verification_counts": {},
            },
        )
        if not isinstance(entry, dict):
            entry = {
                "original_path": file_path,
                "success_count": 0,
                "first_success_at": now,
                "last_success_at": now,
                "last_verification_level": verification_level,
                "verification_counts": {},
            }
            files[normalized_path] = entry

        entry["original_path"] = file_path
        entry["success_count"] = int(entry.get("success_count", 0)) + 1
        entry.setdefault("first_success_at", now)
        entry["last_success_at"] = now
        entry["last_verification_level"] = verification_level
        v_counts = entry.setdefault("verification_counts", {})
        v_counts[verification_level] = int(v_counts.get(verification_level, 0)) + 1

        try:
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            logger.info(
                "Recorded successful ticket for %s (count=%s, level=%s)",
                file_path,
                entry["success_count"],
                verification_level,
            )
        except Exception as exc:
            logger.warning("Failed to persist successful ticket state to %s: %s", state_path, exc)


    def load_default(self, workspace_path: Optional[str] = None) -> List[CapabilityGraphNode]:
        """Load the structured JSON files from the brain/knowledge directory."""
        candidates = []
        if workspace_path:
            candidates.append(Path(workspace_path) / "brain" / "knowledge")
            candidates.append(Path(workspace_path) / ".agents")
            candidates.append(Path(workspace_path) / "knowledge")
        
        env_brain = os.getenv("AVIATOR_BRAIN_DIR")
        if env_brain:
            candidates.append(Path(env_brain))
            
        for cand in candidates:
            if cand.exists() and cand.is_dir():
                return self.load_directory(cand)
                
        logger.warning(
            f"⚠️ Repository Brain directory not found. Searched: {[str(c) for c in candidates]}. "
            "This likely means `build_repository_brain` hasn't been run for this workspace yet. "
            "Falling back to Neo4j graph data natively."
        )
        return []

    def load_directory(self, dir_path: Path) -> List[CapabilityGraphNode]:
        self.semantic_nodes = []
        self.history_map = {}
        self.loaded_directory = dir_path
        skipped = 0

        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".json"):
                    full_path = Path(root) / file
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        # ── Schema 1: CapabilityVersionHistory ───────────────
                        if isinstance(data, dict) and "versions" in data:
                            history = CapabilityVersionHistory(**data)
                            self.history_map[history.capability_id] = history
                            if history.versions:
                                self.semantic_nodes.append(history.versions[-1])

                        # ── Schema 2: CapabilityGraphReport ──────────────────
                        elif isinstance(data, dict) and "capability_nodes" in data:
                            report = CapabilityGraphReport(**data)
                            self.semantic_nodes.extend(report.capability_nodes)

                        # ── Schema 3: List of items ───────────────────────────
                        elif isinstance(data, list):
                            for item in data:
                                node = self._try_load_item(item, str(full_path))
                                if node:
                                    self.semantic_nodes.append(node)

                        # ── Schema 4: Single dict ─────────────────────────────
                        elif isinstance(data, dict):
                            node = self._try_load_item(data, str(full_path))
                            if node:
                                self.semantic_nodes.append(node)

                    except Exception as e:
                        skipped += 1
                        print(f"Failed to load semantic JSON {full_path}: {e}")

        print(f"BRAIN_LOADED_DIRECTORY={self.loaded_directory}")
        print(f"BRAIN_SEMANTIC_NODES_COUNT={len(self.semantic_nodes)}  (skipped={skipped})")
        return self.semantic_nodes

    def _try_load_item(self, item: dict, source: str) -> "CapabilityGraphNode | None":
        """
        Try to load a dict as a CapabilityGraphNode.

        Handles two schemas:
        1. Full CapabilityGraphNode schema (has capability_id)
        2. Directory brain schema (has directory_path, technical_role, intent_terms)
           → mapped to CapabilityGraphNode fields so retrieval works

        Returns None if the item cannot be mapped to a useful node.
        """
        if not isinstance(item, dict):
            return None

        # Schema 1: already a full CapabilityGraphNode
        if "capability_id" in item:
            try:
                return CapabilityGraphNode(**item)
            except Exception:
                pass

        # Schema 2: directory brain format
        if "directory_path" in item:
            dir_path = item.get("directory_path", "")
            domain = item.get("business_domain", "Unknown")
            role = item.get("technical_role", "Unknown")
            resp = item.get("core_responsibilities", "")
            intent = item.get("intent_terms", [])
            artifacts = [str(p).strip() for p in (item.get("key_artifacts", []) or []) if str(p).strip()]

            # Build a synthetic capability_id from the path
            cap_id = dir_path.replace("/", ".").replace("\\", ".").strip(".")
            if not cap_id:
                return None

            try:
                return CapabilityGraphNode(
                    capability_id=cap_id,
                    business_domain=domain,
                    business_feature=role,
                    capability_name=f"{domain} / {role}",
                    purpose=resp or f"Directory: {dir_path}",
                    business_goal="",
                    business_rules=intent,
                    triggers=[],
                    state_modified=[],
                    side_effects=[],
                    depends_on=[],
                    uses=[],
                    publishes=[],
                    consumes=[],
                    affected_by=[],
                    business_concepts=list(dict.fromkeys(list(intent) + artifacts + [dir_path, role, domain])),
                    participating_files=[
                        FileParticipant(
                            file_path=artifact,
                            role="DIRECTORY_MEMBER",
                            change_surfaces=[],
                            artifact_type=Path(artifact).suffix.lstrip(".") or "file",
                        )
                        for artifact in artifacts[:40]
                    ],
                    confidence_sources=["directory_brain"],
                    version=1,
                )
            except Exception:
                return None

        # Unknown schema — skip silently
        return None

    def save_capability_nodes(self, nodes: List[CapabilityGraphNode], save_dir: Optional[Path] = None):
        """
        Saves capability nodes. If the capability already exists, it pushes a new version
        instead of overwriting it completely.
        """
        if not save_dir:
            save_dir = self.loaded_directory
        
        if not save_dir:
            print("No save directory specified or loaded.")
            return

        os.makedirs(save_dir, exist_ok=True)
        
        for node in nodes:
            cap_id = node.capability_id
            history = self.history_map.get(cap_id)
            
            if not history:
                # Load from disk just in case
                file_path = save_dir / f"{cap_id}.json"
                if file_path.exists():
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            if "versions" in data:
                                history = CapabilityVersionHistory(**data)
                    except Exception:
                        pass
                        
            if not history:
                history = CapabilityVersionHistory(capability_id=cap_id)
                self.history_map[cap_id] = history
                
            # Increment version
            history.latest_version_num += 1
            node.version = history.latest_version_num
            history.versions.append(node)
            
            # Save the history object
            file_path = save_dir / f"{cap_id}.json"
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    # using model_dump_json to serialize
                    f.write(history.model_dump_json(indent=2))
            except Exception as e:
                print(f"Failed to save {file_path}: {e}")

    def _normalize_text(self, text: str) -> str:
        if not text: return ""
        return text.lower().strip()

    def _tokenize(self, text: str) -> set:
        if not text: return set()
        words = re.split(r'[^a-zA-Z0-9]+', text)
        words = [w.lower() for w in words if len(w) > 2]
        return set(words)

    def find_nodes_by_semantic_intent(self, query_text: str) -> List[Dict[str, object]]:
        if not self.semantic_nodes:
            return []

        query_tokens = self._tokenize(self._normalize_text(query_text))
        if not query_tokens:
            return []

        matches = []
        for node in self.semantic_nodes:
            score = 0
            
            domain_tokens = self._tokenize(self._normalize_text(node.business_domain))
            feature_tokens = self._tokenize(self._normalize_text(node.business_feature))
            score += len(query_tokens.intersection(domain_tokens)) * 5
            score += len(query_tokens.intersection(feature_tokens)) * 5
            
            name_tokens = self._tokenize(self._normalize_text(node.capability_name))
            score += len(query_tokens.intersection(name_tokens)) * 10
            
            all_concepts = set()
            for concept in node.business_concepts:
                all_concepts.update(self._tokenize(self._normalize_text(concept)))
                
            score += len(query_tokens.intersection(all_concepts)) * 4
                    
            purpose_tokens = self._tokenize(self._normalize_text(node.purpose))
            score += len(query_tokens.intersection(purpose_tokens))

            if score > 0:
                matches.append({
                    "capability_name": node.capability_name,
                    "business_domain": node.business_domain,
                    "business_feature": node.business_feature,
                    "score": score,
                    "matched_reasons": f"Score: {score} based on intent/concepts overlap.",
                    "node": node
                })

        matches.sort(key=lambda x: -float(x["score"]))
        return matches[:15]
