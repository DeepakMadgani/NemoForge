"""
Enterprise Feature Graph — the project's long-term, self-improving knowledge.

Separation of concerns (important):
  - SKILLS know FEATURES  (e.g. "application version")
  - FEATURE GRAPH knows FILES (owners, consumers) and learns them over time

A Feature record holds everything the reasoner needs to act on a class of work
WITHOUT hardcoding repo layout in code:

    feature            : canonical id (e.g. "application_version")
    aliases            : keywords that map a ticket to this feature
    owner_files        : learned canonical source files -> confidence
    consumer_files     : learned consumers/propagation targets -> confidence
    owner_hint_patterns: OVERRIDABLE bootstrap filename hints (data, not code).
                         Used only until real owners are learned; learning wins.
    validators         : outcome checks that must hold
    common_mistakes    : known failure patterns to avoid
    propagation_rules  : how consumers should derive from the owner
    past_tickets       : ids of tickets that touched this feature
    confidence         : how trustworthy this feature's owner map is

Because owners are LEARNED (and hints live in editable data, not in code), the
graph stays correct even if Enterprise's version/config process changes later.

Storage: single JSON document, co-located with the Enterprise brain.

Author: Deepak Madgani
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_IGNORE_DIRS = {
    ".git", "node_modules", "dist", "build", "target", "out", "trace",
    "logs", "coverage", ".venv", "venv", "__pycache__",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _keywords(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9_]{3,}", text or "")]


# Seeded feature DEFINITIONS (no absolute paths, no repo-specific knowledge).
# owner_hint_patterns are generic, editable bootstrap hints — learning overrides them.
_SEED_FEATURES: List[Dict[str, Any]] = [
    {
        "feature": "application_version",
        "aliases": ["version", "release", "bump", "app version", "application version"],
        "owner_files": {},
        "consumer_files": {},
        "owner_hint_patterns": [
            "application.yml", "application.yaml", "application.properties",
            "pom.xml", "build.gradle", "gradle.properties", "package.json",
            "version.txt", "VERSION",
        ],
        "validators": [
            "Old version literal no longer present in the canonical source.",
            "New version literal present in the canonical source.",
            "Consumers derive the version from the canonical source (not hardcoded).",
        ],
        "common_mistakes": [
            "Editing UI/display text directly instead of the canonical source.",
            "Leaving the old version string in some files.",
            "Blind search-and-replace without confirming the owner.",
        ],
        "propagation_rules": [
            "Consumers must read the version from the canonical source at build/runtime.",
        ],
        "past_tickets": [],
        "confidence": 0.3,
    },
    {
        "feature": "rest_api_endpoint",
        "aliases": ["api", "endpoint", "controller", "route", "swagger"],
        "owner_files": {},
        "consumer_files": {},
        "owner_hint_patterns": ["*Controller.*", "*controller*", "*routes*", "*api*"],
        "validators": [
            "Old endpoint removed.",
            "New endpoint present in controller/spec.",
            "Build succeeds.",
        ],
        "common_mistakes": ["Renaming the route but leaving old references/tests."],
        "propagation_rules": ["Update controller + spec/swagger + consumers together."],
        "past_tickets": [],
        "confidence": 0.3,
    },
    {
        "feature": "dynamic_form_component",
        "aliases": ["dynamic form", "register", "upload", "form", "validator", "file browser", "inline message", "uploaddocumentregister"],
        "owner_files": {},
        "consumer_files": {},
        "owner_hint_patterns": [
            "*Register*.json", "*register*.json", "*DynamicForm*",
            "*document-file-browser*.ts", "*document-file-browser*.html",
            "*custom-validator*.ts", "*upload-register*.component.ts", "*upload-register*.component.html"
        ],
        "validators": [
            "Validation rule added to form schema JSON (e.g. fileTypes: ['xlsx']).",
            "Custom form control correctly notifies FormControl with rejected/accepted value.",
            "Dynamic form displays .ot-error when validation fails and disables Next button.",
            "Translation keys added to all 8 locale files in assets/i18n.",
            "Frontend build passes with zero errors."
        ],
        "common_mistakes": [
            "Editing only the parent upload-register component instead of the underlying JSON schema and file-browser child control.",
            "Discarding invalid files silently with null instead of triggering form control validation.",
            "Hardcoding English error messages instead of using ngx-translate keys."
        ],
        "propagation_rules": [
            "Update schema JSON + child input component + custom validator directive + i18n bundles together."
        ],
        "past_tickets": [],
        "confidence": 0.5,
    },
    {
        "feature": "ui_component",
        "aliases": ["ui", "css", "style", "layout", "component", "header", "footer", "angular", "react", "jsx", "tsx", "frontend"],
        "owner_files": {},
        "consumer_files": {},
        "owner_hint_patterns": ["*.component.ts", "*.component.html", "*.scss", "*.css", "*.jsx", "*.tsx", "*.js"],
        "validators": ["Correct component/template edited.", "Frontend build or runtime verification succeeds."],
        "common_mistakes": ["Editing the wrong component; verifying with the wrong frontend stack assumptions."],
        "propagation_rules": ["Edit the owning frontend component/template/style."],
        "past_tickets": [],
        "confidence": 0.3,
    },
]


class FeatureGraph:
    """Feature-centric, self-improving knowledge store for Enterprise."""

    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.path = self._resolve_store_path(workspace_path)
        self.data: Dict[str, Any] = {"project": Path(workspace_path).name or "project", "features": {}, "updated_at": None}
        self._load()
        self._ensure_seeded()

    # ── storage ──────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_store_path(workspace_path: str) -> Path:
        ws = Path(workspace_path)
        candidates = [
            ws / ".nemoforge" / "brain" / "feature_graph.json",
            ws / ".aviator" / "brain" / "feature_graph.json",
            ws / "brain" / "project_feature_graph.json",
            ws / "brain" / "feature_graph.json",
            ws / "brain" / "cloudforge_feature_graph.json",
            ws / "project_feature_graph.json",
            ws / "cloudforge_feature_graph.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        if (ws / ".nemoforge").exists():
            return ws / ".nemoforge" / "brain" / "feature_graph.json"
        if (ws / ".aviator").exists():
            return ws / ".aviator" / "brain" / "feature_graph.json"
        if (ws / "brain").exists():
            return ws / "brain" / "project_feature_graph.json"
        return ws / ".nemoforge" / "brain" / "feature_graph.json"

    def _load(self) -> None:
        try:
            if self.path.exists():
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("features"), dict):
                    self.data.update(loaded)
                    logger.info("Enterprise Feature Graph loaded: %d features", len(self.data["features"]))
        except Exception as exc:
            logger.warning("Feature Graph load failed (%s) — starting fresh", exc)

    def _ensure_seeded(self) -> None:
        """Insert seed feature DEFINITIONS if missing (does not overwrite learned data)."""
        features = self.data.setdefault("features", {})
        changed = False
        for seed in _SEED_FEATURES:
            fid = seed["feature"]
            if fid not in features:
                features[fid] = json.loads(json.dumps(seed))  # deep copy
                changed = True
        if changed:
            self._save()

    def _save(self) -> None:
        self.data["updated_at"] = _now()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            logger.warning("Feature Graph save failed: %s", exc)

    # ── query ────────────────────────────────────────────────────────────────

    def match_feature(self, ticket_text: str, ticket_type: str = "") -> Optional[dict]:
        """Return the feature whose aliases best match the ticket (or None)."""
        text = (ticket_text or "").lower()
        tt = (ticket_type or "").lower()
        best: Optional[dict] = None
        best_score = 0
        for fid, feat in self.data.get("features", {}).items():
            score = 0
            for alias in feat.get("aliases", []):
                if alias in text:
                    score += 2
            if tt and (tt in fid or any(tt in a for a in feat.get("aliases", []))):
                score += 3
            if score > best_score:
                best_score = score
                best = feat
        return best if best_score > 0 else None

    def resolve_owner_candidates(self, feature: dict, workspace_path: Optional[str] = None) -> List[dict]:
        """Return owner file candidates: LEARNED owners first, then hint-pattern matches.

        Learned owners carry their real confidence. Hint-pattern matches are
        low-confidence bootstrap only and are superseded once real owners are learned.
        """
        out: List[dict] = []
        seen: set[str] = set()

        # 1) Learned owners (the durable, correct source).
        learned = feature.get("owner_files", {}) or {}
        for fp, conf in sorted(learned.items(), key=lambda kv: kv[1], reverse=True):
            out.append({"file": fp, "confidence": round(float(conf), 3), "source": "learned"})
            seen.add(fp)

        # 2) Bootstrap hints resolved against the real workspace (data-driven, overridable).
        ws = Path(workspace_path or self.workspace_path)
        patterns = feature.get("owner_hint_patterns", []) or []
        if ws.exists() and patterns:
            exact = {p for p in patterns if "*" not in p}
            globs = [p for p in patterns if "*" in p]
            try:
                for p in ws.rglob("*"):
                    try:
                        if not p.is_file() or any(part in _IGNORE_DIRS for part in p.parts):
                            continue
                        name = p.name
                        matched = name in exact or any(_glob_match(name, g) for g in globs)
                        if not matched:
                            continue
                        rel = str(p.relative_to(ws))
                        if rel in seen:
                            continue
                        seen.add(rel)
                        out.append({"file": rel, "confidence": 0.5, "source": "hint"})
                        if len(out) >= 12:
                            break
                    except Exception:
                        continue
            except Exception:
                pass
        return out

    def feature_summary(self, feature: dict) -> dict:
        return {
            "feature": feature.get("feature"),
            "validators": feature.get("validators", []),
            "common_mistakes": feature.get("common_mistakes", []),
            "propagation_rules": feature.get("propagation_rules", []),
            "confidence": feature.get("confidence", 0.0),
            "learned_owners": list((feature.get("owner_files") or {}).keys())[:8],
            "past_tickets": feature.get("past_tickets", [])[-5:],
        }

    # ── learning ─────────────────────────────────────────────────────────────

    def record_success(
        self,
        feature_id_or_text: str,
        ticket_id: str,
        owner_files: List[str],
        consumer_files: Optional[List[str]] = None,
        confidence: float = 0.8,
    ) -> None:
        """Reinforce a feature's owner/consumer map from a verified ticket."""
        features = self.data.setdefault("features", {})
        feat = features.get(feature_id_or_text)
        if feat is None:
            match = self.match_feature(feature_id_or_text)
            feat = match if match else self._create_feature(feature_id_or_text)
        # Reinforce owners.
        owners = feat.setdefault("owner_files", {})
        for fp in owner_files or []:
            owners[fp] = round(min(1.0, float(owners.get(fp, 0.0)) + 0.25), 3)
        consumers = feat.setdefault("consumer_files", {})
        for fp in consumer_files or []:
            consumers[fp] = round(min(1.0, float(consumers.get(fp, 0.0)) + 0.2), 3)
        past = feat.setdefault("past_tickets", [])
        if ticket_id and ticket_id not in past:
            past.append(ticket_id)
        feat["confidence"] = round(min(1.0, float(feat.get("confidence", 0.3)) + 0.1), 3)
        self._save()

    def _create_feature(self, feature_id: str) -> dict:
        fid = re.sub(r"[^a-z0-9_]+", "_", feature_id.lower()).strip("_") or "unknown_feature"
        feat = {
            "feature": fid,
            "aliases": [fid.replace("_", " ")],
            "owner_files": {},
            "consumer_files": {},
            "owner_hint_patterns": [],
            "validators": [],
            "common_mistakes": [],
            "propagation_rules": [],
            "past_tickets": [],
            "confidence": 0.3,
        }
        self.data.setdefault("features", {})[fid] = feat
        return feat

    def summary(self) -> dict:
        feats = self.data.get("features", {})
        return {
            "project": self.data.get("project", "Enterprise"),
            "feature_count": len(feats),
            "features": {
                fid: {
                    "owners": len(f.get("owner_files", {})),
                    "confidence": f.get("confidence", 0.0),
                    "tickets": len(f.get("past_tickets", [])),
                }
                for fid, f in feats.items()
            },
        }


def _glob_match(name: str, pattern: str) -> bool:
    """Tiny fnmatch-style matcher for '*' wildcard filename patterns."""
    import fnmatch
    return fnmatch.fnmatch(name.lower(), pattern.lower())
