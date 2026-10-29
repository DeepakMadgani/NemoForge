"""Experience store for Enterprise reasoning policy.

Separates dynamic experience from static knowledge:
- decision policies (what worked)
- failure memory (what failed)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperienceStore:
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.path = self._resolve_store_path(workspace_path)
        self.data: Dict[str, Any] = {
            "policies": {},
            "failures": [],
            "updated_at": None,
        }
        self._load()

    @staticmethod
    def _resolve_store_path(workspace_path: str) -> Path:
        ws = Path(workspace_path)
        candidates = [
            ws / ".aviator" / "brain" / "experience.json",
            ws / "brain" / "project_experience.json",
            ws / "brain" / "experience.json",
            ws / "brain" / "cloudforge_experience.json",
            ws / "project_experience.json",
            ws / "cloudforge_experience.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        target = ws / ".aviator" / "brain" / "experience.json"
        if (ws / ".aviator").exists():
            return target
        if (ws / "brain").exists():
            return ws / "brain" / "project_experience.json"
        return target

    def _load(self) -> None:
        try:
            if self.path.exists():
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data.update(loaded)
            else:
                self._save()
        except Exception:
            pass

    def _save(self) -> None:
        self.data["updated_at"] = _now()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def _state_signature(state: Dict[str, float]) -> str:
        keys = [
            "owner_candidates",
            "file_context",
            "patch",
            "build_success",
            "verification",
            "contradictions_open",
        ]
        return "|".join(f"{k}:{int(round(float(state.get(k, 0.0))))}" for k in keys)

    def prior_for_action(self, intent: str, need: str, action: str, state: Dict[str, float]) -> float:
        """Return action prior in [-0.4, 0.4]."""
        policies = self.data.get("policies", {}) or {}
        sig = self._state_signature(state)
        key = f"{intent}::{need}::{sig}::{action}"
        row = policies.get(key) or {}
        wins = int(row.get("wins", 0))
        losses = int(row.get("losses", 0))

        # Beta-smoothed utility prior.
        success_rate = (wins + 1.0) / (wins + losses + 2.0)
        prior = (success_rate - 0.5) * 0.8

        # Failure memory penalty for repeated bad moves in same context.
        penalty = 0.0
        for item in self.data.get("failures", [])[-200:]:
            if (
                str(item.get("intent", "")) == intent
                and str(item.get("need", "")) == need
                and str(item.get("state_signature", "")) == sig
                and str(item.get("action", "")) == action
            ):
                penalty += 0.03

        out = max(-0.4, min(0.4, prior - min(0.25, penalty)))
        return round(out, 4)

    def record_run(
        self,
        *,
        intent: str,
        steps: List[Dict[str, Any]],
        success: bool,
        final_need: str = "",
        final_state: Dict[str, float] | None = None,
    ) -> None:
        policies = self.data.setdefault("policies", {})
        failures = self.data.setdefault("failures", [])

        for st in steps:
            action = str(st.get("action", "")).strip()
            if not action or action == "final":
                continue
            need = str(st.get("need", final_need or "unknown"))
            state_sig = str(st.get("state_signature", "unknown"))
            key = f"{intent}::{need}::{state_sig}::{action}"
            row = policies.setdefault(key, {"wins": 0, "losses": 0, "last_seen": None})

            # Reward actions in successful runs except explicit rejected/unknown actions.
            bad_step = bool(st.get("error")) or bool(st.get("rejected"))
            if success and not bad_step:
                row["wins"] = int(row.get("wins", 0)) + 1
            else:
                row["losses"] = int(row.get("losses", 0)) + 1
                failures.append(
                    {
                        "intent": intent,
                        "need": need,
                        "state_signature": state_sig,
                        "action": action,
                        "reason": st.get("error") or st.get("rejected") or "unsuccessful_run",
                        "timestamp": _now(),
                    }
                )
            row["last_seen"] = _now()

        # Trim retained failure memory.
        if len(failures) > 4000:
            self.data["failures"] = failures[-4000:]

        self._save()
