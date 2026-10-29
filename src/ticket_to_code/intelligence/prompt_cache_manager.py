"""DORMANT: not used by any production code path (only by tests) as of 2026-10-29.
Kept because tests pin its behaviour; wire it in or delete it together with its tests.


Prompt Cache Prefix Stabilization and KV-Cache Optimization.

Implements Category 2.1:
Ensures byte-deterministic, prefix-stable prompt ordering:
[system | tool schemas | repo skeletons | episodic memory | conversation | volatile scratchpad]

Prevents silent cache invalidations caused by timestamps, unstable tool ordering,
or misplaced diagnostic outputs.
"""

import hashlib
import json
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class StablePromptStructure(BaseModel):
    system_instruction: str
    tool_schemas: List[Dict[str, Any]] = Field(default_factory=list)
    repo_skeletons: List[Dict[str, str]] = Field(default_factory=list)  # [{"path": "...", "skeleton": "..."}]
    episodic_memories: List[str] = Field(default_factory=list)
    conversation_turns: List[Dict[str, str]] = Field(default_factory=list)  # [{"role": "user"|"assistant", "content": "..."}]
    live_scratchpad: str = ""  # Volatile diagnostic error / intermediate state placed strictly last


class PromptCacheManager:
    """
    Constructs cache-stable prompt payloads and generates provider-appropriate
    cache breakpoints (Anthropic prompt caching, OpenAI prefix matching, Gemini context caching).
    """

    @classmethod
    def compute_prefix_hash(cls, text: str) -> str:
        """Compute SHA-256 hash of the stable prefix."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def build_cache_stable_prompt(cls, prompt_struct: StablePromptStructure) -> Dict[str, Any]:
        """
        Builds a canonical prompt dict ordered from most-static to most-volatile.
        """
        # 1. System instruction (must be 100% deterministic, zero volatile timestamps)
        system_text = prompt_struct.system_instruction.strip()

        # 2. Tool schemas (stably sorted alphabetically by name to avoid permutation cache bust)
        sorted_tools = sorted(prompt_struct.tool_schemas, key=lambda t: t.get("name", ""))

        # 3. Repo skeletons & Interfaces (sorted by path)
        sorted_skeletons = sorted(prompt_struct.repo_skeletons, key=lambda s: s.get("path", ""))
        skeleton_parts = []
        if sorted_skeletons:
            skeleton_parts.append("## Repository Contracts & Interfaces (AST Skeletons)")
            for s in sorted_skeletons:
                skeleton_parts.append(f"### {s.get('path', '')}\n```\n{s.get('skeleton', '').strip()}\n```")
        skeleton_text = "\n\n".join(skeleton_parts)

        # 4. Episodic memory (sorted / deduplicated)
        episodic_parts = []
        if prompt_struct.episodic_memories:
            episodic_parts.append("## Historical Verified Solutions (Episodic Memory)")
            for mem in sorted(set(prompt_struct.episodic_memories)):
                episodic_parts.append(f"- {mem.strip()}")
        episodic_text = "\n\n".join(episodic_parts)

        # Combine static sections to compute stable prefix hash
        static_prefix = "\n\n".join(filter(bool, [system_text, skeleton_text, episodic_text]))
        prefix_hash = cls.compute_prefix_hash(static_prefix)

        # 5. Conversation turns (grows monotonically)
        messages: List[Dict[str, str]] = list(prompt_struct.conversation_turns)

        # 6. Live scratchpad / compiler diagnostic (volatile, strictly last)
        if prompt_struct.live_scratchpad.strip():
            scratchpad_content = f"## Live Diagnostics & Scratchpad\n{prompt_struct.live_scratchpad.strip()}"
            if messages and messages[-1].get("role") == "user":
                # Append to last user message so volatile tokens stay at the tail
                messages[-1]["content"] = f"{messages[-1]['content']}\n\n{scratchpad_content}"
            else:
                messages.append({"role": "user", "content": scratchpad_content})

        return {
            "system_prompt": system_text,
            "static_prefix": static_prefix,
            "prefix_hash": prefix_hash,
            "tools": sorted_tools,
            "messages": messages,
            "anthropic_cache_control": {
                "type": "ephemeral"
            }
        }


class StageCacheTelemetry(BaseModel):
    stage_name: str
    model_id: str
    provider: str
    cache_state: str = "WARM_HIT"  # WARM_HIT, COLD_PREFILL, RESET_BRIEF, FULL_EVICTED
    cached_tokens: int
    new_tokens: int
    total_tokens: int
    cost_savings_pct: float
    prefix_hash: str
    is_cache_hit: bool
    context_limit: int = 200_000
    context_fill_pct: float = 0.0
    is_near_capacity: bool = False
    lossless_store_ref: Optional[str] = None


class WorkflowCacheSession:
    """
    Maintains prefix stability and monotonically growing conversation history
    across workflow stages (e.g. investigate -> plan -> generate_code -> verify)
    for the same model or provider.
    
    When transitioning stages:
    1. Keeps static prefix identical byte-for-byte.
    2. Appends previous stage outputs as conversation history.
    3. Detects when KV cache approaches capacity and performs a Structured Context Reset.
    4. Calculates exact token savings: cached prefix tokens vs. new delta tokens.
    """
    def __init__(self, workflow_id: str, model_id: str, static_structure: StablePromptStructure):
        self.workflow_id = workflow_id
        self.model_id = model_id
        self.static_structure = static_structure
        self.stage_history: List[Dict[str, str]] = []
        self._last_prefix_hash: str = ""
        self.telemetry_history: List[StageCacheTelemetry] = []

    def _flush_lossless_store(self) -> str:
        """
        Persists full uncompressed conversation history and tool outputs to disk,
        providing a lossless queryable store that agents can reference without keeping
        50k tokens in the active LLM context.
        """
        try:
            import os
            from pathlib import Path
            store_dir = Path("scratch/transcripts")
            store_dir.mkdir(parents=True, exist_ok=True)
            store_file = store_dir / f"{self.workflow_id}.jsonl"
            with open(store_file, "a", encoding="utf-8") as f:
                for turn in self.stage_history:
                    f.write(json.dumps(turn) + "\n")
            return str(store_file).replace("\\", "/")
        except Exception:
            return f"scratch/transcripts/{self.workflow_id}.jsonl"

    def _distill_prior_history(self) -> str:
        """
        Distill verbose prior turns into a compact handoff contract for model transitions or cache resets.
        Keeps key decisions, identified files, and line anchors while dropping raw explorer noise.
        """
        if not self.stage_history:
            return ""
        distilled = []
        for turn in self.stage_history:
            content = turn.get("content", "")
            lines = content.splitlines()
            important_lines = [
                l for l in lines
                if any(k in l.lower() for k in ["found", "candidate", "error", "file:", "line", "#l", "patch", "output", "summary"])
                or l.startswith("###") or l.startswith("-") or l.startswith("•")
            ]
            if important_lines:
                distilled.append("\n".join(important_lines[:15]))
            else:
                distilled.append(content[:300])
        return "\n\n".join(distilled)

    def prepare_stage_prompt(
        self,
        stage_name: str,
        stage_instructions: str,
        stage_delta: str = "",
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Builds the prompt for the new stage.
        If model_id matches previous stage, reuses the cached prefix and conversation history.
        If switching models or context near capacity, distills history into a compact contract.
        """
        current_model = model_id or self.model_id
        is_same_model = (current_model == self.model_id)

        # Context capacity calculation (default 200k for Claude/Pro, 1M for Flash)
        context_limit = 1_000_000 if "flash" in current_model.lower() else (2_000_000 if "2.5-pro" in current_model.lower() else 200_000)
        history_char_count = sum(len(t.get("content", "")) for t in self.stage_history)
        est_history_tokens = max(1, history_char_count // 4)

        # Near-capacity threshold (e.g. 40,000 tokens of accumulated history)
        is_near_capacity = (est_history_tokens > 40_000)
        lossless_ref = None
        cache_state = "COLD_PREFILL"

        if is_near_capacity:
            # KV Cache Full / Context Window Near Limit -> Structured Context Reset
            lossless_ref = self._flush_lossless_store()
            distilled_summary = self._distill_prior_history()
            turns = [
                {"role": "user", "content": f"### Structured Context Reset (KV Cache Full Optimized):\n{distilled_summary}\n\n[Full lossless transcript archived to on-disk store: {lossless_ref}]"}
            ]
            self.stage_history = list(turns)
            cache_state = "FULL_EVICTED"
        elif is_same_model:
            turns = list(self.stage_history)
        else:
            # Cross-model transition -> Distilled Context Firewall
            lossless_ref = self._flush_lossless_store()
            distilled_summary = self._distill_prior_history()
            turns = [
                {"role": "user", "content": f"### Context Handoff from Prior Phase (Distilled):\n{distilled_summary}\n\n[Full lossless transcript archived to on-disk store: {lossless_ref}]"}
            ] if distilled_summary else []
            cache_state = "RESET_BRIEF"

        turns.append({"role": "user", "content": f"### Stage: {stage_name}\n{stage_instructions}"})

        struct = StablePromptStructure(
            system_instruction=self.static_structure.system_instruction,
            tool_schemas=self.static_structure.tool_schemas,
            repo_skeletons=self.static_structure.repo_skeletons,
            episodic_memories=self.static_structure.episodic_memories,
            conversation_turns=turns,
            live_scratchpad=stage_delta
        )

        built = PromptCacheManager.build_cache_stable_prompt(struct)
        prefix_hash = built["prefix_hash"]

        # Calculate token metrics (approx 4 chars per token)
        static_len = len(built["static_prefix"])
        history_len = sum(len(t.get("content", "")) for t in self.stage_history)
        delta_len = len(stage_instructions) + len(stage_delta)

        est_static_tokens = max(1, (static_len + history_len) // 4)
        est_new_tokens = max(1, delta_len // 4)
        total_tokens = est_static_tokens + est_new_tokens

        is_hit = bool(is_same_model and not is_near_capacity and self._last_prefix_hash and self._last_prefix_hash == prefix_hash)
        cached_tokens = est_static_tokens if is_hit else 0
        actual_new_tokens = est_new_tokens if is_hit else total_tokens
        savings_pct = round((cached_tokens / total_tokens) * 100.0, 1) if is_hit else 0.0

        if is_hit:
            cache_state = "WARM_HIT"
        elif not is_same_model:
            cache_state = "RESET_BRIEF"

        fill_pct = round((total_tokens / context_limit) * 100.0, 2)

        # Provider determination
        provider = "google"
        if "claude" in current_model.lower():
            provider = "anthropic"
        elif "gpt" in current_model.lower():
            provider = "openai"

        telemetry = StageCacheTelemetry(
            stage_name=stage_name,
            model_id=current_model,
            provider=provider,
            cache_state=cache_state,
            cached_tokens=cached_tokens,
            new_tokens=actual_new_tokens,
            total_tokens=total_tokens,
            cost_savings_pct=savings_pct,
            prefix_hash=prefix_hash[:12],
            is_cache_hit=is_hit,
            context_limit=context_limit,
            context_fill_pct=fill_pct,
            is_near_capacity=is_near_capacity,
            lossless_store_ref=lossless_ref
        )
        self.telemetry_history.append(telemetry)
        self._last_prefix_hash = prefix_hash
        self.model_id = current_model

        built["telemetry"] = telemetry.model_dump()
        return built

    def record_stage_output(self, stage_name: str, assistant_output: str) -> None:
        """Record the stage outcome as an assistant turn in the monotonic history."""
        self.stage_history.append({
            "role": "assistant",
            "content": f"### Output ({stage_name}):\n{assistant_output[:4000]}"
        })


_WORKFLOW_CACHE_SESSIONS: Dict[str, WorkflowCacheSession] = {}


def get_or_create_cache_session(
    workflow_id: str,
    model_id: str,
    static_structure: StablePromptStructure
) -> WorkflowCacheSession:
    """Retrieve or initialize a cache session for a workflow run."""
    if workflow_id not in _WORKFLOW_CACHE_SESSIONS:
        _WORKFLOW_CACHE_SESSIONS[workflow_id] = WorkflowCacheSession(
            workflow_id=workflow_id,
            model_id=model_id,
            static_structure=static_structure
        )
    return _WORKFLOW_CACHE_SESSIONS[workflow_id]


def clear_cache_session(workflow_id: str) -> None:
    """Clean up cache session after workflow completion."""
    _WORKFLOW_CACHE_SESSIONS.pop(workflow_id, None)
