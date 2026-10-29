"""
Causal Chain Reasoning Engine — Enhancement 3

Builds multi-hop causal chains from symptoms to root causes.
Each hop is validated against indexed code (SQLite/Neo4j).

For bug tickets, switches from breadth-first to depth-first search,
following the chain: symptom → immediate cause → deeper cause → root cause.

Author: Deepak Madgani
Date: July 2026
"""

import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# MODELS
# ============================================================================

class CausalHop(BaseModel):
    """One hop in a causal chain."""
    step: int = Field(..., description="Hop number (1 = symptom)")
    description: str = Field(..., description="What happens at this step")
    file_path: Optional[str] = Field(None, description="File involved")
    line_range: Optional[str] = Field(None, description="Lines e.g. '84-92'")
    function_name: Optional[str] = Field(None, description="Function involved")
    evidence: str = Field("", description="Why we believe this hop")
    verified: bool = Field(False, description="Whether verified against code index")


class CausalChain(BaseModel):
    """A complete causal chain from symptom to root cause."""
    chain_id: str = Field("")
    symptom: str = Field("", description="The observed problem")
    root_cause: str = Field("", description="The determined root cause")
    hops: List[CausalHop] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reasoning: str = Field("", description="LLM reasoning behind this chain")


class CausalAnalysis(BaseModel):
    """Complete causal analysis result."""
    has_causal_data: bool = Field(False)
    chains: List[CausalChain] = Field(default_factory=list)
    primary_chain: Optional[CausalChain] = Field(None, description="Most likely chain")
    suggested_files: List[str] = Field(
        default_factory=list,
        description="Files to investigate based on causal analysis"
    )


class CausalChainEngine:
    """
    Builds multi-hop causal chains from symptoms to root causes.

    Uses LLM to hypothesize chains, then validates each hop
    against the code index.
    """

    MAX_CHAIN_DEPTH = 5  # Maximum hops in a causal chain
    MAX_CHAINS = 3       # Maximum chains to explore

    def __init__(self, workspace_path: str = ""):
        self.workspace_path = workspace_path

    # ------------------------------------------------------------------
    # 1. Build causal chains via LLM
    # ------------------------------------------------------------------

    def build_chains(
        self,
        symptom: str,
        investigation_result: Any = None,
        runtime_diagnosis: Any = None,
        discovered_files: Optional[List[dict]] = None,
    ) -> CausalAnalysis:
        """
        Build causal chains from a symptom description.

        Args:
            symptom: Description of the observed problem.
            investigation_result: Result from investigation_node.
            runtime_diagnosis: Result from runtime_diagnosis_node (Enhancement 1).
            discovered_files: Files discovered by the discovery pipeline.

        Returns:
            CausalAnalysis with chains and suggested files.
        """
        # Gather context
        context_parts = [f"Symptom: {symptom}"]

        if investigation_result:
            root_cause = getattr(investigation_result, "root_cause_hypothesis", "")
            ticket_type = getattr(investigation_result, "ticket_type", None)
            if root_cause:
                context_parts.append(f"Root cause hypothesis: {root_cause}")
            if ticket_type:
                context_parts.append(f"Ticket type: {getattr(ticket_type, 'value', str(ticket_type))}")
            areas = getattr(investigation_result, "investigation_areas", [])
            if areas:
                context_parts.append(f"Investigation areas: {', '.join(areas)}")

        if runtime_diagnosis and getattr(runtime_diagnosis, "has_runtime_data", False):
            context_parts.append(f"Runtime error: {getattr(runtime_diagnosis, 'error_type', 'Unknown')}")
            context_parts.append(f"Error message: {getattr(runtime_diagnosis, 'error_message', '')}")
            context_parts.append(f"Root file: {getattr(runtime_diagnosis, 'root_file', '')}:{getattr(runtime_diagnosis, 'root_line', '')}")

        if discovered_files:
            file_list = ", ".join(d.get("path", "") for d in discovered_files[:10])
            context_parts.append(f"Discovered files: {file_list}")

        context = "\n".join(context_parts)

        # Use LLM to build causal chains
        try:
            return self._llm_build_chains(context, symptom)
        except Exception as exc:
            logger.warning(f"CausalChainEngine: LLM chain building failed: {exc}")
            return self._fallback_chain(symptom, investigation_result, runtime_diagnosis)

    def _llm_build_chains(self, context: str, symptom: str) -> CausalAnalysis:
        """Build chains using LLM."""
        from ticket_to_code.llm_utils import llm_invoke
        from langchain_core.messages import SystemMessage, HumanMessage
        import json

        prompt = f"""Analyze this software problem and build causal chains from symptom to root cause.

{context}

For each chain, provide a sequence of hops:
- Hop 1: The observed symptom
- Hop 2-N: Each intermediate cause
- Final hop: The root cause

Respond with JSON:
{{
  "chains": [
    {{
      "symptom": "What was observed",
      "root_cause": "The actual root cause",
      "hops": [
        {{
          "step": 1,
          "description": "What happens",
          "file_path": "path/to/file if known",
          "function_name": "function if known",
          "evidence": "Why this hop is likely"
        }}
      ],
      "confidence": 0.0-1.0,
      "reasoning": "Why this chain is the most likely explanation"
    }}
  ]
}}

Build up to {self.MAX_CHAINS} chains, ordered by likelihood. Each chain should have {self.MAX_CHAIN_DEPTH} hops max.
Be specific — reference actual file paths and functions when available from the context."""

        result = llm_invoke(
            messages=[
                SystemMessage(content="You are a senior software engineer performing root cause analysis."),
                HumanMessage(content=prompt),
            ],
            label="causal_chain",
        )

        if not result or not result.content:
            return CausalAnalysis(has_causal_data=False)

        raw = result.content.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("CausalChainEngine: JSON parse failed")
            return CausalAnalysis(has_causal_data=False)

        chains = []
        for i, c in enumerate(data.get("chains", [])[:self.MAX_CHAINS]):
            hops = [
                CausalHop(
                    step=h.get("step", j + 1),
                    description=h.get("description", ""),
                    file_path=h.get("file_path"),
                    line_range=h.get("line_range"),
                    function_name=h.get("function_name"),
                    evidence=h.get("evidence", ""),
                )
                for j, h in enumerate(c.get("hops", []))
            ]
            chains.append(CausalChain(
                chain_id=f"chain_{i}",
                symptom=c.get("symptom", symptom),
                root_cause=c.get("root_cause", ""),
                hops=hops,
                confidence=float(c.get("confidence", 0.5)),
                reasoning=c.get("reasoning", ""),
            ))

        # Suggested files from all chains
        suggested = []
        for chain in chains:
            for hop in chain.hops:
                if hop.file_path and hop.file_path not in suggested:
                    suggested.append(hop.file_path)

        primary = chains[0] if chains else None

        return CausalAnalysis(
            has_causal_data=bool(chains),
            chains=chains,
            primary_chain=primary,
            suggested_files=suggested,
        )

    def _fallback_chain(
        self, symptom: str, investigation: Any, runtime: Any
    ) -> CausalAnalysis:
        """Build a simple chain without LLM as fallback."""
        hops = [
            CausalHop(step=1, description=f"Symptom observed: {symptom}"),
        ]

        root_cause = "Unknown"
        if investigation:
            rc = getattr(investigation, "root_cause_hypothesis", "")
            if rc:
                root_cause = rc
                hops.append(CausalHop(
                    step=2,
                    description=f"Investigation hypothesis: {rc}",
                    evidence="From investigation agent",
                ))

        if runtime and getattr(runtime, "has_runtime_data", False):
            hops.append(CausalHop(
                step=len(hops) + 1,
                description=f"Runtime error: {getattr(runtime, 'error_type', '')} in {getattr(runtime, 'root_file', '')}",
                file_path=getattr(runtime, "root_file", None),
                evidence="From runtime log analysis",
            ))

        chain = CausalChain(
            chain_id="fallback_chain",
            symptom=symptom,
            root_cause=root_cause,
            hops=hops,
            confidence=0.3,
            reasoning="Fallback chain — LLM analysis unavailable",
        )

        suggested = [h.file_path for h in hops if h.file_path]

        return CausalAnalysis(
            has_causal_data=True,
            chains=[chain],
            primary_chain=chain,
            suggested_files=suggested,
        )

    # ------------------------------------------------------------------
    # 2. Verify chain hops against code index
    # ------------------------------------------------------------------

    def verify_chain(
        self, chain: CausalChain, sqlite_store=None
    ) -> CausalChain:
        """
        Verify each hop in a causal chain against the code index.
        """
        if not sqlite_store:
            return chain

        try:
            indexed_files = sqlite_store.get_all_files()
            indexed_basenames = {
                Path(f).name.lower(): f for f in indexed_files
            } if indexed_files else {}
        except Exception:
            return chain

        from pathlib import Path as _Path

        for hop in chain.hops:
            if hop.file_path:
                basename = _Path(hop.file_path).name.lower()
                if basename in indexed_basenames:
                    hop.verified = True
                    hop.file_path = indexed_basenames[basename]

        # Adjust confidence based on verification
        verified_count = sum(1 for h in chain.hops if h.verified)
        total_with_files = sum(1 for h in chain.hops if h.file_path)
        if total_with_files > 0:
            chain.confidence = min(
                chain.confidence + 0.1 * (verified_count / total_with_files),
                1.0,
            )

        return chain
