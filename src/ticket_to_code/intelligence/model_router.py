"""
Model Routing and Capability Registry for Ticket-to-Code.

Supports 3 Model Modes:
1. 'default': Uses the user-specified or configured primary model throughout the task.
2. 'auto_stage': Dynamically routes each pipeline stage (node) to the optimal model
   (e.g., FAST for discovery, DEEP for surgical patching, FAST for summarization).
3. 'auto_model': Automatically selects ONE best model for the entire task based on ticket
   intake analysis (complexity, technology, companion needs), avoiding cache invalidation.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from pydantic import BaseModel, Field
from pathlib import Path
import os
import re
import logging

logger = logging.getLogger(__name__)


class ModelMode(str, Enum):
    DEFAULT = "default"
    AUTO_STAGE = "auto_stage"
    AUTO_MODEL = "auto_model"


class ModelTier(str, Enum):
    FAST = "fast"
    MID = "mid"
    DEEP = "deep"


class ModelDescriptor(BaseModel):
    model_id: str
    name: str
    provider: str  # "anthropic", "openai", "google"
    tier: ModelTier
    max_context: int
    supports_parallel_tools: bool = True
    tool_dialect: str = "anthropic"  # "anthropic", "openai", "gemini"
    diff_format: str = "context_anchor_3line"  # "context_anchor_3line", "unified_diff"
    description: str = ""
    recommended_for: List[str] = Field(default_factory=list)
    is_available: bool = True
    status_reason: str = "Verified active on live provider endpoint"


AVAILABLE_MODELS: Dict[str, ModelDescriptor] = {
    "nebius/nvidia/Llama-3.1-Nemotron-70B-Instruct": ModelDescriptor(
        model_id="nebius/nvidia/Llama-3.1-Nemotron-70B-Instruct",
        name="NVIDIA Llama 3.1 Nemotron 70B (Nebius)",
        provider="nebius",
        tier=ModelTier.MID,
        max_context=128_000,
        supports_parallel_tools=True,
        tool_dialect="openai",
        diff_format="context_anchor_3line",
        description="Deprecated/Removed 70B parameter model. Pruned from active routing.",
        recommended_for=[],
        is_available=False,
        status_reason="Defunct / 404: Not hosted on live provider endpoint"
    ),
    "nebius/nvidia/nemotron-3-super-120b-a12b": ModelDescriptor(
        model_id="nebius/nvidia/nemotron-3-super-120b-a12b",
        name="NVIDIA Nemotron 3 Super 120B (Nebius)",
        provider="nebius",
        tier=ModelTier.DEEP,
        max_context=128_000,
        supports_parallel_tools=True,
        tool_dialect="openai",
        diff_format="context_anchor_3line",
        description="Flagship NVIDIA 120B hybrid MoE reasoning model on Nebius Token Factory. Specialized for multi-file code synthesis and AST patching.",
        recommended_for=["code_generation", "surgical_diffs", "deep_architecture", "planning"],
        is_available=True
    ),
    "nebius/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": ModelDescriptor(
        model_id="nebius/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
        name="NVIDIA Nemotron 3 Nano 30B (Nebius)",
        provider="nebius",
        tier=ModelTier.FAST,
        max_context=128_000,
        supports_parallel_tools=True,
        tool_dialect="openai",
        diff_format="context_anchor_3line",
        description="NVIDIA 30B instruction model on Nebius Token Factory. Fast reasoning, ticket localization, and unit test generation.",
        recommended_for=["discovery", "localization", "unit_tests", "code_generation"],
        is_available=True
    ),
    "nebius/nvidia/Nemotron-3_5-Lightning": ModelDescriptor(
        model_id="nebius/nvidia/Nemotron-3_5-Lightning",
        name="NVIDIA Nemotron 3.5 Lightning (Nebius)",
        provider="nebius",
        tier=ModelTier.FAST,
        max_context=128_000,
        supports_parallel_tools=True,
        tool_dialect="openai",
        diff_format="unified_diff",
        description="High-throughput, sub-second latency model on Nebius Token Factory. Ideal for fast AST discovery, log triage, and documentation tasks.",
        recommended_for=["discovery", "intake_classification", "log_analysis", "documentation"],
        is_available=True
    ),
    "nebius/nvidia/Nemotron-3-Ultra-550b-a55b": ModelDescriptor(
        model_id="nebius/nvidia/Nemotron-3-Ultra-550b-a55b",
        name="NVIDIA Nemotron 3 Ultra 550B (Nebius)",
        provider="nebius",
        tier=ModelTier.DEEP,
        max_context=128_000,
        supports_parallel_tools=True,
        tool_dialect="openai",
        diff_format="context_anchor_3line",
        description="Ultra-deep reasoning 550B parameter model on Nebius Token Factory for complex multi-service architecture analysis.",
        recommended_for=["deep_architecture", "cross_service", "planning"],
        is_available=True
    ),
    "gemini-2.5-pro": ModelDescriptor(
        model_id="gemini-2.5-pro",
        name="Gemini 2.5 Pro",
        provider="google",
        tier=ModelTier.DEEP,
        max_context=2_000_000,
        supports_parallel_tools=True,
        tool_dialect="gemini",
        diff_format="context_anchor_3line",
        description="Flagship 2M context deep reasoning & code synthesis model. Excels at monorepo architecture and surgical fixes.",
        recommended_for=["deep_architecture", "code_generation", "error_resolution", "planning"],
        is_available=True
    ),
    "gemini-2.5-flash": ModelDescriptor(
        model_id="gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        provider="google",
        tier=ModelTier.FAST,
        max_context=1_000_000,
        supports_parallel_tools=True,
        tool_dialect="gemini",
        diff_format="unified_diff",
        description="Sub-second low latency with 1M context. Ideal for intake classification, AST search, and build log triage.",
        recommended_for=["discovery", "intake_classification", "log_analysis", "summarization"],
        is_available=True
    ),
    "gemini-2.0-flash": ModelDescriptor(
        model_id="gemini-2.0-flash",
        name="Gemini 2.0 Flash",
        provider="google",
        tier=ModelTier.FAST,
        max_context=1_000_000,
        supports_parallel_tools=True,
        tool_dialect="gemini",
        diff_format="unified_diff",
        description="Low latency with 1M context. Excellent for fast discovery and logs.",
        recommended_for=["discovery", "log_analysis"],
        is_available=True
    ),
    "gemini-1.5-pro": ModelDescriptor(
        model_id="gemini-1.5-pro",
        name="Gemini 1.5 Pro",
        provider="google",
        tier=ModelTier.DEEP,
        max_context=2_000_000,
        supports_parallel_tools=True,
        tool_dialect="gemini",
        diff_format="unified_diff",
        description="Ultra-long context model (2M tokens). Exceptional for large monorepo sweeps and legacy codebase indexing.",
        recommended_for=["monorepo_audit", "long_context_rag"],
        is_available=True
    ),
    "claude-3-7-sonnet": ModelDescriptor(
        model_id="claude-3-7-sonnet",
        name="Claude 3.7 Sonnet",
        provider="anthropic",
        tier=ModelTier.DEEP,
        max_context=200_000,
        supports_parallel_tools=True,
        tool_dialect="anthropic",
        diff_format="context_anchor_3line",
        description="Hybrid reasoning model. Optimal for complex multi-file architectural refactoring and Angular companion sets.",
        recommended_for=["angular_companions", "multi_file_refactor", "complex_bugfix", "planning"],
        is_available=False
    ),
    "claude-3-5-sonnet": ModelDescriptor(
        model_id="claude-3-5-sonnet",
        name="Claude 3.5 Sonnet",
        provider="anthropic",
        tier=ModelTier.DEEP,
        max_context=200_000,
        supports_parallel_tools=True,
        tool_dialect="anthropic",
        diff_format="context_anchor_3line",
        description="High-precision code synthesis model. Gold-standard for fail-closed surgical 3-line anchor diffs.",
        recommended_for=["surgical_diffs", "code_generation", "error_resolution"],
        is_available=False
    ),
    "gpt-4o": ModelDescriptor(
        model_id="gpt-4o",
        name="GPT-4o",
        provider="openai",
        tier=ModelTier.DEEP,
        max_context=128_000,
        supports_parallel_tools=True,
        tool_dialect="openai",
        diff_format="context_anchor_3line",
        description="Multimodal flagship model. Strong in Java Spring Boot backend microservices and polyglot dependency graphs.",
        recommended_for=["java_backend", "cross_service_contracts", "general_code"],
        is_available=False
    ),
    "gpt-4o-mini": ModelDescriptor(
        model_id="gpt-4o-mini",
        name="GPT-4o Mini",
        provider="openai",
        tier=ModelTier.MID,
        max_context=128_000,
        supports_parallel_tools=True,
        tool_dialect="openai",
        diff_format="context_anchor_3line",
        description="Cost-effective, low-latency model. Suited for single-file mechanical modifications and unit test updates.",
        recommended_for=["single_file_edit", "unit_tests", "simple_scripting"],
        is_available=False
    ),
}


def _load_env_vars() -> Dict[str, str]:
    """Load environment variables, merging .env.local if present."""
    env = dict(os.environ)
    for p in [
        Path(__file__).resolve().parents[3] / "chatbot" / ".env.local",
        Path.cwd() / "chatbot" / ".env.local",
        Path.cwd() / ".env.local",
    ]:
        if p.exists():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip()
                        if k not in env:
                            env[k] = v
            except Exception:
                pass
            break
    return env


def get_provider_credentials() -> Dict[str, Dict[str, Any]]:
    """Generic resolver for all configured LLM providers."""
    env = _load_env_vars()
    creds: Dict[str, Dict[str, Any]] = {}

    # Nebius Token Factory
    nebius_key = env.get("NEBIUS_API_KEY", "").strip()
    if nebius_key and len(nebius_key) > 8:
        creds["nebius"] = {
            "api_key": nebius_key,
            "base_url": env.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1").rstrip("/"),
            "protocol": "openai",
            "name": "Nebius Token Factory"
        }

    # OpenAI
    openai_key = env.get("OPENAI_API_KEY", "").strip()
    if openai_key and len(openai_key) > 8:
        creds["openai"] = {
            "api_key": openai_key,
            "base_url": env.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            "protocol": "openai",
            "name": "OpenAI"
        }

    # Google Gemini
    gemini_key = env.get("GEMINI_API_KEY", "").strip() or env.get("GOOGLE_API_KEY", "").strip()
    if gemini_key and len(gemini_key) > 8:
        creds["google"] = {
            "api_key": gemini_key,
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "protocol": "gemini",
            "name": "Google Gemini"
        }

    # Anthropic
    anthropic_key = env.get("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key and len(anthropic_key) > 8:
        creds["anthropic"] = {
            "api_key": anthropic_key,
            "base_url": "https://api.anthropic.com/v1",
            "protocol": "anthropic",
            "name": "Anthropic Claude"
        }

    # Generic dynamic discovery of custom providers via *_API_KEY
    for k, v in env.items():
        if k.endswith("_API_KEY") and v.strip() and len(v.strip()) > 8:
            prov_id = k[:-8].lower()
            if prov_id not in creds and prov_id not in ["gemini"]:
                custom_url = env.get(f"{prov_id.upper()}_BASE_URL", "").rstrip("/")
                if not custom_url:
                    if prov_id == "groq":
                        custom_url = "https://api.groq.com/openai/v1"
                    elif prov_id == "together":
                        custom_url = "https://api.together.xyz/v1"
                    elif prov_id == "openrouter":
                        custom_url = "https://openrouter.ai/api/v1"
                    elif prov_id == "deepseek":
                        custom_url = "https://api.deepseek.com/v1"
                    elif prov_id == "ollama":
                        custom_url = "http://localhost:11434/v1"

                if custom_url:
                    creds[prov_id] = {
                        "api_key": v.strip(),
                        "base_url": custom_url,
                        "protocol": "openai",
                        "name": env.get(f"{prov_id.upper()}_NAME", prov_id.capitalize())
                    }

    return creds


def detect_configured_providers() -> Dict[str, Any]:
    """Detect which LLM provider credentials are configured in the current environment."""
    creds = get_provider_credentials()
    result = {p_id: True for p_id in creds.keys()}
    result["_active_provider"] = os.getenv("LLM_PROVIDER", "nebius").strip().lower()
    return result


_PROVIDER_MODELS_CACHE: Dict[str, Tuple[float, List[str]]] = {}


def discover_provider_models(provider_id: str, creds: Dict[str, Any]) -> List[str]:
    """Universally discover active model IDs from any LLM provider endpoint."""
    global _PROVIDER_MODELS_CACHE
    import time
    now = time.time()
    if provider_id in _PROVIDER_MODELS_CACHE:
        cached_time, cached_list = _PROVIDER_MODELS_CACHE[provider_id]
        if now - cached_time < 300:
            return cached_list

    api_key = creds.get("api_key", "").strip()
    base_url = creds.get("base_url", "").rstrip("/")
    protocol = creds.get("protocol", "openai")
    if not api_key or not base_url:
        return []

    try:
        import httpx
        with httpx.Client(timeout=4.0) as client:
            if protocol == "openai":
                resp = client.get(f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"})
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    raw_ids = [m["id"] for m in data if "id" in m]
                    if provider_id == "nebius":
                        flagship = "nvidia/nemotron-3-super-120b-a12b"
                        if flagship in raw_ids:
                            raw_ids.remove(flagship)
                            raw_ids.insert(0, flagship)
                    _PROVIDER_MODELS_CACHE[provider_id] = (now, raw_ids)
                    return raw_ids
            elif protocol == "gemini":
                resp = client.get(f"{base_url}/models?key={api_key}")
                if resp.status_code == 200:
                    data = resp.json().get("models", [])
                    raw_ids = [m.get("name", "").split("/")[-1] for m in data if "name" in m]
                    _PROVIDER_MODELS_CACHE[provider_id] = (now, raw_ids)
                    return raw_ids
            elif protocol == "anthropic":
                resp = client.get(f"{base_url}/models", headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"})
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    raw_ids = [m["id"] for m in data if "id" in m]
                    _PROVIDER_MODELS_CACHE[provider_id] = (now, raw_ids)
                    return raw_ids
    except Exception as e:
        logger.debug(f"Dynamic model discovery skipped for {provider_id}: {e}")

    return []


def refresh_available_models() -> Dict[str, ModelDescriptor]:
    """
    Universal Capability Intersection across ALL providers:
    1. Intersects configured credentials against live provider /models catalog.
    2. Prunes any model from ANY provider that is not present on its live endpoint.
    3. Dynamically registers new models discovered on the live endpoint.
    """
    all_creds = get_provider_credentials()
    providers = detect_configured_providers()

    # 1. Query live models across all configured providers
    discovered_by_provider: Dict[str, List[str]] = {}
    for prov_id, creds in all_creds.items():
        models = discover_provider_models(prov_id, creds)
        if models:
            discovered_by_provider[prov_id] = models

    # 2. Generic dynamic auto-registration for any newly exposed models
    for prov_id, live_models in discovered_by_provider.items():
        proto = all_creds[prov_id].get("protocol", "openai")
        for m_id in live_models:
            full_id = f"{prov_id}/{m_id}" if not m_id.startswith(f"{prov_id}/") else m_id
            if full_id not in AVAILABLE_MODELS and m_id not in AVAILABLE_MODELS:
                clean_name = m_id.split("/")[-1]
                lowered = m_id.lower()
                tier = (
                    ModelTier.DEEP
                    if any(w in lowered for w in ["super", "ultra", "pro", "opus", "sonnet", "120b", "405b", "235b", "r1"])
                    else (
                        ModelTier.FAST
                        if any(w in lowered for w in ["flash", "nano", "mini", "lightning", "haiku"])
                        else ModelTier.MID
                    )
                )
                AVAILABLE_MODELS[full_id] = ModelDescriptor(
                    model_id=full_id,
                    name=f"{all_creds[prov_id].get('name', prov_id.capitalize())} {clean_name}",
                    provider=prov_id,
                    tier=tier,
                    max_context=128_000,
                    supports_parallel_tools=True,
                    tool_dialect="gemini" if proto == "gemini" else "openai",
                    diff_format="context_anchor_3line",
                    description=f"Active model on {prov_id}: {m_id}",
                    recommended_for=["code_generation", "ast_patching"],
                    is_available=True
                )

    # 3. Universal capability intersection across ALL providers (zero hardcoding)
    explicit_allow = os.getenv("NEMOFORGE_ALLOWED_MODELS", os.getenv("AVIATOR_ALLOWED_MODELS", ""))
    allowed_set = {m.strip() for m in explicit_allow.split(",") if m.strip()} if explicit_allow else None

    for m_id, desc in list(AVAILABLE_MODELS.items()):
        prov_has_creds = providers.get(desc.provider, False)
        if not prov_has_creds:
            desc.is_available = False
            desc.status_reason = f"Provider credentials missing ({desc.provider.upper()}_API_KEY not set)"
            continue

        # If this provider returned a live catalog, enforce live availability
        if desc.provider in discovered_by_provider:
            prov_catalog = discovered_by_provider[desc.provider]
            catalog_set = set(prov_catalog) | {f"{desc.provider}/{x}" for x in prov_catalog}
            norm_id = m_id[len(desc.provider) + 1:] if m_id.startswith(f"{desc.provider}/") else m_id
            if norm_id not in catalog_set and m_id not in catalog_set:
                desc.is_available = False
                desc.status_reason = f"Defunct on {desc.provider}: Not returned by live /models endpoint (pruned from auto-routing)"
                continue

        if allowed_set is not None:
            desc.is_available = (m_id in allowed_set) and prov_has_creds
            desc.status_reason = "Allowed by policy & verified active" if desc.is_available else "Excluded by model whitelist policy"
        else:
            desc.is_available = prov_has_creds
            desc.status_reason = f"Verified active on {desc.provider}" if desc.is_available else "Unavailable"

    return AVAILABLE_MODELS


# Initial discovery on load
refresh_available_models()


def get_default_model_id() -> str:
    """Resolve the default model ID based on environment configuration."""
    env_default = os.getenv("NEMOFORGE_DEFAULT_MODEL", os.getenv("AVIATOR_DEFAULT_MODEL", "")).strip()
    if env_default and env_default in AVAILABLE_MODELS and AVAILABLE_MODELS[env_default].is_available:
        return env_default

    env_model = os.getenv("LLM_MODEL", "").strip()
    if env_model and env_model in AVAILABLE_MODELS and AVAILABLE_MODELS[env_model].is_available:
        return env_model

    # Flagship: NVIDIA Nemotron 3 Super on Nebius Token Factory
    flagship_nebius = "nebius/nvidia/nemotron-3-super-120b-a12b"
    if flagship_nebius in AVAILABLE_MODELS and AVAILABLE_MODELS[flagship_nebius].is_available:
        return flagship_nebius

    # Fallback to any available deep reasoning model
    for m_id, desc in AVAILABLE_MODELS.items():
        if desc.is_available and desc.tier == ModelTier.DEEP:
            return m_id

    # Fallback to any available model
    for m_id, desc in AVAILABLE_MODELS.items():
        if desc.is_available:
            return m_id

    return flagship_nebius


DEFAULT_MODEL_ID = get_default_model_id()


class ModelRouter:
    """
    Orchestrates model selection across the 3 execution modes, strictly adhering
    to models that are actually installed and configured in the environment.
    """

    @classmethod
    def get_available_models(cls) -> Dict[str, ModelDescriptor]:
        """Return all models marked available in the current environment."""
        refresh_available_models()
        avail = {k: v for k, v in AVAILABLE_MODELS.items() if v.is_available}
        if not avail:
            # Fallback to active provider model
            return {k: v for k, v in AVAILABLE_MODELS.items() if v.provider == "nebius"}
        return avail

    @classmethod
    def get_model_descriptor(cls, model_id: Optional[str] = None) -> ModelDescriptor:
        refresh_available_models()
        avail = cls.get_available_models()
        if model_id and model_id in AVAILABLE_MODELS:
            return AVAILABLE_MODELS[model_id]
        default_id = get_default_model_id()
        return avail.get(default_id, next(iter(avail.values())))

    @classmethod
    def list_models(cls, include_all: bool = True) -> List[Dict[str, Any]]:
        refresh_available_models()
        if include_all:
            return [desc.model_dump() for desc in AVAILABLE_MODELS.values()]
        avail = cls.get_available_models()
        return [desc.model_dump() for desc in avail.values()]

    @classmethod
    def list_modes(cls) -> List[Dict[str, str]]:
        return [
            {
                "mode": ModelMode.DEFAULT.value,
                "label": "Default Mode",
                "description": "Uses the specified model continuously throughout the task (zero unexpected switching)."
            },
            {
                "mode": ModelMode.AUTO_STAGE.value,
                "label": "Auto Stage Mode",
                "description": "Switches models dynamically per stage among available models (FAST for discovery, DEEP for code synthesis)."
            },
            {
                "mode": ModelMode.AUTO_MODEL.value,
                "label": "Auto Model Mode",
                "description": "Analyzes ticket complexity at intake and selects ONE optimal available model for the entire task."
            }
        ]

    @classmethod
    def select_model_for_task(
        cls,
        ticket: Any,
        mode: ModelMode = ModelMode.DEFAULT,
        requested_model: Optional[str] = None
    ) -> ModelDescriptor:
        """
        Determine the task-level model.
        In DEFAULT mode, allows user's requested model or default model.
        In AUTO_MODEL mode, inspects ticket properties to assign the single best available model.
        """
        avail = cls.get_available_models()
        default_desc = cls.get_model_descriptor(requested_model or get_default_model_id())

        if mode == ModelMode.DEFAULT:
            if requested_model and requested_model in AVAILABLE_MODELS:
                return AVAILABLE_MODELS[requested_model]
            return default_desc

        if mode == ModelMode.AUTO_STAGE:
            # Stage mode starts with an available fast model for intake / discovery
            fast_candidates = [m for m in avail.values() if m.tier == ModelTier.FAST]
            if fast_candidates:
                return fast_candidates[0]
            return default_desc

        if mode == ModelMode.AUTO_MODEL:
            # Extract ticket text and metadata
            desc = ""
            title = ""
            if hasattr(ticket, "description"):
                desc = getattr(ticket, "description", "") or ""
            elif isinstance(ticket, dict):
                desc = ticket.get("description", "") or ""

            if hasattr(ticket, "title"):
                title = getattr(ticket, "title", "") or ""
            elif isinstance(ticket, dict):
                title = ticket.get("title", "") or ""

            combined_text = f"{title}\n{desc}".lower()

            # Rule 1: Angular / UI / Companion files -> Prefer Claude 3.7 if available, else Gemini 2.5 Pro
            ui_signals = ["component", "angular", "template", "scss", "modal", "dialog", "view", "css", "ts"]
            if any(sig in combined_text for sig in ui_signals):
                if "claude-3-7-sonnet" in avail:
                    return avail["claude-3-7-sonnet"]
                if "gemini-2.5-pro" in avail:
                    return avail["gemini-2.5-pro"]

            # Rule 2: Java Spring / Microservices / Maven -> GPT-4o if available, else Gemini 2.5 Pro
            java_signals = ["spring", "serviceimpl", "controller", "repository", "jpa", "pom.xml", "maven", "microservice"]
            if any(sig in combined_text for sig in java_signals):
                if "gpt-4o" in avail:
                    return avail["gpt-4o"]
                if "gemini-2.5-pro" in avail:
                    return avail["gemini-2.5-pro"]

            # Rule 3: Simple configuration / doc / small fix -> FAST tier (e.g. gpt-4o-mini or gemini-2.5-flash)
            simple_signals = ["readme", "typo", "bump version", "config", "license", "comment"]
            if any(sig in combined_text for sig in simple_signals) and len(combined_text) < 400:
                if "gpt-4o-mini" in avail:
                    return avail["gpt-4o-mini"]
                if "gemini-2.5-flash" in avail:
                    return avail["gemini-2.5-flash"]
                fast_candidates = [m for m in avail.values() if m.tier in (ModelTier.FAST, ModelTier.MID)]
                if fast_candidates:
                    return fast_candidates[0]

            # Rule 4: Deep algorithmic / complex bugfix -> Best available DEEP model
            deep_candidates = [m for m in avail.values() if m.tier == ModelTier.DEEP]
            if deep_candidates:
                return deep_candidates[0]

            return default_desc

        return default_desc

    @classmethod
    def select_model_for_stage(
        cls,
        stage_name: str,
        ticket: Any,
        mode: ModelMode = ModelMode.DEFAULT,
        active_task_model: Optional[ModelDescriptor] = None
    ) -> ModelDescriptor:
        """
        Determine model for an individual execution stage.
        If mode is DEFAULT or AUTO_MODEL, preserves active_task_model to protect cache locality.
        If mode is AUTO_STAGE, assigns specialized model per stage requirements using ONLY available models.
        """
        avail = cls.get_available_models()
        fallback_model = active_task_model or cls.get_model_descriptor(get_default_model_id())

        if mode in (ModelMode.DEFAULT, ModelMode.AUTO_MODEL):
            return fallback_model

        # Helper to pick best available model matching requested preferred ids, or fallback by tier
        def _pick_available(preferred_ids: List[str], target_tier: ModelTier) -> ModelDescriptor:
            for pid in preferred_ids:
                if pid in avail:
                    return avail[pid]
            tier_matches = [m for m in avail.values() if m.tier == target_tier]
            if tier_matches:
                return tier_matches[0]
            return fallback_model

        # AUTO_STAGE mapping (ONLY using available models):
        stage = stage_name.lower()

        # Phase A: Discovery, Intake, Classification, Context Search, Log Summaries -> FAST Tier
        fast_stages = {
            "investigate", "classification", "discovery", "rag_tests", "rag_code",
            "validate_candidates", "localize", "ownership_completeness",
            "memory_update", "shadow_metrics", "outcome_check"
        }
        if stage in fast_stages:
            return _pick_available(["gemini-2.5-flash", "gemini-2.0-flash", "gpt-4o-mini", "nebius/nvidia/Nemotron-3_5-Lightning", "nebius/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"], ModelTier.FAST)

        # Phase B: Planning & Architectural Synthesis -> DEEP Reasoning Tier
        planning_stages = {
            "plan", "remediation_plan", "grounded_understanding",
            "behavior_planning", "dataflow_verification", "plan_validation"
        }
        if stage in planning_stages:
            return _pick_available(["claude-3-7-sonnet", "nebius/nvidia/nemotron-3-super-120b-a12b", "gemini-2.5-pro", "gemini-1.5-pro", "gpt-4o"], ModelTier.DEEP)

        # Phase C: Surgical Code Synthesis, Edit Loops, Error Repair -> DEEP Surgical Tier
        synthesis_stages = {
            "generate_code", "generate_tests", "edit_loop",
            "fix_build", "fix_test", "patch_gate"
        }
        if stage in synthesis_stages:
            return _pick_available(["claude-3-5-sonnet", "nebius/nvidia/nemotron-3-super-120b-a12b", "gemini-2.5-pro", "gpt-4o", "gemini-1.5-pro"], ModelTier.DEEP)

        # Phase D: Mechanical single-file checks
        if "single_file" in stage:
            return _pick_available(["gemini-2.5-flash", "gpt-4o-mini", "nebius/nvidia/Nemotron-3_5-Lightning"], ModelTier.FAST)

        return fallback_model


