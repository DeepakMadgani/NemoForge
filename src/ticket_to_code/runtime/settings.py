"""
Standalone Settings for Ticket-to-Code.

Replaces aviator.settings with a lightweight, env-var-driven configuration
module that has zero dependency on the Aviator platform.
"""

import os
import logging
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM provider identifiers."""
    NEBIUS = "nebius"
    GOOGLE_GENAI = "google-genai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AWS_BEDROCK = "aws-bedrock"
    AZURE_OPENAI = "azure-openai"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value.replace("_", "-") == other.replace("_", "-")
        return super().__eq__(other)


class _Settings:
    """
    Singleton settings object that reads configuration from environment variables.
    """

    def __init__(self):
        self._llm_provider: Optional[LLMProvider] = None
        self._base_dir: Optional[str] = None

    # ── LLM Provider ─────────────────────────────────────────────────────
    @property
    def llm_provider(self) -> Optional[LLMProvider]:
        if self._llm_provider:
            return self._llm_provider
        raw = os.getenv("LLM_PROVIDER", "").strip().lower()
        mapping = {
            "nebius": LLMProvider.NEBIUS,
            "google_genai": LLMProvider.GOOGLE_GENAI,
            "google": LLMProvider.GOOGLE_GENAI,
            "openai": LLMProvider.OPENAI,
            "anthropic": LLMProvider.ANTHROPIC,
            "aws_bedrock": LLMProvider.AWS_BEDROCK,
            "azure_openai": LLMProvider.AZURE_OPENAI,
        }
        return mapping.get(raw)

    @llm_provider.setter
    def llm_provider(self, value):
        self._llm_provider = value

    # ── Model Names ──────────────────────────────────────────────────────
    @property
    def llm_model(self) -> str:
        return os.getenv("LLM_MODEL", "gemini-2.5-pro")

    @property
    def embeddings_model(self) -> str:
        return os.getenv("EMBEDDINGS_MODEL", "text-embedding-004")

    @property
    def embeddings_provider(self) -> Optional[str]:
        return os.getenv("EMBEDDINGS_PROVIDER", "google")

    # ── API Keys (read-only from env) ────────────────────────────────────
    @property
    def openai_api_key(self) -> Optional[str]:
        return os.getenv("OPENAI_API_KEY")

    @property
    def anthropic_api_key(self) -> Optional[str]:
        return os.getenv("ANTHROPIC_API_KEY")

    @property
    def google_api_key(self) -> Optional[str]:
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    # ── Workspace ────────────────────────────────────────────────────────
    @property
    def base_dir(self) -> Optional[str]:
        if self._base_dir:
            return self._base_dir
        return os.getenv("WORKSPACE_PATH") or os.getenv("TARGET_WORKSPACE_PATH")

    @base_dir.setter
    def base_dir(self, value):
        self._base_dir = value

    @property
    def project_base_path(self) -> Optional[str]:
        return self.base_dir

    # ── Database / Infrastructure ────────────────────────────────────────
    @property
    def database_url(self) -> Optional[str]:
        return os.getenv("DATABASE_URL") or self.postgres_connection

    @property
    def postgres_connection(self) -> Optional[str]:
        return os.getenv("POSTGRES_CONNECTION")

    @property
    def neo4j_password(self) -> Optional[str]:
        return os.getenv("NEO4J_PASSWORD")

    @property
    def broker_url(self) -> Optional[str]:
        return os.getenv("BROKER_URL")


# Module-level singleton — same usage pattern as aviator.settings
settings = _Settings()
