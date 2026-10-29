"""Pytest configuration and shared fixtures."""

import os
import sys
import types
from pathlib import Path
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from unittest.mock import Mock, patch

# Ensure src is on sys.path
_src = str(Path(__file__).parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Tests may run without any LLM credentials; allow the offline DummyChatModel.
os.environ.setdefault("AVIATOR_ALLOW_DUMMY_LLM", "1")

import pytest


# ---------------------------------------------------------------------------
# AviatorStubFinder — must be installed at module level BEFORE any
# ticket_to_code import.  The chain ticket_to_code.__init__ → workflow →
# agents.__init__ → investigation_agent eagerly imports aviator.services.llm
# and aviator.vector_store, which aren't available in the test environment.
# ---------------------------------------------------------------------------

class _AviatorStubLoader:
    """Creates an empty namespace module for every aviator.* import."""

    def create_module(self, spec):
        stub = types.ModuleType(spec.name)
        stub.__path__ = []
        stub.__loader__ = self
        stub.__spec__ = spec
        # Provide common stubs that downstream code expects to exist
        stub.LLMRegistry = type("LLMRegistry", (), {
            "get_model": classmethod(lambda cls, *a, **kw: None),
            "get_chat_model": classmethod(lambda cls, *a, **kw: None),
            "get_llm": classmethod(lambda cls, *a, **kw: None),
        })
        return stub

    def exec_module(self, module):
        pass


class AviatorStubFinder(MetaPathFinder):
    """Meta-path finder that intercepts any `aviator.*` import and provides
    a harmless stub module so the rest of ticket_to_code can load."""

    def find_spec(self, fullname, path, target=None):
        if fullname == "aviator" or fullname.startswith("aviator."):
            return ModuleSpec(fullname, _AviatorStubLoader(), is_package=True)
        return None


# Install ONCE at import time — idempotent if conftest is re-evaluated.
if not any(isinstance(f, AviatorStubFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, AviatorStubFinder())


# Mock Google credentials before any imports
@pytest.fixture(scope="session", autouse=True)
def mock_google_credentials():
    """Mock Google Application Default Credentials for all tests."""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "mock_credentials.json"

    with patch("google.auth.default") as mock_auth:
        mock_credentials = Mock()
        mock_credentials.token = "mock_token"
        mock_auth.return_value = (mock_credentials, "mock_project")
        yield


# Mock LLM Registry to avoid actual LLM calls
@pytest.fixture(scope="session", autouse=True)
def mock_llm_registry():
    """Mock LLMRegistry to avoid actual LLM initialization."""
    mock_llm = Mock()
    mock_llm.invoke = Mock(return_value="mock response")
    with patch("ticket_to_code.runtime.llm_factory.LLMRegistry.get_model", return_value=mock_llm), \
         patch("ticket_to_code.runtime.llm_factory.LLMFactory.get_llm", return_value=mock_llm):
        try:
            with patch("aviator.services.llm.LLMRegistry.get_model", return_value=mock_llm):
                yield
        except (ImportError, AttributeError):
            yield


@pytest.fixture
def mock_state():
    """Create a mock state object for testing."""
    state = Mock()
    state.messages = []
    state.query = "test query"
    return state


@pytest.fixture
def sample_weather_data():
    """Sample weather API response."""
    return {
        "location": {"name": "Toronto"},
        "current": {
            "temp_c": 15.0,
            "condition": {"text": "Sunny"},
            "wind_kph": 10.0,
            "humidity": 65,
        },
    }


@pytest.fixture
def sample_holidays_data():
    """Sample holidays API response."""
    return {
        "holidays": [
            {
                "name": "New Year's Day",
                "date": "2026-01-01",
                "observed": "2026-01-01",
                "public": True,
            },
            {
                "name": "Canada Day",
                "date": "2026-07-01",
                "observed": "2026-07-01",
                "public": True,
            },
        ]
    }


@pytest.fixture
async def async_client():
    """Create an async HTTP client for testing."""
    import httpx

    async with httpx.AsyncClient() as client:
        yield client
