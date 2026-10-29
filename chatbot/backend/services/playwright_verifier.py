import sys
from pathlib import Path

_src_path = Path(__file__).resolve().parent.parent.parent.parent / "src"
if _src_path.exists() and str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from ticket_to_code.verification.playwright_verifier import (  # noqa: F401
    PlaywrightUIVerifier,
    UIAssertion,
    UIVerificationResult,
)

from ticket_to_code.verification.environment_manager import VerificationEnvironmentManager

_env_mgr = VerificationEnvironmentManager("default_chat_service")
_caps = _env_mgr.capabilities

# Use discovered browser and headless auto-detection
_channel = _caps.preferred_browser if _caps.preferred_browser in ("msedge", "chrome") else ""
playwright_verifier = PlaywrightUIVerifier(
    headless=_caps.is_headless_required,
    slow_mo=500 if not _caps.is_headless_required else 0,
    browser_channel=_channel,
)

