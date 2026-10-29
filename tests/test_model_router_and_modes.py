"""
Tests for 3-Mode Model Selector & Model Registry.
"""
import pytest
from ticket_to_code.intelligence.model_router import (
    ModelRouter,
    ModelMode,
    ModelTier,
    AVAILABLE_MODELS,
    refresh_available_models
)
from ticket_to_code.models import ValueEdgeTicket, TicketPriority


def test_list_models_and_modes():
    modes = ModelRouter.list_modes()
    assert len(modes) == 3
    mode_values = [m["mode"] for m in modes]
    assert "default" in mode_values
    assert "auto_stage" in mode_values
    assert "auto_model" in mode_values

    models = ModelRouter.list_models()
    assert len(models) >= 5
    model_ids = [m["model_id"] for m in models]
    assert "claude-3-7-sonnet" in model_ids
    assert "claude-3-5-sonnet" in model_ids
    assert "gpt-4o" in model_ids
    assert "gemini-2.0-flash" in model_ids


def test_default_mode_selection():
    ticket = ValueEdgeTicket(
        ticket_id="T-100",
        title="Fix NullPointerException",
        description="Fix in UserService",
        priority=TicketPriority.MEDIUM
    )
    # Explicit model requested
    model = ModelRouter.select_model_for_task(
        ticket, mode=ModelMode.DEFAULT, requested_model="gpt-4o"
    )
    assert model.model_id == "gpt-4o"

    # Default fallback
    model_default = ModelRouter.select_model_for_task(
        ticket, mode=ModelMode.DEFAULT, requested_model=None
    )
    assert model_default.model_id in AVAILABLE_MODELS


def test_auto_model_selection_ui_vs_backend(monkeypatch):
    # Enable all providers for multi-model benchmark test
    monkeypatch.setattr(
        "ticket_to_code.intelligence.model_router.detect_configured_providers",
        lambda: {"google": True, "openai": True, "anthropic": True}
    )
    refresh_available_models()

    # Angular UI ticket -> should select Claude 3.7 Sonnet
    ui_ticket = ValueEdgeTicket(
        ticket_id="T-200",
        title="Fix Angular dialog button spacing in modal component",
        description="Update upload-dialog.component.html and scss",
        priority=TicketPriority.HIGH
    )
    ui_model = ModelRouter.select_model_for_task(ui_ticket, mode=ModelMode.AUTO_MODEL)
    assert ui_model.model_id == "claude-3-7-sonnet"

    # Java Spring backend ticket -> should select GPT-4o
    backend_ticket = ValueEdgeTicket(
        ticket_id="T-300",
        title="Add validation to AreaServiceImpl controller in Spring Boot",
        description="Handle jpa repository query and maven build",
        priority=TicketPriority.MEDIUM
    )
    backend_model = ModelRouter.select_model_for_task(backend_ticket, mode=ModelMode.AUTO_MODEL)
    assert backend_model.model_id == "gpt-4o"

    # Simple typo ticket -> should select GPT-4o-mini
    simple_ticket = ValueEdgeTicket(
        ticket_id="T-400",
        title="Fix typo in README comment",
        description="Fix simple typo in documentation",
        priority=TicketPriority.LOW
    )
    simple_model = ModelRouter.select_model_for_task(simple_ticket, mode=ModelMode.AUTO_MODEL)
    assert simple_model.model_id == "gpt-4o-mini"


def test_auto_stage_selection_routes_per_stage(monkeypatch):
    # Enable all providers for multi-model benchmark test
    monkeypatch.setattr(
        "ticket_to_code.intelligence.model_router.detect_configured_providers",
        lambda: {"google": True, "openai": True, "anthropic": True}
    )
    refresh_available_models()

    ticket = ValueEdgeTicket(
        ticket_id="T-500",
        title="General Feature Implementation",
        description="Add feature",
        priority=TicketPriority.MEDIUM
    )
    # Stage 1: Discovery / investigate -> FAST tier (Gemini 2.5 Flash / 2.0 Flash)
    discovery_model = ModelRouter.select_model_for_stage(
        "investigate", ticket, mode=ModelMode.AUTO_STAGE
    )
    assert discovery_model.tier == ModelTier.FAST
    assert "flash" in discovery_model.model_id

    # Stage 2: Planning -> DEEP tier (Claude 3.7 Sonnet)
    planning_model = ModelRouter.select_model_for_stage(
        "plan", ticket, mode=ModelMode.AUTO_STAGE
    )
    assert planning_model.tier == ModelTier.DEEP
    assert planning_model.model_id == "claude-3-7-sonnet"

    # Stage 3: Code Generation / Edit Loop -> DEEP surgical tier (Claude 3.5 Sonnet)
    codegen_model = ModelRouter.select_model_for_stage(
        "generate_code", ticket, mode=ModelMode.AUTO_STAGE
    )
    assert codegen_model.tier == ModelTier.DEEP
    assert codegen_model.model_id == "claude-3-5-sonnet"

    # Stage 4: Outcome check / memory update -> FAST tier
    outcome_model = ModelRouter.select_model_for_stage(
        "outcome_check", ticket, mode=ModelMode.AUTO_STAGE
    )
    assert outcome_model.tier == ModelTier.FAST


def test_single_provider_google_only_availability(monkeypatch):
    # Simulate user's real enterprise environment: Google GenAI ONLY
    monkeypatch.setattr(
        "ticket_to_code.intelligence.model_router.detect_configured_providers",
        lambda: {"google": True, "openai": False, "anthropic": False}
    )
    refresh_available_models()

    avail = ModelRouter.get_available_models()
    assert all(m.provider == "google" for m in avail.values())
    assert "claude-3-7-sonnet" not in avail
    assert "gpt-4o" not in avail

    ticket = ValueEdgeTicket(
        ticket_id="T-700",
        title="Fix Angular dialog button spacing in modal component",
        description="Update upload-dialog.component.html and scss",
        priority=TicketPriority.HIGH
    )
    # Even on an Angular ticket, AUTO_MODEL must pick Gemini 2.5 Pro because Claude is not available
    task_model = ModelRouter.select_model_for_task(ticket, mode=ModelMode.AUTO_MODEL)
    assert task_model.model_id == "gemini-2.5-pro"
    assert task_model.is_available is True

    # In AUTO_STAGE, planning and synthesis route to Gemini 2.5 Pro (DEEP), discovery to Gemini 2.5 Flash (FAST)
    plan_model = ModelRouter.select_model_for_stage("plan", ticket, mode=ModelMode.AUTO_STAGE)
    assert plan_model.model_id == "gemini-2.5-pro"

    discovery_model = ModelRouter.select_model_for_stage("investigate", ticket, mode=ModelMode.AUTO_STAGE)
    assert discovery_model.model_id in ("gemini-2.5-flash", "gemini-2.0-flash")


def test_default_and_auto_model_preserve_cache_across_stages():
    ticket = ValueEdgeTicket(
        ticket_id="T-600",
        title="Some task",
        description="Details",
        priority=TicketPriority.MEDIUM
    )
    task_model = AVAILABLE_MODELS["gemini-2.5-pro"]

    # In AUTO_MODEL mode, stage calls MUST return task_model to preserve cache
    stage_model = ModelRouter.select_model_for_stage(
        "investigate", ticket, mode=ModelMode.AUTO_MODEL, active_task_model=task_model
    )
    assert stage_model.model_id == "gemini-2.5-pro"

    stage_model_gen = ModelRouter.select_model_for_stage(
        "generate_code", ticket, mode=ModelMode.AUTO_MODEL, active_task_model=task_model
    )
    assert stage_model_gen.model_id == "gemini-2.5-pro"


@pytest.mark.asyncio
async def test_api_models_endpoint():
    import sys
    from pathlib import Path
    backend_dir = Path(__file__).resolve().parents[1] / "chatbot" / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    from main import get_available_models
    res = await get_available_models()
    assert "modes" in res
    assert "models" in res
    assert "default_model" in res
    assert len(res["modes"]) == 3
    assert len(res["models"]) >= 5

