"""Unit tests for Canonical Path Identity & Component Family Normalization."""

import os
from pathlib import Path
import pytest

from ticket_to_code.agents.canonical_path import (
    canonical_repo_path,
    canonical_component_base,
    is_component_family_member,
    COMPONENT_EXTENSIONS,
)


def test_windows_absolute_and_relative_identity(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    target = ws / "xchange-ui" / "src" / "app" / "comp.ts"
    target.parent.mkdir(parents=True)
    target.touch()

    abs_path = str(target)
    rel_path = "xchange-ui/src/app/comp.ts"
    mixed_path = "xchange-ui\\src/app\\comp.ts"

    canon_abs = canonical_repo_path(abs_path, ws)
    canon_rel = canonical_repo_path(rel_path, ws)
    canon_mixed = canonical_repo_path(mixed_path, ws)

    assert canon_abs is not None
    assert canon_abs == canon_rel
    assert canon_abs == canon_mixed
    assert canon_abs.replace("\\", "/") == "xchange-ui/src/app/comp.ts".lower()


def test_unrelated_workspace_never_collapses(tmp_path: Path):
    ws1 = tmp_path / "ws1"
    ws2 = tmp_path / "ws2"
    ws1.mkdir()
    ws2.mkdir()

    file2 = ws2 / "src" / "app.ts"
    file2.parent.mkdir(parents=True)
    file2.touch()

    # file2 is in ws2, so checking against ws1 MUST return None
    assert canonical_repo_path(file2, ws1) is None
    # checking against ws2 should succeed
    assert canonical_repo_path(file2, ws2) == "src/app.ts".lower()


def test_component_base_family_mapping(tmp_path: Path):
    ws = tmp_path / "repo"
    ws.mkdir()

    ts_file = ws / "xchange-ui" / "src" / "add-members.component.ts"
    html_file = "xchange-ui/src/add-members.component.html"
    scss_file = "xchange-ui\\src\\add-members.component.scss"
    service_file = ws / "xchange-ui" / "src" / "member.service.ts"

    base_ts = canonical_component_base(ts_file, ws)
    base_html = canonical_component_base(html_file, ws)
    base_scss = canonical_component_base(scss_file, ws)
    base_service = canonical_component_base(service_file, ws)

    expected_base = "xchange-ui/src/add-members".lower()
    assert base_ts == expected_base
    assert base_html == expected_base
    assert base_scss == expected_base

    # All 3 component files belong to the same family
    assert is_component_family_member(ts_file, html_file, ws)
    assert is_component_family_member(html_file, scss_file, ws)

    # Service is NOT a component companion
    assert base_service is None
    assert not is_component_family_member(ts_file, service_file, ws)
