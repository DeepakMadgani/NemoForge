"""Unit tests for Jato namespace extraction and component localization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ticket_to_code.agents.localization_agent import (
    extract_feature_phrase,
    extract_ticket_namespace,
    feature_anchor_score,
    apply_feature_anchor_boost,
)


def test_jato_namespace_extraction():
    title = "Jato: Scroll Missing data overflow behind footer in All Projects"
    ns = extract_ticket_namespace(title)
    assert ns == "jato"


def test_jato_feature_phrase_extraction():
    title = "Jato: Scroll Missing data overflow behind footer in All Projects"
    tokens = extract_feature_phrase(title)
    assert "projects" in tokens


def test_jprojects_stem_anchor_match():
    # jprojects should match 'projects' via sub-stem expansion
    score_jato = feature_anchor_score(["projects"], "src/jato/projects/jprojects.component.ts")
    assert score_jato > 0.0, f"Expected > 0, got {score_jato}"


def test_jato_namespace_boost_surfaces_jprojects_above_legacy():
    title = "Jato: Scroll Missing data overflow behind footer in All Projects"
    ns = extract_ticket_namespace(title)
    toks = extract_feature_phrase(title)

    candidates = [
        {"path": "src/app/modules/home/projects/projects.component.ts", "unified_score": 0.60, "signals": []},
        {"path": "src/jato/projects/jprojects.component.ts", "unified_score": 0.50, "signals": []},
    ]

    apply_feature_anchor_boost(candidates, toks, namespace=ns)

    # Jato candidate must be #1
    assert candidates[0]["path"] == "src/jato/projects/jprojects.component.ts"
    assert "namespace_match:jato" in candidates[0]["signals"]
    assert "namespace_penalty:jato" in candidates[1]["signals"]
    assert candidates[0]["unified_score"] > candidates[1]["unified_score"]
