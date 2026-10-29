"""Localization ownership + feature-anchor surfacing (Change A).

Proves the invariant:
    feature anchor / similarity  →  SURFACE candidate  (ranking only)
    (never)  filename match  →  PRIMARY_OWNER / write authority

Primary ownership is proven later by inspection (Phase B), NOT by discovery.
Pure logic; no workspace/DB (LocalizationAgent._verify_ownership is called
unbound because it uses no instance state).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ticket_to_code.agents.localization_agent import (
    extract_feature_phrase, feature_anchor_score, verify_ownership,
    apply_feature_anchor_boost, LocalizationAgent,
)

TICKET = ("In the Add Members modal, check whether that user is already a "
          "member of the current project")
FEATURE = "xchange-ui/src/app/modules/members/add-members/add-members.component.ts"
BACKEND = "project-service/src/main/java/com/cloudforge/projectservice/service/ContractMemberService.java"

# Backend is structurally MUCH larger than the feature file.
BACKEND_INV = {"defines": 14, "displays": 19, "calculates": 20, "persists": 2}
FEATURE_INV = {"defines": 2, "displays": 3, "calculates": 0, "persists": 0}


# 1. Backend structural busyness alone cannot make it PRIMARY_OWNER.
def test_backend_structural_busyness_not_primary():
    role, _ = verify_ownership(BACKEND, BACKEND_INV, {"s_vec": 0.72, "s_own": 0.6})
    assert role == "SUPPORTING" and role != "PRIMARY_OWNER"


# 2. No PRIMARY from persistence count or s_own either.
def test_no_primary_from_persistence_or_sown():
    assert verify_ownership("Repo.java", {"persists": 9}, {})[0] != "PRIMARY_OWNER"
    assert verify_ownership("Svc.java", {}, {"s_own": 0.95})[0] != "PRIMARY_OWNER"


# 3. Filename anchor alone cannot grant PRIMARY / write authority.
def test_feature_anchor_alone_not_write_authority():
    # Even a maximal filename anchor is ignored by ownership classification.
    role, _ = verify_ownership(FEATURE, FEATURE_INV, {"s_vec": 0.70, "feature_anchor": 0.99})
    assert role != "PRIMARY_OWNER"


# 4. The feature file surfaces above a structurally-dense backend via the anchor.
def test_feature_surfaced_above_backend():
    cands = [
        {"path": BACKEND, "unified_score": 0.567, "features": {"s_vec": 0.72}},
        {"path": FEATURE, "unified_score": 0.40, "features": {"s_vec": 0.70}},
    ]
    toks = extract_feature_phrase(TICKET)
    apply_feature_anchor_boost(cands, toks)
    assert cands[0]["path"] == FEATURE, "feature file must surface to the top"
    assert cands[0]["features"]["feature_anchor"] > 0.0
    be = next(c for c in cands if c["path"] == BACKEND)
    assert be["features"].get("feature_anchor", 0.0) == 0.0  # backend gets no anchor


# 5. No matching filename must NOT prevent semantic candidate discovery.
def test_no_filename_match_still_discovered():
    comp = "src/components/participant-assignment-panel.tsx"
    cands = [
        {"path": comp, "unified_score": 0.50, "features": {"s_vec": 0.68}},
        {"path": "src/services/ItemService.java", "unified_score": 0.55, "features": {"s_vec": 0.40}},
    ]
    toks = extract_feature_phrase("Improve the Manage participants dialog")
    apply_feature_anchor_boost(cands, toks)
    paths = {c["path"] for c in cands}
    assert comp in paths, "semantic candidate must remain discoverable without a filename match"
    c = next(x for x in cands if x["path"] == comp)
    assert c["features"]["feature_anchor"] == 0.0  # anchor does not fabricate a match
    # its semantic candidacy (s_vec) is preserved for inspection to prove ownership


# 6. The LIVE localization method now delegates (no structural PRIMARY).
def test_live_method_delegates_no_structural_primary():
    role, _ = LocalizationAgent._verify_ownership(
        None, BACKEND, BACKEND_INV, {"s_vec": 0.72, "s_own": 0.6})
    assert role == "SUPPORTING"


# 7. Templates remain DISPLAY_OWNER companions.
def test_template_is_display_owner():
    assert verify_ownership("add-members.component.html", {"displays": 4}, {})[0] == "DISPLAY_OWNER"


# 8. Feature phrase extraction is generic (no hardcoding), works for any wording.
def test_feature_phrase_generic():
    assert "members" in extract_feature_phrase("Add Members modal")
    assert "participants" in extract_feature_phrase("Manage participants dialog")
    # falls back to leading content words when there is no UI-anchor noun
    assert extract_feature_phrase("Refactor invoice export routine")


# 9. Contiguous filename match scores higher than a single-token match.
def test_anchor_contiguous_stronger_than_single():
    two = feature_anchor_score(["add", "members"], "add-members.component.ts")
    one = feature_anchor_score(["members"], "add-members.component.ts")
    assert two >= one and two > 0.0
    assert feature_anchor_score(["add", "members"], "ContractMemberService.java") == 0.0


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
