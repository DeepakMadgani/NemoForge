"""Tightened loose-matching tiers in code_generator.

Tier 5 (fuzzy) now needs >= 3 search lines and >= 95% similarity.
Tier 5.5 (anchor relaxation) caps the back-search window at 300 chars and
refuses to drop any line it was not asked to replace.
"""

from difflib import SequenceMatcher

from ticket_to_code.agents.code_generator import (
    _ANCHOR_RELAXATION_BACK_WINDOW,
    _FUZZY_MATCH_THRESHOLD,
    _FUZZY_MIN_SEARCH_LINES,
    _levenshtein_fuzzy_match,
    _try_anchor_relaxation,
)


# ---------------------------------------------------------------------------
# Tier 5: fuzzy match
# ---------------------------------------------------------------------------

def test_fuzzy_defaults_are_tightened():
    assert _FUZZY_MATCH_THRESHOLD == 0.95
    assert _FUZZY_MIN_SEARCH_LINES == 3


def test_fuzzy_two_line_block_returns_none_even_when_identical():
    old = "    total_price = compute_total(items)\n    return finalize_order(total_price, customer)"
    content = "def checkout(items, customer):\n" + old + "\n"
    assert _levenshtein_fuzzy_match(old, old.replace("items", "cart"), content) is None


def test_fuzzy_92pct_similar_three_line_block_returns_none():
    old = (
        "    total_price = compute_total(items)\n"
        "    discount = apply_discount(total_price, coupon)\n"
        "    return finalize_order(discount, customer)"
    )
    near = (
        "    total_price = compute_total(items)\n"
        "    discount = apply_discount(total_price, coupon)\n"
        "    return finalize_invoice(discount, client)"
    )
    ratio = SequenceMatcher(None, old, near).ratio()
    assert 0.90 < ratio < 0.95, f"fixture drifted: ratio={ratio:.4f}"

    content = "def checkout(items, coupon, client):\n" + near + "\n"
    assert _levenshtein_fuzzy_match(old, old + "  # edited", content) is None
    # The same block was accepted under the old 90% threshold.
    assert _levenshtein_fuzzy_match(old, old + "  # edited", content, threshold=0.90) is not None


def test_fuzzy_three_line_block_above_threshold_still_applies():
    old = (
        "    total_price = compute_total(items)\n"
        "    discount = apply_discount(total_price, coupon)\n"
        "    return finalize_order(discount, customer)"
    )
    # Whitespace-only drift (double space) keeps similarity above 95%.
    near = old.replace("apply_discount(total_price, coupon)", "apply_discount(total_price,  coupon)")
    assert SequenceMatcher(None, old, near).ratio() >= 0.95
    content = "def checkout(items, coupon, customer):\n" + near + "\n"
    new = old.replace("finalize_order", "finalize_checkout")

    result = _levenshtein_fuzzy_match(old, new, content)

    assert result is not None
    assert "finalize_checkout(discount, customer)" in result
    assert "finalize_order" not in result


# ---------------------------------------------------------------------------
# Tier 5.5: anchor relaxation
# ---------------------------------------------------------------------------

def test_anchor_back_window_is_capped_at_300_chars():
    assert _ANCHOR_RELAXATION_BACK_WINDOW == 300


def test_anchor_relaxation_refuses_when_unexpected_lines_would_be_dropped():
    # An earlier hunk inserted `audit_log.record(items)` right after the def line.
    content = (
        "def compute_total(items):\n"
        "    audit_log.record(items)\n"
        "    total = sum(items)\n"
        "    return total\n"
    )
    old = (
        "def compute_total(items):\n"
        "    total = sum(items)\n"
        "    return total\n"
    )
    # new_str rewrites the first line, so the inserted audit line cannot be re-attached.
    new = (
        "def compute_total(items, tax):\n"
        "    total = sum(items)\n"
        "    return total\n"
    )

    assert _try_anchor_relaxation(old, new, content) is None


def test_anchor_relaxation_still_handles_adjacent_insert():
    content = (
        "def compute_total(items):\n"
        "    audit_log.record(items)\n"
        "    total_amount = sum(items)\n"
        "    return total_amount\n"
    )
    old = (
        "def compute_total(items):\n"
        "    total_amount = sum(items)\n"
        "    return total_amount\n"
    )
    new = (
        "def compute_total(items):\n"
        "    total_amount = sum(items)\n"
        "    total_amount = round(total_amount, 2)\n"
        "    return total_amount\n"
    )

    result = _try_anchor_relaxation(old, new, content)

    assert result is not None
    expected = (
        "def compute_total(items):\n"
        "    audit_log.record(items)\n"
        "    total_amount = sum(items)\n"
        "    total_amount = round(total_amount, 2)\n"
        "    return total_amount\n"
    )
    assert result == expected
    # Nothing duplicated, nothing lost.
    assert result.count("total_amount = sum(items)") == 1
    assert result.count("audit_log.record(items)") == 1


def test_anchor_relaxation_does_not_reach_beyond_window():
    # The first line is more than 300 chars before the anchor: refuse rather than
    # swallow a huge unrelated region.
    filler = "".join(f"    unrelated_statement_{i}()\n" for i in range(20))  # ~600 chars
    content = (
        "def compute_total(items):\n"
        + filler
        + "    total_amount = sum(items)\n"
        "    return total_amount\n"
    )
    old = (
        "def compute_total(items):\n"
        "    total_amount = sum(items)\n"
        "    return total_amount\n"
    )
    new = old + "    # trailing comment\n"

    assert _try_anchor_relaxation(old, new, content) is None
