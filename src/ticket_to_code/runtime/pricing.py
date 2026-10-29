"""USD list-price table for LLM cost accounting.

Prices are USD per 1M tokens as (input, output). The table is intentionally
small; unknown models cost 0.0 and emit a single warning per process so the
RunBudget.max_cost_usd ceiling is never silently disabled.

Override/extend via env ``AVIATOR_PRICING_JSON`` pointing at a JSON file with
either ``{"model": [input_per_1m, output_per_1m]}`` or
``{"model": {"input": x, "output": y}}`` entries.
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_PRICES: Dict[str, Tuple[float, float]] = {
    "nvidia/nemotron-3-super-120b-a12b": (0.60, 1.80),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-7-sonnet": (3.00, 15.00),
    "gpt-4o": (2.50, 10.00),
}

_WARNED_UNKNOWN: set = set()


def _coerce_pair(value) -> Optional[Tuple[float, float]]:
    try:
        if isinstance(value, dict):
            return float(value.get("input", 0.0)), float(value.get("output", 0.0))
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return None


@lru_cache(maxsize=1)
def _load_prices() -> Dict[str, Tuple[float, float]]:
    prices: Dict[str, Tuple[float, float]] = dict(_DEFAULT_PRICES)
    path = os.getenv("AVIATOR_PRICING_JSON", "").strip()
    if path:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                for name, raw in data.items():
                    pair = _coerce_pair(raw)
                    if pair is not None:
                        prices[str(name)] = pair
        except Exception as exc:  # never break the harness over pricing
            logger.warning("AVIATOR_PRICING_JSON unreadable (%s): %s", path, exc)
    return prices


def _match(model: str) -> Optional[Tuple[float, float]]:
    prices = _load_prices()
    if model in prices:
        return prices[model]
    lowered = model.lower()
    # Longest key first so "claude-3-5-sonnet" beats a hypothetical "claude".
    for key in sorted(prices, key=len, reverse=True):
        k = key.lower()
        if lowered.startswith(k) or k in lowered:
            return prices[key]
    return None


def estimate_cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate the USD cost of one call. Unknown model -> 0.0 (warned once)."""
    name = (model or "").strip()
    pair = _match(name) if name and name != "unknown" else None
    if pair is None:
        if name not in _WARNED_UNKNOWN:
            _WARNED_UNKNOWN.add(name)
            logger.warning("No pricing entry for model %r; charging $0 (set AVIATOR_PRICING_JSON to override)", name)
        return 0.0
    in_price, out_price = pair
    cost = (max(0, int(tokens_in or 0)) / 1_000_000.0) * in_price + (max(0, int(tokens_out or 0)) / 1_000_000.0) * out_price
    return round(cost, 8)


def reset_pricing_cache() -> None:
    """Test helper: forget loaded prices and the unknown-model warning set."""
    _load_prices.cache_clear()
    _WARNED_UNKNOWN.clear()


__all__ = ["estimate_cost_usd", "reset_pricing_cache"]
