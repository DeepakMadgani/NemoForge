"""Lean deterministic execution path for scoped tickets.

The guarantees that make runaway loops structurally impossible live here, not in call caps:
  - ScopeFence      physical tool-level boundary (scope.py)
  - anchor locate   deterministic file selection, 0 LLM calls (locate.py)
  - snippet window  bounded evidence per file (snippet.py)
  - ConvergenceTracker  halts on repeat/oscillating failure (convergence.py)
  - surgical hunks  anchored search/replace, no full-file rewrites (hunks.py)
"""
