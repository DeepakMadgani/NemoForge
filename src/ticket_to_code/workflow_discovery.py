"""Public entrypoint for the workflow discovery layer.

This module re-exports the indexing implementation without importing the
``ticket_to_code.indexing`` package, which has heavier side-effect imports in
its ``__init__``.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_IMPL_PATH = Path(__file__).resolve().parent / "indexing" / "workflow_discovery.py"
_SPEC = spec_from_file_location("ticket_to_code._workflow_discovery_impl", _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:
	raise ImportError(f"Unable to load workflow discovery implementation from {_IMPL_PATH}")

_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

WorkflowDiscovery = _MODULE.WorkflowDiscovery

__all__ = ["WorkflowDiscovery"]