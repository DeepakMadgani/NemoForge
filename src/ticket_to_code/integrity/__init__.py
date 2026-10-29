"""
Workspace Integrity Subsystem.

Provides non-destructive patch architecture, developer baseline tracking,
three-way merge hierarchy, patch ownership, idempotency, protected regions,
and reversible AI-only rollback.
"""

from ticket_to_code.integrity.structural_manifest import (
    StructuralChangeManifest,
    StructuralDelta,
    DeleteAuthorization,
    DeleteAuthorizationValidator,
    UnauthorizedDeletionError,
)
from ticket_to_code.integrity.workspace_integrity import (
    WorkspaceBaseline,
    FileBaseline,
    ProtectedRegion,
    PatchOwnershipRecord,
    PatchRegistry,
    HierarchicalPatcher,
    PatchConflictError,
    OptimisticConcurrencyError,
    WorkspaceIntegrityManager,
    get_workspace_integrity_manager,
    set_workspace_integrity_manager,
)

__all__ = [
    "StructuralChangeManifest",
    "StructuralDelta",
    "DeleteAuthorization",
    "DeleteAuthorizationValidator",
    "UnauthorizedDeletionError",
    "WorkspaceBaseline",
    "FileBaseline",
    "ProtectedRegion",
    "PatchOwnershipRecord",
    "PatchRegistry",
    "HierarchicalPatcher",
    "PatchConflictError",
    "OptimisticConcurrencyError",
    "WorkspaceIntegrityManager",
    "get_workspace_integrity_manager",
    "set_workspace_integrity_manager",
]
