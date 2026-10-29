"""
Delivery safety utilities: patch scope enforcement, apply-check, secret scan.
"""

from ticket_to_code.delivery.patch_gate import (
    PatchGate,
    ApplyResult,
    ScopeResult,
    SecretHit,
    DeliveryResult,
)

__all__ = [
    "PatchGate",
    "ApplyResult",
    "ScopeResult",
    "SecretHit",
    "DeliveryResult",
]
