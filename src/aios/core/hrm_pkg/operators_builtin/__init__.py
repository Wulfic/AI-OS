"""Built-in Operators Registry Builder.

This module provides the default operator registry with built-in operators
organized by category: system, service, and journal operations.
"""

from __future__ import annotations

from ..api import OperatorRegistry
from .system_operators import register_system_operators
from .service_operators import register_service_operators
from .journal_operators import register_journal_operators


def build_default_registry() -> OperatorRegistry:
    """Create a registry with all built-in operators for smoke/evals."""
    reg = OperatorRegistry()
    
    # Register all operator categories
    register_system_operators(reg)
    register_service_operators(reg)
    register_journal_operators(reg)
    
    return reg


__all__ = ["build_default_registry"]
