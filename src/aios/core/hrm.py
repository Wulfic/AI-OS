from __future__ import annotations

# Thin facade re-exporting the HRM API and built-in operators from the internal package.
# This preserves the public import path aios.core.hrm while allowing modularized internals.

from .hrm_pkg.api import (
    Operator,
    SimpleOperator,
    AsyncOperator,
    OperatorRegistry,
    Manager,
    Recorder,
    build_default_manager_with_recorder,
)
from .hrm_pkg.operators_builtin import build_default_registry

__all__ = [
    "Operator",
    "SimpleOperator",
    "AsyncOperator",
    "OperatorRegistry",
    "Manager",
    "Recorder",
    "build_default_registry",
    "build_default_manager_with_recorder",
]
