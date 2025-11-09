"""Data structures for HF HRM Adapter.

Defines carry state structures used in the HRM-compatible forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class _InnerCarry:
    """Inner carry state for HRM adapter."""
    z_H: Any
    z_L: Any


@dataclass
class _Carry:
    """Top-level carry state passed through HRM steps."""
    inner_carry: _InnerCarry
    steps: Any
    halted: Any
    current_data: Dict[str, Any]
