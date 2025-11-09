"""Core-integrated HRM model implementations (migrated from vendor).

Keep heavy deps optional. Import guarded inside functions to avoid import-time failures.
"""

from .act_v1 import build_model as build_act_v1
from .hf_adapter import build_hf_adapter, build_hf_starter_from_config

__all__ = [
    "build_act_v1",
    "build_hf_adapter",
    "build_hf_starter_from_config",
]
