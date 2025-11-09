"""DDP (Distributed Data Parallel) training support for AI-OS.

This package contains all DDP-related functionality:
- Worker spawning and configuration serialization
- DDP utilities for multi-GPU training
- Worker entry point for child processes
"""

from .worker import ddp_worker, serialize_config_for_spawn
from .utils import maybe_spawn_and_exit_if_parent

__all__ = [
    "ddp_worker",
    "serialize_config_for_spawn",
    "maybe_spawn_and_exit_if_parent",
]
