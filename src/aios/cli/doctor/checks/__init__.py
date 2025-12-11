"""Diagnostic checks for AI-OS Doctor.

This module exports all available diagnostic checks.
"""

from __future__ import annotations

from .platform_checks import check_platform, check_elevation
from .permissions_check import check_permissions
from .dependencies_check import check_dependencies
from .gpu_check import check_gpu
from .disk_check import check_disk_space
from .network_check import check_network
from .env_vars_check import check_environment_variables
from .config_check import check_config_files
from .memory_check import check_memory

__all__ = [
    "check_platform",
    "check_elevation",
    "check_permissions",
    "check_dependencies",
    "check_gpu",
    "check_disk_space",
    "check_network",
    "check_environment_variables",
    "check_config_files",
    "check_memory",
]
