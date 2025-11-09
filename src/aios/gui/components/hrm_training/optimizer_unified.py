"""Unified Optimization System - backward compatibility wrapper.

This file has been refactored into a package: optimizer_unified/

All imports from this file will continue to work via re-export.
The optimizer_unified package contains:
- config.py: OptimizationConfig dataclass
- optimizer.py: UnifiedOptimizer class
- api.py: Public API functions (optimize_from_config, optimize_from_dict, optimize_from_gui, optimize_cli)
- batch_optimization.py: Adaptive batch size optimization logic
- batch_runner.py: Single batch execution logic
- command_builder.py: Command/environment building utilities
- gpu_monitoring.py: GPU monitor integration
- process_manager.py: Process management with heartbeat monitoring
- result_parser.py: Throughput/OOM parsing
"""

from __future__ import annotations

from .optimizer_unified import (
    OptimizationConfig,
    UnifiedOptimizer,
    optimize_from_config,
    optimize_from_dict,
    optimize_from_gui,
    optimize_cli,
)

__all__ = [
    'OptimizationConfig',
    'UnifiedOptimizer',
    'optimize_from_config',
    'optimize_from_dict',
    'optimize_from_gui',
    'optimize_cli',
]
