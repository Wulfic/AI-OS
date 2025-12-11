"""AI-OS Doctor - Comprehensive diagnostic tool.

This module provides a robust, cross-platform diagnostic system for AI-OS
with modular checks, multiple output formats, and auto-repair capabilities.
"""

from __future__ import annotations

from .runner import run_diagnostics, DiagnosticResult, DiagnosticSeverity, save_report_to_log

__all__ = ["run_diagnostics", "DiagnosticResult", "DiagnosticSeverity", "save_report_to_log"]
