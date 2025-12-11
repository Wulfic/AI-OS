"""Core diagnostic runner for AI-OS Doctor.

Orchestrates all diagnostic checks and formats output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class DiagnosticSeverity(str, Enum):
    """Severity levels for diagnostic results."""
    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DiagnosticResult:
    """Result of a single diagnostic check."""
    name: str
    severity: DiagnosticSeverity
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: Optional[str] = None
    auto_fixable: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output."""
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class DiagnosticReport:
    """Complete diagnostic report."""
    timestamp: str
    platform: str
    python_version: str
    aios_version: str
    duration_seconds: float
    results: list[DiagnosticResult] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "timestamp": self.timestamp,
            "platform": self.platform,
            "python_version": self.python_version,
            "aios_version": self.aios_version,
            "duration_seconds": self.duration_seconds,
            "summary": self.summary,
            "results": [r.to_dict() for r in self.results],
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


def _get_aios_version() -> str:
    """Get AI-OS version."""
    try:
        from aios import __version__
        return __version__
    except Exception:
        return "unknown"


def _get_platform_info() -> str:
    """Get detailed platform information."""
    try:
        system = platform.system()
        release = platform.release()
        machine = platform.machine()
        return f"{system} {release} ({machine})"
    except Exception:
        return "unknown"


async def run_diagnostics(
    *,
    check_permissions: bool = True,
    check_dependencies: bool = True,
    check_gpu: bool = True,
    check_disk: bool = True,
    check_network: bool = True,
    check_env_vars: bool = True,
    check_config: bool = True,
    check_memory: bool = True,
    auto_repair: bool = False,
    json_output: bool = False,
) -> DiagnosticReport:
    """Run all diagnostic checks.
    
    Args:
        check_permissions: Check directory permissions
        check_dependencies: Check Python package dependencies
        check_gpu: Check GPU availability and configuration
        check_disk: Check disk space
        check_network: Check network connectivity
        check_env_vars: Display relevant environment variables
        check_config: Validate configuration files
        check_memory: Check system memory
        auto_repair: Attempt to fix issues automatically
        json_output: Format output as JSON
        
    Returns:
        DiagnosticReport with all results
    """
    from datetime import datetime
    
    start_time = time.perf_counter()
    results: list[DiagnosticResult] = []
    
    # Import check modules
    from . import checks
    
    # Collect all checks to run
    check_functions: list[Callable] = []
    
    # Platform checks always run
    check_functions.append(checks.check_platform)
    check_functions.append(checks.check_elevation)
    
    if check_permissions:
        check_functions.append(checks.check_permissions)
    
    if check_dependencies:
        check_functions.append(checks.check_dependencies)
    
    if check_gpu:
        check_functions.append(checks.check_gpu)
    
    if check_disk:
        check_functions.append(checks.check_disk_space)
    
    if check_network:
        check_functions.append(checks.check_network)
    
    if check_env_vars:
        check_functions.append(checks.check_environment_variables)
    
    if check_config:
        check_functions.append(checks.check_config_files)
    
    if check_memory:
        check_functions.append(checks.check_memory)
    
    # Run checks (some may be async)
    for check_fn in check_functions:
        try:
            if asyncio.iscoroutinefunction(check_fn):
                check_results = await check_fn(auto_repair=auto_repair)
            else:
                check_results = check_fn(auto_repair=auto_repair)
            
            if isinstance(check_results, list):
                results.extend(check_results)
            else:
                results.append(check_results)
        except Exception as e:
            logger.exception(f"Check {check_fn.__name__} failed")
            results.append(DiagnosticResult(
                name=check_fn.__name__,
                severity=DiagnosticSeverity.ERROR,
                message=f"Check failed with exception: {e}",
            ))
    
    # Calculate summary
    summary = {
        "ok": 0,
        "info": 0,
        "warning": 0,
        "error": 0,
        "critical": 0,
        "total": len(results),
    }
    for r in results:
        summary[r.severity.value] = summary.get(r.severity.value, 0) + 1
    
    duration = time.perf_counter() - start_time
    
    report = DiagnosticReport(
        timestamp=datetime.now().isoformat(),
        platform=_get_platform_info(),
        python_version=platform.python_version(),
        aios_version=_get_aios_version(),
        duration_seconds=round(duration, 3),
        results=results,
        summary=summary,
    )
    
    return report


def format_report_text(report: DiagnosticReport) -> str:
    """Format report as human-readable text with colors."""
    lines = []
    
    # Header
    lines.append("=" * 60)
    lines.append("AI-OS Doctor - Diagnostic Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Timestamp:      {report.timestamp}")
    lines.append(f"Platform:       {report.platform}")
    lines.append(f"Python:         {report.python_version}")
    lines.append(f"AI-OS Version:  {report.aios_version}")
    lines.append(f"Duration:       {report.duration_seconds:.3f}s")
    lines.append("")
    
    # Results grouped by severity
    severity_order = [
        DiagnosticSeverity.CRITICAL,
        DiagnosticSeverity.ERROR,
        DiagnosticSeverity.WARNING,
        DiagnosticSeverity.INFO,
        DiagnosticSeverity.OK,
    ]
    
    severity_symbols = {
        DiagnosticSeverity.OK: "[+]",
        DiagnosticSeverity.INFO: "[i]",
        DiagnosticSeverity.WARNING: "[!]",
        DiagnosticSeverity.ERROR: "[X]",
        DiagnosticSeverity.CRITICAL: "[!!]",
    }
    
    for severity in severity_order:
        severity_results = [r for r in report.results if r.severity == severity]
        if not severity_results:
            continue
        
        lines.append("-" * 60)
        lines.append(f"{severity.value.upper()} ({len(severity_results)})")
        lines.append("-" * 60)
        
        for r in severity_results:
            symbol = severity_symbols.get(r.severity, "[?]")
            lines.append(f"{symbol} {r.name}: {r.message}")
            
            if r.details:
                for key, value in r.details.items():
                    if isinstance(value, dict):
                        lines.append(f"    {key}:")
                        for k, v in value.items():
                            lines.append(f"      {k}: {v}")
                    elif isinstance(value, list):
                        lines.append(f"    {key}:")
                        for item in value[:10]:  # Limit display
                            lines.append(f"      - {item}")
                        if len(value) > 10:
                            lines.append(f"      ... and {len(value) - 10} more")
                    else:
                        lines.append(f"    {key}: {value}")
            
            if r.suggestion:
                lines.append(f"    -> {r.suggestion}")
            
            if r.auto_fixable:
                lines.append("    [Auto-fixable with --repair]")
        
        lines.append("")
    
    # Summary
    lines.append("=" * 60)
    lines.append("Summary")
    lines.append("=" * 60)
    lines.append(f"Total checks: {report.summary['total']}")
    lines.append(f"  OK:       {report.summary['ok']}")
    lines.append(f"  Info:     {report.summary['info']}")
    lines.append(f"  Warnings: {report.summary['warning']}")
    lines.append(f"  Errors:   {report.summary['error']}")
    lines.append(f"  Critical: {report.summary['critical']}")
    
    if report.summary['critical'] > 0 or report.summary['error'] > 0:
        lines.append("")
        lines.append("⚠️  Issues detected. Review errors above and apply suggestions.")
    elif report.summary['warning'] > 0:
        lines.append("")
        lines.append("⚡ Some warnings detected, but AI-OS should work.")
    else:
        lines.append("")
        lines.append("✅ All checks passed! AI-OS is ready to use.")
    
    return "\n".join(lines)


def save_report_to_log(report: DiagnosticReport, log_path: Optional[Path] = None) -> Path:
    """Save the diagnostic report to a log file.
    
    Args:
        report: The diagnostic report to save
        log_path: Optional custom path. If None, uses the default logs directory.
        
    Returns:
        Path to the saved log file
    """
    if log_path is None:
        # Get the logs directory from system paths
        try:
            from aios.system import paths as system_paths
            logs_dir = system_paths.get_logs_dir()
        except ImportError:
            # Fallback to current directory
            logs_dir = Path.cwd() / "logs"
        
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "doctors_prognosis.log"
    
    # Format the report for the log file
    text_report = format_report_text(report)
    
    # Add a separator and timestamp header for log file readability
    log_content = []
    log_content.append("")
    log_content.append("#" * 70)
    log_content.append(f"# Doctor's Prognosis - {report.timestamp}")
    log_content.append("#" * 70)
    log_content.append("")
    log_content.append(text_report)
    log_content.append("")
    
    # Append to the log file (creates if doesn't exist)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(log_content))
        logger.debug(f"Saved diagnostic report to {log_path}")
    except Exception as e:
        logger.warning(f"Failed to save diagnostic report to {log_path}: {e}")
        raise
    
    return log_path
