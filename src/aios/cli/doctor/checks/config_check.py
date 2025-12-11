"""Configuration file validation for AI-OS Doctor."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from ..runner import DiagnosticResult, DiagnosticSeverity

logger = logging.getLogger(__name__)


def check_config_files(*, auto_repair: bool = False) -> list[DiagnosticResult]:
    """Validate configuration files."""
    results = []
    
    # Get config paths
    config_files = _get_config_files()
    
    for name, path, validator in config_files:
        if not path.exists():
            results.append(DiagnosticResult(
                name=name,
                severity=DiagnosticSeverity.INFO,
                message="Not found (using defaults)",
                details={"path": str(path)},
            ))
            continue
        
        result = _validate_config_file(name, path, validator, auto_repair=auto_repair)
        results.append(result)
    
    # Check for required config files
    required_results = _check_required_configs()
    results.extend(required_results)
    
    return results


def _get_config_files() -> list[tuple[str, Path, str]]:
    """Get list of config files to validate."""
    files = []
    
    try:
        from aios.system import paths as system_paths
        
        config_dir = system_paths.get_user_config_dir()
        install_root = system_paths.get_install_root()
        
        # User configs
        files.append(("User Config (default.yaml)", config_dir / "default.yaml", "yaml"))
        files.append(("Logging Config", config_dir / "logging.yaml", "yaml"))
        files.append(("Tool Permissions", config_dir / "tool_permissions.json", "json"))
        
        # Project configs
        files.append(("Project Config", install_root / "config" / "default.yaml", "yaml"))
        files.append(("DeepSpeed Zero1", install_root / "config" / "deepspeed_zero1.json", "json"))
        files.append(("DeepSpeed Zero2", install_root / "config" / "deepspeed_zero2.json", "json"))
        files.append(("DeepSpeed Zero3", install_root / "config" / "deepspeed_zero3.json", "json"))
        files.append(("MCP Servers", install_root / "config" / "mcp_servers.json", "json"))
        
        # State files
        state_path = system_paths.get_state_file_path()
        files.append(("GUI State", state_path, "json"))
        
    except ImportError:
        # Fallback to current directory
        cwd = Path.cwd()
        files.append(("config/default.yaml", cwd / "config" / "default.yaml", "yaml"))
        files.append(("logging.yaml", cwd / "logging.yaml", "yaml"))
    
    return files


def _validate_config_file(
    name: str,
    path: Path,
    file_type: str,
    *,
    auto_repair: bool = False,
) -> DiagnosticResult:
    """Validate a single config file."""
    details = {"path": str(path), "type": file_type}
    
    try:
        content = path.read_text(encoding="utf-8")
        
        if not content.strip():
            return DiagnosticResult(
                name=name,
                severity=DiagnosticSeverity.WARNING,
                message="File is empty",
                details=details,
            )
        
        if file_type == "json":
            parsed = json.loads(content)
            details["keys"] = list(parsed.keys()) if isinstance(parsed, dict) else f"<{type(parsed).__name__}>"
            
        elif file_type == "yaml":
            try:
                import yaml
                parsed = yaml.safe_load(content)
                if isinstance(parsed, dict):
                    details["keys"] = list(parsed.keys())
            except ImportError:
                # YAML not available, just check syntax roughly
                if not content.strip().startswith(("#", "-", " ", "\n")) and ":" not in content:
                    return DiagnosticResult(
                        name=name,
                        severity=DiagnosticSeverity.WARNING,
                        message="May be invalid YAML (could not parse)",
                        details=details,
                    )
        
        # Get file size
        details["size_bytes"] = path.stat().st_size
        
        return DiagnosticResult(
            name=name,
            severity=DiagnosticSeverity.OK,
            message="Valid",
            details=details,
        )
        
    except json.JSONDecodeError as e:
        details["error"] = str(e)
        details["line"] = e.lineno
        details["column"] = e.colno
        
        return DiagnosticResult(
            name=name,
            severity=DiagnosticSeverity.ERROR,
            message=f"Invalid JSON at line {e.lineno}: {e.msg}",
            details=details,
            suggestion="Fix JSON syntax error or delete file to use defaults",
            auto_fixable=auto_repair,
        )
        
    except Exception as e:
        details["error"] = str(e)
        
        return DiagnosticResult(
            name=name,
            severity=DiagnosticSeverity.ERROR,
            message=f"Error reading file: {e}",
            details=details,
        )


def _check_required_configs() -> list[DiagnosticResult]:
    """Check for required configuration files."""
    results = []
    
    try:
        from aios.system import paths as system_paths
        install_root = system_paths.get_install_root()
        
        # Check pyproject.toml exists (project marker)
        pyproject = install_root / "pyproject.toml"
        if pyproject.exists():
            results.append(DiagnosticResult(
                name="Project File",
                severity=DiagnosticSeverity.OK,
                message="pyproject.toml found",
                details={"path": str(pyproject)},
            ))
        else:
            results.append(DiagnosticResult(
                name="Project File",
                severity=DiagnosticSeverity.WARNING,
                message="pyproject.toml not found",
                details={"expected_path": str(pyproject)},
                suggestion="AI-OS may not be properly installed",
            ))
        
        # Check brains registry
        brains_root = system_paths.get_brains_root()
        masters_json = brains_root / "masters.json"
        
        if masters_json.exists():
            try:
                content = json.loads(masters_json.read_text(encoding="utf-8"))
                brain_count = len(content) if isinstance(content, (list, dict)) else 0
                results.append(DiagnosticResult(
                    name="Brain Registry",
                    severity=DiagnosticSeverity.OK,
                    message=f"{brain_count} brain(s) registered",
                    details={"path": str(masters_json)},
                ))
            except Exception as e:
                results.append(DiagnosticResult(
                    name="Brain Registry",
                    severity=DiagnosticSeverity.WARNING,
                    message=f"Could not parse: {e}",
                    details={"path": str(masters_json)},
                ))
        else:
            results.append(DiagnosticResult(
                name="Brain Registry",
                severity=DiagnosticSeverity.INFO,
                message="No brains registered yet",
                details={"path": str(masters_json)},
            ))
            
    except ImportError:
        pass
    
    return results
