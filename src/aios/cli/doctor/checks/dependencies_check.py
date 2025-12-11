"""Dependency checking for AI-OS Doctor."""

from __future__ import annotations

import importlib.metadata
import logging
import sys
from typing import Any, Optional

from ..runner import DiagnosticResult, DiagnosticSeverity

logger = logging.getLogger(__name__)


# Core dependencies that must be installed
CORE_DEPENDENCIES = [
    ("torch", "PyTorch", "2.1.0", "pip install torch"),
    ("transformers", "Transformers", "4.41.0", "pip install transformers"),
    ("accelerate", "Accelerate", "0.31.0", "pip install accelerate"),
    ("peft", "PEFT", "0.11.1", "pip install peft"),
    ("datasets", "Datasets", "2.14.0", "pip install datasets"),
    ("huggingface_hub", "HuggingFace Hub", "0.19.0", "pip install huggingface_hub"),
    ("safetensors", "Safetensors", "0.3.1", "pip install safetensors"),
]

# UI dependencies
UI_DEPENDENCIES = [
    ("tkinterweb", "TkinterWeb", "3.23.8", "pip install tkinterweb>=3.23.8"),
    ("matplotlib", "Matplotlib", "3.7.0", "pip install matplotlib"),
    ("PIL", "Pillow", "10.0.0", "pip install Pillow>=10.0.0"),
]

# Optional but recommended
OPTIONAL_DEPENDENCIES = [
    ("lm_eval", "LM Eval", "0.4.0", "pip install lm-eval[api]"),
    ("psutil", "psutil", None, "pip install psutil"),
    ("flash_attn", "Flash Attention", "2.3.0", "pip install flash-attn (requires CUDA)"),
    ("bitsandbytes", "bitsandbytes", "0.43.0", "pip install bitsandbytes"),
    ("deepspeed", "DeepSpeed", "0.14.0", "pip install deepspeed (Linux only)"),
    ("sentencepiece", "SentencePiece", "0.1.99", "pip install sentencepiece"),
    ("protobuf", "Protobuf", "3.20.0", "pip install protobuf"),
]

# Utility dependencies
UTILITY_DEPENDENCIES = [
    ("typer", "Typer", None, "pip install typer"),
    ("rich", "Rich", None, "pip install rich"),
    ("pydantic", "Pydantic", "2.0.0", "pip install pydantic>=2"),
    ("httpx", "HTTPX", "0.27", "pip install httpx>=0.27"),
    ("orjson", "orjson", None, "pip install orjson"),
    ("yaml", "PyYAML", None, "pip install PyYAML"),
    ("tenacity", "Tenacity", None, "pip install tenacity"),
    ("watchdog", "Watchdog", None, "pip install watchdog"),
]


def check_dependencies(*, auto_repair: bool = False) -> list[DiagnosticResult]:
    """Check all Python package dependencies."""
    results = []
    
    # Check core dependencies (required)
    results.append(DiagnosticResult(
        name="Core Dependencies",
        severity=DiagnosticSeverity.INFO,
        message="Checking required packages...",
    ))
    
    for module_name, display_name, min_version, install_cmd in CORE_DEPENDENCIES:
        result = _check_package(module_name, display_name, min_version, install_cmd, required=True)
        results.append(result)
    
    # Check UI dependencies
    results.append(DiagnosticResult(
        name="UI Dependencies",
        severity=DiagnosticSeverity.INFO,
        message="Checking UI packages...",
    ))
    
    for module_name, display_name, min_version, install_cmd in UI_DEPENDENCIES:
        result = _check_package(module_name, display_name, min_version, install_cmd, required=False)
        results.append(result)
    
    # Check optional dependencies
    results.append(DiagnosticResult(
        name="Optional Dependencies",
        severity=DiagnosticSeverity.INFO,
        message="Checking optional packages...",
    ))
    
    for module_name, display_name, min_version, install_cmd in OPTIONAL_DEPENDENCIES:
        result = _check_package(module_name, display_name, min_version, install_cmd, required=False, optional=True)
        results.append(result)
    
    # Check utility dependencies
    results.append(DiagnosticResult(
        name="Utility Dependencies",
        severity=DiagnosticSeverity.INFO,
        message="Checking utility packages...",
    ))
    
    for module_name, display_name, min_version, install_cmd in UTILITY_DEPENDENCIES:
        result = _check_package(module_name, display_name, min_version, install_cmd, required=False)
        results.append(result)
    
    return results


def _check_package(
    module_name: str,
    display_name: str,
    min_version: Optional[str],
    install_cmd: str,
    required: bool = True,
    optional: bool = False,
) -> DiagnosticResult:
    """Check if a package is installed and meets version requirements."""
    
    # Handle special module name mappings
    package_name_map = {
        "PIL": "Pillow",
        "yaml": "PyYAML",
        "cv2": "opencv-python",
        "sklearn": "scikit-learn",
    }
    
    package_name = package_name_map.get(module_name, module_name.replace("_", "-"))
    
    # Try to get version from metadata first
    version = None
    try:
        version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        # Try alternative names
        for alt_name in [module_name, module_name.replace("-", "_"), module_name.replace("_", "-")]:
            try:
                version = importlib.metadata.version(alt_name)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
    
    # If still no version, try importing
    if version is None:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
        except ImportError:
            pass
    
    if version is None:
        # Package not found
        if required:
            return DiagnosticResult(
                name=display_name,
                severity=DiagnosticSeverity.ERROR,
                message="MISSING (required)",
                suggestion=install_cmd,
                auto_fixable=True,
            )
        elif optional:
            return DiagnosticResult(
                name=display_name,
                severity=DiagnosticSeverity.INFO,
                message="Not installed (optional)",
                suggestion=install_cmd,
            )
        else:
            return DiagnosticResult(
                name=display_name,
                severity=DiagnosticSeverity.WARNING,
                message="MISSING",
                suggestion=install_cmd,
                auto_fixable=True,
            )
    
    # Package found, check version
    details = {"version": version, "package": package_name}
    
    if min_version and version != "unknown":
        try:
            if _compare_versions(version, min_version) < 0:
                return DiagnosticResult(
                    name=display_name,
                    severity=DiagnosticSeverity.WARNING,
                    message=f"v{version} (outdated, need {min_version}+)",
                    details=details,
                    suggestion=f"Upgrade: {install_cmd}",
                    auto_fixable=True,
                )
        except Exception:
            pass  # Version comparison failed, assume OK
    
    return DiagnosticResult(
        name=display_name,
        severity=DiagnosticSeverity.OK,
        message=f"v{version}",
        details=details,
    )


def _compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings. Returns -1 if v1 < v2, 0 if equal, 1 if v1 > v2."""
    def normalize(v):
        # Strip common prefixes and suffixes
        v = v.lstrip("v").split("+")[0].split("-")[0]
        parts = []
        for part in v.split("."):
            # Handle alpha/beta/rc suffixes
            if part.isdigit():
                parts.append(int(part))
            else:
                # Extract numeric prefix
                num = ""
                for c in part:
                    if c.isdigit():
                        num += c
                    else:
                        break
                if num:
                    parts.append(int(num))
        return parts
    
    v1_parts = normalize(v1)
    v2_parts = normalize(v2)
    
    # Pad to same length
    max_len = max(len(v1_parts), len(v2_parts))
    v1_parts.extend([0] * (max_len - len(v1_parts)))
    v2_parts.extend([0] * (max_len - len(v2_parts)))
    
    for a, b in zip(v1_parts, v2_parts):
        if a < b:
            return -1
        elif a > b:
            return 1
    return 0
