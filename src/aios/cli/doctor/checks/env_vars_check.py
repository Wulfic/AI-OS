"""Environment variable check for AI-OS Doctor."""

from __future__ import annotations

import os
from typing import Optional

from ..runner import DiagnosticResult, DiagnosticSeverity


# AIOS-specific environment variables
AIOS_VARS = [
    "AIOS_INSTALL_ROOT",
    "AIOS_PROGRAM_DATA",
    "AIOS_USER_DATA",
    "AIOS_CACHE_DIR",
    "AIOS_CONFIG_DIR",
    "AIOS_ARTIFACTS_DIR",
    "AIOS_HF_CACHE",
    "AIOS_CONFIG",
    "AIOS_LOG_LEVEL",
    "AIOS_DIAGNOSTICS_DISABLED",
    "AIOS_DDP_BACKEND",
    "AIOS_DDP_TIMEOUT_SEC",
]

# HuggingFace environment variables
HF_VARS = [
    "HF_HOME",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_CACHE",
    "HUGGING_FACE_HUB_TOKEN",
    "TRANSFORMERS_CACHE",
    "HF_DATASETS_CACHE",
]

# CUDA/GPU environment variables
GPU_VARS = [
    "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME",
    "CUDA_PATH",
    "NVIDIA_VISIBLE_DEVICES",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCH_CUDA_ARCH_LIST",
]

# Python environment variables
PYTHON_VARS = [
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
]

# Proxy variables
PROXY_VARS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
]


def check_environment_variables(*, auto_repair: bool = False) -> list[DiagnosticResult]:
    """Check and report relevant environment variables."""
    results = []
    
    # AIOS variables
    aios_result = _check_var_group("AIOS Settings", AIOS_VARS)
    results.append(aios_result)
    
    # HuggingFace variables
    hf_result = _check_var_group("HuggingFace Settings", HF_VARS)
    results.append(hf_result)
    
    # GPU variables
    gpu_result = _check_var_group("GPU Settings", GPU_VARS)
    results.append(gpu_result)
    
    # Python environment
    python_result = _check_var_group("Python Environment", PYTHON_VARS)
    results.append(python_result)
    
    # Proxy settings
    proxy_result = _check_var_group("Proxy Settings", PROXY_VARS)
    results.append(proxy_result)
    
    # Check for potential issues
    issues = _check_env_issues()
    results.extend(issues)
    
    return results


def _check_var_group(name: str, variables: list[str]) -> DiagnosticResult:
    """Check a group of environment variables."""
    found = {}
    
    for var in variables:
        value = os.environ.get(var)
        if value:
            # Mask sensitive values
            if "TOKEN" in var.upper() or "KEY" in var.upper() or "SECRET" in var.upper():
                if len(value) > 8:
                    value = value[:4] + "..." + value[-4:]
                else:
                    value = "***"
            elif "PASSWORD" in var.upper():
                value = "***"
            found[var] = value
    
    if found:
        return DiagnosticResult(
            name=name,
            severity=DiagnosticSeverity.INFO,
            message=f"{len(found)} variable(s) set",
            details={"variables": found},
        )
    else:
        return DiagnosticResult(
            name=name,
            severity=DiagnosticSeverity.INFO,
            message="No custom settings",
        )


def _check_env_issues() -> list[DiagnosticResult]:
    """Check for potential environment variable issues."""
    results = []
    
    # Check for conflicting CUDA settings
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    nvidia_visible = os.environ.get("NVIDIA_VISIBLE_DEVICES")
    
    if cuda_visible and nvidia_visible and cuda_visible != nvidia_visible:
        results.append(DiagnosticResult(
            name="CUDA Config Conflict",
            severity=DiagnosticSeverity.WARNING,
            message="CUDA_VISIBLE_DEVICES and NVIDIA_VISIBLE_DEVICES differ",
            details={
                "CUDA_VISIBLE_DEVICES": cuda_visible,
                "NVIDIA_VISIBLE_DEVICES": nvidia_visible,
            },
            suggestion="Set both to the same value or remove one",
        ))
    
    # Check for deprecated TRANSFORMERS_CACHE
    if os.environ.get("TRANSFORMERS_CACHE"):
        results.append(DiagnosticResult(
            name="Deprecated Variable",
            severity=DiagnosticSeverity.INFO,
            message="TRANSFORMERS_CACHE is deprecated",
            suggestion="Use HF_HOME instead: export HF_HOME=/path/to/cache",
        ))
    
    # Check CUDA_VISIBLE_DEVICES format
    if cuda_visible:
        try:
            # Should be comma-separated integers or empty
            if cuda_visible.strip() and cuda_visible != "-1":
                parts = cuda_visible.split(",")
                for part in parts:
                    int(part.strip())
        except ValueError:
            results.append(DiagnosticResult(
                name="CUDA Config Issue",
                severity=DiagnosticSeverity.WARNING,
                message=f"Invalid CUDA_VISIBLE_DEVICES format: {cuda_visible}",
                suggestion="Should be comma-separated GPU indices (e.g., '0,1,2')",
            ))
    
    # Check if virtual environment is active
    venv = os.environ.get("VIRTUAL_ENV")
    conda = os.environ.get("CONDA_PREFIX")
    
    if not venv and not conda:
        results.append(DiagnosticResult(
            name="Python Environment",
            severity=DiagnosticSeverity.INFO,
            message="No virtual environment detected",
            suggestion="Consider using a virtual environment for isolation",
        ))
    
    return results
