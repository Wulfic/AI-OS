"""Device and dtype utilities for HF HRM Adapter.

Provides helpers for device selection and dtype configuration across different backends
(CUDA, XPU, MPS, DirectML, CPU).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple


def _require_torch_transformers():  # pragma: no cover - optional dependency
    """Validate that PyTorch and Transformers are installed."""
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for HF adapter but not installed") from e
    try:
        import transformers  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("transformers is required for HF adapter but not installed") from e


def pick_device(device: Optional[str] = None) -> Tuple[Any, bool, Any]:
    """Select optimal device for model inference.
    
    Args:
        device: Requested device ('auto', 'cuda', 'xpu', 'mps', 'dml', 'cpu', None)
        
    Returns:
        Tuple of (torch_device, is_dml, dml_device_object)
        - torch_device: torch.device instance or DML device object
        - is_dml: True if using DirectML backend
        - dml_device_object: DirectML device or None
        
    Device priority (auto mode): CUDA > XPU > MPS > DirectML > CPU
    """
    import torch
    
    req_dev = (device or "auto").strip().lower()
    
    # Explicit device selection
    if req_dev in ("cpu", "cuda", "xpu", "mps"):
        try:
            return torch.device(req_dev), False, None
        except Exception:
            return torch.device("cpu"), False, None
    
    if req_dev == "dml":
        try:
            import torch_directml as _dml  # type: ignore
            dml_dev = _dml.device()
            return dml_dev, True, dml_dev
        except Exception:
            return torch.device("cpu"), False, None
    
    # Auto detection (prefer CUDA > XPU > MPS > DirectML > CPU)
    try:
        if torch.cuda.is_available():
            return torch.device("cuda"), False, None
    except Exception:
        pass
    
    try:
        if getattr(torch, "xpu", None) and torch.xpu.is_available():  # type: ignore[attr-defined]
            return torch.device("xpu"), False, None
    except Exception:
        pass
    
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps"), False, None
    except Exception:
        pass
    
    try:
        import torch_directml as _dml  # type: ignore
        dml_dev = _dml.device()
        return dml_dev, True, dml_dev
    except Exception:
        pass
    
    return torch.device("cpu"), False, None


def choose_dtype(forward_dtype: str, device: Any, is_dml: bool) -> Any:
    """Select appropriate dtype based on device capabilities.
    
    Args:
        forward_dtype: Requested dtype name ('bfloat16', 'float32', etc.)
        device: torch.device instance
        is_dml: Whether using DirectML backend
        
    Returns:
        torch.dtype instance (bfloat16, float32, etc.)
        
    Note: BF16 is avoided on CPU, DirectML, and MPS for compatibility.
    """
    import torch
    
    chosen_dtype = getattr(torch, forward_dtype, torch.bfloat16)
    
    # Avoid BF16 on CPU/DML/MPS by default
    try:
        device_type = getattr(device, "type", "cpu")
        if device_type == "cpu" or is_dml or device_type == "mps":
            chosen_dtype = torch.float32
    except Exception:
        chosen_dtype = torch.float32
    
    return chosen_dtype
