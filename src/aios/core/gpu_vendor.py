"""GPU vendor identification and detection utilities.

This module provides utilities for identifying GPU vendors (NVIDIA, AMD, Intel)
from device names and system information. Used by torch-info, GUI, and training
components to correctly label and handle different GPU backends.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Tuple

# GPU vendor type for type hints
GpuVendor = Literal["NVIDIA", "AMD", "Intel", "Unknown"]

# Vendor keywords for detection (case-insensitive matching)
_NVIDIA_KEYWORDS = frozenset([
    "nvidia", "geforce", "rtx", "gtx", "quadro", "tesla", 
    "a100", "h100", "l40", "a40", "v100", "p100", "t4"
])

_AMD_KEYWORDS = frozenset([
    "amd", "radeon", "rx ", "vega", "navi", "mi50", "mi100", 
    "mi200", "mi210", "mi250", "mi300", "instinct", "w7900", "w6800"
])

_INTEL_KEYWORDS = frozenset([
    "intel", "arc ", "a770", "a750", "a380", "xe", "iris", "uhd", "max"
])


def identify_gpu_vendor(gpu_name: str, check_rocm: bool = False) -> GpuVendor:
    """Identify GPU vendor from device name.
    
    Checks the GPU name against known vendor keywords and optionally
    checks for ROCm build indicators to identify AMD GPUs that might
    be reported as generic CUDA devices.
    
    Args:
        gpu_name: GPU device name from torch or system info
        check_rocm: Also check torch.version.hip attribute for ROCm builds
        
    Returns:
        'NVIDIA' | 'AMD' | 'Intel' | 'Unknown'
        
    Examples:
        >>> identify_gpu_vendor("NVIDIA GeForce RTX 4090")
        'NVIDIA'
        >>> identify_gpu_vendor("AMD Radeon RX 7900 XTX")
        'AMD'
        >>> identify_gpu_vendor("Intel Arc A770")
        'Intel'
        >>> identify_gpu_vendor("gfx1100", check_rocm=True)  # AMD ROCm identifier
        'AMD'
    """
    name_lower = gpu_name.lower()
    
    # NVIDIA detection
    if any(kw in name_lower for kw in _NVIDIA_KEYWORDS):
        return "NVIDIA"
    
    # AMD detection - also check for gfx patterns (ROCm device identifiers)
    if any(kw in name_lower for kw in _AMD_KEYWORDS):
        return "AMD"
    if name_lower.startswith("gfx"):  # ROCm GPU identifiers like gfx1100
        return "AMD"
    
    # Intel detection
    if any(kw in name_lower for kw in _INTEL_KEYWORDS):
        return "Intel"
    
    # Check ROCm build as fallback for AMD detection
    if check_rocm:
        try:
            import torch
            if getattr(torch.version, "hip", None):
                return "AMD"
        except Exception:
            pass
    
    return "Unknown"


def calculate_vendor_summary(devices: List[Dict]) -> Dict[str, int]:
    """Calculate vendor breakdown from list of device dicts.
    
    Args:
        devices: List of device dicts with optional 'vendor' key
        
    Returns:
        Dict mapping vendor name to count, e.g. {"NVIDIA": 2, "AMD": 1}
    """
    vendor_counts: Dict[str, int] = {}
    for dev in devices:
        vendor = dev.get("vendor", "Unknown")
        if not vendor:
            vendor = "Unknown"
        vendor_counts[vendor] = vendor_counts.get(vendor, 0) + 1
    return vendor_counts


def detect_xpu_devices() -> Tuple[bool, List[Dict]]:
    """Detect Intel XPU devices if available.
    
    Returns:
        Tuple of (xpu_available: bool, devices: list of device dicts)
        Each device dict has: id, name, total_mem_mb, vendor
    """
    xpu_available = False
    xpu_devices: List[Dict] = []
    
    try:
        import torch
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            xpu_available = True
            device_count = torch.xpu.device_count()
            for i in range(device_count):
                try:
                    props = torch.xpu.get_device_properties(i)
                    name = getattr(props, "name", f"Intel XPU {i}")
                    total_mem = getattr(props, "total_memory", 0)
                    total_mem_mb = int(total_mem // (1024 * 1024)) if total_mem else 0
                    xpu_devices.append({
                        "id": i,
                        "name": name,
                        "total_mem_mb": total_mem_mb,
                        "vendor": "Intel",
                        "backend": "xpu"
                    })
                except Exception:
                    # Fallback for devices we can't query
                    xpu_devices.append({
                        "id": i,
                        "name": f"Intel XPU {i}",
                        "total_mem_mb": 0,
                        "vendor": "Intel",
                        "backend": "xpu"
                    })
    except ImportError:
        # intel-extension-for-pytorch not installed
        pass
    except Exception:
        # torch.xpu not available or other error
        pass
    
    return xpu_available, xpu_devices


def get_xpu_memory_info(device_id: int = 0) -> Tuple[float, float]:
    """Get XPU memory info for a specific device.
    
    Args:
        device_id: XPU device index
        
    Returns:
        Tuple of (total_gb, available_gb)
    """
    try:
        import torch
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            props = torch.xpu.get_device_properties(device_id)
            total_gb = getattr(props, "total_memory", 0) / (1024 ** 3)
            
            # Try to get available memory
            try:
                free_mem, total_mem = torch.xpu.mem_get_info(device_id)
                available_gb = free_mem / (1024 ** 3)
            except Exception:
                # Estimate 90% available if mem_get_info not available
                available_gb = total_gb * 0.9
            
            return total_gb, available_gb
    except Exception:
        pass
    
    return 0.0, 0.0


def detect_all_gpus() -> Dict:
    """Detect all available GPUs across all backends.
    
    Returns comprehensive GPU information including CUDA (NVIDIA/AMD ROCm),
    XPU (Intel), and vendor summary.
    
    Returns:
        Dict with:
        - cuda_available: bool
        - cuda_devices: list of device dicts with vendor field
        - xpu_available: bool
        - xpu_devices: list of device dicts
        - vendor_summary: dict of vendor counts
        - rocm: bool (whether ROCm build is detected)
    """
    result = {
        "cuda_available": False,
        "cuda_devices": [],
        "xpu_available": False,
        "xpu_devices": [],
        "vendor_summary": {},
        "rocm": False
    }
    
    try:
        import torch
        
        # Check for ROCm build
        rocm = bool(getattr(torch.version, "hip", None))
        result["rocm"] = rocm
        
        # Detect CUDA devices (includes NVIDIA and AMD ROCm)
        if torch.cuda.is_available():
            result["cuda_available"] = True
            device_count = torch.cuda.device_count()
            
            for i in range(device_count):
                try:
                    name = torch.cuda.get_device_name(i)
                except Exception:
                    name = f"CUDA {i}"
                
                total_mem_mb = None
                try:
                    props = torch.cuda.get_device_properties(i)
                    total_mem_mb = int(props.total_memory // (1024 ** 2))
                except Exception:
                    pass
                
                vendor = identify_gpu_vendor(name, check_rocm=rocm)
                
                dev = {
                    "id": i,
                    "name": name,
                    "vendor": vendor,
                    "backend": "cuda"
                }
                if total_mem_mb:
                    dev["total_mem_mb"] = total_mem_mb
                    
                result["cuda_devices"].append(dev)
        
        # Detect XPU devices
        xpu_available, xpu_devices = detect_xpu_devices()
        result["xpu_available"] = xpu_available
        result["xpu_devices"] = xpu_devices
        
        # Calculate vendor summary across all devices
        all_devices = result["cuda_devices"] + result["xpu_devices"]
        result["vendor_summary"] = calculate_vendor_summary(all_devices)
        
    except ImportError:
        # torch not installed
        pass
    except Exception:
        pass
    
    return result
