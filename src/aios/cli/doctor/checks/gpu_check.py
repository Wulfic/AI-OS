"""GPU and hardware detection check for AI-OS Doctor."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from typing import Any, Optional

from ..runner import DiagnosticResult, DiagnosticSeverity

logger = logging.getLogger(__name__)


def check_gpu(*, auto_repair: bool = False) -> list[DiagnosticResult]:
    """Check GPU availability and configuration."""
    results = []
    
    # Check for torch first
    torch_result, torch = _check_torch()
    results.append(torch_result)
    
    if torch is None:
        return results
    
    # Check CUDA
    cuda_results = _check_cuda(torch)
    results.extend(cuda_results)
    
    # Check other backends
    results.extend(_check_other_backends(torch))
    
    # Check nvidia-smi if available
    nvsmi_result = _check_nvidia_smi()
    if nvsmi_result:
        results.append(nvsmi_result)
    
    return results


def _check_torch() -> tuple[DiagnosticResult, Optional[Any]]:
    """Check if PyTorch is installed and get version info."""
    try:
        import torch
        
        details = {
            "version": torch.__version__,
            "cuda_built": torch.cuda.is_available() if hasattr(torch, "cuda") else False,
        }
        
        # Check if built with CUDA
        cuda_version = getattr(torch.version, "cuda", None)
        if cuda_version:
            details["cuda_version"] = cuda_version
        
        cudnn_version = getattr(torch.backends, "cudnn", None)
        if cudnn_version and hasattr(cudnn_version, "version"):
            try:
                details["cudnn_version"] = cudnn_version.version()
            except Exception:
                pass
        
        return DiagnosticResult(
            name="PyTorch",
            severity=DiagnosticSeverity.OK,
            message=f"v{torch.__version__}",
            details=details,
        ), torch
        
    except ImportError:
        return DiagnosticResult(
            name="PyTorch",
            severity=DiagnosticSeverity.ERROR,
            message="PyTorch not installed",
            suggestion="Install PyTorch: pip install torch",
        ), None


def _check_cuda(torch) -> list[DiagnosticResult]:
    """Check CUDA availability and device information."""
    results = []
    
    if not torch.cuda.is_available():
        results.append(DiagnosticResult(
            name="CUDA",
            severity=DiagnosticSeverity.INFO,
            message="CUDA not available",
            details={"available": False},
            suggestion="Install CUDA-enabled PyTorch for GPU acceleration",
        ))
        return results
    
    # CUDA is available
    try:
        device_count = torch.cuda.device_count()
        cuda_version = torch.version.cuda or "unknown"
        
        results.append(DiagnosticResult(
            name="CUDA Runtime",
            severity=DiagnosticSeverity.OK,
            message=f"CUDA {cuda_version} with {device_count} GPU(s)",
            details={
                "available": True,
                "version": cuda_version,
                "device_count": device_count,
            },
        ))
        
        # Check cuDNN
        if hasattr(torch.backends, "cudnn"):
            cudnn = torch.backends.cudnn
            if cudnn.is_available():
                try:
                    cudnn_version = cudnn.version()
                    results.append(DiagnosticResult(
                        name="cuDNN",
                        severity=DiagnosticSeverity.OK,
                        message=f"cuDNN {cudnn_version}",
                        details={
                            "version": cudnn_version,
                            "enabled": cudnn.enabled,
                        },
                    ))
                except Exception:
                    results.append(DiagnosticResult(
                        name="cuDNN",
                        severity=DiagnosticSeverity.OK,
                        message="cuDNN available",
                    ))
        
        # Get individual GPU information
        for i in range(device_count):
            gpu_result = _get_gpu_info(torch, i)
            results.append(gpu_result)
        
    except Exception as e:
        results.append(DiagnosticResult(
            name="CUDA",
            severity=DiagnosticSeverity.ERROR,
            message=f"Error querying CUDA: {e}",
        ))
    
    return results


def _get_gpu_info(torch, device_id: int) -> DiagnosticResult:
    """Get information for a specific GPU."""
    try:
        name = torch.cuda.get_device_name(device_id)
        props = torch.cuda.get_device_properties(device_id)
        
        # Memory info
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(device_id)
            used_mem = total_mem - free_mem
            mem_gb = total_mem / (1024**3)
            used_gb = used_mem / (1024**3)
            free_gb = free_mem / (1024**3)
        except Exception:
            mem_gb = props.total_memory / (1024**3)
            used_gb = 0
            free_gb = mem_gb
        
        details = {
            "name": name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": round(mem_gb, 2),
            "free_memory_gb": round(free_gb, 2),
            "used_memory_gb": round(used_gb, 2),
            "multi_processor_count": props.multi_processor_count,
        }
        
        # Determine severity based on VRAM
        if mem_gb < 4:
            severity = DiagnosticSeverity.WARNING
            suggestion = "GPU has limited VRAM. Consider using smaller models or CPU."
        elif mem_gb < 8:
            severity = DiagnosticSeverity.OK
            suggestion = None
        else:
            severity = DiagnosticSeverity.OK
            suggestion = None
        
        return DiagnosticResult(
            name=f"GPU {device_id}",
            severity=severity,
            message=f"{name} ({mem_gb:.1f} GB VRAM)",
            details=details,
            suggestion=suggestion,
        )
        
    except Exception as e:
        return DiagnosticResult(
            name=f"GPU {device_id}",
            severity=DiagnosticSeverity.ERROR,
            message=f"Error querying GPU: {e}",
        )


def _check_other_backends(torch) -> list[DiagnosticResult]:
    """Check for other compute backends (XPU, MPS, DirectML)."""
    results = []
    
    # Intel XPU
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device_count = torch.xpu.device_count()
            results.append(DiagnosticResult(
                name="Intel XPU",
                severity=DiagnosticSeverity.OK,
                message=f"Available ({device_count} device(s))",
                details={"available": True, "device_count": device_count},
            ))
    except Exception:
        pass
    
    # Apple MPS
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            results.append(DiagnosticResult(
                name="Apple MPS",
                severity=DiagnosticSeverity.OK,
                message="Available",
                details={"available": True},
            ))
    except Exception:
        pass
    
    # DirectML (Windows)
    if platform.system() == "Windows":
        try:
            import torch_directml
            dml_device = torch_directml.device()
            results.append(DiagnosticResult(
                name="DirectML",
                severity=DiagnosticSeverity.OK,
                message="Available",
                details={"available": True, "device": str(dml_device)},
            ))
        except ImportError:
            results.append(DiagnosticResult(
                name="DirectML",
                severity=DiagnosticSeverity.INFO,
                message="Not installed",
                details={"available": False},
                suggestion="Install torch-directml for AMD/Intel GPU support on Windows",
            ))
        except Exception:
            pass
    
    return results


def _check_nvidia_smi() -> Optional[DiagnosticResult]:
    """Check nvidia-smi and get driver information."""
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return None
    
    try:
        # Get driver version
        result = subprocess.run(
            [nvsmi, "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if result.returncode == 0 and result.stdout.strip():
            driver_version = result.stdout.strip().split("\n")[0]
            
            # Get more GPU details
            result2 = subprocess.run(
                [nvsmi, "--query-gpu=gpu_name,memory.total,memory.free,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            details = {"driver_version": driver_version, "nvidia_smi_path": nvsmi}
            
            if result2.returncode == 0:
                gpu_details = []
                for line in result2.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        gpu_details.append({
                            "name": parts[0],
                            "total_memory_mb": parts[1],
                            "free_memory_mb": parts[2],
                            "temperature_c": parts[3],
                            "power_draw_w": parts[4],
                        })
                if gpu_details:
                    details["gpus"] = gpu_details
            
            return DiagnosticResult(
                name="NVIDIA Driver",
                severity=DiagnosticSeverity.OK,
                message=f"Driver v{driver_version}",
                details=details,
            )
    except subprocess.TimeoutExpired:
        return DiagnosticResult(
            name="NVIDIA Driver",
            severity=DiagnosticSeverity.WARNING,
            message="nvidia-smi timed out",
            suggestion="GPU may be busy or driver issue",
        )
    except Exception as e:
        return DiagnosticResult(
            name="NVIDIA Driver",
            severity=DiagnosticSeverity.WARNING,
            message=f"Failed to query nvidia-smi: {e}",
        )
    
    return None
