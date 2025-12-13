from __future__ import annotations

from pathlib import Path
from typing import Optional
import shutil
import subprocess as _sp

from rich import print

from aios.core.gpu_vendor import identify_gpu_vendor, detect_xpu_devices, calculate_vendor_summary


def torch_info() -> None:
    try:
        import torch  # type: ignore

        try:
            xpu_avail = bool(getattr(torch, "xpu", None) and torch.xpu.is_available())  # type: ignore[attr-defined]
            xpu_count = int(torch.xpu.device_count()) if xpu_avail else 0  # type: ignore[attr-defined]
        except Exception:
            xpu_avail, xpu_count = False, 0
        try:
            mps_avail = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())  # type: ignore[attr-defined]
        except Exception:
            mps_avail = False
        dml_avail = False
        dml_python_path: Optional[str] = None
        try:
            import torch_directml as _dml  # type: ignore
            _ = _dml.device()
            dml_avail = True
        except Exception:
            dml_avail = False
        if not dml_avail:
            try:
                cfg_path = Path.home() / ".config/aios/dml_python.txt"
                if cfg_path.exists():
                    dml_python_path = cfg_path.read_text(encoding="utf-8").strip()
                    if dml_python_path:
                        rc = _sp.run([dml_python_path, "-c", "import torch_directml, sys; sys.exit(0)"], check=False)
                        dml_avail = (rc.returncode == 0)
            except Exception:
                pass
        try:
            rocm = bool(getattr(torch.version, "hip", None))  # type: ignore[attr-defined]
        except Exception:
            rocm = False

        # CUDA runtime availability and device details (via torch when possible)
        try:
            cuda_runtime = bool(torch.cuda.is_available())
        except Exception:
            cuda_runtime = False
        cuda_count = 0
        cuda_devices: list[dict] = []
        if cuda_runtime:
            try:
                cuda_count = int(torch.cuda.device_count())
                for i in range(max(0, cuda_count)):
                    try:
                        name = str(torch.cuda.get_device_name(i))
                    except Exception:
                        name = f"CUDA {i}"
                    total_mb = None
                    try:
                        props = torch.cuda.get_device_properties(i)
                        total_mb = int(getattr(props, "total_memory", 0) // (1024**2))
                    except Exception:
                        total_mb = None
                    dev: dict = {"id": i, "name": name}
                    if total_mb:
                        dev["total_mem_mb"] = int(total_mb)
                    # Add vendor identification
                    dev["vendor"] = identify_gpu_vendor(name, check_rocm=rocm)
                    cuda_devices.append(dev)
            except Exception:
                cuda_devices = []

        # Detect Intel XPU devices
        xpu_avail_detected, xpu_devices = detect_xpu_devices()
        if xpu_avail_detected:
            xpu_avail = True
            xpu_count = len(xpu_devices)

        # Fallback: detect NVIDIA adapters via nvidia-smi even if torch was built without CUDA
        nvsmi_found = False
        nvsmi_devices: list[dict] = []
        try:
            nvsmi = shutil.which("nvidia-smi")
            if nvsmi:
                nvsmi_found = True
                try:
                    res = _sp.run(
                        [
                            nvsmi,
                            "--query-gpu=index,name,memory.total",
                            "--format=csv,noheader,nounits",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=2.0,
                    )
                    out = (res.stdout or "").strip()
                    if out:
                        for line in out.splitlines():
                            parts = [p.strip() for p in line.split(",")]
                            if len(parts) >= 3:
                                try:
                                    idx = int(parts[0])
                                except Exception:
                                    continue
                                name = parts[1]
                                try:
                                    mem_mb = int(float(parts[2]))
                                except Exception:
                                    mem_mb = None
                                d = {"id": idx, "name": name, "vendor": identify_gpu_vendor(name)}
                                if mem_mb is not None:
                                    d["total_mem_mb"] = mem_mb
                                nvsmi_devices.append(d)
                except Exception:
                    pass
        except Exception:
            pass
        
        # Calculate vendor summary for all devices
        all_gpu_devices = cuda_devices + xpu_devices
        vendor_summary = calculate_vendor_summary(all_gpu_devices)
        
        info = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": bool(cuda_runtime),
            "cuda_device_count": int(cuda_count if cuda_runtime else 0),
            "cuda_devices": cuda_devices,
            "nvidia_smi_found": nvsmi_found,
            "nvidia_smi_devices": nvsmi_devices,
            "xpu_available": xpu_avail,
            "xpu_device_count": xpu_count,
            "xpu_devices": xpu_devices,
            "mps_available": mps_avail,
            "directml_available": dml_avail,
            "directml_python": dml_python_path,
            "rocm": rocm,
            "gpu_vendor_summary": vendor_summary,
        }
        print(info)
    except Exception as e:
        print({"installed": False, "error": str(e)})
