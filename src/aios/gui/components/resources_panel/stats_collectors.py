"""Resource statistics collectors - CPU, RAM, GPU, Network, Disk stats."""

from __future__ import annotations

import os
import shutil
import subprocess as _sp
import time
from typing import Any

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore


def get_cpu_stats() -> tuple[float, str]:
    """Get CPU utilization and temperature.
    
    Returns:
        Tuple of (utilization_percent, temperature_string)
    """
    cpu_usage = 0.0
    cpu_temp = "N/A"
    
    try:
        if psutil:
            cpu_usage = float(psutil.cpu_percent(interval=0.1))
            # Try to get CPU temperature
            try:
                temps = psutil.sensors_temperatures()  # type: ignore[attr-defined]
                if temps:
                    # Try common sensor names
                    for sensor_name in ["coretemp", "k10temp", "zenpower", "cpu_thermal"]:
                        if sensor_name in temps:
                            entries = temps[sensor_name]
                            if entries:
                                cpu_temp = f"{entries[0].current:.0f}°C"
                                break
                    # If no known sensor, use first available
                    if cpu_temp == "N/A":
                        first_sensor = list(temps.values())[0]
                        if first_sensor:
                            cpu_temp = f"{first_sensor[0].current:.0f}°C"
            except Exception:
                pass
    except Exception:
        pass
    
    return cpu_usage, cpu_temp


def get_ram_stats() -> tuple[float, float, float]:
    """Get RAM usage statistics.
    
    Returns:
        Tuple of (usage_percent, used_gb, total_gb)
    """
    ram_usage = 0.0
    ram_used = 0.0
    ram_total = 1.0
    
    try:
        if psutil:
            mem = psutil.virtual_memory()
            ram_usage = float(mem.percent)
            ram_used = float(mem.used) / (1024**3)  # Convert to GB
            ram_total = float(mem.total) / (1024**3)  # Convert to GB
    except Exception:
        pass
    
    return ram_usage, ram_used, ram_total


def get_network_stats(last_net_io: Any = None, last_io_time: float = 0.0) -> tuple[float, float, Any]:
    """Get network upload/download rates in MB/s.
    
    Args:
        last_net_io: Previous network I/O counters
        last_io_time: Time of last measurement
    
    Returns:
        Tuple of (upload_mb_per_sec, download_mb_per_sec, current_net_io)
    """
    upload_rate = 0.0
    download_rate = 0.0
    current_io = None
    
    try:
        if psutil:
            current_io = psutil.net_io_counters()
            current_time = time.time()
            
            if last_net_io is not None and last_io_time > 0:
                time_diff = current_time - last_io_time
                if time_diff > 0:
                    bytes_sent = current_io.bytes_sent - last_net_io.bytes_sent
                    bytes_recv = current_io.bytes_recv - last_net_io.bytes_recv
                    upload_rate = max(0.0, bytes_sent / time_diff / (1024**2))  # MB/s
                    download_rate = max(0.0, bytes_recv / time_diff / (1024**2))  # MB/s
    except Exception:
        pass
    
    return upload_rate, download_rate, current_io


def get_disk_stats(last_disk_io: Any = None, last_io_time: float = 0.0) -> tuple[float, float, Any, float]:
    """Get disk read/write rates in MB/s.
    
    Args:
        last_disk_io: Previous disk I/O counters
        last_io_time: Time of last measurement
    
    Returns:
        Tuple of (read_mb_per_sec, write_mb_per_sec, current_disk_io, current_time)
    """
    read_rate = 0.0
    write_rate = 0.0
    current_io = None
    current_time = time.time()
    
    try:
        if psutil:
            current_io = psutil.disk_io_counters()
            
            if current_io and last_disk_io is not None and last_io_time > 0:
                time_diff = current_time - last_io_time
                if time_diff > 0:
                    bytes_read = current_io.read_bytes - last_disk_io.read_bytes
                    bytes_write = current_io.write_bytes - last_disk_io.write_bytes
                    read_rate = max(0.0, bytes_read / time_diff / (1024**2))  # MB/s
                    write_rate = max(0.0, bytes_write / time_diff / (1024**2))  # MB/s
    except Exception:
        pass
    
    return read_rate, write_rate, current_io, current_time


def get_gpu_stats() -> list[dict[str, Any]]:
    """Get GPU statistics for all available GPUs.
    
    Returns:
        List of dicts with keys: index, name, util, mem_used_mb, mem_total_mb, temp
    """
    gpu_stats: list[dict[str, Any]] = []
    
    # Try nvidia-smi first (gives us temp)
    try:
        nvsmi = shutil.which("nvidia-smi")
        if nvsmi:
            res = _sp.run(
                [
                    nvsmi,
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=1.0,  # Reduced from 2.0s to 1.0s to avoid blocking GUI
            )
            if res.stdout:
                for line in res.stdout.strip().splitlines():
                    try:
                        parts = [x.strip() for x in line.split(",")]
                        if len(parts) >= 6:
                            gpu_stats.append({
                                "index": int(parts[0]),
                                "name": parts[1],
                                "util": float(parts[2]),
                                "mem_used_mb": float(parts[3]),
                                "mem_total_mb": float(parts[4]),
                                "temp": float(parts[5]),
                            })
                    except Exception:
                        continue
                return gpu_stats
    except Exception:
        pass

    # Fallback to torch.cuda (no temperature available)
    try:
        import torch  # type: ignore

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            n = int(torch.cuda.device_count())
            for i in range(n):
                try:
                    name = torch.cuda.get_device_name(i)
                    free_b, total_b = torch.cuda.mem_get_info(i)
                    used_b = max(0, int(total_b - free_b))
                    gpu_stats.append({
                        "index": i,
                        "name": name,
                        "util": 0.0,  # torch doesn't provide utilization
                        "mem_used_mb": used_b / (1024**2),
                        "mem_total_mb": total_b / (1024**2),
                        "temp": None,
                    })
                except Exception:
                    continue
    except Exception:
        pass

    return gpu_stats
