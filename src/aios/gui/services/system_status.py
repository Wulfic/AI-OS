from __future__ import annotations

import os
import shutil
import subprocess as _sp
import threading
import time
from typing import Any


class SystemStatusUpdater:
    """Encapsulates periodic system status updates for the status bar.

    This class is UI-agnostic; it only depends on the provided callbacks
    and a Tk-like scheduler (root.after) so it can be reused or tested.
    """

    def __init__(self, *, root, set_status_cb, resources_panel=None, worker_pool=None):
        self._root = root
        self._set_status = set_status_cb
        self._resources_panel = resources_panel
        self._worker_pool = worker_pool  # Store worker pool for async operations
        self._interval_ms = 1000
        self._running = False
        self._thread = None
        # Optional psutil
        try:
            import psutil  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            psutil = None  # type: ignore
        self._psutil = psutil

    def start(self, interval_ms: int = 1000) -> None:
        """Start the status updater with a background daemon thread.
        
        Args:
            interval_ms: Update interval in milliseconds (default 1000)
        """
        if self._running:
            return
        self._interval_ms = max(100, int(interval_ms))
        self._running = True
        
        if self._worker_pool:
            # Use worker pool for better resource management
            self._thread = self._worker_pool.submit(self._run_loop)
        else:
            # Fallback to threading
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop the status updater."""
        self._running = False

    def _run_loop(self) -> None:
        """Background loop that schedules updates on the main thread."""
        while self._running:
            try:
                self._root.after(0, self._update)
            except Exception:
                pass
            time.sleep(self._interval_ms / 1000.0)

    def _update(self) -> None:
        parts: list[str] = []
        # CPU
        try:
            if self._psutil:
                cpu = self._psutil.cpu_percent(interval=None)
                parts.append(f"CPU: {int(cpu)}%")
            else:
                parts.append("CPU: --")
        except Exception:
            parts.append("CPU: --")
        # RAM
        try:
            if self._psutil:
                vm = self._psutil.virtual_memory()
                used_gb = vm.used / (1024**3)
                total_gb = vm.total / (1024**3)
                parts.append(f"RAM: {int(vm.percent)}% ({used_gb:.1f}/{total_gb:.1f} GB)")
            else:
                parts.append("RAM: --")
        except Exception:
            parts.append("RAM: --")
        # GPUs (prefer nvidia-smi when available)
        gpu_parts: list[str] = []
        try:
            nvsmi = shutil.which("nvidia-smi")
            if nvsmi:
                try:
                    res = _sp.run(
                        [
                            nvsmi,
                            "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=1.5,
                    )
                    if res.stdout:
                        for idx, line in enumerate(res.stdout.strip().splitlines()):
                            util, used, total = [x.strip() for x in line.split(",")]
                            gpu_parts.append(f"GPU{idx}: {util}% {int(float(used))}/{int(float(total))} MB")
                except Exception:
                    pass
        except Exception:
            pass
        if not gpu_parts:
            try:
                import torch  # type: ignore

                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    n = int(torch.cuda.device_count())
                    for i in range(n):
                        try:
                            name = torch.cuda.get_device_name(i)
                        except Exception:
                            name = f"CUDA{i}"
                        try:
                            free_b, total_b = torch.cuda.mem_get_info(i)
                            used_b = max(0, int(total_b - free_b))
                            gpu_parts.append(f"{name}: {used_b // (1024**2)}/{total_b // (1024**2)} MB")
                        except Exception:
                            gpu_parts.append(f"{name}: mem N/A")
                else:
                    # As a hint for non-NVIDIA systems, note DML availability if known
                    try:
                        rp = getattr(self._resources_panel, "dml_var", None)
                        # Ensure rp has a 'get' method (e.g., a Tk variable) before calling
                        if rp is not None and hasattr(rp, "get"):
                            if bool(rp.get()):  # type: ignore[attr-defined]
                                gpu_parts.append("DML: available")
                    except Exception:
                        pass
            except Exception:
                pass
        if not gpu_parts:
            gpu_parts.append("GPU: --")
        parts.append(" | ".join(gpu_parts))

        # Datasets usage: size of training_data directory
        try:
            td_root = None
            try:
                # Project root where pyproject.toml lives
                cur = os.path.abspath(os.getcwd())
                for _ in range(8):
                    if os.path.exists(os.path.join(cur, "pyproject.toml")):
                        td_root = os.path.join(cur, "training_data")
                        break
                    parent = os.path.dirname(cur)
                    if parent == cur:
                        break
                    cur = parent
                if td_root is None:
                    td_root = os.path.join(os.path.abspath(os.getcwd()), "training_data")
            except Exception:
                td_root = os.path.join(os.path.abspath(os.getcwd()), "training_data")
            total_bytes = 0
            if os.path.isdir(td_root):
                for _root, _dirs, files in os.walk(td_root):
                    for fn in files:
                        try:
                            fp = os.path.join(_root, fn)
                            total_bytes += os.path.getsize(fp)
                        except Exception:
                            pass
            usage_gb = total_bytes / (1024.0 ** 3)
            # Dataset cap currently not surfaced here; show usage only
            parts.append(f"Datasets: {usage_gb:.1f}/-- GB")
        except Exception:
            parts.append("Datasets: --")

        try:
            self._set_status(" | ".join(parts))
        except Exception:
            pass
