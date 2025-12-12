from __future__ import annotations

import logging
import os
import shutil
import subprocess as _sp
import time
from typing import Any

from ..utils.resource_management import submit_background

logger = logging.getLogger(__name__)


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
        self._pending_future = None
        self._after_id = None
        # Optional psutil
        try:
            import psutil  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            logger.debug(f"psutil not available, using fallback stats: {e}")
            psutil = None  # type: ignore
        self._psutil = psutil

    def start(self, interval_ms: int = 1000) -> None:
        """Start the status updater with a background daemon thread.
        
        Args:
            interval_ms: Update interval in milliseconds (default 1000)
        """
        if self._running:
            return
        logger.debug(f"Starting system status monitoring thread (interval={interval_ms}ms)")
        self._interval_ms = max(100, int(interval_ms))
        self._running = True
        self._schedule_tick()

    def stop(self) -> None:
        """Stop the status updater."""
        logger.debug("Stopping system status monitoring")
        self._running = False
        self._pending_future = None
        if self._after_id is not None:
            try:
                self._root.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _schedule_tick(self) -> None:
        if not self._running:
            return
        try:
            self._after_id = self._root.after(self._interval_ms, self._queue_update)
        except Exception as e:
            logger.debug(f"Failed to schedule system status update: {e}")

    def _queue_update(self) -> None:
        if not self._running:
            return
        try:
            self._pending_future = submit_background(
                "system-status",
                self._collect_and_dispatch,
                pool=self._worker_pool,
            )
        except RuntimeError as exc:
            logger.error(f"System status queue error: {exc}")
            # As a fallback run synchronously but keep scheduling
            try:
                self._collect_and_dispatch()
            finally:
                self._schedule_tick()

    def _collect_and_dispatch(self) -> None:
        try:
            status_text = self._collect_metrics()
            self._root.after(0, lambda text=status_text: self._apply_and_reschedule(text))
        except Exception as exc:
            logger.error(f"Failed to gather system status: {exc}", exc_info=True)
            self._root.after(0, lambda: self._apply_and_reschedule("System status unavailable"))

    def _apply_and_reschedule(self, text: str) -> None:
        self._apply_status(text)
        self._after_id = None
        self._schedule_tick()

    def _apply_status(self, text: str) -> None:
        if not self._running:
            return
        try:
            self._set_status(text)
        except Exception as e:
            logger.debug(f"Failed to apply status text: {e}")

    def _collect_metrics(self) -> str:
        update_start = time.perf_counter()
        parts: list[str] = []
        # CPU
        try:
            if self._psutil:
                cpu = self._psutil.cpu_percent(interval=None)
                parts.append(f"CPU: {int(cpu)}%")
            else:
                parts.append("CPU: --")
        except Exception as e:
            logger.error(f"Failed to get CPU stats: {e}")
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
        except Exception as e:
            logger.error(f"Failed to get RAM stats: {e}")
            parts.append("RAM: --")
        # GPUs (prefer nvidia-smi when available)
        gpu_section_start = time.perf_counter()
        gpu_parts: list[str] = []
        try:
            nvsmi = shutil.which("nvidia-smi")
            if nvsmi:
                try:
                    nv_start = time.perf_counter()
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
                    nv_duration = time.perf_counter() - nv_start
                    if nv_duration > 0.5:
                        logger.debug(f"nvidia-smi GPU query latency: {nv_duration:.3f}s")
                    if res.stdout:
                        for idx, line in enumerate(res.stdout.strip().splitlines()):
                            util, used, total = [x.strip() for x in line.split(",")]
                            gpu_parts.append(f"GPU{idx}: {util}% {int(float(used))}/{int(float(total))} MB")
                except Exception as e:
                    logger.warning(f"Failed to parse nvidia-smi output: {e}")
        except Exception as e:
            logger.debug(f"nvidia-smi not available or failed: {e}")
        if not gpu_parts:
            try:
                import torch  # type: ignore

                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    n = int(torch.cuda.device_count())
                    for i in range(n):
                        try:
                            name = torch.cuda.get_device_name(i)
                        except Exception as e:
                            logger.debug(f"Failed to get CUDA device {i} name: {e}")
                            name = f"CUDA{i}"
                        try:
                            free_b, total_b = torch.cuda.mem_get_info(i)
                            used_b = max(0, int(total_b - free_b))
                            gpu_parts.append(f"{name}: {used_b // (1024**2)}/{total_b // (1024**2)} MB")
                        except Exception as e:
                            logger.warning(f"Failed to get CUDA device {i} memory: {e}")
                            gpu_parts.append(f"{name}: mem N/A")
            except Exception as e:
                logger.debug(f"Failed to query GPU stats via torch: {e}")
        if not gpu_parts:
            gpu_parts.append("GPU: --")
        parts.append(" | ".join(gpu_parts))
        gpu_duration = time.perf_counter() - gpu_section_start
        if gpu_duration > 0.75:
            logger.debug(f"GPU section latency: {gpu_duration:.3f}s")

        # Datasets usage: size of training_datasets directory
        try:
            td_root = None
            try:
                # Check HF_HOME first for user-configured location
                hf_home = os.environ.get("HF_HOME")
                if hf_home:
                    hf_path = os.path.abspath(hf_home)
                    # Parent of .hf_cache is the datasets root
                    if os.path.basename(hf_path) in (".hf_cache", "hf_cache"):
                        td_root = os.path.dirname(hf_path)
                    else:
                        td_root = os.path.dirname(hf_path)
                
                # Fall back to project training_datasets folder
                if not td_root or not os.path.isdir(td_root):
                    cur = os.path.abspath(os.getcwd())
                    for _ in range(8):
                        if os.path.exists(os.path.join(cur, "pyproject.toml")):
                            td_root = os.path.join(cur, "training_datasets")
                            break
                        parent = os.path.dirname(cur)
                        if parent == cur:
                            break
                        cur = parent
                if td_root is None:
                    td_root = os.path.join(os.path.abspath(os.getcwd()), "training_datasets")
            except Exception as e:
                logger.debug(f"Failed to locate datasets root: {e}")
                td_root = os.path.join(os.path.abspath(os.getcwd()), "training_datasets")
            total_bytes = 0
            if os.path.isdir(td_root):
                for _root, _dirs, files in os.walk(td_root):
                    for fn in files:
                        try:
                            fp = os.path.join(_root, fn)
                            total_bytes += os.path.getsize(fp)
                        except Exception as e:
                            logger.debug(f"Failed to get file size for {fn}: {e}")
            usage_gb = total_bytes / (1024.0 ** 3)
            # Dataset cap currently not surfaced here; show usage only
            parts.append(f"Datasets: {usage_gb:.1f}/-- GB")
        except Exception as e:
            logger.error(f"Failed to calculate dataset usage: {e}")
            parts.append("Datasets: --")
        collection_duration = time.perf_counter() - update_start
        if collection_duration > 1.0:
            logger.debug(f"System status collection latency: {collection_duration:.3f}s")
        return " | ".join(parts)
