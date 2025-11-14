"""Resource monitoring data management and update scheduling."""

from __future__ import annotations

import logging
import queue
import time
import threading
from collections import deque
from concurrent.futures import Future, TimeoutError as FuturesTimeoutError, CancelledError
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ...utils.resource_management import submit_background

from .constants import MATPLOTLIB_AVAILABLE
from . import stats_collectors
from . import chart_widgets
from . import fallback_widgets

if TYPE_CHECKING:
    from .panel_main import ResourcesPanel

logger = logging.getLogger(__name__)


# Global GPU stats cache to avoid blocking main thread
_gpu_stats_cache = []
_gpu_stats_lock = threading.Lock()
_gpu_stats_inflight = False
_GPU_POLL_INTERVAL_MS = 1500
_LOCK_TYPE = type(threading.Lock())


def _is_monitor_active(panel: "ResourcesPanel") -> bool:
    """Return True when monitoring jobs should continue to run for the panel."""
    return bool(getattr(panel, "_monitor_active", True))


def _get_panel_root(panel: "ResourcesPanel") -> Any | None:
    """Best-effort lookup for the Tk root associated with a panel."""
    root = getattr(panel, "_tk_root", None)
    if root is not None:
        return root
    try:
        return panel.winfo_toplevel()
    except Exception:
        return None


def _schedule_gpu_stats_poll(panel: "ResourcesPanel", *, delay_ms: int = 0) -> None:
    if not _is_monitor_active(panel):
        return

    root = _get_panel_root(panel)
    if root is None or delay_ms <= 0:
        _queue_gpu_stats_poll(panel)
        return

    try:
        root.after(delay_ms, lambda: _queue_gpu_stats_poll(panel))
    except Exception:
        _queue_gpu_stats_poll(panel)


def _queue_gpu_stats_poll(panel: "ResourcesPanel") -> None:
    global _gpu_stats_inflight

    worker_pool = getattr(panel, "_worker_pool", None)
    if not _is_monitor_active(panel):
        _gpu_stats_inflight = False
        return
    if _gpu_stats_inflight:
        return

    _gpu_stats_inflight = True

    def _work() -> None:
        global _gpu_stats_cache, _gpu_stats_inflight
        try:
            gpu_stats = stats_collectors.get_gpu_stats()
            with _gpu_stats_lock:
                _gpu_stats_cache = gpu_stats
        except Exception:
            pass
        finally:
            def _reset_and_schedule() -> None:
                global _gpu_stats_inflight
                _gpu_stats_inflight = False
                if _is_monitor_active(panel):
                    _schedule_gpu_stats_poll(panel, delay_ms=_GPU_POLL_INTERVAL_MS)

            root = _get_panel_root(panel)
            if root is not None:
                try:
                    root.after(0, _reset_and_schedule)
                    return
                except Exception:
                    pass
            _reset_and_schedule()

    if worker_pool is None:
        logger.debug("Executing GPU stats poll synchronously; worker pool unavailable")
        _work()
        return

    try:
        future = submit_background("resources-gpu-stats", _work, pool=worker_pool)
        panel._gpu_stats_future = future

        def _clear_future(done_future: Future) -> None:
            if getattr(panel, "_gpu_stats_future", None) is done_future:
                panel._gpu_stats_future = None

        future.add_done_callback(_clear_future)
    except RuntimeError as exc:
        _gpu_stats_inflight = False
        logger.debug("Failed to queue GPU stats poll: %s", exc)
        if _is_monitor_active(panel):
            _schedule_gpu_stats_poll(panel, delay_ms=_GPU_POLL_INTERVAL_MS)


def _get_cached_gpu_stats() -> list:
    """Get cached GPU stats without blocking."""
    with _gpu_stats_lock:
        return _gpu_stats_cache.copy()


def init_monitoring_data(panel: "ResourcesPanel") -> None:
    """Initialize data structures for historical monitoring.
    
    Args:
        panel: ResourcesPanel instance
    """
    panel._monitor_active = True
    panel._gpu_stats_future: Future | None = None
    panel._monitor_future: Future | None = None

    # Start background GPU stats collection
    _schedule_gpu_stats_poll(panel)
    
    # Timeline options and current selection
    panel._timeline_options = {
        "1 minute": 60,
        "5 minutes": 300,
        "15 minutes": 900,
        "1 hour": 3600,
    }
    panel._current_timeline = "1 minute"
    
    # Historical data storage (max 1 hour at 1.5s intervals = ~2400 points)
    max_points = 2400
    panel._history = {
        "timestamps": deque(maxlen=max_points),
        "cpu_util": deque(maxlen=max_points),
        "cpu_temp": deque(maxlen=max_points),
        "ram_used": deque(maxlen=max_points),
        "ram_total": deque(maxlen=max_points),
        "net_upload": deque(maxlen=max_points),
        "net_download": deque(maxlen=max_points),
        "disk_read": deque(maxlen=max_points),
        "disk_write": deque(maxlen=max_points),
        "gpu": {}  # {idx: {util: deque, mem_used: deque, mem_total: deque, temp: deque}}
    }
    
    # Previous values for rate calculations
    panel._last_net_io = None
    panel._last_disk_io = None
    panel._last_io_time = time.time()

    # Storage usage worker coordination
    panel._storage_update_lock = threading.Lock()
    panel._storage_usage_queue: queue.Queue | None = queue.Queue()
    panel._storage_queue_interval_ms = 250
    panel._storage_queue_pump_started = False
    _start_storage_queue_pump(panel)

    # Monitoring snapshot coordination
    panel._monitor_queue: queue.Queue | None = queue.Queue()
    panel._monitor_job_lock = threading.Lock()


def update_storage_usage(panel: "ResourcesPanel") -> None:
    """Update dataset storage usage display without blocking the UI thread."""

    if not _is_monitor_active(panel):
        return

    lock: threading.Lock | None = getattr(panel, "_storage_update_lock", None)
    if lock is not None and not lock.acquire(blocking=False):
        # Another update is already in flight; skip this cycle.
        return

    def _release_lock() -> None:
        if lock is not None and lock.locked():
            lock.release()

    def _compute_usage() -> str:
        try:
            from aios.data.datasets.storage import datasets_storage_usage_gb, datasets_storage_cap_gb

            usage_gb = datasets_storage_usage_gb()
            cap_gb = datasets_storage_cap_gb()

            if cap_gb > 0:
                pct = (usage_gb / cap_gb) * 100 if cap_gb else 0.0
                text = f"Usage: {usage_gb:.2f} GB / {cap_gb:.2f} GB ({pct:.1f}%)"
                if pct >= 95:
                    logger.warning(
                        "Dataset storage critically high: %.2f/%.2f GB (%.1f%%)",
                        usage_gb,
                        cap_gb,
                        pct,
                    )
                elif pct >= 85:
                    logger.warning(
                        "Dataset storage high: %.2f/%.2f GB (%.1f%%)",
                        usage_gb,
                        cap_gb,
                        pct,
                    )
            else:
                text = f"Usage: {usage_gb:.2f} GB (no cap set)"
            return text
        except Exception as exc:
            logger.debug("Failed to calculate storage usage: %s", exc)
            return "Usage: unavailable"

    def _schedule_update(text: str) -> None:
        queue_ref = getattr(panel, "_storage_usage_queue", None)
        if isinstance(queue_ref, queue.Queue):
            queue_ref.put(text)
        else:
            _set_dataset_usage_text(panel, text)

    def _work() -> None:
        try:
            usage_text = _compute_usage()
            _schedule_update(usage_text)
        finally:
            _release_lock()

    worker_pool = getattr(panel, "_worker_pool", None)
    if worker_pool is not None:
        try:
            worker_pool.submit(_work)
            return
        except Exception as exc:
            logger.debug("Worker pool submit failed, falling back to sync storage update: %s", exc)

    try:
        _work()
    except Exception:
        # _work handles its own logging; ensure lock released even if something unexpected bubbles up
        _release_lock()


def _collect_monitor_snapshot(last_net_io, last_disk_io, last_io_time) -> dict | None:
    try:
        snapshot: dict[str, Any] = {
            "timestamp": datetime.now(),
        }

        cpu_usage, cpu_temp = stats_collectors.get_cpu_stats()
        snapshot.update({
            "cpu_usage": cpu_usage,
            "cpu_temp": cpu_temp,
        })

        ram_usage, ram_used, ram_total = stats_collectors.get_ram_stats()
        snapshot.update({
            "ram_usage": ram_usage,
            "ram_used": ram_used,
            "ram_total": ram_total,
        })

        net_up, net_down, net_io = stats_collectors.get_network_stats(last_net_io, last_io_time)
        snapshot.update({
            "net_upload": net_up,
            "net_download": net_down,
            "net_io": net_io,
        })

        disk_read, disk_write, disk_io, io_time = stats_collectors.get_disk_stats(last_disk_io, last_io_time)
        snapshot.update({
            "disk_read": disk_read,
            "disk_write": disk_write,
            "disk_io": disk_io,
            "io_time": io_time,
        })

        snapshot["gpu_stats"] = _get_cached_gpu_stats()
        return snapshot
    except Exception as exc:
        logger.debug("Failed to collect monitor snapshot: %s", exc)
        return None


def _apply_monitor_snapshot(panel: "ResourcesPanel", snapshot: dict) -> None:
    try:
        timestamp = snapshot.get("timestamp", datetime.now())
        panel._history["timestamps"].append(timestamp)

        cpu_usage = float(snapshot.get("cpu_usage", 0.0))
        cpu_temp = snapshot.get("cpu_temp")
        panel._history["cpu_util"].append(cpu_usage)
        panel._history["cpu_temp"].append(cpu_temp)

        if cpu_usage >= 95:
            logger.warning(f"CPU usage critically high: {cpu_usage:.1f}%")
        elif cpu_usage >= 85:
            logger.warning(f"CPU usage high: {cpu_usage:.1f}%")

        if cpu_temp is not None:
            try:
                temp_val = float(cpu_temp)
            except Exception:
                temp_val = None
            if temp_val is not None:
                if temp_val >= 85:
                    logger.warning(f"CPU temperature critically high: {temp_val:.1f}°C")
                elif temp_val >= 75:
                    logger.warning(f"CPU temperature high: {temp_val:.1f}°C")

        ram_usage = float(snapshot.get("ram_usage", 0.0))
        ram_used = float(snapshot.get("ram_used", 0.0))
        ram_total = float(snapshot.get("ram_total", 1.0))
        panel._history["ram_used"].append(ram_used)
        panel._history["ram_total"].append(ram_total)

        if ram_usage >= 95:
            logger.warning(f"RAM usage critically high: {ram_used:.2f}/{ram_total:.2f} GB ({ram_usage:.1f}%)")
        elif ram_usage >= 85:
            logger.warning(f"RAM usage high: {ram_used:.2f}/{ram_total:.2f} GB ({ram_usage:.1f}%)")

        net_up = float(snapshot.get("net_upload", 0.0))
        net_down = float(snapshot.get("net_download", 0.0))
        panel._history["net_upload"].append(net_up)
        panel._history["net_download"].append(net_down)

        net_io = snapshot.get("net_io")
        if net_io is not None:
            panel._last_net_io = net_io

        disk_read = float(snapshot.get("disk_read", 0.0))
        disk_write = float(snapshot.get("disk_write", 0.0))
        panel._history["disk_read"].append(disk_read)
        panel._history["disk_write"].append(disk_write)

        disk_io = snapshot.get("disk_io")
        if disk_io is not None:
            panel._last_disk_io = disk_io

        io_time = snapshot.get("io_time")
        if isinstance(io_time, (int, float)):
            panel._last_io_time = float(io_time)

        try:
            gpu_stats = snapshot.get("gpu_stats") or []
            for gpu in gpu_stats:
                try:
                    idx = gpu["index"]
                except Exception:
                    continue
                if idx not in panel._history["gpu"]:
                    max_points = 2400
                    panel._history["gpu"][idx] = {
                        "util": deque(maxlen=max_points),
                        "mem_used": deque(maxlen=max_points),
                        "mem_total": deque(maxlen=max_points),
                        "temp": deque(maxlen=max_points),
                        "name": gpu.get("name", f"GPU{idx}"),
                    }

                gpu_util = float(gpu.get("util", 0.0))
                gpu_mem_used_gb = float(gpu.get("mem_used_mb", 0.0)) / 1024
                gpu_mem_total_gb = max(0.0001, float(gpu.get("mem_total_mb", 1.0)) / 1024)
                gpu_temp = gpu.get("temp")
                gpu_name = gpu.get("name", f"GPU{idx}")

                panel._history["gpu"][idx]["util"].append(gpu_util)
                panel._history["gpu"][idx]["mem_used"].append(gpu_mem_used_gb)
                panel._history["gpu"][idx]["mem_total"].append(gpu_mem_total_gb)
                panel._history["gpu"][idx]["temp"].append(gpu_temp)
                panel._history["gpu"][idx]["name"] = gpu_name

                gpu_mem_pct = (gpu_mem_used_gb / max(0.1, gpu_mem_total_gb)) * 100
                if gpu_mem_pct >= 95:
                    logger.warning(
                        f"GPU{idx} ({gpu_name}) VRAM critically high: {gpu_mem_used_gb:.2f}/{gpu_mem_total_gb:.2f} GB ({gpu_mem_pct:.1f}%)"
                    )
                elif gpu_mem_pct >= 85:
                    logger.warning(
                        f"GPU{idx} ({gpu_name}) VRAM high: {gpu_mem_used_gb:.2f}/{gpu_mem_total_gb:.2f} GB ({gpu_mem_pct:.1f}%)"
                    )

                if gpu_temp is not None:
                    try:
                        temp_val = float(gpu_temp)
                    except Exception:
                        temp_val = None
                    if temp_val is not None:
                        if temp_val >= 85:
                            logger.warning(
                                f"GPU{idx} ({gpu_name}) temperature critically high: {temp_val:.1f}°C"
                            )
                        elif temp_val >= 75:
                            logger.warning(
                                f"GPU{idx} ({gpu_name}) temperature high: {temp_val:.1f}°C"
                            )
        except Exception as exc:
            logger.debug(f"Failed to process GPU monitoring snapshot: {exc}")

        if not hasattr(panel, "_storage_update_counter"):
            panel._storage_update_counter = 0
        panel._storage_update_counter = (panel._storage_update_counter + 1) % 10
        if panel._storage_update_counter == 0:
            schedule_storage_update(panel)

        if MATPLOTLIB_AVAILABLE:
            chart_widgets.update_charts(panel)
        else:
            fallback_widgets.update_fallback_ui(panel)
    except Exception as exc:
        logger.debug(f"Failed to apply monitor snapshot: {exc}")


def update_monitor(panel: "ResourcesPanel") -> None:
    """Update monitoring display; heavy work runs on a background thread."""

    if not _is_monitor_active(panel):
        return

    monitor_queue = getattr(panel, "_monitor_queue", None)
    if isinstance(monitor_queue, queue.Queue):
        while True:
            try:
                snapshot = monitor_queue.get_nowait()
            except queue.Empty:
                break
            if snapshot is not None:
                _apply_monitor_snapshot(panel, snapshot)

    last_net_io = getattr(panel, "_last_net_io", None)
    last_disk_io = getattr(panel, "_last_disk_io", None)
    last_io_time = getattr(panel, "_last_io_time", 0.0)

    worker_pool = getattr(panel, "_worker_pool", None)
    job_lock = getattr(panel, "_monitor_job_lock", None)

    if worker_pool is not None and job_lock is not None:
        acquired = False
        try:
            acquired = job_lock.acquire(blocking=False)
        except Exception:
            acquired = False

        if acquired:
            def _work() -> None:
                try:
                    snapshot = _collect_monitor_snapshot(last_net_io, last_disk_io, last_io_time)
                    if snapshot is not None and isinstance(monitor_queue, queue.Queue):
                        monitor_queue.put(snapshot)
                finally:
                    try:
                        job_lock.release()
                    except Exception:
                        pass

            try:
                future = worker_pool.submit(_work)
                panel._monitor_future = future

                def _clear_monitor_future(done_future: Future) -> None:
                    if getattr(panel, "_monitor_future", None) is done_future:
                        panel._monitor_future = None

                future.add_done_callback(_clear_monitor_future)
                return
            except Exception as exc:
                try:
                    job_lock.release()
                except Exception:
                    pass
                logger.debug("Worker pool submit failed for monitor update: %s", exc)
        else:
            return

    snapshot = _collect_monitor_snapshot(last_net_io, last_disk_io, last_io_time)
    if snapshot is not None:
        _apply_monitor_snapshot(panel, snapshot)


def schedule_monitor_update(panel: "ResourcesPanel") -> None:
    """Schedule the next monitoring update.
    
    Args:
        panel: ResourcesPanel instance
    """
    if not _is_monitor_active(panel):
        return
    try:
        update_monitor(panel)
    finally:
        try:
            root = _get_panel_root(panel)
            if root is not None:
                panel._monitor_after_id = root.after(1500, lambda: schedule_monitor_update(panel))
        except Exception:
            pass


def schedule_storage_update(panel: "ResourcesPanel", *, delay_ms: int = 0) -> None:
    """Schedule a storage usage refresh using Tk's event loop."""

    if not _is_monitor_active(panel):
        return

    root = _get_panel_root(panel)
    if root is None:
        update_storage_usage(panel)
        return

    try:
        root.after(delay_ms, lambda: update_storage_usage(panel))
    except Exception:
        update_storage_usage(panel)


def _start_storage_queue_pump(panel: "ResourcesPanel") -> None:
    if getattr(panel, "_storage_queue_pump_started", False):
        return

    root = _get_panel_root(panel)
    if root is None or not _is_monitor_active(panel):
        return

    try:
        panel._storage_queue_pump_started = True
        root.after(0, lambda: _pump_storage_queue(panel))
    except Exception:
        panel._storage_queue_pump_started = False


def _pump_storage_queue(panel: "ResourcesPanel") -> None:
    if not _is_monitor_active(panel):
        return

    queue_ref = getattr(panel, "_storage_usage_queue", None)
    if not isinstance(queue_ref, queue.Queue):
        return

    try:
        while True:
            try:
                text = queue_ref.get_nowait()
            except queue.Empty:
                break
            _set_dataset_usage_text(panel, text)
    finally:
        root = _get_panel_root(panel)
        interval = getattr(panel, "_storage_queue_interval_ms", None)
        if root is not None and interval and _is_monitor_active(panel):
            try:
                root.after(interval, lambda: _pump_storage_queue(panel))
            except Exception:
                pass


def _set_dataset_usage_text(panel: "ResourcesPanel", text: str) -> None:
    try:
        if hasattr(panel, "dataset_usage_label"):
            panel.dataset_usage_label.config(text=text)
    except Exception as exc:
        logger.debug("Failed to apply storage usage text: %s", exc)


def shutdown_monitoring_data(panel: "ResourcesPanel") -> None:
    """Signal monitoring helpers to stop scheduling future work."""
    panel._monitor_active = False

    # Prevent workers from queuing more GUI updates.
    queue_ref = getattr(panel, "_storage_usage_queue", None)
    if isinstance(queue_ref, queue.Queue):
        try:
            while True:
                queue_ref.get_nowait()
        except queue.Empty:
            pass
    panel._storage_usage_queue = None

    monitor_queue = getattr(panel, "_monitor_queue", None)
    if isinstance(monitor_queue, queue.Queue):
        try:
            while True:
                monitor_queue.get_nowait()
        except queue.Empty:
            pass
    panel._monitor_queue = None

    global _gpu_stats_inflight
    _gpu_stats_inflight = False

    gpu_future = getattr(panel, "_gpu_stats_future", None)
    if isinstance(gpu_future, Future):
        if not gpu_future.done():
            gpu_future.cancel()
        try:
            gpu_future.result(timeout=0.5)
        except (CancelledError, FuturesTimeoutError):
            pass
        except Exception as exc:
            logger.debug("GPU stats future raised during shutdown: %s", exc)
        finally:
            if getattr(panel, "_gpu_stats_future", None) is gpu_future:
                panel._gpu_stats_future = None

    monitor_future = getattr(panel, "_monitor_future", None)
    if isinstance(monitor_future, Future):
        if not monitor_future.done():
            monitor_future.cancel()
        try:
            monitor_future.result(timeout=0.5)
        except (CancelledError, FuturesTimeoutError):
            pass
        except Exception as exc:
            logger.debug("Monitor future raised during shutdown: %s", exc)
        finally:
            if getattr(panel, "_monitor_future", None) is monitor_future:
                panel._monitor_future = None

    for lock_attr in ("_monitor_job_lock", "_storage_update_lock"):
        lock = getattr(panel, lock_attr, None)
        if isinstance(lock, _LOCK_TYPE):
            acquired = False
            try:
                acquired = lock.acquire(timeout=0.5)
            except TypeError:
                # Python <3.2 compatibility, fallback to blocking acquire with short sleep.
                try:
                    acquired = lock.acquire(False)
                except Exception:
                    acquired = False
            except Exception:
                acquired = False
            if acquired:
                lock.release()


__all__ = [
    "init_monitoring_data",
    "update_monitor",
    "update_storage_usage",
    "schedule_monitor_update",
    "schedule_storage_update",
    "shutdown_monitoring_data",
]
