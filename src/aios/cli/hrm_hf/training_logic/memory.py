"""Memory monitoring utilities for training."""

from __future__ import annotations

from typing import Optional


def sys_mem_used_pct() -> Optional[float]:
    """Get system memory usage percentage."""
    try:
        import psutil  # type: ignore
        v = psutil.virtual_memory()
        return float(v.percent)
    except Exception:
        return None
