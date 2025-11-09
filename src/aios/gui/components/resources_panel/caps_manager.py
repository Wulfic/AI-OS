"""Storage caps management for resources panel."""

from __future__ import annotations

from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aios.gui.components.resources_panel.panel_main import ResourcesPanel


def on_apply_caps(
    panel: "ResourcesPanel",
    apply_fn: Optional[Callable[[float, Optional[float], Optional[float]], dict]]
) -> None:
    """Apply storage capacity limits.
    
    Args:
        panel: ResourcesPanel instance
        apply_fn: Callback function to apply caps
    """
    try:
        if apply_fn is None:
            return
        ds = panel.dataset_cap_var.get().strip()
        dsf = float(ds) if ds else None
        ms = panel.model_cap_var.get().strip()
        msf = float(ms) if ms else None
        pb = panel.per_brain_cap_var.get().strip()
        pbf = float(pb) if pb else None
        # Only pass values that are provided
        apply_fn(float(dsf) if dsf is not None else 0.0, msf, pbf)
        # Update storage usage display after applying caps
        from . import monitoring
        monitoring.update_storage_usage(panel)
    except Exception:
        pass


def on_refresh_caps(
    panel: "ResourcesPanel",
    fetch_fn: Optional[Callable[[], dict]]
) -> None:
    """Refresh storage caps from config.
    
    Args:
        panel: ResourcesPanel instance
        fetch_fn: Callback to fetch current caps
    """
    try:
        if fetch_fn is None:
            return
        data = fetch_fn() or {}
        set_caps(panel, data)
        # Also update storage usage display
        from . import monitoring
        monitoring.update_storage_usage(panel)
    except Exception:
        pass


def set_caps(panel: "ResourcesPanel", caps: dict) -> None:
    """Populate storage caps inputs from a dict.

    Args:
        panel: ResourcesPanel instance
        caps: Dict with keys dataset_cap_gb, model_cap_gb, per_brain_cap_gb
    """
    try:
        v = caps.get("dataset_cap_gb")
        if isinstance(v, (int, float)) and v > 0:
            panel.dataset_cap_var.set(str(v))
    except Exception:
        pass
    try:
        v = caps.get("model_cap_gb")
        if isinstance(v, (int, float)) and v > 0:
            panel.model_cap_var.set(str(v))
    except Exception:
        pass
    try:
        v = caps.get("per_brain_cap_gb")
        if isinstance(v, (int, float)) and v > 0:
            panel.per_brain_cap_var.set(str(v))
    except Exception:
        pass
