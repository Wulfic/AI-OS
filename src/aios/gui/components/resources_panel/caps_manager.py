"""Storage caps management for resources panel."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aios.gui.components.resources_panel.panel_main import ResourcesPanel

from ...utils.resource_management import submit_background
logger = logging.getLogger(__name__)


def _update_status(panel: "ResourcesPanel", message: str) -> None:
    try:
        if hasattr(panel, "_status_label") and panel._status_label is not None:
            panel._status_label.config(text=message)
    except Exception:
        pass


def _set_caps_controls_enabled(panel: "ResourcesPanel", enabled: bool) -> None:
    entry = getattr(panel, "dataset_cap_entry", None)
    if entry is not None:
        try:
            entry_state = "normal" if enabled else "disabled"
            entry.config(state=entry_state)
        except Exception:
            pass


def _enter_caps_busy(panel: "ResourcesPanel", message: str) -> None:
    panel._caps_busy_count = getattr(panel, "_caps_busy_count", 0) + 1
    if panel._caps_busy_count == 1:
        _set_caps_controls_enabled(panel, False)
    _update_status(panel, message)


def _exit_caps_busy(panel: "ResourcesPanel", message: str | None = None) -> None:
    panel._caps_busy_count = max(0, getattr(panel, "_caps_busy_count", 1) - 1)
    if panel._caps_busy_count == 0:
        _set_caps_controls_enabled(panel, True)
        _update_status(panel, message or "")


def _run_in_executor(panel: "ResourcesPanel", work: Callable[[], None]) -> None:
    try:
        submit_background("resources-cap", work, pool=getattr(panel, "_worker_pool", None))
    except RuntimeError as exc:
        logger.error("Failed to queue resources panel task: %s", exc)
        raise


def on_apply_caps(
    panel: "ResourcesPanel",
    apply_fn: Optional[Callable[[float, Optional[float], Optional[float]], dict]]
) -> None:
    """Apply storage capacity limits.
    
    Args:
        panel: ResourcesPanel instance
        apply_fn: Callback function to apply caps
    """
    if apply_fn is None:
        return

    if getattr(panel, "_shutting_down", False):
        logger.debug("Skipping storage caps apply during shutdown")
        return

    try:
        ds = panel.dataset_cap_var.get().strip()
        dsf = float(ds) if ds else None
    except Exception:
        dsf = None
    try:
        ms = panel.model_cap_var.get().strip()
        msf = float(ms) if ms else None
    except Exception:
        msf = None
    try:
        pb = panel.per_brain_cap_var.get().strip()
        pbf = float(pb) if pb else None
    except Exception:
        pbf = None

    def _work() -> None:
        try:
            logger.info(
                "Applying storage caps: dataset=%sGB model=%sGB per_brain=%sGB",
                dsf,
                msf,
                pbf,
            )
            apply_fn(float(dsf) if dsf is not None else 0.0, msf, pbf)
        except Exception as exc:
            logger.error(f"Failed to apply storage caps: {exc}", exc_info=True)

            def _on_error() -> None:
                _exit_caps_busy(panel, "Caps update failed")
            try:
                panel.after(0, _on_error)
            except Exception:
                _on_error()
            return

        def _on_success() -> None:
            from . import monitoring

            logger.info("Storage caps applied successfully")
            _exit_caps_busy(panel, "Caps updated")
            try:
                monitoring.update_storage_usage(panel)
            except Exception as update_exc:
                logger.debug(f"Storage usage update failed: {update_exc}")

        try:
            panel.after(0, _on_success)
        except Exception:
            _on_success()

    _enter_caps_busy(panel, "Applying caps…")
    _run_in_executor(panel, _work)


def on_refresh_caps(
    panel: "ResourcesPanel",
    fetch_fn: Optional[Callable[[], dict]],
    *,
    source: str = "user",
) -> None:
    """Refresh storage caps from config.
    
    Args:
        panel: ResourcesPanel instance
        fetch_fn: Callback to fetch current caps
    """
    if fetch_fn is None:
        return

    if getattr(panel, "_shutting_down", False):
        logger.debug("Skipping storage caps refresh (%s) during shutdown", source)
        return

    try:
        if source == "user":
            logger.debug("Refreshing caps via user request")
        else:
            logger.debug("Refreshing caps via '%s' request", source)
    except Exception:
        pass

    def _work() -> None:
        if getattr(panel, "_shutting_down", False):
            _exit_caps_busy(panel, "")
            return

        try:
            data = fetch_fn() or {}
        except Exception as exc:
            logger.error(f"Failed to refresh caps: {exc}", exc_info=True)

            def _on_error() -> None:
                _exit_caps_busy(panel, "Caps refresh failed")

            try:
                panel.after(0, _on_error)
            except Exception:
                _on_error()
            return

        if getattr(panel, "_shutting_down", False):
            _exit_caps_busy(panel, "")
            return

        def _on_success() -> None:
            set_caps(panel, data)
            from . import monitoring

            try:
                monitoring.update_storage_usage(panel)
            except Exception as update_exc:
                logger.debug(f"Storage usage update failed: {update_exc}")
            _exit_caps_busy(panel, "Caps refreshed")

        try:
            panel.after(0, _on_success)
        except Exception:
            _on_success()

    _enter_caps_busy(panel, "Refreshing caps…")
    _run_in_executor(panel, _work)


def set_caps(panel: "ResourcesPanel", caps: dict) -> None:
    """Populate storage caps inputs from a dict.

    Args:
        panel: ResourcesPanel instance
        caps: Dict with keys dataset_cap_gb, model_cap_gb, per_brain_cap_gb
    """
    applied: dict[str, float] = {}
    try:
        v = caps.get("dataset_cap_gb")
        if isinstance(v, (int, float)) and v > 0:
            panel.dataset_cap_var.set(str(v))
            applied["dataset_cap_gb"] = float(v)
    except Exception:
        pass
    try:
        v = caps.get("model_cap_gb")
        if isinstance(v, (int, float)) and v > 0:
            panel.model_cap_var.set(str(v))
            applied["model_cap_gb"] = float(v)
    except Exception:
        pass
    try:
        v = caps.get("per_brain_cap_gb")
        if isinstance(v, (int, float)) and v > 0:
            panel.per_brain_cap_var.set(str(v))
            applied["per_brain_cap_gb"] = float(v)
    except Exception:
        pass

    if applied:
        try:
            panel._last_caps_snapshot = tuple(sorted(applied.items()))
        except Exception:
            pass
