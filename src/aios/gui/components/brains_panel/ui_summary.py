"""Build summary statistics row for brains panel.

Creates the top row showing aggregate stats for brains and experts.
"""

from __future__ import annotations

from typing import Any, cast

try:  # pragma: no cover
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    tk = cast(Any, None)
    ttk = cast(Any, None)


def build_summary_row(parent: Any, panel: Any) -> None:
    """Build summary statistics row with brains and experts stats.
    
    Creates StringVars on panel instance for data binding and builds the UI row.
    
    Args:
        parent: Parent Tk widget
        panel: BrainsPanel instance (for storing StringVars)
    """
    if tk is None or ttk is None:
        return
    
    # Summary row - combined stats for brains and experts
    top = ttk.Frame(parent)
    top.pack(fill="x", pady=(0, 6))
    
    # Brains stats
    panel.brain_count_var = tk.StringVar(value="0")
    panel.total_mb_var = tk.StringVar(value="0.0")
    panel.total_params_m_var = tk.StringVar(value="0.0")
    
    lbl_brains = ttk.Label(top, text="Brains:")
    lbl_brains.pack(side="left")
    val_brains = ttk.Label(top, textvariable=panel.brain_count_var, width=6)
    val_brains.pack(side="left")
    
    lbl_total = ttk.Label(top, text="Size (MB):")
    lbl_total.pack(side="left", padx=(4, 0))
    val_total = ttk.Label(top, textvariable=panel.total_mb_var, width=10)
    val_total.pack(side="left")
    
    lbl_params = ttk.Label(top, text="Params (M):")
    lbl_params.pack(side="left", padx=(4, 0))
    val_params = ttk.Label(top, textvariable=panel.total_params_m_var, width=10)
    val_params.pack(side="left")
    
    # Separator
    ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=12)
    
    # Experts stats
    ttk.Label(top, text="Experts:").pack(side="left")
    panel.total_experts_var = tk.StringVar(value="0")
    ttk.Label(top, textvariable=panel.total_experts_var, width=6).pack(side="left")
    
    ttk.Label(top, text="Active:").pack(side="left", padx=(4, 0))
    panel.active_experts_var = tk.StringVar(value="0")
    ttk.Label(top, textvariable=panel.active_experts_var, width=6).pack(side="left")
    
    ttk.Label(top, text="Frozen:").pack(side="left", padx=(4, 0))
    panel.frozen_experts_var = tk.StringVar(value="0")
    ttk.Label(top, textvariable=panel.frozen_experts_var, width=6).pack(side="left")
    
    ttk.Label(top, text="Activations:").pack(side="left", padx=(4, 0))
    panel.total_activations_var = tk.StringVar(value="0")
    ttk.Label(top, textvariable=panel.total_activations_var, width=10).pack(side="left")
    
    # Force refresh (bypass throttling) when button clicked
    btn_refresh = ttk.Button(top, text="Refresh All", command=lambda: panel.refresh(force=True))
    btn_refresh.pack(side="right")
    
    # Status indicator for loading state
    panel.status_var = tk.StringVar(value="")
    panel.status_label = ttk.Label(top, textvariable=panel.status_var, foreground="gray")
    panel.status_label.pack(side="right", padx=(0, 8))
    
    # Tooltips for summary row
    try:  # pragma: no cover
        from ..tooltips import add_tooltip
        add_tooltip(lbl_brains, "Total number of brain models in the registry")
        add_tooltip(val_brains, "Current count of brain files")
        add_tooltip(lbl_total, "Aggregate on‑disk size of all brains (megabytes)")
        add_tooltip(val_total, "Computed total MB of all brains")
        add_tooltip(lbl_params, "Approximate parameter count in millions (size_bytes / 4)")
        add_tooltip(val_params, "Estimated parameter total across all brains (millions)")
        add_tooltip(btn_refresh, "Re-scan brains directory and expert registry, update all stats + tables")
    except Exception:
        pass
