"""Subbrains Manager Panel component.

Provides a Tkinter panel for managing dynamic subbrains (experts) with:
- Viewing expert registry with hierarchy
- Creating, deleting, freezing experts
- Binding goals to experts  
- Viewing performance metrics and routing stats

Usage:
    from aios.gui.components.subbrains_manager_panel import SubbrainsManagerPanel
    
    panel = SubbrainsManagerPanel(
        parent,
        run_cli=my_cli_callback,
        append_out=my_output_callback
    )
"""

from __future__ import annotations

from .panel_main import SubbrainsManagerPanel

__all__ = ["SubbrainsManagerPanel"]
