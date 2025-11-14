"""UI structure setup for AI-OS GUI application.

This module creates the main UI structure:
- Notebook widget with tabs
- Tab frames for each section
- Tab change event handlers
- Basic state variables
"""

from __future__ import annotations

import os
import logging
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import tkinter as tk
    from tkinter import ttk
else:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:
        tk = None  # type: ignore
        ttk = None  # type: ignore

# Import safe variable wrappers
from ..utils import safe_variables


logger = logging.getLogger(__name__)


def create_ui_structure(app: Any, root: "tk.Tk") -> None:  # type: ignore[name-defined]
    """Create notebook, tabs, and basic UI structure.
    
    Args:
        app: The AiosTkApp instance
        root: The Tkinter root window
    
    Creates:
        - Notebook widget with 10 tabs
        - Tab frames: Chat, Brains, Datasets, HRM Training, Evaluation, Resources, MCP, Settings, Debug, Help
        - Tab change event handler for auto-refresh
        - Core state variables (cpu_var, cuda_var, etc.)
        - Dataset path variable
        - Command variable
    """
    if tk is None:
        raise RuntimeError("Tkinter is not available")
    
    # Create notebook
    app.nb = ttk.Notebook(root)
    app.nb.pack(fill="both", expand=True)
    
    # Create all tab frames
    app.chat_tab = ttk.Frame(app.nb)
    app.brains_tab = ttk.Frame(app.nb)
    app.datasets_tab = ttk.Frame(app.nb)
    app.training_tab = ttk.Frame(app.nb)
    app.evaluation_tab = ttk.Frame(app.nb)
    app.resources_tab = ttk.Frame(app.nb)
    app.mcp_tab = ttk.Frame(app.nb)
    app.settings_tab = ttk.Frame(app.nb)
    app.debug_tab = ttk.Frame(app.nb)
    app.help_tab = ttk.Frame(app.nb)
    
    # Add tabs in desired order: Chat, Brains, Datasets, HRM Training, Evaluation, Resources, MCP and Tools, Settings, Debug
    app.nb.add(app.chat_tab, text="Chat")
    app.nb.add(app.brains_tab, text="Brains")
    app.nb.add(app.datasets_tab, text="Datasets")
    app.nb.add(app.training_tab, text="HRM Training")
    app.nb.add(app.evaluation_tab, text="Evaluation")
    app.nb.add(app.resources_tab, text="Resources")
    app.nb.add(app.mcp_tab, text="MCP & Tools")
    app.nb.add(app.settings_tab, text="Settings")
    app.nb.add(app.debug_tab, text="Debug")
    app.nb.add(app.help_tab, text="Help")
    
    # Refresh brains when returning to Brains tab
    def _on_tab_changed(event):
        tab_text = None
        try:
            tab_id = app.nb.select()
            tab_text = app.nb.tab(tab_id, "text")
            logger.info("Notebook tab changed to %s", tab_text)
        except Exception:
            logger.debug("Failed to resolve notebook tab selection", exc_info=True)
            return

        try:
            if tab_text == "Brains":
                if hasattr(app, 'brains_panel') and app.brains_panel:
                    app.brains_panel.refresh()  # populated later
            elif tab_text == "MCP & Tools":
                if hasattr(app, 'mcp_panel') and app.mcp_panel:
                    app.mcp_panel.refresh()  # populated later
            elif tab_text == "Resources":
                try:
                    if hasattr(app, 'resources_panel') and app.resources_panel:
                        logger.info("Resources tab activation handler start")
                        app.resources_panel.on_tab_activated()
                        logger.info("Resources tab activation handler done")
                except Exception:
                    logger.exception("Resources tab activation handler failed")
            elif tab_text == "Evaluation":
                try:
                    if hasattr(app, 'evaluation_panel') and app.evaluation_panel:
                        logger.info("Evaluation tab activation handler start")
                        app.evaluation_panel.on_tab_activated()
                        logger.info("Evaluation tab activation handler done")
                except Exception:
                    logger.exception("Evaluation tab activation handler failed")
            elif tab_text == "Help":
                if hasattr(app, 'help_panel') and app.help_panel:
                    app.help_panel.focus_search()
        except Exception:
            logger.exception("Notebook tab change side-effect failed")
    
    cast(Any, app.nb).bind("<<NotebookTabChanged>>", _on_tab_changed)

    # Global guard: prevent Matplotlib's global <MouseWheel> handler from
    # processing events with malformed event.widget (e.g., 'str' on Windows).
    # Returning 'break' here stops further handlers when widget is invalid.
    try:
        def _wheel_guard(event):
            try:
                w = getattr(event, 'widget', None)
                if (w is None) or isinstance(w, str) or (not hasattr(w, 'winfo_containing')):
                    return "break"
            except Exception:
                return "break"
            return None
        root.bind_all("<MouseWheel>", _wheel_guard, add=True)
        root.bind_all("<Shift-MouseWheel>", _wheel_guard, add=True)
        root.bind_all("<Control-MouseWheel>", _wheel_guard, add=True)
    except Exception:
        pass
    
    # --- Top actions bar (Datasets tab) ---
    top = ttk.Frame(app.datasets_tab)
    top.pack(fill="x", padx=8, pady=4)
    
    # Core toggles/state (no direct UI here; ResourcesPanel handles devices)
    app.cpu_var = safe_variables.BooleanVar(value=False)
    app.cuda_var = safe_variables.BooleanVar(value=False)
    app.xpu_var = safe_variables.BooleanVar(value=False)
    app.dml_var = safe_variables.BooleanVar(value=False)
    app.mps_var = safe_variables.BooleanVar(value=False)
    app.dml_py_var = safe_variables.StringVar(value="")
    app.dataset_path_var = safe_variables.StringVar(value="")
    
    # Resource controls state
    try:
        cores = max(1, os.cpu_count() or 1)
    except Exception:
        cores = 1
    
    app._cores = cores
    
    # Command var used by on_run (legacy)
    app.cmd_var = safe_variables.StringVar(value="")
    
    # Create horizontal split container for output and builder
    top_container = ttk.Frame(app.datasets_tab)
    top_container.pack(fill="both", expand=True, padx=8, pady=(4, 8))
    
    # Configure grid weights for 50/50 split
    top_container.grid_columnconfigure(0, weight=1)
    top_container.grid_columnconfigure(1, weight=1)
    top_container.grid_rowconfigure(0, weight=1)
    
    # Left side: Output box (will be created by OutputPanel)
    app._left_frame = ttk.Frame(top_container)
    app._left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
    
    # Right side: Dataset Builder
    app._right_frame = ttk.Frame(top_container)
    app._right_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

    # Global keybinding: F1 opens Help and focuses search
    try:
        def _open_help(event=None):
            try:
                for i in range(app.nb.index("end")):
                    if app.nb.tab(i, "text") == "Help":
                        app.nb.select(i)
                        break
                if hasattr(app, 'help_panel') and app.help_panel:
                    app.help_panel.focus_search()
            except Exception:
                pass
            return "break"
        root.bind_all("<F1>", _open_help)
    except Exception:
        pass
