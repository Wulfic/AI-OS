"""UI structure setup for AI-OS GUI application.

This module creates the main UI structure:
- Notebook widget with tabs
- Tab frames for each section
- Tab change event handlers
- Basic state variables
"""

from __future__ import annotations

import os
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
        try:
            tab_id = app.nb.select()
            tab_text = app.nb.tab(tab_id, "text")
            if tab_text == "Brains":
                try:
                    app.brains_panel.refresh()  # populated later
                except Exception:
                    pass
            elif tab_text == "MCP & Tools":
                try:
                    app.mcp_panel.refresh()  # populated later
                except Exception:
                    pass
            elif tab_text == "Evaluation":
                # Panel already loaded during startup
                pass
            elif tab_text == "Help":
                try:
                    # ensure help panel has focus on search for quick typing
                    app.help_panel.focus_search()
                except Exception:
                    pass
        except Exception:
            pass
    
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
    app.cpu_var = tk.BooleanVar(value=False)
    app.cuda_var = tk.BooleanVar(value=False)
    app.xpu_var = tk.BooleanVar(value=False)
    app.dml_var = tk.BooleanVar(value=False)
    app.mps_var = tk.BooleanVar(value=False)
    app.dml_py_var = tk.StringVar(value="")
    app.dataset_path_var = tk.StringVar(value="")
    
    # Resource controls state
    try:
        cores = max(1, os.cpu_count() or 1)
    except Exception:
        cores = 1
    
    app._cores = cores
    
    # Command var used by on_run (legacy)
    app.cmd_var = tk.StringVar(value="")
    
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
