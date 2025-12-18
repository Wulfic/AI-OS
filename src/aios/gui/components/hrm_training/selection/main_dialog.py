"""Main student selection dialog for HRM training.

This dialog shows a list of existing brains and provides options to:
- Select an existing brain
- Browse for a checkpoint file
- Create a new brain
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog
from typing import Any

from .brain_scanner import scan_existing_brains
from .brain_loader import load_brain_to_panel, load_brain_from_file
from .create_dialog import show_create_dialog
from aios.gui.utils.theme_utils import apply_theme_to_toplevel


def show_selection_dialog(panel: Any) -> None:
    """
    Show the main student selection dialog.
    
    This function creates a Tkinter dialog with:
    - Listbox showing existing brains
    - "Use Selected" button to load selected brain
    - "Browse..." button to pick arbitrary checkpoint file
    - "Create New..." button to create new brain
    - "Cancel" button to close dialog
    
    Args:
        panel: HRM training panel instance
    """
    try:
        top = tk.Toplevel(panel)
        top.title("Select HRM Student")
        top.grab_set()
        
        # Apply theme styling to match the current theme
        apply_theme_to_toplevel(top)
        
        frm = ttk.Frame(top)
        frm.pack(fill="both", expand=True, padx=10, pady=10)
        
        # ===== LISTBOX SHOWING EXISTING BRAINS =====
        ttk.Label(frm, text="Existing brains:").pack(anchor="w")
        lb = tk.Listbox(frm, width=60, height=10)
        lb.pack(fill="both", expand=True)
        
        # Scan for existing brains
        name_to_dir = scan_existing_brains(panel._project_root)
        for name in sorted(name_to_dir.keys()):
            lb.insert("end", name)
        
        # ===== BUTTON ROW =====
        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=(8, 0))
        
        # ===== "USE SELECTED" BUTTON =====
        def _use_selected():
            try:
                idxs = lb.curselection()
                if idxs:
                    sel_name = lb.get(idxs[0])
                    brain_dir = name_to_dir.get(sel_name)
                    if brain_dir:
                        load_brain_to_panel(panel, brain_dir, sel_name)
                top.destroy()
            except Exception:
                top.destroy()
        
        # ===== "BROWSE..." BUTTON =====
        def _browse_any():
            try:
                path = filedialog.askopenfilename(
                    initialdir=panel._project_root,
                    filetypes=[
                        ("PyTorch model", "*.pt"),
                        ("SafeTensors", "*.safetensors"),
                        ("All files", "*.*")
                    ]
                )
                if path:
                    load_brain_from_file(panel, path)
                top.destroy()
            except Exception:
                top.destroy()
        
        # ===== "CREATE NEW..." BUTTON =====
        def _create_new():
            try:
                show_create_dialog(top, panel)
            except Exception as e:
                panel._log(f"[hrm] Error opening create dialog: {e}")
                try:
                    top.destroy()
                except Exception:
                    pass
        
        # ===== ADD BUTTONS =====
        ttk.Button(btns, text="Use Selected", command=_use_selected).pack(side="left")
        ttk.Button(btns, text="Browse…", command=_browse_any).pack(side="left", padx=(6,0))
        ttk.Button(btns, text="Create New…", command=_create_new).pack(side="left", padx=(6,0))
        ttk.Button(btns, text="Cancel", command=top.destroy).pack(side="right")
        
    except Exception as e:
        panel._log(f"[hrm] Failed to open selection dialog: {e}")
