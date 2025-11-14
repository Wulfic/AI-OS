"""Dialog for adding/editing MCP server configuration."""

from __future__ import annotations

# Import safe variable wrappers
from ...utils import safe_variables

from typing import Any, Optional, cast

try:  # pragma: no cover - environment dependent
    import tkinter as tk  # type: ignore
    from tkinter import ttk, messagebox  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    tk = cast(Any, None)
    ttk = cast(Any, None)
    messagebox = cast(Any, None)

from aios.gui.utils.theme_utils import apply_theme_to_toplevel, get_spacing_multiplier


class MCPServerDialog(tk.Toplevel):  # type: ignore[misc]
    """Dialog for adding/editing MCP server configuration."""
    
    def __init__(self, parent, title="Configure MCP Server", initial_data: Optional[dict] = None):
        super().__init__(parent)
        self.title(title)
        self.result = None
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        # Center on parent
        self.geometry("500x400")
        
        # Apply theme to this dialog
        apply_theme_to_toplevel(self)
        
        # Get spacing multiplier for current theme
        spacing = get_spacing_multiplier()
        
        # Create form
        padding = int(10 * spacing)
        form = ttk.Frame(self, padding=padding)
        form.pack(fill="both", expand=True)
        
        row = 0
        pady_val = int(5 * spacing)
        
        # Name
        ttk.Label(form, text="Name:").grid(row=row, column=0, sticky="w", pady=pady_val)
        self.name_var = safe_variables.StringVar(value=initial_data.get("name", "") if initial_data else "")
        ttk.Entry(form, textvariable=self.name_var, width=40).grid(row=row, column=1, sticky="ew", pady=pady_val)
        row += 1
        
        # Type
        ttk.Label(form, text="Type:").grid(row=row, column=0, sticky="w", pady=pady_val)
        self.type_var = safe_variables.StringVar(value=initial_data.get("type", "stdio") if initial_data else "stdio")
        type_combo = ttk.Combobox(form, textvariable=self.type_var, values=("stdio", "http", "https"), state="readonly", width=37)
        type_combo.grid(row=row, column=1, sticky="ew", pady=pady_val)
        row += 1
        
        # Command
        ttk.Label(form, text="Command:").grid(row=row, column=0, sticky="w", pady=pady_val)
        self.command_var = safe_variables.StringVar(value=initial_data.get("command", "") if initial_data else "")
        ttk.Entry(form, textvariable=self.command_var, width=40).grid(row=row, column=1, sticky="ew", pady=pady_val)
        row += 1
        
        # Args
        ttk.Label(form, text="Arguments:").grid(row=row, column=0, sticky="w", pady=pady_val)
        args_text = " ".join(initial_data.get("args", [])) if initial_data else ""
        self.args_var = safe_variables.StringVar(value=args_text)
        ttk.Entry(form, textvariable=self.args_var, width=40).grid(row=row, column=1, sticky="ew", pady=pady_val)
        row += 1
        
        # Description
        ttk.Label(form, text="Description:").grid(row=row, column=0, sticky="nw", pady=pady_val)
        self.description_text = tk.Text(form, height=4, width=40)
        self.description_text.grid(row=row, column=1, sticky="ew", pady=pady_val)
        if initial_data and "description" in initial_data:
            self.description_text.insert("1.0", initial_data["description"])
        row += 1
        
        # Enabled
        self.enabled_var = safe_variables.BooleanVar(value=initial_data.get("enabled", True) if initial_data else True)
        ttk.Checkbutton(form, text="Enabled", variable=self.enabled_var).grid(row=row, column=1, sticky="w", pady=pady_val)
        row += 1
        
        form.columnconfigure(1, weight=1)
        
        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=padding, pady=padding)
        
        padx_val = int(5 * spacing)
        ttk.Button(btn_frame, text="Save", command=self._on_save).pack(side="right", padx=padx_val)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side="right")
        
        # Bind Enter/Escape
        self.bind("<Return>", lambda e: self._on_save())
        self.bind("<Escape>", lambda e: self.destroy())
    
    def _on_save(self):
        """Save and close dialog."""
        name = self.name_var.get().strip()
        if not name:
            if messagebox:
                messagebox.showwarning("Validation Error", "Name is required")
            return
        
        command = self.command_var.get().strip()
        if not command and self.type_var.get() == "stdio":
            if messagebox:
                messagebox.showwarning("Validation Error", "Command is required for stdio servers")
            return
        
        args = self.args_var.get().strip().split() if self.args_var.get().strip() else []
        description = self.description_text.get("1.0", "end").strip()
        
        self.result = {
            "name": name,
            "type": self.type_var.get(),
            "command": command,
            "args": args,
            "description": description,
            "enabled": self.enabled_var.get(),
            "tools_count": 0,  # Will be populated when server is queried
        }
        
        self.destroy()
