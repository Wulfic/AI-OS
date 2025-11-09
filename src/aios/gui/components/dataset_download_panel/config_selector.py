"""
Config Selector Dialog

Helper function for selecting dataset configurations.
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Optional


def show_config_selector(dataset_name: str, configs: List[str], parent: tk.Widget) -> Optional[str]:
    """
    Show dialog to select a dataset config.
    
    Args:
        dataset_name: Name of the dataset
        configs: List of available config names
        parent: Parent widget
        
    Returns:
        Selected config name or None if cancelled
    """
    # Create config selection dialog
    dialog = tk.Toplevel(parent)
    dialog.title(f"Select Config for {dataset_name}")
    dialog.geometry("500x400")
    dialog.transient(parent)
    dialog.grab_set()
    
    selected_config = None
    
    # Instructions
    instructions = ttk.Label(
        dialog,
        text=f"This dataset has multiple configurations.\nPlease select one to download:",
        justify="left",
        font=("TkDefaultFont", 10)
    )
    instructions.pack(pady=15, padx=15)
    
    # Config list frame
    list_frame = ttk.Frame(dialog)
    list_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))
    
    # Scrollbar
    scrollbar = ttk.Scrollbar(list_frame)
    scrollbar.pack(side="right", fill="y")
    
    # Listbox for configs
    config_listbox = tk.Listbox(
        list_frame,
        yscrollcommand=scrollbar.set,
        font=("Consolas", 9),
        height=12
    )
    config_listbox.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=config_listbox.yview)
    
    # Add configs to listbox
    for cfg in configs:
        config_listbox.insert(tk.END, cfg)
    
    # Select first item by default
    if configs:
        config_listbox.selection_set(0)
        config_listbox.activate(0)
    
    # Button frame
    btn_frame = ttk.Frame(dialog)
    btn_frame.pack(fill="x", padx=15, pady=(0, 15))
    
    def on_ok():
        nonlocal selected_config
        selection = config_listbox.curselection()
        if selection:
            selected_config = configs[selection[0]]
        dialog.destroy()
    
    def on_cancel():
        nonlocal selected_config
        selected_config = None
        dialog.destroy()
    
    # Double-click to select
    config_listbox.bind("<Double-1>", lambda e: on_ok())
    
    ttk.Button(
        btn_frame,
        text="✓ OK",
        command=on_ok,
        width=12
    ).pack(side="left", padx=(0, 5))
    
    ttk.Button(
        btn_frame,
        text="✗ Cancel",
        command=on_cancel,
        width=12
    ).pack(side="left")
    
    # Wait for dialog to close
    dialog.wait_window()
    
    return selected_config
