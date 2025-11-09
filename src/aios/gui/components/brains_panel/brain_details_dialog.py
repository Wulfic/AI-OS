"""Create brain details dialog window.

Tkinter UI for displaying brain details in a scrollable dialog.
"""

from __future__ import annotations

from typing import Any, cast

try:  # pragma: no cover
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    tk = cast(Any, None)
    ttk = cast(Any, None)


def show_brain_details_dialog(parent: Any, brain_name: str, details_text: str) -> None:
    """Show brain details in a scrollable dialog window.
    
    Creates a Toplevel window with text widget, scrollbars, and close button.
    
    Args:
        parent: Parent Tk widget
        brain_name: Name of the brain (for window title)
        details_text: Formatted details text to display
    """
    if tk is None or ttk is None:
        return
    
    dialog = tk.Toplevel(parent)
    dialog.title(f"Brain Details: {brain_name}")
    dialog.geometry("700x750")
    dialog.transient(parent)
    
    # Text widget with scrollbar
    frame = ttk.Frame(dialog, padding=10)
    frame.pack(fill="both", expand=True)
    
    # Use monospace font for better formatting
    text_widget = tk.Text(
        frame, 
        wrap="none",  # Don't wrap - use horizontal scrollbar if needed
        width=80, 
        height=40,
        font=("Consolas", 10),
        bg="#f5f5f5",
        fg="#000000",
        padx=10,
        pady=10
    )
    
    # Vertical scrollbar
    v_scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text_widget.yview)
    text_widget.configure(yscrollcommand=v_scrollbar.set)
    
    # Horizontal scrollbar for long lines
    h_scrollbar = ttk.Scrollbar(frame, orient="horizontal", command=text_widget.xview)
    text_widget.configure(xscrollcommand=h_scrollbar.set)
    
    # Grid layout for scrollbars
    text_widget.grid(row=0, column=0, sticky="nsew")
    v_scrollbar.grid(row=0, column=1, sticky="ns")
    h_scrollbar.grid(row=1, column=0, sticky="ew")
    
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    
    text_widget.insert("1.0", details_text)
    text_widget.configure(state="disabled")  # Make read-only
    
    # Close button
    btn_frame = ttk.Frame(dialog, padding=(10, 5))
    btn_frame.pack(fill="x")
    btn_close = ttk.Button(btn_frame, text="Close", command=dialog.destroy, width=12)
    btn_close.pack(side="right")
