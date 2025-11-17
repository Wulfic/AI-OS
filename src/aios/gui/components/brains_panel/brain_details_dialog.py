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

from aios.gui.utils.theme_utils import (
    apply_theme_to_toplevel,
    compute_popup_dimensions,
    get_theme_colors,
)


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
    dialog.transient(parent)
    dialog.resizable(True, True)

    width, height, x, y = compute_popup_dimensions(
        parent,
        width_ratio=0.38,
        height_ratio=0.82,
        min_width=820,
        min_height=760,
    )
    dialog.geometry(f"{width}x{height}+{x}+{y}")
    dialog.minsize(780, 700)

    apply_theme_to_toplevel(dialog)
    colors = get_theme_colors()

    dialog.grid_rowconfigure(0, weight=1)
    dialog.grid_columnconfigure(0, weight=1)

    frame = tk.Frame(dialog, bg=colors["bg"], padx=16, pady=14)
    frame.grid(row=0, column=0, sticky="nsew")

    text_widget = tk.Text(
        frame,
        wrap="none",
        font=("TkFixedFont", 11),
        bg=colors["entry_bg"],
        fg=colors["fg"],
        insertbackground=colors["insert_bg"],
        selectbackground=colors["select_bg"],
        selectforeground=colors.get("select_fg", colors["fg"]),
        padx=14,
        pady=10,
        borderwidth=0,
        highlightthickness=0,
    )

    v_scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text_widget.yview)
    text_widget.configure(yscrollcommand=v_scrollbar.set)

    h_scrollbar = ttk.Scrollbar(frame, orient="horizontal", command=text_widget.xview)
    text_widget.configure(xscrollcommand=h_scrollbar.set)

    text_widget.grid(row=0, column=0, sticky="nsew")
    v_scrollbar.grid(row=0, column=1, sticky="ns")
    h_scrollbar.grid(row=1, column=0, sticky="ew")

    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    text_widget.insert("1.0", details_text)
    text_widget.configure(state="disabled")

    btn_frame = tk.Frame(dialog, bg=colors["bg"], padx=16, pady=12)
    btn_frame.grid(row=1, column=0, sticky="ew")
    btn_frame.grid_columnconfigure(0, weight=1)

    btn_close = ttk.Button(btn_frame, text="Close", command=dialog.destroy, width=14)
    btn_close.grid(row=0, column=0, sticky="e")

    try:
        dialog.bind("<Escape>", lambda event: dialog.destroy())
    except Exception:
        pass

    try:
        dialog.focus_set()
        dialog.grab_set()
    except Exception:
        pass
