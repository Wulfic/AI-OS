"""Logging utilities for the datasets panel.

Provides thread-safe logging to Tkinter text widgets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tkinter as tk


def log_to_text_widget(text_widget: "tk.Text", message: str) -> None:
    """Log a message to a Tkinter Text widget in a thread-safe manner.
    
    Args:
        text_widget: The Text widget to log to
        message: The message to log
        
    This function:
    - Checks if user is at bottom before inserting
    - Only scrolls if user was already at bottom
    - Limits output to 200 lines (keeps last 150)
    - Schedules UI updates on main thread
    """
    def _do_log():
        try:
            # Check if user is at bottom before inserting
            try:
                yview = text_widget.yview()
                at_bottom = yview[1] >= 0.95  # Within ~5% of bottom
            except Exception:
                at_bottom = True  # Default to scrolling if can't check
            
            text_widget.configure(state="normal")
            if not message.endswith("\n"):
                msg = message + "\n"
            else:
                msg = message
            text_widget.insert("end", msg)
            
            # Only scroll if user was at bottom
            if at_bottom:
                text_widget.see("end")
            
            # Limit output size
            lines = int(text_widget.index('end-1c').split('.')[0])
            if lines > 200:
                text_widget.delete("1.0", f"{lines - 150}.0")
        except Exception:
            pass  # Silently fail if widget is destroyed
        finally:
            try:
                text_widget.configure(state="disabled")
            except Exception:
                pass
    
    # Schedule on main thread to avoid Tkinter threading issues
    try:
        text_widget.after(0, _do_log)
    except Exception:
        # Fallback: try direct call (might work if already on main thread)
        _do_log()
