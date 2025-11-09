"""Event handlers for rich chat panel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .panel_main import RichChatPanel


def on_frame_configure(panel: RichChatPanel, event: Any = None) -> None:
    """Update scroll region when frame size changes.
    
    Args:
        panel: Rich chat panel instance
        event: Configure event (unused)
    """
    try:
        panel.canvas.configure(scrollregion=panel.canvas.bbox("all"))
    except Exception:
        pass


def on_canvas_configure(panel: RichChatPanel, event: Any) -> None:
    """Update canvas window width when canvas is resized.
    
    Args:
        panel: Rich chat panel instance
        event: Configure event with width
    """
    try:
        canvas_width = event.width
        panel.canvas.itemconfig(panel.canvas_window, width=canvas_width)
    except Exception:
        pass


def on_mousewheel(panel: RichChatPanel, event: Any) -> None:
    """Handle mouse wheel scrolling.
    
    Args:
        panel: Rich chat panel instance
        event: Mouse wheel event
    """
    try:
        panel.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    except Exception:
        pass


def scroll_to_bottom(panel: RichChatPanel) -> None:
    """Scroll to the bottom of the chat only if user is already viewing the bottom.
    
    This prevents auto-scrolling from interrupting users who have scrolled up
    to view earlier messages.
    
    Args:
        panel: Rich chat panel instance
    """
    try:
        panel.canvas.update_idletasks()
        # Get the current view position (0.0 = top, 1.0 = bottom)
        yview = panel.canvas.yview()
        # If the bottom of the view is at or very close to the end (within ~5%),
        # then auto-scroll. Otherwise, respect the user's scroll position.
        if yview[1] >= 0.95:  # Within ~5% of the bottom
            panel.canvas.yview_moveto(1.0)
    except Exception:
        # Fallback to always scrolling if we can't check position
        try:
            panel.canvas.yview_moveto(1.0)
        except Exception:
            pass


def validate_context_length(panel: RichChatPanel, event: Any = None) -> None:
    """Validate the context length input.
    
    Args:
        panel: Rich chat panel instance
        event: Focus/return event (unused)
    """
    try:
        val_str = panel._context_length_var.get().strip()
        if not val_str:
            panel._context_length_var.set("2048")
            return
        
        val = int(val_str)
        if val < 0:
            panel._context_length_var.set("0")
        # Allow any positive number or 0 (for auto-max)
    except Exception:
        # Invalid input, reset to default
        panel._context_length_var.set("2048")
