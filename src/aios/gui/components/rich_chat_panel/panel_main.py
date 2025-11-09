"""Main RichChatPanel class for enhanced chat with rich message rendering."""

from __future__ import annotations

from typing import Any, Callable, cast
import threading

try:
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    tk = cast(Any, None)
    ttk = cast(Any, None)

from .themes import detect_theme, get_colors
from . import ui_builder, chat_operations, brain_management, export_utils


class RichChatPanel:
    """Enhanced chat panel with support for rich content rendering.
    
    Features:
    - Code blocks with syntax highlighting and copy buttons
    - Inline image display
    - Video preview and controls
    - Clickable links
    - Export to HTML/Markdown
    - Theme-aware colors
    - Performance optimizations
    
    Args:
        parent: Tk container
        on_send: Callback that takes user message and streaming callbacks (on_token, on_done, on_error)
        title: Frame label
        on_load_brain: Optional callback for loading a specific brain
        on_list_brains: Optional callback that returns list of available brain names
        on_unload_model: Optional callback for unloading current model to free memory
        worker_pool: Optional worker pool for async operations
    """

    def __init__(
        self, 
        parent: "tk.Misc",  # type: ignore[name-defined]
        on_send: Callable[[str, Callable[[str], None], Callable[[], None], Callable[[str], None], str | None, int], None],
        *,
        title: str = "Chat",
        on_load_brain: Callable[[str], str] | None = None,
        on_list_brains: Callable[[], list[str]] | None = None,
        on_unload_model: Callable[[], str] | None = None,
        worker_pool: Any = None,
    ) -> None:
        if tk is None:
            raise RuntimeError("Tkinter not available")
        
        # Store callbacks
        self._on_send = on_send
        self._on_load_brain = on_load_brain
        self._on_list_brains = on_list_brains
        self._on_unload_model = on_unload_model
        self._worker_pool = worker_pool
        
        # Initialize state
        self._message_history: list[dict[str, Any]] = []
        self._parent = parent
        self._current_theme = detect_theme()
        self._stop_event = threading.Event()
        self._current_thread: threading.Thread | None = None
        
        # UI variables (will be set by ui_builder)
        self._context_length_var: Any = None
        self._context_entry: Any = None
        self._context_label: Any = None
        
        # Create main frame
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(fill="both", expand=True, padx=4, pady=4)
        self._frame = frame
        
        # Build UI components using helper modules
        ui_builder.build_brain_selector(self, frame)
        ui_builder.build_context_controls(self, frame)
        ui_builder.build_sampling_controls(self, frame)
        ui_builder.build_chat_area(self, frame)
        ui_builder.build_input_area(self, frame)
        
        # Defer initial load of brain list to avoid blocking GUI startup
        # This will be called after the main window is displayed
        # if self._on_load_brain and self._on_list_brains:
        #     brain_management.refresh_brains(self)
    
    def refresh_brain_list(self) -> None:
        """Refresh the brain list - can be called after panel initialization."""
        if self._on_load_brain and self._on_list_brains:
            from . import brain_management
            brain_management.refresh_brains(self)
    
    # Theme management
    def update_theme(self, theme: str) -> None:
        """Update theme and refresh chat colors.
        
        Args:
            theme: Theme name (e.g., 'dark', 'light', 'matrix', 'barbie')
        """
        theme_normalized = theme.lower().replace(" ", "").replace("mode", "")
        if theme_normalized == "dark":
            self._current_theme = "dark"
        elif theme_normalized == "matrix":
            self._current_theme = "matrix"
        elif theme_normalized == "barbie":
            self._current_theme = "barbie"
        else:
            self._current_theme = "light"
        
        # Update canvas background
        try:
            colors = get_colors(self._current_theme)
            self.canvas.config(bg=colors["canvas_bg"])
        except Exception:
            pass
    
    # Chat operations (delegate to chat_operations module)
    def _send(self) -> None:
        """Send user message and get AI response."""
        chat_operations.send_message(self)
    
    def _stop(self) -> None:
        """Stop the current response generation."""
        chat_operations.stop_generation(self)
    
    def clear(self) -> None:
        """Clear all messages from the chat and free memory."""
        chat_operations.clear_chat(self)
    
    def get_context_length(self) -> int:
        """Get the current context length setting.
        
        Returns:
            Context length (0 for auto-max)
        """
        return chat_operations.get_context_length(self)
    
    def update_context_range(
        self, 
        min_val: int = 256, 
        max_val: int = 8192, 
        current: int = 2048
    ) -> None:
        """Update the context input info based on loaded brain capabilities.
        
        Args:
            min_val: Minimum response length
            max_val: Maximum response length
            current: Current value
        """
        chat_operations.update_context_range(self, min_val, max_val, current)
    
    # Brain management (delegate to brain_management module)
    def _refresh_brains(self) -> None:
        """Refresh the list of available brains."""
        brain_management.refresh_brains(self)
    
    def _load_brain(self) -> None:
        """Load the selected brain."""
        brain_management.load_brain(self)
    
    def _unload_model(self) -> None:
        """Unload the current model to free memory."""
        brain_management.unload_model(self)
    
    def update_status(self, status: str) -> None:
        """Update the model status indicator.
        
        Args:
            status: Status text
        """
        brain_management.update_status(self, status)
    
    # Export operations (delegate to export_utils module)
    def _copy_chat(self) -> None:
        """Copy chat history to clipboard."""
        export_utils.copy_chat(self)
    
    def _export_chat(self) -> None:
        """Export chat history to file."""
        export_utils.export_chat(self)
