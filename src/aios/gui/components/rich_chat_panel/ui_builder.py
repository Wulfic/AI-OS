"""UI building utilities for rich chat panel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

try:
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    tk = cast(Any, None)
    ttk = cast(Any, None)

if TYPE_CHECKING:
    from .panel_main import RichChatPanel

from .themes import get_colors
from . import event_handlers


def build_brain_selector(panel: RichChatPanel, frame: Any) -> None:
    """Build the brain selector UI components.
    
    Args:
        panel: Rich chat panel instance
        frame: Parent frame for brain selector
    """
    if ttk is None or tk is None:
        return
    
    if not (panel._on_load_brain and panel._on_list_brains):
        return
    
    brain_bar = ttk.Frame(frame)
    brain_bar.pack(fill="x", padx=4, pady=4)
    
    brain_lbl = ttk.Label(brain_bar, text="Active Brain:")
    brain_lbl.pack(side="left")
    
    panel.brain_var = tk.StringVar(value="<default>")
    panel.brain_combo = ttk.Combobox(
        brain_bar,
        textvariable=panel.brain_var,
        state="readonly",
        width=30
    )
    panel.brain_combo.pack(side="left", padx=(4, 4))
    
    load_brain_btn = ttk.Button(brain_bar, text="Load Brain", command=panel._load_brain)
    load_brain_btn.pack(side="left")
    
    unload_btn = ttk.Button(brain_bar, text="Unload", command=panel._unload_model)
    unload_btn.pack(side="left", padx=(4, 0))
    
    refresh_brains_btn = ttk.Button(brain_bar, text="Refresh", command=panel._refresh_brains)
    refresh_brains_btn.pack(side="left", padx=(4, 0))
    
    # Status indicator
    panel.status_label = ttk.Label(brain_bar, text="Status: No model loaded", foreground="gray")
    panel.status_label.pack(side="left", padx=(12, 0))
    
    # Export and Copy buttons
    copy_btn = ttk.Button(brain_bar, text="📋 Copy Chat", command=panel._copy_chat)
    copy_btn.pack(side="right", padx=(4, 0))
    
    export_btn = ttk.Button(brain_bar, text="💾 Export Chat", command=panel._export_chat)
    export_btn.pack(side="right", padx=(4, 0))
    
    # Tooltips
    try:
        from ..tooltips import add_tooltip
        add_tooltip(brain_lbl, "Select which trained brain to use for chat responses.")
        add_tooltip(panel.brain_combo, "Available trained brains. Select one and click Load Brain.")
        add_tooltip(load_brain_btn, "Load the selected brain into the chat system.")
        add_tooltip(unload_btn, "Unload the current model to free GPU memory.")
        add_tooltip(refresh_brains_btn, "Refresh the list of available brains.")
        add_tooltip(panel.status_label, "Shows the current model loading status.")
        add_tooltip(copy_btn, "Copy entire chat history to clipboard.")
        add_tooltip(export_btn, "Export chat history to HTML or Markdown file.")
    except Exception:
        pass


def build_context_controls(panel: RichChatPanel, frame: Any) -> None:
    """Build the context/response length controls.
    
    Args:
        panel: Rich chat panel instance
        frame: Parent frame for context controls
    """
    if ttk is None or tk is None:
        return
    
    context_bar = ttk.Frame(frame)
    context_bar.pack(fill="x", padx=4, pady=(0, 4))
    
    context_lbl = ttk.Label(context_bar, text="Max Response Length:")
    context_lbl.pack(side="left")
    
    # Input box with default 0 (auto-max), allow any positive number
    panel._context_length_var = tk.StringVar(value="0")
    panel._context_entry = ttk.Entry(
        context_bar,
        textvariable=panel._context_length_var,
        width=10
    )
    panel._context_entry.pack(side="left", padx=(4, 8))
    
    # Label showing info
    panel._context_label = ttk.Label(context_bar, text="chars (0 = auto max)")
    panel._context_label.pack(side="left")
    
    # Bind validation
    panel._context_entry.bind(
        "<FocusOut>", 
        lambda e: event_handlers.validate_context_length(panel, e)
    )
    panel._context_entry.bind(
        "<Return>", 
        lambda e: event_handlers.validate_context_length(panel, e)
    )
    
    # Tooltips
    try:
        from ..tooltips import add_tooltip
        add_tooltip(context_lbl, "Control the maximum length of AI responses.")
        add_tooltip(
            panel._context_entry, 
            "Enter response length in characters. Use 0 to automatically use the model's maximum. Changes take effect on next message."
        )
        add_tooltip(panel._context_label, "Input 0 to automatically use maximum available context.")
    except Exception:
        pass


def build_sampling_controls(panel: RichChatPanel, frame: Any) -> None:
    """Build sampling parameter controls (temperature, top_p, top_k).
    
    Args:
        panel: Rich chat panel instance
        frame: Parent frame for sampling controls
    """
    if ttk is None or tk is None:
        return
    
    sampling_bar = ttk.Frame(frame)
    sampling_bar.pack(fill="x", padx=4, pady=(0, 4))
    
    # Temperature
    temp_lbl = ttk.Label(sampling_bar, text="Temperature:")
    temp_lbl.pack(side="left")
    
    panel._temperature_var = tk.StringVar(value="0.7")
    temp_entry = ttk.Entry(
        sampling_bar,
        textvariable=panel._temperature_var,
        width=6
    )
    temp_entry.pack(side="left", padx=(4, 12))
    
    # Top-p
    topp_lbl = ttk.Label(sampling_bar, text="Top-p:")
    topp_lbl.pack(side="left")
    
    panel._top_p_var = tk.StringVar(value="0.9")
    topp_entry = ttk.Entry(
        sampling_bar,
        textvariable=panel._top_p_var,
        width=6
    )
    topp_entry.pack(side="left", padx=(4, 12))
    
    # Top-k
    topk_lbl = ttk.Label(sampling_bar, text="Top-k:")
    topk_lbl.pack(side="left")
    
    panel._top_k_var = tk.StringVar(value="50")
    topk_entry = ttk.Entry(
        sampling_bar,
        textvariable=panel._top_k_var,
        width=6
    )
    topk_entry.pack(side="left", padx=(4, 0))
    
    # Tooltips
    try:
        from ..tooltips import add_tooltip
        add_tooltip(temp_lbl, "Controls randomness. Lower = more focused, higher = more creative.")
        add_tooltip(temp_entry, "Temperature (0.0-2.0). Default: 0.7. Lower values make output more deterministic.")
        add_tooltip(topp_lbl, "Nucleus sampling threshold. Only tokens with cumulative probability <= top_p are considered.")
        add_tooltip(topp_entry, "Top-p (0.0-1.0). Default: 0.9. Lower values make output more focused.")
        add_tooltip(topk_lbl, "Only the top K tokens are considered for sampling.")
        add_tooltip(topk_entry, "Top-k (0 = disabled, 1-100+). Default: 50. Lower values make output more predictable.")
    except Exception:
        pass


def build_chat_area(panel: RichChatPanel, frame: Any) -> None:
    """Build the scrollable chat area with canvas.
    
    Args:
        panel: Rich chat panel instance
        frame: Parent frame for chat area
    """
    if ttk is None or tk is None:
        return
    
    # Create scrollable chat area
    body = ttk.Frame(frame)
    body.pack(fill="both", expand=True, padx=4, pady=4)
    
    # Canvas with scrollbar for rich content (theme-aware background)
    colors = get_colors(panel._current_theme)
    panel.canvas = tk.Canvas(body, bg=colors["canvas_bg"], highlightthickness=0)
    vsb = ttk.Scrollbar(body, orient="vertical", command=panel.canvas.yview)
    panel.canvas.configure(yscrollcommand=vsb.set)
    
    panel.canvas.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")
    
    # Frame inside canvas to hold messages
    panel.messages_frame = ttk.Frame(panel.canvas)
    panel.canvas_window = panel.canvas.create_window(
        (0, 0),
        window=panel.messages_frame,
        anchor="nw",
        tags="messages_frame"
    )
    
    # Bind canvas resize to update scroll region
    panel.messages_frame.bind(
        "<Configure>", 
        lambda e: event_handlers.on_frame_configure(panel, e)
    )
    panel.canvas.bind(
        "<Configure>", 
        lambda e: event_handlers.on_canvas_configure(panel, e)
    )
    
    # Mouse wheel scrolling
    try:
        panel.canvas.bind_all(
            "<MouseWheel>", 
            lambda e: event_handlers.on_mousewheel(panel, e)
        )
    except Exception:
        pass


def build_input_area(panel: RichChatPanel, frame: Any) -> None:
    """Build the user input area with send/stop/clear buttons.
    
    Args:
        panel: Rich chat panel instance
        frame: Parent frame for input area
    """
    if ttk is None or tk is None:
        return
    
    input_frame = ttk.Frame(frame)
    input_frame.pack(fill="x", padx=4, pady=(0, 4))
    
    you_lbl = ttk.Label(input_frame, text="You:")
    you_lbl.pack(side="left")
    
    panel.text_var = tk.StringVar(value="")
    panel.entry = ttk.Entry(input_frame, textvariable=panel.text_var)
    panel.entry.pack(side="left", fill="x", expand=True, padx=(4, 8))
    
    # Bind Enter key
    try:
        panel.entry.bind("<Return>", lambda e: panel._send())
    except Exception:
        pass
    
    send_btn = ttk.Button(input_frame, text="Send", command=panel._send)
    send_btn.pack(side="left")
    
    panel.stop_btn = ttk.Button(input_frame, text="Stop", command=panel._stop, state="disabled")
    panel.stop_btn.pack(side="left", padx=(4, 0))
    
    clear_btn = ttk.Button(input_frame, text="Clear", command=panel.clear)
    clear_btn.pack(side="left", padx=(4, 0))
    
    # Tooltips
    try:
        from ..tooltips import add_tooltip
        add_tooltip(panel.entry, "Type a command or message. Press Enter or click Send.")
        add_tooltip(send_btn, "Send the message to the AI.")
        add_tooltip(panel.stop_btn, "Stop the current response generation.")
        add_tooltip(clear_btn, "Clear the chat history.")
    except Exception:
        pass
