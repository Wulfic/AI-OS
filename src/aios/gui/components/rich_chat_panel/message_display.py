"""Message display utilities for rich chat panel."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

try:
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    tk = cast(Any, None)
    ttk = cast(Any, None)

from ..message_parser import MessageParser

if TYPE_CHECKING:
    from .panel_main import RichChatPanel

from .themes import get_colors
from . import message_rendering


# Performance settings
MAX_MESSAGES = 100  # Keep last 100 messages to prevent memory bloat


def add_user_message(panel: RichChatPanel, message: str) -> None:
    """Add a user message to the chat with theme-aware colors.
    
    Args:
        panel: Rich chat panel instance
        message: User message text
    """
    if tk is None or ttk is None:
        return
    
    colors = get_colors(panel._current_theme)
    msg_frame = ttk.Frame(panel.messages_frame)
    msg_frame.pack(fill="x", padx=5, pady=(8, 2))
    
    # User label with theme color
    user_label = ttk.Label(
        msg_frame,
        text="You:",
        font=("Segoe UI", 10, "bold"),
        foreground=colors["user_label_fg"]
    )
    user_label.pack(anchor="w", pady=(0, 2))
    
    # Message content with theme colors and improved styling
    content_label = tk.Label(
        msg_frame,
        text=message,
        font=("Segoe UI", 10),
        bg=colors["user_msg_bg"],
        fg=colors["user_msg_fg"],
        justify="left",
        wraplength=600,
        padx=12,
        pady=10,
        relief="flat",
        borderwidth=0,
        highlightthickness=1,
        highlightbackground=colors["user_msg_border"],
        highlightcolor=colors["user_msg_border"]
    )
    content_label.pack(anchor="w", fill="x")
    
    # Store in history and trim if needed
    panel._message_history.append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now()
    })
    trim_old_messages(panel)


def add_ai_message(panel: RichChatPanel, message: str) -> None:
    """Add an AI message with rich rendering and theme-aware colors.
    
    Args:
        panel: Rich chat panel instance
        message: AI response text (may contain code blocks, images, etc.)
    """
    if ttk is None:
        return
    
    colors = get_colors(panel._current_theme)
    msg_frame = ttk.Frame(panel.messages_frame)
    msg_frame.pack(fill="x", padx=5, pady=(8, 2))
    
    # AI label with theme color
    ai_label = ttk.Label(
        msg_frame,
        text="AI:",
        font=("Segoe UI", 10, "bold"),
        foreground=colors["ai_label_fg"]
    )
    ai_label.pack(anchor="w", pady=(0, 2))
    
    # Parse message into segments
    segments = MessageParser.parse(message)
    
    # Render each segment
    for segment in segments:
        if segment.type == "text":
            message_rendering.render_text(panel, msg_frame, segment.content)
        elif segment.type == "code":
            message_rendering.render_code(panel, msg_frame, segment)
        elif segment.type == "image":
            message_rendering.render_image(panel, msg_frame, segment)
        elif segment.type == "video":
            message_rendering.render_video(panel, msg_frame, segment)
    
    # Store in history
    panel._message_history.append({
        "role": "assistant",
        "content": message,
        "timestamp": datetime.now(),
        "segments": segments
    })


def add_system_message(panel: RichChatPanel, message: str) -> None:
    """Add a system message to the chat with theme-aware colors.
    
    Args:
        panel: Rich chat panel instance
        message: System message text
    """
    if tk is None or ttk is None:
        return
    
    colors = get_colors(panel._current_theme)
    msg_frame = ttk.Frame(panel.messages_frame)
    msg_frame.pack(fill="x", padx=5, pady=(4, 2))
    
    content_label = tk.Label(
        msg_frame,
        text=message,
        font=("Segoe UI", 9, "italic"),
        fg=colors["system_fg"],
        bg=colors["system_bg"],
        justify="left",
        wraplength=600,
        padx=10,
        pady=6
    )
    content_label.pack(anchor="w", fill="x")


def add_loading_message(panel: RichChatPanel) -> Any:
    """Add a loading indicator message with theme-aware colors.
    
    Args:
        panel: Rich chat panel instance
    
    Returns:
        The frame containing the loading message (for later removal)
    """
    if tk is None or ttk is None:
        return None
    
    colors = get_colors(panel._current_theme)
    msg_frame = ttk.Frame(panel.messages_frame)
    msg_frame.pack(fill="x", padx=5, pady=(8, 2))
    
    ai_label = ttk.Label(
        msg_frame,
        text="AI:",
        font=("Segoe UI", 10, "bold"),
        foreground=colors["ai_label_fg"]
    )
    ai_label.pack(anchor="w", pady=(0, 2))
    
    loading_label = tk.Label(
        msg_frame,
        text="⏳ Thinking...",
        font=("Segoe UI", 10, "italic"),
        fg=colors["loading_fg"],
        bg=colors["loading_bg"],
        padx=12,
        pady=8
    )
    loading_label.pack(anchor="w")
    
    return msg_frame


def trim_old_messages(panel: RichChatPanel) -> None:
    """Remove old messages if history exceeds MAX_MESSAGES.
    
    Args:
        panel: Rich chat panel instance
    """
    try:
        if len(panel._message_history) > MAX_MESSAGES:
            # Remove oldest messages from display
            widgets = panel.messages_frame.winfo_children()
            to_remove = len(panel._message_history) - MAX_MESSAGES
            for i in range(min(to_remove, len(widgets))):
                widgets[i].destroy()
            # Trim history
            panel._message_history = panel._message_history[-MAX_MESSAGES:]
    except Exception:
        pass
