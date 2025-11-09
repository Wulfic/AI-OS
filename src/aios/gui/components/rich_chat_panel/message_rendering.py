"""Message rendering utilities for rich chat panel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

try:
    import tkinter as tk  # type: ignore
except Exception:  # pragma: no cover
    tk = cast(Any, None)

from ..message_parser import MessageSegment
from ..rich_message_widgets import CodeBlockWidget, ImageWidget, VideoWidget

if TYPE_CHECKING:
    from .panel_main import RichChatPanel

from .themes import get_colors


def render_text(panel: RichChatPanel, parent: Any, text: str) -> None:
    """Render plain text segment with theme-aware colors.
    
    Args:
        panel: Rich chat panel instance
        parent: Parent widget
        text: Text content
    """
    if not text.strip():
        return
    
    if tk is None:
        return
    
    colors = get_colors(panel._current_theme)
    text_label = tk.Label(
        parent,
        text=text.strip(),
        font=("Segoe UI", 10),
        bg=colors["ai_msg_bg"],
        fg=colors["ai_msg_fg"],
        justify="left",
        wraplength=600,
        padx=12,
        pady=10,
        relief="flat",
        borderwidth=0,
        highlightthickness=1,
        highlightbackground=colors["ai_msg_border"],
        highlightcolor=colors["ai_msg_border"]
    )
    text_label.pack(anchor="w", fill="x", pady=2)


def render_code(panel: RichChatPanel, parent: Any, segment: MessageSegment) -> None:
    """Render code block segment.
    
    Args:
        panel: Rich chat panel instance
        parent: Parent widget
        segment: Code segment to render
    """
    language = segment.metadata.get("language", "text") if segment.metadata else "text"
    
    code_widget = CodeBlockWidget(
        parent,
        code=segment.content,
        language=language,
        relief="solid",
        borderwidth=1
    )
    code_widget.pack(anchor="w", fill="x", pady=2)


def render_image(panel: RichChatPanel, parent: Any, segment: MessageSegment) -> None:
    """Render image segment.
    
    Args:
        panel: Rich chat panel instance
        parent: Parent widget
        segment: Image segment to render
    """
    alt_text = segment.metadata.get("alt", "") if segment.metadata else ""
    
    image_widget = ImageWidget(
        parent,
        image_path=segment.content,
        alt_text=alt_text
    )
    image_widget.pack(anchor="w", fill="x", pady=2)


def render_video(panel: RichChatPanel, parent: Any, segment: MessageSegment) -> None:
    """Render video segment.
    
    Args:
        panel: Rich chat panel instance
        parent: Parent widget
        segment: Video segment to render
    """
    video_widget = VideoWidget(
        parent,
        video_path=segment.content
    )
    video_widget.pack(anchor="w", fill="x", pady=2)
