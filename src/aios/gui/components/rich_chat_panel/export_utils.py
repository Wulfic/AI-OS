"""Export utilities for rich chat panel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

try:
    from tkinter import filedialog, messagebox  # type: ignore
except Exception:  # pragma: no cover
    filedialog = cast(Any, None)
    messagebox = cast(Any, None)

if TYPE_CHECKING:
    from .panel_main import RichChatPanel


def copy_chat(panel: RichChatPanel) -> None:
    """Copy chat history to clipboard.
    
    Args:
        panel: Rich chat panel instance
    """
    if not panel._message_history:
        try:
            if messagebox:
                messagebox.showinfo("Copy Chat", "No messages to copy.")
        except Exception:
            pass
        return
    
    try:
        # Generate plain text version of chat
        text_lines = []
        for msg in panel._message_history:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            text_lines.append(f"{role}: {content}")
            text_lines.append("")  # Empty line between messages
        
        chat_text = "\n".join(text_lines)
        
        # Copy to clipboard using parent widget
        if panel._parent and hasattr(panel._parent, 'clipboard_clear'):
            panel._parent.clipboard_clear()
            panel._parent.clipboard_append(chat_text)
            panel._parent.update()  # Keep clipboard content after window closes
            
            try:
                if messagebox:
                    messagebox.showinfo("Copy Chat", f"Copied {len(panel._message_history)} messages to clipboard!")
            except Exception:
                pass
    except Exception as e:
        try:
            if messagebox:
                messagebox.showerror("Copy Error", f"Failed to copy chat: {e}")
        except Exception:
            pass


def export_chat(panel: RichChatPanel) -> None:
    """Export chat history to file.
    
    Args:
        panel: Rich chat panel instance
    """
    if not panel._message_history:
        try:
            if messagebox:
                messagebox.showinfo("Export", "No messages to export.")
        except Exception:
            pass
        return
    
    try:
        if filedialog is None:
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Chat",
            defaultextension=".html",
            filetypes=[
                ("HTML files", "*.html"),
                ("Markdown files", "*.md"),
                ("Text files", "*.txt")
            ]
        )
        
        if not filename:
            return
        
        if filename.endswith(".html"):
            export_html(panel, filename)
        elif filename.endswith(".md"):
            export_markdown(panel, filename)
        else:
            export_text(panel, filename)
        
        try:
            if messagebox:
                messagebox.showinfo("Export", f"Chat exported to {filename}")
        except Exception:
            pass
    except Exception as e:
        try:
            if messagebox:
                messagebox.showerror("Export Error", f"Failed to export: {e}")
        except Exception:
            pass


def export_html(panel: RichChatPanel, filename: str) -> None:
    """Export chat as HTML.
    
    Args:
        panel: Rich chat panel instance
        filename: Output filename
    """
    html = [
        "<!DOCTYPE html>", 
        "<html>", 
        "<head>",
        "<meta charset='utf-8'>",
        "<title>AI-OS Chat Export</title>",
        "<style>",
        "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }",
        ".message { margin: 15px 0; padding: 10px; border-radius: 5px; }",
        ".user { background: #E3F2FD; border-left: 4px solid #0066CC; }",
        ".assistant { background: #F1F8E9; border-left: 4px solid #4CAF50; }",
        ".role { font-weight: bold; margin-bottom: 5px; }",
        ".code { background: #1E1E1E; color: #D4D4D4; padding: 10px; border-radius: 3px; overflow-x: auto; }",
        ".timestamp { font-size: 0.8em; color: #666; }",
        "</style>",
        "</head>", 
        "<body>",
        "<h1>AI-OS Chat Export</h1>"
    ]
    
    for msg in panel._message_history:
        role = msg["role"]
        content = msg["content"]
        timestamp = msg["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        
        role_class = "user" if role == "user" else "assistant"
        role_label = "You" if role == "user" else "AI"
        
        html.append(f"<div class='message {role_class}'>")
        html.append(f"<div class='role'>{role_label} <span class='timestamp'>({timestamp})</span></div>")
        # Escape HTML content
        escaped_content = content.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
        html.append(f"<div>{escaped_content}</div>")
        html.append("</div>")
    
    html.extend(["</body>", "</html>"])
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(html))


def export_markdown(panel: RichChatPanel, filename: str) -> None:
    """Export chat as Markdown.
    
    Args:
        panel: Rich chat panel instance
        filename: Output filename
    """
    md = ["# AI-OS Chat Export", ""]
    
    for msg in panel._message_history:
        role = msg["role"]
        content = msg["content"]
        timestamp = msg["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        
        role_label = "**You**" if role == "user" else "**AI**"
        
        md.append(f"## {role_label} _{timestamp}_")
        md.append("")
        md.append(content)
        md.append("")
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def export_text(panel: RichChatPanel, filename: str) -> None:
    """Export chat as plain text.
    
    Args:
        panel: Rich chat panel instance
        filename: Output filename
    """
    lines = ["AI-OS Chat Export", "=" * 50, ""]
    
    for msg in panel._message_history:
        role = msg["role"]
        content = msg["content"]
        timestamp = msg["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        
        role_label = "You" if role == "user" else "AI"
        
        lines.append(f"{role_label} ({timestamp}):")
        lines.append(content)
        lines.append("-" * 50)
        lines.append("")
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
