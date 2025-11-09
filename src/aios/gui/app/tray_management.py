"""System tray icon management for AI-OS GUI application.

This module handles initialization and management of the system tray icon:
- Tray icon creation
- Tray menu actions
- Minimize to tray behavior
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import tkinter as tk

from ..utils.tray import TrayManager


def init_tray(app: Any) -> None:
    """Initialize system tray icon.
    
    Args:
        app: The AiosTkApp instance
    
    Creates system tray icon with menu for:
    - Show/Hide window
    - Settings access
    - Exit application
    """
    try:
        icon_dir = Path(__file__).parent.parent.parent.parent.parent / "installers"
        ico_path = icon_dir / "AI-OS.ico"
        
        app._tray_manager = TrayManager(
            app.root,
            icon_path=ico_path if ico_path.exists() else None,
            app_name="AI-OS",
            on_settings=lambda: on_tray_settings(app)
        )
        
        if app._tray_manager.create_tray():
            try:
                app._append_out("[tray] System tray icon created")
            except Exception:
                pass
        else:
            try:
                app._append_out("[tray] System tray not available")
            except Exception:
                pass
    except Exception as e:
        try:
            app._append_out(f"[tray] Failed to create tray: {e}")
        except Exception:
            pass


def on_tray_settings(app: Any) -> None:
    """Switch to settings tab (called from tray menu).
    
    Args:
        app: The AiosTkApp instance
    """
    try:
        # Find settings tab index
        for i in range(app.nb.index("end")):
            if app.nb.tab(i, "text") == "Settings":
                app.nb.select(i)
                break
    except Exception as e:
        try:
            app._append_out(f"[tray] Error switching to settings: {e}")
        except Exception:
            pass
