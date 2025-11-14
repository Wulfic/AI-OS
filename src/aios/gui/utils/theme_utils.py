"""Centralized theme utilities for the entire GUI application.

This module provides a single source of truth for theme detection,
color retrieval, and theme-aware widget configuration across all
GUI components including dialogs, popups, and panels.
"""

from __future__ import annotations
import logging
import tkinter as tk
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Theme color configurations - centralized from theme_constants
THEME_COLORS: Dict[str, Dict[str, Any]] = {
    "Dark Mode": {
        "theme_base": "clam",
        "bg_dark": "#2b2b2b",
        "fg_light": "#e0e0e0",
        "select_bg": "#404040",
        "button_bg": "#3c3c3c",
        "entry_bg": "#353535",
        "info_message": "✓ Dark Mode applied",
        "info_color": "#60a060",
    },
    "Matrix Mode": {
        "theme_base": "clam",
        "bg_black": "#000000",
        "fg_green": "#00ff41",
        "fg_dim_green": "#00cc33",
        "select_bg": "#003300",
        "button_bg": "#001a00",
        "entry_bg": "#0a0a0a",
        "accent_green": "#39ff14",
        "info_message": "✓ Matrix Mode activated - Welcome to the Matrix",
        "info_color": "#39ff14",
    },
    "Halloween Mode": {
        "theme_base": "clam",
        "bg_black": "#1a0f00",
        "bg_dark": "#2d1a00",
        "fg_orange": "#ff6600",
        "fg_light": "#ffcc99",
        "accent_orange": "#ff8c00",
        "button_bg": "#4d2600",
        "entry_bg": "#3d1f00",
        "select_bg": "#ff6600",
        "info_message": "🎃 Halloween Mode activated - Spooky season is here!",
        "info_color": "#ff8c00",
    },
    "Barbie Mode": {
        "theme_base": "clam",
        "bg_pink": "#FFB6C1",
        "bg_hot_pink": "#FF69B4",
        "fg_dark": "#8B008B",
        "fg_white": "#FFFFFF",
        "select_bg": "#FF1493",
        "button_bg": "#FF69B4",
        "entry_bg": "#FFF0F5",
        "accent_purple": "#DA70D6",
        "info_message": "✓ Barbie Mode activated - Life in plastic, it's fantastic!",
        "info_color": "#FF1493",
    },
    "Light Mode": {
        "theme_base": None,  # Use default platform theme
        "bg": "#ffffff",
        "fg": "#000000",
        "select_bg": "#0078d7",
        "button_bg": "#f0f0f0",
        "entry_bg": "#ffffff",
        "info_message": "✓ Light Mode applied",
        "info_color": "#6060a0",
    }
}


def detect_current_theme() -> str:
    """Detect the current theme from TTK style settings.
    
    Returns:
        Theme name: 'Light Mode', 'Dark Mode', 'Matrix Mode', 'Halloween Mode', or 'Barbie Mode'
    """
    try:
        from tkinter import ttk
        style = ttk.Style()
        bg = style.lookup(".", "background")
        
        if bg and bg.startswith("#"):
            # Parse RGB values
            r = int(bg[1:3], 16)
            g = int(bg[3:5], 16)
            b = int(bg[5:7], 16)
            brightness = (r + g + b) / 3
            
            # Check for specific themes based on color characteristics
            if brightness < 50:  # Very dark themes
                if g > r and g > b:  # Greenish = Matrix
                    detected_theme = "Matrix Mode"
                elif r > 50 and r > g * 2:  # Orange-ish = Halloween
                    detected_theme = "Halloween Mode"
                else:
                    detected_theme = "Dark Mode"
            elif brightness < 128:  # Dark mode
                detected_theme = "Dark Mode"
            elif r > 200 and g > 150 and b > 150 and r > b:  # Pinkish = Barbie
                detected_theme = "Barbie Mode"
            else:
                detected_theme = "Light Mode"
            
            logger.debug(f"Theme detected: {detected_theme} (bg={bg}, brightness={brightness:.1f})")
            return detected_theme
    except Exception as e:
        logger.error(f"Theme detection failed: {e}", exc_info=True)
    
    logger.debug("Using default theme: Light Mode")
    return "Light Mode"  # Default


def get_theme_colors(theme: Optional[str] = None) -> Dict[str, str]:
    """Get standardized colors for the specified theme.
    
    Args:
        theme: Theme name (if None, auto-detects current theme)
    
    Returns:
        Dictionary with standardized color keys:
        - bg: Background color
        - fg: Foreground/text color
        - select_bg: Selection background
        - select_fg: Selection foreground
        - entry_bg: Entry/text widget background
        - button_bg: Button background
        - insert_bg: Text cursor color
    """
    if theme is None:
        theme = detect_current_theme()
    
    # Normalize theme name
    theme = theme.strip()
    
    # Get base theme colors
    colors = THEME_COLORS.get(theme, THEME_COLORS["Light Mode"])
    
    if theme not in THEME_COLORS:
        logger.error(f"Theme loading failed: {theme} - Theme not found")
        logger.warning("Using default theme")
    
    # Create standardized output dictionary
    if theme == "Dark Mode":
        return {
            "bg": colors["bg_dark"],
            "fg": colors["fg_light"],
            "select_bg": colors["select_bg"],
            "select_fg": colors["fg_light"],
            "entry_bg": colors["entry_bg"],
            "button_bg": colors["button_bg"],
            "insert_bg": colors["fg_light"],
        }
    elif theme == "Matrix Mode":
        return {
            "bg": colors["bg_black"],
            "fg": colors["fg_green"],
            "select_bg": colors["select_bg"],
            "select_fg": colors["accent_green"],
            "entry_bg": colors["entry_bg"],
            "button_bg": colors["button_bg"],
            "insert_bg": colors["accent_green"],
        }
    elif theme == "Halloween Mode":
        return {
            "bg": colors["bg_black"],
            "fg": colors["fg_light"],
            "select_bg": colors["select_bg"],
            "select_fg": colors["bg_black"],
            "entry_bg": colors["entry_bg"],
            "button_bg": colors["button_bg"],
            "insert_bg": colors["fg_orange"],
        }
    elif theme == "Barbie Mode":
        return {
            "bg": colors["bg_pink"],
            "fg": colors["fg_dark"],
            "select_bg": colors["select_bg"],
            "select_fg": colors["fg_white"],
            "entry_bg": colors["entry_bg"],
            "button_bg": colors["button_bg"],
            "insert_bg": colors["fg_dark"],
        }
    else:  # Light Mode
        return {
            "bg": colors.get("bg", "#ffffff"),
            "fg": colors.get("fg", "#000000"),
            "select_bg": colors.get("select_bg", "#0078d7"),
            "select_fg": "#ffffff",
            "entry_bg": colors.get("entry_bg", "#ffffff"),
            "button_bg": colors.get("button_bg", "#f0f0f0"),
            "insert_bg": colors.get("fg", "#000000"),
        }


def apply_theme_to_toplevel(window: tk.Toplevel, theme: Optional[str] = None) -> None:
    """Apply theme colors to a Toplevel window and its widgets.
    
    Args:
        window: The Toplevel window to theme
        theme: Theme name (if None, auto-detects current theme)
    """
    if theme is None:
        theme = detect_current_theme()
    
    logger.debug(f"Applying theme to Toplevel window: {theme}")
    colors = get_theme_colors(theme)
    
    # Configure the toplevel window background
    try:
        window.configure(bg=colors["bg"])
    except Exception as e:
        logger.error(f"Theme application failed: {theme} - {e}")
        logger.warning("Using default theme")
        return
    
    # Apply colors to all child widgets recursively
    try:
        _apply_colors_recursive(window, colors)
        logger.debug(f"Theme '{theme}' applied successfully to Toplevel window")
    except Exception as e:
        logger.error(f"Theme application failed: {e}", exc_info=True)
        logger.warning("Some widgets failed to apply theme")


def _apply_colors_recursive(widget: Any, colors: Dict[str, str]) -> None:
    """Recursively apply colors to widgets.
    
    Args:
        widget: The widget to update
        colors: Color dictionary from get_theme_colors()
    """
    try:
        # Update Text widgets
        if isinstance(widget, tk.Text):
            widget.config(
                bg=colors["entry_bg"],
                fg=colors["fg"],
                insertbackground=colors["insert_bg"],
                selectbackground=colors["select_bg"],
                selectforeground=colors["select_fg"]
            )
        # Update Listbox widgets
        elif isinstance(widget, tk.Listbox):
            widget.config(
                bg=colors["entry_bg"],
                fg=colors["fg"],
                selectbackground=colors["select_bg"],
                selectforeground=colors["select_fg"]
            )
        # Update Entry widgets
        elif isinstance(widget, tk.Entry):
            widget.config(
                bg=colors["entry_bg"],
                fg=colors["fg"],
                insertbackground=colors["insert_bg"],
                selectbackground=colors["select_bg"],
                selectforeground=colors["select_fg"]
            )
        # Update Canvas widgets
        elif isinstance(widget, tk.Canvas):
            widget.config(bg=colors["bg"])
        # Update Frame widgets
        elif isinstance(widget, tk.Frame):
            widget.config(bg=colors["bg"])
        # Update Label widgets
        elif isinstance(widget, tk.Label):
            widget.config(bg=colors["bg"], fg=colors["fg"])
        
        # Recursively update children
        for child in widget.winfo_children():
            _apply_colors_recursive(child, colors)
    except Exception as e:
        logger.warning(f"Failed to apply theme to widget {type(widget).__name__}: {e}")


def configure_global_dialogs(root: tk.Tk, theme: Optional[str] = None) -> None:
    """Configure global dialog settings (messagebox, filedialog, etc.) for theme.
    
    Args:
        root: The root Tk window
        theme: Theme name (if None, auto-detects current theme)
    """
    if theme is None:
        theme = detect_current_theme()
    
    logger.debug(f"Configuring global dialogs for theme: {theme}")
    colors = get_theme_colors(theme)
    
    try:
        # Configure Combobox dropdown colors
        root.option_add("*TCombobox*Listbox*Background", colors["entry_bg"])
        root.option_add("*TCombobox*Listbox*Foreground", colors["fg"])
        root.option_add("*TCombobox*Listbox*selectBackground", colors["select_bg"])
        root.option_add("*TCombobox*Listbox*selectForeground", colors["select_fg"])
        
        # Configure dialog backgrounds
        root.option_add("*Dialog*Background", colors["bg"])
        root.option_add("*Dialog*Foreground", colors["fg"])
        
        # Configure messagebox colors
        root.option_add("*MessageBox*Background", colors["bg"])
        root.option_add("*MessageBox*Foreground", colors["fg"])
        
        # Schedule update asynchronously to avoid blocking
        root.after_idle(lambda: root.update_idletasks())
        logger.debug("Global dialog configuration complete")
    except Exception as e:
        logger.error(f"Failed to configure global dialogs: {e}", exc_info=True)


def is_dark_theme(theme: Optional[str] = None) -> bool:
    """Check if the current or specified theme is a dark theme.
    
    Args:
        theme: Theme name (if None, auto-detects current theme)
    
    Returns:
        True if theme is dark (Dark Mode, Matrix Mode, or Halloween Mode)
    """
    if theme is None:
        theme = detect_current_theme()
    
    return theme in ("Dark Mode", "Matrix Mode", "Halloween Mode")


def get_spacing_multiplier(theme: Optional[str] = None) -> float:
    """Get the spacing multiplier for the current theme.
    
    For dark themes, returns 0.99 to reduce spacing by 1%.
    For light themes, returns 1.0 (no change).
    
    Args:
        theme: Theme name (if None, auto-detects current theme)
    
    Returns:
        Spacing multiplier (0.99 for dark themes, 1.0 for light)
    """
    if is_dark_theme(theme):
        return 0.99
    return 1.0
