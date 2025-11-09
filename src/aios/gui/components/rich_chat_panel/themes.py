"""Theme constants and utilities for rich chat panel."""

from __future__ import annotations

from typing import Any, cast

try:
    import tkinter.ttk as ttk  # type: ignore
except Exception:  # pragma: no cover
    ttk = cast(Any, None)


# Theme-aware color schemes
THEMES = {
    "light": {
        "canvas_bg": "#FFFFFF",
        "user_label_fg": "#0066CC",
        "user_msg_bg": "#E3F2FD",
        "user_msg_fg": "#000000",
        "user_msg_border": "#BBDEFB",
        "ai_label_fg": "#2E7D32",
        "ai_msg_bg": "#F1F8E9",
        "ai_msg_fg": "#000000",
        "ai_msg_border": "#C5E1A5",
        "system_bg": "#F5F5F5",
        "system_fg": "#666666",
        "loading_bg": "#FFFDE7",
        "loading_fg": "#666666",
    },
    "dark": {
        "canvas_bg": "#1E1E1E",
        "user_label_fg": "#4FC3F7",
        "user_msg_bg": "#263238",
        "user_msg_fg": "#E0E0E0",
        "user_msg_border": "#37474F",
        "ai_label_fg": "#81C784",
        "ai_msg_bg": "#2E3B2E",
        "ai_msg_fg": "#E0E0E0",
        "ai_msg_border": "#4CAF50",
        "system_bg": "#2b2b2b",
        "system_fg": "#AAAAAA",
        "loading_bg": "#3E2723",
        "loading_fg": "#BCAAA4",
    },
    "matrix": {
        "canvas_bg": "#000000",
        "user_label_fg": "#00FF41",
        "user_msg_bg": "#001A00",
        "user_msg_fg": "#00FF41",
        "user_msg_border": "#003300",
        "ai_label_fg": "#39FF14",
        "ai_msg_bg": "#0A0A0A",
        "ai_msg_fg": "#00CC33",
        "ai_msg_border": "#00FF41",
        "system_bg": "#0A0A0A",
        "system_fg": "#00CC33",
        "loading_bg": "#001A00",
        "loading_fg": "#00FF41",
    },
    "barbie": {
        "canvas_bg": "#FFB6C1",
        "user_label_fg": "#FF1493",
        "user_msg_bg": "#FFF0F5",
        "user_msg_fg": "#8B008B",
        "user_msg_border": "#FF69B4",
        "ai_label_fg": "#DA70D6",
        "ai_msg_bg": "#FFF0F5",
        "ai_msg_fg": "#8B008B",
        "ai_msg_border": "#DA70D6",
        "system_bg": "#FFE4E1",
        "system_fg": "#C71585",
        "loading_bg": "#FFB6C1",
        "loading_fg": "#8B008B",
    },
}


def detect_theme() -> str:
    """Detect current theme from ttk style or return default.
    
    Returns:
        Theme name: 'light', 'dark', 'matrix', or 'barbie'
    """
    try:
        if ttk is None:
            return "light"
        
        style = ttk.Style()
        # Try to detect from style settings
        bg = style.lookup(".", "background")
        if bg:
            # Convert to RGB to check brightness and color
            try:
                # Simple heuristic: dark themes have dark backgrounds
                if bg.startswith("#"):
                    r = int(bg[1:3], 16)
                    g = int(bg[3:5], 16)
                    b = int(bg[5:7], 16)
                    brightness = (r + g + b) / 3
                    # Check for pink (high red, moderate green, high blue)
                    if r > 200 and g > 150 and b > 150 and r > b:
                        return "barbie"
                    if brightness < 50:  # Very dark = matrix
                        if g > r and g > b:  # Greenish
                            return "matrix"
                        return "dark"
                    elif brightness < 128:  # Dark
                        return "dark"
            except Exception:
                pass
    except Exception:
        pass
    return "light"  # Default


def get_colors(theme: str) -> dict[str, str]:
    """Get colors for the specified theme.
    
    Args:
        theme: Theme name ('light', 'dark', 'matrix', 'barbie')
    
    Returns:
        Dictionary of color values for the theme
    """
    return THEMES.get(theme, THEMES["light"])
