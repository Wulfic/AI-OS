"""Theme color utilities for HRM Training Panel.

Provides theme-aware colors for Text widgets based on current TTK theme.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tkinter import ttk


def get_theme_colors() -> dict[str, str]:
    """Detect current theme and return appropriate colors for Text widgets.
    
    Returns:
        dict[str, str]: Color mapping for bg, fg, selectbg, selectfg, insertbg
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
            
            # Check for specific themes
            if brightness < 50:  # Very dark (Matrix, Halloween, or Dark)
                if g > r and g > b:  # Greenish = Matrix
                    return {
                        "bg": "#000000",
                        "fg": "#00ff41",
                        "selectbg": "#003300",
                        "selectfg": "#00ff41",
                        "insertbg": "#00ff41"
                    }
                elif r > 50 and r > g * 2:  # Orange-ish (low brightness but high red) = Halloween
                    return {
                        "bg": "#1a0f00",
                        "fg": "#ff6600",
                        "selectbg": "#ff6600",
                        "selectfg": "#1a0f00",
                        "insertbg": "#ff6600"
                    }
                else:  # Dark mode
                    return {
                        "bg": "#2b2b2b",
                        "fg": "#e0e0e0",
                        "selectbg": "#404040",
                        "selectfg": "#e0e0e0",
                        "insertbg": "#e0e0e0"
                    }
            elif brightness < 128:  # Dark mode
                return {
                    "bg": "#2b2b2b",
                    "fg": "#e0e0e0",
                    "selectbg": "#404040",
                    "selectfg": "#e0e0e0",
                    "insertbg": "#e0e0e0"
                }
            elif r > 200 and g > 150 and b > 150 and r > b:  # Barbie mode (pinkish)
                return {
                    "bg": "#ffe4f0",
                    "fg": "#c71585",
                    "selectbg": "#ff69b4",
                    "selectfg": "#ffffff",
                    "insertbg": "#c71585"
                }
    except Exception:
        pass
    
    # Default to light theme
    return {
        "bg": "#ffffff",
        "fg": "#000000",
        "selectbg": "#0078d7",
        "selectfg": "#ffffff",
        "insertbg": "#000000"
    }
