"""Theme color definitions for the settings panel."""

from typing import Dict, Any

# Theme color configurations
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
        "theme_base": None,  # Use default platform theme for baseline widgets
        "bg": "#ffffff",
        "fg": "#000000",
        "button_bg": "#f0f0f0",
        "entry_bg": "#ffffff",
        "select_bg": "#0078d7",
        "select_fg": "#ffffff",
        "border": "#c8c8c8",
        "disabled_bg": "#e0e0e0",
        "disabled_fg": "#888888",
        "info_message": "✓ Light Mode applied",
        "info_color": "#6060a0",
    }
}

# Available theme names
THEME_NAMES = ("Light Mode", "Dark Mode", "Matrix Mode", "Barbie Mode", "Halloween Mode")
