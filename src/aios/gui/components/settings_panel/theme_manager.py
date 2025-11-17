"""Theme management and application logic."""

from __future__ import annotations
import logging
import time
import tkinter as tk
import platform
from collections import deque
from typing import TYPE_CHECKING, Any, Sequence

from .theme_constants import THEME_COLORS
from aios.gui.utils.theme_utils import (
    configure_global_dialogs,
    compute_treeview_dimensions,
)

if TYPE_CHECKING:
    from .panel_main import SettingsPanel


logger = logging.getLogger(__name__)

_THEME_STYLE_PREFIX = "AIThemeSelector"
THEME_COMBOBOX_STYLE = f"{_THEME_STYLE_PREFIX}.TCombobox"


def _theme_specific_combobox_style(theme: str) -> str:
    token = theme.lower().replace(" ", "-")
    return f"{_THEME_STYLE_PREFIX}-{token}.TCombobox"


def _apply_combobox_styles_to_widgets(panel: "SettingsPanel", theme_style: str) -> None:
    """Reassign combobox widgets so they immediately pick up updated styles."""
    try:
        if hasattr(panel, "theme_combo"):
            panel.theme_combo.configure(style=theme_style)
    except Exception as exc:
        logger.debug("Theme combo style assignment failed for %s: %s", theme_style, exc)
        try:
            panel.theme_combo.configure(style=THEME_COMBOBOX_STYLE)
        except Exception:
            pass

    try:
        if hasattr(panel, "log_level_combo"):
            panel.log_level_combo.configure(style=THEME_COMBOBOX_STYLE)
    except Exception:
        pass


def _configure_combobox_style(
    style: "tkinter.ttk.Style",
    *,
    field_bg: str,
    fg: str,
    button_bg: str,
    select_bg: str,
    select_fg: str,
    arrowcolor: str,
    bordercolor: str | None = None,
    lightcolor: str | None = None,
    darkcolor: str | None = None,
    insertcolor: str | None = None,
    style_names: Sequence[str] | None = None,
) -> None:
    """Standardize combobox styling so previous themes cannot leak settings."""

    bordercolor = bordercolor or button_bg
    lightcolor = lightcolor or bordercolor
    darkcolor = darkcolor or bordercolor
    insertcolor = insertcolor or select_fg

    targets = style_names or ("TCombobox",)

    for style_name in targets:
        style.configure(
            style_name,
            fieldbackground=field_bg,
            foreground=fg,
            background=button_bg,
            selectbackground=select_bg,
            selectforeground=select_fg,
            arrowcolor=arrowcolor,
            bordercolor=bordercolor,
            lightcolor=lightcolor,
            darkcolor=darkcolor,
            insertcolor=insertcolor,
        )


def _map_combobox_states(
    style: "tkinter.ttk.Style",
    *,
    style_names: Sequence[str] | None = None,
    **state_map: Any,
) -> None:
    """Apply identical ttk map settings to multiple combobox styles."""

    targets = style_names or ("TCombobox",)
    for style_name in targets:
        style.map(style_name, **state_map)


def _configure_button_style(
    style: "tkinter.ttk.Style",
    *,
    background: str,
    foreground: str,
    bordercolor: str | None = None,
    focuscolor: str | None = None,
    highlightcolor: str | None = None,
    relief: str = "groove",
) -> None:
    """Ensure TButton picks up the active theme's border/focus colors."""

    bordercolor = bordercolor or background
    focuscolor = focuscolor or bordercolor
    highlightcolor = highlightcolor or focuscolor

    style.configure(
        "TButton",
        background=background,
        foreground=foreground,
        bordercolor=bordercolor,
        lightcolor=bordercolor,
        darkcolor=bordercolor,
        focuscolor=focuscolor,
        highlightcolor=highlightcolor,
        relief=relief,
    )


def _configure_toggle_styles(
    style: "tkinter.ttk.Style",
    *,
    background: str,
    foreground: str,
    focuscolor: str | None = None,
) -> None:
    """Normalize checkbutton/radiobutton sizing across platforms."""

    system = platform.system()
    padding = (8, 4) if system != "Windows" else (6, 2)

    base_kwargs = {
        "background": background,
        "foreground": foreground,
        "padding": padding,
    }
    if focuscolor:
        base_kwargs["focuscolor"] = focuscolor

    check_kwargs = dict(base_kwargs)
    radio_kwargs = dict(base_kwargs)

    if system == "Linux":
        # Larger indicator improves visibility on GTK-based themes
        check_kwargs["indicatorsize"] = 18
        check_kwargs["indicatormargin"] = 4
        radio_kwargs["indicatorsize"] = 18
        radio_kwargs["indicatormargin"] = 4

    style.configure("TCheckbutton", **check_kwargs)
    style.configure("TRadiobutton", **radio_kwargs)


def _configure_treeview_metrics(style: "tkinter.ttk.Style") -> None:
    """Normalize Treeview row/heading spacing so text never overlaps on HiDPI."""

    row_height, heading_padding = compute_treeview_dimensions()

    try:
        style.configure("Treeview", rowheight=row_height)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.debug("Unable to set Treeview rowheight: %s", exc)

    try:
        style.configure("Treeview", padding=(4, 2))
    except Exception:
        pass

    try:
        style.configure("Treeview.Heading", padding=(8, heading_padding, 8, heading_padding))
    except Exception:
        pass


def apply_theme(panel: "SettingsPanel", theme: str) -> None:
    """Apply the selected theme to the application.
    
    Args:
        panel: The settings panel instance
        theme: The theme name to apply
    """
    logger.info(f"[THEME] Applying theme: {theme}")
    start_time = time.perf_counter()

    try:
        import tkinter.ttk as ttk_style
        import platform
        
        style = ttk_style.Style()
        colors = THEME_COLORS.get(theme, THEME_COLORS["Light Mode"])
        combobox_style_name = _theme_specific_combobox_style(theme)
        combobox_styles = ("TCombobox", THEME_COMBOBOX_STYLE, combobox_style_name)
        
        logger.info(f"[THEME] Using color scheme: bg={colors.get('bg_dark', 'N/A')}, fg={colors.get('fg_light', 'N/A')}")

        if theme == "Dark Mode":
            style.theme_use(colors["theme_base"])
            
            bg_dark = colors["bg_dark"]
            fg_light = colors["fg_light"]
            select_bg = colors["select_bg"]
            button_bg = colors["button_bg"]
            entry_bg = colors["entry_bg"]
            
            # Configure styles
            style.configure(".", background=bg_dark, foreground=fg_light, fieldbackground=entry_bg)
            style.configure("TFrame", background=bg_dark)
            style.configure("TLabel", background=bg_dark, foreground=fg_light)
            _configure_button_style(
                style,
                background=button_bg,
                foreground=fg_light,
                bordercolor=button_bg,
                focuscolor=select_bg,
            )
            style.map("TButton", background=[("active", select_bg), ("pressed", "#505050")])
            style.configure("TEntry", fieldbackground=entry_bg, foreground=fg_light)
            _configure_combobox_style(
                style,
                field_bg=entry_bg,
                fg=fg_light,
                button_bg=button_bg,
                select_bg=select_bg,
                select_fg=fg_light,
                arrowcolor=fg_light,
                insertcolor=fg_light,
                style_names=combobox_styles,
            )
            _map_combobox_states(
                style,
                style_names=combobox_styles,
                fieldbackground=[("readonly", entry_bg), ("disabled", bg_dark)],
                foreground=[("readonly", fg_light), ("disabled", "#888888")],
                background=[("readonly", button_bg), ("disabled", bg_dark)],
                selectbackground=[("readonly", select_bg)],
                selectforeground=[("readonly", fg_light)],
            )
            _configure_toggle_styles(
                style,
                background=bg_dark,
                foreground=fg_light,
                focuscolor=select_bg,
            )
            style.configure("TLabelframe", background=bg_dark, foreground=fg_light)
            style.configure("TLabelframe.Label", background=bg_dark, foreground=fg_light)
            style.configure("TNotebook", background=bg_dark)
            style.configure("TNotebook.Tab", background=button_bg, foreground=fg_light)
            style.map("TNotebook.Tab", 
                     background=[("selected", select_bg)],
                     foreground=[("selected", fg_light), ("!selected", fg_light)])
            style.configure("TScrollbar", background=button_bg, troughcolor=bg_dark)
            style.configure("Treeview", background=entry_bg, foreground=fg_light, fieldbackground=entry_bg)
            style.configure("Treeview.Heading", background=button_bg, foreground=fg_light)
            
            # Configure dropdown listbox colors
            try:
                panel.parent.master.option_add("*TCombobox*Listbox*Background", entry_bg)
                panel.parent.master.option_add("*TCombobox*Listbox*Foreground", fg_light)
                panel.parent.master.option_add("*TCombobox*Listbox*selectBackground", select_bg)
                panel.parent.master.option_add("*TCombobox*Listbox*selectForeground", fg_light)
                # Schedule update asynchronously to avoid blocking
                panel.parent.master.after_idle(lambda: panel.parent.master.update_idletasks())
            except Exception:
                pass
            
            # Update Text widgets
            _schedule_widget_recolor(panel, bg=bg_dark, fg=fg_light, insertbackground=fg_light)
                
        elif theme == "Matrix Mode":
            style.theme_use(colors["theme_base"])
            
            bg_black = colors["bg_black"]
            fg_green = colors["fg_green"]
            fg_dim_green = colors["fg_dim_green"]
            select_bg = colors["select_bg"]
            button_bg = colors["button_bg"]
            entry_bg = colors["entry_bg"]
            accent_green = colors["accent_green"]
            
            style.configure(".", background=bg_black, foreground=fg_green, fieldbackground=entry_bg)
            style.configure("TFrame", background=bg_black)
            style.configure("TLabel", background=bg_black, foreground=fg_green)
            _configure_button_style(
                style,
                background=button_bg,
                foreground=fg_green,
                bordercolor=button_bg,
                focuscolor=accent_green,
                highlightcolor=accent_green,
            )
            style.map("TButton", background=[("active", select_bg), ("pressed", "#004400")], foreground=[("active", accent_green)])
            style.configure("TEntry", fieldbackground=entry_bg, foreground=fg_green, insertbackground=accent_green)
            _configure_combobox_style(
                style,
                field_bg=entry_bg,
                fg=fg_green,
                button_bg=button_bg,
                select_bg=select_bg,
                select_fg=accent_green,
                arrowcolor=fg_green,
                insertcolor=accent_green,
                style_names=combobox_styles,
            )
            _map_combobox_states(
                style,
                style_names=combobox_styles,
                fieldbackground=[("readonly", entry_bg), ("disabled", bg_black)],
                foreground=[("readonly", fg_green), ("disabled", fg_dim_green)],
                background=[("readonly", button_bg), ("disabled", bg_black)],
                selectbackground=[("readonly", select_bg)],
                selectforeground=[("readonly", accent_green)],
            )
            _configure_toggle_styles(
                style,
                background=bg_black,
                foreground=fg_green,
                focuscolor=accent_green,
            )
            style.configure("TLabelframe", background=bg_black, foreground=fg_green)
            style.configure("TLabelframe.Label", background=bg_black, foreground=accent_green)
            style.configure("TNotebook", background=bg_black, bordercolor=bg_black)
            style.configure("TNotebook.Tab", 
                background=button_bg, 
                foreground=fg_green,
                lightcolor=button_bg,
                bordercolor=bg_black)
            style.map("TNotebook.Tab", 
                background=[("selected", select_bg), ("active", "#002200")],
                foreground=[("selected", accent_green), ("active", accent_green)],
                lightcolor=[("selected", select_bg)],
                bordercolor=[("selected", bg_black)])
            style.configure("TScrollbar", background=button_bg, troughcolor=bg_black)
            style.configure("Treeview", background=entry_bg, foreground=fg_green, fieldbackground=entry_bg)
            style.configure("Treeview.Heading", background=button_bg, foreground=accent_green)
            
            try:
                panel.parent.master.option_add("*TCombobox*Listbox*Background", entry_bg)
                panel.parent.master.option_add("*TCombobox*Listbox*Foreground", fg_green)
                panel.parent.master.option_add("*TCombobox*Listbox*selectBackground", select_bg)
                panel.parent.master.option_add("*TCombobox*Listbox*selectForeground", accent_green)
                # Schedule update asynchronously to avoid blocking
                panel.parent.master.after_idle(lambda: panel.parent.master.update_idletasks())
            except Exception:
                pass
            
            _schedule_widget_recolor(panel, bg=bg_black, fg=fg_green, insertbackground=accent_green)
                
        elif theme == "Halloween Mode":
            style.theme_use(colors["theme_base"])
            
            bg_black = colors["bg_black"]
            bg_dark = colors["bg_dark"]
            fg_orange = colors["fg_orange"]
            fg_light = colors["fg_light"]
            accent_orange = colors["accent_orange"]
            button_bg = colors["button_bg"]
            entry_bg = colors["entry_bg"]
            select_bg = colors["select_bg"]
            
            style.configure(".", background=bg_black, foreground=fg_orange, fieldbackground=entry_bg)
            style.configure("TFrame", background=bg_black)
            style.configure("TLabel", background=bg_black, foreground=fg_light)
            _configure_button_style(
                style,
                background=button_bg,
                foreground=fg_orange,
                bordercolor=accent_orange,
                focuscolor=accent_orange,
                highlightcolor=accent_orange,
            )
            style.map("TButton", 
                background=[("active", select_bg), ("pressed", "#3d2000")],
                foreground=[("active", "#ffffff")])
            style.configure("TEntry", fieldbackground=entry_bg, foreground=fg_orange, insertbackground=fg_orange)
            _configure_combobox_style(
                style,
                field_bg=entry_bg,
                fg=fg_orange,
                button_bg=button_bg,
                select_bg=select_bg,
                select_fg=bg_black,
                arrowcolor=accent_orange,
                bordercolor=accent_orange,
                lightcolor=accent_orange,
                darkcolor=accent_orange,
                insertcolor=fg_orange,
                style_names=combobox_styles,
            )
            _map_combobox_states(
                style,
                style_names=combobox_styles,
                fieldbackground=[("readonly", entry_bg), ("disabled", bg_black)],
                foreground=[("readonly", fg_orange), ("disabled", "#8B4513")],
                background=[("readonly", button_bg), ("disabled", bg_black)],
                selectbackground=[("readonly", select_bg)],
                selectforeground=[("readonly", bg_black)],
            )
            _configure_toggle_styles(
                style,
                background=bg_black,
                foreground=fg_light,
                focuscolor=accent_orange,
            )
            style.configure("TLabelframe", background=bg_black, foreground=accent_orange)
            style.configure("TLabelframe.Label", background=bg_black, foreground=accent_orange)
            style.configure("TNotebook", background=bg_black, bordercolor=bg_black)
            style.configure("TNotebook.Tab", 
                background=button_bg, 
                foreground=fg_orange,
                lightcolor=button_bg,
                bordercolor=bg_black)
            style.map("TNotebook.Tab",
                background=[("selected", select_bg), ("active", "#3d2000")],
                foreground=[("selected", "#ffffff"), ("active", accent_orange)],
                lightcolor=[("selected", select_bg)],
                bordercolor=[("selected", bg_black)])
            style.configure("TScrollbar", background=button_bg, troughcolor=bg_black)
            style.configure("Treeview", background=entry_bg, foreground=fg_orange, fieldbackground=entry_bg)
            style.configure("Treeview.Heading", background=button_bg, foreground=accent_orange)
            
            try:
                panel.parent.master.option_add("*TCombobox*Listbox*Background", entry_bg)
                panel.parent.master.option_add("*TCombobox*Listbox*Foreground", fg_orange)
                panel.parent.master.option_add("*TCombobox*Listbox*selectBackground", select_bg)
                panel.parent.master.option_add("*TCombobox*Listbox*selectForeground", "#ffffff")
                # Schedule update asynchronously to avoid blocking
                panel.parent.master.after_idle(lambda: panel.parent.master.update_idletasks())
            except Exception:
                pass
            
            # Update Text widgets
            _schedule_widget_recolor(panel, bg=bg_black, fg=fg_orange, insertbackground=fg_orange)
            
            # Update root and frames recursively
            try:
                root = panel.parent
                while root.master:
                    root = root.master
                root.configure(bg=bg_black)
                
                def update_bg(widget):
                    try:
                        if isinstance(widget, (tk.Frame, tk.Toplevel)):
                            widget.configure(bg=bg_black)
                        for child in widget.winfo_children():
                            update_bg(child)
                    except Exception:
                        pass
                update_bg(root)
            except Exception:
                pass
                
        elif theme == "Barbie Mode":
            style.theme_use(colors["theme_base"])
            
            bg_pink = colors["bg_pink"]
            bg_hot_pink = colors["bg_hot_pink"]
            fg_dark = colors["fg_dark"]
            fg_white = colors["fg_white"]
            select_bg = colors["select_bg"]
            button_bg = colors["button_bg"]
            entry_bg = colors["entry_bg"]
            accent_purple = colors["accent_purple"]
            
            style.configure(".", background=bg_pink, foreground=fg_dark, fieldbackground=entry_bg)
            style.configure("TFrame", background=bg_pink)
            style.configure("TLabel", background=bg_pink, foreground=fg_dark)
            _configure_button_style(
                style,
                background=button_bg,
                foreground=fg_white,
                bordercolor=button_bg,
                focuscolor=select_bg,
                highlightcolor=select_bg,
            )
            style.map("TButton", background=[("active", select_bg)])
            style.configure("TEntry", fieldbackground=entry_bg, foreground=fg_dark, insertbackground=fg_dark)
            _configure_combobox_style(
                style,
                field_bg=entry_bg,
                fg=fg_dark,
                button_bg=button_bg,
                select_bg=select_bg,
                select_fg=fg_white,
                arrowcolor=fg_dark,
                insertcolor=fg_dark,
                style_names=combobox_styles,
            )
            _map_combobox_states(
                style,
                style_names=combobox_styles,
                fieldbackground=[("readonly", entry_bg), ("disabled", bg_pink)],
                foreground=[("readonly", fg_dark), ("disabled", "#C71585")],
                background=[("readonly", button_bg), ("disabled", bg_pink)],
                selectbackground=[("readonly", select_bg)],
                selectforeground=[("readonly", fg_white)],
            )
            _configure_toggle_styles(
                style,
                background=bg_pink,
                foreground=fg_dark,
                focuscolor=select_bg,
            )
            style.configure("TLabelframe", background=bg_pink, foreground=fg_dark)
            style.configure("TLabelframe.Label", background=bg_pink, foreground=select_bg)
            style.configure("TNotebook", background=bg_pink)
            style.configure("TNotebook.Tab", background=button_bg, foreground=fg_white)
            style.map("TNotebook.Tab", 
                background=[("selected", select_bg)],
                foreground=[("selected", fg_white)])
            style.configure("TScrollbar", background=button_bg, troughcolor=bg_pink)
            style.configure("Treeview", background=entry_bg, foreground=fg_dark, fieldbackground=entry_bg)
            style.configure("Treeview.Heading", background=button_bg, foreground=fg_white)
            
            try:
                panel.parent.master.option_add("*TCombobox*Listbox*Background", entry_bg)
                panel.parent.master.option_add("*TCombobox*Listbox*Foreground", fg_dark)
                panel.parent.master.option_add("*TCombobox*Listbox*selectBackground", select_bg)
                panel.parent.master.option_add("*TCombobox*Listbox*selectForeground", fg_white)
                # Schedule update asynchronously to avoid blocking
                panel.parent.master.after_idle(lambda: panel.parent.master.update_idletasks())
            except Exception:
                pass
            
            _schedule_widget_recolor(panel, bg=bg_pink, fg=fg_dark, insertbackground=fg_dark)
                
        else:  # Light Mode
            # Reset to default platform theme
            try:
                if platform.system() == "Darwin":
                    style.theme_use("aqua")
                elif platform.system() == "Windows":
                    style.theme_use("vista")
                else:
                    style.theme_use("clam")
            except Exception:
                style.theme_use("default")

            bg = colors.get("bg", "white")
            fg = colors.get("fg", "black")
            button_bg = colors.get("button_bg", "#f0f0f0")
            entry_bg = colors.get("entry_bg", "white")
            select_bg = colors.get("select_bg", "#0078d7")
            select_fg = colors.get("select_fg", "white")
            border = colors.get("border", "#c8c8c8")
            disabled_bg = colors.get("disabled_bg", "#e0e0e0")
            disabled_fg = colors.get("disabled_fg", "#888888")

            style.configure(".", background=bg, foreground=fg, fieldbackground=entry_bg)
            style.configure("TFrame", background=bg)
            style.configure("TLabel", background=bg, foreground=fg)
            style.configure("TLabelframe", background=bg, foreground=fg)
            style.configure("TLabelframe.Label", background=bg, foreground=fg)
            style.configure("TEntry", fieldbackground=entry_bg, foreground=fg)
            style.configure("TNotebook", background=bg, bordercolor=bg)
            style.configure("TNotebook.Tab", background=button_bg, foreground=fg)
            style.map(
                "TNotebook.Tab",
                background=[("selected", select_bg), ("active", "#e6e6e6")],
                foreground=[("selected", fg)],
            )
            style.configure("TScrollbar", background=button_bg, troughcolor=bg)
            style.configure("Treeview", background=entry_bg, foreground=fg, fieldbackground=entry_bg)
            style.configure("Treeview.Heading", background=button_bg, foreground=fg)
            
            _configure_combobox_style(
                style,
                field_bg=entry_bg,
                fg=fg,
                button_bg=button_bg,
                select_bg=select_bg,
                select_fg=select_fg,
                arrowcolor=fg,
                bordercolor=border,
                lightcolor="#fdfdfd",
                darkcolor="#c0c0c0",
                insertcolor=fg,
                style_names=combobox_styles,
            )
            _configure_button_style(
                style,
                background=button_bg,
                foreground=fg,
                bordercolor=border,
                focuscolor=select_bg,
                highlightcolor=select_bg,
                relief="raised",
            )
            _configure_toggle_styles(
                style,
                background=bg,
                foreground=fg,
                focuscolor=select_bg,
            )
            _map_combobox_states(
                style,
                style_names=combobox_styles,
                fieldbackground=[("readonly", entry_bg), ("disabled", disabled_bg)],
                foreground=[("readonly", fg), ("disabled", disabled_fg)],
                background=[("readonly", button_bg), ("disabled", disabled_bg)],
                selectbackground=[("readonly", select_bg)],
                selectforeground=[("readonly", select_fg)],
            )
            style.map(
                "TButton",
                background=[("active", "#e6e6e6"), ("pressed", "#dcdcdc")],
                foreground=[("disabled", disabled_fg)],
            )
            
            try:
                panel.parent.master.option_add("*TCombobox*Listbox*Background", entry_bg)
                panel.parent.master.option_add("*TCombobox*Listbox*Foreground", fg)
                panel.parent.master.option_add("*TCombobox*Listbox*selectBackground", select_bg)
                panel.parent.master.option_add("*TCombobox*Listbox*selectForeground", select_fg)
                # Schedule update asynchronously to avoid blocking
                panel.parent.master.after_idle(lambda: panel.parent.master.update_idletasks())
            except Exception:
                pass
            
            _schedule_widget_recolor(panel, bg=bg, fg=fg, insertbackground=fg)
        
        _configure_treeview_metrics(style)

        # Force update of notebook tabs immediately
        try:
            root = panel.parent
            while root.master:
                root = root.master

            def _configure_dialogs_async() -> None:
                dialog_start = time.perf_counter()
                configure_global_dialogs(root, theme)
                dialog_elapsed = time.perf_counter() - dialog_start
                if dialog_elapsed >= 0.1:
                    logger.debug(f"[THEME] Global dialog configuration took {dialog_elapsed:.3f}s")

            try:
                root.after_idle(root.update_idletasks)
            except Exception:
                try:
                    root.update_idletasks()
                except Exception:
                    pass

            try:
                root.after_idle(_configure_dialogs_async)
            except Exception:
                _configure_dialogs_async()
        except Exception:
            pass
        
        _apply_combobox_styles_to_widgets(panel, combobox_style_name)

        # Update theme info label
        panel.theme_info.config(text=colors["info_message"], foreground=colors["info_color"])
        
        # Notify chat panel of theme change
        if panel._chat_panel and hasattr(panel._chat_panel, 'update_theme'):
            try:
                notify_start = time.perf_counter()
                panel._chat_panel.update_theme(theme)
                notify_elapsed = time.perf_counter() - notify_start
                if notify_elapsed >= 0.1:
                    logger.debug(f"[THEME] Chat panel update_theme took {notify_elapsed:.3f}s")
            except Exception:
                pass
        
        # Notify help panel of theme change
        help_panel = getattr(panel, '_help_panel', None)
        if help_panel and hasattr(help_panel, 'update_theme'):
            try:
                notify_start = time.perf_counter()
                help_panel.update_theme(theme)
                notify_elapsed = time.perf_counter() - notify_start
                if notify_elapsed >= 0.1:
                    logger.debug(f"[THEME] Help panel update_theme took {notify_elapsed:.3f}s")
            except Exception:
                pass
        
        # Notify HRM training panel of theme change
        hrm_panel = getattr(panel, '_hrm_training_panel', None)
        if hrm_panel and hasattr(hrm_panel, 'update_theme'):
            try:
                notify_start = time.perf_counter()
                hrm_panel.update_theme()
                notify_elapsed = time.perf_counter() - notify_start
                if notify_elapsed >= 0.1:
                    logger.debug(f"[THEME] HRM panel update_theme took {notify_elapsed:.3f}s")
            except Exception:
                pass

        try:
            panel._last_theme_applied = theme
            panel._last_theme_applied_at = time.perf_counter()
        except Exception:
            pass

    except Exception as e:
        panel.theme_info.config(text=f"⚠ Error applying theme: {e}", foreground="red")
    finally:
        elapsed_total = time.perf_counter() - start_time
        logger.info(f"[THEME] Theme '{theme}' applied in {elapsed_total:.3f}s")


def _apply_widget_colors(widget: Any, bg: str, fg: str, insertbackground: str) -> None:
    """Apply palette colors to a single widget if supported."""
    try:
        if isinstance(widget, tk.Text):
            widget.config(bg=bg, fg=fg, insertbackground=insertbackground)
        elif isinstance(widget, tk.Listbox):
            widget.config(bg=bg, fg=fg)
        elif isinstance(widget, tk.Canvas):
            widget.config(bg=bg)
        elif isinstance(widget, tk.Frame):
            widget.config(bg=bg)
        elif isinstance(widget, tk.Label):
            widget.config(bg=bg, fg=fg)
    except Exception:
        pass


def update_widget_colors(widget: Any, bg: str, fg: str, insertbackground: str) -> None:
    """Recursively update widget colors for Text, Listbox, etc."""
    try:
        _apply_widget_colors(widget, bg, fg, insertbackground)
        for child in widget.winfo_children():
            update_widget_colors(child, bg, fg, insertbackground)
    except Exception:
        pass


def _schedule_widget_recolor(panel: "SettingsPanel", bg: str, fg: str, insertbackground: str) -> None:
    """Iteratively recolor widgets in chunks to avoid UI stalls."""
    try:
        root = panel.parent.master
    except Exception:
        root = None
    if root is None:
        return

    try:
        queue = deque([root])
    except Exception:
        return

    start = time.perf_counter()
    batch_size = 40

    def _process() -> None:
        processed = 0
        while queue and processed < batch_size:
            widget = queue.popleft()
            _apply_widget_colors(widget, bg, fg, insertbackground)
            try:
                for child in widget.winfo_children():
                    queue.append(child)
            except Exception:
                pass
            processed += 1

        if queue:
            try:
                root.after(1, _process)
            except Exception:
                pass
        else:
            elapsed = time.perf_counter() - start
            if elapsed >= 0.2:
                logger.debug(f"[THEME] Widget recolor traversal completed in {elapsed:.3f}s")

    try:
        root.after_idle(_process)
    except Exception:
        _process()
