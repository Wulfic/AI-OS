"""Theme management and application logic."""

from __future__ import annotations
import tkinter as tk
from typing import TYPE_CHECKING, Any

from .theme_constants import THEME_COLORS
from aios.gui.utils.theme_utils import configure_global_dialogs

if TYPE_CHECKING:
    from .panel_main import SettingsPanel


def apply_theme(panel: "SettingsPanel", theme: str) -> None:
    """Apply the selected theme to the application.
    
    Args:
        panel: The settings panel instance
        theme: The theme name to apply
    """
    try:
        import tkinter.ttk as ttk_style
        import platform
        
        style = ttk_style.Style()
        colors = THEME_COLORS.get(theme, THEME_COLORS["Light Mode"])

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
            style.configure("TButton", background=button_bg, foreground=fg_light)
            style.map("TButton", background=[("active", select_bg), ("pressed", "#505050")])
            style.configure("TEntry", fieldbackground=entry_bg, foreground=fg_light)
            style.configure("TCombobox", 
                fieldbackground=entry_bg, 
                foreground=fg_light, 
                background=button_bg,
                selectbackground=select_bg,
                selectforeground=fg_light,
                arrowcolor=fg_light)
            style.map("TCombobox",
                fieldbackground=[("readonly", entry_bg), ("disabled", bg_dark)],
                foreground=[("readonly", fg_light), ("disabled", "#888888")],
                background=[("readonly", button_bg), ("disabled", bg_dark)],
                selectbackground=[("readonly", select_bg)],
                selectforeground=[("readonly", fg_light)])
            style.configure("TCheckbutton", background=bg_dark, foreground=fg_light)
            style.configure("TRadiobutton", background=bg_dark, foreground=fg_light)
            style.configure("TLabelframe", background=bg_dark, foreground=fg_light)
            style.configure("TLabelframe.Label", background=bg_dark, foreground=fg_light)
            style.configure("TNotebook", background=bg_dark)
            style.configure("TNotebook.Tab", background=button_bg, foreground=fg_light)
            style.map("TNotebook.Tab", background=[("selected", select_bg)])
            style.configure("TScrollbar", background=button_bg, troughcolor=bg_dark)
            style.configure("Treeview", background=entry_bg, foreground=fg_light, fieldbackground=entry_bg)
            style.configure("Treeview.Heading", background=button_bg, foreground=fg_light)
            
            # Configure dropdown listbox colors
            try:
                panel.parent.master.option_add("*TCombobox*Listbox*Background", entry_bg)
                panel.parent.master.option_add("*TCombobox*Listbox*Foreground", fg_light)
                panel.parent.master.option_add("*TCombobox*Listbox*selectBackground", select_bg)
                panel.parent.master.option_add("*TCombobox*Listbox*selectForeground", fg_light)
                # Also update any existing comboboxes
                panel.parent.master.update_idletasks()
            except Exception:
                pass
            
            # Update Text widgets
            try:
                for widget in panel.parent.master.winfo_children():
                    update_widget_colors(widget, bg=bg_dark, fg=fg_light, insertbackground=fg_light)
            except Exception:
                pass
                
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
            style.configure("TButton", background=button_bg, foreground=fg_green)
            style.map("TButton", background=[("active", select_bg), ("pressed", "#004400")], foreground=[("active", accent_green)])
            style.configure("TEntry", fieldbackground=entry_bg, foreground=fg_green, insertbackground=accent_green)
            style.configure("TCombobox", 
                fieldbackground=entry_bg, 
                foreground=fg_green, 
                background=button_bg,
                selectbackground=select_bg,
                selectforeground=accent_green,
                arrowcolor=fg_green)
            style.map("TCombobox",
                fieldbackground=[("readonly", entry_bg), ("disabled", bg_black)],
                foreground=[("readonly", fg_green), ("disabled", fg_dim_green)],
                background=[("readonly", button_bg), ("disabled", bg_black)],
                selectbackground=[("readonly", select_bg)],
                selectforeground=[("readonly", accent_green)])
            style.configure("TCheckbutton", background=bg_black, foreground=fg_green)
            style.configure("TRadiobutton", background=bg_black, foreground=fg_green)
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
                # Also update any existing comboboxes
                panel.parent.master.update_idletasks()
            except Exception:
                pass
            
            try:
                for widget in panel.parent.master.winfo_children():
                    update_widget_colors(widget, bg=bg_black, fg=fg_green, insertbackground=accent_green)
            except Exception:
                pass
                
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
            style.configure("TButton", background=button_bg, foreground=fg_orange)
            style.map("TButton", 
                background=[("active", select_bg), ("pressed", "#3d2000")],
                foreground=[("active", "#ffffff")])
            style.configure("TEntry", fieldbackground=entry_bg, foreground=fg_orange, insertbackground=fg_orange)
            style.configure("TCombobox", 
                fieldbackground=entry_bg,
                background=button_bg,
                foreground=fg_orange,
                arrowcolor=accent_orange,
                bordercolor=accent_orange,
                lightcolor=accent_orange,
                darkcolor=accent_orange,
                insertcolor=fg_orange,
                selectbackground=select_bg,
                selectforeground=bg_black)
            style.map("TCombobox",
                fieldbackground=[("readonly", entry_bg), ("disabled", bg_black)],
                foreground=[("readonly", fg_orange), ("disabled", "#8B4513")],
                background=[("readonly", button_bg), ("disabled", bg_black)],
                selectbackground=[("readonly", select_bg)],
                selectforeground=[("readonly", bg_black)])
            style.configure("TCheckbutton", background=bg_black, foreground=fg_light)
            style.configure("TRadiobutton", background=bg_black, foreground=fg_light)
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
                # Also update any existing comboboxes
                panel.parent.master.update_idletasks()
            except Exception:
                pass
            
            # Update Text widgets
            try:
                for widget in panel.parent.master.winfo_children():
                    update_widget_colors(widget, bg=bg_black, fg=fg_orange, insertbackground=fg_orange)
            except Exception:
                pass
            
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
            style.configure("TButton", background=button_bg, foreground=fg_white)
            style.map("TButton", background=[("active", select_bg)])
            style.configure("TEntry", fieldbackground=entry_bg, foreground=fg_dark, insertbackground=fg_dark)
            style.configure("TCombobox", 
                fieldbackground=entry_bg, 
                foreground=fg_dark, 
                background=button_bg,
                selectbackground=select_bg,
                selectforeground=fg_white,
                arrowcolor=fg_dark)
            style.map("TCombobox",
                fieldbackground=[("readonly", entry_bg), ("disabled", bg_pink)],
                foreground=[("readonly", fg_dark), ("disabled", "#C71585")],
                background=[("readonly", button_bg), ("disabled", bg_pink)],
                selectbackground=[("readonly", select_bg)],
                selectforeground=[("readonly", fg_white)])
            style.configure("TCheckbutton", background=bg_pink, foreground=fg_dark)
            style.configure("TRadiobutton", background=bg_pink, foreground=fg_dark)
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
                # Also update any existing comboboxes
                panel.parent.master.update_idletasks()
            except Exception:
                pass
            
            try:
                for widget in panel.parent.master.winfo_children():
                    update_widget_colors(widget, bg=bg_pink, fg=fg_dark, insertbackground=fg_dark)
            except Exception:
                pass
                
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
            
            style.map("TButton", background=[("active", "#e1e1e1"), ("pressed", "#d0d0d0")])
            
            try:
                panel.parent.master.option_add("*TCombobox*Listbox*Background", "white")
                panel.parent.master.option_add("*TCombobox*Listbox*Foreground", "black")
                panel.parent.master.option_add("*TCombobox*Listbox*selectBackground", "#0078d7")
                panel.parent.master.option_add("*TCombobox*Listbox*selectForeground", "white")
                # Also update any existing comboboxes
                panel.parent.master.update_idletasks()
            except Exception:
                pass
            
            try:
                for widget in panel.parent.master.winfo_children():
                    update_widget_colors(widget, bg="white", fg="black", insertbackground="black")
            except Exception:
                pass
        
        # Force update of notebook tabs immediately
        try:
            root = panel.parent
            while root.master:
                root = root.master
            root.update_idletasks()
            
            # Configure global dialogs for this theme
            configure_global_dialogs(root, theme)
        except Exception:
            pass
        
        # Update theme info label
        panel.theme_info.config(text=colors["info_message"], foreground=colors["info_color"])
        
        # Notify chat panel of theme change
        if panel._chat_panel and hasattr(panel._chat_panel, 'update_theme'):
            try:
                panel._chat_panel.update_theme(theme)
            except Exception:
                pass
        
        # Notify help panel of theme change
        help_panel = getattr(panel, '_help_panel', None)
        if help_panel and hasattr(help_panel, 'update_theme'):
            try:
                help_panel.update_theme(theme)
            except Exception:
                pass
        
        # Notify HRM training panel of theme change
        hrm_panel = getattr(panel, '_hrm_training_panel', None)
        if hrm_panel and hasattr(hrm_panel, 'update_theme'):
            try:
                hrm_panel.update_theme()
            except Exception:
                pass

    except Exception as e:
        panel.theme_info.config(text=f"⚠ Error applying theme: {e}", foreground="red")


def update_widget_colors(widget: Any, bg: str, fg: str, insertbackground: str) -> None:
    """Recursively update widget colors for Text, Listbox, etc.
    
    Args:
        widget: The widget to update
        bg: Background color
        fg: Foreground color
        insertbackground: Insert cursor color
    """
    try:
        # Update Text widgets
        if isinstance(widget, tk.Text):
            widget.config(bg=bg, fg=fg, insertbackground=insertbackground)
        # Update Listbox widgets
        elif isinstance(widget, tk.Listbox):
            widget.config(bg=bg, fg=fg)
        # Update Canvas widgets
        elif isinstance(widget, tk.Canvas):
            widget.config(bg=bg)
        
        # Recursively update children
        for child in widget.winfo_children():
            update_widget_colors(child, bg, fg, insertbackground)
    except Exception:
        pass
