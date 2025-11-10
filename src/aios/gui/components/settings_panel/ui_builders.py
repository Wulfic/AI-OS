"""UI builder functions for settings panel."""

from __future__ import annotations
import os
import logging
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from ..tooltips import add_tooltip
from .theme_constants import THEME_NAMES

if TYPE_CHECKING:
    from .panel_main import SettingsPanel

logger = logging.getLogger(__name__)


def create_title(container: ttk.Frame) -> None:
    """Create the title section.
    
    Args:
        container: The main container frame
    """
    title = ttk.Label(container, text="⚙️ Application Settings", font=("TkDefaultFont", 12, "bold"))
    title.pack(anchor="w", pady=(0, 10))


def create_appearance_section(panel: "SettingsPanel", container: ttk.Frame) -> None:
    """Create the appearance section with theme selection.
    
    Args:
        panel: The settings panel instance
        container: The main container frame
    """
    appearance_frame = ttk.LabelFrame(container, text="Appearance", padding=8)
    appearance_frame.pack(fill="both", expand=True, pady=(0, 8))

    # Theme selection
    theme_row = ttk.Frame(appearance_frame)
    theme_row.pack(fill="x", pady=5)

    theme_label = ttk.Label(theme_row, text="Theme:", width=15, anchor="e")
    theme_label.pack(side="left", padx=(0, 10))

    panel.theme_var = tk.StringVar(value="Dark Mode")
    theme_combo = ttk.Combobox(
        theme_row,
        textvariable=panel.theme_var,
        values=THEME_NAMES,
        state="readonly",
        width=20
    )
    theme_combo.pack(side="left")

    add_tooltip(
        theme_combo,
        "Select the application color theme.\n\n"
        "• Light Mode: Traditional bright theme with light backgrounds\n"
        "  Best for well-lit environments, easier on battery\n\n"
        "• Dark Mode: Dark theme with reduced brightness\n"
        "  Reduces eye strain in low-light conditions, saves power on OLED screens\n\n"
        "• Matrix Mode: Green-on-black hacker aesthetic\n"
        "  Terminal-style theme inspired by The Matrix, perfect for late-night coding\n\n"
        "• Barbie Mode: Hot pink and vibrant theme\n"
        "  Fun, colorful theme with pink and purple accents for a playful experience\n\n"
        "• Halloween Mode: Spooky orange and black theme\n"
        "  Festive theme with pumpkin orange and midnight black for the Halloween season\n\n"
        "Note: Theme changes will take effect after restarting the application."
    )

    # Theme change callback
    def _on_theme_change(*args):
        # Don't trigger during state restoration to avoid race conditions
        if hasattr(panel, '_restoring_state') and panel._restoring_state:
            return
        
        theme = panel.theme_var.get()
        panel._apply_theme(theme)
        if panel._save_state_fn:
            panel._save_state_fn()

    panel.theme_var.trace_add("write", _on_theme_change)

    # Apply button
    apply_btn = ttk.Button(theme_row, text="Apply Now", command=lambda: panel._apply_theme(panel.theme_var.get()))
    apply_btn.pack(side="left", padx=(10, 0))

    add_tooltip(
        apply_btn,
        "Apply the selected theme immediately to the current session.\n"
        "Note: Some elements may require restarting the application for full effect."
    )

    # Theme info label
    panel.theme_info = ttk.Label(appearance_frame, text="", foreground="blue", font=("TkDefaultFont", 8, "italic"))
    panel.theme_info.pack(anchor="w", pady=(5, 0))


def create_general_settings_section(panel: "SettingsPanel", container: ttk.Frame) -> None:
    """Create the general settings section.
    
    Args:
        panel: The settings panel instance
        container: The main container frame
    """
    general_frame = ttk.LabelFrame(container, text="General Settings", padding=8)
    general_frame.pack(fill="x", pady=(0, 8))

    # Start at boot checkbox (Windows only)
    panel.startup_var = tk.BooleanVar(value=False)
    startup_check = ttk.Checkbutton(
        general_frame,
        text="Start AI-OS at Windows boot",
        variable=panel.startup_var,
        command=panel._on_startup_changed
    )
    startup_check.pack(anchor="w", pady=5)

    add_tooltip(
        startup_check,
        "Automatically start AI-OS when Windows starts up.\n\n"
        "When enabled, AI-OS will launch automatically when you log in to Windows.\n"
        "This setting modifies the Windows registry:\n"
        "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\n\n"
        "Note: Only works on Windows. On other platforms, this option has no effect."
    )

    # Startup status label
    panel.startup_info = ttk.Label(general_frame, text="", font=("TkDefaultFont", 8, "italic"))
    panel.startup_info.pack(anchor="w", pady=(0, 5))

    # Start minimized checkbox
    panel.start_minimized_var = tk.BooleanVar(value=False)
    start_minimized_check = ttk.Checkbutton(
        general_frame,
        text="Start minimized to system tray",
        variable=panel.start_minimized_var,
        command=panel._on_start_minimized_changed
    )
    start_minimized_check.pack(anchor="w", pady=5)

    add_tooltip(
        start_minimized_check,
        "When enabled, AI-OS will start minimized to the system tray.\n\n"
        "The application will launch silently in the background,\n"
        "showing only the tray icon. Double-click the tray icon\n"
        "to show the main window.\n\n"
        "Works best when combined with 'Start at Windows boot'."
    )

    # Minimize to tray on close checkbox
    panel.minimize_to_tray_var = tk.BooleanVar(value=False)
    minimize_to_tray_check = ttk.Checkbutton(
        general_frame,
        text="Minimize to tray instead of closing",
        variable=panel.minimize_to_tray_var,
        command=panel._on_minimize_to_tray_changed
    )
    minimize_to_tray_check.pack(anchor="w", pady=5)

    add_tooltip(
        minimize_to_tray_check,
        "When enabled, clicking the X button will minimize to tray\n"
        "instead of closing the application.\n\n"
        "The app will remain running in the background.\n"
        "Use the tray menu 'Exit' option to fully quit."
    )


def create_help_section(panel: "SettingsPanel", container: ttk.Frame) -> None:
    """Create the help documentation management section.
    
    Args:
        panel: The settings panel instance
        container: The main container frame
    """
    help_frame = ttk.LabelFrame(container, text="Help Documentation", padding=8)
    help_frame.pack(fill="both", expand=True, pady=(0, 8))

    help_desc_label = ttk.Label(
        help_frame,
        text="Manage the in-app documentation search index."
    )
    help_desc_label.pack(anchor="w", pady=(0, 10))

    # Index status row
    index_status_row = ttk.Frame(help_frame)
    index_status_row.pack(fill="x", pady=(0, 10))

    index_status_label_text = ttk.Label(index_status_row, text="Index Status:", width=20, anchor="e")
    index_status_label_text.pack(side="left", padx=(0, 10))

    panel.help_index_status_label = ttk.Label(index_status_row, text="Ready")
    panel.help_index_status_label.pack(side="left", fill="x", expand=True)

    add_tooltip(
        panel.help_index_status_label,
        "Status of the documentation search index.\n\n"
        "• Ready: Index is built and ready to use\n"
        "• Building: Index is currently being rebuilt\n"
        "• Not Found: Index needs to be built\n\n"
        "The index is automatically rebuilt when the Help panel opens for the first time.\n"
        "Use 'Rebuild Index' if links aren't working or documentation has been updated."
    )

    # Button row
    help_btn_row = ttk.Frame(help_frame)
    help_btn_row.pack(fill="x", pady=(0, 5))

    # Spacer to align with status above
    spacer = ttk.Label(help_btn_row, text="", width=20)
    spacer.pack(side="left", padx=(0, 10))

    # Rebuild index button
    rebuild_index_btn = ttk.Button(
        help_btn_row,
        text="  Rebuild Index",
        command=panel._rebuild_help_index,
        width=15
    )
    rebuild_index_btn.pack(side="left", padx=(0, 5))

    add_tooltip(
        rebuild_index_btn,
        "Rebuild the documentation search index.\n\n"
        "Use this if:\n"
        "• Links in documentation aren't working\n"
        "• Documentation files have been updated\n"
        "• Search results seem incomplete or outdated\n\n"
        "This will delete the existing index and build a new one\n"
        "from all .md and .mdx files in the docs/ folder.\n\n"
        "The rebuild happens in the background and takes a few seconds."
    )

    # Info label
    help_info_label = ttk.Label(
        help_frame,
        text="Note: The index is rebuilt automatically when needed. Use this button\n"
             "if you experience issues with documentation links or search.",
        foreground="gray",
        font=("TkDefaultFont", 8, "italic")
    )
    help_info_label.pack(anchor="w", pady=(10, 0))


def create_cache_section(panel: "SettingsPanel", container: ttk.Frame) -> None:
    """Create the cache management section.
    
    Args:
        panel: The settings panel instance
        container: The main container frame
    """
    cache_frame = ttk.LabelFrame(container, text="Dataset Cache Management", padding=8)
    cache_frame.pack(fill="both", expand=True, pady=(0, 8))

    cache_desc_label = ttk.Label(
        cache_frame,
        text="Streaming datasets are cached to speed up subsequent training runs."
    )
    cache_desc_label.pack(anchor="w", pady=(0, 10))

    # Cache size configuration row
    cache_size_row = ttk.Frame(cache_frame)
    cache_size_row.pack(fill="x", pady=(0, 10))

    cache_size_label = ttk.Label(cache_size_row, text="Max Cache Size (MB):", width=20, anchor="e")
    cache_size_label.pack(side="left", padx=(0, 10))

    panel.cache_size_var = tk.StringVar(value="100")
    cache_size_entry = ttk.Entry(
        cache_size_row, 
        textvariable=panel.cache_size_var, 
        width=10,
        font=("TkDefaultFont", 10)
    )
    cache_size_entry.pack(side="left", padx=(0, 10), ipady=3)

    save_size_btn = ttk.Button(
        cache_size_row,
        text="Save",
        command=panel._save_cache_size,
        width=8
    )
    save_size_btn.pack(side="left")

    add_tooltip(
        cache_size_entry,
        "Maximum total size for cached dataset blocks.\n\n"
        "Default: 100 MB\n"
        "When this limit is reached, oldest blocks are automatically removed.\n\n"
        "Set higher for faster training iterations (more cached data)\n"
        "or lower to save disk space."
    )

    add_tooltip(
        save_size_btn,
        "Save the cache size limit to config/default.yaml.\n"
        "Changes take effect on next training run."
    )

    # Cache stats row
    cache_stats_row = ttk.Frame(cache_frame)
    cache_stats_row.pack(fill="x", pady=(0, 10))

    cache_stats_label_text = ttk.Label(cache_stats_row, text="Current Usage:", width=20, anchor="e")
    cache_stats_label_text.pack(side="left", padx=(0, 10))

    panel.cache_stats_label = ttk.Label(cache_stats_row, text="Loading...")
    panel.cache_stats_label.pack(side="left", fill="x", expand=True)

    add_tooltip(
        panel.cache_stats_label,
        "Current cache usage statistics.\n\n"
        "Shows: size used vs. limit | number of blocks | number of datasets cached.\n"
        "Click Refresh to update these numbers."
    )

    # Button row
    cache_btn_row = ttk.Frame(cache_frame)
    cache_btn_row.pack(fill="x", pady=(0, 5))

    # Spacer to align with input above
    spacer = ttk.Label(cache_btn_row, text="", width=20)
    spacer.pack(side="left", padx=(0, 10))

    # Refresh stats button
    refresh_cache_btn = ttk.Button(
        cache_btn_row,
        text="  Refresh",
        command=panel._refresh_cache_stats,
        width=12
    )
    refresh_cache_btn.pack(side="left", padx=(0, 5))

    add_tooltip(
        refresh_cache_btn,
        "Refresh cache statistics to see current usage."
    )

    # Clear cache button
    clear_cache_btn = ttk.Button(
        cache_btn_row,
        text="  Clear Cache",
        command=panel._clear_cache,
        width=15
    )
    clear_cache_btn.pack(side="left", padx=(0, 0))

    add_tooltip(
        clear_cache_btn,
        "Clear all cached dataset blocks.\n\n"
        "This will free up disk space but subsequent training runs\n"
        "will need to re-download data from HuggingFace.\n\n"
        "Cache will be automatically rebuilt as needed."
    )


def create_support_section(panel: "SettingsPanel", container: ttk.Frame) -> None:
    """Create the support section with Ko-Fi button.
    
    Args:
        panel: The settings panel instance
        container: The main container frame
    """
    support_frame = ttk.LabelFrame(container, text="Support the Project", padding=8)
    support_frame.pack(fill="x", pady=(0, 8))

    support_label = ttk.Label(
        support_frame,
        text="If you find AI-OS useful, consider supporting its development!",
        foreground="gray"
    )
    support_label.pack(anchor="w", pady=(0, 10))

    # Ko-Fi button
    try:
        from PIL import Image, ImageTk
        
        # Get the project root directory
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))))
        kofi_image_path = os.path.join(project_root, "kofi.png")
        
        if os.path.exists(kofi_image_path):
            kofi_img = Image.open(kofi_image_path)
            target_height = 35
            original_width, original_height = kofi_img.size
            aspect_ratio = original_width / original_height
            target_width = int(target_height * aspect_ratio)
            kofi_img = kofi_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            kofi_photo = ImageTk.PhotoImage(kofi_img)
            
            kofi_btn = tk.Button(
                support_frame,
                image=kofi_photo,
                command=panel._open_kofi_link,
                cursor="hand2",
                borderwidth=0,
                highlightthickness=0
            )
            kofi_btn.image = kofi_photo  # type: ignore[attr-defined]
            kofi_btn.pack(anchor="w")
            
            add_tooltip(
                kofi_btn,
                "Support AI-OS development on Ko-fi!\n\n"
                "Your support helps keep this project alive and growing.\n"
                "Click to visit: https://ko-fi.com/wulfic"
            )
        else:
            _create_text_kofi_button(support_frame, panel._open_kofi_link)  # type: ignore[arg-type]
    except ImportError:
        _create_text_kofi_button(support_frame, panel._open_kofi_link)  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"Error loading Ko-fi button: {e}")
        _create_text_kofi_button(support_frame, panel._open_kofi_link)  # type: ignore[arg-type]


def _create_text_kofi_button(parent: ttk.Frame, command: callable) -> None:  # type: ignore[valid-type]
    """Create a text-based Ko-fi button fallback.
    
    Args:
        parent: The parent frame
        command: The callback function
    """
    kofi_text_btn = ttk.Button(
        parent,
        text="☕ Support on Ko-fi",
        command=command
    )
    kofi_text_btn.pack(anchor="w")
    
    add_tooltip(
        kofi_text_btn,
        "Support AI-OS development on Ko-fi!\n\n"
        "Your support helps keep this project alive and growing.\n"
        "Click to visit: https://ko-fi.com/wulfic"
    )
