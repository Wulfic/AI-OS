"""UI builder functions for settings panel."""

from __future__ import annotations
import os
import logging
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

# Import safe variable wrappers
from ...utils import safe_variables

from ..tooltips import add_tooltip
from .theme_constants import THEME_NAMES
from .theme_manager import THEME_COMBOBOX_STYLE

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

    panel.theme_var = safe_variables.StringVar(value="Dark Mode")
    theme_combo = ttk.Combobox(
        theme_row,
        textvariable=panel.theme_var,
        values=THEME_NAMES,
        state="readonly",
        width=20
    )
    theme_combo.configure(style=THEME_COMBOBOX_STYLE)
    theme_combo.pack(side="left")
    panel.theme_combo = theme_combo

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

    # Theme info label (using gray for better readability across all themes)
    panel.theme_info = ttk.Label(appearance_frame, text="", foreground="gray", font=("TkDefaultFont", 8, "italic"))
    panel.theme_info.pack(anchor="w", pady=(5, 0))


def create_general_settings_section(panel: "SettingsPanel", container: ttk.Frame) -> None:
    """Create the general settings section.
    
    Args:
        panel: The settings panel instance
        container: The main container frame
    """
    general_frame = ttk.LabelFrame(container, text="General Settings", padding=8)
    general_frame.pack(fill="x", pady=(0, 8))

    # Start at login checkbox (platform aware)
    from ...utils.startup import is_windows, is_linux

    if is_windows():
        startup_label = "Start AI-OS at Windows login"
        startup_tooltip = (
            "Automatically start AI-OS after Windows login.\n\n"
            "When enabled, AI-OS launches automatically when you sign in.\n"
            "This setting writes to the user-specific registry key:\n"
            "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\n\n"
            "Disable this option if you prefer to launch the application manually."
        )
    elif is_linux():
        startup_label = "Start AI-OS when you log in"
        startup_tooltip = (
            "Automatically start AI-OS after your Linux desktop session begins.\n\n"
            "When enabled, AI-OS creates ~/.config/autostart/ai-os.desktop\n"
            "following the freedesktop.org autostart specification (tested on Ubuntu).\n\n"
            "Remove that file or disable this option to stop launching on login."
        )
    else:
        startup_label = "Start AI-OS when you log in"
        startup_tooltip = (
            "Autostart is currently supported on Windows and Ubuntu-based desktops.\n"
            "On this platform, the option remains disabled."
        )

    panel.startup_var = safe_variables.BooleanVar(value=False)
    startup_check = ttk.Checkbutton(
        general_frame,
        text=startup_label,
        variable=panel.startup_var,
        command=panel._on_startup_changed
    )
    startup_check.pack(anchor="w", pady=5)
    panel.startup_check = startup_check

    add_tooltip(
        startup_check,
        startup_tooltip
    )

    # Startup status label
    panel.startup_info = ttk.Label(general_frame, text="", font=("TkDefaultFont", 8, "italic"))
    panel.startup_info.pack(anchor="w", pady=(0, 5))

    # Start minimized checkbox
    panel.start_minimized_var = safe_variables.BooleanVar(value=False)
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
        "Works best when combined with the autostart option above."
    )

    # Minimize to tray on close checkbox
    panel.minimize_to_tray_var = safe_variables.BooleanVar(value=False)
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


def create_logging_section(panel: "SettingsPanel", container: ttk.Frame) -> None:
    """Create the logging configuration section.
    
    Args:
        panel: The settings panel instance
        container: The main container frame
    """
    logging_frame = ttk.LabelFrame(container, text="Logging Configuration", padding=8)
    logging_frame.pack(fill="x", pady=(0, 8))

    # Logging level description
    log_desc_label = ttk.Label(
        logging_frame,
        text="Configure how much detail to show in the Debug panel.\nChanges apply immediately to the current session."
    )
    log_desc_label.pack(anchor="w", pady=(0, 10))

    # Logging level selection row
    log_level_row = ttk.Frame(logging_frame)
    log_level_row.pack(fill="x", pady=5)

    log_level_label = ttk.Label(log_level_row, text="Logging Level:", width=15, anchor="e")
    log_level_label.pack(side="left", padx=(0, 10))

    panel.log_level_var = safe_variables.StringVar(value="Normal")
    log_levels = ["DEBUG", "Advanced", "Normal"]
    log_level_combo = ttk.Combobox(
        log_level_row,
        textvariable=panel.log_level_var,
        values=log_levels,
        state="readonly",
        width=15
    )
    log_level_combo.configure(style=THEME_COMBOBOX_STYLE)
    log_level_combo.pack(side="left")
    panel.log_level_combo = log_level_combo

    add_tooltip(
        log_level_combo,
        "Select the logging detail level:\n\n"
        "• Normal: Highlights essential system, training, chat, and dataset updates\n"
        "  Shows INFO, WARNING, and ERROR logs from core features while hiding low-level noise.\n\n"
        "• Advanced: Expands Normal mode to include INFO/WARNING logs from every category\n"
        "  Useful when troubleshooting panels beyond the core workflows.\n\n"
        "• DEBUG: Shows absolutely everything\n"
        "  For developers and deep debugging - includes all internal\n"
        "  operations, API calls, and low-level system details.\n\n"
        "Note: Changes apply immediately to already-running operations.\n"
        "The setting is saved and persists across application restarts."
    )

    # Logging level change callback
    def _on_log_level_change(*args):
        # Don't trigger during state restoration to avoid race conditions
        if hasattr(panel, '_restoring_state') and panel._restoring_state:
            return
        
        # Apply the logging level immediately to debug panel
        level = panel.log_level_var.get()
        if hasattr(panel, '_apply_log_level'):
            panel._apply_log_level(level)
        
        if panel._save_state_fn:
            panel._save_state_fn()

    panel.log_level_var.trace_add("write", _on_log_level_change)

    # Info label explaining each level (using gray for better readability across all themes)
    level_info_text = (
        "Normal: Essentials (INFO+)  |  Advanced: All INFO/WARN  |  DEBUG: Everything"
    )
    level_info_label = ttk.Label(
        logging_frame,
        text=level_info_text,
        foreground="gray",
        font=("TkDefaultFont", 8, "italic")
    )
    level_info_label.pack(anchor="w", pady=(5, 0))

    # Log management row: Clear logs button + folder size
    log_mgmt_row = ttk.Frame(logging_frame)
    log_mgmt_row.pack(fill="x", pady=(10, 0))

    clear_logs_btn = ttk.Button(
        log_mgmt_row,
        text="Clear Old Logs",
        command=lambda: panel._clear_old_logs() if hasattr(panel, '_clear_old_logs') else None
    )
    clear_logs_btn.pack(side="left")

    add_tooltip(
        clear_logs_btn,
        "Clear old log files to free disk space.\n\n"
        "This will delete rotated/archived log files from previous sessions,\n"
        "but will NOT delete the current session's log files.\n\n"
        "Current session logs (aios.log, aios_debug.log) are preserved."
    )

    # Log folder size indicator
    panel.log_size_var = safe_variables.StringVar(value="Calculating...")
    log_size_label = ttk.Label(
        log_mgmt_row,
        textvariable=panel.log_size_var,
        foreground="gray",
        font=("TkDefaultFont", 9)
    )
    log_size_label.pack(side="left", padx=(15, 0))

    add_tooltip(
        log_size_label,
        "Total size of all log files in the logs directory.\n"
        "Includes current and archived log files."
    )

    # Log size will be calculated when Settings tab is activated


def create_help_section(panel: "SettingsPanel", container: ttk.Frame) -> None:
    """Create the help documentation management section.
    
    Args:
        panel: The settings panel instance
        container: The main container frame
    """
    help_frame = ttk.LabelFrame(container, text="Help Documentation", padding=8)
    help_frame.pack(fill="x", expand=False, pady=(0, 8))

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
    logger.debug("Creating cache management section")
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

    panel.cache_size_var = safe_variables.StringVar(value="1000")
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
        "Default: 1000 MB\n"
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
    
    logger.debug("Cache management section created successfully")


def create_dataset_storage_section(panel: "SettingsPanel", container: ttk.Frame) -> None:
    """Create the dataset storage configuration section (Phase 3).
    
    Args:
        panel: The settings panel instance
        container: The main container frame
    """
    logger.debug("Creating dataset storage section")
    storage_frame = ttk.LabelFrame(container, text="Dataset Storage", padding=8)
    storage_frame.pack(fill="x", pady=(0, 8))

    storage_desc_label = ttk.Label(
        storage_frame,
        text="Configure dataset download and storage limits."
    )
    storage_desc_label.pack(anchor="w", pady=(0, 10))

    # Dataset cap configuration row
    cap_row = ttk.Frame(storage_frame)
    cap_row.pack(fill="x", pady=(0, 10))

    cap_label = ttk.Label(cap_row, text="Dataset Cap (GB):", width=20, anchor="e")
    cap_label.pack(side="left", padx=(0, 10))

    panel.dataset_cap_var = safe_variables.StringVar(value="")
    cap_entry = ttk.Entry(
        cap_row,
        textvariable=panel.dataset_cap_var,
        width=10,
        font=("TkDefaultFont", 10)
    )
    cap_entry.pack(side="left", padx=(0, 10), ipady=3)

    # Usage display
    panel.dataset_usage_label = ttk.Label(
        cap_row,
        text="",
        font=("TkDefaultFont", 8),
        foreground="gray"
    )
    panel.dataset_usage_label.pack(side="left", padx=(6, 0))

    add_tooltip(
        cap_entry,
        "Maximum disk space for dataset downloads (GB).\n\n"
        "Leave empty for unlimited storage.\n"
        "Enforced in training_datasets folder.\n\n"
        "Changes are applied immediately."
    )

    # Artifacts directory configuration (Phase 3.2)
    artifacts_row = ttk.Frame(storage_frame)
    artifacts_row.pack(fill="x", pady=(10, 0))

    artifacts_label = ttk.Label(artifacts_row, text="Artifacts Directory:", width=20, anchor="e")
    artifacts_label.pack(side="left", padx=(0, 10))
    
    # Will be populated with default in _load_artifacts_path
    panel.artifacts_dir_var = safe_variables.StringVar(value="")
    artifacts_entry = ttk.Entry(
        artifacts_row,
        textvariable=panel.artifacts_dir_var,
        font=("TkDefaultFont", 10)
    )
    artifacts_entry.pack(side="left", fill="x", expand=True, padx=(0, 5), ipady=3)
    
    # Bind focus out for validation
    artifacts_entry.bind("<FocusOut>", lambda _e: panel._validate_artifacts_dir())

    # Buttons row
    artifacts_btn_row = ttk.Frame(storage_frame)
    artifacts_btn_row.pack(fill="x", pady=(5, 0))

    # Spacer to align with entry above
    spacer = ttk.Label(artifacts_btn_row, text="", width=20)
    spacer.pack(side="left", padx=(0, 10))

    browse_btn = ttk.Button(
        artifacts_btn_row,
        text="Browse…",
        command=panel._browse_artifacts_dir,
        width=10
    )
    browse_btn.pack(side="left", padx=(0, 5))

    reset_btn = ttk.Button(
        artifacts_btn_row,
        text="Use Default",
        command=panel._reset_artifacts_dir,
        width=12
    )
    reset_btn.pack(side="left")

    add_tooltip(
        artifacts_entry,
        "Custom location for artifacts and brains.\n\n"
        "Leave blank to use default location:\n"
        "ProgramData/AI-OS/artifacts (Windows)\n"
        "~/.local/share/AI-OS/artifacts (Linux)\n\n"
        "Changes take effect immediately."
    )

    add_tooltip(
        reset_btn,
        "Revert to the default artifacts path."
    )

    # Status label for artifacts directory (hidden but kept for compatibility)
    panel._artifacts_status_var = safe_variables.StringVar(value="")
    panel._artifacts_status_label = None

    # Download location configuration (Phase 3.3)
    download_row = ttk.Frame(storage_frame)
    download_row.pack(fill="x", pady=(10, 0))

    download_label = ttk.Label(download_row, text="Download Location:", width=20, anchor="e")
    download_label.pack(side="left", padx=(0, 10))

    panel.download_location_var = safe_variables.StringVar(value="training_datasets")
    download_entry = ttk.Entry(
        download_row,
        textvariable=panel.download_location_var,
        font=("TkDefaultFont", 10)
    )
    download_entry.pack(side="left", fill="x", expand=True, padx=(0, 5), ipady=3)

    # Download location buttons row
    download_btn_row = ttk.Frame(storage_frame)
    download_btn_row.pack(fill="x", pady=(5, 0))

    # Spacer to align with entry above
    spacer2 = ttk.Label(download_btn_row, text="", width=20)
    spacer2.pack(side="left", padx=(0, 10))

    browse_download_btn = ttk.Button(
        download_btn_row,
        text="📁 Browse",
        command=panel._browse_download_location,
        width=10
    )
    browse_download_btn.pack(side="left", padx=(0, 5))

    default_download_btn = ttk.Button(
        download_btn_row,
        text="Use Default",
        command=panel._reset_download_location,
        width=12
    )
    default_download_btn.pack(side="left")

    add_tooltip(
        download_entry,
        "Directory where downloaded datasets will be saved.\n\n"
        "Default: training_datasets (in aios root folder)\n"
        "Relative paths are relative to the aios root.\n\n"
        "Changes take effect for new downloads."
    )

    add_tooltip(
        default_download_btn,
        "Reset to default: training_datasets folder in aios root."
    )

    logger.debug("Dataset storage section created successfully")


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

    # Button container for Ko-Fi and GitHub buttons
    button_container = ttk.Frame(support_frame)
    button_container.pack(anchor="w", fill="x")

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
                button_container,
                image=kofi_photo,
                command=panel._open_kofi_link,
                cursor="hand2",
                borderwidth=0,
                highlightthickness=0
            )
            kofi_btn.image = kofi_photo  # type: ignore[attr-defined]
            kofi_btn.pack(side="left", padx=(0, 10))
            
            add_tooltip(
                kofi_btn,
                "Support AI-OS development on Ko-fi!\n\n"
                "Your support helps keep this project alive and growing.\n"
                "Click to visit: https://ko-fi.com/wulfic"
            )
        else:
            _create_text_kofi_button(button_container, panel._open_kofi_link)  # type: ignore[arg-type]
    except ImportError:
        _create_text_kofi_button(button_container, panel._open_kofi_link)  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"Error loading Ko-fi button: {e}")
        _create_text_kofi_button(button_container, panel._open_kofi_link)  # type: ignore[arg-type]
    
    # GitHub button (Phase 2.4)
    _create_github_button(button_container, panel._open_github_link)  # type: ignore[arg-type]


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
    kofi_text_btn.pack(side="left", padx=(0, 10))
    
    add_tooltip(
        kofi_text_btn,
        "Support AI-OS development on Ko-fi!\n\n"
        "Your support helps keep this project alive and growing.\n"
        "Click to visit: https://ko-fi.com/wulfic"
    )


def _create_github_button(parent: ttk.Frame, command: callable) -> None:  # type: ignore[valid-type]
    """Create a GitHub button for project repository.
    
    Args:
        parent: The parent frame
        command: The callback function
    """
    github_btn = ttk.Button(
        parent,
        text="⭐ GitHub Repository",
        command=command
    )
    github_btn.pack(side="left")
    
    add_tooltip(
        github_btn,
        "Visit the AI-OS GitHub repository!\n\n"
        "Star the project, report issues, or contribute.\n"
        "Click to visit: https://github.com/Wulfic/AI-OS"
    )
