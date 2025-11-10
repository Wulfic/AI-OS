"""Main settings panel class."""

from __future__ import annotations
import webbrowser
import logging
from typing import Any, Callable
from tkinter import ttk
import tkinter as tk

from . import ui_builders, theme_manager, startup_settings, cache_management

logger = logging.getLogger(__name__)


class SettingsPanel:
    """Settings panel with theme selection and other preferences."""

    def __init__(
        self,
        parent: Any,
        save_state_fn: Callable[[], None] | None = None,
        chat_panel: Any | None = None,
        help_panel: Any | None = None,
    ) -> None:
        self.parent = parent
        self._save_state_fn = save_state_fn
        self._chat_panel = chat_panel
        self._help_panel = help_panel
        
        # Flag to prevent trace callbacks during state restoration
        self._restoring_state = False

        # Main container with canvas for scrolling
        main_container = ttk.Frame(parent)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Build UI with two-column layout
        ui_builders.create_title(main_container)
        
        # Create two columns
        columns_frame = ttk.Frame(main_container)
        columns_frame.pack(fill="both", expand=True)
        
        left_column = ttk.Frame(columns_frame)
        left_column.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        right_column = ttk.Frame(columns_frame)
        right_column.pack(side="left", fill="both", expand=True, padx=(5, 0))
        
        # Left column: Appearance, General Settings, Support
        ui_builders.create_appearance_section(self, left_column)
        ui_builders.create_general_settings_section(self, left_column)
        ui_builders.create_support_section(self, left_column)
        
        # Right column: Help, Cache
        ui_builders.create_help_section(self, right_column)
        ui_builders.create_cache_section(self, right_column)
        
        # Load initial settings
        self._load_startup_status()
        self._load_cache_size()
        self._refresh_cache_stats()

    def _apply_theme(self, theme: str) -> None:
        """Apply the selected theme to the application."""
        theme_manager.apply_theme(self, theme)

    def _open_kofi_link(self) -> None:
        """Open Ko-fi link in default browser."""
        try:
            webbrowser.open("https://ko-fi.com/wulfic")
        except Exception as e:
            logger.error(f"Error opening Ko-fi link: {e}")

    def _load_startup_status(self) -> None:
        """Load current startup status from Windows registry."""
        startup_settings.load_startup_status(self)

    def _on_startup_changed(self) -> None:
        """Handle startup checkbox toggle."""
        startup_settings.on_startup_changed(self)

    def _on_start_minimized_changed(self) -> None:
        """Handle start minimized checkbox toggle."""
        startup_settings.on_start_minimized_changed(self)

    def _on_minimize_to_tray_changed(self) -> None:
        """Handle minimize to tray checkbox toggle."""
        startup_settings.on_minimize_to_tray_changed(self)

    def _load_cache_size(self) -> None:
        """Load cache size configuration from config file."""
        cache_management.load_cache_size(self)

    def _save_cache_size(self) -> None:
        """Save cache size configuration to config file."""
        cache_management.save_cache_size(self)

    def _refresh_cache_stats(self) -> None:
        """Refresh and display cache statistics."""
        cache_management.refresh_cache_stats(self)

    def _clear_cache(self) -> None:
        """Clear all cached dataset blocks."""
        cache_management.clear_cache(self)

    def _rebuild_help_index(self) -> None:
        """Rebuild the help documentation search index."""
        import threading
        from pathlib import Path
        
        def rebuild():
            try:
                # Update status
                self.help_index_status_label.config(text="Building...")
                
                # Find docs root
                from ...gui.components.help_panel import utils
                project_root = utils.find_project_root(Path(__file__))
                docs_root = utils.resolve_docs_root(project_root)
                
                # Delete old index
                index_file = docs_root / "search_index.json"
                if index_file.exists():
                    index_file.unlink()
                    logger.info("Deleted old search index")
                
                # Build new index
                from ...gui.components.help_panel.search_engine import SearchEngine
                engine = SearchEngine(docs_root)
                success = engine.build_index()
                
                if success:
                    self.help_index_status_label.config(text=f"✓ Ready ({len(engine.index)} docs)")
                    logger.info(f"Rebuilt help index with {len(engine.index)} documents")
                else:
                    self.help_index_status_label.config(text="✗ Failed to rebuild")
                    logger.error("Failed to rebuild help index")
            except Exception as e:
                self.help_index_status_label.config(text="✗ Error")
                logger.error(f"Error rebuilding help index: {e}")
        
        # Run in background thread
        threading.Thread(target=rebuild, daemon=True).start()

    def get_state(self) -> dict[str, Any]:
        """Return current settings state for persistence."""
        return {
            "theme": self.theme_var.get(),
            "startup_enabled": self.startup_var.get(),
            "start_minimized": self.start_minimized_var.get(),
            "minimize_to_tray": self.minimize_to_tray_var.get(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore settings state from saved data."""
        # Set flag to prevent trace callbacks during restoration
        self._restoring_state = True
        
        try:
            theme = state.get("theme")
            if theme in ("Light Mode", "Dark Mode", "Matrix Mode", "Barbie Mode", "Halloween Mode"):
                self.theme_var.set(theme)
                # Apply immediately on load
                self._apply_theme(theme)
                logger.info(f"Restored and applied theme: {theme}")
        except Exception as e:
            logger.error(f"Failed to restore theme: {e}", exc_info=True)
            # Ensure theme_var has a valid default value even if restore fails
            if not self.theme_var.get() or self.theme_var.get() not in ("Light Mode", "Dark Mode", "Matrix Mode", "Barbie Mode", "Halloween Mode"):
                logger.warning("No valid theme found, defaulting to Dark Mode")
                self.theme_var.set("Dark Mode")
                self._apply_theme("Dark Mode")
        
        try:
            startup = state.get("startup_enabled")
            start_minimized = state.get("start_minimized", False)
            if startup is not None:
                self.startup_var.set(bool(startup))
                # Apply startup setting immediately
                from ...utils.startup import set_startup_enabled, is_windows
                if is_windows():
                    set_startup_enabled(bool(startup), minimized=bool(start_minimized))
        except Exception:
            pass
        
        try:
            start_minimized = state.get("start_minimized")
            if start_minimized is not None:
                self.start_minimized_var.set(bool(start_minimized))
        except Exception:
            pass
        
        try:
            minimize_to_tray = state.get("minimize_to_tray")
            if minimize_to_tray is not None:
                self.minimize_to_tray_var.set(bool(minimize_to_tray))
        except Exception:
            pass
        
        # Clear flag after restoration is complete
        self._restoring_state = False
