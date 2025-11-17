"""Startup preference management for platform autostart."""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .panel_main import SettingsPanel

logger = logging.getLogger(__name__)


def load_startup_status(panel: "SettingsPanel") -> None:
    """Load current startup status from Windows registry.
    
    Args:
        panel: The settings panel instance
    """
    try:
        from ...utils.startup import (
            is_startup_enabled,
            get_startup_path,
            is_windows,
            is_linux,
        )

        checkbox = getattr(panel, "startup_check", None)
        supported = is_windows() or is_linux()

        if not supported:
            panel.startup_var.set(False)
            panel.startup_info.config(
                text="ℹ Not available on this platform",
                foreground="gray"
            )
            if checkbox:
                try:
                    checkbox.state(["disabled"])
                except Exception:
                    pass
            return

        if checkbox:
            try:
                checkbox.state(["!disabled"])
            except Exception:
                pass

        enabled = is_startup_enabled()
        panel.startup_var.set(enabled)

        enabled_context = ""
        disabled_context = ""
        if is_windows():
            enabled_context = " at Windows login"
            disabled_context = " during Windows login"
        elif is_linux():
            enabled_context = " at login"
            disabled_context = " after you log in"

        if enabled:
            path = get_startup_path()
            if path:
                display_path = path if len(path) <= 60 else path[:57] + "..."
                panel.startup_info.config(
                    text=f"✓ Enabled{enabled_context}: {display_path}",
                    foreground="green"
                )
            else:
                panel.startup_info.config(
                    text=f"✓ Enabled{enabled_context}",
                    foreground="green"
                )
        else:
            panel.startup_info.config(
                text=f"○ Disabled - AI-OS will not launch automatically{disabled_context}",
                foreground="gray"
            )
    except Exception as e:
        panel.startup_info.config(
            text=f"⚠ Error loading startup status: {e}",
            foreground="red"
        )


def on_startup_changed(panel: "SettingsPanel") -> None:
    """Handle startup checkbox toggle.
    
    Args:
        panel: The settings panel instance
    """
    try:
        from ...utils.startup import (
            set_startup_enabled,
            verify_startup_command,
            is_windows,
            is_linux,
        )

        enabled = panel.startup_var.get()
        minimized = panel.start_minimized_var.get()

        context_suffix = ""
        if is_windows():
            context_suffix = " (Windows login)"
        elif is_linux():
            context_suffix = " (login)"

        logger.info(
            "User action: %s autostart%s (minimized: %s)",
            "Enabling" if enabled else "Disabling",
            context_suffix,
            minimized,
        )

        success = set_startup_enabled(enabled, minimized=minimized)

        if success:
            if enabled:
                is_valid, message = verify_startup_command()
                if is_valid:
                    logger.info("Successfully enabled autostart%s", context_suffix)
                    load_startup_status(panel)
                else:
                    logger.warning("Autostart enabled but validation failed: %s", message)
                    panel.startup_info.config(
                        text=f"⚠ Warning: {message}",
                        foreground="orange"
                    )
            else:
                logger.info("Successfully disabled autostart%s", context_suffix)
                load_startup_status(panel)

            if panel._save_state_fn:
                panel._save_state_fn()
        else:
            logger.error("Failed to update autostart setting - check permissions")
            panel.startup_var.set(not enabled)
            panel.startup_info.config(
                text="⚠ Failed to update autostart setting. Check permissions.",
                foreground="red"
            )
    except Exception as e:
        logger.error(f"Error changing startup setting: {e}", exc_info=True)
        # Error occurred - revert checkbox
        panel.startup_var.set(not panel.startup_var.get())
        panel.startup_info.config(
            text=f"⚠ Error: {e}",
            foreground="red"
        )


def on_start_minimized_changed(panel: "SettingsPanel") -> None:
    """Handle start minimized checkbox toggle.
    
    Args:
        panel: The settings panel instance
    """
    try:
        minimized = panel.start_minimized_var.get()
        logger.info(f"User action: {'Enabling' if minimized else 'Disabling'} start minimized to tray")
        
        # Update the startup command if startup is enabled
        from ...utils.startup import is_startup_enabled, set_startup_enabled
        if is_startup_enabled():
            # Re-set startup with updated minimized flag
            enabled = panel.startup_var.get()
            if set_startup_enabled(enabled, minimized=minimized):
                logger.debug("Updated autostart command with minimized flag: %s", minimized)
            else:
                logger.warning("Failed to update autostart command with minimized flag")
        
        # Save state
        if panel._save_state_fn:
            panel._save_state_fn()
    except Exception as e:
        logger.error(f"Error updating start minimized setting: {e}", exc_info=True)


def on_minimize_to_tray_changed(panel: "SettingsPanel") -> None:
    """Handle minimize to tray checkbox toggle.
    
    Args:
        panel: The settings panel instance
    """
    try:
        minimize_to_tray = panel.minimize_to_tray_var.get()
        logger.info(f"User action: {'Enabling' if minimize_to_tray else 'Disabling'} minimize to tray on close")
        
        # Update the app's minimize behavior
        if panel._save_state_fn:
            panel._save_state_fn()
        
        # Notify parent app to sync settings
        if hasattr(panel.parent.master, '_sync_tray_settings'):
            try:
                app = panel.parent.master
                while app and not hasattr(app, '_sync_tray_settings'):
                    app = getattr(app, 'master', None)
                if app and hasattr(app, '_sync_tray_settings'):
                    app._sync_tray_settings()
                    logger.debug("Synced tray settings with main app")
            except Exception as e:
                logger.debug(f"Could not sync tray settings: {e}")
                pass
    except Exception as e:
        logger.error(f"Error updating minimize to tray setting: {e}", exc_info=True)
