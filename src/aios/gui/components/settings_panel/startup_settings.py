"""Windows startup and registry management."""

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
        from ...utils.startup import is_startup_enabled, get_startup_path, is_windows
        from tkinter import ttk
        
        if not is_windows():
            panel.startup_info.config(
                text="ℹ Not available on this platform (Windows only)",
                foreground="gray"
            )
            # Disable checkbox on non-Windows
            try:
                for child in panel.parent.winfo_children():
                    if isinstance(child, ttk.LabelFrame) and "General Settings" in str(child.cget("text")):
                        for widget in child.winfo_children():
                            if isinstance(widget, ttk.Checkbutton):
                                widget.config(state="disabled")
                                break
                        break
            except Exception:
                pass
            return
        
        enabled = is_startup_enabled()
        panel.startup_var.set(enabled)
        
        if enabled:
            path = get_startup_path()
            if path:
                # Truncate path if too long
                display_path = path if len(path) < 60 else path[:57] + "..."
                panel.startup_info.config(
                    text=f"✓ Enabled: {display_path}",
                    foreground="green"
                )
            else:
                panel.startup_info.config(
                    text="✓ Enabled",
                    foreground="green"
                )
        else:
            panel.startup_info.config(
                text="○ Disabled - AI-OS will not start on boot",
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
        from ...utils.startup import set_startup_enabled, get_startup_path, verify_startup_command
        
        enabled = panel.startup_var.get()
        # Check if should start minimized
        minimized = panel.start_minimized_var.get()
        
        logger.info(f"User action: {'Enabling' if enabled else 'Disabling'} startup at Windows boot (minimized: {minimized})")
        
        success = set_startup_enabled(enabled, minimized=minimized)
        
        if success:
            logger.info(f"Successfully {'enabled' if enabled else 'disabled'} startup at Windows boot")
            if enabled:
                # Verify the command is valid
                is_valid, message = verify_startup_command()
                if is_valid:
                    path = get_startup_path()
                    if path:
                        display_path = path if len(path) < 60 else path[:57] + "..."
                        panel.startup_info.config(
                            text=f"✓ Enabled: {display_path}",
                            foreground="green"
                        )
                        logger.debug(f"Startup registry path: {path}")
                    else:
                        panel.startup_info.config(
                            text="✓ Enabled successfully",
                            foreground="green"
                        )
                else:
                    logger.warning(f"Startup enabled but command validation failed: {message}")
                    panel.startup_info.config(
                        text=f"⚠ Warning: {message}",
                        foreground="orange"
                    )
            else:
                panel.startup_info.config(
                    text="○ Disabled - AI-OS will not start on boot",
                    foreground="gray"
                )
            
            # Save state
            if panel._save_state_fn:
                panel._save_state_fn()
        else:
            logger.error("Failed to modify Windows registry for startup setting - check permissions")
            # Failed to set - revert checkbox
            panel.startup_var.set(not enabled)
            panel.startup_info.config(
                text="⚠ Failed to modify Windows registry. Check permissions.",
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
            set_startup_enabled(enabled, minimized=minimized)
            logger.debug(f"Updated Windows startup command with minimized flag: {minimized}")
        
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
