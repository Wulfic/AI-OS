"""Brain management utilities for rich chat panel."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .panel_main import RichChatPanel

from . import message_display
from .event_handlers import scroll_to_bottom
from ...utils.resource_management import submit_background


logger = logging.getLogger(__name__)


def refresh_brains(panel: RichChatPanel) -> None:
    """Refresh the list of available brains.
    
    This runs asynchronously via worker pool if available to prevent GUI blocking.
    
    Args:
        panel: Rich chat panel instance
    """
    if not panel._on_list_brains:
        return
    
    # Check if brain UI components exist (they may not if callbacks weren't provided)
    if not hasattr(panel, 'brain_combo') or not hasattr(panel, 'brain_var'):
        return
    
    def _do_refresh():
        """Background refresh operation."""
        try:
            logger.info("Brain refresh task running")
            brains = panel._on_list_brains()
            
            # Schedule UI update on main thread
            def _update_ui():
                try:
                    if brains:
                        panel.brain_combo["values"] = brains
                        if not panel.brain_var.get() or panel.brain_var.get() == "<default>":
                            panel.brain_var.set(brains[0] if brains else "<default>")
                    else:
                        panel.brain_combo["values"] = ["<no brains>"]
                        panel.brain_var.set("<no brains>")
                    logger.info("Brain list updated (%d brain(s))", len(brains))
                except Exception as e:
                    if hasattr(panel, 'canvas'):
                        message_display.add_system_message(panel, f"Failed to update brain list: {e}")
            
            try:
                panel.canvas.after(0, _update_ui)
            except Exception:
                pass
        except Exception as e:
            logger.error("Failed to refresh brain list: %s", e, exc_info=True)
            # Schedule error handling on main thread
            def _handle_error():
                if hasattr(panel, 'canvas'):
                    message_display.add_system_message(panel, f"Failed to refresh brains: {e}")
            
            try:
                panel.canvas.after(0, _handle_error)
            except Exception:
                pass
    
    try:
        submit_background("chat-brain-refresh", _do_refresh, pool=panel._worker_pool)
    except RuntimeError as exc:
        logger.error("Failed to queue brain refresh: %s", exc)
        message_display.add_system_message(panel, f"Failed to refresh brains: {exc}")


def load_brain(panel: RichChatPanel) -> None:
    """Load the selected brain.
    
    Args:
        panel: Rich chat panel instance
    """
    if not panel._on_load_brain:
        return
    
    # Check if brain UI components exist
    if not hasattr(panel, 'brain_var'):
        return
    
    brain_name = panel.brain_var.get()
    if not brain_name or brain_name in {"<default>", "<no brains>"}:
        message_display.add_system_message(panel, "No brain selected.")
        return
    
    try:
        message_display.add_system_message(panel, f"Loading {brain_name}...")
        update_status(panel, "Loading...")
        scroll_to_bottom(panel)
        
        load_callback = panel._on_load_brain
        
        def _work():
            try:
                result = load_callback(brain_name)
                success = "error" not in result.lower()
            except Exception as e:
                result = f"Error: {e}"
                success = False
            
            def _render():
                message_display.add_system_message(panel, result)
                if success:
                    update_status(panel, f"Loaded - {brain_name}")
                else:
                    update_status(panel, "Load failed")
                scroll_to_bottom(panel)
            
            try:
                panel.canvas.after(0, _render)
            except Exception:
                pass
        
        try:
            submit_background("chat-brain-load", _work, pool=panel._worker_pool)
        except RuntimeError as exc:
            message_display.add_system_message(panel, f"Failed to load brain: {exc}")
            update_status(panel, "Load failed")
    except Exception as e:
        message_display.add_system_message(panel, f"Failed to load brain: {e}")
        update_status(panel, "Load failed")


def unload_model(panel: RichChatPanel) -> None:
    """Unload the current model to free memory.
    
    Args:
        panel: Rich chat panel instance
    """
    if not panel._on_unload_model:
        message_display.add_system_message(panel, "Unload not available.")
        return
    try:
        message_display.add_system_message(panel, "Unloading model...")
        scroll_to_bottom(panel)
        
        unload_callback = panel._on_unload_model
        
        def _work():
            try:
                result = unload_callback()
            except Exception as e:
                result = f"Error: {e}"
            
            def _render():
                message_display.add_system_message(panel, result)
                update_status(panel, "No model loaded")
                scroll_to_bottom(panel)
            
            try:
                panel.canvas.after(0, _render)
            except Exception:
                pass
        
        try:
            submit_background("chat-brain-unload", _work, pool=panel._worker_pool)
        except RuntimeError as exc:
            message_display.add_system_message(panel, f"Failed to unload: {exc}")
            update_status(panel, "Unload failed")
    except Exception as e:
        message_display.add_system_message(panel, f"Failed to unload: {e}")


def update_status(panel: RichChatPanel, status: str) -> None:
    """Update the model status indicator.
    
    Args:
        panel: Rich chat panel instance
        status: Status text (e.g., 'Loaded - brain_name', 'No model loaded', 'Loading...')
    """
    try:
        if hasattr(panel, 'status_label'):
            panel.status_label.config(text=f"Status: {status}")
            # Color code based on status
            if "loaded" in status.lower() and "no model" not in status.lower():
                panel.status_label.config(foreground="green")
            elif "loading" in status.lower():
                panel.status_label.config(foreground="orange")
            else:
                panel.status_label.config(foreground="gray")
    except Exception:
        pass
