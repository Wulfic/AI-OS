"""Chat operation utilities for rich chat panel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .panel_main import RichChatPanel

from . import message_display
from .event_handlers import scroll_to_bottom
from ...utils.resource_management import submit_background


def send_message(panel: RichChatPanel) -> None:
    """Send user message and get AI response.
    
    Args:
        panel: Rich chat panel instance
    """
    msg = (panel.text_var.get() or "").strip()
    if not msg:
        return
    
    try:
        # Add user message
        message_display.add_user_message(panel, msg)
        
        # Clear input
        try:
            panel.text_var.set("")
        except Exception:
            pass
        
        # Clear stop event and enable stop button
        panel._stop_event.clear()
        try:
            panel.stop_btn.config(state="normal")
        except Exception:
            pass
        
        # Show loading indicator
        loading_frame = message_display.add_loading_message(panel)
        scroll_to_bottom(panel)
        
        # Track accumulated response
        response_parts = []
        
        def _on_token(token: str) -> None:
            """Handle each token from streaming."""
            if not panel._stop_event.is_set():
                response_parts.append(token)
        
        def _on_done() -> None:
            """Handle completion of streaming."""
            def _render():
                try:
                    # Remove loading indicator
                    if loading_frame:
                        loading_frame.destroy()
                except Exception:
                    pass
                
                # Join all tokens into full response
                resp = "".join(response_parts) if response_parts else "[No response]"
                
                # Check if stopped
                if panel._stop_event.is_set():
                    resp = "[Response stopped by user]"
                
                # Add AI response with rich rendering
                message_display.add_ai_message(panel, resp)
                scroll_to_bottom(panel)
                
                # Disable stop button after response
                try:
                    panel.stop_btn.config(state="disabled")
                except Exception:
                    pass
                panel._current_thread = None
            
            try:
                panel.canvas.after(0, _render)
            except Exception:
                pass
        
        def _on_error(error: str) -> None:
            """Handle error during streaming."""
            def _render():
                try:
                    # Remove loading indicator
                    if loading_frame:
                        loading_frame.destroy()
                except Exception:
                    pass
                
                # Add error message
                message_display.add_ai_message(panel, f"Error: {error}")
                scroll_to_bottom(panel)
                
                # Disable stop button after error
                try:
                    panel.stop_btn.config(state="disabled")
                except Exception:
                    pass
                panel._current_thread = None
            
            try:
                panel.canvas.after(0, _render)
            except Exception:
                pass
        
        def _work():
            try:
                # Get max response chars from panel (0 = unlimited)
                max_chars = get_context_length(panel)
                # Get sampling parameters
                sampling_params = get_sampling_params(panel)
                # Call streaming API with callbacks, char limit, and sampling params
                panel._on_send(msg, _on_token, _on_done, _on_error, None, max_chars, sampling_params)
            except Exception as e:
                _on_error(str(e))

        try:
            panel._current_thread = submit_background(
                "chat-send",
                _work,
                pool=getattr(panel, "_worker_pool", None),
            )
        except RuntimeError as exc:
            _on_error(f"queue full: {exc}")
    except Exception as e:
        message_display.add_system_message(panel, f"Error: {e}")


def stop_generation(panel: RichChatPanel) -> None:
    """Stop the current response generation.
    
    Args:
        panel: Rich chat panel instance
    """
    try:
        panel._stop_event.set()
        message_display.add_system_message(panel, "[Stopping response...]")
        scroll_to_bottom(panel)
        # Disable stop button immediately
        try:
            panel.stop_btn.config(state="disabled")
        except Exception:
            pass
        try:
            if panel._current_thread is not None:
                panel._current_thread.cancel()
        except Exception:
            pass
    except Exception as e:
        try:
            message_display.add_system_message(panel, f"Error stopping: {e}")
        except Exception:
            pass


def clear_chat(panel: RichChatPanel) -> None:
    """Clear all messages from the chat and free memory.
    
    Args:
        panel: Rich chat panel instance
    """
    try:
        # Destroy widgets efficiently
        for widget in panel.messages_frame.winfo_children():
            widget.destroy()
        panel._message_history.clear()
        # Force garbage collection hint
        panel.canvas.update_idletasks()
    except Exception:
        pass


def get_sampling_params(panel: RichChatPanel) -> dict[str, float]:
    """Get the current sampling parameter settings.
    
    Args:
        panel: Rich chat panel instance
    
    Returns:
        Dictionary with temperature, top_p, top_k
    """
    params = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
    }
    
    try:
        if hasattr(panel, '_temperature_var'):
            temp_str = panel._temperature_var.get().strip()
            params["temperature"] = max(0.0, min(2.0, float(temp_str)))
    except Exception:
        pass
    
    try:
        if hasattr(panel, '_top_p_var'):
            topp_str = panel._top_p_var.get().strip()
            params["top_p"] = max(0.0, min(1.0, float(topp_str)))
    except Exception:
        pass
    
    try:
        if hasattr(panel, '_top_k_var'):
            topk_str = panel._top_k_var.get().strip()
            params["top_k"] = max(0, int(float(topk_str)))
    except Exception:
        pass
    
    return params


def get_context_length(panel: RichChatPanel) -> int:
    """Get the current context length setting.
    
    Args:
        panel: Rich chat panel instance
    
    Returns:
        Context length (0 for auto-max)
    """
    try:
        if panel._context_length_var:
            val_str = panel._context_length_var.get().strip()
            val = int(val_str) if val_str else 0  # Default to auto-max
            return val  # Can be 0 for auto-max
    except Exception:
        pass
    return 0  # Default to auto-max (no limit)


def update_context_range(
    panel: RichChatPanel, 
    min_val: int = 256, 
    max_val: int = 8192, 
    current: int = 2048
) -> None:
    """Update the context input info based on loaded brain capabilities.
    
    Args:
        panel: Rich chat panel instance
        min_val: Minimum response length (default 256)
        max_val: Maximum response length based on model (gen_max_new_tokens * 4)
        current: Current/default value
    """
    try:
        # Update label with max info
        if panel._context_label:
            panel._context_label.config(text=f"chars (0 = auto, max: {max_val})")
        
        # Set current value if within range
        if panel._context_length_var:
            current_val = get_context_length(panel)
            if current_val == 0 or (min_val <= current_val <= max_val):
                # Keep current value
                pass
            else:
                # Clamp to range
                clamped = max(min_val, min(current_val, max_val))
                panel._context_length_var.set(str(clamped))
                current = clamped
        
        # Add system message about updated range
        message_display.add_system_message(
            panel,
            f"Response length range: {min_val}-{max_val} chars (0=auto max, current: {current})"
        )
    except Exception:
        pass
