from __future__ import annotations

from typing import Any, Callable, cast
import logging
import threading

logger = logging.getLogger(__name__)

try:
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    tk = cast(Any, None)
    ttk = cast(Any, None)

# Import safe variable wrappers
from ..utils import safe_variables


class ChatPanel:
    """Simple chat/command panel that sends user text via a callback.

    Args:
        parent: Tk container
        on_send: Callable that takes the raw user input and returns a string response
        title: Frame label
        on_load_brain: Optional callback for loading a specific brain (takes brain name)
        on_list_brains: Optional callback that returns list of available brain names
        on_unload_model: Optional callback for unloading current model to free memory
    """

    def __init__(
        self, 
        parent: "tk.Misc",  # type: ignore[name-defined]
        on_send: Callable[[str], str],
        *,
        title: str = "Chat",
        on_load_brain: Callable[[str], str] | None = None,
        on_list_brains: Callable[[], list[str]] | None = None,
        on_unload_model: Callable[[], str] | None = None,
    ) -> None:
        if tk is None:
            raise RuntimeError("Tkinter not available")
        
        logger.info("Initializing Chat Panel")
        
        self._on_send = on_send
        self._on_load_brain = on_load_brain
        self._on_list_brains = on_list_brains
        self._on_unload_model = on_unload_model
        self._stop_event = threading.Event()
        self._current_thread: threading.Thread | None = None

        frame = ttk.LabelFrame(parent, text=title)
        # Expand chat panel to use available space on the tab
        frame.pack(fill="both", expand=True)

        # Brain selector row (if callbacks provided)
        if on_load_brain and on_list_brains:
            brain_bar = ttk.Frame(frame)
            brain_bar.pack(fill="x", padx=4, pady=4)
            
            brain_lbl = ttk.Label(brain_bar, text="Active Brain:")
            brain_lbl.pack(side="left")
            
            self.brain_var = safe_variables.StringVar(value="<default>")
            self.brain_combo = ttk.Combobox(brain_bar, textvariable=self.brain_var, state="readonly", width=30)
            self.brain_combo.pack(side="left", padx=(4, 4))
            
            load_brain_btn = ttk.Button(brain_bar, text="Load Brain", command=self._load_brain)
            load_brain_btn.pack(side="left")
            
            unload_btn = ttk.Button(brain_bar, text="Unload", command=self._unload_model)
            unload_btn.pack(side="left", padx=(4, 0))
            
            refresh_brains_btn = ttk.Button(brain_bar, text="Refresh", command=self._refresh_brains)
            refresh_brains_btn.pack(side="left", padx=(4, 0))
            
            # Status indicator
            self.status_label = ttk.Label(brain_bar, text="Status: No model loaded", foreground="gray")
            self.status_label.pack(side="left", padx=(12, 0))
            
            # Tooltips
            try:  # pragma: no cover - UI enhancement only
                from .tooltips import add_tooltip
                add_tooltip(brain_lbl, "Select which trained brain to use for chat responses.")
                add_tooltip(self.brain_combo, "Available trained brains. Select one and click Load Brain.")
                add_tooltip(load_brain_btn, "Load the selected brain into the chat system.")
                add_tooltip(unload_btn, "Unload the current model to free GPU memory.")
                add_tooltip(refresh_brains_btn, "Refresh the list of available brains.")
                add_tooltip(self.status_label, "Shows the current model loading status.")
            except Exception:
                pass
            
            # Initial load of brain list
            self._refresh_brains()

        body = ttk.Frame(frame)
        body.pack(fill="both", expand=True)
        # Make the chat log taller for better readability
        self.log = tk.Text(body, wrap="word", height=16)
        vsb = ttk.Scrollbar(body, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=vsb.set)
        self.log.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        bar = ttk.Frame(frame)
        bar.pack(fill="x")
        you_lbl = ttk.Label(bar, text="You:")
        you_lbl.pack(side="left")
        self.text_var = safe_variables.StringVar(value="status")
        self.entry = ttk.Entry(bar, textvariable=self.text_var)
        self.entry.pack(side="left", fill="x", expand=True, padx=(4, 8))
        try:
            # Enter to send (UX improvement)
            self.entry.bind("<Return>", lambda e: self._send())
        except Exception:
            pass
        send_btn = ttk.Button(bar, text="Send", command=self._send)
        send_btn.pack(side="left")
        self.stop_btn = ttk.Button(bar, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(4, 0))
        clear_btn = ttk.Button(bar, text="Clear", command=self.clear)
        clear_btn.pack(side="left", padx=(4, 0))

        # Tooltips (best-effort; ignore if tooltip module not available)
        try:  # pragma: no cover - UI enhancement only
            from .tooltips import add_tooltip
            add_tooltip(self.log, "Conversation transcript. Scroll to review earlier messages.")
            add_tooltip(self.entry, "Type a command or message. Press Enter or click Send to submit.")
            add_tooltip(send_btn, "Send the current text to the system (executes command / prompts model).")
            add_tooltip(self.stop_btn, "Stop the current response generation (if any).")
            add_tooltip(clear_btn, "Clear the chat log (does not affect underlying state).")
            add_tooltip(you_lbl, "Label for your input line.")
        except Exception:
            pass
        
        logger.info("Chat Panel initialized successfully")

    # public helpers
    def clear(self) -> None:
        logger.info("User action: Clearing chat log")
        try:
            self.log.delete("1.0", tk.END)
            logger.debug("Chat log cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear chat log: {e}")

    def _stop(self) -> None:
        """Stop the current response generation."""
        logger.info("User action: Stopping response generation")
        try:
            self._stop_event.set()
            self.log.insert(tk.END, "\n[Stopping...]\n")
            self.log.see(tk.END)
            # Disable stop button immediately
            try:
                self.stop_btn.config(state="disabled")
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Error during stop operation: {e}")
            try:
                self.log.insert(tk.END, f"Error stopping: {e}\n")
            except Exception:
                pass

    def _unload_model(self) -> None:
        """Unload the current model to free memory."""
        if not self._on_unload_model:
            self.log.insert(tk.END, "Unload not available.\n")
            return
        try:
            logger.info("User action: Unloading model to free memory")
            self.log.insert(tk.END, "Unloading model...\n")
            self.log.see(tk.END)
            
            unload_callback = self._on_unload_model
            
            def _work():
                try:
                    result = unload_callback()
                    logger.info("Model unloaded successfully")
                except Exception as e:
                    logger.error(f"Failed to unload model: {e}", exc_info=True)
                    result = f"Error: {e}"
                
                def _render():
                    self.log.insert(tk.END, f"{result}\n")
                    self.update_status("No model loaded")
                    self.log.see(tk.END)
                
                try:
                    self.log.after(0, _render)
                except Exception:
                    pass
            
            threading.Thread(target=_work, daemon=True).start()
        except Exception as e:
            logger.error(f"Failed to initiate model unload: {e}", exc_info=True)
            self.log.insert(tk.END, f"Failed to unload: {e}\n")
    
    def update_status(self, status: str) -> None:
        """Update the model status indicator.
        
        Args:
            status: Status text (e.g., 'Loaded - brain_name', 'No model loaded', 'Loading...')
        """
        try:
            if hasattr(self, 'status_label'):
                self.status_label.config(text=f"Status: {status}")
                # Color code based on status
                if "loaded" in status.lower() and "no model" not in status.lower():
                    self.status_label.config(foreground="green")
                elif "loading" in status.lower():
                    self.status_label.config(foreground="orange")
                else:
                    self.status_label.config(foreground="gray")
        except Exception:
            pass
    
    def _refresh_brains(self) -> None:
        """Refresh the list of available brains."""
        if not self._on_list_brains:
            return
        try:
            logger.debug("Refreshing brain list")
            brains = self._on_list_brains()
            if brains:
                self.brain_combo["values"] = brains
                logger.info(f"Brain list refreshed: {len(brains)} brain(s) available")
                # Select first brain if available
                if not self.brain_var.get() or self.brain_var.get() == "<default>":
                    self.brain_var.set(brains[0] if brains else "<default>")
                    logger.debug(f"Auto-selected brain: {brains[0] if brains else '<default>'}")
            else:
                self.brain_combo["values"] = ["<no brains>"]
                self.brain_var.set("<no brains>")
                logger.warning("No brains available")
        except Exception as e:
            logger.error(f"Failed to refresh brain list: {e}", exc_info=True)
            self.log.insert(tk.END, f"[ui] Failed to refresh brains: {e}\n")

    def _load_brain(self) -> None:
        """Load the selected brain."""
        if not self._on_load_brain:
            return
        brain_name = self.brain_var.get()
        if not brain_name or brain_name in {"<default>", "<no brains>"}:
            logger.warning("Attempted to load brain but no valid brain selected")
            self.log.insert(tk.END, "No brain selected.\n")
            return
        try:
            logger.info(f"User action: Loading brain '{brain_name}'")
            self.log.insert(tk.END, f"Loading {brain_name}...\n")
            self.update_status("Loading...")
            self.log.see(tk.END)
            
            # Capture callback in local variable to satisfy type checker
            load_callback = self._on_load_brain
            
            def _work():
                try:
                    result = load_callback(brain_name)  # Now guaranteed non-None
                    success = "error" not in result.lower()
                    if success:
                        logger.info(f"Brain '{brain_name}' loaded successfully")
                    else:
                        logger.warning(f"Brain '{brain_name}' load failed: {result}")
                except Exception as e:
                    logger.error(f"Failed to load brain '{brain_name}': {e}", exc_info=True)
                    result = f"Error: {e}"
                    success = False
                
                def _render():
                    self.log.insert(tk.END, f"{result}\n")
                    if success:
                        self.update_status(f"Loaded - {brain_name}")
                    else:
                        self.update_status("Load failed")
                    self.log.see(tk.END)
                
                try:
                    self.log.after(0, _render)
                except Exception:
                    pass
            
            threading.Thread(target=_work, daemon=True).start()
        except Exception as e:
            logger.error(f"Failed to initiate brain load for '{brain_name}': {e}", exc_info=True)
            self.log.insert(tk.END, f"Failed to load brain: {e}\n")
            self.update_status("Load failed")

    def update_theme(self, theme: str) -> None:
        """Update the chat panel theme.
        
        Args:
            theme: Theme name (e.g., 'Light Mode', 'Dark Mode', 'Matrix Mode', etc.)
        """
        logger.debug(f"Applying theme '{theme}' to chat panel")
        # Get theme colors
        colors = self._get_theme_colors(theme)
        logger.debug(f"Theme colors: bg={colors['bg']}, fg={colors['fg']}")
        
        # Update Text widget
        try:
            self.log.config(
                bg=colors["bg"],
                fg=colors["fg"],
                insertbackground=colors["fg"]
            )
        except Exception as e:
            logger.error(f"Failed to apply theme to chat panel: {e}")
    
    def _get_theme_colors(self, theme: str) -> dict[str, str]:
        """Get colors for the specified theme.
        
        Args:
            theme: Theme name
            
        Returns:
            Dictionary with color values
        """
        theme_normalized = theme.lower().replace(" ", "").replace("mode", "")
        
        if theme_normalized == "dark":
            return {"bg": "#2b2b2b", "fg": "#e0e0e0"}
        elif theme_normalized == "matrix":
            return {"bg": "#000000", "fg": "#00ff41"}
        elif theme_normalized == "halloween":
            return {"bg": "#1a0f00", "fg": "#ffcc99"}
        elif theme_normalized == "barbie":
            return {"bg": "#FFB6C1", "fg": "#8B008B"}
        else:  # Light mode
            return {"bg": "#ffffff", "fg": "#000000"}

    # internals
    def _send(self) -> None:
        msg = (self.text_var.get() or "").strip()
        if not msg:
            return
        try:
            logger.info(f"User message: '{msg[:100]}{'...' if len(msg) > 100 else ''}' ({len(msg)} chars)")
            # Echo user
            self.log.insert(tk.END, f"You: {msg}\n")
            # Insert a temporary loading line (brief - model stays loaded now)
            loading_idx = self.log.index(tk.END)
            self.log.insert(tk.END, "AI: ...\n")
            self.log.see(tk.END)
            # Clear input immediately for better UX
            try:
                self.text_var.set("")
            except Exception:
                pass
            
            # Clear stop event and enable stop button
            self._stop_event.clear()
            try:
                self.stop_btn.config(state="normal")
            except Exception:
                pass

            def _work():
                try:
                    logger.debug("Calling chat handler")
                    resp = self._on_send(msg)
                    # Check if stopped
                    if self._stop_event.is_set():
                        logger.info("Chat response stopped by user")
                        resp = "[Response stopped by user]"
                    else:
                        logger.info(f"Chat response received: {len(str(resp))} chars")
                except Exception as e:  # capture errors and render in UI thread
                    logger.error(f"Chat handler error: {e}", exc_info=True)
                    resp = f"Error: {e}"

                def _render():
                    try:
                        # Replace loading line with the actual response
                        self.log.delete(loading_idx, tk.END)
                    except Exception:
                        pass
                    if isinstance(resp, str) and "\n" in resp:
                        self.log.insert(tk.END, "AI:\n")
                        self.log.insert(tk.END, resp.rstrip() + "\n\n")
                    else:
                        self.log.insert(tk.END, f"AI: {resp}\n\n")
                    self.log.see(tk.END)
                    # Disable stop button after response completes
                    try:
                        self.stop_btn.config(state="disabled")
                    except Exception:
                        pass

                try:
                    # Schedule UI update on Tk thread
                    self.log.after(0, _render)
                except Exception:
                    pass

            self._current_thread = threading.Thread(target=_work, daemon=True)
            self._current_thread.start()
        except Exception as e:
            logger.error(f"Failed to send message: {e}", exc_info=True)
            self.log.insert(tk.END, f"Error: {e}\n")

    def get_state(self) -> dict[str, Any]:
        """Get current chat panel state for persistence.
        
        Returns:
            Dictionary containing chat panel state
        """
        state = {}
        
        # Save selected brain if brain selector exists
        if hasattr(self, 'brain_var'):
            try:
                selected_brain = self.brain_var.get()
                if selected_brain and selected_brain != "<default>":
                    state['selected_brain'] = selected_brain
                    logger.debug(f"Chat panel state: selected_brain={selected_brain}")
            except Exception as e:
                logger.warning(f"Failed to get brain_var state: {e}")
        
        return state
    
    def set_state(self, state: dict[str, Any]) -> None:
        """Restore chat panel state from saved data.
        
        Args:
            state: Dictionary containing chat panel state
        """
        if not state:
            return
        
        logger.info(f"Restoring chat panel state with {len(state)} parameters")
        
        # Restore selected brain
        if 'selected_brain' in state and hasattr(self, 'brain_var'):
            try:
                brain_name = state['selected_brain']
                self.brain_var.set(brain_name)
                logger.debug(f"Restored selected brain: {brain_name}")
            except Exception as e:
                logger.warning(f"Failed to restore selected brain: {e}")
        
        logger.debug("Chat panel state restoration complete")
    
    def cleanup(self) -> None:
        """Clean up chat panel resources on shutdown."""
        logger.info("Cleaning up Chat Panel")
        
        # Stop any active response generation
        if self._current_thread and self._current_thread.is_alive():
            logger.info("Stopping active chat response generation")
            try:
                self._stop_event.set()
                self._current_thread.join(timeout=2.0)
                if self._current_thread.is_alive():
                    logger.warning("Chat response thread did not terminate within timeout")
            except Exception as e:
                logger.error(f"Error stopping chat thread during cleanup: {e}")
        
        logger.info("Chat Panel cleanup complete")
