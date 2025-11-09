"""Enhanced chat panel with rich message rendering support."""

from __future__ import annotations

from typing import Any, Callable, cast
import threading
from datetime import datetime

try:
    import tkinter as tk  # type: ignore
    from tkinter import ttk, filedialog  # type: ignore
except Exception:  # pragma: no cover
    tk = cast(Any, None)
    ttk = cast(Any, None)
    filedialog = cast(Any, None)

from .message_parser import MessageParser, MessageSegment
from .rich_message_widgets import CodeBlockWidget, ImageWidget, VideoWidget, LinkWidget


class RichChatPanel:
    """Enhanced chat panel with support for rich content rendering.
    
    Features:
    - Code blocks with syntax highlighting and copy buttons
    - Inline image display
    - Video preview and controls
    - Clickable links
    - Export to HTML/Markdown
    - Theme-aware colors
    - Performance optimizations
    
    Args:
        parent: Tk container
        on_send: Callback that takes user input and returns a string response
        title: Frame label
        on_load_brain: Optional callback for loading a specific brain
        on_list_brains: Optional callback that returns list of available brain names
        on_unload_model: Optional callback for unloading current model to free memory
        worker_pool: Optional worker pool for async operations
    """
    
    # Performance settings
    MAX_MESSAGES = 100  # Keep last 100 messages to prevent memory bloat
    
    # Theme-aware color schemes
    THEMES = {
        "light": {
            "canvas_bg": "#FFFFFF",
            "user_label_fg": "#0066CC",
            "user_msg_bg": "#E3F2FD",
            "user_msg_fg": "#000000",
            "user_msg_border": "#BBDEFB",
            "ai_label_fg": "#2E7D32",
            "ai_msg_bg": "#F1F8E9",
            "ai_msg_fg": "#000000",
            "ai_msg_border": "#C5E1A5",
            "system_bg": "#F5F5F5",
            "system_fg": "#666666",
            "loading_bg": "#FFFDE7",
            "loading_fg": "#666666",
        },
        "dark": {
            "canvas_bg": "#1E1E1E",
            "user_label_fg": "#4FC3F7",
            "user_msg_bg": "#263238",
            "user_msg_fg": "#E0E0E0",
            "user_msg_border": "#37474F",
            "ai_label_fg": "#81C784",
            "ai_msg_bg": "#2E3B2E",
            "ai_msg_fg": "#E0E0E0",
            "ai_msg_border": "#4CAF50",
            "system_bg": "#2b2b2b",
            "system_fg": "#AAAAAA",
            "loading_bg": "#3E2723",
            "loading_fg": "#BCAAA4",
        },
        "matrix": {
            "canvas_bg": "#000000",
            "user_label_fg": "#00FF41",
            "user_msg_bg": "#001A00",
            "user_msg_fg": "#00FF41",
            "user_msg_border": "#003300",
            "ai_label_fg": "#39FF14",
            "ai_msg_bg": "#0A0A0A",
            "ai_msg_fg": "#00CC33",
            "ai_msg_border": "#00FF41",
            "system_bg": "#0A0A0A",
            "system_fg": "#00CC33",
            "loading_bg": "#001A00",
            "loading_fg": "#00FF41",
        },
        "barbie": {
            "canvas_bg": "#FFB6C1",
            "user_label_fg": "#FF1493",
            "user_msg_bg": "#FFF0F5",
            "user_msg_fg": "#8B008B",
            "user_msg_border": "#FF69B4",
            "ai_label_fg": "#DA70D6",
            "ai_msg_bg": "#FFF0F5",
            "ai_msg_fg": "#8B008B",
            "ai_msg_border": "#DA70D6",
            "system_bg": "#FFE4E1",
            "system_fg": "#C71585",
            "loading_bg": "#FFB6C1",
            "loading_fg": "#8B008B",
        },
    }

    def __init__(
        self, 
        parent: "tk.Misc",  # type: ignore[name-defined]
        on_send: Callable[[str], str],
        *,
        title: str = "Chat",
        on_load_brain: Callable[[str], str] | None = None,
        on_list_brains: Callable[[], list[str]] | None = None,
        on_unload_model: Callable[[], str] | None = None,
        worker_pool: Any = None,
    ) -> None:
        if tk is None:
            raise RuntimeError("Tkinter not available")
        
        self._on_send = on_send
        self._on_load_brain = on_load_brain
        self._on_list_brains = on_list_brains
        self._worker_pool = worker_pool  # Store worker pool for async operations
        self._message_history: list[dict[str, Any]] = []
        self._context_length_var: Any = None
        self._context_slider: Any = None
        self._context_label: Any = None
        self._parent = parent
        self._current_theme = self._detect_theme()
        self._stop_event = threading.Event()
        self._current_thread: threading.Thread | None = None
        self._on_unload_model = on_unload_model
        
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(fill="both", expand=True, padx=4, pady=4)
        self._frame = frame
        
        # Brain selector row (if callbacks provided)
        if on_load_brain and on_list_brains:
            brain_bar = ttk.Frame(frame)
            brain_bar.pack(fill="x", padx=4, pady=4)
            
            brain_lbl = ttk.Label(brain_bar, text="Active Brain:")
            brain_lbl.pack(side="left")
            
            self.brain_var = tk.StringVar(value="<default>")
            self.brain_combo = ttk.Combobox(
                brain_bar,
                textvariable=self.brain_var,
                state="readonly",
                width=30
            )
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
            
            # Export and Copy buttons
            copy_btn = ttk.Button(brain_bar, text="📋 Copy Chat", command=self._copy_chat)
            copy_btn.pack(side="right", padx=(4, 0))
            
            export_btn = ttk.Button(brain_bar, text="💾 Export Chat", command=self._export_chat)
            export_btn.pack(side="right", padx=(4, 0))
            
            # Tooltips
            try:
                from .tooltips import add_tooltip
                add_tooltip(brain_lbl, "Select which trained brain to use for chat responses.")
                add_tooltip(self.brain_combo, "Available trained brains. Select one and click Load Brain.")
                add_tooltip(load_brain_btn, "Load the selected brain into the chat system.")
                add_tooltip(unload_btn, "Unload the current model to free GPU memory.")
                add_tooltip(refresh_brains_btn, "Refresh the list of available brains.")
                add_tooltip(self.status_label, "Shows the current model loading status.")
                add_tooltip(copy_btn, "Copy entire chat history to clipboard.")
                add_tooltip(export_btn, "Export chat history to HTML or Markdown file.")
            except Exception:
                pass
            
            # Initial load of brain list
            self._refresh_brains()
        
        # Context/Response Length input (below brain selector)
        context_bar = ttk.Frame(frame)
        context_bar.pack(fill="x", padx=4, pady=(0, 4))
        
        context_lbl = ttk.Label(context_bar, text="Max Response Length:")
        context_lbl.pack(side="left")
        
        # Input box with default 0 (auto-max), allow any positive number
        self._context_length_var = tk.StringVar(value="0")
        self._context_entry = ttk.Entry(
            context_bar,
            textvariable=self._context_length_var,
            width=10
        )
        self._context_entry.pack(side="left", padx=(4, 8))
        
        # Label showing info
        self._context_label = ttk.Label(context_bar, text="chars (0 = auto max)")
        self._context_label.pack(side="left")
        
        # Bind validation
        self._context_entry.bind("<FocusOut>", self._validate_context_length)
        self._context_entry.bind("<Return>", self._validate_context_length)
        
        # Tooltips
        try:
            from .tooltips import add_tooltip
            add_tooltip(context_lbl, "Control the maximum length of AI responses.")
            add_tooltip(self._context_entry, "Enter response length in characters. Use 0 to automatically use the model's maximum. Changes take effect on next message.")
            add_tooltip(self._context_label, "Input 0 to automatically use maximum available context.")
        except Exception:
            pass
        
        # Create scrollable chat area
        body = ttk.Frame(frame)
        body.pack(fill="both", expand=True, padx=4, pady=4)
        
        # Canvas with scrollbar for rich content (theme-aware background)
        colors = self.THEMES[self._current_theme]
        self.canvas = tk.Canvas(body, bg=colors["canvas_bg"], highlightthickness=0)
        vsb = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        
        # Frame inside canvas to hold messages
        self.messages_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.messages_frame,
            anchor="nw",
            tags="messages_frame"
        )
        
        # Bind canvas resize to update scroll region
        self.messages_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Mouse wheel scrolling
        try:
            self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        except Exception:
            pass
        
        # Input area
        input_frame = ttk.Frame(frame)
        input_frame.pack(fill="x", padx=4, pady=(0, 4))
        
        you_lbl = ttk.Label(input_frame, text="You:")
        you_lbl.pack(side="left")
        
        self.text_var = tk.StringVar(value="")
        self.entry = ttk.Entry(input_frame, textvariable=self.text_var)
        self.entry.pack(side="left", fill="x", expand=True, padx=(4, 8))
        
        # Bind Enter key
        try:
            self.entry.bind("<Return>", lambda e: self._send())
        except Exception:
            pass
        
        send_btn = ttk.Button(input_frame, text="Send", command=self._send)
        send_btn.pack(side="left")
        
        self.stop_btn = ttk.Button(input_frame, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(4, 0))
        
        clear_btn = ttk.Button(input_frame, text="Clear", command=self.clear)
        clear_btn.pack(side="left", padx=(4, 0))
        
        # Tooltips
        try:
            from .tooltips import add_tooltip
            add_tooltip(self.entry, "Type a command or message. Press Enter or click Send.")
            add_tooltip(send_btn, "Send the message to the AI.")
            add_tooltip(self.stop_btn, "Stop the current response generation.")
            add_tooltip(clear_btn, "Clear the chat history.")
        except Exception:
            pass
    
    def _detect_theme(self) -> str:
        """Detect current theme from ttk style or return default."""
        try:
            import tkinter.ttk as ttk_style
            style = ttk_style.Style()
            # Try to detect from style settings
            bg = style.lookup(".", "background")
            if bg:
                # Convert to RGB to check brightness and color
                try:
                    # Simple heuristic: dark themes have dark backgrounds
                    if bg.startswith("#"):
                        r = int(bg[1:3], 16)
                        g = int(bg[3:5], 16)
                        b = int(bg[5:7], 16)
                        brightness = (r + g + b) / 3
                        # Check for pink (high red, moderate green, high blue)
                        if r > 200 and g > 150 and b > 150 and r > b:
                            return "barbie"
                        if brightness < 50:  # Very dark = matrix
                            if g > r and g > b:  # Greenish
                                return "matrix"
                            return "dark"
                        elif brightness < 128:  # Dark
                            return "dark"
                except Exception:
                    pass
        except Exception:
            pass
        return "light"  # Default
    
    def _get_colors(self) -> dict[str, str]:
        """Get current theme colors."""
        return self.THEMES[self._current_theme]
    
    def update_theme(self, theme: str) -> None:
        """Update theme and refresh chat colors."""
        theme_normalized = theme.lower().replace(" ", "").replace("mode", "")
        if theme_normalized == "dark":
            self._current_theme = "dark"
        elif theme_normalized == "matrix":
            self._current_theme = "matrix"
        elif theme_normalized == "barbie":
            self._current_theme = "barbie"
        else:
            self._current_theme = "light"
        
        # Update canvas background
        try:
            colors = self._get_colors()
            self.canvas.config(bg=colors["canvas_bg"])
        except Exception:
            pass
    
    def _on_frame_configure(self, event: Any = None) -> None:
        """Update scroll region when frame size changes."""
        try:
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        except Exception:
            pass
    
    def _on_canvas_configure(self, event: Any) -> None:
        """Update canvas window width when canvas is resized."""
        try:
            canvas_width = event.width
            self.canvas.itemconfig(self.canvas_window, width=canvas_width)
        except Exception:
            pass
    
    def _on_mousewheel(self, event: Any) -> None:
        """Handle mouse wheel scrolling."""
        try:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass
    
    def _scroll_to_bottom(self) -> None:
        """Scroll to the bottom of the chat only if user is already viewing the bottom.
        
        This prevents auto-scrolling from interrupting users who have scrolled up
        to view earlier messages.
        """
        try:
            self.canvas.update_idletasks()
            # Get the current view position (0.0 = top, 1.0 = bottom)
            yview = self.canvas.yview()
            # If the bottom of the view is at or very close to the end (within ~5%),
            # then auto-scroll. Otherwise, respect the user's scroll position.
            if yview[1] >= 0.95:  # Within ~5% of the bottom
                self.canvas.yview_moveto(1.0)
        except Exception:
            # Fallback to always scrolling if we can't check position
            try:
                self.canvas.yview_moveto(1.0)
            except Exception:
                pass
    
    def _validate_context_length(self, event: Any = None) -> None:
        """Validate the context length input."""
        try:
            val_str = self._context_length_var.get().strip()
            if not val_str:
                self._context_length_var.set("2048")
                return
            
            val = int(val_str)
            if val < 0:
                self._context_length_var.set("0")
            # Allow any positive number or 0 (for auto-max)
        except Exception:
            # Invalid input, reset to default
            self._context_length_var.set("2048")
    
    def clear(self) -> None:
        """Clear all messages from the chat and free memory."""
        try:
            # Destroy widgets efficiently
            for widget in self.messages_frame.winfo_children():
                widget.destroy()
            self._message_history.clear()
            # Force garbage collection hint
            self.canvas.update_idletasks()
        except Exception:
            pass
    
    def _trim_old_messages(self) -> None:
        """Remove old messages if history exceeds MAX_MESSAGES."""
        try:
            if len(self._message_history) > self.MAX_MESSAGES:
                # Remove oldest messages from display
                widgets = self.messages_frame.winfo_children()
                to_remove = len(self._message_history) - self.MAX_MESSAGES
                for i in range(min(to_remove, len(widgets))):
                    widgets[i].destroy()
                # Trim history
                self._message_history = self._message_history[-self.MAX_MESSAGES:]
        except Exception:
            pass
    
    def get_context_length(self) -> int:
        """Get the current context length setting. Returns 0 for auto-max."""
        try:
            if self._context_length_var:
                val_str = self._context_length_var.get().strip()
                val = int(val_str) if val_str else 2048
                return val  # Can be 0 for auto-max
        except Exception:
            pass
        return 2048  # Default
    
    def update_context_range(self, min_val: int = 256, max_val: int = 8192, current: int = 2048) -> None:
        """Update the context input info based on loaded brain capabilities.
        
        Args:
            min_val: Minimum response length (default 256)
            max_val: Maximum response length based on model (gen_max_new_tokens * 4)
            current: Current/default value
        """
        try:
            # Update label with max info
            if self._context_label:
                self._context_label.config(text=f"chars (0 = auto, max: {max_val})")
            
            # Set current value if within range
            if self._context_length_var:
                current_val = self.get_context_length()
                if current_val == 0 or (min_val <= current_val <= max_val):
                    # Keep current value
                    pass
                else:
                    # Clamp to range
                    clamped = max(min_val, min(current_val, max_val))
                    self._context_length_var.set(str(clamped))
                    current = clamped
            
            # Add system message about updated range
            self._add_system_message(f"Response length range: {min_val}-{max_val} chars (0=auto max, current: {current})")
        except Exception as e:
            pass
    
    def _stop(self) -> None:
        """Stop the current response generation."""
        try:
            self._stop_event.set()
            self._add_system_message("[Stopping response...]")
            self._scroll_to_bottom()
            # Disable stop button immediately
            try:
                self.stop_btn.config(state="disabled")
            except Exception:
                pass
        except Exception as e:
            try:
                self._add_system_message(f"Error stopping: {e}")
            except Exception:
                pass
    
    def _unload_model(self) -> None:
        """Unload the current model to free memory."""
        if not self._on_unload_model:
            self._add_system_message("Unload not available.")
            return
        try:
            self._add_system_message("Unloading model...")
            self._scroll_to_bottom()
            
            unload_callback = self._on_unload_model
            
            def _work():
                try:
                    result = unload_callback()
                except Exception as e:
                    result = f"Error: {e}"
                
                def _render():
                    self._add_system_message(result)
                    self.update_status("No model loaded")
                    self._scroll_to_bottom()
                
                try:
                    self.canvas.after(0, _render)
                except Exception:
                    pass
            
            if self._worker_pool:
                self._worker_pool.submit(_work)
            else:
                threading.Thread(target=_work, daemon=True).start()
        except Exception as e:
            self._add_system_message(f"Failed to unload: {e}")
    
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
            brains = self._on_list_brains()
            if brains:
                self.brain_combo["values"] = brains
                if not self.brain_var.get() or self.brain_var.get() == "<default>":
                    self.brain_var.set(brains[0] if brains else "<default>")
            else:
                self.brain_combo["values"] = ["<no brains>"]
                self.brain_var.set("<no brains>")
        except Exception as e:
            self._add_system_message(f"Failed to refresh brains: {e}")
    
    def _load_brain(self) -> None:
        """Load the selected brain."""
        if not self._on_load_brain:
            return
        brain_name = self.brain_var.get()
        if not brain_name or brain_name in {"<default>", "<no brains>"}:
            self._add_system_message("No brain selected.")
            return
        
        try:
            self._add_system_message(f"Loading {brain_name}...")
            self.update_status("Loading...")
            self._scroll_to_bottom()
            
            load_callback = self._on_load_brain
            
            def _work():
                try:
                    result = load_callback(brain_name)
                    success = "error" not in result.lower()
                except Exception as e:
                    result = f"Error: {e}"
                    success = False
                
                def _render():
                    self._add_system_message(result)
                    if success:
                        self.update_status(f"Loaded - {brain_name}")
                    else:
                        self.update_status("Load failed")
                    self._scroll_to_bottom()
                
                try:
                    self.canvas.after(0, _render)
                except Exception:
                    pass
            
            if self._worker_pool:
                self._worker_pool.submit(_work)
            else:
                # Fallback to threading if no worker pool available
                threading.Thread(target=_work, daemon=True).start()
        except Exception as e:
            self._add_system_message(f"Failed to load brain: {e}")
            self.update_status("Load failed")
    
    def _send(self) -> None:
        """Send user message and get AI response."""
        msg = (self.text_var.get() or "").strip()
        if not msg:
            return
        
        try:
            # Add user message
            self._add_user_message(msg)
            
            # Clear input
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
            
            # Show loading indicator
            loading_frame = self._add_loading_message()
            self._scroll_to_bottom()
            
            def _work():
                try:
                    resp = self._on_send(msg)
                    # Check if stopped
                    if self._stop_event.is_set():
                        resp = "[Response stopped by user]"
                except Exception as e:
                    resp = f"Error: {e}"
                
                def _render():
                    try:
                        # Remove loading indicator
                        loading_frame.destroy()
                    except Exception:
                        pass
                    
                    # Add AI response with rich rendering
                    self._add_ai_message(resp)
                    self._scroll_to_bottom()
                    
                    # Disable stop button after response
                    try:
                        self.stop_btn.config(state="disabled")
                    except Exception:
                        pass
                
                try:
                    self.canvas.after(0, _render)
                except Exception:
                    pass
            
            self._current_thread = threading.Thread(target=_work, daemon=True)
            self._current_thread.start()
        except Exception as e:
            self._add_system_message(f"Error: {e}")
    
    def _add_user_message(self, message: str) -> None:
        """Add a user message to the chat with theme-aware colors.
        
        Args:
            message: User message text
        """
        colors = self._get_colors()
        msg_frame = ttk.Frame(self.messages_frame)
        msg_frame.pack(fill="x", padx=5, pady=(8, 2))
        
        # User label with theme color
        user_label = ttk.Label(
            msg_frame,
            text="You:",
            font=("Segoe UI", 10, "bold"),
            foreground=colors["user_label_fg"]
        )
        user_label.pack(anchor="w", pady=(0, 2))
        
        # Message content with theme colors and improved styling
        content_label = tk.Label(
            msg_frame,
            text=message,
            font=("Segoe UI", 10),
            bg=colors["user_msg_bg"],
            fg=colors["user_msg_fg"],
            justify="left",
            wraplength=600,
            padx=12,
            pady=10,
            relief="flat",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=colors["user_msg_border"],
            highlightcolor=colors["user_msg_border"]
        )
        content_label.pack(anchor="w", fill="x")
        
        # Store in history and trim if needed
        self._message_history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now()
        })
        self._trim_old_messages()
    
    def _add_ai_message(self, message: str) -> None:
        """Add an AI message with rich rendering and theme-aware colors.
        
        Args:
            message: AI response text (may contain code blocks, images, etc.)
        """
        colors = self._get_colors()
        msg_frame = ttk.Frame(self.messages_frame)
        msg_frame.pack(fill="x", padx=5, pady=(8, 2))
        
        # AI label with theme color
        ai_label = ttk.Label(
            msg_frame,
            text="AI:",
            font=("Segoe UI", 10, "bold"),
            foreground=colors["ai_label_fg"]
        )
        ai_label.pack(anchor="w", pady=(0, 2))
        
        # Parse message into segments
        segments = MessageParser.parse(message)
        
        # Render each segment
        for segment in segments:
            if segment.type == "text":
                self._render_text(msg_frame, segment.content)
            elif segment.type == "code":
                self._render_code(msg_frame, segment)
            elif segment.type == "image":
                self._render_image(msg_frame, segment)
            elif segment.type == "video":
                self._render_video(msg_frame, segment)
        
        # Store in history
        self._message_history.append({
            "role": "assistant",
            "content": message,
            "timestamp": datetime.now(),
            "segments": segments
        })
    
    def _add_system_message(self, message: str) -> None:
        """Add a system message to the chat with theme-aware colors.
        
        Args:
            message: System message text
        """
        colors = self._get_colors()
        msg_frame = ttk.Frame(self.messages_frame)
        msg_frame.pack(fill="x", padx=5, pady=(4, 2))
        
        content_label = tk.Label(
            msg_frame,
            text=message,
            font=("Segoe UI", 9, "italic"),
            fg=colors["system_fg"],
            bg=colors["system_bg"],
            justify="left",
            wraplength=600,
            padx=10,
            pady=6
        )
        content_label.pack(anchor="w", fill="x")
    
    def _add_loading_message(self) -> Any:
        """Add a loading indicator message with theme-aware colors.
        
        Returns:
            The frame containing the loading message (for later removal)
        """
        colors = self._get_colors()
        msg_frame = ttk.Frame(self.messages_frame)
        msg_frame.pack(fill="x", padx=5, pady=(8, 2))
        
        ai_label = ttk.Label(
            msg_frame,
            text="AI:",
            font=("Segoe UI", 10, "bold"),
            foreground=colors["ai_label_fg"]
        )
        ai_label.pack(anchor="w", pady=(0, 2))
        
        loading_label = tk.Label(
            msg_frame,
            text="⏳ Thinking...",
            font=("Segoe UI", 10, "italic"),
            fg=colors["loading_fg"],
            bg=colors["loading_bg"],
            padx=12,
            pady=8
        )
        loading_label.pack(anchor="w")
        
        return msg_frame
    
    def _render_text(self, parent: Any, text: str) -> None:
        """Render plain text segment with theme-aware colors.
        
        Args:
            parent: Parent widget
            text: Text content
        """
        if not text.strip():
            return
        
        colors = self._get_colors()
        text_label = tk.Label(
            parent,
            text=text.strip(),
            font=("Segoe UI", 10),
            bg=colors["ai_msg_bg"],
            fg=colors["ai_msg_fg"],
            justify="left",
            wraplength=600,
            padx=12,
            pady=10,
            relief="flat",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=colors["ai_msg_border"],
            highlightcolor=colors["ai_msg_border"]
        )
        text_label.pack(anchor="w", fill="x", pady=2)
    
    def _render_code(self, parent: Any, segment: MessageSegment) -> None:
        """Render code block segment.
        
        Args:
            parent: Parent widget
            segment: Code segment to render
        """
        language = segment.metadata.get("language", "text") if segment.metadata else "text"
        
        code_widget = CodeBlockWidget(
            parent,
            code=segment.content,
            language=language,
            relief="solid",
            borderwidth=1
        )
        code_widget.pack(anchor="w", fill="x", pady=2)
    
    def _render_image(self, parent: Any, segment: MessageSegment) -> None:
        """Render image segment.
        
        Args:
            parent: Parent widget
            segment: Image segment to render
        """
        alt_text = segment.metadata.get("alt", "") if segment.metadata else ""
        
        image_widget = ImageWidget(
            parent,
            image_path=segment.content,
            alt_text=alt_text
        )
        image_widget.pack(anchor="w", fill="x", pady=2)
    
    def _render_video(self, parent: Any, segment: MessageSegment) -> None:
        """Render video segment.
        
        Args:
            parent: Parent widget
            segment: Video segment to render
        """
        video_widget = VideoWidget(
            parent,
            video_path=segment.content
        )
        video_widget.pack(anchor="w", fill="x", pady=2)
    
    def _copy_chat(self) -> None:
        """Copy chat history to clipboard."""
        if not self._message_history:
            try:
                from tkinter import messagebox
                messagebox.showinfo("Copy Chat", "No messages to copy.")
            except Exception:
                pass
            return
        
        try:
            # Generate plain text version of chat
            text_lines = []
            for msg in self._message_history:
                role = msg.get("role", "unknown").capitalize()
                content = msg.get("content", "")
                text_lines.append(f"{role}: {content}")
                text_lines.append("")  # Empty line between messages
            
            chat_text = "\n".join(text_lines)
            
            # Copy to clipboard using parent widget
            if self._parent and hasattr(self._parent, 'clipboard_clear'):
                self._parent.clipboard_clear()
                self._parent.clipboard_append(chat_text)
                self._parent.update()  # Keep clipboard content after window closes
                
                try:
                    from tkinter import messagebox
                    messagebox.showinfo("Copy Chat", f"Copied {len(self._message_history)} messages to clipboard!")
                except Exception:
                    pass
        except Exception as e:
            try:
                from tkinter import messagebox
                messagebox.showerror("Copy Error", f"Failed to copy chat: {e}")
            except Exception:
                pass
    
    def _export_chat(self) -> None:
        """Export chat history to file."""
        if not self._message_history:
            try:
                from tkinter import messagebox
                messagebox.showinfo("Export", "No messages to export.")
            except Exception:
                pass
            return
        
        try:
            if filedialog is None:
                return
            
            filename = filedialog.asksaveasfilename(
                title="Export Chat",
                defaultextension=".html",
                filetypes=[
                    ("HTML files", "*.html"),
                    ("Markdown files", "*.md"),
                    ("Text files", "*.txt")
                ]
            )
            
            if not filename:
                return
            
            if filename.endswith(".html"):
                self._export_html(filename)
            elif filename.endswith(".md"):
                self._export_markdown(filename)
            else:
                self._export_text(filename)
            
            try:
                from tkinter import messagebox
                messagebox.showinfo("Export", f"Chat exported to {filename}")
            except Exception:
                pass
        except Exception as e:
            try:
                from tkinter import messagebox
                messagebox.showerror("Export Error", f"Failed to export: {e}")
            except Exception:
                pass
    
    def _export_html(self, filename: str) -> None:
        """Export chat as HTML.
        
        Args:
            filename: Output filename
        """
        html = ["<!DOCTYPE html>", "<html>", "<head>",
                "<meta charset='utf-8'>",
                "<title>AI-OS Chat Export</title>",
                "<style>",
                "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }",
                ".message { margin: 15px 0; padding: 10px; border-radius: 5px; }",
                ".user { background: #E3F2FD; border-left: 4px solid #0066CC; }",
                ".assistant { background: #F1F8E9; border-left: 4px solid #4CAF50; }",
                ".role { font-weight: bold; margin-bottom: 5px; }",
                ".code { background: #1E1E1E; color: #D4D4D4; padding: 10px; border-radius: 3px; overflow-x: auto; }",
                ".timestamp { font-size: 0.8em; color: #666; }",
                "</style>",
                "</head>", "<body>",
                "<h1>AI-OS Chat Export</h1>"]
        
        for msg in self._message_history:
            role = msg["role"]
            content = msg["content"]
            timestamp = msg["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            
            role_class = "user" if role == "user" else "assistant"
            role_label = "You" if role == "user" else "AI"
            
            html.append(f"<div class='message {role_class}'>")
            html.append(f"<div class='role'>{role_label} <span class='timestamp'>({timestamp})</span></div>")
            # Can't use backslash in f-string, so escape first
            escaped_content = content.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
            html.append(f"<div>{escaped_content}</div>")
            html.append("</div>")
        
        html.extend(["</body>", "</html>"])
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
    
    def _export_markdown(self, filename: str) -> None:
        """Export chat as Markdown.
        
        Args:
            filename: Output filename
        """
        md = ["# AI-OS Chat Export", ""]
        
        for msg in self._message_history:
            role = msg["role"]
            content = msg["content"]
            timestamp = msg["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            
            role_label = "**You**" if role == "user" else "**AI**"
            
            md.append(f"## {role_label} _{timestamp}_")
            md.append("")
            md.append(content)
            md.append("")
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
    
    def _export_text(self, filename: str) -> None:
        """Export chat as plain text.
        
        Args:
            filename: Output filename
        """
        lines = ["AI-OS Chat Export", "=" * 50, ""]
        
        for msg in self._message_history:
            role = msg["role"]
            content = msg["content"]
            timestamp = msg["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            
            role_label = "You" if role == "user" else "AI"
            
            lines.append(f"{role_label} ({timestamp}):")
            lines.append(content)
            lines.append("-" * 50)
            lines.append("")
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
