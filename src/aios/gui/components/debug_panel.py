from __future__ import annotations

# Import safe variable wrappers
from ..utils import safe_variables

from typing import Any, cast
import re
import logging
from datetime import datetime

try:  # pragma: no cover - environment dependent
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    tk = cast(Any, None)
    ttk = cast(Any, None)

try:  # pragma: no cover
    from .tooltips import add_tooltip
except Exception:
    def add_tooltip(*args, **kwargs):
        pass


class DebugPanel(ttk.LabelFrame):  # type: ignore[misc]
    """Debug console with multi-level logging and advanced filtering capabilities.

    Features:
      - Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
      - Category-based filtering (Chat, Training, Thought, Debug, Error, System, Dataset)
      - Dataset-specific quick filter button
      - Improved formatting with timestamps and visual hierarchy
      - Line wrapping and proper text spacing
    
    Methods:
      - write(text, category=None, level=None): append a line with optional category and log level
      - set_error(text): show last exception text
      - clear(): clear the console
    """

    def __init__(self, parent: Any) -> None:
        super().__init__(parent, text="🐛 Debug & System Logs (Enhanced)")
        if tk is None:
            raise RuntimeError("Tkinter not available")
        self.pack(fill="both", expand=True, padx=8, pady=8)

        # Global logging level (set from settings panel)
        self.global_log_level = "Normal"  # Default to Normal
        
        # Top control bar with quick actions
        top_bar = ttk.Frame(self)
        top_bar.pack(fill="x", padx=4, pady=(4, 2))
        
        # Formatting options
        self.show_timestamps_var = safe_variables.BooleanVar(value=True)
        ttk.Checkbutton(
            top_bar,
            text="⏰ Timestamps",
            variable=self.show_timestamps_var,
            command=self._apply_filters
        ).pack(side="left", padx=2)
        
        # Search bar row
        search_frame = ttk.Frame(self)
        search_frame.pack(fill="x", padx=4, pady=(2, 4))
        
        ttk.Label(search_frame, text="Search:").pack(side="left", padx=(0, 4))
        
        self.search_var = safe_variables.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=30)
        search_entry.pack(side="left", padx=(0, 8))
        self.search_var.trace_add("write", lambda *args: self._apply_filters())
        add_tooltip(search_entry, "Filter debug output to only show lines containing this text (case-insensitive)")
        
        clear_search_btn = ttk.Button(search_frame, text="✗", command=self._clear_search, width=3)
        clear_search_btn.pack(side="left", padx=(0, 8))
        add_tooltip(clear_search_btn, "Clear search filter")
        
        # Filter controls row
        filter_frame = ttk.Frame(self)
        filter_frame.pack(fill="x", padx=4, pady=(2, 4))
        
        ttk.Label(filter_frame, text="Show Categories:").pack(side="left", padx=(0, 8))
        
        # Filter checkboxes with icons
        self.filter_vars = {}
        categories = [
            ("💬 Chat", "chat"),
            ("🎓 Training", "training"),
            ("💭 Thought", "thought"),
            ("🔍 Debug", "debug"),
            ("❌ Error", "error"),
            ("⚙️ System", "system"),
            ("📊 Dataset", "dataset"),
        ]
        
        for label, category in categories:
            var = safe_variables.BooleanVar(value=True)
            self.filter_vars[category] = var
            cb = ttk.Checkbutton(
                filter_frame,
                text=label,
                variable=var,
                command=self._apply_filters
            )
            cb.pack(side="left", padx=2)
            add_tooltip(cb, f"Show/hide {category} category messages")
        
        all_btn = ttk.Button(filter_frame, text="✓ All", command=self._select_all_filters, width=8)
        all_btn.pack(side="left", padx=(8, 2))
        add_tooltip(all_btn, "Enable all category filters")
        
        none_btn = ttk.Button(filter_frame, text="✗ None", command=self._deselect_all_filters, width=8)
        none_btn.pack(side="left", padx=2)
        add_tooltip(none_btn, "Disable all category filters")
        
        clear_btn = ttk.Button(filter_frame, text="Clear", command=self.clear, width=8)
        clear_btn.pack(side="left", padx=2)
        add_tooltip(clear_btn, "Clear all debug messages from the console")

        # Text widget with scrollbar and enhanced styling
        text_frame = ttk.Frame(self)
        text_frame.pack(fill="both", expand=True, padx=4, pady=4)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")
        
        # Use monospace font for better formatting and readability
        self._text = tk.Text(
            text_frame, 
            height=25, 
            wrap="word", 
            yscrollcommand=scrollbar.set,
            font=("Consolas", 9),
            bg="#1E1E1E",
            fg="#D4D4D4",
            insertbackground="white",
            selectbackground="#264F78",
            spacing1=2,  # Space above lines
            spacing3=2,  # Space below lines
        )
        self._text.configure(state="disabled")
        self._text.pack(side="left", fill="both", expand=True)
        # REMOVED: Tooltip on text widget causes debug messages on every mouse move
        # add_tooltip(self._text, "Debug console showing system logs...")
        
        scrollbar.config(command=self._text.yview)
        
        # Configure tags for different categories with enhanced colors and styling
        try:
            # Category colors (VS Code Dark+ theme inspired)
            self._text.tag_configure("chat", foreground="#4EC9B0", font=("Consolas", 9, "normal"))  # Cyan-ish
            self._text.tag_configure("training", foreground="#B5CEA8", font=("Consolas", 9, "normal"))  # Green
            self._text.tag_configure("thought", foreground="#C586C0", font=("Consolas", 9, "italic"))  # Purple
            self._text.tag_configure("debug", foreground="#858585", font=("Consolas", 9, "normal"))  # Gray
            self._text.tag_configure("error", foreground="#F48771", font=("Consolas", 9, "bold"))  # Red
            self._text.tag_configure("system", foreground="#9CDCFE", font=("Consolas", 9, "normal"))  # Light blue
            self._text.tag_configure("dataset", foreground="#DCDCAA", font=("Consolas", 9, "normal"))  # Yellow
            
            # Log level tags
            self._text.tag_configure("level_DEBUG", foreground="#858585")
            self._text.tag_configure("level_INFO", foreground="#4FC1FF")
            self._text.tag_configure("level_WARNING", foreground="#FFD700")
            self._text.tag_configure("level_ERROR", foreground="#F48771", font=("Consolas", 9, "bold"))
            self._text.tag_configure("level_CRITICAL", foreground="#FF0000", font=("Consolas", 9, "bold"), background="#3B1F1F")
            
            # Timestamp tag
            self._text.tag_configure("timestamp", foreground="#608B4E", font=("Consolas", 8))  # Dark green
            
            # Separator tag
            self._text.tag_configure("separator", foreground="#3E3E3E")
        except Exception:
            pass

        # Bottom bar with statistics and error display
        self._last_error_var = safe_variables.StringVar(value="")
        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(4, 0))
        
        # Statistics
        self._stats_var = safe_variables.StringVar(value="Messages: 0 | Filtered: 0")
        ttk.Label(bar, textvariable=self._stats_var).pack(side="left", padx=(0, 8))
        
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=4)
        
        # Last error display
        ttk.Label(bar, text="Last Error:").pack(side="left", padx=(4, 4))
        self._err_entry = ttk.Entry(bar, textvariable=self._last_error_var, state="readonly")
        self._err_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        add_tooltip(self._err_entry, "Most recent error message captured by the system")
        
        # Export button
        export_btn = ttk.Button(bar, text="💾 Export Logs", command=self._export_logs, width=12)
        export_btn.pack(side="left", padx=2)
        add_tooltip(export_btn, "Export all debug messages to a text file")
        
        # Store all messages with their categories, levels, and timestamps for filtering
        self._all_messages: list[tuple[str, str, str, datetime]] = []  # (message, category, level, timestamp)
        
        # Track if user is scrolled to bottom (for auto-scroll behavior)
        self._auto_scroll = True
        self._max_text_lines = 5000  # Limit text widget to prevent performance issues
        
        # Thread safety for text widget access
        import threading
        self._text_lock = threading.RLock()  # Reentrant lock for nested calls
        self._pending_writes = []  # Buffer for pending writes
        self._write_scheduled = False
        self._destroyed = False  # Track if widget has been destroyed
        
        # Log level to numeric mapping
        self._log_level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        
        # Debouncing for filter updates
        self._filter_update_scheduled = False
        
        # Schedule initial scroll to bottom after widget is fully initialized
        # Use longer delay to ensure all initial messages are loaded
        self.after(500, self._initial_scroll_to_bottom)
        
        # Bind destroy event to clean up
        self.bind("<Destroy>", self._on_destroy)
    
    def _on_destroy(self, event=None) -> None:
        """Clean up when widget is destroyed."""
        self._destroyed = True
        try:
            if self._text_lock.acquire(blocking=False):
                try:
                    self._pending_writes.clear()
                finally:
                    self._text_lock.release()
        except Exception:
            pass
    
    def _initial_scroll_to_bottom(self) -> None:
        """Scroll to bottom on initial load."""
        if self._destroyed:
            return
        try:
            self._text.configure(state="normal")
            self._text.see("end")
            self._text.configure(state="disabled")
            self._auto_scroll = True
            # Schedule another scroll in case more messages arrived
            self.after(200, self._ensure_scrolled_to_bottom)
        except Exception:
            pass
    
    def _ensure_scrolled_to_bottom(self) -> None:
        """Ensure we're scrolled to bottom (second attempt)."""
        if self._destroyed:
            return
        try:
            self._text.see("end")
        except Exception:
            pass

    def write(self, text: str, category: str | None = None, level: str | None = None) -> None:
        """Write a message to the debug panel with optional category and log level.
        
        Args:
            text: The message to write
            category: Category (chat, training, thought, debug, error, system, dataset)
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        if self._destroyed or not text:
            return
        
        # Detect category from message if not provided
        if category is None:
            category = self._detect_category(text)
        
        # Detect log level from message if not provided
        if level is None:
            level = self._detect_log_level(text)
        
        # Store message with category, level, and timestamp
        timestamp = datetime.now()
        
        try:
            if self._text_lock.acquire(blocking=False):
                try:
                    self._all_messages.append((text, category, level, timestamp))
                    
                    # Limit stored messages to prevent memory issues
                    if len(self._all_messages) > 10000:
                        self._all_messages = self._all_messages[-8000:]
                finally:
                    self._text_lock.release()
        except Exception:
            # Failed to acquire lock, skip storing this message to prevent blocking
            pass
        
        # Update statistics (non-blocking)
        self._update_stats()
        
        # Only display if passes all filters including search
        if self._should_display_message_with_search(text, category, level):
            # Use batched writing to reduce UI updates
            self._schedule_write(text, category, level, timestamp)
    
    def _schedule_write(self, text: str, category: str, level: str, timestamp: datetime) -> None:
        """Schedule a write to be batched with other pending writes."""
        if self._destroyed:
            return
            
        try:
            # Use non-blocking lock to prevent UI freezes
            if self._text_lock.acquire(blocking=False):
                try:
                    self._pending_writes.append((text, category, level, timestamp))
                    
                    # Schedule batch write if not already scheduled
                    if not self._write_scheduled:
                        self._write_scheduled = True
                        # Use after_idle to batch multiple rapid writes into one UI update
                        try:
                            self.after_idle(self._flush_pending_writes)
                        except (tk.TclError, RuntimeError):
                            # Widget might be destroyed, clear flag
                            self._write_scheduled = False
                finally:
                    self._text_lock.release()
            # If we can't acquire the lock immediately, just drop this message
            # to prevent UI freezes - it's better to miss a log message than hang
        except Exception:
            # Failed to acquire lock or widget destroyed, skip this write
            pass
    
    def _flush_pending_writes(self) -> None:
        """Flush all pending writes to the text widget in one batch."""
        if self._destroyed:
            return
            
        writes = []
        try:
            # Use non-blocking lock to prevent UI freezes
            if self._text_lock.acquire(blocking=False):
                try:
                    if not self._pending_writes:
                        self._write_scheduled = False
                        return
                    
                    # Get all pending writes
                    writes = self._pending_writes[:]
                    self._pending_writes.clear()
                    self._write_scheduled = False
                finally:
                    self._text_lock.release()
            else:
                # Couldn't acquire lock, reschedule with shorter delay
                self.after(50, self._flush_pending_writes)
                return
        except Exception:
            self._write_scheduled = False
            return
        
        if not writes:
            return
        
        # Perform the actual write (outside lock to avoid deadlock)
        try:
            # Check if user is at the bottom before adding new text
            at_bottom = self._is_scrolled_to_bottom()
            
            self._text.configure(state="normal")
            
            # Limit text widget size to prevent performance issues
            try:
                line_count = int(self._text.index('end-1c').split('.')[0])
                if line_count > self._max_text_lines:
                    # Delete oldest 1000 lines to keep widget responsive
                    self._text.delete('1.0', f'{1000}.0')
            except (tk.TclError, ValueError):
                pass
            
            # Write all pending messages at once
            for text, category, level, timestamp in writes:
                try:
                    # Format timestamp if enabled
                    if self.show_timestamps_var.get():
                        ts_str = timestamp.strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
                        self._text.insert("end", f"[{ts_str}] ", "timestamp")
                    
                    # Format log level with appropriate coloring
                    level_str = f"[{level:8s}] "  # Fixed width for alignment
                    self._text.insert("end", level_str, f"level_{level}")
                    
                    # Format category with icon
                    category_icons = {
                        "chat": "💬",
                        "training": "🎓",
                        "thought": "💭",
                        "debug": "🔍",
                        "error": "❌",
                        "system": "⚙️",
                        "dataset": "📊",
                    }
                    icon = category_icons.get(category, "📝")
                    category_str = f"{icon} [{category.upper():8s}] "
                    self._text.insert("end", category_str, category)
                    
                    # Insert the actual message
                    clean_text = text.strip()
                    self._text.insert("end", clean_text + "\n", category)
                    
                    # Add subtle separator for errors and critical messages
                    if level in ["ERROR", "CRITICAL"]:
                        self._text.insert("end", "─" * 80 + "\n", "separator")
                except (tk.TclError, RuntimeError):
                    # Skip individual message if widget is in invalid state
                    continue
                except Exception:
                    # Skip individual message if it fails
                    continue
            
            self._text.configure(state="disabled")
            
            # Only auto-scroll if user was already at bottom
            if at_bottom:
                try:
                    self._text.see("end")
                except Exception:
                    pass
        except (tk.TclError, RuntimeError):
            # Widget in invalid state, skip entire batch
            pass
        except Exception as e:
            # Log error but don't crash
            import sys
            print(f"Error flushing debug panel writes: {e}", file=sys.stderr)
        finally:
            try:
                self._text.configure(state="disabled")
            except Exception:
                pass
    
    def _append_to_text(self, text: str, category: str, level: str, timestamp: datetime) -> None:
        """Append text to the text widget with enhanced formatting.
        
        DEPRECATED: Use _schedule_write instead for better batching.
        Kept for compatibility with _apply_filters.
        
        Format: [TIMESTAMP] [LEVEL] [CATEGORY] Message
        """
        # Redirect to scheduled write for consistency
        self._schedule_write(text, category, level, timestamp)
    
    def _is_scrolled_to_bottom(self) -> bool:
        """Check if the text widget is scrolled to the bottom.
        
        Non-blocking check that safely handles exceptions.
        """
        if self._destroyed:
            return True
        try:
            # Get the current scroll position
            # yview returns (top_fraction, bottom_fraction)
            result = self._text.yview()
            if result and len(result) >= 2:
                _, bottom = result
                # If bottom fraction is very close to 1.0, we're at the bottom
                return bottom >= 0.98
        except (tk.TclError, RuntimeError, AttributeError):
            # Widget might be in an invalid state (e.g., during destruction)
            pass
        except Exception:
            # Catch any other unexpected exceptions
            pass
        return True  # Default to auto-scroll if we can't determine position
    
    def _detect_category(self, text: str) -> str:
        """Detect category from message content."""
        text_lower = text.lower()
        
        # Check for explicit tags
        tag_match = re.match(r'^\[([^\]]+)\]', text)
        if tag_match:
            tag = tag_match.group(1).lower()
            if 'chat' in tag:
                return 'chat'
            elif 'dataset' in tag or 'download' in tag:
                return 'dataset'
            elif 'train' in tag or 'hrm' in tag or 'optimization' in tag:
                return 'training'
            elif 'error' in tag or 'exception' in tag:
                return 'error'
            elif 'system' in tag or 'status' in tag:
                return 'system'
            elif 'thought' in tag or 'thinking' in tag:
                return 'thought'
        
        # Check for error indicators
        if any(x in text_lower for x in ['error', 'exception', 'traceback', 'failed', 'critical']):
            return 'error'
        
        # Check for dataset keywords (important to catch these)
        if any(x in text_lower for x in ['dataset', 'download', 'fetching', 'sha256', 'huggingface']):
            return 'dataset'
        
        # Check for chat keywords
        if any(x in text_lower for x in ['chat', 'message', 'response']):
            return 'chat'
        
        # Check for training keywords
        if any(x in text_lower for x in ['training', 'epoch', 'loss', 'batch', 'optimizer', 'gradient']):
            return 'training'
        
        # Check for thought process
        if any(x in text_lower for x in ['thinking', 'reasoning', 'analyzing', 'considering']):
            return 'thought'
        
        # Check for system keywords
        if any(x in text_lower for x in ['system', 'status', 'memory', 'cpu', 'gpu', 'device']):
            return 'system'
        
        return 'debug'
    
    def _detect_log_level(self, text: str) -> str:
        """Detect log level from message content."""
        text_lower = text.lower()
        
        # Check for explicit log level indicators
        if 'critical' in text_lower or 'fatal' in text_lower:
            return 'CRITICAL'
        elif 'error' in text_lower or 'exception' in text_lower or 'failed' in text_lower:
            return 'ERROR'
        elif 'warning' in text_lower or 'warn' in text_lower:
            return 'WARNING'
        elif 'info' in text_lower:
            return 'INFO'
        else:
            return 'DEBUG'
    
    def set_global_log_level(self, level: str) -> None:
        """Set the global logging level from settings panel.
        
        Args:
            level: One of "Normal", "Advanced", or "DEBUG"
        """
        self.global_log_level = level
        self._apply_filters()
    
    def _should_display_message(self, category: str, level: str) -> bool:
        """Check if message should be displayed based on current filters."""
        # Check global log level filter (set from settings)
        # Normal: Only CRITICAL errors + essential outputs
        # Advanced: Normal + WARNING + INFO
        # DEBUG: Everything
        msg_level = self._log_level_map.get(level, logging.DEBUG)
        
        if self.global_log_level == "Normal":
            # Only show CRITICAL and essential messages (errors)
            if msg_level < logging.ERROR:
                return False
        elif self.global_log_level == "Advanced":
            # Show INFO and above (filters out DEBUG)
            if msg_level < logging.INFO:
                return False
        # DEBUG shows everything (no filtering by level)
        
        # Check category filter
        if not self.filter_vars.get(category, safe_variables.BooleanVar(value=True)).get():
            return False
        
        return True
    
    def _should_display_message_with_search(self, message: str, category: str, level: str) -> bool:
        """Check if message should be displayed based on current filters including search.
        
        Args:
            message: The message text
            category: Message category
            level: Message log level
            
        Returns:
            True if message should be displayed
        """
        # First check the regular filters
        if not self._should_display_message(category, level):
            return False
        
        # Then check search filter (case-insensitive)
        search_text = self.search_var.get().strip()
        if search_text:
            if search_text.lower() not in message.lower():
                return False
        
        return True
    
    def _clear_search(self) -> None:
        """Clear the search filter."""
        self.search_var.set("")
    
    def _update_stats(self) -> None:
        """Update the statistics display."""
        total = len(self._all_messages)
        filtered = sum(1 for msg, cat, lvl, ts in self._all_messages 
                      if self._should_display_message_with_search(msg, cat, lvl))
        self._stats_var.set(f"Messages: {total} | Visible: {filtered}")
    
    def _apply_filters(self) -> None:
        """Reapply filters to show/hide messages based on current filter settings.
        
        This is debounced to prevent excessive UI updates during rapid filter changes.
        """
        if self._destroyed:
            return
        
        # Debounce: cancel any pending filter update and schedule a new one
        if self._filter_update_scheduled:
            # Already scheduled, let that one handle it
            return
        
        self._filter_update_scheduled = True
        # Delay the actual filter application slightly to batch rapid changes
        self.after(100, self._apply_filters_impl)
    
    def _apply_filters_impl(self) -> None:
        """Internal implementation of filter application (debounced)."""
        self._filter_update_scheduled = False
        
        if self._destroyed:
            return
            
        # Get a snapshot of messages to avoid holding lock during UI update
        messages_snapshot = []
        try:
            # Use non-blocking lock acquisition to prevent hangs
            if self._text_lock.acquire(blocking=False):
                try:
                    messages_snapshot = self._all_messages[:]
                finally:
                    self._text_lock.release()
            else:
                # Couldn't get lock immediately, reschedule
                self._filter_update_scheduled = True
                self.after(50, self._apply_filters_impl)
                return
        except Exception:
            return
        
        try:
            # Remember if user was at bottom before rebuilding
            was_at_bottom = self._is_scrolled_to_bottom()
            
            self._text.configure(state="normal")
            self._text.delete("1.0", "end")
            
            # Re-add messages that match current filters (with line limit for performance)
            messages_to_display = [
                (msg, cat, lvl, ts) for msg, cat, lvl, ts in messages_snapshot
                if self._should_display_message_with_search(msg, cat, lvl)
            ]
            
            # Only show the last N messages to keep UI responsive
            display_limit = min(len(messages_to_display), self._max_text_lines)
            for message, category, level, timestamp in messages_to_display[-display_limit:]:
                # Build formatted line in memory first (faster than multiple inserts)
                line_parts = []
                
                if self.show_timestamps_var.get():
                    ts_str = timestamp.strftime("%H:%M:%S.%f")[:-3]
                    line_parts.append(f"[{ts_str}] ")
                
                line_parts.append(f"[{level:8s}] ")
                
                category_icons = {
                    "chat": "💬", "training": "🎓", "thought": "💭",
                    "debug": "🔍", "error": "❌", "system": "⚙️", "dataset": "📊",
                }
                icon = category_icons.get(category, "📝")
                line_parts.append(f"{icon} [{category.upper():8s}] ")
                line_parts.append(message.strip() + "\n")
                
                if level in ["ERROR", "CRITICAL"]:
                    line_parts.append("─" * 80 + "\n")
                
                # Insert the complete line at once
                line = "".join(line_parts)
                self._text.insert("end", line, category)
            
            self._text.configure(state="disabled")
            
            # Restore scroll position
            if was_at_bottom:
                try:
                    self._text.see("end")
                except Exception:
                    pass
            
            # Update statistics
            self._update_stats()
            
        except (tk.TclError, RuntimeError):
            # Widget might be in invalid state, skip this update
            pass
        except Exception as e:
            import sys
            print(f"Error applying filters: {e}", file=sys.stderr)
        finally:
            try:
                self._text.configure(state="disabled")
            except Exception:
                pass
    
    def _select_all_filters(self) -> None:
        """Enable all category filters."""
        for var in self.filter_vars.values():
            var.set(True)
        self._apply_filters()
    
    def _deselect_all_filters(self) -> None:
        """Disable all category filters."""
        for var in self.filter_vars.values():
            var.set(False)
        self._apply_filters()

    def set_error(self, text: str) -> None:
        """Set and display an error message."""
        self._last_error_var.set(text or "")
        if text:
            self.write(text, category='error', level='ERROR')

    def clear(self) -> None:
        """Clear all messages and reset the display."""
        if self._destroyed:
            return
            
        try:
            if self._text_lock.acquire(blocking=False):
                try:
                    self._last_error_var.set("")
                    self._all_messages.clear()
                    self._pending_writes.clear()
                finally:
                    self._text_lock.release()
            else:
                # Couldn't get lock, skip clear operation
                return
        except Exception:
            return
        
        try:
            self._text.configure(state="normal")
            self._text.delete("1.0", "end")
            self._text.configure(state="disabled")
        except (tk.TclError, RuntimeError):
            # Widget in invalid state
            pass
        except Exception:
            pass
        
        try:
            self._update_stats()
        except Exception:
            pass
    
    def _export_logs(self) -> None:
        """Export all logs to a file."""
        logger.info("User action: Exporting debug logs")
        try:
            from tkinter import filedialog
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filedialog.asksaveasfilename(
                defaultextension=".log",
                filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
                initialfile=f"aios_debug_{timestamp_str}.log"
            )
            if filename:
                logger.debug(f"Exporting {len(self._all_messages)} log messages to {filename}")
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"AI-OS Debug Log Export\n")
                    f.write(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total Messages: {len(self._all_messages)}\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for message, category, level, timestamp in self._all_messages:
                        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        f.write(f"[{ts_str}] [{level:8s}] [{category.upper():8s}] {message.strip()}\n")
                
                logger.info(f"Successfully exported {len(self._all_messages)} log messages to {filename}")
                self.write(f"Logs exported successfully to: {filename}", category="system", level="INFO")
            else:
                logger.debug("Log export cancelled by user")
        except Exception as e:
            logger.error(f"Failed to export logs: {e}", exc_info=True)
            self.write(f"Failed to export logs: {str(e)}", category="error", level="ERROR")
