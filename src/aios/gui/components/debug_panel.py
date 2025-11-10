from __future__ import annotations

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
        
        # Quick filter: Hide Dataset Logs button
        self.hide_dataset_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            top_bar,
            text="🚫 Hide Dataset Logs",
            variable=self.hide_dataset_var,
            command=self._apply_filters
        ).pack(side="left", padx=2)
        
        ttk.Separator(top_bar, orient="vertical").pack(side="left", fill="y", padx=8)
        
        # Formatting options
        self.show_timestamps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            top_bar,
            text="⏰ Timestamps",
            variable=self.show_timestamps_var,
            command=self._apply_filters
        ).pack(side="left", padx=2)
        
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
            var = tk.BooleanVar(value=True)
            self.filter_vars[category] = var
            cb = ttk.Checkbutton(
                filter_frame,
                text=label,
                variable=var,
                command=self._apply_filters
            )
            cb.pack(side="left", padx=2)
            add_tooltip(cb, f"Show/hide {category} category messages")
        
        all_btn = ttk.Button(filter_frame, text="✓ All", command=self._select_all_filters, width=6)
        all_btn.pack(side="left", padx=(8, 2))
        add_tooltip(all_btn, "Enable all category filters")
        
        none_btn = ttk.Button(filter_frame, text="✗ None", command=self._deselect_all_filters, width=6)
        none_btn.pack(side="left", padx=2)
        add_tooltip(none_btn, "Disable all category filters")
        
        clear_btn = ttk.Button(filter_frame, text="🗑️ Clear", command=self.clear, width=6)
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
        add_tooltip(self._text, "Debug console showing system logs, training output, chat messages, and errors. Use filters above to focus on specific categories.")
        
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
        self._last_error_var = tk.StringVar(value="")
        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(4, 0))
        
        # Statistics
        self._stats_var = tk.StringVar(value="Messages: 0 | Filtered: 0")
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
        
        # Log level to numeric mapping
        self._log_level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

    def write(self, text: str, category: str | None = None, level: str | None = None) -> None:
        """Write a message to the debug panel with optional category and log level.
        
        Args:
            text: The message to write
            category: Category (chat, training, thought, debug, error, system, dataset)
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        if not text:
            return
        
        # Detect category from message if not provided
        if category is None:
            category = self._detect_category(text)
        
        # Detect log level from message if not provided
        if level is None:
            level = self._detect_log_level(text)
        
        # Store message with category, level, and timestamp
        timestamp = datetime.now()
        self._all_messages.append((text, category, level, timestamp))
        
        # Limit stored messages to prevent memory issues
        if len(self._all_messages) > 10000:
            self._all_messages = self._all_messages[-8000:]
        
        # Update statistics
        self._update_stats()
        
        # Only display if passes all filters
        if self._should_display_message(category, level):
            self._append_to_text(text, category, level, timestamp)
    
    def _append_to_text(self, text: str, category: str, level: str, timestamp: datetime) -> None:
        """Append text to the text widget with enhanced formatting.
        
        Format: [TIMESTAMP] [LEVEL] [CATEGORY] Message
        """
        try:
            self._text.configure(state="normal")
            
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
            
            # Auto-scroll to bottom
            self._text.see("end")
        finally:
            self._text.configure(state="disabled")
    
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
        
        # Check dataset filter
        if self.hide_dataset_var.get() and category == "dataset":
            return False
        
        # Check category filter
        if not self.filter_vars.get(category, tk.BooleanVar(value=True)).get():
            return False
        
        return True
    
    def _update_stats(self) -> None:
        """Update the statistics display."""
        total = len(self._all_messages)
        filtered = sum(1 for msg, cat, lvl, ts in self._all_messages 
                      if self._should_display_message(cat, lvl))
        self._stats_var.set(f"Messages: {total} | Visible: {filtered}")
    
    def _apply_filters(self) -> None:
        """Reapply filters to show/hide messages based on current filter settings."""
        try:
            self._text.configure(state="normal")
            self._text.delete("1.0", "end")
            
            # Re-add all messages that match current filters
            for message, category, level, timestamp in self._all_messages:
                if self._should_display_message(category, level):
                    self._append_to_text(message, category, level, timestamp)
            
            # Update statistics
            self._update_stats()
            
            self._text.see("end")
        finally:
            self._text.configure(state="disabled")
    
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
        self._last_error_var.set("")
        self._all_messages.clear()
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")
        self._update_stats()
    
    def _export_logs(self) -> None:
        """Export all logs to a file."""
        try:
            from tkinter import filedialog
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filedialog.asksaveasfilename(
                defaultextension=".log",
                filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
                initialfile=f"aios_debug_{timestamp_str}.log"
            )
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"AI-OS Debug Log Export\n")
                    f.write(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total Messages: {len(self._all_messages)}\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for message, category, level, timestamp in self._all_messages:
                        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        f.write(f"[{ts_str}] [{level:8s}] [{category.upper():8s}] {message.strip()}\n")
                
                self.write(f"Logs exported successfully to: {filename}", category="system", level="INFO")
        except Exception as e:
            self.write(f"Failed to export logs: {str(e)}", category="error", level="ERROR")
