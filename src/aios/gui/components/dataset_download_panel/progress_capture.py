"""
Real-time Progress Capture for Dataset Downloads

Captures stderr writes and forwards them to a log function in real-time.
Enables live progress tracking for tqdm progress bars during dataset downloads.
"""

import re
import tkinter as tk
from typing import Callable


class RealTimeProgressCapture:
    """
    A file-like object that captures stderr writes and forwards them to a log function in real-time.
    This enables live progress tracking for tqdm progress bars during dataset downloads.
    """
    
    def __init__(self, log_func: Callable[[str], None], parent_widget: tk.Widget):
        """
        Initialize progress capture.
        
        Args:
            log_func: Function to call with progress messages
            parent_widget: Parent tkinter widget for scheduling GUI updates
        """
        self.log_func = log_func
        self.parent = parent_widget
        self.buffer = ""
        self.last_progress_line = ""
    
    def write(self, text: str) -> int:
        """Write text to the capture, forwarding to log function immediately."""
        if not text:
            return 0
        
        # Handle carriage returns (tqdm uses \r to update the same line)
        if '\r' in text:
            # Split by \r and process each part
            parts = text.split('\r')
            for part in parts[:-1]:
                if part.strip():
                    clean = re.sub(r'\x1b\[[0-9;]*m', '', part)
                    if clean.strip() and ('Downloading' in clean or '%' in clean or 'MB/s' in clean or 'files' in clean):
                        # Schedule GUI update in main thread - check if widget exists
                        try:
                            if self.parent.winfo_exists():
                                self.parent.after(0, lambda msg=clean.strip(): self.log_func(f"   {msg}"))
                        except Exception:
                            pass  # Widget destroyed or mainloop not started
            # Keep the last part as current line
            self.last_progress_line = parts[-1]
            if self.last_progress_line.strip():
                clean = re.sub(r'\x1b\[[0-9;]*m', '', self.last_progress_line)
                if clean.strip() and ('Downloading' in clean or '%' in clean or 'MB/s' in clean or 'files' in clean):
                    # Schedule GUI update in main thread - check if widget exists
                    try:
                        if self.parent.winfo_exists():
                            self.parent.after(0, lambda msg=clean.strip(): self.log_func(f"   {msg}"))
                    except Exception:
                        pass  # Widget destroyed or mainloop not started
        elif '\n' in text:
            # Process complete lines
            self.buffer += text
            lines = self.buffer.split('\n')
            for line in lines[:-1]:
                if line.strip():
                    clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
                    if clean.strip() and ('Downloading' in clean or '%' in clean or 'MB/s' in clean or 'files' in clean or 'Generating' in clean):
                        # Schedule GUI update in main thread - check if widget exists
                        try:
                            if self.parent.winfo_exists():
                                self.parent.after(0, lambda msg=clean.strip(): self.log_func(f"   {msg}"))
                        except Exception:
                            pass  # Widget destroyed or mainloop not started
            self.buffer = lines[-1]
        else:
            # Accumulate partial line
            self.buffer += text
        
        return len(text)
    
    def flush(self):
        """Flush any remaining buffered content."""
        if self.buffer.strip():
            clean = re.sub(r'\x1b\[[0-9;]*m', '', self.buffer)
            if clean.strip() and ('Downloading' in clean or '%' in clean or 'MB/s' in clean or 'files' in clean):
                # Schedule GUI update in main thread - check if widget exists
                try:
                    if self.parent.winfo_exists():
                        self.parent.after(0, lambda msg=clean.strip(): self.log_func(f"   {msg}"))
                except Exception:
                    pass  # Widget destroyed or mainloop not started
            self.buffer = ""
    
    def isatty(self):
        """Return False - we're not a terminal."""
        return False
