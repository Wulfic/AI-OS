"""
Real-time Progress Capture for Dataset Downloads

Captures stderr writes and forwards them to a log function in real-time.
Enables live progress tracking for tqdm progress bars during dataset downloads.
"""

import re
import tkinter as tk
from typing import Callable, Optional


class RealTimeProgressCapture:
    """
    A file-like object that captures stderr writes and forwards them to a log function in real-time.
    This enables live progress tracking for tqdm progress bars during dataset downloads.
    """
    
    def __init__(self, log_func: Callable[[str], None], parent_widget: tk.Widget, progress_tracker: Optional['DownloadProgressTracker'] = None):
        """
        Initialize progress capture.
        
        Args:
            log_func: Function to call with progress messages
            parent_widget: Parent tkinter widget for scheduling GUI updates
            progress_tracker: Optional progress tracker to update with parsed download progress
        """
        self.log_func = log_func
        self.parent = parent_widget
        self.buffer = ""
        self.last_progress_line = ""
        self.progress_tracker = progress_tracker
        self._last_parsed_bytes = 0
        self._total_bytes = 0
    
    def _parse_hf_progress(self, text: str) -> None:
        """Parse HuggingFace download progress and update tracker."""
        if not self.progress_tracker:
            return
        
        # Try to extract download information from HuggingFace progress bars
        # Format examples:
        # "Downloading: 100%|██████████| 1.23G/1.23G [00:30<00:00, 41.0MB/s]"
        # "Downloading data files:  50%|█████     | 1/2 [00:15<00:15, 15.0s/it]"
        # "Generating train split: 1000 examples [00:05, 180.50 examples/s]"
        
        # Extract bytes downloaded (e.g., "1.23G/2.46G" or "512M/1G")
        bytes_match = re.search(r'(\d+\.?\d*)(K|M|G|T)B?/(\d+\.?\d*)(K|M|G|T)B?', text)
        if bytes_match:
            current_value = float(bytes_match.group(1))
            current_unit = bytes_match.group(2)
            total_value = float(bytes_match.group(3))
            total_unit = bytes_match.group(4)
            
            # Convert to bytes
            unit_multipliers = {'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4}
            current_bytes = int(current_value * unit_multipliers.get(current_unit, 1))
            total_bytes = int(total_value * unit_multipliers.get(total_unit, 1))
            
            # Update total if we detected it
            if total_bytes > self._total_bytes:
                self._total_bytes = total_bytes
                self.progress_tracker.set_totals(total_bytes=total_bytes)
            
            # Calculate delta since last update
            if current_bytes > self._last_parsed_bytes:
                delta = current_bytes - self._last_parsed_bytes
                self._last_parsed_bytes = current_bytes
                self.progress_tracker.update_bytes(delta)
        
        # Alternative: Extract percentage (e.g., "100%" or "50%")
        elif '%' in text and self._total_bytes > 0:
            pct_match = re.search(r'(\d+)%', text)
            if pct_match:
                pct = int(pct_match.group(1))
                current_bytes = int((pct / 100.0) * self._total_bytes)
                if current_bytes > self._last_parsed_bytes:
                    delta = current_bytes - self._last_parsed_bytes
                    self._last_parsed_bytes = current_bytes
                    self.progress_tracker.update_bytes(delta)
    
    def write(self, text: str) -> int:
        """Write text to the capture, forwarding to log function immediately."""
        if not text:
            return 0
        
        # Parse HuggingFace progress information
        self._parse_hf_progress(text)
        
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
                    # Parse progress from this line too
                    self._parse_hf_progress(clean)
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
