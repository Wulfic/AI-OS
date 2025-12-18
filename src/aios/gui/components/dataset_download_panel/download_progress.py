"""
Download Progress Tracker

Tracks download progress including percentage, speed, and remaining time.
Supports both bits and bytes speed display with automatic unit scaling.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import Enum


class SpeedUnit(Enum):
    """Speed display unit preference."""
    BITS = "bits"    # Megabits, Gigabits, etc.
    BYTES = "bytes"  # Megabytes, Gigabytes, etc.


@dataclass
class DownloadStats:
    """Current download statistics."""
    bytes_downloaded: int = 0
    total_bytes: int = 0
    samples_downloaded: int = 0
    total_samples: int = 0
    blocks_completed: float = 0.0  # Can be fractional for smooth progress
    total_blocks: int = 0
    speed_bytes_per_sec: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0
    is_complete: bool = False
    is_cancelled: bool = False


class DownloadProgressTracker:
    """
    Tracks download progress with speed calculation and ETA.
    
    Features:
    - Smooth speed averaging over configurable window
    - Automatic unit scaling (KB/s to TB/s)
    - ETA calculation based on average speed
    - Thread-safe updates
    """
    
    def __init__(
        self,
        total_bytes: int = 0,
        total_samples: int = 0,
        total_blocks: int = 0,
        speed_window_seconds: float = 3.0,
        update_callback: Optional[Callable[[DownloadStats], None]] = None,
    ):
        """
        Initialize progress tracker.
        
        Args:
            total_bytes: Expected total download size (0 if unknown)
            total_samples: Expected total samples (0 if unknown)
            total_blocks: Expected total blocks (0 if unknown)
            speed_window_seconds: Time window for speed averaging
            update_callback: Called when progress is updated
        """
        self.total_bytes = total_bytes
        self.total_samples = total_samples
        self.total_blocks = total_blocks
        self.speed_window = speed_window_seconds
        self.update_callback = update_callback
        
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None
        self._bytes_downloaded = 0
        self._samples_downloaded = 0
        self._blocks_completed = 0.0  # Can be fractional for smooth progress
        self._is_complete = False
        self._is_cancelled = False
        
        # Speed calculation with sliding window
        self._speed_samples: list[tuple[float, int]] = []  # (timestamp, cumulative_bytes)
        self._last_speed: float = 0.0
    
    def start(self) -> None:
        """Start tracking (call when download begins)."""
        with self._lock:
            self._start_time = time.monotonic()
            self._speed_samples = [(self._start_time, 0)]
    
    def update_bytes(self, bytes_delta: int) -> DownloadStats:
        """Update with bytes downloaded since last update."""
        with self._lock:
            self._bytes_downloaded += bytes_delta
            return self._calculate_stats()
    
    def update_samples(self, samples_delta: int, bytes_per_sample: int = 500) -> DownloadStats:
        """Update with samples downloaded (estimates bytes from sample count)."""
        with self._lock:
            self._samples_downloaded += samples_delta
            self._bytes_downloaded += samples_delta * bytes_per_sample
            return self._calculate_stats()
    
    def set_progress(self, bytes_downloaded: int = 0, samples_downloaded: int = 0, blocks_completed: float = 0.0) -> DownloadStats:
        """Set absolute progress values. blocks_completed can be fractional for smooth progress."""
        with self._lock:
            self._bytes_downloaded = bytes_downloaded
            self._samples_downloaded = samples_downloaded
            self._blocks_completed = blocks_completed
            return self._calculate_stats()
    
    def set_totals(self, total_bytes: int = 0, total_samples: int = 0, total_blocks: int = 0) -> None:
        """Update expected totals (can be called mid-download)."""
        with self._lock:
            if total_bytes > 0:
                self.total_bytes = total_bytes
            if total_samples > 0:
                self.total_samples = total_samples
            if total_blocks > 0:
                self.total_blocks = total_blocks
    
    def complete(self) -> DownloadStats:
        """Mark download as complete."""
        with self._lock:
            self._is_complete = True
            return self._calculate_stats()
    
    def cancel(self) -> DownloadStats:
        """Mark download as cancelled."""
        with self._lock:
            self._is_cancelled = True
            return self._calculate_stats()
    
    def get_stats(self) -> DownloadStats:
        """Get current download statistics."""
        with self._lock:
            return self._calculate_stats()
    
    def _calculate_stats(self) -> DownloadStats:
        """Calculate current stats (must be called with lock held)."""
        now = time.monotonic()
        
        # Calculate elapsed time
        elapsed = 0.0
        if self._start_time:
            elapsed = now - self._start_time
        
        # Update speed samples
        self._speed_samples.append((now, self._bytes_downloaded))
        
        # Remove samples outside the window
        cutoff = now - self.speed_window
        self._speed_samples = [(t, b) for t, b in self._speed_samples if t >= cutoff]
        
        # Calculate speed from window
        speed = 0.0
        if len(self._speed_samples) >= 2:
            oldest = self._speed_samples[0]
            newest = self._speed_samples[-1]
            time_diff = newest[0] - oldest[0]
            bytes_diff = newest[1] - oldest[1]
            if time_diff > 0:
                speed = bytes_diff / time_diff
                self._last_speed = speed
        else:
            speed = self._last_speed
        
        # Calculate ETA
        eta = 0.0
        if speed > 0 and self.total_bytes > 0:
            remaining_bytes = self.total_bytes - self._bytes_downloaded
            if remaining_bytes > 0:
                eta = remaining_bytes / speed
        
        stats = DownloadStats(
            bytes_downloaded=self._bytes_downloaded,
            total_bytes=self.total_bytes,
            samples_downloaded=self._samples_downloaded,
            total_samples=self.total_samples,
            blocks_completed=self._blocks_completed,
            total_blocks=self.total_blocks,
            speed_bytes_per_sec=speed,
            elapsed_seconds=elapsed,
            eta_seconds=eta,
            is_complete=self._is_complete,
            is_cancelled=self._is_cancelled,
        )
        
        # Notify callback
        if self.update_callback:
            try:
                self.update_callback(stats)
            except Exception:
                pass
        
        return stats


def format_speed(bytes_per_sec: float, unit: SpeedUnit = SpeedUnit.BYTES) -> str:
    """
    Format speed with appropriate unit scaling.
    
    Supports automatic scaling from bits/bytes up to terabits/terabytes.
    
    Args:
        bytes_per_sec: Speed in bytes per second
        unit: Whether to display as bits or bytes
        
    Returns:
        Formatted speed string (e.g., "125.5 MB/s" or "1.00 Gbps")
    """
    if unit == SpeedUnit.BITS:
        # Convert to bits per second
        bits_per_sec = bytes_per_sec * 8
        
        if bits_per_sec >= 1e12:  # Terabits
            return f"{bits_per_sec / 1e12:.2f} Tbps"
        elif bits_per_sec >= 1e9:  # Gigabits
            return f"{bits_per_sec / 1e9:.2f} Gbps"
        elif bits_per_sec >= 1e6:  # Megabits
            return f"{bits_per_sec / 1e6:.2f} Mbps"
        elif bits_per_sec >= 1e3:  # Kilobits
            return f"{bits_per_sec / 1e3:.2f} Kbps"
        else:
            return f"{bits_per_sec:.0f} bps"
    else:
        # Display as bytes per second
        if bytes_per_sec >= 1e12:  # Terabytes
            return f"{bytes_per_sec / 1e12:.2f} TB/s"
        elif bytes_per_sec >= 1e9:  # Gigabytes
            return f"{bytes_per_sec / 1e9:.2f} GB/s"
        elif bytes_per_sec >= 1e6:  # Megabytes
            return f"{bytes_per_sec / 1e6:.2f} MB/s"
        elif bytes_per_sec >= 1e3:  # Kilobytes
            return f"{bytes_per_sec / 1e3:.2f} KB/s"
        else:
            return f"{bytes_per_sec:.0f} B/s"


def format_eta(seconds: float) -> str:
    """Format ETA in human-readable format."""
    if seconds <= 0:
        return "calculating..."
    elif seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_size(bytes_count: int) -> str:
    """Format size in human-readable format."""
    if bytes_count >= 1e12:
        return f"{bytes_count / 1e12:.2f} TB"
    elif bytes_count >= 1e9:
        return f"{bytes_count / 1e9:.2f} GB"
    elif bytes_count >= 1e6:
        return f"{bytes_count / 1e6:.2f} MB"
    elif bytes_count >= 1e3:
        return f"{bytes_count / 1e3:.2f} KB"
    else:
        return f"{bytes_count} B"


def format_progress_display(
    stats: DownloadStats,
    speed_unit: SpeedUnit = SpeedUnit.BYTES,
) -> str:
    """
    Format complete progress display string.
    
    Args:
        stats: Current download statistics
        speed_unit: Unit preference for speed display
        
    Returns:
        Formatted string like "45.2% | 125.5 MB/s | ETA: 2m 30s"
    """
    parts = []
    
    # Percentage (prefer blocks for accuracy)
    if stats.total_blocks > 0 and stats.blocks_completed > 0:
        pct = (stats.blocks_completed / stats.total_blocks) * 100
        parts.append(f"{pct:.1f}%")
    elif stats.total_bytes > 0:
        pct = (stats.bytes_downloaded / stats.total_bytes) * 100
        parts.append(f"{pct:.1f}%")
    elif stats.total_samples > 0 and stats.samples_downloaded > 0:
        pct = (stats.samples_downloaded / stats.total_samples) * 100
        parts.append(f"{pct:.1f}%")
    else:
        parts.append(format_size(stats.bytes_downloaded))
    
    # Speed
    if stats.speed_bytes_per_sec > 0:
        parts.append(format_speed(stats.speed_bytes_per_sec, speed_unit))
    
    # ETA
    if stats.eta_seconds > 0:
        parts.append(f"ETA: {format_eta(stats.eta_seconds)}")
    
    return " | ".join(parts)
