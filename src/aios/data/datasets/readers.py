"""Basic text file reading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List
import os
import time
import signal
from contextlib import contextmanager

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# Default timeout for file reading operations (seconds)
# Can be overridden via AIOS_DATASET_LOAD_TIMEOUT environment variable
DEFAULT_READ_TIMEOUT = int(os.environ.get("AIOS_DATASET_LOAD_TIMEOUT", "300"))  # 5 minutes default


class TimeoutError(Exception):
    """Raised when file reading operation times out."""
    pass


@contextmanager
def timeout_context(seconds: int, error_message: str = "Operation timed out"):
    """Context manager for timing out operations.
    
    Note: Only works on Unix systems with signal support.
    On Windows, timeout is checked manually during iteration.
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(error_message)
    
    # Check if signal is available (Unix systems)
    has_signal = hasattr(signal, 'SIGALRM')
    
    if has_signal:
        # Set the signal handler and alarm
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
    
    try:
        yield
    finally:
        if has_signal:
            # Disable the alarm and restore old handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def read_text_lines_sample(path: str | Path, max_lines: int = 1000, timeout: int = DEFAULT_READ_TIMEOUT) -> List[str]:
    """Read up to max_lines of UTF-8 text lines from a dataset file.

    Designed for quick sampling to seed small replay buffers.
    Shows progress bar for files larger than 10MB.
    
    Args:
        path: Path to the text file
        max_lines: Maximum number of lines to read
        timeout: Maximum time in seconds before timing out (default: 300)
        
    Returns:
        List of text lines (up to max_lines)
        
    Raises:
        TimeoutError: If reading takes longer than timeout seconds
    """
    p = Path(path)
    out: List[str] = []
    
    # Track start time for manual timeout on Windows
    start_time = time.time()
    has_signal = hasattr(signal, 'SIGALRM')
    
    try:
        # Get file size for progress bar and size checks
        file_size = p.stat().st_size
        show_progress = TQDM_AVAILABLE and file_size > 10 * 1024 * 1024  # Show for files >10MB
        
        # Warning for extremely large files
        if file_size > 5 * 1024 * 1024 * 1024:  # >5GB
            print(f"Warning: Large file detected ({file_size / (1024**3):.2f} GB). Consider splitting or using streaming.")
        
        # Use timeout context (Unix) or manual checks (Windows)
        timeout_error_msg = f"Dataset loading timed out after {timeout} seconds. File: {p.name} ({file_size / (1024**2):.2f} MB)"
        
        if has_signal:
            # Unix: Use signal-based timeout
            with timeout_context(timeout, timeout_error_msg):
                out = _read_lines_with_progress(p, max_lines, file_size, show_progress, None)
        else:
            # Windows: Manual timeout checking during iteration
            out = _read_lines_with_progress(p, max_lines, file_size, show_progress, start_time, timeout)
                
    except TimeoutError as e:
        # Provide helpful error message with workarounds
        print(f"\n{'='*60}")
        print(f"ERROR: {str(e)}")
        print(f"{'='*60}")
        print("Workarounds:")
        print("  1. Split large file into smaller chunks")
        print("  2. Use plain text format (not CSV/JSON)")
        print("  3. Increase timeout: export AIOS_DATASET_LOAD_TIMEOUT=600")
        print("  4. Use HuggingFace streaming: hf://dataset_name")
        print(f"{'='*60}\n")
        raise
    except UnicodeDecodeError as e:
        # Handle encoding issues gracefully
        print(f"Warning: Encoding error in {p.name}: {e}")
        print("Trying with 'latin-1' encoding fallback...")
        try:
            # Retry with latin-1 encoding
            with p.open("r", encoding="latin-1", errors="replace") as f:
                for i, ln in enumerate(f):
                    out.append(ln.strip())
                    if (i + 1) >= max_lines:
                        break
        except Exception:
            return []
    except Exception as e:
        print(f"Warning: Failed to read {p.name}: {e}")
        return []
    
    return out


def _read_lines_with_progress(
    p: Path,
    max_lines: int,
    file_size: int,
    show_progress: bool,
    start_time: float = None,
    timeout: int = None
) -> List[str]:
    """Helper function to read lines with optional progress bar and timeout checking.
    
    Args:
        p: Path to file
        max_lines: Maximum lines to read
        file_size: Size of file in bytes
        show_progress: Whether to show progress bar
        start_time: Start time for manual timeout checking (Windows)
        timeout: Timeout in seconds (Windows)
        
    Returns:
        List of lines
        
    Raises:
        TimeoutError: If manual timeout is exceeded (Windows only)
    """
    out: List[str] = []
    
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        if show_progress:
            # Estimate total lines for progress (rough estimate: 100 bytes per line)
            estimated_lines = min(max_lines, file_size // 100)
            with tqdm(total=estimated_lines, desc=f"Loading {p.name}", unit="lines") as pbar:
                for i, ln in enumerate(f):
                    # Manual timeout check for Windows
                    if start_time and timeout and (time.time() - start_time) > timeout:
                        raise TimeoutError(
                            f"Dataset loading timed out after {timeout} seconds. "
                            f"File: {p.name} ({file_size / (1024**2):.2f} MB), "
                            f"Lines read: {len(out)}"
                        )
                    
                    out.append(ln.strip())
                    if i % 100 == 0:  # Update progress every 100 lines to avoid overhead
                        pbar.update(min(100, estimated_lines - pbar.n))
                    if (i + 1) >= max_lines:
                        break
        else:
            # No progress bar for small files
            for i, ln in enumerate(f):
                # Manual timeout check for Windows
                if start_time and timeout and (time.time() - start_time) > timeout:
                    raise TimeoutError(
                        f"Dataset loading timed out after {timeout} seconds. "
                        f"File: {p.name} ({file_size / (1024**2):.2f} MB), "
                        f"Lines read: {len(out)}"
                    )
                
                out.append(ln.strip())
                if (i + 1) >= max_lines:
                    break
    
    return out
