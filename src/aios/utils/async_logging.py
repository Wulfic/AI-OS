"""Async logging utilities for high-performance non-blocking logging.

This module provides optimized logging handlers for debug mode that prevent
performance degradation from heavy logging:

1. QueueHandler - Non-blocking handler that puts log records in a queue
2. QueueListener - Background thread that processes the queue asynchronously
3. MemoryHandler - Buffers log records in memory and flushes in batches
4. Optimized formatters - Lighter weight formatting for debug mode
5. LazyLogString - Deferred string formatting to avoid overhead when logging is disabled

Usage:
    from aios.utils.async_logging import setup_async_logging, shutdown_async_logging, LazyLogString
    
    # During initialization
    listener = setup_async_logging(handlers)
    
    # For expensive string formatting, use lazy evaluation:
    logger.debug(LazyLogString(lambda: f"Expensive {expensive_computation()}"))
    
    # During shutdown
    shutdown_async_logging(listener)
"""

from __future__ import annotations

import logging
import logging.handlers
from queue import Queue
from typing import Optional, List, Callable, Any
import atexit
import glob
import json
import os
import re
import time
import signal
import sys
import threading
from pathlib import Path

try:
    from aios.system import paths as system_paths
except Exception:  # pragma: no cover - fallback for bootstrap
    system_paths = None

# Import platform-specific file locking
if os.name == 'nt':
    import msvcrt  # Windows file locking
else:
    import fcntl  # Unix/Linux file locking


# Global session tracking - ensures all handlers in the same program run use the same session
_SESSION_CACHE = {}  # {base_filename: session_number}

# Global session number for the entire run - ALL log files use this
_GLOBAL_SESSION_NUMBER = None

# Threading lock to protect global session number assignment in multi-threaded environments
_SESSION_LOCK = threading.Lock()

# Track the main process ID to know when we're in a new run
_MAIN_PROCESS_ID = os.getpid()

# Session file to coordinate across all threads in the SAME PROGRAM RUN
# We use a timestamp-based unique ID to ensure each program execution gets its own lock file
_SESSION_RUN_ID = None  # Set once per program execution

# Track the lock file for THIS specific program run
_CURRENT_LOCK_FILE = None

# Track if we're in the process of shutting down (to prevent recreating sessions)
_SHUTTING_DOWN = False

# Track if cleanup has been registered
_CLEANUP_REGISTERED = False

# Track how many times session number has been set (for debugging)
_SESSION_SET_COUNT = 0


_SESSION_ENV_VAR = "AIOS_SESSION_NUMBER"
_SESSION_RUN_ENV_VAR = "AIOS_SESSION_RUN_ID"
_SESSION_DATE_ENV_VAR = "AIOS_SESSION_DATE"


def _resolve_logs_dir() -> str:
    if system_paths is not None:
        try:
            return str(system_paths.get_logs_dir())
        except Exception:
            logging.getLogger(__name__).debug("Failed to resolve logs dir via helper", exc_info=True)
    return os.path.join(os.getcwd(), "logs")


LOG_DIR = _resolve_logs_dir()
os.makedirs(LOG_DIR, exist_ok=True)

if system_paths is not None:
    try:
        _SESSION_STATE_FILE = str(system_paths.get_session_state_file())
        _LOCK_ROOT = Path(system_paths.get_session_lock_dir())
    except Exception:
        logging.getLogger(__name__).debug("Failed to resolve session files via helper", exc_info=True)
        _SESSION_STATE_FILE = os.path.join(LOG_DIR, ".session_state.json")
        _LOCK_ROOT = Path(LOG_DIR)
else:
    _SESSION_STATE_FILE = os.path.join(LOG_DIR, ".session_state.json")
    _LOCK_ROOT = Path(LOG_DIR)


def _cleanup_old_lock_files_at_import() -> None:
    """Clean up stale session artifacts before the first session initializes."""
    try:
        # If session metadata already exists, another process has initialized
        # logging for this run. Skip cleanup so we don't remove active artifacts.
        if os.environ.get(_SESSION_RUN_ENV_VAR):
            return

        lock_pattern = str(_LOCK_ROOT / ".session_*.lock")
        lock_files = glob.glob(lock_pattern)
        for lock_file in lock_files:
            try:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
            except (IOError, OSError):
                pass  # Ignore errors during cleanup

        if os.path.exists(_SESSION_STATE_FILE):
            try:
                os.remove(_SESSION_STATE_FILE)
            except (IOError, OSError):
                pass
    except Exception:
        pass  # Don't crash if cleanup fails


# CRITICAL: Clean up old lock files at module import time, before anything else
_cleanup_old_lock_files_at_import()


def _lock_file(file_handle):
    """Lock a file (cross-platform)."""
    if os.name == 'nt':  # Windows
        msvcrt.locking(file_handle.fileno(), msvcrt.LK_LOCK, 1)
    else:  # Unix/Linux
        fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(file_handle):
    """Unlock a file (cross-platform)."""
    if os.name == 'nt':  # Windows
        msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:  # Unix/Linux
        fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)


def _get_session_lock_file(date_str: str, run_id: str = None) -> str:
    """Get the session lock file path for a given date and run ID.
    
    Each program execution gets its own lock file with a unique run ID.
    This allows all threads in the same program to share the same session number
    while keeping different program executions separate.
    """
    global _SESSION_RUN_ID
    
    if run_id is None:
        # Generate run ID once per program execution (thread-safe)
        if _SESSION_RUN_ID is None:
            import uuid
            import sys
            _SESSION_RUN_ID = str(uuid.uuid4())[:8]
            print(f"[LOCK FILE] Generated new RUN ID: {_SESSION_RUN_ID} (PID: {os.getpid()})", file=sys.stderr)
        run_id = _SESSION_RUN_ID
    
    lock_path = _LOCK_ROOT / f".session_{date_str}_{run_id}.lock"
    import sys
    print(f"[LOCK FILE] Using lock file: {lock_path} (PID: {os.getpid()})", file=sys.stderr)
    return str(lock_path)


def _read_session_from_lock(lock_file: str) -> Optional[int]:
    """Read session number from lock file with proper locking."""
    try:
        if os.path.exists(lock_file):
            with open(lock_file, 'r+') as f:
                try:
                    _lock_file(f)
                except (IOError, OSError):
                    # File might be locked, try reading without lock
                    pass
                    
                try:
                    content = f.read().strip()
                    if content:
                        return int(content)
                finally:
                    try:
                        _unlock_file(f)
                    except (IOError, OSError):
                        pass  # Ignore unlock errors
    except (ValueError, IOError, OSError):
        pass
    return None


def _write_session_to_lock(lock_file: str, session_num: int) -> None:
    """Write session number to lock file with proper locking.
    
    The lock file persists for the entire program lifetime (across all threads)
    and is only cleaned up on program termination. This ensures all threads
    in the same program run share the same session number.
    """
    global _CURRENT_LOCK_FILE
    
    try:
        os.makedirs(os.path.dirname(lock_file), exist_ok=True)
        
        # Register cleanup handlers once (if not already done)
        # Note: _CURRENT_LOCK_FILE is already set in get_next_session_number()
        if not _CLEANUP_REGISTERED:
            _register_cleanup_handlers()
        
        # Open in r+ mode if exists, w+ if new
        mode = 'r+' if os.path.exists(lock_file) else 'w+'
        with open(lock_file, mode) as f:
            try:
                _lock_file(f)
            except (IOError, OSError):
                # File might be locked by another process, skip write
                return
                
            try:
                # Read current value to check if another process beat us to it
                if mode == 'r+':
                    f.seek(0)
                    content = f.read().strip()
                    if content:
                        # Another process already wrote, use that value
                        # Don't return - we still want to make file hidden
                        pass
                    else:
                        # File exists but is empty, write our session number
                        f.seek(0)
                        f.truncate()
                        f.write(str(session_num))
                        f.flush()
                else:
                    # New file, write our session number
                    f.write(str(session_num))
                    f.flush()
            finally:
                try:
                    _unlock_file(f)
                except (IOError, OSError):
                    pass  # Ignore unlock errors
        
        # Make the file hidden on Windows
        if os.name == 'nt' and os.path.exists(lock_file):
            try:
                import ctypes
                FILE_ATTRIBUTE_HIDDEN = 0x02
                ctypes.windll.kernel32.SetFileAttributesW(lock_file, FILE_ATTRIBUTE_HIDDEN)
            except Exception:
                pass  # Ignore errors setting hidden attribute
            
    except (IOError, OSError) as e:
        import sys
        print(f"Warning: Could not write session lock file: {e}", file=sys.stderr)


def _register_cleanup_handlers() -> None:
    """Register cleanup handlers for atexit and signal handling."""
    global _CLEANUP_REGISTERED
    
    if _CLEANUP_REGISTERED:
        return
    
    # Register atexit cleanup - this is called when the process exits normally
    atexit.register(_cleanup_on_exit)
    
    # Register signal handlers for graceful cleanup on CTRL+C, etc.
    def _signal_handler(signum, frame):
        """Handle signals by cleaning up and exiting."""
        _cleanup_on_exit()
        sys.exit(0)
    
    try:
        # Register common termination signals
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, _signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, _signal_handler)
        # On Windows, also handle CTRL_BREAK
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, _signal_handler)
    except (ValueError, OSError):
        # Ignore errors - might not be able to register in all contexts
        pass
    
    _CLEANUP_REGISTERED = True


def get_next_session_number(base_filename: str) -> int:
    """Get the next session number by finding the highest existing session number for today.
    
    Session numbers reset each day. ALL log files in the same run use the same session number.
    
    This function coordinates session numbers across all threads in the SAME PROGRAM RUN
    using a lock file with a unique run ID. Each new program execution gets a new session number.
    
    Thread-safe: Uses _SESSION_LOCK to prevent race conditions in multi-threaded environments.
    
    Args:
        base_filename: Base log file path (e.g., "logs/aios.log") - used only for scanning
        
    Returns:
        Session number for today (same for all log files in this run)
    """
    global _GLOBAL_SESSION_NUMBER, _SESSION_RUN_ID, _CURRENT_LOCK_FILE, _SESSION_SET_COUNT

    if _SHUTTING_DOWN:
        if _GLOBAL_SESSION_NUMBER is not None:
            return _GLOBAL_SESSION_NUMBER
        return 1

    today = time.strftime("%d-%m-%Y")

    if _GLOBAL_SESSION_NUMBER is not None:
        return _GLOBAL_SESSION_NUMBER

    # Fast path: reuse session metadata propagated via environment variables.
    env_session = os.environ.get(_SESSION_ENV_VAR)
    env_run_id = os.environ.get(_SESSION_RUN_ENV_VAR)
    env_date = os.environ.get(_SESSION_DATE_ENV_VAR)
    if env_session and env_run_id and env_date == today:
        try:
            env_session_num = int(env_session)
        except ValueError:
            env_session_num = None
        if env_session_num is not None:
            _SESSION_RUN_ID = env_run_id
            lock_file = _get_session_lock_file(today, env_run_id)
            _CURRENT_LOCK_FILE = lock_file
            _GLOBAL_SESSION_NUMBER = env_session_num
            _SESSION_CACHE[base_filename] = env_session_num
            return env_session_num

    lock_file = None
    session_num = 1
    run_id = None

    with _SESSION_LOCK:
        if _GLOBAL_SESSION_NUMBER is not None:
            _SESSION_CACHE[base_filename] = _GLOBAL_SESSION_NUMBER
            return _GLOBAL_SESSION_NUMBER

        try:
            os.makedirs(os.path.dirname(_SESSION_STATE_FILE), exist_ok=True)

            state_data: dict[str, Any] = {}
            new_session_created = False

            state_file_handle = None
            try:
                state_file_handle = open(_SESSION_STATE_FILE, "a+", encoding="utf-8")
                state_file_handle.seek(0)
                _lock_file(state_file_handle)
                state_file_handle.seek(0)
                raw_state = state_file_handle.read().strip()
                if raw_state:
                    try:
                        state_data = json.loads(raw_state)
                    except json.JSONDecodeError:
                        state_data = {}

                stored_date = state_data.get("date")
                stored_session = state_data.get("session")
                stored_run_id = state_data.get("run_id")

                if stored_date == today and stored_session is not None and stored_run_id:
                    try:
                        session_num = int(stored_session)
                        run_id = str(stored_run_id)
                    except (TypeError, ValueError):
                        session_num = None
                        run_id = None

                if run_id is None:
                    session_num = 1
                    log_dir = LOG_DIR
                    if os.path.exists(log_dir):
                        pattern = os.path.join(log_dir, f"*_{today}_*.log")
                        existing_files = glob.glob(pattern)
                        if existing_files:
                            session_pattern = re.compile(r'_\d{2}-\d{2}-\d{4}_(\d{3})_\d{3}\.log$')
                            session_numbers = []
                            for filepath in existing_files:
                                match = session_pattern.search(filepath)
                                if match:
                                    session_num_found = int(match.group(1))
                                    session_numbers.append(session_num_found)
                            if session_numbers:
                                session_num = max(session_numbers) + 1

                    import uuid
                    run_id = str(uuid.uuid4())[:8]
                    new_session_created = True

                    if state_file_handle is not None:
                        state_file_handle.seek(0)
                        state_file_handle.truncate()
                        json.dump({"date": today, "session": session_num, "run_id": run_id}, state_file_handle)
                        state_file_handle.flush()
            finally:
                if state_file_handle is not None:
                    try:
                        _unlock_file(state_file_handle)
                    except (IOError, OSError):
                        pass
                    state_file_handle.close()

            if run_id is None:
                session_num = session_num or 1
                import uuid
                run_id = str(uuid.uuid4())[:8]
                new_session_created = True

            _SESSION_RUN_ID = run_id
            lock_file = _get_session_lock_file(today, run_id)
            _CURRENT_LOCK_FILE = lock_file

            _SESSION_SET_COUNT += 1
            _GLOBAL_SESSION_NUMBER = session_num
            _SESSION_CACHE[base_filename] = session_num

            import sys
            print(
                f"[SESSION #{_SESSION_SET_COUNT}] Set to {session_num} for {base_filename} (total files in cache: {len(_SESSION_CACHE)})",
                file=sys.stderr,
            )

            if new_session_created or not os.path.exists(lock_file):
                try:
                    _write_session_to_lock(lock_file, session_num)
                except Exception as write_error:
                    print(f"[SESSION] Warning: Failed to write lock file: {write_error}", file=sys.stderr)

        except Exception as e:
            import sys
            print(f"Warning: Could not determine session number: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            if _GLOBAL_SESSION_NUMBER is not None:
                session_num = _GLOBAL_SESSION_NUMBER
            else:
                session_num = 1
                _GLOBAL_SESSION_NUMBER = session_num
            _SESSION_CACHE[base_filename] = session_num
            return session_num

    os.environ[_SESSION_ENV_VAR] = str(session_num)
    os.environ[_SESSION_RUN_ENV_VAR] = run_id
    os.environ[_SESSION_DATE_ENV_VAR] = today

    return session_num


def cleanup_old_sessions(base_filename: str, max_sessions: int = 10) -> None:
    """Clean up old log sessions, keeping only the most recent sessions.
    
    Groups log files by session (all parts of a session count as one session)
    and deletes entire old sessions when the limit is exceeded.
    
    Args:
        base_filename: Base log file path (e.g., "logs/aios.log")
        max_sessions: Maximum number of sessions to keep (default: 10)
    """
    try:
        from collections import defaultdict
        
        # Extract the base filename without path and extension
        base_dir = os.path.dirname(base_filename)
        base_name = os.path.basename(base_filename)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Pattern to match all log files: filename_DD-MM-YYYY_SSS_PPP.log
        pattern = os.path.join(base_dir, f"{name_without_ext}_*.log") if base_dir else f"{name_without_ext}_*.log"
        existing_files = glob.glob(pattern)
        
        if not existing_files:
            return
        
        # Group files by session (date + session number)
        # Key: (date, session_number), Value: list of file paths
        sessions = defaultdict(list)
        
        # Pattern: filename_DD-MM-YYYY_SSS_PPP.log
        file_pattern = re.compile(
            rf'{re.escape(name_without_ext)}_(\d{{2}}-\d{{2}}-\d{{4}})_(\d{{3}})_(\d{{3}})\.log$'
        )
        
        for filepath in existing_files:
            match = file_pattern.search(filepath)
            if match:
                date_str = match.group(1)
                session_num = int(match.group(2))
                # Group by (date, session)
                session_key = (date_str, session_num)
                sessions[session_key].append(filepath)
        
        # If we have more sessions than the limit, delete the oldest ones
        if len(sessions) > max_sessions:
            # Sort sessions by the newest file in each session
            # This ensures we keep the most recently modified sessions
            session_list = []
            for session_key, files in sessions.items():
                # Get the newest modification time from all parts in this session
                newest_time = max(os.path.getmtime(f) for f in files)
                session_list.append((newest_time, session_key, files))
            
            # Sort by modification time (newest first)
            session_list.sort(reverse=True)
            
            # Keep only the most recent max_sessions sessions
            sessions_to_delete = session_list[max_sessions:]
            
            # Delete all files from old sessions
            for _, session_key, files in sessions_to_delete:
                for filepath in files:
                    try:
                        os.remove(filepath)
                        print(f"Deleted old log file: {os.path.basename(filepath)}")
                    except Exception as e:
                        import sys
                        print(f"Warning: Could not delete {filepath}: {e}", file=sys.stderr)
        
    except Exception as e:
        # Don't crash if cleanup fails
        import sys
        print(f"Warning: Error during session cleanup: {e}", file=sys.stderr)


class LazyLogString:
    """Deferred string formatting for log messages.
    
    Only evaluates the format function when the string is actually needed,
    avoiding overhead when the log level filters out the message.
    
    Example:
        logger.debug(LazyLogString(lambda: f"Expensive {call()}"))
    """
    
    __slots__ = ('_func',)
    
    def __init__(self, func: Callable[[], str]):
        """Initialize lazy log string.
        
        Args:
            func: Function that returns the log message string
        """
        self._func = func
    
    def __str__(self) -> str:
        """Evaluate the format function and return the string."""
        try:
            return self._func()
        except Exception as e:
            return f"<LazyLogString error: {e}>"


# ---------------------------------------------------------------------------
# Logging filters
# ---------------------------------------------------------------------------


class DebugAndTraceFilter(logging.Filter):
    """Allow DEBUG records and any record that carries exception/stack info."""

    def __init__(self, name: str = "", level: int | str = logging.DEBUG):
        super().__init__(name)
        if isinstance(level, str):
            self.level = logging.getLevelName(level.upper())
            if not isinstance(self.level, int):
                self.level = logging.DEBUG
        else:
            self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        has_trace = (
            bool(record.exc_info)
            or bool(getattr(record, "stack_info", None))
            or bool(getattr(record, "exc_text", None))
        )
        return record.levelno == self.level or has_trace


# Global registry to track queue listeners for cleanup
_QUEUE_LISTENERS: List[logging.handlers.QueueListener] = []
_ROOT_HANDLERS_BEFORE_QUEUE: List[logging.Handler] = []
_AIOS_HANDLERS_BEFORE_QUEUE: List[logging.Handler] = []


class AsyncMemoryHandler(logging.handlers.MemoryHandler):
    """Memory handler optimized for async logging with time-based and size-based flushing."""
    
    def __init__(
        self,
        capacity: int = 10000,
        flushLevel: int = logging.ERROR,
        target: Optional[logging.Handler] = None,
        flushOnClose: bool = True,
        flushInterval: float = 5.0  # Flush every 5 seconds
    ):
        """Initialize async memory handler.
        
        Args:
            capacity: Number of records to buffer before flushing (default: 10000)
            flushLevel: Level at which to auto-flush (default: ERROR)
            target: Target handler to flush to
            flushOnClose: Whether to flush on close
            flushInterval: Time interval in seconds between automatic flushes (default: 5.0)
        """
        super().__init__(capacity, flushLevel, target, flushOnClose)
        self.buffer_size = capacity
        self.flushInterval = flushInterval
        import time
        self.last_flush_time = time.time()
    
    def shouldFlush(self, record: logging.LogRecord) -> bool:
        """Determine if buffer should be flushed.
        
        Flushes when:
        - Buffer is full
        - Record level >= flushLevel (WARNING by default)
        - Time since last flush exceeds flushInterval
        """
        import time
        current_time = time.time()
        time_based_flush = (current_time - self.last_flush_time) >= self.flushInterval
        
        should_flush = (
            (len(self.buffer) >= self.capacity) or 
            (record.levelno >= self.flushLevel) or
            time_based_flush
        )
        
        if should_flush:
            self.last_flush_time = current_time
        
        return should_flush


class FastDebugFormatter(logging.Formatter):
    """Lightweight formatter optimized for debug logging.
    
    Skips expensive operations like time formatting and stack traces
    for better performance in debug mode.
    """
    
    def __init__(self, simple_mode: bool = False, ultra_simple: bool = False):
        """Initialize fast debug formatter.
        
        Args:
            simple_mode: If True, use minimal formatting (just message)
            ultra_simple: If True, use ultra-minimal formatting (levelname + message only)
        """
        self.ultra_simple = ultra_simple
        if ultra_simple:
            fmt = "%(levelname)s: %(message)s"
        elif simple_mode:
            fmt = "%(levelname)s | %(name)s | %(message)s"
        else:
            fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        super().__init__(fmt)
        self.simple_mode = simple_mode or ultra_simple
    
    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        """Fast time formatting using default format."""
        if self.simple_mode or self.ultra_simple:
            return ""
        # Use simple time format to avoid expensive strftime calls
        return self.default_time_format % (record.created,)
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with minimal overhead."""
        # Handle LazyLogString
        if hasattr(record.msg, '_func'):
            # It's a LazyLogString, evaluate it now
            record.msg = str(record.msg)
        return super().format(record)
    
    default_time_format = "%.3f"


class NonBlockingFormatter(logging.Formatter):
    """Compatibility formatter that mirrors logging.Formatter for config references."""

    def format(self, record: logging.LogRecord) -> str:
        # Ensure LazyLogString instances get evaluated.
        if hasattr(record.msg, "_func"):
            record.msg = str(record.msg)
        return super().format(record)


def create_queue_handler() -> tuple[logging.Handler, Queue]:
    """Create a QueueHandler with its queue.
    
    Returns:
        Tuple of (handler, queue)
    """
    log_queue: Queue = Queue(maxsize=-1)  # Unlimited queue size
    queue_handler = logging.handlers.QueueHandler(log_queue)
    return queue_handler, log_queue


def create_queue_listener(
    queue: Queue,
    *handlers: logging.Handler,
    respect_handler_level: bool = True
) -> logging.handlers.QueueListener:
    """Create a QueueListener to process log records asynchronously.
    
    Args:
        queue: The queue to listen to
        *handlers: Target handlers to send records to
        respect_handler_level: Whether to respect individual handler levels
    
    Returns:
        QueueListener instance
    """
    listener = logging.handlers.QueueListener(
        queue,
        *handlers,
        respect_handler_level=respect_handler_level
    )
    
    # Register for cleanup
    _QUEUE_LISTENERS.append(listener)
    
    return listener


def setup_async_logging(
    base_handlers: List[logging.Handler],
    use_memory_buffer: bool = True,
    memory_capacity: int = 5000,
    debug_mode: bool = False
) -> Optional[logging.handlers.QueueListener]:
    """Set up async logging with queue and optional memory buffering.
    
    This replaces synchronous file handlers with async handlers that:
    1. Put log records in a queue (non-blocking)
    2. Process them in a background thread
    3. Optionally buffer in memory before writing to disk
    
    Args:
        base_handlers: List of base handlers (file, stream, etc.)
        use_memory_buffer: Whether to wrap handlers in MemoryHandler
        memory_capacity: Number of records to buffer in memory (default: 5000)
        debug_mode: Whether we're in debug mode (enables optimizations)
    
    Returns:
        QueueListener instance if successful, None otherwise
    """
    global _ROOT_HANDLERS_BEFORE_QUEUE, _AIOS_HANDLERS_BEFORE_QUEUE

    try:
        # Create queue handler with UNLIMITED queue size to prevent blocking
        # The queue will be processed asynchronously, so we don't want to block the main thread
        log_queue: Queue = Queue(maxsize=0)  # Unlimited queue size (non-blocking)
        queue_handler = logging.handlers.QueueHandler(log_queue)
        
        # Optionally wrap handlers in memory buffers
        target_handlers: List[logging.Handler] = []
        log = logging.getLogger("aios.utils.async_logging")

        for handler in base_handlers:
            if use_memory_buffer and isinstance(handler, (
                logging.FileHandler,
                logging.handlers.RotatingFileHandler,
                logging.handlers.TimedRotatingFileHandler
            )):
                # Wrap file handlers in memory buffer
                # Use smaller capacity in debug mode for faster flushes
                buffer_capacity = min(memory_capacity, 1000) if debug_mode else memory_capacity

                # Choose flush level/interval so INFO logs reach disk promptly.
                flush_level = logging.WARNING
                flush_interval = 2.0
                if isinstance(handler, NonBlockingTimedRotatingFileHandler):
                    flush_level = logging.INFO
                    flush_interval = 1.0
                elif isinstance(handler, NonBlockingRotatingFileHandler):
                    flush_level = logging.DEBUG if debug_mode else logging.INFO
                    flush_interval = 1.0 if debug_mode else 2.0

                memory_handler = AsyncMemoryHandler(
                    capacity=buffer_capacity,
                    flushLevel=flush_level,
                    target=handler,
                    flushOnClose=True,
                    flushInterval=flush_interval,
                )
                memory_handler.setLevel(handler.level)

                # Flush immediately so the target handler creates its file early.
                try:
                    memory_handler.flush()
                except Exception:
                    pass

                target_handlers.append(memory_handler)
                log.info(
                    "Wrapped handler %s with AsyncMemoryHandler (flushLevel=%s interval=%s)",
                    type(handler).__name__,
                    logging.getLevelName(memory_handler.flushLevel),
                    getattr(memory_handler, "flushInterval", None),
                )
            else:
                # Use handler directly (e.g., StreamHandler for console)
                target_handlers.append(handler)
                log.info("Using handler without wrapper: %s", type(handler).__name__)
        
        # Create and start listener
        listener = create_queue_listener(
            log_queue,
            *target_handlers,
            respect_handler_level=True
        )
        log.info(
            "QueueListener initialized with handlers: %s",
            [type(h).__name__ for h in target_handlers],
        )
        listener.start()
        
        # Replace root logger handlers with queue handler
        root_logger = logging.getLogger()
        _ROOT_HANDLERS_BEFORE_QUEUE = root_logger.handlers[:]
        root_logger.handlers.clear()
        root_logger.addHandler(queue_handler)
        log.info("Root logger now uses QueueHandler")
        
        # Also update aios logger
        aios_logger = logging.getLogger("aios")
        _AIOS_HANDLERS_BEFORE_QUEUE = aios_logger.handlers[:]
        aios_logger.handlers.clear()
        aios_logger.addHandler(queue_handler)
        log.info("Aios logger now uses QueueHandler")
        
        return listener
        
    except Exception as e:
        logging.getLogger("aios.utils.async_logging").error(
            f"Failed to set up async logging: {e}",
            exc_info=True
        )
        return None


def shutdown_async_logging(listener: Optional[logging.handlers.QueueListener] = None) -> None:
    """Shutdown async logging and flush all buffers.
    
    Args:
        listener: Specific listener to shutdown, or None to shutdown all
    """
    global _ROOT_HANDLERS_BEFORE_QUEUE, _AIOS_HANDLERS_BEFORE_QUEUE

    if listener:
        try:
            listener.stop()
            if listener in _QUEUE_LISTENERS:
                _QUEUE_LISTENERS.remove(listener)
        except Exception as e:
            logging.getLogger("aios.utils.async_logging").error(
                f"Error stopping queue listener: {e}"
            )
    else:
        # Shutdown all registered listeners
        for listener in _QUEUE_LISTENERS[:]:
            try:
                listener.stop()
                _QUEUE_LISTENERS.remove(listener)
            except Exception as e:
                logging.getLogger("aios.utils.async_logging").error(
                    f"Error stopping queue listener: {e}"
                )

    # Restore original handlers so late logs during shutdown still flush normally.
    try:
        root_logger = logging.getLogger()
        if root_logger.handlers and any(isinstance(h, logging.handlers.QueueHandler) for h in root_logger.handlers):
            root_logger.handlers.clear()
            for handler in _ROOT_HANDLERS_BEFORE_QUEUE:
                root_logger.addHandler(handler)
        aios_logger = logging.getLogger("aios")
        if aios_logger.handlers and any(isinstance(h, logging.handlers.QueueHandler) for h in aios_logger.handlers):
            aios_logger.handlers.clear()
            for handler in _AIOS_HANDLERS_BEFORE_QUEUE:
                if handler not in aios_logger.handlers:
                    aios_logger.addHandler(handler)
    except Exception as restore_error:
        logging.getLogger("aios.utils.async_logging").debug(
            "Failed to restore synchronous handlers during shutdown", exc_info=restore_error
        )
    finally:
        _ROOT_HANDLERS_BEFORE_QUEUE = []
        _AIOS_HANDLERS_BEFORE_QUEUE = []


def _cleanup_on_exit() -> None:
    """Cleanup function called on program exit via atexit.
    
    Deletes this run's lock file to prevent accumulation of lock files from
    multiple program runs on the same day.
    """
    global _SHUTTING_DOWN
    _SHUTTING_DOWN = True
    
    # Shutdown async logging (safe to call multiple times)
    try:
        shutdown_async_logging()
    except Exception:
        pass
    
    # Delete this run's lock file (not lock files from other concurrent runs)
    global _CURRENT_LOCK_FILE
    if _CURRENT_LOCK_FILE is not None:
        try:
            if os.path.exists(_CURRENT_LOCK_FILE):
                os.remove(_CURRENT_LOCK_FILE)
        except (IOError, OSError):
            pass  # Ignore errors during cleanup


# Note: cleanup handlers are registered lazily when first lock file is created
# This is done in _register_cleanup_handlers() called from _write_session_to_lock()


class NonBlockingRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Rotating file handler with optimized settings for async logging.
    
    This handler adds:
    - Timestamped backup file names with session and part numbers 
      (e.g., aios_debug_11-11-2025_001_000.log)
    - Size-based rotation
    - Error handling to prevent blocking
    - Session numbers reset each day
    """
    
    def __init__(
        self,
        filename: str,
        mode: str = 'a',
        maxBytes: int = 10 * 1024 * 1024,  # 10 MB default
        backupCount: int = 5,
        encoding: Optional[str] = None,
        delay: bool = True  # Changed to True to avoid race conditions
    ):
        """Initialize non-blocking rotating file handler.
        
        Args:
            filename: Log file path
            mode: File open mode
            maxBytes: Maximum file size before rotation (default: 10 MB)
            backupCount: Number of backup files to keep
            encoding: File encoding
            delay: Whether to delay file opening (default: True to avoid creating empty files)
        """
        # Store the base filename BEFORE getting session number
        self._base_filename = filename
        
        # Get the session number ONCE for the entire program run
        # This ensures all handlers use the same session number
        self._session_number = get_next_session_number(filename)
        self._part_number = 0  # Start at 0 for each session
        
        # Create the actual filename with date, session, and part
        import time
        date_str = time.strftime("%d-%m-%Y")
        base_dir = os.path.dirname(filename)
        base_name = os.path.basename(filename)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Create session filename: filename_DD-MM-YYYY_SSS_PPP.log
        session_filename = f"{name_without_ext}_{date_str}_{self._session_number:03d}_{self._part_number:03d}.log"
        actual_filename = os.path.join(base_dir, session_filename) if base_dir else session_filename
        
        # Force UTF-8 encoding to handle Unicode characters (✓, ✗, etc.)
        if encoding is None:
            encoding = 'utf-8'
        
        # Initialize parent with the session-based filename
        super().__init__(
            actual_filename,
            mode=mode,
            maxBytes=maxBytes,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay
        )
    
    def doRollover(self) -> None:
        """Do a rollover with date-stamped file names, session, and part numbers.
        
        Creates backup files with format: filename_DD-MM-YYYY_SSS_PPP.log
        where SSS is the session number (resets daily) and PPP is the part number.
        Example: aios_debug_11-11-2025_001_000.log
        """
        try:
            if self.stream:
                self.stream.close()
                self.stream = None
            
            # Current file already has the correct name, just increment part number
            self._part_number += 1
            
            # Create new filename with incremented part number
            import time
            date_str = time.strftime("%d-%m-%Y")
            base_dir = os.path.dirname(self._base_filename)
            base_name = os.path.basename(self._base_filename)
            name_without_ext = os.path.splitext(base_name)[0]
            
            # Create the new filename: filename_DD-MM-YYYY_SSS_PPP.log
            new_filename = f"{name_without_ext}_{date_str}_{self._session_number:03d}_{self._part_number:03d}.log"
            new_path = os.path.join(base_dir, new_filename) if base_dir else new_filename
            
            # Update the baseFilename to the new part number
            self.baseFilename = new_path
            
            # Clean up old sessions (keeps last 10 sessions, all parts included)
            if self.backupCount > 0:
                cleanup_old_sessions(self._base_filename, max_sessions=self.backupCount)
            
            # Open new log file
            if not self.delay:
                self.stream = self._open()
            
        except Exception as e:
            # Log the error but don't crash
            import sys
            print(f"Error during log rotation: {e}", file=sys.stderr)
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record with error handling to prevent blocking."""
        try:
            super().emit(record)
        except UnicodeEncodeError:
            # Handle Unicode encoding errors by sanitizing the message
            try:
                # Try to encode message as ASCII, replacing problematic characters
                if hasattr(record, 'msg'):
                    original_msg = record.msg
                    # Replace common Unicode symbols with ASCII equivalents
                    sanitized_msg = str(original_msg).encode('ascii', errors='replace').decode('ascii')
                    record.msg = sanitized_msg
                super().emit(record)
                record.msg = original_msg  # Restore original message
            except Exception:
                self.handleError(record)
        except Exception:
            # Don't let logging errors crash the app
            self.handleError(record)


class NonBlockingTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """Timed rotating file handler with optimized settings for async logging.
    
    This handler adds:
    - Date-stamped backup file names with session and part numbers 
      (e.g., aios_11-11-2025_001_000.log)
    - Proper rotation at specified intervals OR when file size exceeds maxBytes
    - Error handling to prevent blocking
    - Session numbers reset each day
    """
    
    def __init__(
        self,
        filename: str,
        when: str = 'midnight',
        interval: int = 1,
        backupCount: int = 100,
        encoding: Optional[str] = None,
        delay: bool = False,
        utc: bool = False,
        maxBytes: int = 0,  # 0 = no size limit, otherwise rotate when file exceeds this size
        include_tracebacks: bool = True
    ):
        """Initialize non-blocking timed rotating file handler.
        
        Args:
            filename: Log file path
            when: When to rotate (midnight, H, D, etc.)
            interval: Rotation interval
            backupCount: Number of backup files to keep
            encoding: File encoding
            delay: Whether to delay file opening
            utc: Whether to use UTC time
            maxBytes: Maximum file size in bytes before rotation (0 = no limit)
            include_tracebacks: Whether to include exception tracebacks in output
        """
        # Store base filename and session info before calling parent __init__
        self._base_filename = filename
        self._session_number = get_next_session_number(filename)
        self._part_number = 0  # Start at 0 for each session
        self.include_tracebacks = include_tracebacks
        
        # Create the actual filename with date, session, and part
        import time
        date_str = time.strftime("%d-%m-%Y")
        base_dir = os.path.dirname(filename)
        base_name = os.path.basename(filename)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Create session filename: filename_DD-MM-YYYY_SSS_PPP.log
        session_filename = f"{name_without_ext}_{date_str}_{self._session_number:03d}_{self._part_number:03d}.log"
        actual_filename = os.path.join(base_dir, session_filename) if base_dir else session_filename
        
        # Force UTF-8 encoding to handle Unicode characters (✓, ✗, etc.)
        if encoding is None:
            encoding = 'utf-8'
        
        # Initialize parent with the session-based filename
        super().__init__(
            actual_filename,
            when=when,
            interval=interval,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay,
            utc=utc
        )
        self.maxBytes = maxBytes
    
    def shouldRollover(self, record: logging.LogRecord) -> bool:
        """Determine if rollover should occur.
        
        Rollover happens if:
        - The scheduled time has been reached (time-based), OR
        - The file size exceeds maxBytes (size-based)
        
        Args:
            record: The log record being processed
            
        Returns:
            True if rollover should occur
        """
        # Check time-based rollover (from parent class)
        import time
        t = int(time.time())
        if t >= self.rolloverAt:
            return True
        
        # Check size-based rollover if maxBytes is set
        if self.maxBytes > 0:
            if self.stream is None:
                self.stream = self._open()
            if self.stream:
                self.stream.seek(0, 2)  # Seek to end
                if self.stream.tell() + len(record.getMessage()) >= self.maxBytes:
                    return True
        
        return False
    
    def doRollover(self) -> None:
        """Do a rollover with date-stamped file names, session, and part numbers.
        
        Override to ensure proper file naming with date, session, and part numbers.
        Creates backup files with format: filename_DD-MM-YYYY_SSS_PPP.log
        where SSS is the session number (resets daily) and PPP is the part number.
        Example: aios_11-11-2025_001_000.log
        """
        try:
            if self.stream:
                self.stream.close()
                self.stream = None
            
            # Current file already has the correct name, just increment part number
            self._part_number += 1
            
            # Get the current date
            import time
            current_time = int(time.time())
            time_tuple = time.localtime(current_time) if not self.utc else time.gmtime(current_time)
            date_str = time.strftime("%d-%m-%Y", time_tuple)
            
            # Extract filename without extension
            base_dir = os.path.dirname(self._base_filename)
            base_name = os.path.basename(self._base_filename)
            name_without_ext = os.path.splitext(base_name)[0]
            
            # Create the new filename: filename_DD-MM-YYYY_SSS_PPP.log
            new_filename = f"{name_without_ext}_{date_str}_{self._session_number:03d}_{self._part_number:03d}.log"
            new_path = os.path.join(base_dir, new_filename) if base_dir else new_filename
            
            # Update the baseFilename to the new part number
            self.baseFilename = new_path
            
            # Clean up old sessions (keeps last 10 sessions, all parts included)
            if self.backupCount > 0:
                cleanup_old_sessions(self._base_filename, max_sessions=self.backupCount)
            
            # Open new log file
            if not self.delay:
                self.stream = self._open()
            
            # Update rollover time
            self.rolloverAt = self.computeRollover(current_time)
            
        except Exception as e:
            # Log the error but don't crash
            import sys
            print(f"Error during log rotation: {e}", file=sys.stderr)
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record with optional traceback suppression and error handling."""
        restore_exc = False
        original_exc = None
        original_stack = None
        original_exc_text = None
        has_stack_attr = hasattr(record, "stack_info")
        has_exc_text_attr = hasattr(record, "exc_text")

        if not self.include_tracebacks:
            strip_trace = bool(record.exc_info) or (
                has_stack_attr and bool(getattr(record, "stack_info", None))
            ) or bool(getattr(record, "exc_text", None))
            if strip_trace:
                restore_exc = True
                original_exc = record.exc_info
                original_stack = getattr(record, "stack_info", None)
                record.exc_info = None
                if has_stack_attr:
                    record.stack_info = None
                if has_exc_text_attr:
                    original_exc_text = getattr(record, "exc_text", None)
                    record.exc_text = None

        try:
            try:
                super().emit(record)
            except UnicodeEncodeError:
                # Handle Unicode encoding errors by sanitizing the message
                try:
                    # Try to encode message as ASCII, replacing problematic characters
                    if hasattr(record, 'msg'):
                        original_msg = record.msg
                        # Replace common Unicode symbols with ASCII equivalents
                        sanitized_msg = str(original_msg).encode('ascii', errors='replace').decode('ascii')
                        record.msg = sanitized_msg
                    super().emit(record)
                    record.msg = original_msg  # Restore original message
                except Exception:
                    self.handleError(record)
            except Exception:
                # Don't let logging errors crash the app
                self.handleError(record)
        finally:
            if restore_exc:
                record.exc_info = original_exc
                if has_stack_attr:
                    record.stack_info = original_stack
                if has_exc_text_attr:
                    record.exc_text = original_exc_text


class UTF8StreamHandler(logging.StreamHandler):
    """StreamHandler that safely handles Unicode characters on Windows.
    
    Windows console uses cp1252 by default which can't handle Unicode characters
    like ✓, ✗, etc. This handler sanitizes such characters to prevent crashes.
    """

    def __init__(self, stream=None, include_tracebacks: bool = True):
        super().__init__(stream)
        self.include_tracebacks = include_tracebacks

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record with Unicode error handling and optional trace suppression."""
        restore_exc = False
        original_exc = None
        original_stack = None
        original_exc_text = None
        has_stack_attr = hasattr(record, "stack_info")
        has_exc_text_attr = hasattr(record, "exc_text")

        if not self.include_tracebacks:
            strip_trace = bool(record.exc_info) or (
                has_stack_attr and bool(getattr(record, "stack_info", None))
            ) or bool(getattr(record, "exc_text", None))
            if strip_trace:
                restore_exc = True
                original_exc = record.exc_info
                original_stack = getattr(record, "stack_info", None)
                record.exc_info = None
                if has_stack_attr:
                    record.stack_info = None
                if has_exc_text_attr:
                    original_exc_text = getattr(record, "exc_text", None)
                    record.exc_text = None

        try:
            try:
                super().emit(record)
            except UnicodeEncodeError:
                # Handle Unicode encoding errors by sanitizing the message
                try:
                    if hasattr(record, 'msg'):
                        original_msg = record.msg
                        # Replace Unicode characters that can't be encoded with ASCII equivalents
                        # Common replacements: ✓→√ or [OK], ✗→X, etc.
                        sanitized_msg = str(original_msg)
                        sanitized_msg = sanitized_msg.replace('✓', '[OK]')
                        sanitized_msg = sanitized_msg.replace('✗', '[X]')
                        sanitized_msg = sanitized_msg.replace('⚠', '[!]')
                        sanitized_msg = sanitized_msg.replace('🔍', '[SEARCH]')
                        sanitized_msg = sanitized_msg.replace('📦', '[CACHE]')
                        sanitized_msg = sanitized_msg.replace('🔄', '[REFRESH]')
                        # Fallback: replace any remaining problematic characters
                        sanitized_msg = sanitized_msg.encode('ascii', errors='replace').decode('ascii')
                        record.msg = sanitized_msg
                    super().emit(record)
                    record.msg = original_msg  # Restore original message
                except Exception:
                    self.handleError(record)
            except Exception:
                self.handleError(record)
        finally:
            if restore_exc:
                record.exc_info = original_exc
                if has_stack_attr:
                    record.stack_info = original_stack
                if has_exc_text_attr:
                    record.exc_text = original_exc_text
