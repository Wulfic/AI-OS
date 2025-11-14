from __future__ import annotations

import asyncio
import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Optional
import io

import yaml
from aios.utils.async_logging import NonBlockingTimedRotatingFileHandler

# Global reference to UTF-8 wrapped streams for Windows
_utf8_stdout = None
_utf8_stderr = None

# Wrap stdout/stderr with UTF-8 on Windows at module import time
# This MUST happen before any logging handlers are created
if sys.platform == 'win32':
    try:
        # Check if stdout has a buffer attribute (may not in some environments)
        if hasattr(sys.stdout, 'buffer'):
            _utf8_stdout = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding='utf-8',
                errors='replace',
                line_buffering=True
            )
            sys.stdout = _utf8_stdout
    except (AttributeError, ValueError) as e:
        # If wrapping fails, continue with original stdout
        pass
    
    try:
        # Check if stderr has a buffer attribute
        if hasattr(sys.stderr, 'buffer'):
            _utf8_stderr = io.TextIOWrapper(
                sys.stderr.buffer,
                encoding='utf-8',
                errors='replace',
                line_buffering=True
            )
            sys.stderr = _utf8_stderr
    except (AttributeError, ValueError) as e:
        # If wrapping fails, continue with original stderr
        pass

logger = logging.getLogger(__name__)

# Global reference to queue listener for cleanup
_async_logging_listener = None

# Global flag to track if logging has been set up in this process
_logging_initialized = False

# Track whether debug mode was active when logging was configured
_is_debug_mode = False


def load_config(path: Optional[str] = None) -> dict:
    env_path = os.environ.get("AIOS_CONFIG")
    p = Path(path or env_path or Path.home() / ".config/aios/config.yaml")
    if not p.exists():
        # fall back to repo default if available
        repo_default = Path(__file__).resolve().parents[3] / "config" / "default.yaml"
        if repo_default.exists():
            logger.debug(f"Config not found at {p}, using repo default: {repo_default}")
            with open(repo_default, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        logger.warning(f"No config file found at {p} or repo default, using empty config")
        return {}
    logger.info(f"Loading configuration from {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_logging(
    cfg: dict,
    *,
    level: Optional[str] = None,
    json_preferred: Optional[bool] = None,
    stream: Optional[str] = None,
    logfile: Optional[str] = None,
    enable_async: bool = True,
    use_memory_buffer: bool = True,
    memory_capacity: int = 5000,
) -> None:
    """Initialize logging with sensible defaults and VS Code-friendly overrides.

    Honors environment variables:
      - AIOS_LOG_LEVEL: DEBUG|INFO|WARNING|ERROR (default DEBUG in VS Code debug sessions)
      - AIOS_LOG_JSON: 1/0 to force json vs text formatter (default json when logging.yaml specifies it)
      - AIOS_LOG_STREAM: stdout|stderr (default stderr for debug sessions)
      - AIOS_LOG_FILE: path to log file (overrides configured filename if provided)
      - AIOS_DEBUG: any non-empty value enables DEBUG level and stderr stream
      - AIOS_ASYNC_LOGGING: 0 to disable async logging (enabled by default)
      - AIOS_LOG_MEMORY_BUFFER: 0 to disable memory buffering (enabled by default)
      - AIOS_LOG_BUFFER_SIZE: number of records to buffer in memory (default: 5000)
      - AIOS_LOG_ULTRA_SIMPLE: 1 to enable ultra-minimal formatting (faster but less info)
    
    Args:
        cfg: Configuration dictionary
        level: Override log level
        json_preferred: Prefer JSON formatting
        stream: Override stream (stdout/stderr)
        logfile: Override log file path
        enable_async: Enable async logging with QueueHandler (default: True)
        use_memory_buffer: Enable memory buffering (default: True)
        memory_capacity: Memory buffer capacity (default: 5000)
    """
    global _async_logging_listener, _logging_initialized, _is_debug_mode
    
    # Prevent re-initialization - logging should only be set up once per process
    if _logging_initialized:
        # Use print to stderr since logger might not be configured yet
        import sys
        print(f"[setup_logging PID:{os.getpid()}] Logging already initialized, skipping", file=sys.stderr)
        return
    
    # Use print to stderr to show initialization
    import sys
    print(f"[setup_logging PID:{os.getpid()}] Initializing logging for the first time", file=sys.stderr)
    _logging_initialized = True
    
    cfg_path = cfg.get("logging", {}).get("config_path") if isinstance(cfg, dict) else None
    
    # If cfg_path is relative, resolve it relative to the repo root
    if cfg_path and not Path(cfg_path).is_absolute():
        repo_root = Path(__file__).resolve().parents[3]  # src/aios/cli/utils.py -> repo root
        cfg_path = str(repo_root / cfg_path)

    # Detect if we're running under a VS Code debug session and prefer DEBUG to stderr
    is_vscode = bool(os.environ.get("VSCODE_PID"))
    is_debug = bool(os.environ.get("AIOS_DEBUG")) or is_vscode
    _is_debug_mode = is_debug
    
    # Log what mode we're in (will go to stderr initially)
    import sys
    print(f"[setup_logging] Debug mode: {is_debug} (AIOS_DEBUG={os.environ.get('AIOS_DEBUG')}, VSCODE_PID={os.environ.get('VSCODE_PID')})", file=sys.stderr)

    # Gather env overrides
    env_level = (level or os.environ.get("AIOS_LOG_LEVEL") or ("DEBUG" if is_debug else "")).upper()
    env_json = str(json_preferred) if json_preferred is not None else os.environ.get("AIOS_LOG_JSON")
    env_stream = (stream or os.environ.get("AIOS_LOG_STREAM") or ("stderr" if is_debug else "")).lower()
    env_file = logfile or os.environ.get("AIOS_LOG_FILE")
    
    # Check async logging environment variables
    env_async = os.environ.get("AIOS_ASYNC_LOGGING", "1").strip() not in ("0", "false", "False")
    env_buffer = os.environ.get("AIOS_LOG_MEMORY_BUFFER", "1").strip() not in ("0", "false", "False")
    env_buffer_size = int(os.environ.get("AIOS_LOG_BUFFER_SIZE", str(memory_capacity)))
    env_ultra_simple = os.environ.get("AIOS_LOG_ULTRA_SIMPLE", "0").strip() not in ("0", "false", "False")
    
    enable_async = enable_async and env_async
    use_memory_buffer = use_memory_buffer and env_buffer
    memory_capacity = env_buffer_size

    # Load config dict from YAML if available; otherwise craft a minimal default
    conf: dict
    if cfg_path and Path(cfg_path).exists():
        logger.debug(f"Loading logging configuration from {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            conf = yaml.safe_load(f) or {}
    else:
        logger.debug("No logging config path found, using default configuration")
        conf = {
            "version": 1,
            "formatters": {
                "json": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"},
                "verbose": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
                },
                "fast_debug": {
                    "format": "%(levelname)s | %(name)s | %(message)s",
                },
                "ultra_simple": {
                    "format": "%(levelname)s: %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "json",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "aios": {"level": "INFO", "handlers": ["console"], "propagate": False},
            },
            "root": {"level": "WARNING", "handlers": ["console"]},
        }

    # Ensure keys exist
    conf.setdefault("formatters", {})
    conf.setdefault("handlers", {})
    conf.setdefault("loggers", {})
    conf.setdefault("root", {})
    conf.setdefault("filters", {})

    # Add/ensure a good text formatter for debug sessions
    conf["formatters"].setdefault(
        "verbose",
        {"format": "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"},
    )
    
    # Add fast debug formatter for high-volume debug logging
    conf["formatters"].setdefault(
        "fast_debug",
        {
            "format": "%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    )
    
    # Add ultra-simple formatter for maximum performance
    conf["formatters"].setdefault(
        "ultra_simple",
        {"format": "%(levelname)s: %(message)s"},
    )

    # Determine desired console formatter (json vs verbose text)
    use_json = None
    if env_json is not None:
        use_json = env_json.strip() not in ("0", "false", "False")
    # If not explicitly set, keep whatever logging.yaml configured
    console = conf["handlers"].get("console", {})
    current_formatter = console.get("formatter") or "json"
    
    # Use ultra_simple formatter in debug mode for maximum performance
    if is_debug and env_ultra_simple:
        desired_formatter = "ultra_simple"
    elif is_debug:
        desired_formatter = "fast_debug" if not use_json else current_formatter
    else:
        desired_formatter = ("json" if use_json else "verbose") if use_json is not None else current_formatter

    # Determine desired stream for console
    desired_stream = console.get("stream") or "ext://sys.stdout"
    if env_stream in ("stdout", "stderr"):
        desired_stream = f"ext://sys.{env_stream}"
    elif is_debug:
        desired_stream = "ext://sys.stderr"

    # Determine desired logging level
    def _normalize_level(val: str | int | None) -> str:
        if val is None:
            return ""
        if isinstance(val, int):
            return logging.getLevelName(val)
        return val.upper()

    existing_root_level = _normalize_level(conf.get("root", {}).get("level")) or "WARNING"
    desired_level = env_level or existing_root_level
    if not desired_level:
        desired_level = "DEBUG" if is_debug else "INFO"

    explicit_level_source = level or os.environ.get("AIOS_LOG_LEVEL")
    explicit_level_normalized = _normalize_level(explicit_level_source) if explicit_level_source else ""
    if explicit_level_normalized:
        console_level = explicit_level_normalized
    else:
        console_level = "INFO" if is_debug else (desired_level or "INFO")
    if not console_level:
        console_level = "INFO"

    logger.debug(
        "Logging configuration: level=%s, console_level=%s, formatter=%s, stream=%s, async=%s, buffer=%s, capacity=%s, ultra_simple=%s",
        desired_level,
        console_level,
        desired_formatter,
        desired_stream,
        enable_async,
        use_memory_buffer,
        memory_capacity,
        env_ultra_simple,
    )
    
    # Apply overrides to console handler - use UTF8StreamHandler to handle Unicode safely
    console_handler = conf["handlers"].setdefault("console", {"class": "aios.utils.async_logging.UTF8StreamHandler"})
    console_handler["level"] = console_level
    console_handler["formatter"] = desired_formatter
    console_handler["stream"] = desired_stream

    console_class = console_handler.get("class", "")
    if not console_class:
        console_class = "aios.utils.async_logging.UTF8StreamHandler"
        console_handler["class"] = console_class
    class_tail = console_class.rsplit(".", 1)[-1]
    if class_tail == "UTF8StreamHandler":
        console_handler["include_tracebacks"] = False

    # If a file handler exists or an override is provided, ensure it uses at least INFO unless explicitly set
    if "file" in conf["handlers"] or env_file:
        fh = conf["handlers"].setdefault("file", {"class": "logging.handlers.RotatingFileHandler"})
        
        # Only set maxBytes/backupCount for RotatingFileHandler, not TimedRotatingFileHandler
        handler_class = fh.get("class", "")
        if "TimedRotatingFileHandler" in handler_class or "NonBlockingTimedRotatingFileHandler" in handler_class:
            # TimedRotatingFileHandler uses when/interval/backupCount
            fh.setdefault("backupCount", 100)
            # Don't add maxBytes - it's not a valid parameter for TimedRotatingFileHandler
        else:
            # RotatingFileHandler uses maxBytes/backupCount
            fh.setdefault("maxBytes", 10485760)  # 10 MB
            fh.setdefault("backupCount", 5)
        
        fh.setdefault("formatter", "ultra_simple" if (is_debug and env_ultra_simple) else (desired_formatter if not is_debug else "fast_debug"))
        if env_file:
            fh["filename"] = env_file
        fh["level"] = console_level or "INFO"
        class_tail = handler_class.rsplit(".", 1)[-1] if handler_class else ""
        if class_tail == "NonBlockingTimedRotatingFileHandler":
            fh["include_tracebacks"] = False
    
    # Handle debug file handler based on debug mode
    if is_debug:
        # Ensure debug handler exists (either from YAML or create it)
        if "file_rotating" not in conf["handlers"]:
            conf["handlers"]["file_rotating"] = {
                "class": "aios.utils.async_logging.NonBlockingRotatingFileHandler",
                "level": "DEBUG",
                "formatter": "ultra_simple" if env_ultra_simple else "fast_debug",
                "filename": "logs/aios_debug.log",
                "maxBytes": 20971520,  # 20 MB
                "backupCount": 10,
                "encoding": "utf-8",
            }
        # Update formatter if needed for debug mode
        conf["handlers"]["file_rotating"]["formatter"] = "ultra_simple" if env_ultra_simple else "fast_debug"
        conf["handlers"]["file_rotating"].setdefault("level", "DEBUG")

        conf["filters"].setdefault(
            "debug_trace_filter",
            {"()": "aios.utils.async_logging.DebugAndTraceFilter"},
        )
        existing_filters = conf["handlers"]["file_rotating"].get("filters", [])
        if isinstance(existing_filters, str):
            existing_filters = [existing_filters]
        if "debug_trace_filter" not in existing_filters:
            existing_filters.append("debug_trace_filter")
        conf["handlers"]["file_rotating"]["filters"] = existing_filters
    else:
        # Remove debug handler if not in debug mode
        if "file_rotating" in conf["handlers"]:
            del conf["handlers"]["file_rotating"]

    # Raise 'aios' logger and root to desired level for debug visibility
    conf["loggers"].setdefault("aios", {"handlers": ["console"], "propagate": False})
    
    # Set logger level to the MINIMUM of all handler levels to allow handler-level filtering
    # This ensures DEBUG handlers can receive DEBUG messages even when console is INFO
    min_handler_level = desired_level
    if "file_rotating" in conf["handlers"]:
        # file_rotating handler is DEBUG, so logger must be DEBUG too
        min_handler_level = "DEBUG"
    
    conf["loggers"]["aios"]["level"] = min_handler_level
    
    # Add file handler to aios logger if it exists in handlers config
    if "file" in conf["handlers"] and "file" not in conf["loggers"]["aios"]["handlers"]:
        conf["loggers"]["aios"]["handlers"].append("file")
    
    # Add file_rotating handler if it exists in handlers config (regardless of debug mode)
    # The handler's own level (DEBUG) will filter appropriately
    if "file_rotating" in conf["handlers"] and "file_rotating" not in conf["loggers"]["aios"]["handlers"]:
        conf["loggers"]["aios"]["handlers"].append("file_rotating")
    
    # Set root logger level to minimum as well
    conf["root"]["level"] = min_handler_level
    conf["root"].setdefault("handlers", ["console"])
    
    # Add file handler to root logger if it exists
    if "file" in conf["handlers"] and "file" not in conf["root"]["handlers"]:
        conf["root"]["handlers"].append("file")
    
    # Add file_rotating handler to root logger if it exists
    if "file_rotating" in conf["handlers"] and "file_rotating" not in conf["root"]["handlers"]:
        conf["root"]["handlers"].append("file_rotating")

    # Install configuration (sys.stdout/stderr are already UTF-8 wrapped on Windows at module import time)
    logging.config.dictConfig(conf)

    root_logger = logging.getLogger()
    aios_logger = logging.getLogger("aios")
    logger.debug(
        "Root handlers after dictConfig: %s",
        [type(h).__name__ for h in root_logger.handlers],
    )
    logger.debug(
        "Aios handlers after dictConfig: %s",
        [type(h).__name__ for h in aios_logger.handlers],
    )
    
    # Set up async logging if enabled
    if enable_async:
        try:
            from aios.utils.async_logging import setup_async_logging
            
            # Collect all configured handlers
            handlers = []
            handlers.extend(root_logger.handlers[:])

            for h in aios_logger.handlers:
                if h not in handlers:
                    handlers.append(h)

            logger.debug(
                "Collected %d base handler(s) for async logging: %s",
                len(handlers),
                [type(h).__name__ for h in handlers],
            )
            
            # Set up async logging
            _async_logging_listener = setup_async_logging(
                handlers,
                use_memory_buffer=use_memory_buffer,
                memory_capacity=memory_capacity,
                debug_mode=is_debug
            )
            
            if _async_logging_listener:
                logger.info(f"Async logging enabled: buffer={use_memory_buffer}, capacity={memory_capacity}")
            else:
                logger.warning("Failed to enable async logging, falling back to synchronous logging")
                
        except Exception as e:
            logger.warning(f"Could not enable async logging: {e}, falling back to synchronous logging")
    
    # Log environment info after logging is configured
    env_vars = ["AIOS_LOG_LEVEL", "AIOS_LOG_JSON", "AIOS_LOG_STREAM", "AIOS_LOG_FILE", "AIOS_DEBUG", "VSCODE_PID", "AIOS_ASYNC_LOGGING", "AIOS_LOG_MEMORY_BUFFER", "AIOS_LOG_BUFFER_SIZE", "AIOS_LOG_ULTRA_SIMPLE"]
    active_env = {k: os.environ.get(k) for k in env_vars if os.environ.get(k)}
    if active_env:
        logger.debug(f"Active logging environment variables: {active_env}")
    
    # Capture warnings module messages into logging
    try:
        logging.captureWarnings(True)
    except Exception:
        pass

    # Global exception hooks to surface unexpected errors into logs (and thus Debug Console)
    def _excepthook(exc_type, exc, tb):
        log = logging.getLogger("aios")
        log.error("Uncaught exception: %s", exc)
        if _is_debug_mode:
            import traceback

            stack = "".join(traceback.format_exception(exc_type, exc, tb))
            log.debug("Stack trace for uncaught exception:\n%s", stack)
    sys.excepthook = _excepthook

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    if loop and not loop.is_closed():
        def _handle_asyncio(loop, context):  # type: ignore[no-redef]
            msg = context.get("message") or "Unhandled asyncio exception"
            err = context.get("exception")
            if err:
                logging.getLogger("aios").exception(msg, exc_info=err)
            else:
                logging.getLogger("aios").error(msg)
        try:
            loop.set_exception_handler(_handle_asyncio)
        except Exception:
            pass


def dml_cfg_path() -> Path:
    return Path.home() / ".config/aios/dml_python.txt"


def shutdown_logging() -> None:
    """Shutdown async logging gracefully and flush all buffers.
    
    Call this function before application exit to ensure all log records
    are written to disk.
    """
    global _async_logging_listener
    
    if _async_logging_listener:
        try:
            from aios.utils.async_logging import shutdown_async_logging
            shutdown_async_logging(_async_logging_listener)
            _async_logging_listener = None
            logging.getLogger("aios").info("Async logging shutdown complete")
        except Exception as e:
            logging.getLogger("aios").error(f"Error during logging shutdown: {e}")
