"""Main settings panel class."""

from __future__ import annotations
import webbrowser
import logging
import os
import time
from typing import Any, Callable
from tkinter import ttk
import tkinter as tk

from . import ui_builders, theme_manager, startup_settings, cache_management
from ...utils.resource_management import submit_background

logger = logging.getLogger(__name__)


class SettingsPanel:
    """Settings panel with theme selection and other preferences."""

    def __init__(
        self,
        parent: Any,
        save_state_fn: Callable[[], None] | None = None,
        chat_panel: Any | None = None,
        help_panel: Any | None = None,
        debug_panel: Any | None = None,
        worker_pool: Any | None = None,
    ) -> None:
        self.parent = parent
        self._save_state_fn = save_state_fn
        self._chat_panel = chat_panel
        self._help_panel = help_panel
        self._debug_panel = debug_panel
        self._worker_pool = worker_pool
        self._debug_file_handler: logging.Handler | None = None
        self._last_theme_applied: str | None = None
        self._last_theme_applied_at: float = 0.0
        
        # Flag to prevent trace callbacks during state restoration
        self._restoring_state = False

        # Main container with canvas for scrolling
        main_container = ttk.Frame(parent)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Build UI with two-column layout
        ui_builders.create_title(main_container)
        
        # Create two columns
        columns_frame = ttk.Frame(main_container)
        columns_frame.pack(fill="both", expand=True)
        
        left_column = ttk.Frame(columns_frame)
        left_column.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        right_column = ttk.Frame(columns_frame)
        right_column.pack(side="left", fill="both", expand=True, padx=(5, 0))
        
        # Left column: Appearance, General Settings, Support
        ui_builders.create_appearance_section(self, left_column)
        ui_builders.create_general_settings_section(self, left_column)
        ui_builders.create_support_section(self, left_column)
        
        # Right column: Logging, Help, Cache
        ui_builders.create_logging_section(self, right_column)
        ui_builders.create_help_section(self, right_column)
        ui_builders.create_cache_section(self, right_column)
        
        # Load initial settings
        self._load_startup_status()
        self._load_cache_size()
        self._refresh_cache_stats()

    def _apply_theme(self, theme: str) -> None:
        """Apply the selected theme to the application."""
        theme_manager.apply_theme(self, theme)

    def _open_kofi_link(self) -> None:
        """Open Ko-fi link in default browser."""
        try:
            webbrowser.open("https://ko-fi.com/wulfic")
        except Exception as e:
            logger.error(f"Error opening Ko-fi link: {e}")

    def _load_startup_status(self) -> None:
        """Load current startup status from Windows registry."""
        startup_settings.load_startup_status(self)

    def _on_startup_changed(self) -> None:
        """Handle startup checkbox toggle."""
        startup_settings.on_startup_changed(self)

    def _on_start_minimized_changed(self) -> None:
        """Handle start minimized checkbox toggle."""
        startup_settings.on_start_minimized_changed(self)

    def _on_minimize_to_tray_changed(self) -> None:
        """Handle minimize to tray checkbox toggle."""
        startup_settings.on_minimize_to_tray_changed(self)

    def _load_cache_size(self) -> None:
        """Load cache size configuration from config file."""
        cache_management.load_cache_size(self)

    def _save_cache_size(self) -> None:
        """Save cache size configuration to config file."""
        cache_management.save_cache_size(self)

    def _refresh_cache_stats(self) -> None:
        """Refresh and display cache statistics."""
        cache_management.refresh_cache_stats(self)

    def _clear_cache(self) -> None:
        """Clear all cached dataset blocks."""
        cache_management.clear_cache(self)

    def _rebuild_help_index(self) -> None:
        """Rebuild the help documentation search index."""
        from pathlib import Path

        def _dispatch(callback: Callable[[], None]) -> None:
            try:
                self.parent.after(0, callback)
            except Exception:
                try:
                    callback()
                except Exception:
                    logger.debug("Help index UI update failed", exc_info=True)

        def rebuild() -> None:
            status_text = "✗ Failed to rebuild"
            try:
                from ...gui.components.help_panel import utils
                project_root = utils.find_project_root(Path(__file__))
                docs_root = utils.resolve_docs_root(project_root)

                index_file = docs_root / "search_index.json"
                if index_file.exists():
                    index_file.unlink()
                    logger.info("Deleted old search index")

                from ...gui.components.help_panel.search_engine import SearchEngine
                engine = SearchEngine(docs_root)
                success = engine.build_index()
                if success:
                    doc_count = len(engine.index)
                    status_text = f"✓ Ready ({doc_count} docs)"
                    logger.info("Rebuilt help index with %s documents", doc_count)
                else:
                    status_text = "✗ Failed to rebuild"
                    logger.error("Failed to rebuild help index")
            except Exception as exc:
                status_text = "✗ Error"
                logger.error("Error rebuilding help index: %s", exc, exc_info=True)
            finally:
                _dispatch(lambda: self.help_index_status_label.config(text=status_text))

        _dispatch(lambda: self.help_index_status_label.config(text="Building..."))

        try:
            submit_background("settings-help-index", rebuild, pool=self._worker_pool)
        except RuntimeError as exc:
            logger.error("Failed to queue help index rebuild: %s", exc)
            _dispatch(lambda: self.help_index_status_label.config(text=f"✗ Queue error: {exc}"))

    def get_state(self) -> dict[str, Any]:
        """Return current settings state for persistence."""
        return {
            "theme": self.theme_var.get(),
            "startup_enabled": self.startup_var.get(),
            "start_minimized": self.start_minimized_var.get(),
            "minimize_to_tray": self.minimize_to_tray_var.get(),
            "log_level": self.log_level_var.get(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore settings state from saved data."""
        # Set flag to prevent trace callbacks during restoration
        self._restoring_state = True
        
        try:
            theme = state.get("theme")
            if theme in ("Light Mode", "Dark Mode", "Matrix Mode", "Barbie Mode", "Halloween Mode"):
                self.theme_var.set(theme)
                # Apply immediately on load
                self._apply_theme(theme)
                logger.info(f"Restored and applied theme: {theme}")
        except Exception as e:
            logger.error(f"Failed to restore theme: {e}", exc_info=True)
            # Ensure theme_var has a valid default value even if restore fails
            if not self.theme_var.get() or self.theme_var.get() not in ("Light Mode", "Dark Mode", "Matrix Mode", "Barbie Mode", "Halloween Mode"):
                logger.warning("No valid theme found, defaulting to Dark Mode")
                self.theme_var.set("Dark Mode")
                self._apply_theme("Dark Mode")
        
        try:
            startup = state.get("startup_enabled")
            start_minimized = state.get("start_minimized", False)
            if startup is not None:
                self.startup_var.set(bool(startup))
                # Apply startup setting immediately
                from ...utils.startup import set_startup_enabled, is_windows
                if is_windows():
                    set_startup_enabled(bool(startup), minimized=bool(start_minimized))
        except Exception:
            pass
        
        try:
            start_minimized = state.get("start_minimized")
            if start_minimized is not None:
                self.start_minimized_var.set(bool(start_minimized))
        except Exception:
            pass
        
        try:
            minimize_to_tray = state.get("minimize_to_tray")
            if minimize_to_tray is not None:
                self.minimize_to_tray_var.set(bool(minimize_to_tray))
        except Exception:
            pass
        
        try:
            log_level = state.get("log_level")
            if log_level in ("DEBUG", "Advanced", "Normal"):
                self.log_level_var.set(log_level)
                logger.info(f"Restored logging level: {log_level}")
                # Apply to debug panel immediately if available
                self._apply_log_level(log_level)
            else:
                # Default to Normal if not set or invalid
                self.log_level_var.set("Normal")
                logger.info("No valid logging level found, defaulting to Normal")
                self._apply_log_level("Normal")
        except Exception:
            self.log_level_var.set("Normal")  # Default to Normal on error
            self._apply_log_level("Normal")
        
        # Clear flag after restoration is complete
        self._restoring_state = False

    def _apply_log_level(self, level: str) -> None:
        """Apply the logging level to the debug panel.
        
        Args:
            level: One of "Normal", "Advanced", or "DEBUG"
        """
        enable_debug_logs = level == "DEBUG"
        self._manage_debug_file_handler(enable_debug_logs)

        if self._debug_panel and hasattr(self._debug_panel, 'set_global_log_level'):
            try:
                self._debug_panel.set_global_log_level(level)
                logger.debug(f"Applied logging level to debug panel: {level}")
            except Exception as e:
                logger.error(f"Failed to apply logging level: {e}")

    def _manage_debug_file_handler(self, enable_debug: bool) -> None:
        """Enable or disable the rotating debug file handler.

        Integrates with async logging (QueueListener) when available to avoid
        duplicate handlers while keeping timestamp formatting consistent.
        """

        start_time = time.perf_counter()
        action = "unchanged"

        try:
            from logging.handlers import QueueHandler
            from aios.utils.async_logging import (
                NonBlockingRotatingFileHandler,
                NonBlockingTimedRotatingFileHandler,
                AsyncMemoryHandler,
                DebugAndTraceFilter,
                _QUEUE_LISTENERS,
            )

            root_logger = logging.getLogger()
            aios_logger = logging.getLogger("aios")

            def _ensure_formatter(handler: logging.Handler) -> None:
                handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s",
                        "%Y-%m-%d %H:%M:%S",
                    )
                )

            def _ensure_debug_filter(handler: logging.Handler) -> None:
                has_filter = any(isinstance(f, DebugAndTraceFilter) for f in getattr(handler, "filters", []))
                if not has_filter:
                    handler.addFilter(DebugAndTraceFilter(level=logging.DEBUG))

            def _is_debug_handler(handler: logging.Handler) -> bool:
                base = getattr(handler, "baseFilename", "")
                return bool(base) and "aios_debug" in os.path.basename(str(base))

            queue_mode = any(isinstance(h, QueueHandler) for h in root_logger.handlers)

            if queue_mode and _QUEUE_LISTENERS:
                debug_handlers: list[tuple[logging.handlers.QueueListener, NonBlockingRotatingFileHandler]] = []
                for listener in list(_QUEUE_LISTENERS):
                    handlers = list(getattr(listener, "handlers", ()))
                    logger.debug(
                        "QueueListener %s has handlers: %s",
                        listener,
                        [type(h).__name__ for h in handlers],
                    )
                    if not handlers:
                        logger.warning("QueueListener has no handlers; regular log file cannot be created")
                    for existing in handlers:
                        candidate = getattr(existing, "target", existing)
                        if isinstance(candidate, NonBlockingTimedRotatingFileHandler):
                            logger.debug(
                                "Standard handler state: base=%s level=%s flushLevel=%s interval=%s",
                                getattr(candidate, "baseFilename", "<unknown>"),
                                logging.getLevelName(candidate.level),
                                getattr(existing, "flushLevel", None),
                                getattr(existing, "flushInterval", None),
                            )
                        if isinstance(candidate, NonBlockingRotatingFileHandler):
                            logger.debug(
                                "Debug handler state: base=%s level=%s flushLevel=%s interval=%s",
                                getattr(candidate, "baseFilename", "<unknown>"),
                                logging.getLevelName(candidate.level),
                                getattr(existing, "flushLevel", None),
                                getattr(existing, "flushInterval", None),
                            )
                    has_standard = False
                    for existing in handlers:
                        target = getattr(existing, "target", None)
                        if isinstance(existing, AsyncMemoryHandler) and isinstance(target, NonBlockingTimedRotatingFileHandler):
                            existing.flushLevel = logging.INFO
                            existing.flushInterval = min(getattr(existing, "flushInterval", 5.0), 1.0)
                            existing.flush()
                        if isinstance(existing, NonBlockingTimedRotatingFileHandler) or (
                            isinstance(target, NonBlockingTimedRotatingFileHandler)
                        ):
                            has_standard = True
                            break

                    if not has_standard:
                        # Make sure the primary timed rotating file handler remains reachable via the listener queue.
                        os.makedirs("logs", exist_ok=True)
                        standard = NonBlockingTimedRotatingFileHandler(
                            filename="logs/aios.log",
                            when="midnight",
                            interval=1,
                            backupCount=10,
                            encoding="utf-8",
                            utc=False,
                            maxBytes=20971520,
                            include_tracebacks=False,
                        )
                        standard.setLevel(logging.INFO)
                        standard.setFormatter(
                            logging.Formatter(
                                "%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
                                "%Y-%m-%d %H:%M:%S",
                            )
                        )
                        logger.debug("Created NonBlockingTimedRotatingFileHandler for aios.log")

                        memory_wrapper = AsyncMemoryHandler(
                            capacity=5000,
                            flushLevel=logging.INFO,
                            target=standard,
                            flushOnClose=True,
                            flushInterval=1.0,
                        )
                        memory_wrapper.setLevel(standard.level)
                        logger.debug(
                            "Wrapped standard handler in AsyncMemoryHandler (flushLevel=%s, interval=%s)",
                            logging.getLevelName(memory_wrapper.flushLevel),
                            getattr(memory_wrapper, "flushInterval", None),
                        )

                        listener.stop()
                        handlers.append(memory_wrapper)
                        listener.handlers = tuple(handlers)
                        listener.start()
                        logger.info("Added standard log file handler (aios.log)")
                        handlers = list(getattr(listener, "handlers", ()))
                        if not handlers:
                            logger.warning("Standard handler failed to attach to QueueListener")
                        for wrapper in handlers:
                            if isinstance(wrapper, AsyncMemoryHandler):
                                logger.debug(
                                    "Post-add memory handler flushLevel=%s interval=%s buffer=%s",
                                    logging.getLevelName(wrapper.flushLevel),
                                    getattr(wrapper, "flushInterval", None),
                                    len(getattr(wrapper, "buffer", [])),
                                )

                    for handler in handlers:
                        candidate = handler
                        if isinstance(handler, AsyncMemoryHandler):
                            target = getattr(handler, "target", None)
                            if isinstance(target, logging.Handler):
                                handler.setLevel(target.level)
                            if isinstance(handler.target, NonBlockingRotatingFileHandler):
                                handler.flushLevel = logging.DEBUG
                                handler.flushInterval = min(getattr(handler, "flushInterval", 5.0), 1.0)
                                handler.flush()
                        if not isinstance(handler, NonBlockingRotatingFileHandler):
                            candidate = getattr(handler, "target", None)
                        if isinstance(candidate, NonBlockingRotatingFileHandler) and _is_debug_handler(candidate):
                            debug_handlers.append((listener, candidate))

                if enable_debug:
                    if debug_handlers:
                        for _listener, handler in debug_handlers:
                            handler.setLevel(logging.DEBUG)
                            _ensure_formatter(handler)
                            _ensure_debug_filter(handler)
                        self._debug_file_handler = debug_handlers[0][1]
                        logger.debug("Re-enabled existing debug file handler (async mode)")
                        return
                    else:
                        debug_base = NonBlockingRotatingFileHandler(
                            filename="logs/aios_debug.log",
                            maxBytes=20971520,
                            backupCount=10,
                        )
                        _ensure_formatter(debug_base)
                        _ensure_debug_filter(debug_base)
                        debug_base.setLevel(logging.DEBUG)

                        listener = _QUEUE_LISTENERS[0]
                        listener.stop()
                        handlers = list(getattr(listener, "handlers", ()))
                        memory_wrapper = AsyncMemoryHandler(
                            capacity=1000,
                            flushLevel=logging.DEBUG,
                            target=debug_base,
                            flushOnClose=True,
                            flushInterval=1.0,
                        )
                        memory_wrapper.setLevel(debug_base.level)
                        logger.debug(
                            "Wrapped debug handler in AsyncMemoryHandler (flushLevel=%s, interval=%s)",
                            logging.getLevelName(memory_wrapper.flushLevel),
                            getattr(memory_wrapper, "flushInterval", None),
                        )
                        handlers.append(memory_wrapper)
                        listener.handlers = tuple(handlers)
                        listener.start()

                        self._debug_file_handler = debug_base
                        action = "added"
                        elapsed = time.perf_counter() - start_time
                        logger.info(f"Added debug file handler in {elapsed:.3f}s: logs/aios_debug.log")
                        return
                else:
                    if debug_handlers:
                        for _listener, handler in debug_handlers:
                            handler.setLevel(logging.CRITICAL + 1)
                        self._debug_file_handler = debug_handlers[0][1]
                        logger.info("Disabled debug file handler output (async mode)")
                        action = "disabled"
                    return

            # Fallback: async logging disabled, manage handlers directly on loggers.
            NBHandler = NonBlockingRotatingFileHandler

            existing_handler = None
            for handler in list(root_logger.handlers) + list(aios_logger.handlers):
                if isinstance(handler, NBHandler) and _is_debug_handler(handler):
                    existing_handler = handler
                    break

            if enable_debug:
                debug_handler = existing_handler or self._debug_file_handler
                if debug_handler is None:
                    os.makedirs("logs", exist_ok=True)
                    debug_handler = NBHandler(
                        filename="logs/aios_debug.log",
                        maxBytes=20971520,
                        backupCount=10,
                    )
                    action = "added"
                else:
                    action = "enabled"

                debug_handler.setLevel(logging.DEBUG)
                _ensure_formatter(debug_handler)
                _ensure_debug_filter(debug_handler)

                if debug_handler not in aios_logger.handlers:
                    aios_logger.addHandler(debug_handler)
                if debug_handler not in root_logger.handlers:
                    root_logger.addHandler(debug_handler)

                self._debug_file_handler = debug_handler

            else:
                if existing_handler:
                    existing_handler.setLevel(logging.CRITICAL + 1)
                    self._debug_file_handler = existing_handler
                    action = "disabled"
                return

        except Exception:
            action = "error"
            logger.exception("Failed to manage debug file handler")
        finally:
            elapsed = time.perf_counter() - start_time
            if action != "error":
                logger.debug(
                    "Debug file handler action=%s (enable_debug=%s) in %.3fs",
                    action,
                    enable_debug,
                    elapsed,
                )
