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

try:
    from aios.system import paths as system_paths
except Exception:  # pragma: no cover - GUI fallback
    system_paths = None


def _resolve_log_dir() -> str:
    if system_paths is not None:
        try:
            return str(system_paths.get_logs_dir())
        except Exception:
            logger = logging.getLogger(__name__)
            logger.debug("Failed to resolve logs dir via helper", exc_info=True)
    return "logs"


LOG_DIR = _resolve_log_dir()
os.makedirs(LOG_DIR, exist_ok=True)

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
        app: Any | None = None,
    ) -> None:
        self.parent = parent
        self._save_state_fn = save_state_fn
        self._chat_panel = chat_panel
        self._help_panel = help_panel
        self._debug_panel = debug_panel
        self._worker_pool = worker_pool
        self._app = app
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
        
        # Left column: Appearance, General Settings, Help, Support
        ui_builders.create_appearance_section(self, left_column)
        ui_builders.create_general_settings_section(self, left_column)
        ui_builders.create_help_section(self, left_column)
        ui_builders.create_support_section(self, left_column)
        
        # Right column: Logging, Cache, Dataset Storage
        ui_builders.create_logging_section(self, right_column)
        ui_builders.create_cache_section(self, right_column)
        ui_builders.create_dataset_storage_section(self, right_column)
        
        # Load initial settings (non-blocking only)
        self._load_startup_status()
        self._load_cache_size()
        self._refresh_cache_stats()
        # NOTE: _load_dataset_cap() and _load_artifacts_path() are called via 
        # load_settings_deferred() after mainloop starts to avoid blocking
        # the UI thread with CLI subprocess calls during initialization.

    def _apply_theme(self, theme: str) -> None:
        """Apply the selected theme to the application."""
        theme_manager.apply_theme(self, theme)

    def load_settings_deferred(self) -> None:
        """Load settings that require CLI calls (deferred to after mainloop starts).
        
        This method is called from panel_setup after mainloop is running to avoid
        blocking the UI thread with subprocess calls during initialization.
        """
        try:
            logger.debug("Loading deferred settings (dataset cap, artifacts path)...")
            self._load_dataset_cap()
            self._load_artifacts_path()
            logger.debug("Deferred settings loaded successfully")
        except Exception as e:
            logger.error(f"Error loading deferred settings: {e}", exc_info=True)

    def _open_kofi_link(self) -> None:
        """Open Ko-fi link in default browser."""
        try:
            webbrowser.open("https://ko-fi.com/wulfic")
        except Exception as e:
            logger.error(f"Error opening Ko-fi link: {e}")

    def _open_github_link(self) -> None:
        """Open GitHub repository in default browser (Phase 2.4)."""
        try:
            webbrowser.open("https://github.com/Wulfic/AI-OS")
        except Exception as e:
            logger.error(f"Error opening GitHub link: {e}")

    def _load_dataset_cap(self) -> None:
        """Load dataset cap from config (Phase 3.1)."""
        try:
            # Prevent re-entry and duplicate calls
            if getattr(self, '_dataset_cap_loaded', False):
                return
            self._dataset_cap_loaded = True
            
            # Flag to prevent save during initial load
            self._loading_dataset_cap = True
            
            # Get cap from CLI (use allow_sync_cli for initialization)
            if hasattr(self, "_app") and hasattr(self._app, "_run_cli"):
                with self._app.allow_sync_cli(reason="settings_panel_init"):
                    result_str = self._app._run_cli(["datasets-get-cap"])
                    # Parse the CLI output string to dictionary
                    result = self._app._parse_cli_dict(result_str) if hasattr(self._app, "_parse_cli_dict") else {}
                    if result and "cap_gb" in result:
                        cap = result.get("cap_gb", 0)
                        if cap and cap > 0:
                            self.dataset_cap_var.set(str(cap))
                        else:
                            self.dataset_cap_var.set("")
            
            # Setup auto-save on change (with debouncing) - only once
            if not getattr(self, '_cap_trace_added', False):
                self._cap_trace_added = True
                self._cap_save_pending = False
                self._last_cap_value = self.dataset_cap_var.get()
                
                def _on_cap_change(*args):
                    # Skip if loading or value unchanged
                    if getattr(self, '_loading_dataset_cap', False):
                        return
                    new_val = self.dataset_cap_var.get()
                    if new_val == getattr(self, '_last_cap_value', ''):
                        return
                    self._last_cap_value = new_val
                    
                    # Debounce: only save after 1000ms of no changes
                    if getattr(self, '_cap_save_pending', False):
                        return
                    self._cap_save_pending = True
                    
                    def _do_save():
                        self._cap_save_pending = False
                        self._save_dataset_cap()
                    
                    if hasattr(self, '_app') and hasattr(self._app, 'root'):
                        self._app.root.after(1000, _do_save)
                    else:
                        _do_save()
                
                self.dataset_cap_var.trace_add("write", _on_cap_change)
            
            # Mark loading complete
            self._loading_dataset_cap = False
            
            # Refresh usage display only once (async to not block)
            if hasattr(self, '_app') and hasattr(self._app, 'root'):
                if not getattr(self, '_usage_refresh_scheduled', False):
                    self._usage_refresh_scheduled = True
                    self._app.root.after(500, self._refresh_dataset_usage)
        except Exception as e:
            self._loading_dataset_cap = False
            logger.error(f"Error loading dataset cap: {e}", exc_info=True)

    def _save_dataset_cap(self) -> None:
        """Save dataset cap to config (Phase 3.1)."""
        try:
            if not hasattr(self, "_app") or not hasattr(self._app, "_run_cli"):
                return
            
            cap_str = self.dataset_cap_var.get().strip()
            if cap_str:
                try:
                    cap_gb = float(cap_str)
                    if cap_gb > 0:
                        with self._app.allow_sync_cli(reason="save_dataset_cap"):
                            self._app._run_cli(["datasets-set-cap", str(cap_gb)])
                        logger.info(f"Dataset cap set to {cap_gb} GB")
                        # Schedule refresh only if not already scheduled
                        if hasattr(self, '_app') and hasattr(self._app, 'root'):
                            if not getattr(self, '_usage_refresh_scheduled', False):
                                self._usage_refresh_scheduled = True
                                self._app.root.after(500, self._refresh_dataset_usage)
                except ValueError:
                    logger.warning(f"Invalid dataset cap value: {cap_str}")
            else:
                # Empty = unlimited
                with self._app.allow_sync_cli(reason="save_dataset_cap_unlimited"):
                    self._app._run_cli(["datasets-set-cap", "0"])
                logger.info("Dataset cap set to unlimited")
                # Schedule refresh only if not already scheduled
                if hasattr(self, '_app') and hasattr(self._app, 'root'):
                    if not getattr(self, '_usage_refresh_scheduled', False):
                        self._usage_refresh_scheduled = True
                        self._app.root.after(500, self._refresh_dataset_usage)
        except Exception as e:
            logger.error(f"Error saving dataset cap: {e}", exc_info=True)

    def _refresh_dataset_usage(self) -> None:
        """Refresh dataset storage usage display (Phase 3.1)."""
        try:
            # Clear the scheduled flag
            self._usage_refresh_scheduled = False
            
            if not hasattr(self, "_app") or not hasattr(self._app, "_run_cli"):
                return
            
            # Use the correct CLI command: datasets-stats
            with self._app.allow_sync_cli(reason="refresh_dataset_usage"):
                result_str = self._app._run_cli(["datasets-stats"])
            # Parse the JSON response from the CLI output
            import json
            result = None
            try:
                # Try to parse as JSON directly
                result = json.loads(result_str)
            except Exception:
                # If direct parse fails, try to extract JSON from CLI output
                import re
                match = re.search(r'\{.*\}', result_str, re.DOTALL)
                if match:
                    result = json.loads(match.group(0))
            
            if result and isinstance(result, dict):
                usage = result.get("usage_gb", 0)
                cap = result.get("cap_gb", 0)
                
                if cap and cap > 0:
                    usage_text = f"Using {usage:.2f} GB / {cap:.2f} GB"
                else:
                    usage_text = f"Using {usage:.2f} GB (unlimited)"
                
                if hasattr(self, "dataset_usage_label"):
                    self.dataset_usage_label.config(text=usage_text)
        except Exception as e:
            logger.debug(f"Error refreshing dataset usage: {e}")

    def _resolve_default_artifacts_dir(self) -> str:
        """Resolve the default artifacts directory (Phase 3.2)."""
        try:
            from aios.system import paths as system_paths
            if system_paths is not None:
                try:
                    return str(system_paths.get_artifacts_root())
                except Exception:
                    logger.debug("Failed to resolve ProgramData artifacts root", exc_info=True)
        except ImportError:
            logger.debug("aios.system.paths not available, using fallback", exc_info=True)
        try:
            from pathlib import Path
            return str(Path(__file__).resolve().parents[5] / "artifacts")
        except Exception:
            from pathlib import Path
            return str(Path.cwd() / "artifacts")

    def _set_artifacts_status(self, message: str, color: str = "gray") -> None:
        """Set artifacts directory status message (Phase 3.2)."""
        try:
            self._artifacts_status_var.set(message)
            if self._artifacts_status_label is not None:
                self._artifacts_status_label.configure(foreground=color)
        except Exception:
            logger.debug("Failed to update artifacts status label", exc_info=True)

    def _probe_directory_writable(self, path) -> str | None:
        """Test if directory is writable (Phase 3.2)."""
        from pathlib import Path
        import os
        
        path = Path(path)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return f"Failed to create directory: {exc}"

        probe = path / f".aios-write-test-{os.getpid()}"
        try:
            probe.write_text("ok", encoding="utf-8")
        except Exception as exc:
            try:
                probe.unlink(missing_ok=True)
            except Exception:
                pass
            return f"Failed to write probe file: {exc}"

        try:
            probe.unlink(missing_ok=True)
        except Exception:
            pass
        return None

    def _apply_artifacts_override(self, path) -> None:
        """Apply artifacts directory override (Phase 3.2)."""
        try:
            from aios.system.paths import system_paths
        except ImportError:
            import aios.system.paths as system_paths
        import os
        
        if system_paths is not None:
            try:
                system_paths.set_artifacts_root_override(path)
            except Exception:
                logger.debug("Failed to apply artifacts override", exc_info=True)

        if path is None:
            os.environ.pop("AIOS_ARTIFACTS_DIR", None)
        else:
            os.environ["AIOS_ARTIFACTS_DIR"] = str(path)

    def _validate_artifacts_dir(self, value: str | None = None, *, apply_override: bool = True) -> bool:
        """Validate artifacts directory path (Phase 3.2)."""
        from pathlib import Path
        try:
            from aios.system.paths import system_paths
        except ImportError:
            import aios.system.paths as system_paths
        
        if not hasattr(self, "_artifacts_default_dir"):
            self._artifacts_default_dir = self._resolve_default_artifacts_dir()
        
        raw_value = (value if value is not None else self.artifacts_dir_var.get() or "").strip()
        if not raw_value:
            self._artifacts_dir_is_valid = True
            self._set_artifacts_status(f"Using default path: {self._artifacts_default_dir}", "gray")
            if apply_override:
                self._apply_artifacts_override(None)
            return True

        candidate = Path(raw_value).expanduser()
        if not candidate.is_absolute():
            self._artifacts_dir_is_valid = False
            self._set_artifacts_status("Enter an absolute path (e.g., D:\\AI-OS\\artifacts)", "#c0392b")
            return False

        if system_paths is not None:
            try:
                error = system_paths.test_directory_writable(candidate)
            except Exception as exc:
                error = str(exc)
        else:
            error = self._probe_directory_writable(candidate)

        if error:
            self._artifacts_dir_is_valid = False
            self._set_artifacts_status(error, "#c0392b")
            return False

        self._artifacts_dir_is_valid = True
        self._set_artifacts_status(f"Custom path OK: {candidate}", "#1d8348")
        if apply_override:
            self._apply_artifacts_override(candidate)
        return True

    def _browse_artifacts_dir(self) -> None:
        """Browse for artifacts directory (Phase 3.2)."""
        from tkinter import filedialog
        
        if not hasattr(self, "_artifacts_default_dir"):
            self._artifacts_default_dir = self._resolve_default_artifacts_dir()
        
        initial = self.artifacts_dir_var.get().strip() or self._artifacts_default_dir
        try:
            selected = filedialog.askdirectory(initialdir=initial, title="Select artifacts directory")
        except Exception:
            logger.warning("Failed to open artifacts folder picker", exc_info=True)
            return

        if selected:
            self.artifacts_dir_var.set(selected)
            self._validate_artifacts_dir()

    def _reset_artifacts_dir(self) -> None:
        """Reset artifacts directory to default (Phase 3.2)."""
        self.artifacts_dir_var.set("")
        self._validate_artifacts_dir()

    def _load_artifacts_path(self) -> None:
        """Load artifacts directory configuration (Phase 3.2)."""
        try:
            # Initialize defaults
            self._artifacts_default_dir = self._resolve_default_artifacts_dir()
            self._artifacts_dir_is_valid = True
            
            # Check for existing override from environment
            import os
            override = os.environ.get("AIOS_ARTIFACTS_DIR", "").strip()
            if override:
                self.artifacts_dir_var.set(override)
            
            # Validate initial path
            self._validate_artifacts_dir(apply_override=False)
            
            # Setup auto-save on change
            def _on_artifacts_change(*args):
                if not self._restoring_state:
                    self._validate_artifacts_dir()
            
            self.artifacts_dir_var.trace_add("write", _on_artifacts_change)
            
        except Exception as e:
            logger.error(f"Error loading artifacts path: {e}", exc_info=True)

    def _browse_download_location(self) -> None:
        """Browse for download location (Phase 3.3)."""
        from tkinter import filedialog
        
        initial = self.download_location_var.get().strip() or "training_datasets"
        try:
            selected = filedialog.askdirectory(initialdir=initial, title="Select dataset download location")
        except Exception:
            logger.warning("Failed to open download location picker", exc_info=True)
            return

        if selected:
            self.download_location_var.set(selected)
            logger.info(f"Download location set to: {selected}")

    def _reset_download_location(self) -> None:
        """Reset download location to default (Phase 3.3)."""
        self.download_location_var.set("training_datasets")
        logger.info("Download location reset to default: training_datasets")

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
                from ..help_panel import utils
                project_root = utils.find_project_root(Path(__file__))
                docs_root = utils.resolve_docs_root(project_root)

                index_file = docs_root / "search_index.json"
                if index_file.exists():
                    index_file.unlink()
                    logger.info("Deleted old search index")

                from ..help_panel.search_engine import SearchEngine
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
            "download_location": self.download_location_var.get() if hasattr(self, 'download_location_var') else "training_datasets",
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
                from ...utils.startup import set_startup_enabled, is_windows, is_linux
                if is_windows() or is_linux():
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
            download_location = state.get("download_location")
            if download_location and hasattr(self, 'download_location_var'):
                self.download_location_var.set(download_location)
                logger.info(f"Restored download location: {download_location}")
        except Exception as e:
            logger.error(f"Failed to restore download location: {e}")
        
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
                        os.makedirs(LOG_DIR, exist_ok=True)
                        standard = NonBlockingTimedRotatingFileHandler(
                            filename=os.path.join(LOG_DIR, "aios.log"),
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
                            filename=os.path.join(LOG_DIR, "aios_debug.log"),
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
                        logger.info(
                            "Added debug file handler in %.3fs: %s",
                            elapsed,
                            os.path.join(LOG_DIR, "aios_debug.log"),
                        )
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
                    os.makedirs(LOG_DIR, exist_ok=True)
                    debug_handler = NBHandler(
                        filename=os.path.join(LOG_DIR, "aios_debug.log"),
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

    def _clear_old_logs(self) -> None:
        """Clear old log files (non-current session).
        
        Deletes rotated/archived log files to free disk space while preserving
        the current session's log files.
        """
        try:
            from tkinter import messagebox
            import glob
            
            logger.info("User action: Clearing old log files")
            
            # Get all log files in the logs directory
            log_files = []
            patterns = [
                os.path.join(LOG_DIR, "aios.log.*"),  # Rotated standard logs
                os.path.join(LOG_DIR, "aios_debug.log.*"),  # Rotated debug logs
            ]
            
            for pattern in patterns:
                log_files.extend(glob.glob(pattern))
            
            if not log_files:
                messagebox.showinfo(
                    "Clear Old Logs",
                    "No old log files found to clear.\n\n"
                    "Only rotated/archived logs (e.g., aios.log.1, aios.log.2) are cleared.\n"
                    "Current session logs are always preserved."
                )
                return
            
            # Ask for confirmation
            file_count = len(log_files)
            total_size = sum(os.path.getsize(f) for f in log_files if os.path.exists(f))
            size_mb = total_size / (1024 * 1024)
            
            ok = messagebox.askyesno(
                "Clear Old Logs",
                f"Found {file_count} old log file(s) totaling {size_mb:.2f} MB.\n\n"
                "This will delete rotated/archived logs from previous sessions.\n"
                "Current session logs (aios.log, aios_debug.log) will be preserved.\n\n"
                "Continue?"
            )
            
            if not ok:
                logger.info("User cancelled log clearing operation")
                return
            
            # Delete old log files
            deleted = 0
            errors = []
            for log_file in log_files:
                try:
                    os.remove(log_file)
                    deleted += 1
                    logger.debug(f"Deleted old log file: {log_file}")
                except Exception as e:
                    errors.append(f"{os.path.basename(log_file)}: {e}")
                    logger.warning(f"Failed to delete {log_file}: {e}")
            
            # Show result
            if errors:
                messagebox.showwarning(
                    "Clear Old Logs",
                    f"Deleted {deleted} of {file_count} log files.\n\n"
                    f"Failed to delete {len(errors)} file(s):\n" +
                    "\n".join(errors[:5]) +
                    ("\n..." if len(errors) > 5 else "")
                )
            else:
                messagebox.showinfo(
                    "Clear Old Logs",
                    f"Successfully deleted {deleted} old log file(s) ({size_mb:.2f} MB freed)."
                )
            
            logger.info(f"Cleared {deleted} old log files, {len(errors)} errors")
            
            # Update log size display
            self._update_log_size()
            
        except Exception as e:
            logger.error(f"Error clearing old logs: {e}", exc_info=True)
            try:
                from tkinter import messagebox
                messagebox.showerror("Clear Old Logs", f"Error: {e}")
            except Exception:
                pass

    def _update_log_size(self) -> None:
        """Update the log folder size indicator."""
        def _calculate():
            try:
                total_size = 0
                file_count = 0
                
                # Walk through logs directory
                if os.path.isdir(LOG_DIR):
                    for root, dirs, files in os.walk(LOG_DIR):
                        for file in files:
                            if file.endswith('.log') or '.log.' in file:
                                file_path = os.path.join(root, file)
                                try:
                                    total_size += os.path.getsize(file_path)
                                    file_count += 1
                                except Exception:
                                    pass
                
                # Format size
                size_mb = total_size / (1024 * 1024)
                size_str = f"Log Folder: {size_mb:.2f} MB ({file_count} files)"
                
                # Update UI on main thread
                if hasattr(self, 'log_size_var'):
                    try:
                        self.log_size_var.set(size_str)
                    except Exception:
                        pass
                
                logger.debug(f"Log folder size: {size_mb:.2f} MB ({file_count} files)")
                
            except Exception as e:
                logger.warning(f"Error calculating log size: {e}")
                if hasattr(self, 'log_size_var'):
                    try:
                        self.log_size_var.set("Log Folder: ? MB")
                    except Exception:
                        pass
        
        # Run in background
        if hasattr(self, '_worker_pool'):
            try:
                submit_background("log-size-calc", _calculate, pool=self._worker_pool)
            except Exception:
                _calculate()  # Fallback to sync
        else:
            _calculate()
