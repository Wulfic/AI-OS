from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess as _sp
import sys
import threading
import queue
import urllib.request as _urlreq
import urllib.error as _urlerr
from pathlib import Path
import logging
from typing import Any, cast

# New modular components
from .components import OutputPanel, ChatPanel, RichChatPanel, DatasetBuilderPanel, ResourcesPanel, StatusBar, DebugPanel, BrainsPanel, HRMTrainingPanel, EvaluationPanel, DatasetDownloadPanel, SettingsPanel, MCPManagerPanel

# Optional imports: allow module import without Tk installed (for CI/tests)
try:  # pragma: no cover - environment dependent
    import tkinter as tk  # type: ignore
    from tkinter import filedialog  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    # Headless fallback: set to None so run() can no-op cleanly
    tk = cast(Any, None)
    filedialog = cast(Any, None)
    ttk = cast(Any, None)

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore


from .mixins.cli_bridge import CliBridgeMixin
from .mixins.debug import DebugMixin
from .services import SystemStatusUpdater, save_app_state, load_app_state, chat_route, render_chat_output, LogRouter, LogCategory
from .utils.resource_management import ManagedThread, ProcessReaper, TimerManager, ResourceMonitor, AsyncWorkerPool, AsyncEventLoop
from aios.cli.utils import load_config
from aios.memory.store import db_connection

# Auto-training orchestrator components
from aios.core.datasets.registry import DatasetRegistry
from aios.core.hrm_models.expert_metadata import ExpertRegistry
from aios.core.auto_training.orchestrator import AutoTrainingOrchestrator, NoDatasetFoundError

# Tray utilities
from .utils.tray import TrayManager


logger = logging.getLogger(__name__)


class AiosTkApp(DebugMixin, CliBridgeMixin):
    # Predeclare dynamic UI attributes for static analyzers
    known_ds_var: Any
    known_ds_combo: Any
    goals_panel: Any
    _known_ds_items: list[dict] | None
    _known_ds_cache: list[dict] | None
    _download_thread: threading.Thread | None
    _download_cancel: threading.Event | None

    def __init__(self, root: "tk.Tk", start_minimized: bool = False) -> None:  # type: ignore[name-defined]
        if tk is None:
            raise RuntimeError("Tkinter is not available in this environment")
        self.root = root
        self.root.title("AI-OS Control Panel")
        self._start_minimized = start_minimized
        self._tray_manager: Any = None
        self._minimize_to_tray_on_close = False
        
        # Resource management utilities
        self._managed_threads: list[ManagedThread] = []
        self._process_reaper = ProcessReaper()
        self._timer_manager = TimerManager(root)
        self._resource_monitor = ResourceMonitor()
        
        # Async worker pool (configurable via environment variable)
        # Default: (cpu_count * 2) + 1 for I/O-bound GUI operations
        worker_count = None
        try:
            env_workers = os.environ.get("AIOS_WORKER_THREADS")
            if env_workers and env_workers.isdigit():
                worker_count = int(env_workers)
                logger.info(f"Using custom worker count from AIOS_WORKER_THREADS: {worker_count}")
        except Exception:
            pass
        
        self._worker_pool = AsyncWorkerPool(max_workers=worker_count)
        logger.info(f"Initialized worker pool with {self._worker_pool.max_workers} workers")
        
        # Async event loop for GUI responsiveness
        self._async_loop = AsyncEventLoop()
        self._async_loop.start()
        logger.info("Async event loop started for GUI responsiveness")
        
        # Register emergency cleanup
        import atexit
        atexit.register(self._emergency_cleanup)
        
        # Set window icon
        try:
            # Get the path to the icon files
            icon_dir = Path(__file__).parent.parent.parent.parent / "installers"
            ico_path = icon_dir / "AI-OS.ico"
            png_path = icon_dir / "AI-OS.png"
            
            # Try .ico for Windows
            if ico_path.exists():
                try:
                    self.root.iconbitmap(str(ico_path))
                except Exception:
                    pass
            
            # Try .png as fallback (for Linux/macOS)
            if png_path.exists():
                try:
                    from PIL import Image, ImageTk  # type: ignore
                    img = Image.open(png_path)
                    photo = ImageTk.PhotoImage(img)
                    self.root.iconphoto(True, photo)
                except Exception:
                    pass
        except Exception:
            # Silently fail if icons can't be loaded
            pass
        
        # Start maximized
        try:
            self.root.state('zoomed')  # Windows
        except Exception:
            try:
                self.root.attributes('-zoomed', True)  # Linux
            except Exception:
                pass
        # Graceful close protocol
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)  # type: ignore[attr-defined]
        except Exception:
            pass
        # Route Tk callback exceptions to our debug panel instead of stderr
        try:
            # type: ignore[attr-defined]
            self.root.report_callback_exception = self._tk_exception_hook  # pyright: ignore[reportAttributeAccessIssue]
        except Exception:
            pass

        # Internal state - use platform-appropriate config directory
        if sys.platform == "win32":
            # Windows: Use %LOCALAPPDATA%\aios
            appdata = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
            config_dir = Path(appdata) / "aios"
        else:
            # Unix-like: Use ~/.config/aios
            config_dir = Path(os.path.expanduser("~/.config/aios"))
        
        config_dir.mkdir(parents=True, exist_ok=True)
        self._state_path: Path = config_dir / "gui_state.json"
        self._save_after_id: int | None = None
        self._raw_buffer: list[str] = []
        self._raw_max: int = 5000
        self._proc: _sp.Popen | None = None
        self._q: queue.Queue[str] | None = None
        self._known_ds_items = []
        self._known_ds_cache = None
        self._download_thread = None
        self._download_cancel = None

        # Persistent chat router to keep model loaded
        self._chat_router: Any = None
        self._chat_registry: Any = None
        
        # Auto-training orchestrator for learning intent detection
        self._orchestrator: Any = None
        
        # Initialize centralized log router
        self._log_router = LogRouter()
        
        # Suppress verbose transformers warnings
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

        # Notebook: Control + Chat + Resources + Goals pages
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True)
        
        # Create all tab frames
        self.chat_tab = ttk.Frame(self.nb)
        self.brains_tab = ttk.Frame(self.nb)
        self.main_tab = ttk.Frame(self.nb)
        self.hrm_train_tab = ttk.Frame(self.nb)
        self.evaluation_tab = ttk.Frame(self.nb)
        self.resources_tab = ttk.Frame(self.nb)
        self.mcp_tab = ttk.Frame(self.nb)
        self.settings_tab = ttk.Frame(self.nb)
        self.debug_tab = ttk.Frame(self.nb)
        
        # Add tabs in desired order: Chat, Brains, Datasets, HRM Training, Evaluation, Resources, MCP and Tools, Settings, Debug
        self.nb.add(self.chat_tab, text="Chat")
        self.nb.add(self.brains_tab, text="Brains")
        self.nb.add(self.main_tab, text="Datasets")
        self.nb.add(self.hrm_train_tab, text="HRM Training")
        self.nb.add(self.evaluation_tab, text="Evaluation")
        self.nb.add(self.resources_tab, text="Resources")
        self.nb.add(self.mcp_tab, text="MCP & Tools")
        self.nb.add(self.settings_tab, text="Settings")
        self.nb.add(self.debug_tab, text="Debug")
        # refresh brains when returning to Brains tab
        def _on_tab_changed(event):
            try:
                tab_id = self.nb.select()
                tab_text = self.nb.tab(tab_id, "text")
                if tab_text == "Brains":
                    try:
                        self.brains_panel.refresh()  # populated later
                    except Exception:
                        pass
                elif tab_text == "MCP & Tools":
                    try:
                        self.mcp_panel.refresh()  # populated later
                    except Exception:
                        pass
            except Exception:
                pass
        cast(Any, self.nb).bind("<<NotebookTabChanged>>", _on_tab_changed)

        # --- Top actions bar (Datasets tab) ---
        # Simplified: remove Status/Artifacts/Budgets buttons from this page
        # (Controls remain available on their dedicated tabs if needed.)
        top = ttk.Frame(self.main_tab)
        top.pack(fill="x", padx=8, pady=4)
        # First page refocus: remove Train button; dataset building lives here, HRM Training has its own tab

        # Core toggles/state (no direct UI here; ResourcesPanel handles devices)
        self.cpu_var = tk.BooleanVar(value=False)
        self.cuda_var = tk.BooleanVar(value=False)
        self.xpu_var = tk.BooleanVar(value=False)
        self.dml_var = tk.BooleanVar(value=False)
        self.mps_var = tk.BooleanVar(value=False)
        self.dml_py_var = tk.StringVar(value="")
        self.dataset_path_var = tk.StringVar(value="")

        # Resource controls state
        try:
            cores = max(1, os.cpu_count() or 1)
        except Exception:
            cores = 1

        # Command var used by on_run (legacy)
        self.cmd_var = tk.StringVar(value="")

        # Removed legacy Datasets and Crawl panels from first page per redesign

        # Create horizontal split container for output and builder
        top_container = ttk.Frame(self.main_tab)
        top_container.pack(fill="both", expand=False, padx=8, pady=(4, 8))
        
        # Configure grid weights for 50/50 split
        top_container.grid_columnconfigure(0, weight=1)
        top_container.grid_columnconfigure(1, weight=1)
        
        # Left side: Output box (will be created by OutputPanel)
        left_frame = ttk.Frame(top_container)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        
        # Right side: Dataset Builder
        right_frame = ttk.Frame(top_container)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        
        # Dataset Download panel - new feature for downloading training datasets
        # Create this first so we can use its output
        self.dataset_download_panel = DatasetDownloadPanel(
            self.main_tab,
            log_callback=self._log_router.log,
            output_parent=left_frame  # Pass left frame for output box placement
        )
        
        # Bridge to log router and dataset panel output
        self._append_out = lambda s: self._log_router.log(s, category=None)  # Auto-detect category
        self._update_out = lambda s: self.dataset_download_panel.log(s)  # Use dataset panel's output
        
        # Dataset Builder panel (images) - now in right side of top container
        self.dataset_builder_panel = DatasetBuilderPanel(
            right_frame,
            run_cli=self._run_cli,
            dataset_path_var=self.dataset_path_var,
            append_out=self._append_out,
            update_out=self._update_out,
            worker_pool=self._worker_pool,
        )

        # Resources panel
        def _detect_devices_info() -> dict:
            info = self._parse_cli_dict(self._run_cli(["torch-info"]) or "{}")
            return info if isinstance(info, dict) else {}

        def _apply_caps(dataset_cap_gb: float, model_cap_gb: float | None, per_brain_cap_gb: float | None) -> dict:
            out = {}
            try:
                if dataset_cap_gb and dataset_cap_gb > 0:
                    raw = self._run_cli(["datasets-set-cap", str(dataset_cap_gb)])
                    out["datasets"] = self._parse_cli_dict(raw)
            except Exception:
                pass
            try:
                # brains config_set expects GB values for storage_limit and per-brain width
                args = ["brains", "config-set"]
                args += ["--config", "config/default.yaml"]
                if model_cap_gb is not None and model_cap_gb > 0:
                    args += ["--storage_limit_gb", str(model_cap_gb)]
                if per_brain_cap_gb is not None and per_brain_cap_gb > 0:
                    args += ["--per_brain_limit_gb", str(per_brain_cap_gb)]
                if len(args) > 4:
                    raw2 = self._run_cli(args)
                    out["brains"] = self._parse_cli_dict(raw2)
            except Exception:
                pass
            return out

        def _fetch_caps() -> dict:
            data = {}
            try:
                ds = self._parse_cli_dict(self._run_cli(["datasets-stats"]) or "{}")
                if isinstance(ds, dict):
                    cap = ds.get("cap_gb")
                    if isinstance(cap, (int, float)):
                        data["dataset_cap_gb"] = float(cap)
            except Exception:
                pass
            try:
                cfg = self._parse_cli_dict(self._run_cli(["brains", "config-show"]) or "{}")
                brains = cfg.get("brains") if isinstance(cfg, dict) else {}
                if isinstance(brains, dict):
                    mg = brains.get("storage_limit_gb")
                    if not isinstance(mg, (int, float)):
                        mb = brains.get("storage_limit_mb")
                        if isinstance(mb, (int, float)):
                            mg = float(mb) / 1024.0
                    if isinstance(mg, (int, float)):
                        data["model_cap_gb"] = float(mg)
                    tovr = brains.get("trainer_overrides") or {}
                    if isinstance(tovr, dict):
                        pbg = tovr.get("width_storage_limit_gb")
                        if not isinstance(pbg, (int, float)):
                            pbm = tovr.get("width_storage_limit_mb")
                            if isinstance(pbm, (int, float)):
                                pbg = float(pbm) / 1024.0
                        if isinstance(pbg, (int, float)):
                            data["per_brain_cap_gb"] = float(pbg)
            except Exception:
                pass
            return data

        self.resources_panel = ResourcesPanel(
            self.resources_tab,
            cores=cores,
            detect_fn=_detect_devices_info,
            apply_caps_fn=_apply_caps,
            fetch_caps_fn=_fetch_caps,
            save_state_fn=self._save_state,
            root=root,
        )
        try:
            self.resources_panel.set_caps(_fetch_caps())
        except Exception:
            pass
        
        # CRITICAL FIX: Detect devices IMMEDIATELY before loading state
        # This ensures GPU rows exist when settings are applied, fixing persistence issues
        try:
            info = _detect_devices_info()
            if isinstance(info, dict):
                self.resources_panel.set_detected(info)
                logger.info(f"Initial device detection completed: {len(info.get('cuda_devices', []))} GPU(s) found")
        except Exception as e:
            logger.warning(f"Initial device detection failed: {e}")
            # Continue anyway - devices can be detected later manually

        # Debug panel
        self.debug_panel = DebugPanel(self.debug_tab)
        
        # Settings panel (chat_panel will be set after it's created)
        self.settings_panel = SettingsPanel(
            self.settings_tab, 
            save_state_fn=self._save_state,
            chat_panel=None
        )
        
        # Register debug panel as handler for all log categories
        for category in LogCategory:
            # Skip DATASET category for the debug panel initially - datasets have their own output
            # But still allow it to appear if user enables the filter
            if category != LogCategory.DATASET:
                self._log_router.register_handler(
                    category, 
                    lambda msg, lvl=None, cat=category: self.debug_panel.write(msg, cat.value, lvl)
                )
            else:
                # For datasets, route to debug panel with "dataset" category
                self._log_router.register_handler(
                    category, 
                    lambda msg, lvl=None: self.debug_panel.write(msg, "dataset", lvl)
                )

        # Bridge Python logging to the Debug tab so logs appear live in the GUI
        try:
            class _TkDebugHandler(logging.Handler):
                def __init__(self, root: "tk.Tk", log_router, err_cb):  # type: ignore[name-defined]
                    super().__init__()
                    self._root = root
                    self._log_router = log_router
                    self._err = err_cb
                    # Verbose, developer-friendly formatter
                    self.setFormatter(logging.Formatter(
                        "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
                    ))

                def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
                    try:
                        msg = self.format(record)
                    except Exception:
                        msg = record.getMessage()
                    
                    # Map Python logging level to our string format
                    level_name = record.levelname  # DEBUG, INFO, WARNING, ERROR, CRITICAL
                    
                    # Determine category based on log level and content
                    category = None
                    if record.levelno >= logging.ERROR:
                        category = LogCategory.ERROR
                    elif "chat" in record.name.lower() or "chat" in msg.lower():
                        category = LogCategory.CHAT
                    elif "train" in record.name.lower() or "hrm" in record.name.lower():
                        category = LogCategory.TRAINING
                    elif "dataset" in record.name.lower():
                        category = LogCategory.DATASET
                    else:
                        category = LogCategory.DEBUG
                    
                    def _do_write() -> None:
                        try:
                            # Pass the message with both category and log level
                            self._log_router.log(msg, category, level_name)
                            if record.levelno >= logging.ERROR:
                                # Show last error field for quick visibility
                                try:
                                    exc_text = getattr(record, "exc_text", None)
                                except Exception:
                                    exc_text = None
                                self._err(exc_text or msg)
                        except Exception:
                            pass
                    try:
                        self._root.after(0, _do_write)
                    except Exception:
                        _do_write()

            self._debug_log_handler = _TkDebugHandler(self.root, self._log_router, self.debug_panel.set_error)  # type: ignore[attr-defined]
            self._debug_log_handler.setLevel(logging.DEBUG)
            # Attach to the main project logger and root so we see everything
            logging.getLogger("aios").addHandler(self._debug_log_handler)
            logging.getLogger().addHandler(self._debug_log_handler)
        except Exception:
            pass

        # Chat panel (using RichChatPanel for enhanced content support)
        self.chat_panel = RichChatPanel(
            self.chat_tab, 
            self._on_chat_route_and_run,
            on_load_brain=self._on_load_brain,
            on_list_brains=self._on_list_brains,
            on_unload_model=self._on_unload_model,
            worker_pool=self._worker_pool,
        )
        
        # Now that chat_panel exists, connect it to settings panel for theme updates
        self.settings_panel._chat_panel = self.chat_panel

        # Brains panel (now includes goals management)
        self.brains_panel = BrainsPanel(
            self.brains_tab,
            run_cli=self._run_cli,
            append_out=self._append_out,
            on_goal_add=self._on_goal_add_for_brain,
            on_goals_list=self._on_goals_list_for_brain,
            on_goal_remove=self._on_goal_remove,
        )
        try:
            self.brains_panel.refresh()
        except Exception:
            pass

        # MCP Servers & Tools Manager panel
        self.mcp_panel = MCPManagerPanel(
            self.mcp_tab,
            run_cli=self._run_cli,
            append_out=self._append_out,
            save_state_fn=self._save_state,
        )
        try:
            self.mcp_panel.refresh()
        except Exception:
            pass

        # HRM Training panel
        try:
            self.hrm_training_panel = HRMTrainingPanel(self.hrm_train_tab, run_cli=self._run_cli, append_out=self._append_out, save_state_fn=self._save_state, worker_pool=self._worker_pool)
            # Provide resources panel so HRM training derives devices from global settings
            try:
                self.hrm_training_panel._resources_panel = self.resources_panel  # type: ignore[attr-defined]
            except Exception:
                pass
            # Provide goal callbacks for auto-adding default goals on training start
            try:
                self.hrm_training_panel._on_goal_add_for_brain = self._on_goal_add_for_brain  # type: ignore[attr-defined]
                self.hrm_training_panel._on_goals_list_for_brain = self._on_goals_list_for_brain  # type: ignore[attr-defined]
            except Exception:
                pass
            # Connect to brains panel for auto-refresh after training
            try:
                self.hrm_training_panel._brains_panel = self.brains_panel  # type: ignore[attr-defined]
            except Exception:
                pass
            # Connect to settings panel for theme updates
            try:
                self.settings_panel._hrm_training_panel = self.hrm_training_panel  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception:
            pass

        # Evaluation panel
        try:
            self.evaluation_panel = EvaluationPanel(
                self.evaluation_tab,
                run_cli=self._run_cli,
                append_out=self._append_out,
                save_state_fn=self._save_state,
                worker_pool=self._worker_pool,
                on_list_brains=self._on_list_brains
            )
            # Connect to resources panel for device settings
            try:
                self.evaluation_panel._resources_panel = self.resources_panel  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception:
            pass

        # System status bar
        self.status_bar = StatusBar(self.root)
        try:
            self._status_updater = SystemStatusUpdater(
                root=self.root,
                set_status_cb=self.status_bar.set,
                resources_panel=self.resources_panel,
                worker_pool=self._worker_pool,
            )
            # Note: SystemStatusUpdater now uses managed worker pool for better performance
            self._status_updater.start(1000)
        except Exception:
            pass

        # Load persisted UI state (best-effort)
        try:
            self._load_state()
        except Exception:
            pass

        # Initialize persistent chat router (lazy - will be created on first use)
        self._init_chat_router()
        
        # Sync initial active goals to router
        self._update_router_goals()
        
        # Create system tray icon
        self._init_tray()
        
        # Handle start minimized flag
        if self._start_minimized and self._tray_manager:
            self.root.after(500, self._tray_manager.hide_window)

        # Defer warmup tasks slightly so initial UI paints before any heavy CLI work
        def _deferred_warmups():
            try:
                self._debug_write("[gui] deferred warmup starting…")  # type: ignore[attr-defined]
            except Exception:
                pass
            # Base text brain warmup
            try:
                self._run_cli(["chat", "__warmup__"])
            except Exception:
                pass
            # Device refresh (update device info without resetting settings)
            # NOTE: Initial detection already happened, this is just a refresh
            try:
                info = self._parse_cli_dict(self._run_cli(["torch-info"]) or "{}")
                if isinstance(info, dict):
                    try:
                        # Use refresh_detected to update without resetting settings
                        self.resources_panel.refresh_detected(info)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                self._debug_write("[gui] deferred warmup done")  # type: ignore[attr-defined]
            except Exception:
                pass
        try:
            # Use worker pool instead of ad-hoc threading - delay 3 seconds to allow GUI to fully render
            self.root.after(3000, lambda: self._worker_pool.submit(_deferred_warmups))
        except Exception:
            pass
        
        # Schedule periodic cleanup (every 15 minutes - reduced to prevent performance impact)
        try:
            self._timer_manager.set_timer("periodic_cleanup", 900000, self._periodic_cleanup)
        except Exception:
            pass

    def _periodic_cleanup(self) -> None:
        """Periodic maintenance tasks to prevent resource leaks.
        
        Runs every 15 minutes to:
        - Force Python garbage collection
        - Clear CUDA cache if available
        
        Note: VACUUM removed as it blocks all database operations
        """
        try:
            # Force Python garbage collection
            import gc
            collected = gc.collect()
            logging.debug(f"Periodic cleanup: collected {collected} objects")
            
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logging.debug("Periodic cleanup: cleared CUDA cache")
            except Exception:
                pass
            
            # Log basic resource usage for monitoring (lightweight)
            try:
                import psutil
                process = psutil.Process()
                mem_mb = process.memory_info().rss / 1024 / 1024
                threads = threading.active_count()
                logging.debug(f"Resource usage: {mem_mb:.0f}MB RAM, {threads} threads")
            except Exception:
                pass
        
        except Exception as e:
            logging.error(f"Periodic cleanup error: {e}")
        
        finally:
            # Reschedule for next cycle
            try:
                if not hasattr(self, '_emergency_cleanup_triggered'):
                    self._timer_manager.set_timer("periodic_cleanup", 900000, self._periodic_cleanup)
            except Exception:
                pass

    def _tk_exception_hook(self, exc: BaseException, val: BaseException, tb) -> None:  # type: ignore[override]
        """Route unhandled Tk callback exceptions to the Debug tab."""
        try:
            import traceback as _tb
            etype = exc if isinstance(exc, type) else type(val)
            parts = []
            try:
                parts.extend(_tb.format_exception_only(etype, val))
            except Exception:
                pass
            try:
                parts.extend(_tb.format_tb(tb))
            except Exception:
                pass
            self._debug_set_error("".join(parts))
        except Exception:
            pass

    # CLI helpers provided by CliBridgeMixin

    # dataset and crawl (removed from first page)
    def on_crawl_only(self) -> None:
        try:
            self._append_out("[ui] Crawl & Store has been removed from this page. Use Dataset Builder or CLI.")
        except Exception:
            pass

    # goals helpers (brain-aware)
    def _on_goal_add_for_brain(self, brain_name: str, text: str) -> None:
        """Add a goal for a specific brain."""
        txt = (text or "").strip()
        if not txt:
            self._append_out("[goals] Empty goal text")
            return
        # Add brain name as context to the goal (store in metadata)
        # For now, use standard goal-add with brain prefix in text
        goal_text = f"[{brain_name}] {txt}"
        self._append_out(f"[goals] Adding goal: {goal_text}")
        result = self._run_cli(["goals-add", goal_text])
        self._append_out(f"[goals] Result: {result}")

    def _on_goal_remove(self, goal_id: int) -> None:
        self._append_out(self._run_cli(["goals-remove", str(int(goal_id))]))

    def _on_goals_list_for_brain(self, brain_name: str) -> list[str]:
        """Get goals for a specific brain."""
        # Get all goals and filter by brain name
        all_goals = self._on_goals_list()
        self._append_out(f"[goals] Filter debug: brain_name='{brain_name}', all_goals count={len(all_goals)}")
        if all_goals:
            self._append_out(f"[goals] Filter debug: sample goals: {all_goals[:3]}")
        # Filter goals that have [brain_name] prefix
        brain_goals = [g for g in all_goals if f"[{brain_name}]" in g]
        self._append_out(f"[goals] Filter debug: filtered to {len(brain_goals)} goals for '{brain_name}'")
        if brain_goals:
            self._append_out(f"[goals] Filter debug: sample filtered: {brain_goals[:3]}")
        return brain_goals

    def _on_goals_list(self) -> list[str]:
        raw = self._run_cli(["goals-list"]) or "{}"
        # Use _parse_cli_dict to handle the [cli] header that _run_cli prepends
        data = self._parse_cli_dict(raw)
        if not data:
            # Fallback for legacy list format
            try:
                data = json.loads(raw)
            except Exception:
                try:
                    data = ast.literal_eval(raw)
                except Exception:
                    data = {}
        out: list[str] = []
        # Preferred modern shape: { count, directives: [str], items: [ {id,text,created_ts,protected} ] }
        if isinstance(data, dict):
            items = data.get("items") if isinstance(data.get("items"), list) else None
            if items is not None:
                primary = []
                others = []
                for it in items:
                    try:
                        did = it.get("id") or it.get("directive_id")
                        txt = it.get("text") or it.get("goal") or ""
                        created = it.get("created_ts") or it.get("created") or it.get("ts") or ""
                        prot = bool(it.get("protected"))
                        entry = {
                            "id": did,
                            "text": txt,
                            "created": created,
                            "protected": prot,
                        }
                        if prot:
                            primary.append(entry)
                        else:
                            others.append(entry)
                    except Exception:
                        continue
                # Numbered list, primary first
                ordered = primary + others
                for idx, it in enumerate(ordered, start=1):
                    try:
                        label = f"{idx}) #{it['id']} • {it['text']}"
                        if it.get("protected"):
                            label += "  [primary]"
                        if it.get("created"):
                            label += f"  (created {it['created']})"
                        out.append(label)
                    except Exception:
                        continue
            else:
                # Legacy fallback: list of directive strings
                try:
                    ds = data.get("directives") or []
                    for i, txt in enumerate(ds, start=1):
                        out.append(f"{i}) • {txt}")
                except Exception:
                    pass
        elif isinstance(data, list):
            # Legacy older shape: list of dicts
            for i, it in enumerate(data, start=1):
                try:
                    did = it.get("id")
                    txt = it.get("text") or it.get("goal") or ""
                    created = it.get("created") or it.get("ts") or ""
                    prot = bool(it.get("protected"))
                    label = f"{i}) #{did} • {txt}"
                    if prot:
                        label += "  [primary]"
                    if created:
                        label += f"  (created {created})"
                    out.append(label)
                except Exception:
                    continue
        return out

    def on_crawl_and_train(self) -> None:
        # Training now lives in the HRM Training tab; Crawl panel removed from this page
        try:
            self._append_out("[ui] Training has moved. Use the 'HRM Training' tab for training.")
        except Exception:
            pass

    def _save_state(self) -> None:
        # Dataset builder state (images)
        try:
            builder_state = self.dataset_builder_panel.get_state()
        except Exception:
            builder_state = {}
        # Resources values
        try:
            resources_values = self.resources_panel.get_values()
        except Exception:
            resources_values = {}
        # HRM Training state
        hrm_training_state: dict[str, Any] | None = None
        try:
            panel = getattr(self, "hrm_training_panel", None)
            if panel is not None and hasattr(panel, "get_state"):
                hv = panel.get_state()  # type: ignore[attr-defined]
                if isinstance(hv, dict):
                    hrm_training_state = hv
        except Exception:
            pass
        # Evaluation panel state
        evaluation_state: dict[str, Any] | None = None
        try:
            panel = getattr(self, "evaluation_panel", None)
            if panel is not None and hasattr(panel, "get_state"):
                ev = panel.get_state()  # type: ignore[attr-defined]
                if isinstance(ev, dict):
                    evaluation_state = ev
        except Exception:
            pass
        # Settings state
        settings_state: dict[str, Any] | None = None
        try:
            panel = getattr(self, "settings_panel", None)
            if panel is not None and hasattr(panel, "get_state"):
                sv = panel.get_state()  # type: ignore[attr-defined]
                if isinstance(sv, dict):
                    settings_state = sv
                # Sync tray settings whenever we save
                self._sync_tray_settings()
        except Exception:
            pass
        # Persist
        save_app_state(
            self._state_path,
            core_toggles={
                "cpu": bool(self.cpu_var.get()),
                "cuda": bool(self.cuda_var.get()),
                "xpu": bool(self.xpu_var.get()),
                "dml": bool(self.dml_var.get()),
                "mps": bool(self.mps_var.get()),
                "dml_python": self.dml_py_var.get(),
            },
            dataset_path=self.dataset_path_var.get(),
            builder_state=builder_state or {},
            resources_values=resources_values or {},
            hrm_training_state=hrm_training_state,
            evaluation_state=evaluation_state,
            settings_state=settings_state,
        )

    def _save_state_debounced(self, delay_ms: int = 300) -> None:
        """Save state with debouncing using TimerManager."""
        self._timer_manager.set_timer("save_state", delay_ms, self._save_state)

    def _load_state(self) -> None:
        data = load_app_state(self._state_path)
        if not data:
            return
        if not isinstance(data, dict):
            return
        # Core toggles
        try:
            self.cpu_var.set(bool(data.get("cpu", False)))
            self.cuda_var.set(bool(data.get("cuda", False)))
            self.xpu_var.set(bool(data.get("xpu", False)))
            self.dml_var.set(bool(data.get("dml", False)))
            self.mps_var.set(bool(data.get("mps", False)))
        except Exception:
            pass
        # Dataset builder state
        try:
            b = self.dataset_builder_panel
            v = str(data.get("builder_type") or "").strip()
            if v:
                b.type_var.set(v)
            q = str(data.get("builder_query") or "").strip()
            if q:
                b.query_var.set(q)
            v = str(data.get("builder_max_images") or "").strip()
            if v:
                b.max_images_var.set(v)
            v = str(data.get("builder_per_site") or "").strip()
            if v:
                b.per_site_var.set(v)
            v = str(data.get("builder_search_results") or "").strip()
            if v:
                b.search_results_var.set(v)
            v = str(data.get("builder_dataset_name") or "").strip()
            if v:
                b.ds_name_var.set(v)
            try:
                b.overwrite_var.set(bool(data.get("builder_overwrite", False)))
            except Exception:
                pass
        except Exception:
            pass
        # Dataset path
        try:
            v = str(data.get("dataset_path") or "").strip()
            if v:
                self.dataset_path_var.set(v)
        except Exception:
            pass
        # Crawl settings removed from first page
        # Resources
        try:
            vals = {k: v for k, v in data.items() if k in {
                "cpu_threads", "gpu_mem_pct", "cpu_util_pct", "gpu_util_pct",
                "train_device", "run_device",
                "train_cuda_selected", "train_cuda_mem_pct",
                "train_cuda_util_pct",
                "run_cuda_selected", "run_cuda_mem_pct", "run_cuda_util_pct",
                "dataset_cap",
            }}
            if vals:
                # Backfill ensure ints exist where needed
                if "cpu_threads" in vals and isinstance(vals["cpu_threads"], (int, float)):
                    self.resources_panel.cpu_threads_var.set(str(int(vals["cpu_threads"])) if int(vals["cpu_threads"]) > 0 else self.resources_panel.cpu_threads_var.get())
                if "gpu_mem_pct" in vals and isinstance(vals["gpu_mem_pct"], (int, float)):
                    self.resources_panel.gpu_mem_pct_var.set(str(int(vals["gpu_mem_pct"])) if int(vals["gpu_mem_pct"]) > 0 else self.resources_panel.gpu_mem_pct_var.get())
                if "cpu_util_pct" in vals and isinstance(vals["cpu_util_pct"], (int, float)):
                    self.resources_panel.cpu_util_pct_var.set(str(int(vals["cpu_util_pct"])) if int(vals["cpu_util_pct"]) >= 0 else self.resources_panel.cpu_util_pct_var.get())
                if "gpu_util_pct" in vals and isinstance(vals["gpu_util_pct"], (int, float)):
                    self.resources_panel.gpu_util_pct_var.set(str(int(vals["gpu_util_pct"])) if int(vals["gpu_util_pct"]) >= 0 else self.resources_panel.gpu_util_pct_var.get())
                # Let panel apply the rest comprehensively
                self.resources_panel.set_values(vals)
        except Exception:
            pass

        # HRM Training panel state
        try:
            hv = data.get("hrm_training")
            if isinstance(hv, dict) and hasattr(getattr(self, "hrm_training_panel", None), "set_state"):
                try:
                    self.hrm_training_panel.set_state(hv)  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            pass

        # Evaluation panel state
        try:
            ev = data.get("evaluation")
            if isinstance(ev, dict) and hasattr(getattr(self, "evaluation_panel", None), "set_state"):
                try:
                    self.evaluation_panel.set_state(ev)  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            pass

        # Settings panel state
        try:
            sv = data.get("settings")
            if isinstance(sv, dict) and hasattr(getattr(self, "settings_panel", None), "set_state"):
                try:
                    self.settings_panel.set_state(sv)  # type: ignore[attr-defined]
                    # Update tray behavior from settings
                    self._sync_tray_settings()
                except Exception:
                    pass
        except Exception:
            pass
    
    def _sync_tray_settings(self) -> None:
        """Sync tray behavior from settings panel."""
        try:
            if hasattr(self, "settings_panel"):
                self._minimize_to_tray_on_close = self.settings_panel.minimize_to_tray_var.get()
        except Exception:
            pass

    def _emergency_cleanup(self) -> None:
        """Emergency cleanup if normal shutdown fails (called by atexit)."""
        try:
            logging.info("Emergency cleanup triggered")
            
            # Shutdown async worker pool
            try:
                if hasattr(self, '_worker_pool'):
                    self._worker_pool.shutdown(wait=False, timeout=1.0)
            except Exception:
                pass
            
            # Stop async event loop
            try:
                if hasattr(self, '_async_loop'):
                    self._async_loop.stop(timeout=1.0)
            except Exception:
                pass
            
            # Cancel all timers
            try:
                if hasattr(self, '_timer_manager'):
                    self._timer_manager.cancel_all()
            except Exception:
                pass
            
            # Stop all managed threads with short timeout
            try:
                if hasattr(self, '_managed_threads'):
                    for thread in self._managed_threads:
                        try:
                            thread.stop(timeout=1.0)
                        except Exception:
                            pass
            except Exception:
                pass
            
            # Kill all processes
            try:
                if hasattr(self, '_process_reaper'):
                    self._process_reaper.cleanup_all(timeout=2.0)
            except Exception:
                pass
        except Exception as e:
            logging.error(f"Emergency cleanup error: {e}")
    
    def _on_close(self) -> None:
        """Enhanced cleanup on application exit - kills all processes and threads."""
        # Check if we should minimize to tray instead of closing
        if self._minimize_to_tray_on_close and self._tray_manager:
            self._tray_manager.hide_window()
            return
        
        try:
            # Destroy tray icon first
            if self._tray_manager:
                self._tray_manager.destroy()
            
            # Shutdown async worker pool with timeout
            try:
                logging.info("Shutting down worker pool...")
                self._worker_pool.shutdown(wait=True, timeout=5.0)
            except Exception as e:
                logging.error(f"Worker pool shutdown error: {e}")
            
            # Stop async event loop
            try:
                logging.info("Stopping async event loop...")
                self._async_loop.stop(timeout=3.0)
            except Exception as e:
                logging.error(f"Async loop stop error: {e}")
            
            # Cancel all timers
            try:
                self._timer_manager.cancel_all()
            except Exception:
                pass
            
            # Stop metrics polling
            try:
                if hasattr(self, "hrm_training_panel") and getattr(self.hrm_training_panel, "_metrics_polling_active", False):  # type: ignore[attr-defined]
                    self.hrm_training_panel._metrics_polling_active = False  # type: ignore[attr-defined]
            except Exception:
                pass
            
            # Stop HRM training/optimization (uses enhanced universal stop)
            try:
                panel = getattr(self, "hrm_training_panel", None)
                if panel is not None:
                    # Use _stop_all which handles both training and optimization
                    if callable(getattr(panel, "_stop_all", None)):
                        panel._stop_all()
                    # Also set stop flag to ensure background threads exit
                    if hasattr(panel, "_stop_requested"):
                        panel._stop_requested = True
                    
                    # CRITICAL: Force-kill training subprocess immediately on window close
                    # Don't wait for graceful shutdown - user expects immediate termination
                    proc = getattr(panel, "_proc", None)
                    if proc is not None and proc.poll() is None:
                        logging.info(f"Force-terminating training process on window close (PID: {proc.pid})")
                        try:
                            import platform
                            import subprocess as _subproc
                            
                            # Windows-specific: Use taskkill for forceful termination
                            if platform.system() == "Windows":
                                # Kill entire process tree (including child processes)
                                try:
                                    _subproc.run(
                                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                        capture_output=True,
                                        timeout=3
                                    )
                                    logging.info(f"Sent taskkill to PID {proc.pid}")
                                except Exception as e:
                                    logging.error(f"taskkill failed: {e}, falling back to terminate()")
                                    proc.terminate()
                            else:
                                # Unix: Use standard terminate
                                proc.terminate()
                            
                            # Wait briefly for terminate
                            import time
                            for _ in range(10):  # 2.5 seconds max
                                if proc.poll() is not None:
                                    logging.info("Training process terminated successfully")
                                    break
                                time.sleep(0.25)
                            
                            # If still alive, escalate to kill (Unix only - Windows taskkill is already forceful)
                            if proc.poll() is None:
                                if platform.system() != "Windows":
                                    proc.kill()
                                    logging.info("Escalated to kill()")
                                for _ in range(4):
                                    if proc.poll() is not None:
                                        break
                                    time.sleep(0.25)
                                
                                if proc.poll() is None:
                                    logging.warning(f"Process {proc.pid} did not terminate!")
                        except Exception as e:
                            logging.error(f"Failed to terminate training process: {e}")
            except Exception:
                pass
            
            # Stop all managed threads (including status updater)
            try:
                for thread in self._managed_threads:
                    try:
                        thread.stop(timeout=2.0)
                    except Exception as e:
                        logging.error(f"Failed to stop managed thread: {e}")
            except Exception:
                pass
            
            # Save state before exit
            self._save_state()
            
            # Use ProcessReaper for comprehensive process cleanup
            try:
                self._process_reaper.cleanup_all(timeout=3.0)
            except Exception as e:
                logging.error(f"Process cleanup error: {e}")
            
            # Close logging handlers to prevent leaks
            try:
                if hasattr(self, "_debug_log_handler"):
                    handler = self._debug_log_handler
                    if handler:
                        try:
                            logging.getLogger("aios").removeHandler(handler)
                            logging.getLogger().removeHandler(handler)
                            handler.close()
                        except Exception:
                            pass
            except Exception:
                pass
            
        finally:
            try:
                self.root.destroy()
            except Exception:
                pass

    # _parse_cli_dict provided by CliBridgeMixin

    # Button callbacks
    def on_status(self) -> None:
        try:
            self._update_out(self._run_cli(["status", "--recent", "1"]))
        except Exception as e:
            import traceback
            self._debug_set_error(traceback.format_exc())

    def on_artifacts(self) -> None:
        try:
            self._update_out(self._run_cli(["artifacts-list", "--limit", "5"]))
        except Exception:
            import traceback
            self._debug_set_error(traceback.format_exc())

    def on_budgets(self) -> None:
        try:
            info = {
                "budgets": json.loads(self._run_cli(["budgets-show"]) or "{}"),
                "usage": json.loads(self._run_cli(["budgets-usage"]) or "{}"),
            }
            self._update_out(json.dumps(info, indent=2))
        except Exception:
            import traceback
            self._debug_set_error(traceback.format_exc())

    def on_train(self) -> None:
        try:
            # Training moved: guide user to HRM Training tab
            self._append_out("[ui] Training controls have moved. Use the 'HRM Training' tab.")
            self._save_state()
        except Exception:
            import traceback
            self._debug_set_error(traceback.format_exc())

    def on_run(self) -> None:
        try:
            user = self.cmd_var.get().strip()
            args = user.split() if user else []
            self._update_out(self._run_cli(args))
        except Exception:
            import traceback
            self._debug_set_error(traceback.format_exc())

    # on_copy_output now provided by OutputPanel

    # --- chat helpers ---
    def _init_chat_router(self) -> None:
        """Initialize persistent Router/BrainRegistry for chat to keep model loaded."""
        try:
            from aios.core.brains import BrainRegistry, Router
            cfg = load_config()
            brains_cfg = (cfg.get("brains") or {}) if isinstance(cfg, dict) else {}
            
            # Build registry with config
            storage_limit_mb = float(brains_cfg.get("storage_limit_mb", 0) or 0) or None
            self._chat_registry = BrainRegistry()
            self._chat_registry.total_storage_limit_mb = storage_limit_mb
            self._chat_registry.store_dir = str(brains_cfg.get("store_dir", "artifacts/brains"))
            
            # Load persisted pins/masters
            try:
                self._chat_registry.load_pinned()
                self._chat_registry.load_masters()
            except Exception:
                pass
            
            # Build router with config overrides
            create_cfg = dict(brains_cfg.get("trainer_overrides", {}))
            gen_cfg = dict(brains_cfg.get("generation", {}) or {})
            if gen_cfg:
                create_cfg = dict(create_cfg or {})
                create_cfg["generation"] = gen_cfg
            if "system_prompt" in brains_cfg:
                create_cfg = dict(create_cfg or {})
                create_cfg["system_prompt"] = brains_cfg.get("system_prompt")
            if "history_max_turns" in brains_cfg:
                create_cfg = dict(create_cfg or {})
                create_cfg["history_max_turns"] = int(brains_cfg.get("history_max_turns") or 0)
            
            # Get device settings from resources panel for inference/chat
            try:
                rvals = self.resources_panel.get_values()
                run_device = rvals.get("run_device", "auto")
                run_cuda_selected = rvals.get("run_cuda_selected", [])
                
                # Determine inference device based on resources panel settings
                inference_device = None
                if run_device == "cuda" and isinstance(run_cuda_selected, list) and len(run_cuda_selected) > 0:
                    # Use first selected GPU for inference
                    inference_device = f"cuda:{int(run_cuda_selected[0])}"
                elif run_device == "cuda":
                    inference_device = "cuda"
                elif run_device == "cpu":
                    inference_device = "cpu"
                # else: auto (let adapter decide)
                
                if inference_device:
                    create_cfg = dict(create_cfg or {})
                    create_cfg["inference_device"] = inference_device
                    self._debug_write(f"[gui] Chat inference device set to: {inference_device}")  # type: ignore[attr-defined]
            except Exception as e:
                # Non-critical: just log and continue with auto-detect
                try:
                    self._debug_write(f"[gui] Failed to get inference device from resources panel: {e}")  # type: ignore[attr-defined]
                except Exception:
                    pass
            
            # Determine expert registry path
            expert_registry_path = os.path.join("artifacts", "experts", "registry.json")
            
            self._chat_router = Router(
                registry=self._chat_registry,
                default_modalities=list(brains_cfg.get("default_modalities", ["text"])),
                brain_prefix=str(brains_cfg.get("prefix", "brain")),
                create_cfg=create_cfg,
                strategy=str(brains_cfg.get("strategy", "hash")),
                modality_overrides=dict(brains_cfg.get("modality_overrides", {})),
                expert_registry_path=expert_registry_path,
            )
            
            # Initialize auto-training orchestrator
            try:
                dataset_registry = DatasetRegistry()
                expert_registry = ExpertRegistry()
                
                self._orchestrator = AutoTrainingOrchestrator(
                    dataset_registry=dataset_registry,
                    expert_registry=expert_registry,
                    min_confidence=0.5,
                )
                self._debug_write("[gui] Auto-training orchestrator initialized")  # type: ignore[attr-defined]
            except Exception as orch_err:
                self._orchestrator = None
                self._debug_write(f"[gui] Failed to init auto-training orchestrator: {orch_err}")  # type: ignore[attr-defined]
            
        except Exception as e:
            # Log error but don't crash - fallback to CLI subprocess
            try:
                self._debug_write(f"[gui] Failed to init chat router: {e}")  # type: ignore[attr-defined]
            except Exception:
                pass

    def _init_tray(self) -> None:
        """Initialize system tray icon."""
        try:
            icon_dir = Path(__file__).parent.parent.parent.parent / "installers"
            ico_path = icon_dir / "AI-OS.ico"
            
            self._tray_manager = TrayManager(
                self.root,
                icon_path=ico_path if ico_path.exists() else None,
                app_name="AI-OS",
                on_settings=self._on_tray_settings
            )
            
            if self._tray_manager.create_tray():
                self._append_out("[tray] System tray icon created")
            else:
                self._append_out("[tray] System tray not available")
        except Exception as e:
            self._append_out(f"[tray] Failed to create tray: {e}")
    
    def _on_tray_settings(self) -> None:
        """Switch to settings tab (called from tray menu)."""
        try:
            # Find settings tab index
            for i in range(self.nb.index("end")):
                if self.nb.tab(i, "text") == "Settings":
                    self.nb.select(i)
                    break
        except Exception as e:
            self._append_out(f"[tray] Error switching to settings: {e}")
    
    def _update_router_goals(self) -> None:
        """Update router with current active goals from directives DB."""
        if self._chat_router is None:
            return
        
        try:
            from aios.core.directives import list_directives
            from aios.memory.store import get_db, init_db
            conn = get_db()
            init_db(conn)
            try:
                active_directives = list_directives(conn, active_only=True)
                active_goal_ids = [str(d.directive_id) for d in active_directives]
                
                # Update router with active goals
                result = self._chat_router.update_active_goals(active_goal_ids)
                
                # Log changes if any
                if result.get("newly_activated") or result.get("newly_deactivated"):
                    activated_count = len(result.get("newly_activated", []))
                    deactivated_count = len(result.get("newly_deactivated", []))
                    linked_count = len(result.get("linked_experts", []))
                    try:
                        self._debug_write(  # type: ignore[attr-defined]
                            f"[gui] Goals updated: +{activated_count} -{deactivated_count} "
                            f"({linked_count} linked experts)"
                        )
                    except Exception:
                        pass
            finally:
                conn.close()
        except Exception as e:
            try:
                self._debug_write(f"[gui] Failed to update router goals: {e}")  # type: ignore[attr-defined]
            except Exception:
                pass

    def _unload_chat_model(self) -> None:
        """Optionally unload chat model to free GPU memory (e.g., before training)."""
        try:
            if self._chat_registry is not None:
                # Clear all loaded brains from memory (they can be restored from disk)
                self._chat_registry.brains.clear()
                self._debug_write("[gui] Chat model unloaded to free GPU memory")  # type: ignore[attr-defined]
        except Exception as e:
            try:
                self._debug_write(f"[gui] Failed to unload chat model: {e}")  # type: ignore[attr-defined]
            except Exception:
                pass

    def _on_unload_model(self) -> str:
        """Callback for unload button - unloads chat model and returns status message."""
        try:
            self._unload_chat_model()
            return "✓ Model unloaded successfully. GPU memory freed."
        except Exception as e:
            return f"✗ Failed to unload model: {e}"

    def _chat_route(self, user: str) -> list[str]:
        return chat_route(user)

    def _on_chat_route_and_run(self, msg: str) -> str:
        """Handle chat message using persistent router (keeps model loaded)."""
        msg_stripped = (msg or "").strip()
        if not msg_stripped:
            return "Please enter a message."
        
        # Apply context slider value to the brain's max_response_chars
        try:
            context_length = self.chat_panel.get_context_length()
            if self._chat_router and hasattr(self._chat_router, 'registry'):
                registry = self._chat_router.registry
                # Apply to all active brains
                for brain in registry.brains.values():
                    if hasattr(brain, 'max_response_chars'):
                        # If context_length is 0, skip override (use brain's auto-calculated value)
                        if context_length <= 0:
                            continue
                        # Ensure within model's limits
                        max_tokens = getattr(brain, 'gen_max_new_tokens', 256)
                        max_chars = max_tokens * 4
                        brain.max_response_chars = max(256, min(context_length, max_chars))
        except Exception as e:
            # Non-critical: just log and continue
            try:
                self._debug_write(f"[gui] Failed to apply context setting: {e}")  # type: ignore[attr-defined]
            except Exception:
                pass
        
        # Check for learning intent (auto-training orchestrator)
        intent_message = ""
        if self._orchestrator is not None and not msg_stripped.startswith("/"):
            try:
                task = self._orchestrator.create_learning_task(
                    user_message=msg_stripped,
                    auto_start=True,
                )
                
                if task:
                    # Build user-friendly status message
                    categories_str = ', '.join(task.intent.categories)
                    intent_message = (
                        f"✨ Great! I'll learn about {task.intent.extracted_topic} for you.\n\n"
                        f"📊 Training Details:\n"
                        f"  • Domain: {task.intent.domain}\n"
                        f"  • Categories: {categories_str}\n"
                        f"  • Dataset: {task.dataset_id}\n"
                        f"  • Expert ID: {task.expert_id}\n\n"
                        f"⚡ Training is running in the background. I'll continue to assist you while I learn!\n\n"
                        f"{'─' * 60}\n\n"
                    )
                    self._debug_write(f"[gui] Auto-training task started: {task.task_id}")  # type: ignore[attr-defined]
                    
            except NoDatasetFoundError as e:
                intent_message = (
                    f"📚 I detected you want to learn about something, but I couldn't find a suitable dataset.\n"
                    f"Details: {str(e)}\n\n"
                    f"{'─' * 60}\n\n"
                )
                self._debug_write(f"[gui] Auto-training: No dataset found - {e}")  # type: ignore[attr-defined]
                
            except Exception as e:
                # Log error but don't interrupt normal chat
                try:
                    self._debug_write(f"[gui] Auto-training error: {e}")  # type: ignore[attr-defined]
                except Exception:
                    pass
        
        # Check for special commands
        if msg_stripped.startswith("/"):
            # Execute as CLI command
            args = msg_stripped[1:].split()
            out = self._run_cli(args)
            try:
                if args and args[0] in {"goals-add", "goals-list"}:
                    self.goals_panel.refresh()
                    self._update_router_goals()
            except Exception:
                pass
            return render_chat_output(out, args)
        
        # Use persistent router for chat
        try:
            if self._chat_router is not None:
                # Direct router call - keeps model loaded!
                result = self._chat_router.handle({"modalities": ["text"], "payload": msg_stripped})
                # result is a dict like {"ok": True, "text": "..."}
                response = render_chat_output(result, ["chat", msg_stripped])
                
                # Prepend intent message if learning task was created
                if intent_message:
                    response = intent_message + response
                
                return response
        except Exception as e:
            # Log and fall back to CLI subprocess
            try:
                self._debug_write(f"[gui] Chat router error: {e}")  # type: ignore[attr-defined]
            except Exception:
                pass
        
        # Fallback to CLI subprocess
        args = self._chat_route(msg_stripped)
        out = self._run_cli(args)
        if not out or not str(out).strip():
            try:
                out = self._run_cli(["status", "--recent", "1"]) or ""
            except Exception:
                out = ""
        try:
            if args and args[0] in {"goals-add", "goals-list"}:
                self.goals_panel.refresh()
                self._update_router_goals()
        except Exception:
            pass
        
        response = render_chat_output(out, args)
        
        # Prepend intent message if learning task was created
        if intent_message:
            response = intent_message + response
        
        return response

    def _on_list_brains(self) -> list[str]:
        """List available brains from the registry, excluding temporary/internal brains."""
        try:
            # Use absolute path to artifacts/brains directory
            import os
            import re
            store_dir = os.path.join(os.getcwd(), "artifacts", "brains")
            out = self._run_cli(["brains", "list-brains", "--store-dir", store_dir])
            # Parse JSON output: {"brains": {"name1": {...}, "name2": {...}, ...}}
            data = self._parse_cli_dict(out)
            if isinstance(data, dict) and "brains" in data:
                brains_dict = data["brains"]
                brain_names = []
                if isinstance(brains_dict, dict):
                    brain_names = list(brains_dict.keys())
                elif isinstance(brains_dict, list):
                    brain_names = brains_dict
                
                # Filter out temporary/internal brains (same logic as BrainsPanel):
                # - Brains starting with '_' (like _ddp used for DDP training)
                # - Brains matching 'brain-<modality>-<hash>' pattern (temporary chat router brains)
                def _is_temporary(name: str) -> bool:
                    if name.startswith('_'):
                        return True
                    # Check for router-generated temporary brains: brain-text-de5aae40, brain-image-abc123, etc.
                    if re.match(r'^brain-[a-z]+-[0-9a-f]{8}$', name):
                        return True
                    return False
                
                return [name for name in brain_names if not _is_temporary(name)]
            return []
        except Exception:
            return []

    def _on_load_brain(self, brain_name: str) -> str:
        """Load a specific brain by updating the config to use it as master."""
        try:
            # Use the brains CLI to set this brain as master (--enabled is default)
            import os
            store_dir = os.path.join(os.getcwd(), "artifacts", "brains")
            out = self._run_cli(["brains", "set-master", brain_name, "--enabled", "--store-dir", store_dir])
            
            # Update config with the loaded brain's settings
            try:
                brain_json_path = os.path.join(store_dir, "actv1", brain_name, "brain.json")
                if os.path.exists(brain_json_path):
                    import json
                    import yaml
                    from pathlib import Path
                    
                    # Read brain.json to get trained settings
                    with open(brain_json_path, "r", encoding="utf-8") as f:
                        brain_data = json.load(f)
                    
                    # Read current config
                    config_path = Path("config/default.yaml")
                    if config_path.exists():
                        with open(config_path, "r", encoding="utf-8") as f:
                            config = yaml.safe_load(f) or {}
                    else:
                        config = {}
                    
                    # Update config with brain's trained settings
                    if "brains" not in config:
                        config["brains"] = {}
                    if "trainer_overrides" not in config["brains"]:
                        config["brains"]["trainer_overrides"] = {}
                    
                    # Update starter_config to point to this brain
                    config["brains"]["trainer_overrides"]["starter_config"] = brain_json_path
                    
                    # Update max_seq_len to match brain's trained context length
                    trained_max_seq_len = brain_data.get("max_seq_len")
                    if trained_max_seq_len is not None:
                        config["brains"]["trainer_overrides"]["max_seq_len"] = int(trained_max_seq_len)
                    
                    # Update MoE settings if present
                    if "use_moe" in brain_data:
                        if "moe" not in config["brains"]["trainer_overrides"]:
                            config["brains"]["trainer_overrides"]["moe"] = {}
                        config["brains"]["trainer_overrides"]["moe"]["enabled"] = brain_data.get("use_moe", True)
                        config["brains"]["trainer_overrides"]["moe"]["num_experts"] = brain_data.get("num_experts", 8)
                        config["brains"]["trainer_overrides"]["moe"]["experts_per_tok"] = brain_data.get("num_experts_per_tok", 2)
                    
                    # Save updated config
                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(config_path, "w", encoding="utf-8") as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                    
                    print(f"[Config] Updated with brain '{brain_name}' settings (context: {trained_max_seq_len} tokens)")
            except Exception as e:
                print(f"[Config] Warning: Could not update config with brain settings: {e}")
            
            # Update context slider range based on loaded brain's capabilities
            try:
                if self._chat_router and hasattr(self._chat_router, 'registry'):
                    registry = self._chat_router.registry
                    if brain_name in registry.brains:
                        brain = registry.brains[brain_name]
                        # Get max tokens for generation (gen_max_new_tokens)
                        max_tokens = getattr(brain, 'gen_max_new_tokens', 256)
                        max_chars = max_tokens * 4  # Rough estimate: 4 chars per token
                        current_chars = getattr(brain, 'max_response_chars', 2048)
                        
                        # Update slider range in chat panel
                        self.chat_panel.update_context_range(
                            min_val=256,
                            max_val=max(256, max_chars),
                            current=current_chars
                        )
            except Exception as e:
                # Non-critical: just log and continue
                try:
                    self._debug_write(f"[gui] Failed to update context slider: {e}")  # type: ignore[attr-defined]
                except Exception:
                    pass
            
            # Check if successful
            if "master" in str(out).lower() or "set" in str(out).lower():
                return f"Loaded brain '{brain_name}' successfully. It will be used for chat."
            return f"Result: {out}"
        except Exception as e:
            return f"Failed to load brain: {e}"

    def on_detect_devices(self) -> None:
        """Probe torch backends and update toggles and DML path display."""
        try:
            info = self._parse_cli_dict(self._run_cli(["torch-info"]) or "{}")
        except Exception:
            import traceback
            self._debug_set_error(traceback.format_exc())
            info = {}
        try:
            self.cuda_var.set(bool(info.get("cuda_available")))
        except Exception:
            pass
        try:
            self.xpu_var.set(bool(info.get("xpu_available")))
        except Exception:
            pass
        try:
            self.mps_var.set(bool(info.get("mps_available")))
        except Exception:
            pass
        try:
            self.dml_var.set(bool(info.get("directml_available")))
        except Exception:
            pass
        dml_py = info.get("directml_python") or ""
        if isinstance(dml_py, str):
            self.dml_py_var.set(dml_py)
        else:
            self.dml_py_var.set("")
        try:
            self.resources_panel.set_detected(info if isinstance(info, dict) else {})
        except Exception:
            pass
        summary = {
            "cuda": bool(info.get("cuda_available")),
            "xpu": bool(info.get("xpu_available")),
            "mps": bool(info.get("mps_available")),
            "directml": bool(info.get("directml_available")),
            "dml_python": self.dml_py_var.get(),
        }
        self._update_out("Detected devices:\n" + json.dumps(summary, indent=2))

    def on_train_parallel(self) -> None:
        # Training moved: guide user to HRM Training tab
        try:
            self._append_out("[ui] Training controls have moved. Use the 'HRM Training' tab.")
            self._save_state()
        except Exception:
            pass

    def on_stop(self) -> None:
        # No global stop for training here anymore
        try:
            self._append_out("[ui] Stop: If running training, use controls in 'HRM Training' tab.")
        except Exception:
            pass

    def on_test_dml(self) -> None:
        dml_py = self.dml_py_var.get().strip()
        args = ["dml-test"]
        if dml_py:
            args.extend(["--python", dml_py])
        out = self._run_cli(args)
        self._update_out(out)

    def on_browse_dataset(self) -> None:
        # removed from first page
        self._append_out("[ui] Dataset browser removed. Use Dataset Builder output path or CLI.")

    def on_load_known_datasets(self) -> None:
        self._append_out("[ui] Known dataset list removed from this page.")

    def on_use_known_dataset(self) -> None:
        self._append_out("[ui] Known dataset selection removed from this page.")

    def on_download_known_dataset(self) -> None:
        self._append_out("[ui] Known dataset download removed from this page.")

    def on_cancel_download(self) -> None:
        self._append_out("[ui] No active dataset downloads on this page.")

    def _fetch_known_datasets_from_github(self, *, max_items: int = 200, max_size_gb: int = 15) -> list[dict]:
        # unused in main app; dataset fetching is handled by DatasetsPanel
        return []

    def _fetch_known_datasets_from_awesomedata_nlp(self, *, max_items: int = 120, max_size_gb: int = 15) -> list[dict]:
        # unused in main app; dataset fetching is handled by DatasetsPanel
        return []

    def _fetch_known_datasets_from_aws_open_data_registry(self, *, max_items: int = 60, max_size_gb: int = 15) -> list[dict]:
        # unused in main app; dataset fetching is handled by DatasetsPanel
        return []

    # Removed legacy Vendor/HRM/DREAM handlers – functionality is automated or CLI only

    # system status is now provided by SystemStatusUpdater

    def run(self) -> None:
        """Start the Tkinter main loop."""
        self.root.mainloop()

    def _schedule_save_state(self) -> None:
        if self._save_after_id is not None:
            self.root.after_cancel(self._save_after_id)
        self._save_after_id = self.root.after(2000, self._save_state)


def run(exit_after: float | None = None, minimized: bool = False):
    """Start the Tkinter app.

    Args:
        exit_after: if provided (>0), auto-close the window after N seconds (CI/headless smoke).
        minimized: if True, start with window minimized to tray
    """
    if tk is None:
        return
    try:
        print("GUI main() called")
        root = tk.Tk()
    except Exception:
        # Headless or Tk init failure; no-op for smoke tests
        return
    try:
        app = AiosTkApp(root, start_minimized=minimized)
        print("App created, starting mainloop...")
        app.run()
        print("Mainloop finished.")
    except Exception:
        try:
            root.destroy()
        except Exception:
            pass
        return
    if exit_after and exit_after > 0:
        try:
            ms = int(float(exit_after) * 1000)
            root.after(max(1, ms), root.destroy)
        except Exception:
            pass
    root.mainloop()
