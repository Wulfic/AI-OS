"""Panel initialization module for AI-OS GUI.

This module initializes all panel components including:
- Dataset Download panel
- Dataset Builder panel
- Resources panel (with device detection)
- Debug panel  
- Settings panel
- Chat panel (RichChatPanel)
- Brains panel
- MCP Manager panel
- HRM Training panel
- Evaluation panel
- Status bar
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING
import logging
import time

if TYPE_CHECKING:
    import tkinter as tk

# Lazy imports - panels are imported only when instantiated to minimize startup time
# This reduces initial import overhead by ~10-15 seconds
# Components will be imported in their respective initialization functions

from ..services import LogCategory

# Apply safety monkeypatches early (no-ops if not applicable)
try:
    from ..monkeypatches import matplotlib_tk_guard  # noqa: F401
except Exception:
    pass

logger = logging.getLogger(__name__)


def initialize_panels(app: Any) -> None:
    """
    Initialize all panel components.
    
    This function creates all UI panels in dependency order:
    1. Output panels (for bridging)
    2. Dataset panels
    3. Resources panel (with immediate device detection)
    4. Debug panel  
    5. Settings panel
    6. Logging configuration (Python logging → Debug panel)
    7. Chat panel (RichChatPanel)
    8. Brains panel
    9. MCP Manager panel
    10. HRM Training panel
    11. Evaluation panel
    12. Status bar
    
    Args:
        app: AiosTkApp instance with tabs already created
    """
    
    start_time = time.time()
    last_time = start_time
    
    def update_loading(text: str) -> None:
        """Update loading screen if available."""
        try:
            if hasattr(app, '_loading_canvas') and app._loading_canvas:
                # Get the status text ID from canvas (set by update_loading_canvas)
                if hasattr(app._loading_canvas, '_status_text_id'):
                    app._loading_canvas.itemconfig(app._loading_canvas._status_text_id, text=text)
                app.root.update_idletasks()
        except Exception:
            pass
    
    def log_timing(panel_name: str) -> None:
        """Log timing for each panel initialization."""
        nonlocal last_time
        current = time.time()
        step_duration = current - last_time
        msg = f"[PANEL TIMING] {panel_name}: {step_duration:.3f}s"
        logger.info(msg)
        print(msg)  # Also print to console for visibility
        last_time = current
    
    # ===== DATASET PANELS =====
    # Create shared output panel first (will be in left frame)
    update_loading("Loading Dataset Output panel...")
    try:
        import tkinter as tk
        from tkinter import scrolledtext, ttk
        
        # Create output frame in left frame (50% width)
        app.dataset_output_frame = ttk.LabelFrame(app._left_frame, text="📋 Output", padding=5)
        app.dataset_output_frame.pack(fill="both", expand=True, padx=5, pady=(5, 5))
        
        # Create scrolled text widget for output
        app.dataset_output_text = scrolledtext.ScrolledText(
            app.dataset_output_frame,
            wrap="word",
            height=15,  # Fixed height in lines
            font=("Consolas", 9)
        )
        app.dataset_output_text.pack(fill="both", expand=True)
        
        # Create output callback that writes to this widget
        def _dataset_output_callback(msg: str) -> None:
            try:
                app.dataset_output_text.insert(tk.END, msg + "\n")
                app.dataset_output_text.see(tk.END)
            except Exception:
                pass
        
        app._dataset_output = _dataset_output_callback
        log_timing("Dataset output panel")
    except Exception as e:
        logger.error(f"Failed to create dataset output panel: {e}")
        app._dataset_output = app._append_out  # Fallback
    
    # Dataset download panel (search and download controls - will use shared output)
    update_loading("Loading Dataset Download panel...")
    try:
        from ..components import OutputPanel, DatasetDownloadPanel  # Lazy import
        
        # Create bridge for streaming download progress to output panel
        def _create_download_bridge():
            queue_bridge = OutputPanel(app.root)
            
            def _bridge_write(text: str, tag: str | None = None) -> None:
                try:
                    # Append streaming lines to the output panel; ignore tag
                    queue_bridge.append(text)
                except Exception:
                    pass
            
            return queue_bridge, _bridge_write
        
        app._download_output_bridge_panel, app._download_output_bridge = _create_download_bridge()
        
        # Don't pass output_parent - the panel will skip creating output since we're using log_callback
        app.dataset_download_panel = DatasetDownloadPanel(
            app.datasets_tab,
            log_callback=app._dataset_output,  # Use shared output callback
            output_parent=None,  # Panel won't create its own output
        )
        log_timing("Dataset download panel")
    except Exception as e:
        logger.error(f"Failed to initialize dataset download panel: {e}")
    
    # Dataset builder panel (will be in right frame)
    update_loading("Loading Dataset Builder panel...")
    try:
        from ..components import DatasetBuilderPanel  # Lazy import
        
        app.dataset_builder_panel = DatasetBuilderPanel(
            app._right_frame,
            run_cli=app._run_cli,
            dataset_path_var=app.dataset_path_var,
            append_out=app._dataset_output,  # Use shared output
            update_out=app._dataset_output,  # Use same callback for both
            worker_pool=app._worker_pool,
        )
        log_timing("Dataset builder panel")
    except Exception as e:
        logger.error(f"Failed to initialize dataset builder panel: {e}")
    
    # ===== RESOURCES PANEL =====
    update_loading("Loading Resources panel...")
    _initialize_resources_panel(app)
    log_timing("Resources panel")
    
    # ===== DEBUG PANEL =====
    update_loading("Loading Debug panel...")
    try:
        from ..components import DebugPanel  # Lazy import
        
        app.debug_panel = DebugPanel(app.debug_tab)
        log_timing("Debug panel")
    except Exception as e:
        logger.error(f"Failed to initialize debug panel: {e}")
    
    # ===== SETTINGS PANEL =====
    update_loading("Loading Settings panel...")
    try:
        from ..components import SettingsPanel  # Lazy import
        
        app.settings_panel = SettingsPanel(
            app.settings_tab,
            save_state_fn=app._save_state,
            chat_panel=None,  # Will be set after chat_panel is created
            help_panel=None   # Will be set after help_panel is created
        )
        log_timing("Settings panel")
    except Exception as e:
        logger.error(f"Failed to initialize settings panel: {e}")
    
    # ===== LOGGING CONFIGURATION =====
    _configure_python_logging(app)
    log_timing("Logging configuration")
    
    # ===== CHAT PANEL =====
    update_loading("Loading Chat panel...")
    try:
        from ..components import RichChatPanel  # Lazy import
        
        app.chat_panel = RichChatPanel(
            app.chat_tab,
            app._on_chat_route_and_run,
            on_load_brain=app._on_load_brain,
            on_list_brains=app._on_list_brains,
            on_unload_model=app._on_unload_model,
            worker_pool=app._worker_pool,
        )
        
        # Connect chat_panel to settings panel for theme updates
        app.settings_panel._chat_panel = app.chat_panel
        log_timing("Chat panel")
    except Exception as e:
        logger.error(f"Failed to initialize chat panel: {e}")
    
    # ===== BRAINS PANEL =====
    update_loading("Loading Brains panel...")
    try:
        from ..components import BrainsPanel  # Lazy import
        
        app.brains_panel = BrainsPanel(
            app.brains_tab,
            run_cli=app._run_cli,
            append_out=app._append_out,
            on_goal_add=app._on_goal_add_for_brain,
            on_goals_list=app._on_goals_list_for_brain,
            on_goal_remove=app._on_goal_remove,
            worker_pool=app._worker_pool,
        )
        log_timing("Brains panel")
    except Exception as e:
        logger.error(f"Failed to initialize brains panel: {e}")
    
    # ===== MCP MANAGER PANEL =====
    update_loading("Loading MCP Manager panel...")
    try:
        from ..components import MCPManagerPanel  # Lazy import
        
        app.mcp_panel = MCPManagerPanel(
            app.mcp_tab,
            run_cli=app._run_cli,
            append_out=app._append_out,
            save_state_fn=app._save_state,
        )
        log_timing("MCP Manager panel")
    except Exception as e:
        logger.error(f"Failed to initialize MCP panel: {e}")
    
    # ===== HRM TRAINING PANEL =====
    update_loading("Loading HRM Training panel...")
    try:
        from ..components import HRMTrainingPanel  # Lazy import
        
        app.hrm_training_panel = HRMTrainingPanel(
            app.training_tab,
            run_cli=app._run_cli,
            append_out=app._append_out,
            save_state_fn=app._save_state,
            worker_pool=app._worker_pool,
            resources_panel=app.resources_panel,
        )
        log_timing("HRM Training panel")
    except Exception as e:
        logger.error(f"Failed to initialize HRM training panel: {e}")
    
    # ===== EVALUATION PANEL =====
    # Load evaluation panel during startup
    update_loading("Loading Evaluation panel...")
    try:
        from ..components import EvaluationPanel  # Lazy import
        
        app.evaluation_panel = EvaluationPanel(
            app.evaluation_tab,
            run_cli=app._run_cli,
            append_out=app._append_out,
            save_state_fn=app._save_state,
            worker_pool=app._worker_pool,
            on_list_brains=app._on_list_brains if hasattr(app, '_on_list_brains') else None,
        )
        log_timing("Evaluation panel")
    except Exception as e:
        logger.error(f"Failed to initialize evaluation panel: {e}")
        app.evaluation_panel = None
    
    # ===== HELP PANEL =====
    update_loading("Loading Help panel...")
    try:
        from ..components import HelpPanel  # Lazy import
        
        app.help_panel = HelpPanel(app.help_tab, project_root=app._project_root)
        
        # Connect help_panel to settings panel for theme updates
        if hasattr(app, 'settings_panel') and app.settings_panel:
            app.settings_panel._help_panel = app.help_panel
        log_timing("Help panel")
    except Exception as e:
        logger.error(f"Failed to initialize help panel: {e}")

    # ===== STATUS BAR =====
    try:
        from ..components import StatusBar  # Lazy import
        
        app.status_bar = StatusBar(app.root)
        log_timing("Status bar")
    except Exception as e:
        logger.error(f"Failed to initialize status bar: {e}")
    
    total_time = time.time() - start_time
    msg = f"[PANEL TIMING] Total panel initialization time: {total_time:.3f}s"
    logger.info(msg)
    print(msg)  # Also print to console for visibility


def deferred_panel_initialization(app: Any) -> None:
    """
    Perform deferred panel initialization tasks after GUI is displayed.
    
    DEPRECATED: All data now loads synchronously during startup.
    This function is no longer called but kept for reference.
    
    Args:
        app: AiosTkApp instance with panels already created
    """
    pass  # All loading now happens synchronously


def _load_chat_brains_sync(app: Any) -> None:
    """Load chat panel brain list synchronously during startup.
    
    This runs in a background thread. Data is loaded here, but UI updates
    are stored for the main thread to apply later.
    
    Args:
        app: AiosTkApp instance
    """
    try:
        if hasattr(app, 'chat_panel') and app.chat_panel:
            # Call on_list_brains directly (this is the slow part)
            if app.chat_panel._on_list_brains:
                brains = app.chat_panel._on_list_brains()
                
                # Store results for main thread to apply
                # NO Tkinter calls from this thread!
                app.chat_panel._pending_brains = brains
                
                logger.debug(f"Chat panel brain list loaded: {len(brains) if brains else 0} brains")
    except Exception as e:
        logger.warning(f"Failed to load chat panel brain list: {e}")


def _load_brains_panel_sync(app: Any) -> None:
    """Load brains panel data synchronously during startup.
    
    This runs in a background thread. Data is loaded here, but UI updates
    are stored for the main thread to apply later.
    
    Args:
        app: AiosTkApp instance
    """
    try:
        if hasattr(app, 'brains_panel') and app.brains_panel:
            # Load brains data directly (this calls CLI - the slow part)
            from ..components.brains_panel.data_loading import refresh_brains_data, refresh_experts_data
            
            refresh_brains_data(app.brains_panel)
            refresh_experts_data(app.brains_panel)
            
            # Store completion flag for main thread
            # NO Tkinter calls from this thread!
            app.brains_panel._data_loaded = True
            
            logger.debug(f"Brains panel data loaded from CLI")
    except Exception as e:
        logger.warning(f"Failed to load brains panel data: {e}")


def _load_mcp_panel_sync(app: Any) -> None:
    """Load MCP panel data synchronously during startup.
    
    This function runs in a background thread. It loads the config data
    but schedules the UI update on the main thread.
    
    Args:
        app: AiosTkApp instance
    """
    try:
        if hasattr(app, 'mcp_panel') and app.mcp_panel:
            # Load config data directly (file I/O - can be done in background)
            servers = app.mcp_panel._load_servers_config()
            tools = app.mcp_panel._load_tools_config()
            
            # Store data for main thread to apply
            # NO Tkinter calls from this thread!
            app.mcp_panel._pending_servers = servers
            app.mcp_panel._pending_tools = tools
            app.mcp_panel._data_loaded = True
            
            logger.debug("MCP panel data loaded")
    except Exception as e:
        logger.warning(f"Failed to load MCP panel data: {e}")


def _load_resources_panel_sync(app: Any) -> None:
    """Load resources panel data (caps and device detection) in background.
    
    This function runs in a background thread. It loads caps from CLI
    and detects devices, then stores results for main thread to apply.
    
    Args:
        app: AiosTkApp instance
    """
    try:
        if not hasattr(app, 'resources_panel') or not app.resources_panel:
            return
        
        # Load caps (CLI calls - slow)
        caps_data = None
        if hasattr(app, '_resources_fetch_caps_fn'):
            try:
                caps_data = app._resources_fetch_caps_fn()
                logger.debug(f"Resource caps loaded: {caps_data}")
            except Exception as e:
                logger.warning(f"Failed to load resource caps: {e}")
        
        # Detect devices (torch calls - can be slow)
        device_info = None
        if hasattr(app, '_resources_detect_devices_fn'):
            try:
                device_info = app._resources_detect_devices_fn()
                logger.debug(f"Device detection complete: {len(device_info.get('cuda_devices', []))} GPU(s)")
            except Exception as e:
                logger.warning(f"Device detection failed: {e}")
        
        # Store results for main thread to apply
        # NO Tkinter calls from this thread!
        app.resources_panel._pending_caps = caps_data
        app.resources_panel._pending_devices = device_info
        app.resources_panel._data_loaded = True
        
    except Exception as e:
        logger.warning(f"Failed to load resources panel data: {e}")


def _load_evaluation_panel_sync(app: Any) -> None:
    """Load evaluation panel data (history database) in background.
    
    This function runs in a background thread. It initializes the database
    and stores the history instance for main thread to apply.
    
    Args:
        app: AiosTkApp instance
    """
    try:
        if not hasattr(app, 'evaluation_panel') or not app.evaluation_panel:
            return
        
        # Initialize history database (disk I/O - can be slow)
        history_db_path = app.evaluation_panel._history_db_path
        try:
            from aios.core.evaluation import EvaluationHistory
            history = EvaluationHistory(history_db_path)
            
            # Store for main thread to apply
            # NO Tkinter calls from this thread!
            app.evaluation_panel._pending_history = history
            app.evaluation_panel._data_loaded = True
            
            logger.debug("Evaluation history database initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize evaluation history: {e}")
        
    except Exception as e:
        logger.warning(f"Failed to load evaluation panel data: {e}")


def _load_help_panel_sync(app: Any) -> None:
    """Load help panel data (search index) in background.
    
    This function runs in a background thread. It builds/loads the search
    index for documentation.
    
    Args:
        app: AiosTkApp instance
    """
    try:
        if not hasattr(app, 'help_panel') or not app.help_panel:
            return
        
        # Build search index (file I/O - can be slow)
        try:
            success = app.help_panel._search_engine.build_index()
            
            # Store result for main thread
            # NO Tkinter calls from this thread!
            app.help_panel._index_loaded = success
            app.help_panel._data_loaded = True
            
            logger.debug(f"Help panel search index loaded: {success}")
        except Exception as e:
            logger.warning(f"Failed to load help panel search index: {e}")
        
    except Exception as e:
        logger.warning(f"Failed to load help panel data: {e}")


def _load_hrm_training_panel_sync(app: Any) -> None:
    """Load HRM training panel data (last safe batches, VRAM estimate) in background.
    
    This function runs in a background thread. It loads state and performs
    initial VRAM estimation.
    
    Args:
        app: AiosTkApp instance
    """
    try:
        if not hasattr(app, 'hrm_training_panel') or not app.hrm_training_panel:
            return
        
        # Prefill from last safe batches (file I/O)
        try:
            from ..components.hrm_training_panel.state_management import prefill_last_safe_batches
            prefill_last_safe_batches(app.hrm_training_panel)
            logger.debug("HRM training panel: last safe batches loaded")
        except Exception as e:
            logger.warning(f"Failed to prefill last safe batches: {e}")
        
        # Perform initial VRAM estimate (can be slow)
        try:
            from ..components.hrm_training_panel.memory_estimation import update_vram_estimate
            update_vram_estimate(app.hrm_training_panel)
            logger.debug("HRM training panel: VRAM estimate completed")
        except Exception as e:
            logger.warning(f"Failed to perform initial VRAM estimate: {e}")
        
        # Mark as loaded
        app.hrm_training_panel._data_loaded = True
        
    except Exception as e:
        logger.warning(f"Failed to load HRM training panel data: {e}")


def create_evaluation_panel_on_demand(app: Any) -> None:
    """
    Create the evaluation panel on-demand when user first accesses it.
    
    This avoids the 11-second startup penalty from importing lm_eval.
    
    Args:
        app: AiosTkApp instance
    """
    if app.evaluation_panel is not None:
        return  # Already created
    
    try:
        logger.info("Creating evaluation panel on-demand...")
        from ..components.evaluation_panel import EvaluationPanel
        
        app.evaluation_panel = EvaluationPanel(
            app.evaluation_tab,
            run_cli=app._run_cli,
            append_out=app._append_out,
            save_state_fn=app._save_state,
            worker_pool=app._worker_pool,
            on_list_brains=app._on_list_brains if hasattr(app, '_on_list_brains') else None,
        )
        logger.info("Evaluation panel created successfully")
    except Exception as e:
        logger.error(f"Failed to create evaluation panel: {e}", exc_info=True)


def _initialize_resources_panel(app: Any) -> None:
    """Initialize Resources panel with device detection."""
    import os
    
    # Detect CPU cores
    cores = os.cpu_count() or 4
    
    # Device detection function
    def _detect_devices_info() -> dict:
        # Devices command not yet implemented - use default device detection
        try:
            import torch
            cuda_available = torch.cuda.is_available() if hasattr(torch, "cuda") else False
            cuda_devices = []
            
            if cuda_available:
                try:
                    device_count = torch.cuda.device_count()
                    for i in range(device_count):
                        try:
                            name = torch.cuda.get_device_name(i)
                            props = torch.cuda.get_device_properties(i)
                            total_mem_mb = props.total_memory // (1024 * 1024)
                            cuda_devices.append({
                                "id": i,
                                "name": name,
                                "total_mem_mb": total_mem_mb,
                            })
                        except Exception:
                            cuda_devices.append({"id": i, "name": f"CUDA Device {i}", "total_mem_mb": 0})
                except Exception:
                    pass
            
            return {
                "cuda_available": cuda_available,
                "cuda_devices": cuda_devices,
                "nvidia_smi_devices": cuda_devices,  # Same for now
            }
        except Exception:
            return {"cuda_available": False, "cuda_devices": [], "nvidia_smi_devices": []}
    
    # Apply resource caps function
    def _apply_caps(dataset_cap_gb, model_cap_gb, per_brain_cap_gb) -> dict:
        out = {}
        try:
            args = ["datasets-config-caps"]
            if dataset_cap_gb is not None and dataset_cap_gb > 0:
                args += ["--cap_gb", str(dataset_cap_gb)]
            if len(args) > 1:
                raw = app._run_cli(args)
                out["datasets"] = app._parse_cli_dict(raw)
        except Exception:
            pass
        try:
            args = ["brains", "config-set"]
            args += ["--config", "config/default.yaml"]
            if model_cap_gb is not None and model_cap_gb > 0:
                args += ["--storage_limit_gb", str(model_cap_gb)]
            if per_brain_cap_gb is not None and per_brain_cap_gb > 0:
                args += ["--per_brain_limit_gb", str(per_brain_cap_gb)]
            if len(args) > 4:
                raw2 = app._run_cli(args)
                out["brains"] = app._parse_cli_dict(raw2)
        except Exception:
            pass
        return out
    
    # Fetch current caps function
    def _fetch_caps() -> dict:
        data = {}
        try:
            ds = app._parse_cli_dict(app._run_cli(["datasets-stats"]) or "{}")
            if isinstance(ds, dict):
                cap = ds.get("cap_gb")
                if isinstance(cap, (int, float)):
                    data["dataset_cap_gb"] = float(cap)
        except Exception:
            pass
        try:
            cfg = app._parse_cli_dict(app._run_cli(["brains", "config-show"]) or "{}")
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
    
    # Create resources panel
    try:
        from ..components import ResourcesPanel  # Lazy import
        
        app.resources_panel = ResourcesPanel(
            app.resources_tab,
            cores=cores,
            detect_fn=_detect_devices_info,
            apply_caps_fn=_apply_caps,
            fetch_caps_fn=_fetch_caps,
            save_state_fn=app._save_state,
            root=app.root,
        )
        
        # Store functions for async loading
        app._resources_fetch_caps_fn = _fetch_caps
        app._resources_detect_devices_fn = _detect_devices_info
            
    except Exception as e:
        logger.error(f"Failed to initialize resources panel: {e}")
        return
    
    # Load resources settings from config file (source of truth)
    try:
        from ..components.resources_panel.config_persistence import load_resources_from_config
        config_resources = load_resources_from_config()
        if config_resources:
            app.resources_panel.set_values(config_resources)
            logger.info("Loaded resources settings from config/default.yaml")
    except Exception as e:
        logger.warning(f"Failed to load resources from config: {e}")


def _configure_python_logging(app: Any) -> None:
    """Configure Python logging to route to Debug panel."""
    import logging
    import tkinter as tk
    
    # Register debug panel as handler for all log categories
    try:
        for category in LogCategory:
            if category != LogCategory.DATASET:
                app._log_router.register_handler(
                    category,
                    lambda msg, lvl=None, cat=category: app.debug_panel.write(msg, cat.value, lvl)
                )
            else:
                # Dataset logs route to debug panel with "dataset" category
                app._log_router.register_handler(
                    category,
                    lambda msg, lvl=None: app.debug_panel.write(msg, "dataset", lvl)
                )
    except Exception as e:
        logger.error(f"Failed to register log handlers: {e}")
    
    # Bridge Python logging to Debug tab
    try:
        class _TkDebugHandler(logging.Handler):
            def __init__(self, root: tk.Tk, log_router, err_cb):
                super().__init__()
                self._root = root
                self._log_router = log_router
                self._err = err_cb
                self.setFormatter(logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
                ))
            
            def emit(self, record: logging.LogRecord) -> None:
                try:
                    msg = self.format(record)
                except Exception:
                    msg = record.getMessage()
                
                level_name = record.levelname
                
                # Determine category
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
                        self._log_router.log(msg, category, level_name)
                        if record.levelno >= logging.ERROR:
                            exc_text = getattr(record, "exc_text", None)
                            self._err(exc_text or msg)
                    except Exception:
                        pass
                
                try:
                    self._root.after(0, _do_write)
                except Exception:
                    _do_write()
        
        app._debug_log_handler = _TkDebugHandler(app.root, app._log_router, app.debug_panel.set_error)
        app._debug_log_handler.setLevel(logging.DEBUG)
        logging.getLogger("aios").addHandler(app._debug_log_handler)
        logging.getLogger().addHandler(app._debug_log_handler)
    except Exception as e:
        logger.error(f"Failed to configure Python logging handler: {e}")


def load_all_panel_data(app: Any, update_status_fn: Any) -> None:
    """Load all panel data using maximum available threads for optimal performance.
    
    This function loads data for all panels that require it, using the system's
    maximum available threads to parallelize the work while keeping the loading 
    screen visible and updating status messages. This ensures all data is ready 
    before the user can interact with the application.
    
    Threads only do data loading - NO Tkinter calls. UI updates happen on main
    thread after data loading completes.
    
    Args:
        app: AiosTkApp instance with panels already created
        update_status_fn: Function to call to update loading status message
    """
    import threading
    import concurrent.futures
    import os
    
    logger.info("Loading all panel data with threading...")
    
    # Use maximum available threads (CPU count or default to 4)
    max_workers = os.cpu_count() or 4
    logger.info(f"Using {max_workers} worker threads for data loading")
    
    # Track completion status
    results = {
        'chat': {'done': False, 'error': None},
        'brains': {'done': False, 'error': None},
        'mcp': {'done': False, 'error': None},
        'resources': {'done': False, 'error': None},
        'evaluation': {'done': False, 'error': None},
        'help': {'done': False, 'error': None},
        'hrm_training': {'done': False, 'error': None},
    }
    
    # Thread functions that update results
    def load_chat():
        try:
            _load_chat_brains_sync(app)
            results['chat']['done'] = True
        except Exception as e:
            results['chat']['error'] = str(e)
            logger.error(f"Error loading chat data: {e}")
        return 'chat'
    
    def load_brains():
        try:
            _load_brains_panel_sync(app)
            results['brains']['done'] = True
        except Exception as e:
            results['brains']['error'] = str(e)
            logger.error(f"Error loading brains data: {e}")
        return 'brains'
    
    def load_mcp():
        try:
            _load_mcp_panel_sync(app)
            results['mcp']['done'] = True
        except Exception as e:
            results['mcp']['error'] = str(e)
            logger.error(f"Error loading MCP data: {e}")
        return 'mcp'
    
    def load_resources():
        try:
            _load_resources_panel_sync(app)
            results['resources']['done'] = True
        except Exception as e:
            results['resources']['error'] = str(e)
            logger.error(f"Error loading resources data: {e}")
        return 'resources'
    
    def load_evaluation():
        try:
            _load_evaluation_panel_sync(app)
            results['evaluation']['done'] = True
        except Exception as e:
            results['evaluation']['error'] = str(e)
            logger.error(f"Error loading evaluation data: {e}")
        return 'evaluation'
    
    def load_help():
        try:
            _load_help_panel_sync(app)
            results['help']['done'] = True
        except Exception as e:
            results['help']['error'] = str(e)
            logger.error(f"Error loading help data: {e}")
        return 'help'
    
    def load_hrm_training():
        try:
            _load_hrm_training_panel_sync(app)
            results['hrm_training']['done'] = True
        except Exception as e:
            results['hrm_training']['error'] = str(e)
            logger.error(f"Error loading HRM training data: {e}")
        return 'hrm_training'
    
    # Start loading with thread pool executor
    update_status_fn("Loading data in parallel...")
    app.root.update_idletasks()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(load_chat): 'chat',
            executor.submit(load_brains): 'brains',
            executor.submit(load_mcp): 'mcp',
            executor.submit(load_resources): 'resources',
            executor.submit(load_evaluation): 'evaluation',
            executor.submit(load_help): 'help',
            executor.submit(load_hrm_training): 'hrm_training',
        }
        
        # Wait for completion with VERY infrequent UI updates to prevent hangs
        completed = set()
        import time
        last_update = time.time()
        
        while len(completed) < len(futures):
            # Check for newly completed tasks
            for future in list(futures.keys()):
                if future not in completed and future.done():
                    completed.add(future)
                    task_name = futures[future]
                    logger.debug(f"Task '{task_name}' completed")
            
            # Update status based on what's still loading (only every 500ms)
            current_time = time.time()
            if current_time - last_update >= 0.5:  # 500ms = 2 updates/sec
                loading_items = []
                if not results['chat']['done']:
                    loading_items.append("chat")
                if not results['brains']['done']:
                    loading_items.append("brains")
                if not results['mcp']['done']:
                    loading_items.append("MCP")
                if not results['resources']['done']:
                    loading_items.append("resources")
                if not results['evaluation']['done']:
                    loading_items.append("evaluation")
                if not results['help']['done']:
                    loading_items.append("help")
                if not results['hrm_training']['done']:
                    loading_items.append("training")
                
                if loading_items:
                    update_status_fn(f"Loading {', '.join(loading_items)}...")
                
                # Only update_idletasks, not full update
                app.root.update_idletasks()
                last_update = current_time
            
            # Sleep to avoid busy-waiting and reduce CPU usage
            time.sleep(0.1)  # 100ms sleep
    
    # NOW apply UI updates on main thread (after threads complete, before mainloop)
    try:
        # Apply chat panel updates
        if hasattr(app, 'chat_panel') and hasattr(app.chat_panel, '_pending_brains'):
            brains = app.chat_panel._pending_brains
            if hasattr(app.chat_panel, 'brain_combo') and hasattr(app.chat_panel, 'brain_var'):
                if brains:
                    app.chat_panel.brain_combo["values"] = brains
                    if not app.chat_panel.brain_var.get() or app.chat_panel.brain_var.get() == "<default>":
                        app.chat_panel.brain_var.set(brains[0] if brains else "<default>")
                else:
                    app.chat_panel.brain_combo["values"] = ["<no brains>"]
                    app.chat_panel.brain_var.set("<no brains>")
            delattr(app.chat_panel, '_pending_brains')
        
        # Apply brains panel updates
        if hasattr(app, 'brains_panel') and hasattr(app.brains_panel, '_data_loaded'):
            total_experts = int(app.brains_panel.total_experts_var.get() or "0")
            total_brains = int(app.brains_panel.brain_count_var.get() or "0")
            if hasattr(app.brains_panel, 'status_var'):
                app.brains_panel.status_var.set(f"{total_brains} brains, {total_experts} experts")
            delattr(app.brains_panel, '_data_loaded')
        
        # Apply MCP panel updates
        if hasattr(app, 'mcp_panel') and hasattr(app.mcp_panel, '_data_loaded'):
            # Populate UI with loaded data
            from ..components.mcp_manager_panel.ui_updaters import populate_servers_tree, populate_tools_tree, update_summary
            
            servers = getattr(app.mcp_panel, '_pending_servers', None)
            tools = getattr(app.mcp_panel, '_pending_tools', None)
            
            if servers is not None:
                populate_servers_tree(app.mcp_panel.servers_tree, servers)
            if tools is not None:
                category_filter = app.mcp_panel.tool_category_var.get()
                populate_tools_tree(app.mcp_panel.tools_tree, tools, category_filter)
            
            # Update summary
            if servers is not None and tools is not None:
                update_summary(
                    servers,
                    tools,
                    app.mcp_panel.servers_count_var,
                    app.mcp_panel.servers_active_var,
                    app.mcp_panel.tools_enabled_var,
                )
            
            # Clean up
            if hasattr(app.mcp_panel, '_pending_servers'):
                delattr(app.mcp_panel, '_pending_servers')
            if hasattr(app.mcp_panel, '_pending_tools'):
                delattr(app.mcp_panel, '_pending_tools')
            delattr(app.mcp_panel, '_data_loaded')
        
        # Apply resources panel updates
        if hasattr(app, 'resources_panel') and hasattr(app.resources_panel, '_data_loaded'):
            # Apply caps if loaded
            caps_data = getattr(app.resources_panel, '_pending_caps', None)
            if caps_data:
                app.resources_panel.set_caps(caps_data)
                logger.debug("Resource caps applied to UI")
            
            # Apply device detection if loaded
            device_info = getattr(app.resources_panel, '_pending_devices', None)
            if device_info:
                app.resources_panel.set_detected(device_info)
                logger.info(f"Device detection applied: {len(device_info.get('cuda_devices', []))} GPU(s) found")
            
            # Load resources settings from config file (source of truth)
            try:
                from ..components.resources_panel.config_persistence import load_resources_from_config
                config_resources = load_resources_from_config()
                if config_resources:
                    app.resources_panel.set_values(config_resources)
                    logger.info("Loaded resources settings from config/default.yaml")
            except Exception as e:
                logger.warning(f"Failed to load resources from config: {e}")
            
            # Clean up
            if hasattr(app.resources_panel, '_pending_caps'):
                delattr(app.resources_panel, '_pending_caps')
            if hasattr(app.resources_panel, '_pending_devices'):
                delattr(app.resources_panel, '_pending_devices')
            delattr(app.resources_panel, '_data_loaded')
        
        # Apply evaluation panel updates
        if hasattr(app, 'evaluation_panel') and hasattr(app.evaluation_panel, '_data_loaded'):
            # Set history instance
            history = getattr(app.evaluation_panel, '_pending_history', None)
            if history:
                app.evaluation_panel._set_history(history)
                logger.debug("Evaluation history database initialized and set")
            
            # Clean up
            if hasattr(app.evaluation_panel, '_pending_history'):
                delattr(app.evaluation_panel, '_pending_history')
            delattr(app.evaluation_panel, '_data_loaded')
        
        # Apply help panel updates
        if hasattr(app, 'help_panel') and hasattr(app.help_panel, '_data_loaded'):
            # Index is already loaded in background thread
            # Just mark as complete - no UI updates needed
            logger.debug("Help panel search index loaded and ready")
            
            # Clean up
            delattr(app.help_panel, '_data_loaded')
        
        # Apply HRM training panel updates  
        if hasattr(app, 'hrm_training_panel') and hasattr(app.hrm_training_panel, '_data_loaded'):
            # Data already loaded in background thread (prefill and VRAM estimate)
            # Just mark as complete - no UI updates needed
            logger.debug("HRM training panel data loaded and ready")
            
            # Clean up
            delattr(app.hrm_training_panel, '_data_loaded')
    except Exception as e:
        logger.warning(f"Error applying UI updates: {e}")
    
    # Final status update
    update_status_fn("Data loading complete!")
    app.root.update_idletasks()
    
    logger.info("All panel data loaded successfully")
