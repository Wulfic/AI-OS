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

# Import all panel components EXCEPT EvaluationPanel (lazy loaded)
from ..components import (
    OutputPanel, ChatPanel, RichChatPanel, DatasetBuilderPanel, ResourcesPanel,
    StatusBar, DebugPanel, BrainsPanel, HRMTrainingPanel,
    DatasetDownloadPanel, SettingsPanel, MCPManagerPanel, HelpPanel
)
# EvaluationPanel imported lazily when needed to avoid loading heavy lm_eval deps at startup

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
    try:
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
    try:
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
    _initialize_resources_panel(app)
    log_timing("Resources panel")
    
    # ===== DEBUG PANEL =====
    try:
        app.debug_panel = DebugPanel(app.debug_tab)
        log_timing("Debug panel")
    except Exception as e:
        logger.error(f"Failed to initialize debug panel: {e}")
    
    # ===== SETTINGS PANEL =====
    try:
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
    try:
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
    try:
        app.brains_panel = BrainsPanel(
            app.brains_tab,
            run_cli=app._run_cli,
            append_out=app._append_out,
            on_goal_add=app._on_goal_add_for_brain,
            on_goals_list=app._on_goals_list_for_brain,
            on_goal_remove=app._on_goal_remove,
            worker_pool=app._worker_pool,
        )
        # Defer refresh to avoid blocking GUI startup with CLI subprocess calls
        log_timing("Brains panel")
    except Exception as e:
        logger.error(f"Failed to initialize brains panel: {e}")
    
    # ===== MCP MANAGER PANEL =====
    try:
        app.mcp_panel = MCPManagerPanel(
            app.mcp_tab,
            run_cli=app._run_cli,
            append_out=app._append_out,
            save_state_fn=app._save_state,
        )
        # Defer refresh to avoid blocking GUI startup with CLI subprocess calls
        log_timing("MCP Manager panel")
    except Exception as e:
        logger.error(f"Failed to initialize MCP panel: {e}")
    
    # ===== HRM TRAINING PANEL =====
    try:
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
    # Defer evaluation panel creation completely to avoid 11s startup delay from lm_eval imports
    # It will be created on-demand when user first switches to the Evaluation tab
    app.evaluation_panel = None
    log_timing("Evaluation panel (deferred)")
    
    # ===== HELP PANEL =====
    try:
        app.help_panel = HelpPanel(app.help_tab, project_root=app._project_root)
        
        # Connect help_panel to settings panel for theme updates
        if hasattr(app, 'settings_panel') and app.settings_panel:
            app.settings_panel._help_panel = app.help_panel
        log_timing("Help panel")
    except Exception as e:
        logger.error(f"Failed to initialize help panel: {e}")

    # ===== STATUS BAR =====
    try:
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
    
    This includes:
    - Refreshing panels that make CLI subprocess calls
    - Loading data that requires external processes
    
    Called asynchronously after the main window is shown to avoid blocking startup.
    All refresh operations are submitted to worker pool to prevent GUI blocking.
    
    Args:
        app: AiosTkApp instance with panels already created
    """
    logger.info("Starting deferred panel initialization...")
    
    # Submit all refresh operations to worker pool to prevent blocking GUI thread
    def _refresh_chat_panel():
        """Refresh chat panel brain list in background."""
        try:
            if hasattr(app, 'chat_panel') and app.chat_panel:
                app.chat_panel.refresh_brain_list()
                logger.debug("Chat panel brain list refreshed")
        except Exception as e:
            logger.warning(f"Failed to refresh chat panel brain list: {e}")
    
    def _refresh_brains_panel():
        """Refresh brains panel in background."""
        try:
            if hasattr(app, 'brains_panel') and app.brains_panel:
                app.brains_panel.refresh()
                logger.debug("Brains panel refreshed")
        except Exception as e:
            logger.warning(f"Failed to refresh brains panel: {e}")
    
    def _refresh_mcp_panel():
        """Refresh MCP panel in background."""
        try:
            if hasattr(app, 'mcp_panel') and app.mcp_panel:
                app.mcp_panel.refresh()
                logger.debug("MCP panel refreshed")
        except Exception as e:
            logger.warning(f"Failed to refresh MCP panel: {e}")
    
    # Submit to worker pool for async execution
    if hasattr(app, '_worker_pool') and app._worker_pool:
        app._worker_pool.submit(_refresh_chat_panel)
        app._worker_pool.submit(_refresh_brains_panel)
        app._worker_pool.submit(_refresh_mcp_panel)
        logger.info("Deferred panel initialization submitted to worker pool")
    else:
        # Fallback to synchronous if worker pool not available
        logger.warning("Worker pool not available, running deferred initialization synchronously")
        _refresh_chat_panel()
        _refresh_brains_panel()
        _refresh_mcp_panel()
        logger.info("Deferred panel initialization complete (synchronous fallback)")


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
        app.resources_panel = ResourcesPanel(
            app.resources_tab,
            cores=cores,
            detect_fn=_detect_devices_info,
            apply_caps_fn=_apply_caps,
            fetch_caps_fn=_fetch_caps,
            save_state_fn=app._save_state,
            root=app.root,
        )
        # Defer caps fetching to avoid blocking GUI startup with CLI subprocess calls
        # This will be called asynchronously after GUI is displayed
        app.root.after(100, lambda: app.resources_panel.set_caps(_fetch_caps()))
    except Exception as e:
        logger.error(f"Failed to initialize resources panel: {e}")
        return
    
    # Defer device detection to avoid blocking GUI startup with torch initialization
    # This ensures the window appears quickly, then devices are detected
    def _deferred_device_detection():
        try:
            info = _detect_devices_info()
            if isinstance(info, dict):
                app.resources_panel.set_detected(info)
                logger.info(f"Device detection complete: {len(info.get('cuda_devices', []))} GPU(s) found")
        except Exception as e:
            logger.warning(f"Device detection failed: {e}")
    
    # Schedule device detection after a brief delay
    app.root.after(200, _deferred_device_detection)
    
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
