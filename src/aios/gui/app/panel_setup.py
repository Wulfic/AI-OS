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
from typing import Any, TYPE_CHECKING, Callable
from contextlib import suppress
import logging
import time
from pathlib import Path

import yaml

if TYPE_CHECKING:
    import tkinter as tk

# Lazy imports - panels are imported only when instantiated to minimize startup time
# This reduces initial import overhead by ~10-15 seconds
# Components will be imported in their respective initialization functions

from ..services import LogCategory
from ..utils.resource_management import submit_background

# Apply safety monkeypatches early (no-ops if not applicable)
try:
    from ..monkeypatches import matplotlib_tk_guard  # noqa: F401
except Exception:
    pass

logger = logging.getLogger(__name__)


def _get_debounced_save_state(app: Any, delay_ms: int = 500) -> Callable[[], None]:
    """Return a zero-argument callback that schedules a debounced state save."""
    schedule_fn = getattr(app, "schedule_state_save", None)
    save_fn = getattr(app, "_save_state", None)

    if callable(schedule_fn):
        def _callback() -> None:
            try:
                schedule_fn(delay_ms)
            except TypeError:
                schedule_fn()
        return _callback

    if callable(save_fn):
        return save_fn  # type: ignore[return-value]

    return lambda: None


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
    import time as time_module
    
    logger.info("Starting panel initialization")
    start_time = time_module.time()
    last_time = start_time
    
    def update_loading(text: str) -> None:
        """Update loading screen if available with minimal UI blocking."""
        logger.debug(f"Panel loading: {text}")
        try:
            if hasattr(app, '_loading_canvas') and app._loading_canvas:
                if hasattr(app._loading_canvas, '_status_text_id'):
                    app._loading_canvas.itemconfig(app._loading_canvas._status_text_id, text=text)
                    # Use after_idle instead of update_idletasks to prevent blocking
                    app.root.after_idle(lambda: None)
        except Exception:
            pass
    
    def log_timing(panel_name: str) -> None:
        """Log timing for each panel initialization."""
        nonlocal last_time
        current = time_module.time()
        step_duration = current - last_time
        msg = f"[PANEL TIMING] {panel_name}: {step_duration:.3f}s"
        logger.info(msg)
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
            """Append *msg* to the shared dataset output pane on the UI thread."""

            text = f"{msg}\n"

            def _write() -> None:
                try:
                    app.dataset_output_text.insert(tk.END, text)
                    app.dataset_output_text.see(tk.END)
                except Exception:
                    pass

            try:
                app.post_to_ui(_write)
            except Exception:
                _write()
        
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
            from tkinter import ttk  # Lazy import to keep startup fast

            hidden_parent = ttk.Frame(app.root)
            queue_bridge = OutputPanel(hidden_parent, show_summary_toggle=False)
            hidden_parent.pack_forget()  # Keep bridge headless so it does not surface in the UI
            
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
            worker_pool=app._worker_pool,
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
    resources_save_cb = _get_debounced_save_state(app, delay_ms=900)
    settings_save_cb = _get_debounced_save_state(app, delay_ms=600)
    general_save_cb = _get_debounced_save_state(app, delay_ms=700)
    hrm_save_cb = _get_debounced_save_state(app, delay_ms=1200)

    _initialize_resources_panel(app, resources_save_cb)
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
            save_state_fn=settings_save_cb,
            chat_panel=None,  # Will be set after chat_panel is created
            help_panel=None,  # Will be set after help_panel is created
            debug_panel=app.debug_panel if hasattr(app, 'debug_panel') else None,  # Connect to debug panel
            worker_pool=app._worker_pool,
        )
        log_timing("Settings panel")
    except Exception as e:
        logger.error(f"Failed to initialize settings panel: {e}", exc_info=True)
        # Ensure app.settings_panel exists even if initialization failed
        app.settings_panel = None
    
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
        if hasattr(app, 'settings_panel') and app.settings_panel:
            app.settings_panel._chat_panel = app.chat_panel
        log_timing("Chat panel")
    except Exception as e:
        logger.error(f"Failed to initialize chat panel: {e}", exc_info=True)
    
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
            save_state_fn=general_save_cb,
            post_to_ui=app.post_to_ui,
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
            save_state_fn=hrm_save_cb,
            worker_pool=app._worker_pool,
            resources_panel=app.resources_panel,
            post_to_ui=app.post_to_ui,
        )
        log_timing("HRM Training panel")
    except Exception as e:
        logger.error(f"Failed to initialize HRM training panel: {e}", exc_info=True)
        app.hrm_training_panel = None
    
    # ===== EVALUATION PANEL =====
    # Load evaluation panel during startup
    update_loading("Loading Evaluation panel...")
    
    # CRITICAL: Temporarily disable canvas updates during heavy panel creation
    # This prevents Configure events from blocking the main thread
    try:
        if hasattr(app, '_loading_canvas'):
            app._loading_canvas.unbind('<Configure>')
    except Exception:
        pass
    
    try:
        import time
        eval_start = time.time()
        
        from ..components import EvaluationPanel  # Lazy import
        import_time = time.time() - eval_start
        logger.info(f"[EVAL TIMING] Import time: {import_time:.3f}s")
        
        panel_start = time.time()
        app.evaluation_panel = EvaluationPanel(
            app.evaluation_tab,
            run_cli=app._run_cli,
            append_out=app._append_out,
            save_state_fn=general_save_cb,
            worker_pool=app._worker_pool,
            on_list_brains=app._on_list_brains if hasattr(app, '_on_list_brains') else None,
            resources_panel=getattr(app, 'resources_panel', None),
        )
        panel_create_time = time.time() - panel_start
        logger.info(f"[EVAL TIMING] Panel creation time: {panel_create_time:.3f}s")
        
        log_timing("Evaluation panel")
    except Exception as e:
        logger.error(f"Failed to initialize evaluation panel: {e}")
        app.evaluation_panel = None
    finally:
        # Re-enable canvas updates after panel creation
        try:
            if hasattr(app, '_loading_canvas') and hasattr(app, '_canvas_update_handler'):
                app._loading_canvas.bind('<Configure>', app._canvas_update_handler)
        except Exception:
            pass
    
    # ===== HELP PANEL =====
    update_loading("Loading Help panel...")
    try:
        from ..components import HelpPanel  # Lazy import
        
        app.help_panel = HelpPanel(
            app.help_tab,
            project_root=app._project_root,
            worker_pool=app._worker_pool,
        )
        
        # Connect help_panel to settings panel for theme updates and vice versa
        if hasattr(app, 'settings_panel') and app.settings_panel:
            app.settings_panel._help_panel = app.help_panel
            app.help_panel._settings_panel = app.settings_panel
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
    
    total_time = time_module.time() - start_time
    msg = f"[PANEL TIMING] Total panel initialization time: {total_time:.3f}s"
    logger.info(msg)
    logger.info(f"Panel initialization complete: {total_time:.3f}s")


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
    import time

    try:
        if not hasattr(app, 'resources_panel') or not app.resources_panel:
            return

        logger.debug("Resources panel data load started")
        caps_data: dict[str, Any] = {}
        device_info: dict[str, Any] | None = None
        needs_device_refresh = False
        caps_refresh_needed = True

        if hasattr(app, '_resources_fetch_caps_fn'):
            logger.debug("Resource caps loading deferred to background refresh")
        else:
            logger.debug("Resource caps fetch function unavailable during startup; skipping immediate load")

        detect_started = time.time()

        cached_loader = getattr(app, '_resources_cached_devices_fn', None)
        if callable(cached_loader):
            try:
                cached = cached_loader() or {}
                if cached:
                    gpu_count = len(cached.get('cuda_devices', [])) if isinstance(cached, dict) else 0
                    logger.debug(
                        "Cached device info loaded in %.3fs: %d GPU(s)",
                        time.time() - detect_started,
                        gpu_count,
                    )
                    device_info = dict(cached)
                    needs_device_refresh = True  # always refresh cached data in background
                else:
                    logger.debug(
                        "No cached device info available after %.3fs",
                        time.time() - detect_started,
                    )
                    device_info = {}
                    needs_device_refresh = True
            except Exception as e:
                logger.warning("Failed to load cached device info: %s", e, exc_info=True)
                device_info = {}
                needs_device_refresh = True
        else:
            logger.debug("Cached device loader unavailable; scheduling fresh detection")
            device_info = {}
            needs_device_refresh = True

        # Store results for main thread to apply
        # NO Tkinter calls from this thread!
        pending_caps: dict[str, Any] = dict(caps_data)
        if caps_refresh_needed:
            pending_caps.setdefault('_needs_refresh', True)
        app.resources_panel._pending_caps = pending_caps
        if needs_device_refresh:
            if not isinstance(device_info, dict):
                device_info = {}
            device_info.setdefault('_needs_refresh', True)
        app.resources_panel._pending_devices = device_info if isinstance(device_info, dict) else {}
    except Exception as e:
        logger.warning("Failed to load resources panel data: %s", e, exc_info=True)
    finally:
        # Ensure UI knows loading completed even if partial data
        try:
            if hasattr(app, 'resources_panel') and app.resources_panel:
                app.resources_panel._data_loaded = True
        except Exception:
            logger.debug("Unable to mark resources panel as loaded", exc_info=True)


def _load_settings_panel_sync(app: Any) -> None:
    """Load settings panel config values from config file.
    
    This loads settings from config/default.yaml so that config values
    take precedence over hardcoded defaults.
    
    Args:
        app: AiosTkApp instance
    """
    try:
        if not hasattr(app, 'settings_panel') or not app.settings_panel:
            logger.debug("Settings panel not available, skipping config load")
            return
        
        from ..components.settings_panel.config_persistence import load_settings_from_config
        settings = load_settings_from_config()
        
        if not settings:
            logger.debug("No settings found in config file")
            return
        
        # Store loaded settings for main thread to apply
        app.settings_panel._pending_config = settings
        logger.info(f"Loaded settings from config: {settings}")
        
    except Exception as e:
        logger.warning(f"Failed to load settings from config: {e}", exc_info=True)


def _load_evaluation_panel_sync(app: Any) -> None:
    """Load evaluation panel data (brains list, history database + benchmark tree) in background.
    
    This function runs in a background thread. It loads brains list,
    initializes the database, prepares benchmark tree data, and stores 
    for main thread to apply.
    
    Args:
        app: AiosTkApp instance
    """
    try:
        if not hasattr(app, 'evaluation_panel') or not app.evaluation_panel:
            return
        
        # Load brains list (subprocess call - VERY slow ~9.6s)
        try:
            if app.evaluation_panel._on_list_brains:
                brains = app.evaluation_panel._on_list_brains()
                
                # Store results for main thread to apply
                # NO Tkinter calls from this thread!
                app.evaluation_panel._pending_brains = brains
                
                logger.debug(f"Evaluation panel brain list loaded: {len(brains) if brains else 0} brains")
        except Exception as e:
            logger.warning(f"Failed to load evaluation panel brain list: {e}")
        
        # Initialize history database (disk I/O - can be slow)
        history_db_path = app.evaluation_panel._history_db_path
        try:
            from aios.core.evaluation import EvaluationHistory
            history = EvaluationHistory(history_db_path)
            
            # Store for main thread to apply
            # NO Tkinter calls from this thread!
            app.evaluation_panel._pending_history = history
            
            logger.debug("Evaluation history database initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize evaluation history: {e}")
        
        # Prepare benchmark tree data (CPU-intensive - build data structure)
        try:
            from ..components.evaluation_panel.benchmark_data import BENCHMARKS
            
            # Build tree items structure without Tkinter calls
            tree_items = {}
            tree_structure = []  # List of (parent_id, item_id, text, values, item_info)
            
            for category, benchmarks in BENCHMARKS.items():
                # Prepare category data
                cat_id = f"cat_{category}"
                tree_structure.append((
                    "",  # parent (root)
                    cat_id,
                    "☐",  # text
                    (category, "", ""),  # values
                    {"type": "category", "name": category, "checked": False}
                ))
                tree_items[cat_id] = {"type": "category", "name": category, "checked": False}
                
                # Prepare benchmark data
                for bench_name, bench_desc in benchmarks:
                    item_id = f"bench_{category}_{bench_name}"
                    tree_structure.append((
                        cat_id,  # parent (category)
                        item_id,
                        "☐",  # text
                        ("", bench_name, bench_desc),  # values
                        {
                            "type": "benchmark",
                            "name": bench_name,
                            "category": category,
                            "checked": False
                        }
                    ))
                    tree_items[item_id] = {
                        "type": "benchmark",
                        "name": bench_name,
                        "category": category,
                        "checked": False
                    }
            
            # Store prepared data for main thread
            app.evaluation_panel._pending_tree_structure = tree_structure
            app.evaluation_panel._pending_tree_items = tree_items
            
            logger.debug(f"Prepared {len(tree_structure)} benchmark tree items")
        except Exception as e:
            logger.warning(f"Failed to prepare benchmark tree: {e}")
        
        app.evaluation_panel._data_loaded = True
        
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
            resources_panel=getattr(app, 'resources_panel', None),
        )
        logger.info("Evaluation panel created successfully")
    except Exception as e:
        logger.error(f"Failed to create evaluation panel: {e}", exc_info=True)


def _initialize_resources_panel(app: Any, save_state_cb: Callable[[], None] | None = None) -> None:
    """Initialize Resources panel with device detection."""
    import os
    
    logger.info("Initializing Resources panel")
    
    # Detect CPU cores
    cores = os.cpu_count() or 4
    logger.debug(f"Detected {cores} CPU cores")
    
    project_root_value = getattr(app, "_project_root", None)
    try:
        project_root_path = Path(project_root_value) if project_root_value else Path(__file__).resolve().parents[3]
    except Exception:
        project_root_path = Path(__file__).resolve().parents[3]

    from ..components.resources_panel import device_cache

    cache_max_age_seconds = 12 * 3600

    def _perform_device_detection() -> dict:
        import time as _time
        import subprocess
        import shutil
        
        from aios.core.gpu_vendor import identify_gpu_vendor, detect_xpu_devices, calculate_vendor_summary

        detection_result: dict[str, Any] = {
            "cuda_available": False,
            "cuda_devices": [],
            "nvidia_smi_devices": [],
            "xpu_available": False,
            "xpu_devices": [],
            "vendor_summary": {},
            "rocm": False,
            "detected_at": _time.time(),
            "source": "torch",
        }

        cuda_devices: list[dict[str, Any]] = []
        torch_reported_devices = False
        rocm = False

        try:
            import torch
            
            # Check for ROCm build
            rocm = bool(getattr(torch.version, "hip", None))
            detection_result["rocm"] = rocm

            cuda_available = torch.cuda.is_available() if hasattr(torch, "cuda") else False
            detection_result["cuda_available"] = bool(cuda_available)

            if cuda_available:
                logger.debug("CUDA is available, enumerating devices")
                try:
                    device_count = torch.cuda.device_count()
                    logger.info(f"Found {device_count} CUDA device(s)")
                    for i in range(device_count):
                        try:
                            name = torch.cuda.get_device_name(i)
                            props = torch.cuda.get_device_properties(i)
                            total_mem_mb = int(props.total_memory // (1024 * 1024))
                            vendor = identify_gpu_vendor(name, check_rocm=rocm)
                            cuda_devices.append({
                                "id": i,
                                "name": name,
                                "total_mem_mb": total_mem_mb,
                                "vendor": vendor,
                            })
                            logger.debug(f"CUDA device {i}: [{vendor}] {name} ({total_mem_mb} MB)")
                        except Exception as exc:
                            logger.warning(f"Failed to get info for CUDA device {i}: {exc}")
                            cuda_devices.append({"id": i, "name": f"CUDA Device {i}", "total_mem_mb": 0, "vendor": "Unknown"})
                except Exception as exc:
                    logger.error(f"Failed to enumerate CUDA devices: {exc}")
            else:
                logger.info("CUDA is not available on this system")

            detection_result["cuda_devices"] = cuda_devices
            detection_result["nvidia_smi_devices"] = list(cuda_devices)
            torch_reported_devices = bool(cuda_devices)
            
            # Detect Intel XPU devices
            xpu_available, xpu_devices = detect_xpu_devices()
            detection_result["xpu_available"] = xpu_available
            detection_result["xpu_devices"] = xpu_devices
            if xpu_available:
                logger.info(f"Found {len(xpu_devices)} Intel XPU device(s)")
                
        except Exception as exc:
            logger.error(f"Device detection failed: {exc}", exc_info=True)

        nvidia_devices: list[dict[str, Any]] = []
        try:
            nvidia_smi_path = shutil.which("nvidia-smi")
            if nvidia_smi_path:
                cmd = [
                    nvidia_smi_path,
                    "--query-gpu=index,name,memory.total",
                    "--format=csv,noheader,nounits",
                ]
                env = os.environ.copy()
                for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
                    env.pop(key, None)

                result = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    env=env,
                )
                if result.stdout:
                    for line in result.stdout.strip().splitlines():
                        parts = [piece.strip() for piece in line.split(",")]
                        if len(parts) >= 3:
                            try:
                                idx = int(parts[0])
                            except Exception:
                                continue
                            name = parts[1]
                            try:
                                total_mem_mb = int(float(parts[2]))
                            except Exception:
                                total_mem_mb = 0
                            vendor = identify_gpu_vendor(name)
                            nvidia_devices.append({
                                "id": idx,
                                "name": name,
                                "total_mem_mb": total_mem_mb,
                                "vendor": vendor,
                            })
        except subprocess.TimeoutExpired:
            logger.warning("nvidia-smi query timed out during device detection")
        except Exception as exc:
            logger.debug("nvidia-smi detection failed: %s", exc, exc_info=True)

        if nvidia_devices:
            detection_result["nvidia_smi_devices"] = nvidia_devices
            if not detection_result.get("cuda_devices"):
                detection_result["cuda_devices"] = list(nvidia_devices)
            detection_result["cuda_available"] = True
            detection_result["source"] = "torch+nvidia-smi" if torch_reported_devices else "nvidia-smi"
        else:
            detection_result.setdefault("nvidia_smi_devices", [])

        # Calculate vendor summary for all detected devices
        all_devices = detection_result.get("cuda_devices", []) + detection_result.get("xpu_devices", [])
        detection_result["vendor_summary"] = calculate_vendor_summary(all_devices)

        return detection_result

    def _load_cached_devices() -> dict | None:
        try:
            return device_cache.load_device_cache(project_root_path, max_age_seconds=cache_max_age_seconds)
        except Exception:
            logger.debug("Failed to load cached CUDA device info", exc_info=True)
            return None

    def _detect_devices_info(force_refresh: bool = False) -> dict:
        if not force_refresh:
            cached = _load_cached_devices()
            if cached:
                devices = cached.get("cuda_devices") or cached.get("nvidia_smi_devices") or []
                if devices:
                    age = cached.get("_cache_age_seconds")
                    if isinstance(age, (int, float)):
                        logger.info("Using cached CUDA device info (age %.1fs)", age)
                    else:
                        logger.info("Using cached CUDA device info")
                    cached.setdefault("source", "cache")
                    return cached
                logger.info("Cached CUDA device info empty; forcing fresh detection")

        fresh_info = _perform_device_detection()
        fresh_info["_from_cache"] = False
        try:
            device_cache.save_device_cache(project_root_path, fresh_info)
        except Exception:
            logger.debug("Failed to persist CUDA detection cache", exc_info=True)
        return fresh_info
    
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
        """Fetch current storage caps without invoking the CLI."""

        start_time = time.perf_counter()
        dataset_duration = 0.0
        config_duration = 0.0
        data: dict[str, float] = {}

        try:
            dataset_start = time.perf_counter()
            from aios.data.datasets import datasets_storage_cap_gb

            cap_val = datasets_storage_cap_gb()
            dataset_duration = time.perf_counter() - dataset_start
            if isinstance(cap_val, (int, float)):
                data["dataset_cap_gb"] = float(cap_val)
        except Exception as exc:
            dataset_duration = time.perf_counter() - dataset_start
            logger.debug("Dataset cap lookup failed: %s", exc, exc_info=True)

        try:
            config_start = time.perf_counter()
            cfg_path = Path("config/default.yaml")
            if cfg_path.exists():
                with cfg_path.open("r", encoding="utf-8") as fh:
                    cfg = yaml.safe_load(fh) or {}
                brains_cfg = cfg.get("brains") if isinstance(cfg, dict) else {}
                if isinstance(brains_cfg, dict):
                    global_cap = brains_cfg.get("storage_limit_gb")
                    if not isinstance(global_cap, (int, float)):
                        cap_mb = brains_cfg.get("storage_limit_mb")
                        if isinstance(cap_mb, (int, float)):
                            global_cap = float(cap_mb) / 1024.0
                    if isinstance(global_cap, (int, float)):
                        data["model_cap_gb"] = float(global_cap)

                    overrides = brains_cfg.get("trainer_overrides") or {}
                    if isinstance(overrides, dict):
                        per_brain = overrides.get("width_storage_limit_gb")
                        if not isinstance(per_brain, (int, float)):
                            per_brain_mb = overrides.get("width_storage_limit_mb")
                            if isinstance(per_brain_mb, (int, float)):
                                per_brain = float(per_brain_mb) / 1024.0
                        if isinstance(per_brain, (int, float)):
                            data["per_brain_cap_gb"] = float(per_brain)
            config_duration = time.perf_counter() - config_start
        except Exception as exc:
            config_duration = time.perf_counter() - config_start
            logger.debug("Brains config parse failed: %s", exc, exc_info=True)

        if logger.isEnabledFor(logging.DEBUG):
            total_ms = (time.perf_counter() - start_time) * 1000
            dataset_ms = dataset_duration * 1000
            config_ms = config_duration * 1000
            logger.debug(
                "Fetched storage caps without CLI in %.1f ms (dataset %.1f ms, config %.1f ms)",
                total_ms,
                dataset_ms,
                config_ms,
            )

        return data
    
    # Create resources panel
    try:
        from ..components import ResourcesPanel  # Lazy import
        
        logger.debug("Creating ResourcesPanel instance")
        save_state_fn = save_state_cb or getattr(app, "_save_state", None)

        app.resources_panel = ResourcesPanel(
            app.resources_tab,
            cores=cores,
            detect_fn=_detect_devices_info,
            apply_caps_fn=_apply_caps,
            fetch_caps_fn=_fetch_caps,
            save_state_fn=save_state_fn,
            root=app.root,
            worker_pool=app._worker_pool,
            post_to_ui=getattr(app, "post_to_ui", None),
        )
        
        # Store functions for async loading
        app._resources_fetch_caps_fn = _fetch_caps
        app._resources_detect_devices_fn = _detect_devices_info
        app._resources_cached_devices_fn = _load_cached_devices
        app._resources_device_refresh_submitted = False
        app._resources_caps_refresh_submitted = False
        logger.info("Resources panel created successfully")
            
    except Exception as e:
        logger.error(f"Failed to initialize resources panel: {e}", exc_info=True)
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
        def _register(category: LogCategory, category_name: str | None = None) -> None:
            target_name = category_name or category.value

            def _handler(message: str, level: str | None = None, _cat=target_name) -> None:
                if not hasattr(app, 'debug_panel') or not app.debug_panel:
                    return
                try:
                    app.post_to_ui(app.debug_panel.write, message, _cat, level)
                except Exception:
                    try:
                        app.debug_panel.write(message, _cat, level)
                    except Exception:
                        pass

            app._log_router.register_handler(category, _handler)

        for category in LogCategory:
            if category == LogCategory.DATASET:
                _register(category, "dataset")
            else:
                _register(category)
    except Exception as e:
        logger.error(f"Failed to register log handlers: {e}")
    
    # Bridge Python logging to Debug tab
    try:
        import threading

        class _TkDebugHandler(logging.Handler):
            def __init__(self, root: tk.Tk, log_router, err_cb, dispatcher):
                super().__init__()
                self._root = root
                self._log_router = log_router
                self._err = err_cb
                self._dispatcher = dispatcher
                self._ui_thread_id = threading.get_ident()
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
                
                try:
                    self._log_router.log(msg, category, level_name)
                except Exception:
                    return

                if record.levelno >= logging.ERROR:
                    exc_text = getattr(record, "exc_text", None)
                    self._dispatch_error(exc_text or msg)

            def _dispatch_error(self, message: str) -> None:
                if self._err is None:
                    return
                try:
                    if self._dispatcher is not None:
                        self._dispatcher.dispatch(self._err, message)
                        return
                except Exception:
                    pass

                try:
                    self._root.after(0, lambda: self._err(message))
                except Exception:
                    if threading.get_ident() == self._ui_thread_id:
                        try:
                            self._err(message)
                        except Exception:
                            pass
        
        app._debug_log_handler = _TkDebugHandler(app.root, app._log_router, app.debug_panel.set_error, getattr(app, '_ui_dispatcher', None))
        app._debug_log_handler.setLevel(logging.DEBUG)
        logging.getLogger("aios").addHandler(app._debug_log_handler)
        logging.getLogger().addHandler(app._debug_log_handler)
    except Exception as e:
        logger.error(f"Failed to configure Python logging handler: {e}")


def _apply_loaded_panel_data(app: Any) -> None:
    """Apply any panel data that finished loading in background threads."""
    start_total = time.perf_counter()

    def _log_duration(label: str, start_marker: float) -> None:
        """Emit timing logs for sections that risk blocking the UI."""
        elapsed_ms = (time.perf_counter() - start_marker) * 1000.0
        if elapsed_ms >= 220.0:
            logger.warning("UI apply '%s' took %.1f ms", label, elapsed_ms)
        elif elapsed_ms >= 140.0:
            logger.info("UI apply '%s' took %.1f ms", label, elapsed_ms)

    try:
        section_start = time.perf_counter()
        # Apply any queued brains panel updates that were scheduled before Tk mainloop started
        if hasattr(app, 'brains_panel') and hasattr(app.brains_panel, '_startup_callbacks'):
            callbacks = getattr(app.brains_panel, '_startup_callbacks') or []
            for cb in list(callbacks):
                try:
                    cb()
                except Exception as cb_err:
                    logger.warning(f"Failed to apply queued brains panel update: {cb_err}")
            try:
                delattr(app.brains_panel, '_startup_callbacks')
            except Exception:
                pass
        _log_duration("brains panel startup callbacks", section_start)

        section_start = time.perf_counter()
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
        _log_duration("chat panel apply", section_start)

        section_start = time.perf_counter()
        # Apply brains panel updates
        if hasattr(app, 'brains_panel') and hasattr(app.brains_panel, '_data_loaded'):
            total_experts = int(app.brains_panel.total_experts_var.get() or "0")
            total_brains = int(app.brains_panel.brain_count_var.get() or "0")
            if hasattr(app.brains_panel, 'status_var'):
                app.brains_panel.status_var.set(f"{total_brains} brains, {total_experts} experts")
            with suppress(AttributeError):
                delattr(app.brains_panel, '_data_loaded')
        _log_duration("brains panel apply", section_start)

        section_start = time.perf_counter()
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
            with suppress(AttributeError):
                delattr(app.mcp_panel, '_data_loaded')
        _log_duration("mcp panel apply", section_start)

        section_start = time.perf_counter()
        # Apply resources panel updates
        if hasattr(app, 'resources_panel') and hasattr(app.resources_panel, '_data_loaded'):
            # Apply caps if loaded
            caps_needs_refresh = False
            caps_data = getattr(app.resources_panel, '_pending_caps', None)
            caps_start = time.perf_counter()
            if isinstance(caps_data, dict):
                caps_copy = dict(caps_data)
                caps_needs_refresh = bool(caps_copy.pop('_needs_refresh', False))
                if caps_copy:
                    caps_signature = tuple(sorted(caps_copy.items()))
                    last_caps = getattr(app.resources_panel, '_last_caps_snapshot', None)
                    if caps_signature != last_caps:
                        app.resources_panel.set_caps(caps_copy)
                        try:
                            app.resources_panel._last_caps_snapshot = caps_signature
                        except Exception:
                            pass
                        logger.debug("Resource caps applied to UI")
                    else:
                        logger.debug("Resource caps unchanged; skipping UI update")
                elif caps_needs_refresh:
                    logger.info("Resource caps refresh deferred during startup; awaiting background refresh")
            elif caps_data is not None:
                logger.debug("Ignoring unexpected resource caps payload: %r", type(caps_data).__name__)
            _log_duration("resources panel apply[caps]", caps_start)

            # Apply device detection if loaded
            device_info = getattr(app.resources_panel, '_pending_devices', None)
            if device_info:
                detect_start = time.perf_counter()
                info_copy = dict(device_info)
                needs_refresh = bool(info_copy.pop('_needs_refresh', False))
                from_cache = bool(info_copy.pop('_from_cache', False))
                cache_age = info_copy.pop('_cache_age_seconds', None)
                info_copy.pop('_cache_timestamp', None)

                if info_copy:
                    set_detected_start = time.perf_counter()
                    app.resources_panel.set_detected(info_copy)
                    _log_duration("resources panel apply[set_detected]", set_detected_start)

                    gpu_count = len(info_copy.get('cuda_devices', []))
                    source_label = "cache" if from_cache else info_copy.get('source', 'fresh')
                    if from_cache and isinstance(cache_age, (int, float)):
                        logger.info(
                            "Device detection applied from %s (age %.1fs): %d GPU(s) found",
                            source_label,
                            cache_age,
                            gpu_count,
                        )
                    else:
                        logger.info(
                            "Device detection applied from %s: %d GPU(s) found",
                            source_label,
                            gpu_count,
                        )
                else:
                    if needs_refresh:
                        logger.info("Device detection deferred during startup; awaiting background refresh")

                if (from_cache or needs_refresh) and not getattr(app, '_resources_device_refresh_submitted', False):
                    app._resources_device_refresh_submitted = True

                    def _run_device_refresh() -> None:
                        try:
                            fresh = app._resources_detect_devices_fn(force_refresh=True)
                        except Exception:
                            logger.debug("Deferred device refresh failed", exc_info=True)
                            return

                        if not fresh:
                            return

                        fresh_copy = dict(fresh)
                        fresh_copy.pop('_from_cache', None)
                        fresh_copy.pop('_cache_age_seconds', None)
                        fresh_copy.pop('_cache_timestamp', None)

                        def _apply_refresh() -> None:
                            try:
                                if hasattr(app, 'resources_panel') and app.resources_panel:
                                    app.resources_panel.refresh_detected(fresh_copy)
                                    gpu_total = len(fresh_copy.get('cuda_devices', []))
                                    logger.info(
                                        "Device detection refreshed from torch: %d GPU(s) found",
                                        gpu_total,
                                    )
                            except Exception:
                                logger.debug("Failed to apply refreshed device info", exc_info=True)

                        app.post_to_ui(_apply_refresh)

                    try:
                        submit_background(
                            "panel-setup-device-refresh",
                            _run_device_refresh,
                            pool=getattr(app, "_worker_pool", None),
                        )
                    except RuntimeError as exc:
                        logger.error("Failed to queue device refresh task: %s", exc)
                        _run_device_refresh()

                _log_duration("resources panel apply[device_info block]", detect_start)

            if caps_needs_refresh and not getattr(app, '_resources_caps_refresh_submitted', False):
                app._resources_caps_refresh_submitted = True

                def _begin_caps_refresh() -> None:
                    try:
                        if hasattr(app, 'resources_panel') and app.resources_panel:
                            logger.info("Starting deferred resource caps refresh")
                            app.resources_panel.schedule_caps_refresh(delay_ms=1500, source="startup")
                    except Exception:
                        logger.debug("Deferred resource caps refresh failed", exc_info=True)

                try:
                    app.root.after(200, _begin_caps_refresh)
                except Exception:
                    _begin_caps_refresh()

            # Clean up
            if hasattr(app.resources_panel, '_pending_caps'):
                with suppress(AttributeError):
                    delattr(app.resources_panel, '_pending_caps')
            if hasattr(app.resources_panel, '_pending_devices'):
                with suppress(AttributeError):
                    delattr(app.resources_panel, '_pending_devices')
            with suppress(AttributeError):
                delattr(app.resources_panel, '_data_loaded')
        _log_duration("resources panel apply", section_start)

        section_start = time.perf_counter()
        # Apply settings panel config
        if hasattr(app, 'settings_panel') and hasattr(app.settings_panel, '_pending_config'):
            try:
                settings_config = app.settings_panel._pending_config
                if settings_config:
                    if 'cache_max_size_mb' in settings_config and hasattr(app.settings_panel, 'cache_size_var'):
                        app.settings_panel.cache_size_var.set(str(int(settings_config['cache_max_size_mb'])))
                        logger.info(f"Applied cache size from config: {settings_config['cache_max_size_mb']} MB")
            except Exception as e:
                logger.warning(f"Failed to apply settings from config: {e}", exc_info=True)
            finally:
                if hasattr(app.settings_panel, '_pending_config'):
                    delattr(app.settings_panel, '_pending_config')
        _log_duration("settings panel apply", section_start)

        section_start = time.perf_counter()
        # Apply evaluation panel updates
        if hasattr(app, 'evaluation_panel') and hasattr(app.evaluation_panel, '_data_loaded'):
            panel = app.evaluation_panel

            if hasattr(panel, '_pending_brains'):
                brains_start = time.perf_counter()
                brains = panel._pending_brains
                if hasattr(panel, 'brain_combo') and hasattr(panel, 'model_name_var'):
                    if brains:
                        panel.brain_combo["values"] = brains
                        current = panel.model_name_var.get()
                        if not current or current not in brains:
                            panel.model_name_var.set(brains[0])
                        logger.debug(f"Applied {len(brains)} brains to evaluation panel")
                    else:
                        panel.brain_combo["values"] = []
                _log_duration("evaluation panel apply[brains]", brains_start)
                delattr(panel, '_pending_brains')

            history = getattr(panel, '_pending_history', None)
            if history:
                history_start = time.perf_counter()
                panel._set_history(history)
                logger.debug("Evaluation history database initialized and set")
                _log_duration("evaluation panel apply[history]", history_start)

            if hasattr(panel, '_pending_tree_structure') and hasattr(panel, '_pending_tree_items'):
                tree_start = time.perf_counter()
                try:
                    tree_structure = list(panel._pending_tree_structure)
                    tree_items = dict(panel._pending_tree_items)
                    delattr(panel, '_pending_tree_structure')
                    delattr(panel, '_pending_tree_items')

                    tree = panel.bench_tree
                    items_cache = panel._tree_items
                    try:
                        delete_start = time.perf_counter()
                        tree.delete(*tree.get_children())
                        _log_duration("evaluation panel apply[tree delete]", delete_start)
                    except Exception:
                        pass
                    items_cache.clear()

                    id_mapping: dict[str, str] = {}
                    total_items = len(tree_structure)
                    batch_size = 18
                    delay_ms = 8

                    if total_items > 0 and logger.isEnabledFor(logging.INFO):
                        logger.info("Evaluation panel enqueuing %d tree items (batch %d, delay %dms)", total_items, batch_size, delay_ms)

                    def _insert_batch(start: int = 0) -> None:
                        end = min(start + batch_size, total_items)
                        for parent_id, item_id, text, values, item_info in tree_structure[start:end]:
                            actual_parent = id_mapping.get(parent_id, parent_id)
                            try:
                                actual_id = tree.insert(actual_parent, "end", text=text, values=values)
                            except Exception:
                                actual_id = tree.insert("", "end", text=text, values=values)
                            id_mapping[item_id] = actual_id
                            items_cache[actual_id] = item_info

                        if end < total_items:
                            try:
                                tree.after(delay_ms, lambda idx=end: _insert_batch(idx))
                            except Exception:
                                _insert_batch(end)
                        else:
                            panel._tree_populated = True
                            logger.debug(f"Populated benchmark tree with {total_items} items (batched)")

                    _insert_batch(0)
                except Exception as e:
                    logger.warning(f"Failed to populate benchmark tree: {e}")
                finally:
                    _log_duration("evaluation panel apply[tree structure]", tree_start)

            if hasattr(panel, '_pending_history'):
                with suppress(AttributeError):
                    delattr(panel, '_pending_history')
            with suppress(AttributeError):
                delattr(panel, '_data_loaded')
        _log_duration("evaluation panel apply", section_start)

        section_start = time.perf_counter()
        # Apply help panel updates
        if hasattr(app, 'help_panel') and hasattr(app.help_panel, '_data_loaded'):
            logger.debug("Help panel search index loaded and ready")
            with suppress(AttributeError):
                delattr(app.help_panel, '_data_loaded')
        _log_duration("help panel apply", section_start)

        section_start = time.perf_counter()
        # Apply HRM training panel updates
        if hasattr(app, 'hrm_training_panel') and hasattr(app.hrm_training_panel, '_data_loaded'):
            logger.debug("HRM training panel data loaded and ready")
            with suppress(AttributeError):
                delattr(app.hrm_training_panel, '_data_loaded')
        _log_duration("hrm training panel apply", section_start)
    except Exception as e:
        logger.warning(f"Error applying UI updates: {e}")
    finally:
        _log_duration("apply panel totals", start_total)


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
    # Import crash logging for early diagnostics
    try:
        from .app_main import _write_crash_log
    except ImportError:
        def _write_crash_log(msg, exc=None):
            pass
    
    _write_crash_log("load_all_panel_data: Starting parallel data load")
    logger.info("Loading all panel data with threading...")

    pool = getattr(app, "_worker_pool", None)
    if pool is None:
        _write_crash_log("load_all_panel_data: Worker pool unavailable!")
        raise RuntimeError("Worker pool unavailable during panel data loading")

    pool_workers = getattr(pool, "max_workers", "unknown")
    _write_crash_log(f"load_all_panel_data: Worker pool has {pool_workers} workers")
    logger.info("Submitting panel data loads to shared worker pool (%s workers)", pool_workers)
    
    # Track completion status
    results = {
        'chat': {'done': False, 'error': None},
        'brains': {'done': False, 'error': None},
        'mcp': {'done': False, 'error': None},
        'resources': {'done': False, 'error': None},
        'settings': {'done': False, 'error': None},
        'evaluation': {'done': False, 'error': None},
        'help': {'done': False, 'error': None},
        'hrm_training': {'done': False, 'error': None},
    }
    
    # Thread functions that update results - each has crash logging for diagnostics
    def load_chat():
        try:
            _write_crash_log("load_chat: Starting")
            import time
            start = time.time()
            _load_chat_brains_sync(app)
            duration = time.time() - start
            logger.info(f"[DATA LOAD] Chat brains: {duration:.3f}s")
            results['chat']['done'] = True
            _write_crash_log(f"load_chat: Completed in {duration:.3f}s")
        except Exception as e:
            results['chat']['error'] = str(e)
            logger.error(f"Error loading chat data: {e}")
            _write_crash_log(f"load_chat: FAILED", e)
        return 'chat'
    
    def load_brains():
        try:
            _write_crash_log("load_brains: Starting")
            import time
            start = time.time()
            _load_brains_panel_sync(app)
            duration = time.time() - start
            logger.info(f"[DATA LOAD] Brains panel: {duration:.3f}s")
            results['brains']['done'] = True
            _write_crash_log(f"load_brains: Completed in {duration:.3f}s")
        except Exception as e:
            results['brains']['error'] = str(e)
            logger.error(f"Error loading brains data: {e}")
            _write_crash_log(f"load_brains: FAILED", e)
        return 'brains'
    
    def load_mcp():
        try:
            _write_crash_log("load_mcp: Starting")
            import time
            start = time.time()
            _load_mcp_panel_sync(app)
            duration = time.time() - start
            logger.info(f"[DATA LOAD] MCP panel: {duration:.3f}s")
            results['mcp']['done'] = True
            _write_crash_log(f"load_mcp: Completed in {duration:.3f}s")
        except Exception as e:
            results['mcp']['error'] = str(e)
            logger.error(f"Error loading MCP data: {e}")
            _write_crash_log(f"load_mcp: FAILED", e)
        return 'mcp'
    
    def load_resources():
        try:
            _write_crash_log("load_resources: Starting")
            import time
            start = time.time()
            _load_resources_panel_sync(app)
            duration = time.time() - start
            logger.info(f"[DATA LOAD] Resources panel: {duration:.3f}s")
            results['resources']['done'] = True
            _write_crash_log(f"load_resources: Completed in {duration:.3f}s")
        except Exception as e:
            results['resources']['error'] = str(e)
            logger.error(f"Error loading resources data: {e}")
            _write_crash_log(f"load_resources: FAILED", e)
        return 'resources'
    
    def load_settings():
        try:
            _write_crash_log("load_settings: Starting")
            import time
            start = time.time()
            _load_settings_panel_sync(app)
            duration = time.time() - start
            logger.info(f"[DATA LOAD] Settings panel: {duration:.3f}s")
            results['settings']['done'] = True
            _write_crash_log(f"load_settings: Completed in {duration:.3f}s")
        except Exception as e:
            results['settings']['error'] = str(e)
            logger.error(f"Error loading settings data: {e}")
            _write_crash_log(f"load_settings: FAILED", e)
        return 'settings'
    
    def load_evaluation():
        try:
            _write_crash_log("load_evaluation: Starting")
            import time
            start = time.time()
            _load_evaluation_panel_sync(app)
            duration = time.time() - start
            logger.info(f"[DATA LOAD] Evaluation panel: {duration:.3f}s")
            results['evaluation']['done'] = True
            _write_crash_log(f"load_evaluation: Completed in {duration:.3f}s")
        except Exception as e:
            results['evaluation']['error'] = str(e)
            logger.error(f"Error loading evaluation data: {e}")
            _write_crash_log(f"load_evaluation: FAILED", e)
        return 'evaluation'
    
    def load_help():
        try:
            _write_crash_log("load_help: Starting")
            import time
            start = time.time()
            _load_help_panel_sync(app)
            duration = time.time() - start
            logger.info(f"[DATA LOAD] Help panel: {duration:.3f}s")
            results['help']['done'] = True
            _write_crash_log(f"load_help: Completed in {duration:.3f}s")
        except Exception as e:
            results['help']['error'] = str(e)
            logger.error(f"Error loading help data: {e}")
            _write_crash_log(f"load_help: FAILED", e)
        return 'help'
    
    def load_hrm_training():
        try:
            _write_crash_log("load_hrm_training: Starting")
            import time
            start = time.time()
            _load_hrm_training_panel_sync(app)
            duration = time.time() - start
            logger.info(f"[DATA LOAD] HRM training panel: {duration:.3f}s")
            results['hrm_training']['done'] = True
            _write_crash_log(f"load_hrm_training: Completed in {duration:.3f}s")
        except Exception as e:
            results['hrm_training']['error'] = str(e)
            logger.error(f"Error loading HRM training data: {e}")
            _write_crash_log(f"load_hrm_training: FAILED", e)
        return 'hrm_training'
    
    # Start loading with available executor
    _write_crash_log("load_all_panel_data: Submitting tasks to worker pool")
    update_status_fn("Loading data in parallel...")
    app.root.update_idletasks()
    futures = {
        pool.submit(load_chat): 'chat',
        pool.submit(load_brains): 'brains',
        pool.submit(load_mcp): 'mcp',
        pool.submit(load_resources): 'resources',
        pool.submit(load_settings): 'settings',
        pool.submit(load_evaluation): 'evaluation',
        pool.submit(load_help): 'help',
        pool.submit(load_hrm_training): 'hrm_training',
    }

    import threading

    remaining = set(results.keys())
    results_lock = threading.Lock()
    completion_logged = False

    def _compose_status_message() -> str | None:
        labels = {
            'chat': "chat models",
            'brains': "brain registry",
            'mcp': "MCP servers",
            'resources': "GPU devices",
            'settings': "settings",
            'evaluation': "benchmarks",
            'help': "documentation",
            'hrm_training': "training data",
        }
        with results_lock:
            pending_keys = [key for key, info in results.items() if not info['done']]
        if not pending_keys:
            return None
        display_items = [labels.get(key, key) for key in pending_keys[:3]]
        if len(pending_keys) > 3:
            return f"Loading {', '.join(display_items)} +{len(pending_keys) - 3} more..."
        return f"Loading {', '.join(display_items)}..."

    def _handle_completion(task_name: str):
        def _callback(future):
            nonlocal completion_logged
            try:
                future.result()
            except Exception as exc:
                logger.exception(f"Task '{task_name}' failed", exc_info=True)
                with results_lock:
                    if not results[task_name]['error']:
                        results[task_name]['error'] = str(exc)

            with results_lock:
                results[task_name]['done'] = True
                remaining.discard(task_name)
                all_done = not remaining

            logger.debug(f"Task '{task_name}' completed (all_done={all_done})")

            def _on_ui_complete() -> None:
                nonlocal completion_logged
                _apply_loaded_panel_data(app)

                if all_done and not completion_logged:
                    completion_logged = True
                    update_status_fn("Data loading complete!")
                    try:
                        app.root.after_idle(app.root.update_idletasks)
                    except Exception:
                        try:
                            app.root.update_idletasks()
                        except Exception:
                            pass
                    logger.info("All panel data loaded successfully")
                    _write_crash_log("load_all_panel_data: All tasks completed successfully")
                elif not all_done:
                    message = _compose_status_message()
                    if message:
                        update_status_fn(message)

            app.post_to_ui(_on_ui_complete)

        return _callback

    for future, task_name in futures.items():
        future.add_done_callback(_handle_completion(task_name))

    _write_crash_log("load_all_panel_data: All tasks submitted, waiting for completion callbacks")
    message = _compose_status_message()
    if message:
        update_status_fn(message)
