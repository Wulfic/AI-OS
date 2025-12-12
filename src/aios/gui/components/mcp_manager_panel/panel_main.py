"""Main MCP Servers & Tools Manager Panel component.

Provides a visual interface for managing MCP servers and tool permissions.
"""

from __future__ import annotations

# Import safe variable wrappers
from ...utils import safe_variables

import logging
import os
import threading
from typing import Any, Callable, Optional, cast

logger = logging.getLogger(__name__)

try:  # pragma: no cover - environment dependent
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    tk = cast(Any, None)
    ttk = cast(Any, None)

from .data_manager import (
    get_default_servers,
    get_default_tools,
    load_servers_config,
    load_tools_config,
    save_servers_config,
    save_tools_config,
)
from .server_dialog import MCPServerDialog
from .server_handlers import (
    handle_add_server,
    handle_delete_server,
    handle_disable_server,
    handle_edit_server,
    handle_enable_server,
    handle_test_server,
)
from .tool_handlers import (
    handle_disable_all_tools,
    handle_enable_all_tools,
    handle_reset_tools,
    handle_tool_toggle,
)
from .ui_builders import (
    create_header,
    create_notebook,
    create_servers_section,
    create_tools_section,
)
from .ui_updaters import (
    get_active_servers,
    get_enabled_tools,
    populate_servers_tree,
    populate_tools_tree,
    update_summary,
)


class MCPManagerPanel(ttk.LabelFrame):  # type: ignore[misc]
    """Panel for managing MCP servers and tool permissions."""

    def __init__(
        self,
        parent: Any,
        *,
        run_cli: Optional[Callable[[list[str]], str]] = None,
        append_out: Optional[Callable[[str], None]] = None,
        save_state_fn: Optional[Callable[[], None]] = None,
        post_to_ui: Optional[Callable[[Callable[..., None]], None]] = None,
    ) -> None:
        """Initialize the MCP manager panel.
        
        Args:
            parent: Parent Tkinter widget
            run_cli: Optional callback to run CLI commands
            append_out: Optional callback for debug output
            save_state_fn: Optional callback to save state
        """
        super().__init__(parent, text="MCP Servers & Tools Manager")
        if tk is None:
            raise RuntimeError("Tkinter not available")
        
        self.pack(fill="both", expand=True, padx=8, pady=8)

        self._run_cli = run_cli or (lambda args: "")
        self._append_out = append_out or (lambda s: None)
        self._save_state_fn = save_state_fn or (lambda: None)
        self._post_to_ui = post_to_ui
        
        # Detect project root
        self._project_root = self._detect_project_root()
        self._config_dir = os.path.join(self._project_root, "config")
        self._servers_config_path = os.path.join(self._config_dir, "mcp_servers.json")
        self._tools_config_path = os.path.join(self._config_dir, "tool_permissions.json")
        
        # Ensure config directory exists
        os.makedirs(self._config_dir, exist_ok=True)
        
        # Create state variables
        self.servers_count_var = safe_variables.StringVar(value="0")
        self.servers_active_var = safe_variables.StringVar(value="0")
        self.tools_enabled_var = safe_variables.StringVar(value="0")
        self.tool_category_var = safe_variables.StringVar(value="All")
        
        # Worker pool for async operations (to prevent blocking GUI)
        try:
            from ...utils.resource_management import get_worker_pool
            self._worker_pool = get_worker_pool()
        except Exception:
            self._worker_pool = None
            logger.warning("Worker pool not available, operations will be synchronous")
        
        # Refresh throttling
        self._last_refresh_time = 0.0
        self._refresh_in_progress = False
        
        # Create UI sections
        create_header(
            self,
            self.servers_count_var,
            self.servers_active_var,
            self.tools_enabled_var,
            self.refresh,
        )
        
        def create_servers_tab(parent):
            self.servers_tree = create_servers_section(
                parent,
                self._on_add_server,
                self._on_edit_server,
                self._on_delete_server,
                self._on_test_server,
                self._on_enable_server,
                self._on_disable_server,
                self._on_server_select,
            )
        
        def create_tools_tab(parent):
            self.tools_tree = create_tools_section(
                parent,
                self.tool_category_var,
                lambda e: self._filter_tools(),
                self._on_enable_all_tools,
                self._on_disable_all_tools,
                self._on_reset_tools,
                self._on_tool_toggle,
            )
        
        self.notebook, self.servers_tab, self.tools_tab = create_notebook(
            self,
            create_servers_tab,
            create_tools_tab,
        )
        
        # Initial load
        self.refresh()

    def _dispatch_to_ui(self, callback: Callable[[], None]) -> bool:
        """Schedule ``callback`` to execute on the Tk UI thread."""
        if self._post_to_ui is not None:
            try:
                self._post_to_ui(callback)
                return True
            except Exception:
                logger.debug("Failed to dispatch via post_to_ui", exc_info=True)

        if threading.current_thread() is threading.main_thread():
            try:
                self.after(0, callback)
                return True
            except Exception:
                logger.debug("Failed to schedule callback with after", exc_info=True)

        return False

    def _current_tool_category(self) -> str:
        """Return the currently selected tool category from the UI thread."""
        if threading.current_thread() is threading.main_thread():
            try:
                return self.tool_category_var.get()
            except Exception:
                logger.debug("Failed to read tool category from Tk variable", exc_info=True)
        else:
            logger.debug("Tool category read requested off main thread; defaulting to 'All'")
        return "All"
    
    @staticmethod
    def _detect_project_root() -> str:
        """Detect the project root directory.
        
        First tries from source file location, then falls back to CWD.
        """
        try:
            # First, try to find from source file location (more reliable)
            source_dir = os.path.dirname(os.path.abspath(__file__))
            cur = source_dir
            for _ in range(10):  # Traverse up to find project root
                if os.path.exists(os.path.join(cur, "pyproject.toml")):
                    return cur
                parent_dir = os.path.dirname(cur)
                if parent_dir == cur:
                    break
                cur = parent_dir
            
            # Fallback: try from CWD
            cur = os.path.abspath(os.getcwd())
            for _ in range(8):
                if os.path.exists(os.path.join(cur, "pyproject.toml")):
                    return cur
                parent_dir = os.path.dirname(cur)
                if parent_dir == cur:
                    break
                cur = parent_dir
            return os.path.abspath(os.getcwd())
        except Exception:
            return os.path.abspath(os.getcwd())

    # ========== Data Management ==========
    
    def _load_servers_config(self):
        """Load MCP servers configuration."""
        return load_servers_config(
            self._servers_config_path,
            lambda: get_default_servers(self._project_root),
            self._append_out,
        )

    def _save_servers_config(self, servers):
        """Save MCP servers configuration."""
        save_servers_config(
            self._servers_config_path,
            servers,
            self._save_state_fn,
            self._append_out,
        )

    def _load_tools_config(self):
        """Load tool permissions configuration."""
        return load_tools_config(
            self._tools_config_path,
            get_default_tools,
            self._append_out,
        )

    def _save_tools_config(self, tools):
        """Save tool permissions configuration."""
        save_tools_config(
            self._tools_config_path,
            tools,
            self._save_state_fn,
            self._append_out,
        )

    # ========== Server Event Handlers ==========
    
    def _on_add_server(self):
        """Handle adding a new MCP server."""
        handle_add_server(
            self,
            MCPServerDialog,
            self._load_servers_config,
            self._save_servers_config,
            self.refresh,
            self._append_out,
        )

    def _on_edit_server(self):
        """Handle editing selected server."""
        handle_edit_server(
            self,
            self.servers_tree,
            MCPServerDialog,
            self._load_servers_config,
            self._save_servers_config,
            self.refresh,
            self._append_out,
        )

    def _on_delete_server(self):
        """Handle deleting selected server."""
        handle_delete_server(
            self.servers_tree,
            self._load_servers_config,
            self._save_servers_config,
            self.refresh,
            self._append_out,
        )

    def _on_test_server(self):
        """Handle testing server connection."""
        handle_test_server(self.servers_tree, self._append_out)

    def _on_enable_server(self):
        """Enable selected server."""
        handle_enable_server(
            self.servers_tree,
            self._load_servers_config,
            self._save_servers_config,
            self.refresh,
            self._append_out,
        )

    def _on_disable_server(self):
        """Disable selected server."""
        handle_disable_server(
            self.servers_tree,
            self._load_servers_config,
            self._save_servers_config,
            self.refresh,
            self._append_out,
        )

    def _on_server_select(self, event=None):
        """Handle server selection."""
        # Could show server details in a side panel in future
        pass

    # ========== Tool Event Handlers ==========
    
    def _on_tool_toggle(self, event=None):
        """Toggle enabled status of selected tool."""
        handle_tool_toggle(
            self.tools_tree,
            self._load_tools_config,
            self._save_tools_config,
            self.refresh,
            self._append_out,
        )

    def _on_enable_all_tools(self):
        """Enable all tools in current filter."""
        handle_enable_all_tools(
            self.tool_category_var.get(),
            self._load_tools_config,
            self._save_tools_config,
            self.refresh,
            self._append_out,
        )

    def _on_disable_all_tools(self):
        """Disable all tools in current filter."""
        handle_disable_all_tools(
            self.tool_category_var.get(),
            self._load_tools_config,
            self._save_tools_config,
            self.refresh,
            self._append_out,
        )

    def _on_reset_tools(self):
        """Reset tool permissions to defaults."""
        handle_reset_tools(
            get_default_tools,
            self._save_tools_config,
            self.refresh,
            self._append_out,
        )

    def _filter_tools(self):
        """Filter tools tree by selected category."""
        self._populate_tools_tree()

    # ========== UI Population ==========
    
    def refresh(self):
        """Refresh all data from disk (async to prevent blocking GUI).

        This method runs in a background thread to avoid blocking the GUI
        when switching tabs or manually refreshing.
        """
        category_filter = self._current_tool_category()
        # Throttle refreshes - only refresh if >2 seconds since last refresh
        import time
        current_time = time.time()
        if current_time - self._last_refresh_time < 2.0:
            # Too soon - skip refresh
            logger.debug("MCP refresh throttled (< 2s since last refresh)")
            return
        
        if self._refresh_in_progress:
            # Already refreshing, skip
            logger.debug("MCP refresh already in progress, skipping")
            return
        
        self._last_refresh_time = current_time
        self._refresh_in_progress = True
        
        logger.info("Refreshing MCP servers and tools from configuration")
        
        def _do_refresh(selected_category: str):
            """Background refresh operation."""
            try:
                # Load data in background thread (file I/O can be slow)
                servers = self._load_servers_config()
                tools = self._load_tools_config()
                
                # Schedule UI update on main thread
                def _update_ui():
                    try:
                        logger.debug(f"Updating MCP UI with {len(servers)} servers and {len(tools)} tools")
                        populate_servers_tree(self.servers_tree, servers)
                        populate_tools_tree(self.tools_tree, tools, selected_category)
                        update_summary(
                            servers,
                            tools,
                            self.servers_count_var,
                            self.servers_active_var,
                            self.tools_enabled_var,
                        )
                        logger.debug("MCP manager refresh complete")
                    except Exception as e:
                        logger.error(f"Error updating MCP UI: {e}", exc_info=True)
                    finally:
                        self._refresh_in_progress = False
                
                if not self._dispatch_to_ui(_update_ui):
                    self._refresh_in_progress = False
            except Exception as exc:
                logger.error(f"Error refreshing MCP data: {exc}", exc_info=True)
                error_message = str(exc)

                def _handle_error():
                    self._append_out(f"[MCP] Refresh failed: {error_message}")
                    self._refresh_in_progress = False
                
                if not self._dispatch_to_ui(_handle_error):
                    self._refresh_in_progress = False
        
        # Submit to worker pool if available
        if self._worker_pool:
            self._worker_pool.submit(_do_refresh, category_filter)
        else:
            # Fallback to direct execution (will block, but better than crashing)
            logger.warning("Worker pool not available, MCP refresh will block GUI")
            _do_refresh(category_filter)

    def _populate_servers_tree(self):
        """Populate MCP servers tree view."""
        servers = self._load_servers_config()
        logger.debug(f"Populating servers tree with {len(servers)} configured servers")
        populate_servers_tree(self.servers_tree, servers)

    def _populate_tools_tree(self):
        """Populate tools tree view with category grouping."""
        tools = self._load_tools_config()
        category_filter = self.tool_category_var.get()
        logger.debug(f"Populating tools tree with {len(tools)} tools (filter: {category_filter})")
        populate_tools_tree(self.tools_tree, tools, category_filter)

    def _update_summary(self):
        """Update header summary statistics."""
        servers = self._load_servers_config()
        tools = self._load_tools_config()
        update_summary(
            servers,
            tools,
            self.servers_count_var,
            self.servers_active_var,
            self.tools_enabled_var,
        )

    def get_enabled_tools(self):
        """Get list of enabled tool names."""
        tools = self._load_tools_config()
        return get_enabled_tools(tools)

    def get_active_servers(self):
        """Get list of active MCP servers."""
        servers = self._load_servers_config()
        return get_active_servers(servers)
