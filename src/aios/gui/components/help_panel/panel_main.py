"""HelpPanel - In-app documentation browser with fuzzy search and robust links.

Indexes markdown under docs/ and provides a left-hand TOC/search and a right
HTML preview. Supports in-app navigation for .md/.mdx links and opens external
http/https links in the system browser.

This module has been refactored into smaller, modular components:
- search_engine.py: Document indexing and search
- link_handler.py: Link interception and navigation
- markdown_renderer.py: Markdown to HTML conversion
- toc_builder.py: Table of contents tree building
- utils.py: Helper functions
"""

import logging
import os
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING, Callable, cast

try:
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    tk = cast(Any, None)
    ttk = cast(Any, None)

# Optional rich rendering support
try:  # pragma: no cover - optional dependency
    from tkinterweb import HtmlFrame  # type: ignore
except Exception:  # pragma: no cover
    HtmlFrame = None  # type: ignore

# Import safe Variable wrappers
from ...utils import safe_variables

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...utils.resource_management.async_pool import AsyncWorkerPool

# Import our modular components
from . import utils
from .search_engine import SearchEngine
from .link_handler import LinkHandler
from .markdown_renderer import MarkdownRenderer
from .toc_builder import TOCBuilder

logger = logging.getLogger(__name__)


class HelpPanel(ttk.LabelFrame):  # type: ignore[misc]
    """Main Help Panel widget with search, TOC, and HTML preview."""
    
    def __init__(
        self,
        parent: Any,
        project_root: Optional[Path] = None,
        worker_pool: Optional["AsyncWorkerPool"] = None,
    ) -> None:
        """Initialize the Help Panel.
        
        Args:
            parent: Parent Tkinter widget
            project_root: Optional project root path (auto-detected if not provided)
        """
        logger.info("[HelpPanel] __init__ called - starting initialization")
        super().__init__(parent, text="Help & Docs")
        if tk is None:
            raise RuntimeError("Tkinter not available")
        self.pack(fill="both", expand=True, padx=8, pady=8)

        # Resolve project/docs roots
        self._project_root = utils.find_project_root(project_root or Path(__file__))
        self._docs_root = utils.resolve_docs_root(self._project_root)
        logger.info(f"[HelpPanel] Docs root: {self._docs_root}")

        # Initialize components
        self._search_engine = SearchEngine(self._docs_root)
        self._md_renderer = MarkdownRenderer()
        self._worker_pool = worker_pool
        self._render_request_id = 0
        self._pending_render: Optional[dict[str, Any]] = None
        
        # Initialize references to other panels (set later by app)
        self._settings_panel = None
        
        # Build UI
        self._build_ui()
        
        # Initialize state
        self._mode = "toc"  # 'toc' or 'search'
        self._nav_history: list[tuple[str, int]] = []
        self._nav_index = -1
        self._nav_in_progress = False
        self._current_doc_rel = ""
        self._debounce_after_id: Optional[str] = None
        self._loading_doc = False
        self._current_theme = "light"  # Default theme
        self._html_frame_initialized = False  # Track if HtmlFrame has been created
        
        # Initialize link handler (will be set after HtmlFrame is created)
        self._link_handler = None
        
        # Initialize TOC builder
        self._toc_builder = TOCBuilder(self.results)
        self._pending_scroll: Optional[tuple[str, Optional[str]]] = None
        self._index_future = None
        self._pending_index_payload: Optional[dict[str, Any]] = None
        
        self._update_nav_buttons()
        
        logger.info("[HelpPanel] __init__ complete - will initialize HTML frame and index after main loop starts")

    def start_deferred_initialization(self) -> None:
        """Start deferred initialization after main loop is running.
        
        This should be called after root.mainloop() has started to avoid
        "main thread is not in main loop" errors.
        """
        logger.info("[HelpPanel] Starting deferred initialization (HTML frame + index)")
        
        # Initialize HtmlFrame first (needs to be ready before index results are displayed)
        logger.info("[HelpPanel] Scheduling HtmlFrame initialization (100ms delay)...")
        self.after(100, self._initialize_html_frame_safe)
        
        # Start index building on main thread (not background thread to avoid deadlock)
        logger.info("[HelpPanel] Scheduling index build (1000ms delay)...")
        self.after(1000, self._schedule_index_build)

    def _build_ui(self) -> None:
        """Build the user interface components."""
        # Top bar with search and controls
        self._build_top_bar()
        
        # Paned window with TOC/search results and HTML preview
        self._build_paned_view()
        
        # Layout helpers
        self.after(200, self._set_initial_sash)
        self._sash_set_once = False
        try:
            self._pane.bind("<Configure>", self._on_pane_configure)
            self.bind("<Map>", lambda e: self._ensure_left_visible())
            self._pane.bind("<Map>", lambda e: self._ensure_left_visible())
        except Exception:
            pass

    def _build_top_bar(self) -> None:
        """Build the top control bar with search and navigation."""
        top = ttk.Frame(self)
        top.pack(fill="x")
        
        # Search box
        ttk.Label(top, text="Search docs:").pack(side="left")
        self.search_var = safe_variables.StringVar()
        self.search_entry = ttk.Entry(top, textvariable=self.search_var)
        self.search_entry.pack(side="left", fill="x", expand=True, padx=6, pady=6)
        self.search_entry.bind("<Return>", self._on_search)
        self.search_entry.bind("<KeyRelease>", self._on_key_release)

        # Navigation buttons
        self.back_btn = ttk.Button(top, text="← Back", command=self._nav_back)
        self.back_btn.pack(side="left", padx=(6, 0))
        self.forward_btn = ttk.Button(top, text="Forward →", command=self._nav_forward)
        self.forward_btn.pack(side="left", padx=(6, 0))
        self.open_btn = ttk.Button(top, text="Open Externally", command=self._on_open_external)
        self.open_btn.pack(side="left", padx=(6, 0))
        self.clear_btn = ttk.Button(top, text="Clear", command=self._on_clear)
        self.clear_btn.pack(side="left", padx=(6, 0))

        # Status label
        try:
            self._status_var = safe_variables.StringVar(value="")
            ttk.Label(top, textvariable=self._status_var).pack(side="left", padx=(12, 0))
        except Exception:
            self._status_var = None

    def _build_paned_view(self) -> None:
        """Build the paned view with TOC/results and HTML preview."""
        self._pane = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        self._pane.pack(fill="both", expand=True)
        
        # Left pane: TOC/search results
        left = ttk.Frame(self._pane)
        self._pane.add(left, weight=1)
        self._left_frame = left
        
        # Right pane: HTML preview
        right = ttk.Frame(self._pane)
        self._pane.add(right, weight=3)
        self._right_frame = right
        
        # Configure minimum sizes
        try:
            self._pane.paneconfigure(self._left_frame, minsize=220)
            self._pane.paneconfigure(self._right_frame, minsize=300)
        except Exception:
            pass

        # Build tree view for TOC/results
        self._build_tree_view(left)
        
        # Build HTML preview
        self._build_html_preview(right)

    def _build_tree_view(self, parent: Any) -> None:
        """Build the tree view for TOC and search results.
        
        Args:
            parent: Parent widget
        """
        # Treeview with hidden metadata columns
        self.results = ttk.Treeview(
            parent,
            columns=("path", "line"),
            show="tree",
            displaycolumns=(),  # Hide metadata columns
        )
        self.results.pack(fill="both", expand=True)
        
        # Event bindings
        self.results.bind("<<TreeviewSelect>>", self._on_tree_single_click)
        self.results.bind("<Double-1>", self._on_select)
        self.results.bind("<Return>", self._on_select)
        
        # Context menu
        try:
            self._results_menu = tk.Menu(self.results, tearoff=0)
            self._results_menu.add_command(label="Open", command=lambda: self._on_select(None))
            self._results_menu.add_command(label="Open Externally", command=self._on_open_external)
            
            def _popup_menu(event: Any) -> None:
                try:
                    iid = self.results.identify_row(event.y)
                    if iid:
                        self.results.selection_set(iid)
                    self._results_menu.tk_popup(event.x_root, event.y_root)
                finally:
                    try:
                        self._results_menu.grab_release()
                    except Exception:
                        pass
            
            self.results.bind("<Button-3>", _popup_menu)
        except Exception:
            pass

    def _build_html_preview(self, parent: Any) -> None:
        """Build the HTML preview pane.
        
        Args:
            parent: Parent widget
        """
        self.html_frame = ttk.Frame(parent)
        self.html_frame.pack(fill="both", expand=True)
        # HtmlFrame will be created later in _initialize_html_frame() to avoid threading issues
        self.html_view = None
        
        # Show placeholder message
        self._placeholder_label = ttk.Label(
            self.html_frame,
            text="Loading documentation viewer..."
        )
        self._placeholder_label.pack(fill="both", expand=True, padx=12, pady=12)

    def _initialize_html_frame_safe(self) -> None:
        """Safely initialize HtmlFrame with progressive timeout checks.
        
        This ensures the UI remains responsive even if HtmlFrame creation is slow.
        """
        logger.info("[HelpPanel] _initialize_html_frame_safe() - starting")
        
        # Track initialization state
        self._html_init_started = True
        self._html_init_complete = False
        
        # Break up initialization into steps to allow UI responsiveness
        def step1_remove_placeholder():
            """Step 1: Remove placeholder."""
            logger.info("[HelpPanel] Step 1: Removing placeholder")
            try:
                if hasattr(self, '_placeholder_label') and self._placeholder_label:
                    self._placeholder_label.destroy()
            except Exception as e:
                logger.error(f"[HelpPanel] Error removing placeholder: {e}")
            
            # Schedule next step
            self.after(10, step2_check_htmlframe_available)
        
        def step2_check_htmlframe_available():
            """Step 2: Check if HtmlFrame is available."""
            logger.info("[HelpPanel] Step 2: Checking HtmlFrame availability")
            
            if HtmlFrame is None:
                logger.warning("[HelpPanel] HtmlFrame not available (tkinterweb not installed)")
                self._show_html_fallback()
                return
            
            # Schedule next step
            self.after(10, step3_create_htmlframe)
        
        def step3_create_htmlframe():
            """Step 3: Create HtmlFrame widget."""
            logger.info("[HelpPanel] Step 3: Creating HtmlFrame widget...")
            
            try:
                # Monkey-patch tkinterweb to suppress threading errors
                try:
                    import tkinterweb.bindings as twb
                    if hasattr(twb, 'TkinterWeb'):
                        original_post_event = twb.TkinterWeb.post_event
                        
                        def safe_post_event(self, event):
                            """Wrap post_event to catch RuntimeError from background threads."""
                            try:
                                return original_post_event(self, event)
                            except RuntimeError as e:
                                if "main thread is not in main loop" in str(e):
                                    pass  # Silently ignore during startup
                                else:
                                    raise
                        
                        twb.TkinterWeb.post_event = safe_post_event
                        logger.info("[HelpPanel] Applied tkinterweb monkey-patch")
                except Exception as e:
                    logger.warning(f"[HelpPanel] Could not apply monkey-patch: {e}")
                
                # Create HtmlFrame (must be on main thread)
                logger.info("[HelpPanel] Creating HtmlFrame instance...")
                try:
                    self.html_view = HtmlFrame(
                        self.html_frame,
                        horizontal_scrollbar="auto",
                        messages_enabled=False,
                        on_link_click=self._handle_link_click
                    )
                except Exception:
                    try:
                        self.html_view = HtmlFrame(
                            self.html_frame,
                            on_link_click=self._handle_link_click
                        )
                    except Exception:
                        self.html_view = HtmlFrame(self.html_frame)
                
                logger.info("[HelpPanel] HtmlFrame instance created, packing...")
                self.html_view.pack(fill="both", expand=True)
                
                # Force UI update to ensure widget is rendered
                self.update_idletasks()
                logger.info("[HelpPanel] HtmlFrame packed and UI updated")
                
                # Schedule finalization
                self.after(10, step4_finalize)
                
            except Exception as e:
                logger.error(f"[HelpPanel] Failed to create HtmlFrame: {e}", exc_info=True)
                self._show_html_fallback()
        
        def step4_finalize():
            """Step 4: Finalize initialization."""
            logger.info("[HelpPanel] Step 4: Finalizing HtmlFrame initialization")
            
            try:
                # Protect mousewheel events
                self._protect_mousewheel_events()
                
                # Initialize link handler
                if self.html_view is not None:
                    self._link_handler = LinkHandler(
                        self.html_view,
                        self._docs_root,
                        self._search_engine.index if self._search_engine else [],
                        self._open_doc
                    )
                    logger.info("[HelpPanel] Link handler initialized")
                
                self._html_frame_initialized = True
                self._html_init_complete = True
                logger.info("[HelpPanel] HtmlFrame initialization COMPLETE")

                pending_payload = getattr(self, "_pending_index_payload", None)
                if pending_payload:
                    logger.info("[HelpPanel] Applying deferred index payload after HtmlFrame init")
                    self._apply_index_payload(pending_payload)
                
            except Exception as e:
                logger.error(f"[HelpPanel] Error during finalization: {e}", exc_info=True)
        
        # Start the initialization sequence
        self.after(10, step1_remove_placeholder)

    def _show_html_fallback(self) -> None:
        """Show fallback message if HtmlFrame fails to load."""
        try:
            logger.info("[HelpPanel] Showing HTML fallback message")
            for w in self.html_frame.winfo_children():
                try:
                    w.destroy()
                except Exception:
                    pass
            ttk.Label(
                self.html_frame,
                text="HTML viewer could not be initialized.\nDocumentation is available in the tree view on the left.\n\nInstall tkinterweb for rich HTML rendering:\npip install tkinterweb",
                justify="center"
            ).pack(fill="both", expand=True, padx=12, pady=12)
            self._html_frame_initialized = True  # Mark as "done" even if failed
        except Exception as e:
            logger.error(f"[HelpPanel] Could not show fallback message: {e}")

    def _initialize_html_frame(self) -> None:
        """Initialize the HtmlFrame widget after mainloop has started.
        
        DEPRECATED: Use _initialize_html_frame_safe() instead.
        This method is kept for backward compatibility but should not be called directly.
        """
        logger.warning("[HelpPanel] _initialize_html_frame() called directly - use _initialize_html_frame_safe() instead")
        self._initialize_html_frame_safe()


    # ========== Indexing ==========
    
    def _build_index_main_thread(self) -> None:
        """Build the search index on the main thread (no background thread).
        
        This avoids deadlock issues that can occur with background threads
        and GUI interactions.
        """
        logger.info("[HelpPanel] _build_index_main_thread() called - building on main thread")
        try:
            logger.info("[HelpPanel] Calling build_index()...")
            success = self._search_engine.build_index()
            logger.info("[HelpPanel] build_index() returned")
            stats = self._search_engine.get_index_stats() if success else ""
            payload = {
                "success": success,
                "stats": stats,
            }
            if not success:
                payload["error"] = "Failed to index documentation"
            self._on_index_build_complete(payload)
        except Exception as e:
            logger.error(f"[HelpPanel] Exception during index build: {e}", exc_info=True)
            self._on_index_build_complete({"success": False, "error": str(e)})

    def _schedule_index_build(self) -> None:
        """Choose the appropriate strategy for building the search index."""
        logger.info("[HelpPanel] _schedule_index_build() invoked")
        if self._worker_pool and not getattr(self._worker_pool, "is_shutdown", False):
            if self._index_future is not None:
                logger.debug("[HelpPanel] Index build already scheduled")
                return
            logger.info("[HelpPanel] Submitting index build to worker pool")
            try:
                self._index_future = self._worker_pool.submit(self._build_index_background)
            except Exception as e:
                logger.error(f"[HelpPanel] Failed to submit background index build: {e}", exc_info=True)
                self._index_future = self._launch_index_thread()
                self.after(120, self._poll_index_future)
                return
            self.after(120, self._poll_index_future)
            return
        logger.info("[HelpPanel] Worker pool unavailable, running index build in dedicated thread")
        self._index_future = self._launch_index_thread()
        self.after(120, self._poll_index_future)

    def _launch_index_thread(self) -> Future:
        """Start a dedicated daemon thread to build the index."""

        logger.info("[HelpPanel] Launching dedicated index build thread")
        future: Future = Future()

        def _run() -> None:
            try:
                result = self._build_index_background()
            except Exception as exc:  # pragma: no cover - defensive logging
                future.set_exception(exc)
            else:
                future.set_result(result)

        thread = threading.Thread(target=_run, name="HelpIndexBuild", daemon=True)
        thread.start()
        return future

    def _poll_index_future(self) -> None:
        """Check whether the background index build has completed."""
        future = self._index_future
        if future is None:
            return
        if future.done():
            logger.info("[HelpPanel] Background index build completed")
            try:
                payload = future.result()
            except Exception as e:
                logger.error(f"[HelpPanel] Background index build failed: {e}", exc_info=True)
                payload = {"success": False, "error": str(e)}
            finally:
                self._index_future = None
            self._on_index_build_complete(payload)
            return
        self.after(120, self._poll_index_future)

    def _build_index_background(self) -> dict[str, Any]:
        """Run the index build off the UI thread."""
        logger.info("[HelpPanel] _build_index_background() running in worker thread")
        success = self._search_engine.build_index()
        stats = self._search_engine.get_index_stats() if success else ""
        payload: dict[str, Any] = {
            "success": success,
            "stats": stats,
        }
        if not success:
            payload["error"] = "Failed to index documentation"
        return payload

    def _on_index_build_complete(self, payload: dict[str, Any]) -> None:
        """Apply index build results once HTML frame is ready."""
        if not getattr(self, "_html_frame_initialized", False):
            logger.info("[HelpPanel] HtmlFrame not ready – deferring index payload application")
            self._pending_index_payload = payload
            return
        self._apply_index_payload(payload)

    def _apply_index_payload(self, payload: dict[str, Any]) -> None:
        """Finalize UI updates after an index build completes."""
        self._pending_index_payload = None
        success = bool(payload.get("success"))
        if not success:
            logger.warning("[HelpPanel] Index build reported failure")
            message = payload.get("error") or "Failed to index documentation"
            self._set_html_message(message)
            return

        stats = payload.get("stats", "")
        logger.info(f"[HelpPanel] Index build succeeded – stats: {stats}")

        try:
            if self._link_handler:
                self._link_handler.index = self._search_engine.index
                logger.debug(f"[HelpPanel] Link handler updated with {len(self._search_engine.index)} docs")
        except Exception as link_error:
            logger.error(f"[HelpPanel] Failed to update link handler: {link_error}", exc_info=True)

        if self._status_var is not None:
            try:
                self._status_var.set(stats)
            except Exception as status_error:
                logger.debug(f"[HelpPanel] Failed to update status label: {status_error}")

        try:
            self._update_settings_panel_status()
        except Exception as settings_error:
            logger.debug(f"[HelpPanel] Failed to propagate index status to settings panel: {settings_error}")

        logger.info("[HelpPanel] Scheduling TOC population after index build")
        self.after(250, self._populate_toc_and_default)
    
    def _build_index(self) -> None:
        """Build the search index in a background thread."""
        logger.info("[HelpPanel] _build_index() called - starting index build")

        html_ready = getattr(self, "_html_frame_initialized", False)
        retry_state = getattr(self, "_build_index_retry_state", {"count": 0, "total_delay": 0})

        if not html_ready:
            delay_ms = min(200 * (2 ** retry_state["count"]), 2000)
            next_total = retry_state["total_delay"] + delay_ms

            if next_total >= 5000:
                logger.warning(
                    "[HelpPanel] HtmlFrame still not initialized after %.2fs; proceeding with index build",
                    next_total / 1000.0,
                )
                html_ready = True  # fall through to scheduling logic
            else:
                retry_state["count"] += 1
                retry_state["total_delay"] = next_total
                self._build_index_retry_state = retry_state
                logger.info(
                    "[HelpPanel] HtmlFrame not ready, retrying in %dms (attempt %d)",
                    delay_ms,
                    retry_state["count"],
                )
                try:
                    self.after(delay_ms, self._build_index)
                except Exception as after_error:
                    logger.error(
                        "[HelpPanel] Failed to schedule HtmlFrame readiness check: %s",
                        after_error,
                        exc_info=True,
                    )
                return

        if html_ready:
            self._build_index_retry_state = {"count": 0, "total_delay": 0}

        if self._index_future is not None:
            logger.debug("[HelpPanel] Index build already in progress; skipping duplicate request")
            return

        try:
            self._schedule_index_build()
        except Exception as schedule_error:
            logger.error(
                "[HelpPanel] Failed to schedule index build: %s", schedule_error, exc_info=True
            )
            logger.info("[HelpPanel] Falling back to dedicated index build thread")
            self._index_future = self._launch_index_thread()
            self.after(120, self._poll_index_future)

    # ========== Search Handlers ==========
    
    def _on_search(self, event: Any = None) -> None:
        """Handle search button or Enter key."""
        query = (self.search_var.get() or "").strip()
        if query:
            logger.info(f"Help search: query=\"{query}\"")
            self._mode = 'search'
            
            import time
            start_time = time.time()
            self._run_search(query)
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Log search results
            results_count = len(self.results.get_children())
            logger.info(f"Found {results_count} results in {duration_ms}ms")
            
            # Log top result if available
            if results_count > 0:
                try:
                    top_item = self.results.get_children()[0]
                    top_title = self.results.item(top_item, "text")
                    logger.debug(f"Top result: {top_title}")
                except Exception:
                    pass
        else:
            self._mode = 'toc'
            self._populate_toc_and_default(populate_only=True)

    def _on_key_release(self, event: Any) -> None:
        """Handle key release with debouncing."""
        try:
            if self._debounce_after_id is not None:
                self.after_cancel(self._debounce_after_id)
        except Exception:
            pass
        
        try:
            self._debounce_after_id = self.after(250, self._on_search)
        except Exception:
            self._on_search()

    def _on_clear(self) -> None:
        """Clear search and return to TOC view."""
        logger.debug("Search cleared")
        
        self.search_var.set("")
        for i in self.results.get_children():
            self.results.delete(i)
        
        try:
            self._mode = 'toc'
            self._populate_toc_and_default(populate_only=True)
        except Exception:
            pass
        
        self._set_html_message(
            "Type to search the documentation. Press F1 anywhere for Help."
        )

    def _run_search(self, query: str) -> None:
        """Run a search and display results.
        
        Args:
            query: Search query string
        """
        # Get search results
        results = self._search_engine.search(query)
        
        # Populate tree with results
        self._toc_builder.populate_search_results(results)
        
        # Update status message
        if results:
            self._set_html_message(
                f"Found {len(results)} relevant documents for '{query}'. "
                f"Expand nodes to see matching sections. Select to preview."
            )
        else:
            self._set_html_message(
                f"No results found for '{query}'. Try different search terms."
            )

    # ========== Tree Navigation ==========
    
    def _on_tree_single_click(self, event: Any = None) -> None:
        """Handle single-click on tree items."""
        sel = self.results.selection()
        if not sel:
            return
        
        item = sel[0]
        
        # Check if clicking toggle icon (let default behavior handle it)
        try:
            cursor_x = self.results.winfo_pointerx() - self.results.winfo_rootx()
            cursor_y = self.results.winfo_pointery() - self.results.winfo_rooty()
            element = self.results.identify_element(cursor_x, cursor_y)
            if element == 'Treeitem.indicator':
                return
        except Exception:
            pass
        
        # Toggle folder expand/collapse or open file
        if self.results.get_children(item):
            is_open = self.results.item(item, 'open')
            self.results.item(item, open=not is_open)
        else:
            self._on_select(event)

    def _on_select(self, event: Any = None) -> None:
        """Handle selection of a tree item."""
        sel = self.results.selection()
        if not sel:
            return
        
        item = sel[0]
        rel_path = self.results.set(item, "path") or self.results.item(item, "text")
        
        # Get heading line if available
        heading_val = self.results.set(item, "line")
        try:
            heading_line = int(heading_val)
        except Exception:
            heading_line = -1
        
        # Skip if it's a directory
        try:
            if "." not in os.path.basename(rel_path) and self.results.get_children(item):
                self.results.item(item, open=True)
                return
        except Exception:
            pass
        
        # Open document
        if rel_path:
            title = self.results.item(item, "text")
            logger.info(f"Help document selected: {title}")
            logger.debug(f"Document path: {rel_path}")
            self._open_doc(rel_path, heading_line)

    # ========== TOC Building ==========
    
    def _populate_toc_and_default(self, populate_only: bool = False) -> None:
        """Populate the TOC and optionally load a default document.
        
        Args:
            populate_only: If True, only populate TOC without loading a document
        """
        logger.info(f"[HelpPanel] _populate_toc_and_default called, populate_only={populate_only}")
        
        # Safety check: ensure tree widget exists and is ready
        try:
            if not self.results.winfo_exists():
                logger.error("[HelpPanel] ERROR: Tree widget doesn't exist yet!")
                return
        except Exception as e:
            logger.error(f"[HelpPanel] ERROR: Cannot check tree widget existence: {e}")
            return
        
        # Wait for HtmlFrame to finish initializing if we're going to load a document
        if not populate_only and not getattr(self, '_html_frame_initialized', False):
            logger.info("[HelpPanel] HtmlFrame not ready yet, waiting 500ms before loading default doc...")
            self.after(500, lambda: self._populate_toc_and_default(populate_only=populate_only))
            return
        
        try:
            # Build TOC tree
            logger.info(f"[HelpPanel] Building TOC with {len(self._search_engine.index)} documents")
            top_items = self._toc_builder.build_toc(self._search_engine.index)
            logger.info(f"[HelpPanel] TOC builder returned {len(top_items)} top-level items")
            
            # Update status
            try:
                if self._status_var is not None:
                    self._status_var.set(
                        f"Indexed {len(self._search_engine.index)} docs under {self._docs_root} | "
                        f"TOC roots: {len(self.results.get_children())}"
                    )
            except Exception:
                pass
        except Exception:
            pass

        if populate_only:
            return
        
        # Reinforce left pane visibility
        try:
            self.after(50, self._set_initial_sash)
        except Exception:
            pass
        
        # Load default document
        try:
            logger.info("[HelpPanel] Looking for default document to load...")
            preferred = ['guide/INDEX.MD', 'README.md']
            index_paths = [p for (p, _t, _g, _h) in self._search_engine.index]
            
            rel = None
            for cand in preferred:
                if cand in index_paths:
                    rel = cand
                    break
            
            if rel is None and index_paths:
                rel = index_paths[0]
            
            if rel:
                logger.info(f"[HelpPanel] Loading default document: {rel}")
                self._open_doc(rel)
                logger.info("[HelpPanel] Default document loaded successfully")
            else:
                logger.warning("[HelpPanel] No default document found to load")
        except Exception as e:
            logger.error(f"[HelpPanel] Error loading default document: {e}", exc_info=True)
            pass

    # ========== Document Display ==========
    
    def _open_doc(
        self, 
        rel: str, 
        heading_line: int = -1, 
        from_history: bool = False
    ) -> None:
        """Open and display a document.
        
        Args:
            rel: Relative path to document
            heading_line: Optional line number to scroll to
            from_history: True if navigating from history (don't push to history)
        """
        if self._loading_doc:
            return
        
        self._loading_doc = True
        try:
            # Find document in index
            found = next(
                ((p, c, t, h) for (p, c, t, h) in self._search_engine.index if p == rel),
                None
            )
            if not found:
                self._set_html_message(f"Document not found: {rel}")
                return
            
            _p, content, _t, headings = found
            
            # Find anchor if heading line is specified
            target_anchor = None
            if heading_line >= 0:
                for line_num, heading_text in headings:
                    if line_num == heading_line:
                        target_anchor = utils.slugify_github_anchor(heading_text)
                        break
            
            # Update link handler state
            self._current_doc_rel = rel
            if self._link_handler:
                self._link_handler.set_current_doc(rel)
                pending = self._link_handler.get_pending_anchor()
            else:
                pending = None
            if not target_anchor and pending:
                target_anchor = pending
            
            # Render and display
            if target_anchor:
                self._schedule_scroll("anchor", target_anchor)
            else:
                self._schedule_scroll("bottom")
            self._set_rendered_html(content, on_loaded=self._apply_pending_scroll)
            
            # Update navigation history
            if not from_history:
                self._push_history(rel, heading_line)
            
            self._update_nav_buttons()
        except Exception as e:
            self._set_html_message(f"Failed to open document: {e}")
        finally:
            self._loading_doc = False

    def _set_html_message(self, msg: str) -> None:
        """Display a simple message in the HTML pane.
        
        Args:
            msg: Message to display
        """
        try:
            logger.info(f"[HelpPanel] _set_html_message START: {msg[:100] if msg else 'None'}")
            
            # Check if HtmlFrame is ready
            html_init_status = getattr(self, '_html_frame_initialized', False)
            logger.info(f"[HelpPanel] _html_frame_initialized = {html_init_status}")
            
            if not html_init_status:
                logger.info("[HelpPanel] HtmlFrame not initialized yet, skipping message display")
                return
            
            logger.info(f"[HelpPanel] html_view is None: {self.html_view is None}")
            
            if self.html_view is not None:
                # Get theme colors
                logger.info("[HelpPanel] Getting theme colors...")
                colors = self._get_theme_colors()
                logger.info("[HelpPanel] Got theme colors, building HTML page...")
                html_page = (
                    "<html><head><meta charset='utf-8'><style>"
                    f"body{{font-family:'Segoe UI', Arial, sans-serif; "
                    f"background:{colors['bg']}; color:{colors['fg']}; padding:16px; line-height:1.6;}}"
                    "p{margin:0;}"
                    "</style></head><body><p>" + (msg or "") + "</p></body></html>"
                )
                logger.info("[HelpPanel] Loading HTML into HtmlFrame...")
                try:
                    # Try to load HTML - this may block on some systems
                    self.html_view.load_html(html_page)  # type: ignore[attr-defined]
                    logger.info("[HelpPanel] Message loaded into HtmlFrame successfully")
                except Exception as load_error:
                    logger.error(f"[HelpPanel] Failed to load HTML: {load_error}", exc_info=True)
                    # Fall back to updating on next event loop iteration
                    try:
                        self.after(50, lambda: self.html_view.load_html(html_page) if self.html_view else None)
                        logger.info("[HelpPanel] Scheduled delayed HTML load")
                    except Exception:
                        pass
            else:
                logger.info("[HelpPanel] HtmlFrame is None, using label fallback")
                for w in self.html_frame.winfo_children():
                    try:
                        w.destroy()
                    except Exception:
                        pass
                ttk.Label(self.html_frame, text=msg or "").pack(padx=12, pady=12, anchor="w")
                logger.info("[HelpPanel] Label fallback complete")
        except Exception as e:
            logger.error(f"[HelpPanel] Error in _set_html_message: {e}", exc_info=True)

    def _set_rendered_html(
        self,
        md_text: str,
        on_loaded: Optional[Callable[[], None]] = None,
    ) -> None:
        """Render markdown and display as HTML, preferring async rendering."""
        if not md_text:
            if on_loaded:
                self._invoke_render_callback(on_loaded)
            return

        try:
            if self._worker_pool and not getattr(self._worker_pool, "is_shutdown", False):
                self._render_markdown_async(md_text, on_loaded)
            else:
                html_page, base_url = self._render_markdown_sync(md_text)
                self._apply_rendered_html(html_page, base_url)
                self._invoke_render_callback(on_loaded)
        except Exception as exc:
            logger.error(f"[HelpPanel] Failed to render markdown: {exc}", exc_info=True)

    def _render_markdown_sync(self, md_text: str) -> tuple[str, str]:
        """Render markdown to styled HTML synchronously."""
        html = self._md_renderer.render(md_text)

        if self._link_handler:
            html = self._link_handler.transform_html_links(html)
            base_url = self._link_handler.get_base_url()
        else:
            base_url = ""

        html_page = self._create_styled_html_page(html)
        return html_page, base_url

    def _apply_rendered_html(self, html_page: str, base_url: str) -> None:
        """Apply pre-rendered HTML content to the HtmlFrame on the main thread."""
        if self.html_view is None:
            return

        try:
            self.html_view.load_html(html_page, base_url=base_url)  # type: ignore[attr-defined]
        except Exception as load_error:
            logger.error(f"[HelpPanel] Failed to load HTML: {load_error}", exc_info=True)
            try:
                self.after(50, lambda: self.html_view.load_html(html_page, base_url=base_url) if self.html_view else None)  # type: ignore[attr-defined]
                logger.info("[HelpPanel] Scheduled delayed HTML load after failure")
            except Exception:
                pass

        try:
            self._protect_mousewheel_events()
        except Exception:
            pass

    def _schedule_scroll(self, action: str, data: Optional[str] = None) -> None:
        """Remember a scroll action to run once the HTML widget is ready."""
        self._pending_scroll = (action, data)

    def _apply_pending_scroll(self) -> None:
        """Execute any pending scroll action with retries for slow renders."""
        target = self._pending_scroll
        if not target:
            return

        self._pending_scroll = None
        action, data = target

        def _attempt_scroll(attempt: int = 0) -> None:
            success = False
            try:
                if action == "anchor" and data:
                    if self._link_handler:
                        self._link_handler._scroll_to_anchor(data)
                        success = True
                    else:
                        success = True
                elif action == "bottom":
                    success = self._scroll_html_to_bottom()
            except Exception as exc:
                logger.debug(f"[HelpPanel] Scroll attempt failed: {exc}")

            if not success and attempt < 5:
                try:
                    self.after(100, lambda: _attempt_scroll(attempt + 1))
                except Exception:
                    pass

        try:
            self.after(50, _attempt_scroll)
        except Exception:
            _attempt_scroll()

    def _scroll_html_to_bottom(self) -> bool:
        """Scroll the documentation view to the bottom of the page."""
        if not self.html_view:
            return False

        try:
            if hasattr(self.html_view, "execute_script"):
                self.html_view.execute_script("window.scrollTo(0, document.body.scrollHeight);")  # type: ignore[attr-defined]
                return True
            if hasattr(self.html_view, "run_script"):
                self.html_view.run_script("window.scrollTo(0, document.body.scrollHeight);")  # type: ignore[attr-defined]
                return True

            document = getattr(self.html_view, "document", None)
            if document is not None:
                body = getattr(document, "body", None)
                if body is not None:
                    try:
                        body.scrollTop = getattr(body, "scrollHeight", 0)  # type: ignore[attr-defined]
                        return True
                    except Exception:
                        pass
        except Exception as exc:
            logger.debug(f"[HelpPanel] Scroll to bottom via script failed: {exc}")

        return False

    def _invoke_render_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Invoke a render completion callback on the Tk event loop."""
        if not callback:
            return
        try:
            self.after_idle(callback)
        except Exception:
            try:
                callback()
            except Exception:
                pass

    def _render_markdown_async(
        self,
        md_text: str,
        on_loaded: Optional[Callable[[], None]] = None,
    ) -> None:
        """Render markdown on a worker thread, then apply on the main thread."""
        if not self._worker_pool or getattr(self._worker_pool, "is_shutdown", False):
            html_page, base_url = self._render_markdown_sync(md_text)
            self._apply_rendered_html(html_page, base_url)
            self._invoke_render_callback(on_loaded)
            return

        if self._pending_render:
            future = self._pending_render.get("future")
            try:
                if future and not future.done():
                    future.cancel()
            except Exception:
                pass

        self._render_request_id += 1
        request_id = self._render_request_id
        start_ts = time.perf_counter()

        def _render_job() -> tuple[str, str]:
            return self._render_markdown_sync(md_text)

        future = self._worker_pool.submit(_render_job)
        self._pending_render = {
            "id": request_id,
            "future": future,
            "start": start_ts,
            "callback": on_loaded,
        }
        self.after(5, self._poll_render_future)

    def _poll_render_future(self) -> None:
        """Poll the pending markdown render future and apply results when ready."""
        task = self._pending_render
        if not task:
            return

        request_id = task.get("id")
        future = task.get("future")
        callback = task.get("callback")
        start_ts = task.get("start", time.perf_counter())

        if request_id != self._render_request_id:
            if future and not future.done():
                try:
                    future.cancel()
                except Exception:
                    pass
            self._pending_render = None
            return

        if future is None:
            self._pending_render = None
            return

        if not future.done():
            self.after(10, self._poll_render_future)
            return

        try:
            html_page, base_url = future.result()
        except Exception as exc:
            logger.error(f"[HelpPanel] Markdown render failed: {exc}", exc_info=True)
            self._pending_render = None
            return

        elapsed = time.perf_counter() - start_ts
        logger.debug(f"[HelpPanel] Markdown rendered asynchronously in {elapsed:.3f}s")
        self._pending_render = None

        def _apply() -> None:
            self._apply_rendered_html(html_page, base_url)
            self._invoke_render_callback(callback)

        try:
            self.after_idle(_apply)
        except Exception:
            _apply()

    def _create_styled_html_page(self, html_content: str) -> str:
        """Wrap HTML content in a styled page.
        
        Args:
            html_content: HTML body content
            
        Returns:
            Complete HTML page with styling
        """
        # Get theme colors
        colors = self._get_theme_colors()
        
        return (
            "<html><head><meta charset='utf-8'><style>"
            "body{font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', "
            "Helvetica, Arial, sans-serif, 'Apple Color Emoji','Segoe UI Emoji'; "
            f"background:{colors['bg']}; color:{colors['fg']}; padding:16px; line-height:1.6;}}"
            f"h1,h2,h3{{border-bottom:1px solid {colors['border']}; padding-bottom:.3em; margin-top:1.5em;}}"
            f"pre{{background:{colors['code_bg']}; padding:12px; overflow:auto; border-radius:6px; "
            f"border:1px solid {colors['border']};}}"
            f"code{{background:{colors['code_bg']}; padding:2px 6px; border-radius:4px; "
            "font-family:'Consolas','Monaco','Courier New',monospace; font-size:0.9em;}"
            "pre code{background:transparent; padding:0; border:none;}"
            "table{border-collapse:collapse; display:block; width:100%; overflow:auto;} "
            f"td,th{{border:1px solid {colors['border']}; padding:6px 13px;}}"
            f"a{{color:{colors['link']}; text-decoration:none; cursor:pointer;}} "
            "a:hover{text-decoration:underline;}"
            "ul,ol{padding-left:2em;}"
            f".warn{{background:{colors['warn_bg']};border:1px solid {colors['warn_border']};padding:10px;"
            "border-radius:6px;margin-bottom:12px;}"
            "</style></head><body>" + html_content + "</body></html>"
        )

    # ========== Link Handling ==========
    
    def _handle_link_click(self, url: str) -> None:
        """Handle link clicks from the HTML view.
        
        Args:
            url: URL that was clicked
        """
        if self._link_handler:
            self._link_handler.handle_link_click(url)

    # ========== Navigation History ==========
    
    def _update_nav_buttons(self) -> None:
        """Update the state of navigation buttons."""
        try:
            can_back = self._nav_index > 0
            can_fwd = (
                self._nav_history 
                and self._nav_index >= 0 
                and self._nav_index < len(self._nav_history) - 1
            )
            
            self.back_btn.configure(state=tk.NORMAL if can_back else tk.DISABLED)
            self.forward_btn.configure(state=tk.NORMAL if can_fwd else tk.DISABLED)
        except Exception:
            pass

    def _push_history(self, rel: str, heading_line: int) -> None:
        """Push a navigation entry to history.
        
        Args:
            rel: Document path
            heading_line: Heading line number (-1 for top)
        """
        try:
            if self._nav_in_progress:
                return
            
            # Don't duplicate current entry
            if self._nav_index >= 0 and self._nav_index < len(self._nav_history):
                cur_rel, cur_line = self._nav_history[self._nav_index]
                if cur_rel == rel and cur_line == heading_line:
                    return
            
            # Truncate forward history
            if self._nav_index < len(self._nav_history) - 1:
                self._nav_history = self._nav_history[: self._nav_index + 1]
            
            self._nav_history.append((rel, heading_line))
            self._nav_index = len(self._nav_history) - 1
            self._update_nav_buttons()
        except Exception:
            pass

    def _nav_back(self) -> None:
        """Navigate back in history."""
        try:
            if self._nav_index <= 0:
                return
            
            self._nav_index -= 1
            rel, line = self._nav_history[self._nav_index]
            
            self._nav_in_progress = True
            try:
                self._open_doc(rel, line, from_history=True)
            finally:
                self._nav_in_progress = False
            
            self._update_nav_buttons()
        except Exception:
            pass

    def _nav_forward(self) -> None:
        """Navigate forward in history."""
        try:
            if self._nav_index >= len(self._nav_history) - 1:
                return
            
            self._nav_index += 1
            rel, line = self._nav_history[self._nav_index]
            
            self._nav_in_progress = True
            try:
                self._open_doc(rel, line, from_history=True)
            finally:
                self._nav_in_progress = False
            
            self._update_nav_buttons()
        except Exception:
            pass

    # ========== External Actions ==========
    
    def _on_open_external(self) -> None:
        """Open the current document externally."""
        rel = self._current_selection_path()
        if not rel:
            return
        
        logger.info(f"Opening external link: {rel}")
        
        # Get anchor if heading is selected
        anchor = None
        try:
            sel = self.results.selection()
            if sel:
                txt = self.results.item(sel[0], "text") or ""
                if isinstance(txt, str) and txt.startswith("# "):
                    anchor = utils.slugify_github_anchor(txt[2:])
        except Exception:
            anchor = None
        
        # Open externally
        if self._link_handler:
            self._link_handler.open_external_in_browser(rel, anchor)

    def _current_selection_path(self) -> Optional[str]:
        """Get the path of the currently selected item.
        
        Returns:
            Document path, or None if no selection
        """
        try:
            sel = self.results.selection()
            if not sel:
                return None
            item = sel[0]
            return self.results.set(item, "path") or self.results.item(item, "text")
        except Exception:
            return None

    # ========== Layout Helpers ==========
    
    def _set_initial_sash(self) -> None:
        """Set the initial sash position for the paned window."""
        try:
            w = 0
            try:
                w = int(self._pane.winfo_width())  # type: ignore[attr-defined]
            except Exception:
                pass
            if not w or w <= 0:
                w = self.winfo_width()
            if w and w > 0:
                pos = int(w * 0.25)
                try:
                    self._pane.sashpos(0, pos)
                    self._sash_set_once = True
                except Exception:
                    pass
        except Exception:
            pass

    def _on_pane_configure(self, event: Any) -> None:
        """Handle pane configure event to set sash once."""
        try:
            if getattr(self, "_sash_set_once", False):
                return
            
            w = int(event.width) if hasattr(event, "width") else 0
            if not w or w <= 0:
                try:
                    w = int(self._pane.winfo_width())  # type: ignore[attr-defined]
                except Exception:
                    w = 0
            
            if w and w > 0:
                pos = int(w * 0.25)
                try:
                    self._pane.sashpos(0, pos)
                    self._sash_set_once = True
                except Exception:
                    pass
        except Exception:
            pass

    def _ensure_left_visible(self) -> None:
        """Ensure the left TOC pane has visible width."""
        try:
            self.update_idletasks()
        except Exception:
            pass
        
        try:
            lw = int(self._left_frame.winfo_width())
            pw = int(self._pane.winfo_width()) if hasattr(self._pane, 'winfo_width') else 0
        except Exception:
            lw, pw = 0, 0
        
        try:
            if (lw <= 140) and (pw > 0):
                pos = max(200, int(pw * 0.25))
                try:
                    self._pane.sashpos(0, pos)
                except Exception:
                    pass
                try:
                    self.after(250, self._ensure_left_visible)
                except Exception:
                    pass
        except Exception:
            pass

    def _protect_mousewheel_events(self) -> None:
        """Protect HTML view from global mousewheel handlers."""
        if self.html_view is None:
            return

        def _swallow(w: Any) -> None:
            try:
                for seq in ("<MouseWheel>", "<Shift-MouseWheel>", "<Control-MouseWheel>"):
                    try:
                        w.bind(seq, lambda e: "break")
                    except Exception:
                        pass
            except Exception:
                pass

        def _recurse_bind(widget: Any) -> None:
            _swallow(widget)
            try:
                for ch in widget.winfo_children():
                    _recurse_bind(ch)
            except Exception:
                pass

        _recurse_bind(self.html_view)
        try:
            self.after(200, lambda: _recurse_bind(self.html_view))
            self.after(800, lambda: _recurse_bind(self.html_view))
            self.after(2000, lambda: _recurse_bind(self.html_view))
        except Exception:
            pass

    # ========== Public API ==========
    
    def focus_search(self) -> None:
        """Focus the search entry (called when F1 is pressed)."""
        try:
            self.search_entry.focus_set()
        except Exception:
            pass
        try:
            self._ensure_left_visible()
        except Exception:
            pass
    
    def update_theme(self, theme: str) -> None:
        """Update the help panel theme.
        
        Args:
            theme: Theme name (e.g., 'Light Mode', 'Dark Mode', 'Matrix Mode', etc.)
        """
        # Normalize theme name
        theme_normalized = theme.lower().replace(" ", "").replace("mode", "")
        if theme_normalized == "dark":
            self._current_theme = "dark"
        elif theme_normalized == "matrix":
            self._current_theme = "matrix"
        elif theme_normalized == "halloween":
            self._current_theme = "halloween"
        elif theme_normalized == "barbie":
            self._current_theme = "barbie"
        else:
            self._current_theme = "light"
        
        # Re-render current document if one is loaded
        if self._current_doc_rel:
            try:
                found = next(
                    ((p, c, t, h) for (p, c, t, h) in self._search_engine.index if p == self._current_doc_rel),
                    None
                )
                if found:
                    _p, content, _t, _h = found
                    self._set_rendered_html(content)
            except Exception:
                pass
    
    def _get_theme_colors(self) -> dict[str, str]:
        """Get HTML colors for the current theme.
        
        Returns:
            Dictionary with color values for HTML rendering
        """
        if self._current_theme == "dark":
            return {
                "bg": "#2b2b2b",
                "fg": "#e0e0e0",
                "border": "#404040",
                "code_bg": "#353535",
                "link": "#58a6ff",
                "warn_bg": "#3d3600",
                "warn_border": "#6b6000",
            }
        elif self._current_theme == "matrix":
            return {
                "bg": "#000000",
                "fg": "#00ff41",
                "border": "#003300",
                "code_bg": "#0a0a0a",
                "link": "#00ff00",  # Bright pure green for better contrast
                "warn_bg": "#001a00",
                "warn_border": "#00cc33",
            }
        elif self._current_theme == "halloween":
            return {
                "bg": "#1a0f00",
                "fg": "#ffcc99",
                "border": "#4d2600",
                "code_bg": "#3d1f00",
                "link": "#ff8c00",
                "warn_bg": "#2d1a00",
                "warn_border": "#ff6600",
            }
        elif self._current_theme == "barbie":
            return {
                "bg": "#FFB6C1",
                "fg": "#8B008B",
                "border": "#FF69B4",
                "code_bg": "#FFF0F5",
                "link": "#FF1493",
                "warn_bg": "#FFE4E1",
                "warn_border": "#FF69B4",
            }
        else:  # Light mode (default)
            return {
                "bg": "#ffffff",
                "fg": "#24292f",
                "border": "#d0d7de",
                "code_bg": "#e8ebed",
                "link": "#0969da",
                "warn_bg": "#fff8c5",
                "warn_border": "#e3cc00",
            }
    
    def _update_settings_panel_status(self) -> None:
        """Update the settings panel's help index status label if available."""
        try:
            logger.info("[HelpPanel] _update_settings_panel_status() START")
            
            # CRITICAL FIX: Skip settings panel update during initialization to prevent hangs
            # The settings panel will be updated later when it's fully initialized
            if not getattr(self, '_settings_panel', None):
                logger.info("[HelpPanel] Skipping settings panel update - not initialized yet")
                return
            
            # Check if we have a reference to the settings panel
            if hasattr(self, '_settings_panel') and self._settings_panel:
                logger.info("[HelpPanel] Settings panel exists, checking for status label...")
                if hasattr(self._settings_panel, 'help_index_status_label'):
                    logger.info("[HelpPanel] Status label found, updating...")
                    doc_count = len(self._search_engine.index) if self._search_engine else 0
                    status_text = f"✓ Ready ({doc_count} docs)"
                    
                    # Schedule the label update on next event loop to avoid blocking
                    def _update_label():
                        try:
                            self._settings_panel.help_index_status_label.config(text=status_text)
                            logger.debug(f"Updated settings panel help index status: {status_text}")
                        except Exception as label_error:
                            logger.debug(f"Could not update label: {label_error}")
                    
                    self.after(0, _update_label)
                    logger.info("[HelpPanel] Scheduled label update")
                else:
                    logger.info("[HelpPanel] Status label not found on settings panel")
            else:
                logger.info("[HelpPanel] Settings panel not available")
            
            logger.info("[HelpPanel] _update_settings_panel_status() END")
        except Exception as e:
            # Silently fail - this is just a status update
            logger.debug(f"Could not update settings panel status: {e}")
