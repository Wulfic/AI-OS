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

import os
import threading
from pathlib import Path
from typing import Any, Optional, cast

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

# Import our modular components
from . import utils
from .search_engine import SearchEngine
from .link_handler import LinkHandler
from .markdown_renderer import MarkdownRenderer
from .toc_builder import TOCBuilder


class HelpPanel(ttk.LabelFrame):  # type: ignore[misc]
    """Main Help Panel widget with search, TOC, and HTML preview."""
    
    def __init__(self, parent: Any, project_root: Optional[Path] = None) -> None:
        """Initialize the Help Panel.
        
        Args:
            parent: Parent Tkinter widget
            project_root: Optional project root path (auto-detected if not provided)
        """
        super().__init__(parent, text="Help & Docs")
        if tk is None:
            raise RuntimeError("Tkinter not available")
        self.pack(fill="both", expand=True, padx=8, pady=8)

        # Resolve project/docs roots
        self._project_root = utils.find_project_root(project_root or Path(__file__))
        self._docs_root = utils.resolve_docs_root(self._project_root)

        # Initialize components
        self._search_engine = SearchEngine(self._docs_root)
        self._md_renderer = MarkdownRenderer()
        
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
        
        # Initialize link handler (after html_view is created)
        # Note: index will be empty initially, updated after build_index completes
        if self.html_view is not None:
            self._link_handler = LinkHandler(
                self.html_view,
                self._docs_root,
                [],  # Empty initially, will be updated after indexing
                self._open_doc
            )
        else:
            self._link_handler = None
        
        # Initialize TOC builder
        self._toc_builder = TOCBuilder(self.results)
        
        self._update_nav_buttons()
        
        # Index in background
        threading.Thread(target=self._build_index, daemon=True).start()

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
        self.search_var = tk.StringVar()
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
            self._status_var = tk.StringVar(value="")
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
        self.html_view = None
        
        if HtmlFrame is not None:
            try:
                # Use on_link_click callback to handle link clicks
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
                    )  # type: ignore[call-arg]
                except Exception:
                    self.html_view = HtmlFrame(self.html_frame)  # type: ignore[call-arg]
            
            self.html_view.pack(fill="both", expand=True)
            
            try:
                self._protect_mousewheel_events()
            except Exception:
                pass
        else:
            ttk.Label(
                self.html_frame,
                text="Install tkinterweb to enable rich rendering"
            ).pack(padx=12, pady=12)

    # ========== Indexing ==========
    
    def _build_index(self) -> None:
        """Build the search index in a background thread."""
        try:
            success = self._search_engine.build_index()
            if success:
                # Update link handler with the built index
                if self._link_handler:
                    self._link_handler.index = self._search_engine.index
                
                stats = self._search_engine.get_index_stats()
                self._set_html_message(stats)
                try:
                    if self._status_var is not None:
                        self._status_var.set(stats)
                except Exception:
                    pass
                self._populate_toc_and_default()
            else:
                self._set_html_message("Failed to index documentation")
        except Exception as e:
            self._set_html_message(f"Failed to index docs: {e}")

    # ========== Search Handlers ==========
    
    def _on_search(self, event: Any = None) -> None:
        """Handle search button or Enter key."""
        query = (self.search_var.get() or "").strip()
        if query:
            self._mode = 'search'
            self._run_search(query)
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
            self._open_doc(rel_path, heading_line)

    # ========== TOC Building ==========
    
    def _populate_toc_and_default(self, populate_only: bool = False) -> None:
        """Populate the TOC and optionally load a default document.
        
        Args:
            populate_only: If True, only populate TOC without loading a document
        """
        try:
            # Build TOC tree
            top_items = self._toc_builder.build_toc(self._search_engine.index)
            
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
                self._open_doc(rel)
        except Exception:
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
            
            # Render and display
            self._set_rendered_html(content)
            
            # Scroll to anchor if needed
            if target_anchor and self._link_handler:
                # Schedule scroll after render
                self.after(100, lambda: self._link_handler._scroll_to_anchor(target_anchor))
            
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
            if self.html_view is not None:
                # Get theme colors
                colors = self._get_theme_colors()
                html_page = (
                    "<html><head><meta charset='utf-8'><style>"
                    f"body{{font-family:'Segoe UI', Arial, sans-serif; "
                    f"background:{colors['bg']}; color:{colors['fg']}; padding:16px; line-height:1.6;}}"
                    "p{margin:0;}"
                    "</style></head><body><p>" + (msg or "") + "</p></body></html>"
                )
                self.html_view.load_html(html_page)  # type: ignore[attr-defined]
            else:
                for w in self.html_frame.winfo_children():
                    try:
                        w.destroy()
                    except Exception:
                        pass
                ttk.Label(self.html_frame, text=msg or "").pack(padx=12, pady=12, anchor="w")
        except Exception:
            pass

    def _set_rendered_html(self, md_text: str) -> None:
        """Render markdown and display as HTML.
        
        Args:
            md_text: Markdown content
        """
        try:
            if self.html_view is None:
                return
            
            # Convert markdown to HTML
            html = self._md_renderer.render(md_text)
            
            # Transform links for interception
            if self._link_handler:
                html = self._link_handler.transform_html_links(html)
            
            # Create styled HTML page
            html_page = self._create_styled_html_page(html)
            
            # Get base URL
            base_url = self._link_handler.get_base_url() if self._link_handler else ""
            
            # Load HTML
            self.html_view.load_html(html_page, base_url=base_url)  # type: ignore[attr-defined]
            
            # Re-protect mousewheel and ensure link handler
            try:
                self._protect_mousewheel_events()
            except Exception:
                pass
        except Exception:
            pass

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
