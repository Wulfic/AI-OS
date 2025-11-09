"""
UI Builder for Dataset Download Panel

Functions for building the user interface components.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from typing import Callable

# Import tooltip helper
try:
    from ..tooltips import add_tooltip
except ImportError:
    def add_tooltip(widget, text, **kwargs):  # type: ignore
        pass  # No-op if not available


def build_ui(panel):
    """
    Build the complete UI for the dataset download panel.
    
    Args:
        panel: DatasetDownloadPanel instance to build UI into
    """
    # Create main frame
    panel.frame = ttk.LabelFrame(panel.parent, text="📦 Dataset Search & Downloads", padding=10)
    panel.frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    # Info label
    info_label = ttk.Label(
        panel.frame,
        text="Search HuggingFace for datasets, favorite them, and download for AI-OS training",
        font=("", 9)
    )
    info_label.pack(anchor="w", pady=(0, 5))
    
    # Build control bar
    _build_control_bar(panel)
    
    # Build search frame
    _build_search_frame(panel)
    
    # Build results tree
    _build_results_frame(panel)
    
    # Build action buttons
    _build_action_buttons(panel)
    
    # Build download location
    _build_location_frame(panel)
    
    # Build status label
    panel.status_label = ttk.Label(panel.frame, text="Ready", font=("", 9, "italic"))
    panel.status_label.pack(anchor="w", pady=(5, 0))
    
    # Build output text area (skip if output_parent is None and we have a log callback)
    # This allows using a shared output panel at the top of the tab
    if panel._output_parent is not None or not hasattr(panel, 'log'):
        _build_output_frame(panel)
    else:
        # Using shared output via log callback - create dummy widget to avoid errors
        panel.output_text = None


def _build_control_bar(panel):
    """Build the top control bar with favorites and HF auth buttons."""
    control_frame = ttk.Frame(panel.frame)
    control_frame.pack(fill="x", pady=(0, 5))
    
    # Favorites button
    panel.favorites_btn = ttk.Button(
        control_frame,
        text="⭐ View Favorites",
        command=panel._show_favorites_popup,
        width=18
    )
    add_tooltip(panel.favorites_btn, "View and manage your favorited datasets")
    panel.favorites_btn.pack(side="left", padx=(0, 10))
    
    # HuggingFace authentication
    hf_frame = ttk.Frame(control_frame)
    hf_frame.pack(side="right")
    
    panel.hf_auth_btn = ttk.Button(
        hf_frame,
        text="🔐 Login to HF",
        command=panel._show_login_dialog,
        width=15
    )
    add_tooltip(panel.hf_auth_btn, "Login to HuggingFace to access gated/private datasets")
    panel.hf_auth_btn.pack(side="left", padx=(0, 5))
    
    panel.hf_status_label = ttk.Label(
        hf_frame,
        text="Not logged in",
        font=("", 8, "italic"),
        foreground="gray"
    )
    add_tooltip(panel.hf_status_label, "HuggingFace login status. Some datasets require authentication.")
    panel.hf_status_label.pack(side="left")


def _build_search_frame(panel):
    """Build the search frame with query and filters."""
    panel.search_frame = ttk.LabelFrame(panel.frame, text="🔍 Search HuggingFace Datasets", padding=5)
    panel.search_frame.pack(fill="x", pady=(0, 5))
    
    search_controls = ttk.Frame(panel.search_frame)
    search_controls.pack(fill="x", pady=(0, 5))
    
    # Search query field
    ttk.Label(search_controls, text="Search:").pack(side="left", padx=(0, 5))
    
    panel.search_var = tk.StringVar()
    search_entry = ttk.Entry(search_controls, textvariable=panel.search_var, width=35)
    add_tooltip(search_entry, "Search HuggingFace datasets by name, topic, or task")
    search_entry.pack(side="left", padx=(0, 10))
    search_entry.bind("<Return>", lambda e: panel._do_search())
    
    # Max size filter field
    ttk.Label(search_controls, text="Max Size:").pack(side="left", padx=(0, 5))
    
    panel.max_size_var = tk.StringVar(value="")  # Empty = no limit
    max_size_entry = ttk.Entry(search_controls, textvariable=panel.max_size_var, width=8)
    add_tooltip(max_size_entry, "Maximum dataset size in GB (leave empty for no limit)")
    max_size_entry.pack(side="left", padx=(0, 2))
    max_size_entry.bind("<Return>", lambda e: panel._do_search())
    
    # Size unit dropdown
    panel.size_unit_var = tk.StringVar(value="GB")
    size_unit_combo = ttk.Combobox(
        search_controls,
        textvariable=panel.size_unit_var,
        values=["MB", "GB", "TB"],
        state="readonly",
        width=5
    )
    size_unit_combo.pack(side="left", padx=(0, 10))
    
    # Search buttons
    search_btn = ttk.Button(
        search_controls,
        text="🔍 Search",
        command=panel._do_search,
        width=12
    )
    add_tooltip(search_btn, "Search for datasets matching your query")
    search_btn.pack(side="left", padx=(0, 2))
    
    clear_btn = ttk.Button(
        search_controls,
        text="🔄 Clear",
        command=panel._clear_search,
        width=10
    )
    add_tooltip(clear_btn, "Clear search and show all datasets")
    clear_btn.pack(side="left")
    
    panel.search_status_label = ttk.Label(
        panel.search_frame,
        text="Enter a search query or leave blank to browse popular datasets",
        font=("", 8, "italic"),
        foreground="gray"
    )
    panel.search_status_label.pack(anchor="w", pady=(0, 2))


def _build_results_frame(panel):
    """Build the results treeview frame."""
    results_frame = ttk.LabelFrame(panel.frame, text="📋 Results", padding=5)
    results_frame.pack(fill="both", expand=True, pady=(0, 5))
    
    # Create treeview for results
    tree_scroll = ttk.Scrollbar(results_frame)
    tree_scroll.pack(side="right", fill="y")
    
    panel.results_tree = ttk.Treeview(
        results_frame,
        columns=("downloads", "size", "rows", "blocks", "description"),
        yscrollcommand=tree_scroll.set,
        height=8,
        selectmode="browse"
    )
    add_tooltip(panel.results_tree, "Search results. Click columns to sort. Select a dataset to download or view details.")
    panel.results_tree.pack(side="left", fill="both", expand=True)
    tree_scroll.config(command=panel.results_tree.yview)
    
    # Track sort state
    panel._sort_column = None
    panel._sort_reverse = False
    
    # Configure columns with sortable headers
    panel.results_tree.heading("#0", text="Dataset ▼", command=lambda: panel._sort_by_column("#0"))
    panel.results_tree.heading("downloads", text="Downloads", command=lambda: panel._sort_by_column("downloads"))
    panel.results_tree.heading("size", text="Size", command=lambda: panel._sort_by_column("size"))
    panel.results_tree.heading("rows", text="Rows", command=lambda: panel._sort_by_column("rows"))
    panel.results_tree.heading("blocks", text="Blocks", command=lambda: panel._sort_by_column("blocks"))
    panel.results_tree.heading("description", text="Description", command=lambda: panel._sort_by_column("description"))
    
    panel.results_tree.column("#0", width=200, minwidth=150)
    panel.results_tree.column("downloads", width=90, minwidth=70)
    panel.results_tree.column("size", width=90, minwidth=70)
    panel.results_tree.column("rows", width=110, minwidth=80)
    panel.results_tree.column("blocks", width=80, minwidth=60)
    panel.results_tree.column("description", width=300, minwidth=200)


def _build_action_buttons(panel):
    """Build the result action buttons."""
    actions_frame = ttk.Frame(panel.frame)
    actions_frame.pack(fill="x", pady=(0, 5))
    
    download_btn = ttk.Button(
        actions_frame,
        text="📥 Download Selected",
        command=panel._download_selected,
        width=20
    )
    add_tooltip(download_btn, "Download selected dataset with streaming (default 100k samples)")
    download_btn.pack(side="left", padx=(0, 5))
    
    favorite_btn = ttk.Button(
        actions_frame,
        text="⭐ Add to Favorites",
        command=panel._favorite_selected,
        width=20
    )
    add_tooltip(favorite_btn, "Add or remove selected dataset from favorites")
    favorite_btn.pack(side="left", padx=(0, 5))
    
    details_btn = ttk.Button(
        actions_frame,
        text="ℹ️ View Details",
        command=panel._view_dataset_details,
        width=15
    )
    add_tooltip(details_btn, "View detailed information about selected dataset")
    details_btn.pack(side="left", padx=(0, 5))
    
    panel.cancel_btn = ttk.Button(
        actions_frame,
        text="❌ Cancel Download",
        command=panel._cancel_download,
        state="disabled",
        width=18
    )
    add_tooltip(panel.cancel_btn, "Cancel current download operation")
    panel.cancel_btn.pack(side="right")


def _build_location_frame(panel):
    """Build the download location selector."""
    location_frame = ttk.LabelFrame(panel.frame, text="📁 Download Location", padding=5)
    location_frame.pack(fill="x", pady=(0, 5))
    
    location_controls = ttk.Frame(location_frame)
    location_controls.pack(fill="x")
    
    ttk.Label(location_controls, text="Save to:").pack(side="left", padx=(0, 5))
    
    panel.download_location = tk.StringVar(value="training_datasets")
    location_entry = ttk.Entry(
        location_controls,
        textvariable=panel.download_location,
        width=50
    )
    add_tooltip(location_entry, "Directory where downloaded datasets will be saved")
    location_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
    
    browse_btn = ttk.Button(
        location_controls,
        text="📁 Browse",
        command=panel._browse_location,
        width=10
    )
    add_tooltip(browse_btn, "Browse for dataset download location")
    browse_btn.pack(side="left")


def _build_output_frame(panel):
    """Build the output text area."""
    # Output text area - skip if output_parent already has the output widget
    # (when shared output panel is used at top of tab)
    if panel._output_parent is not None:
        # Output widget is already created and managed by parent
        # Just find and use it
        for child in panel._output_parent.winfo_children():
            if isinstance(child, scrolledtext.ScrolledText):
                panel.output_text = child
                return
        
        # If we didn't find it, create it in the output_parent
        panel.output_text = scrolledtext.ScrolledText(
            panel._output_parent,
            wrap="word",
            height=10,
            font=("Consolas", 9)
        )
        panel.output_text.pack(fill="both", expand=True)
    else:
        # Create output in this panel's frame
        output_frame = ttk.LabelFrame(panel.frame, text="📋 Output", padding=5)
        output_frame.pack(fill="both", expand=True, pady=(5, 0))
        
        panel.output_text = scrolledtext.ScrolledText(
            output_frame,
            wrap="word",
            height=10,
            font=("Consolas", 9)
        )
        panel.output_text.pack(fill="both", expand=True)


def browse_location(panel):
    """Browse for download location."""
    directory = filedialog.askdirectory(
        title="Select Download Location",
        initialdir=panel.download_location.get()
    )
    
    if directory:
        panel.download_location.set(directory)
        panel.log(f"📁 Download location set to: {directory}")
