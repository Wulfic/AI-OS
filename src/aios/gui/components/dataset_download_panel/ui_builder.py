"""
UI Builder for Dataset Download Panel

Functions for building the user interface components.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog

# Import safe variable wrappers
from ...utils import safe_variables

# Import progress formatting
from .download_progress import SpeedUnit

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

    # Inner content frame with grid-based layout so lower controls stay visible
    panel.content_frame = ttk.Frame(panel.frame)
    panel.content_frame.pack(fill="both", expand=True)
    panel.content_frame.grid_columnconfigure(0, weight=1)

    current_row = 0
    info_label = ttk.Label(
        panel.content_frame,
        text="Search HuggingFace for datasets, favorite them, and download for AI-OS training",
        font=("", 9)
    )
    info_label.grid(row=current_row, column=0, sticky="w", pady=(0, 5))

    current_row += 1
    control_frame = _build_control_bar(panel, panel.content_frame)
    control_frame.grid(row=current_row, column=0, sticky="ew", pady=(0, 5))

    current_row += 1
    search_frame = _build_search_frame(panel, panel.content_frame)
    search_frame.grid(row=current_row, column=0, sticky="ew", pady=(0, 5))

    current_row += 1
    results_frame = _build_results_frame(panel, panel.content_frame)
    panel.content_frame.grid_rowconfigure(current_row, weight=1)
    results_frame.grid(row=current_row, column=0, sticky="nsew", pady=(0, 5))

    current_row += 1
    actions_frame = _build_action_buttons(panel, panel.content_frame)
    actions_frame.grid(row=current_row, column=0, sticky="ew", pady=(0, 4))

    # Location UI moved to Settings tab (Phase 3.3)
    # current_row += 1
    # location_frame = _build_location_frame(panel, panel.content_frame)
    # location_frame.grid(row=current_row, column=0, sticky="ew", pady=(0, 5))

    current_row += 1
    progress_frame = _build_progress_frame(panel, panel.content_frame)
    progress_frame.grid(row=current_row, column=0, sticky="ew", pady=(2, 0))

    # Build output text area unless the panel is configured to use the shared output
    if not getattr(panel, "_use_shared_output", False):
        _build_output_frame(panel, panel.content_frame, current_row + 1)


def _build_control_bar(panel, parent):
    """Build the top control bar with favorites and HF auth buttons."""
    control_frame = ttk.Frame(parent)
    
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

    return control_frame


def _on_modality_changed(panel):
    """Handle modality dropdown selection change."""
    modality = panel.modality_var.get()
    
    # Show warning for non-text modalities since only text is supported during training
    if modality != "Text" and modality != "All":
        from tkinter import messagebox
        messagebox.showwarning(
            "Modality Warning",
            f"Note: Only 'Text' datasets are currently supported for model training.\n\n"
            f"You selected '{modality}' which may not be usable for training.\n"
            f"You can still browse and download these datasets for inspection."
        )
    
    # Trigger a new search with the updated modality filter
    panel._do_search()


def _build_search_frame(panel, parent):
    """Build the search frame with query and filters."""
    panel.search_frame = ttk.LabelFrame(parent, text="🔍 Search HuggingFace Datasets", padding=5)
    
    search_controls = ttk.Frame(panel.search_frame)
    search_controls.pack(fill="x", pady=(0, 5))
    
    # Search query field
    ttk.Label(search_controls, text="Search:").pack(side="left", padx=(0, 5))
    
    panel.search_var = safe_variables.StringVar()
    search_entry = ttk.Entry(search_controls, textvariable=panel.search_var, width=30)
    add_tooltip(search_entry, "Search HuggingFace datasets by name, topic, or task")
    search_entry.pack(side="left", padx=(0, 8))
    search_entry.bind("<Return>", lambda e: panel._do_search())
    
    # Modality filter dropdown
    ttk.Label(search_controls, text="Type:").pack(side="left", padx=(0, 5))
    
    # Define modalities based on HuggingFace Hub modality options
    # Text is default since only text is currently supported during training
    panel.modality_var = safe_variables.StringVar(value="Text")
    panel._modality_options = ["All", "Text", "Audio", "Document", "Geospatial", "Image", "Tabular", "Time-series", "Video", "3D"]
    modality_combo = ttk.Combobox(
        search_controls,
        textvariable=panel.modality_var,
        values=panel._modality_options,
        state="readonly",
        width=10
    )
    add_tooltip(modality_combo, "Filter by data type. Text is default (only text is supported for training)")
    modality_combo.pack(side="left", padx=(0, 8))
    modality_combo.bind("<<ComboboxSelected>>", lambda e: _on_modality_changed(panel))
    
    # Max size filter field
    ttk.Label(search_controls, text="Max Size:").pack(side="left", padx=(0, 5))
    
    panel.max_size_var = safe_variables.StringVar(value="")  # Empty = no limit
    max_size_entry = ttk.Entry(search_controls, textvariable=panel.max_size_var, width=8)
    add_tooltip(max_size_entry, "Maximum dataset size in GB (leave empty for no limit)")
    max_size_entry.pack(side="left", padx=(0, 2))
    max_size_entry.bind("<Return>", lambda e: panel._do_search())
    
    # Size unit dropdown
    panel.size_unit_var = safe_variables.StringVar(value="GB")
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

    return panel.search_frame


def _build_results_frame(panel, parent):
    """Build the results treeview frame."""
    results_frame = ttk.LabelFrame(parent, text="📋 Results", padding=5)
    
    # Create treeview for results
    tree_scroll = ttk.Scrollbar(results_frame)
    tree_scroll.pack(side="right", fill="y")
    
    panel.results_tree = ttk.Treeview(
        results_frame,
        columns=("downloads", "size", "rows", "blocks", "description"),
        yscrollcommand=tree_scroll.set,
        height=6,
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

    return results_frame


def _build_action_buttons(panel, parent):
    """Build the result action buttons."""
    actions_frame = ttk.Frame(parent)
    
    download_btn = ttk.Button(
        actions_frame,
        text="📥 Download Selected",
        command=panel._download_selected,
        width=20
    )
    add_tooltip(download_btn, "Download selected dataset with streaming (downloads entire dataset)")
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
        width=8
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

    return actions_frame


def _build_progress_frame(panel, parent):
    """Build the download progress frame with status, percentage, speed, and unit toggle."""
    progress_frame = ttk.Frame(parent)
    
    # Left side: Status label and progress info
    left_frame = ttk.Frame(progress_frame)
    left_frame.pack(side="left", fill="x", expand=True)
    
    # Status label (e.g., "Ready" or "Downloading dataset...")
    panel.status_label = ttk.Label(left_frame, text="Ready", font=("", 9, "italic"))
    panel.status_label.pack(side="left")
    
    # Progress separator (hidden when not downloading)
    panel.progress_separator = ttk.Label(left_frame, text=" | ", font=("", 9))
    panel.progress_separator.pack(side="left")
    panel.progress_separator.pack_forget()  # Hidden initially
    
    # Progress percentage label
    panel.progress_pct_label = ttk.Label(left_frame, text="0%", font=("", 9, "bold"))
    panel.progress_pct_label.pack(side="left")
    panel.progress_pct_label.pack_forget()  # Hidden initially
    
    # Speed label
    panel.speed_label = ttk.Label(left_frame, text="", font=("", 9))
    panel.speed_label.pack(side="left", padx=(8, 0))
    panel.speed_label.pack_forget()  # Hidden initially
    
    # ETA label
    panel.eta_label = ttk.Label(left_frame, text="", font=("", 9, "italic"), foreground="gray")
    panel.eta_label.pack(side="left", padx=(8, 0))
    panel.eta_label.pack_forget()  # Hidden initially
    
    # Right side: Speed unit toggle
    right_frame = ttk.Frame(progress_frame)
    right_frame.pack(side="right")
    
    # Speed unit toggle button
    panel.speed_unit = SpeedUnit.BYTES  # Default to MB/s (more intuitive for downloads)
    panel.speed_unit_btn = ttk.Button(
        right_frame,
        text="MB/s",
        command=lambda: _toggle_speed_unit(panel),
        width=6
    )
    add_tooltip(panel.speed_unit_btn, "Toggle speed display between MB/s (Megabytes) and Mbps (Megabits)")
    panel.speed_unit_btn.pack(side="right")
    
    return progress_frame


def _toggle_speed_unit(panel):
    """Toggle between bits and bytes speed display."""
    if panel.speed_unit == SpeedUnit.BYTES:
        panel.speed_unit = SpeedUnit.BITS
        panel.speed_unit_btn.config(text="Mbps")
    else:
        panel.speed_unit = SpeedUnit.BYTES
        panel.speed_unit_btn.config(text="MB/s")
    
    # Update display if currently showing speed
    if hasattr(panel, '_current_download_stats') and panel._current_download_stats:
        _update_progress_display(panel, panel._current_download_stats)


def _update_progress_display(panel, stats):
    """Update the progress display with current stats."""
    from .download_progress import format_speed, format_eta, format_size
    
    # Store for unit toggle updates
    panel._current_download_stats = stats
    
    # Show progress elements if hidden
    try:
        if not panel.progress_separator.winfo_ismapped():
            panel.progress_separator.pack(side="left")
        if not panel.progress_pct_label.winfo_ismapped():
            panel.progress_pct_label.pack(side="left")
        if not panel.speed_label.winfo_ismapped():
            panel.speed_label.pack(side="left", padx=(8, 0))
        if not panel.eta_label.winfo_ismapped():
            panel.eta_label.pack(side="left", padx=(8, 0))
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"Progress display pack error: {e}")
    
    # Update percentage (prefer blocks for accuracy)
    try:
        if stats.total_blocks > 0 and stats.blocks_completed >= 0:
            pct = (stats.blocks_completed / stats.total_blocks) * 100
            # Clamp to 100% to prevent showing >100%
            pct = min(pct, 100.0)
            panel.progress_pct_label.config(text=f"{pct:.1f}%")
        elif stats.total_bytes > 0:
            pct = (stats.bytes_downloaded / stats.total_bytes) * 100
            pct = min(pct, 100.0)
            panel.progress_pct_label.config(text=f"{pct:.1f}%")
        elif stats.total_samples > 0 and stats.samples_downloaded > 0:
            pct = (stats.samples_downloaded / stats.total_samples) * 100
            pct = min(pct, 100.0)
            panel.progress_pct_label.config(text=f"{pct:.1f}%")
        else:
            # Unknown total - show downloaded amount
            panel.progress_pct_label.config(text=format_size(stats.bytes_downloaded))
        
        # Update speed
        if stats.speed_bytes_per_sec > 0:
            speed_str = format_speed(stats.speed_bytes_per_sec, panel.speed_unit)
            panel.speed_label.config(text=f"⚡ {speed_str}")
        else:
            panel.speed_label.config(text="⚡ calculating...")
        
        # Update ETA
        if stats.eta_seconds > 0:
            panel.eta_label.config(text=f"ETA: {format_eta(stats.eta_seconds)}")
        else:
            panel.eta_label.config(text="")
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"Progress display update error: {e}")


def hide_progress_display(panel):
    """Hide the progress display elements."""
    try:
        panel.progress_separator.pack_forget()
        panel.progress_pct_label.pack_forget()
        panel.speed_label.pack_forget()
        panel.eta_label.pack_forget()
        panel._current_download_stats = None
    except Exception:
        pass


def _build_location_frame(panel, parent):
    """Build the download location selector."""
    location_frame = ttk.LabelFrame(parent, text="📁 Download Location", padding=5)
    
    location_controls = ttk.Frame(location_frame)
    location_controls.pack(fill="x")
    
    ttk.Label(location_controls, text="Save to:").pack(side="left", padx=(0, 5))
    
    panel.download_location = safe_variables.StringVar(value="training_datasets")
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

    return location_frame


def _build_output_frame(panel, parent, grid_row):
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
        output_frame = ttk.LabelFrame(parent, text="📋 Output", padding=5)
        output_frame.grid(row=grid_row, column=0, sticky="nsew", pady=(5, 0))
        parent.grid_rowconfigure(grid_row, weight=1)
        
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
