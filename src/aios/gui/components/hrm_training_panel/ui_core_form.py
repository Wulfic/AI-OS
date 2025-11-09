"""Core form UI builders for HRM Training Panel.

Handles dataset selection, batch size, steps, brain name, and related UI components.
"""

from __future__ import annotations
import os
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .panel_main import HRMTrainingPanel


def build_core_form(panel: HRMTrainingPanel, parent: any) -> None:  # type: ignore[valid-type]
    """Build the core training configuration form.
    
    Args:
        panel: The HRMTrainingPanel instance
        parent: Parent widget (typically a Frame)
    """
    from .helpers import log
    
    form = ttk.Frame(parent)
    form.pack(fill="x")
    panel._form_row = 0
    
    def _row(label: str, widget: any, browse_for: str | None = None) -> None:  # type: ignore[valid-type]
        r = panel._form_row
        ttk.Label(form, text=label, anchor="e").grid(row=r, column=0, sticky="e", padx=(0, 6), pady=2)
        widget.grid(row=r, column=1, sticky="we", pady=2)
        if browse_for:
            def _browse_cmd(kind: str = browse_for, target_widget: any = widget):  # type: ignore[valid-type]
                try:
                    from tkinter import filedialog
                    path = ""
                    if kind == "file":
                        path = filedialog.askopenfilename(initialdir=panel._project_root)
                    elif kind == "dir":
                        path = filedialog.askdirectory(initialdir=panel._project_root)
                    elif kind == "dataset":
                        path = filedialog.askdirectory(
                            initialdir=panel._project_root,
                            title="Select dataset directory (or cancel to pick a file)"
                        )
                        if not path:
                            path = filedialog.askopenfilename(
                                initialdir=panel._project_root,
                                title="Select dataset file",
                                filetypes=[
                                    ("Text files", "*.txt"),
                                    ("CSV files", "*.csv"),
                                    ("JSON files", "*.json *.jsonl"),
                                    ("All files", "*.*")
                                ]
                            )
                    if path:
                        try:
                            target_widget.delete(0, "end")
                            target_widget.insert(0, path)
                        except Exception:
                            pass
                except Exception:
                    pass
            ttk.Button(form, text="Browse", command=_browse_cmd).grid(row=r, column=2, sticky="w", padx=(6,0))
        form.columnconfigure(1, weight=1)
        panel._form_row += 1
    
    # ROW 1: Dataset selection with Select and Browse buttons
    dataset_frame = ttk.Frame(form)
    dataset_frame.grid(row=panel._form_row, column=1, sticky="we", pady=2)
    dataset_entry = ttk.Entry(dataset_frame, textvariable=panel.dataset_var, width=60)
    dataset_entry.pack(side="left", fill="x", expand=True)
    
    ttk.Label(form, text="Dataset file/dir:", anchor="e").grid(row=panel._form_row, column=0, sticky="e", padx=(0, 6), pady=2)
    
    # Add both Select and Browse buttons
    btn_container = ttk.Frame(form)
    btn_container.grid(row=panel._form_row, column=2, sticky="w", padx=(6,0))
    select_btn = ttk.Button(btn_container, text="Select", command=lambda: show_dataset_selector(panel), width=7)
    select_btn.pack(side="left", padx=(0, 2))
    
    def _browse_dataset():
        try:
            from tkinter import filedialog
            path = filedialog.askdirectory(
                initialdir=panel._project_root,
                title="Select dataset directory (or cancel to pick a file)"
            )
            if not path:
                path = filedialog.askopenfilename(
                    initialdir=panel._project_root,
                    title="Select dataset file",
                    filetypes=[
                        ("Text files", "*.txt"),
                        ("CSV files", "*.csv"),
                        ("JSON files", "*.json *.jsonl"),
                        ("All files", "*.*")
                    ]
                )
            if path:
                panel.dataset_var.set(path)
        except Exception as e:
            log(panel, f"[hrm] Browse error: {e}")
    
    browse_btn = ttk.Button(btn_container, text="Browse", command=_browse_dataset, width=7)
    browse_btn.pack(side="left")
    
    form.columnconfigure(1, weight=1)
    panel._form_row += 1
    
    # ROW 2: Context length, Batch size, Steps (with Auto), Chunk size - all in one row
    params_frame = ttk.Frame(form)
    params_frame.grid(row=panel._form_row, column=1, columnspan=2, sticky="we", pady=2)
    
    # Context length
    ttk.Label(params_frame, text="Context:", anchor="e").pack(side="left", padx=(0, 2))
    context_entry = ttk.Entry(params_frame, textvariable=panel.max_seq_var, width=8)
    context_entry.pack(side="left", padx=(0, 10))
    
    # Batch size
    ttk.Label(params_frame, text="Batch:", anchor="e").pack(side="left", padx=(0, 2))
    batch_entry = ttk.Entry(params_frame, textvariable=panel.batch_var, width=6)
    batch_entry.pack(side="left", padx=(0, 8))
    
    # Gradient accumulation
    ttk.Label(params_frame, text="×").pack(side="left", padx=(0, 2))
    accum_combo = ttk.Combobox(params_frame, textvariable=panel.gradient_accumulation_var, width=4, state="readonly")
    accum_combo['values'] = ('1', '2', '4', '8', '16', '32')
    accum_combo.pack(side="left", padx=(0, 8))
    
    # Effective batch (read-only display)
    ttk.Label(params_frame, text="=").pack(side="left", padx=(0, 2))
    panel.effective_batch_entry = ttk.Entry(params_frame, width=6, state="readonly")
    panel.effective_batch_entry.pack(side="left", padx=(0, 10))
    
    # Steps with Auto button
    ttk.Label(params_frame, text="Steps:", anchor="e").pack(side="left", padx=(0, 2))
    steps_entry = ttk.Entry(params_frame, textvariable=panel.steps_var, width=8)
    steps_entry.pack(side="left")
    
    from .helpers import auto_calculate_steps
    auto_steps_btn = ttk.Button(params_frame, text="Auto", width=6, command=lambda: auto_calculate_steps(panel))
    auto_steps_btn.pack(side="left", padx=(2, 10))
    panel._auto_steps_btn = auto_steps_btn
    
    # Chunk size
    ttk.Label(params_frame, text="Chunk Size:", anchor="e").pack(side="left", padx=(0, 2))
    chunk_entry = ttk.Entry(params_frame, textvariable=panel.dataset_chunk_size_var, width=8)
    chunk_entry.pack(side="left")
    
    ttk.Label(form, text="Training params:", anchor="e").grid(row=panel._form_row, column=0, sticky="e", padx=(0, 6), pady=2)
    panel._form_row += 1
    
    # Add tooltips for all params
    try:
        from ..tooltips import add_tooltip
        add_tooltip(dataset_entry, "Dataset file or directory to feed into HRM training.")
        add_tooltip(select_btn, "Select from available datasets")
        add_tooltip(browse_btn, "Browse for dataset file or folder")
        add_tooltip(context_entry, "Maximum sequence length (context window) in tokens")
        add_tooltip(batch_entry, "Physical batch size: Number of samples per training step")
        add_tooltip(accum_combo, 
            "Gradient Accumulation: Accumulate gradients over N batches\n"
            "before updating weights.\n\n"
            "Benefits:\n"
            "• Fixes loss instability from small batches\n"
            "• No VRAM increase\n"
            "• Smoother training dynamics\n\n"
            "Effective Batch = Physical Batch × Accum Steps\n"
            "Example: batch=8, accum=4 → effective=32\n\n"
            "Recommended:\n"
            "• 1: No accumulation (default)\n"
            "• 4: Balanced (recommended)\n"
            "• 8-16: High stability (small batches)\n"
            "• 32: Maximum stability (batch=1-2)")
        add_tooltip(panel.effective_batch_entry, "Effective batch size = Physical batch × Gradient accumulation\nThis is the actual batch size the optimizer sees")
        add_tooltip(steps_entry, "Total training steps to run")
        add_tooltip(auto_steps_btn, "Automatically calculate optimal steps based on dataset size")
        add_tooltip(chunk_entry, 
            "Chunk Size: Samples processed in each training batch cycle.\n\n"
            "Controls memory usage during training:\n"
            "• 100: Low VRAM, many small chunks per block\n"
            "• 4000: Default balanced\n"
            "• 10000+: High VRAM, fewer chunks per block\n\n"
            "Note: Blocks (100k samples) are downloaded from HuggingFace,\n"
            "then split into chunks of this size for training.\n\n"
            "The Auto button will calculate optimal steps based on this chunk size.")
    except Exception:
        pass
    
    # ROW 3: Brain name and Dataset mode checkbox
    brain_mode_frame = ttk.Frame(form)
    brain_mode_frame.grid(row=panel._form_row, column=1, columnspan=2, sticky="we", pady=2)
    
    # Brain name (read-only - use Select Student button to change)
    ttk.Label(brain_mode_frame, text="Name:", anchor="e").pack(side="left", padx=(0, 2))
    brain_entry = ttk.Entry(brain_mode_frame, textvariable=panel.brain_name_var, width=20, state="readonly")
    brain_entry.pack(side="left", padx=(0, 15))
    
    # Dataset mode checkbox
    linear_check = ttk.Checkbutton(
        brain_mode_frame, 
        text="Linear progression (sequential order, enables position tracking for pause/resume)",
        variable=panel.linear_dataset_var
    )
    linear_check.pack(side="left")
    
    ttk.Label(form, text="Brain & mode:", anchor="e").grid(row=panel._form_row, column=0, sticky="e", padx=(0, 6), pady=2)
    panel._form_row += 1
    
    # Setup callback to update effective batch display
    def update_effective_batch_display(*args):
        try:
            batch = int(panel.batch_var.get() or 8)
            accum = int(panel.gradient_accumulation_var.get() or 1)
            effective = batch * accum
            panel.effective_batch_entry.configure(state="normal")
            panel.effective_batch_entry.delete(0, "end")
            panel.effective_batch_entry.insert(0, str(effective))
            panel.effective_batch_entry.configure(state="readonly")
        except Exception:
            pass
    
    panel.gradient_accumulation_var.trace_add("write", update_effective_batch_display)
    panel.batch_var.trace_add("write", update_effective_batch_display)
    
    # Initial update
    update_effective_batch_display()
    
    # Add tooltips for brain and mode
    try:
        from ..tooltips import add_tooltip
        add_tooltip(brain_entry, "Unique identifier for this brain. Used to organize training artifacts and goals.")
        add_tooltip(linear_check, 
            "Linear (default): Process data sequentially [0,1,2,...]. Tracks position for pause/resume.\n"
            "Shuffled: Randomize order each epoch. Better for generalization.\n\n"
            "Use linear for: sequential data (stories), curriculum learning, precise tracking.\n"
            "Use shuffled for: general training, classification, better model generalization.")
    except Exception:
        pass


def populate_dataset_dropdown(panel: HRMTrainingPanel) -> list:
    """Find available datasets in common locations and HuggingFace favorites.
    
    Args:
        panel: The HRMTrainingPanel instance
        
    Returns:
        list: List of dataset tuples (type, path, display_name, [fav_data])
    """
    from .helpers import log
    
    datasets = []
    
    # Check common dataset locations
    locations = [
        ("Z:/training_datasets", "Z: drive"),
        ("training_data", "Local training_data"),
        ("artifacts/datasets", "Local artifacts"),
    ]
    
    for base_path, label in locations:
        try:
            if os.path.isdir(base_path):
                for entry in sorted(os.listdir(base_path)):
                    full_path = os.path.join(base_path, entry)
                    if os.path.isdir(full_path):
                        # Check if it looks like a dataset directory
                        if any(os.path.exists(os.path.join(full_path, f)) 
                              for f in ["dataset_info.json", "data", "dataset.arrow"]):
                            datasets.append(("local", full_path, f"{entry} ({label})"))
        except Exception:
            pass
    
    # Add HuggingFace favorites
    try:
        from ..dataset_download_panel.favorites_manager import load_favorites
        favorites = load_favorites()
        for fav in favorites:
            hf_path = fav.get("path", "")
            if hf_path:
                display_name = f"🤗 {fav.get('full_name', fav.get('name', 'Unknown'))} (HuggingFace)"
                datasets.append(("huggingface", hf_path, display_name, fav))
    except Exception as e:
        log(panel, f"[hrm] Could not load HuggingFace favorites: {e}")
    
    return datasets


def show_dataset_selector(panel: HRMTrainingPanel) -> None:
    """Show a dialog to quickly select from available datasets.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    from .helpers import log
    
    try:
        datasets = populate_dataset_dropdown(panel)
        if not datasets:
            log(panel, "[hrm] No datasets found in common locations or HuggingFace favorites")
            return
        
        # Create selection dialog
        dialog = tk.Toplevel(panel)
        dialog.title("Select Dataset")
        dialog.grab_set()
        dialog.geometry("700x450")
        
        frame = ttk.Frame(dialog)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ttk.Label(
            frame, 
            text="Available datasets (local directories and HuggingFace favorites):"
        ).pack(anchor="w", pady=(0, 5))
        
        # Listbox with scrollbar
        list_frame = ttk.Frame(frame)
        list_frame.pack(fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        
        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, width=90, height=15)
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=listbox.yview)
        
        # Populate listbox
        dataset_info = []
        for item in datasets:
            if item[0] == "local":
                _, path, display_name = item
                listbox.insert("end", display_name)
                dataset_info.append(("local", path, None))
            elif item[0] == "huggingface":
                _, hf_path, display_name, fav_data = item
                listbox.insert("end", display_name)
                dataset_info.append(("huggingface", hf_path, fav_data))
        
        # Info label
        info_label = ttk.Label(
            frame,
            text="💡 Tip: HuggingFace datasets will be streamed during training",
            font=("", 8, "italic"),
            foreground="gray"
        )
        info_label.pack(anchor="w", pady=(5, 0))
        
        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=(10, 0))
        
        def _select():
            selection = listbox.curselection()
            if selection:
                dataset_type, dataset_path, dataset_data = dataset_info[selection[0]]
                
                if dataset_type == "local":
                    panel.dataset_var.set(dataset_path)
                    log(panel, f"[hrm] Selected local dataset: {dataset_path}")
                elif dataset_type == "huggingface":
                    hf_identifier = f"hf://{dataset_path}"
                    config = dataset_data.get("config") if dataset_data else None
                    split = dataset_data.get("split") if dataset_data else None
                    
                    if not config:
                        config = "default"
                    hf_identifier += f":{config}"
                    
                    if not split:
                        split = "train"
                    hf_identifier += f":{split}"
                    
                    panel.dataset_var.set(hf_identifier)
                    log(panel, f"[hrm] Selected HuggingFace dataset: {dataset_path}")
                    log(panel, f"[hrm] Using config='{config}', split='{split}'")
                    log(panel, f"[hrm] Dataset will be streamed from HuggingFace Hub")
                    
                    # Detect and display dataset size info
                    try:
                        from .helpers import detect_and_display_dataset_info
                        panel.after(100, lambda: detect_and_display_dataset_info(panel))
                    except Exception as e:
                        log(panel, f"[hrm] Could not auto-detect dataset size: {e}")
                
            dialog.destroy()
        
        def _browse_instead():
            dialog.destroy()
            try:
                from tkinter import filedialog
                path = filedialog.askdirectory(
                    initialdir=panel._project_root,
                    title="Select dataset directory"
                )
                if path:
                    panel.dataset_var.set(path)
            except Exception as e:
                log(panel, f"[hrm] Browse error: {e}")
        
        ttk.Button(btn_frame, text="Select", command=_select).pack(side="left", padx=(0, 5))
        ttk.Button(btn_frame, text="Browse...", command=_browse_instead).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side="right")
        
        # Double-click to select
        listbox.bind("<Double-Button-1>", lambda e: _select())
        
    except Exception as e:
        log(panel, f"[hrm] Dataset selector error: {e}")
