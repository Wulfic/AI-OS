"""Core form UI builders for HRM Training Panel.

Handles dataset selection, batch size, steps, brain name, and related UI components.
"""

from __future__ import annotations
import os
import string
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Iterable, Tuple

if TYPE_CHECKING:
    from .panel_main import HRMTrainingPanel


def _iter_linux_mount_roots() -> Iterable[Path]:
    """Yield likely mount root directories for Linux and Unix environments."""

    # Incorporate standard mount roots plus user-specific GVFS mounts.
    candidates = {
        Path("/mnt"),
        Path("/media"),
        Path("/Volumes"),  # macOS
        Path.home() / "mnt",
        Path.home() / "media",
    }

    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        candidates.add(Path(runtime_dir) / "gvfs")

    uid = os.getuid() if hasattr(os, "getuid") else None
    if uid is not None:
        candidates.add(Path("/run/user") / str(uid) / "gvfs")

    for candidate in candidates:
        if candidate.is_dir():
            yield candidate


def get_mounted_volumes() -> list[Tuple[str, str]]:
    """Return a sorted list of accessible mounted volumes for dataset browsing."""

    volumes: list[Tuple[str, str]] = []
    seen_paths: set[str] = set()

    def _add_volume(label: str, path: Path) -> None:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path

        norm_path = str(resolved)
        if norm_path in seen_paths:
            return

        try:
            if resolved.is_dir() and os.access(norm_path, os.R_OK):
                volumes.append((label, norm_path))
                seen_paths.add(norm_path)
        except Exception:
            return

    if os.name == "nt":
        for letter in string.ascii_uppercase:
            root = Path(f"{letter}:/")
            if root.exists():
                _add_volume(f"Drive {letter}:", root)
        return sorted(volumes, key=lambda item: item[0].lower())

    # POSIX-style mounts, including GVFS for SMB shares.
    for root in _iter_linux_mount_roots():
        try:
            # Include the root itself when it is a mount point.
            _add_volume(f"{root.name or root}/", root)

            for entry in sorted(root.iterdir()):
                if not entry.is_dir():
                    continue

                label = entry.name
                raw = entry.name
                if raw.startswith("smb-share:"):
                    suffix = raw.split(":", 1)[-1]
                    parts = {}
                    for chunk in suffix.split(","):
                        if "=" in chunk:
                            key, value = chunk.split("=", 1)
                            parts[key] = value
                    server = parts.get("server", "smb")
                    share = parts.get("share", "share")
                    label = f"SMB {server}/{share}"
                elif raw.startswith("sftp:"):
                    label = f"SFTP {raw.split(':', 1)[-1]}"

                _add_volume(label, entry)
        except Exception:
            continue

    return sorted(volumes, key=lambda item: item[0].lower())


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

    # Track the last directory the user visited so subsequent dialogs reuse it.
    last_browse_dir = panel.dataset_var.get() or panel._project_root

    def _remember_last_dir(path: str) -> None:
        nonlocal last_browse_dir
        if not path:
            return
        candidate = path if os.path.isdir(path) else os.path.dirname(path)
        if candidate:
            last_browse_dir = candidate

    def _resolve_initial_dir(hint: str | None = None) -> str:
        if hint and os.path.isdir(hint):
            return hint

        current = panel.dataset_var.get()
        if current:
            if os.path.isdir(current):
                return current
            parent_dir = os.path.dirname(current)
            if parent_dir and os.path.isdir(parent_dir):
                return parent_dir

        if last_browse_dir and os.path.isdir(last_browse_dir):
            return last_browse_dir

        try:
            if panel._project_root and os.path.isdir(panel._project_root):
                return panel._project_root
        except Exception:
            pass

        return os.getcwd()

    def _choose_directory(initial: str | None = None) -> None:
        initial_dir = _resolve_initial_dir(initial)
        try:
            path = filedialog.askdirectory(
                initialdir=initial_dir,
                title="Select dataset directory",
            )
        except Exception as exc:  # pragma: no cover - tkinter message loop safety
            log(panel, f"[hrm] Directory picker error: {exc}")
            return

        if path:
            panel.dataset_var.set(path)
            _remember_last_dir(path)

    def _choose_file(initial: str | None = None) -> None:
        initial_dir = _resolve_initial_dir(initial)
        filetypes = [
            ("Text files", "*.txt"),
            ("CSV files", "*.csv"),
            ("JSON files", "*.json *.jsonl"),
            ("All files", "*.*"),
        ]

        try:
            path = filedialog.askopenfilename(
                initialdir=initial_dir,
                title="Select dataset file",
                filetypes=filetypes,
            )
        except Exception as exc:  # pragma: no cover - tkinter message loop safety
            log(panel, f"[hrm] File picker error: {exc}")
            return

        if path:
            panel.dataset_var.set(path)
            _remember_last_dir(path)

    def _show_drive_picker() -> None:
        volumes = get_mounted_volumes()
        if not volumes:
            messagebox.showinfo(
                "No drives detected",
                "No mounted drives were detected. Mount a drive or network share and try again.",
            )
            return

        dialog = tk.Toplevel(panel)
        dialog.title("Available Drives")
        dialog.geometry("960x720")
        dialog.transient(panel)
        dialog.grab_set()
        dialog.minsize(640, 480)
        dialog.resizable(True, True)
        
        # Apply theme styling
        from aios.gui.utils.theme_utils import apply_theme_to_toplevel
        apply_theme_to_toplevel(dialog)

        container = ttk.Frame(dialog, padding=12)
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text="Mounted drives and network shares detected on this system:",
        ).pack(anchor="w")

        listbox = tk.Listbox(container, activestyle="dotbox")
        listbox.pack(fill="both", expand=True, pady=8)

        for label, path in volumes:
            listbox.insert("end", f"{label} — {path}")

        if volumes:
            listbox.selection_set(0)

        hint_label = ttk.Label(
            container,
            text="Double-click a drive to browse inside it.",
            font=("", 8, "italic"),
            foreground="gray",
        )
        hint_label.pack(anchor="w", pady=(0, 6))

        button_row = ttk.Frame(container)
        button_row.pack(fill="x")

        def _use_selected(open_dialog: bool) -> None:
            selection = listbox.curselection()
            if not selection:
                return
            selected_path = volumes[selection[0]][1]
            if open_dialog:
                dialog.destroy()
                panel.after(10, lambda: _choose_directory(selected_path))
                return
            panel.dataset_var.set(selected_path)
            _remember_last_dir(selected_path)
            dialog.destroy()

        ttk.Button(button_row, text="Use Path", command=lambda: _use_selected(False)).pack(side="left")
        ttk.Button(button_row, text="Browse Inside", command=lambda: _use_selected(True)).pack(side="left", padx=6)
        ttk.Button(button_row, text="Cancel", command=dialog.destroy).pack(side="right")

        listbox.bind("<Double-Button-1>", lambda _event: _use_selected(True))

    # Expose helpers for other dialogs within the panel.
    panel._choose_dataset_directory = _choose_directory  # type: ignore[attr-defined]
    panel._choose_dataset_file = _choose_file  # type: ignore[attr-defined]
    panel._show_dataset_drive_picker = _show_drive_picker  # type: ignore[attr-defined]
    
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

    browse_menu = tk.Menu(btn_container, tearoff=False)
    browse_menu.add_command(label="Folder...", command=_choose_directory)
    browse_menu.add_command(label="Single File...", command=_choose_file)
    browse_menu.add_separator()
    browse_menu.add_command(label="Available Drives...", command=_show_drive_picker)

    def _open_browse_menu() -> None:
        try:
            browse_menu.tk_popup(
                browse_btn.winfo_rootx(),
                browse_btn.winfo_rooty() + browse_btn.winfo_height(),
            )
        finally:
            browse_menu.grab_release()

    browse_btn = ttk.Button(btn_container, text="Browse ▾", command=_open_browse_menu, width=9)
    browse_btn.pack(side="left")
    panel._dataset_browse_menu = browse_menu  # type: ignore[attr-defined]
    
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
        add_tooltip(browse_btn, "Browse for dataset folders, single files, or mounted drives")
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
    
    # Check configured dataset download location and fallback locations
    locations = []
    
    # Try to get configured download location from settings
    try:
        if hasattr(panel, '_app') and panel._app and hasattr(panel._app, 'settings_panel'):
            settings_panel = panel._app.settings_panel
            if settings_panel and hasattr(settings_panel, 'download_location_var'):
                configured_location = settings_panel.download_location_var.get().strip()
                if configured_location:
                    locations.append((configured_location, "Local"))
                    log(panel, f"[hrm] Using configured dataset location: {configured_location}")
    except Exception as e:
        log(panel, f"[hrm] Could not get configured download location: {e}")
    
    # Add default fallback locations
    locations.extend([
        ("training_datasets", "Local"),
        ("artifacts/datasets", "Local"),
    ])
    
    for base_path, label in locations:
        try:
            # Resolve relative paths to absolute paths
            if not os.path.isabs(base_path):
                # Try relative to project root first
                if hasattr(panel, '_project_root') and panel._project_root:
                    abs_base_path = os.path.join(panel._project_root, base_path)
                else:
                    abs_base_path = os.path.abspath(base_path)
            else:
                abs_base_path = base_path
            
            if os.path.isdir(abs_base_path):
                for entry in sorted(os.listdir(abs_base_path)):
                    full_path = os.path.join(abs_base_path, entry)
                    if os.path.isdir(full_path):
                        # Check if it looks like a dataset directory
                        # Include block_manifest.json for downloaded datasets
                        if any(os.path.exists(os.path.join(full_path, f)) 
                              for f in ["dataset_info.json", "data", "dataset.arrow", "block_manifest.json"]):
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
                display_name = f"🤗 {fav.get('full_name', fav.get('name', 'Unknown'))} (Streaming)"
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
        
        # Apply theme styling
        from aios.gui.utils.theme_utils import apply_theme_to_toplevel
        apply_theme_to_toplevel(dialog)
        
        frame = ttk.Frame(dialog)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ttk.Label(
            frame, 
            text="Available datasets (local directories and streaming favorites):"
        ).pack(anchor="w", pady=(0, 5))
        
        # Listbox with scrollbar
        list_frame = ttk.Frame(frame)
        list_frame.pack(fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        
        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, width=90, height=15)
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=listbox.yview)
        
        # Populate listbox with dividers
        dataset_info = []
        
        # Add local datasets with header
        local_items = [item for item in datasets if item[0] == "local"]
        streaming_items = [item for item in datasets if item[0] == "huggingface"]
        
        if local_items:
            listbox.insert("end", "─── LOCAL DATASETS ───────────────────────────────────────────────")
            dataset_info.append(("divider", None, None))
            for item in local_items:
                _, path, display_name = item
                listbox.insert("end", display_name)
                dataset_info.append(("local", path, None))
        
        if streaming_items:
            if local_items:  # Add spacing between sections
                listbox.insert("end", "")
                dataset_info.append(("divider", None, None))
            listbox.insert("end", "─── STREAMING DATASETS ───────────────────────────────────────────")
            dataset_info.append(("divider", None, None))
            for item in streaming_items:
                _, hf_path, display_name, fav_data = item
                listbox.insert("end", display_name)
                dataset_info.append(("huggingface", hf_path, fav_data))
        
        # Info label
        info_label = ttk.Label(
            frame,
            text="💡 Tip: Streaming datasets will be loaded from HuggingFace Hub during training",
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
                
                # Skip dividers
                if dataset_type == "divider":
                    return
                
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
            chooser = getattr(panel, "_choose_dataset_directory", None)
            if callable(chooser):
                panel.after(10, chooser)
            else:
                try:
                    path = filedialog.askdirectory(
                        initialdir=panel._project_root,
                        title="Select dataset directory"
                    )
                    if path:
                        panel.dataset_var.set(path)
                except Exception as e:
                    log(panel, f"[hrm] Browse error: {e}")

        def _browse_file():
            dialog.destroy()
            chooser = getattr(panel, "_choose_dataset_file", None)
            if callable(chooser):
                panel.after(10, chooser)
            else:
                try:
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

        def _open_drive_picker():
            dialog.destroy()
            picker = getattr(panel, "_show_dataset_drive_picker", None)
            if callable(picker):
                panel.after(10, picker)
            else:
                messagebox.showinfo(
                    "Drive picker unavailable",
                    "Drive discovery is not yet available in this view. Use the Browse button instead.",
                )
        
        ttk.Button(btn_frame, text="Select", command=_select).pack(side="left", padx=(0, 5))
        ttk.Button(btn_frame, text="Folder...", command=_browse_instead).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="File...", command=_browse_file).pack(side="left")
        ttk.Button(btn_frame, text="Drives...", command=_open_drive_picker).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side="right")
        
        # Double-click to select
        listbox.bind("<Double-Button-1>", lambda e: _select())
        
    except Exception as e:
        log(panel, f"[hrm] Dataset selector error: {e}")
