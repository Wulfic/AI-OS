"""Main datasets panel component.

Provides a Tkinter panel for managing datasets with:
- Local file browsing
- Known dataset selection
- Dataset downloading with progress tracking
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable, cast

try:  # pragma: no cover - environment dependent
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    tk = cast(Any, None)
    ttk = cast(Any, None)

from .dataset_fetchers import (
    fetch_from_aws_open_data_registry,
    fetch_from_awesomedata_nlp,
    fetch_from_github,
)
from .download_manager import start_dataset_download
from .event_handlers import (
    handle_browse_dataset,
    handle_cancel_download,
    handle_load_known_datasets,
    handle_use_known_dataset,
)
from .logging_utils import log_to_text_widget


class DatasetsPanel(ttk.LabelFrame):  # type: ignore[misc]
    """Tkinter panel for dataset management.
    
    Features:
    - Browse for local dataset files
    - Load and display known public datasets
    - Download datasets with progress tracking
    - Thread-safe logging to local output area
    """
    
    def __init__(
        self,
        parent: Any,
        *,
        dataset_path_var,
        append_out: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize the datasets panel.
        
        Args:
            parent: Parent Tkinter widget
            dataset_path_var: Tkinter StringVar for dataset file path
            append_out: Optional callback for debug output
        """
        super().__init__(parent, text="Datasets")
        self.pack(fill="x", padx=8, pady=(0, 8))
        self.dataset_path_var = dataset_path_var
        self._append_out = append_out or (lambda s: None)

        # Top row: dataset file chooser
        r1 = ttk.Frame(self)
        r1.pack(fill="x", padx=4, pady=(2, 2))
        ttk.Label(r1, text="Dataset file:").pack(side="left")
        ttk.Entry(r1, textvariable=self.dataset_path_var, width=60).pack(
            side="left", fill="x", expand=True, padx=(2, 6)
        )
        ttk.Button(r1, text="Browse", command=self.on_browse_dataset).pack(side="left")

        # Second row: known datasets + mini progress
        r2 = ttk.Frame(self)
        r2.pack(fill="x", padx=4, pady=(2, 2))
        ttk.Label(r2, text="Known datasets:").pack(side="left")
        self.known_ds_var = tk.StringVar(value="")
        try:
            self.known_ds_combo = ttk.Combobox(r2, textvariable=self.known_ds_var, width=40)
        except Exception:
            self.known_ds_combo = ttk.Entry(r2, textvariable=self.known_ds_var, width=40)  # type: ignore[assignment]
        self.known_ds_combo.pack(side="left", padx=(2, 6))
        ttk.Button(r2, text="Load List", command=self.load_known_datasets).pack(side="left")
        ttk.Button(r2, text="Use", command=self.use_known_dataset).pack(side="left", padx=(4, 0))
        self.btn_download = ttk.Button(r2, text="Download", command=self.download_known_dataset)
        self.btn_download.pack(side="left", padx=(4, 0))
        self.btn_cancel = ttk.Button(r2, text="Cancel", command=self.cancel_download)
        try:
            self.btn_cancel.configure(state="disabled")
        except Exception:
            pass
        self.btn_cancel.pack(side="left", padx=(4, 0))

        # Inline progress for downloads
        r3 = ttk.Frame(self)
        r3.pack(fill="x", padx=4, pady=(0, 2))
        self.progress = ttk.Progressbar(r3, orient="horizontal", mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.progress_status = tk.StringVar(value="Idle")
        ttk.Label(r3, textvariable=self.progress_status, width=18).pack(side="left")

        # Dataset-specific output area (compact)
        output_frame = ttk.LabelFrame(self, text="Dataset Activity")
        output_frame.pack(fill="both", expand=True, padx=4, pady=(4, 2))
        
        output_scroll = ttk.Scrollbar(output_frame)
        output_scroll.pack(side="right", fill="y")
        
        self._output_text = tk.Text(output_frame, height=6, wrap="word", yscrollcommand=output_scroll.set)
        self._output_text.configure(state="disabled")
        self._output_text.pack(side="left", fill="both", expand=True)
        
        output_scroll.config(command=self._output_text.yview)

        # internal state
        self._known_ds_items: list[dict] | None = []
        self._known_ds_cache: list[dict] | None = None
        self._download_thread: threading.Thread | None = None
        self._download_cancel: threading.Event | None = None
    
    def _log_dataset(self, message: str) -> None:
        """Log a dataset-specific message to the local output area.
        
        Thread-safe: schedules UI update on main thread if called from background thread.
        
        Args:
            message: The message to log
        """
        log_to_text_widget(self._output_text, message)

    def on_browse_dataset(self) -> None:
        """Handle browse button click to select a dataset file."""
        handle_browse_dataset(self.dataset_path_var)

    def load_known_datasets(self) -> None:
        """Load and display list of known public datasets."""
        def fetch_all():
            items_all: list[dict] = []
            try:
                items_all += fetch_from_github(max_items=120, max_size_gb=15)
            except Exception:
                pass
            try:
                items_all += fetch_from_awesomedata_nlp(max_items=120, max_size_gb=15)
            except Exception:
                pass
            try:
                items_all += fetch_from_aws_open_data_registry(max_items=60, max_size_gb=15)
            except Exception:
                pass
            
            # De-duplicate by URL
            seen: set[str] = set()
            items: list[dict] = []
            for it in items_all:
                url = str(it.get("url") or "").strip()
                if not url:
                    continue
                key = url.lower()
                if key in seen:
                    continue
                seen.add(key)
                items.append(it)
            return items
        
        self._known_ds_items, self._known_ds_cache = handle_load_known_datasets(
            known_ds_combo=self.known_ds_combo,
            known_ds_cache=self._known_ds_cache,
            fetch_datasets_callback=fetch_all,
            log_callback=self._log_dataset,
            append_out_callback=self._append_out,
        )

    def use_known_dataset(self) -> None:
        """Use the currently selected known dataset."""
        handle_use_known_dataset(
            known_ds_var=self.known_ds_var,
            known_ds_items=self._known_ds_items,
            dataset_path_var=self.dataset_path_var,
            log_callback=self._log_dataset,
        )

    def download_known_dataset(self) -> None:
        """Download the currently selected known dataset."""
        name = self.known_ds_var.get().strip()
        if not name:
            self._log_dataset("Please select a dataset from the dropdown first")
            return
        
        url = ""
        try:
            for it in self._known_ds_items or []:
                base = it.get("name", "")
                if base == name or name.startswith(base):
                    url = it.get("url", "")
                    name = base or name
                    break
        except Exception:
            url = ""
        
        if not url:
            self._log_dataset(f"No URL found for '{name}'")
            return
        
        base = os.path.expanduser("~/.local/share/aios/datasets")
        try:
            os.makedirs(os.path.join(base, name), exist_ok=True)
        except Exception:
            pass
        
        fname = os.path.basename(url) or "data"
        dest = os.path.join(base, name, fname)
        
        self._download_cancel = threading.Event()
        self._download_thread = start_dataset_download(
            name=name,
            url=url,
            dest=dest,
            progress_widget=self.progress,
            progress_status_var=self.progress_status,
            btn_download=self.btn_download,
            btn_cancel=self.btn_cancel,
            download_cancel_event=self._download_cancel,
            dataset_path_var=self.dataset_path_var,
            log_callback=self._log_dataset,
            append_out_callback=self._append_out,
        )

    def cancel_download(self) -> None:
        """Cancel the currently active download."""
        handle_cancel_download(
            download_cancel=self._download_cancel,
            btn_cancel=self.btn_cancel,
            log_callback=self._log_dataset,
        )
