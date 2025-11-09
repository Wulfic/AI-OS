"""Event handlers for the datasets panel.

Provides UI event handling methods for:
- Browsing for local dataset files
- Loading list of known datasets
- Using a selected dataset
- Cancelling downloads
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

try:  # pragma: no cover - environment dependent
    from tkinter import filedialog  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    filedialog = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import threading


def handle_browse_dataset(dataset_path_var: Any) -> None:
    """Handle browse button click to select a dataset file.
    
    Args:
        dataset_path_var: Tkinter StringVar to update with selected path
    """
    if filedialog is None:
        return
    try:
        path = filedialog.askopenfilename(title="Select dataset file")
    except Exception:
        path = ""
    if path:
        try:
            dataset_path_var.set(path)
        except Exception:
            pass


def handle_load_known_datasets(
    *,
    known_ds_combo: Any,
    known_ds_cache: list[dict] | None,
    fetch_datasets_callback: Any,
    log_callback: Any,
    append_out_callback: Any,
) -> tuple[list[dict] | None, list[dict] | None]:
    """Handle loading the list of known datasets.
    
    Args:
        known_ds_combo: The combobox widget to populate
        known_ds_cache: Cached dataset list (None if not cached)
        fetch_datasets_callback: Callback to fetch datasets if not cached
        log_callback: Callback for logging messages
        append_out_callback: Callback for debug output
        
    Returns:
        Tuple of (known_ds_items, known_ds_cache)
    """
    # Use cached list if present; allow manual refresh in future
    if known_ds_cache is not None:
        items = list(known_ds_cache)
    else:
        items = []
        try:
            items = fetch_datasets_callback()
        except Exception as e:
            log_callback(f"[datasets] Fetch error: {e}")
            append_out_callback(f"[error] Dataset fetch error: {e}")
            items = []
        
        if not items:
            # curated fallback
            try:
                from aios.data.datasets import known_datasets as _known

                items = [
                    {"name": kd.name, "url": kd.url, "size_bytes": int(kd.approx_size_gb * (1024 ** 3))}
                    for kd in _known(max_size_gb=15)
                ]
            except Exception:
                items = []
        
        known_ds_cache = list(items)
    
    names: list[str] = []
    for it in items:
        nm = it.get("name") or ""
        sz = it.get("size_bytes")
        if isinstance(sz, int) and sz > 0:
            gb = sz / (1024**3)
            nm = f"{nm}  ({gb:.2f} GB)"
        names.append(nm)
    
    try:
        if hasattr(known_ds_combo, "configure"):
            known_ds_combo.configure(values=names)  # type: ignore[call-arg]
    except Exception:
        pass
    
    return items, known_ds_cache


def handle_use_known_dataset(
    *,
    known_ds_var: Any,
    known_ds_items: list[dict] | None,
    dataset_path_var: Any,
    log_callback: Any,
) -> None:
    """Handle using a selected known dataset.
    
    Args:
        known_ds_var: Tkinter StringVar with selected dataset name
        known_ds_items: List of dataset dictionaries
        dataset_path_var: Tkinter StringVar to update with path hint
        log_callback: Callback for logging messages
    """
    sel = known_ds_var.get().strip()
    name = sel
    if not name:
        return
    
    url = ""
    try:
        for it in known_ds_items or []:
            base = it.get("name", "")
            if base == name or sel.startswith(base):
                url = it.get("url", "")
                break
    except Exception:
        url = ""
    
    try:
        base = os.path.expanduser("~/.local/share/aios/datasets")
        fname = os.path.basename(url) if url else "data.jsonl"
        hint = os.path.join(base, name, fname)
        dataset_path_var.set(hint)
    except Exception:
        pass
    
    if url:
        log_callback(f"Selected: {name}\nURL: {url}")


def handle_cancel_download(
    *,
    download_cancel: threading.Event | None,
    btn_cancel: Any,
    log_callback: Any,
) -> None:
    """Handle cancelling an active download.
    
    Args:
        download_cancel: Threading event to signal cancellation
        btn_cancel: Cancel button widget to disable
        log_callback: Callback for logging messages
    """
    if download_cancel and not download_cancel.is_set():
        download_cancel.set()
        log_callback("Cancelling download...")
    try:
        btn_cancel.configure(state="disabled")
    except Exception:
        pass
