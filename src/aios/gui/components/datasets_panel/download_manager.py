"""Download manager for datasets.

Provides threaded download functionality with progress tracking.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
import urllib.request as _urlreq
from typing import Any


def start_dataset_download(
    *,
    name: str,
    url: str,
    dest: str,
    progress_widget: Any,
    progress_status_var: Any,
    btn_download: Any,
    btn_cancel: Any,
    download_cancel_event: threading.Event,
    dataset_path_var: Any,
    log_callback: Any,
    append_out_callback: Any,
) -> threading.Thread:
    """Start a dataset download in a background thread.
    
    Args:
        name: Dataset name for logging
        url: URL to download from
        dest: Destination file path
        progress_widget: Tkinter Progressbar widget
        progress_status_var: Tkinter StringVar for status text
        btn_download: Download button to disable/enable
        btn_cancel: Cancel button to enable/disable
        download_cancel_event: Threading event for cancellation
        dataset_path_var: Tkinter StringVar to update with final path
        log_callback: Callback for logging messages
        append_out_callback: Callback for debug output
        
    Returns:
        The started download thread
    """
    log_callback(f"Downloading: {name}\n{url}\n→ {dest}")
    
    try:
        btn_download.configure(state="disabled")
        btn_cancel.configure(state="normal")
    except Exception:
        pass

    def _dl():
        sha = hashlib.sha256()
        total = None
        read_bytes = 0
        start_ts = time.time()
        
        # Helper to update UI from background thread
        def _update_progress_ui(mode=None, value=None, status=None, start_indeterminate=False):
            def _do_update():
                try:
                    if mode is not None:
                        progress_widget.configure(mode=mode)
                        if mode == "determinate" and value is not None:
                            progress_widget.configure(maximum=100, value=value)
                    if start_indeterminate:
                        progress_widget.start(50)
                    if value is not None and mode != "indeterminate":
                        progress_widget.configure(value=value)
                    if status is not None:
                        progress_status_var.set(status)
                except Exception:
                    pass
            try:
                progress_widget.after(0, _do_update)
            except Exception:
                _do_update()
        
        try:
            req = _urlreq.Request(url)
            with _urlreq.urlopen(req, timeout=30) as r, open(dest + ".part", "wb") as f:
                cl = r.headers.get("Content-Length")
                if cl and cl.isdigit():
                    total = int(cl)
                    _update_progress_ui(mode="determinate", value=0, status="Downloading...")
                    log_callback(f"Download started: {read_bytes}/{total} bytes")
                else:
                    _update_progress_ui(mode="indeterminate", status="Downloading...", start_indeterminate=True)
                    log_callback("Download started (size unknown)")
                
                chunk = 1024 * 256
                last_log_bytes = 0
                while True:
                    if download_cancel_event.is_set():
                        raise RuntimeError("cancelled")
                    buf = r.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    sha.update(buf)
                    read_bytes += len(buf)
                    
                    # Log progress every 10MB
                    if read_bytes - last_log_bytes >= 10 * 1024 * 1024:
                        if total:
                            log_callback(f"Progress: {read_bytes / (1024*1024):.1f} MB / {total / (1024*1024):.1f} MB")
                        else:
                            log_callback(f"Progress: {read_bytes / (1024*1024):.1f} MB downloaded")
                        last_log_bytes = read_bytes
                    
                    if total:
                        pct = max(0, min(100, int(read_bytes * 100 / max(1, total))))
                        # ETA calculation
                        elapsed = max(0.001, time.time() - start_ts)
                        rate = read_bytes / elapsed  # bytes/sec
                        remain = max(0, (total - read_bytes))
                        eta = remain / rate if rate > 0 else 0
                        _update_progress_ui(value=pct, status=f"{pct}%  ETA {int(eta)}s")
            os.replace(dest + ".part", dest)
        except Exception as e:
            log_callback(f"Download error: {e}")
            append_out_callback(f"[error] Dataset download error: {e}")
            try:
                if os.path.exists(dest + ".part"):
                    os.remove(dest + ".part")
            except Exception:
                pass
            
            def _cleanup_ui_error():
                try:
                    progress_widget.stop()
                    progress_widget.configure(mode="determinate", value=0)
                    progress_status_var.set("Idle")
                    btn_download.configure(state="normal")
                    btn_cancel.configure(state="disabled")
                except Exception:
                    pass
            
            try:
                progress_widget.after(0, _cleanup_ui_error)
            except Exception:
                _cleanup_ui_error()
            return
        
        # Success path
        h = sha.hexdigest()
        log_callback(f"Download complete!\nsha256={h}\nFile: {dest}")
        
        def _cleanup_ui_success():
            try:
                dataset_path_var.set(dest)
            except Exception:
                pass
            try:
                progress_widget.stop()
                progress_widget.configure(mode="determinate", value=0)
                progress_status_var.set("Complete")
                btn_download.configure(state="normal")
                btn_cancel.configure(state="disabled")
            except Exception:
                pass
        
        try:
            progress_widget.after(0, _cleanup_ui_success)
        except Exception:
            _cleanup_ui_success()

    download_thread = threading.Thread(target=_dl, daemon=True)
    download_thread.start()
    return download_thread
