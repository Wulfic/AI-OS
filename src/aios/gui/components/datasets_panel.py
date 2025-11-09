from __future__ import annotations

import os
import threading
import urllib.request as _urlreq
from typing import Any, Callable, cast

try:  # pragma: no cover - environment dependent
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
    from tkinter import filedialog  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    tk = cast(Any, None)
    ttk = cast(Any, None)
    filedialog = cast(Any, None)


class DatasetsPanel(ttk.LabelFrame):  # type: ignore[misc]
    def __init__(
        self,
        parent: Any,
        *,
        dataset_path_var,
        append_out: Callable[[str], None] | None = None,
    ) -> None:
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
        """
        def _do_log():
            try:
                # Check if user is at bottom before inserting
                try:
                    yview = self._output_text.yview()
                    at_bottom = yview[1] >= 0.95  # Within ~5% of bottom
                except Exception:
                    at_bottom = True  # Default to scrolling if can't check
                
                self._output_text.configure(state="normal")
                if not message.endswith("\n"):
                    msg = message + "\n"
                else:
                    msg = message
                self._output_text.insert("end", msg)
                
                # Only scroll if user was at bottom
                if at_bottom:
                    self._output_text.see("end")
                
                # Limit output size
                lines = int(self._output_text.index('end-1c').split('.')[0])
                if lines > 200:
                    self._output_text.delete("1.0", f"{lines - 150}.0")
            except Exception:
                pass  # Silently fail if widget is destroyed
            finally:
                try:
                    self._output_text.configure(state="disabled")
                except Exception:
                    pass
        
        # Schedule on main thread to avoid Tkinter threading issues
        try:
            self._output_text.after(0, _do_log)
        except Exception:
            # Fallback: try direct call (might work if already on main thread)
            _do_log()

    # UI handlers
    def on_browse_dataset(self) -> None:
        if filedialog is None:
            return
        try:
            path = filedialog.askopenfilename(title="Select dataset file")
        except Exception:
            path = ""
        if path:
            try:
                self.dataset_path_var.set(path)
            except Exception:
                pass

    def load_known_datasets(self) -> None:
        # Use cached list if present; allow manual refresh in future
        if self._known_ds_cache is not None:
            items = list(self._known_ds_cache)
        else:
            items = []
            try:
                items_all: list[dict] = []
                try:
                    items_all += self._fetch_known_datasets_from_github(max_items=120, max_size_gb=15)
                except Exception:
                    pass
                try:
                    items_all += self._fetch_known_datasets_from_awesomedata_nlp(max_items=120, max_size_gb=15)
                except Exception:
                    pass
                try:
                    items_all += self._fetch_known_datasets_from_aws_open_data_registry(max_items=60, max_size_gb=15)
                except Exception:
                    pass
                # De-duplicate by URL
                seen: set[str] = set()
                for it in items_all:
                    url = str(it.get("url") or "").strip()
                    if not url:
                        continue
                    key = url.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    items.append(it)
                self._known_ds_cache = list(items)
            except Exception as e:
                self._log_dataset(f"[datasets] Fetch error: {e}")
                self._append_out(f"[error] Dataset fetch error: {e}")  # Also log to debug
                items = []
            if not items:
                # curated fallback
                try:
                    from aios.data.datasets import known_datasets as _known

                    items = [
                        {"name": kd.name, "url": kd.url, "size_bytes": int(kd.approx_size_gb * (1024 ** 3))}
                        for kd in _known(max_size_gb=15)
                    ]
                    self._known_ds_cache = list(items)
                except Exception:
                    items = []
        names: list[str] = []
        for it in items:
            nm = it.get("name") or ""
            sz = it.get("size_bytes")
            if isinstance(sz, int) and sz > 0:
                gb = sz / (1024**3)
                nm = f"{nm}  ({gb:.2f} GB)"
            names.append(nm)
        try:
            if hasattr(self.known_ds_combo, "configure"):
                self.known_ds_combo.configure(values=names)  # type: ignore[call-arg]
        except Exception:
            pass
        self._known_ds_items = items

    def use_known_dataset(self) -> None:
        sel = self.known_ds_var.get().strip()
        name = sel
        if not name:
            return
        url = ""
        try:
            for it in self._known_ds_items or []:
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
            self.dataset_path_var.set(hint)
        except Exception:
            pass
        if url:
            self._log_dataset(f"Selected: {name}\nURL: {url}")

    def download_known_dataset(self) -> None:
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
        self._log_dataset(f"Downloading: {name}\n{url}\n→ {dest}")
        try:
            self.btn_download.configure(state="disabled")
            self.btn_cancel.configure(state="normal")
        except Exception:
            pass
        self._download_cancel = threading.Event()

        def _dl():
            import hashlib
            import time

            sha = hashlib.sha256()
            total = None
            read_bytes = 0
            start_ts = time.time()
            
            # Helper to update UI from background thread
            def _update_progress_ui(mode=None, value=None, status=None, start_indeterminate=False):
                def _do_update():
                    try:
                        if mode is not None:
                            self.progress.configure(mode=mode)
                            if mode == "determinate" and value is not None:
                                self.progress.configure(maximum=100, value=value)
                        if start_indeterminate:
                            self.progress.start(50)
                        if value is not None and mode != "indeterminate":
                            self.progress.configure(value=value)
                        if status is not None:
                            self.progress_status.set(status)
                    except Exception:
                        pass
                try:
                    self.progress.after(0, _do_update)
                except Exception:
                    _do_update()
            
            try:
                req = _urlreq.Request(url)
                with _urlreq.urlopen(req, timeout=30) as r, open(dest + ".part", "wb") as f:
                    cl = r.headers.get("Content-Length")
                    if cl and cl.isdigit():
                        total = int(cl)
                        _update_progress_ui(mode="determinate", value=0, status="Downloading...")
                        self._log_dataset(f"Download started: {read_bytes}/{total} bytes")
                    else:
                        _update_progress_ui(mode="indeterminate", status="Downloading...", start_indeterminate=True)
                        self._log_dataset("Download started (size unknown)")
                    
                    chunk = 1024 * 256
                    last_log_bytes = 0
                    while True:
                        if self._download_cancel and self._download_cancel.is_set():
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
                                self._log_dataset(f"Progress: {read_bytes / (1024*1024):.1f} MB / {total / (1024*1024):.1f} MB")
                            else:
                                self._log_dataset(f"Progress: {read_bytes / (1024*1024):.1f} MB downloaded")
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
                self._log_dataset(f"Download error: {e}")
                self._append_out(f"[error] Dataset download error: {e}")
                try:
                    if os.path.exists(dest + ".part"):
                        os.remove(dest + ".part")
                except Exception:
                    pass
                
                def _cleanup_ui_error():
                    try:
                        self.progress.stop()
                        self.progress.configure(mode="determinate", value=0)
                        self.progress_status.set("Idle")
                        self.btn_download.configure(state="normal")
                        self.btn_cancel.configure(state="disabled")
                    except Exception:
                        pass
                
                try:
                    self.progress.after(0, _cleanup_ui_error)
                except Exception:
                    _cleanup_ui_error()
                return
            
            # Success path
            h = sha.hexdigest()
            self._log_dataset(f"Download complete!\nsha256={h}\nFile: {dest}")
            
            def _cleanup_ui_success():
                try:
                    self.dataset_path_var.set(dest)
                except Exception:
                    pass
                try:
                    self.progress.stop()
                    self.progress.configure(mode="determinate", value=0)
                    self.progress_status.set("Complete")
                    self.btn_download.configure(state="normal")
                    self.btn_cancel.configure(state="disabled")
                except Exception:
                    pass
            
            try:
                self.progress.after(0, _cleanup_ui_success)
            except Exception:
                _cleanup_ui_success()

        self._download_thread = threading.Thread(target=_dl, daemon=True)
        self._download_thread.start()

    def cancel_download(self) -> None:
        if self._download_cancel and not self._download_cancel.is_set():
            self._download_cancel.set()
            self._log_dataset("Cancelling download...")
        try:
            self.btn_cancel.configure(state="disabled")
        except Exception:
            pass

    # --- helper fetchers ---
    def _fetch_known_datasets_from_github(self, *, max_items: int = 200, max_size_gb: int = 15) -> list[dict]:
        RAW_URL = "https://raw.githubusercontent.com/niderhoff/nlp-datasets/master/README.md"
        resp = _urlreq.urlopen(RAW_URL, timeout=5)
        md = resp.read().decode("utf-8", errors="ignore")
        import re as _re

        cand: list[tuple[str, str]] = []
        for m in _re.finditer(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", md):
            title = m.group(1).strip()
            url = m.group(2).strip()
            if any(s in url.lower() for s in ["github.com/niderhoff/nlp-datasets", "#"]):
                continue
            if not _re.search(r"\.(jsonl?|csv|tsv|txt|zip|tgz|tar\.gz|gz|bz2|xz)$", url, _re.I):
                continue
            cand.append((title, url))
        items: list[dict] = []
        for title, url in cand[: max_items * 2]:
            ok = False
            size_ok = True
            size_val = None
            try:
                req = _urlreq.Request(url, method="HEAD")  # type: ignore[arg-type]
                with _urlreq.urlopen(req, timeout=4) as r2:  # type: ignore[call-arg]
                    code = getattr(r2, "status", 200)
                    ok = 200 <= int(code) < 400
                    cl = r2.headers.get("Content-Length")
                    if cl and cl.isdigit():
                        size = int(cl)
                        size_val = size
                        if size > max(1, max_size_gb) * (1024**3):
                            size_ok = False
                if not ok:
                    req2 = _urlreq.Request(url, headers={"Range": "bytes=0-0"})
                    with _urlreq.urlopen(req2, timeout=6) as r3:
                        code = getattr(r3, "status", 206)
                        ok = 200 <= int(code) < 400
            except Exception:
                ok = False
            if ok and size_ok:
                d: dict[str, Any] = {"name": title, "url": url}
                if isinstance(size_val, int):
                    d["size_bytes"] = int(size_val)
                items.append(d)
            if len(items) >= max_items:
                break
        return items

    def _fetch_known_datasets_from_awesomedata_nlp(self, *, max_items: int = 120, max_size_gb: int = 15) -> list[dict]:
        RAW_URL = "https://raw.githubusercontent.com/awesomedata/awesome-public-datasets/master/README.md"
        import re as _re

        resp = _urlreq.urlopen(RAW_URL, timeout=6)
        md = resp.read().decode("utf-8", errors="ignore")
        m = _re.search(r"^##\s*Natural\s+Language\b.*?(?=^##\s+|\Z)", md, _re.M | _re.S)
        block = m.group(0) if m else md
        cand: list[tuple[str, str]] = []
        for mm in _re.finditer(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", block):
            title = mm.group(1).strip()
            url = mm.group(2).strip()
            if "github.com/awesomedata/awesome-public-datasets" in url.lower():
                continue
            if not _re.search(r"\.(jsonl?|csv|tsv|txt|zip|tgz|tar\.gz|gz|bz2|xz)$", url, _re.I):
                continue
            cand.append((title, url))
        items: list[dict] = []
        for title, url in cand[: max_items * 2]:
            ok = False
            size_ok = True
            size_val = None
            try:
                req = _urlreq.Request(url, method="HEAD")  # type: ignore[arg-type]
                with _urlreq.urlopen(req, timeout=4) as r2:  # type: ignore[call-arg]
                    code = getattr(r2, "status", 200)
                    ok = 200 <= int(code) < 400
                    cl = r2.headers.get("Content-Length")
                    if cl and cl.isdigit():
                        size = int(cl)
                        size_val = size
                        if size > max(1, max_size_gb) * (1024**3):
                            size_ok = False
                if not ok:
                    req2 = _urlreq.Request(url, headers={"Range": "bytes=0-0"})
                    with _urlreq.urlopen(req2, timeout=6) as r3:
                        code = getattr(r3, "status", 206)
                        ok = 200 <= int(code) < 400
            except Exception:
                ok = False
            if ok and size_ok:
                d: dict[str, Any] = {"name": title, "url": url}
                if isinstance(size_val, int):
                    d["size_bytes"] = int(size_val)
                items.append(d)
            if len(items) >= max_items:
                break
        return items

    def _fetch_known_datasets_from_aws_open_data_registry(self, *, max_items: int = 60, max_size_gb: int = 15) -> list[dict]:
        RAW_URL = "https://raw.githubusercontent.com/awslabs/open-data-registry/main/datasets/"
        # list of YAML files index (simple scrape)
        index_url = RAW_URL + "README.md"
        md = _urlreq.urlopen(index_url, timeout=6).read().decode("utf-8", errors="ignore")
        import re as _re

        ymls = [m.group(1) for m in _re.finditer(r"\((datasets/[^)]+\.ya?ml)\)", md)]
        items: list[dict] = []
        for rel in ymls[: max_items * 2]:
            url = "https://raw.githubusercontent.com/awslabs/open-data-registry/main/" + rel
            try:
                txt = _urlreq.urlopen(url, timeout=6).read().decode("utf-8", errors="ignore")
                # crude scan for http/https URLs
                for mm in _re.finditer(r"https?://[^\s'\"]+", txt):
                    link = mm.group(0)
                    # best-effort HEAD validate
                    ok = False
                    size_ok = True
                    size_val = None
                    try:
                        req = _urlreq.Request(link, method="HEAD")  # type: ignore[arg-type]
                        with _urlreq.urlopen(req, timeout=4) as r2:  # type: ignore[call-arg]
                            code = getattr(r2, "status", 200)
                            ok = 200 <= int(code) < 400
                            cl = r2.headers.get("Content-Length")
                            if cl and cl.isdigit():
                                size = int(cl)
                                size_val = size
                                if size > max(1, max_size_gb) * (1024**3):
                                    size_ok = False
                        if not ok:
                            req2 = _urlreq.Request(link, headers={"Range": "bytes=0-0"})
                            with _urlreq.urlopen(req2, timeout=6) as r3:
                                code = getattr(r3, "status", 206)
                                ok = 200 <= int(code) < 400
                    except Exception:
                        ok = False
                    if ok and size_ok:
                        items.append({"name": os.path.basename(link), "url": link, "size_bytes": size_val})
                        if len(items) >= max_items:
                            break
            except Exception:
                pass
            if len(items) >= max_items:
                break
        return items
