from __future__ import annotations

import threading
from typing import Any, Callable, cast

# Import safe variable wrappers
from ..utils import safe_variables

from aios.python_exec import get_preferred_python_executable
from ..utils.resource_management import submit_background

try:  # pragma: no cover - environment dependent
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    tk = cast(Any, None)
    ttk = cast(Any, None)


class DatasetBuilderPanel(ttk.LabelFrame):  # type: ignore[misc]
    """UI to build image datasets by query via datasets-build-images CLI.

    Emits progress and updates status; on completion, sets the dataset path var.
    """

    def __init__(
        self,
        parent: Any,
        *,
        run_cli: Callable[[list[str]], str],
        dataset_path_var: Any,
        append_out: Callable[[str], None],
        update_out: Callable[[str], None],
        worker_pool: Any = None,
    ) -> None:
        super().__init__(parent, text="Dataset Builder (Images)")
        if tk is None or ttk is None:
            raise RuntimeError("Tkinter is not available in this environment")
        # Mirror the shared output frame padding so both sides align vertically
        self.pack(fill="both", expand=True, padx=5, pady=(5, 5))
        self._run_cli = run_cli
        self._append_out = append_out
        self._update_out = update_out
        self.dataset_path_var = dataset_path_var
        self._worker_pool = worker_pool  # Store worker pool for async operations

        # Controls
        # Row 0: dataset type selector
        r0 = ttk.Frame(self)
        r0.pack(fill="x", pady=(0, 4))
        ttk.Label(r0, text="Type:").pack(side="left")
        self.type_var = safe_variables.StringVar(value="images")
        self.type_combo = ttk.Combobox(r0, textvariable=self.type_var, values=[
            "images", "videos", "text", "websites", "raw"
        ], state="readonly", width=12)
        self.type_combo.pack(side="left", padx=(4, 8))
        try:
            from .tooltips import add_tooltip
            add_tooltip(self.type_combo, "Dataset content type to build.")
        except Exception:
            pass

        r1 = ttk.Frame(self)
        r1.pack(fill="x")
        ttk.Label(r1, text="Query:").pack(side="left")
        self.query_var = safe_variables.StringVar(value="boats")
        q_entry = ttk.Entry(r1, textvariable=self.query_var, width=40)
        q_entry.pack(side="left", padx=(4, 8))
        try:
            from .tooltips import add_tooltip
            add_tooltip(q_entry, "Search query or topic.")
        except Exception:
            pass
        ttk.Label(r1, text="Max images:").pack(side="left")
        self.max_images_var = safe_variables.StringVar(value="200")
        max_entry = ttk.Entry(r1, textvariable=self.max_images_var, width=8)
        max_entry.pack(side="left", padx=(4, 8))
        try:
            from .tooltips import add_tooltip
            add_tooltip(max_entry, "Target number of items to download.")
        except Exception:
            pass
        ttk.Label(r1, text="Per-site:").pack(side="left")
        self.per_site_var = safe_variables.StringVar(value="40")
        per_site_entry = ttk.Entry(r1, textvariable=self.per_site_var, width=6)
        per_site_entry.pack(side="left", padx=(4, 8))
        try:
            from .tooltips import add_tooltip
            add_tooltip(per_site_entry, "Maximum items taken from a single domain.")
        except Exception:
            pass
        ttk.Label(r1, text="Search results:").pack(side="left")
        self.search_results_var = safe_variables.StringVar(value="10")
        search_results_entry = ttk.Entry(r1, textvariable=self.search_results_var, width=6)
        search_results_entry.pack(side="left", padx=(4, 8))
        try:
            from .tooltips import add_tooltip
            add_tooltip(search_results_entry, "Number of search result pages to explore per query.")
        except Exception:
            pass

        r2 = ttk.Frame(self)
        r2.pack(fill="x", pady=(4, 0))
        ttk.Label(r2, text="Dataset name:").pack(side="left")
        self.ds_name_var = safe_variables.StringVar(value="")
        name_entry = ttk.Entry(r2, textvariable=self.ds_name_var, width=30)
        name_entry.pack(side="left", padx=(4, 8))
        try:
            from .tooltips import add_tooltip
            add_tooltip(name_entry, "Optional dataset name for persistent storage.")
        except Exception:
            pass
        self.overwrite_var = safe_variables.BooleanVar(value=False)
        ow_cb = ttk.Checkbutton(r2, text="Overwrite", variable=self.overwrite_var)
        ow_cb.pack(side="left")
        try:
            from .tooltips import add_tooltip
            add_tooltip(ow_cb, "Replace existing dataset with same name if it exists.")
        except Exception:
            pass

        # Row: optional allow-ext filter
        r2b = ttk.Frame(self)
        r2b.pack(fill="x", pady=(2, 0))
        ttk.Label(r2b, text="Extensions filter (comma):").pack(side="left")
        self.allow_ext_var = safe_variables.StringVar(value="")
        self.allow_ext_entry = ttk.Entry(r2b, textvariable=self.allow_ext_var, width=40)
        self.allow_ext_entry.pack(side="left", padx=(4, 8))
        self.allow_ext_hint = ttk.Label(r2b, text="e.g. jpg,png or pdf,txt")
        self.allow_ext_hint.pack(side="left")
        try:
            from .tooltips import add_tooltip
            add_tooltip(self.allow_ext_entry, "Comma separated whitelist of extensions to keep (leave blank for any).")
        except Exception:
            pass

        r3 = ttk.Frame(self)
        r3.pack(fill="x", pady=(6, 0))
        self.btn_build = ttk.Button(r3, text="Build Dataset", command=self.start_build)
        self.btn_build.pack(side="left")
        try:
            from .tooltips import add_tooltip
            add_tooltip(self.btn_build, "Start background build; progress and events stream below.")
        except Exception:
            pass
        self.progress = ttk.Progressbar(r3, orient="horizontal", mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=(8, 6))
        self.progress_lbl = ttk.Label(r3, text="idle")
        self.progress_lbl.pack(side="left")
        try:
            from .tooltips import add_tooltip
            add_tooltip(self.progress_lbl, "Status of current build (searching, downloading, done).")
        except Exception:
            pass

        # Footnote to highlight the panel's experimental status
        self._footnote = ttk.Label(self, text="Dataset Builder is experimental (WIP).")
        try:
            self._footnote.configure(font=("TkDefaultFont", 8, "italic"))
        except Exception:
            pass
        self._footnote.pack(fill="x", padx=4, pady=(6, 0))

        # internal proc state
        self._proc = None  # type: ignore[assignment]
        # update filter visibility with type
        def _on_type_change(*_):
            t = (self.type_var.get() or "").strip()
            show = t in ("images", "text", "videos")
            try:
                r2b.pack_forget() if not show else r2b.pack(fill="x", pady=(2, 0))
                if t == "images":
                    self.allow_ext_hint.configure(text="e.g. jpg,png,webp")
                elif t == "text":
                    self.allow_ext_hint.configure(text="e.g. pdf,txt,docx")
                elif t == "videos":
                    self.allow_ext_hint.configure(text="e.g. mp4,webm,m4v,mov")
            except Exception:
                pass
        try:
            self.type_var.trace_add("write", lambda *_: _on_type_change())
            _on_type_change()
        except Exception:
            pass

    # === Internal helpers ===

    def _post_to_ui(self, callback: Callable[[], None]) -> None:
        """Schedule *callback* on the Tk event loop (best effort)."""
        try:
            if self.winfo_exists():
                self.after(0, callback)
        except Exception:
            pass

    def _run_background(self, label: str, work: Callable[[], None]) -> None:
        """Execute *work* using the shared worker pool or a fallback thread."""
        try:
            submit_background(label, work, pool=self._worker_pool)
        except RuntimeError:
            threading.Thread(target=work, name=label, daemon=True).start()

    def _safe_append_out(self, message: str) -> None:
        if not message:
            return

        def _apply() -> None:
            try:
                self._append_out(message)
            except Exception:
                pass

        self._post_to_ui(_apply)

    def _set_progress_label(self, text: str) -> None:
        def _apply() -> None:
            try:
                self.progress_lbl.configure(text=text)
            except Exception:
                pass

        self._post_to_ui(_apply)

    def _set_progress_value(self, downloaded: float, target: float) -> None:
        def _apply() -> None:
            try:
                self.progress.configure(mode="determinate", maximum=float(target))
                self.progress["value"] = float(downloaded)
            except Exception:
                pass

        self._post_to_ui(_apply)

    def _set_dataset_path(self, path: str) -> None:
        def _apply() -> None:
            try:
                self.dataset_path_var.set(path)
            except Exception:
                pass

        self._post_to_ui(_apply)

    def _on_build_complete(self, return_code: int) -> None:
        def _apply() -> None:
            try:
                self.btn_build.configure(state="normal")
            except Exception:
                pass
            try:
                self.progress["value"] = self.progress["maximum"]
            except Exception:
                pass
            try:
                self.progress_lbl.configure(text=f"done (rc={return_code})")
            except Exception:
                pass

        self._post_to_ui(_apply)

    def get_state(self) -> dict:
        return {
            "type": self.type_var.get(),
            "query": self.query_var.get(),
            "max_images": self.max_images_var.get(),
            "per_site": self.per_site_var.get(),
            "search_results": self.search_results_var.get(),
            "dataset_name": self.ds_name_var.get(),
            "overwrite": bool(self.overwrite_var.get()),
        }

    def start_build(self) -> None:
        import json
        import shlex
        import subprocess as _sp

        q = (self.query_var.get() or "").strip()
        if not q:
            self._append_out("[builder] query is empty")
            return
        dtype = (self.type_var.get() or "images").strip()
        # Choose command per type. For non-image types, call stub commands for now.
        if dtype == "images":
            args = [
                "datasets-build-images",
                q,
                "--max-images",
                (self.max_images_var.get().strip() or "200"),
                "--per-site",
                (self.per_site_var.get().strip() or "40"),
                "--search-results",
                (self.search_results_var.get().strip() or "10"),
            ]
            allow = (self.allow_ext_var.get() or "").strip()
            if allow:
                args += ["--allow-ext", allow]
        elif dtype == "videos":
            args = [
                "datasets-build-videos", q,
                "--max-videos", (self.max_images_var.get().strip() or "50"),
                "--per-site", (self.per_site_var.get().strip() or "10"),
                "--search-results", (self.search_results_var.get().strip() or "10"),
            ]
            allow = (self.allow_ext_var.get() or "").strip()
            if allow:
                args += ["--allow-ext", allow]
        elif dtype == "text":
            args = [
                "datasets-build-text", q,
                "--max-docs", (self.max_images_var.get().strip() or "200"),
                "--per-site", (self.per_site_var.get().strip() or "40"),
                "--search-results", (self.search_results_var.get().strip() or "10"),
            ]
            allow = (self.allow_ext_var.get() or "").strip()
            if allow:
                args += ["--allow-ext", allow]
        elif dtype == "websites":
            args = [
                "datasets-build-websites", q,
                "--max-pages", (self.max_images_var.get().strip() or "200"),
                "--per-site", (self.per_site_var.get().strip() or "40"),
                "--search-results", (self.search_results_var.get().strip() or "10"),
            ]
        else:
            args = [
                "datasets-build-raw", q,
                "--max-files", (self.max_images_var.get().strip() or "100"),
                "--per-site", (self.per_site_var.get().strip() or "20"),
                "--search-results", (self.search_results_var.get().strip() or "10"),
            ]
        ds = (self.ds_name_var.get() or "").strip()
        if ds:
            args += ["--store-dataset", ds]
        if bool(self.overwrite_var.get()):
            args.append("--overwrite")
        # Launch in background to stream JSONL
        cmd = [get_preferred_python_executable(), "-u", "-m", "aios.cli.aios", *args]
        self._update_out("Building dataset: aios " + " ".join(args) + "\n")
        try:
            proc = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.PIPE, text=True, bufsize=1)
            self._proc = proc
            self.btn_build.configure(state="disabled")
            self.progress.configure(maximum=100.0, value=0.0)
            self.progress_lbl.configure(text="starting…")
        except Exception as e:
            self._append_out(f"[builder] start error: {e}")
            return

        def _reader(pipe, label):
            def run():
                if pipe is None:
                    return
                for line in iter(pipe.readline, ""):
                    ln = line.strip()
                    if not ln:
                        continue
                    # parse JSON lines
                    try:
                        data = json.loads(ln)
                        ev = data.get("event") if isinstance(data, dict) else None
                        if ev == "search":
                            sites = len(data.get("sites") or [])
                            self._set_progress_label(f"searching… {sites} sites")
                            continue
                        elif ev == "page":
                            images_found = data.get("images_found", 0)
                            self._set_progress_label(f"{images_found} images on page")
                            continue
                        elif ev in ("image", "doc", "html", "video", "file"):
                            d = int(data.get("downloaded", 0))
                            t = int(data.get("target", 100))
                            self._set_progress_value(d, t)
                            continue
                        else:
                            # final summary
                            if isinstance(data, dict) and data.get("stored"):
                                dp = data.get("dataset_path") or ""
                                if isinstance(dp, str) and dp:
                                    self._set_dataset_path(str(dp))
                                self._safe_append_out(ln)
                                continue
                    except Exception:
                        pass
                    self._safe_append_out(ln)
            self._run_background(f"dataset-builder-stream-{label}", run)
        _reader(proc.stdout, "out")
        _reader(proc.stderr, "err")

        def _waiter():
            rc = proc.wait()
            self._on_build_complete(rc)

        self._run_background("dataset-builder-wait", _waiter)
