from __future__ import annotations

import json
import queue
import re
import subprocess as _sp
import sys
import threading
from typing import Any, Callable

from aios.python_exec import get_preferred_python_executable

from typing import cast

try:  # pragma: no cover - environment dependent
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    tk = cast(Any, None)  # type: ignore
    ttk = cast(Any, None)  # type: ignore


class TrainingPanel:
    """Encapsulates training controls (single and parallel) and logic.

    Responsibilities:
    - Expose UI for steps/batch/feature-dim/emit-metrics
    - Provide Start (single) and Start Parallel buttons + Stop and progress UI
    - Build CLI args from current state and call provided run_cli
    - Read resource limits via provided resources_panel
    - Share dataset_path_var with DatasetsPanel
    """

    def __init__(
        self,
        parent: Any,
        *,
        run_cli: Callable[[list[str]], str],
        resources_panel: Any,
        dataset_path_var: Any,
        append_out: Callable[[str], None],
        update_out: Callable[[str], None],
        debug_set_error: Callable[[str], None],
        domains_default: str = "english",
    ) -> None:
        if tk is None or ttk is None:
            raise RuntimeError("Tkinter is not available in this environment")

        self.parent = parent
        self._run_cli = run_cli
        self._resources_panel = resources_panel
        self.dataset_path_var = dataset_path_var
        self._append_out = append_out
        self._update_out = update_out
        self._debug_set_error = debug_set_error

        # Training params
        self.steps_var = tk.StringVar(value="200")
        self.batch_var = tk.StringVar(value="64")
        self.mem_var = tk.StringVar(value="0.9")
        self.tag_var = tk.StringVar(value="parallel")
        self.train_flags_var = tk.StringVar(value="")
        self.domains_var = tk.StringVar(value=domains_default)
        self.hybrid_var = tk.BooleanVar(value=False)
        # English-first controls
        self.feature_dim_var = tk.StringVar(value="1024")
        self.emit_metrics_var = tk.BooleanVar(value=True)

        # Process state (parallel only)
        self._proc: _sp.Popen | None = None
        self._q: queue.Queue[str] | None = None

        # Build UI
        self._build_ui()

    # UI
    def _build_ui(self) -> None:
        # Devices & Training Params
        dev_frame = ttk.LabelFrame(self.parent, text="Devices & Training Params")
        dev_frame.pack(fill="x", padx=8, pady=(0, 8))
        row = ttk.Frame(dev_frame)
        row.pack(fill="x", padx=8, pady=(6, 4))
        ttk.Label(row, text="Steps:").pack(side="left")
        steps_entry = ttk.Entry(row, textvariable=self.steps_var, width=8)
        steps_entry.pack(side="left", padx=(4, 12))
        ttk.Label(row, text="Batch size:").pack(side="left")
        batch_entry = ttk.Entry(row, textvariable=self.batch_var, width=8)
        batch_entry.pack(side="left", padx=(4, 12))
        ttk.Label(row, text="Feature dim:").pack(side="left")
        feat_entry = ttk.Entry(row, textvariable=self.feature_dim_var, width=10)
        feat_entry.pack(side="left", padx=(4, 12))
        ttk.Checkbutton(row, text="Emit metrics", variable=self.emit_metrics_var).pack(side="left")
        # Automated pipeline controls
        auto_row = ttk.Frame(dev_frame)
        auto_row.pack(fill="x", padx=8, pady=(0, 6))
        auto_btn = ttk.Button(auto_row, text="Auto Train (Search → Crawl → Train)", command=self.start_auto_train)
        auto_btn.pack(side="left")

        # Parallel Training block
        par = ttk.LabelFrame(self.parent, text="Parallel Training")
        par.pack(fill="x", padx=8, pady=(0, 8))
        btns = ttk.Frame(par)
        btns.pack(fill="x")
        self.btn_train_parallel = ttk.Button(btns, text="Start Parallel", command=self.start_parallel)
        self.btn_train_parallel.pack(side="left")
        self.btn_stop = ttk.Button(btns, text="Stop", command=self.stop)
        try:
            self.btn_stop.configure(state="disabled")
        except Exception:
            pass
        self.btn_stop.pack(side="left", padx=(6, 0))

        flags_row = ttk.Frame(par)
        flags_row.pack(fill="x", padx=(8, 0), pady=(4, 0))
        ttk.Label(flags_row, text="Extra train flags:").pack(side="left")
        flags_entry = ttk.Entry(flags_row, textvariable=self.train_flags_var, width=60)
        flags_entry.pack(side="left", fill="x", expand=True, padx=(6, 0))

        pwrap = ttk.Frame(par)
        pwrap.pack(fill="x", padx=(8, 0))
        self.progress = ttk.Progressbar(pwrap, orient="horizontal", mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True)
        self.progress_status = tk.StringVar(value="Idle")
        ttk.Label(pwrap, textvariable=self.progress_status, width=12).pack(side="left", padx=(8, 0))
        # Add tooltips (best-effort; ignore failure)
        try:  # pragma: no cover - UI enhancement only
            from .tooltips import add_tooltip
            add_tooltip(steps_entry, "Total training steps (iterations over batches).")
            add_tooltip(batch_entry, "Examples per batch; adjust for memory constraints.")
            add_tooltip(feat_entry, "Feature dimension for English-first encoder layers.")
            add_tooltip(auto_btn, "Run discovery + crawl + training automatically.")
            add_tooltip(flags_entry, "Extra raw flags appended to training CLI invocation.")
            add_tooltip(self.btn_train_parallel, "Start multi-process parallel training run.")
            add_tooltip(self.btn_stop, "Stop the active parallel training run (graceful terminate).")
            add_tooltip(self.progress, "Parallel training estimated progress (percentage of steps).")
            add_tooltip(self.progress_status, "Current parallel training status message.")
        except Exception:
            pass

    # Public API
    def get_state(self) -> dict:
        return {
            "steps": self.steps_var.get(),
            "batch": self.batch_var.get(),
            "mem": self.mem_var.get(),
            "tag": self.tag_var.get(),
            "domains": self.domains_var.get(),
            "dataset_path": self.dataset_path_var.get(),
            "hybrid": bool(self.hybrid_var.get()),
            "feature_dim": self.feature_dim_var.get(),
            "emit_metrics": bool(self.emit_metrics_var.get()),
            "train_flags": self.train_flags_var.get(),
        }

    # Actions
    def start_single(self) -> None:
        steps = self.steps_var.get().strip() or "200"
        batch = self.batch_var.get().strip() or "64"
        self._update_out(f"Starting training ({steps} steps, batch {batch})...")
        args: list[str] = ["train", "--steps", steps, "--batch-size", batch, "--progress"]
        # Resource limits
        vals = self._resources_panel.get_values()
        th = int(vals.get("cpu_threads") or 0)
        if th > 0:
            args.extend(["--num-threads", str(th)])
        try:
            frac = str(round(max(0.1, min(0.99, float(self.mem_var.get() or "0.9"))), 2))
            args.extend(["--gpu-mem-frac", frac])
        except Exception:
            pass
        doms = (self.domains_var.get().strip() or "english")
        if doms:
            args.extend(["--domains", doms])
        ds = self.dataset_path_var.get().strip()
        if not ds:
            try:
                from aios.data.datasets import datasets_base_dir  # local import
                default_path = (datasets_base_dir() / "web_crawl" / "data.jsonl")
                if default_path.exists():
                    ds = str(default_path)
            except Exception:
                ds = ds
        if ds:
            args.extend(["--dataset-file", ds])
            if self.hybrid_var.get():
                args.append("--hybrid")
        # English-first flags
        try:
            fd = (self.feature_dim_var.get().strip() or "")
            if fd.isdigit() and int(fd) > 0:
                args.extend(["--feature-dim", fd])
        except Exception:
            pass
        try:
            if bool(self.emit_metrics_var.get()):
                args.append("--emit-metrics")
        except Exception:
            pass
        try:
            out = self._run_cli(args)
            self._update_out(out)
        except Exception:
            import traceback
            self._debug_set_error(traceback.format_exc())

    def start_auto_train(self) -> None:
        """End-to-end: get topics → search → crawl into dataset → start training.

        Non-blocking: runs search/crawl steps in a background thread, then triggers training on the UI thread.
        """
        import threading, ast, json

        def _bg():
            try:
                # 1) Discover topics from active directives via status
                status_raw = self._run_cli(["status", "--recent", "1"]) or "{}"
                # Parse dict-like output (JSON or Python literal)
                st: dict
                try:
                    st = json.loads(status_raw)
                except Exception:
                    try:
                        st = ast.literal_eval(status_raw)
                        if not isinstance(st, dict):
                            st = {}
                    except Exception:
                        st = {}
                directives = []
                try:
                    dlist = st.get("directives") or []
                    if isinstance(dlist, list):
                        directives = [str(x).strip() for x in dlist if str(x).strip()]
                except Exception:
                    directives = []
                if not directives:
                    directives = ["english writing basics", "python tutorials"]

                # 2) For each directive, run a minimal search (via CLI smoketest) and collect URLs
                urls: list[str] = []
                for q in directives[:3]:  # cap breadth
                    res = self._run_cli(["smoke", "--query", q]) or "[]"
                    arr_any = None
                    try:
                        arr_any = json.loads(res)
                    except Exception:
                        try:
                            arr_any = ast.literal_eval(res)
                        except Exception:
                            arr_any = None
                    if isinstance(arr_any, list):
                        for it in arr_any[:2]:  # top 2 per query
                            u = (it.get("url") if isinstance(it, dict) else None) or ""
                            if isinstance(u, str) and u and u not in urls:
                                urls.append(u)
                if not urls:
                    # Fallback: example
                    urls = ["https://example.com"]

                # 3) Crawl each URL into a single dataset
                ds_name = "auto_train"
                dataset_path: str | None = None
                first = True
                for u in urls[:6]:  # cap depth
                    flags = [
                        "--progress",
                        "--recursive",
                        "--max-pages",
                        "20",
                        "--max-depth",
                        "2",
                        "--same-domain",
                        "--rps",
                        "1",
                        "--store-dataset",
                        ds_name,
                    ]
                    if first:
                        flags.append("--overwrite")
                    self._append_out(f"[auto] Crawling {u}\n")
                    out = self._run_cli(["crawl", u, *flags]) or ""
                    # parse any JSON line containing dataset_path
                    try:
                        for ln in str(out).splitlines():
                            if "\"dataset_path\"" in ln:
                                try:
                                    data = json.loads(ln)
                                    dp = data.get("dataset_path")
                                    if isinstance(dp, str) and dp:
                                        dataset_path = dp
                                except Exception:
                                    continue
                    except Exception:
                        pass
                    first = False

                if not dataset_path:
                    self._append_out("[auto] No dataset_path returned by crawl; aborting auto-train.\n")
                    return

                # 4) Launch training on the prepared dataset
                def _kickoff(dp=dataset_path):
                    try:
                        self.dataset_path_var.set(dp)
                    except Exception:
                        pass
                    self._append_out(f"[auto] Starting training on {dp}\n")
                    self.start_single()

                try:
                    # schedule on UI thread
                    self.parent.after(0, _kickoff)
                except Exception:
                    _kickoff()
            except Exception as e:
                self._append_out(f"[auto] error: {e}\n")

        threading.Thread(target=_bg, daemon=True).start()

    def start_parallel(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._append_out("A run is already in progress. Stop it first.")
            return
        steps = self.steps_var.get().strip() or "200"
        batch = self.batch_var.get().strip() or "64"
        tag = self.tag_var.get().strip() or "parallel"
        # Resource selections
        rvals = self._resources_panel.get_values()
        try:
            memfrac = str(round(max(0.1, min(0.99, float(rvals.get("gpu_mem_pct") or 90) / 100.0)), 2))
        except Exception:
            memfrac = "0.9"
        args: list[str] = [
            "train-parallel",
            "--steps",
            steps,
            "--batch-size",
            batch,
            "--tag",
            tag,
            "--gpu-mem-frac",
            memfrac,
        ]
        # Per-train extra flags
        extra = (self.train_flags_var.get().strip() or "")
        try:
            fd = (self.feature_dim_var.get().strip() or "")
            if fd.isdigit() and int(fd) > 0 and "--feature-dim" not in extra:
                extra = (extra + f" --feature-dim {fd}").strip()
        except Exception:
            pass
        try:
            if bool(self.emit_metrics_var.get()) and "--emit-metrics" not in extra:
                extra = (extra + " --emit-metrics").strip()
        except Exception:
            pass
        # Ensure progress printing enabled so UI can parse step updates
        if "--progress" not in extra:
            extra = (extra + " --progress").strip()
        if extra:
            args.extend(["--train-flags", extra])
        # Devices and compute options (simplified)
        train_dev = str(rvals.get("train_device") or "auto")
        # Default: if auto and GPUs selected, use CUDA; else CPU
        cuda_sel = rvals.get("train_cuda_selected") or []
        use_cuda = False
        use_cpu = False
        try:
            if train_dev == "cpu":
                use_cpu = True
            elif train_dev == "cuda":
                use_cuda = True
            else:  # auto
                use_cuda = bool(cuda_sel)
                use_cpu = not use_cuda
        except Exception:
            use_cpu = True
            use_cuda = False
        args.append("--cpu" if use_cpu else "--no-cpu")
        args.append("--cuda" if use_cuda else "--no-cuda")
        try:
            if isinstance(cuda_sel, list) and cuda_sel:
                ids = ",".join(str(int(i)) for i in cuda_sel)
                args.extend(["--cuda-ids", ids])
            cuda_map = rvals.get("train_cuda_mem_pct")
            if isinstance(cuda_map, dict) and cuda_map:
                mp = {int(k): max(0.1, min(0.99, float(v) / 100.0)) for k, v in cuda_map.items()}
                import json as _json
                args.extend(["--cuda-mem-map", _json.dumps(mp)])
        except Exception:
            pass
        doms = (self.domains_var.get().strip() or "english")
        if doms:
            args.extend(["--domains", doms])
        ds = self.dataset_path_var.get().strip()
        if not ds:
            try:
                from aios.data.datasets import datasets_base_dir  # local import
                default_path = (datasets_base_dir() / "web_crawl" / "data.jsonl")
                if default_path.exists():
                    ds = str(default_path)
            except Exception:
                ds = ds
        if ds:
            args.extend(["--dataset-file", ds])
            if self.hybrid_var.get():
                args.append("--hybrid")

        # Notify and spawn
        self._update_out("Starting parallel training...\n" + "aios " + " ".join(args))
        cmd = [get_preferred_python_executable(), "-u", "-m", "aios.cli.aios", *args]
        try:
            self._proc = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.PIPE, text=True, bufsize=1)
        except Exception as e:
            self._append_out(f"error starting: {e}")
            import traceback
            self._debug_set_error(traceback.format_exc())
            return
        try:
            self.btn_train_parallel.configure(state="disabled")
            self.btn_stop.configure(state="normal")
            self.progress.configure(maximum=100, value=0)
            self.progress_status.set("Running...")
        except Exception:
            pass
        self._q = queue.Queue()

        def _reader(pipe, label):
            try:
                for line in iter(pipe.readline, ""):
                    if self._q is not None:
                        self._q.put(f"[{label}] {line.rstrip()}")
            except Exception:
                pass
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass

        threading.Thread(target=_reader, args=(self._proc.stdout, "out"), daemon=True).start()
        threading.Thread(target=_reader, args=(self._proc.stderr, "err"), daemon=True).start()

        def _drain():
            if self._q:
                try:
                    while True:
                        msg = self._q.get_nowait()
                        self._append_out(msg)
                        try:
                            lower = msg.lower()
                            m = re.search(r"step\s+(\d+)\s*/\s*(\d+)", lower)
                            if m:
                                n = int(m.group(1))
                                d = max(1, int(m.group(2)))
                                pct = max(0, min(100, int(n * 100 / d)))
                                self.progress.configure(value=pct)
                                self.progress_status.set(f"{pct}% ({n}/{d})")
                            else:
                                m2 = re.search(r"(\d{1,3})%", lower)
                                if m2:
                                    pct2 = max(0, min(100, int(m2.group(1))))
                                    self.progress.configure(value=pct2)
                                    self.progress_status.set(f"{pct2}%")
                        except Exception:
                            pass
                except queue.Empty:
                    pass
            if self._proc and self._proc.poll() is None:
                # schedule again
                try:
                    self.parent.after(100, _drain)
                except Exception:
                    pass
            else:
                rc = None
                try:
                    rc = self._proc.returncode if self._proc else None
                except Exception:
                    rc = None
                self._append_out(f"[done] returncode={rc}")
                try:
                    self.btn_train_parallel.configure(state="normal")
                    self.btn_stop.configure(state="disabled")
                    self.progress.configure(value=0)
                    self.progress_status.set("Idle")
                except Exception:
                    pass
                self._proc = None
                self._q = None

        try:
            self.parent.after(100, _drain)
        except Exception:
            pass

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._append_out("[stopping] terminate sent")
            except Exception as e:
                self._append_out(f"error stopping: {e}")
        try:
            self.btn_train_parallel.configure(state="normal")
            self.btn_stop.configure(state="disabled")
        except Exception:
            pass
