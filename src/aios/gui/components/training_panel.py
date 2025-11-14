from __future__ import annotations

# Import safe variable wrappers
from ..utils import safe_variables

import json
import logging
import queue
import re
import subprocess as _sp
import sys
import time
from typing import Any, Callable, Optional

from aios.python_exec import get_preferred_python_executable
from ..utils.resource_management import submit_background

from typing import cast

try:  # pragma: no cover - environment dependent
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    tk = cast(Any, None)  # type: ignore
    ttk = cast(Any, None)  # type: ignore

logger = logging.getLogger(__name__)


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
        run_cli_async: Optional[Callable[..., Any]] = None,
        worker_pool: Any | None = None,
        ui_dispatcher: Any | None = None,
    ) -> None:
        if tk is None or ttk is None:
            raise RuntimeError("Tkinter is not available in this environment")

        logger.info("Initializing Training Panel")
        
        self.parent = parent
        self._run_cli = run_cli
        self._run_cli_async = run_cli_async
        self._resources_panel = resources_panel
        self.dataset_path_var = dataset_path_var
        self._append_out = append_out
        self._update_out = update_out
        self._debug_set_error = debug_set_error
        self._worker_pool = worker_pool
        self._ui_dispatcher = ui_dispatcher
        self._single_run_inflight = False

        # Training params
        self.steps_var = safe_variables.StringVar(value="200")
        self.batch_var = safe_variables.StringVar(value="64")
        self.mem_var = safe_variables.StringVar(value="0.9")
        self.tag_var = safe_variables.StringVar(value="parallel")
        self.train_flags_var = safe_variables.StringVar(value="")
        self.domains_var = safe_variables.StringVar(value=domains_default)
        self.hybrid_var = safe_variables.BooleanVar(value=False)
        # English-first controls
        self.feature_dim_var = safe_variables.StringVar(value="1024")
        self.emit_metrics_var = safe_variables.BooleanVar(value=True)

        # Process state (parallel only)
        self._proc: _sp.Popen | None = None
        self._q: queue.Queue[str] | None = None

        # Build UI
        logger.debug("Building Training Panel UI")
        self._build_ui()
        logger.info("Training Panel initialized successfully")

    def _dispatch_ui(self, func: Callable[[], None]) -> None:
        """Execute *func* on the Tk UI thread when possible."""
        try:
            self.parent.after(0, func)
        except Exception:
            func()

    def _run_in_background(self, task: Callable[[], Any]) -> None:
        """Run a callable on the worker pool or a fallback thread."""
        try:
            submit_background("training-panel", task, pool=self._worker_pool)
        except RuntimeError as exc:
            logger.error("Failed to queue training task: %s", exc)
            self._append_out(f"[train] queue saturated: {exc}")

    def _submit_cli(
        self,
        args: list[str],
        *,
        use_cache: bool = False,
        on_success: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        """Execute a CLI command without blocking the UI thread."""
        if callable(self._run_cli_async):
            try:
                self._run_cli_async(
                    args,
                    use_cache=use_cache,
                    worker_pool=self._worker_pool,
                    ui_dispatcher=self._ui_dispatcher,
                    on_success=on_success,
                    on_error=on_error,
                )
                return
            except TypeError:
                # Older async helpers without keyword support; fall back to default path
                pass

        def _worker() -> None:
            try:
                result = self._run_cli(args)
            except Exception as exc:
                if on_error is not None:
                    self._dispatch_ui(lambda: on_error(exc))
            else:
                if on_success is not None:
                    self._dispatch_ui(lambda: on_success(result))

        self._run_in_background(_worker)

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
        self.progress_status = safe_variables.StringVar(value="Idle")
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
        logger.info(f"User action: Starting single training (steps={steps}, batch={batch})")
        self._update_out(f"Starting training ({steps} steps, batch {batch})...")
        args: list[str] = ["train", "--steps", steps, "--batch-size", batch, "--progress"]
        # Resource limits
        vals = self._resources_panel.get_values()
        th = int(vals.get("cpu_threads") or 0)
        if th > 0:
            args.extend(["--num-threads", str(th)])
            logger.debug(f"Training will use {th} CPU threads")
        try:
            frac = str(round(max(0.1, min(0.99, float(self.mem_var.get() or "0.9"))), 2))
            args.extend(["--gpu-mem-frac", frac])
            logger.debug(f"Training will use {frac} GPU memory fraction")
        except Exception:
            pass
        doms = (self.domains_var.get().strip() or "english")
        if doms:
            args.extend(["--domains", doms])
            logger.debug(f"Training domains: {doms}")
        ds = self.dataset_path_var.get().strip()
        if not ds:
            try:
                from aios.data.datasets import datasets_base_dir  # local import
                default_path = (datasets_base_dir() / "web_crawl" / "data.jsonl")
                if default_path.exists():
                    ds = str(default_path)
                    logger.debug(f"Using default dataset path: {ds}")
            except Exception:
                ds = ds
        if ds:
            args.extend(["--dataset-file", ds])
            logger.info(f"Training on dataset: {ds}")
            if self.hybrid_var.get():
                args.append("--hybrid")
                logger.debug("Hybrid mode enabled")
        else:
            logger.warning("No dataset path specified for training")
        # English-first flags
        try:
            fd = (self.feature_dim_var.get().strip() or "")
            if fd.isdigit() and int(fd) > 0:
                args.extend(["--feature-dim", fd])
                logger.debug(f"Feature dimension: {fd}")
        except Exception:
            pass
        try:
            if bool(self.emit_metrics_var.get()):
                args.append("--emit-metrics")
                logger.debug("Metrics emission enabled")
        except Exception:
            pass
        if self._single_run_inflight:
            self._append_out("[train] A training run is already active.")
            return

        self._single_run_inflight = True

        def _on_success(result: str) -> None:
            self._update_out(result)
            logger.info("Single training completed successfully")
            self._single_run_inflight = False

        def _on_error(exc: Exception) -> None:
            error_context = "Training execution failed"
            error_str = str(exc).lower()
            if "cuda" in error_str or "gpu" in error_str:
                suggestion = "GPU error. Check CUDA installation, GPU availability, or try using CPU mode"
            elif "memory" in error_str or "out of memory" in error_str:
                suggestion = "Out of memory. Reduce batch size, enable gradient accumulation, or use a smaller model"
            elif "dataset" in error_str or "data" in error_str:
                suggestion = "Dataset error. Verify dataset path and format are correct"
            elif "permission" in error_str:
                suggestion = "Permission denied. Check write permissions for output directories"
            elif "module" in error_str or "import" in error_str:
                suggestion = "Missing dependencies. Run 'pip install -r requirements.txt' to install required packages"
            else:
                suggestion = "Check logs for details. Verify dataset, GPU availability, and system resources"

            logger.error(f"{error_context}: {exc}. Suggestion: {suggestion}", exc_info=True)
            try:
                import traceback
                self._debug_set_error(f"{traceback.format_exc()}\n\nSuggestion: {suggestion}")
            except Exception:
                self._append_out(f"[train] Error: {exc}\nSuggestion: {suggestion}")
            self._single_run_inflight = False

        logger.debug(f"Queueing training CLI: {' '.join(args)}")
        try:
            self._submit_cli(args, use_cache=False, on_success=_on_success, on_error=_on_error)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Failed to queue training CLI: {exc}", exc_info=True)
            self._single_run_inflight = False
            _on_error(exc)

    def start_auto_train(self) -> None:
        """End-to-end: get topics → search → crawl into dataset → start training.

        Non-blocking: runs search/crawl steps in a background thread, then triggers training on the UI thread.
        """
        import ast, json

        logger.info("User action: Starting auto-train workflow (topics → search → crawl → train)")
        start_time = time.time()

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
                    elapsed = time.time() - start_time
                    logger.warning(f"Auto-train aborted: No dataset path returned after {elapsed:.1f}s")
                    self._append_out("[auto] No dataset_path returned by crawl; aborting auto-train.\n")
                    return

                # 4) Launch training on the prepared dataset
                def _kickoff(dp=dataset_path):
                    try:
                        self.dataset_path_var.set(dp)
                    except Exception:
                        pass
                    elapsed = time.time() - start_time
                    logger.info(f"Auto-train dataset ready after {elapsed:.1f}s, starting training on {dp}")
                    self._append_out(f"[auto] Starting training on {dp}\n")
                    self.start_single()

                try:
                    # schedule on UI thread
                    self.parent.after(0, _kickoff)
                except Exception:
                    _kickoff()
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Auto-train workflow failed after {elapsed:.1f}s: {e}", exc_info=True)
                self._append_out(f"[auto] error: {e}\n")
            finally:
                logger.debug("Auto-train background thread completed")

        logger.debug("Queueing auto-train workflow")
        self._run_in_background(_bg)

    def start_parallel(self) -> None:
        if self._proc and self._proc.poll() is None:
            logger.warning("Cannot start parallel training - run already in progress")
            self._append_out("A run is already in progress. Stop it first.")
            return
        steps = self.steps_var.get().strip() or "200"
        batch = self.batch_var.get().strip() or "64"
        tag = self.tag_var.get().strip() or "parallel"
        
        logger.info(f"User action: Starting parallel training (steps={steps}, batch={batch}, tag={tag})")
        
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
        logger.debug(f"Launching parallel training with command: {' '.join(args)}")
        cmd = [get_preferred_python_executable(), "-u", "-m", "aios.cli.aios", *args]
        try:
            self._proc = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.PIPE, text=True, bufsize=1)
            logger.info(f"Parallel training process started (PID: {self._proc.pid})")
        except Exception as e:
            error_context = "Failed to start parallel training process"
            
            # Provide contextual suggestions
            if "not found" in str(e).lower() or "no such file" in str(e).lower():
                suggestion = "Python executable not found. Reinstall AI-OS or check your Python installation"
            elif "permission" in str(e).lower():
                suggestion = "Permission denied. Run as administrator or check file permissions"
            elif "memory" in str(e).lower() or "resource" in str(e).lower():
                suggestion = "Insufficient system resources. Close other applications and try again"
            else:
                suggestion = "Check system resources and Python installation, then try again"
            
            logger.error(f"{error_context}: {e}. Suggestion: {suggestion}", exc_info=True)
            self._append_out(f"error starting: {e}\nSuggestion: {suggestion}")
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
            logger.debug(f"Starting training output reader thread: {label}")
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
                logger.debug(f"Training output reader thread exiting: {label}")

        def _queue_reader(pipe, label):
            try:
                submit_background(f"training-{label}-reader", _reader, pipe, label, pool=self._worker_pool)
                logger.debug("Queued training output reader task: %s", label)
            except RuntimeError as exc:
                logger.error("Failed to queue training output reader (%s): %s", label, exc)
                # Fallback to synchronous drain to avoid losing logs; runs on caller thread.
                _reader(pipe, label)

        _queue_reader(self._proc.stdout, "out")
        _queue_reader(self._proc.stderr, "err")

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
                
                if rc == 0:
                    logger.info(f"Parallel training completed successfully (returncode={rc})")
                elif rc is not None:
                    logger.warning(f"Parallel training exited with non-zero returncode={rc}")
                else:
                    logger.info("Parallel training process finished")
                
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
            logger.info(f"User requested training stop (PID: {self._proc.pid})")
            logger.debug("Cleaning up training resources")
            try:
                self._proc.terminate()
                self._append_out("[stopping] terminate sent")
                logger.info("Training process termination signal sent")
            except Exception as e:
                logger.error(f"Error terminating training process: {e}")
                self._append_out(f"error stopping: {e}")
        else:
            logger.debug("Stop requested but no active training process")
        try:
            self.btn_train_parallel.configure(state="normal")
            self.btn_stop.configure(state="disabled")
        except Exception:
            pass

    def cleanup(self) -> None:
        """Clean up training panel resources on shutdown."""
        logger.info("Cleaning up Training Panel")
        if self._proc and self._proc.poll() is None:
            logger.info(f"Terminating active training process (PID: {self._proc.pid})")
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3.0)
                logger.debug("Training process terminated successfully")
            except Exception as e:
                logger.warning(f"Error during training process cleanup: {e}")
                try:
                    self._proc.kill()
                    logger.debug("Training process killed (force)")
                except Exception as kill_error:
                    logger.error(f"Failed to kill training process: {kill_error}")
        self._proc = None
        self._q = None
        logger.info("Training Panel cleanup complete")
