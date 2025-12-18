from __future__ import annotations

# Import safe variable wrappers
from ..utils import safe_variables

import logging
from typing import Any, Callable, cast

logger = logging.getLogger(__name__)

try:  # pragma: no cover - environment dependent
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    tk = cast(Any, None)
    ttk = cast(Any, None)


class CrawlPanel(ttk.LabelFrame):  # type: ignore[misc]
    """Encapsulates the Crawl & Store UI and logic.

    Args:
        parent: container
        run_cli: callable to execute CLI with list[str] and return text output
        append_out/update_out: output helpers
    """

    def __init__(
        self,
        parent: Any,
        *,
        run_cli: Callable[[list[str]], str],
        append_out: Callable[[str], None],
        update_out: Callable[[str], None],
    ) -> None:
        super().__init__(parent, text="Crawl & Store")
        if tk is None:
            raise RuntimeError("Tkinter not available")
        self.pack(fill="x", padx=8, pady=(0, 8))

        self._run_cli = run_cli
        self._append_out = append_out
        self._update_out = update_out

        # URL + action bar
        self.url_var = safe_variables.StringVar(value="https://example.com")
        r1 = ttk.Frame(self)
        r1.pack(fill="x")
        ttk.Label(r1, text="URL:").pack(side="left")
        ttk.Entry(r1, textvariable=self.url_var).pack(side="left", fill="x", expand=True, padx=(4, 6))
        ttk.Button(r1, text="Crawl & Store", command=self.run_crawl).pack(side="left")

        # Options
        self.crawl_no_robots_var = safe_variables.BooleanVar(value=False)
        self.crawl_ttl_var = safe_variables.StringVar(value="0")
        self.crawl_render_var = safe_variables.BooleanVar(value=False)
        self.crawl_traf_var = safe_variables.BooleanVar(value=False)
        self.crawl_recursive_var = safe_variables.BooleanVar(value=False)
        self.crawl_pages_var = safe_variables.StringVar(value="50")
        self.crawl_depth_var = safe_variables.StringVar(value="2")
        self.crawl_same_domain_var = safe_variables.BooleanVar(value=True)
        self.crawl_store_var = safe_variables.StringVar(value="web_crawl")
        self.crawl_overwrite_var = safe_variables.BooleanVar(value=False)
        # New: free-form extra flags for crawl CLI
        self.crawl_extra_flags_var = safe_variables.StringVar(value="--rps 1 --progress")

        r2 = ttk.Frame(self)
        r2.pack(fill="x", pady=(4, 0))
        ttk.Checkbutton(r2, text="No robots", variable=self.crawl_no_robots_var).pack(side="left")
        ttk.Checkbutton(r2, text="Render", variable=self.crawl_render_var).pack(side="left", padx=(6, 0))
        ttk.Checkbutton(r2, text="Trafilatura", variable=self.crawl_traf_var).pack(side="left", padx=(6, 0))
        ttk.Checkbutton(r2, text="Recursive", variable=self.crawl_recursive_var).pack(side="left", padx=(6, 0))
        ttk.Label(r2, text="Pages:").pack(side="left", padx=(8, 2))
        ttk.Entry(r2, textvariable=self.crawl_pages_var, width=6).pack(side="left")
        ttk.Label(r2, text="Depth:").pack(side="left", padx=(8, 2))
        ttk.Entry(r2, textvariable=self.crawl_depth_var, width=4).pack(side="left")
        ttk.Checkbutton(r2, text="Same domain", variable=self.crawl_same_domain_var).pack(side="left", padx=(8, 0))
        ttk.Label(r2, text="TTL(s):").pack(side="left", padx=(8, 2))
        ttk.Entry(r2, textvariable=self.crawl_ttl_var, width=6).pack(side="left")

        r3 = ttk.Frame(self)
        r3.pack(fill="x", pady=(4, 0))
        ttk.Label(r3, text="Store dataset:").pack(side="left")
        ttk.Entry(r3, textvariable=self.crawl_store_var, width=18).pack(side="left", padx=(4, 8))
        ttk.Checkbutton(r3, text="Overwrite", variable=self.crawl_overwrite_var).pack(side="left")

        # Extra flags input row + progress + stop
        r4 = ttk.Frame(self)
        r4.pack(fill="x", pady=(4, 0))
        ttk.Label(r4, text="Extra crawl flags:").pack(side="left")
        ttk.Entry(r4, textvariable=self.crawl_extra_flags_var).pack(side="left", fill="x", expand=True, padx=(6, 8))
        # Progress bar and Stop button
        self._crawl_progress = ttk.Progressbar(r4, orient="horizontal", mode="determinate", length=160)
        self._crawl_progress.pack(side="left", padx=(0, 6))
        self._crawl_stop_btn = ttk.Button(r4, text="Stop", command=self._stop_crawl, state="disabled")
        self._crawl_stop_btn.pack(side="left")
        self._crawl_proc = None  # type: ignore[assignment]
        self._crawl_max = 0
        self._crawl_curr = 0

    # public API
    def run_crawl(self) -> str:
        """Run crawl with current options in background; streams live progress to output."""
        import threading, json, subprocess as _sp, sys, shlex
        url = (self.url_var.get() or "").strip()
        if not url:
            logger.warning("User attempted to start crawl with empty URL")
            self._append_out("[ui] URL is empty")
            return ""
        
        # Build crawl command
        args = ["crawl", url, "--progress"]
        if self.crawl_no_robots_var.get():
            args.append("--no-robots")
        ttl = (self.crawl_ttl_var.get() or "0").strip()
        if ttl and ttl.isdigit() and int(ttl) > 0:
            args += ["--ttl-sec", ttl]
        if self.crawl_render_var.get():
            args.append("--render")
        if self.crawl_traf_var.get():
            args.append("--trafilatura")
        if self.crawl_recursive_var.get():
            args.append("--recursive")
            pages = self.crawl_pages_var.get().strip() or "50"
            depth = self.crawl_depth_var.get().strip() or "2"
            args += ["--max-pages", pages, "--max-depth", depth]
            args.append("--same-domain" if self.crawl_same_domain_var.get() else "--any-domain")
        ds = (self.crawl_store_var.get() or "web_crawl").strip()
        if ds:
            args += ["--store-dataset", ds]
            if self.crawl_overwrite_var.get():
                args.append("--overwrite")
        # Extra flags (free-form)
        extra = (self.crawl_extra_flags_var.get() or "").strip()
        if extra:
            try:
                args += shlex.split(extra)
            except Exception:
                # best effort: append raw as one arg
                args.append(extra)
        
        # Log crawl start
        crawl_options = {
            'url': url,
            'recursive': self.crawl_recursive_var.get(),
            'max_pages': pages if self.crawl_recursive_var.get() else 'N/A',
            'max_depth': depth if self.crawl_recursive_var.get() else 'N/A',
            'dataset': ds,
            'render': self.crawl_render_var.get(),
        }
        logger.info(f"User action: Starting web crawl - {url}")
        logger.debug(f"Crawl options: {crawl_options}")
        
        self._update_out(f"Crawling: {url}\n")
        # reset progress UI
        try:
            self._crawl_progress.config(mode="determinate", maximum=100.0)
            self._crawl_progress['value'] = 0.0
            self._crawl_max = 0
            self._crawl_curr = 0
            self._crawl_stop_btn.config(state="disabled")
        except Exception:
            pass
        # Run in background to keep UI responsive and stream JSONL progress
        def _bg():
            cmd = [sys.executable, "-u", "-m", "aios.cli.aios", *args]
            try:
                # On Windows, use CREATE_NO_WINDOW to prevent CMD popups
                creationflags = _sp.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                proc = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.PIPE, text=True, bufsize=1, creationflags=creationflags)
                # expose proc for cancellation
                def _enable_stop():
                    try:
                        self._crawl_stop_btn.config(state="normal")
                    except Exception:
                        pass
                self._crawl_proc = proc
                self.after(0, _enable_stop)
            except Exception as e:
                self._append_out(f"[crawl] start error: {e}")
                return
            try:
                if proc.stdout is None:
                    # unlikely, but guard against None stdout
                    output_iter = []
                else:
                    output_iter = iter(proc.stdout.readline, "")
                for line in output_iter:
                    ln = line.strip()
                    if not ln:
                        continue
                    # Try to parse progress JSON lines
                    try:
                        data = json.loads(ln)
                        if isinstance(data, dict) and data.get("event") == "page":
                            n = data.get("n")
                            mx = data.get("max")
                            u = data.get("url")
                            t = data.get("title")
                            self._append_out(f"[{n}/{mx}] {t or ''} — {u}")
                            # update progress bar on UI thread
                            def _upd(nv=n, mxv=mx):
                                try:
                                    if isinstance(mxv, int) and mxv > 0:
                                        self._crawl_max = mxv
                                        self._crawl_progress.config(maximum=float(mxv))
                                    if isinstance(nv, int):
                                        self._crawl_curr = nv
                                        self._crawl_progress['value'] = float(nv)
                                except Exception:
                                    pass
                            try:
                                self.after(0, _upd)
                            except Exception:
                                pass
                        else:
                            self._append_out(ln)
                    except Exception:
                        self._append_out(ln)
            finally:
                rc = None
                try:
                    rc = proc.wait(timeout=1.0)
                except Exception:
                    pass
                
                # Log completion
                pages_crawled = self._crawl_curr if hasattr(self, '_crawl_curr') else 0
                if rc == 0:
                    logger.info(f"Web crawl completed successfully: {pages_crawled} pages crawled")
                else:
                    logger.warning(f"Web crawl ended with return code {rc}: {pages_crawled} pages crawled")
                
                # disable stop button and finalize progress
                def _finish():
                    try:
                        self._crawl_stop_btn.config(state="disabled")
                        if self._crawl_max and self._crawl_curr < self._crawl_max:
                            self._crawl_progress['value'] = float(self._crawl_max)
                    except Exception:
                        pass
                try:
                    self.after(0, _finish)
                except Exception:
                    pass
                self._append_out(f"[crawl done] returncode={rc}")

        threading.Thread(target=_bg, daemon=True).start()
        return ""

    def _stop_crawl(self) -> None:
        """Stop a running crawl if any."""
        logger.info("User action: Stopping web crawl")
        try:
            proc = getattr(self, "_crawl_proc", None)
            if proc is not None:
                try:
                    proc.terminate()
                    logger.debug(f"Terminated crawl process (PID: {proc.pid})")
                except Exception as e:
                    logger.error(f"Error terminating crawl process: {e}")
        finally:
            try:
                self._crawl_stop_btn.config(state="disabled")
            except Exception:
                pass

    def get_state(self) -> dict[str, Any]:
        return {
            "crawl_recursive": bool(self.crawl_recursive_var.get()),
            "crawl_pages": self.crawl_pages_var.get(),
            "crawl_depth": self.crawl_depth_var.get(),
            "crawl_same_domain": bool(self.crawl_same_domain_var.get()),
            "crawl_ttl": self.crawl_ttl_var.get(),
            "crawl_render": bool(self.crawl_render_var.get()),
            "crawl_trafilatura": bool(self.crawl_traf_var.get()),
            "crawl_store_dataset": self.crawl_store_var.get(),
            "crawl_overwrite": bool(self.crawl_overwrite_var.get()),
            "crawl_no_robots": bool(self.crawl_no_robots_var.get()),
            "crawl_extra_flags": self.crawl_extra_flags_var.get(),
            "url": self.url_var.get(),
        }
