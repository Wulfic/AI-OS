from __future__ import annotations

import os
from typing import Any


def open_rank_logs(panel: Any) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
        import glob
        log_dir = os.environ.get("AIOS_DDP_LOG_DIR")
        if not log_dir or not os.path.isdir(log_dir):
            panel._log("[hrm] No per-rank log directory available yet.")
            return
        top = tk.Toplevel(panel)
        top.title("Per-Rank Logs")
        frm = ttk.Frame(top)
        frm.pack(fill="both", expand=True)
        txt = tk.Text(frm, height=30, width=120, wrap="none")
        txt.pack(fill="both", expand=True)
        txt.insert("end", f"Monitoring: {log_dir}\n")
        txt.configure(state="disabled")

        prev_sizes: dict[str,int] = {}

        def _refresh() -> None:
            try:
                files = sorted(glob.glob(os.path.join(log_dir, "rank*.out.log")))
                out_lines = []
                for fpath in files:
                    try:
                        sz = os.path.getsize(fpath)
                        last_sz = prev_sizes.get(fpath, 0)
                        if sz < last_sz:
                            prev_sizes[fpath] = 0
                            last_sz = 0
                        read_from = max(0, sz - 8000)
                        with open(fpath, "rb") as fh:
                            if read_from:
                                fh.seek(read_from)
                            data = fh.read().decode("utf-8", errors="replace")
                        out_lines.append(f"=== {os.path.basename(fpath)} (tail) ===\n{data}\n")
                        prev_sizes[fpath] = sz
                    except Exception as fe:
                        out_lines.append(f"[error reading {fpath}: {fe}]")
                txt.configure(state="normal")
                txt.delete("1.0", "end")
                txt.insert("end", "\n".join(out_lines) or "(no rank logs yet)")
                txt.configure(state="disabled")
            except Exception:
                pass
            try:
                if top.winfo_exists():
                    top.after(2000, _refresh)
            except Exception:
                pass

        top.after(500, _refresh)
    except Exception:
        panel._log("[hrm] Failed to open rank log viewer.")
