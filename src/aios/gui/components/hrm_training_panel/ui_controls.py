"""Control buttons, progress bar, and log output UI for HRM Training Panel."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .panel_main import HRMTrainingPanel


def build_controls(panel: HRMTrainingPanel, parent: any) -> None:
    """Build control buttons section."""
    from ..hrm_training.optimizer_progressive import optimize_from_gui_progressive
    
    btns = ttk.Frame(parent)
    btns.pack(fill="x", pady=(6, 0))
    panel.start_btn = ttk.Button(btns, text="Start HRM Training", command=panel._on_start)
    panel.start_btn.pack(side="left")

    # Optimize button with experimental label
    opt_frame = ttk.Frame(btns)
    opt_frame.pack(side="left", padx=(6, 0))
    opt_btn = ttk.Button(opt_frame, text="Optimize Settings", command=panel._on_optimize)
    opt_btn.pack(side="left")
    ttk.Label(opt_frame, text="(Experimental)", font=("", 7), foreground="gray").pack(side="left", padx=(4, 0))

    ttk.Button(btns, text="Select Student", command=panel._on_select_student).pack(side="left", padx=(6, 0))
    panel.stop_btn = ttk.Button(btns, text="Stop", command=panel._stop_all)
    panel.stop_btn.pack(side="left", padx=(6, 0))

    # Iterate checkbox
    iterate_check = ttk.Checkbutton(btns, text="Iterate Mode", variable=panel.iterate_var)
    iterate_check.pack(side="left", padx=(6, 0))
    
    # Stop after block checkbox
    stop_after_block_check = ttk.Checkbutton(btns, text="Stop After Block", variable=panel.stop_after_block_var)
    stop_after_block_check.pack(side="left", padx=(6, 0))
    
    # Stop after epoch checkbox
    stop_after_epoch_check = ttk.Checkbutton(btns, text="Stop After Epoch", variable=panel.stop_after_epoch_var)
    stop_after_epoch_check.pack(side="left", padx=(6, 0))

    # Clear Output button
    clear_btn = ttk.Button(btns, text="Clear Output", command=panel._clear_output)
    clear_btn.pack(side="right")
    
    try:
        from ..tooltips import add_tooltip
        add_tooltip(panel.start_btn, "Launch training with current settings.")
        add_tooltip(panel.stop_btn, "Stop training:\n• First click (during training): Gracefully finish current chunk then exit (button turns red)\n• Second click or first click (during optimization): Immediate stop")
        add_tooltip(iterate_check, "Iterate Mode: Continuous training cycles until manually stopped.")
        add_tooltip(stop_after_block_check, "Stop After Block: Complete the current block then stop gracefully.\nBlock = downloaded dataset chunk (e.g., 100k samples). Useful for testing or incremental training.")
        add_tooltip(stop_after_epoch_check, "Stop After Epoch: Complete the current epoch (full dataset pass) then stop gracefully.\nEpoch = one complete pass through ALL blocks in the entire dataset. Training will continue until all blocks are visited, then stop automatically.")
        add_tooltip(clear_btn, "Clear the training output log.")
    except Exception:
        pass


def build_progress_bar(panel: HRMTrainingPanel, parent: any) -> None:
    """Build progress bar section."""
    p = ttk.Frame(parent)
    p.pack(fill="x", pady=(6, 0))
    ttk.Label(p, text="Progress:").pack(side="left")
    panel.progress = ttk.Progressbar(p, orient="horizontal", mode="determinate", length=240, maximum=100)
    panel.progress.pack(side="left", fill="x", expand=True, padx=(6, 6))
    panel.progress_lbl = ttk.Label(p, text="idle")
    panel.progress_lbl.pack(side="left")
    
    try:
        from ..tooltips import add_tooltip
        add_tooltip(panel.progress, "Relative progress (if determinable) or indeterminate spinner during startup.")
    except Exception:
        pass


def build_log_output(panel: HRMTrainingPanel, parent: any) -> None:
    """Build log output Text widget with scrollbar."""
    from .theme_utils import get_theme_colors
    
    log_frame = ttk.Frame(parent)
    log_frame.pack(fill="both", expand=True, pady=(8, 0))
    
    log_scrollbar = ttk.Scrollbar(log_frame)
    log_scrollbar.pack(side="right", fill="y")
    
    # Apply theme-aware colors to Text widget
    theme_colors = get_theme_colors()
    panel.log = tk.Text(
        log_frame, 
        height=8, 
        wrap="word", 
        yscrollcommand=log_scrollbar.set,
        bg=theme_colors["bg"],
        fg=theme_colors["fg"],
        selectbackground=theme_colors["selectbg"],
        selectforeground=theme_colors["selectfg"],
        insertbackground=theme_colors["insertbg"]
    )
    panel.log.pack(side="left", fill="both", expand=True)
    
    log_scrollbar.config(command=panel.log.yview)
    
    try:
        from ..tooltips import add_tooltip
        add_tooltip(panel.log, "Live CLI output and training logs.")
    except Exception:
        pass
