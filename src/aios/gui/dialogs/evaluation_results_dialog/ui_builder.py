"""UI construction utilities for evaluation results dialog."""

from __future__ import annotations

import json
import tkinter as tk
from tkinter import ttk
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from aios.core.evaluation import EvaluationResult

from .chart_utils import is_matplotlib_available


def build_dialog_ui(
    dialog: tk.Toplevel,  # type: ignore[name-defined]
    result: "EvaluationResult",
) -> dict[str, Any]:
    """Build the evaluation results dialog UI.
    
    Args:
        dialog: Parent dialog window
        result: EvaluationResult to display
        
    Returns:
        Dict of UI widgets that need to be populated: {
            'model_label': Label,
            'status_label': Label,
            'score_label': Label,
            'duration_label': Label,
            'tasks_label': Label,
            'scores_tree': Treeview,
            'charts_container': Frame (if matplotlib available),
        }
    """
    widgets = {}
    
    # Main container
    main_frame = ttk.Frame(dialog, padding=10)
    main_frame.pack(fill="both", expand=True)
    
    # Top: Summary section
    summary_frame = ttk.LabelFrame(main_frame, text="Summary", padding=10)
    summary_frame.pack(fill="x", pady=(0, 10))
    
    # Summary grid
    summary_grid = ttk.Frame(summary_frame)
    summary_grid.pack(fill="x")
    
    # Row 1: Model and Status
    ttk.Label(summary_grid, text="Model:", font=("TkDefaultFont", 9, "bold")).grid(
        row=0, column=0, sticky="w", padx=(0, 5)
    )
    widgets['model_label'] = ttk.Label(summary_grid, text="")
    widgets['model_label'].grid(row=0, column=1, sticky="w", padx=(0, 20))
    
    ttk.Label(summary_grid, text="Status:", font=("TkDefaultFont", 9, "bold")).grid(
        row=0, column=2, sticky="w", padx=(0, 5)
    )
    widgets['status_label'] = ttk.Label(summary_grid, text="")
    widgets['status_label'].grid(row=0, column=3, sticky="w")
    
    # Row 2: Overall Score and Duration
    ttk.Label(summary_grid, text="Overall Score:", font=("TkDefaultFont", 9, "bold")).grid(
        row=1, column=0, sticky="w", padx=(0, 5), pady=(5, 0)
    )
    widgets['score_label'] = ttk.Label(
        summary_grid, text="", font=("TkDefaultFont", 11, "bold"), foreground="blue"
    )
    widgets['score_label'].grid(row=1, column=1, sticky="w", padx=(0, 20), pady=(5, 0))
    
    ttk.Label(summary_grid, text="Duration:", font=("TkDefaultFont", 9, "bold")).grid(
        row=1, column=2, sticky="w", padx=(0, 5), pady=(5, 0)
    )
    widgets['duration_label'] = ttk.Label(summary_grid, text="")
    widgets['duration_label'].grid(row=1, column=3, sticky="w", pady=(5, 0))
    
    # Row 3: Tasks Evaluated
    ttk.Label(summary_grid, text="Tasks Evaluated:", font=("TkDefaultFont", 9, "bold")).grid(
        row=2, column=0, sticky="w", padx=(0, 5), pady=(5, 0)
    )
    widgets['tasks_label'] = ttk.Label(summary_grid, text="")
    widgets['tasks_label'].grid(row=2, column=1, columnspan=3, sticky="w", pady=(5, 0))
    
    # Middle: Notebook with tabs
    notebook = ttk.Notebook(main_frame)
    notebook.pack(fill="both", expand=True, pady=(0, 10))
    
    # Tab 1: Scores Table
    scores_frame = ttk.Frame(notebook)
    notebook.add(scores_frame, text="Detailed Scores")
    
    # Scores TreeView
    tree_frame = ttk.Frame(scores_frame)
    tree_frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    # Scrollbars
    vsb = ttk.Scrollbar(tree_frame, orient="vertical")
    hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
    
    widgets['scores_tree'] = ttk.Treeview(
        tree_frame,
        columns=("metric", "value", "stderr"),
        show="tree headings",
        yscrollcommand=vsb.set,
        xscrollcommand=hsb.set,
    )
    
    vsb.config(command=widgets['scores_tree'].yview)
    hsb.config(command=widgets['scores_tree'].xview)
    
    # Configure columns
    widgets['scores_tree'].heading("#0", text="Benchmark")
    widgets['scores_tree'].heading("metric", text="Metric")
    widgets['scores_tree'].heading("value", text="Score")
    widgets['scores_tree'].heading("stderr", text="Std Error")
    
    widgets['scores_tree'].column("#0", width=200, minwidth=150)
    widgets['scores_tree'].column("metric", width=150, minwidth=100)
    widgets['scores_tree'].column("value", width=100, minwidth=80, anchor="center")
    widgets['scores_tree'].column("stderr", width=100, minwidth=80, anchor="center")
    
    # Pack tree and scrollbars
    widgets['scores_tree'].grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    
    tree_frame.grid_rowconfigure(0, weight=1)
    tree_frame.grid_columnconfigure(0, weight=1)
    
    # Tab 2: Charts
    if is_matplotlib_available():
        charts_frame = ttk.Frame(notebook)
        notebook.add(charts_frame, text="Visualizations")
        
        # Chart container
        widgets['charts_container'] = ttk.Frame(charts_frame)
        widgets['charts_container'].pack(fill="both", expand=True)
    else:
        # Show message about matplotlib
        no_charts_frame = ttk.Frame(notebook)
        notebook.add(no_charts_frame, text="Visualizations")
        ttk.Label(
            no_charts_frame,
            text="Matplotlib not available.\nInstall with: pip install matplotlib",
            justify="center",
        ).pack(expand=True)
    
    # Tab 3: Raw JSON
    json_frame = ttk.Frame(notebook)
    notebook.add(json_frame, text="Raw Results")
    
    # JSON text widget
    json_text_frame = ttk.Frame(json_frame)
    json_text_frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    json_vsb = ttk.Scrollbar(json_text_frame, orient="vertical")
    json_hsb = ttk.Scrollbar(json_text_frame, orient="horizontal")
    
    json_text = tk.Text(
        json_text_frame,
        wrap="none",
        yscrollcommand=json_vsb.set,
        xscrollcommand=json_hsb.set,
        font=("Consolas", 9),
    )
    
    json_vsb.config(command=json_text.yview)
    json_hsb.config(command=json_text.xview)
    
    json_text.grid(row=0, column=0, sticky="nsew")
    json_vsb.grid(row=0, column=1, sticky="ns")
    json_hsb.grid(row=1, column=0, sticky="ew")
    
    json_text_frame.grid_rowconfigure(0, weight=1)
    json_text_frame.grid_columnconfigure(0, weight=1)
    
    # Populate JSON
    json_text.insert("1.0", json.dumps(result.raw_results, indent=2))
    json_text.config(state="disabled")
    
    return widgets


def add_dialog_buttons(
    dialog: tk.Toplevel,  # type: ignore[name-defined]
    main_frame: ttk.Frame,
    export_csv_callback,
    export_json_callback,
    export_html_callback,
) -> None:
    """Add export and close buttons to dialog.
    
    Args:
        dialog: Parent dialog window
        main_frame: Main container frame
        export_csv_callback: Callback for CSV export
        export_json_callback: Callback for JSON export
        export_html_callback: Callback for HTML export
    """
    # Bottom: Buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill="x")
    
    # Export buttons
    ttk.Button(
        button_frame, text="Export CSV", command=export_csv_callback
    ).pack(side="left", padx=(0, 5))
    
    ttk.Button(
        button_frame, text="Export JSON", command=export_json_callback
    ).pack(side="left", padx=(0, 5))
    
    ttk.Button(
        button_frame, text="Export HTML Report", command=export_html_callback
    ).pack(side="left", padx=(0, 5))
    
    # Close button
    ttk.Button(
        button_frame, text="Close", command=dialog.destroy
    ).pack(side="right")
