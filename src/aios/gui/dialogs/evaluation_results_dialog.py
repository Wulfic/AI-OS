"""Detailed evaluation results viewer dialog."""

from __future__ import annotations

import csv
import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import numpy as np

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # type: ignore[assignment]
    from matplotlib.figure import Figure  # type: ignore[assignment]
    import numpy as np  # type: ignore[assignment]
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Figure = None  # type: ignore[assignment,misc]
    FigureCanvasTkAgg = None  # type: ignore[assignment,misc]
    np = None  # type: ignore[assignment]

from aios.core.evaluation import EvaluationResult


class EvaluationResultsDialog(tk.Toplevel):  # type: ignore[misc]
    """Dialog for viewing detailed evaluation results with charts and export."""
    
    def __init__(
        self,
        parent: Any,
        result: EvaluationResult,
        model_name: str = "",
    ) -> None:
        """Initialize the results dialog.
        
        Args:
            parent: Parent window
            result: EvaluationResult to display
            model_name: Name of evaluated model
        """
        super().__init__(parent)
        
        self.result = result
        self.model_name = model_name
        
        # Configure window
        self.title(f"Evaluation Results: {model_name or 'Unknown Model'}")
        self.geometry("1000x700")
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        
        # Build UI
        self._build_ui()
        
        # Populate data
        self._populate_summary()
        self._populate_scores_tree()
        if MATPLOTLIB_AVAILABLE:
            self._create_charts()
        
        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
    
    def _build_ui(self) -> None:
        """Build the dialog UI."""
        # Main container
        main_frame = ttk.Frame(self, padding=10)
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
        self.model_label = ttk.Label(summary_grid, text="")
        self.model_label.grid(row=0, column=1, sticky="w", padx=(0, 20))
        
        ttk.Label(summary_grid, text="Status:", font=("TkDefaultFont", 9, "bold")).grid(
            row=0, column=2, sticky="w", padx=(0, 5)
        )
        self.status_label = ttk.Label(summary_grid, text="")
        self.status_label.grid(row=0, column=3, sticky="w")
        
        # Row 2: Overall Score and Duration
        ttk.Label(summary_grid, text="Overall Score:", font=("TkDefaultFont", 9, "bold")).grid(
            row=1, column=0, sticky="w", padx=(0, 5), pady=(5, 0)
        )
        self.score_label = ttk.Label(
            summary_grid, text="", font=("TkDefaultFont", 11, "bold"), foreground="blue"
        )
        self.score_label.grid(row=1, column=1, sticky="w", padx=(0, 20), pady=(5, 0))
        
        ttk.Label(summary_grid, text="Duration:", font=("TkDefaultFont", 9, "bold")).grid(
            row=1, column=2, sticky="w", padx=(0, 5), pady=(5, 0)
        )
        self.duration_label = ttk.Label(summary_grid, text="")
        self.duration_label.grid(row=1, column=3, sticky="w", pady=(5, 0))
        
        # Row 3: Tasks Evaluated
        ttk.Label(summary_grid, text="Tasks Evaluated:", font=("TkDefaultFont", 9, "bold")).grid(
            row=2, column=0, sticky="w", padx=(0, 5), pady=(5, 0)
        )
        self.tasks_label = ttk.Label(summary_grid, text="")
        self.tasks_label.grid(row=2, column=1, columnspan=3, sticky="w", pady=(5, 0))
        
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
        
        self.scores_tree = ttk.Treeview(
            tree_frame,
            columns=("metric", "value", "stderr"),
            show="tree headings",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
        )
        
        vsb.config(command=self.scores_tree.yview)
        hsb.config(command=self.scores_tree.xview)
        
        # Configure columns
        self.scores_tree.heading("#0", text="Benchmark")
        self.scores_tree.heading("metric", text="Metric")
        self.scores_tree.heading("value", text="Score")
        self.scores_tree.heading("stderr", text="Std Error")
        
        self.scores_tree.column("#0", width=200, minwidth=150)
        self.scores_tree.column("metric", width=150, minwidth=100)
        self.scores_tree.column("value", width=100, minwidth=80, anchor="center")
        self.scores_tree.column("stderr", width=100, minwidth=80, anchor="center")
        
        # Pack tree and scrollbars
        self.scores_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Tab 2: Charts
        if MATPLOTLIB_AVAILABLE:
            charts_frame = ttk.Frame(notebook)
            notebook.add(charts_frame, text="Visualizations")
            
            # Chart container
            self.charts_container = ttk.Frame(charts_frame)
            self.charts_container.pack(fill="both", expand=True)
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
        
        self.json_text = tk.Text(
            json_text_frame,
            wrap="none",
            yscrollcommand=json_vsb.set,
            xscrollcommand=json_hsb.set,
            font=("Consolas", 9),
        )
        
        json_vsb.config(command=self.json_text.yview)
        json_hsb.config(command=self.json_text.xview)
        
        self.json_text.grid(row=0, column=0, sticky="nsew")
        json_vsb.grid(row=0, column=1, sticky="ns")
        json_hsb.grid(row=1, column=0, sticky="ew")
        
        json_text_frame.grid_rowconfigure(0, weight=1)
        json_text_frame.grid_columnconfigure(0, weight=1)
        
        # Populate JSON
        self.json_text.insert("1.0", json.dumps(self.result.raw_results, indent=2))
        self.json_text.config(state="disabled")
        
        # Bottom: Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x")
        
        # Export buttons
        ttk.Button(
            button_frame, text="Export CSV", command=self._export_csv
        ).pack(side="left", padx=(0, 5))
        
        ttk.Button(
            button_frame, text="Export JSON", command=self._export_json
        ).pack(side="left", padx=(0, 5))
        
        ttk.Button(
            button_frame, text="Export HTML Report", command=self._export_html
        ).pack(side="left", padx=(0, 5))
        
        # Close button
        ttk.Button(
            button_frame, text="Close", command=self.destroy
        ).pack(side="right")
    
    def _populate_summary(self) -> None:
        """Populate summary section."""
        self.model_label.config(text=self.model_name or "Unknown")
        self.status_label.config(text=self.result.status.upper())
        self.score_label.config(text=f"{self.result.overall_score:.2%}")
        self.duration_label.config(text=self.result.duration_str)
        
        # Tasks
        task_names = list(self.result.benchmark_scores.keys())
        tasks_text = f"{len(task_names)} tasks: {', '.join(task_names[:5])}"
        if len(task_names) > 5:
            tasks_text += f" ... (+{len(task_names) - 5} more)"
        self.tasks_label.config(text=tasks_text)
    
    def _populate_scores_tree(self) -> None:
        """Populate scores TreeView."""
        # Clear existing
        for item in self.scores_tree.get_children():
            self.scores_tree.delete(item)
        
        # Add benchmarks
        for task_name, task_data in sorted(self.result.benchmark_scores.items()):
            # Add task as parent
            task_id = self.scores_tree.insert("", "end", text=task_name, values=("", "", ""))
            
            # Add metrics as children
            scores_dict = task_data.get("scores", {})
            raw_data = task_data.get("raw", {})
            
            for metric_name, score_value in sorted(scores_dict.items()):
                # Try to get stderr if available
                stderr = ""
                if metric_name in raw_data:
                    stderr_val = raw_data[metric_name]
                    if isinstance(stderr_val, dict) and "stderr" in stderr_val:
                        stderr = f"{stderr_val['stderr']:.4f}"
                elif f"{metric_name}_stderr" in raw_data:
                    stderr = f"{raw_data[f'{metric_name}_stderr']:.4f}"
                
                # Format score
                if isinstance(score_value, float):
                    score_str = f"{score_value:.2%}" if score_value <= 1.0 else f"{score_value:.4f}"
                else:
                    score_str = str(score_value)
                
                self.scores_tree.insert(
                    task_id, "end", text="", values=(metric_name, score_str, stderr)
                )
            
            # Expand all tasks
            self.scores_tree.item(task_id, open=True)
    
    def _create_charts(self) -> None:
        """Create matplotlib charts."""
        if not MATPLOTLIB_AVAILABLE or Figure is None or np is None or FigureCanvasTkAgg is None:
            return
        
        # Create figure with subplots
        fig = Figure(figsize=(10, 8), dpi=100)
        
        # Extract data
        task_names = []
        task_scores = []
        
        for task_name, task_data in sorted(self.result.benchmark_scores.items()):
            scores_dict = task_data.get("scores", {})
            if scores_dict:
                # Use first score as primary metric
                primary_score = list(scores_dict.values())[0]
                task_names.append(task_name)
                task_scores.append(float(primary_score) * 100)  # Convert to percentage
        
        if not task_names:
            return
        
        # Chart 1: Bar chart (top half)
        ax1 = fig.add_subplot(2, 1, 1)
        bars = ax1.bar(range(len(task_names)), task_scores, color='steelblue', alpha=0.8)
        ax1.set_xlabel('Benchmark', fontsize=10)
        ax1.set_ylabel('Score (%)', fontsize=10)
        ax1.set_title('Benchmark Scores Comparison', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(task_names)))
        ax1.set_xticklabels(task_names, rotation=45, ha='right', fontsize=8)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 1,
                f'{height:.1f}%',
                ha='center',
                va='bottom',
                fontsize=8,
            )
        
        # Chart 2: Radar chart (bottom half) - only if 3+ benchmarks
        if len(task_names) >= 3:
            ax2 = fig.add_subplot(2, 1, 2, projection='polar')
            
            # Prepare data for radar chart
            angles = np.linspace(0, 2 * np.pi, len(task_names), endpoint=False).tolist()
            scores_radar = task_scores + [task_scores[0]]  # Close the circle
            angles_radar = angles + [angles[0]]
            
            # Plot
            ax2.plot(angles_radar, scores_radar, 'o-', linewidth=2, color='steelblue')
            ax2.fill(angles_radar, scores_radar, alpha=0.25, color='steelblue')
            ax2.set_xticks(angles)
            ax2.set_xticklabels(task_names, fontsize=8)
            ax2.set_ylim(0, 100)
            ax2.set_ylabel('Score (%)', fontsize=10)
            ax2.set_title('Performance Radar Chart', fontsize=12, fontweight='bold', pad=20)
            ax2.grid(True)
        
        fig.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.charts_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _export_csv(self) -> None:
        """Export results to CSV."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"eval_results_{self.model_name.replace('/', '_')}.csv",
        )
        
        if not filepath:
            return
        
        try:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow(["Model", self.model_name])
                writer.writerow(["Overall Score", f"{self.result.overall_score:.4f}"])
                writer.writerow(["Duration", self.result.duration_str])
                writer.writerow(["Status", self.result.status])
                writer.writerow([])
                
                # Benchmark scores
                writer.writerow(["Benchmark", "Metric", "Score", "Std Error"])
                
                for task_name, task_data in sorted(self.result.benchmark_scores.items()):
                    scores_dict = task_data.get("scores", {})
                    raw_data = task_data.get("raw", {})
                    
                    for metric_name, score_value in sorted(scores_dict.items()):
                        stderr = ""
                        if f"{metric_name}_stderr" in raw_data:
                            stderr = f"{raw_data[f'{metric_name}_stderr']:.6f}"
                        
                        writer.writerow([
                            task_name,
                            metric_name,
                            f"{score_value:.6f}",
                            stderr,
                        ])
            
            messagebox.showinfo("Export Successful", f"Results exported to:\n{filepath}")
        
        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to export CSV:\n{e}")
    
    def _export_json(self) -> None:
        """Export results to JSON."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"eval_results_{self.model_name.replace('/', '_')}.json",
        )
        
        if not filepath:
            return
        
        try:
            export_data = {
                "model": self.model_name,
                "overall_score": self.result.overall_score,
                "duration_seconds": self.result.duration,
                "duration_str": self.result.duration_str,
                "status": self.result.status,
                "error_message": self.result.error_message,
                "benchmark_scores": self.result.benchmark_scores,
                "raw_results": self.result.raw_results,
            }
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)
            
            messagebox.showinfo("Export Successful", f"Results exported to:\n{filepath}")
        
        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to export JSON:\n{e}")
    
    def _export_html(self) -> None:
        """Export results to HTML report."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")],
            initialfile=f"eval_report_{self.model_name.replace('/', '_')}.html",
        )
        
        if not filepath:
            return
        
        try:
            html = self._generate_html_report()
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html)
            
            messagebox.showinfo("Export Successful", f"Report exported to:\n{filepath}")
        
        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to export HTML:\n{e}")
    
    def _generate_html_report(self) -> str:
        """Generate HTML report."""
        # Build benchmark rows
        benchmark_rows = []
        for task_name, task_data in sorted(self.result.benchmark_scores.items()):
            scores_dict = task_data.get("scores", {})
            
            for metric_name, score_value in sorted(scores_dict.items()):
                score_pct = f"{score_value:.2%}" if score_value <= 1.0 else f"{score_value:.4f}"
                benchmark_rows.append(
                    f"<tr><td>{task_name}</td><td>{metric_name}</td><td>{score_pct}</td></tr>"
                )
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Report: {self.model_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-item {{
            padding: 15px;
            background: #f9f9f9;
            border-left: 4px solid #4CAF50;
            border-radius: 4px;
        }}
        .summary-item label {{
            display: block;
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .summary-item value {{
            display: block;
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #4CAF50;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Evaluation Report</h1>
        
        <div class="summary">
            <div class="summary-item">
                <label>Model</label>
                <value>{self.model_name}</value>
            </div>
            <div class="summary-item">
                <label>Overall Score</label>
                <value>{self.result.overall_score:.2%}</value>
            </div>
            <div class="summary-item">
                <label>Status</label>
                <value>{self.result.status.upper()}</value>
            </div>
            <div class="summary-item">
                <label>Duration</label>
                <value>{self.result.duration_str}</value>
            </div>
        </div>
        
        <h2>📊 Benchmark Scores</h2>
        <table>
            <thead>
                <tr>
                    <th>Benchmark</th>
                    <th>Metric</th>
                    <th>Score</th>
                </tr>
            </thead>
            <tbody>
                {"".join(benchmark_rows)}
            </tbody>
        </table>
        
        <div class="footer">
            Generated by AI-OS Evaluation System | {self.result.duration_str} evaluation time
        </div>
    </div>
</body>
</html>"""
        
        return html
