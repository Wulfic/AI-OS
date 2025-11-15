"""Evaluation history viewer dialog."""

from __future__ import annotations

# Import safe variable wrappers
from ..utils import safe_variables

import logging
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk
from typing import Any, Callable, Optional

from aios.core.evaluation import EvaluationHistory
from aios.gui.utils.model_display import get_model_display_name
from aios.gui.utils.theme_utils import apply_theme_to_toplevel, get_spacing_multiplier

logger = logging.getLogger(__name__)


class EvaluationHistoryDialog(tk.Toplevel):  # type: ignore[misc]
    """Dialog for viewing and managing evaluation history."""
    
    def __init__(
        self,
        parent: Any,
        history: EvaluationHistory,
        on_view_details: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Initialize the history dialog.
        
        Args:
            parent: Parent window
            history: EvaluationHistory instance
            on_view_details: Callback when user wants to view details (receives eval_id)
        """
        super().__init__(parent)
        
        logger.info("Opening evaluation history dialog")
        
        self.history = history
        self.on_view_details = on_view_details
        
        # Configure window
        self.title("Evaluation History")
        self.geometry("1100x600")
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        
        # Apply theme to this dialog
        apply_theme_to_toplevel(self)
        
        # Build UI
        self._build_ui()
        
        # Load initial data
        self._load_history()
        
        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
    
    def _build_ui(self) -> None:
        """Build the dialog UI."""
        # Get spacing multiplier for current theme
        spacing = get_spacing_multiplier()
        
        # Main container
        padding = int(10 * spacing)
        main_frame = ttk.Frame(self, padding=padding)
        main_frame.pack(fill="both", expand=True)
        
        # Top: Filters and statistics
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill="x", pady=(0, int(10 * spacing)))
        
        # Left: Filters
        filter_frame = ttk.LabelFrame(top_frame, text="Filters", padding=int(10 * spacing))
        filter_frame.pack(side="left", fill="x", expand=True, padx=(0, int(10 * spacing)))
        
        filter_grid = ttk.Frame(filter_frame)
        filter_grid.pack(fill="x")
        
        # Model filter
        ttk.Label(filter_grid, text="Model:").grid(row=0, column=0, sticky="w", padx=(0, int(5 * spacing)))
        self.model_filter_var = safe_variables.StringVar()
        self.model_filter_entry = ttk.Entry(filter_grid, textvariable=self.model_filter_var, width=20)
        self.model_filter_entry.grid(row=0, column=1, sticky="w", padx=(0, int(10 * spacing)))
        
        # Status filter
        ttk.Label(filter_grid, text="Status:").grid(row=0, column=2, sticky="w", padx=(0, int(5 * spacing)))
        self.status_filter_var = safe_variables.StringVar(value="all")
        status_combo = ttk.Combobox(
            filter_grid,
            textvariable=self.status_filter_var,
            values=["all", "completed", "failed", "cancelled"],
            state="readonly",
            width=12,
        )
        status_combo.grid(row=0, column=3, sticky="w", padx=(0, int(10 * spacing)))
        
        # Limit
        ttk.Label(filter_grid, text="Limit:").grid(row=0, column=4, sticky="w", padx=(0, int(5 * spacing)))
        self.limit_var = safe_variables.StringVar(value="50")
        limit_spin = ttk.Spinbox(filter_grid, textvariable=self.limit_var, from_=10, to=500, width=8)
        limit_spin.grid(row=0, column=5, sticky="w", padx=(0, int(10 * spacing)))
        
        # Apply button
        ttk.Button(
            filter_grid, text="Apply", command=self._load_history
        ).grid(row=0, column=6, sticky="w")
        
        # Right: Statistics
        stats_frame = ttk.LabelFrame(top_frame, text="Statistics", padding=int(10 * spacing))
        stats_frame.pack(side="right", fill="x")
        
        self.stats_label = ttk.Label(stats_frame, text="Loading...", justify="left")
        self.stats_label.pack()
        
        # Middle: History TreeView
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill="both", expand=True, pady=(0, int(10 * spacing)))
        
        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        
        self.tree = ttk.Treeview(
            tree_frame,
            columns=("model", "score", "status", "tasks", "duration", "date"),
            show="tree headings",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
        )
        
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)
        
        # Configure columns
        self.tree.heading("#0", text="ID")
        self.tree.heading("model", text="Model")
        self.tree.heading("score", text="Score")
        self.tree.heading("status", text="Status")
        self.tree.heading("tasks", text="Tasks")
        self.tree.heading("duration", text="Duration")
        self.tree.heading("date", text="Date")
        
        self.tree.column("#0", width=60, minwidth=50)
        self.tree.column("model", width=180, minwidth=100)
        self.tree.column("score", width=80, minwidth=60, anchor="center")
        self.tree.column("status", width=100, minwidth=80, anchor="center")
        self.tree.column("tasks", width=250, minwidth=150)
        self.tree.column("duration", width=80, minwidth=60, anchor="center")
        self.tree.column("date", width=150, minwidth=120)
        
        # Pack tree and scrollbars
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Double-click to view details
        self.tree.bind("<Double-Button-1>", self._on_double_click)
        
        # Bottom: Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x")
        
        padx_val = (0, int(5 * spacing))
        # Left buttons
        ttk.Button(
            button_frame, text="View Details", command=self._on_view_selected
        ).pack(side="left", padx=padx_val)
        
        ttk.Button(
            button_frame, text="View Samples", command=self._on_view_samples
        ).pack(side="left", padx=padx_val)
        
        ttk.Button(
            button_frame, text="Compare Selected", command=self._on_compare_selected
        ).pack(side="left", padx=padx_val)
        
        ttk.Button(
            button_frame, text="Delete Selected", command=self._on_delete_selected
        ).pack(side="left", padx=padx_val)
        
        ttk.Button(
            button_frame, text="Refresh", command=self._load_history
        ).pack(side="left", padx=padx_val)
        
        # Right buttons
        ttk.Button(
            button_frame, text="Close", command=self.destroy
        ).pack(side="right")
    
    def _load_history(self) -> None:
        """Load evaluation history."""
        # Clear existing
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Get filters
        model_filter = self.model_filter_var.get().strip()
        status_filter = self.status_filter_var.get()
        if status_filter == "all":
            status_filter = None
        
        try:
            limit = int(self.limit_var.get())
        except ValueError:
            limit = 50
        
        logger.debug(f"Loading evaluation history: model_filter={model_filter or 'none'}, status_filter={status_filter or 'all'}, limit={limit}")
        
        # Load evaluations
        try:
            evaluations = self.history.get_recent_evaluations(
                limit=limit,
                model_name=model_filter if model_filter else None,
                status=status_filter,
            )
            
            logger.info(f"Loaded {len(evaluations)} evaluation history entries")
            logger.debug(f"Evaluation IDs: {[e['id'] for e in evaluations]}")
            
            for eval_data in evaluations:
                # Format data
                eval_id = eval_data["id"]
                model = eval_data["model_name"]
                model_display = get_model_display_name(model)
                
                score = eval_data.get("overall_score", 0.0)
                score_str = f"{score:.2%}" if score else "N/A"
                
                status = eval_data["status"]
                
                tasks = eval_data.get("tasks", [])
                if isinstance(tasks, list):
                    tasks_str = ", ".join(tasks[:3])
                    if len(tasks) > 3:
                        tasks_str += f" (+{len(tasks) - 3})"
                else:
                    tasks_str = str(tasks)
                
                duration = eval_data.get("duration", 0)
                if duration:
                    if duration < 60:
                        duration_str = f"{duration:.1f}s"
                    elif duration < 3600:
                        duration_str = f"{duration/60:.1f}m"
                    else:
                        duration_str = f"{duration/3600:.1f}h"
                else:
                    duration_str = "N/A"
                
                created_at = eval_data.get("created_at", 0)
                date_str = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M") if created_at else "N/A"
                
                # Add to tree
                self.tree.insert(
                    "",
                    "end",
                    text=str(eval_id),
                    values=(model_display, score_str, status, tasks_str, duration_str, date_str),
                )
            
            # Update statistics
            self._update_statistics()
        
        except Exception as e:
            logger.error(f"Failed to load evaluation history: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to load history:\n{e}")
    
    def _update_statistics(self) -> None:
        """Update statistics display."""
        try:
            stats = self.history.get_statistics()
            
            stats_text = (
                f"Total: {stats['total_evaluations']} | "
                f"Models: {stats['unique_models']} | "
                f"Avg Score: {stats['average_score']:.1%} | "
                f"Recent (7d): {stats['recent_count']}"
            )
            
            self.stats_label.config(text=stats_text)
        
        except Exception:
            self.stats_label.config(text="Statistics unavailable")
    
    def _on_double_click(self, event: Any) -> None:
        """Handle double-click on tree item."""
        self._on_view_selected()
    
    def _on_view_selected(self) -> None:
        """View details of selected evaluation."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an evaluation to view.")
            return
        
        # Get evaluation ID
        item = selection[0]
        eval_id_str = self.tree.item(item, "text")
        
        try:
            eval_id = int(eval_id_str)
            
            logger.info(f"Viewing evaluation details: eval_id={eval_id}")
            
            if self.on_view_details:
                self.on_view_details(eval_id)
            else:
                # Show basic info
                eval_data = self.history.get_evaluation(eval_id)
                if eval_data:
                    info = (
                        f"Model: {eval_data['model_name']}\n"
                        f"Score: {eval_data['overall_score']:.2%}\n"
                        f"Status: {eval_data['status']}\n"
                        f"Tasks: {', '.join(eval_data.get('tasks', []))}\n"
                        f"Date: {datetime.fromtimestamp(eval_data['created_at']).strftime('%Y-%m-%d %H:%M')}"
                    )
                    messagebox.showinfo("Evaluation Details", info)
        
        except Exception as e:
            logger.error(f"Failed to view evaluation details: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to view details:\n{e}")
    
    def _on_view_samples(self) -> None:
        """View samples from selected evaluation."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an evaluation to view samples.")
            return
        
        # Get evaluation ID
        item = selection[0]
        eval_id_str = self.tree.item(item, "text")
        
        try:
            eval_id = int(eval_id_str)
            
            logger.info(f"Viewing evaluation samples: eval_id={eval_id}")
            
            # Get evaluation data
            eval_data = self.history.get_evaluation(eval_id)
            if not eval_data:
                logger.error(f"Evaluation not found: eval_id={eval_id}")
                messagebox.showerror("Error", "Evaluation not found.")
                return
            
            # Check if samples are available
            samples_path = eval_data.get("samples_path", "")
            if not samples_path:
                logger.warning(f"No samples available for evaluation {eval_id}")
                messagebox.showinfo(
                    "No Samples",
                    "This evaluation does not have logged samples.\n\n"
                    "To log samples, enable the 'Log samples' option before running an evaluation."
                )
                return
            
            # Get tasks
            tasks = eval_data.get("tasks", [])
            if isinstance(tasks, str):
                tasks = tasks.split(",")
            
            logger.debug(f"Opening samples dialog for {len(tasks)} tasks: {tasks}")
            
            # Open samples viewer dialog
            from aios.gui.dialogs.evaluation_samples_dialog import EvaluationSamplesDialog
            EvaluationSamplesDialog(self, samples_path, tasks)
        
        except Exception as e:
            logger.error(f"Failed to view samples: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to view samples:\n{e}")
    
    def _on_compare_selected(self) -> None:
        """Compare selected evaluations."""
        selection = self.tree.selection()
        if len(selection) < 2:
            messagebox.showwarning(
                "Select Multiple",
                "Please select at least 2 evaluations to compare."
            )
            return
        
        try:
            # Get evaluation IDs
            eval_ids = []
            for item in selection:
                eval_id_str = self.tree.item(item, "text")
                eval_ids.append(int(eval_id_str))
            
            logger.info(f"Comparing {len(eval_ids)} evaluations: {eval_ids}")
            
            # Get comparison
            comparison = self.history.compare_evaluations(eval_ids)
            
            if not comparison:
                logger.warning(f"No comparison data available for evaluations: {eval_ids}")
                messagebox.showwarning("No Data", "No comparison data available.")
                return
            
            logger.debug(f"Comparison benchmarks: {list(comparison.get('benchmarks', {}).keys())}")
            
            # Build comparison text
            lines = ["Evaluation Comparison\n" + "=" * 50 + "\n"]
            
            lines.append("Models:\n")
            for i, eval_info in enumerate(comparison["evaluations"], 1):
                lines.append(
                    f"  {i}. {eval_info['model_name']} "
                    f"(Score: {eval_info['overall_score']:.2%})\n"
                )
            
            lines.append("\nBenchmark Comparison:\n")
            for benchmark, scores in comparison["benchmarks"].items():
                lines.append(f"\n{benchmark}:\n")
                for i, score_data in enumerate(scores, 1):
                    if score_data:
                        lines.append(
                            f"  Model {i}: {score_data['score']:.2%} "
                            f"({score_data['metric']})\n"
                        )
                    else:
                        lines.append(f"  Model {i}: N/A\n")
            
            # Show in messagebox (could be a separate dialog in the future)
            comparison_text = "".join(lines)
            
            # Create comparison dialog
            comp_dialog = tk.Toplevel(self)
            comp_dialog.title("Evaluation Comparison")
            comp_dialog.geometry("600x500")
            comp_dialog.transient(self)
            
            text_widget = tk.Text(comp_dialog, wrap="word", font=("Consolas", 10))
            text_widget.pack(fill="both", expand=True, padx=10, pady=10)
            text_widget.insert("1.0", comparison_text)
            text_widget.config(state="disabled")
            
            ttk.Button(comp_dialog, text="Close", command=comp_dialog.destroy).pack(pady=5)
        
        except Exception as e:
            logger.error(f"Failed to compare evaluations: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to compare:\n{e}")
    
    def _on_delete_selected(self) -> None:
        """Delete selected evaluations."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select evaluations to delete.")
            return
        
        # Confirm
        count = len(selection)
        if not messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete {count} evaluation(s)?\n"
            "This action cannot be undone."
        ):
            logger.debug(f"User cancelled deletion of {count} evaluations")
            return
        
        try:
            eval_ids = []
            for item in selection:
                eval_id_str = self.tree.item(item, "text")
                eval_ids.append(int(eval_id_str))
            
            logger.info(f"Deleting {count} evaluations: {eval_ids}")
            
            deleted = 0
            for eval_id in eval_ids:
                if self.history.delete_evaluation(eval_id):
                    deleted += 1
            
            logger.info(f"Successfully deleted {deleted} of {count} evaluations")
            messagebox.showinfo("Success", f"Deleted {deleted} evaluation(s).")
            self._load_history()
        
        except Exception as e:
            logger.error(f"Failed to delete evaluations: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to delete:\n{e}")
