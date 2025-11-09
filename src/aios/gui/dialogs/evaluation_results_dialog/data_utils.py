"""Data population utilities for evaluation results dialog."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tkinter import ttk
    from aios.core.evaluation import EvaluationResult


def populate_summary(
    result: "EvaluationResult",
    model_name: str,
    model_label: "ttk.Label",
    status_label: "ttk.Label",
    score_label: "ttk.Label",
    duration_label: "ttk.Label",
    tasks_label: "ttk.Label",
) -> None:
    """Populate summary section with evaluation data.
    
    Args:
        result: EvaluationResult to display
        model_name: Name of evaluated model
        model_label: Label widget for model name
        status_label: Label widget for status
        score_label: Label widget for overall score
        duration_label: Label widget for duration
        tasks_label: Label widget for task list
    """
    model_label.config(text=model_name or "Unknown")
    status_label.config(text=result.status.upper())
    score_label.config(text=f"{result.overall_score:.2%}")
    duration_label.config(text=result.duration_str)
    
    # Tasks
    task_names = list(result.benchmark_scores.keys())
    tasks_text = f"{len(task_names)} tasks: {', '.join(task_names[:5])}"
    if len(task_names) > 5:
        tasks_text += f" ... (+{len(task_names) - 5} more)"
    tasks_label.config(text=tasks_text)


def populate_scores_tree(result: "EvaluationResult", scores_tree: "ttk.Treeview") -> None:
    """Populate scores TreeView with benchmark data.
    
    Args:
        result: EvaluationResult to display
        scores_tree: Treeview widget to populate
    """
    # Clear existing
    for item in scores_tree.get_children():
        scores_tree.delete(item)
    
    # Add benchmarks
    for task_name, task_data in sorted(result.benchmark_scores.items()):
        # Add task as parent
        task_id = scores_tree.insert("", "end", text=task_name, values=("", "", ""))
        
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
            
            scores_tree.insert(
                task_id, "end", text="", values=(metric_name, score_str, stderr)
            )
        
        # Expand all tasks
        scores_tree.item(task_id, open=True)
