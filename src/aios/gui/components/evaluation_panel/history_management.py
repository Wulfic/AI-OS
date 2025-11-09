"""History management for evaluation results."""

from __future__ import annotations
from tkinter import messagebox
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .panel_main import EvaluationPanel


def view_history(panel: "EvaluationPanel") -> None:
    """View evaluation history.
    
    Args:
        panel: The evaluation panel instance
    """
    from aios.core.evaluation import EvaluationResult
    from aios.gui.dialogs import EvaluationHistoryDialog, EvaluationResultsDialog
    
    try:
        def on_view_details(eval_id: int) -> None:
            """Handle viewing details from history."""
            try:
                eval_data = panel._history.get_evaluation(eval_id)
                if not eval_data:
                    messagebox.showerror("Error", "Evaluation not found.")
                    return
                
                # Reconstruct EvaluationResult
                result = EvaluationResult(
                    overall_score=eval_data.get("overall_score", 0.0),
                    status=eval_data.get("status", "unknown"),
                    error_message=eval_data.get("error_message", ""),
                    start_time=eval_data.get("start_time", 0.0),
                    end_time=eval_data.get("end_time", 0.0),
                    output_path=eval_data.get("output_path", ""),
                    raw_results=eval_data.get("raw_results", {}),
                )
                
                # Reconstruct benchmark_scores from scores list
                benchmark_scores = {}
                for score in eval_data.get("scores", []):
                    benchmark = score["benchmark_name"]
                    if benchmark not in benchmark_scores:
                        benchmark_scores[benchmark] = {"scores": {}, "raw": {}}
                    
                    metric = score["metric_name"]
                    benchmark_scores[benchmark]["scores"][metric] = score["score"]
                    if score.get("stderr"):
                        benchmark_scores[benchmark]["raw"][f"{metric}_stderr"] = score["stderr"]
                
                result.benchmark_scores = benchmark_scores
                
                # Open results dialog
                model_name = eval_data.get("model_name", "Unknown")
                EvaluationResultsDialog(panel, result, model_name)
            
            except Exception as e:
                panel._log(f"[eval] Error viewing historical result: {e}")
                messagebox.showerror("Error", f"Failed to view details:\n{e}")
        
        # Open history dialog
        dialog = EvaluationHistoryDialog(
            panel,
            panel._history,
            on_view_details=on_view_details
        )
        panel._log("[eval] Opened evaluation history")
    
    except Exception as e:
        panel._log(f"[eval] Error opening history: {e}")
        messagebox.showerror("Error", f"Failed to open history:\n{e}")
