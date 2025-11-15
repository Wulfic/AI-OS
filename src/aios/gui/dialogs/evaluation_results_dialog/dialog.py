"""Main evaluation results dialog class."""

from __future__ import annotations

import tkinter as tk
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from aios.core.evaluation import EvaluationResult

from aios.gui.utils.model_display import get_model_display_name
from aios.gui.utils.theme_utils import apply_theme_to_toplevel

from .ui_builder import build_dialog_ui, add_dialog_buttons
from .data_utils import populate_summary, populate_scores_tree
from .chart_utils import create_evaluation_charts, is_matplotlib_available
from .export_utils import export_to_csv, export_to_json, export_to_html


class EvaluationResultsDialog(tk.Toplevel):  # type: ignore[misc]
    """Dialog for viewing detailed evaluation results with charts and export."""
    
    def __init__(
        self,
        parent: Any,
        result: "EvaluationResult",
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
        self._raw_model_name = model_name
        self.model_name = get_model_display_name(model_name)
        
        # Configure window
        self.title(f"Evaluation Results: {self.model_name or 'Unknown Model'}")
        self.geometry("1000x700")
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        
        # Apply theme to this dialog
        apply_theme_to_toplevel(self)
        
        # Build UI and get widgets
        widgets = build_dialog_ui(self, result)
        self.model_label = widgets['model_label']
        self.status_label = widgets['status_label']
        self.score_label = widgets['score_label']
        self.duration_label = widgets['duration_label']
        self.tasks_label = widgets['tasks_label']
        self.scores_tree = widgets['scores_tree']
        
        # Add buttons (need to get main_frame from build_dialog_ui - for simplicity, rebuild)
        # In practice, we'd pass this through, but for now we'll access via winfo_children
        main_frame = self.winfo_children()[0]
        add_dialog_buttons(
            self,
            main_frame,
            self._export_csv,
            self._export_json,
            self._export_html,
        )
        
        # Populate data
        populate_summary(
            result,
            self.model_name,
            self.model_label,
            self.status_label,
            self.score_label,
            self.duration_label,
            self.tasks_label,
        )
        populate_scores_tree(result, self.scores_tree)
        
        # Create charts if available
        if is_matplotlib_available() and 'charts_container' in widgets:
            create_evaluation_charts(result, widgets['charts_container'])
        
        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
    
    def _export_csv(self) -> None:
        """Export results to CSV."""
        export_to_csv(self.result, self._raw_model_name or self.model_name)
    
    def _export_json(self) -> None:
        """Export results to JSON."""
        export_to_json(self.result, self._raw_model_name or self.model_name)
    
    def _export_html(self) -> None:
        """Export results to HTML report."""
        export_to_html(self.result, self._raw_model_name or self.model_name)
