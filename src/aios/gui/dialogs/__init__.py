"""GUI dialogs for AI-OS.

Dialogs are lazily imported to avoid loading torch/transformers during startup.
"""

__all__ = ["EvaluationResultsDialog", "EvaluationHistoryDialog", "EvaluationSamplesDialog"]


def __getattr__(name: str):
    """Lazy import dialogs to avoid loading heavy dependencies during startup."""
    if name == "EvaluationResultsDialog":
        from aios.gui.dialogs.evaluation_results_dialog import EvaluationResultsDialog
        return EvaluationResultsDialog
    elif name == "EvaluationHistoryDialog":
        from aios.gui.dialogs.evaluation_history_dialog import EvaluationHistoryDialog
        return EvaluationHistoryDialog
    elif name == "EvaluationSamplesDialog":
        from aios.gui.dialogs.evaluation_samples_dialog import EvaluationSamplesDialog
        return EvaluationSamplesDialog
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
