"""Student selection package for HRM training.

This package provides dialogs for selecting/creating HRM student models.

Main entry point:
    select_student(panel) - Shows the selection dialog

The selection dialog allows users to:
- Select from existing trained/training brains
- Browse for arbitrary checkpoint files
- Create new brains with custom architecture/tokenizer configuration

Example usage:
    >>> from aios.gui.components.hrm_training.selection import select_student
    >>> 
    >>> # In HRM training panel
    >>> select_student(self)  # Shows selection dialog
"""

from .main_dialog import show_selection_dialog


def select_student(panel) -> None:
    """
    Show student selection dialog.
    
    This is the main entry point for backward compatibility with
    the original selection.py module.
    
    Args:
        panel: HRM training panel instance
    """
    show_selection_dialog(panel)


__all__ = ["select_student"]
