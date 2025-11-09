"""Datasets panel component.

Provides a Tkinter panel for managing datasets with:
- Local file browsing
- Known dataset selection and downloading
- Progress tracking for downloads
- Thread-safe logging

Usage:
    from aios.gui.components.datasets_panel import DatasetsPanel
    
    panel = DatasetsPanel(parent, dataset_path_var=my_var)
"""

from __future__ import annotations

from .panel_main import DatasetsPanel

__all__ = ["DatasetsPanel"]
