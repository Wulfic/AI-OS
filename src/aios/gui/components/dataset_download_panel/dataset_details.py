"""
Dataset Details Dialog

Functions for showing detailed information about a dataset.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Dict, Any

from .favorites_manager import is_favorited
from aios.gui.utils.theme_utils import apply_theme_to_toplevel, get_theme_colors, compute_popup_dimensions


def show_dataset_details_dialog(dataset: Dict[str, Any], parent_window: tk.Widget):
    """
    Show detailed information about a dataset in a dialog.
    
    Args:
        dataset: Dataset information dictionary
        parent_window: Parent window for the dialog
    """
    # Create details dialog
    dialog = tk.Toplevel(parent_window)
    dialog.title(f"Dataset Details: {dataset.get('name', 'Unknown')}")
    dialog.transient(parent_window)
    apply_theme_to_toplevel(dialog)
    min_width = 880
    min_height = 620
    width, height, pos_x, pos_y = compute_popup_dimensions(
        parent_window,
        width_ratio=0.5,
        height_ratio=0.74,
        min_width=min_width,
        min_height=min_height,
    )
    dialog.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
    dialog.minsize(min_width, min_height)
    dialog.resizable(True, True)
    colors = get_theme_colors()
    
    # Create scrolled text for details
    details_text = scrolledtext.ScrolledText(
      dialog,
      wrap="word",
      font=("Consolas", 10),
      bg=colors["entry_bg"],
      fg=colors["fg"],
      insertbackground=colors["insert_bg"],
      selectbackground=colors["select_bg"],
      selectforeground=colors["select_fg"],
    )
    details_text.pack(fill="both", expand=True, padx=12, pady=(10, 8))
    
    # Format details
    details = f"""Dataset: {dataset.get('full_name', dataset.get('name', 'Unknown'))}
Author: {dataset.get('author', 'Unknown')}
ID: {dataset.get('id', 'N/A')}

📊 Statistics:
  Downloads: {dataset.get('downloads', 0):,}
  Likes: {dataset.get('likes', 0):,}
  
🏷️ Tags: {', '.join(dataset.get('tags', [])[:10]) if dataset.get('tags') else 'None'}

🔒 Access:
  Private: {'Yes' if dataset.get('private', False) else 'No'}
  Gated: {'Yes' if dataset.get('gated', False) else 'No'}

📝 Description:
{dataset.get('description', 'No description available')}

⚙️ Download Settings:
  Path: {dataset.get('path', 'N/A')}
  Config: {dataset.get('config', 'default')}
  Split: {dataset.get('split', 'train')}
  Streaming: {dataset.get('streaming', True)}
  Max Samples: {dataset.get('max_samples', 100000):,}

⭐ Favorited: {'Yes' if is_favorited(dataset.get('id', '')) else 'No'}
"""
    
    details_text.insert("1.0", details)
    details_text.config(state="disabled")
    
    # Close button
    ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=(0, 12))
