"""
Dataset Details Dialog

Functions for showing detailed information about a dataset.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Dict, Any

from .favorites_manager import is_favorited


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
    dialog.geometry("600x500")
    dialog.transient(parent_window)
    
    # Create scrolled text for details
    details_text = scrolledtext.ScrolledText(dialog, wrap="word", font=("Consolas", 9))
    details_text.pack(fill="both", expand=True, padx=10, pady=10)
    
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
    ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
