"""
Favorites Dialog

Functions for showing and managing favorited datasets.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any, Callable

from .favorites_manager import load_favorites, remove_favorite
from .dataset_details import show_dataset_details_dialog
from aios.gui.utils.theme_utils import apply_theme_to_toplevel, get_theme_colors


def _configure_favorites_styles(dialog: tk.Toplevel) -> Dict[str, str]:
    """Create dedicated ttk styles so the popup matches the active theme."""
    colors = get_theme_colors()
    style = ttk.Style(dialog)
    prefix = "FavoritesPopup"
    frame_style = f"{prefix}.TFrame"
    label_style = f"{prefix}.TLabel"
    button_style = f"{prefix}.TButton"
    tree_style = f"{prefix}.Treeview"
    scrollbar_style = f"{prefix}.Vertical.TScrollbar"
    style.configure(frame_style, background=colors["bg"])
    style.configure(label_style, background=colors["bg"], foreground=colors["fg"])
    style.configure(
        button_style,
        background=colors["button_bg"],
        foreground=colors["fg"],
        padding=(10, 4)
    )
    style.map(
        button_style,
        background=[("active", colors["select_bg"]), ("pressed", colors["select_bg"])],
        foreground=[("active", colors["select_fg"]), ("pressed", colors["select_fg"])],
    )
    style.configure(
        tree_style,
        background=colors["entry_bg"],
        foreground=colors["fg"],
        fieldbackground=colors["entry_bg"],
        rowheight=22,
        borderwidth=0
    )
    style.map(
        tree_style,
        background=[("selected", colors["select_bg"])],
        foreground=[("selected", colors["select_fg"])],
    )
    style.configure(
        f"{tree_style}.Heading",
        background=colors["button_bg"],
        foreground=colors["fg"],
        relief="flat"
    )
    style.configure(
        scrollbar_style,
        troughcolor=colors["bg"],
        background=colors["button_bg"],
        arrowcolor=colors["fg"]
    )
    return {
        "frame": frame_style,
        "label": label_style,
        "button": button_style,
        "tree": tree_style,
        "scrollbar": scrollbar_style,
    }


def show_favorites_popup(parent: tk.Widget, log_func: Callable[[str], None], download_callback: Callable[[Dict[str, Any]], None]):
    """
    Show a popup dialog with the list of favorited datasets.
    
    Args:
        parent: Parent widget
        log_func: Function to call for logging messages
        download_callback: Function to call when user wants to download a dataset
    """
    favorites = load_favorites()
    
    if not favorites:
        messagebox.showinfo(
            "No Favorites",
            "You haven't favorited any datasets yet!\n\n"
            "Search for datasets and click '⭐ Add to Favorites' to save them."
        )
        return
    
    # Create popup dialog
    dialog = tk.Toplevel(parent)
    dialog.title("⭐ Favorite Datasets")
    dialog.geometry("900x500")
    dialog.transient(parent)
    apply_theme_to_toplevel(dialog)
    styles = _configure_favorites_styles(dialog)
    
    # Info label
    info_label = ttk.Label(
        dialog,
        text=f"You have {len(favorites)} favorite dataset(s). Double-click to download.",
        font=("", 9),
        style=styles["label"],
    )
    info_label.pack(pady=10, padx=10)
    
    # Create treeview for favorites
    tree_frame = ttk.Frame(dialog, style=styles["frame"])
    tree_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    
    tree_scroll = ttk.Scrollbar(tree_frame, style=styles["scrollbar"])
    tree_scroll.pack(side="right", fill="y")
    
    fav_tree = ttk.Treeview(
        tree_frame,
        columns=("downloads", "likes", "description"),
        yscrollcommand=tree_scroll.set,
        selectmode="browse",
        style=styles["tree"],
    )
    fav_tree.pack(side="left", fill="both", expand=True)
    tree_scroll.config(command=fav_tree.yview)
    
    # Configure columns
    fav_tree.heading("#0", text="Dataset")
    fav_tree.heading("downloads", text="Downloads")
    fav_tree.heading("likes", text="Likes")
    fav_tree.heading("description", text="Description")
    
    fav_tree.column("#0", width=250, minwidth=150)
    fav_tree.column("downloads", width=100, minwidth=80)
    fav_tree.column("likes", width=80, minwidth=60)
    fav_tree.column("description", width=400, minwidth=200)
    
    # Add favorites to tree
    fav_dict = {}  # Store dataset info by item ID
    for ds in favorites:
        downloads = ds.get("downloads", 0)
        if downloads >= 1_000_000:
            downloads_str = f"{downloads / 1_000_000:.1f}M"
        elif downloads >= 1_000:
            downloads_str = f"{downloads / 1_000:.1f}K"
        else:
            downloads_str = str(downloads)
        
        likes = ds.get("likes", 0)
        description = ds.get("description", "No description")[:100]
        if len(ds.get("description", "")) > 100:
            description += "..."
        
        dataset_name = ds.get("full_name", ds.get("name", "Unknown"))
        
        item_id = fav_tree.insert(
            "",
            "end",
            text=dataset_name,
            values=(downloads_str, likes, description)
        )
        fav_dict[item_id] = ds
    
    # Button frame
    btn_frame = ttk.Frame(dialog, style=styles["frame"])
    btn_frame.pack(fill="x", padx=10, pady=(0, 10))
    
    def download_selected():
        selection = fav_tree.selection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select a dataset to download.")
            return
        
        dataset = fav_dict.get(selection[0])
        if dataset:
            dialog.destroy()
            download_callback(dataset)
    
    def remove_selected():
        selection = fav_tree.selection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select a dataset to remove.")
            return
        
        dataset = fav_dict.get(selection[0])
        if dataset:
            dataset_id = dataset.get("id", "")
            dataset_name = dataset.get("full_name", dataset.get("name", "Unknown"))
            
            if messagebox.askyesno(
                "Confirm Remove",
                f"Remove '{dataset_name}' from favorites?"
            ):
                if remove_favorite(dataset_id):
                    fav_tree.delete(selection[0])
                    del fav_dict[selection[0]]
                    log_func(f"💔 Removed '{dataset_name}' from favorites")
                    
                    if len(fav_dict) == 0:
                        dialog.destroy()
                        messagebox.showinfo("All Clear", "All favorites removed!")
    
    def view_details():
        selection = fav_tree.selection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select a dataset to view details.")
            return
        
        dataset = fav_dict.get(selection[0])
        if dataset:
            show_dataset_details_dialog(dataset, dialog)
    
    # Double-click to download
    fav_tree.bind("<Double-1>", lambda e: download_selected())
    
    ttk.Button(
        btn_frame,
        text="📥 Download",
        command=download_selected,
        width=15,
        style=styles["button"],
    ).pack(side="left", padx=(0, 5))
    
    ttk.Button(
        btn_frame,
        text="💔 Remove",
        command=remove_selected,
        width=15,
        style=styles["button"],
    ).pack(side="left", padx=(0, 5))
    
    ttk.Button(
        btn_frame,
        text="ℹ️ Details",
        command=view_details,
        width=15,
        style=styles["button"],
    ).pack(side="left", padx=(0, 5))
    
    ttk.Button(
        btn_frame,
        text="Close",
        command=dialog.destroy,
        width=10,
        style=styles["button"],
    ).pack(side="right")
