"""
Dataset Download Panel - Main Class

Main orchestrator for the dataset download panel.
"""

import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Callable, Optional, Dict, Any

# Import all helper modules
from . import hf_cache_setup  # Must be first to set env vars
from .hf_auth import get_hf_login_status, logout_from_hf, show_login_dialog
from .favorites_dialog import show_favorites_popup
from .favorites_manager import add_favorite, remove_favorite, is_favorited
from .dataset_details import show_dataset_details_dialog
from .search_operations import do_search, get_selected_dataset, sort_results_by_column
from .cache_manager import load_search_cache, save_search_cache
from .download_core import download_dataset
from .ui_builder import build_ui, browse_location


class DatasetDownloadPanel:
    """Panel for searching, downloading, and managing favorite HuggingFace datasets."""
    
    def __init__(self, parent, log_callback: Callable[[str], None], output_parent=None):
        """
        Initialize the dataset download panel.
        
        Args:
            parent: Parent tkinter widget
            log_callback: Function to call for logging messages
            output_parent: Optional parent for output text widget
        """
        self.parent = parent
        self._original_log = log_callback  # Keep reference to original
        self._output_parent = output_parent  # Optional parent for output box
        self.download_thread = None
        self.cancel_download = False
        self.search_results = []
        self.current_view = "search"  # "search" or "favorites"
        self._hf_logged_in = False  # Track HF login state
        
        # Stream coordination for preventing concurrent streaming conflicts
        self.download_pause_event = threading.Event()  # For pausing downloads during training
        self.current_download_dataset_id: Optional[str] = None  # Track which dataset is being downloaded
        
        # Build UI
        build_ui(self)
        
        # Make the original log callback write to our output widget
        self.log = self._log_to_output
        
        # Initialize
        self._update_hf_status()
        self.log("💡 Tip: Search for datasets by topic, task, or language (e.g., 'python code', 'wikipedia', 'sentiment')")
        self.log("💡 Use the Favorites button to view and manage your saved datasets")
        
        # Load cached results immediately if available
        self._load_cached_results()
        
        # Schedule fresh search 10 seconds after program opens
        def _delayed_fresh_search():
            try:
                if self.parent.winfo_exists():
                    self.log("🔄 Refreshing dataset catalog...")
                    do_search(self, cache_results=True)
            except Exception:
                pass  # Widget destroyed or mainloop not started
        
        try:
            if self.parent.winfo_exists():
                self.parent.after(10000, _delayed_fresh_search)  # 10 second delay for fresh search
        except Exception:
            pass  # Widget destroyed or mainloop not started
    
    def _load_cached_results(self):
        """Load cached search results from disk."""  
        from .search_operations import display_search_results
        
        try:
            cache_data = load_search_cache()
            if cache_data:
                self.search_results = cache_data.get('results', [])
                if self.search_results:
                    display_search_results(
                        self.search_results, cache_data.get('query', ''), len(self.search_results),
                        self.results_tree, self.search_status_label, self.status_label, self.log
                    )
                    self.log(f"📦 Loaded {len(self.search_results)} datasets from cache")
                    return
            
            # If no valid cache, show a quick message and trigger initial search
            self.search_status_label.config(text="Loading...", foreground="gray")
            # Trigger immediate initial search (no delay)
            if self.parent.winfo_exists():
                self.parent.after(100, lambda: do_search(self, cache_results=True))
        except Exception as e:
            self.log(f"⚠️  Could not load search cache: {e}")
    
    def _show_favorites_popup(self):
        """Show a popup dialog with the list of favorited datasets."""
        show_favorites_popup(self.parent, self.log, self._download_dataset_direct)
    
    def _do_search(self, cache_results: bool = False):
        """Perform dataset search on HuggingFace."""
        do_search(self, cache_results)
    
    def _clear_search(self):
        """Clear search and show default results."""
        self.search_var.set("")
        if hasattr(self, 'max_size_var'):
            self.max_size_var.set("")
        do_search(self)
    
    def _sort_by_column(self, col: str):
        """Sort treeview by column."""
        sort_results_by_column(self, col)
    
    def _download_selected(self):
        """Download the selected dataset."""
        self.log("📥 Download button clicked")
        dataset = get_selected_dataset(self)
        if not dataset:
            self.log("❌ Cannot download: No dataset selected or found")
            messagebox.showinfo(
                "No Selection", 
                "Please select a dataset from the list to download.\n\n"
                "Tip: Click on a dataset in the results table to select it."
            )
            return
        
        # Confirm download
        dataset_name = dataset.get("full_name", dataset.get("name", "Unknown"))
        size_info = f" (~{dataset.get('size_gb', 0):.1f} GB)" if dataset.get('size_gb', 0) > 0 else ""
        
        msg = f"Download '{dataset_name}'{size_info}?\n\n"
        if dataset.get("gated", False):
            msg += "⚠️ This is a gated dataset. You may need to accept terms on HuggingFace.\n\n"
        if dataset.get("private", False):
            msg += "⚠️ This is a private dataset. You need appropriate access.\n\n"
        
        msg += "The dataset will be streamed with a default limit of 100,000 samples."
        
        if not messagebox.askyesno("Confirm Download", msg):
            return
        
        # Start download
        self.cancel_download = False
        self.cancel_btn.config(state="normal")
        self.status_label.config(text=f"Downloading {dataset_name}...")
        
        self.download_thread = threading.Thread(
            target=download_dataset,
            args=(self, dataset),
            daemon=True
        )
        self.download_thread.start()
    
    def _favorite_selected(self):
        """Add or remove the selected dataset from favorites."""
        self.log("⭐ Favorite button clicked")
        dataset = get_selected_dataset(self)
        if not dataset:
            self.log("❌ Cannot favorite: No dataset selected or found")
            messagebox.showinfo(
                "No Selection", 
                "Please select a dataset from the list to favorite.\n\n"
                "Tip: Click on a dataset in the results table to select it."
            )
            return
        
        dataset_id = dataset.get("id", dataset.get("path", ""))
        dataset_name = dataset.get("full_name", dataset.get("name", "Unknown"))
        
        if is_favorited(dataset_id):
            # Remove from favorites
            if remove_favorite(dataset_id):
                self.log(f"💔 Removed '{dataset_name}' from favorites")
                messagebox.showinfo("Success", f"Removed from favorites:\n{dataset_name}")
                # Refresh to update star indicator
                do_search(self)
            else:
                messagebox.showerror("Error", "Failed to remove from favorites")
        else:
            # Add to favorites
            if add_favorite(dataset):
                self.log(f"⭐ Added '{dataset_name}' to favorites")
                messagebox.showinfo("Success", f"Added to favorites:\n{dataset_name}")
                # Refresh to show star
                do_search(self)
            else:
                messagebox.showinfo("Info", "Dataset is already in favorites")
    
    def _view_dataset_details(self):
        """Show detailed information about the selected dataset."""
        dataset = get_selected_dataset(self)
        if not dataset:
            messagebox.showinfo("No Selection", "Please select a dataset to view details.")
            return
        
        show_dataset_details_dialog(dataset, self.parent)
    
    def _download_dataset_direct(self, dataset: Dict[str, Any]):
        """Start downloading a dataset directly (used by favorites popup)."""
        # Confirm download
        dataset_name = dataset.get("full_name", dataset.get("name", "Unknown"))
        size_info = f" (~{dataset.get('size_gb', 0):.1f} GB)" if dataset.get('size_gb', 0) > 0 else ""
        
        msg = f"Download '{dataset_name}'{size_info}?\n\n"
        if dataset.get("gated", False):
            msg += "⚠️ This is a gated dataset. You may need to accept terms on HuggingFace.\n\n"
        if dataset.get("private", False):
            msg += "⚠️ This is a private dataset. You need appropriate access.\n\n"
        
        msg += "The dataset will be streamed with a default limit of 100,000 samples."
        
        if not messagebox.askyesno("Confirm Download", msg):
            return
        
        # Start download
        self.cancel_download = False
        self.cancel_btn.config(state="normal")
        self.status_label.config(text=f"Downloading {dataset_name}...")
        
        self.download_thread = threading.Thread(
            target=download_dataset,
            args=(self, dataset),
            daemon=True
        )
        self.download_thread.start()
    
    def _log_to_output(self, message: str) -> None:
        """Write log message to the output text widget."""
        # Only write to local output_text if it exists (not using shared output)
        if self.output_text is not None:
            try:
                # Check if user is at bottom before inserting
                try:
                    yview = self.output_text.yview()
                    at_bottom = yview[1] >= 0.95  # Within ~5% of bottom
                except Exception:
                    at_bottom = True  # Default to scrolling if can't check
                
                # Write to our local output widget
                self.output_text.insert(tk.END, message + "\n")
                
                # Only scroll if user was at bottom
                if at_bottom:
                    self.output_text.see(tk.END)
                
                self.output_text.update_idletasks()
            except Exception:
                pass  # Fail silently if widget not available
        
        # Always send to the original log callback (shared output or debug panel)
        try:
            if self._original_log:
                self._original_log(message)
        except Exception:
            pass
    
    def _browse_location(self):
        """Browse for download location."""
        browse_location(self)
    
    def _update_hf_status(self):
        """Update HuggingFace authentication status."""
        is_logged_in, status_msg = get_hf_login_status()
        
        self._hf_logged_in = is_logged_in
        self.hf_status_label.config(text=status_msg, foreground="green" if is_logged_in else "gray")
        
        if is_logged_in:
            self.hf_auth_btn.config(text="🔓 Logout", state="normal")
        else:
            self.hf_auth_btn.config(text="🔐 Login to HF", state="normal")
    
    def _show_login_dialog(self):
        """Show dialog to login or logout from HuggingFace."""
        # Check if user wants to logout
        if self._hf_logged_in:
            if messagebox.askyesno("Logout", "Are you sure you want to logout from HuggingFace?"):
                if logout_from_hf(self.log):
                    self._update_hf_status()
            return
        
        # Login flow
        username = show_login_dialog(self.parent, self.log)
        if username:
            self._update_hf_status()
    
    def _cancel_download(self):
        """Cancel ongoing download."""
        self.cancel_download = True
        self.log("❌ Cancelling download...")
