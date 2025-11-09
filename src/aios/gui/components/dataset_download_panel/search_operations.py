"""
Search Operations

Helper functions for searching, filtering, displaying, and sorting dataset search results.
"""

import threading
import tkinter as tk
from tkinter import messagebox
from typing import Dict, List, Any, Optional

from .hf_search import search_huggingface_datasets, list_datasets
from .favorites_manager import is_favorited
from .cache_manager import save_search_cache
from .hf_size_detection import enrich_dataset_with_size, format_size_display


def apply_size_filter_threadsafe(results: List[Dict[str, Any]], max_size_text: str, size_unit: str, log_func) -> List[Dict[str, Any]]:
    """
    Thread-safe filter that doesn't access Tkinter variables.
    
    Args:
        results: List of dataset info dictionaries
        max_size_text: Maximum size as string (e.g., "10")
        size_unit: Size unit ("MB", "GB", or "TB")
        log_func: Function to call for logging
        
    Returns:
        Filtered list of datasets
    """
    try:
        if not max_size_text:
            return results  # No filter applied
        
        max_size = float(max_size_text)
        
        # Convert to GB for comparison
        if size_unit == "MB":
            max_size_gb = max_size / 1024
        elif size_unit == "TB":
            max_size_gb = max_size * 1024
        else:  # GB
            max_size_gb = max_size
        
        # Filter results - keep datasets with unknown size (0) or within limit
        filtered = [
            ds for ds in results
            if ds.get('size_gb', 0) == 0 or ds.get('size_gb', 0) <= max_size_gb
        ]
        
        log_func(f"🔍 Size filter: {len(filtered)}/{len(results)} datasets <= {max_size} {size_unit}")
        return filtered
        
    except (ValueError, AttributeError) as e:
        log_func(f"⚠️ Size filter error: {e}, returning unfiltered results")
        return results


def display_search_results(
    results: List[Dict[str, Any]], 
    query: str, 
    total_before_filter: Optional[int],
    results_tree: tk.Widget,
    search_status_label: tk.Widget,
    status_label: tk.Widget,
    log_func
):
    """
    Display search results in the treeview.
    
    Args:
        results: List of dataset info dictionaries
        query: Search query that was used
        total_before_filter: Total results before filtering (for display)
        results_tree: Treeview widget to populate
        search_status_label: Label widget for search status
        status_label: Label widget for general status
        log_func: Function to call for logging
    """
    # Clear existing items
    for item in results_tree.get_children():
        results_tree.delete(item)
    
    if not results:
        if total_before_filter and total_before_filter > 0:
            search_status_label.config(
                text=f"All {total_before_filter} results filtered out by size limit",
                foreground="orange"
            )
        else:
            search_status_label.config(
                text=f"No results found for '{query}'",
                foreground="orange"
            )
        status_label.config(text="Ready")
        return
    
    # Add results to tree
    for ds in results:
        # Format numbers
        downloads = ds.get("downloads", 0)
        likes = ds.get("likes", 0)
        
        # Format downloads (e.g., 1.2M, 5.3K)
        if downloads >= 1_000_000:
            downloads_str = f"{downloads / 1_000_000:.1f}M"
        elif downloads >= 1_000:
            downloads_str = f"{downloads / 1_000:.1f}K"
        else:
            downloads_str = str(downloads)
        
        # Add favorite star if favorited
        dataset_name = ds.get("full_name", ds.get("name", "Unknown"))
        dataset_id = ds.get("id", "")
        if is_favorited(dataset_id):
            dataset_name = f"⭐ {dataset_name}"
        
        # Format size, rows, and blocks
        size_str, rows_str, blocks_str = format_size_display(ds)
        
        # Truncate description
        description = ds.get("description", "No description")[:80]
        if len(ds.get("description", "")) > 80:
            description += "..."
        
        # Insert into tree with dataset ID as tag
        item_id = results_tree.insert(
            "",
            "end",
            text=dataset_name,
            values=(downloads_str, size_str, rows_str, blocks_str, description),
            tags=(dataset_id,)
        )
        log_func(f"📦 Added dataset to tree: {dataset_name} (ID: {dataset_id}, item: {item_id})")
    
    query_text = f"'{query}'" if query else "popular datasets"
    filter_text = ""
    if total_before_filter and total_before_filter > len(results):
        filter_text = f" (filtered from {total_before_filter})"
    search_status_label.config(
        text=f"✅ Found {len(results)} results for {query_text}{filter_text}",
        foreground="green"
    )
    status_label.config(text="Ready")
    log_func(f"✅ Search complete: {len(results)} datasets found and added to tree")


def do_search(
    panel,  # DatasetDownloadPanel instance
    cache_results: bool = False
):
    """
    Perform dataset search on HuggingFace in background thread.
    
    Args:
        panel: DatasetDownloadPanel instance with all required attributes
        cache_results: If True, save results to cache after successful search
    """
    query = panel.search_var.get().strip()
    
    if list_datasets is None:
        messagebox.showerror(
            "Library Missing",
            "huggingface_hub library is not installed.\n\n"
            "Install it with: pip install huggingface_hub"
        )
        return
    
    panel.search_status_label.config(text="🔄 Searching...", foreground="blue")
    panel.status_label.config(text="Searching HuggingFace Hub...")
    panel.frame.update()
    
    # Capture filter values in main thread before spawning background thread
    try:
        max_size_text = panel.max_size_var.get().strip()
        size_unit = panel.size_unit_var.get()
    except Exception:
        max_size_text = ""
        size_unit = "GB"
    
    def search_thread():
        try:
            results = search_huggingface_datasets(query, limit=50)
            
            # Enrich datasets with size information in background
            panel.log(f"🔍 Enriching {len(results)} datasets with size information...")
            for ds in results:
                try:
                    enrich_dataset_with_size(ds)
                except Exception as e:
                    panel.log(f"⚠️ Could not get size for {ds.get('id', 'unknown')}: {e}")
            
            # Apply size filter if specified
            filtered_results = apply_size_filter_threadsafe(results, max_size_text, size_unit, panel.log)
            
            panel.search_results = filtered_results
            
            # Update UI in main thread - check if widget still exists
            try:
                if panel.frame.winfo_exists():
                    panel.frame.after(0, lambda: display_search_results(
                        filtered_results, query, len(results),
                        panel.results_tree, panel.search_status_label, 
                        panel.status_label, panel.log
                    ))
                    # Save to cache if requested
                    if cache_results:
                        panel.frame.after(100, lambda: save_search_cache(query, filtered_results))
            except Exception:
                pass  # Widget destroyed, ignore
        except Exception as e:
            error_msg = str(e)
            # Update UI in main thread - check if widget still exists
            try:
                if panel.frame.winfo_exists():
                    panel.frame.after(0, lambda: search_error(error_msg, panel.search_status_label, panel.status_label, panel.log))
            except Exception:
                pass  # Widget destroyed, ignore
    
    threading.Thread(target=search_thread, daemon=True, name="Thread-HF-Search").start()


def search_error(error_msg: str, search_status_label: tk.Widget, status_label: tk.Widget, log_func):
    """Handle search error."""
    search_status_label.config(
        text=f"❌ Search failed: {error_msg[:50]}",
        foreground="red"
    )
    status_label.config(text="Ready")
    log_func(f"❌ Search error: {error_msg}")


def sort_results_by_column(panel, col: str):
    """
    Sort treeview by column.
    
    Args:
        panel: DatasetDownloadPanel instance
        col: Column identifier to sort by
    """
    # Toggle sort direction if clicking same column
    if panel._sort_column == col:
        panel._sort_reverse = not panel._sort_reverse
    else:
        panel._sort_column = col
        panel._sort_reverse = False
    
    # Get all items
    items = [(panel.results_tree.set(k, col) if col != "#0" else panel.results_tree.item(k, "text"), k) 
             for k in panel.results_tree.get_children('')]
    
    # Sort items - handle numeric columns
    if col == "downloads":
        # Parse download strings like "1.2M" or "5.3K"
        def parse_downloads(val):
            val_str = str(val)
            if 'M' in val_str:
                return float(val_str.replace('M', '')) * 1_000_000
            elif 'K' in val_str:
                return float(val_str.replace('K', '')) * 1_000
            return float(val_str) if val_str.replace('.', '').isdigit() else 0
        items = [(parse_downloads(val), k) for val, k in items]
    elif col == "rows":
        # Parse row strings like "25,000 rows" or "Unknown"
        def parse_rows(val):
            val_str = str(val)
            if 'Unknown' in val_str:
                return 0
            # Extract number from "25,000 rows" or "25,000 rows (est.)"
            num_str = val_str.split()[0].replace(',', '')
            return int(num_str) if num_str.isdigit() else 0
        items = [(parse_rows(val), k) for val, k in items]
    elif col == "blocks":
        # Parse block strings like "3 blocks" or "Unknown"
        def parse_blocks(val):
            val_str = str(val)
            if 'Unknown' in val_str:
                return 0
            num_str = val_str.split()[0]
            return int(num_str) if num_str.isdigit() else 0
        items = [(parse_blocks(val), k) for val, k in items]
    elif col == "size":
        # Parse size strings like "1.50 GB" or "45.2 MB"
        def parse_size(val):
            val_str = str(val)
            if 'Unknown' in val_str:
                return 0
            parts = val_str.split()
            if len(parts) < 2:
                return 0
            num = float(parts[0]) if parts[0].replace('.', '').isdigit() else 0
            if 'GB' in val_str:
                return num * 1024  # Convert to MB for comparison
            return num  # Already in MB
        items = [(parse_size(val), k) for val, k in items]
    
    items.sort(reverse=panel._sort_reverse)
    
    # Rearrange items in sorted order
    for index, (val, k) in enumerate(items):
        panel.results_tree.move(k, '', index)
    
    # Update column headings to show sort direction
    for column in ("#0", "downloads", "size", "rows", "blocks", "description"):
        if column == col:
            arrow = " ▼" if panel._sort_reverse else " ▲"
            current_text = panel.results_tree.heading(column)["text"]
            # Remove existing arrows
            base_text = current_text.replace(" ▲", "").replace(" ▼", "")
            panel.results_tree.heading(column, text=base_text + arrow)
        else:
            current_text = panel.results_tree.heading(column)["text"]
            base_text = current_text.replace(" ▲", "").replace(" ▼", "")
            panel.results_tree.heading(column, text=base_text)


def get_selected_dataset(panel) -> Optional[Dict[str, Any]]:
    """
    Get the currently selected dataset from the tree.
    
    Args:
        panel: DatasetDownloadPanel instance
        
    Returns:
        Dataset info dict or None if no selection
    """
    selection = panel.results_tree.selection()
    if not selection:
        panel.log("⚠️ No dataset selected in tree")
        return None
    
    item = selection[0]
    tags = panel.results_tree.item(item, "tags")
    
    # Tags is a tuple, get the first element which is the dataset ID
    if not tags or len(tags) == 0 or not tags[0]:
        panel.log(f"⚠️ Selected item has no dataset ID in tags: {tags}")
        return None
    
    dataset_id = tags[0]
    panel.log(f"🔍 Looking for dataset ID: {dataset_id}")
    
    # Find dataset in current results or favorites
    if panel.current_view == "search":
        for ds in panel.search_results:
            if ds.get("id") == dataset_id:
                panel.log(f"✅ Found dataset in search results: {ds.get('name', 'Unknown')}")
                return ds
        panel.log(f"❌ Dataset ID {dataset_id} not found in {len(panel.search_results)} search results")
    else:
        from .favorites_manager import load_favorites
        favorites = load_favorites()
        for ds in favorites:
            if ds.get("id") == dataset_id:
                panel.log(f"✅ Found dataset in favorites: {ds.get('name', 'Unknown')}")
                return ds
        panel.log(f"❌ Dataset ID {dataset_id} not found in {len(favorites)} favorites")
    
    return None
