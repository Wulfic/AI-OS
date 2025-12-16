"""
Dataset Download Panel - Main Class

Main orchestrator for the dataset download panel.
"""

import logging
import threading
import time
import tkinter as tk
from concurrent.futures import Future
from queue import Queue, Empty
from pathlib import Path
from tkinter import messagebox
from typing import Callable, Optional, Dict, Any

# Import all helper modules
from . import hf_cache_setup  # Must be first to set env vars
from .hf_auth import get_hf_login_status, logout_from_hf, show_login_dialog
from .favorites_dialog import show_favorites_popup
from .favorites_manager import add_favorite, remove_favorite, is_favorited
from .dataset_details import show_dataset_details_dialog
from .search_operations import (
    build_display_payload,
    display_search_results,
    do_search,
    get_selected_dataset,
    sort_results_by_column,
)
from .cache_manager import load_search_cache, save_search_cache
from .download_core import download_dataset
from .ui_builder import build_ui, browse_location
from .pause_token import PauseToken
from ...utils.resource_management import submit_background

logger = logging.getLogger(__name__)


class DatasetDownloadPanel:
    """Panel for searching, downloading, and managing favorite HuggingFace datasets."""
    
    def __init__(self, parent, log_callback: Callable[[str], None], output_parent=None, worker_pool=None):
        """
        Initialize the dataset download panel.
        
        Args:
            parent: Parent tkinter widget
            log_callback: Function to call for logging messages
            output_parent: Optional parent for output text widget
        """
        logger.info("Initializing Dataset Download Panel")
        
        self.parent = parent
        self._original_log = log_callback  # Keep reference to original
        self._output_parent = output_parent  # Optional parent for output box
        self._use_shared_output = output_parent is None and callable(log_callback)
        self._download_job: Optional[Future] = None
        self.cancel_download = False
        self.search_results = []
        self.current_view = "search"  # "search" or "favorites"
        self._hf_logged_in = False  # Track HF login state
        self._worker_pool = worker_pool
        self._log_queue: "Queue[str]" = Queue()
        self._panel_active = True
        self._background_jobs: set[Future] = set()
        self._background_lock = threading.Lock()
        
        # Determine HF cache directory
        try:
            import os
            hf_cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            logger.info(f"HuggingFace cache directory: {hf_cache_dir}")
        except Exception as e:
            logger.warning(f"Could not determine HF cache directory: {e}")
        
        # Stream coordination for preventing concurrent streaming conflicts
        self.download_pause_event = PauseToken()  # For pausing downloads during training
        self.current_download_dataset_id: Optional[str] = None  # Track which dataset is being downloaded
        
        # Build UI
        logger.debug("Building dataset download panel UI")
        self.output_text = None
        build_ui(self)
        
        # Create download_location variable (UI is in Settings panel but we need the variable for sync)
        from ...utils import safe_variables
        if not hasattr(self, 'download_location'):
            self.download_location = safe_variables.StringVar(value="training_datasets")
            logger.debug("Created download_location variable with default value")
        
        # Make the original log callback write to our output widget
        self.log = self._log_to_output
        
        # Initialize
        self._update_hf_status()
        self.log("💡 Tip: Search for datasets by topic, task, or language (e.g., 'python code', 'wikipedia', 'sentiment')")
        self.log("💡 Use the Favorites button to view and manage your saved datasets")
        
        # Load cached results immediately if available
        self._load_cached_results()

        # Start draining queued log messages on the Tk thread
        self._start_log_queue_pump()
        
        # Schedule fresh search 10 seconds after program opens
        def _delayed_fresh_search():
            try:
                if self._panel_active and self.parent.winfo_exists():
                    self.log("🔄 Refreshing dataset catalog...")
                    do_search(self, cache_results=True)
            except Exception:
                pass  # Widget destroyed or mainloop not started
        
        try:
            if self._panel_active and self.parent.winfo_exists():
                self.parent.after(10000, _delayed_fresh_search)  # 10 second delay for fresh search
        except Exception:
            pass  # Widget destroyed or mainloop not started
        
        logger.info("Dataset Download Panel initialized successfully")

    def _load_cached_results(self):
        """Load cached search results from disk and render without blocking the UI."""

        logger.debug("_load_cached_results invoked")
        try:
            cache_data = load_search_cache()
            if cache_data:
                cached_results = cache_data.get("results", []) or []
                if cached_results:
                    query = cache_data.get("query", "")
                    total = len(cached_results)
                    self.search_results = cached_results

                    logger.debug(
                        "Applying cached search results: %d entries (query='%s')",
                        total,
                        query,
                    )

                    def _apply_payload(payload):
                        try:
                            display_search_results(
                                payload,
                                query,
                                total,
                                self.results_tree,
                                self.search_status_label,
                                self.status_label,
                                self.log,
                                completion_message="📦 Cached datasets restored ({count})",
                            )
                            self.log(f"📦 Loaded {payload.count} datasets from cache")
                        except Exception:
                            logger.debug("Failed to apply cached results payload", exc_info=True)

                    def _build_payload() -> None:
                        payload = build_display_payload(cached_results)
                        if self.parent.winfo_exists():
                            try:
                                self.parent.after(0, lambda p=payload: _apply_payload(p))
                            except Exception:
                                logger.debug("Failed to schedule cached payload render", exc_info=True)

                    try:
                        future = submit_background("dataset-cache-render", _build_payload, pool=self._worker_pool)
                        self._register_background_future(future)
                    except RuntimeError:
                        logger.debug("Worker pool unavailable for cache render; falling back to sync payload build")
                        payload = build_display_payload(cached_results)
                        if self.parent.winfo_exists():
                            self.parent.after(0, lambda p=payload: _apply_payload(p))
                        else:
                            logger.debug("Parent widget unavailable; skipping cached render")
                    return

            # If no valid cache, show a quick message and trigger immediate async search
            logger.debug("No cached search results available; scheduling immediate search")
            self.search_status_label.config(text="Loading...", foreground="gray")
            if self.parent.winfo_exists():
                self.parent.after(0, lambda: do_search(self, cache_results=True))
        except Exception as e:
            logger.exception("Failed to load cached search results")
            self.log(f"⚠️  Could not load search cache: {e}")
    
    def _show_favorites_popup(self):
        """Show a popup dialog with the list of favorited datasets."""
        show_favorites_popup(self.parent, self.log, self._download_dataset_direct)
    
    def _do_search(self, cache_results: bool = False):
        """Perform dataset search on HuggingFace."""
        query = self.search_var.get().strip() if hasattr(self, 'search_var') else ""
        logger.info(f"Searching datasets: query='{query}', cache_results={cache_results}")
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
            logger.warning("Download attempted but no dataset selected")
            self.log("❌ Cannot download: No dataset selected or found")
            messagebox.showinfo(
                "No Selection", 
                "Please select a dataset from the list to download.\n\n"
                "Tip: Click on a dataset in the results table to select it."
            )
            return
        
        # Get dataset info for logging
        dataset_name = dataset.get("full_name", dataset.get("name", "Unknown"))
        size_mb = dataset.get('size_gb', 0) * 1024
        
        logger.info(f"Starting download: dataset={dataset_name}, size={size_mb:.1f}MB")
        
        # Capture download location in main thread (tkinter vars can't be accessed from background)
        download_location = self.download_location.get()
        
        # Start download (validation and confirmation happen in download_dataset)
        self.cancel_download = False
        self.cancel_btn.config(state="normal")
        self.status_label.config(text=f"Downloading {dataset_name}...")
        
        logger.debug(f"Launching download task for {dataset_name}")
        try:
            job = submit_background(
                "dataset-download",
                download_dataset,
                self,
                dataset,
                download_location,  # Pass pre-captured location
                pool=self._worker_pool,
            )
            self._register_background_future(job)
        except RuntimeError:
            logger.error("Failed to queue dataset download", exc_info=True)
            self.cancel_btn.config(state="disabled")
            self.status_label.config(text="Ready")
            self.log("❌ Unable to start download: background workers unavailable")
            try:
                messagebox.showerror(
                    "Download Unavailable",
                    "Background workers are unavailable. Please try again once the dispatcher is ready.",
                )
            except Exception:
                pass
            return

        self._download_job = job
        job.add_done_callback(lambda _: setattr(self, "_download_job", None))
    
    def _favorite_selected(self):
        """Add or remove the selected dataset from favorites."""
        self.log("⭐ Favorite button clicked")
        dataset = get_selected_dataset(self)
        if not dataset:
            logger.warning("Favorite attempted but no dataset selected")
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
            logger.info(f"Removing dataset from favorites: {dataset_name}")
            if remove_favorite(dataset_id):
                self.log(f"💔 Removed '{dataset_name}' from favorites")
                messagebox.showinfo("Success", f"Removed from favorites:\n{dataset_name}")
                # Refresh to update star indicator
                do_search(self)
            else:
                logger.error(f"Failed to remove dataset from favorites: {dataset_name}")
                messagebox.showerror("Error", "Failed to remove from favorites")
        else:
            # Add to favorites
            logger.info(f"Adding dataset to favorites: {dataset_name}")
            if add_favorite(dataset):
                self.log(f"⭐ Added '{dataset_name}' to favorites")
                messagebox.showinfo("Success", f"Added to favorites:\n{dataset_name}")
                # Refresh to show star
                do_search(self)
            else:
                logger.debug(f"Dataset already in favorites: {dataset_name}")
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
        
        try:
            job = submit_background(
                "dataset-download",
                download_dataset,
                self,
                dataset,
                pool=self._worker_pool,
            )
            self._register_background_future(job)
        except RuntimeError:
            logger.error("Failed to queue dataset download from favorites", exc_info=True)
            self.cancel_btn.config(state="disabled")
            self.status_label.config(text="Ready")
            self.log("❌ Unable to start download: background workers unavailable")
            try:
                messagebox.showerror(
                    "Download Unavailable",
                    "Background workers are unavailable. Please try again once the dispatcher is ready.",
                )
            except Exception:
                pass
            return

        self._download_job = job
        job.add_done_callback(lambda _: setattr(self, "_download_job", None))
    
    def _start_log_queue_pump(self) -> None:
        """Begin periodic draining of the log queue on the Tk thread."""
        try:
            if self._panel_active and self.parent.winfo_exists():
                self.parent.after(0, self._drain_log_queue)
        except Exception:
            # If parent is unavailable just skip; logging will be best-effort
            pass

    def _drain_log_queue(self) -> None:
        """Drain queued log messages and write them on the main thread."""
        if not self._panel_active:
            return

        max_batch = 75
        processed = 0
        has_more = False

        while processed < max_batch:
            try:
                message = self._log_queue.get_nowait()
            except Empty:
                break
            except Exception:
                break

            self._write_log_message(message)
            processed += 1

        if processed == max_batch:
            has_more = True
        else:
            try:
                has_more = not self._log_queue.empty()
            except Exception:
                has_more = False

        delay_ms = 10 if has_more else 50

        try:
            if self._panel_active and self.parent.winfo_exists():
                self.parent.after(delay_ms, self._drain_log_queue)
        except Exception:
            pass

    def _write_log_message(self, message: str) -> None:
        """Render a single log message in the panel output and shared log."""
        if not self._panel_active:
            try:
                if self._original_log:
                    self._original_log(message)
            except Exception:
                pass
            return

        if self.output_text is not None:
            try:
                try:
                    yview = self.output_text.yview()
                    at_bottom = yview[1] >= 0.95
                except Exception:
                    at_bottom = True

                self.output_text.insert(tk.END, message + "\n")
                if at_bottom:
                    self.output_text.see(tk.END)
            except Exception:
                pass

        try:
            if self._original_log:
                self._original_log(message)
        except Exception:
            pass

    def _log_to_output(self, message: str) -> None:
        """Queue log message so UI updates happen on the Tk thread."""
        if not self._panel_active:
            try:
                if self._original_log:
                    self._original_log(message)
            except Exception:
                pass
            return

        try:
            self._log_queue.put(message, block=False)
        except Exception:
            # Fallback to direct write if queueing fails
            self._write_log_message(message)
    
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
        logger.info("User cancelled download")
        self.cancel_download = True
        self.log("❌ Cancelling download...")

    def cleanup(self) -> None:
        """Signal background tasks to stop and release panel resources."""
        self._panel_active = False
        self.cancel_download = True
        self._cancel_background_jobs()

        job = getattr(self, "_download_job", None)
        if job is not None:
            try:
                job.cancel()
            except Exception:
                pass

        queue_ref = getattr(self, "_log_queue", None)
        if isinstance(queue_ref, Queue):
            try:
                while True:
                    queue_ref.get_nowait()
            except Empty:
                pass
            except Exception:
                pass
        self._log_queue = None

        noop = (lambda *_args, **_kwargs: None)
        self.log = self._original_log or noop

    def _register_background_future(self, future: Optional[Future]) -> None:
        if future is None:
            return

        with self._background_lock:
            self._background_jobs.add(future)

        def _cleanup(done_future: Future) -> None:
            # Log any exceptions from the background task
            try:
                exc = done_future.exception()
                if exc:
                    logger.error(f"Background task failed with exception: {exc}", exc_info=exc)
                    self.log(f"❌ Background task error: {exc}")
            except Exception as e:
                logger.debug(f"Could not retrieve exception from future: {e}")
            
            with self._background_lock:
                self._background_jobs.discard(done_future)

        future.add_done_callback(_cleanup)

    def _cancel_background_jobs(self, wait_timeout: float = 2.0) -> None:
        with self._background_lock:
            jobs = list(self._background_jobs)

        if not jobs:
            return

        for future in jobs:
            try:
                if not future.done():
                    future.cancel()
            except Exception:
                pass

        if wait_timeout <= 0:
            return

        deadline = time.monotonic() + wait_timeout
        while time.monotonic() < deadline:
            if all(job.done() for job in jobs):
                break
            time.sleep(0.05)

        with self._background_lock:
            self._background_jobs = {job for job in self._background_jobs if not job.done()}
