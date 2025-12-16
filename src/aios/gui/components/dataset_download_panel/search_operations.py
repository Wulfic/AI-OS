"""
Search Operations

Helper functions for searching, filtering, displaying, and sorting dataset search results.
"""

import logging
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from tkinter import messagebox
from typing import Dict, List, Any, Optional

from .hf_search import search_huggingface_datasets, list_datasets
from .favorites_manager import get_favorite_ids
from .cache_manager import save_search_cache
from .hf_size_detection import enrich_dataset_with_size, format_size_display
from ...utils.resource_management import submit_background

logger = logging.getLogger(__name__)

# Enrichment settings - these control how many datasets get size/row info
MAX_ENRICH_DATASETS = 50  # Increased from 12 to cover more search results
ENRICH_CONCURRENCY = 8    # Increased from 4 for faster enrichment
ENRICH_TIME_BUDGET = 30.0  # seconds - increased from 20s for more datasets


@dataclass(frozen=True)
class SearchDisplayPayload:
    rows: List[tuple[str, str, str, str, str, str, str]]
    preview_names: List[str]
    count: int


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


def build_display_payload(results: List[Dict[str, Any]]) -> SearchDisplayPayload:
    """Prepare immutable payload for rendering dataset search results on the UI thread."""

    favorites_lookup = get_favorite_ids()
    preview_limit = 5
    preview_names: List[str] = []
    rows: List[tuple[str, str, str, str, str, str, str]] = []

    for ds in results:
        downloads = ds.get("downloads", 0)
        if downloads >= 1_000_000:
            downloads_str = f"{downloads / 1_000_000:.1f}M"
        elif downloads >= 1_000:
            downloads_str = f"{downloads / 1_000:.1f}K"
        else:
            downloads_str = str(downloads)

        dataset_name = ds.get("full_name", ds.get("name", "Unknown"))
        dataset_id = ds.get("id") or ds.get("path", "")
        if dataset_id and str(dataset_id) in favorites_lookup:
            dataset_name = f"⭐ {dataset_name}"

        size_str, rows_str, blocks_str = format_size_display(ds)

        description = ds.get("description", "No description")
        if len(description) > 80:
            description = description[:80] + "..."

        rows.append(
            (
                dataset_name,
                downloads_str,
                size_str,
                rows_str,
                blocks_str,
                description,
                str(dataset_id or ""),
            )
        )

        if len(preview_names) < preview_limit:
            preview_names.append(dataset_name)

    return SearchDisplayPayload(rows=rows, preview_names=preview_names, count=len(results))


def display_search_results(
    payload: SearchDisplayPayload,
    query: str, 
    total_before_filter: Optional[int],
    results_tree: tk.Widget,
    search_status_label: tk.Widget,
    status_label: tk.Widget,
    log_func,
    *,
    log_previews: bool = True,
    completion_message: Optional[str] = None,
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
    start_time = time.perf_counter()
    logger.debug(
        "display_search_results start: %d rows (query='%s', total_before_filter=%s)",
        payload.count,
        query,
        total_before_filter,
    )

    # Clear existing items (handle gracefully if tree is mid-update)
    existing_items = list(results_tree.get_children())
    if existing_items:
        try:
            results_tree.delete(*existing_items)
        except Exception:
            for item in existing_items:
                try:
                    results_tree.delete(item)
                except Exception:
                    logger.debug("Failed to delete tree item %s", item, exc_info=True)
    
    if payload.count == 0:
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
    
    query_text = f"'{query}'" if query else "popular datasets"
    filter_text = ""
    if total_before_filter and total_before_filter > payload.count:
        filter_text = f" (filtered from {total_before_filter})"

    def _finalize_ui() -> None:
        search_status_label.config(
            text=f"✅ Found {payload.count} results for {query_text}{filter_text}",
            foreground="green"
        )
        status_label.config(text="Ready")

        if log_previews and payload.preview_names:
            remaining = payload.count - len(payload.preview_names)
            if remaining > 0:
                log_func(
                    "📦 Added datasets: "
                    + ", ".join(payload.preview_names)
                    + f" ... (+{remaining} more)"
                )
            else:
                log_func("📦 Added datasets: " + ", ".join(payload.preview_names))
        try:
            message_template = completion_message or "✅ Search complete: {count} datasets loaded"
            log_func(message_template.format(count=payload.count))
        except Exception:
            log_func("✅ Search results updated")

        elapsed = time.perf_counter() - start_time
        logger.debug(
            "display_search_results end: inserted %d items in %.3fs",
            payload.count,
            elapsed,
        )

    batch_size = 12
    after_delay_ms = 8

    def _insert_batch(start: int = 0) -> None:
        end = min(start + batch_size, len(payload.rows))
        for dataset_name, downloads_str, size_str, rows_str, blocks_str, description, dataset_id in payload.rows[start:end]:
            try:
                results_tree.insert(
                    "",
                    "end",
                    text=dataset_name,
                    values=(downloads_str, size_str, rows_str, blocks_str, description),
                    tags=(dataset_id,),
                )
            except Exception:
                logger.debug("Tree insert failed for dataset %s", dataset_name, exc_info=True)
        if end < len(payload.rows):
            try:
                results_tree.after(after_delay_ms, lambda: _insert_batch(end))
            except Exception:
                _insert_batch(end)
        else:
            _finalize_ui()

    _insert_batch(0)


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
    if not getattr(panel, "_panel_active", True):
        return

    query = panel.search_var.get().strip()
    
    if list_datasets is None:
        messagebox.showerror(
            "Library Missing",
            "huggingface_hub library is not installed.\n\n"
            "Install it with: pip install huggingface_hub"
        )
        return
    
    # Use gray for "in progress" status to be theme-friendly
    panel.search_status_label.config(text="🔄 Searching...", foreground="gray")
    panel.status_label.config(text="Searching HuggingFace Hub...")
    try:
        panel.frame.update_idletasks()
    except Exception:
        pass
    
    # Capture filter values in main thread before spawning background thread
    try:
        max_size_text = panel.max_size_var.get().strip()
        size_unit = panel.size_unit_var.get()
    except Exception:
        max_size_text = ""
        size_unit = "GB"
    
    # Capture modality filter
    try:
        modality = panel.modality_var.get() if hasattr(panel, 'modality_var') else "Text"
    except Exception:
        modality = "Text"  # Default to Text
    
    def _submit_background_task(fn, name: str):
        if not getattr(panel, "_panel_active", True):
            return
        try:
            future = submit_background(name, fn, pool=getattr(panel, "_worker_pool", None))
            register = getattr(panel, "_register_background_future", None)
            if callable(register):
                register(future)
            logger.debug("Submitted background task '%s' to worker pool", name)
        except RuntimeError as exc:
            logger.error("Failed to queue background task '%s': %s", name, exc)
            panel.log(f"❌ Background task queue is full: {name}")

    def search_thread():
        if not getattr(panel, "_panel_active", True):
            return
        logger.debug(f"Starting dataset search thread for query: {query}, modality: {modality}")
        try:
            results = search_huggingface_datasets(query, limit=50, modality=modality)
            logger.debug(f"Dataset search found {len(results)} results")
            
            # Apply size filter immediately (before enrichment)
            filtered_results = apply_size_filter_threadsafe(results, max_size_text, size_unit, panel.log)
            logger.debug(f"Dataset search completed with {len(filtered_results)} filtered results")
            
            if not getattr(panel, "_panel_active", True):
                return

            panel.search_results = filtered_results
            payload = build_display_payload(filtered_results)

            # Display results IMMEDIATELY - don't wait for enrichment
            try:
                if getattr(panel, "_panel_active", True) and panel.frame.winfo_exists():
                    panel.frame.after(
                        0,
                        lambda p=payload: display_search_results(
                            p,
                            query,
                            len(results),
                            panel.results_tree,
                            panel.search_status_label,
                            panel.status_label,
                            panel.log,
                        ),
                    )
                    # Save to cache if requested
                    if cache_results:
                        panel.frame.after(100, lambda: save_search_cache(query, filtered_results))
            except Exception:
                pass  # Widget destroyed, ignore
            
            # NOW enrich datasets in the background (non-blocking, fire-and-forget)
            def background_enrich():
                """Enrich datasets in background without blocking UI."""
                if not getattr(panel, "_panel_active", True):
                    return

                datasets_to_enrich = filtered_results[:MAX_ENRICH_DATASETS]
                total = len(datasets_to_enrich)
                overflow = max(0, len(filtered_results) - total)

                if total == 0:
                    return

                panel.log(
                    f"🔍 Enriching {total} dataset{'s' if total != 1 else ''} with size information..."
                )
                if overflow:
                    panel.log(
                        f"ℹ️ Showing lightweight metadata for {overflow} additional result(s); refine your search to enrich everything."
                    )

                start_time = time.perf_counter()
                deadline = start_time + ENRICH_TIME_BUDGET
                enriched_count = 0
                failed_count = 0
                abort_reason: Optional[str] = None

                def _should_abort() -> bool:
                    if not getattr(panel, "_panel_active", True):
                        return True
                    return time.perf_counter() >= deadline

                def _worker(payload: tuple[int, Dict[str, Any]]) -> tuple[int, bool]:
                    idx, dataset = payload
                    dataset_id = dataset.get("id", "unknown")
                    if not getattr(panel, "_panel_active", True):
                        return idx, False
                    try:
                        logger.debug(f"📊 [WORKER {idx}] Starting enrichment for {dataset_id}")
                        enrich_dataset_with_size(dataset, timeout=2.0)
                        # Check if enrichment actually worked
                        has_size = "size_gb" in dataset and dataset.get("size_gb", 0) > 0
                        has_rows = "num_rows" in dataset and dataset.get("num_rows", 0) > 0
                        if has_size or has_rows:
                            logger.debug(f"✅ [WORKER {idx}] Enriched {dataset_id} successfully")
                            return idx, True
                        else:
                            logger.warning(f"⚠️ [WORKER {idx}] Enrichment returned but no size/rows for {dataset_id}")
                            return idx, False
                    except Exception as exc:  # pragma: no cover - defensive logging
                        logger.warning(f"❌ [WORKER {idx}] Could not enrich {dataset_id}: {type(exc).__name__}: {exc}")
                        return idx, False

                executor = ThreadPoolExecutor(
                    max_workers=min(ENRICH_CONCURRENCY, total),
                    thread_name_prefix="DatasetEnrich",
                )
                futures = {
                    executor.submit(_worker, (idx, dataset)): idx
                    for idx, dataset in enumerate(datasets_to_enrich, start=1)
                }

                try:
                    pending = set(futures.keys())

                    while pending:
                        if _should_abort():
                            abort_reason = "timeout" if time.perf_counter() >= deadline else "panel"
                            break

                        remaining = deadline - time.perf_counter()
                        if remaining <= 0:
                            abort_reason = "timeout"
                            break

                        wait_timeout = min(0.75, remaining)
                        done, pending = wait(pending, timeout=wait_timeout, return_when=FIRST_COMPLETED)

                        if not done:
                            continue

                        for future in done:
                            idx = futures.get(future)
                            if idx is None:
                                continue
                            try:
                                idx, ok = future.result()
                            except Exception:
                                ok = False
                            if ok:
                                enriched_count += 1
                                # Log progress more frequently and update display periodically
                                if enriched_count % 10 == 0 and getattr(panel, "_panel_active", True):
                                    panel.log(f"ℹ️ Enriched {enriched_count}/{total} datasets so far")
                                    # Update display mid-enrichment for better UX
                                    if panel.frame.winfo_exists():
                                        try:
                                            mid_payload = build_display_payload(filtered_results)
                                            panel.frame.after(
                                                0,
                                                lambda p=mid_payload: display_search_results(
                                                    p,
                                                    query,
                                                    len(results),
                                                    panel.results_tree,
                                                    panel.search_status_label,
                                                    panel.status_label,
                                                    panel.log,
                                                    log_previews=False,
                                                    completion_message=None,
                                                ),
                                            )
                                        except Exception:
                                            pass  # Non-critical
                            else:
                                failed_count += 1

                    if abort_reason is not None:
                        for future in pending:
                            try:
                                future.cancel()
                            except Exception:
                                pass
                finally:
                    executor.shutdown(wait=abort_reason is None, cancel_futures=abort_reason is not None)

                elapsed = time.perf_counter() - start_time
                logger.info(
                    "📈 [ENRICHMENT] Complete: %d/%d succeeded in %.1fs (failed=%d, abort=%s)",
                    enriched_count,
                    total,
                    elapsed,
                    failed_count,
                    abort_reason,
                )
                
                # Log which datasets failed to enrich
                if failed_count > 0:
                    failed_names = []
                    for dataset in datasets_to_enrich:
                        if "size_gb" not in dataset or dataset.get("size_gb", 0) == 0:
                            failed_names.append(dataset.get("id", "unknown"))
                    if failed_names[:5]:  # Show first 5
                        logger.warning(f"⚠️ [ENRICHMENT] Failed datasets (showing {min(5, len(failed_names))}/{len(failed_names)}): {', '.join(failed_names[:5])}")

                if abort_reason == "timeout" and getattr(panel, "_panel_active", True):
                    panel.log("⚠️ Size enrichment timed out; leaving remaining datasets unprocessed")
                elif abort_reason == "panel":
                    panel.log("ℹ️ Size enrichment stopped during shutdown")

                if getattr(panel, "_panel_active", True) and panel.frame.winfo_exists():
                    final_payload = build_display_payload(filtered_results)
                    try:
                        panel.frame.after(
                            0,
                            lambda p=final_payload: display_search_results(
                                p,
                                query,
                                len(results),
                                panel.results_tree,
                                panel.search_status_label,
                                panel.status_label,
                                panel.log,
                                log_previews=False,
                                completion_message="🔁 Search metadata refreshed ({count} datasets)",
                            ),
                        )
                    except Exception:
                        logger.debug("Failed to schedule final enrichment update", exc_info=True)

                if getattr(panel, "_panel_active", True):
                    panel.log(
                        f"🔁 Dataset metadata refreshed ({enriched_count}/{total} detailed entries)"
                    )
            
            # Start background enrichment using worker pool when available
            _submit_background_task(background_enrich, "Dataset-Enrich")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Dataset search thread failed: {error_msg}")
            # Update UI in main thread - check if widget still exists
            try:
                if getattr(panel, "_panel_active", True) and panel.frame.winfo_exists():
                    panel.frame.after(0, lambda: search_error(error_msg, panel.search_status_label, panel.status_label, panel.log))
            except Exception:
                pass  # Widget destroyed, ignore
        finally:
            logger.debug("Dataset search thread exiting")
    
    logger.debug(f"Starting dataset search background thread for: {query}")
    _submit_background_task(search_thread, "HF-Search")


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
