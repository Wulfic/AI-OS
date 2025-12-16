"""
Dataset Download Core Logic

Core functions for downloading datasets from HuggingFace Hub.
Supports streaming downloads with automatic 100k block creation.
Includes pre-download validation for storage limits and disk space.
Tracks download progress with speed and ETA display.
"""

import logging
import os
import sys
import threading
import time
from pathlib import Path
from tkinter import messagebox
from typing import Dict, Any, Optional

from .progress_capture import RealTimeProgressCapture
from .config_selector import show_config_selector
from .block_processor import StreamingBlockWriter, DEFAULT_BLOCK_SIZE
from .download_validation import validate_download_prerequisites, show_download_confirmation_dialog
from .download_progress import DownloadProgressTracker, DownloadStats

logger = logging.getLogger(__name__)

# Import stream manager for coordinating concurrent dataset access
try:
    from ....data.stream_manager import get_stream_manager
except ImportError:
    get_stream_manager = None  # type: ignore


def download_dataset(panel, dataset: Dict[str, Any], download_location: str):
    """
    Download a single dataset in background thread.
    
    Includes pre-download validation for:
    - Dataset size within storage cap
    - Sufficient disk space available
    - User confirmation with size/row information
    
    Args:
        panel: DatasetDownloadPanel instance with all required attributes
        dataset: Dataset information dictionary
        download_location: Pre-captured download location path (captured in main thread)
    """
    from datasets import load_dataset, Dataset, get_dataset_config_names
    
    dataset_name = dataset.get("full_name", dataset.get("name", "Unknown"))
    dataset_path = dataset.get("path")
    
    logger.info(f"Starting download: dataset={dataset_name}, path={dataset_path}")
    
    try:
        # Use the global HF cache directory
        cache_dir = Path(os.environ.get("HF_HOME", str(Path.cwd() / "training_datasets" / "hf_cache")))
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Output path - use the pre-captured download_location (not panel.download_location.get() which would crash)
        output_name = dataset_name.replace("/", "_").replace(" ", "_").lower()
        output_path = Path(download_location) / output_name
        
        logger.info(f"Destination: {output_path}")
    except Exception as e:
        logger.exception(f"Error setting up paths for {dataset_name}: {e}")
        panel.log(f"❌ Error setting up download paths: {e}")
        return
    
    # PRE-DOWNLOAD VALIDATION
    panel.log(f"\n{'='*60}")
    panel.log(f"📥 Preparing download: {dataset_name}")
    panel.log(f"{'='*60}")
    
    try:
        # Validate prerequisites (storage cap, disk space)
        logger.info(f"Validating download prerequisites for {dataset_name}...")
        can_proceed, error_msg = validate_download_prerequisites(
            dataset,
            output_path,
            parent_widget=panel.frame
        )
        logger.info(f"Validation result: can_proceed={can_proceed}, error={error_msg}")
        
        if not can_proceed:
            panel.log(f"   ❌ Download blocked: {error_msg}")
            logger.warning(f"Download blocked for {dataset_name}: {error_msg}")
            return
        
        # Show confirmation dialog with size/row info
        logger.info(f"Showing confirmation dialog for {dataset_name}...")
        confirmed = show_download_confirmation_dialog(dataset, parent_widget=panel.frame)
        logger.info(f"Confirmation result: {confirmed}")
        if not confirmed:
            panel.log(f"   ❌ Download cancelled by user")
            logger.info(f"Download cancelled by user: {dataset_name}")
            return
    except Exception as e:
        logger.exception(f"Validation/confirmation failed for {dataset_name}: {e}")
        panel.log(f"   ❌ Validation error: {e}")
        return
    
    # Proceed with download
    logger.info(f"Proceeding with download for {dataset_name}")
    panel.log(f"\n📥 Downloading: {dataset_name}")
    panel.log(f"   Path: {dataset_path}")
    panel.log(f"   Output: {output_path}")
    
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check if config is needed and not provided
        config_name = dataset.get("config")
        if not config_name and dataset_path:
            try:
                # Try to get available configs
                configs = get_dataset_config_names(str(dataset_path))
                if configs and len(configs) > 0:
                    # Multiple configs available - need user to select
                    panel.log(f"   ℹ️ Dataset has {len(configs)} configs available")
                    
                    # Prompt user to select config in main thread
                    selected_config: Optional[str] = None
                    config_event = threading.Event()

                    def _select_config() -> None:
                        nonlocal selected_config
                        try:
                            selected_config = show_config_selector(dataset_name, configs, panel.frame)
                        finally:
                            config_event.set()

                    # Schedule with safety check
                    try:
                        if panel.frame.winfo_exists():
                            panel.frame.after(0, _select_config)
                        else:
                            panel.log("   ❌ Download cancelled: Widget destroyed")
                            return
                    except Exception:
                        panel.log("   ❌ Download cancelled: Widget destroyed")
                        return

                    if not config_event.wait(timeout=30):
                        panel.log("   ❌ Download cancelled: No config selected")
                        return

                    if not selected_config:
                        panel.log("   ❌ Download cancelled: No config selected")
                        return

                    config_name = selected_config
                    panel.log(f"   ✓ Using config: {config_name}")
            except Exception as e:
                # If we can't get configs, try without config (might work for some datasets)
                panel.log(f"   ⚠️ Could not check configs: {e}")
        
        # Download with streaming - save in 100k blocks for efficient training
        max_samples = dataset.get("max_samples", 0)  # 0 = unlimited
        use_block_format = max_samples == 0 or max_samples >= DEFAULT_BLOCK_SIZE
        
        if max_samples == 0:
            panel.log(f"   Streaming entire dataset (saving in {DEFAULT_BLOCK_SIZE:,}-sample blocks)...")
        else:
            panel.log(f"   Streaming up to {max_samples:,} samples...")
            if use_block_format:
                panel.log(f"   📦 Using block format ({DEFAULT_BLOCK_SIZE:,} samples per block)...")
        
        # Get streaming cache for chunk caching
        chunk_cache = None
        try:
            from ....data.streaming_cache import get_cache
            chunk_cache = get_cache()
        except ImportError:
            pass  # Cache not available
        
        # Calculate chunk index for caching (rotate through chunks)
        chunk_index = 0
        if max_samples > 0:
            chunk_index = (max_samples // 10000) % 5  # Rotate through 5 chunks
        
        load_kwargs = {
            "path": dataset_path,
            "split": dataset.get("split", "train"),
            "streaming": True,
            "trust_remote_code": False,
            "cache_dir": str(cache_dir),
        }
        
        if config_name:
            load_kwargs["name"] = config_name
        
        # Capture progress
        progress_capture = RealTimeProgressCapture(panel.log, panel.frame)
        old_stderr = sys.stderr
        
        try:
            sys.stderr = progress_capture
            dataset_stream = load_dataset(**load_kwargs)
        finally:
            sys.stderr = old_stderr
            progress_capture.flush()
        
        # Register download with stream manager
        if get_stream_manager and dataset_path:
            stream_mgr = get_stream_manager()
            panel.download_pause_event.clear()
            panel.current_download_dataset_id = str(dataset_path)
            
            if not stream_mgr.register_download(str(dataset_path), panel.download_pause_event):
                panel.log("   ⚠️ Cannot download: training is active on this dataset")
                panel.log("   💡 Wait for training to complete or use a different dataset")
                return
        
        # Initialize block writer for streaming block-format downloads
        block_writer: Optional[StreamingBlockWriter] = None
        if use_block_format:
            block_writer = StreamingBlockWriter(
                output_dir=output_path,
                dataset_name=dataset_name,
                block_size=DEFAULT_BLOCK_SIZE,
                log_callback=panel.log,
            )
        
        # Stream and save samples
        samples = []
        last_update = 0
        last_progress_ui_update = -1.0  # Start negative to ensure immediate first update
        # For unlimited downloads, update every 10k samples; otherwise every 1% of max
        update_interval = 10000 if max_samples == 0 else max(1000, max_samples // 100)
        check_pause_interval = 100  # Check for pause every 100 samples
        ui_update_interval = 0.1  # Update UI every 100ms for smooth display
        
        # Initialize progress tracker
        total_bytes_estimate = dataset.get("num_bytes", 0) or int(dataset.get("size_gb", 0) * 1024 * 1024 * 1024)
        total_samples_count = dataset.get("num_rows", 0) or max_samples
        
        # Estimate bytes per sample based on dataset type
        bytes_per_sample = 500  # Default for text
        if dataset.get("has_images"):
            bytes_per_sample = 100_000  # 100KB for images
        elif dataset.get("has_audio"):
            bytes_per_sample = 500_000  # 500KB for audio
        elif dataset.get("has_video"):
            bytes_per_sample = 10_000_000  # 10MB for video
        
        def on_progress_update(stats: DownloadStats):
            """Callback to update UI with progress."""
            nonlocal last_progress_ui_update
            now = time.monotonic()
            if now - last_progress_ui_update >= ui_update_interval:
                last_progress_ui_update = now
                try:
                    if panel.frame.winfo_exists():
                        from .ui_builder import _update_progress_display
                        panel.frame.after(0, lambda s=stats: _update_progress_display(panel, s))
                except Exception as e:
                    logger.debug(f"Progress UI update error: {e}")
        
        # Calculate total blocks based on dataset size
        total_blocks = dataset.get("total_blocks", 0)
        if not total_blocks and total_samples_count > 0:
            # Calculate from samples (100k per block)
            total_blocks = (total_samples_count + DEFAULT_BLOCK_SIZE - 1) // DEFAULT_BLOCK_SIZE
        
        progress_tracker = DownloadProgressTracker(
            total_bytes=total_bytes_estimate,
            total_samples=total_samples_count,
            total_blocks=total_blocks,
            speed_window_seconds=3.0,
            update_callback=on_progress_update,
        )
        progress_tracker.start()
        
        # Log initial tracking info
        if total_bytes_estimate > 0:
            panel.log(f"   📊 Tracking: {total_bytes_estimate / (1024*1024*1024):.2f} GB estimated")
        elif total_samples_count > 0:
            panel.log(f"   📊 Tracking: {total_samples_count:,} samples expected")
        else:
            panel.log(f"   📊 Tracking: unknown size (streaming mode)")
        
        for j, sample in enumerate(dataset_stream, 1):
            if panel.cancel_download:
                panel.log("   ❌ Download cancelled by user")
                break
            
            # Check if download should be paused (e.g., training started)
            if get_stream_manager and j % check_pause_interval == 0:
                if panel.download_pause_event.is_set():
                    panel.log("   ⏸️ Download paused: training started on same dataset")
                    panel.log("   ⏳ Waiting for training to complete...")
                    resumed = panel.download_pause_event.wait_for_resume(lambda: panel.cancel_download)
                    if resumed and not panel.cancel_download:
                        panel.log("   ▶️ Download resumed: training completed")
            
            # Add sample to block writer OR collect in memory
            if block_writer:
                block_metadata = block_writer.add_sample(dict(sample))
                # Update blocks_completed when a block is flushed
                if block_metadata:
                    progress_tracker.set_progress(
                        bytes_downloaded=progress_tracker._bytes_downloaded,
                        samples_downloaded=j,
                        blocks_completed=len(block_writer.blocks)
                    )
            else:
                samples.append(sample)
            
            # Update progress tracker (estimates bytes from sample count)
            progress_tracker.update_samples(1, bytes_per_sample)
            
            # Progress updates (log messages)
            if j - last_update >= update_interval:
                if max_samples == 0:
                    # Unlimited: show absolute count
                    blocks_saved = len(block_writer.blocks) if block_writer else 0
                    panel.log(f"   Progress: {j:,} samples ({blocks_saved} blocks saved)...")
                else:
                    # Limited: show percentage
                    pct = int(100 * j / max_samples)
                    panel.log(f"   Progress: {j:,}/{max_samples:,} ({pct}%)")
                last_update = j
            
            # Stop if we reached the limit (only if max_samples > 0)
            if max_samples > 0 and j >= max_samples:
                panel.log(f"   ℹ️ Reached max_samples limit ({max_samples:,})")
                break
        
        # Finalize and get stats
        total_samples_downloaded = 0
        
        # Mark progress as complete or cancelled
        if panel.cancel_download:
            progress_tracker.cancel()
        else:
            progress_tracker.complete()
        
        if block_writer:
            # Finalize block writer (flushes remaining samples)
            if not panel.cancel_download:
                block_info = block_writer.finalize()
                total_samples_downloaded = block_info.total_samples
                panel.log(f"   ✅ Saved {block_info.total_blocks} blocks ({total_samples_downloaded:,} samples)")
            else:
                # Save what we have even if cancelled
                try:
                    block_info = block_writer.finalize()
                    total_samples_downloaded = block_info.total_samples
                    panel.log(f"   ⚠️ Partial save: {block_info.total_blocks} blocks ({total_samples_downloaded:,} samples)")
                except Exception:
                    pass
        else:
            # Original flow for small datasets (< 100k samples)
            total_samples_downloaded = len(samples)
            
            # Check if stream ended naturally
            if max_samples > 0 and len(samples) < max_samples and not panel.cancel_download:
                panel.log(f"   ℹ️ Dataset exhausted at {len(samples):,} samples (fewer than max {max_samples:,})")
            elif max_samples == 0 and not panel.cancel_download:
                panel.log(f"   ℹ️ Dataset complete: {len(samples):,} total samples")
            
            if not panel.cancel_download and samples:
                # Convert and save (legacy format for small datasets)
                samples_count = len(samples)
                logger.info(f"Download progress: {samples_count:,} samples collected")
                panel.log(f"   💾 Saving {samples_count:,} samples...")
                ds = Dataset.from_dict({
                    key: [s[key] for s in samples]
                    for key in samples[0].keys()
                })
                ds.save_to_disk(str(output_path))
        
        # Calculate final stats
        if total_samples_downloaded > 0:
            file_size_mb = sum(f.stat().st_size for f in Path(output_path).rglob('*') if f.is_file()) / (1024 * 1024)
            logger.info(f"Download completed: {dataset_name} ({total_samples_downloaded:,} samples, {file_size_mb:.1f} MB)")
            
            # Cache samples for future use (only for non-block format)
            if chunk_cache and samples and not block_writer:
                try:
                    # Extract text from samples for caching
                    text_columns = ['text', 'content', 'sentence', 'document', 'article', 'body', 'input', 'output']
                    found_column = None
                    for col in text_columns:
                        if col in samples[0]:
                            found_column = col
                            break
                    if not found_column and samples[0]:
                        found_column = list(samples[0].keys())[0]
                    
                    if found_column:
                        text_lines = [str(s.get(found_column, "")).strip() for s in samples if s.get(found_column)]
                        if text_lines:
                            chunk_cache.save_chunk(
                                dataset_path=str(dataset_path),
                                config=config_name,
                                split=dataset.get("split", "train"),
                                chunk_index=chunk_index,
                                lines=text_lines
                            )
                            logger.debug(f"Cached chunk for future use: chunk {chunk_index}, {len(text_lines)} lines")
                            panel.log(f"   ✓ Cached chunk for future use (chunk {chunk_index}, {len(text_lines)} lines)")
                except Exception as e:
                    # Non-critical failure
                    logger.debug(f"Failed to cache chunk: {e}")
            
            # Show success message
            if not panel.cancel_download:
                block_info_str = ""
                if block_writer and hasattr(block_writer, 'blocks'):
                    block_info_str = f"\nBlocks: {len(block_writer.blocks)}"
                
                try:
                    panel.frame.after(0, lambda: messagebox.showinfo(
                        "Download Complete",
                        f"Successfully downloaded:\n{dataset_name}\n\n"
                        f"Samples: {total_samples_downloaded:,}{block_info_str}\n"
                        f"Size: {file_size_mb:.1f} MB\n"
                        f"Location: {output_path}"
                    ))
                except Exception:
                    pass
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Download failed: {dataset_name} - {error_msg}")
        panel.log(f"   ❌ Download failed: {error_msg}")
        try:
            panel.frame.after(0, lambda: messagebox.showerror(
                "Download Failed",
                f"Failed to download {dataset_name}:\n\n{error_msg[:200]}"
            ))
        except Exception:
            pass
    
    finally:
        # Unregister download from stream manager
        if get_stream_manager and panel.current_download_dataset_id:
            try:
                stream_mgr = get_stream_manager()
                stream_mgr.unregister_download(panel.current_download_dataset_id)
                panel.current_download_dataset_id = None
                logger.debug(f"Unregistered download from stream manager: {dataset_name}")
            except Exception as e:
                logger.debug(f"Failed to unregister download: {e}")
        
        # Mark download complete
        def _finalize_ui() -> None:
            try:
                panel.cancel_btn.config(state="disabled")
                panel.status_label.config(text="Ready")
                panel._download_job = None
                # Hide progress display
                from .ui_builder import hide_progress_display
                hide_progress_display(panel)
            except Exception:
                pass

        try:
            panel.frame.after(0, _finalize_ui)
        except Exception:
            _finalize_ui()

        logger.debug("Download operation completed")
