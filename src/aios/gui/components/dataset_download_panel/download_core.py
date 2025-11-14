"""
Dataset Download Core Logic

Core functions for downloading datasets from HuggingFace Hub.
"""

import logging
import os
import sys
import threading
from pathlib import Path
from tkinter import messagebox
from typing import Dict, Any, Optional

from .progress_capture import RealTimeProgressCapture
from .config_selector import show_config_selector

logger = logging.getLogger(__name__)

# Import stream manager for coordinating concurrent dataset access
try:
    from ....data.stream_manager import get_stream_manager
except ImportError:
    get_stream_manager = None  # type: ignore


def download_dataset(panel, dataset: Dict[str, Any]):
    """
    Download a single dataset in background thread.
    
    Args:
        panel: DatasetDownloadPanel instance with all required attributes
        dataset: Dataset information dictionary
    """
    from datasets import load_dataset, Dataset, get_dataset_config_names
    
    dataset_name = dataset.get("full_name", dataset.get("name", "Unknown"))
    dataset_path = dataset.get("path")
    
    logger.info(f"Starting download: dataset={dataset_name}, path={dataset_path}")
    
    # Use the global HF cache directory
    cache_dir = Path(os.environ.get("HF_HOME", str(Path.cwd() / "training_data" / "hf_cache")))
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Output path
    output_name = dataset_name.replace("/", "_").replace(" ", "_").lower()
    output_path = Path(panel.download_location.get()) / output_name
    
    logger.info(f"Destination: {output_path}")
    
    panel.log(f"\n{'='*60}")
    panel.log(f"📥 Downloading: {dataset_name}")
    panel.log(f"   Path: {dataset_path}")
    panel.log(f"   Output: {output_path}")
    panel.log(f"{'='*60}")
    
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
        
        # Download with streaming
        max_samples = dataset.get("max_samples", 0)  # 0 = unlimited
        if max_samples == 0:
            panel.log(f"   Streaming entire dataset (unlimited)...")
        else:
            panel.log(f"   Streaming up to {max_samples:,} samples...")
        
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
        
        # Stream and save samples
        samples = []
        last_update = 0
        # For unlimited downloads, update every 10k samples; otherwise every 1% of max
        update_interval = 10000 if max_samples == 0 else max(1000, max_samples // 100)
        check_pause_interval = 100  # Check for pause every 100 samples
        
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
            
            samples.append(sample)
            
            # Progress updates
            if j - last_update >= update_interval:
                if max_samples == 0:
                    # Unlimited: show absolute count
                    panel.log(f"   Progress: {j:,} samples downloaded...")
                else:
                    # Limited: show percentage
                    pct = int(100 * j / max_samples)
                    panel.log(f"   Progress: {j:,}/{max_samples:,} ({pct}%)")
                last_update = j
            
            # Stop if we reached the limit (only if max_samples > 0)
            if max_samples > 0 and j >= max_samples:
                panel.log(f"   ℹ️ Reached max_samples limit ({max_samples:,})")
                break
        
        # Check if stream ended naturally
        if max_samples > 0 and len(samples) < max_samples and not panel.cancel_download:
            panel.log(f"   ℹ️ Dataset exhausted at {len(samples):,} samples (fewer than max {max_samples:,})")
        elif max_samples == 0 and not panel.cancel_download:
            panel.log(f"   ℹ️ Dataset complete: {len(samples):,} total samples")
        
        if not panel.cancel_download and samples:
            # Convert and save
            samples_count = len(samples)
            logger.info(f"Download progress: {samples_count:,} samples collected")
            panel.log(f"   💾 Saving {samples_count:,} samples...")
            ds = Dataset.from_dict({
                key: [s[key] for s in samples]
                for key in samples[0].keys()
            })
            ds.save_to_disk(str(output_path))
            file_size_mb = sum(f.stat().st_size for f in Path(output_path).rglob('*') if f.is_file()) / (1024 * 1024)
            logger.info(f"Download completed: {dataset_name} ({samples_count:,} samples, {file_size_mb:.1f} MB)")
            panel.log(f"   ✅ Success: {samples_count:,} samples saved to {output_path}")
            
            # Cache the downloaded chunk for future use
            if chunk_cache and samples:
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
            
            try:
                panel.frame.after(0, lambda: messagebox.showinfo(
                    "Download Complete",
                    f"Successfully downloaded:\n{dataset_name}\n\n"
                    f"Samples: {samples_count:,}\n"
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
            except Exception:
                pass

        try:
            panel.frame.after(0, _finalize_ui)
        except Exception:
            _finalize_ui()

        logger.debug("Download operation completed")
