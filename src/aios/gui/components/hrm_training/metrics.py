from __future__ import annotations

import json
import os
from typing import Any

"""
Training Metrics and Progress Monitoring

This module handles metrics updates from the JSONL log file. It polls the log file
every second and updates UI elements with training metrics, epoch tracking, and status.

IMPORTANT: Progress bar updates are handled by process_manager.py (_parse_and_update_progress)
which parses stdout in real-time. This module focuses on:
- Metrics display (loss, perplexity, token accuracy, etc.)
- Epoch/chunk/block tracking
- Dataset information
- Generation progress
- Training status messages

The dual monitoring approach:
1. process_manager.py: Real-time stdout parsing → progress bar updates
2. This module: JSONL file polling → metrics and tracking displays
"""


def poll_metrics(panel: Any) -> None:
    if not panel._metrics_polling_active:
        return
    path = panel.log_file_var.get().strip()
    if path and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                try:
                    f.seek(-4096, os.SEEK_END)
                except Exception:
                    pass
                data = f.read().decode("utf-8", errors="ignore").splitlines()
            last = ""
            for line in reversed(data):
                if line.strip():
                    last = line.strip()
                    break
            if last:
                _update_from_json(panel, last)
        except Exception:
            pass
    
    # Heartbeat monitoring and Force Stop management
    try:
        import time as _time
        now = _time.time()
        
        # Force stop button removed - regular stop is now immediate and universal
        
        # Heartbeat timeout detection
        if getattr(panel, "_run_in_progress", False) and not getattr(panel, "_stop_requested", False):
            last_hb = getattr(panel, "_last_heartbeat", None)
            hb_timeout = getattr(panel, "_heartbeat_timeout", 30)
            if last_hb is not None and (now - last_hb) > hb_timeout:
                panel._log(f"[hrm] Warning: No heartbeat for {int(now - last_hb)}s (timeout={hb_timeout}s)")
                # Reset to avoid spam
                panel._last_heartbeat = now
    except Exception:
        pass
        
    try:
        panel.after(1000, panel._poll_metrics)
    except Exception:
        panel._metrics_polling_active = False


def _update_from_json(panel: Any, line: str) -> None:
    try:
        obj = json.loads(line)
    except Exception:
        return
    if not isinstance(obj, dict):
        return
    
    # Update heartbeat timestamp for any training-related event
    try:
        if obj.get("event") in ("train", "step", "gen_progress", "gen_start", "gen_done") or "step" in obj:
            import time as _time
            panel._last_heartbeat = _time.time()
    except Exception:
        pass
        
    try:
        # Track steps across all GPUs
        # For multi-GPU training, we need to aggregate steps from each GPU
        # session_steps = steps trained in current session (samples/rows trained) - aggregated across GPUs
        # total_steps = all-time cumulative steps (samples/rows trained)
        
        # Handle per-GPU step updates (for multi-GPU aggregation)
        if "gpu_session_steps" in obj:
            gpu_id = obj.get("gpu_id", 0)
            gpu_steps = obj.get("gpu_session_steps", 0)
            
            # Initialize per-GPU tracking dict if needed
            if not hasattr(panel, "_gpu_session_steps"):
                panel._gpu_session_steps = {}
            
            # Update this GPU's step count
            panel._gpu_session_steps[gpu_id] = gpu_steps
            
            # Calculate total session steps across all GPUs
            total_session_steps = sum(panel._gpu_session_steps.values())
            panel._session_steps = total_session_steps
            panel._seen_session_steps = True
            
            if hasattr(panel, "met_step"):
                panel.met_step.config(text=str(total_session_steps))
        
        # Update Training Progress display with current session steps (aggregated total from chunk_complete)
        elif "session_steps" in obj:
            panel._session_steps = obj.get("session_steps", 0)
            panel._seen_session_steps = True  # Flag that we've seen session steps
            if hasattr(panel, "met_step"):
                panel.met_step.config(text=str(panel._session_steps))
        elif "total_optimizer_steps" in obj:
            # Only use total_optimizer_steps if we haven't seen session_steps yet
            # (for backward compatibility with non-parallel training modes)
            if not getattr(panel, "_seen_session_steps", False):
                panel._total_optimizer_steps = obj.get("total_optimizer_steps", 0)
                panel._seen_total_optimizer_steps = True  # Flag that we've seen this field
                if hasattr(panel, "met_step"):
                    panel.met_step.config(text=str(panel._total_optimizer_steps))
        # Backward compatibility: handle legacy field names
        elif "session_true_steps" in obj:
            if not getattr(panel, "_seen_session_steps", False):
                panel._session_steps = obj.get("session_true_steps", 0)
                panel._seen_session_steps = True
                if hasattr(panel, "met_step"):
                    panel.met_step.config(text=str(panel._session_steps))
        elif "total_gpu_steps" in obj:
            if not getattr(panel, "_seen_session_steps", False) and not getattr(panel, "_seen_total_optimizer_steps", False):
                panel._total_optimizer_steps = obj.get("total_gpu_steps", 0)
                panel._seen_total_optimizer_steps = True
                if hasattr(panel, "met_step"):
                    panel.met_step.config(text=str(panel._total_optimizer_steps))
        elif "step" in obj and hasattr(panel, "met_step"):
            # If we've ever seen session_steps or total_optimizer_steps in this session, ignore step values
            if not getattr(panel, "_seen_total_optimizer_steps", False) and not getattr(panel, "_seen_session_steps", False):
                step_val = obj.get("step", "-")
                panel.met_step.config(text=str(step_val))
        
        # Update architecture display's Steps field with all-time total
        if "total_steps" in obj and hasattr(panel, "trained_steps_entry"):
            total_steps = obj.get("total_steps", 0)
            try:
                panel.trained_steps_entry.config(state="normal")
                panel.trained_steps_entry.delete(0, "end")
                panel.trained_steps_entry.insert(0, f"{total_steps:,}")
                panel.trained_steps_entry.config(state="readonly")
            except Exception:
                pass
        # Backward compatibility: handle legacy field name
        elif "total_true_steps" in obj and hasattr(panel, "trained_steps_entry"):
            total_steps = obj.get("total_true_steps", 0)
            try:
                panel.trained_steps_entry.config(state="normal")
                panel.trained_steps_entry.delete(0, "end")
                panel.trained_steps_entry.insert(0, f"{total_steps:,}")
                panel.trained_steps_entry.config(state="readonly")
            except Exception:
                pass
        
        if "loss" in obj and hasattr(panel, "met_loss"):
            v = obj.get("loss")
            panel.met_loss.config(text=(f"{v:.4f}" if isinstance(v, (int,float)) else str(v)))
        if "ce_token" in obj and hasattr(panel, "met_ce"):
            panel.met_ce.config(text=str(obj.get("ce_token")))
        if "ppl" in obj and hasattr(panel, "met_ppl"):
            panel.met_ppl.config(text=str(obj.get("ppl")))
        if "token_acc" in obj and hasattr(panel, "met_tok"):
            panel.met_tok.config(text=str(obj.get("token_acc")))
        if "exact_match" in obj and hasattr(panel, "met_exact"):
            panel.met_exact.config(text=str(obj.get("exact_match")))
    except Exception:
        pass
    # Epoch tracking updates
    _update_epoch_tracking(panel, obj)
    # Progress/state messages
    _update_progress(panel, obj)


def _update_epoch_tracking(panel: Any, obj: dict) -> None:
    """Update epoch tracking display from JSON log events."""
    try:
        # Initialize epoch tracking from training start
        if obj.get("epoch_tracking") == "initialized":
            panel._epoch_tracking_initialized = True
            panel._dataset_total_samples = obj.get("dataset_total_samples")
            panel._samples_per_block = obj.get("samples_per_block")
            panel._total_blocks = obj.get("total_blocks")
            panel._current_epoch = 0
            panel._samples_processed_this_epoch = 0
            panel._blocks_processed = 0
            panel._current_block_samples = 0
            panel._chunks_completed = 0
            panel._total_chunks = None
            panel._chunks_in_current_block = 0
            panel._current_block_id = 0
            
            # Extract dataset name from panel's dataset field
            try:
                dataset_path = getattr(panel, 'dataset_var', None)
                if dataset_path:
                    dataset_str = dataset_path.get().strip()
                    # Extract name from path or HF dataset
                    if 'hf://' in dataset_str:
                        panel._dataset_name = dataset_str.split('hf://')[-1].split(':')[0]
                    else:
                        import os
                        # Get basename and remove common extensions
                        name = os.path.basename(dataset_str)
                        for ext in ['.txt', '.csv', '.json', '.jsonl']:
                            name = name.replace(ext, '')
                        panel._dataset_name = name
                else:
                    panel._dataset_name = "unknown"
            except Exception:
                panel._dataset_name = "unknown"
            
            # Use chunks_per_block from backend (preferred) or calculate as fallback
            panel._chunks_per_block = obj.get("chunks_per_block")
            if panel._chunks_per_block is None:
                try:
                    chunk_var = getattr(panel, 'dataset_chunk_size_var', None)
                    if chunk_var:
                        chunk_size = int(chunk_var.get())
                        samples_per_block = int(panel._samples_per_block or 100000)
                        # ceil division to match trainer logic
                        chunks_per_block = max(1, (samples_per_block + chunk_size - 1) // chunk_size)
                        panel._chunks_per_block = chunks_per_block
                except Exception:
                    pass
            
            # Update displays
            if hasattr(panel, "epoch_dataset_lbl"):
                panel.epoch_dataset_lbl.config(text=panel._dataset_name or "unknown")
            if hasattr(panel, "epoch_number_lbl"):
                panel.epoch_number_lbl.config(text="0")
            if hasattr(panel, "epoch_blocks_lbl"):
                if panel._total_blocks:
                    panel.epoch_blocks_lbl.config(text=f"0/{panel._total_blocks}")
                else:
                    panel.epoch_blocks_lbl.config(text="0/???")
            if hasattr(panel, "epoch_chunk_lbl"):
                if panel._chunks_per_block:
                    panel.epoch_chunk_lbl.config(text=f"0/{panel._chunks_per_block}")
                else:
                    panel.epoch_chunk_lbl.config(text="0/???")
            
            # Ensure steps display is initialized
            if hasattr(panel, "met_step"):
                # Initialize with 0 if we have total_optimizer_steps, otherwise use current step
                if hasattr(panel, "_total_optimizer_steps") and panel._total_optimizer_steps > 0:
                    panel.met_step.config(text=str(panel._total_optimizer_steps))
                else:
                    panel.met_step.config(text="0")
            
            # Reset session tracking flags and per-GPU step tracking for new training session
            panel._seen_session_steps = False
            panel._seen_total_optimizer_steps = False
            panel._gpu_session_steps = {}  # Reset per-GPU step aggregation
        
        # Epoch tracking disabled - still initialize chunk/block tracking from GUI settings
        elif obj.get("epoch_tracking") == "disabled":
            # Initialize tracking variables even without full epoch tracking
            panel._epoch_tracking_initialized = False
            panel._current_block_id = 0
            panel._blocks_processed = 0
            panel._chunks_completed = 0
            panel._chunks_in_current_block = 0
            panel._total_blocks = None  # Unknown
            panel._samples_per_block = 100000  # Standard block size
            
            # Calculate chunks per block from GUI settings (100k samples / chunk size)
            try:
                chunk_var = getattr(panel, 'dataset_chunk_size_var', None)
                if chunk_var:
                    chunk_size = int(chunk_var.get())
                    # Standard block is 100k samples
                    chunks_per_block = max(1, (100000 + chunk_size - 1) // chunk_size)
                    panel._chunks_per_block = chunks_per_block
                else:
                    panel._chunks_per_block = 25  # Default: 100k / 4000
            except Exception:
                panel._chunks_per_block = 25  # Default fallback
            
            # Extract dataset name
            try:
                dataset_path = getattr(panel, 'dataset_var', None)
                if dataset_path:
                    dataset_str = dataset_path.get().strip()
                    if 'hf://' in dataset_str:
                        panel._dataset_name = dataset_str.split('hf://')[-1].split(':')[0]
                    else:
                        import os
                        name = os.path.basename(dataset_str)
                        for ext in ['.txt', '.csv', '.json', '.jsonl']:
                            name = name.replace(ext, '')
                        panel._dataset_name = name
                else:
                    panel._dataset_name = "unknown"
            except Exception:
                panel._dataset_name = "unknown"
            
            # Check if ChunkTracker state is included in this event (resume case)
            if "chunks_trained" in obj or "chunks_completed" in obj:
                # Resuming with existing chunks
                panel._chunks_completed = obj.get("chunks_trained", obj.get("chunks_completed", 0))
                panel._current_block_id = obj.get("current_block_id", 0)
                panel._blocks_processed = obj.get("blocks_completed", 0)
                
                # Calculate current position within block
                if panel._chunks_per_block:
                    panel._chunks_in_current_block = panel._chunks_completed % panel._chunks_per_block
            
            # Initialize displays
            if hasattr(panel, "epoch_dataset_lbl"):
                panel.epoch_dataset_lbl.config(text=panel._dataset_name or "unknown")
            if hasattr(panel, "epoch_chunk_lbl"):
                chunks_in_block = getattr(panel, '_chunks_in_current_block', 0)
                chunks_per_block = getattr(panel, '_chunks_per_block', 0)
                if chunks_per_block:
                    panel.epoch_chunk_lbl.config(text=f"{chunks_in_block}/{chunks_per_block}")
                else:
                    panel.epoch_chunk_lbl.config(text="0/???")
            if hasattr(panel, "epoch_blocks_lbl"):
                blocks = getattr(panel, '_current_block_id', 0)
                panel.epoch_blocks_lbl.config(text=str(blocks))  # Just show count, no total
        
        # Restore epoch tracking from checkpoint
        elif obj.get("epoch_tracking") == "restored":
            panel._epoch_tracking_initialized = True
            panel._dataset_total_samples = obj.get("dataset_total_samples")
            panel._samples_per_block = obj.get("samples_per_block")
            panel._total_blocks = obj.get("total_blocks")
            panel._current_epoch = obj.get("current_epoch", 0)
            panel._samples_processed_this_epoch = obj.get("samples_processed_this_epoch", 0)
            panel._current_block_samples = obj.get("current_block_samples", 0)
            panel._chunks_completed = obj.get("chunks_completed", 0)
            panel._total_chunks = obj.get("total_chunks")
            panel._chunks_in_current_block = 0
            panel._current_block_id = obj.get("current_block_id", 0)
            
            # Parse blocks processed
            blocks_str = obj.get("blocks_processed_this_epoch", "")
            panel._blocks_processed = len(blocks_str.split(",")) if blocks_str else 0
            
            # Extract dataset name
            try:
                dataset_path = getattr(panel, 'dataset_var', None)
                if dataset_path:
                    dataset_str = dataset_path.get().strip()
                    if 'hf://' in dataset_str:
                        panel._dataset_name = dataset_str.split('hf://')[-1].split(':')[0]
                    else:
                        import os
                        name = os.path.basename(dataset_str)
                        for ext in ['.txt', '.csv', '.json', '.jsonl']:
                            name = name.replace(ext, '')
                        panel._dataset_name = name
                else:
                    panel._dataset_name = "unknown"
            except Exception:
                panel._dataset_name = "unknown"
            
            # Use chunks_per_block from backend (preferred) or calculate as fallback
            panel._chunks_per_block = obj.get("chunks_per_block")
            if panel._chunks_per_block is None:
                try:
                    chunk_var = getattr(panel, 'dataset_chunk_size_var', None)
                    if chunk_var:
                        chunk_size = int(chunk_var.get())
                        samples_per_block = int(panel._samples_per_block or 100000)
                        # ceil division to match trainer logic
                        chunks_per_block = max(1, (samples_per_block + chunk_size - 1) // chunk_size)
                        panel._chunks_per_block = chunks_per_block
                except Exception:
                    pass
            
            # Update displays
            if hasattr(panel, "epoch_dataset_lbl"):
                panel.epoch_dataset_lbl.config(text=panel._dataset_name or "unknown")
            if hasattr(panel, "epoch_number_lbl"):
                panel.epoch_number_lbl.config(text=str(panel._current_epoch))
            if hasattr(panel, "epoch_blocks_lbl"):
                if panel._total_blocks:
                    panel.epoch_blocks_lbl.config(text=f"{panel._blocks_processed}/{panel._total_blocks}")
                else:
                    panel.epoch_blocks_lbl.config(text=f"{panel._blocks_processed}/???")
            if hasattr(panel, "epoch_chunk_lbl"):
                # Calculate current chunk within current block
                current_chunk_in_block = panel._chunks_completed % panel._chunks_per_block if panel._chunks_per_block else 0
                if panel._chunks_per_block:
                    panel.epoch_chunk_lbl.config(text=f"{current_chunk_in_block}/{panel._chunks_per_block}")
                else:
                    panel.epoch_chunk_lbl.config(text=f"{panel._chunks_completed}/???")
        
        # Update progress within epoch
        elif "epoch_progress" in obj:
            progress_data = obj.get("epoch_progress", {})
            if isinstance(progress_data, dict):
                panel._current_epoch = progress_data.get("epoch", panel._current_epoch)
                panel._samples_processed_this_epoch = progress_data.get("samples_processed", 0)
                panel._blocks_processed = progress_data.get("blocks_done", 0)
                panel._chunks_completed = progress_data.get("chunks_done", panel._chunks_completed)
                progress_pct = progress_data.get("progress_pct", 0)
                
                # Update displays
                if hasattr(panel, "epoch_number_lbl"):
                    panel.epoch_number_lbl.config(text=str(panel._current_epoch))
                if hasattr(panel, "epoch_blocks_lbl"):
                    if panel._total_blocks:
                        panel.epoch_blocks_lbl.config(text=f"{panel._blocks_processed}/{panel._total_blocks}")
                    else:
                        panel.epoch_blocks_lbl.config(text=f"{panel._blocks_processed}/???")
                if hasattr(panel, "epoch_chunk_lbl"):
                    # Show chunk position within current block
                    chunks_per_block = getattr(panel, '_chunks_per_block', None)
                    if chunks_per_block:
                        current_chunk_in_block = panel._chunks_completed % chunks_per_block
                        panel.epoch_chunk_lbl.config(text=f"{current_chunk_in_block}/{chunks_per_block}")
                    else:
                        panel.epoch_chunk_lbl.config(text=f"{panel._chunks_completed}/???")
        
        # Block completed (single-GPU and parallel events)
        elif obj.get("block_complete") is True or obj.get("event") == "block_complete":
            try:
                panel._blocks_processed = int(getattr(panel, "_blocks_processed", 0)) + 1
            except Exception:
                panel._blocks_processed = int(getattr(panel, "_blocks_processed", 0))
            
            # Reset chunks in current block when moving to next block
            panel._chunks_in_current_block = 0
            
            # Update current block ID if provided
            if "block_id" in obj:
                try:
                    # Block just completed, so next block is current+1
                    completed_block = int(obj.get("block_id"))
                    panel._current_block_id = completed_block + 1
                except Exception:
                    pass
            
            # Update blocks display
            if hasattr(panel, "epoch_blocks_lbl"):
                current_block = getattr(panel, '_current_block_id', panel._blocks_processed)
                if getattr(panel, "_total_blocks", None):
                    panel.epoch_blocks_lbl.config(text=f"{current_block}/{panel._total_blocks}")
                else:
                    panel.epoch_blocks_lbl.config(text=f"{current_block}/???")
            
            # Reset chunk display for new block
            if hasattr(panel, "epoch_chunk_lbl"):
                chunks_per_block = getattr(panel, '_chunks_per_block', None)
                if chunks_per_block:
                    panel.epoch_chunk_lbl.config(text=f"0/{chunks_per_block}")
                else:
                    panel.epoch_chunk_lbl.config(text="0/???")
        
        # Chunk tracking updates (parallel and single-GPU)
        elif obj.get("event") == "chunk_complete" or "total_chunks_trained" in obj or "chunk_complete" in obj:
            # Ensure tracking is initialized (even if epoch tracking failed)
            if not hasattr(panel, '_chunks_per_block') or panel._chunks_per_block is None:
                # Emergency initialization from GUI settings
                try:
                    chunk_var = getattr(panel, 'dataset_chunk_size_var', None)
                    if chunk_var:
                        chunk_size = int(chunk_var.get())
                        panel._chunks_per_block = max(1, (100000 + chunk_size - 1) // chunk_size)
                    else:
                        panel._chunks_per_block = 25
                except Exception:
                    panel._chunks_per_block = 25
                
                # Initialize other tracking vars if missing
                if not hasattr(panel, '_chunks_completed'):
                    panel._chunks_completed = 0
                if not hasattr(panel, '_current_block_id'):
                    panel._current_block_id = 0
                if not hasattr(panel, '_chunks_in_current_block'):
                    panel._chunks_in_current_block = 0
            
            # Update current block ID from event
            if "block_id" in obj:
                try:
                    panel._current_block_id = int(obj.get("block_id"))
                except Exception:
                    pass
            
            # Update global chunk counter
            if "total_chunks_trained" in obj:
                panel._chunks_completed = obj.get("total_chunks_trained", getattr(panel, "_chunks_completed", 0))
            else:
                # Increment by one for a chunk completion event
                try:
                    panel._chunks_completed = int(getattr(panel, "_chunks_completed", 0)) + 1
                except Exception:
                    panel._chunks_completed = int(getattr(panel, "_chunks_completed", 0))
            panel._total_chunks = obj.get("total_chunks", getattr(panel, "_total_chunks", None))
            
            # Track chunks trained in current block
            if "chunk_id" in obj:
                try:
                    # chunk_id is 0-indexed within the block, so +1 gives us "chunks trained"
                    panel._chunks_in_current_block = int(obj.get("chunk_id")) + 1
                except Exception:
                    pass

            # Update chunk display - show chunks trained in current block / total per block
            if hasattr(panel, "epoch_chunk_lbl"):
                chunks_per_block = getattr(panel, '_chunks_per_block', None)
                chunks_in_block = getattr(panel, '_chunks_in_current_block', 0)
                
                if chunks_per_block and chunks_in_block > 0:
                    # Show: "25/100" = 25 chunks trained out of 100 total in this block
                    panel.epoch_chunk_lbl.config(text=f"{chunks_in_block}/{chunks_per_block}")
                elif chunks_per_block:
                    # Fallback to modulo calculation if chunk_id not available
                    # Add 1 because we just completed a chunk
                    current_chunk_in_block = (panel._chunks_completed % chunks_per_block)
                    if current_chunk_in_block == 0 and panel._chunks_completed > 0:
                        current_chunk_in_block = chunks_per_block
                    panel.epoch_chunk_lbl.config(text=f"{current_chunk_in_block}/{chunks_per_block}")
                else:
                    # Show total chunks if no per-block info available
                    panel.epoch_chunk_lbl.config(text=f"{panel._chunks_completed}/???")
            
            # Update block display - show completed blocks / total blocks
            if hasattr(panel, "epoch_blocks_lbl"):
                # Use blocks_completed from event if available, otherwise infer from current_block_id
                blocks_completed = obj.get("blocks_completed", getattr(panel, '_blocks_processed', 0))
                
                # Update total_blocks from event if provided (for lazy-detected datasets)
                if "total_blocks" in obj and obj["total_blocks"] is not None:
                    panel._total_blocks = obj["total_blocks"]
                
                total_blocks = getattr(panel, '_total_blocks', None)
                if total_blocks:
                    panel.epoch_blocks_lbl.config(text=f"{blocks_completed}/{total_blocks}")
                else:
                    # Just show completed blocks without total
                    panel.epoch_blocks_lbl.config(text=str(blocks_completed))
            
            # Update steps display with session_steps
            # Update Training Progress with session steps, architecture display with total steps
            if "session_steps" in obj and hasattr(panel, "met_step"):
                panel._session_steps = obj.get("session_steps", getattr(panel, "_session_steps", 0))
                panel._seen_session_steps = True
                panel.met_step.config(text=str(panel._session_steps))
            elif "total_optimizer_steps" in obj and hasattr(panel, "met_step"):
                if not getattr(panel, "_seen_session_steps", False):
                    panel._total_optimizer_steps = obj.get("total_optimizer_steps", getattr(panel, "_total_optimizer_steps", 0))
                    panel.met_step.config(text=str(panel._total_optimizer_steps))
            # Backward compat: handle legacy field names
            elif "session_true_steps" in obj and hasattr(panel, "met_step"):
                if not getattr(panel, "_seen_session_steps", False):
                    panel._session_steps = obj.get("session_true_steps", 0)
                    panel._seen_session_steps = True
                    panel.met_step.config(text=str(panel._session_steps))
            elif "total_gpu_steps" in obj and hasattr(panel, "met_step"):
                if not getattr(panel, "_seen_session_steps", False) and not getattr(panel, "_seen_total_optimizer_steps", False):
                    panel._total_optimizer_steps = obj.get("total_gpu_steps", 0)
                    panel.met_step.config(text=str(panel._total_optimizer_steps))
            
            # Update architecture display with all-time total
            if "total_steps" in obj and hasattr(panel, "trained_steps_entry"):
                total_steps = obj.get("total_steps", 0)
                try:
                    panel.trained_steps_entry.config(state="normal")
                    panel.trained_steps_entry.delete(0, "end")
                    panel.trained_steps_entry.insert(0, f"{total_steps:,}")
                    panel.trained_steps_entry.config(state="readonly")
                except Exception:
                    pass
            # Backward compat: handle legacy field name
            elif "total_true_steps" in obj and hasattr(panel, "trained_steps_entry"):
                total_steps = obj.get("total_true_steps", 0)
                try:
                    panel.trained_steps_entry.config(state="normal")
                    panel.trained_steps_entry.delete(0, "end")
                    panel.trained_steps_entry.insert(0, f"{total_steps:,}")
                    panel.trained_steps_entry.config(state="readonly")
                except Exception:
                    pass

        # Cycle complete (iterate mode) - may contain chunk/block progress
        elif obj.get("event") == "cycle_complete":
            # Ensure tracking is initialized
            if not hasattr(panel, '_chunks_per_block') or panel._chunks_per_block is None:
                try:
                    chunk_var = getattr(panel, 'dataset_chunk_size_var', None)
                    if chunk_var:
                        chunk_size = int(chunk_var.get())
                        panel._chunks_per_block = max(1, (100000 + chunk_size - 1) // chunk_size)
                    else:
                        panel._chunks_per_block = 25
                except Exception:
                    panel._chunks_per_block = 25
            
            if not hasattr(panel, '_chunks_completed'):
                panel._chunks_completed = 0
            if not hasattr(panel, '_current_block_id'):
                panel._current_block_id = 0
            if not hasattr(panel, '_samples_in_current_block'):
                panel._samples_in_current_block = 0
            
            # Update steps if available (prioritize session_steps)
            if "session_steps" in obj and hasattr(panel, "met_step"):
                panel._session_steps = obj.get("session_steps", getattr(panel, "_session_steps", 0))
                panel._seen_session_steps = True
                panel.met_step.config(text=str(panel._session_steps))
            elif "total_optimizer_steps" in obj and hasattr(panel, "met_step"):
                if not getattr(panel, "_seen_session_steps", False):
                    panel._total_optimizer_steps = obj.get("total_optimizer_steps", getattr(panel, "_total_optimizer_steps", 0))
                    panel.met_step.config(text=str(panel._total_optimizer_steps))
            # Backward compat
            elif "session_true_steps" in obj and hasattr(panel, "met_step"):
                if not getattr(panel, "_seen_session_steps", False):
                    panel._session_steps = obj.get("session_true_steps", 0)
                    panel._seen_session_steps = True
                    panel.met_step.config(text=str(panel._session_steps))
            elif "total_steps" in obj and hasattr(panel, "met_step"):
                if not getattr(panel, "_seen_session_steps", False):
                    panel._total_steps = obj.get("total_steps", getattr(panel, "_total_steps", 0))
                    panel.met_step.config(text=str(panel._total_steps))
            
            # Update chunk and block progress from cycle data
            chunk_samples = obj.get("chunk_samples", 0)
            samples_in_block = obj.get("block_progress", 0)
            samples_per_block = obj.get("block_size", 100000)
            
            if chunk_samples > 0:
                # Calculate chunks completed from chunk_samples
                try:
                    chunk_var = getattr(panel, 'dataset_chunk_size_var', None)
                    if chunk_var:
                        chunk_size = int(chunk_var.get())
                        # How many chunks fit in chunk_samples?
                        chunks_this_cycle = max(1, (chunk_samples + chunk_size - 1) // chunk_size)
                        panel._chunks_completed = getattr(panel, '_chunks_completed', 0) + chunks_this_cycle
                except Exception:
                    pass
            
            # Track samples in current block
            panel._samples_in_current_block = samples_in_block
            
            # Calculate current block from samples processed
            if samples_per_block > 0 and samples_in_block >= samples_per_block:
                # Moved to next block
                panel._current_block_id = getattr(panel, '_current_block_id', 0) + 1
                panel._samples_in_current_block = 0
            
            # Update chunk display - show chunks within current block
            if hasattr(panel, "epoch_chunk_lbl"):
                chunks_per_block = getattr(panel, '_chunks_per_block', None)
                if chunks_per_block and samples_in_block > 0:
                    try:
                        chunk_var = getattr(panel, 'dataset_chunk_size_var', None)
                        if chunk_var:
                            chunk_size = int(chunk_var.get())
                            # Chunks completed within current block
                            chunks_in_block = (samples_in_block // chunk_size)
                            panel.epoch_chunk_lbl.config(text=f"{chunks_in_block}/{chunks_per_block}")
                    except Exception:
                        pass
            
            # Update block display
            if hasattr(panel, "epoch_blocks_lbl"):
                current_block = getattr(panel, '_current_block_id', 0)
                panel.epoch_blocks_lbl.config(text=str(current_block))
        
        # Blocks detected (total blocks determined for the first time)
        elif obj.get("event") == "blocks_detected":
            total_blocks = obj.get("total_blocks")
            if total_blocks is not None:
                panel._total_blocks = total_blocks
                logger.info(f"Detected {total_blocks} total blocks in dataset")
                
                # Update block display with new total
                if hasattr(panel, "epoch_blocks_lbl"):
                    current_block = obj.get("current_block", getattr(panel, '_current_block_id', 0))
                    panel.epoch_blocks_lbl.config(text=f"{current_block}/{total_blocks}")
        
        # Chunk claimed (optional immediate UI update)
        elif obj.get("event") == "chunk_claimed":
            if hasattr(panel, "epoch_chunk_lbl"):
                chunks_per_block = getattr(panel, '_chunks_per_block', None)
                try:
                    c_id = int(obj.get("chunk_id")) if obj.get("chunk_id") is not None else None
                except Exception:
                    c_id = None
                if chunks_per_block and c_id is not None:
                    panel.epoch_chunk_lbl.config(text=f"{c_id}/{chunks_per_block}")
                elif c_id is not None:
                    panel.epoch_chunk_lbl.config(text=f"{c_id}/???")
        
        # Progress stats from ChunkTracker
        elif "progress_stats" in obj:
            stats = obj.get("progress_stats", {})
            if isinstance(stats, dict):
                # Update Training Progress display with session steps
                if "session_steps" in stats:
                    panel._session_steps = stats.get("session_steps", getattr(panel, "_session_steps", 0))
                    panel._seen_session_steps = True
                elif "total_optimizer_steps" in stats:
                    if not getattr(panel, "_seen_session_steps", False):
                        panel._total_optimizer_steps = stats.get("total_optimizer_steps", getattr(panel, "_total_optimizer_steps", 0))
                # Backward compat
                elif "session_true_steps" in stats:
                    if not getattr(panel, "_seen_session_steps", False):
                        panel._session_steps = stats.get("session_true_steps", 0)
                elif "total_gpu_steps" in stats:
                    if not getattr(panel, "_seen_session_steps", False):
                        panel._total_optimizer_steps = stats.get("total_gpu_steps", 0)
                
                panel._chunks_completed = stats.get("total_chunks_trained", panel._chunks_completed)
                panel._blocks_processed = stats.get("blocks_completed", panel._blocks_processed)
                panel._current_epoch = stats.get("current_epoch", panel._current_epoch)
                panel._current_block_id = stats.get("current_block_id", getattr(panel, "_current_block_id", 0))
                
                # Update Training Progress display
                if hasattr(panel, "met_step"):
                    session_val = getattr(panel, "_session_steps", None) or getattr(panel, "_total_optimizer_steps", 0)
                    panel.met_step.config(text=str(session_val))
                
                # Update architecture display with all-time total
                if "total_steps" in stats and hasattr(panel, "trained_steps_entry"):
                    total_steps = stats.get("total_steps", 0)
                    try:
                        panel.trained_steps_entry.config(state="normal")
                        panel.trained_steps_entry.delete(0, "end")
                        panel.trained_steps_entry.insert(0, f"{total_steps:,}")
                        panel.trained_steps_entry.config(state="readonly")
                    except Exception:
                        pass
                # Backward compat
                elif "total_true_steps" in stats and hasattr(panel, "trained_steps_entry"):
                    total_steps = stats.get("total_true_steps", 0)
                    try:
                        panel.trained_steps_entry.config(state="normal")
                        panel.trained_steps_entry.delete(0, "end")
                        panel.trained_steps_entry.insert(0, f"{total_steps:,}")
                        panel.trained_steps_entry.config(state="readonly")
                    except Exception:
                        pass
                
                if hasattr(panel, "epoch_number_lbl"):
                    panel.epoch_number_lbl.config(text=str(panel._current_epoch))
                if hasattr(panel, "epoch_blocks_lbl"):
                    current_block = getattr(panel, '_current_block_id', panel._blocks_processed)
                    if panel._total_blocks:
                        panel.epoch_blocks_lbl.config(text=f"{current_block}/{panel._total_blocks}")
                    else:
                        panel.epoch_blocks_lbl.config(text=f"{current_block}/???")
                if hasattr(panel, "epoch_chunk_lbl"):
                    chunks_per_block = getattr(panel, '_chunks_per_block', None)
                    chunks_in_block = getattr(panel, '_chunks_in_current_block', 0)
                    if chunks_per_block and chunks_in_block > 0:
                        panel.epoch_chunk_lbl.config(text=f"{chunks_in_block}/{chunks_per_block}")
                    elif chunks_per_block:
                        current_chunk_in_block = panel._chunks_completed % chunks_per_block
                        if current_chunk_in_block == 0 and panel._chunks_completed > 0:
                            current_chunk_in_block = chunks_per_block
                        panel.epoch_chunk_lbl.config(text=f"{current_chunk_in_block}/{chunks_per_block}")
                    else:
                        panel.epoch_chunk_lbl.config(text=f"{panel._chunks_completed}/???")
        
        # Epoch completed
        elif obj.get("epoch_complete") is True:
            panel._current_epoch = obj.get("epoch_number", panel._current_epoch + 1)
            panel._samples_processed_this_epoch = 0
            panel._blocks_processed = 0
            panel._chunks_completed = 0
            panel._chunks_in_current_block = 0
            panel._current_block_id = 0
            
            # Update displays
            if hasattr(panel, "epoch_number_lbl"):
                panel.epoch_number_lbl.config(text=str(panel._current_epoch))
            if hasattr(panel, "epoch_blocks_lbl") and panel._total_blocks:
                panel.epoch_blocks_lbl.config(text=f"0/{panel._total_blocks}")
            if hasattr(panel, "epoch_chunk_lbl"):
                chunks_per_block = getattr(panel, '_chunks_per_block', None)
                if chunks_per_block:
                    panel.epoch_chunk_lbl.config(text=f"0/{chunks_per_block}")
                else:
                    panel.epoch_chunk_lbl.config(text="0/???")
    except Exception:
        pass


def _update_progress(panel: Any, obj: dict) -> None:
    try:
        evt = obj.get("event")
    except Exception:
        return
    if evt == "gen_start":
        try:
            tot = obj.get("total")
            panel._last_gen_total = int(tot) if tot is not None else None
        except Exception:
            panel._last_gen_total = None
        try:
            panel._gen_hist = []
            panel.progress_lbl.config(text="generating…")
            panel.progress.configure(mode="indeterminate", value=0)
            panel.progress.start(10)
        except Exception:
            pass
    elif evt == "gen_auto_batch":
        try:
            gb = obj.get("gen_batch")
            if gb is not None:
                panel.progress_lbl.config(text=f"tuning gen batch (bs={int(gb)})…")
            panel.progress.configure(mode="indeterminate", value=0)
            panel.progress.start(10)
        except Exception:
            pass
    elif evt == "gen_progress":
        try:
            gen_raw = obj.get("generated", 0)
            gen_i = int(gen_raw if gen_raw is not None else 0)
            tot_raw = obj.get("total")
            if tot_raw is None:
                tot_raw = getattr(panel, "_last_gen_total", 0)
            tot_i = int(tot_raw if tot_raw is not None else 0)
            if tot_i > 0:
                import time as _time
                now = _time.time()
                try:
                    panel._gen_hist.append((gen_i, now))
                    if len(panel._gen_hist) > 10:
                        panel._gen_hist = panel._gen_hist[-10:]
                    eta_txt = ""
                    if len(panel._gen_hist) >= 2:
                        s0, t0 = panel._gen_hist[0]
                        s1, t1 = panel._gen_hist[-1]
                        ds = max(1, s1 - s0)
                        dt = max(1e-3, t1 - t0)
                        rate = ds / dt
                        rem = max(0, tot_i - gen_i)
                        eta_sec = int(rem / max(1e-6, rate))
                        mm, ss = divmod(eta_sec, 60)
                        eta_txt = f" ETA {mm:02d}:{ss:02d}"
                except Exception:
                    eta_txt = ""
                try:
                    panel.progress.stop()
                    panel.progress.configure(mode="determinate")
                except Exception:
                    pass
                panel.progress.configure(value=int(max(0, min(100, round((gen_i / max(1, tot_i)) * 100)))))
                panel.progress_lbl.config(text=f"gen {gen_i}/{tot_i} ({panel.progress['value']}%){eta_txt}")
        except Exception:
            pass
    elif evt == "gen_done":
        try:
            panel.progress.stop()
        except Exception:
            pass
        panel.progress.configure(mode="determinate", value=100)
        panel.progress_lbl.config(text="generation done")
    elif evt == "iterate_cycle":
        try:
            cyc = obj.get("cycle")
            if cyc is not None:
                panel.progress_lbl.config(text=f"cycle {int(cyc)}…")
        except Exception:
            panel.progress_lbl.config(text="cycling…")
    elif evt in ("train", "step"):
        # Progress bar updates handled by _parse_and_update_progress in process_manager.py
        pass
    elif evt == "graceful_stopped":
        # Handle graceful stop completion
        try:
            panel.progress.stop()
        except Exception:
            pass
        panel.progress.configure(mode="determinate", value=0)
        panel.progress_lbl.config(text="stopped (gracefully)")
        # Reset stop button appearance
        try:
            panel.stop_btn.config(text="Stop", style="TButton")
        except Exception:
            pass
        panel._graceful_stop_requested = False
        if not getattr(panel, "_stopped_dialog_shown", False):
            panel._stopped_dialog_shown = True
            try:
                panel._show_stopped_dialog()
            except Exception:
                pass
    elif evt in ("stopped", "stop"):
        try:
            panel.progress.stop()
        except Exception:
            pass
        panel.progress.configure(mode="determinate", value=0)
        panel.progress_lbl.config(text="stopped")
        # Reset stop button appearance
        try:
            panel.stop_btn.config(text="Stop", style="TButton")
        except Exception:
            pass
        panel._graceful_stop_requested = False
        if not getattr(panel, "_stopped_dialog_shown", False):
            panel._stopped_dialog_shown = True
            try:
                panel._show_stopped_dialog()
            except Exception:
                pass


def show_stopped_dialog(panel: Any) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
        top = tk.Toplevel(panel)
        top.title("Training Stopped")
        top.grab_set()
        msg = ttk.Label(top, text="Training was stopped. What would you like to do?", wraplength=380, justify="left")
        msg.pack(padx=12, pady=(10, 8), anchor="w")
        btns = ttk.Frame(top)
        btns.pack(fill="x", padx=12, pady=(0, 12))
        def _open_dir():
            try:
                s = panel.student_init_var.get().strip()
                d = os.path.dirname(s) if s else os.path.dirname(panel.log_file_var.get().strip())
                if d and os.path.isdir(d):
                    if os.name == "nt":
                        try:
                            os.startfile(d)  # type: ignore[attr-defined]
                        except Exception:
                            import subprocess
                            subprocess.Popen(["explorer", d])
                    else:
                        import subprocess
                        subprocess.Popen(["xdg-open", d], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        def _open_log():
            try:
                lf = panel.log_file_var.get().strip()
                if lf and os.path.exists(lf):
                    if os.name == "nt":
                        try:
                            os.startfile(lf)  # type: ignore[attr-defined]
                        except Exception:
                            import subprocess
                            subprocess.Popen(["notepad", lf])
                    else:
                        import subprocess
                        subprocess.Popen(["xdg-open", lf], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        def _clear_stop():
            try:
                sf = os.path.join(panel._project_root, "training_datasets", "actv1", "STOP")
                if sf and os.path.exists(sf):
                    os.remove(sf)
                    panel._log(f"[hrm] Cleared stop file: {sf}")
            except Exception:
                pass
        ttk.Button(btns, text="Open bundle folder", command=_open_dir).pack(side="left")
        ttk.Button(btns, text="Open metrics log", command=_open_log).pack(side="left", padx=(8,0))
        ttk.Button(btns, text="Clear STOP file", command=_clear_stop).pack(side="left", padx=(8,0))
        ttk.Button(btns, text="Dismiss", command=top.destroy).pack(side="right")
    except Exception:
        pass
