"""Helper utility functions for HRM Training Panel.

Provides logging, state management, and utility functions that require panel access.
"""

from __future__ import annotations
import os
from typing import TYPE_CHECKING, Any

from ...utils.resource_management import submit_background

# Import safe variable wrappers
from ...utils import safe_variables

if TYPE_CHECKING:
    from .panel_main import HRMTrainingPanel


def log(panel: HRMTrainingPanel, msg: str) -> None:
    """Append a line of text to the panel log and external output.

    This method is safe to call from background threads; UI updates are
    marshalled onto the Tk main loop using `after(0, ...)`.
    
    Args:
        panel: The HRMTrainingPanel instance
        msg: Message to log
    """
    # Forward to the shared output sink (Debug tab) if provided
    try:
        if callable(getattr(panel, "_append_out", None)):
            panel._append_out(msg)
    except Exception:
        pass

    line = (msg if isinstance(msg, str) else str(msg))
    if not line.endswith("\n"):
        line += "\n"

    def _ui_append() -> None:
        try:
            # Only scroll to bottom if user is already viewing the bottom
            try:
                yview = panel.log.yview()
                at_bottom = yview[1] >= 0.95  # Within ~5% of bottom
            except Exception:
                at_bottom = True  # Default to scrolling if can't check
            
            panel.log.insert("end", line)
            
            if at_bottom:
                panel.log.see("end")
        except Exception:
            pass

    try:
        # Schedule on the Tk event loop to avoid cross-thread UI access
        panel.after(0, _ui_append)
    except Exception:
        # Best-effort direct append if `after` is unavailable
        _ui_append()


def schedule_save(panel: HRMTrainingPanel, delay_ms: int = 400) -> None:
    """Schedule a debounced save of panel state.
    
    Args:
        panel: The HRMTrainingPanel instance
        delay_ms: Delay in milliseconds before saving
    """
    try:
        if not callable(panel._save_state_fn):
            return
        if panel._save_after_id is not None:
            try:
                panel.after_cancel(panel._save_after_id)
            except Exception:
                pass
        def _trigger_save() -> None:
            panel._save_after_id = None
            try:
                if callable(panel._save_state_fn):
                    panel._save_state_fn()
            except Exception:
                pass

        panel._save_after_id = panel.after(delay_ms, _trigger_save)
    except Exception:
        pass


def mk_bool(val: bool) -> Any:
    """Create a BooleanVar or fallback mock for headless environments.
    
    Args:
        val: Initial boolean value
        
    Returns:
        tk.BooleanVar or mock object
    """
    try:
        import tkinter as tk
        return safe_variables.BooleanVar(value=val)
    except Exception:  # pragma: no cover - headless
        class _B:
            def __init__(self, v): 
                self._v = bool(v)
            def get(self): 
                return self._v
            def set(self, v): 
                self._v = bool(v)
        return _B(val)


def detect_and_display_dataset_info(panel: HRMTrainingPanel) -> Optional[Dict[str, Any]]:
    """Detect dataset size and update display with rows, size, and blocks.
    
    Args:
        panel: HRMTrainingPanel instance
    
    Returns:
        Dictionary with dataset metadata or None if detection fails
    """
    try:
        dataset_path = panel.dataset_var.get().strip()
        if not dataset_path:
            return None
        
        # Parse HuggingFace dataset format: hf://dataset_name:config:split
        if dataset_path.startswith("hf://"):
            hf_path = dataset_path[5:]  # Remove 'hf://' prefix
            parts = hf_path.split(":")
            
            dataset_name = parts[0]
            config = parts[1] if len(parts) > 1 and parts[1] != "default" else None
            split = parts[2] if len(parts) > 2 else "train"
            
            # Import size detection module
            try:
                from ..dataset_download_panel.hf_size_detection import get_hf_dataset_metadata
                
                log(panel, f"🔍 Detecting size for HuggingFace dataset: {dataset_name}")
                if config:
                    log(panel, f"   Config: {config}, Split: {split}")
                
                metadata = get_hf_dataset_metadata(dataset_name, config, split)
                
                if metadata:
                    num_rows = metadata.get('num_rows', 0)
                    size_gb = metadata.get('size_gb', 0)
                    size_mb = metadata.get('size_mb', 0)
                    total_blocks = metadata.get('total_blocks', 0)
                    is_estimated = metadata.get('num_rows_estimated', False)
                    source = metadata.get('source', 'unknown')
                    
                    # Format size display
                    if size_gb >= 1.0:
                        size_str = f"{size_gb:.2f} GB"
                    else:
                        size_str = f"{size_mb:.1f} MB"
                    
                    est_tag = " (estimated)" if is_estimated else ""
                    
                    log(panel, f"✅ Dataset info detected:")
                    log(panel, f"   Rows: {num_rows:,}{est_tag}")
                    log(panel, f"   Size: {size_str}")
                    log(panel, f"   Blocks (100k samples): {total_blocks}")
                    log(panel, f"   Source: {source}")
                    
                    # Update epoch tracking display if available
                    if hasattr(panel, 'epoch_blocks_lbl') and total_blocks > 0:
                        panel.epoch_blocks_lbl.config(text=f"0/{total_blocks}")
                        log(panel, f"📊 Epoch tracking updated: 0/{total_blocks} blocks")
                    
                    # Store for training config
                    panel._detected_dataset_info = metadata
                    
                    return metadata
                else:
                    log(panel, "⚠️ Could not detect dataset size (may need download to determine)")
                    return None
                    
            except ImportError:
                log(panel, "⚠️ Dataset size detection not available (module not found)")
                return None
            except Exception as e:
                log(panel, f"⚠️ Error detecting dataset size: {e}")
                return None
        else:
            # Local dataset - could integrate local detection here
            log(panel, "ℹ️ Local dataset - size detection available during training")
            return None
            
    except Exception as e:
        log(panel, f"⚠️ Error in dataset detection: {e}")
        return None


def auto_calculate_steps(panel: HRMTrainingPanel) -> None:
    """Auto-calculate optimal steps based on dataset size and batch size.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    if panel._auto_steps_calculating:
        return
    
    try:
        panel._auto_steps_calculating = True
        
        # Disable button during calculation
        if hasattr(panel, '_auto_steps_btn'):
            panel._auto_steps_btn.config(state="disabled", text="...")
        
        # Get dataset chunk size
        try:
            dataset_chunk_size = int(panel.dataset_chunk_size_var.get())
            if dataset_chunk_size < 1:
                dataset_chunk_size = 4000
        except:
            dataset_chunk_size = 4000
        
        # Simply set steps = chunk_size
        def _update_ui():
            panel.steps_var.set(str(dataset_chunk_size))
            log(panel, f"[Auto Steps] Set steps to chunk size: {dataset_chunk_size:,}")
            log(panel, f"[Auto Steps] Training will process {dataset_chunk_size} examples per cycle")
            log(panel, f"[Auto Steps] Note: Datasets are streamed in 100k blocks, broken into {dataset_chunk_size}-sample chunks")
            
            # Re-enable button
            if hasattr(panel, '_auto_steps_btn'):
                panel._auto_steps_btn.config(state="normal", text="Auto")
            
            panel._auto_steps_calculating = False
        
        panel.after(0, _update_ui)
        return
        
        # Old dataset counting logic (kept as fallback, never reached)
        def _count_and_update():
            try:
                from pathlib import Path
                import os
                
                dataset_count = 0
                
                # Handle HuggingFace dataset paths (hf://dataset_name:split:subset format)
                if dataset_path.startswith("hf://"):
                    log(panel, f"[Auto Steps] Detecting HuggingFace dataset: {dataset_path}")
                    try:
                        # Parse hf://PleIAs/common_corpus:default:train format
                        hf_path = dataset_path[5:]  # Remove "hf://" prefix
                        parts = hf_path.split(":")
                        
                        if len(parts) >= 3:
                            dataset_name = parts[0]
                            config_name = parts[1] if parts[1] != "default" else None
                            split_name = parts[2]
                        elif len(parts) == 2:
                            dataset_name = parts[0]
                            config_name = None
                            split_name = parts[1]
                        else:
                            dataset_name = hf_path
                            config_name = None
                            split_name = "train"
                        
                        log(panel, f"[Auto Steps] Loading HF dataset: {dataset_name}, config={config_name}, split={split_name}")
                        
                        # Try to load dataset info without downloading full data
                        try:
                            from datasets import load_dataset_builder
                            builder = load_dataset_builder(dataset_name, config_name)
                            if builder.info.splits:
                                split_info = builder.info.splits.get(split_name)
                                if split_info:
                                    dataset_count = split_info.num_examples
                                    log(panel, f"[Auto Steps] Found {dataset_count:,} examples in HF dataset (from metadata)")
                                else:
                                    raise ValueError(f"Split '{split_name}' not found")
                            else:
                                raise ValueError("No split information available")
                        except Exception as e1:
                            # Fallback: try streaming mode to avoid downloading full dataset
                            log(panel, f"[Auto Steps] Metadata unavailable ({e1}), trying streaming mode...")
                            from datasets import load_dataset
                            dataset = load_dataset(dataset_name, config_name, split=split_name, streaming=True)
                            # Handle both sized and iterable datasets
                            if hasattr(dataset, '__len__'):
                                dataset_count = len(dataset)  # type: ignore[arg-type]
                                log(panel, f"[Auto Steps] Loaded {dataset_count:,} examples from HF dataset")
                            else:
                                # For IterableDataset, we can't get length without downloading
                                log(panel, f"[Auto Steps] Error: Cannot determine size for streaming dataset without downloading")
                                log(panel, f"[Auto Steps] Solution: Download dataset first via Datasets tab, or manually set steps")
                                log(panel, f"[Auto Steps] Tip: For large datasets, use estimated steps based on training time")
                                return
                    
                    except ImportError:
                        log(panel, "[Auto Steps] Error: 'datasets' library not installed. Install with: pip install datasets")
                        return
                    except Exception as e:
                        log(panel, f"[Auto Steps] Error loading HuggingFace dataset: {e}")
                        return
                
                else:
                    # Handle local file/directory paths
                    path = Path(dataset_path)
                    
                    # Check if path exists
                    if not path.exists():
                        log(panel, f"[Auto Steps] Error: Dataset path does not exist: {dataset_path}")
                        log(panel, f"[Auto Steps] Tip: For HuggingFace datasets, use format: hf://dataset_name:config:split")
                        return
                    
                    # Count samples based on path type
                    if path.is_file():
                        # Single file: count lines
                        try:
                            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                                dataset_count = sum(1 for line in f if line.strip())
                        except Exception as e:
                            log(panel, f"[Auto Steps] Error reading file: {e}")
                            return
                    elif path.is_dir():
                        # Directory: count lines in all .txt files
                        txt_files = list(path.rglob('*.txt'))
                        if not txt_files:
                            log(panel, f"[Auto Steps] Warning: No .txt files found in {dataset_path}")
                            return
                        
                        for txt_file in txt_files:
                            try:
                                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                                    dataset_count += sum(1 for line in f if line.strip())
                            except Exception:
                                continue
                    else:
                        log(panel, f"[Auto Steps] Error: Invalid dataset path type")
                        return
                
                if dataset_count == 0:
                    log(panel, f"[Auto Steps] Warning: Dataset appears to be empty")
                    return
                
                # Get dataset chunk size
                try:
                    dataset_chunk_size = int(panel.dataset_chunk_size_var.get())
                    if dataset_chunk_size < 1:
                        dataset_chunk_size = 4000
                except:
                    dataset_chunk_size = 4000
                
                # Check if iterate mode is enabled
                iterate_enabled = panel.iterate_var.get() if hasattr(panel, 'iterate_var') else False
                
                if iterate_enabled:
                    # In iterate mode: steps = chunk_size (1 step per example)
                    optimal_steps = dataset_chunk_size
                    chunks_needed = max(1, (dataset_count + dataset_chunk_size - 1) // dataset_chunk_size)
                    
                    # Update UI on main thread
                    def _update_ui():
                        panel.steps_var.set(str(optimal_steps))
                        log(panel, f"[Auto Steps] Set to chunk size: {optimal_steps} steps (1 step per example)")
                        log(panel, f"[Auto Steps] Dataset: {dataset_count:,} samples → {chunks_needed} chunks")
                        log(panel, f"[Auto Steps] Total training: {chunks_needed} cycles × {optimal_steps} steps = {chunks_needed * optimal_steps:,} total steps")
                else:
                    # Non-iterate mode: calculate steps for full dataset (1 step per example)
                    optimal_steps = dataset_count
                    
                    # Update UI on main thread
                    def _update_ui():
                        panel.steps_var.set(str(optimal_steps))
                        log(panel, f"[Auto Steps] Set to dataset size: {optimal_steps:,} steps (1 step per example)")
                        log(panel, f"[Auto Steps] Coverage: 100% of dataset")
                    
                    # Re-enable button
                    if hasattr(panel, '_auto_steps_btn'):
                        panel._auto_steps_btn.config(state="normal", text="Auto")
                    
                    panel._auto_steps_calculating = False
                
                panel.after(0, _update_ui)
                
            except Exception as e:
                def _show_error():
                    log(panel, f"[Auto Steps] Error: {e}")
                    if hasattr(panel, '_auto_steps_btn'):
                        panel._auto_steps_btn.config(state="normal", text="Auto")
                    panel._auto_steps_calculating = False
                panel.after(0, _show_error)
        
        try:
            submit_background(
                "hrm-auto-steps",
                _count_and_update,
                pool=getattr(panel, "_worker_pool", None),
            )
        except RuntimeError as exc:
            log(panel, f"[Auto Steps] Queue error: {exc}")
            if hasattr(panel, '_auto_steps_btn'):
                panel._auto_steps_btn.config(state="normal", text="Auto")
            panel._auto_steps_calculating = False
    
    except Exception as e:
        log(panel, f"[Auto Steps] Error: {e}")
        if hasattr(panel, '_auto_steps_btn'):
            panel._auto_steps_btn.config(state="normal", text="Auto")
        panel._auto_steps_calculating = False


def clear_output(panel: HRMTrainingPanel) -> None:
    """Clear the training output log.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    try:
        panel.log.delete("1.0", "end")
        log(panel, "[hrm] Output cleared")
    except Exception:
        pass


def apply_preset(panel: HRMTrainingPanel, name: str) -> None:
    """Apply a rough architecture preset by name.

    Presets are approximate and meant for convenience; users can still edit
    the fields afterwards if needed.
    
    NOTE: All presets now use MoE by default (8 experts, 2 active).
    Hidden sizes are adjusted to account for MoE overhead while hitting
    target parameter counts: 1M, 5M, 10M, 20M, 50M.
    
    Args:
        panel: The HRMTrainingPanel instance
        name: Preset name (1M, 5M, 10M, 20M, 50M)
    """
    p = (name or "").strip().upper()
    # Defaults with MoE enabled
    spec = {
        "h_layers": "2",
        "l_layers": "2",
        "hidden": "512",
        "expansion": "2.0",
        "heads": "8",
        "h_cycles": "2",
        "l_cycles": "2",
        "pos": "rope",
    }
    # Adjusted for MoE (8 experts, ~1.6x param multiplier)
    # Target: actual params with MoE = desired total
    if p == "1M":
        spec.update({"hidden": "128", "heads": "4"})  # ~1M with MoE
    elif p == "5M":
        spec.update({"hidden": "256", "heads": "8"})  # ~5M with MoE
    elif p == "10M":
        spec.update({"hidden": "384", "heads": "8"})  # ~10M with MoE
    elif p == "20M":
        spec.update({"hidden": "512", "heads": "8"})  # ~20M with MoE
    elif p == "50M":
        spec.update({"hidden": "768", "heads": "12"})  # ~50M with MoE

    try:
        panel.h_layers_var.set(spec["h_layers"])  # type: ignore[index]
        panel.l_layers_var.set(spec["l_layers"])  # type: ignore[index]
        panel.hidden_size_var.set(spec["hidden"])  # type: ignore[index]
        panel.expansion_var.set(spec["expansion"])  # type: ignore[index]
        panel.num_heads_var.set(spec["heads"])  # type: ignore[index]
        panel.h_cycles_var.set(spec["h_cycles"])  # type: ignore[index]
        panel.l_cycles_var.set(spec["l_cycles"])  # type: ignore[index]
        panel.pos_enc_var.set(spec["pos"])  # type: ignore[index]
        log(panel, f"[hrm] Applied preset: {p}")
    except Exception:
        pass


def default_stop_file(panel: HRMTrainingPanel) -> str:
    """Get default stop file path.
    
    Args:
        panel: The HRMTrainingPanel instance
        
    Returns:
        str: Default stop file path
    """
    try:
        return os.path.join(panel._project_root, "training_datasets", "actv1", "STOP")
    except Exception:
        return "training_datasets/actv1/STOP"


def get_moe_num_experts(panel: HRMTrainingPanel) -> int:
    """Get MoE num_experts from readonly display field, with fallback to default.
    
    Args:
        panel: The HRMTrainingPanel instance
        
    Returns:
        int: Number of experts (default: 8)
    """
    try:
        val = panel.moe_num_experts_entry.get().strip()
        if val and val != "N/A" and val != "-":
            return int(val)
    except Exception:
        pass
    return 8  # Default


def get_moe_active_experts(panel: HRMTrainingPanel) -> int:
    """Get MoE active experts per token from readonly display field, with fallback to default.
    
    Args:
        panel: The HRMTrainingPanel instance
        
    Returns:
        int: Active experts per token (default: 2)
    """
    try:
        val = panel.moe_active_experts_entry.get().strip()
        if val and val != "N/A" and val != "-":
            return int(val)
    except Exception:
        pass
    return 2  # Default


def project_root() -> str:
    """Try to detect the project root (directory containing pyproject.toml).
    
    First tries to find from source file location (__file__), then falls back to CWD.
    This ensures the GUI works correctly even when launched from a different directory.
    
    Returns:
        str: Project root directory path
    """
    try:
        # First, try to find from source file location (more reliable for installed/launched GUI)
        source_dir = os.path.dirname(os.path.abspath(__file__))
        cur = source_dir
        for _ in range(10):  # src/aios/gui/components/hrm_training_panel -> 5 levels up + buffer
            if os.path.exists(os.path.join(cur, "pyproject.toml")):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        
        # Fallback: try from CWD
        cur = os.path.abspath(os.getcwd())
        for _ in range(8):
            if os.path.exists(os.path.join(cur, "pyproject.toml")):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        
        return os.path.abspath(os.getcwd())
    except Exception:
        return os.path.abspath(os.getcwd())
