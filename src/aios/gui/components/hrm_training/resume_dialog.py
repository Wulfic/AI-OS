"""Resume Training Dialog for HRM Training Panel.

Prompts user to resume from checkpoint when one exists.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime
import logging
import sys

from aios.gui.utils.theme_utils import apply_theme_to_toplevel, get_spacing_multiplier

logger = logging.getLogger(__name__)


class ResumeDialog(tk.Toplevel):
    """Dialog to ask user if they want to resume from checkpoint."""
    
    def __init__(self, parent, brain_name: str, save_dir: str, dataset_file: str, has_checkpoint: bool = False, parent_panel=None):
        super().__init__(parent)
        
        self.result: Optional[bool] = None  # None=cancelled, True=resume, False=fresh
        self.resume_info: Optional[Dict[str, Any]] = None
        self.has_checkpoint = has_checkpoint
        self.parent_panel = parent_panel
        
        # Phase 6.4: Start position controls
        self.start_block_id = tk.IntVar(value=0)
        self.start_chunk_id = tk.IntVar(value=0)
        
        # Configure window
        if has_checkpoint:
            self.title("Resume Training?")
        else:
            self.title("Start Training")
        width, height = 1200, 800
        if sys.platform.startswith("win"):
            width //= 2
            height = (height // 2) + 100
        self.geometry(f"{width}x{height}")
        self.minsize(width, height)
        self.resizable(False, False)
        
        # Make modal
        self.transient(parent)
        
        # Apply theme to this dialog
        apply_theme_to_toplevel(self)
        
        # Center on parent
        try:
            self.update_idletasks()
            parent_x = parent.winfo_x()
            parent_y = parent.winfo_y()
            parent_width = parent.winfo_width()
            parent_height = parent.winfo_height()
            
            x = parent_x + (parent_width // 2) - (self.winfo_reqwidth() // 2)
            y = parent_y + (parent_height // 2) - (self.winfo_reqheight() // 2)
            self.geometry(f"+{x}+{y}")
        except Exception as e:
            logger.debug(f"Could not center dialog: {e}")
            # Fallback to center of screen
            try:
                self.update_idletasks()
                screen_width = self.winfo_screenwidth()
                screen_height = self.winfo_screenheight()
                x = (screen_width // 2) - (self.winfo_reqwidth() // 2)
                y = (screen_height // 2) - (self.winfo_reqheight() // 2)
                self.geometry(f"+{x}+{y}")
            except Exception:
                pass
        
        # Load checkpoint info
        self._load_checkpoint_info(brain_name, save_dir, dataset_file)
        
        # Build UI
        self._build_ui()
        
        # Ensure dialog is visible and on top AFTER building UI
        self.update_idletasks()
        self.deiconify()
        self.lift()
        self.focus_force()
        
        # CRITICAL: Set grab AFTER window is fully visible
        try:
            self.wait_visibility()
            self.grab_set()
        except Exception as e:
            logger.debug(f"Could not grab focus: {e}")
            try:
                self.grab_set()
            except Exception:
                pass
        
    def _load_checkpoint_info(self, brain_name: str, save_dir: str, dataset_file: str):
        """Load checkpoint metadata from brain.json or chunk_tracker_state.json."""
        try:
            brain_path = Path(save_dir) / brain_name
            brain_json_path = brain_path / "brain.json"
            chunk_tracker_path = brain_path / "chunk_tracker_state.json"
            
            # Check if checkpoint file exists
            checkpoint_path = brain_path / "actv1_student.safetensors"
            legacy_checkpoint_path = brain_path / "final_model.safetensors"
            
            # Use new checkpoint if exists, otherwise fall back to legacy
            if not checkpoint_path.exists() and legacy_checkpoint_path.exists():
                checkpoint_path = legacy_checkpoint_path
            
            # Store current dataset for later reference
            self.current_dataset_file = dataset_file
            
            # If no checkpoint exists, skip to creating default resume_info
            if not checkpoint_path.exists():
                checkpoint_path = None
            
            # Try to load from brain.json first (legacy/complete metadata)
            last_session = None
            if checkpoint_path and brain_json_path.exists():
                try:
                    with brain_json_path.open("r", encoding="utf-8") as f:
                        brain_data = json.load(f)
                    last_session = brain_data.get("last_session")
                    if last_session and isinstance(last_session, dict):
                        # Have complete metadata from brain.json
                        saved_dataset = last_session.get("dataset_file")
                        # If saved_dataset is None/unknown (old checkpoints), assume match (benefit of doubt)
                        # Otherwise, validate that datasets match
                        dataset_matches = (not saved_dataset) or (str(saved_dataset) == str(dataset_file))
                        checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
                        
                        timestamp = last_session.get("timestamp", 0)
                        if timestamp:
                            dt = datetime.fromtimestamp(timestamp)
                            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                        else:
                            time_str = "Unknown"
                        
                        self.resume_info = {
                            "checkpoint_path": str(checkpoint_path),
                            "checkpoint_size_mb": checkpoint_size_mb,
                            "total_steps": last_session.get("total_steps", 0),
                            "steps_completed": last_session.get("steps_completed", 0),
                            "stopped_early": last_session.get("stopped_early", False),
                            "saved_dataset": saved_dataset,
                            "current_dataset": dataset_file,
                            "dataset_matches": dataset_matches,
                            "timestamp": time_str,
                            "config": last_session.get("config", {}),
                        }
                        return
                except Exception as e:
                    logger.debug(f"Error reading brain.json: {e}")
            
            # Fall back to chunk_tracker_state.json (from parallel training)
            if checkpoint_path and chunk_tracker_path.exists():
                try:
                    with chunk_tracker_path.open("r", encoding="utf-8") as f:
                        tracker_state = json.load(f)
                    
                    completed_chunks = tracker_state.get("completed_chunks", [])
                    total_samples = tracker_state.get("total_samples_trained", 0)
                    current_epoch = tracker_state.get("current_epoch", 0)
                    total_blocks_in_dataset = tracker_state.get("total_blocks_in_dataset")
                    saved_dataset_name = tracker_state.get("dataset_name")
                    
                    # Calculate total steps from completed chunks
                    total_steps = sum(chunk.get("step", 0) for chunk in completed_chunks)
                    
                    # Get checkpoint size and timestamp
                    checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
                    checkpoint_mtime = checkpoint_path.stat().st_mtime
                    dt = datetime.fromtimestamp(checkpoint_mtime)
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Calculate chunk size from samples_trained (use first chunk if available)
                    chunk_size = None
                    if completed_chunks:
                        first_chunk_samples = completed_chunks[0].get("samples_trained", 0)
                        if first_chunk_samples > 0:
                            chunk_size = first_chunk_samples
                    
                    # Calculate total chunks per block (standard block is 100k samples)
                    total_chunks_per_block = None
                    if chunk_size and chunk_size > 0:
                        samples_per_block = 100000  # Standard block size
                        total_chunks_per_block = max(1, (samples_per_block + chunk_size - 1) // chunk_size)
                    
                    self.resume_info = {
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_size_mb": checkpoint_size_mb,
                        "total_steps": total_steps,
                        "steps_completed": total_steps,
                        "stopped_early": False,
                        "saved_dataset": dataset_file,  # Assume current dataset
                        "saved_dataset_name": saved_dataset_name,
                        "current_dataset": dataset_file,
                        "dataset_matches": True,
                        "timestamp": time_str,
                        "config": {},
                        "chunks_trained": len(completed_chunks),
                        "samples_trained": total_samples,
                        "epoch": current_epoch,
                        "total_blocks": total_blocks_in_dataset,
                        "chunk_size": chunk_size,
                        "chunks_per_block": total_chunks_per_block,
                    }
                    return
                except Exception as e:
                    logger.debug(f"Error reading chunk_tracker_state.json: {e}")
            
            # No valid metadata found - create default empty info
            # Calculate total blocks and chunks from current settings
            chunk_size = self._get_chunk_size_from_panel()
            total_blocks = self._estimate_total_blocks(dataset_file)
            chunks_per_block = None
            if chunk_size and chunk_size > 0:
                samples_per_block = 100000  # Standard block size
                chunks_per_block = max(1, (samples_per_block + chunk_size - 1) // chunk_size)
            
            self.resume_info = {
                "checkpoint_path": "N/A",
                "checkpoint_size_mb": 0.0,
                "total_steps": 0,
                "steps_completed": 0,
                "stopped_early": False,
                "saved_dataset": dataset_file,
                "saved_dataset_name": Path(dataset_file).name if dataset_file else "unknown",
                "current_dataset": dataset_file,
                "dataset_matches": True,
                "timestamp": "Never",
                "config": {},
                "chunks_trained": 0,
                "samples_trained": 0,
                "epoch": 0,
                "total_blocks": total_blocks,
                "chunk_size": chunk_size,
                "chunks_per_block": chunks_per_block,
            }
            
        except Exception as e:
            logger.error(f"Error loading checkpoint info: {e}", exc_info=True)
            # Create default empty info on error
            # Calculate total blocks and chunks from current settings
            chunk_size = self._get_chunk_size_from_panel()
            total_blocks = self._estimate_total_blocks(dataset_file)
            chunks_per_block = None
            if chunk_size and chunk_size > 0:
                samples_per_block = 100000  # Standard block size
                chunks_per_block = max(1, (samples_per_block + chunk_size - 1) // chunk_size)
            
            self.resume_info = {
                "checkpoint_path": "N/A",
                "checkpoint_size_mb": 0.0,
                "total_steps": 0,
                "steps_completed": 0,
                "stopped_early": False,
                "saved_dataset": dataset_file,
                "saved_dataset_name": Path(dataset_file).name if dataset_file else "unknown",
                "current_dataset": dataset_file,
                "dataset_matches": True,
                "timestamp": "Never",
                "config": {},
                "chunks_trained": 0,
                "samples_trained": 0,
                "epoch": 0,
                "total_blocks": total_blocks,
                "chunk_size": chunk_size,
                "chunks_per_block": chunks_per_block,
            }
    
    def _get_chunk_size_from_panel(self) -> Optional[int]:
        """Get chunk size from parent panel."""
        try:
            if self.parent_panel and hasattr(self.parent_panel, 'dataset_chunk_size_var'):
                chunk_size = int(self.parent_panel.dataset_chunk_size_var.get())
                return chunk_size if chunk_size > 0 else 4000
        except Exception:
            pass
        return 4000  # Default
    
    def _estimate_total_blocks(self, dataset_file: str) -> Optional[int]:
        """Get total blocks for the dataset from cache or chunk tracker."""
        # Try to get from chunk_tracker_state.json even if no checkpoint exists
        try:
            from pathlib import Path
            brain_path = Path(self.parent_panel.brain_bundle_dir) if hasattr(self.parent_panel, 'brain_bundle_dir') else None
            if not brain_path and hasattr(self.parent_panel, 'bundle_dir_var'):
                brain_path = Path(self.parent_panel.bundle_dir_var.get()) / self.parent_panel.brain_name_var.get()
            
            if brain_path:
                chunk_tracker_path = brain_path / "chunk_tracker_state.json"
                if chunk_tracker_path.exists():
                    import json
                    with chunk_tracker_path.open("r", encoding="utf-8") as f:
                        tracker_state = json.load(f)
                    total_blocks = tracker_state.get("total_blocks_in_dataset")
                    if total_blocks is not None:
                        logger.debug(f"Found total_blocks from chunk_tracker: {total_blocks}")
                        return total_blocks
        except Exception as e:
            logger.debug(f"Could not read total_blocks from chunk_tracker: {e}")
        
        # Try to get from streaming cache stats (if blocks have been cached)
        try:
            from ...data.streaming_cache import get_cache
            cache = get_cache()
            stats = cache.get_cache_stats()
            
            # Check if we have this dataset in cache
            if dataset_file:
                from pathlib import Path
                dataset_name = Path(dataset_file).stem if not dataset_file.startswith("hf://") else dataset_file
                
                # Look in chunks_per_dataset
                chunks_per_dataset = stats.get('chunks_per_dataset', {})
                for cached_dataset, chunk_count in chunks_per_dataset.items():
                    # Match dataset name
                    if dataset_name in cached_dataset or cached_dataset in str(dataset_file):
                        logger.debug(f"Found {chunk_count} cached blocks for dataset")
                        return chunk_count
        except Exception as e:
            logger.debug(f"Could not read from streaming cache: {e}")
        
        # Return None if we can't determine - let it be calculated during training
        return None
    
    def _build_ui(self):
        """Build the dialog UI."""
        # Get spacing multiplier for current theme
        spacing = get_spacing_multiplier()
        
        # Main container
        padding = int(20 * spacing)
        main_frame = ttk.Frame(self, padding=padding)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, int(15 * spacing)))
        
        if self.has_checkpoint:
            ttk.Label(
                header_frame,
                text="🔄 Checkpoint Found",
                font=("TkDefaultFont", 14, "bold")
            ).pack()
            
            ttk.Label(
                header_frame,
                text="Would you like to resume training from where you left off?",
                font=("TkDefaultFont", 10)
            ).pack(pady=(int(5 * spacing), 0))
        else:
            ttk.Label(
                header_frame,
                text="🚀 No Checkpoint Found",
                font=("TkDefaultFont", 14, "bold")
            ).pack()
            
            ttk.Label(
                header_frame,
                text="Start fresh training for this brain and dataset",
                font=("TkDefaultFont", 10)
            ).pack(pady=(int(5 * spacing), 0))
        
        # Container for side-by-side info sections
        padding_15 = int(15 * spacing)
        info_container = ttk.Frame(main_frame)
        info_container.pack(fill=tk.BOTH, expand=True, pady=(0, padding_15))
        
        # Configure grid to give equal space to both columns
        info_container.grid_columnconfigure(0, weight=1, uniform="info")
        info_container.grid_columnconfigure(1, weight=1, uniform="info")
        info_container.grid_rowconfigure(0, weight=1)
        
        # Checkpoint Information (left side)
        info_frame = ttk.LabelFrame(info_container, text="Checkpoint Information", padding=padding_15)
        info_frame.grid(row=0, column=0, sticky="nsew", padx=(0, int(5 * spacing)))
        
        # Create info grid
        info_labels = []
        
        # Add chunk/sample info if available (from parallel training)
        if "chunks_trained" in self.resume_info:
            info_labels.extend([
                ("Chunks Trained:", f"{self.resume_info['chunks_trained']} chunks"),
                ("Samples Trained:", f"{self.resume_info['samples_trained']:,} samples"),
                ("Training Steps:", f"{self.resume_info['total_steps']:,} steps"),
                ("Current Epoch:", f"{self.resume_info.get('epoch', 0)}"),
            ])
        else:
            # Legacy format (from brain.json)
            info_labels.extend([
                ("Training Steps:", f"{self.resume_info['total_steps']:,} steps"),
                ("Last Session:", f"{self.resume_info['steps_completed']} steps"),
                ("Status:", "Stopped Early" if self.resume_info['stopped_early'] else "Completed"),
            ])
        
        # Always show timestamp and size
        info_labels.extend([
            ("Last Trained:", self.resume_info['timestamp']),
            ("Checkpoint Size:", f"{self.resume_info['checkpoint_size_mb']:.2f} MB"),
        ])
        
        for i, (label_text, value_text) in enumerate(info_labels):
            ttk.Label(
                info_frame,
                text=label_text,
                font=("TkDefaultFont", 9, "bold")
            ).grid(row=i, column=0, sticky=tk.W, padx=(0, int(10 * spacing)), pady=int(3 * spacing))
            
            ttk.Label(
                info_frame,
                text=value_text,
                font=("TkDefaultFont", 9)
            ).grid(row=i, column=1, sticky=tk.W, pady=int(3 * spacing))
        
        # Dataset Information Section (right side)
        dataset_info_frame = ttk.LabelFrame(info_container, text="Dataset Information", padding=padding_15)
        dataset_info_frame.grid(row=0, column=1, sticky="nsew", padx=(int(5 * spacing), 0))
        
        # Dataset name
        dataset_name_frame = ttk.Frame(dataset_info_frame)
        dataset_name_frame.pack(fill=tk.X, pady=int(5 * spacing))
        
        ttk.Label(
            dataset_name_frame,
            text="Dataset:",
            font=("TkDefaultFont", 9, "bold"),
            width=15
        ).pack(side=tk.LEFT, padx=(0, int(10 * spacing)))
        
        dataset_display_name = self.resume_info.get('saved_dataset_name') or Path(self.resume_info['current_dataset']).name
        ttk.Label(
            dataset_name_frame,
            text=dataset_display_name,
            font=("TkDefaultFont", 9)
        ).pack(side=tk.LEFT)
        
        # Total blocks
        blocks_frame = ttk.Frame(dataset_info_frame)
        blocks_frame.pack(fill=tk.X, pady=int(5 * spacing))
        
        ttk.Label(
            blocks_frame,
            text="Total Blocks:",
            font=("TkDefaultFont", 9, "bold"),
            width=15
        ).pack(side=tk.LEFT, padx=(0, int(10 * spacing)))
        
        total_blocks = self.resume_info.get('total_blocks')
        blocks_text = f"{total_blocks} blocks" if total_blocks is not None else "Unknown (will be determined)"
        ttk.Label(
            blocks_frame,
            text=blocks_text,
            font=("TkDefaultFont", 9)
        ).pack(side=tk.LEFT)
        
        # Total chunks per block
        if self.resume_info.get('chunks_per_block'):
            chunks_frame = ttk.Frame(dataset_info_frame)
            chunks_frame.pack(fill=tk.X, pady=int(5 * spacing))
            
            ttk.Label(
                chunks_frame,
                text="Chunks/Block:",
                font=("TkDefaultFont", 9, "bold"),
                width=15
            ).pack(side=tk.LEFT, padx=(0, int(10 * spacing)))
            
            chunk_size_text = f"{self.resume_info['chunks_per_block']} chunks"
            if self.resume_info.get('chunk_size'):
                chunk_size_text += f" ({self.resume_info['chunk_size']:,} samples/chunk)"
            
            ttk.Label(
                chunks_frame,
                text=chunk_size_text,
                font=("TkDefaultFont", 9)
            ).pack(side=tk.LEFT)
        
        # Chunk size warning note
        note_frame = ttk.Frame(dataset_info_frame)
        note_frame.pack(fill=tk.X, pady=(int(10 * spacing), 0))
        
        ttk.Label(
            note_frame,
            text="ℹ️  Note: Changing chunk size requires starting a new training run",
            font=("TkDefaultFont", 8),
            foreground="blue"
        ).pack(anchor=tk.W)
        
        ttk.Label(
            note_frame,
            text="(New chunk sizes must be computed from the beginning)",
            font=("TkDefaultFont", 7),
            foreground="gray"
        ).pack(anchor=tk.W, padx=(int(15 * spacing), 0))
        
        # Dataset validation
        dataset_frame = ttk.Frame(info_frame)
        dataset_frame.grid(row=len(info_labels), column=0, columnspan=2, sticky=tk.W + tk.E, pady=(int(10 * spacing), 0))
        
        saved_dataset = self.resume_info['saved_dataset']
        dataset_was_unknown = not saved_dataset  # Old checkpoint without dataset tracking
        
        if self.resume_info['dataset_matches']:
            if dataset_was_unknown:
                # Old checkpoint - show info message instead of green checkmark
                ttk.Label(
                    dataset_frame,
                    text="ℹ️  Dataset info not saved in checkpoint (resuming anyway)",
                    font=("TkDefaultFont", 9),
                    foreground="blue"
                ).pack(anchor=tk.W)
                ttk.Label(
                    dataset_frame,
                    text=f"Current: {Path(self.resume_info['current_dataset']).name}",
                    font=("TkDefaultFont", 8),
                    foreground="gray"
                ).pack(anchor=tk.W, padx=(int(10 * spacing), 0))
            else:
                # Dataset matches - all good
                ttk.Label(
                    dataset_frame,
                    text="✅ Dataset matches checkpoint",
                    font=("TkDefaultFont", 9),
                    foreground="green"
                ).pack(anchor=tk.W)
        else:
            # Dataset actually differs
            ttk.Label(
                dataset_frame,
                text="⚠️  Dataset differs from checkpoint",
                font=("TkDefaultFont", 9, "bold"),
                foreground="orange"
            ).pack(anchor=tk.W)
            
            ttk.Label(
                dataset_frame,
                text=f"Saved: {Path(saved_dataset).name}",
                font=("TkDefaultFont", 8),
                foreground="gray"
            ).pack(anchor=tk.W, padx=(int(10 * spacing), 0))
            
            ttk.Label(
                dataset_frame,
                text=f"Current: {Path(self.resume_info['current_dataset']).name}",
                font=("TkDefaultFont", 8),
                foreground="gray"
            ).pack(anchor=tk.W, padx=(int(10 * spacing), 0))
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, padding_15))
        
        padx_val = (0, int(10 * spacing))
        
        # Resume button (disabled if no checkpoint)
        resume_btn = ttk.Button(
            button_frame,
            text="🔄 Resume Training",
            command=self._resume,
            width=20,
            state=tk.NORMAL if self.has_checkpoint else tk.DISABLED
        )
        resume_btn.pack(side=tk.LEFT, padx=padx_val)
        
        # Start fresh button
        fresh_btn = ttk.Button(
            button_frame,
            text="🆕 Start Fresh",
            command=self._start_fresh,
            width=20
        )
        fresh_btn.pack(side=tk.LEFT, padx=padx_val)
        
        # Manual Start button
        manual_btn = ttk.Button(
            button_frame,
            text="⚙️ Manual Start",
            command=self._manual_start,
            width=20
        )
        manual_btn.pack(side=tk.LEFT, padx=padx_val)
        
        # Cancel button
        cancel_btn = ttk.Button(
            button_frame,
            text="Cancel",
            command=self._cancel,
            width=15
        )
        cancel_btn.pack(side=tk.RIGHT)
        
        # Manual Start Position Section
        position_frame = ttk.LabelFrame(main_frame, text="Manual Start Position", padding=padding_15)
        position_frame.pack(fill=tk.X, pady=(0, padding_15))
        
        # Input container for side-by-side layout
        inputs_frame = ttk.Frame(position_frame)
        inputs_frame.pack(fill=tk.X)
        
        # Block ID input
        ttk.Label(
            inputs_frame,
            text="Start Block:",
            font=("TkDefaultFont", 9)
        ).pack(side=tk.LEFT, padx=(0, int(5 * spacing)))
        
        ttk.Entry(
            inputs_frame,
            textvariable=self.start_block_id,
            width=10
        ).pack(side=tk.LEFT, padx=(0, int(15 * spacing)))
        
        # Chunk ID input
        ttk.Label(
            inputs_frame,
            text="Start Chunk:",
            font=("TkDefaultFont", 9)
        ).pack(side=tk.LEFT, padx=(0, int(5 * spacing)))
        
        ttk.Entry(
            inputs_frame,
            textvariable=self.start_chunk_id,
            width=10
        ).pack(side=tk.LEFT)
        
        # Warning if dataset mismatch
        if not self.resume_info['dataset_matches']:
            warning_frame = ttk.Frame(main_frame)
            warning_frame.pack(fill=tk.X, pady=(15, 0))
            
            ttk.Label(
                warning_frame,
                text="⚠️  Warning: Dataset mismatch may lead to suboptimal training results.",
                font=("TkDefaultFont", 8),
                foreground="orange"
            ).pack()
            
            ttk.Label(
                warning_frame,
                text="Consider starting fresh if you changed datasets intentionally.",
                font=("TkDefaultFont", 8),
                foreground="gray"
            ).pack()
    
    def _resume(self):
        """User chose to resume training."""
        # Return tuple: (resume=True, start_block_id=0, start_chunk_id=0)
        self.result = (True, 0, 0)
        self.destroy()
    
    def _start_fresh(self):
        """User chose to start fresh training."""
        # Return tuple: (resume=False, start_block_id=0, start_chunk_id=0)
        self.result = (False, 0, 0)
        self.destroy()
    
    def _manual_start(self):
        """User chose manual start with specific position."""
        # Validate inputs
        block_id = self.start_block_id.get()
        chunk_id = self.start_chunk_id.get()
        
        # Get max values
        max_blocks = self.resume_info.get('total_blocks')
        max_chunks = self.resume_info.get('chunks_per_block')
        
        # Validate block_id
        if block_id < 0 or (max_blocks is not None and block_id >= max_blocks):
            logger.warning(f"Invalid block_id {block_id}, resetting to 0 (max: {max_blocks})")
            block_id = 0
            self.start_block_id.set(0)
        
        # Validate chunk_id
        if chunk_id < 0 or (max_chunks is not None and chunk_id >= max_chunks):
            logger.warning(f"Invalid chunk_id {chunk_id}, resetting to 0 (max: {max_chunks})")
            chunk_id = 0
            self.start_chunk_id.set(0)
        
        # Return tuple: (resume=False, start_block_id, start_chunk_id)
        self.result = (False, block_id, chunk_id)
        self.destroy()
    
    def _cancel(self):
        """User cancelled."""
        self.result = None
        self.destroy()


def show_resume_dialog(
    parent, 
    brain_name: str, 
    save_dir: str, 
    dataset_file: str,
    has_checkpoint: bool = False,
    parent_panel = None
) -> Optional[tuple[bool, int, int]]:
    """Show resume/start dialog and return user's choice with start position.
    
    Phase 6.4: Training Resume Start Position Selector
    
    Args:
        parent: Parent window
        brain_name: Name of the brain being trained
        save_dir: Directory where brains are saved
        dataset_file: Current dataset file path
        has_checkpoint: Whether a resumable checkpoint exists
        parent_panel: Parent panel to get training configuration from
    
    Returns:
        Tuple of (resume_choice, start_block_id, start_chunk_id) where:
        - resume_choice: True if resume, False if start fresh
        - start_block_id: Starting block ID (0 = beginning)
        - start_chunk_id: Starting chunk ID (0 = beginning)
        Returns None if cancelled
    """
    try:
        logger.debug(f"Creating dialog for brain: {brain_name}, has_checkpoint={has_checkpoint}")
        dialog = ResumeDialog(parent, brain_name, save_dir, dataset_file, has_checkpoint, parent_panel)
        
        logger.debug("Waiting for dialog to close...")
        # Wait for dialog to close
        parent.wait_window(dialog)
        
        logger.debug(f"Dialog closed with result: {dialog.result}")
        return dialog.result
    except Exception as e:
        logger.error(f"Error showing dialog: {e}", exc_info=True)
        return None
