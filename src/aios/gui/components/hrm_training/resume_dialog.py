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

from aios.gui.utils.theme_utils import apply_theme_to_toplevel, get_spacing_multiplier

logger = logging.getLogger(__name__)


class ResumeDialog(tk.Toplevel):
    """Dialog to ask user if they want to resume from checkpoint."""
    
    def __init__(self, parent, brain_name: str, save_dir: str, dataset_file: str):
        super().__init__(parent)
        
        self.result: Optional[bool] = None  # None=cancelled, True=resume, False=fresh
        self.resume_info: Optional[Dict[str, Any]] = None
        
        # Configure window
        self.title("Resume Training?")
        # Dialog defaults were previously 600x400; double both dimensions for readability.
        self.geometry("1200x800")
        self.minsize(1200, 800)
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
            
            if not checkpoint_path.exists():
                self.resume_info = None
                return
            
            # Try to load from brain.json first (legacy/complete metadata)
            last_session = None
            if brain_json_path.exists():
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
            if chunk_tracker_path.exists():
                try:
                    with chunk_tracker_path.open("r", encoding="utf-8") as f:
                        tracker_state = json.load(f)
                    
                    completed_chunks = tracker_state.get("completed_chunks", [])
                    total_samples = tracker_state.get("total_samples_trained", 0)
                    current_epoch = tracker_state.get("current_epoch", 0)
                    
                    # Calculate total steps from completed chunks
                    total_steps = sum(chunk.get("step", 0) for chunk in completed_chunks)
                    
                    # Get checkpoint size and timestamp
                    checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
                    checkpoint_mtime = checkpoint_path.stat().st_mtime
                    dt = datetime.fromtimestamp(checkpoint_mtime)
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    
                    self.resume_info = {
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_size_mb": checkpoint_size_mb,
                        "total_steps": total_steps,
                        "steps_completed": total_steps,
                        "stopped_early": False,
                        "saved_dataset": dataset_file,  # Assume current dataset
                        "current_dataset": dataset_file,
                        "dataset_matches": True,
                        "timestamp": time_str,
                        "config": {},
                        "chunks_trained": len(completed_chunks),
                        "samples_trained": total_samples,
                        "epoch": current_epoch,
                    }
                    return
                except Exception as e:
                    logger.debug(f"Error reading chunk_tracker_state.json: {e}")
            
            # No valid metadata found
            self.resume_info = None
            
        except Exception as e:
            logger.error(f"Error loading checkpoint info: {e}", exc_info=True)
            self.resume_info = None
    
    def _build_ui(self):
        """Build the dialog UI."""
        # Get spacing multiplier for current theme
        spacing = get_spacing_multiplier()
        
        # Main container
        padding = int(20 * spacing)
        main_frame = ttk.Frame(self, padding=padding)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        if not self.resume_info:
            # No checkpoint found - shouldn't happen, but handle gracefully
            ttk.Label(
                main_frame,
                text="No checkpoint found.",
                font=("TkDefaultFont", 11)
            ).pack(pady=int(20 * spacing))
            
            ttk.Button(
                main_frame,
                text="Start Fresh Training",
                command=self._start_fresh,
                width=25
            ).pack(pady=int(10 * spacing))
            return
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, int(15 * spacing)))
        
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
        
        # Info frame
        padding_15 = int(15 * spacing)
        info_frame = ttk.LabelFrame(main_frame, text="Checkpoint Information", padding=padding_15)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=(0, padding_15))
        
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
        button_frame.pack(fill=tk.X, pady=(int(10 * spacing), 0))
        
        padx_val = (0, int(10 * spacing))
        # Resume button (primary action)
        resume_btn = ttk.Button(
            button_frame,
            text="🔄 Resume Training",
            command=self._resume,
            width=20
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
        
        # Cancel button
        cancel_btn = ttk.Button(
            button_frame,
            text="Cancel",
            command=self._cancel,
            width=15
        )
        cancel_btn.pack(side=tk.RIGHT)
        
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
        self.result = True
        self.destroy()
    
    def _start_fresh(self):
        """User chose to start fresh training."""
        self.result = False
        self.destroy()
    
    def _cancel(self):
        """User cancelled."""
        self.result = None
        self.destroy()


def show_resume_dialog(parent, brain_name: str, save_dir: str, dataset_file: str) -> Optional[bool]:
    """Show resume dialog and return user's choice.
    
    Args:
        parent: Parent window
        brain_name: Name of the brain being trained
        save_dir: Directory where brains are saved
        dataset_file: Current dataset file path
    
    Returns:
        True if resume, False if start fresh, None if cancelled
    """
    try:
        logger.debug(f"Creating dialog for brain: {brain_name}")
        dialog = ResumeDialog(parent, brain_name, save_dir, dataset_file)
        
        logger.debug("Waiting for dialog to close...")
        # Wait for dialog to close
        parent.wait_window(dialog)
        
        logger.debug(f"Dialog closed with result: {dialog.result}")
        return dialog.result
    except Exception as e:
        logger.error(f"Error showing dialog: {e}", exc_info=True)
        return None
