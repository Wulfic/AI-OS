"""Signal handling and checkpoint saving utilities."""
from __future__ import annotations

import logging
import signal
import sys
from typing import TYPE_CHECKING, Optional, Any, Callable
from pathlib import Path

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aios.core.hrm_training.training_config import TrainingConfig


class CheckpointSaver:
    """Handles checkpoint saving and signal interruption."""
    
    def __init__(
        self,
        model: Any,
        save_dir: str,
        config: "TrainingConfig",
        print_fn: Callable,
    ):
        self.model = model
        self.save_dir = Path(save_dir)
        self.config = config
        self.print_fn = print_fn
        self.current_step = 0
        self.current_cycle = 0
        self.interrupted = False
        self._original_sigint = None
        self._original_sigterm = None
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        self.print_fn({"checkpoint_saver": "signal_handlers_installed"})
        
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals by saving checkpoint."""
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        self.print_fn({
            "checkpoint_saver": "signal_received",
            "signal": signal_name,
            "current_step": self.current_step,
            "current_cycle": self.current_cycle,
        })
        
        if not self.interrupted:
            self.interrupted = True
            self.print_fn({"checkpoint_saver": "saving_interrupt_checkpoint"})
            self.save_checkpoint(
                reason="interrupt",
                step=self.current_step,
                cycle=self.current_cycle,
            )
            self.print_fn({"checkpoint_saver": "checkpoint_saved", "exiting": True})
        
        # Restore original handler and re-raise
        if signum == signal.SIGINT and self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)
        elif signum == signal.SIGTERM and self._original_sigterm:
            signal.signal(signal.SIGTERM, self._original_sigterm)
        
        # Exit gracefully
        sys.exit(0)
    
    def update_progress(self, step: int, cycle: int = 0):
        """Update current training progress."""
        self.current_step = step
        self.current_cycle = cycle
        
    def save_checkpoint(
        self,
        reason: str = "periodic",
        step: Optional[int] = None,
        cycle: Optional[int] = None,
    ) -> bool:
        """Save a checkpoint.
        
        Args:
            reason: Why checkpoint is being saved (periodic, chunk_complete, interrupt, etc.)
            step: Current training step (uses self.current_step if None)
            cycle: Current training cycle (uses self.current_cycle if None)
            
        Returns:
            True if save successful, False otherwise
        """
        step = step if step is not None else self.current_step
        cycle = cycle if cycle is not None else self.current_cycle
        
        try:
            import torch
            from safetensors.torch import save_file as save_safetensors
            
            self.save_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = self.save_dir / "actv1_student.safetensors"
            tmp_path = self.save_dir / "actv1_student.safetensors.tmp"
            
            logger.info(
                f"Saving checkpoint: reason={reason}, step={step}, "
                f"cycle={cycle}, path={checkpoint_path}"
            )
            self.print_fn({
                "checkpoint_save": "starting",
                "reason": reason,
                "step": step,
                "cycle": cycle,
                "path": str(checkpoint_path),
            })
            
            # Get model state dict (handle DeepSpeed wrapper)
            if hasattr(self.model, 'module'):
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            
            # Log state dict info
            num_tensors = len(state_dict)
            total_params = sum(p.numel() for p in state_dict.values())
            logger.debug(f"State dict: {num_tensors} tensors, {total_params:,} parameters")
            
            # Save to temporary file first
            logger.debug(f"Writing to temporary file: {tmp_path}")
            save_safetensors(state_dict, str(tmp_path))
            
            # Log file size
            tmp_size_mb = tmp_path.stat().st_size / (1024 * 1024)
            logger.info(f"Checkpoint file size: {tmp_size_mb:.1f} MB")
            
            # Backup existing checkpoint if it exists
            if checkpoint_path.exists():
                backup_path = self.save_dir / "actv1_student.safetensors.prev"
                try:
                    if backup_path.exists():
                        backup_path.unlink()
                    checkpoint_path.rename(backup_path)
                    logger.debug("Backed up previous checkpoint")
                    self.print_fn({"checkpoint_save": "old_checkpoint_backed_up"})
                except Exception as e:
                    logger.warning(f"Failed to backup old checkpoint: {e}")
                    self.print_fn({"checkpoint_save": "backup_warning", "error": str(e)})
            
            # Move temp file to final location
            tmp_path.rename(checkpoint_path)
            logger.debug("Moved temporary file to final location")
            
            # Save checkpoint metadata
            metadata = {
                "step": step,
                "cycle": cycle,
                "reason": reason,
                "timestamp": str(__import__('datetime').datetime.now()),
            }
            metadata_path = self.save_dir / "checkpoint_metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update brain.json last_session.checkpoint_path to ensure resume works
            # This is critical for single-GPU training which doesn't always go through finalization
            try:
                brain_json_path = self.save_dir / "brain.json"
                if brain_json_path.exists():
                    with open(brain_json_path, 'r') as f:
                        brain_data = json.load(f)
                    
                    # Update last_session with current checkpoint info
                    if "last_session" not in brain_data:
                        brain_data["last_session"] = {}
                    
                    # Calculate cumulative total_steps
                    # step parameter is the CURRENT session steps, need to add to previous total
                    prev_total = brain_data.get("training_steps", 0)
                    prev_session_steps = brain_data.get("last_session", {}).get("steps_completed", 0)
                    
                    # If step is less than prev_session_steps, we're starting fresh (counter reset)
                    # Otherwise, add the increment to cumulative total
                    if step < prev_session_steps:
                        # Fresh start detected - use step as new total
                        current_total = step
                    else:
                        # Add new steps to previous total
                        current_total = prev_total + (step - prev_session_steps)
                    
                    brain_data["last_session"]["checkpoint_path"] = str(checkpoint_path)
                    brain_data["last_session"]["steps_completed"] = step
                    brain_data["last_session"]["total_steps"] = current_total  # Required for resume detection
                    brain_data["last_session"]["timestamp"] = float(__import__('time').time())
                    brain_data["last_session"]["iterate_cycle"] = cycle  # Required for iterate mode resume
                    brain_data["last_session"]["stopped_early"] = False  # Not a crash, orderly checkpoint
                    brain_data["last_session"]["checkpoint_format"] = "safetensors"
                    brain_data["last_session"]["dataset_file"] = str(self.config.dataset_file) if self.config.dataset_file else None  # Required for resume dataset validation
                    brain_data["checkpoint_file"] = "actv1_student.safetensors"
                    brain_data["checkpoint_format"] = "safetensors"
                    brain_data["last_trained"] = float(__import__('time').time())
                    # NOTE: Do NOT update training_steps here - finalization will handle cumulative total
                    # Only update last_session.total_steps for resume detection
                    
                    # Write updated brain.json
                    with open(brain_json_path, 'w') as f:
                        json.dump(brain_data, f, indent=2)
                    
                    self.print_fn({"brain_json": "updated", "checkpoint_path": str(checkpoint_path)})
            except Exception as e:
                logger.warning(f"Failed to update brain.json: {e}")
                self.print_fn({"brain_json_update": "warning", "error": str(e)})
            
            final_size_mb = round(checkpoint_path.stat().st_size / (1024**2), 1)
            logger.info(f"Checkpoint saved successfully: {final_size_mb} MB")
            
            self.print_fn({
                "checkpoint_save": "SUCCESS",
                "reason": reason,
                "step": step,
                "cycle": cycle,
                "size_mb": final_size_mb,
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")
            self.print_fn({
                "checkpoint_save": "FAILED",
                "reason": reason,
                "error": str(e),
            })
            return False
    
    def cleanup(self):
        """Restore original signal handlers."""
        if self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm:
            signal.signal(signal.SIGTERM, self._original_sigterm)
        logger.debug("Restored original signal handlers")
        self.print_fn({"checkpoint_saver": "signal_handlers_restored"})
