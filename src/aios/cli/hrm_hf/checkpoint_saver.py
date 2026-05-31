"""Signal handling and checkpoint saving utilities."""
from __future__ import annotations

import logging
import signal
import sys
import os
from typing import TYPE_CHECKING, Optional, Any, Callable
from pathlib import Path

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aios.core.hrm_training.training_config import TrainingConfig


def prepare_state_dict_for_safetensors(state_dict: dict) -> dict:
    """Return a state dict that is safe to pass to ``safetensors.save_file``.

    safetensors raises at save time for tensors that are on a non-CPU device,
    are non-contiguous (common with gradient checkpointing / slicing), or share
    storage with another tensor (tied embeddings). This moves every tensor to
    CPU, makes it contiguous, and clones it to break any shared storage.
    """
    import torch

    cleaned: dict = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            # clone() breaks shared/tied storage so safetensors won't reject it
            cleaned[key] = tensor.clone()
        else:
            cleaned[key] = value
    return cleaned


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
        """Handle interrupt signals by saving checkpoint.

        A second signal arriving while the checkpoint is being written could
        terminate the process mid-write and corrupt the file. To prevent that
        we ignore further SIGINT/SIGTERM for the duration of the save, then
        exit once the on-disk state is consistent.
        """
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"

        # Ignore further interrupts while we flush state to disk so a second
        # Ctrl-C cannot interrupt the (atomic) save half-way through.
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        except Exception:
            pass

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

        # Exit gracefully now that on-disk state is consistent.
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
            
            # Get model state dict (handle DDP / DeepSpeed wrappers which expose .module)
            if hasattr(self.model, 'module'):
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()

            # Make the state dict safe for safetensors (CPU, contiguous, unshared)
            state_dict = prepare_state_dict_for_safetensors(state_dict)

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
            metadata_tmp = metadata_path.with_suffix(".json.tmp")
            with open(metadata_tmp, 'w') as f:
                json.dump(metadata, f, indent=2)
            os.replace(metadata_tmp, metadata_path)
            
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
                    
                    # Calculate cumulative total_steps.
                    # `step` is the CURRENT session's step count (session-relative).
                    # `training_steps` is the cumulative total as of the last
                    # finalization (i.e. the total before this session began) and is
                    # intentionally NOT updated here, so it stays constant across all
                    # intra-session checkpoint writes. The running cumulative total is
                    # therefore simply (previous cumulative total + current session steps).
                    # NOTE: the previous heuristic that subtracted last_session.steps_completed
                    # and guessed "fresh start" misfired whenever a session was shorter
                    # than the prior one, corrupting the resume step count.
                    prev_total = brain_data.get("training_steps", 0)
                    current_total = prev_total + step
                    
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
                    
                    # Write updated brain.json (atomic: tmp + replace)
                    brain_json_tmp = brain_json_path.with_suffix(".json.tmp")
                    with open(brain_json_tmp, 'w') as f:
                        json.dump(brain_data, f, indent=2)
                    os.replace(brain_json_tmp, brain_json_path)
                    
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
