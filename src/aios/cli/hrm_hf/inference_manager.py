"""Inference Model Manager for Multi-GPU Training.

This module provides InferenceModelManager which manages a separate inference model
on a different GPU while training proceeds on another GPU. Supports hot-reloading
the inference model from training checkpoints.
"""

from __future__ import annotations

import logging
from typing import Optional, Any, Dict, Callable
from pathlib import Path
import torch
import gc

logger = logging.getLogger(__name__)


class InferenceModelManager:
    """Manages a separate inference model on a different GPU for multi-GPU systems.
    
    Allows running inference on one GPU while training on another, with periodic
    hot-reloading of the trained weights.
    
    Example:
        >>> manager = InferenceModelManager(
        ...     inference_device="cuda:1",
        ...     model_factory=lambda: build_actv1_model(...),
        ...     checkpoint_dir=Path(r"C:/ProgramData/AI-OS/artifacts/brains/actv1/MyBrain")
        ... )
        >>> manager.initialize()
        >>> # During training...
        >>> if step % hot_reload_steps == 0:
        ...     manager.reload_from_checkpoint("actv1_student.safetensors")
    """
    
    def __init__(
        self,
        inference_device: str,
        model_factory: Callable[[], Any],
        checkpoint_dir: Optional[Path] = None,
        training_device: Optional[str] = None,
    ):
        """Initialize the inference model manager.
        
        Args:
            inference_device: Device string for inference (e.g., "cuda:1")
            model_factory: Callable that returns a new model instance
            checkpoint_dir: Directory containing checkpoints to reload from
            training_device: Device string for training (for validation, e.g., "cuda:0")
        """
        logger.info(f"Initializing InferenceModelManager: inference_device={inference_device}, training_device={training_device}")
        logger.debug(f"Checkpoint directory: {checkpoint_dir}")
        
        self.inference_device = inference_device
        self.model_factory = model_factory
        self.checkpoint_dir = checkpoint_dir
        self.training_device = training_device
        self.model: Optional[Any] = None
        self._initialized = False
        
        # Validate devices
        if torch.cuda.is_available():
            self._validate_devices()
        else:
            logger.warning("CUDA not available, inference manager will use CPU")
    
    def _validate_devices(self) -> None:
        """Validate that the requested devices are available."""
        if not self.inference_device:
            logger.debug("No inference device specified, skipping validation")
            return
            
        # Parse device index from "cuda:X"
        try:
            if ":" in self.inference_device:
                device_idx = int(self.inference_device.split(":")[1])
            else:
                device_idx = 0
                
            device_count = torch.cuda.device_count()
            logger.debug(f"Device validation: requested device {device_idx}, available GPUs: {device_count}")
            
            if device_idx >= device_count:
                logger.error(f"Inference device {self.inference_device} not available - only {device_count} GPU(s) detected")
                raise ValueError(
                    f"Inference device {self.inference_device} not available. "
                    f"Only {device_count} GPU(s) detected."
                )
            
            # Validate training device if provided
            if self.training_device and ":" in self.training_device:
                train_idx = int(self.training_device.split(":")[1])
                if train_idx >= device_count:
                    logger.error(f"Training device {self.training_device} not available - only {device_count} GPU(s) detected")
                    raise ValueError(
                        f"Training device {self.training_device} not available. "
                        f"Only {device_count} GPU(s) detected."
                    )
                if train_idx == device_idx:
                    logger.warning(f"Inference and training devices are the same ({self.inference_device}) - multi-GPU separation disabled")
                    print({
                        "warning": "inference_device and training_device are the same",
                        "note": "Multi-GPU separation disabled",
                        "device": self.inference_device
                    })
            
            logger.info(f"Device validation successful: inference={self.inference_device}, training={self.training_device}")
        except ValueError as e:
            raise e
        except Exception as e:
            logger.error(f"Device validation error: {e}", exc_info=True)
            print({"device_validation_error": str(e)})
    
    def initialize(self) -> bool:
        """Initialize the inference model on the inference device.
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        if self._initialized:
            logger.debug("Inference model already initialized")
            return True
            
        try:
            logger.info(f"Initializing inference model on device: {self.inference_device}")
            print({
                "inference_model_init": "starting",
                "device": self.inference_device
            })
            
            # Create model on inference device with proper context
            with torch.cuda.device(self.inference_device):
                self.model = self.model_factory()
                if self.model is not None:
                    # Move model to inference device
                    device_obj = torch.device(self.inference_device)
                    self.model.to(device_obj)
                    self.model.eval()  # Set to eval mode
                    logger.debug(f"Model moved to {self.inference_device} and set to eval mode")
                    
            self._initialized = True
            logger.info(f"Inference model initialization successful on {self.inference_device}")
            print({
                "inference_model_init": "success",
                "device": self.inference_device
            })
            return True
            
        except Exception as e:
            logger.error(f"Inference model initialization failed on {self.inference_device}: {e}", exc_info=True)
            print({
                "inference_model_init": "failed",
                "device": self.inference_device,
                "error": str(e)
            })
            return False
    
    def reload_from_checkpoint(self, checkpoint_name: str = "actv1_student.safetensors") -> bool:
        """Reload the inference model from a training checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint file to load
            
        Returns:
            True if reload succeeded, False otherwise
        """
        if not self._initialized or self.model is None:
            logger.warning("Inference reload skipped - model not initialized")
            print({"inference_reload": "skipped", "reason": "not initialized"})
            return False
            
        if not self.checkpoint_dir:
            logger.warning("Inference reload skipped - no checkpoint directory configured")
            print({"inference_reload": "skipped", "reason": "no checkpoint_dir"})
            return False
            
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            logger.warning(f"Inference reload skipped - checkpoint not found: {checkpoint_path}")
            print({
                "inference_reload": "skipped",
                "reason": "checkpoint not found",
                "path": str(checkpoint_path)
            })
            return False
            
        try:
            logger.info(f"Reloading inference model from checkpoint: {checkpoint_name}")
            print({
                "inference_reload": "starting",
                "checkpoint": str(checkpoint_path),
                "device": self.inference_device
            })
            
            # Load checkpoint on inference device (use safetensors)
            with torch.cuda.device(self.inference_device):
                try:
                    from safetensors.torch import load_file as load_safetensors
                    state_dict = load_safetensors(str(checkpoint_path), device=str(self.inference_device))
                    logger.debug(f"Loaded checkpoint using safetensors: {checkpoint_name}")
                    print({"inference_reload": "loaded_safetensors", "checkpoint": checkpoint_name})
                except Exception as st_error:
                    # Fallback to torch.load for backwards compatibility
                    logger.debug(f"Safetensors load failed, falling back to torch.load: {st_error}")
                    print({"inference_reload": "safetensors_failed", "trying_torch": True, "error": str(st_error)})
                    state_dict = torch.load(
                        checkpoint_path,
                        map_location=self.inference_device
                    )
                
                # Handle wrapped models (DDP, DeepSpeed, etc.)
                if hasattr(self.model, "module"):
                    logger.debug("Loading state dict into wrapped model (module)")
                    self.model.module.load_state_dict(state_dict, strict=False)
                else:
                    logger.debug("Loading state dict into model")
                    self.model.load_state_dict(state_dict, strict=False)
                    
                self.model.eval()
                
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache after checkpoint reload")
                
            logger.info(f"Inference model reloaded successfully from {checkpoint_name}")
            print({
                "inference_reload": "success",
                "checkpoint": checkpoint_name
            })
            return True
            
        except Exception as e:
            logger.error(f"Inference reload failed for {checkpoint_name}: {e}", exc_info=True)
            print({
                "inference_reload": "failed",
                "checkpoint": checkpoint_name,
                "error": str(e)
            })
            return False
    
    def generate(self, *args, **kwargs) -> Any:
        """Run inference using the loaded model.
        
        Forward all arguments to the model's generate method.
        """
        if not self._initialized or self.model is None:
            logger.error("Attempted to generate with uninitialized inference model")
            raise RuntimeError("Inference model not initialized. Call initialize() first.")
        
        logger.debug(f"Running inference generation on {self.inference_device}")
        with torch.no_grad():
            with torch.cuda.device(self.inference_device):
                return self.model.generate(*args, **kwargs)
    
    def forward(self, *args, **kwargs) -> Any:
        """Run forward pass using the loaded model.
        
        Forward all arguments to the model's forward method.
        """
        if not self._initialized or self.model is None:
            logger.error("Attempted to forward with uninitialized inference model")
            raise RuntimeError("Inference model not initialized. Call initialize() first.")
            
        with torch.no_grad():
            with torch.cuda.device(self.inference_device):
                return self.model(*args, **kwargs)
    
    def cleanup(self) -> None:
        """Clean up the inference model and free GPU memory."""
        if self.model is not None:
            logger.info(f"Cleaning up inference model on {self.inference_device}")
            try:
                # Move model to CPU before deletion to free GPU memory
                self.model.cpu()
                del self.model
                self.model = None
                
                # Force garbage collection and CUDA cache cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug(f"Model moved to CPU and CUDA cache cleared on {self.inference_device}")
                    
                logger.info("Inference model cleanup complete")
                print({
                    "inference_model_cleanup": "success",
                    "device": self.inference_device
                })
            except Exception as e:
                logger.error(f"Error during inference model cleanup: {e}", exc_info=True)
                print({
                    "inference_model_cleanup": "failed",
                    "error": str(e)
                })
                
        self._initialized = False
