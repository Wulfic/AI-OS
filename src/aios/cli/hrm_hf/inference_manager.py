"""Inference Model Manager for Multi-GPU Training.

This module provides InferenceModelManager which manages a separate inference model
on a different GPU while training proceeds on another GPU. Supports hot-reloading
the inference model from training checkpoints.
"""

from __future__ import annotations

from typing import Optional, Any, Dict, Callable
from pathlib import Path
import torch
import gc


class InferenceModelManager:
    """Manages a separate inference model on a different GPU for multi-GPU systems.
    
    Allows running inference on one GPU while training on another, with periodic
    hot-reloading of the trained weights.
    
    Example:
        >>> manager = InferenceModelManager(
        ...     inference_device="cuda:1",
        ...     model_factory=lambda: build_actv1_model(...),
        ...     checkpoint_dir=Path("artifacts/brains/actv1/MyBrain")
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
        self.inference_device = inference_device
        self.model_factory = model_factory
        self.checkpoint_dir = checkpoint_dir
        self.training_device = training_device
        self.model: Optional[Any] = None
        self._initialized = False
        
        # Validate devices
        if torch.cuda.is_available():
            self._validate_devices()
    
    def _validate_devices(self) -> None:
        """Validate that the requested devices are available."""
        if not self.inference_device:
            return
            
        # Parse device index from "cuda:X"
        try:
            if ":" in self.inference_device:
                device_idx = int(self.inference_device.split(":")[1])
            else:
                device_idx = 0
                
            device_count = torch.cuda.device_count()
            if device_idx >= device_count:
                raise ValueError(
                    f"Inference device {self.inference_device} not available. "
                    f"Only {device_count} GPU(s) detected."
                )
            
            # Validate training device if provided
            if self.training_device and ":" in self.training_device:
                train_idx = int(self.training_device.split(":")[1])
                if train_idx >= device_count:
                    raise ValueError(
                        f"Training device {self.training_device} not available. "
                        f"Only {device_count} GPU(s) detected."
                    )
                if train_idx == device_idx:
                    print({
                        "warning": "inference_device and training_device are the same",
                        "note": "Multi-GPU separation disabled",
                        "device": self.inference_device
                    })
        except ValueError as e:
            raise e
        except Exception as e:
            print({"device_validation_error": str(e)})
    
    def initialize(self) -> bool:
        """Initialize the inference model on the inference device.
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        if self._initialized:
            return True
            
        try:
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
                    
            self._initialized = True
            print({
                "inference_model_init": "success",
                "device": self.inference_device
            })
            return True
            
        except Exception as e:
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
            print({"inference_reload": "skipped", "reason": "not initialized"})
            return False
            
        if not self.checkpoint_dir:
            print({"inference_reload": "skipped", "reason": "no checkpoint_dir"})
            return False
            
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            print({
                "inference_reload": "skipped",
                "reason": "checkpoint not found",
                "path": str(checkpoint_path)
            })
            return False
            
        try:
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
                    print({"inference_reload": "loaded_safetensors", "checkpoint": checkpoint_name})
                except Exception as st_error:
                    # Fallback to torch.load for backwards compatibility
                    print({"inference_reload": "safetensors_failed", "trying_torch": True, "error": str(st_error)})
                    state_dict = torch.load(
                        checkpoint_path,
                        map_location=self.inference_device
                    )
                
                # Handle wrapped models (DDP, DeepSpeed, etc.)
                if hasattr(self.model, "module"):
                    self.model.module.load_state_dict(state_dict, strict=False)
                else:
                    self.model.load_state_dict(state_dict, strict=False)
                    
                self.model.eval()
                
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print({
                "inference_reload": "success",
                "checkpoint": checkpoint_name
            })
            return True
            
        except Exception as e:
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
            raise RuntimeError("Inference model not initialized. Call initialize() first.")
            
        with torch.no_grad():
            with torch.cuda.device(self.inference_device):
                return self.model.generate(*args, **kwargs)
    
    def forward(self, *args, **kwargs) -> Any:
        """Run forward pass using the loaded model.
        
        Forward all arguments to the model's forward method.
        """
        if not self._initialized or self.model is None:
            raise RuntimeError("Inference model not initialized. Call initialize() first.")
            
        with torch.no_grad():
            with torch.cuda.device(self.inference_device):
                return self.model(*args, **kwargs)
    
    def cleanup(self) -> None:
        """Clean up the inference model and free GPU memory."""
        if self.model is not None:
            try:
                # Move model to CPU before deletion to free GPU memory
                self.model.cpu()
                del self.model
                self.model = None
                
                # Force garbage collection and CUDA cache cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                print({
                    "inference_model_cleanup": "success",
                    "device": self.inference_device
                })
            except Exception as e:
                print({
                    "inference_model_cleanup": "failed",
                    "error": str(e)
                })
                
        self._initialized = False
