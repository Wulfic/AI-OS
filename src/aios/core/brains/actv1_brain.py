"""ACTv1Brain - HRM brain with custom hierarchical architecture and 3rd party tokenizers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging
import os
import json
import time

logger = logging.getLogger(__name__)


@dataclass
class ACTv1Brain:
    """HRM brain using custom ACTv1 architecture with 3rd party tokenizers.
    
    This brain loads the custom hierarchical reasoning model architecture
    with H-layers and L-layers, using tokenizers from HuggingFace but with
    our own trained weights.
    """

    name: str
    modalities: List[str]
    checkpoint_path: str  # Path to actv1_student.safetensors
    brain_config_path: Optional[str] = None  # Path to brain.json
    max_seq_len: Optional[int] = None  # Explicit max_seq_len override
    inference_device: Optional[str] = None  # Specific device for inference
    
    # Sampling parameters (configurable)
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    
    # Loaded components (lazy-loaded)
    _model: Optional[Any] = field(default=None, repr=False)
    _tokenizer: Optional[Any] = field(default=None, repr=False)
    _config: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _device: Optional[str] = field(default=None, repr=False)
    _oom_warning_printed: bool = field(default=False, repr=False)
    
    def set_sampling_params(
        self, 
        temperature: float | None = None, 
        top_p: float | None = None, 
        top_k: int | None = None
    ) -> None:
        """Update sampling parameters.
        
        Args:
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling threshold (0.0-1.0)
            top_k: Top-k filtering (0 = disabled)
        """
        if temperature is not None:
            self.temperature = max(0.0, min(2.0, temperature))
        if top_p is not None:
            self.top_p = max(0.0, min(1.0, top_p))
        if top_k is not None:
            self.top_k = max(0, int(top_k))

    def _convert_dtype(self, dtype_str: str) -> str:
        """Convert dtype string to torch dtype name.
        
        Args:
            dtype_str: Dtype string like "fp32", "bf16", etc.
            
        Returns:
            Torch dtype name like "float32", "bfloat16", etc.
        """
        dtype_map = {
            "fp32": "float32",
            "fp16": "float16",
            "bf16": "bfloat16",
            "float32": "float32",
            "float16": "float16",
            "bfloat16": "bfloat16",
        }
        return dtype_map.get(dtype_str.lower(), "float32")

    def _ensure_loaded(self) -> None:
        """Ensure ACTv1 model and tokenizer are loaded."""
        if self._model is not None:
            return
        
        logger.info(f"Loading ACTv1 brain: {self.name}")
        
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as e:
            logger.error("PyTorch and transformers are required but not installed")
            raise RuntimeError(
                "PyTorch and transformers are required for ACTv1 brains but not installed."
            ) from e
        
        # Validate checkpoint path
        if not self.checkpoint_path:
            logger.error(f"Brain {self.name} has no checkpoint_path specified")
            raise ValueError(
                f"ACTv1 brain {self.name} has no checkpoint_path. "
                "Brain must be created with a valid checkpoint path."
            )
        
        logger.debug(f"Checkpoint path: {self.checkpoint_path}")
        
        # Load brain configuration
        config_path = self.brain_config_path
        if not config_path:
            # Try to find brain.json next to checkpoint
            checkpoint_dir = os.path.dirname(self.checkpoint_path)
            config_path = os.path.join(checkpoint_dir, "brain.json")
        
        if not os.path.exists(config_path):
            logger.error(f"brain.json not found at {config_path}")
            raise FileNotFoundError(
                f"brain.json not found at {config_path}. "
                "ACTv1 brains require brain.json configuration."
            )
        
        logger.info(f"Loading brain configuration from: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = json.load(f)
        
        # Log configuration details
        logger.debug(f"Brain config: {json.dumps(self._config, indent=2)}")
        
        # Determine context length
        if self.max_seq_len is not None:
            max_seq_len = self.max_seq_len
        else:
            max_seq_len = self._config.get("max_seq_len", 2048)  # Default to 2048 if not specified
        
        logger.info(f"Context length: {max_seq_len}")
        
        # Load tokenizer
        tokenizer_path = self._config.get("tokenizer_model")
        if not tokenizer_path:
            logger.error("brain.json missing 'tokenizer_model' field")
            raise ValueError(
                f"brain.json must specify 'tokenizer_model' path for ACTv1 brain {self.name}"
            )
        
        # Convert to absolute path if relative
        if not os.path.isabs(tokenizer_path):
            tokenizer_path = os.path.abspath(tokenizer_path)
        
        if not os.path.exists(tokenizer_path):
            logger.error(f"Tokenizer path does not exist: {tokenizer_path}")
            raise FileNotFoundError(
                f"Tokenizer path does not exist: {tokenizer_path}. "
                f"Check that the tokenizer files are present."
            )
        
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        print(f"[ACTv1Brain] Loading tokenizer from: {tokenizer_path}")
        try:
            # Try loading with fast tokenizer first (works with tokenizer.json)
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                use_fast=True,  # Use fast tokenizer which works with tokenizer.json
            )
            logger.debug("Successfully loaded fast tokenizer")
        except Exception as e:
            logger.warning(f"Fast tokenizer failed, falling back to slow tokenizer: {e}")
            # Fall back to slow tokenizer if fast fails
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    trust_remote_code=True,
                    use_fast=False,
                )
                logger.debug("Successfully loaded slow tokenizer")
            except Exception as e2:
                logger.error(f"Both fast and slow tokenizer loading failed: {e}, {e2}")
                raise RuntimeError(
                    f"Failed to load tokenizer from {tokenizer_path}: {e}. "
                    f"Also tried slow tokenizer: {e2}. "
                    f"Ensure the tokenizer files (tokenizer.json or tokenizer.model) are present."
                ) from e
        
        # Log tokenizer details
        vocab_size = len(self._tokenizer)
        logger.info(f"Tokenizer loaded: vocab_size={vocab_size}")
        
        # Ensure tokenizer has pad token
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            logger.debug("Set pad_token = eos_token")
        
        # Determine device
        if self.inference_device:
            self._device = self.inference_device
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
        
        logger.info(f"Target device: {self._device}")
        print(f"[ACTv1Brain] Loading model from: {self.checkpoint_path}")
        print(f"[ACTv1Brain] Device: {self._device}, Context length: {max_seq_len}")
        
        # Build ACTv1 model configuration
        from aios.core.hrm_models.act_v1 import build_model
        
        model_config = {
            # Required fields
            "batch_size": 1,  # Inference batch size
            "seq_len": max_seq_len,
            "num_puzzle_identifiers": 0,  # Not used in chat inference
            "vocab_size": self._config.get("vocab_size", len(self._tokenizer)),
            
            # Architecture - map from brain.json to model config
            "H_cycles": self._config.get("h_cycles", self._config.get("halt_max_steps", 2)),
            "L_cycles": self._config.get("l_cycles", self._config.get("halt_max_steps", 2)),
            "H_layers": self._config.get("h_layers", 2),
            "L_layers": self._config.get("l_layers", 2),
            
            # Model dimensions
            "hidden_size": self._config.get("hidden_size", 512),
            "num_heads": self._config.get("num_heads", self._config.get("heads", 8)),
            "expansion": self._config.get("expansion", 4.0),
            
            # Halting
            "halt_max_steps": self._config.get("halt_max_steps", self._config.get("h_cycles", 2)),
            "halt_exploration_prob": self._config.get("halt_exploration_prob", 0.1),
            
            # Positional encoding
            "pos_encodings": self._config.get("pos_encoding", "rope"),
            
            # MoE configuration - using static MoE (dynamic experts not yet implemented)
            "use_moe": self._config.get("use_moe", False),
            "num_experts": self._config.get("num_experts", 8),
            "num_experts_per_tok": self._config.get("num_experts_per_tok", 2),
            
            # Device - pass device to model for efficient initialization
            "device": self._device,
            
            # Optimization - convert dtype names to torch dtype names
            "forward_dtype": self._convert_dtype(self._config.get("dtype", "float32")),
        }
        
        # Log model architecture details
        logger.info(
            f"Building model: H{model_config['H_layers']}/L{model_config['L_layers']} layers, "
            f"hidden={model_config['hidden_size']}, heads={model_config['num_heads']}, "
            f"vocab={model_config['vocab_size']}, dtype={model_config['forward_dtype']}"
        )
        
        # Calculate approximate parameter count
        hidden = model_config['hidden_size']
        h_layers = model_config['H_layers']
        l_layers = model_config['L_layers']
        vocab = model_config['vocab_size']
        approx_params = (
            vocab * hidden +  # Embedding
            (h_layers + l_layers) * (4 * hidden * hidden) +  # Transformer layers (approx)
            vocab * hidden  # Output projection
        )
        logger.info(f"Estimated parameters: ~{approx_params / 1e6:.1f}M")
        
        # Build the model (creates on CPU by default)
        self._model = build_model(model_config)
        
        # Move model to device BEFORE loading checkpoint for faster initialization
        logger.info(f"Moving model to {self._device}...")
        print(f"[ACTv1Brain] Moving model to {self._device}...")
        self._model.to(self._device)
        self._model.eval()
        
        # Load checkpoint weights
        try:
            from safetensors.torch import load_file as load_safetensors
            
            if not os.path.exists(self.checkpoint_path):
                logger.error(f"Checkpoint not found: {self.checkpoint_path}")
                raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
            
            # Log checkpoint file size
            checkpoint_size_bytes = os.path.getsize(self.checkpoint_path)
            checkpoint_size_mb = checkpoint_size_bytes / (1024 * 1024)
            logger.info(f"Loading checkpoint ({checkpoint_size_mb:.1f} MB)...")
            print(f"[ACTv1Brain] Loading checkpoint weights...")

            state_dict = load_safetensors(self.checkpoint_path, device=str(self._device))
            
            # Log state dict info
            num_tensors = len(state_dict)
            logger.debug(f"Checkpoint contains {num_tensors} tensors")
            
            # Check for vocabulary size mismatch and resize embeddings if needed
            checkpoint_vocab_size = None
            model_vocab_size = model_config['vocab_size']
            
            if "inner.embed_tokens.embedding_weight" in state_dict:
                checkpoint_vocab_size = state_dict["inner.embed_tokens.embedding_weight"].shape[0]
            elif "inner.embed_tokens.weight" in state_dict:
                checkpoint_vocab_size = state_dict["inner.embed_tokens.weight"].shape[0]
            
            if checkpoint_vocab_size is not None and checkpoint_vocab_size != model_vocab_size:
                logger.warning(
                    f"Vocab size mismatch: checkpoint={checkpoint_vocab_size}, "
                    f"model={model_vocab_size}, resizing embeddings..."
                )
                print(f"[ACTv1Brain] WARNING: Vocab size mismatch detected!")
                print(f"[ACTv1Brain]   Checkpoint vocab: {checkpoint_vocab_size}, Model vocab: {model_vocab_size}")
                print(f"[ACTv1Brain]   Resizing embeddings to match model...")
                
                # Resize embedding layer in state_dict
                for key in ["inner.embed_tokens.embedding_weight", "inner.embed_tokens.weight"]:
                    if key in state_dict:
                        old_embeddings = state_dict[key]
                        hidden_size = old_embeddings.shape[1]
                        
                        # Create new embeddings with correct vocab size
                        new_embeddings = torch.zeros(
                            model_vocab_size, 
                            hidden_size,
                            dtype=old_embeddings.dtype,
                            device=old_embeddings.device
                        )
                        
                        # Copy existing embeddings
                        min_vocab = min(checkpoint_vocab_size, model_vocab_size)
                        new_embeddings[:min_vocab] = old_embeddings[:min_vocab]
                        
                        # Initialize new tokens with small random values (same strategy as training)
                        if model_vocab_size > checkpoint_vocab_size:
                            with torch.no_grad():
                                init_std = 1.0 / (hidden_size ** 0.5)
                                new_embeddings[checkpoint_vocab_size:].normal_(mean=0.0, std=init_std)
                                print(f"[ACTv1Brain]   Initialized {model_vocab_size - checkpoint_vocab_size} new token embeddings")
                        
                        state_dict[key] = new_embeddings
                
                # Resize output projection (lm_head) in state_dict
                for key in ["inner.lm_head.weight", "lm_head.weight"]:
                    if key in state_dict:
                        old_lm_head = state_dict[key]
                        hidden_size = old_lm_head.shape[1]
                        
                        # Create new lm_head with correct vocab size
                        new_lm_head = torch.zeros(
                            model_vocab_size,
                            hidden_size,
                            dtype=old_lm_head.dtype,
                            device=old_lm_head.device
                        )
                        
                        # Copy existing weights
                        min_vocab = min(checkpoint_vocab_size, model_vocab_size)
                        new_lm_head[:min_vocab] = old_lm_head[:min_vocab]
                        
                        # Initialize new output projections with small random values
                        if model_vocab_size > checkpoint_vocab_size:
                            with torch.no_grad():
                                init_std = 1.0 / (hidden_size ** 0.5)
                                new_lm_head[checkpoint_vocab_size:].normal_(mean=0.0, std=init_std)
                        
                        state_dict[key] = new_lm_head
            
            # Load state dict with resized embeddings
            missing, unexpected = self._model.load_state_dict(state_dict, strict=False)
            
            # Log load results
            if missing:
                logger.warning(f"Missing keys in checkpoint: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected}")
            
            logger.info(f"Successfully loaded {self.name}")
            logger.info(
                f"Architecture: H{model_config['H_layers']}/L{model_config['L_layers']}, "
                f"Vocab: {model_config['vocab_size']}, Hidden: {model_config['hidden_size']}"
            )
            
            print(f"[ACTv1Brain] Loaded {self.name} successfully")
            print(f"[ACTv1Brain]   Architecture: H{model_config['H_layers']}/L{model_config['L_layers']}")
            print(f"[ACTv1Brain]   Vocab: {model_config['vocab_size']}, Hidden: {model_config['hidden_size']}")
            
        except Exception as e:
            logger.error(f"Failed to load ACTv1 checkpoint: {e}")
            raise RuntimeError(f"Failed to load ACTv1 checkpoint: {e}") from e

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference with the ACTv1 model.
        
        Args:
            task: Task dict with 'payload' containing prompt or conversation
            
        Returns:
            Result dict with generated response
        """
        self._ensure_loaded()
        
        try:
            import torch
            
            payload = task.get("payload", {})
            
            # Handle both string payload and dict payload
            if isinstance(payload, str):
                prompt = payload
                max_new_tokens = 100  # Default
            elif isinstance(payload, dict):
                prompt = payload.get("prompt", "")
                max_new_tokens = payload.get("max_tokens", 100)
            else:
                prompt = str(payload)
                max_new_tokens = 100
            
            # Tokenize input on CPU first to avoid carrying stale CUDA state
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self._config.get("max_seq_len", 2048),
            )

            input_ids = inputs["input_ids"]

            requested_device = torch.device(self._device or "cpu")
            runtime_device = requested_device

            def _move_to_device(tensor: torch.Tensor) -> torch.Tensor:
                nonlocal runtime_device

                if tensor.device == runtime_device:
                    return tensor

                attempts = 0
                while True:
                    try:
                        return tensor.to(runtime_device)
                    except RuntimeError as err:  # pragma: no cover - defensive path
                        err_msg = str(err).lower()
                        if (
                            "out of memory" in err_msg
                            and runtime_device.type == "cuda"
                            and torch.cuda.is_available()
                            and attempts == 0
                        ):
                            logger.warning(
                                "[ACTv1Brain] CUDA OOM while moving tensors; clearing cache and retrying once."
                            )
                            attempts += 1
                            try:
                                torch.cuda.empty_cache()
                            except RuntimeError:  # pragma: no cover - defensive path
                                logger.warning(
                                    "[ACTv1Brain] Failed to clear CUDA cache; falling back to CPU."
                                )
                                runtime_device = torch.device("cpu")
                                self._device = "cpu"
                                if hasattr(self._model, "to"):
                                    self._model.to(runtime_device)
                                return tensor.to(runtime_device)
                            time.sleep(0.1)
                            continue

                        if (
                            "out of memory" in err_msg
                            and runtime_device.type == "cuda"
                            and torch.cuda.is_available()
                        ):
                            logger.warning(
                                "[ACTv1Brain] Falling back to CPU after repeated CUDA OOM during tensor staging."
                            )
                            runtime_device = torch.device("cpu")
                            self._device = "cpu"
                            if hasattr(self._model, "to"):
                                self._model.to(runtime_device)
                            return tensor.to(runtime_device)

                        raise

            input_ids = _move_to_device(input_ids)
            
            # Generate response using autoregressive decoding
            # ACTv1 processes full sequences, so we'll give it the growing context each time
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    forward_attempts = 0
                    while True:
                        if input_ids.device != runtime_device:
                            input_ids = input_ids.to(runtime_device)

                        target_device = runtime_device

                        # Prepare batch with current sequence
                        batch = {
                            "input_ids": input_ids,
                            "inputs": input_ids,
                            "puzzle_identifiers": torch.zeros(
                                input_ids.shape[0], dtype=torch.long, device=input_ids.device
                            ),
                        }

                        # Fresh carry state for each forward pass
                        # This allows the model to process the full context including previous generations
                        try:
                            carry = self._model.initial_carry(batch)

                            # Forward pass - model sees full context
                            carry, outputs = self._model(carry, batch)
                            break
                        except RuntimeError as err:  # pragma: no cover - GPU error path
                            err_msg = str(err).lower()
                            if (
                                "out of memory" in err_msg
                                and target_device.type == "cuda"
                                and torch.cuda.is_available()
                                and forward_attempts == 0
                            ):
                                logger.warning(
                                    "[ACTv1Brain] CUDA OOM during forward pass; clearing cache and falling back to CPU."
                                )
                                forward_attempts += 1
                                try:
                                    torch.cuda.empty_cache()
                                except RuntimeError:
                                    logger.warning(
                                        "[ACTv1Brain] Failed to clear CUDA cache after forward OOM; forcing CPU fallback."
                                    )
                                runtime_device = torch.device("cpu")
                                self._device = "cpu"
                                if hasattr(self._model, "to"):
                                    self._model.to(runtime_device)
                                continue
                            raise
                    
                    # Get logits for next token prediction
                    logits = outputs.get("logits")
                    if logits is None:
                        return {"ok": False, "error": "Model did not return logits"}
                    
                    # Get the last token's logits and sample next token
                    next_token_logits = logits[0, -1, :]  # [vocab_size]
                    
                    # Apply sampling parameters (use instance attributes)
                    next_token_logits = next_token_logits / self.temperature
                    
                    # Apply top-k filtering
                    if self.top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, self.top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Apply top-p (nucleus) filtering
                    if self.top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > self.top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample from the filtered distribution
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).unsqueeze(0)  # [1, 1]
                    
                    # Stop if EOS token
                    if next_token_id.item() == self._tokenizer.eos_token_id:
                        break
                    
                    # Append to input sequence
                    input_ids = torch.cat([input_ids, next_token_id], dim=1)
                    
                    # Stop if exceeds max length
                    if input_ids.shape[1] >= self._config.get("max_seq_len", 2048):
                        break
                
                # Decode only the new tokens (skip the original prompt)
                prompt_length = inputs["input_ids"].shape[1]
                generated_ids = input_ids[0, prompt_length:]
                response = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                return {
                    "ok": True,
                    "response": response,
                    "text": response,  # Compatibility with chat handler
                    "model": self.name,
                }
                
        except Exception as e:
            err_msg = str(e).lower()
            if "out of memory" in err_msg:
                if not self._oom_warning_printed:
                    logger.warning(
                        "[ACTv1Brain] Returning empty response after CUDA OOM; consider using smaller batch or CPU device."
                    )
                    self._oom_warning_printed = True
                else:
                    logger.debug(
                        "[ACTv1Brain] Suppressing repeated CUDA OOM warning; returning empty response."
                    )
                return {
                    "ok": True,
                    "response": "",
                    "text": "",
                    "model": self.name,
                    "warning": "cuda_oom_fallback",
                }

            import traceback
            error_trace = traceback.format_exc()
            print(f"[ACTv1Brain] Error during inference: {e}")
            print(f"[ACTv1Brain] Traceback:\n{error_trace}")
            return {
                "ok": False,
                "error": f"ACTv1 inference failed: {e}",
                "traceback": error_trace
            }

    def train(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Training not supported through Brain interface.
        
        Use the HRM training CLI/GUI for training ACTv1 brains.
        """
        return {
            "ok": False,
            "error": "Training not supported through Brain interface. Use HRM training CLI/GUI.",
        }

    def size_bytes(self) -> int:
        """Estimate memory footprint of loaded ACTv1 model.
        
        Returns:
            Estimated size in bytes
        """
        if self._model is None:
            # Estimate from config if not loaded
            if self._config:
                hidden = self._config.get("hidden_size", 512)
                h_layers = self._config.get("h_layers", 2)
                l_layers = self._config.get("l_layers", 2)
                vocab_size = self._config.get("vocab_size", 32000)
                
                # Rough estimate: embedding + layer params
                # Each layer has attention + FFN params
                params_per_layer = (hidden * hidden * 4) + (hidden * hidden * 4 * 4)  # attn + FFN
                total_params = (vocab_size * hidden) + (h_layers * params_per_layer) + (l_layers * params_per_layer)
                
                # 4 bytes per float32 parameter + overhead
                return int(total_params * 4 * 1.2)
            return 100 * 1024 * 1024  # Default 100MB estimate
        
        try:
            import torch
            total = sum(p.numel() * p.element_size() for p in self._model.parameters())
            return int(total * 1.2)  # Add 20% overhead
        except Exception:
            return 100 * 1024 * 1024

    def last_trained(self) -> float:
        """Get last training timestamp from brain.json.
        
        Returns:
            Unix timestamp of last training, or 0 if unknown
        """
        if self._config:
            return float(self._config.get("last_trained", 0))
        
        # Try to load from brain.json if not loaded
        config_path = self.brain_config_path
        if not config_path and self.checkpoint_path:
            checkpoint_dir = os.path.dirname(self.checkpoint_path)
            config_path = os.path.join(checkpoint_dir, "brain.json")
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    return float(config.get("last_trained", 0))
            except Exception:
                pass
        
        return 0.0
