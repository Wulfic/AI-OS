"""lm_eval adapter for AI-OS native brains.

This module provides a custom lm_eval model class that wraps AI-OS brains
(actv1, etc.) so they can be evaluated using the lm-evaluation-harness framework.
"""

from __future__ import annotations

from typing import Any, Optional, Union
import torch
from pathlib import Path
import json

try:
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
except ImportError:
    # lm_eval not installed, skip registration
    LM = object
    def register_model(name):
        def decorator(cls):
            return cls
        return decorator


@register_model("aios")
class AIOSBrainModel(LM):
    """lm_eval model adapter for AI-OS native brains.
    
    This allows AI-OS brains (actv1, etc.) to be evaluated using standard
    benchmarks through the lm-evaluation-harness framework.
    
    Usage:
        lm_eval --model aios --model_args brain_path=/path/to/brain --tasks hellaswag
    """
    
    def __init__(
        self,
        brain_path: str,
        device: str = "cuda",
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize the AI-OS brain model.
        
        Args:
            brain_path: Path to the brain directory (e.g., artifacts/brains/actv1/English-v1)
            device: Device to run on (cuda, cpu, etc.)
            batch_size: Batch size for evaluation
            **kwargs: Additional arguments
        """
        super().__init__()
        
        self.brain_path = Path(brain_path)
        self.device = device
        self._batch_size = batch_size
        
        # Load brain
        self._load_brain()
    
    def _load_brain(self) -> None:
        """Load the AI-OS brain."""
        # Check for brain.json
        brain_json_path = self.brain_path / "brain.json"
        if not brain_json_path.exists():
            raise FileNotFoundError(f"brain.json not found at {brain_json_path}")
        
        with open(brain_json_path, 'r') as f:
            brain_config = json.load(f)
        
        brain_type = brain_config.get("type", "unknown")
        brain_name = brain_config.get("name", self.brain_path.name)
        
        if brain_type == "actv1":
            # Load ACTv1 brain
            from aios.core.brains.actv1_brain import ACTv1Brain
            
            checkpoint_path = str(self.brain_path / brain_config.get("checkpoint_file", "actv1_student.safetensors"))
            
            self.brain = ACTv1Brain(
                name=brain_name,
                modalities=["text"],
                checkpoint_path=checkpoint_path,
                brain_config_path=str(brain_json_path),
                inference_device=self.device,
            )
            
            # Pre-load the brain
            self.brain._ensure_loaded()
            
            # Get tokenizer
            self.tokenizer = self.brain._tokenizer
            
            # Set vocab size
            self._vocab_size = len(self.tokenizer)
            
        else:
            raise ValueError(f"Unsupported brain type: {brain_type}")
    
    @property
    def eot_token_id(self) -> int:
        """End of text token ID."""
        return self.tokenizer.eos_token_id
    
    @property
    def max_length(self) -> int:
        """Maximum sequence length."""
        if hasattr(self.brain, '_config') and self.brain._config:
            return self.brain._config.get("max_seq_len", 2048)
        return 2048
    
    @property
    def max_gen_toks(self) -> int:
        """Maximum tokens to generate."""
        return 256
    
    @property
    def batch_size(self) -> int:
        """Batch size for evaluation."""
        return self._batch_size
    
    @property
    def device(self) -> str:
        """Device string."""
        return self._device
    
    @device.setter
    def device(self, value: str) -> None:
        """Set device."""
        self._device = value
    
    def tok_encode(self, string: str, **kwargs: Any) -> list[int]:
        """Encode a string to token IDs.
        
        Args:
            string: String to encode
            **kwargs: Additional arguments
            
        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(string, add_special_tokens=False)
    
    def tok_decode(self, tokens: list[int], **kwargs: Any) -> str:
        """Decode token IDs to a string.
        
        Args:
            tokens: List of token IDs
            **kwargs: Additional arguments
            
        Returns:
            Decoded string
        """
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def loglikelihood(
        self,
        requests: list,
        **kwargs: Any,
    ) -> list[tuple[float, bool]]:
        """Compute log-likelihoods for continuation requests.
        
        Args:
            requests: List of Instance objects or (context, continuation) tuples
            **kwargs: Additional arguments
            
        Returns:
            List of (log_likelihood, is_greedy) tuples
        """
        print(f"[AIOS Adapter] Processing {len(requests)} loglikelihood requests...")
        results = []
        
        for idx, request in enumerate(requests):
            if idx % 10 == 0:  # More frequent updates
                print(f"[AIOS Adapter] Progress: {idx}/{len(requests)}", flush=True)
            # Handle both Instance objects and raw tuples
            if hasattr(request, 'arguments'):
                # New lm_eval API: Instance objects
                context, continuation = request.arguments
            else:
                # Legacy API: raw tuples
                context, continuation = request
            # Encode context and continuation
            context_tokens = self.tok_encode(context)
            continuation_tokens = self.tok_encode(continuation)
            
            # Combine for full sequence
            full_tokens = context_tokens + continuation_tokens
            
            # Check if sequence exceeds max length
            max_len = self.max_length
            if len(full_tokens) > max_len:
                # Truncate context from the left, keeping continuation intact
                overflow = len(full_tokens) - max_len
                context_tokens = context_tokens[overflow:]
                full_tokens = context_tokens + continuation_tokens
            
            # Convert to tensor
            input_ids = torch.tensor([full_tokens], device=self.device)
            
            # Get model output
            with torch.no_grad():
                try:
                    # For ACTv1 models, use the carry-state architecture
                    if hasattr(self.brain._model, 'initial_carry'):
                        batch = {
                            "input_ids": input_ids,
                            "inputs": input_ids,
                            "puzzle_identifiers": torch.zeros(input_ids.shape[0], dtype=torch.long, device=self.device)
                        }
                        carry = self.brain._model.initial_carry(batch)
                        carry, outputs = self.brain._model(carry, batch)
                        logits = outputs.get("logits")
                    elif hasattr(self.brain._model, 'forward'):
                        outputs = self.brain._model(input_ids)
                        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                    else:
                        # Fallback - use approximation
                        results.append((0.0, False))
                        continue
                    
                    if logits is None:
                        results.append((0.0, False))
                        continue
                    
                    # Calculate log probabilities
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    
                    # Get log likelihood of continuation
                    context_len = len(context_tokens)
                    continuation_len = len(continuation_tokens)
                    
                    total_log_prob = 0.0
                    is_greedy = True
                    
                    for i in range(continuation_len):
                        pos = context_len + i - 1  # -1 because logits are shifted
                        if pos >= 0 and pos < log_probs.shape[1]:
                            token_id = continuation_tokens[i]
                            token_log_prob = log_probs[0, pos, token_id].item()
                            total_log_prob += token_log_prob
                            
                            # Check if this was the greedy choice
                            greedy_token = log_probs[0, pos].argmax().item()
                            if greedy_token != token_id:
                                is_greedy = False
                    
                    results.append((total_log_prob, is_greedy))
                    
                except Exception as e:
                    # If there's an error, return conservative estimate
                    print(f"Warning: Error computing log likelihood: {e}")
                    results.append((0.0, False))
        
        print(f"[AIOS Adapter] Completed {len(results)} loglikelihood requests")
        return results
    
    def loglikelihood_rolling(
        self,
        requests: list,
        **kwargs: Any,
    ) -> list[float]:
        """Compute rolling log-likelihoods.
        
        Args:
            requests: List of Instance objects or strings
            **kwargs: Additional arguments
            
        Returns:
            List of log-likelihoods
        """
        results = []
        
        for request in requests:
            # Handle both Instance objects and raw strings
            if hasattr(request, 'arguments'):
                # New lm_eval API: Instance objects
                string = request.arguments[0]
            else:
                # Legacy API: raw strings
                string = request
            tokens = self.tok_encode(string)
            
            # Truncate if too long
            max_len = self.max_length
            if len(tokens) > max_len:
                tokens = tokens[-max_len:]  # Take the last max_len tokens
            
            input_ids = torch.tensor([tokens], device=self.device)
            
            with torch.no_grad():
                try:
                    # For ACTv1 models, use the carry-state architecture
                    if hasattr(self.brain._model, 'initial_carry'):
                        batch = {
                            "input_ids": input_ids,
                            "inputs": input_ids,
                            "puzzle_identifiers": torch.zeros(input_ids.shape[0], dtype=torch.long, device=self.device)
                        }
                        carry = self.brain._model.initial_carry(batch)
                        carry, outputs = self.brain._model(carry, batch)
                        logits = outputs.get("logits")
                    elif hasattr(self.brain._model, 'forward'):
                        outputs = self.brain._model(input_ids)
                        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                    else:
                        results.append(0.0)
                        continue
                    
                    if logits is None:
                        results.append(0.0)
                        continue
                    
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    
                    total_log_prob = 0.0
                    for i in range(len(tokens) - 1):
                        next_token = tokens[i + 1]
                        token_log_prob = log_probs[0, i, next_token].item()
                        total_log_prob += token_log_prob
                    
                    results.append(total_log_prob)
                    
                except Exception as e:
                    print(f"Warning: Error computing rolling log likelihood: {e}")
                    results.append(0.0)
        
        return results
    
    def generate_until(
        self,
        requests: list,
        **kwargs: Any,
    ) -> list[str]:
        """Generate text until stopping criteria.
        
        Args:
            requests: List of Instance objects or (context, generation_kwargs) tuples
            **kwargs: Additional arguments
            
        Returns:
            List of generated strings
        """
        results = []
        
        for request in requests:
            # Handle both Instance objects and raw tuples
            if hasattr(request, 'arguments'):
                # New lm_eval API: Instance objects
                context, gen_kwargs = request.arguments
            else:
                # Legacy API: raw tuples
                context, gen_kwargs = request
            # Extract generation parameters
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            until = gen_kwargs.get("until", [])
            
            # Use brain's run method for generation
            task = {
                "payload": {
                    "prompt": context,
                    "max_tokens": max_gen_toks,
                }
            }
            
            result = self.brain.run(task)
            generated_text = result.get("response", "")
            
            # Handle "until" stopping criteria
            if until:
                for stop_seq in until:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]
                        break
            
            results.append(generated_text)
        
        return results


def register_aios_model() -> None:
    """Register the AI-OS model adapter with lm_eval.
    
    This should be called before running evaluations with AI-OS brains.
    """
    # Registration happens via decorator, but this function
    # can be used to ensure the module is imported
    pass


# Auto-register when module is imported
if LM != object:
    register_aios_model()
