"""HuggingFace Causal LM → HRM Adapter.

This module wraps a pretrained causal LM to present the HRM interface expected by
segment_rollout: model.initial_carry(batch) and model(carry, batch) -> (carry, outputs).

Outputs contain:
  - logits: [B, S, V]
  - q_halt_logits: [B]
  - q_continue_logits: [B]

Imports of heavy dependencies (torch, transformers) are guarded to keep baseline installs lightweight.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .types import _Carry, _InnerCarry
from .device_utils import _require_torch_transformers, pick_device, choose_dtype
from .moe_utils import replace_ffn_with_moe


class HFCausalLM_HRMAdapter:
    """Wrap a HF causal LM to expose HRM-compatible API.

    forward(carry, batch) returns (new_carry, outputs) where outputs has token logits
    and simple halting heads derived from the hidden state at position 0.
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_len: int,
        halt_max_steps: int = 1,
        halt_exploration_prob: float = 0.0,
        device: Optional[str] = None,
        forward_dtype: str = "bfloat16",
        # LoRA options
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[Tuple[str, ...]] = ("c_attn", "c_proj"),
        lora_inference_mode: bool = False,
        # Sparse MoE options
        use_moe: bool = False,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        moe_capacity_factor: float = 1.25,
    ) -> None:
        _require_torch_transformers()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import os

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # Ensure pad token exists for batching; fall back to eos if missing
        if self.tokenizer.pad_token_id is None and hasattr(self.tokenizer, "eos_token_id"):
            try:
                self.tokenizer.pad_token = self.tokenizer.eos_token  # type: ignore[assignment]
            except Exception:
                pass

        # Decide device and dtype upfront
        picked, is_dml, dml_device_obj = pick_device(device)
        self._is_dml = is_dml
        self._dml_device = dml_device_obj
        
        # Normalize device field
        if is_dml:
            # DirectML device object - set placeholder for checks
            self.device = torch.device("cpu")
        else:
            self.device = picked  # type: ignore[assignment]

        chosen_dtype = choose_dtype(forward_dtype, self.device if not is_dml else None, is_dml)

        # Set HF_HOME to suppress deprecated TRANSFORMERS_CACHE warning
        if "TRANSFORMERS_CACHE" in os.environ and "HF_HOME" not in os.environ:
            os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=chosen_dtype,
        )
        base_model.config.output_hidden_states = True  # ensure we get hidden states
        
        # Extract hidden_size early to avoid AttributeError during progressive loading
        self.hidden_size = int(getattr(base_model.config, "hidden_size", 0) or getattr(base_model.config, "n_embd", 0))
        if self.hidden_size <= 0:
            raise RuntimeError("Could not determine hidden size for HF model")
        
        # Store MoE configuration
        self.use_moe = bool(use_moe)
        self.num_experts = int(num_experts) if use_moe else 1
        self.num_experts_per_tok = int(num_experts_per_tok) if use_moe else 1
        self.moe_capacity_factor = float(moe_capacity_factor) if use_moe else 1.0
        
        # Optionally wrap with LoRA (PEFT)
        self.is_peft = False
        self.lora_targets = tuple(lora_target_modules) if lora_target_modules else tuple()
        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except Exception as e:
                raise RuntimeError("LoRA requested but 'peft' is not installed. Install with: pip install peft") from e
            lcfg = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=list(self.lora_targets) if self.lora_targets else None,
                inference_mode=bool(lora_inference_mode),
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(base_model, lcfg)
            self.is_peft = True
        else:
            self.model = base_model
            
        # Move model to target device BEFORE MoE replacement
        if self._is_dml and self._dml_device is not None:
            self.model.to(self._dml_device)
        else:
            self.model.to(self.device)  # type: ignore[arg-type]
        self.model.eval()
        
        # Apply Sparse MoE if enabled
        if self.use_moe and self.num_experts > 1:
            dev = self._dml_device if self._is_dml and self._dml_device is not None else self.device
            replace_ffn_with_moe(
                model=self.model,
                is_peft=self.is_peft,
                hidden_size=self.hidden_size,
                num_experts=self.num_experts,
                num_experts_per_tok=self.num_experts_per_tok,
                moe_capacity_factor=self.moe_capacity_factor,
                device=dev,
            )

        # Store configuration
        self.max_seq_len = int(max(1, max_seq_len))
        self.halt_max_steps = int(max(1, halt_max_steps))
        self.halt_exploration_prob = float(max(0.0, min(1.0, halt_exploration_prob)))

        # Simple halting head on top of pooled hidden state - create on CPU first
        self.q_head = torch.nn.Linear(self.hidden_size, 2, bias=True)
        # Initialize weights before moving to device
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)
        # Now move to the same device as the model
        if self._is_dml and self._dml_device is not None:
            self.q_head = self.q_head.to(self._dml_device)
        else:
            self.q_head = self.q_head.to(self.device)

        # Cache the chosen forward dtype for tensor inits
        self.forward_dtype = chosen_dtype

        # Expose a lightweight training flag for segment_rollout
        self.training = True

    # Minimal Module-like API expected by training utilities
    def train(self, mode: bool = True) -> None:
        """Set training flag on the adapter; does not change base LM unless caller does so."""
        self.training = bool(mode)

    def eval(self) -> None:
        """Set evaluation mode on the adapter; base LM eval is managed separately by caller."""
        self.training = False

    # HRM API
    def initial_carry(self, batch: Dict[str, Any]) -> _Carry:
        """Create initial carry state for HRM inference."""
        import torch
        B, S = batch["inputs"].shape
        H = self.hidden_size
        dev = (self._dml_device if self._is_dml and self._dml_device is not None else self.device)
        z = torch.zeros((B, S, H), dtype=self.forward_dtype, device=dev)
        return _Carry(
            inner_carry=_InnerCarry(z_H=z.clone(), z_L=z.clone()),
            steps=torch.zeros((B,), dtype=torch.int32, device=dev),
            halted=torch.ones((B,), dtype=torch.bool, device=dev),
            current_data={k: v for k, v in batch.items()},
        )

    def forward(self, carry: _Carry, batch: Dict[str, Any]) -> Tuple[_Carry, Dict[str, Any]]:
        """Run one HRM inference step."""
        import torch
        # Get target device - prefer DML if available, otherwise use standard device
        dev = (self._dml_device if self._is_dml and self._dml_device is not None else self.device)
        
        # Move input tensors to device
        input_ids = batch["inputs"].to(dev)
        attn_mask = (input_ids != self.tokenizer.pad_token_id).to(dev) if self.tokenizer.pad_token_id is not None else None
        
        # Run model inference
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False, output_hidden_states=True, return_dict=True)
            # Keep intermediate tensors on the same device as model
            logits = out.logits[:, -input_ids.shape[1]:]  # [B,S,V] - keep on model device
            last_hidden = out.hidden_states[-1][:, -1, :]  # [B,H] take last token state
        
        # Ensure last_hidden is on the same device as q_head before passing through
        q_head_device = next(self.q_head.parameters()).device
        last_hidden = last_hidden.to(q_head_device)
        
        # Apply q_head and convert to float32 for consistency
        q_logits = self.q_head(last_hidden)  # [B,2] - on q_head device
        
        # Convert outputs to float32 for stability (move logits to same device as q_logits)
        logits = logits.to(q_head_device).to(torch.float32)
        q_logits = q_logits.to(torch.float32)
        q_h, q_c = q_logits[..., 0], q_logits[..., 1]

        B, S = input_ids.shape
        H = self.hidden_size
        # Produce a detached new carry; values unused by this adapter but required by segment_rollout
        z = torch.zeros((B, S, H), dtype=self.forward_dtype, device=dev)
        new_carry = _Carry(
            inner_carry=_InnerCarry(z_H=z.detach(), z_L=z.detach()),
            steps=(carry.steps + 1) if hasattr(carry, "steps") else torch.ones((B,), dtype=torch.int32, device=dev),
            halted=torch.zeros((B,), dtype=torch.bool, device=dev),
            current_data={k: v for k, v in batch.items()},
        )
        outputs = {
            "logits": logits,
            "q_halt_logits": q_h,
            "q_continue_logits": q_c,
        }
        return new_carry, outputs

    # Allow calling the adapter directly like a module: model(carry, batch)
    __call__ = forward

    # Persistence helpers for q_head weights
    def load_q_head(self, path: str) -> None:
        """Load q_head weights from disk."""
        import torch
        map_dev = (self._dml_device if self._is_dml and self._dml_device is not None else self.device)
        state = torch.load(path, map_location=map_dev)
        self.q_head.load_state_dict(state)
        if self._is_dml and self._dml_device is not None:
            self.q_head.to(self._dml_device)
        else:
            self.q_head.to(self.device)

    def save_brain(self, out_dir: str, save_model: bool = False) -> Dict[str, str]:
        """Save q_head and optionally the base LM into out_dir."""
        import torch
        out = {}
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        q_path = p / "q_head.pt"
        torch.save(self.q_head.state_dict(), str(q_path))
        out["q_head"] = str(q_path)
        # Save LoRA adapter separately to avoid mixing with q_head
        if getattr(self, "is_peft", False):
            try:
                lora_dir = p / "lora"
                lora_dir.mkdir(exist_ok=True)
                # For PEFT models, save_pretrained saves the adapter
                self.model.save_pretrained(str(lora_dir))
                out["peft_dir"] = str(lora_dir)
            except Exception:
                pass
        if save_model:
            try:
                # Save the full model or PEFT-wrapped model
                self.model.save_pretrained(str(p / "model"))
                out["model_dir"] = str(p / "model")
            except Exception:
                pass
        return out
