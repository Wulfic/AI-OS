"""HuggingFace HRM Adapter - Public API.

Provides adapters and factory functions for integrating HuggingFace models
with the HRM (Halting Recurrent Model) interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

from .adapter import HFCausalLM_HRMAdapter
from .types import _Carry, _InnerCarry


def build_hf_adapter(
    model_name_or_path: str,
    max_seq_len: int,
    halt_max_steps: int = 1,
    halt_exploration_prob: float = 0.0,
    device: Optional[str] = None,
    forward_dtype: str = "bfloat16",
    # LoRA
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[Tuple[str, ...]] = ("c_attn", "c_proj"),
    lora_inference_mode: bool = False,
    # Sparse MoE configuration
    use_moe: bool = False,
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
    moe_capacity_factor: float = 1.25,
) -> HFCausalLM_HRMAdapter:
    """Factory for HFCausalLM_HRMAdapter.

    Returns a module exposing the HRM interface. Tokenization remains external; callers
    should prepare batch dicts with token ids under key 'inputs' and 'targets'.
    
    Args:
        model_name_or_path: HuggingFace model identifier or path
        max_seq_len: Maximum sequence length for context window
        halt_max_steps: Maximum HRM steps before halting
        halt_exploration_prob: Exploration probability for halting
        device: Target device ('auto', 'cuda', 'cpu', etc.)
        forward_dtype: Data type for forward pass ('bfloat16', 'float32', etc.)
        use_lora: Enable LoRA adapters for efficient fine-tuning
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        lora_target_modules: Module names to apply LoRA to
        lora_inference_mode: Enable LoRA inference mode
        use_moe: Enable sparse Mixture of Experts architecture
        num_experts: Total number of expert networks (ignored if use_moe=False)
        num_experts_per_tok: Top-k experts to activate per token (ignored if use_moe=False)
        moe_capacity_factor: Expert capacity factor for load balancing (ignored if use_moe=False)
    """
    return HFCausalLM_HRMAdapter(
        model_name_or_path=model_name_or_path,
        max_seq_len=max_seq_len,
        halt_max_steps=halt_max_steps,
        halt_exploration_prob=halt_exploration_prob,
        device=device,
        forward_dtype=forward_dtype,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        lora_inference_mode=lora_inference_mode,
        use_moe=use_moe,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_capacity_factor=moe_capacity_factor,
    )


def build_hf_starter_from_config(config_path: str) -> HFCausalLM_HRMAdapter:
    """Restore an HF HRM adapter from a starter-brain JSON config.

    Config schema minimal keys:
        - model: HF repo id or local dir for the model
        - q_head: path to q_head.pt
        - max_seq_len (int)
        - halt_max_steps (int)
    Optional:
        - device: 'auto'|'cpu'|'cuda'
        - forward_dtype: 'bfloat16'|'float32'|...
        - peft_dir: path to LoRA adapter
    """
    cfg_p = Path(config_path)
    data = json.loads(cfg_p.read_text(encoding="utf-8"))

    model = data["model"]
    max_seq_len = int(data.get("max_seq_len", 128))
    halt_max_steps = int(data.get("halt_max_steps", 1))
    device = data.get("device")
    forward_dtype = data.get("forward_dtype", "bfloat16")
    q_head = data.get("q_head")

    adapter = build_hf_adapter(
        model_name_or_path=model,
        max_seq_len=max_seq_len,
        halt_max_steps=halt_max_steps,
        halt_exploration_prob=0.0,
        device=device,
        forward_dtype=forward_dtype,
    )
    if q_head:
        adapter.load_q_head(q_head)
    # If LoRA adapter exists in config, attach it
    peft_dir = data.get("peft_dir")
    if peft_dir:
        try:
            from peft import PeftModel
            # Attach adapter onto the current base model
            adapter.model = PeftModel.from_pretrained(adapter.model, peft_dir)
            adapter.is_peft = True
            adapter.model.to(adapter.device)
            adapter.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load LoRA adapter from {peft_dir}: {e}")
    return adapter


__all__ = [
    "HFCausalLM_HRMAdapter",
    "build_hf_adapter",
    "build_hf_starter_from_config",
    "_Carry",
    "_InnerCarry",
]
