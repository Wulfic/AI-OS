"""HF brain loading utilities - progressive context reduction and adapter initialization."""

from __future__ import annotations

import json
import os
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aios.core.brains.hf_brain import HFHRMBrain


def load_hf_brain(brain: "HFHRMBrain") -> None:
    """Load HF adapter with progressive context reduction fallback.
    
    Args:
        brain: HFHRMBrain instance to load
        
    Raises:
        RuntimeError: If all context size attempts fail
    """
    if brain._adapter is not None:
        return
    
    # Try loading from starter config with progressive context reduction
    model_path = None
    
    # Use explicit max_seq_len if provided, otherwise read from config
    if brain.max_seq_len is not None:
        base_max_seq_len = max(1024, brain.max_seq_len)
        print(f"[Brain] Using explicit context length: {base_max_seq_len} tokens")
    else:
        base_max_seq_len = 2048  # Default to 2k tokens (matches config default)
        
        # Read model path and max_seq_len from config
        try:
            with open(brain.starter_config, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
                mp = str(data.get("model") or "").strip()
                model_path = mp or None
                # Also read max_seq_len from config if present
                try:
                    config_max_seq = int(data.get("max_seq_len", 2048))
                    base_max_seq_len = max(1024, config_max_seq)  # Ensure at least 1k
                    print(f"[Brain] Using context length from config: {base_max_seq_len} tokens")
                except Exception:
                    pass
        except Exception:
            model_path = None
    
    from aios.core.hrm_models import build_hf_adapter  # type: ignore
    
    # Progressive context window reduction starting from base_max_seq_len
    max_seq_attempts = []
    current = base_max_seq_len
    while current >= 1024:
        max_seq_attempts.append(current)
        current -= 1024  # Reduce by 1k per attempt
    
    # Ensure we have at least one attempt
    if not max_seq_attempts:
        max_seq_attempts = [2048, 1024]
    
    last_error = None
    
    for max_seq_len in max_seq_attempts:
        try:
            brain._adapter = build_hf_adapter(
                model_name_or_path=model_path or "gpt2",  # Use valid HF model as fallback
                max_seq_len=max_seq_len,
                halt_max_steps=1,
                halt_exploration_prob=0.0,
                device=brain.inference_device,  # Use specific device if provided for multi-GPU support
                forward_dtype="bfloat16",
                use_lora=False,
                # Sparse MoE configuration for efficient inference
                use_moe=brain.use_moe,
                num_experts=brain.num_experts if brain.use_moe else 1,
                num_experts_per_tok=brain.num_experts_per_tok if brain.use_moe else 1,
                moe_capacity_factor=brain.moe_capacity_factor if brain.use_moe else 1.0,
            )
            brain._adapter.train(True)
            # Success! Store the loaded context window and use 50% for generation (no hard cap)
            brain._loaded_max_seq_len = max_seq_len
            # Use up to 75% of context window for generation, respecting max_response_chars setting
            default_gen_tokens = (max_seq_len * 3) // 4  # 75% of context for generation
            # If max_response_chars is set (not default 8192), use it to calculate token limit
            if brain.max_response_chars != 8192:  # User has customized it
                tokens_from_chars = brain.max_response_chars // 4  # Estimate tokens from chars
                brain.gen_max_new_tokens = min(default_gen_tokens, max(256, tokens_from_chars))
            else:
                # Use default 75% of context for longer responses
                brain.gen_max_new_tokens = default_gen_tokens
            # Update max_response_chars based on actual token limit
            brain.max_response_chars = brain.gen_max_new_tokens * 4
            print(f"[Brain] Loaded with context window: {max_seq_len} tokens (max generation: {brain.gen_max_new_tokens} tokens, ~{brain.max_response_chars} chars)")
            return
        except Exception as e:
            last_error = e
            continue  # Try next smaller context size
    
    # All attempts failed - provide helpful guidance
    raise RuntimeError(
        f"Failed to load brain with any context size (tried {len(max_seq_attempts)} sizes from {base_max_seq_len} to 1024 tokens).\n"
        f"Last error: {last_error}\n"
        f"Suggestion: The model may not support the requested context length. Try reducing the max_seq_len in your brain configuration."
    )


def get_brain_size_estimate(starter_config: str) -> int:
    """Estimate brain size from starter config directory.
    
    Args:
        starter_config: Path to starter_brain.json
        
    Returns:
        Estimated size in bytes
    """
    base = 512 * 1024  # ~0.5 MB for q_head
    try:
        # If starter dir has lora/, include its on-disk size
        from pathlib import Path
        p = Path(starter_config).parent / "lora"
        if p.exists():
            total = 0
            for root, _, files in os.walk(p):
                for f in files:
                    total += os.path.getsize(os.path.join(root, f))
            return int(base + total)
    except Exception:
        pass
    return int(base)
