"""HFHRMBrain - HRM-style brain backed by Hugging Face LM adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aios.core.brains.hf_brain_loading import load_hf_brain, get_brain_size_estimate
from aios.core.brains.hf_brain_generation import generate_response, run_minimal_rollout


@dataclass
class HFHRMBrain:
    """HRM-style brain backed by a Hugging Face LM adapter (starter brain).

    Loads a starter_brain.json (q_head, optional LoRA adapter) and exposes a minimal
    Brain interface. Training here is represented by a tiny HRM-style rollout on a
    toy batch so callers and GUI remain unchanged.
    """

    name: str
    modalities: List[str]
    starter_config: str = "artifacts/hf_starter/starter_brain.json"
    max_seq_len: Optional[int] = None  # Explicit max_seq_len override (takes precedence over config file)
    inference_device: Optional[str] = None  # Specific device for inference (e.g., "cuda:1" for multi-GPU)
    
    # Sparse MoE configuration for efficient expert routing
    use_moe: bool = True  # Enable sparse Mixture of Experts architecture by default for 75% compute reduction
    num_experts: int = 8  # Total number of expert networks (default 8 for good specialization)
    num_experts_per_tok: int = 2  # Top-k experts activated per token (sparse activation)
    moe_capacity_factor: float = 1.25  # Expert capacity factor for load balancing
    
    # Generation controls and lightweight chat memory
    system_prompt: Optional[str] = None
    history_max_turns: int = 20  # Increased from 3 to support longer conversations
    history: List[Dict[str, str]] = field(default_factory=list)
    gen_max_new_tokens: int = 2048  # Increased from 256 for much longer responses
    gen_temperature: float = 0.7  # Slightly higher for more varied output
    gen_top_p: float = 0.9
    gen_top_k: int = 50
    gen_repetition_penalty: float = 1.2  # Higher to prevent repetition loops
    max_response_chars: int = 8192  # User-configurable max response length (min 256, max based on model context)
    
    _adapter: Any = None
    _loaded_max_seq_len: int = 0  # Track actual loaded context window

    def _ensure_loaded(self) -> None:
        """Ensure HF adapter is loaded with progressive context reduction fallback."""
        load_hf_brain(self)

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle either chat generation or a minimal HRM-style rollout for liveness.

        If payload contains a user message (e.g., {"user": str} or {"text": str}),
        generate a short reply using the HF model. Otherwise, run a tiny rollout to
        produce a loss metric (used by idle/self-improvement probes).
        """
        self._ensure_loaded()
        payload = task.get("payload") if isinstance(task, dict) else None
        
        # Chat mode: generate text when a user message is present
        try:
            if isinstance(payload, str):
                user_msg = payload
            elif isinstance(payload, dict):
                user_msg = payload.get("user") or payload.get("text")
            else:
                user_msg = None
        except Exception:
            user_msg = None

        if isinstance(user_msg, str) and user_msg.strip():
            return generate_response(self, user_msg)
        
        # Non-chat path: run a minimal rollout to return a scalar loss
        return run_minimal_rollout(self, payload)

    def size_bytes(self) -> int:
        """Estimate size by small adapter head + optional LoRA files if present."""
        return get_brain_size_estimate(self.starter_config)
