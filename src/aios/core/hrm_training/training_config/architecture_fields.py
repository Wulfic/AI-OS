"""Model architecture configuration fields.

HRM ACTv1 architecture parameters and MoE settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ArchitectureFields:
    """Model architecture parameters for HRM ACTv1."""
    
    # ============================================================================
    # Model Architecture
    # ============================================================================
    h_layers: int = 2
    """Number of High-level layers in the HRM architecture."""
    
    l_layers: int = 2
    """Number of Low-level layers in the HRM architecture."""
    
    hidden_size: int = 512
    """Model hidden dimension.
    
    Examples:
    - 512 → ~50M params
    - 768 → ~87M params
    - 1024 → ~150M params
    - 1536 → ~230M params
    - 2048 → ~378M params
    """
    
    expansion: float = 2.0
    """FFN (Feed-Forward Network) expansion factor.
    
    Intermediate dimension = hidden_size * expansion.
    Typical values: 2.0 to 4.0.
    """
    
    num_heads: int = 8
    """Number of attention heads.
    
    Must divide hidden_size evenly (e.g., 768/8 = 96 per head).
    """
    
    h_cycles: int = 2
    """High-level recurrent cycles per segment."""
    
    l_cycles: int = 2
    """Low-level recurrent cycles per segment."""
    
    pos_encodings: str = "rope"
    """Position encodings type.
    
    Options:
    - "rope": Rotary Position Embeddings (recommended, handles long contexts)
    - "learned": Learned absolute position embeddings
    """
    
    use_flash_attn: bool = False
    """Enable Flash Attention 2 for optimized attention computation.
    
    Flash Attention 2 is a memory-efficient attention algorithm that:
    - Reduces memory from O(n²) to O(n)
    - Faster than standard attention (~20-30% speedup)
    - Enables longer contexts without OOM
    
    Requirements:
    - CUDA GPU with Ampere architecture or newer (RTX 30-series+)
    - flash-attn package installed
    
    If requirements not met, automatically falls back to PyTorch SDPA.
    
    Note: This is separate from window_size. Flash Attention can be used
    with or without sliding windows.
    
    Default: False (user must explicitly enable)
    """
    
    window_size: Optional[int] = None
    """Sliding window attention size (None = full attention).
    
    Limits attention to a local window for memory efficiency.
    Converts O(n²) attention to O(n×window_size).
    
    This is INDEPENDENT of use_flash_attn:
    - use_flash_attn controls the attention ALGORITHM
    - window_size controls the attention RANGE
    
    You can use window_size with or without Flash Attention.
    Flash Attention makes sliding windows MORE efficient.
    
    Recommended values for extreme contexts:
    - 256: Minimal memory, ~6GB for 100K tokens
    - 512: Balanced, ~12GB for 100K tokens  
    - 1024: High quality, ~24GB for 100K tokens
    - 2048: Near-full attention, ~48GB for 100K tokens
    - None: Full attention (default)
    
    WARNING: Smaller windows may reduce model quality.
    Test quality after enabling to ensure acceptable performance.
    """
    
    # ============================================================================
    # Sparse Mixture of Experts (MoE) Options
    # ============================================================================
    use_moe: bool = True
    """Enable sparse Mixture of Experts architecture for efficient inference.
    
    Replaces standard FFN layers with MoE layers that activate only a subset of
    expert networks per token. Provides significant compute savings (~75%) with
    minimal quality loss.
    
    Benefits:
    - ~75% reduction in FFN compute (2/8 experts active by default)
    - Maintains model quality through specialized experts
    - Scales model capacity without proportional compute increase
    - Each block gets its own set of experts for fine-grained specialization
    
    Architecture impact:
    - Each ACTV1 block (H and L layers) gets independent MoE layer
    - Total MoE layers = h_layers + l_layers (e.g., 6+6 = 12 MoE layers)
    - Each MoE layer has num_experts total, num_experts_per_tok active
    
    Memory trade-off:
    - More parameters (8 experts vs 1 FFN per layer)
    - But compute remains same as ~2 FFN layers (sparse activation)
    
    Compatible with:
    - All optimization features (gradient checkpointing, AMP, etc.)
    - PEFT/LoRA (adapters can target expert parameters)
    - Chunked training and extreme contexts
    
    Default: True (enabled for efficiency)
    """
    
    num_experts: int = 8
    """Number of expert networks per MoE layer.
    
    Each MoE layer contains this many expert FFN networks. Higher counts allow
    more specialization but increase parameter count proportionally.
    
    Typical values:
    - 4: Minimal (less specialization)
    - 8: Balanced (recommended, good specialization vs params)
    - 16: High specialization (2x parameters)
    - 32: Very high (4x parameters, diminishing returns)
    
    Memory impact (per MoE layer, hidden_size=512, expansion=2.0):
    - 4 experts: ~4M params
    - 8 experts: ~8M params (default)
    - 16 experts: ~16M params
    
    Note: Total model size scales with num_experts, but compute cost depends
    only on num_experts_per_tok (sparse activation).
    
    Default: 8 experts (good balance)
    """
    
    num_experts_per_tok: int = 2
    """Number of experts activated per token (top-k routing).
    
    The router selects the top-k most relevant experts for each token, activating
    only those experts. Lower values save more compute but may reduce quality.
    
    Compute reduction = 1 - (num_experts_per_tok / num_experts)
    
    Examples (with num_experts=8):
    - k=1: 87.5% compute reduction (very sparse, may hurt quality)
    - k=2: 75% compute reduction (recommended, good balance)
    - k=4: 50% compute reduction (less sparse)
    - k=8: 0% compute reduction (defeats purpose of MoE)
    
    Quality considerations:
    - k=1 works well for simple, repetitive tasks
    - k=2 (default) balances efficiency and quality for most tasks
    - k>=4 for very complex tasks requiring multiple perspectives
    
    Note: Must be <= num_experts. Typical ratio is k = num_experts / 4.
    
    Default: 2 (75% compute reduction)
    """
    
    moe_capacity_factor: float = 1.25
    """Expert capacity factor for load balancing in MoE layers.
    
    Controls how many tokens each expert can process in a batch. Higher values
    allow better load balancing when some experts are very popular, but use
    more memory.
    
    Capacity = (batch_tokens / num_experts) * capacity_factor
    
    Typical values:
    - 1.0: Strict capacity (may drop tokens if expert overloaded)
    - 1.25: Balanced (default, allows 25% overflow)
    - 1.5: Generous (allows 50% overflow, better load balancing)
    - 2.0: Very generous (2x capacity, rarely needed)
    
    Trade-offs:
    - Higher factor → better load balancing, less token dropping
    - Higher factor → more memory usage
    - Lower factor → memory efficient, but may drop tokens
    
    Note: Token dropping means some tokens don't get processed by any expert
    (fall back to residual connection), which can slightly hurt quality.
    
    Default: 1.25 (good balance)
    """
