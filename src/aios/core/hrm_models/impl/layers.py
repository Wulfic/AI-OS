from typing import Dict, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F

from .common import trunc_normal_init_

CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Compute in the activation dtype to leverage AMP/bfloat16 benefits.
        # Cast parameters to input dtype on-the-fly (creates ephemeral views) while
        # keeping parameters stored in their original dtype for stability.
        input_dtype = input.dtype
        w = self.weight.to(input_dtype)
        b = self.bias.to(input_dtype) if self.bias is not None else None
        return F.linear(input, w, b)


class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        # Initialize parameter in the target dtype so we don't need to cast it in forward
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim), dtype=self.cast_to), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Use parameter directly to keep it attached to autograd graph
        return F.embedding(input, self.embedding_weight)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        # Avoid nn.Buffer (vendor-specific). Use registered buffers.
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False, use_flash_attn=False, window_size=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.use_flash_attn = use_flash_attn  # User-controlled Flash Attention enable
        self.window_size = window_size  # Sliding window size (None = full attention)
        
        # Cache for attention masks to avoid recomputation
        self._mask_cache: Dict[Tuple[int, str, str], torch.Tensor] = {}

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def _create_sliding_window_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Create attention mask for sliding window + causal attention.
        
        Returns a mask of shape (seq_len, seq_len) where:
        - True = attention allowed
        - False = attention blocked
        
        Uses caching to avoid recomputing masks for common sequence lengths.
        """
        # Check cache first
        cache_key = (seq_len, str(device), str(dtype))
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]
        
        # Start with causal mask if needed
        if self.causal:
            # Causal mask: can attend to current and previous positions
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        else:
            # No causal constraint: attend to all positions
            mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        
        if self.window_size is not None and self.window_size > 0:
            # Apply sliding window constraint
            # Each position can only attend to positions within window_size distance
            # OPTIMIZATION: Skip if window_size >= seq_len (full attention anyway)
            if self.window_size < seq_len:
                row_idx = torch.arange(seq_len, device=device).view(-1, 1)
                col_idx = torch.arange(seq_len, device=device).view(1, -1)
                
                # Distance from current position
                distance = row_idx - col_idx
                
                # Allow attention only within window (current position and window_size previous positions)
                window_mask = (distance >= 0) & (distance < self.window_size)
                
                # Combine with existing mask
                mask = mask & window_mask
            # else: window_size >= seq_len, so full attention (no masking needed)
        
        # Convert to attention mask format (True -> 0.0, False -> -inf)
        attn_mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        attn_mask.masked_fill_(~mask, float('-inf'))
        
        # Cache the mask (limit cache size to prevent memory issues)
        if len(self._mask_cache) < 100:  # Max 100 cached masks
            self._mask_cache[cache_key] = attn_mask
        
        return attn_mask

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Preferred path: use memory-efficient kernels if available (SDPA/Flash)
        use_optimized = False
        if self.use_flash_attn:
            # Priority: FlashAttention-2 > FlashAttention-1 > PyTorch SDPA
            
            # Try FlashAttention-2 first (best performance, native sliding window support)
            try:
                try:
                    from flash_attn_interface import flash_attn_func  # type: ignore
                except ImportError:
                    from flash_attn import flash_attn_func  # type: ignore
                
                # FlashAttention-2 supports window_size parameter for efficient sliding window
                if self.window_size is not None and self.window_size > 0:
                    # window_size format: (left, right) - for causal it's (window_size-1, 0)
                    window = (self.window_size - 1, 0) if self.causal else (self.window_size // 2, self.window_size // 2)
                    attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal, window_size=window)
                else:
                    attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)
                
                if isinstance(attn_output, tuple):
                    attn_output = attn_output[0]
                use_optimized = True
            except Exception as e:
                # FlashAttention-2 not available, try FlashAttention-1
                import logging
                try:
                    from flash_attn.flash_attention import FlashAttention  # type: ignore
                    
                    # FlashAttention-1 requires different format: [B, S, H, D]
                    # Query, key, value are already in correct format from qkv projection
                    flash_attn_1 = FlashAttention(softmax_scale=1.0/math.sqrt(self.head_dim), attention_dropout=0.0)
                    
                    # FlashAttention-1 doesn't natively support sliding window, but works for causal
                    if self.window_size is None or self.window_size == 0:
                        attn_output, _ = flash_attn_1(query, key, value, causal=self.causal)
                        use_optimized = True
                        logging.info(
                            f"Using FlashAttention-1 (FlashAttention-2 unavailable: {type(e).__name__}). "
                            f"For best performance, upgrade: pip install flash-attn --no-build-isolation"
                        )
                    else:
                        # FlashAttention-1 doesn't support sliding window, fall through to SDPA
                        logging.warning(
                            f"FlashAttention-1 doesn't support sliding window (window_size={self.window_size}). "
                            f"Falling back to PyTorch SDPA. Install FlashAttention-2 for sliding window support."
                        )
                except Exception as e2:
                    # FlashAttention-1 also not available, fall back to PyTorch SDPA
                    logging.warning(
                        f"Flash Attention requested but unavailable (FA2: {type(e).__name__}, FA1: {type(e2).__name__}). "
                        f"Falling back to PyTorch SDPA (minimal memory savings). "
                        f"Install flash-attn for better performance: pip install flash-attn --no-build-isolation"
                    )
                    use_optimized = False
        
            if not use_optimized:
                # Fallback to PyTorch's scaled_dot_product_attention (still optimized)
                # Reorder to [B, H, S, D]
                q = query.permute(0, 2, 1, 3)
                k = key.permute(0, 2, 1, 3)
                v = value.permute(0, 2, 1, 3)

                # PyTorch SDPA supports sliding window via explicit mask
                if self.window_size is not None and self.window_size > 0:
                    attn_mask = self._create_sliding_window_mask(seq_len, hidden_states.device, hidden_states.dtype)
                    attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
                else:
                    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)

                # Back to [B, S, H*D]
                attn_output = attn_out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.output_size)
        else:
            # Even when FlashAttention is disabled, use SDPA by default for memory efficiency.
            # Only fall back to a manual implementation if SDPA is unavailable (older PyTorch).
            try:
                q = query.permute(0, 2, 1, 3)  # [B, H, S, D]
                k = key.permute(0, 2, 1, 3)
                v = value.permute(0, 2, 1, 3)

                if self.window_size is not None and self.window_size > 0:
                    attn_mask = self._create_sliding_window_mask(seq_len, hidden_states.device, hidden_states.dtype)
                    attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
                else:
                    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)

                attn_output = attn_out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.output_size)
            except Exception:
                # Ultimate fallback: manual attention (higher memory use)
                q = query.permute(0, 2, 1, 3)  # [B, H, S, D]
                k = key.permute(0, 2, 1, 3)
                v = value.permute(0, 2, 1, 3)

                attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                if self.causal:
                    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool), diagonal=1)
                    attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
                if self.window_size is not None and self.window_size > 0:
                    attn_mask = self._create_sliding_window_mask(seq_len, hidden_states.device, hidden_states.dtype)
                    attn_scores = attn_scores.masked_fill(attn_mask == float('-inf'), float('-inf'))
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_out = torch.matmul(attn_weights, v)
                attn_output = attn_out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.output_size)

        attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class MoESwiGLU(nn.Module):
    """SwiGLU-compatible Mixture of Experts layer for ACTV1 architecture.
    
    Provides the same interface as SwiGLU but uses sparse expert routing
    for 75% compute reduction while maintaining model quality.
    
    NOTE: Uses static MoE for inference - dynamic expert addition is not yet implemented.
    """
    
    def __init__(
        self,
        hidden_size: int,
        expansion: float,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        # Calculate intermediate size matching SwiGLU
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        
        # Use static MoELayer instead of DynamicMoELayer for inference
        from aios.core.hrm_models.moe_layer import MoELayer
        
        self.moe = MoELayer(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            intermediate_size=inter,
            capacity_factor=capacity_factor,
            dropout=0.0,  # No dropout during inference
        )
        
        self.hidden_size = hidden_size
        self.inter = inter
    
    def forward(self, x):
        """Forward pass matching SwiGLU interface.
        
        Args:
            x: Input tensor [batch, seq, hidden]
            
        Returns:
            Output tensor [batch, seq, hidden]
        """
        # Check input for NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[MoESwiGLU] WARNING: NaN/Inf detected in input, shape={x.shape}")
            # Replace NaN/Inf with zeros to prevent propagation
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # MoELayer returns (output, router_logits)
        output, router_logits = self.moe(x)
        
        # Check output for NaN/Inf
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"[MoESwiGLU] WARNING: NaN/Inf detected in output, shape={output.shape}")
            output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # Store router logits for auxiliary load balancing loss (optional)
        self.last_router_logits = router_logits
        
        return output


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)
