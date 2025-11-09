from typing import Tuple, List, Dict, cast, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from .common import trunc_normal_init_
from .layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from .sparse_embedding import CastedSparseEmbedding


@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str = "rope"

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    window_size: int | None = None  # Sliding window attention size (None = full attention)
    use_flash_attn: bool = False  # Enable Flash Attention 2 optimization (requires Ampere+ GPU)
    use_gradient_checkpointing: bool = False  # Enable gradient checkpointing for ~25% VRAM savings
    
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"
    
    # Sparse MoE configuration for efficient expert routing
    use_moe: bool = True  # Enable sparse Mixture of Experts by default
    num_experts: int = 8  # Total number of experts per MoE layer
    num_experts_per_tok: int = 2  # Top-k experts activated per token (sparse)
    moe_capacity_factor: float = 1.25  # Expert capacity factor for load balancing
    moe_load_balance_loss_coef: float = 0.01  # Load balancing loss coefficient (reduced to 0.01 for stability)


class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config, window_size: int | None = None) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
            use_flash_attn=config.use_flash_attn,  # User-controlled Flash Attention enable
            window_size=window_size  # Add sliding window support
        )
        
        # Use MoE or standard SwiGLU based on configuration
        if config.use_moe:
            from .layers import MoESwiGLU
            self.mlp = MoESwiGLU(
                hidden_size=config.hidden_size,
                expansion=config.expansion,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                capacity_factor=config.moe_capacity_factor,
            )
        else:
            self.mlp = SwiGLU(
                hidden_size=config.hidden_size,
                expansion=config.expansion,
            )
        
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block], use_gradient_checkpointing: bool = False):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        
        # Apply gradient checkpointing during training to save memory
        if self.use_gradient_checkpointing and self.training:
            import logging
            from torch.utils.checkpoint import checkpoint
            
            # DEBUG: Log that gradient checkpointing is active
            if not hasattr(self, '_gradcheck_logged'):
                logging.warning(f"[GRADCHECK] Gradient checkpointing ENABLED for {len(self.layers)} layers")
                self._gradcheck_logged = True
            
            # Extract cos_sin from kwargs for checkpointing
            cos_sin = kwargs.get('cos_sin')
            
            # Checkpoint ALL layers to maximize memory savings (~25% VRAM reduction)
            for layer in self.layers:
                # Checkpoint expects (function, *args), layer.forward expects (cos_sin, hidden_states)
                # Use reentrant=True for compatibility with MoE and dynamic routing
                hidden_states = checkpoint(layer, cos_sin, hidden_states, use_reentrant=True)
        else:
            # Standard forward pass (no checkpointing)
            for layer in self.layers:
                hidden_states = layer(hidden_states=hidden_states, **kwargs)
        
        return hidden_states


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # Use smaller embedding scale for small hidden sizes to prevent numerical issues
        # Original Transformer uses sqrt(d_model), but for small d_model this can cause instability
        self.embed_scale = math.sqrt(self.config.hidden_size) if self.config.hidden_size >= 512 else 1.0
        embed_init_std = 1.0 / math.sqrt(self.config.hidden_size)

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            raise NotImplementedError()

        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[HierarchicalReasoningModel_ACTV1Block(self.config, window_size=self.config.window_size) for _i in range(self.config.H_layers)],
            use_gradient_checkpointing=self.config.use_gradient_checkpointing
        )
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[HierarchicalReasoningModel_ACTV1Block(self.config, window_size=self.config.window_size) for _i in range(self.config.L_layers)],
            use_gradient_checkpointing=self.config.use_gradient_checkpointing
        )
        
        # Log MoE configuration if enabled
        if self.config.use_moe:
            total_moe_layers = self.config.H_layers + self.config.L_layers
            print(f"[ACTV1] Sparse MoE enabled: {total_moe_layers} MoE layers, {self.config.num_experts} experts each, {self.config.num_experts_per_tok} active per token")
            print(f"[ACTV1] Expected compute reduction: ~{(1 - self.config.num_experts_per_tok / self.config.num_experts) * 100:.1f}%")
        
        # vendor used nn.Buffer; use register_buffer instead
        self.register_buffer("H_init", trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.register_buffer("L_init", trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        with torch.no_grad():
            # Use smaller initialization for q_head to prevent gradient explosion
            self.q_head.weight.normal_(mean=0.0, std=0.001)  # Very small weights
            if self.q_head.bias is not None:
                self.q_head.bias.fill_(-2)  # Less extreme bias to prevent saturation
            
            # Initialize lm_head more conservatively for stability
            # Use smaller std for low hidden dimensions to prevent extreme logits
            lm_head_std = min(0.02, 1.0 / math.sqrt(self.config.hidden_size))
            self.lm_head.weight.normal_(mean=0.0, std=lm_head_std)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Apply embedding scale and clip to prevent extreme values
        scaled_embedding = self.embed_scale * embedding
        # Clip embeddings to reasonable range to prevent NaN propagation
        return torch.clamp(scaled_embedding, min=-10.0, max=10.0)

    def empty_carry(self, batch_size: int, seq_len: Optional[int] = None):
        """Create empty carry state. If seq_len not provided, uses config.seq_len."""
        # Default to module's device (moved with .to(device))
        dev = next(self.parameters()).device
        actual_seq_len = seq_len if seq_len is not None else self.config.seq_len
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(
                batch_size,
                actual_seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=dev,
            ),
            z_L=torch.empty(
                batch_size,
                actual_seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=dev,
            ),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV1InnerCarry):
        # Ensure the boolean condition lives on the same device as carry tensors
        rf = reset_flag.to(carry.z_H.device)
        cond = rf.view(-1, 1, 1)
        # Broadcast initial states to match [B, S, H] without calling expand/repeat
        H0 = torch.reshape(cast(torch.Tensor, self.H_init), (1, 1, -1))
        L0 = torch.reshape(cast(torch.Tensor, self.L_init), (1, 1, -1))
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(cond, H0, carry.z_H),
            z_L=torch.where(cond, L0, carry.z_L),
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]):
        # Get actual sequence length from batch
        actual_seq_len = batch["inputs"].shape[1]
        
        # Slice rotary embeddings to match actual sequence length
        if hasattr(self, "rotary_emb"):
            cos_full, sin_full = self.rotary_emb()
            cos_sin = (cos_full[:actual_seq_len], sin_full[:actual_seq_len])
        else:
            cos_sin = None
        
        seq_info = dict(cos_sin=cos_sin)

        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        
        # Safety check: catch NaN in embeddings early
        if torch.isnan(input_embeddings).any() or torch.isinf(input_embeddings).any():
            print(f"[ACTV1 WARNING] NaN/Inf detected in input embeddings! Setting to zero.")
            input_embeddings = torch.nan_to_num(input_embeddings, nan=0.0, posinf=10.0, neginf=-10.0)

        # Start from carry tensors (they come from previous step without grad)
        z_H, z_L = carry.z_H, carry.z_L

        # Reset L at the start of each H cycle per paper (new computational phase)
        for _H_step in range(self.config.H_cycles):
            # Reset L to start a new computational phase; broadcast to [B, S, H]
            L0 = torch.reshape(cast(torch.Tensor, self.L_init), (1, 1, -1))
            z_L = torch.ones_like(z_H) * L0
            for _L_step in range(self.config.L_cycles):
                if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

            if not (_H_step == self.config.H_cycles - 1):
                z_H = self.H_level(z_H, z_L, **seq_info)
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # Keep computational graph for outputs; carry can be detached to avoid graph blow-up across outer segments
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        
        # Generate logits and clip to prevent numerical instability
        raw_output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        
        # Debug: Check for NaN/Inf in raw logits
        if torch.isnan(raw_output).any() or torch.isinf(raw_output).any():
            print(f"[ACTV1 CRITICAL] NaN/Inf in lm_head output! Stats: min={raw_output.min():.2f}, max={raw_output.max():.2f}, mean={raw_output.mean():.2f}")
            print(f"[ACTV1 CRITICAL] z_H stats: min={z_H.min():.2f}, max={z_H.max():.2f}, mean={z_H.mean():.2f}")
            # Replace NaN/Inf with safe values
            raw_output = torch.nan_to_num(raw_output, nan=0.0, posinf=20.0, neginf=-20.0)
        
        # Clip logits to prevent overflow in cross-entropy (exp overflow at ~88)
        # Cast to FP32 for numerical stability with loss calculation (critical for AMP)
        output = torch.clamp(raw_output, min=-20.0, max=20.0).to(torch.float32)

        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV1(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        seq_len = batch["inputs"].shape[1]  # Get actual sequence length from batch
        device = batch["inputs"].device

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size, seq_len=seq_len),
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=device),
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]):
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                halted = halted | (q_halt_logits > q_continue_logits)

                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)

                halted = halted & (new_steps >= min_halt_steps)

                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
                
                outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return HierarchicalReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
