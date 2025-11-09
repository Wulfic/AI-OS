from __future__ import annotations

"""
Hugging Face causal LM → HRM adapter.

This wraps a pretrained causal LM so it presents the HRM interface expected by
segment_rollout: model.initial_carry(batch) and model(carry, batch) -> (carry, outputs)
with outputs containing:
  - logits: [B, S, V]
  - q_halt_logits: [B]
  - q_continue_logits: [B]

Imports of heavy dependencies (torch, transformers) are guarded inside functions
to keep baseline installs lightweight.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import json
from pathlib import Path


def _require_torch_transformers():  # pragma: no cover - optional dependency
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for HF adapter but not installed") from e
    try:
        import transformers  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("transformers is required for HF adapter but not installed") from e


@dataclass
class _InnerCarry:
    z_H: Any
    z_L: Any


@dataclass
class _Carry:
    inner_carry: _InnerCarry
    steps: Any
    halted: Any
    current_data: Dict[str, Any]


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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # Ensure pad token exists for batching; fall back to eos if missing
        if self.tokenizer.pad_token_id is None and hasattr(self.tokenizer, "eos_token_id"):
            try:
                self.tokenizer.pad_token = self.tokenizer.eos_token  # type: ignore[assignment]
            except Exception:
                pass

        # Decide device and dtype upfront (prefer CUDA > XPU > MPS > DirectML > CPU)
        self._is_dml = False
        self._dml_device = None
        req_dev = (device or "auto").strip().lower()
        def _pick_device() -> Any:
            # explicit selection first
            if req_dev in ("cpu", "cuda", "xpu", "mps"):
                try:
                    return torch.device(req_dev)
                except Exception:
                    return torch.device("cpu")
            if req_dev == "dml":
                try:
                    import torch_directml as _dml  # type: ignore
                    return _dml.device()
                except Exception:
                    return torch.device("cpu")
            # auto detection
            try:
                if torch.cuda.is_available():
                    return torch.device("cuda")
            except Exception:
                pass
            try:
                if getattr(torch, "xpu", None) and torch.xpu.is_available():  # type: ignore[attr-defined]
                    return torch.device("xpu")
            except Exception:
                pass
            try:
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
                    return torch.device("mps")
            except Exception:
                pass
            try:
                import torch_directml as _dml  # type: ignore
                return _dml.device()
            except Exception:
                pass
            return torch.device("cpu")

        picked = _pick_device()
        # Normalize fields
        if str(picked).lower().startswith("<torch_directml.") or "DirectML" in str(type(picked)):
            # torch_directml device object
            self._is_dml = True
            self._dml_device = picked
            # set a placeholder torch device for checks
            self.device = torch.device("cpu")
        else:
            self.device = picked  # type: ignore[assignment]

        chosen_dtype = getattr(torch, forward_dtype, torch.bfloat16)
        # Avoid BF16 on CPU/DML/MPS by default
        try:
            if (not hasattr(self, "device")) or (getattr(self.device, "type", "cpu") == "cpu") or self._is_dml or (getattr(self.device, "type", "") == "mps"):
                chosen_dtype = torch.float32
        except Exception:
            chosen_dtype = torch.float32

        # Set HF_HOME to suppress deprecated TRANSFORMERS_CACHE warning
        import os
        if "TRANSFORMERS_CACHE" in os.environ and "HF_HOME" not in os.environ:
            os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]

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
        # Move model to target device BEFORE MoE replacement to ensure all components are on the correct device
        if self._is_dml and self._dml_device is not None:
            self.model.to(self._dml_device)
        else:
            self.model.to(self.device)  # type: ignore[arg-type]
        self.model.eval()
        
        # Apply Sparse MoE if enabled: replace FFN layers with MoE layers
        if self.use_moe and self.num_experts > 1:
            self._replace_ffn_with_moe()

        self.max_seq_len = int(max(1, max_seq_len))
        self.halt_max_steps = int(max(1, halt_max_steps))
        self.halt_exploration_prob = float(max(0.0, min(1.0, halt_exploration_prob)))
        # hidden_size already set earlier after base_model loading

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

        # Expose a lightweight training flag and API expected by segment_rollout
        # (without requiring nn.Module as a superclass at import time).
        self.training = True

    def _replace_ffn_with_moe(self) -> None:
        """Replace feed-forward layers with sparse MoE layers for efficiency.
        
        This method identifies FFN/MLP blocks in the transformer and replaces them
        with DynamicMoELayer instances for sparse expert routing.
        """
        try:
            from aios.core.hrm_models.dynamic_moe import DynamicMoELayer
            import torch.nn as nn
        except ImportError as e:
            raise RuntimeError(f"MoE support requires dynamic_moe module: {e}") from e
        
        # Define wrapper class once
        class MoEWrapper(nn.Module):
            """Wrapper to make MoE layer compatible with original MLP interface."""
            def __init__(self, moe: DynamicMoELayer):
                super().__init__()
                self.moe = moe
            
            def forward(self, x):
                output, _ = self.moe(x)
                return output
        
        # Determine device for MoE layers
        dev = self._dml_device if self._is_dml and self._dml_device is not None else self.device
        
        # Get the underlying base model (unwrap PEFT if present)
        base_model = self.model.base_model if self.is_peft else self.model
        
        # Common transformer architectures have layers at different paths
        # Try to identify and replace FFN/MLP modules
        replaced_count = 0
        
        # Try GPT-2 style (model.transformer.h[i].mlp)
        if hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'h'):
            for layer_idx, layer in enumerate(base_model.transformer.h):  # type: ignore
                if hasattr(layer, 'mlp'):
                    # Get intermediate size from existing MLP
                    try:
                        # GPT-2 uses c_fc (input projection) to determine intermediate size
                        if hasattr(layer.mlp, 'c_fc'):
                            intermediate_size = layer.mlp.c_fc.out_features
                        else:
                            # Fallback: 4x hidden size (standard transformer FFN)
                            intermediate_size = self.hidden_size * 4
                    except Exception:
                        intermediate_size = self.hidden_size * 4
                    
                    # Create MoE layer
                    moe_layer = DynamicMoELayer(
                        hidden_size=self.hidden_size,
                        intermediate_size=intermediate_size,
                        num_experts_per_tok=self.num_experts_per_tok,
                        capacity_factor=self.moe_capacity_factor,
                        lazy_loading=False,  # Keep experts in memory for now
                        device=str(dev),
                    )
                    
                    # Add default experts
                    for expert_idx in range(self.num_experts):
                        from aios.core.hrm_models.expert_metadata import create_expert_metadata
                        metadata = create_expert_metadata(
                            expert_id=f"layer{layer_idx}_expert{expert_idx}",
                            name=f"Layer {layer_idx} Expert {expert_idx}",
                            description=f"Auto-created MoE expert for layer {layer_idx}",
                            category="general",
                        )
                        moe_layer.add_expert(f"layer{layer_idx}_expert{expert_idx}", metadata=metadata)
                    
                    # Convert all experts to match the model's dtype (critical for bfloat16/fp16 models)
                    model_dtype = next(base_model.parameters()).dtype
                    for expert_id in list(moe_layer.experts.keys()):
                        moe_layer.experts[expert_id] = moe_layer.experts[expert_id].to(dtype=model_dtype)
                    
                    # Also convert the router to match model dtype
                    moe_layer.router = moe_layer.router.to(dtype=model_dtype)
                    
                    # Replace the MLP with wrapped MoE
                    layer.mlp = MoEWrapper(moe_layer)
                    replaced_count += 1
        
        # Try LLaMA style (model.model.layers[i].mlp)
        elif hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
            for layer_idx, layer in enumerate(base_model.model.layers):  # type: ignore
                if hasattr(layer, 'mlp'):
                    # Get intermediate size
                    try:
                        if hasattr(layer.mlp, 'gate_proj'):
                            intermediate_size = layer.mlp.gate_proj.out_features
                        elif hasattr(layer.mlp, 'up_proj'):
                            intermediate_size = layer.mlp.up_proj.out_features
                        else:
                            intermediate_size = self.hidden_size * 4
                    except Exception:
                        intermediate_size = self.hidden_size * 4
                    
                    moe_layer = DynamicMoELayer(
                        hidden_size=self.hidden_size,
                        intermediate_size=intermediate_size,
                        num_experts_per_tok=self.num_experts_per_tok,
                        capacity_factor=self.moe_capacity_factor,
                        lazy_loading=False,
                        device=str(dev),
                    )
                    
                    for expert_idx in range(self.num_experts):
                        from aios.core.hrm_models.expert_metadata import create_expert_metadata
                        metadata = create_expert_metadata(
                            expert_id=f"layer{layer_idx}_expert{expert_idx}",
                            name=f"Layer {layer_idx} Expert {expert_idx}",
                            description=f"Auto-created MoE expert for layer {layer_idx}",
                            category="general",
                        )
                        moe_layer.add_expert(f"layer{layer_idx}_expert{expert_idx}", metadata=metadata)
                    
                    # Convert all experts to match the model's dtype (critical for bfloat16/fp16 models)
                    model_dtype = next(base_model.parameters()).dtype
                    for expert_id in list(moe_layer.experts.keys()):
                        moe_layer.experts[expert_id] = moe_layer.experts[expert_id].to(dtype=model_dtype)
                    
                    # Also convert the router to match model dtype
                    moe_layer.router = moe_layer.router.to(dtype=model_dtype)
                    
                    layer.mlp = MoEWrapper(moe_layer)
                    replaced_count += 1
        
        if replaced_count > 0:
            print(f"[HFAdapter] Replaced {replaced_count} FFN layers with sparse MoE ({self.num_experts} experts, top-{self.num_experts_per_tok} routing)")
        else:
            print(f"[HFAdapter] Warning: MoE enabled but no FFN layers found to replace in {type(base_model).__name__}")

    # Minimal Module-like API expected by training utilities
    def train(self, mode: bool = True) -> None:
        """Set training flag on the adapter; does not change base LM unless caller does so."""
        self.training = bool(mode)

    def eval(self) -> None:
        """Set evaluation mode on the adapter; base LM eval is managed separately by caller."""
        self.training = False

    # HRM API
    def initial_carry(self, batch: Dict[str, Any]) -> _Carry:
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
        # This prevents "mat2 is on cpu, different from other tensors on cuda" errors
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
                # Save the full model or PEFT-wrapped model (PEFT will save adapter files)
                self.model.save_pretrained(str(p / "model"))
                out["model_dir"] = str(p / "model")
            except Exception:
                pass
        return out


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
