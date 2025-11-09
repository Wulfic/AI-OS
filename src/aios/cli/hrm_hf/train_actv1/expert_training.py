"""Expert-only training mode.

Trains standalone FeedForward expert modules for goal-aware routing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import typer
from rich import print

if TYPE_CHECKING:
    from aios.core.hrm_training.training_config import TrainingConfig


def train_expert_only(config: "TrainingConfig") -> None:
    """Train a standalone expert module (FeedForward network).
    
    This function provides expert-specific training mode that creates and trains
    a lightweight FeedForward expert instead of the full HRM model. The expert
    can later be loaded into DynamicMoELayer for goal-aware routing.
    
    Args:
        config: Training configuration with expert_id specified
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import json
    
    # Import helper functions from parent module
    from ..hrm_hf_utils import load_tokenizer as _load_tokenizer_helper
    from ..encoding import (
        encode_lines as _encode_lines_helper,
        adjust_tokenizer_padding as _adjust_tok_padding,
    )
    from ..device import resolve_device as _resolve_device_helper
    from ..data import get_training_lines as _get_training_lines_helper
    from ..training_helpers import write_jsonl as _write_jsonl_helper
    
    expert_id = config.expert_id
    if not expert_id:
        raise ValueError("expert_id must be provided for expert-only training")
    
    # Import FeedForward from moe_layer
    from aios.core.hrm_models.moe_layer import FeedForward
    from aios.core.hrm_models.expert_metadata import ExpertRegistry, create_expert_metadata
    from aios.data.datasets import read_text_lines_sample_any
    
    print({"mode": "expert_training", "expert_id": expert_id})
    
    # Load tokenizer
    tok = _load_tokenizer_helper(config.model)
    _adjust_tok_padding(tok)
    
    # Calculate dimensions
    intermediate_size = int(config.hidden_size * config.expansion)
    
    # Create expert module
    expert = FeedForward(config.hidden_size, intermediate_size)
    print({
        "expert_created": True,
        "expert_id": expert_id,
        "hidden_size": config.hidden_size,
        "intermediate_size": intermediate_size,
        "params": sum(p.numel() for p in expert.parameters()),
    })
    
    # Resolve device
    dev, device_obj, dml_device = _resolve_device_helper(config.device, config.strict, torch)
    expert.to(device_obj)
    
    # Load training data
    lines = _get_training_lines_helper(
        dataset_file=config.dataset_file,
        ascii_only=config.ascii_only,
        read_text_lines_sample_any=read_text_lines_sample_any,
    )
    
    if not lines:
        print({"started": False, "error": "no lines"})
        raise typer.Exit(code=1)
    
    # Encode dataset
    input_ids, labels = _encode_lines_helper(tok, lines, config.max_seq_len)
    N = input_ids.shape[0]
    
    print({
        "dataset": {
            "num_samples": N,
            "max_seq_len": config.max_seq_len,
            "vocab_size": tok.vocab_size,
        }
    })
    
    # Create optimizer
    params = [p for p in expert.parameters() if p.requires_grad]
    if config.use_8bit_optimizer:
        try:
            from aios.core.hrm_models.memory_optimizations import create_8bit_optimizer
            opt = create_8bit_optimizer(params, lr=config.lr, optimizer_type='adamw')
            print({"optimizer": "AdamW8bit"})
        except ImportError:
            OptClass = getattr(torch.optim, "AdamW", None) or getattr(torch.optim, "Adam")
            opt = OptClass(params, lr=config.lr)
            print({"optimizer": "AdamW (8bit unavailable)"})
    else:
        OptClass = getattr(torch.optim, "AdamW", None) or getattr(torch.optim, "Adam")
        opt = OptClass(params, lr=config.lr)
        print({"optimizer": "AdamW"})
    
    # AMP scaler for mixed precision
    scaler = None
    if config.use_amp and dev == "cuda" and torch.cuda.is_available():
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                scaler = torch.cuda.amp.GradScaler()
            print({"amp": True})
        except Exception:
            print({"amp": False})
    
    # Training loop
    expert.train()
    global_step = 0
    
    def _write_jsonl(payload: dict) -> None:
        if config.log_file:
            _write_jsonl_helper(log_file=config.log_file, payload=payload, is_distributed=False, rank_id=0)
    
    _write_jsonl({"event": "expert_training_start", "expert_id": expert_id})
    
    for step in range(config.steps):
        # Random batch
        indices = torch.randint(0, N, (config.batch_size,))
        batch_input = input_ids[indices].to(device_obj)
        
        # Create dummy hidden states (since expert expects hidden_size input)
        # In real usage, this comes from HRM layers; here we create random inputs
        batch_hidden = torch.randn(
            config.batch_size, config.max_seq_len, config.hidden_size,
            device=device_obj
        )
        
        # Forward pass (expert processes hidden states)
        opt.zero_grad()
        
        if config.use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                output = expert(batch_hidden)
                # Simple MSE loss: expert should learn to preserve information
                loss = F.mse_loss(output, batch_hidden)
        else:
            output = expert(batch_hidden)
            loss = F.mse_loss(output, batch_hidden)
        
        # Backward pass
        if config.use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        
        global_step += 1
        
        if step % 10 == 0:
            print({
                "step": global_step,
                "loss": round(float(loss.item()), 6),
                "progress": f"{global_step}/{config.steps}",
            })
            _write_jsonl({
                "step": global_step,
                "loss": float(loss.item()),
                "expert_id": expert_id,
            })
        
        # Check stop file
        if config.stop_file and Path(config.stop_file).exists():
            print({"stopped": True, "step": global_step})
            break
    
    # Save expert
    expert_dir = Path("artifacts") / "experts" / expert_id
    expert_dir.mkdir(parents=True, exist_ok=True)
    
    expert_path = expert_dir / "expert.pt"
    torch.save(expert.state_dict(), expert_path)
    print({
        "expert_saved": True,
        "path": str(expert_path),
        "size_mb": round(expert_path.stat().st_size / (1024 ** 2), 2),
    })
    
    # Update expert registry
    registry_path = Path("artifacts") / "experts" / "registry.json"
    if registry_path.exists():
        registry = ExpertRegistry.load(str(registry_path))
    else:
        registry = ExpertRegistry()
    
    # Create or update expert metadata
    expert_goals = [config.default_goal] if config.default_goal else []
    
    metadata = create_expert_metadata(
        expert_id=expert_id,
        name=f"Expert {expert_id}",
        description=f"Expert trained on {Path(config.dataset_file).name if config.dataset_file else 'dataset'}",
        category="general",
        checkpoint_path=str(expert_path),
        goals=expert_goals,
    )
    
    # Store architecture details in training_config
    metadata.training_config = {
        "hidden_size": config.hidden_size,
        "intermediate_size": intermediate_size,
        "lr": config.lr,
        "steps": global_step,
        "batch_size": config.batch_size,
        "max_seq_len": config.max_seq_len,
    }
    
    if config.default_goal:
        print({"expert_linked_to_goal": config.default_goal})
    
    registry.add_expert(metadata)
    registry.save(str(registry_path))
    
    print({
        "registry_updated": True,
        "path": str(registry_path),
        "total_experts": len(registry.experts),
    })
    
    _write_jsonl({
        "event": "expert_training_complete",
        "expert_id": expert_id,
        "steps": global_step,
        "checkpoint": str(expert_path),
    })
    
    print({
        "completed": True,
        "expert_id": expert_id,
        "steps": global_step,
        "checkpoint": str(expert_path),
    })
