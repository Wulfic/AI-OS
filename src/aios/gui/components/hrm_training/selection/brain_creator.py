"""Brain creation logic for initializing new HRM student models.

This module handles creating brain directories and writing brain.json metadata files.
"""

from __future__ import annotations
import logging
import os
import json
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def create_brain_directory(
    project_root: str,
    brain_name: str,
    tokenizer_id: str,
    tokenizer_model: str,
    vocab_size: int,
    default_goal: str = "",
    use_moe: bool = False,
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
    architecture: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a new brain directory with brain.json metadata.
    
    Args:
        project_root: Project root path
        brain_name: Name for the brain (used as directory name)
        tokenizer_id: Tokenizer registry ID (e.g., "gpt2-base-model")
        tokenizer_model: Path to tokenizer model
        vocab_size: Tokenizer vocabulary size
        default_goal: Default training goal/directive
        use_moe: Whether using sparse MoE
        num_experts: Number of MoE experts (if MoE enabled)
        num_experts_per_tok: Active experts per token (if MoE enabled)
        architecture: Optional dict with architecture params (hidden_size, h_layers, etc.)
    
    Returns:
        Path to created brain directory
    """
    logger.info(f"Creating new brain directory: {brain_name}")
    
    # Sanitize brain name (replace spaces with underscores)
    safe_name = brain_name.replace(" ", "_")
    
    # Create brain directory
    brain_dir = os.path.join(project_root, "artifacts", "brains", "actv1", safe_name)
    os.makedirs(brain_dir, exist_ok=True)
    logger.debug(f"Directory path: {brain_dir}")
    
    # Build brain metadata
    brain_metadata = {
        "name": safe_name,
        "type": "actv1",
        "tokenizer_id": tokenizer_id,
        "tokenizer_model": tokenizer_model,
        "vocab_size": vocab_size,
        "checkpoint_file": "actv1_student.safetensors",
        "log_file": "metrics.jsonl",
        "default_goal": default_goal if default_goal else "Learn and improve through training.",
        "created_at": time.time(),
        # MoE configuration
        "use_moe": use_moe,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
    }
    
    logger.debug(f"Brain config: vocab_size={vocab_size}, MoE={use_moe}")
    
    # Add architecture parameters if provided
    if architecture:
        brain_metadata.update({
            "hidden_size": architecture.get("hidden_size"),
            "h_layers": architecture.get("h_layers"),
            "l_layers": architecture.get("l_layers"),
            "num_heads": architecture.get("num_heads"),
            "expansion": architecture.get("expansion"),
            "h_cycles": architecture.get("h_cycles"),
            "l_cycles": architecture.get("l_cycles"),
            "pos_encoding": architecture.get("pos_encoding"),
            "dtype": architecture.get("dtype"),
        })
        
        logger.debug(f"Architecture: {architecture.get('h_layers')}H/{architecture.get('l_layers')}L layers, "
                    f"{architecture.get('hidden_size')} hidden, {architecture.get('num_heads')} heads")
        
        # Calculate and store total parameters (source of truth)
        if all(k in architecture for k in ["hidden_size", "h_layers", "l_layers"]):
            try:
                from aios.cli.hrm_hf.model_building import calculate_actv1_params
                total_params = calculate_actv1_params(
                    vocab_size=vocab_size,
                    hidden_size=architecture.get("hidden_size"),
                    h_layers=architecture.get("h_layers"),
                    l_layers=architecture.get("l_layers"),
                    expansion=architecture.get("expansion", 2.0),
                    use_moe=use_moe,
                    num_experts=num_experts if use_moe else 1
                )
                brain_metadata["total_params"] = total_params
                brain_metadata["model_size_mb"] = (total_params * 4) / (1024 * 1024)
                logger.debug(f"Calculated parameters: {total_params:,} ({brain_metadata['model_size_mb']:.1f} MB)")
            except Exception as e:
                logger.warning(f"Could not calculate model parameters: {e}")
    
    # Write brain.json
    brain_json_path = os.path.join(brain_dir, "brain.json")
    with open(brain_json_path, 'w', encoding='utf-8') as f:
        json.dump(brain_metadata, f, indent=2)
    
    logger.info(f"Brain directory created successfully: {safe_name}")
    return brain_dir


def apply_preset_to_panel(panel: Any, preset: str) -> None:
    """
    Apply architecture preset to HRM panel UI variables.
    
    Args:
        panel: HRM training panel instance
        preset: Preset name ("1M", "5M", "10M", "20M", "50M")
    """
    # Preset configurations
    presets = {
        "1M": {
            "hidden_size": "128",
            "h_layers": "2",
            "l_layers": "2",
            "num_heads": "8",
            "expansion": "2.0",
            "h_cycles": "2",
            "l_cycles": "2",
        },
        "5M": {
            "hidden_size": "256",
            "h_layers": "2",
            "l_layers": "2",
            "num_heads": "8",
            "expansion": "2.0",
            "h_cycles": "2",
            "l_cycles": "2",
        },
        "10M": {
            "hidden_size": "384",
            "h_layers": "2",
            "l_layers": "2",
            "num_heads": "8",
            "expansion": "2.0",
            "h_cycles": "2",
            "l_cycles": "2",
        },
        "20M": {
            "hidden_size": "512",
            "h_layers": "2",
            "l_layers": "2",
            "num_heads": "8",
            "expansion": "2.0",
            "h_cycles": "2",
            "l_cycles": "2",
        },
        "50M": {
            "hidden_size": "768",
            "h_layers": "2",
            "l_layers": "2",
            "num_heads": "12",
            "expansion": "2.0",
            "h_cycles": "2",
            "l_cycles": "2",
        },
    }
    
    config = presets.get(preset)
    if not config:
        return  # Unknown preset, do nothing
    
    # Update panel variables
    panel.hidden_size_var.set(config["hidden_size"])
    panel.h_layers_var.set(config["h_layers"])
    panel.l_layers_var.set(config["l_layers"])
    panel.num_heads_var.set(config["num_heads"])
    panel.expansion_var.set(config["expansion"])
    panel.h_cycles_var.set(config["h_cycles"])
    panel.l_cycles_var.set(config["l_cycles"])
