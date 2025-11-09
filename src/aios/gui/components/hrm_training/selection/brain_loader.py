"""Brain loading logic for populating HRM panel with existing brain configuration.

This module handles loading brain.json metadata and updating the HRM training panel UI.
"""

from __future__ import annotations
import os
import json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # Panel type is external


def load_brain_to_panel(panel: Any, brain_dir: str, brain_name: str) -> None:
    """
    Load an existing brain into the HRM training panel.
    
    This function:
    1. Sets student_init_var to the checkpoint path
    2. Sets log_file_var to metrics.jsonl
    3. Sets brain_name_var to the brain name
    4. Reads brain.json and populates architecture/MoE settings
    5. Disables architecture widgets (can't change loaded brain)
    
    Args:
        panel: HRM training panel instance
        brain_dir: Path to brain directory
        brain_name: Name of the brain
    """
    # Set basic paths
    pt_path = os.path.join(brain_dir, "actv1_student.safetensors")
    panel.student_init_var.set(pt_path)
    panel.log_file_var.set(os.path.join(brain_dir, "metrics.jsonl"))
    panel.brain_name_var.set(brain_name)
    
    # Read brain.json to populate training steps and architecture
    brain_json_path = os.path.join(brain_dir, "brain.json")
    if not os.path.isfile(brain_json_path):
        panel._log(f"[hrm] Warning: No brain.json found for {brain_name}")
        panel._set_arch_widgets_state("disabled")
        return
    
    try:
        with open(brain_json_path, 'r', encoding='utf-8') as f:
            brain_data = json.load(f)
        
        # Update training steps if available
        if "training_steps" in brain_data:
            panel.steps_var.set(str(brain_data["training_steps"]))
            panel._log(f"[hrm] Loaded training steps: {brain_data['training_steps']}")
        
        # Update tokenizer model from brain data
        if "tokenizer_model" in brain_data:
            panel.model_var.set(brain_data["tokenizer_model"])
            panel._log(f"[hrm] Loaded tokenizer: {brain_data['tokenizer_model']}")
            # Save state immediately to ensure tokenizer path is persisted
            if callable(getattr(panel, "_save_state_fn", None)):
                try:
                    panel._save_state_fn()
                except Exception:
                    pass  # Silent fail - state save is not critical
        
        # Update architecture fields from brain data
        _update_architecture_fields(panel, brain_data)
        
        # Update MoE fields first (from brain data)
        _update_moe_fields(panel, brain_data)
        
        # Update all brain stats (MoE/tok/params/MB/steps)
        try:
            from ...hrm_training_panel.memory_estimation import update_moe_stats_display
            update_moe_stats_display(panel)
            panel._log(f"[hrm] Brain stats populated from metadata")
        except Exception as e:
            panel._log(f"[hrm] Warning: Could not populate brain stats: {e}")
            import traceback
            panel._log(f"[hrm] Traceback: {traceback.format_exc()}")
        
        # Try to update VRAM/RAM estimates (this may fail if architecture vars incomplete)
        try:
            from ...hrm_training_panel.memory_estimation import update_vram_estimate
            update_vram_estimate(panel)
        except Exception as e:
            panel._log(f"[hrm] Note: VRAM/RAM estimates not available (will update after training start): {e}")
        
    except Exception as e:
        panel._log(f"[hrm] Warning: Could not read brain.json: {e}")
    
    # Disable architecture widgets (can't modify loaded brain architecture)
    panel._set_arch_widgets_state("disabled")
    panel._log(f"[hrm] Selected brain: {brain_name}")


def _update_architecture_fields(panel: Any, brain_data: dict) -> None:
    """
    Update architecture UI fields from brain metadata.
    
    Args:
        panel: HRM training panel instance
        brain_data: Parsed brain.json data
    """
    # Get architecture from brain data (try both root and 'arch' sub-key)
    arch = brain_data.get("arch", {})
    
    # H/L layers
    h_layers = arch.get("H_layers") or brain_data.get("h_layers") or 2
    l_layers = arch.get("L_layers") or brain_data.get("l_layers") or 2
    panel.h_layers_var.set(str(h_layers))
    panel.l_layers_var.set(str(l_layers))
    
    # Hidden size
    hidden_size = arch.get("hidden_size") or brain_data.get("hidden_size") or 512
    panel.hidden_size_var.set(str(hidden_size))
    
    # Number of heads
    num_heads = arch.get("num_heads") or brain_data.get("num_heads") or 8
    panel.num_heads_var.set(str(num_heads))
    
    # Expansion factor
    expansion = arch.get("expansion") or brain_data.get("expansion") or 2.0
    panel.expansion_var.set(str(expansion))
    
    # H/L cycles
    h_cycles = arch.get("H_cycles") or brain_data.get("h_cycles") or 2
    l_cycles = arch.get("L_cycles") or brain_data.get("l_cycles") or 2
    panel.h_cycles_var.set(str(h_cycles))
    panel.l_cycles_var.set(str(l_cycles))
    
    # Position encoding
    pos_enc = arch.get("pos_encodings") or brain_data.get("pos_encodings") or "rope"
    panel.pos_enc_var.set(pos_enc)
    
    panel._log(f"[hrm] Loaded architecture: {h_layers}H/{l_layers}L layers, {hidden_size} hidden, {num_heads} heads")


def _update_moe_fields(panel: Any, brain_data: dict) -> None:
    """
    Update MoE UI fields from brain metadata.
    
    Args:
        panel: HRM training panel instance
        brain_data: Parsed brain.json data
    """
    use_moe = brain_data.get("use_moe", False)
    
    if use_moe:
        # MoE enabled - show actual values
        num_experts = brain_data.get("num_experts", 8)
        active_per_tok = brain_data.get("num_experts_per_tok", 2)
        
        # Update num_experts field
        panel.moe_num_experts_entry.config(state="normal")
        panel.moe_num_experts_entry.delete(0, "end")
        panel.moe_num_experts_entry.insert(0, str(num_experts))
        panel.moe_num_experts_entry.config(state="readonly")
        
        # Update active experts field
        panel.moe_active_experts_entry.config(state="normal")
        panel.moe_active_experts_entry.delete(0, "end")
        panel.moe_active_experts_entry.insert(0, str(active_per_tok))
        panel.moe_active_experts_entry.config(state="readonly")
    else:
        # MoE disabled - show N/A
        panel.moe_num_experts_entry.config(state="normal")
        panel.moe_num_experts_entry.delete(0, "end")
        panel.moe_num_experts_entry.insert(0, "N/A")
        panel.moe_num_experts_entry.config(state="readonly")
        
        panel.moe_active_experts_entry.config(state="normal")
        panel.moe_active_experts_entry.delete(0, "end")
        panel.moe_active_experts_entry.insert(0, "N/A")
        panel.moe_active_experts_entry.config(state="readonly")


def load_brain_from_file(panel: Any, checkpoint_path: str) -> None:
    """
    Load a brain from an arbitrary checkpoint file (via Browse button).
    
    This is a simpler version that doesn't load full brain.json metadata,
    just sets the checkpoint path and infers directory structure.
    
    Args:
        panel: HRM training panel instance
        checkpoint_path: Path to .pt or .safetensors checkpoint file
    """
    panel.student_init_var.set(checkpoint_path)
    panel._set_arch_widgets_state("disabled")
    panel._log(f"[hrm] Selected student init: {checkpoint_path}")
    
    # Try to infer brain directory and set log file
    try:
        brain_dir = os.path.dirname(checkpoint_path)
        panel.log_file_var.set(os.path.join(brain_dir, "metrics.jsonl"))
        panel.brain_name_var.set(os.path.basename(brain_dir))
    except Exception:
        pass  # Silently fail if can't infer
