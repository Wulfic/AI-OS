"""State persistence for HRM Training Panel.

Handles get_state/set_state for saving and loading panel configuration.
"""

from __future__ import annotations
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .panel_main import HRMTrainingPanel


def get_state(panel: HRMTrainingPanel) -> dict:
    """Return a dict of current UI settings for persistence.
    
    Args:
        panel: The HRMTrainingPanel instance
        
    Returns:
        dict: Current panel state
    """
    try:
        return {
            # core
            "dataset": panel.dataset_var.get(),
            "dataset_chunk_size": panel.dataset_chunk_size_var.get(),
            "model": panel.model_var.get(),
            "max_seq": panel.max_seq_var.get(),
            "batch": panel.batch_var.get(),
            "gradient_accumulation_steps": panel.gradient_accumulation_var.get(),
            "steps": panel.steps_var.get(),
            "lr": panel.lr_var.get(),
            "auto_adjust_lr": panel.auto_adjust_lr_var.get(),
            "halt_steps": panel.halt_steps_var.get(),
            "gradient_checkpointing": bool(panel.gradient_checkpointing_var.get()),
            "use_amp": bool(getattr(panel, "use_amp_var", None).get() if hasattr(panel, "use_amp_var") else True),
            "use_cpu_offload": bool(getattr(panel, "use_cpu_offload_var", None).get() if hasattr(panel, "use_cpu_offload_var") else False),
            "use_8bit_optimizer": bool(getattr(panel, "use_8bit_optimizer_var", None).get() if hasattr(panel, "use_8bit_optimizer_var") else False),
            "use_flash_attn": bool(getattr(panel, "use_flash_attn_var", None).get() if hasattr(panel, "use_flash_attn_var") else False),
            "flash_attn_window": getattr(panel, "flash_attn_window_var", None).get() if hasattr(panel, "flash_attn_window_var") else "512",
            "use_chunked_training": bool(getattr(panel, "use_chunked_training_var", None).get() if hasattr(panel, "use_chunked_training_var") else False),
            "chunk_size": panel.chunk_size_var.get(),
            "use_peft": bool(getattr(panel, "use_peft_var", None).get() if hasattr(panel, "use_peft_var") else False),
            "peft_method": getattr(panel, "peft_method_var", None).get() if hasattr(panel, "peft_method_var") else "lora",
            "lora_r": getattr(panel, "lora_r_var", None).get() if hasattr(panel, "lora_r_var") else "16",
            "lora_alpha": getattr(panel, "lora_alpha_var", None).get() if hasattr(panel, "lora_alpha_var") else "32",
            "lora_dropout": getattr(panel, "lora_dropout_var", None).get() if hasattr(panel, "lora_dropout_var") else "0.05",
            "lora_target_modules": getattr(panel, "lora_target_modules_var", None).get() if hasattr(panel, "lora_target_modules_var") else "q_proj,v_proj",
            "kl": panel.kl_var.get(),
            "kl_temp": panel.kl_temp_var.get(),
            "stop_file": panel.stop_file_var.get(),
            "log_file": panel.log_file_var.get(),
            "student_init": panel.student_init_var.get(),
            "h_layers": panel.h_layers_var.get(),
            "l_layers": panel.l_layers_var.get(),
            "hidden_size": panel.hidden_size_var.get(),
            "expansion": panel.expansion_var.get(),
            "num_heads": panel.num_heads_var.get(),
            "h_cycles": panel.h_cycles_var.get(),
            "l_cycles": panel.l_cycles_var.get(),
            "pos_enc": panel.pos_enc_var.get(),
            "brain_name": panel.brain_name_var.get(),
            "default_goal": panel.default_goal_var.get(),
            "bundle_dir": panel.bundle_dir_var.get(),
            "ascii_only": bool(panel.ascii_only_var.get()),
            "linear_dataset": bool(panel.linear_dataset_var.get()),
            "zero_stage": panel.zero_stage_var.get(),
            "ddp_abort_on_fail": bool(panel.ddp_abort_on_fail_var.get()),
            # flow
            "iterate": bool(getattr(panel, "iterate_var", None).get() if hasattr(panel, "iterate_var") else False),
            "stop_after_block": bool(getattr(panel, "stop_after_block_var", None).get() if hasattr(panel, "stop_after_block_var") else False),
            "stop_after_epoch": bool(getattr(panel, "stop_after_epoch_var", None).get() if hasattr(panel, "stop_after_epoch_var") else False),
        }
    except Exception:
        return {}


def set_state(panel: HRMTrainingPanel, state: dict) -> None:
    """Apply settings from a dict produced by get_state().
    
    Args:
        panel: The HRMTrainingPanel instance
        state: State dictionary to apply
    """
    if not isinstance(state, dict):
        return
    
    # helpers
    def _set_str(var, key):
        try:
            v = state.get(key)
            if isinstance(v, (str, int, float)):
                # Format learning rate values as regular decimals instead of scientific notation
                if key == "lr" and isinstance(v, (int, float)):
                    formatted = f"{float(v):.6f}".rstrip('0').rstrip('.')
                    var.set(formatted)
                else:
                    var.set(str(v))
        except Exception:
            pass
    
    def _set_bool(var, key):
        try:
            v = state.get(key)
            if isinstance(v, bool):
                var.set(v)
            elif isinstance(v, (int, float)):
                var.set(bool(v))
        except Exception:
            pass

    # core
    _set_str(panel.dataset_var, "dataset")
    _set_str(panel.dataset_chunk_size_var, "dataset_chunk_size")
    _set_str(panel.model_var, "model")
    _set_str(panel.max_seq_var, "max_seq")
    _set_str(panel.batch_var, "batch")
    _set_str(panel.gradient_accumulation_var, "gradient_accumulation_steps")
    _set_str(panel.steps_var, "steps")
    _set_str(panel.lr_var, "lr")
    _set_bool(panel.auto_adjust_lr_var, "auto_adjust_lr")
    _set_str(panel.halt_steps_var, "halt_steps")
    _set_bool(panel.gradient_checkpointing_var, "gradient_checkpointing")
    if hasattr(panel, "use_amp_var"):
        _set_bool(panel.use_amp_var, "use_amp")
    if hasattr(panel, "use_cpu_offload_var"):
        _set_bool(panel.use_cpu_offload_var, "use_cpu_offload")
    if hasattr(panel, "use_8bit_optimizer_var"):
        _set_bool(panel.use_8bit_optimizer_var, "use_8bit_optimizer")
    if hasattr(panel, "use_flash_attn_var"):
        _set_bool(panel.use_flash_attn_var, "use_flash_attn")
    if hasattr(panel, "flash_attn_window_var"):
        _set_str(panel.flash_attn_window_var, "flash_attn_window")
    if hasattr(panel, "use_chunked_training_var"):
        _set_bool(panel.use_chunked_training_var, "use_chunked_training")
    _set_str(panel.chunk_size_var, "chunk_size")
    
    # PEFT options
    if hasattr(panel, "use_peft_var"):
        _set_bool(panel.use_peft_var, "use_peft")
    if hasattr(panel, "peft_method_var"):
        _set_str(panel.peft_method_var, "peft_method")
    if hasattr(panel, "lora_r_var"):
        _set_str(panel.lora_r_var, "lora_r")
    if hasattr(panel, "lora_alpha_var"):
        _set_str(panel.lora_alpha_var, "lora_alpha")
    if hasattr(panel, "lora_dropout_var"):
        _set_str(panel.lora_dropout_var, "lora_dropout")
    if hasattr(panel, "lora_target_modules_var"):
        _set_str(panel.lora_target_modules_var, "lora_target_modules")
    
    _set_str(panel.kl_var, "kl")
    _set_str(panel.kl_temp_var, "kl_temp")
    _set_str(panel.stop_file_var, "stop_file")
    _set_str(panel.log_file_var, "log_file")
    _set_str(panel.student_init_var, "student_init")
    _set_str(panel.brain_name_var, "brain_name")
    _set_str(panel.default_goal_var, "default_goal")
    # arch
    _set_str(panel.h_layers_var, "h_layers")
    _set_str(panel.l_layers_var, "l_layers")
    _set_str(panel.hidden_size_var, "hidden_size")
    _set_str(panel.expansion_var, "expansion")
    _set_str(panel.num_heads_var, "num_heads")
    _set_str(panel.h_cycles_var, "h_cycles")
    _set_str(panel.l_cycles_var, "l_cycles")
    _set_str(panel.pos_enc_var, "pos_enc")
    _set_str(panel.brain_name_var, "brain_name")
    _set_str(panel.bundle_dir_var, "bundle_dir")
    _set_bool(panel.ascii_only_var, "ascii_only")
    _set_bool(panel.linear_dataset_var, "linear_dataset")
    _set_str(panel.zero_stage_var, "zero_stage")
    _set_bool(panel.ddp_abort_on_fail_var, "ddp_abort_on_fail")
    # flow
    if hasattr(panel, "iterate_var"):
        _set_bool(panel.iterate_var, "iterate")
    if hasattr(panel, "stop_after_block_var"):
        _set_bool(panel.stop_after_block_var, "stop_after_block")
    if hasattr(panel, "stop_after_epoch_var"):
        _set_bool(panel.stop_after_epoch_var, "stop_after_epoch")


def prefill_last_safe_batches(panel: HRMTrainingPanel) -> None:
    """Prefill batch size from persisted last safe batches if available.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    try:
        base = os.path.join(panel._project_root, "artifacts", "brains", "actv1", "last_safe.json")
        if os.path.exists(base):
            import json
            with open(base, "r", encoding="utf-8") as f:
                data = json.loads(f.read())
            if isinstance(data, dict):
                tb = data.get("train_batch")
                if isinstance(tb, int) and tb > 0:
                    panel.batch_var.set(str(tb))
                elif isinstance(tb, str) and tb.isdigit():
                    panel.batch_var.set(tb)
    except Exception:
        pass
