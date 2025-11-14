"""Variable initialization and trace setup for HRM Training Panel.

Handles creation of all tk.StringVar and tk.BooleanVar instances,
trace setup for auto-save, and PEFT module mapping.
"""

from __future__ import annotations
import logging
import tkinter as tk
from typing import TYPE_CHECKING

# Import safe variable wrappers
from ...utils import safe_variables

if TYPE_CHECKING:
    from .panel_main import HRMTrainingPanel

logger = logging.getLogger(__name__)


def setup_variables(panel: HRMTrainingPanel) -> None:
    """Initialize all tkinter variables for the panel.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    # Core inputs
    panel.dataset_var = safe_variables.StringVar(value="training_data")
    panel.dataset_chunk_size_var = safe_variables.StringVar(value="4000")
    panel.model_var = safe_variables.StringVar(value="artifacts/hf_implant/base_model")
    panel.max_seq_var = safe_variables.StringVar(value="128")
    panel.batch_var = safe_variables.StringVar(value="4")
    panel.gradient_accumulation_var = safe_variables.StringVar(value="1")
    panel.steps_var = safe_variables.StringVar(value="100")
    panel._auto_steps_calculating = False
    panel.lr_var = safe_variables.StringVar(value="0.00005")
    panel.auto_adjust_lr_var = safe_variables.BooleanVar(value=True)
    panel.halt_steps_var = safe_variables.StringVar(value="1")
    panel.gradient_checkpointing_var = safe_variables.BooleanVar(value=True)
    panel.use_amp_var = safe_variables.BooleanVar(value=True)
    panel.use_cpu_offload_var = safe_variables.BooleanVar(value=False)
    panel.use_8bit_optimizer_var = safe_variables.BooleanVar(value=False)
    panel.use_flash_attn_var = safe_variables.BooleanVar(value=False)
    panel.flash_attn_window_var = safe_variables.StringVar(value="512")
    panel.use_chunked_training_var = safe_variables.BooleanVar(value=False)
    panel.chunk_size_var = safe_variables.StringVar(value="2048")
    
    # PEFT (Parameter-Efficient Fine-Tuning) options
    panel.use_peft_var = safe_variables.BooleanVar(value=False)
    panel.peft_method_var = safe_variables.StringVar(value="lora")
    panel.lora_r_var = safe_variables.StringVar(value="16")
    panel.lora_alpha_var = safe_variables.StringVar(value="32")
    panel.lora_dropout_var = safe_variables.StringVar(value="0.05")
    panel.lora_target_modules_var = safe_variables.StringVar(value="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    
    panel.kl_var = safe_variables.StringVar(value="0.0")
    panel.kl_temp_var = safe_variables.StringVar(value="1.0")
    panel.strict_var = safe_variables.BooleanVar(value=True)
    panel.eval_file_var = safe_variables.StringVar(value="")
    panel.eval_batches_var = safe_variables.StringVar(value="10")
    panel.stop_file_var = safe_variables.StringVar(value="training_data/actv1/STOP")
    panel.log_file_var = safe_variables.StringVar(value="artifacts/brains/actv1/default/metrics.jsonl")
    panel.student_init_var = safe_variables.StringVar(value="artifacts/brains/actv1/default/actv1_student.safetensors")
    panel.brain_name_var = safe_variables.StringVar(value="default")
    panel.default_goal_var = safe_variables.StringVar(value="")
    panel.bundle_dir_var = safe_variables.StringVar(value="artifacts/brains/actv1")
    
    # Architecture knobs
    panel.h_layers_var = safe_variables.StringVar(value="2")
    panel.l_layers_var = safe_variables.StringVar(value="2")
    panel.hidden_size_var = safe_variables.StringVar(value="512")
    panel.expansion_var = safe_variables.StringVar(value="2.0")
    panel.num_heads_var = safe_variables.StringVar(value="8")
    panel.h_cycles_var = safe_variables.StringVar(value="2")
    panel.l_cycles_var = safe_variables.StringVar(value="2")
    panel.pos_enc_var = safe_variables.StringVar(value="rope")
    
    # Data filtering
    panel.ascii_only_var = safe_variables.BooleanVar(value=False)
    
    # Dataset progression mode
    panel.linear_dataset_var = safe_variables.BooleanVar(value=True)
    
    # DeepSpeed ZeRO optimization
    panel.zero_stage_var = safe_variables.StringVar(value="none")
    
    # DDP behavior toggles (always abort on DDP failure by default)
    panel.ddp_abort_on_fail_var = safe_variables.BooleanVar(value=True)
    
    # Flow control
    panel.iterate_var = safe_variables.BooleanVar(value=False)
    panel.stop_after_block_var = safe_variables.BooleanVar(value=False)
    panel.stop_after_epoch_var = safe_variables.BooleanVar(value=False)
    
    # Epoch tracking (internal - updated by metrics polling)
    panel._epoch_tracking_initialized = False
    panel._dataset_total_samples = None
    panel._samples_per_block = None
    panel._total_blocks = None
    panel._current_epoch = 0
    panel._samples_processed_this_epoch = 0
    panel._blocks_processed = 0
    panel._current_block_samples = 0
    panel._chunks_completed = 0
    panel._total_chunks = None
    panel._current_chunk_id = None
    panel._total_steps_all_gpus = 0
    panel._dataset_name = None


def setup_variable_traces(panel: HRMTrainingPanel) -> None:
    """Setup trace callbacks for auto-save and VRAM updates.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    from .helpers import schedule_save
    
    # Helper to create traced variable callback with logging
    def make_trace_callback(var_name: str, var_obj):
        """Create a traced callback that logs parameter changes."""
        def callback(*args):
            try:
                new_val = var_obj.get()
                logger.debug(f"HRM training parameter changed: {var_name}={new_val}")
            except Exception:
                pass
            schedule_save(panel)
        return callback
    
    # Auto-persist on changes (best-effort)
    try:
        _vars_to_watch = [
            ("learning_rate", panel.lr_var),
            ("batch_size", panel.batch_var),
            ("gradient_accumulation", panel.gradient_accumulation_var),
            ("dataset", panel.dataset_var),
            ("model", panel.model_var),
            ("max_seq_len", panel.max_seq_var),
            ("steps", panel.steps_var),
            ("halt_steps", panel.halt_steps_var),
            ("gradient_checkpointing", panel.gradient_checkpointing_var),
            ("use_amp", panel.use_amp_var),
            ("use_cpu_offload", panel.use_cpu_offload_var),
            ("use_8bit_optimizer", panel.use_8bit_optimizer_var),
            ("use_flash_attn", panel.use_flash_attn_var),
            ("use_chunked_training", panel.use_chunked_training_var),
            ("chunk_size", panel.chunk_size_var),
            ("use_peft", panel.use_peft_var),
            ("peft_method", panel.peft_method_var),
            ("lora_r", panel.lora_r_var),
            ("lora_alpha", panel.lora_alpha_var),
            ("lora_dropout", panel.lora_dropout_var),
            ("lora_target_modules", panel.lora_target_modules_var),
            ("kl_coef", panel.kl_var),
            ("kl_temp", panel.kl_temp_var),
            ("stop_file", panel.stop_file_var),
            ("log_file", panel.log_file_var),
            ("student_init", panel.student_init_var),
            ("brain_name", panel.brain_name_var),
            ("default_goal", panel.default_goal_var),
            ("bundle_dir", panel.bundle_dir_var),
            ("ascii_only", panel.ascii_only_var),
            ("linear_dataset", panel.linear_dataset_var),
            ("h_layers", panel.h_layers_var),
            ("l_layers", panel.l_layers_var),
            ("hidden_size", panel.hidden_size_var),
            ("expansion_factor", panel.expansion_var),
            ("num_heads", panel.num_heads_var),
            ("h_cycles", panel.h_cycles_var),
            ("l_cycles", panel.l_cycles_var),
            ("pos_encoding", panel.pos_enc_var),
            ("zero_stage", panel.zero_stage_var),
            ("ddp_abort_on_fail", panel.ddp_abort_on_fail_var),
        ]
        for var_name, var_obj in _vars_to_watch:
            try:
                var_obj.trace_add("write", make_trace_callback(var_name, var_obj))
            except Exception:
                pass
        
        # Also update VRAM estimate when key parameters change
        _vram_vars = [
            panel.max_seq_var, panel.batch_var, panel.h_layers_var, panel.l_layers_var,
            panel.hidden_size_var, panel.expansion_var, panel.num_heads_var,
            panel.h_cycles_var, panel.l_cycles_var, panel.halt_steps_var,
            panel.zero_stage_var,
        ]
        for v in _vram_vars:
            try:
                v.trace_add("write", lambda *args: _schedule_vram_update(panel))
            except Exception:
                pass
        
        # Update dataset name display when dataset changes
        try:
            panel.dataset_var.trace_add("write", lambda *args: _update_dataset_display(panel))
        except Exception:
            pass
    except Exception:
        pass


def _schedule_vram_update(panel: HRMTrainingPanel, *args) -> None:
    """Schedule VRAM and stats update with small delay to avoid rapid-fire updates.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    # Cancel any pending update
    if hasattr(panel, '_vram_update_after_id') and panel._vram_update_after_id is not None:
        try:
            panel.after_cancel(panel._vram_update_after_id)
        except Exception:
            pass
    # Schedule new update after 300ms delay (debouncing)
    try:
        from .memory_estimation import update_vram_estimate, update_moe_stats_display
        def _update_all():
            update_vram_estimate(panel)
            update_moe_stats_display(panel)
        panel._vram_update_after_id = panel.after(300, _update_all)
    except Exception:
        pass


def _update_dataset_display(panel: HRMTrainingPanel, *args) -> None:
    """Update dataset name display when dataset field changes.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    try:
        if not hasattr(panel, "epoch_dataset_lbl"):
            return
        
        dataset_str = panel.dataset_var.get().strip()
        if not dataset_str or dataset_str == "training_data":
            panel.epoch_dataset_lbl.config(text="-")
            panel._dataset_name = None
            return
        
        # Extract name from path or HF dataset
        if 'hf://' in dataset_str:
            dataset_name = dataset_str.split('hf://')[-1].split(':')[0]
        else:
            import os
            name = os.path.basename(dataset_str)
            for ext in ['.txt', '.csv', '.json', '.jsonl']:
                name = name.replace(ext, '')
            dataset_name = name if name else "unknown"
        
        panel.epoch_dataset_lbl.config(text=dataset_name)
        panel._dataset_name = dataset_name
    except Exception:
        pass


def setup_lora_module_mapping(panel: HRMTrainingPanel, lora_modules_combo: any) -> None:
    """Setup LoRA module mapping for combobox.
    
    Args:
        panel: The HRMTrainingPanel instance
        lora_modules_combo: The ttk.Combobox widget
    """
    # Map display names to actual module strings
    panel._lora_module_map = {
        'Minimal': 'q_proj,v_proj',
        'Balanced': 'q_proj,k_proj,v_proj,o_proj',
        'Full': 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj'
    }
    panel._lora_module_reverse_map = {v: k for k, v in panel._lora_module_map.items()}
    
    # Set initial display value
    initial_val = panel.lora_target_modules_var.get()
    if initial_val in panel._lora_module_reverse_map:
        lora_modules_combo.set(panel._lora_module_reverse_map[initial_val])
    else:
        lora_modules_combo.set('Full')
        panel.lora_target_modules_var.set('q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    
    # Bind selection event to update actual value
    def _on_module_select(event=None):
        display_val = lora_modules_combo.get()
        if display_val in panel._lora_module_map:
            panel.lora_target_modules_var.set(panel._lora_module_map[display_val])
    lora_modules_combo.bind('<<ComboboxSelected>>', _on_module_select)
    
    # Bind variable changes to update combo display (for config loading)
    def _on_var_change(*args):
        actual_val = panel.lora_target_modules_var.get()
        if actual_val in panel._lora_module_reverse_map:
            lora_modules_combo.set(panel._lora_module_reverse_map[actual_val])
    panel.lora_target_modules_var.trace_add('write', _on_var_change)
