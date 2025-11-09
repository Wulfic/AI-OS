"""Variable initialization and trace setup for HRM Training Panel.

Handles creation of all tk.StringVar and tk.BooleanVar instances,
trace setup for auto-save, and PEFT module mapping.
"""

from __future__ import annotations
import tkinter as tk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .panel_main import HRMTrainingPanel


def setup_variables(panel: HRMTrainingPanel) -> None:
    """Initialize all tkinter variables for the panel.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    # Core inputs
    panel.dataset_var = tk.StringVar(value="training_data")
    panel.dataset_chunk_size_var = tk.StringVar(value="4000")
    panel.model_var = tk.StringVar(value="artifacts/hf_implant/base_model")
    panel.max_seq_var = tk.StringVar(value="128")
    panel.batch_var = tk.StringVar(value="4")
    panel.gradient_accumulation_var = tk.StringVar(value="1")
    panel.steps_var = tk.StringVar(value="100")
    panel._auto_steps_calculating = False
    panel.lr_var = tk.StringVar(value="0.00005")
    panel.auto_adjust_lr_var = tk.BooleanVar(value=True)
    panel.halt_steps_var = tk.StringVar(value="1")
    panel.gradient_checkpointing_var = tk.BooleanVar(value=True)
    panel.use_amp_var = tk.BooleanVar(value=True)
    panel.use_cpu_offload_var = tk.BooleanVar(value=False)
    panel.use_8bit_optimizer_var = tk.BooleanVar(value=False)
    panel.use_flash_attn_var = tk.BooleanVar(value=False)
    panel.flash_attn_window_var = tk.StringVar(value="512")
    panel.use_chunked_training_var = tk.BooleanVar(value=False)
    panel.chunk_size_var = tk.StringVar(value="2048")
    
    # PEFT (Parameter-Efficient Fine-Tuning) options
    panel.use_peft_var = tk.BooleanVar(value=False)
    panel.peft_method_var = tk.StringVar(value="lora")
    panel.lora_r_var = tk.StringVar(value="16")
    panel.lora_alpha_var = tk.StringVar(value="32")
    panel.lora_dropout_var = tk.StringVar(value="0.05")
    panel.lora_target_modules_var = tk.StringVar(value="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    
    panel.kl_var = tk.StringVar(value="0.0")
    panel.kl_temp_var = tk.StringVar(value="1.0")
    panel.strict_var = tk.BooleanVar(value=True)
    panel.eval_file_var = tk.StringVar(value="")
    panel.eval_batches_var = tk.StringVar(value="10")
    panel.stop_file_var = tk.StringVar(value="training_data/actv1/STOP")
    panel.log_file_var = tk.StringVar(value="artifacts/brains/actv1/default/metrics.jsonl")
    panel.student_init_var = tk.StringVar(value="artifacts/brains/actv1/default/actv1_student.safetensors")
    panel.brain_name_var = tk.StringVar(value="default")
    panel.default_goal_var = tk.StringVar(value="")
    panel.bundle_dir_var = tk.StringVar(value="artifacts/brains/actv1")
    
    # Architecture knobs
    panel.h_layers_var = tk.StringVar(value="2")
    panel.l_layers_var = tk.StringVar(value="2")
    panel.hidden_size_var = tk.StringVar(value="512")
    panel.expansion_var = tk.StringVar(value="2.0")
    panel.num_heads_var = tk.StringVar(value="8")
    panel.h_cycles_var = tk.StringVar(value="2")
    panel.l_cycles_var = tk.StringVar(value="2")
    panel.pos_enc_var = tk.StringVar(value="rope")
    
    # Data filtering
    panel.ascii_only_var = tk.BooleanVar(value=False)
    
    # Dataset progression mode
    panel.linear_dataset_var = tk.BooleanVar(value=True)
    
    # DeepSpeed ZeRO optimization
    panel.zero_stage_var = tk.StringVar(value="none")
    
    # DDP behavior toggles (always abort on DDP failure by default)
    panel.ddp_abort_on_fail_var = tk.BooleanVar(value=True)
    
    # Flow control
    panel.iterate_var = tk.BooleanVar(value=False)
    panel.stop_after_block_var = tk.BooleanVar(value=False)
    panel.stop_after_epoch_var = tk.BooleanVar(value=False)
    
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
    
    # Auto-persist on changes (best-effort)
    try:
        _vars_to_watch = [
            panel.dataset_var, panel.model_var, panel.max_seq_var, panel.batch_var,
            panel.gradient_accumulation_var, panel.steps_var, panel.lr_var, panel.halt_steps_var, panel.gradient_checkpointing_var, 
            panel.use_amp_var, panel.use_cpu_offload_var, panel.use_8bit_optimizer_var, panel.use_flash_attn_var,
            panel.use_chunked_training_var, panel.chunk_size_var,
            panel.use_peft_var, panel.peft_method_var, panel.lora_r_var, panel.lora_alpha_var, panel.lora_dropout_var, panel.lora_target_modules_var,
            panel.kl_var, panel.kl_temp_var, panel.stop_file_var, panel.log_file_var, panel.student_init_var,
            panel.brain_name_var, panel.default_goal_var, panel.bundle_dir_var, panel.ascii_only_var,
            panel.linear_dataset_var,
            panel.h_layers_var, panel.l_layers_var, panel.hidden_size_var, panel.expansion_var,
            panel.num_heads_var, panel.h_cycles_var, panel.l_cycles_var, panel.pos_enc_var,
            panel.zero_stage_var,
            panel.ddp_abort_on_fail_var,
        ]
        for v in _vars_to_watch:
            try:
                v.trace_add("write", lambda *args: schedule_save(panel))
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
