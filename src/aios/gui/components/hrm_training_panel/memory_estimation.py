"""Memory estimation logic for HRM Training Panel.

Handles VRAM/RAM estimation, MoE stats display, and model parameter calculation.

Note: Panel attributes (h_layers_var, moe_num_experts_entry, etc.) are added 
dynamically by variable_setup.py and ui_builder modules at runtime.
Type checker cannot see these attributes statically.
"""
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations
import os
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .panel_main import HRMTrainingPanel


def estimate_model_params(panel: HRMTrainingPanel) -> int:  # pyright: ignore[reportAttributeAccessIssue]
    """Estimate the number of model parameters based on architecture settings.
    
    Uses the centralized calculate_actv1_params() function for accuracy.
    
    Args:
        panel: The HRMTrainingPanel instance
        
    Returns:
        int: Estimated total parameters
    """
    try:
        from aios.cli.hrm_hf.model_building import calculate_actv1_params
        
        # Parse architecture parameters
        h_layers = int(panel.h_layers_var.get() or 2)
        l_layers = int(panel.l_layers_var.get() or 2)
        hidden_size = int(panel.hidden_size_var.get() or 512)
        expansion = float(panel.expansion_var.get() or 2.0)
        
        # Use default vocab size to avoid slow tokenizer loading during startup
        # Tokenizer loading can add 15-20 seconds to GUI startup time
        vocab_size = 50257  # Default GPT-2 vocab
        
        # Get MoE settings
        use_moe = False
        num_experts = 1
        try:
            moe_experts_text = panel.moe_num_experts_entry.get()
            if moe_experts_text and moe_experts_text != "N/A" and moe_experts_text != "-":
                num_experts = int(moe_experts_text)
                use_moe = True
        except:
            pass
        
        # Calculate using centralized function
        return calculate_actv1_params(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            h_layers=h_layers,
            l_layers=l_layers,
            expansion=expansion,
            use_moe=use_moe,
            num_experts=num_experts
        )
    except Exception:
        return 0


def get_effective_chunk_size(total_params: int, seq_len: int) -> int:
    """Calculate effective chunk size based on model size and sequence length.
    
    Args:
        total_params: Total model parameters
        seq_len: Sequence length
        
    Returns:
        int: Effective chunk size
    """
    # Auto-chunking only for long sequences
    if seq_len <= 8192:
        return seq_len
    
    # Base chunk sizes
    if total_params > 200_000_000:  # 200M+ params
        base_chunk = 512
    elif total_params > 100_000_000:  # 100M-200M params
        base_chunk = 1024
    else:  # <100M params
        base_chunk = 2048
    
    # For very long sequences, use even smaller chunks
    if seq_len >= 100_000:
        return min(base_chunk, 512)
    elif seq_len >= 50_000:
        return min(base_chunk, 1024)
    elif seq_len >= 20_000:
        return min(base_chunk, 1536)
    else:
        return base_chunk


def estimate_dataset_memory(batch_size: int, seq_len: int) -> float:
    """Estimate dataset memory overhead.
    
    Args:
        batch_size: Training batch size
        seq_len: Sequence length
        
    Returns:
        float: Estimated dataset memory in GB
    """
    # With streaming enabled for large datasets, memory is minimal
    bytes_per_token = 4  # int32
    
    # Current batch memory
    current_batch_memory = batch_size * seq_len * bytes_per_token * 2  # input_ids + labels
    
    # Streaming keeps memory low
    streaming_overhead_gb = (current_batch_memory * 1.2) / (1024**3)
    
    # Cap at reasonable minimum
    return max(0.05, streaming_overhead_gb)


def estimate_teacher_params(model_name: str) -> int:
    """Estimate teacher model parameters based on model name.
    
    Args:
        model_name: Model name/path
        
    Returns:
        int: Estimated parameters
    """
    model_name_lower = model_name.lower()
    
    if "gpt2-xl" in model_name_lower or "gpt2xl" in model_name_lower:
        return 1_500_000_000
    elif "gpt2-medium" in model_name_lower or "gpt2medium" in model_name_lower:
        return 345_000_000
    elif "distilgpt2" in model_name_lower:
        return 82_000_000
    elif "gpt-j" in model_name_lower or "gptj" in model_name_lower:
        return 6_000_000_000
    elif "llama-7b" in model_name_lower or "7b" in model_name_lower:
        return 7_000_000_000
    elif "llama-13b" in model_name_lower or "13b" in model_name_lower:
        return 13_000_000_000
    elif "phi" in model_name_lower:
        if "2.7" in model_name_lower:
            return 2_700_000_000
        else:
            return 1_300_000_000
    elif "mistral" in model_name_lower:
        return 7_000_000_000
    else:
        return 124_000_000


def update_vram_estimate(panel: HRMTrainingPanel) -> None:  # pyright: ignore[reportAttributeAccessIssue]
    """Update memory estimates (VRAM and RAM) using accurate MemoryEstimator.
    
    Accounts for all optimizations: AMP, gradient checkpointing, LoRA/PEFT,
    CPU offload, 8-bit optimizer, DeepSpeed ZeRO, chunked training.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    from .helpers import log, mk_bool
    
    try:
        from ..hrm_training.memory_estimator import MemoryEstimator
        
        # Get parameters
        batch_size = int(panel.batch_var.get() or 4)
        seq_len = int(panel.max_seq_var.get() or 128)
        total_params = estimate_model_params(panel)
        hidden_size = int(panel.hidden_size_var.get() or 512)
        h_layers = int(panel.h_layers_var.get() or 2)
        l_layers = int(panel.l_layers_var.get() or 2)
        num_layers = h_layers + l_layers
        
        if total_params == 0:
            # Invalid params - show placeholder
            panel.vram_model_lbl.config(text="-")
            panel.vram_optimizer_lbl.config(text="-")
            panel.vram_activations_lbl.config(text="-")
            panel.vram_total_lbl.config(text="-")
            panel.ram_dataset_lbl.config(text="-")
            panel.ram_offload_lbl.config(text="-")
            panel.ram_total_lbl.config(text="-")
            return
        
        # Get optimization settings
        use_amp = bool(getattr(panel, "use_amp_var", mk_bool(True)).get())
        use_gradient_checkpointing = bool(getattr(panel, "gradient_checkpointing_var", mk_bool(True)).get())
        use_lora = bool(getattr(panel, "use_peft_var", mk_bool(False)).get())
        lora_r = int(panel.lora_r_var.get() or 16) if use_lora else 16
        use_cpu_offload = bool(getattr(panel, "use_cpu_offload_var", mk_bool(False)).get())
        use_8bit_optimizer = bool(getattr(panel, "use_8bit_optimizer_var", mk_bool(False)).get())
        zero_stage = panel.zero_stage_var.get()
        use_chunking = seq_len > 8192
        
        # Warn about long sequences without AMP
        if seq_len > 2048 and not use_amp:
            try:
                log(panel, f"[hrm] ⚠️  WARNING: Long sequence ({seq_len} tokens) without AMP will use massive memory!")
                log(panel, f"[hrm] 💡 STRONGLY RECOMMENDED: Enable AMP to reduce memory by ~50%")
                log(panel, f"[hrm] FP32 attention memory for seq_len={seq_len}: ~{(seq_len * seq_len * 4 * num_layers) / 1e9:.1f} GB per batch!")
            except:
                pass
        
        # Get number of GPUs
        num_gpus = 1
        try:
            rp = getattr(panel, "_resources_panel", None)
            if rp is not None:
                rvals = rp.get_values()
                sel_train = rvals.get("train_cuda_selected") or []
                if isinstance(sel_train, list) and len(sel_train) > 0:
                    num_gpus = len(sel_train)
        except Exception:
            pass
        
        # Get number of attention heads
        num_heads = int(panel.num_heads_var.get() or 8)
        
        # Create memory estimator
        estimator = MemoryEstimator(
            total_params=total_params,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            batch_size=batch_size,
            seq_len=seq_len,
            use_amp=use_amp,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_lora=use_lora,
            lora_r=lora_r,
            offload_optimizer=use_cpu_offload,
            use_8bit_optimizer=use_8bit_optimizer,
            zero_stage=zero_stage,
            num_gpus=num_gpus,
            use_chunking=use_chunking,
        )
        
        # Get estimates
        summary = estimator.get_summary()
        vram = summary["vram"]
        ram = summary["ram"]
        
        # Update VRAM display
        panel.vram_model_lbl.config(text=f"{vram['model_gb']:.2f} GB")
        panel.vram_optimizer_lbl.config(text=f"{vram['optimizer_gb']:.2f} GB")
        
        act_grad_total = vram['activations_gb'] + vram['gradients_gb']
        panel.vram_activations_lbl.config(text=f"{act_grad_total:.2f} GB")
        
        # Color-code total
        per_gpu_vram_gb = vram['total_gb']
        if per_gpu_vram_gb <= 8:
            color = "green"
            recommendation = "✓ Fits most GPUs"
        elif per_gpu_vram_gb <= 12:
            color = "#DAA520"
            recommendation = "⚠ Needs 12GB+ GPU"
        elif per_gpu_vram_gb <= 16:
            color = "orange"
            recommendation = "⚠ Needs 16GB+ GPU"
        elif per_gpu_vram_gb <= 24:
            color = "red"
            recommendation = "⚠ Needs 24GB+ GPU"
        else:
            color = "red"
            recommendation = "❌ Needs 40GB+ GPU"
        
        gpu_text = f"{per_gpu_vram_gb:.2f} GB"
        panel.vram_total_lbl.config(text=gpu_text, foreground=color)
        
        # Update RAM display
        panel.ram_dataset_lbl.config(text=f"{ram['dataset_gb']:.2f} GB")
        panel.ram_offload_lbl.config(text=f"{ram['optimizer_gb']:.2f} GB")
        ram_total_text = f"{ram['total_gb']:.1f} GB"
        panel.ram_total_lbl.config(text=ram_total_text)
        
        # Update MoE stats and tokenizer display
        update_moe_stats_display(panel)
        
        # Add tooltips with detailed breakdown
        try:
            from ..tooltips import add_tooltip
            
            cfg = summary["config"]
            
            # Build multi-GPU mode note
            if num_gpus > 1:
                if cfg.get('zero_stage', 'none') != 'none':
                    gpu_mode = f"DDP mode with {cfg['zero_stage'].upper()} (memory partitioned)"
                else:
                    gpu_mode = f"Parallel mode ({num_gpus} independent instances)"
            else:
                gpu_mode = "Single GPU"
            
            tooltip_lines = [
                "══════ MEMORY ESTIMATE ══════",
                "",
                f"🎯 VRAM per GPU: {per_gpu_vram_gb:.2f} GB",
                f"   • Model: {vram['model_gb']:.2f} GB",
                f"   • Optimizer: {vram['optimizer_gb']:.2f} GB",
                f"   • Gradients: {vram['gradients_gb']:.2f} GB",
                f"   • Activations: {vram['activations_gb']:.2f} GB",
                f"   • Overhead: {vram['overhead_gb']:.2f} GB",
                "",
                f"💾 System RAM: {ram['total_gb']:.1f} GB",
                f"   • Dataset: {ram['dataset_gb']:.2f} GB",
                f"   • CPU Offload: {ram['optimizer_gb']:.2f} GB",
                f"   • PyTorch: {ram['pytorch_gb']:.2f} GB",
                "",
                "⚙️  Active Optimizations:",
                f"   • AMP (FP16): {'✓' if cfg['use_amp'] else '✗'}",
                f"   • Gradient Checkpointing: {'✓' if cfg['use_gradient_checkpointing'] else '✗'}",
                f"   • LoRA/PEFT: {'✓' if cfg['use_lora'] else '✗'}",
                f"   • CPU Offload: {'✓' if cfg['offload_optimizer'] else '✗'}",
                f"   • 8-bit Optimizer: {'✓' if cfg['use_8bit_optimizer'] else '✗'}",
                f"   • DeepSpeed ZeRO: {cfg['zero_stage']}",
                f"   • Chunked Training: {'✓' if cfg['use_chunking'] else '✗'}",
                "",
                f"📊 Configuration:",
                f"   • Total params: {total_params/1e6:.1f}M",
                f"   • Trainable params: {vram['breakdown']['trainable_params']/1e6:.1f}M",
                f"   • Batch size: {batch_size}",
                f"   • Sequence length: {seq_len}",
                f"   • Effective chunk: {vram['breakdown']['effective_seq']}",
                f"   • Multi-GPU: {gpu_mode}",
                "",
                f"💡 {recommendation}",
            ]
            
            if per_gpu_vram_gb > 12:
                tooltip_lines.extend(["", "🔧 Suggestions to reduce VRAM:"])
                if not cfg['use_amp']:
                    tooltip_lines.append("   • Enable AMP → Save ~40%")
                if not cfg['use_gradient_checkpointing']:
                    tooltip_lines.append("   • Enable Grad Checkpoint → Save ~60% activations")
                if not cfg['use_lora']:
                    tooltip_lines.append("   • Enable LoRA → Save ~99% optimizer/gradients")
                if not cfg['offload_optimizer'] and vram['optimizer_gb'] > 0:
                    tooltip_lines.append(f"   • Enable CPU Offload → Move {vram['optimizer_gb']:.1f} GB to RAM")
                if batch_size > 1:
                    tooltip_lines.append(f"   • Reduce batch size → Direct VRAM savings")
            
            add_tooltip(panel.vram_total_lbl, "\n".join(tooltip_lines))
            
            ram_tooltip = [
                "══════ SYSTEM RAM ══════",
                "",
                f"Total: {ram['total_gb']:.1f} GB",
                "",
                "Breakdown:",
                f"  • Dataset buffer: {ram['dataset_gb']:.2f} GB",
                f"  • CPU offloaded optimizer: {ram['optimizer_gb']:.2f} GB",
                f"  • PyTorch/Python: {ram['pytorch_gb']:.2f} GB",
                "",
                "💡 Enable CPU Offload to move optimizer",
                "   state from VRAM to RAM (slower but saves VRAM)",
            ]
            add_tooltip(panel.ram_total_lbl, "\n".join(ram_tooltip))
            
        except Exception:
            pass
            
    except Exception as e:
        # Log the actual error for debugging
        try:
            from .helpers import log
            import traceback
            log(panel, f"[hrm] Memory estimation error: {e}")
            log(panel, f"[hrm] Traceback: {traceback.format_exc()}")
        except Exception:
            pass
        
        try:
            panel.vram_model_lbl.config(text="-")
            panel.vram_optimizer_lbl.config(text="-")
            panel.vram_activations_lbl.config(text="-")
            panel.vram_total_lbl.config(text="-")
            panel.ram_dataset_lbl.config(text="-")
            panel.ram_offload_lbl.config(text="-")
            panel.ram_total_lbl.config(text="-")
        except Exception:
            pass


def update_moe_stats_display(panel: HRMTrainingPanel) -> None:  # pyright: ignore[reportAttributeAccessIssue]
    """Update MoE stats, tokenizer, and model stats from brain.json.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    try:
        from .helpers import log
        
        # Try to find brain.json
        brain_json_path = None
        brain_dir = None
        
        # Method 1: From student_init path
        student_path = panel.student_init_var.get()
        if student_path and os.path.exists(student_path):
            brain_dir = os.path.dirname(student_path)
            brain_json_path = os.path.join(brain_dir, "brain.json")
        
        # Method 2: From brain name (fallback)
        if not brain_json_path or not os.path.exists(brain_json_path):
            brain_name = panel.brain_name_var.get()
            # Skip if brain name is the default placeholder value
            if brain_name and brain_name != "default":
                brain_dir = os.path.join("artifacts", "brains", "actv1", brain_name)
                brain_json_path = os.path.join(brain_dir, "brain.json")
                log(panel, f"[hrm] Looking for brain.json at: {brain_json_path}")
                log(panel, f"[hrm] Brain name: {brain_name}, Exists: {os.path.exists(brain_json_path) if brain_json_path else False}")
        
        if brain_json_path and os.path.exists(brain_json_path):
            with open(brain_json_path) as f:
                brain_meta = json.load(f)
            
            # MoE configuration
            use_moe = brain_meta.get("use_moe", False)
            num_experts = brain_meta.get("num_experts", 8)
            active_per_tok = brain_meta.get("num_experts_per_tok", 2)
            
            if use_moe:
                panel.moe_num_experts_entry.config(state="normal")
                panel.moe_num_experts_entry.delete(0, "end")
                panel.moe_num_experts_entry.insert(0, str(num_experts))
                panel.moe_num_experts_entry.config(state="readonly")
                
                panel.moe_active_experts_entry.config(state="normal")
                panel.moe_active_experts_entry.delete(0, "end")
                panel.moe_active_experts_entry.insert(0, str(active_per_tok))
                panel.moe_active_experts_entry.config(state="readonly")
            else:
                panel.moe_num_experts_entry.config(state="normal")
                panel.moe_num_experts_entry.delete(0, "end")
                panel.moe_num_experts_entry.insert(0, "N/A")
                panel.moe_num_experts_entry.config(state="readonly")
                
                panel.moe_active_experts_entry.config(state="normal")
                panel.moe_active_experts_entry.delete(0, "end")
                panel.moe_active_experts_entry.insert(0, "N/A")
                panel.moe_active_experts_entry.config(state="readonly")
            
            # Tokenizer detection
            tokenizer_id = brain_meta.get("tokenizer_id")
            tokenizer_display = None
            
            if tokenizer_id:
                try:
                    from aios.core.tokenizers import TokenizerRegistry
                    tokenizer_info = TokenizerRegistry.get(tokenizer_id)
                    if tokenizer_info:
                        tokenizer_display = tokenizer_info.name
                except Exception:
                    pass
                
                if not tokenizer_display:
                    tokenizer_display = tokenizer_id
            else:
                tokenizer_display = brain_meta.get("tokenizer_name", "Unknown")
            
            if "/" in tokenizer_display:
                tokenizer_display = tokenizer_display.split("/")[-1]
            
            panel.tokenizer_entry.config(state="normal")
            panel.tokenizer_entry.delete(0, "end")
            panel.tokenizer_entry.insert(0, tokenizer_display)
            panel.tokenizer_entry.config(state="readonly")
            
            # Calculate total params from architecture
            arch = brain_meta.get("arch", {})
            h_layers = arch.get("H_layers") or brain_meta.get("h_layers")
            l_layers = arch.get("L_layers") or brain_meta.get("l_layers")
            hidden_size = arch.get("hidden_size") or brain_meta.get("hidden_size")
            vocab_size = arch.get("vocab_size") or brain_meta.get("vocab_size")
            expansion = arch.get("expansion") or brain_meta.get("expansion") or 2.0
            
            # Prefer stored total_params (source of truth) over calculation
            total_params = brain_meta.get("total_params", 0)
            total_params_str = "-"
            
            if not total_params and h_layers and l_layers and hidden_size and vocab_size:
                # Calculate if not stored in brain.json (legacy brains)
                from aios.cli.hrm_hf.model_building import calculate_actv1_params
                total_params = calculate_actv1_params(
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    h_layers=h_layers,
                    l_layers=l_layers,
                    expansion=expansion,
                    use_moe=use_moe,
                    num_experts=num_experts if use_moe else 1
                )
            
            if total_params:
                if total_params >= 1_000_000_000:
                    total_params_str = f"{total_params / 1_000_000_000:.2f}B"
                elif total_params >= 1_000_000:
                    total_params_str = f"{total_params / 1_000_000:.2f}M"
                else:
                    total_params_str = f"{total_params:,.0f}"
            
            # Get file size and current params
            current_params = 0
            size_mb_str = "-"
            current_params_str = "-"
            if brain_dir:
                student_pt_path = os.path.join(brain_dir, brain_meta.get("checkpoint_file", "actv1_student.safetensors"))
                if os.path.exists(student_pt_path):
                    size_bytes = os.path.getsize(student_pt_path)
                    size_mb = size_bytes / (1024 * 1024)
                    size_mb_str = f"{size_mb:.2f}"
                    
                    current_params = size_bytes / 4
                    if current_params >= 1_000_000_000:
                        current_params_str = f"{current_params / 1_000_000_000:.2f}B"
                    elif current_params >= 1_000_000:
                        current_params_str = f"{current_params / 1_000_000:.2f}M"
                    else:
                        current_params_str = f"{current_params:,.0f}"
            
            # Get training steps
            training_steps = brain_meta.get("training_steps", 0)
            
            if training_steps == 0 and brain_dir:
                metrics_path = os.path.join(brain_dir, brain_meta.get("log_file", "metrics.jsonl"))
                if os.path.exists(metrics_path):
                    try:
                        with open(metrics_path, 'r') as f:
                            max_step = 0
                            for line in f:
                                try:
                                    entry = json.loads(line.strip())
                                    if "step" in entry:
                                        max_step = max(max_step, entry["step"])
                                except:
                                    continue
                            training_steps = max_step
                    except Exception:
                        pass
            
            trained_steps_str = f"{training_steps:,}" if training_steps else "0"
            
            # Combine params display: show current with total in parentheses if different
            # Use 5% tolerance to avoid showing duplicates due to rounding/calculation differences
            # (Increased from 1% to handle normal float32/file size calculation variances)
            if current_params and total_params:
                params_diff_pct = abs(current_params - total_params) / max(current_params, total_params)
                if params_diff_pct > 0.05:  # More than 5% difference
                    params_str = f"{current_params_str} ({total_params_str})"
                else:
                    # They're essentially the same, just show total (architectural params)
                    params_str = total_params_str
            elif current_params:
                params_str = current_params_str
            elif total_params:
                params_str = total_params_str
            else:
                params_str = "-"
            
            # Update stats fields
            panel.params_entry.config(state="normal")
            panel.params_entry.delete(0, "end")
            panel.params_entry.insert(0, params_str)
            panel.params_entry.config(state="readonly")
            
            panel.size_mb_entry.config(state="normal")
            panel.size_mb_entry.delete(0, "end")
            panel.size_mb_entry.insert(0, size_mb_str)
            panel.size_mb_entry.config(state="readonly")
            
            panel.trained_steps_entry.config(state="normal")
            panel.trained_steps_entry.delete(0, "end")
            panel.trained_steps_entry.insert(0, trained_steps_str)
            panel.trained_steps_entry.config(state="readonly")
        else:
            # Brain JSON not found - calculate from GUI values instead
            log(panel, f"[hrm] Brain JSON not found - calculating params from GUI values")
            
            # Try to calculate params from current GUI architecture settings
            try:
                h_layers = int(panel.h_layers_var.get())
                l_layers = int(panel.l_layers_var.get())
                hidden_size = int(panel.hidden_size_var.get())
                expansion = float(panel.expansion_var.get())
                
                # Use default vocab size to avoid slow tokenizer loading
                vocab_size = 50257  # Default GPT-2 vocab
                
                # Get MoE settings from readonly display fields (from brain.json)
                use_moe = False
                num_experts = 1
                try:
                    moe_experts_text = panel.moe_num_experts_entry.get()
                    if moe_experts_text and moe_experts_text != "N/A" and moe_experts_text != "-":
                        num_experts = int(moe_experts_text)
                        use_moe = True
                except:
                    pass
                
                # Calculate total params using shared function
                from aios.cli.hrm_hf.model_building import calculate_actv1_params
                total_params = calculate_actv1_params(
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    h_layers=h_layers,
                    l_layers=l_layers,
                    expansion=expansion,
                    use_moe=use_moe,
                    num_experts=num_experts
                )
                
                if total_params >= 1_000_000_000:
                    total_params_str = f"{total_params / 1_000_000_000:.2f}B"
                elif total_params >= 1_000_000:
                    total_params_str = f"{total_params / 1_000_000:.2f}M"
                else:
                    total_params_str = f"{total_params:,.0f}"
                
                # Calculate size in MB (4 bytes per param for float32)
                size_mb = (total_params * 4) / (1024 * 1024)
                size_mb_str = f"{size_mb:.2f}"
                
                # Update params and size fields
                panel.params_entry.config(state="normal")
                panel.params_entry.delete(0, "end")
                panel.params_entry.insert(0, total_params_str)
                panel.params_entry.config(state="readonly")
                
                panel.size_mb_entry.config(state="normal")
                panel.size_mb_entry.delete(0, "end")
                panel.size_mb_entry.insert(0, size_mb_str)
                panel.size_mb_entry.config(state="readonly")
                
                # Clear steps field since no brain exists yet
                panel.trained_steps_entry.config(state="normal")
                panel.trained_steps_entry.delete(0, "end")
                panel.trained_steps_entry.insert(0, "-")
                panel.trained_steps_entry.config(state="readonly")
                
                log(panel, f"[hrm] Calculated params from GUI: {total_params_str} ({size_mb_str} MB)")
            except Exception as e:
                log(panel, f"[hrm] Could not calculate params from GUI: {e}")
                _clear_stats_display(panel)
    except Exception as e:
        try:
            # Log error if panel has _log method
            if hasattr(panel, '_log') and callable(panel._log):
                panel._log(f"[hrm] Error updating MoE stats: {e}")
                import traceback
                panel._log(f"[hrm] Traceback: {traceback.format_exc()}")
        except:
            pass
        _clear_stats_display(panel)


def _clear_stats_display(panel: HRMTrainingPanel) -> None:  # pyright: ignore[reportAttributeAccessIssue]
    """Clear all stats display fields.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    try:
        for entry in [panel.moe_num_experts_entry, panel.moe_active_experts_entry, 
                      panel.tokenizer_entry, panel.params_entry, 
                      panel.size_mb_entry, panel.trained_steps_entry]:
            entry.config(state="normal")
            entry.delete(0, "end")
            entry.insert(0, "-")
            entry.config(state="readonly")
    except Exception:
        pass
