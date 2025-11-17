"""Memory estimation logic for HRM Training Panel.

Handles VRAM/RAM estimation, MoE stats display, and model parameter calculation.

Note: Panel attributes (h_layers_var, moe_num_experts_entry, etc.) are added 
dynamically by variable_setup.py and ui_builder modules at runtime.
Type checker cannot see these attributes statically.
"""
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations
import json
import logging
import os
import threading
import traceback
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    from .panel_main import HRMTrainingPanel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _VramInputs:
    batch_size: int
    seq_len: int
    total_params: int
    hidden_size: int
    h_layers: int
    l_layers: int
    num_layers: int
    num_heads: int
    use_amp: bool
    use_gradient_checkpointing: bool
    use_lora: bool
    lora_r: int
    use_cpu_offload: bool
    use_8bit_optimizer: bool
    zero_stage: str
    use_chunking: bool
    num_gpus: int
    chunk_size: int


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
    """Update memory estimates (VRAM and RAM) using background worker threads."""

    if threading.current_thread() is not threading.main_thread():
        if not _dispatch_to_ui(panel, lambda: update_vram_estimate(panel)):
            logger.debug("Failed to dispatch VRAM estimate to UI thread")
        return

    if hasattr(panel, "_vram_update_after_id"):
        panel._vram_update_after_id = None

    inputs = _collect_vram_inputs(panel)
    if inputs is None:
        _set_vram_placeholders(panel)
        return

    if inputs.total_params <= 0:
        logger.warning("Cannot estimate VRAM: total_params=0 (invalid configuration)")
        _set_vram_placeholders(panel)
        return

    pool = getattr(panel, "_worker_pool", None)
    panel._vram_task_id = getattr(panel, "_vram_task_id", 0) + 1
    task_id = panel._vram_task_id

    logger.debug(
        "Submitting VRAM estimation task (batch=%s, seq_len=%s, params=%s, task_id=%s)",
        inputs.batch_size,
        inputs.seq_len,
        f"{inputs.total_params/1e6:.1f}M",
        task_id,
    )

    try:
        if pool is not None and not getattr(pool, "is_shutdown", False):
            future = pool.submit(_compute_vram_summary, inputs)
        else:
            future = Future()

            def _run_sync() -> None:
                try:
                    future.set_result(_compute_vram_summary(inputs))
                except Exception as exc:  # pragma: no cover - defensive
                    future.set_exception(exc)

            threading.Thread(target=_run_sync, name="VRAMEstimate", daemon=True).start()
    except Exception as submit_exc:  # pragma: no cover - defensive
        logger.error("Failed to submit VRAM estimate task", exc_info=True)
        trace = "".join(traceback.format_exception(type(submit_exc), submit_exc, submit_exc.__traceback__))
        _set_vram_placeholders(panel)
        _log_panel_error(panel, submit_exc, trace)
        return

    panel._vram_estimate_future = future

    def _on_complete(done: Future) -> None:
        try:
            summary = done.result()
            error: Optional[Exception] = None
            trace_str: Optional[str] = None
        except Exception as exc:  # pragma: no cover - defensive
            summary = None
            error = exc
            trace_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

        def _apply_result() -> None:
            if getattr(panel, "_vram_task_id", 0) != task_id:
                logger.debug("Ignoring stale VRAM result (task_id=%s)", task_id)
                return

            if getattr(panel, "_vram_estimate_future", None) is done:
                panel._vram_estimate_future = None

            if error is not None or summary is None:
                logger.warning("VRAM estimation failed for task_id=%s: %s", task_id, error)
                _set_vram_placeholders(panel)
                if error is not None:
                    _log_panel_error(panel, error, trace_str)
                return

            _apply_vram_summary(panel, summary, inputs)

        if not _dispatch_to_ui(panel, _apply_result):
            logger.debug("Failed to dispatch VRAM result to UI thread")

    future.add_done_callback(_on_complete)


def _collect_vram_inputs(panel: HRMTrainingPanel) -> Optional[_VramInputs]:  # pyright: ignore[reportAttributeAccessIssue]
    try:
        def _parse_int(value: Any, default: int) -> int:
            try:
                sval = str(value).strip()
                return int(sval or default)
            except Exception:
                return default

        batch_size = max(1, _parse_int(getattr(panel.batch_var, "get", lambda: 4)(), 4))
        seq_len = max(1, _parse_int(getattr(panel.max_seq_var, "get", lambda: 128)(), 128))
        total_params = estimate_model_params(panel)
        hidden_size = max(1, _parse_int(getattr(panel.hidden_size_var, "get", lambda: 512)(), 512))
        h_layers = max(0, _parse_int(getattr(panel.h_layers_var, "get", lambda: 2)(), 2))
        l_layers = max(0, _parse_int(getattr(panel.l_layers_var, "get", lambda: 2)(), 2))
        num_layers = max(1, h_layers + l_layers)
        num_heads = max(1, _parse_int(getattr(panel.num_heads_var, "get", lambda: 8)(), 8))

        def _get_bool(var_name: str, default: bool) -> bool:
            var = getattr(panel, var_name, None)
            if var is None or not hasattr(var, "get"):
                return default
            try:
                return bool(var.get())
            except Exception:
                return default

        use_amp = _get_bool("use_amp_var", True)
        use_gradient_checkpointing = _get_bool("gradient_checkpointing_var", True)
        use_lora = _get_bool("use_peft_var", False)
        lora_r = max(1, _parse_int(getattr(panel.lora_r_var, "get", lambda: 16)(), 16)) if use_lora else 16
        use_cpu_offload = _get_bool("use_cpu_offload_var", False)
        use_8bit_optimizer = _get_bool("use_8bit_optimizer_var", False)
        zero_stage = getattr(panel, "zero_stage_var", None)
        zero_stage_val = zero_stage.get() if zero_stage is not None and hasattr(zero_stage, "get") else "none"

        chunking_requested = _get_bool("use_chunked_training_var", False)
        use_chunking = chunking_requested or seq_len > 8192
        chunk_var = getattr(panel, "chunk_size_var", None)
        chunk_var_value = chunk_var.get() if chunk_var is not None and hasattr(chunk_var, "get") else seq_len
        chunk_size_candidate = _parse_int(chunk_var_value, seq_len)
        if not use_chunking:
            chunk_size = seq_len
        else:
            if chunk_size_candidate <= 0:
                chunk_size = get_effective_chunk_size(total_params, seq_len)
            else:
                chunk_size = max(1, chunk_size_candidate)

        num_gpus = 1
        try:
            resources_panel = getattr(panel, "_resources_panel", None)
            if resources_panel is not None and hasattr(resources_panel, "get_values"):
                rvals = resources_panel.get_values()
                sel_train = rvals.get("train_cuda_selected") or []
                if isinstance(sel_train, list) and sel_train:
                    num_gpus = len(sel_train)
        except Exception:
            logger.debug("Failed to read GPU selection for VRAM inputs", exc_info=True)

        logger.debug(
            "Collected VRAM inputs: batch=%s, seq_len=%s, params=%s, layers=%s, heads=%s, chunking=%s",
            batch_size,
            seq_len,
            total_params,
            num_layers,
            num_heads,
            use_chunking,
        )

        return _VramInputs(
            batch_size=batch_size,
            seq_len=seq_len,
            total_params=total_params,
            hidden_size=hidden_size,
            h_layers=h_layers,
            l_layers=l_layers,
            num_layers=num_layers,
            num_heads=num_heads,
            use_amp=use_amp,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_lora=use_lora,
            lora_r=lora_r,
            use_cpu_offload=use_cpu_offload,
            use_8bit_optimizer=use_8bit_optimizer,
            zero_stage=zero_stage_val,
            use_chunking=use_chunking,
            num_gpus=num_gpus,
            chunk_size=chunk_size,
        )
    except Exception:
        logger.debug("Failed to collect VRAM inputs", exc_info=True)
        return None


def _compute_vram_summary(inputs: _VramInputs) -> Dict[str, Any]:
    from ..hrm_training.memory_estimator import MemoryEstimator

    estimator = MemoryEstimator(
        total_params=inputs.total_params,
        hidden_size=inputs.hidden_size,
        num_layers=inputs.num_layers,
        num_heads=inputs.num_heads,
        batch_size=inputs.batch_size,
        seq_len=inputs.seq_len,
        num_gpus=inputs.num_gpus,
        use_amp=inputs.use_amp,
        use_gradient_checkpointing=inputs.use_gradient_checkpointing,
        use_lora=inputs.use_lora,
        lora_r=inputs.lora_r,
        use_8bit_optimizer=inputs.use_8bit_optimizer,
        offload_optimizer=inputs.use_cpu_offload,
        zero_stage=inputs.zero_stage,
        use_chunking=inputs.use_chunking,
        chunk_size=inputs.chunk_size if inputs.use_chunking else None,
    )

    return estimator.get_summary()


def _apply_vram_summary(panel: HRMTrainingPanel, summary: Dict[str, Any], inputs: _VramInputs) -> None:  # pyright: ignore[reportAttributeAccessIssue]
    try:
        vram = summary.get("vram", {})
        ram = summary.get("ram", {})

        model_gb = float(vram.get("model_gb", 0.0))
        optimizer_gb = float(vram.get("optimizer_gb", 0.0))
        gradients_gb = float(vram.get("gradients_gb", 0.0))
        activations_gb = float(vram.get("activations_gb", 0.0))
        overhead_gb = float(vram.get("overhead_gb", 0.0))
        act_grad_total = gradients_gb + activations_gb
        per_gpu_vram_gb = float(vram.get("total_gb", 0.0))

        panel.vram_model_lbl.config(text=f"{model_gb:.2f} GB")
        panel.vram_optimizer_lbl.config(text=f"{optimizer_gb:.2f} GB")
        panel.vram_activations_lbl.config(text=f"{act_grad_total:.2f} GB")

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

        panel.vram_total_lbl.config(text=f"{per_gpu_vram_gb:.2f} GB (wip)", foreground=color)

        dataset_ram_gb = float(ram.get("dataset_gb", 0.0))
        optimizer_ram_gb = float(ram.get("optimizer_gb", 0.0))
        total_ram_gb = float(ram.get("total_gb", 0.0))
        panel.ram_dataset_lbl.config(text=f"{dataset_ram_gb:.2f} GB")
        panel.ram_offload_lbl.config(text=f"{optimizer_ram_gb:.2f} GB")
        panel.ram_total_lbl.config(text=f"{total_ram_gb:.1f} GB")

        update_moe_stats_display(panel)

        logger.info("VRAM estimate: %.2f GB per GPU (%s GPU(s))", per_gpu_vram_gb, inputs.num_gpus)
        logger.debug(
            "VRAM breakdown: model=%.2fGB, optimizer=%.2fGB, activations+grads=%.2fGB, overhead=%.2fGB",
            model_gb,
            optimizer_gb,
            act_grad_total,
            overhead_gb,
        )

        if per_gpu_vram_gb > 24:
            logger.warning("Estimate exceeds available VRAM on most GPUs: %.2f GB", per_gpu_vram_gb)

        _apply_tooltips(panel, summary, inputs, per_gpu_vram_gb, recommendation)
        _log_long_seq_warning(panel, inputs)
    except Exception:
        logger.error("Failed to apply VRAM summary", exc_info=True)
        _set_vram_placeholders(panel)


def _apply_tooltips(
    panel: HRMTrainingPanel,
    summary: Dict[str, Any],
    inputs: _VramInputs,
    per_gpu_vram_gb: float,
    recommendation: str,
) -> None:
    try:
        from ..tooltips import add_tooltip
    except Exception:
        return

    vram = summary.get("vram", {})
    ram = summary.get("ram", {})
    breakdown = vram.get("breakdown", {})
    cfg = summary.get("config", {})

    trainable_params = float(breakdown.get("trainable_params", inputs.total_params))
    effective_seq = int(breakdown.get("effective_seq", inputs.chunk_size if inputs.use_chunking else inputs.seq_len))

    gpu_mode = "Single GPU"
    zero_stage = cfg.get("zero_stage", inputs.zero_stage)
    if inputs.num_gpus > 1:
        if isinstance(zero_stage, str) and zero_stage != "none":
            gpu_mode = f"DDP mode with {zero_stage.upper()} (memory partitioned)"
        else:
            gpu_mode = f"Parallel mode ({inputs.num_gpus} independent instances)"

    tooltip_lines = [
        "══════ MEMORY ESTIMATE ══════",
        "",
        f"🎯 VRAM per GPU: {per_gpu_vram_gb:.2f} GB",
        f"   • Model: {float(vram.get('model_gb', 0.0)):.2f} GB",
        f"   • Optimizer: {float(vram.get('optimizer_gb', 0.0)):.2f} GB",
        f"   • Gradients: {float(vram.get('gradients_gb', 0.0)):.2f} GB",
        f"   • Activations: {float(vram.get('activations_gb', 0.0)):.2f} GB",
        f"   • Overhead: {float(vram.get('overhead_gb', 0.0)):.2f} GB",
        "",
        f"💾 System RAM: {float(ram.get('total_gb', 0.0)):.1f} GB",
        f"   • Dataset: {float(ram.get('dataset_gb', 0.0)):.2f} GB",
        f"   • CPU Offload: {float(ram.get('optimizer_gb', 0.0)):.2f} GB",
        f"   • PyTorch: {float(ram.get('pytorch_gb', 0.0)):.2f} GB",
        "",
        "⚙️  Active Optimizations:",
        f"   • AMP (FP16): {'✓' if cfg.get('use_amp', inputs.use_amp) else '✗'}",
        f"   • Gradient Checkpointing: {'✓' if cfg.get('use_gradient_checkpointing', inputs.use_gradient_checkpointing) else '✗'}",
        f"   • LoRA/PEFT: {'✓' if cfg.get('use_lora', inputs.use_lora) else '✗'}",
        f"   • CPU Offload: {'✓' if cfg.get('offload_optimizer', inputs.use_cpu_offload) else '✗'}",
        f"   • 8-bit Optimizer: {'✓' if cfg.get('use_8bit_optimizer', inputs.use_8bit_optimizer) else '✗'}",
        f"   • DeepSpeed ZeRO: {zero_stage}",
        f"   • Chunked Training: {'✓' if cfg.get('use_chunking', inputs.use_chunking) else '✗'}",
        "",
        "📊 Configuration:",
        f"   • Total params: {inputs.total_params/1e6:.1f}M",
        f"   • Trainable params: {trainable_params/1e6:.1f}M",
        f"   • Batch size: {inputs.batch_size}",
        f"   • Sequence length: {inputs.seq_len}",
        f"   • Effective chunk: {effective_seq}",
        f"   • Multi-GPU: {gpu_mode}",
        "",
        f"💡 {recommendation}",
    ]

    if per_gpu_vram_gb > 12:
        tooltip_lines.extend(["", "🔧 Suggestions to reduce VRAM:"])
        if not cfg.get("use_amp", inputs.use_amp):
            tooltip_lines.append("   • Enable AMP → Save ~40%")
        if not cfg.get("use_gradient_checkpointing", inputs.use_gradient_checkpointing):
            tooltip_lines.append("   • Enable Grad Checkpoint → Save ~60% activations")
        if not cfg.get("use_lora", inputs.use_lora):
            tooltip_lines.append("   • Enable LoRA → Save ~99% optimizer/gradients")
        if not cfg.get("offload_optimizer", inputs.use_cpu_offload) and float(vram.get("optimizer_gb", 0.0)) > 0:
            tooltip_lines.append(f"   • Enable CPU Offload → Move {float(vram.get('optimizer_gb', 0.0)):.1f} GB to RAM")
        if inputs.batch_size > 1:
            tooltip_lines.append("   • Reduce batch size → Direct VRAM savings")

    add_tooltip(panel.vram_total_lbl, "\n".join(tooltip_lines))

    ram_tooltip = [
        "══════ SYSTEM RAM ══════",
        "",
        f"Total: {float(ram.get('total_gb', 0.0)):.1f} GB",
        "",
        "Breakdown:",
        f"  • Dataset buffer: {float(ram.get('dataset_gb', 0.0)):.2f} GB",
        f"  • CPU offloaded optimizer: {float(ram.get('optimizer_gb', 0.0)):.2f} GB",
        f"  • PyTorch/Python: {float(ram.get('pytorch_gb', 0.0)):.2f} GB",
        "",
        "💡 Enable CPU Offload to move optimizer",
        "   state from VRAM to RAM (slower but saves VRAM)",
    ]
    add_tooltip(panel.ram_total_lbl, "\n".join(ram_tooltip))


def _set_vram_placeholders(panel: HRMTrainingPanel) -> None:  # pyright: ignore[reportAttributeAccessIssue]
    try:
        panel.vram_model_lbl.config(text="-")
        panel.vram_optimizer_lbl.config(text="-")
        panel.vram_activations_lbl.config(text="-")
        panel.vram_total_lbl.config(text="- (wip)", foreground="black")
        panel.ram_dataset_lbl.config(text="-")
        panel.ram_offload_lbl.config(text="-")
        panel.ram_total_lbl.config(text="-")
    except Exception:
        pass


def _log_panel_error(panel: HRMTrainingPanel, error: Exception | str, trace: Optional[str]) -> None:  # pyright: ignore[reportAttributeAccessIssue]
    try:
        from .helpers import log

        log(panel, f"[hrm] Memory estimation error: {error}")
        if trace:
            log(panel, f"[hrm] Traceback: {trace}")
    except Exception:
        pass


def _log_long_seq_warning(panel: HRMTrainingPanel, inputs: _VramInputs) -> None:  # pyright: ignore[reportAttributeAccessIssue]
    if inputs.seq_len <= 2048 or inputs.use_amp:
        panel._vram_warned_long_seq = None
        return

    key = (inputs.seq_len, inputs.use_amp)
    if getattr(panel, "_vram_warned_long_seq", None) == key:
        return

    panel._vram_warned_long_seq = key

    try:
        from .helpers import log

        attention_gb = (inputs.seq_len * inputs.seq_len * 4 * inputs.num_layers) / 1e9
        log(panel, f"[hrm] ⚠️  WARNING: Long sequence ({inputs.seq_len} tokens) without AMP will use massive memory!")
        log(panel, "[hrm] 💡 STRONGLY RECOMMENDED: Enable AMP to reduce memory by ~50%")
        log(panel, f"[hrm] FP32 attention memory for seq_len={inputs.seq_len}: ~{attention_gb:.1f} GB per batch!")
    except Exception:
        pass


def _dispatch_to_ui(panel: HRMTrainingPanel, callback: Callable[[], None]) -> bool:  # pyright: ignore[reportAttributeAccessIssue]
    dispatcher = getattr(panel, "dispatch_to_ui", None)
    if callable(dispatcher):
        try:
            return bool(dispatcher(callback))
        except Exception:
            logger.debug("dispatch_to_ui failed", exc_info=True)
    return False


def update_moe_stats_display(panel: HRMTrainingPanel) -> None:  # pyright: ignore[reportAttributeAccessIssue]
    """Update MoE stats, tokenizer, and model stats from brain.json.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    if threading.current_thread() is not threading.main_thread():
        try:
            panel.after(0, lambda: update_moe_stats_display(panel))
        except Exception as schedule_err:
            logger.warning("Failed to schedule MoE stats update on main thread: %s", schedule_err)
        return
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
