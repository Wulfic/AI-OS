"""Training configuration builder for HRM Training Panel.

Builds TrainingConfig from GUI state with proper device/CUDA/DDP detection.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
import logging

from .path_defaults import get_default_bundle_dir, get_default_model_base

if TYPE_CHECKING:
    from .panel_main import HRMTrainingPanel

logger = logging.getLogger(__name__)


def build_training_config(panel: HRMTrainingPanel) -> Any:
    """Build a TrainingConfig object from current GUI state.
    
    Args:
        panel: The HRMTrainingPanel instance
        
    Returns:
        TrainingConfig: Configuration object ready for training
    """
    from aios.core.hrm_training import TrainingConfig
    from .helpers import log, get_moe_num_experts, get_moe_active_experts
    
    # Helper to safely get int value
    def _int(var, default=0):
        try:
            return int(var.get().strip() or default)
        except Exception:
            return default
    
    # Helper to safely get float value
    def _float(var, default=0.0):
        try:
            return float(var.get().strip() or default)
        except Exception:
            return default
    
    # Helper to safely get string value
    def _str(var, default=""):
        try:
            return var.get().strip() or default
        except Exception:
            return default
    
    # Helper to safely get boolean from a variable that might not exist
    def _bool(attr_name: str, default: bool = False) -> bool:
        var = getattr(panel, attr_name, None)
        if var is None:
            return default
        try:
            return bool(var.get())
        except Exception:
            return default
    
    # Helper to safely get window size (works with SDPA or FlashAttention)
    # If the entry exists, honor it regardless of FlashAttention toggle so that
    # PyTorch scaled_dot_product_attention can exploit a sliding window on all platforms.
    def _get_window_size(panel):
        window_var = getattr(panel, "flash_attn_window_var", None)
        if window_var is None:
            return None
        try:
            val = window_var.get().strip()
            return int(val) if val else None
        except Exception:
            return None
    
    # Resolve adaptive LR profile from dropdown (fallback to legacy boolean)
    adaptive_mode_label = ""
    try:
        adaptive_mode_label = getattr(panel, "adaptive_lr_mode_var", panel.lr_var).get().strip()
    except Exception:
        adaptive_mode_label = ""

    profiles_by_label = getattr(panel, "_adaptive_lr_profiles_by_label", {}) or {}
    selected_profile = profiles_by_label.get(adaptive_mode_label)

    # Off => disable adaptive LR; everything else enables it.
    auto_adjust_value = True
    if adaptive_mode_label.strip().lower() in {"off", "none", "disabled"}:
        auto_adjust_value = False
    logger.debug(
        "Building config: adaptive_lr_mode=%r derived auto_adjust_lr=%r",
        adaptive_mode_label,
        auto_adjust_value,
    )

    adaptive_lr_config_path = None
    try:
        if auto_adjust_value and selected_profile is not None:
            from .adaptive_lr_profiles import resolve_profile_config_path

            adaptive_lr_config_path = resolve_profile_config_path(selected_profile)
    except Exception:
        adaptive_lr_config_path = None
    
    # Get device from resources panel if available
    device = "auto"
    try:
        rp = getattr(panel, "_resources_panel", None)
        if rp is not None:
            rvals = rp.get_values()
            td = str(rvals.get("train_device") or "auto").lower()
            if td in {"cpu", "cuda", "xpu", "mps", "dml"}:
                device = td
            elif isinstance(rvals.get("train_cuda_selected"), list) and len(rvals.get("train_cuda_selected", [])) > 0:
                device = "cuda"
    except Exception:
        pass
    
    # Get CUDA IDs and DDP/Parallel settings from resources panel
    cuda_ids = None
    ddp = False
    parallel_independent = False
    world_size = None
    try:
        rp = getattr(panel, "_resources_panel", None)
        logger.debug(f"Resources panel found: {rp is not None}")
        if rp is not None:
            rvals = rp.get_values()
            sel_train = rvals.get("train_cuda_selected") or []
            training_mode = rvals.get("training_mode", "none")  # Get training mode from resources
            logger.debug(f"train_cuda_selected from resources: {sel_train}")
            logger.debug(f"training_mode from resources: {training_mode}")
            if isinstance(sel_train, list) and len(sel_train) > 0:
                cuda_ids = ",".join(str(int(i)) for i in sel_train)
                if len(sel_train) > 1:
                    # Use training mode to determine backend coordination
                    if training_mode == "parallel":
                        parallel_independent = True
                        ddp = False
                    elif training_mode == "zero3":
                        parallel_independent = True
                        ddp = False
                    elif training_mode == "none":
                        parallel_independent = False
                        ddp = False
                    else:
                        ddp = True
                        parallel_independent = False
                    world_size = len(sel_train) if ddp else None
                logger.debug(f"Multi-GPU config: cuda_ids={cuda_ids}, mode={training_mode}, ddp={ddp}, parallel={parallel_independent}, world_size={world_size}")
            else:
                logger.debug("No GPUs selected or invalid selection")
    except Exception as e:
        logger.error(f"Error getting GPU config: {e}", exc_info=True)
        pass
    
    # Get sys_mem_cap_pct from resources CPU util
    sys_mem_cap_pct = None
    try:
        rp = getattr(panel, "_resources_panel", None)
        if rp is not None:
            rvals = rp.get_values()
            cap = int(rvals.get("cpu_util_pct") or 0)
            if cap > 0:
                sys_mem_cap_pct = cap
    except Exception:
        pass
    
    # Fix old HuggingFace dataset format (hf://dataset:split -> hf://dataset:config:split)
    dataset_file = _str(panel.dataset_var)
    if dataset_file and dataset_file.startswith("hf://"):
        parts = dataset_file[5:].split(":")
        if len(parts) == 2:
            # Old format: hf://dataset:split -> Fix to hf://dataset:default:split
            dataset_path, split = parts
            dataset_file = f"hf://{dataset_path}:default:{split}"
            panel.dataset_var.set(dataset_file)  # Update for next time
            log(panel, f"[hrm] Auto-corrected dataset format to: {dataset_file}")
    
    # Check for incompatible DDP + ZeRO-3 configuration
    zero_stage_val = _str(panel.zero_stage_var, "none")
    try:
        rp_zero = None
        if rp is not None and hasattr(rp, "zero_stage_var"):
            rp_zero = rp.zero_stage_var.get()
        elif rp is not None:
            rvals = rp.get_values()
            rp_zero = rvals.get("zero_stage")
        if isinstance(rp_zero, str) and rp_zero.strip() == "zero3":
            zero_stage_val = "zero3"
    except Exception:
        pass
    logger.debug(f"Before ZeRO check: ddp={ddp}, world_size={world_size}, zero_stage={zero_stage_val}")
    if zero_stage_val == "zero3":
        if ddp:
            log(panel, "[hrm] ZeRO-3 requested with DDP enabled; ensure DeepSpeed orchestration supports this path.")
            logger.debug("ZeRO-3 requested alongside DDP; downstream components must handle compatibility.")
        elif parallel_independent:
            log(panel, "[hrm] ZeRO-3 standalone mode selected; training will run without DDP while DeepSpeed handles stage-3 sharding.")
            logger.debug("ZeRO-3 standalone mode active (parallel independent path)")
        else:
            log(panel, "[hrm] ZeRO-3 mode selected on a single GPU; training will run without DDP.")
            logger.debug("ZeRO-3 single-GPU path active (no DDP)")
    logger.debug(f"After ZeRO check: ddp={ddp}, world_size={world_size}, cuda_ids={cuda_ids}")
    
    # Build config
    default_bundle = str(get_default_bundle_dir())
    config = TrainingConfig(
        model=_str(panel.model_var, str(get_default_model_base())),
        dataset_file=dataset_file,
        dataset_chunk_size=_int(panel.dataset_chunk_size_var, 4000),
        max_seq_len=_int(panel.max_seq_var, 128),
        batch_size=_int(panel.batch_var, 4),
        gradient_accumulation_steps=_int(panel.gradient_accumulation_var, 1),
        steps=_int(panel.steps_var, 100),
        lr=_float(panel.lr_var, 5e-5),
        auto_adjust_lr=auto_adjust_value,
        adaptive_lr_config=adaptive_lr_config_path,
        device=device,
        halt_max_steps=_int(panel.halt_steps_var, 1),
        save_dir=_str(panel.bundle_dir_var, default_bundle),
        ascii_only=bool(panel.ascii_only_var.get()),
        linear_dataset=bool(panel.linear_dataset_var.get()),
        dataset_start_offset=0,  # GUI always starts from 0; resume handled by checkpoint loading
        sys_mem_cap_pct=sys_mem_cap_pct,
        stop_file=_str(panel.stop_file_var) or None,
        log_file=_str(panel.log_file_var) or None,
        student_init=_str(panel.student_init_var) or None,
        brain_name=_str(panel.brain_name_var) or None,
        default_goal=_str(panel.default_goal_var) or None,
        bundle_dir=_str(panel.bundle_dir_var, default_bundle),
        h_layers=_int(panel.h_layers_var, 2),
        l_layers=_int(panel.l_layers_var, 2),
        hidden_size=_int(panel.hidden_size_var, 512),
        expansion=_float(panel.expansion_var, 2.0),
        num_heads=_int(panel.num_heads_var, 8),
        h_cycles=_int(panel.h_cycles_var, 2),
        l_cycles=_int(panel.l_cycles_var, 2),
        pos_encodings=_str(panel.pos_enc_var, "rope"),
        
        # MoE (Mixture of Experts) configuration
        use_moe=get_moe_num_experts(panel) > 1,
        num_experts=get_moe_num_experts(panel),
        num_experts_per_tok=get_moe_active_experts(panel),
        moe_capacity_factor=1.25,  # Default
        
        cuda_ids=cuda_ids,
        iterate=_bool("iterate_var", False),
        stop_after_block=_bool("stop_after_block_var", False),
        stop_after_epoch=_bool("stop_after_epoch_var", False),
        optimize=False,  # Never auto-optimize during training (use Optimize button instead)
        parallel_independent=parallel_independent,  # Use parallel independent training mode
        gradient_checkpointing=bool(panel.gradient_checkpointing_var.get()),
        use_amp=_bool("use_amp_var", True),
        use_cpu_offload=_bool("use_cpu_offload_var", False),
        use_8bit_optimizer=_bool("use_8bit_optimizer_var", False),
        use_chunked_training=_bool("use_chunked_training_var", False),
        chunk_size=_int(panel.chunk_size_var, 2048),
        use_flash_attn=_bool("use_flash_attn_var", False),
        window_size=_get_window_size(panel),
        
        # PEFT options
        use_peft=_bool("use_peft_var", False),
        peft_method=_str(panel.peft_method_var, "lora"),
        lora_r=_int(panel.lora_r_var, 16),
        lora_alpha=_int(panel.lora_alpha_var, 32),
        lora_dropout=_float(panel.lora_dropout_var, 0.05),
        lora_target_modules=_str(panel.lora_target_modules_var, "q_proj,v_proj"),
        
        zero_stage=_str(panel.zero_stage_var, "none"),
        ddp=ddp,
        world_size=world_size,
        strict=True,  # Always strict mode in GUI
    )
    
    return config
