from __future__ import annotations

from typing import Optional
from pathlib import Path

import typer

from aios.core.hrm_training.training_config import TrainingConfig
from .hrm_hf.implant import implant_brain_impl as _implant_brain_impl
from .hrm_hf.starter_brain import (
    save_starter_brain_impl as _save_starter_brain_impl,
    load_starter_brain_impl as _load_starter_brain_impl,
)
from .hrm_hf.arch_summary import arch_summary_impl as _arch_summary_impl
from .hrm_hf.train_actv1 import train_actv1_impl as _train_actv1_impl
from .hrm_hf.preprocess_dataset_cmd import preprocess_dataset_cmd as _preprocess_dataset_cmd

try:  # pragma: no cover - fallback for bootstrapping
    from aios.system import paths as system_paths
except Exception:  # pragma: no cover
    system_paths = None


def _resolve_artifact(relative: str) -> str:
    if system_paths is not None:
        return str(system_paths.resolve_artifact_path(relative))
    return relative


def _default_bundle_dir() -> str:
    if system_paths is not None:
        return str(system_paths.get_brain_family_dir("actv1"))
    fallback = Path(__file__).resolve().parents[3] / "artifacts" / "brains" / "actv1"
    return str(fallback.resolve())


def _default_model_dir() -> str:
    return _resolve_artifact("hf_implant/base_model")


def _default_implant_save_dir() -> str:
    return _resolve_artifact("hf_implant")


def _default_starter_dir() -> str:
    return _resolve_artifact("hf_starter")


def _default_training_save_dir() -> str:
    return _resolve_artifact("training_datasets/actv1")


app = typer.Typer(help="Implant a pretrained HF LLM as the HRM 'brain' and adapt it.")


@app.command("implant")
def implant_brain(
    model: str = typer.Option("base_model", "--model", help="HF model name or path"),
    dataset_file: str = typer.Option(..., "--dataset-file", help="Path to text/CSV/archive/dir to sample lines from"),
    max_seq_len: int = typer.Option(128, "--max-seq-len", help="Token sequence length"),
    batch_size: int = typer.Option(4, "--batch-size", help="Batch size"),
    steps: int = typer.Option(100, "--steps", help="Optimization steps"),
    lr: float = typer.Option(5e-5, "--lr", help="Learning rate"),
    device: str = typer.Option("auto", "--device", help="Device: auto|cpu|cuda"),
    save_dir: Optional[str] = typer.Option(_default_implant_save_dir(), "--save-dir", help="Where to save adapter (and model if fine-tuned)"),
    train_lm: bool = typer.Option(False, "--train-lm/--freeze-lm", help="If true, fine-tune the base LM too (VRAM heavy)"),
    halt_max_steps: int = typer.Option(1, "--halt-max-steps", help="Max ACT segments during training"),
    # LoRA knobs
    use_lora: bool = typer.Option(False, "--lora/--no-lora", help="Enable LoRA for parameter-efficient finetuning"),
    lora_r: int = typer.Option(16, "--lora-r"),
    lora_alpha: int = typer.Option(32, "--lora-alpha"),
    lora_dropout: float = typer.Option(0.05, "--lora-drop"),
    lora_target: str = typer.Option("c_attn,c_proj", "--lora-target", help="Comma-separated target module names"),
):
    """Implant a pretrained HF LLM as the HRM 'brain' and adapt it."""
    return _implant_brain_impl(
        model=model,
        dataset_file=dataset_file,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        steps=steps,
        lr=lr,
        device=device,
        save_dir=save_dir,
        train_lm=train_lm,
        halt_max_steps=halt_max_steps,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target=lora_target,
    )


@app.command("save-starter-brain")
def save_starter_brain(
    model: str = typer.Option("base_model", "--model", help="HF model name or path (or existing fine-tuned dir)"),
    out_dir: str = typer.Option(_default_starter_dir(), "--out-dir", help="Output dir for starter brain"),
    max_seq_len: int = typer.Option(128, "--max-seq-len"),
    halt_max_steps: int = typer.Option(1, "--halt-max-steps"),
    save_model: bool = typer.Option(False, "--include-model/--no-include-model"),
    device: str = typer.Option("auto", "--device"),
    # LoRA knobs (if saving adapters)
    use_lora: bool = typer.Option(False, "--lora/--no-lora"),
    lora_r: int = typer.Option(16, "--lora-r"),
    lora_alpha: int = typer.Option(32, "--lora-alpha"),
    lora_dropout: float = typer.Option(0.05, "--lora-drop"),
    lora_target: str = typer.Option("c_attn,c_proj", "--lora-target"),
):
    """Create a starter-brain bundle (JSON + q_head, optionally model weights)."""
    return _save_starter_brain_impl(
        model=model,
        out_dir=out_dir,
        max_seq_len=max_seq_len,
        halt_max_steps=halt_max_steps,
        save_model=save_model,
        device=device,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target=lora_target,
    )


@app.command("load-starter-brain")
def load_starter_brain(
    config: str = typer.Option(..., "--config", help="Path to starter_brain.json"),
):
    """Load a starter brain config and print a brief summary (dry-run check)."""
    return _load_starter_brain_impl(config=config)


@app.command("arch-summary")
def arch_summary(
    model: str = typer.Option("base_model", "--model", help="HF model name or local dir for tokenizer (for vocab size)"),
    max_seq_len: int = typer.Option(128, "--max-seq-len"),
    halt_max_steps: int = typer.Option(2, "--halt-max-steps"),
    # Architecture knobs
    h_layers: int = typer.Option(2, "--h-layers"),
    l_layers: int = typer.Option(2, "--l-layers"),
    hidden_size: int = typer.Option(512, "--hidden-size"),
    expansion: float = typer.Option(2.0, "--expansion"),
    num_heads: int = typer.Option(8, "--num-heads"),
    h_cycles: int = typer.Option(2, "--h-cycles"),
    l_cycles: int = typer.Option(2, "--l-cycles"),
    pos_encodings: str = typer.Option("rope", "--pos-encodings"),
):
    """Print parameter counts for the ACT V1 student given an architecture, without training."""
    return _arch_summary_impl(
        model=model,
        max_seq_len=max_seq_len,
        halt_max_steps=halt_max_steps,
        h_layers=h_layers,
        l_layers=l_layers,
        hidden_size=hidden_size,
        expansion=expansion,
        num_heads=num_heads,
        h_cycles=h_cycles,
        l_cycles=l_cycles,
        pos_encodings=pos_encodings,
    )


@app.command("train-actv1")
def train_actv1(
    model: str = typer.Option(_default_model_dir(), "--model", help="HF model name or local dir for tokenizer"),
    dataset_file: Optional[str] = typer.Option(None, "--dataset-file", help="Text/CSV/archive/dir to sample lines from"),
    dataset_chunk_size: int = typer.Option(4000, "--dataset-chunk-size", help="Samples per training cycle in iterate mode. Smaller=less VRAM (2000), default=balanced (4000), larger=faster (8000+). Adjust based on VRAM."),
    max_seq_len: int = typer.Option(128, "--max-seq-len"),
    batch_size: int = typer.Option(8, "--batch-size"),
    gradient_accumulation_steps: int = typer.Option(
        1, 
        "--gradient-accumulation-steps",
        help="Accumulate gradients over N batches before updating weights. "
             "Effective batch size = batch_size × gradient_accumulation_steps. "
             "Use to train with larger effective batches without increasing VRAM. "
             "Example: batch=8, accum=4 → effective_batch=32"
    ),
    steps: int = typer.Option(200, "--steps"),
    lr: float = typer.Option(2e-4, "--lr"),
    device: str = typer.Option("auto", "--device", help="auto|cpu|cuda|xpu|mps|dml"),
    halt_max_steps: int = typer.Option(2, "--halt-max-steps"),
    save_dir: str = typer.Option(_default_training_save_dir(), "--save-dir"),
    ascii_only: bool = typer.Option(False, "--ascii-only/--no-ascii-only", help="Filter dataset to ASCII-only lines for English focus"),
    eval_file: Optional[str] = typer.Option(None, "--eval-file", help="Held-out file/dir for final evaluation after training"),
    eval_batches: int = typer.Option(10, "--eval-batches", help="Max eval batches for final evaluation (0=disabled)"),
    sys_mem_cap_pct: Optional[int] = typer.Option(None, "--sys-mem-cap-pct", help="Soft cap for system memory usage percent; auto-reduce CPU batch size if exceeded"),
    stop_file: Optional[str] = typer.Option(None, "--stop-file", help="If file exists, training stops gracefully"),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="Optional JSONL file to append training/eval metrics"),
    student_init: Optional[str] = typer.Option(None, "--student-init", help="Optional path to existing ACTV1 student checkpoint (.safetensors) to continue training"),
    # Brain bundle
    brain_name: Optional[str] = typer.Option(None, "--brain-name", help="Optional brain bundle name; when provided, saves to --bundle-dir/<brain-name>"),
    default_goal: Optional[str] = typer.Option(None, "--default-goal", help="Default training goal/directive for this brain. This goal guides what the brain learns during training."),
    bundle_dir: str = typer.Option(_default_bundle_dir(), "--bundle-dir", help="Base directory for ACTV1 brain bundles"),
    # Expert training
    expert_id: Optional[str] = typer.Option(None, "--expert-id", help="Train a specific expert module (FeedForward network) instead of full HRM model. Saves to artifacts/experts/<expert-id>/"),
    # Architecture knobs
    h_layers: int = typer.Option(2, "--h-layers", help="Number of High-level layers"),
    l_layers: int = typer.Option(2, "--l-layers", help="Number of Low-level layers"),
    hidden_size: int = typer.Option(512, "--hidden-size", help="Model hidden dimension (not total params; e.g., 768→~57M params, 1536→~230M params, 2048→~378M params)"),
    expansion: float = typer.Option(2.0, "--expansion", help="FFN expansion factor (intermediate = hidden*expansion)"),
    num_heads: int = typer.Option(8, "--num-heads", help="Number of attention heads (must divide hidden_size evenly)"),
    h_cycles: int = typer.Option(2, "--h-cycles", help="High-level recurrent cycles per segment"),
    l_cycles: int = typer.Option(2, "--l-cycles", help="Low-level recurrent cycles per segment"),
    pos_encodings: str = typer.Option("rope", "--pos-encodings", help="Position encodings: rope|learned"),
    use_flash_attn: bool = typer.Option(False, "--use-flash-attn/--no-flash-attn", help="Enable Flash Attention 2 for optimized attention computation. Faster and more memory-efficient. Requires Ampere GPU or newer. Falls back to PyTorch SDPA if unavailable."),
    window_size: Optional[int] = typer.Option(None, "--window-size", help="Sliding window attention size (None=full attention). Use 256-512 for extreme contexts (50K-100K tokens). Independent of --use-flash-attn."),
    cuda_ids: Optional[str] = typer.Option(None, "--cuda-ids", help="Comma-separated CUDA device indices to use (e.g., '0,1')"),
    iterate: bool = typer.Option(False, "--iterate/--no-iterate", help="Repeat generation + training cycles indefinitely until stopped"),
    stop_after_epoch: bool = typer.Option(False, "--stop-after-epoch/--no-stop-after-epoch", help="Stop training after completing the current epoch. Useful for pausing at natural checkpoints."),
    resume: bool = typer.Option(False, "--resume/--no-resume", help="Resume from the last checkpoint recorded in the brain bundle when available."),
    # Dataset progression mode
    linear_dataset: bool = typer.Option(True, "--linear-dataset/--no-linear-dataset", help="Process dataset linearly (sequential order) without shuffling. Enables position tracking for pause/resume. Default: linear mode"),
    dataset_start_offset: int = typer.Option(0, "--dataset-start-offset", help="Starting sample index for resuming linear dataset training (use with --linear-dataset). Default: 0"),
    # Automatic optimization
    optimize: bool = typer.Option(False, "--optimize/--no-optimize", help="Automatically find optimal settings for max context (up to 100K) and batch size based on available VRAM. Overrides max-seq-len and batch-size if enabled."),
    # Memory optimization
    gradient_checkpointing: bool = typer.Option(True, "--gradient-checkpointing/--no-gradient-checkpointing", help="Enable gradient checkpointing to reduce VRAM usage (trades ~20% speed for 30-50% less memory). Default: enabled"),
    use_amp: bool = typer.Option(True, "--amp/--no-amp", help="Use automatic mixed precision (FP16/BF16) for activations. Saves ~40-50%% memory with minimal quality loss. Default: enabled"),
    use_cpu_offload: bool = typer.Option(False, "--cpu-offload/--no-cpu-offload", help="Offload carry states to CPU between chunks (for extreme contexts >500K tokens). Slower but uses less VRAM."),
    use_8bit_optimizer: bool = typer.Option(False, "--use-8bit-optimizer/--no-8bit-optimizer", help="Use 8-bit optimizer (bitsandbytes) for 75%% optimizer memory savings. Great for large models. Requires: pip install bitsandbytes"),
    use_chunked_training: bool = typer.Option(False, "--use-chunked-training/--no-chunked-training", help="Enable chunked training for extreme contexts. Splits sequences into chunks to reduce memory. User must explicitly enable - not auto-enforced based on context length."),
    chunk_size: int = typer.Option(2048, "--chunk-size", help="Chunk size for chunked training (1024-4096 tokens). Smaller = less VRAM, slower. Default: 2048"),
    # Sparse MoE (Mixture of Experts) options
    use_moe: bool = typer.Option(True, "--use-moe/--no-moe", help="Enable sparse Mixture of Experts architecture for 75%% compute reduction and improved specialization. Default: enabled"),
    num_experts: int = typer.Option(8, "--num-experts", help="Total number of expert networks in MoE layers. More experts = better specialization but more VRAM. Default: 8"),
    num_experts_per_tok: int = typer.Option(2, "--num-experts-per-tok", help="Top-k experts activated per token (sparse activation). Lower = faster, less capacity. Default: 2"),
    moe_capacity_factor: float = typer.Option(1.25, "--moe-capacity-factor", help="Expert capacity factor for load balancing. Higher = better load balance, more memory. Default: 1.25"),
    auto_adjust_lr: bool = typer.Option(True, "--auto-adjust-lr/--no-auto-adjust-lr", help="Automatically adjust learning rate based on model configuration (MoE, size, etc). When enabled: MoE models use 1e-5 to 2e-5 for router stability. Prevents gradient issues. Default: enabled"),
    # DeepSpeed ZeRO optimization
    zero_stage: str = typer.Option("none", "--zero-stage", help="DeepSpeed ZeRO optimization: none, zero1 (↓25%% VRAM), zero2 (↓50%% VRAM, recommended), zero3 (↓75%% VRAM). Auto-selected if --optimize is used."),
    # PEFT (Parameter-Efficient Fine-Tuning) options
    use_peft: bool = typer.Option(False, "--use-peft/--no-peft", help="Enable PEFT (LoRA/AdaLoRA) for parameter-efficient training. Reduces trainable params by 95-99%% and saves 5-10GB VRAM."),
    peft_method: str = typer.Option("lora", "--peft-method", help="PEFT method: 'lora' (default), 'adalora', 'ia3', 'loha', 'lokr'."),
    lora_r: int = typer.Option(16, "--lora-r", help="LoRA rank (adapter dimension). Higher = more capacity. Typical: 8-64. Default: 16."),
    lora_alpha: int = typer.Option(32, "--lora-alpha", help="LoRA scaling factor (alpha/r = scaling). Typical: 2×rank. Default: 32."),
    lora_dropout: float = typer.Option(0.05, "--lora-dropout", help="Dropout for LoRA layers. Default: 0.05."),
    lora_target_modules: str = typer.Option("q_proj,v_proj", "--lora-target-modules", help="Comma-separated modules for LoRA: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj. Default: 'q_proj,v_proj'."),
    
    # Model Precision and Quantization options
    model_dtype: str = typer.Option("fp32", "--model-dtype", help="Model weight precision: 'fp32' (full, default), 'fp16' (half, 50%% memory), 'bf16' (bfloat16, 50%% memory, more stable). Separate from AMP."),
    load_in_8bit: bool = typer.Option(False, "--load-in-8bit/--no-load-in-8bit", help="Load model in 8-bit (INT8) precision. 75%% memory reduction. Requires bitsandbytes + CUDA. Great with PEFT."),
    load_in_4bit: bool = typer.Option(False, "--load-in-4bit/--no-load-in-4bit", help="Load model in 4-bit precision (QLoRA). 87.5%% memory reduction. Most aggressive quantization. Requires compatible GPU."),
    
    # Distributed (optional)
    ddp: bool = typer.Option(False, "--ddp/--no-ddp", help="Enable multi-GPU training via torch.distributed (CUDA only)"),
    world_size: Optional[int] = typer.Option(None, "--world-size", help="Number of processes/GPUs to use for DDP (defaults to number of --cuda-ids or all visible GPUs)"),
    strict: bool = typer.Option(False, "--strict/--no-strict", help="Disallow device fallbacks (e.g., no CPU fallback if CUDA requested); error instead."),
    parallel_independent: bool = typer.Option(False, "--parallel-independent/--no-parallel-independent", help="Use parallel independent training (Windows-compatible multi-GPU). Trains different data blocks on different GPUs sequentially, then merges checkpoints. Bypasses DDP."),
    
    # Multi-GPU Inference + Training
    inference_device: Optional[str] = typer.Option(None, "--inference-device", help="Specific GPU for inference while training on another (e.g., 'cuda:1'). Enables simultaneous training and inference. Requires 2+ GPUs."),
    hot_reload_steps: int = typer.Option(0, "--hot-reload-steps", help="Frequency (in steps) to reload inference model from training checkpoint. Use with --inference-device. 0=disabled."),
):
    """Train the built-in ACT V1 HRM model as a student using HRM segment_rollout.

    Trains the student model on sequences from the provided dataset file.
    The training uses sequence cross-entropy loss and trains the student's 
    halt/continue heads jointly.
    
    REQUIRED: --dataset-file must be provided with training data.
    """
    
    # Build TrainingConfig from CLI parameters
    config = TrainingConfig(
        model=model,
        dataset_file=dataset_file,
        dataset_chunk_size=dataset_chunk_size,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        steps=steps,
        lr=lr,
        device=device,
        halt_max_steps=halt_max_steps,
        save_dir=save_dir,
        ascii_only=ascii_only,
        eval_file=eval_file,
        eval_batches=eval_batches,
        sys_mem_cap_pct=sys_mem_cap_pct,
        stop_file=stop_file,
        log_file=log_file,
        student_init=student_init,
        brain_name=brain_name,
        default_goal=default_goal,
        expert_id=expert_id,
        bundle_dir=bundle_dir,
        h_layers=h_layers,
        l_layers=l_layers,
        hidden_size=hidden_size,
        expansion=expansion,
        num_heads=num_heads,
        h_cycles=h_cycles,
        l_cycles=l_cycles,
        pos_encodings=pos_encodings,
        use_flash_attn=use_flash_attn,
        window_size=window_size,
        cuda_ids=cuda_ids,
        iterate=iterate,
        stop_after_epoch=stop_after_epoch,
    resume=resume,
        linear_dataset=linear_dataset,
        dataset_start_offset=dataset_start_offset,
        optimize=optimize,
        gradient_checkpointing=gradient_checkpointing,
        use_amp=use_amp,
        use_cpu_offload=use_cpu_offload,
        use_8bit_optimizer=use_8bit_optimizer,
        use_chunked_training=use_chunked_training,
        chunk_size=chunk_size,
        # Sparse MoE options
        use_moe=use_moe,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_capacity_factor=moe_capacity_factor,
        auto_adjust_lr=auto_adjust_lr,
        # DeepSpeed ZeRO
        zero_stage=zero_stage,
        # PEFT options
        use_peft=use_peft,
        peft_method=peft_method,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        # Model precision and quantization options
        model_dtype=model_dtype,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        # Distributed
        ddp=ddp,
        world_size=world_size,
        strict=strict,
        parallel_independent=parallel_independent,
        # Multi-GPU Inference + Training
        inference_device=inference_device,
        hot_reload_steps=hot_reload_steps,
    )
    
    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(1)
    
    # Call implementation with TrainingConfig with exception handler
    try:
        _train_actv1_impl(config=config)
        # Successful completion - exit with code 0
        return 0
    except typer.Exit:
        # Re-raise typer.Exit (already has correct exit code)
        raise
    except Exception as e:
        import traceback
        print({
            "FATAL_ERROR": "train_actv1_impl_crashed",
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc(),
        })
        raise  # Re-raise to show error code


@app.command("preprocess-dataset")
def preprocess_dataset(
    dataset_path: str = typer.Argument(..., help="Path to dataset directory to preprocess"),
    block_size: int = typer.Option(100000, "--block-size", help="Number of samples per block"),
    ascii_only: bool = typer.Option(False, "--ascii-only", help="Filter to ASCII-only text"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing preprocessed structure"),
):
    """Preprocess a downloaded dataset into optimized block-based structure.
    
    This command organizes datasets into blocks (default: 100k samples per block),
    enabling fast size detection and efficient training with progress tracking.
    
    Examples:
        aios hrm-hf preprocess-dataset training_datasets/tinystories
        aios hrm-hf preprocess-dataset ~/datasets/my_dataset --block-size 50000
        aios hrm-hf preprocess-dataset /data/corpus --ascii-only --overwrite
    """
    return _preprocess_dataset_cmd(
        dataset_path=dataset_path,
        block_size=block_size,
        ascii_only=ascii_only,
        overwrite=overwrite,
    )

def register(app_root: typer.Typer) -> None:
    app_root.add_typer(app, name="hrm-hf")
