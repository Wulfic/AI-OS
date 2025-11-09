"""Unified configuration for HRM ActV1 training.

This module provides a single source of truth for all training parameters,
ensuring consistency across CLI, GUI, and any future interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


@dataclass
class TrainingConfig:
    """Unified configuration for HRM ActV1 training.
    
    Single source of truth for all training parameters.
    Used by CLI, GUI, and any future interfaces.
    
    This ensures feature parity across interfaces and makes it easy to add
    new optimizations (Flash Attention, DeepSpeed ZeRO, etc.) that automatically
    work everywhere.
    """
    
    # ============================================================================
    # Model and Data
    # ============================================================================
    model: str = "artifacts/hf_implant/base_model"
    """HF model name or local path for tokenizer."""
    
    dataset_file: Optional[str] = None
    """Text/CSV/archive/dir to sample training lines from (REQUIRED)."""
    
    max_seq_len: int = 128
    """Token sequence length (context length)."""
    
    batch_size: int = 8
    """Training batch size (may be auto-adjusted by OOM backoff)."""
    
    steps: int = 200
    """Number of training steps."""
    
    # ============================================================================
    # Optimization
    # ============================================================================
    lr: float = 2e-4
    """Learning rate."""
    
    auto_adjust_moe_lr: bool = True
    """Automatically reduce learning rate for MoE models to prevent instability.
    
    When enabled and use_moe=True:
    - If lr >= 1e-4: reduce to 1e-6
    - If lr >= 5e-5: reduce to 2e-6
    
    MoE models are sensitive to high learning rates due to router networks.
    Disable this to use your exact lr value.
    
    Default: True (recommended for MoE stability)
    """
    
    halt_max_steps: int = 2
    """Maximum ACT (Adaptive Computation Time) segments during training."""
    
    gradient_checkpointing: bool = True
    """Enable gradient checkpointing to reduce VRAM usage.
    
    Trades ~20% speed for 30-50% less memory. Recommended for large contexts.
    Default: enabled for better VRAM efficiency.
    """
    
    use_amp: bool = True
    """Use automatic mixed precision (FP16/BF16) for activations.
    
    Saves ~40-50% memory with minimal quality loss. Default: enabled.
    """
    
    use_cpu_offload: bool = False
    """Offload carry states to CPU between chunks.
    
    For extreme contexts (>500K tokens). Slower but uses less VRAM.
    Default: disabled (only beneficial for very large contexts).
    """
    
    use_8bit_optimizer: bool = False
    """Use 8-bit optimizer (bitsandbytes) for 75% optimizer memory savings.
    
    Reduces optimizer state memory from FP32 to INT8, saving ~75% memory
    with minimal impact on training quality. Particularly effective for
    large models (>100M parameters).
    
    Memory savings example:
    - 60M params with AdamW: ~360 MB saved
    - 500M params with AdamW: ~3 GB saved
    - 1B params with AdamW: ~6 GB saved
    
    Requires: bitsandbytes library (pip install bitsandbytes)
    Default: disabled (enable for memory-constrained scenarios)
    """
    
    use_chunked_training: bool = False
    """Enable chunked training for extreme context lengths (8K+ tokens).
    
    Splits long sequences into smaller chunks to reduce memory usage.
    Automatically enabled when max_seq_len > 8192 tokens.
    
    Benefits:
    - Allows training on sequences that would otherwise OOM
    - Reduces peak memory usage
    - Maintains gradient flow across chunks
    
    Trade-offs:
    - Slightly slower training (~10-20% overhead)
    - More complex training loop
    
    Default: disabled (auto-enabled for long contexts)
    """
    
    chunk_size: int = 2048
    """Size of chunks when using chunked training.
    
    Smaller chunks use less memory but are slower.
    Typical values: 1024-4096 tokens.
    
    Recommendations:
    - 2048: Good balance for most scenarios
    - 1024: Very constrained memory (10GB VRAM)
    - 4096: More memory available (24GB+ VRAM)
    
    Default: 2048 tokens
    """
    
    # ============================================================================
    # DeepSpeed ZeRO Optimization
    # ============================================================================
    zero_stage: str = "none"
    """DeepSpeed ZeRO optimization stage.
    
    Options:
    - "none": No DeepSpeed (standard PyTorch training)
    - "zero1": Optimizer state partitioning (~25% VRAM reduction)
    - "zero2": Optimizer + gradient partitioning (~50% VRAM reduction, recommended)
    - "zero3": Optimizer + gradient + parameter partitioning (~75% VRAM reduction)
    
    Note: Requires DeepSpeed library installed. Auto-selected if --optimize is used.
    """
    
    # ============================================================================
    # Distributed Training
    # ============================================================================
    device: str = "auto"
    """Device for training: auto|cpu|cuda|xpu|mps|dml.
    
    "auto" will use CUDA if available, otherwise CPU.
    """
    
    cuda_ids: Optional[str] = None
    """Comma-separated CUDA device indices to use (e.g., '0,1' for 2 GPUs).
    
    If multiple IDs provided, DDP is automatically enabled.
    """
    
    ddp: bool = False
    """Enable multi-GPU training via torch.distributed (CUDA only).
    
    Automatically enabled when multiple cuda_ids are specified.
    """
    
    world_size: Optional[int] = None
    """Number of processes/GPUs to use for DDP.
    
    Defaults to number of cuda_ids or all visible GPUs.
    """
    
    strict: bool = False
    """Disallow device fallbacks.
    
    If True, error instead of falling back (e.g., no CPU fallback if CUDA requested).
    """
    
    inference_device: Optional[str] = None
    """Specific GPU device for inference while training on another GPU.
    
    Example: "cuda:1" to run inference on GPU 1 while training on GPU 0.
    Enables simultaneous training and inference on multi-GPU systems.
    Requires at least 2 GPUs available.
    
    Note: Leave None to disable separate inference GPU (inference and training on same device).
    """
    
    hot_reload_steps: int = 0
    """Frequency (in training steps) to reload inference model from training checkpoint.
    
    If > 0 and inference_device is set, the inference model will be reloaded
    every N steps with the latest training weights. This enables real-time testing
    of the model during training on a separate GPU.
    
    Example: hot_reload_steps=100 reloads every 100 training steps.
    Default: 0 (disabled)
    """
    
    # ============================================================================
    # Evaluation
    # ============================================================================
    eval_file: Optional[str] = None
    """Held-out file/dir for final evaluation after training.
    
    If provided, enables evaluation metrics on validation data.
    """
    
    eval_batches: int = 10
    """Maximum eval batches for final evaluation.
    
    Set to 0 to disable evaluation. Default: 10 batches.
    """
    
    # ============================================================================
    # I/O and Persistence
    # ============================================================================
    save_dir: str = "training_data/actv1"
    """Directory where checkpoints and training state are saved."""
    
    stop_file: Optional[str] = None
    """If file exists, training stops gracefully.
    
    Useful for remote monitoring and controlled shutdown.
    """
    
    log_file: Optional[str] = None
    """Optional JSONL file to append training/eval metrics.
    
    Each line is a JSON object with step, loss, etc.
    """
    
    student_init: Optional[str] = None
    """Optional path to existing ACTV1 student state_dict (.pt) to continue training.
    
    If provided, training resumes from this checkpoint.
    """
    
    brain_name: Optional[str] = None
    """Optional brain bundle name.
    
    When provided, saves to --bundle-dir/<brain-name>.
    """
    
    default_goal: Optional[str] = None
    """Default training goal/directive for this brain.
    
    This goal guides what the brain learns during training.
    Automatically added to the brain's goals when training starts.
    """
    
    expert_id: Optional[str] = None
    """Expert ID for expert-specific training mode.
    
    When provided, trains only a FeedForward expert module instead of full HRM model.
    The expert is saved to artifacts/experts/{expert_id}/ and registered in the
    expert registry. This enables training specialized experts for specific goals
    without affecting the base model.
    
    Expert training features:
    - Trains lightweight FeedForward module (hidden_size → intermediate_size → hidden_size)
    - Saves to artifacts/experts/{expert_id}/expert.pt
    - Auto-updates expert_registry.json with metadata
    - Can be loaded at runtime via LazyExpertLoader
    - Supports all optimization features (AMP, gradient checkpointing, etc.)
    
    If None (default), trains full HRM ACT-v1 model normally.
    """
    
    bundle_dir: str = "artifacts/brains/actv1"
    """Base directory for ACTV1 brain bundles.
    
    Brain-specific subdirectories are created here.
    
    Note: All checkpoints are saved in safetensors format for security and performance.
    Safetensors provides:
    - Security: No arbitrary code execution (unlike Pickle-based .pt)
    - Performance: Faster load/save with memory-mapped I/O
    - Interoperability: Standard across HuggingFace ecosystem
    """
    
    # ============================================================================
    # Model Architecture
    # ============================================================================
    h_layers: int = 2
    """Number of High-level layers in the HRM architecture."""
    
    l_layers: int = 2
    """Number of Low-level layers in the HRM architecture."""
    
    hidden_size: int = 512
    """Model hidden dimension.
    
    Examples:
    - 512 → ~50M params
    - 768 → ~87M params
    - 1024 → ~150M params
    - 1536 → ~230M params
    - 2048 → ~378M params
    """
    
    expansion: float = 2.0
    """FFN (Feed-Forward Network) expansion factor.
    
    Intermediate dimension = hidden_size * expansion.
    Typical values: 2.0 to 4.0.
    """
    
    num_heads: int = 8
    """Number of attention heads.
    
    Must divide hidden_size evenly (e.g., 768/8 = 96 per head).
    """
    
    h_cycles: int = 2
    """High-level recurrent cycles per segment."""
    
    l_cycles: int = 2
    """Low-level recurrent cycles per segment."""
    
    pos_encodings: str = "rope"
    """Position encodings type.
    
    Options:
    - "rope": Rotary Position Embeddings (recommended, handles long contexts)
    - "alibi": ALiBi position bias
    - "none": No position encodings (not recommended)
    """
    
    window_size: Optional[int] = None
    """Sliding window attention size (None = full attention).
    
    Limits attention to a local window for memory efficiency.
    Converts O(n²) attention to O(n×window_size).
    
    Recommended values for extreme contexts:
    - 256: Minimal memory, ~6GB for 100K tokens
    - 512: Balanced, ~12GB for 100K tokens  
    - 1024: High quality, ~24GB for 100K tokens
    - 2048: Near-full attention, ~48GB for 100K tokens
    - None: Full attention (default)
    
    WARNING: Smaller windows may reduce model quality.
    Test quality after enabling to ensure acceptable performance.
    """
    
    # ============================================================================
    # Sparse Mixture of Experts (MoE) Options
    # ============================================================================
    use_moe: bool = True
    """Enable sparse Mixture of Experts architecture for efficient inference.
    
    Replaces standard FFN layers with MoE layers that activate only a subset of
    expert networks per token. Provides significant compute savings (~75%) with
    minimal quality loss.
    
    Benefits:
    - ~75% reduction in FFN compute (2/8 experts active by default)
    - Maintains model quality through specialized experts
    - Scales model capacity without proportional compute increase
    - Each block gets its own set of experts for fine-grained specialization
    
    Architecture impact:
    - Each ACTV1 block (H and L layers) gets independent MoE layer
    - Total MoE layers = h_layers + l_layers (e.g., 6+6 = 12 MoE layers)
    - Each MoE layer has num_experts total, num_experts_per_tok active
    
    Memory trade-off:
    - More parameters (8 experts vs 1 FFN per layer)
    - But compute remains same as ~2 FFN layers (sparse activation)
    
    Compatible with:
    - All optimization features (gradient checkpointing, AMP, etc.)
    - PEFT/LoRA (adapters can target expert parameters)
    - Chunked training and extreme contexts
    
    Default: True (enabled for efficiency)
    """
    
    num_experts: int = 8
    """Number of expert networks per MoE layer.
    
    Each MoE layer contains this many expert FFN networks. Higher counts allow
    more specialization but increase parameter count proportionally.
    
    Typical values:
    - 4: Minimal (less specialization)
    - 8: Balanced (recommended, good specialization vs params)
    - 16: High specialization (2x parameters)
    - 32: Very high (4x parameters, diminishing returns)
    
    Memory impact (per MoE layer, hidden_size=512, expansion=2.0):
    - 4 experts: ~4M params
    - 8 experts: ~8M params (default)
    - 16 experts: ~16M params
    
    Note: Total model size scales with num_experts, but compute cost depends
    only on num_experts_per_tok (sparse activation).
    
    Default: 8 experts (good balance)
    """
    
    num_experts_per_tok: int = 2
    """Number of experts activated per token (top-k routing).
    
    The router selects the top-k most relevant experts for each token, activating
    only those experts. Lower values save more compute but may reduce quality.
    
    Compute reduction = 1 - (num_experts_per_tok / num_experts)
    
    Examples (with num_experts=8):
    - k=1: 87.5% compute reduction (very sparse, may hurt quality)
    - k=2: 75% compute reduction (recommended, good balance)
    - k=4: 50% compute reduction (less sparse)
    - k=8: 0% compute reduction (defeats purpose of MoE)
    
    Quality considerations:
    - k=1 works well for simple, repetitive tasks
    - k=2 (default) balances efficiency and quality for most tasks
    - k>=4 for very complex tasks requiring multiple perspectives
    
    Note: Must be <= num_experts. Typical ratio is k = num_experts / 4.
    
    Default: 2 (75% compute reduction)
    """
    
    moe_capacity_factor: float = 1.25
    """Expert capacity factor for load balancing in MoE layers.
    
    Controls how many tokens each expert can process in a batch. Higher values
    allow better load balancing when some experts are very popular, but use
    more memory.
    
    Capacity = (batch_tokens / num_experts) * capacity_factor
    
    Typical values:
    - 1.0: Strict capacity (may drop tokens if expert overloaded)
    - 1.25: Balanced (default, allows 25% overflow)
    - 1.5: Generous (allows 50% overflow, better load balancing)
    - 2.0: Very generous (2x capacity, rarely needed)
    
    Trade-offs:
    - Higher factor → better load balancing, less token dropping
    - Higher factor → more memory usage
    - Lower factor → memory efficient, but may drop tokens
    
    Note: Token dropping means some tokens don't get processed by any expert
    (fall back to residual connection), which can slightly hurt quality.
    
    Default: 1.25 (good balance)
    """
    
    # ============================================================================
    # Advanced Options
    # ============================================================================
    resume: bool = False
    """Resume training from last checkpoint.
    
    When enabled and a checkpoint exists:
    - Loads model weights from checkpoint
    - Restores training configuration (dataset, batch size, learning rate, etc.)
    - Continues step counter from where training left off
    - Validates dataset matches to prevent breaking training continuity
    
    If checkpoint doesn't exist or dataset doesn't match, starts fresh training.
    
    Use cases:
    - Continue training after interruption
    - Add more training steps to an existing brain
    - Resume long-running training sessions
    
    Safety:
    - Dataset-aware: Warns if dataset changed
    - Config validation: Alerts if critical settings differ
    - Graceful degradation: Falls back to fresh start if resume fails
    
    Default: False (start fresh training)
    """
    
    iterate: bool = False
    """Repeat generation + training cycles indefinitely until stopped.
    
    WARNING: This is an experimental feature for continuous training loops.
    """
    
    stop_after_epoch: bool = False
    """Stop training after completing the current epoch.
    
    When enabled, training will complete the current epoch (all steps) and then
    stop gracefully, even if in iterate mode. This allows for clean checkpointing
    after a full pass through the training steps.
    
    Use cases:
    - Stop multi-cycle training after the current cycle completes
    - Ensure checkpoint is saved at a clean epoch boundary
    - Pause long-running training sessions at natural stopping points
    
    Note: This is different from the stop_file mechanism, which can interrupt
    training at any step. stop_after_epoch waits for epoch completion.
    
    Default: False (train for all specified steps/cycles)
    """
    
    optimize: bool = False
    """Automatically find optimal settings for max context and batch size.
    
    When enabled, searches for the largest context length (up to 100K) and
    batch size that fit in available VRAM. Overrides max-seq-len and batch-size.
    
    WARNING: This runs a search process that may take several minutes.
    """
    
    ascii_only: bool = False
    """Filter dataset to ASCII-only lines for English focus.
    
    Useful when training on mixed-language datasets but wanting English-only output.
    """
    
    linear_dataset: bool = False
    """Process dataset linearly without shuffling.
    
    When enabled (linear_dataset=True):
    - Dataset samples are processed in sequential order
    - Training position is tracked and saved in checkpoints
    - Can pause and resume without retraining the same data
    - Ensures 100% dataset coverage before wrapping to the beginning
    
    When disabled (linear_dataset=False, default):
    - Dataset is shuffled each epoch using deterministic seeds
    - Better for model generalization (recommended for most cases)
    - Each epoch presents data in a different order
    
    Use cases for linear mode:
    - When dataset order matters (sequential story, curriculum learning)
    - When you need precise tracking of training progress through dataset
    - When resuming training and want to continue exactly where left off
    - When debugging or validating training on specific data samples
    
    Note: Position tracking is saved to checkpoints automatically when using
    streaming datasets (>4GB). For smaller datasets, the entire dataset is
    loaded into memory and sampling is random (position tracking not applicable).
    
    Default: False (shuffled mode for better generalization)
    """
    
    dataset_start_offset: int = 0
    """Starting sample index for resuming linear dataset training.
    
    Only used when linear_dataset=True. Allows manually specifying where
    to start in the dataset, useful for:
    - Resuming interrupted training runs
    - Training on specific portions of the dataset
    - Continuing from a known checkpoint position
    
    Value should be between 0 and (num_samples - 1).
    Default: 0 (start from beginning)
    """
    
    sys_mem_cap_pct: Optional[int] = None
    """Soft cap for system memory usage percent.
    
    If system RAM usage exceeds this percentage, CPU batch size is auto-reduced.
    Typical values: 70-90. Default: None (no cap).
    """
    
    # ============================================================================
    # PEFT (Parameter-Efficient Fine-Tuning) Options
    # ============================================================================
    use_peft: bool = False
    """Enable PEFT (e.g., LoRA) for parameter-efficient training.
    
    When enabled, only adds small adapter layers (~1-2M params) instead of training
    all 87M+ parameters. Reduces memory by 5-10GB and speeds up training significantly.
    
    Benefits:
    - 95-99% reduction in trainable parameters
    - 40-60% reduction in VRAM usage
    - Faster training and convergence
    - Easy to merge adapters back into base model
    
    Compatible with all other optimizations (gradient checkpointing, AMP, etc.).
    """
    
    peft_method: str = "lora"
    """PEFT method to use.
    
    Supported methods:
    - 'lora': Low-Rank Adaptation (recommended, most stable)
    - 'adalora': Adaptive LoRA (adjusts rank dynamically)
    - 'ia3': Infused Adapter by Inhibiting and Amplifying Inner Activations
    - 'loha': Low-Rank Hadamard Product
    - 'lokr': Low-Rank Kronecker Product
    
    Default: 'lora' (best balance of quality and efficiency)
    """
    
    lora_r: int = 16
    """LoRA rank (adapter matrix dimension).
    
    Higher rank = more model capacity but more parameters and memory.
    
    Typical values:
    - r=8: Minimal (0.5M params, ~1-2GB VRAM) - good for fine-tuning
    - r=16: Balanced (2M params, ~2-3GB VRAM) - recommended default
    - r=32: High capacity (8M params, ~4-5GB VRAM) - for complex tasks
    - r=64: Very high (32M params, ~8-10GB VRAM) - rarely needed
    
    Rule of thumb: Use 2× rank for lora_alpha (e.g., r=16, alpha=32).
    """
    
    lora_alpha: int = 32
    """LoRA scaling parameter.
    
    Controls the scaling of adapter outputs. The effective learning rate of
    adapters is: (lora_alpha / lora_r) * base_lr
    
    Typical values:
    - lora_alpha = 2 × lora_r (standard scaling)
    - Higher alpha = stronger adapter influence
    - Lower alpha = more conservative adaptation
    
    Default: 32 (works well with r=16).
    """
    
    lora_dropout: float = 0.05
    """Dropout probability for LoRA adapter layers.
    
    Standard dropout applied to LoRA adapters for regularization.
    Typical values: 0.0-0.1. Default: 0.05.
    """
    
    lora_target_modules: str = "q_proj,v_proj"
    """Comma-separated list of module names to apply LoRA adapters to.
    
    HRM ACTv1 architecture modules:
    - Attention projections: q_proj, k_proj, v_proj, o_proj (recommended)
    - MLP projections: gate_proj, up_proj, down_proj (optional, more params)
    
    Recommendations:
    - Minimal: 'q_proj,v_proj' (default) - attention queries and values only
    - Balanced: 'q_proj,k_proj,v_proj,o_proj' - full attention
    - Full: 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj' - attention + MLP
    
    Note: lm_head and q_head (HRM-specific halting head) always remain trainable
    to preserve architecture integrity.
    
    Default: 'q_proj,v_proj' (good balance of efficiency and quality).
    """
    
    # ============================================================================
    # Model Precision and Quantization Options
    # ============================================================================
    
    model_dtype: str = "fp32"
    """Model weight precision/data type.
    
    Controls the precision of model weights (parameters). Lower precision reduces
    memory usage and can speed up training, but may impact numerical stability.
    
    Options:
    - "fp32" (float32): Full precision, maximum stability, highest memory (default)
    - "fp16" (float16): Half precision, 50% memory reduction, faster on modern GPUs
    - "bf16" (bfloat16): Brain float 16, 50% memory reduction, better stability than fp16
    
    Recommendations:
    - fp32: Use for small models or when maximum precision is required
    - fp16: Best for NVIDIA GPUs with Tensor Cores (V100, A100, RTX 20xx+)
    - bf16: Best for Ampere+ GPUs (A100, RTX 30xx+) or TPUs, more stable than fp16
    
    Memory savings example (87M params model):
    - fp32: ~348 MB
    - fp16/bf16: ~174 MB (50% reduction)
    
    Note: This is separate from AMP (use_amp), which only affects activations.
    Model dtype affects the stored weights, while AMP affects computation.
    
    Compatibility:
    - Works with all optimizations (gradient checkpointing, PEFT, etc.)
    - bf16 requires PyTorch 1.10+ and compatible hardware
    - Combine with use_amp=True for maximum memory efficiency
    """
    
    load_in_8bit: bool = False
    """Load model weights in 8-bit precision using bitsandbytes.
    
    Quantizes model weights to INT8, reducing memory by ~75% with minimal quality loss.
    Particularly effective when combined with PEFT/LoRA (QLoRA pattern).
    
    Benefits:
    - 75% reduction in model memory (87M params: 348 MB → 87 MB)
    - Enables training larger models on consumer GPUs
    - Minimal accuracy degradation (<1% typically)
    - Works well with PEFT adapters
    
    Requirements:
    - bitsandbytes library (pip install bitsandbytes>=0.43.0)
    - CUDA-capable GPU (not supported on CPU/DML)
    - PyTorch 2.0+
    
    Memory example (87M params):
    - fp32: 348 MB
    - 8-bit: ~87 MB (75% reduction)
    - 8-bit + LoRA (r=16): ~90 MB total (only adapters trainable)
    
    Note: Quantization happens at model loading time. Training gradients remain
    in higher precision for numerical stability.
    
    Trade-offs:
    - Slower training (~10-20% overhead from quantization/dequantization)
    - May not work with custom model architectures (HRM may have limitations)
    - Cannot be combined with load_in_4bit
    
    Recommended for:
    - Memory-constrained scenarios (8-12 GB VRAM)
    - Fine-tuning with PEFT on large models
    - When model memory dominates total memory usage
    """
    
    load_in_4bit: bool = False
    """Load model weights in 4-bit precision using bitsandbytes (QLoRA).
    
    Quantizes model weights to 4-bit NormalFloat, reducing memory by ~87.5%.
    The most aggressive quantization option, enabling very large models on consumer hardware.
    
    Benefits:
    - 87.5% reduction in model memory (87M params: 348 MB → 43 MB)
    - Enables training models 4x larger than 8-bit
    - Surprisingly good quality with proper configuration
    - Perfect for QLoRA (4-bit base + LoRA adapters)
    
    Requirements:
    - bitsandbytes library (pip install bitsandbytes>=0.43.0)  
    - CUDA-capable GPU with compute capability 7.0+ (V100, T4, RTX 20xx+)
    - PyTorch 2.0+
    
    Memory example (87M params):
    - fp32: 348 MB
    - 4-bit: ~43 MB (87.5% reduction)
    - 4-bit + LoRA (r=16): ~45 MB total
    
    4-bit Quantization Types:
    - NF4 (NormalFloat4): Optimized for normally distributed weights (default)
    - FP4: Standard 4-bit float
    
    Note: 4-bit uses double quantization by default (quantizing the quantization
    constants) for additional memory savings with minimal quality impact.
    
    Trade-offs:
    - Slower than 8-bit (~20-30% training overhead)
    - More aggressive quality degradation (monitor val loss carefully)
    - Requires compatible GPU architecture
    - Cannot be combined with load_in_8bit
    - May not work with custom architectures
    
    Recommended for:
    - Extremely memory-constrained scenarios (4-8 GB VRAM)
    - When you need to fit a model 4x larger
    - QLoRA fine-tuning pattern
    - Experimentation before committing to full precision training
    """
    
    # ============================================================================
    # Deprecated (Kept for Backward Compatibility)
    # ============================================================================
    kl: float = 0.0
    """[DEPRECATED] KL weight toward teacher distribution.
    
    Teacher-student distillation has been removed. This parameter is ignored.
    """
    
    kl_temp: float = 1.0
    """[DEPRECATED] Temperature for teacher/student softmax in KL.
    
    Teacher-student distillation has been removed. This parameter is ignored.
    """
    
    teacher: Optional[str] = None
    """[DEPRECATED] Optional teacher LM name/path.
    
    Teacher-student distillation has been removed. This parameter is ignored.
    """
    
    teacher_device: str = "cuda"
    """[DEPRECATED] Device for teacher model.
    
    Teacher-student distillation has been removed. This parameter is ignored.
    """
    
    # ============================================================================
    # Methods
    # ============================================================================
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid.
        """
        if self.dataset_file is None:
            raise ValueError("dataset_file is required")
        
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        
        if self.steps < 1:
            raise ValueError("steps must be positive")
        
        if self.zero_stage not in {"none", "zero1", "zero2", "zero3"}:
            raise ValueError(
                f"Invalid zero_stage: {self.zero_stage}. "
                f"Must be one of: none, zero1, zero2, zero3"
            )
        
        if self.pos_encodings not in {"rope", "alibi", "none"}:
            raise ValueError(
                f"Invalid pos_encodings: {self.pos_encodings}. "
                f"Must be one of: rope, alibi, none"
            )
        
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        
        if self.lr <= 0:
            raise ValueError("lr (learning rate) must be positive")
        
        if self.eval_batches < 0:
            raise ValueError("eval_batches must be non-negative (0 to disable)")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of the configuration.
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        """Create from dictionary.
        
        Args:
            d: Dictionary with configuration parameters.
            
        Returns:
            New TrainingConfig instance.
        """
        return cls(**d)
    
    def to_cli_args(self) -> list[str]:
        """Convert to CLI arguments for subprocess launching.
        
        This is primarily used by the GUI to launch training as a subprocess.
        
        Returns:
            List of CLI arguments like: ["hrm-hf", "train-actv1", "--model", "...", ...]
        """
        args = ["hrm-hf", "train-actv1"]
        
        # Required/primary parameters
        args.extend(["--model", self.model])
        args.extend(["--max-seq-len", str(self.max_seq_len)])
        args.extend(["--batch-size", str(self.batch_size)])
        args.extend(["--steps", str(self.steps)])
        
        # Optional string/path parameters
        if self.dataset_file:
            args.extend(["--dataset-file", self.dataset_file])
        if self.cuda_ids:
            args.extend(["--cuda-ids", self.cuda_ids])
        if self.eval_file:
            args.extend(["--eval-file", self.eval_file])
        if self.stop_file:
            args.extend(["--stop-file", self.stop_file])
        if self.log_file:
            args.extend(["--log-file", self.log_file])
        if self.student_init:
            args.extend(["--student-init", self.student_init])
        if self.brain_name:
            args.extend(["--brain-name", self.brain_name])
        if self.default_goal:
            args.extend(["--default-goal", self.default_goal])
        if self.expert_id:
            args.extend(["--expert-id", self.expert_id])
        
        # Numeric parameters
        args.extend(["--lr", str(self.lr)])
        args.extend(["--halt-max-steps", str(self.halt_max_steps)])
        args.extend(["--eval-batches", str(self.eval_batches)])
        if self.sys_mem_cap_pct is not None:
            args.extend(["--sys-mem-cap-pct", str(self.sys_mem_cap_pct)])
        
        # Device/distributed
        args.extend(["--device", self.device])
        if self.ddp:
            args.append("--ddp")
        if self.world_size is not None:
            args.extend(["--world-size", str(self.world_size)])
        
        # Boolean flags
        if self.gradient_checkpointing:
            args.append("--gradient-checkpointing")
        else:
            args.append("--no-gradient-checkpointing")
        
        if self.use_amp:
            args.append("--amp")
        else:
            args.append("--no-amp")
        
        if self.use_cpu_offload:
            args.append("--cpu-offload")
        else:
            args.append("--no-cpu-offload")
        
        if self.use_8bit_optimizer:
            args.append("--use-8bit-optimizer")
        
        if self.use_chunked_training:
            args.append("--use-chunked-training")
            args.extend(["--chunk-size", str(self.chunk_size)])
        
        if self.resume:
            args.append("--resume")
        if self.iterate:
            args.append("--iterate")
        if self.stop_after_epoch:
            args.append("--stop-after-epoch")
        if self.optimize:
            args.append("--optimize")
        if self.strict:
            args.append("--strict")
        if self.ascii_only:
            args.append("--ascii-only")
        
        # Window size for Flash Attention / sliding window
        if self.window_size is not None:
            args.extend(["--window-size", str(self.window_size)])
        
        # DeepSpeed ZeRO
        if self.zero_stage != "none":
            args.extend(["--zero-stage", self.zero_stage])
        
        # Architecture parameters
        args.extend(["--h-layers", str(self.h_layers)])
        args.extend(["--l-layers", str(self.l_layers)])
        args.extend(["--hidden-size", str(self.hidden_size)])
        args.extend(["--expansion", str(self.expansion)])
        args.extend(["--num-heads", str(self.num_heads)])
        args.extend(["--h-cycles", str(self.h_cycles)])
        args.extend(["--l-cycles", str(self.l_cycles)])
        args.extend(["--pos-encodings", self.pos_encodings])
        
        # MoE parameters
        if self.use_moe:
            args.append("--use-moe")
        else:
            args.append("--no-moe")
        args.extend(["--num-experts", str(self.num_experts)])
        args.extend(["--num-experts-per-tok", str(self.num_experts_per_tok)])
        args.extend(["--moe-capacity-factor", str(self.moe_capacity_factor)])
        
        # MoE learning rate adjustment
        if self.auto_adjust_moe_lr:
            args.append("--auto-adjust-moe-lr")
        else:
            args.append("--no-auto-adjust-moe-lr")
        
        # Directories
        args.extend(["--save-dir", self.save_dir])
        args.extend(["--bundle-dir", self.bundle_dir])
        
        return args
    
    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"TrainingConfig(\n"
            f"  model={self.model!r},\n"
            f"  dataset={self.dataset_file!r},\n"
            f"  seq_len={self.max_seq_len}, batch={self.batch_size}, steps={self.steps},\n"
            f"  arch={self.h_layers}h/{self.l_layers}l, hidden={self.hidden_size}, heads={self.num_heads},\n"
            f"  device={self.device!r}, ddp={self.ddp}, zero={self.zero_stage!r}\n"
            f")"
        )
