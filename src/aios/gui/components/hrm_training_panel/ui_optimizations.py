"""Optimizations section UI for HRM Training Panel.

Builds the comprehensive optimizations section with memory options, PEFT, advanced features.
"""

from __future__ import annotations
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .panel_main import HRMTrainingPanel


def build_optimizations_section(panel: HRMTrainingPanel, parent: any) -> None:  # type: ignore[valid-type]
    """Build the optimizations section with all optimization toggles and settings.
    
    Args:
        panel: The HRMTrainingPanel instance
        parent: Parent widget
    """
    from .variable_setup import setup_lora_module_mapping
    
    opt_frame = ttk.LabelFrame(parent, text="🚀 Optimizations", padding=10)
    opt_frame.pack(fill="x", pady=(10, 5))
    
    # Row 1: Memory Optimizations
    opt_row1 = ttk.Frame(opt_frame)
    opt_row1.pack(fill="x", pady=2)
    ttk.Label(opt_row1, text="Memory:", width=15, anchor="e", font=("TkDefaultFont", 9, "bold")).pack(side="left")
    grad_ckpt_btn = ttk.Checkbutton(opt_row1, text="Grad Checkpoint ✓", variable=panel.gradient_checkpointing_var)
    grad_ckpt_btn.pack(side="left")
    amp_btn = ttk.Checkbutton(opt_row1, text="AMP ✓", variable=panel.use_amp_var)
    amp_btn.pack(side="left", padx=(8, 0))
    opt8bit_btn = ttk.Checkbutton(opt_row1, text="8-bit Optimizer", variable=panel.use_8bit_optimizer_var)
    opt8bit_btn.pack(side="left", padx=(8, 0))
    cpu_offload_btn = ttk.Checkbutton(opt_row1, text="CPU Offload", variable=panel.use_cpu_offload_var)
    cpu_offload_btn.pack(side="left", padx=(8, 0))
    
    # Row 2: PEFT (Parameter-Efficient Fine-Tuning)
    peft_row = ttk.Frame(opt_frame)
    peft_row.pack(fill="x", pady=2)
    ttk.Label(peft_row, text="PEFT:", width=15, anchor="e", font=("TkDefaultFont", 9, "bold")).pack(side="left")
    peft_enable_btn = ttk.Checkbutton(peft_row, text="Enable LoRA", variable=panel.use_peft_var)
    peft_enable_btn.pack(side="left")
    ttk.Label(peft_row, text="Method:").pack(side="left", padx=(10, 2))
    peft_method_combo = ttk.Combobox(peft_row, textvariable=panel.peft_method_var, width=8, state="readonly")
    peft_method_combo['values'] = ('lora', 'adalora', 'ia3')
    peft_method_combo.pack(side="left")
    ttk.Label(peft_row, text="Rank:").pack(side="left", padx=(8, 2))
    lora_r_entry = ttk.Entry(peft_row, textvariable=panel.lora_r_var, width=6)
    lora_r_entry.pack(side="left")
    ttk.Label(peft_row, text="Alpha:").pack(side="left", padx=(6, 2))
    lora_alpha_entry = ttk.Entry(peft_row, textvariable=panel.lora_alpha_var, width=6)
    lora_alpha_entry.pack(side="left")
    ttk.Label(peft_row, text="Dropout:").pack(side="left", padx=(6, 2))
    lora_dropout_entry = ttk.Entry(peft_row, textvariable=panel.lora_dropout_var, width=6)
    lora_dropout_entry.pack(side="left")
    ttk.Label(peft_row, text="Modules:").pack(side="left", padx=(8, 2))
    lora_modules_combo = ttk.Combobox(peft_row, width=10, state="readonly")
    lora_modules_combo['values'] = ('Full', 'Balanced', 'Minimal')
    lora_modules_combo.pack(side="left")
    
    # Setup LoRA module mapping
    setup_lora_module_mapping(panel, lora_modules_combo)
    
    # Row 3: Advanced Optimizations
    opt_row2 = ttk.Frame(opt_frame)
    opt_row2.pack(fill="x", pady=2)
    ttk.Label(opt_row2, text="Advanced:", width=15, anchor="e", font=("TkDefaultFont", 9, "bold")).pack(side="left")
    flash_attn_btn = ttk.Checkbutton(opt_row2, text="FlashAttn-2", variable=panel.use_flash_attn_var)
    flash_attn_btn.pack(side="left")
    ttk.Label(opt_row2, text="Window:").pack(side="left", padx=(10, 2))
    flash_window_entry = ttk.Entry(opt_row2, textvariable=panel.flash_attn_window_var, width=6)
    flash_window_entry.pack(side="left")
    
    # Row 4: Context Chunking (swapped with ZeRO)
    chunk_row = ttk.Frame(opt_frame)
    chunk_row.pack(fill="x", pady=2)
    ttk.Label(chunk_row, text="Context Chunking:", width=15, anchor="e", font=("TkDefaultFont", 9, "bold")).pack(side="left")
    chunk_enable_btn = ttk.Checkbutton(chunk_row, text="Enable", variable=panel.use_chunked_training_var)
    chunk_enable_btn.pack(side="left")
    ttk.Label(chunk_row, text="Chunk Size:").pack(side="left", padx=(10, 2))
    chunk_size_combo = ttk.Combobox(chunk_row, textvariable=panel.chunk_size_var, width=8, state="readonly")
    chunk_size_combo['values'] = ('32', '64', '128', '256', '512', '1024', '2048', '4096', '8192')
    chunk_size_combo.pack(side="left")
    panel.chunk_info_lbl = ttk.Label(chunk_row, text="")
    panel.chunk_info_lbl.pack(side="left", padx=(10, 0))
    
    # Row 5: DeepSpeed ZeRO (swapped with Context Chunking)
    zero_row = ttk.Frame(opt_frame)
    zero_row.pack(fill="x", pady=2)
    ttk.Label(zero_row, text="DeepSpeed:", width=15, anchor="e", font=("TkDefaultFont", 9, "bold")).pack(side="left")
    ttk.Label(zero_row, text="ZeRO Stage:").pack(side="left", padx=(0, 2))
    zero_combo = ttk.Combobox(zero_row, textvariable=panel.zero_stage_var, width=8, state="readonly")
    zero_combo['values'] = ('none', 'zero1', 'zero2', 'zero3')
    zero_combo.pack(side="left")
    panel.zero_savings_lbl = ttk.Label(zero_row, text="")
    panel.zero_savings_lbl.pack(side="left", padx=(10, 0))
    
    # Store reference to zero_combo for dynamic state management
    panel.zero_combo = zero_combo
    
    # Row 6: MoE Learning Rate Auto-Adjust
    moe_lr_row = ttk.Frame(opt_frame)
    moe_lr_row.pack(fill="x", pady=2)
    ttk.Label(moe_lr_row, text="MoE LR:", width=15, anchor="e", font=("TkDefaultFont", 9, "bold")).pack(side="left")
    moe_lr_auto_btn = ttk.Checkbutton(moe_lr_row, text="Auto-adjust learning rate", variable=panel.auto_adjust_lr_var)
    moe_lr_auto_btn.pack(side="left")
    ttk.Label(moe_lr_row, text="Manual LR:").pack(side="left", padx=(10, 2))
    lr_entry = ttk.Entry(moe_lr_row, textvariable=panel.lr_var, width=10)
    lr_entry.pack(side="left")
    panel.moe_lr_info_lbl = ttk.Label(moe_lr_row, text="")
    panel.moe_lr_info_lbl.pack(side="left", padx=(10, 0))
    
    # Setup tooltips
    try:
        from ..tooltips import add_tooltip
        add_tooltip(grad_ckpt_btn, "Gradient Checkpointing: Trades computation for memory by\nrecomputing activations during backward pass\n↓30-50% VRAM usage • Enabled by default")
        add_tooltip(amp_btn, "Automatic Mixed Precision: Uses FP16/BF16 for faster training\n↓40-50% VRAM • +20% speed • Enabled by default")
        add_tooltip(opt8bit_btn, "8-bit Optimizer: Stores optimizer states in INT8\n↓75% optimizer memory • Minimal accuracy impact")
        add_tooltip(cpu_offload_btn, "CPU Offload: Moves optimizer states to system RAM\nSaves VRAM • ~30% slower training")
        add_tooltip(peft_enable_btn, "Enable PEFT: Use Low-Rank Adaptation (LoRA) for efficient fine-tuning\n↓95-99% trainable parameters (87M → 500K-2M)")
        add_tooltip(peft_method_combo, "PEFT Method:\n• LoRA: Low-Rank Adaptation (best balance)\n• AdaLoRA: Adaptive rank allocation\n• IA3: Fewer parameters than LoRA")
        add_tooltip(lora_r_entry, "LoRA Rank: Controls adapter capacity\n8=minimal • 16=balanced (default) • 32=high quality")
        add_tooltip(lora_alpha_entry, "LoRA Alpha: Scaling factor for adapter weights\nTypically 2× rank (default: 32 for rank 16)")
        add_tooltip(lora_dropout_entry, "LoRA Dropout: Regularization to prevent overfitting\n0.0=none • 0.05=default • 0.1-0.3=high regularization")
        add_tooltip(lora_modules_combo, "Target Modules: Which layers apply LoRA\nMinimal (~500K) • Balanced (~2M) • Full (~8M params)")
        add_tooltip(flash_attn_btn, "Flash Attention 2: Optimized attention for long contexts\n• OFF: Standard attention (no optimization)\n• ON: Try FA2 → fallback to PyTorch SDPA\nBest for 50K+ tokens • Requires Ampere+ GPU")
        add_tooltip(flash_window_entry, "Window Size: Sliding window for attention (works with FA2 and SDPA)\nDefault: None (full attention) • Range: 256-8192\nLarger = more context, more memory")
        add_tooltip(zero_combo, "DeepSpeed ZeRO: Distributed memory optimization\n• none: Standard training\n• zero1: Partition optimizer states (↓25% VRAM)\n• zero2: Partition optimizer + gradients (↓50% VRAM) [RECOMMENDED]\n• zero3: Partition everything (↓75% VRAM, slower)")
        add_tooltip(chunk_enable_btn, "Context Chunking: Split long sequences into smaller chunks\nReduces memory for extreme contexts (8K+ tokens)")
        add_tooltip(chunk_size_combo, "Chunk Size: Powers of 2 from 32-8192 tokens\nSmaller = less VRAM, slower • 2048-4096 typical")
        add_tooltip(moe_lr_auto_btn, "MoE Auto-Adjust: Automatically adjusts learning rate for MoE models to improve router stability\n\nWhen enabled for MoE models:\n• High LR (≥0.002) → adjusted to 0.001 (standard MoE starting rate)\n• Very Low LR (<0.0001) → increased to 0.0001 (minimum threshold)\n• Optimal range (0.0001-0.002) → used as-is\n\nAdjustments use 0.0001 increments for fine control.\n\nUnchecked = use manual LR value below\nChecked = automatic adjustment for stability (recommended)")
        add_tooltip(lr_entry, "Learning Rate: Controls training step size during optimization\n\nRecommended values:\n• Dense models: 0.00005 (5e-5) typical\n• MoE models: 0.0001 to 0.001 range\n  - Start: 0.001 for initial training\n  - Adjust by: ±0.0001 as needed\n  - Minimum: 0.0001\n\nNote: Auto-adjust checkbox above will optimize this value\nfor MoE models when enabled (recommended)")
    except Exception:
        pass
    
    # Setup label update callbacks
    panel.zero_stage_var.trace_add("write", lambda *args: update_zero_label(panel))
    panel.use_chunked_training_var.trace_add("write", lambda *args: update_chunk_label(panel))
    panel.chunk_size_var.trace_add("write", lambda *args: update_chunk_label(panel))
    
    # Initial updates
    update_zero_label(panel)
    update_chunk_label(panel)


def update_zero_label(panel: HRMTrainingPanel) -> None:
    """Update ZeRO savings label based on selected stage.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    try:
        stage = panel.zero_stage_var.get()
        if stage == "none":
            panel.zero_savings_lbl.config(text="")
        elif stage == "zero1":
            panel.zero_savings_lbl.config(text="↓25% VRAM")
        elif stage == "zero2":
            panel.zero_savings_lbl.config(text="↓50% VRAM (recommended)")
        elif stage == "zero3":
            panel.zero_savings_lbl.config(text="↓75% VRAM")
    except Exception:
        pass


def update_chunk_label(panel: HRMTrainingPanel) -> None:
    """Update chunk info label based on settings.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    try:
        if panel.use_chunked_training_var.get():
            chunk_size = panel.chunk_size_var.get().strip() or "2048"
            panel.chunk_info_lbl.config(text=f"Active: {chunk_size} token chunks")
        else:
            panel.chunk_info_lbl.config(text="")
    except Exception:
        pass
