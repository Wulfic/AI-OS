"""
Main train command orchestrator.

Coordinates dataset loading, training setup, and execution.
"""

from typing import List, Optional

import typer

from aios.cli.utils import load_config, setup_logging
from aios.core.replay import ReplayBuffer
from aios.core.train import Trainer

from .dataset_loading import load_dataset_file
from .supervised_loading import load_supervised_csv
from .training_setup import (
    create_train_config,
    setup_gpu_memory,
    setup_cuda_devices,
    setup_cost_budget,
    load_checkpoint_if_provided,
)
from .training_execution import (
    run_training_with_checkpointing,
    run_training_simple,
    run_training_with_budget,
)


def train(
    steps: int = typer.Option(50, "--steps", "-s", help="Training steps to run"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    use_torch: bool = typer.Option(False, "--torch", help="Use PyTorch backend if available"),
    device: str = typer.Option("auto", "--device", help="Torch device: auto|cpu|cuda|mps|xpu|dml (when --torch)"),
    amp: bool = typer.Option(True, "--amp/--no-amp", help="Enable AMP (mixed precision) when CUDA is available"),
    num_threads: int = typer.Option(0, "--num-threads", help="Torch CPU threads (0=auto)"),
    data_parallel: bool = typer.Option(True, "--data-parallel/--no-data-parallel", help="Enable torch.nn.DataParallel when multiple CUDA devices"),
    ddp: bool = typer.Option(False, "--ddp/--no-ddp", help="Enable DistributedDataParallel in this process (when launched under torchrun)"),
    cuda_devices: Optional[str] = typer.Option(None, "--cuda-devices", help="Comma-separated CUDA device IDs to use when --device=cuda"),
    dynamic_width: bool = typer.Option(False, "--dynamic-width/--no-dynamic-width", help="Enable dynamic hidden width growth/shrink"),
    width_min: int = typer.Option(8, "--width-min", help="Minimum hidden width when --dynamic-width"),
    width_max: int = typer.Option(1024, "--width-max", help="Maximum hidden width when --dynamic-width"),
    grow_patience: int = typer.Option(200, "--grow-patience", help="Window size (steps) before considering growth"),
    shrink_patience: int = typer.Option(400, "--shrink-patience", help="Window size (steps) before considering shrink"),
    grow_factor: float = typer.Option(2.0, "--grow-factor", help="Multiplier when growing width"),
    shrink_factor: float = typer.Option(1.5, "--shrink-factor", help="Divisor when shrinking width"),
    grow_threshold: float = typer.Option(1e-4, "--grow-threshold", help="Min improvement to avoid growth (plateau heuristic)"),
    sleep_downscale: float = typer.Option(0.01, "--sleep-downscale", help="Multiplicative downscale applied to weights during sleep"),
    sleep_steps: int = typer.Option(50, "--sleep-steps", help="Consolidation steps to run during each sleep cycle"),
    sleep_every: int = typer.Option(0, "--sleep-every", help="If >0, run one sleep cycle after every N training steps"),
    cost_budget: Optional[float] = typer.Option(None, "--budget", help="Optional total cost budget; if set, enforce budget"),
    cost_coef: float = typer.Option(0.1, "--cost-coef", help="Scale factor for synthetic cost"),
    save_ckpt: Optional[str] = typer.Option(None, "--save-ckpt", help="Optional path to save a checkpoint (.npz; will also write .pt if torch)"),
    load_ckpt: Optional[str] = typer.Option(None, "--load-ckpt", help="Optional path to load a checkpoint (.npz) before training"),
    checkpoint_every: int = typer.Option(0, "--checkpoint-every", help="If >0, save a checkpoint every N steps into --checkpoint-dir"),
    checkpoint_dir: Optional[str] = typer.Option(None, "--checkpoint-dir", help="Directory to save periodic checkpoints (defaults to ~/.local/share/aios/checkpoints)"),
    tag: Optional[str] = typer.Option(None, "--tag", help="Optional tag/prefix for periodic checkpoints"),
    emit_metrics: bool = typer.Option(False, "--emit-metrics", help="Persist a training_metrics artifact with recent losses"),
    gpu_mem_frac: float = typer.Option(0.9, "--gpu-mem-frac", help="When using CUDA, cap per-process GPU memory fraction (0.1-0.99)"),
    domains: Optional[str] = typer.Option(None, "--domains", help="Comma-separated domains/languages to bias synthetic training (e.g., english,python,bash)"),
    dataset_file: Optional[str] = typer.Option(None, "--dataset-file", help="Optional path to a text/CSV/archive/dir to sample English lines from"),
    text_feats: Optional[str] = typer.Option(None, "--text-feats", help="Alias of --dataset-file; provide a file/dir/archive for English text features"),
    hybrid: bool = typer.Option(False, "--hybrid/--no-hybrid", help="If true and --dataset-file is provided, combine dataset samples with synthetic domain-biased samples"),
    feature_dim: Optional[int] = typer.Option(None, "--feature-dim", help="Override input feature dimension (e.g., 512/1024) for text hashing"),
    supervised_csv: Optional[str] = typer.Option(None, "--supervised-csv", help="Path to labeled CSV for quick supervised English tasks"),
    csv_text_col: Optional[str] = typer.Option(None, "--csv-text-col", help="CSV text column (name or 0-based index)"),
    csv_label_col: Optional[str] = typer.Option(None, "--csv-label-col", help="CSV label column (name or 0-based index)"),
    task: Optional[str] = typer.Option(None, "--task", help="Supervised task spec; currently supports 'extract:key=REGEX' to reward lines matching REGEX"),
    instr: Optional[str] = typer.Option(None, "--instr", help="Instruction adherence spec, e.g., 'require=foo,bar;max_words=60;no_passive=true' (rewards pass=1.0 else 0.0)"),
    progress: bool = typer.Option(False, "--progress/--no-progress", help="Print training progress (step N / TOTAL) to stdout"),
):
    """Main training command entry point."""
    cfg = load_config(None)
    setup_logging(cfg)
    
    # Initialize replay buffer
    rb = ReplayBuffer(capacity=max(256, batch_size * 8))
    
    # Parse domains
    doms: List[str] = []
    if domains:
        doms = [d.strip() for d in str(domains).split(",") if d.strip()]
    if not doms:
        doms = ["generic"]
    
    # Handle text_feats alias
    if (not dataset_file) and text_feats:
        dataset_file = text_feats
    
    # Load datasets
    used_dataset = False
    english_corpus_metrics = None
    adherence_metrics = None
    
    if dataset_file:
        used_dataset, english_corpus_metrics, adherence_metrics = load_dataset_file(
            dataset_file, batch_size, rb, feature_dim, task, instr
        )
    
    if supervised_csv and feature_dim and (csv_text_col is not None) and (csv_label_col is not None):
        csv_used, csv_corpus, csv_adherence = load_supervised_csv(
            supervised_csv, csv_text_col, csv_label_col, batch_size, rb, feature_dim, task, instr
        )
        used_dataset = used_dataset or csv_used
        if csv_corpus:
            english_corpus_metrics = csv_corpus
        if csv_adherence:
            adherence_metrics = csv_adherence
    
    # Add synthetic data if hybrid or no dataset
    if hybrid or not used_dataset:
        domain_actions = {name: (idx % 5) for idx, name in enumerate(doms)}
        for i in range(128):
            dom = doms[i % len(doms)]
            a = domain_actions[dom]
            reward = float((i % 3) + (0.5 if dom != "generic" else 0.0))
            rb.push([0], a, reward, [0], False)
    
    # Create training config
    tcfg = create_train_config(
        use_torch, steps, batch_size, cost_coef, device, amp, num_threads,
        data_parallel, ddp, dynamic_width, width_min, width_max, grow_patience,
        shrink_patience, grow_factor, shrink_factor, grow_threshold,
        sleep_downscale, sleep_steps, feature_dim
    )
    
    # Setup GPU and devices
    setup_gpu_memory(use_torch, gpu_mem_frac)
    setup_cuda_devices(tcfg, cuda_devices)
    setup_cost_budget(tcfg, cost_budget)
    
    # Create trainer
    tr = Trainer(tcfg)
    load_checkpoint_if_provided(tr, load_ckpt)
    
    # Run training based on mode
    if checkpoint_every and checkpoint_every > 0:
        result = run_training_with_checkpointing(
            tr, rb, steps, checkpoint_every, checkpoint_dir, tag, sleep_every,
            cost_budget, progress, emit_metrics, english_corpus_metrics, task, adherence_metrics
        )
        print(result)
        
        # Run evaluation if dataset provided
        if dataset_file:
            try:
                from .english_eval_cmd import english_eval
                english_eval(
                    dataset_file=dataset_file,
                    checkpoint=result.get("last_checkpoint"),
                    feature_dim=feature_dim,
                    task=task,
                    instr=instr,
                    max_lines=max(2000, batch_size * 20),
                    label=tag or "ckpt",
                )
            except Exception:
                pass
    
    elif cost_budget is not None:
        result = run_training_with_budget(
            tr, rb, steps, progress, save_ckpt, emit_metrics, tag,
            english_corpus_metrics, task, adherence_metrics
        )
        print(result)
        
        # Run evaluation if dataset provided
        if dataset_file:
            try:
                from .english_eval_cmd import english_eval
                english_eval(
                    dataset_file=dataset_file,
                    checkpoint=save_ckpt,
                    feature_dim=feature_dim,
                    task=task,
                    instr=instr,
                    max_lines=max(2000, batch_size * 20),
                    label=tag or "train",
                )
            except Exception:
                pass
    
    else:
        result = run_training_simple(
            tr, rb, steps, sleep_every, progress, save_ckpt, emit_metrics, tag,
            english_corpus_metrics, task, adherence_metrics
        )
        print(result)
        
        # Run evaluation if dataset provided
        if dataset_file:
            try:
                from .english_eval_cmd import english_eval
                english_eval(
                    dataset_file=dataset_file,
                    checkpoint=save_ckpt,
                    feature_dim=feature_dim,
                    task=task,
                    instr=instr,
                    max_lines=max(2000, batch_size * 20),
                    label=tag or "train",
                )
            except Exception:
                pass
