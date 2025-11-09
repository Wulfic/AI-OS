"""
CLI options for train command.

Defines all typer options for the training CLI.
"""

from typing import Optional

import typer


def get_train_options():
    """Return dictionary of all training CLI options.
    
    This function defines all the typer.Option() objects for the train command.
    Returns them as a dictionary that can be unpacked into the function signature.
    """
    return {
        "steps": typer.Option(50, "--steps", "-s", help="Training steps to run"),
        "batch_size": typer.Option(32, "--batch-size", "-b", help="Batch size"),
        "use_torch": typer.Option(False, "--torch", help="Use PyTorch backend if available"),
        "device": typer.Option("auto", "--device", help="Torch device: auto|cpu|cuda|mps|xpu|dml (when --torch)"),
        "amp": typer.Option(True, "--amp/--no-amp", help="Enable AMP (mixed precision) when CUDA is available"),
        "num_threads": typer.Option(0, "--num-threads", help="Torch CPU threads (0=auto)"),
        "data_parallel": typer.Option(True, "--data-parallel/--no-data-parallel", help="Enable torch.nn.DataParallel when multiple CUDA devices"),
        "ddp": typer.Option(False, "--ddp/--no-ddp", help="Enable DistributedDataParallel in this process (when launched under torchrun)"),
        "cuda_devices": typer.Option(None, "--cuda-devices", help="Comma-separated CUDA device IDs to use when --device=cuda"),
        "dynamic_width": typer.Option(False, "--dynamic-width/--no-dynamic-width", help="Enable dynamic hidden width growth/shrink"),
        "width_min": typer.Option(8, "--width-min", help="Minimum hidden width when --dynamic-width"),
        "width_max": typer.Option(1024, "--width-max", help="Maximum hidden width when --dynamic-width"),
        "grow_patience": typer.Option(200, "--grow-patience", help="Window size (steps) before considering growth"),
        "shrink_patience": typer.Option(400, "--shrink-patience", help="Window size (steps) before considering shrink"),
        "grow_factor": typer.Option(2.0, "--grow-factor", help="Multiplier when growing width"),
        "shrink_factor": typer.Option(1.5, "--shrink-factor", help="Divisor when shrinking width"),
        "grow_threshold": typer.Option(1e-4, "--grow-threshold", help="Min improvement to avoid growth (plateau heuristic)"),
        "sleep_downscale": typer.Option(0.01, "--sleep-downscale", help="Multiplicative downscale applied to weights during sleep"),
        "sleep_steps": typer.Option(50, "--sleep-steps", help="Consolidation steps to run during each sleep cycle"),
        "sleep_every": typer.Option(0, "--sleep-every", help="If >0, run one sleep cycle after every N training steps"),
        "cost_budget": typer.Option(None, "--budget", help="Optional total cost budget; if set, enforce budget"),
        "cost_coef": typer.Option(0.1, "--cost-coef", help="Scale factor for synthetic cost"),
        "save_ckpt": typer.Option(None, "--save-ckpt", help="Optional path to save a checkpoint (.npz; will also write .pt if torch)"),
        "load_ckpt": typer.Option(None, "--load-ckpt", help="Optional path to load a checkpoint (.npz) before training"),
        "checkpoint_every": typer.Option(0, "--checkpoint-every", help="If >0, save a checkpoint every N steps into --checkpoint-dir"),
        "checkpoint_dir": typer.Option(None, "--checkpoint-dir", help="Directory to save periodic checkpoints (defaults to ~/.local/share/aios/checkpoints)"),
        "tag": typer.Option(None, "--tag", help="Optional tag/prefix for periodic checkpoints"),
        "emit_metrics": typer.Option(False, "--emit-metrics", help="Persist a training_metrics artifact with recent losses"),
        "gpu_mem_frac": typer.Option(0.9, "--gpu-mem-frac", help="When using CUDA, cap per-process GPU memory fraction (0.1-0.99)"),
        "domains": typer.Option(None, "--domains", help="Comma-separated domains/languages to bias synthetic training (e.g., english,python,bash)"),
        "dataset_file": typer.Option(None, "--dataset-file", help="Optional path to a text/CSV/archive/dir to sample English lines from"),
        "text_feats": typer.Option(None, "--text-feats", help="Alias of --dataset-file; provide a file/dir/archive for English text features"),
        "hybrid": typer.Option(False, "--hybrid/--no-hybrid", help="If true and --dataset-file is provided, combine dataset samples with synthetic domain-biased samples"),
        "feature_dim": typer.Option(None, "--feature-dim", help="Override input feature dimension (e.g., 512/1024) for text hashing"),
        "supervised_csv": typer.Option(None, "--supervised-csv", help="Path to labeled CSV for quick supervised English tasks"),
        "csv_text_col": typer.Option(None, "--csv-text-col", help="CSV text column (name or 0-based index)"),
        "csv_label_col": typer.Option(None, "--csv-label-col", help="CSV label column (name or 0-based index)"),
        "task": typer.Option(None, "--task", help="Supervised task spec; currently supports 'extract:key=REGEX' to reward lines matching REGEX"),
        "instr": typer.Option(None, "--instr", help="Instruction adherence spec, e.g., 'require=foo,bar;max_words=60;no_passive=true' (rewards pass=1.0 else 0.0)"),
        "progress": typer.Option(False, "--progress/--no-progress", help="Print training progress (step N / TOTAL) to stdout"),
    }
