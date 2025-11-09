from __future__ import annotations

import subprocess as _sp
import sys
from typing import Optional

import typer


def train_ddp(
    steps: int = typer.Option(500, "--steps", help="Training steps per process"),
    batch_size: int = typer.Option(64, "--batch-size", help="Batch size per process"),
    world_size: int = typer.Option(0, "--world-size", help="Number of processes (0=auto from CUDA device_count)"),
    amp: bool = typer.Option(True, "--amp/--no-amp", help="Enable AMP on CUDA"),
    tag: str = typer.Option("ddp", "--tag", help="Checkpoint tag prefix"),
    checkpoint_dir: Optional[str] = typer.Option(None, "--checkpoint-dir", help="Dir for periodic checkpoints"),
):
    use_cuda = False
    dev_count = 0
    try:
        import torch  # type: ignore
        use_cuda = torch.cuda.is_available()
        dev_count = torch.cuda.device_count() if use_cuda else 0
    except Exception:
        use_cuda = False
        dev_count = 0
    if world_size <= 0:
        world_size = max(1, dev_count)

    device = "cuda" if use_cuda and world_size > 0 else "cpu"
    args = [
        sys.executable,
        "-m",
        "aios.cli.aios",
        "train",
        "--torch",
        "--ddp",
        "--device",
        device,
        "--steps",
        str(int(steps)),
        "--batch-size",
        str(int(batch_size)),
        "--tag",
        tag,
        "--checkpoint-every",
        "0",
    ]
    args.append("--amp" if amp and device == "cuda" else "--no-amp")
    if checkpoint_dir:
        args.extend(["--checkpoint-dir", checkpoint_dir])

    if world_size == 1:
        res = _sp.run(args, check=False)
        typer.echo({"world_size": world_size, "returncode": res.returncode})
        raise typer.Exit(code=res.returncode)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc_per_node",
        str(int(world_size)),
    ] + args
    typer.echo({"cmd": " ".join(cmd)})
    res = _sp.run(cmd, check=False)
    typer.echo({"world_size": world_size, "returncode": res.returncode})
    raise typer.Exit(code=res.returncode)
