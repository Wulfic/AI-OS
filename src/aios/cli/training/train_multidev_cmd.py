from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import typer

from aios.core.replay import ReplayBuffer
from aios.core.train import TrainConfig, Trainer, average_checkpoints_npz
from aios.memory.store import get_db, init_db


def train_multidev(
    steps_per_round: int = typer.Option(200, "--steps-per-round", help="Steps to train on each device per round"),
    rounds: int = typer.Option(2, "--rounds", help="Number of multi-device rounds"),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size"),
    dynamic_width: bool = typer.Option(False, "--dynamic-width/--no-dynamic-width", help="Enable dynamic width"),
    tag: str = typer.Option("multidev", "--tag", help="Checkpoint tag prefix"),
    emit_metrics: bool = typer.Option(True, "--emit-metrics/--no-emit-metrics", help="Emit training_metrics per segment"),
    gpu_mem_frac: float = typer.Option(0.9, "--gpu-mem-frac", help="Cap per-process GPU memory fraction for CUDA/ROCm"),
):
    tmp = Path(tempfile.gettempdir())
    devs = ["cpu"]
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            devs.append("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
            devs.append("xpu")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            devs.append("mps")
        try:
            import torch_directml as _dml  # type: ignore
            _ = _dml.device()
            devs.append("dml")
        except Exception:
            pass
        try:
            if getattr(torch.version, "hip", None):  # type: ignore[attr-defined]
                devs.append("rocm")
        except Exception:
            pass
        try:
            import torch_directml as _dml  # type: ignore
            _ = _dml.device()
            devs.append("dml")
        except Exception:
            pass
    except Exception:
        pass

    rb = ReplayBuffer(capacity=max(128, batch_size * 4))
    for i in range(64):
        a = i % 5
        rb.push([0], a, float(a), [0], False)

    merged_ckpt: Optional[Path] = None
    last_loss = None
    for r in range(int(rounds)):
        round_ckpts: list[str] = []
        for d in devs:
            try:
                import multiprocessing as _mp
                cores = max(1, _mp.cpu_count())
                half = max(1, int(cores // 2))
                os.environ["OMP_NUM_THREADS"] = str(half)
                os.environ["OPENBLAS_NUM_THREADS"] = str(half)
                os.environ["MKL_NUM_THREADS"] = str(half)
                num_threads_seg = half
            except Exception:
                num_threads_seg = 0

            if d in ("cuda", "rocm"):
                try:
                    import torch  # type: ignore
                    frac = float(max(0.1, min(0.99, float(gpu_mem_frac))))
                    if torch.cuda.is_available():
                        ndev = int(torch.cuda.device_count())
                        for dd in range(ndev):
                            try:
                                torch.cuda.set_per_process_memory_fraction(frac, device=dd)  # type: ignore[attr-defined]
                            except Exception:
                                pass
                except Exception:
                    pass
            tcfg = TrainConfig(use_torch=True, device=d, batch_size=batch_size, max_steps=steps_per_round, dynamic_width=dynamic_width, num_threads=num_threads_seg)
            tr = Trainer(tcfg)
            if merged_ckpt and merged_ckpt.exists():
                try:
                    tr.load_checkpoint(str(merged_ckpt))
                except Exception:
                    pass
            last_loss = tr.train(rb, steps=steps_per_round)
            ck_name = f"{tag}-r{r+1}-{d}-{int(time.time())}.npz"
            ck_path = tmp / ck_name
            tr.save_checkpoint(str(ck_path), {"round": r + 1, "device": d})
            round_ckpts.append(str(ck_path))
        merged_path = tmp / f"{tag}-merged-r{r+1}.npz"
        ok = average_checkpoints_npz(round_ckpts, str(merged_path))
        if ok:
            merged_ckpt = merged_path
        if emit_metrics:
            try:
                conn = get_db()
                init_db(conn)
                from aios.memory.store import save_artifact
                save_artifact(conn, kind="training_metrics", label=tag, data={"round": r + 1, "devices": devs, "last_loss": float(last_loss) if last_loss is not None else None})
            finally:
                try:
                    conn.close()  # type: ignore[name-defined]
                except Exception:
                    pass
    final: Optional[Path] = None
    if merged_ckpt and merged_ckpt.exists():
        ck_dir = Path.home() / ".local/share/aios/checkpoints"
        ck_dir.mkdir(parents=True, exist_ok=True)
        final = ck_dir / f"{tag}-final-{int(time.time())}.npz"
        shutil.copy2(str(merged_ckpt), str(final))
        try:
            conn = get_db()
            init_db(conn)
            from aios.memory.store import save_artifact
            save_artifact(conn, kind="training_checkpoint", label=tag, data={"path": str(final)})
        finally:
            try:
                conn.close()  # type: ignore[name-defined]
            except Exception:
                pass
    print({"devices": devs, "rounds": int(rounds), "last_loss": float(last_loss) if last_loss is not None else None, "final_checkpoint": (str(final) if final is not None else None)})
