from __future__ import annotations

import json
import os
import re
import subprocess as _sp
import sys
import tempfile as _tmp
import time
from pathlib import Path
from typing import Optional

import typer

from aios.core.train import average_checkpoints_npz
from aios.memory.store import get_db, init_db


def train_parallel(
    steps: int = typer.Option(500, "--steps", help="Training steps per device"),
    batch_size: int = typer.Option(64, "--batch-size", help="Batch size per device"),
    tag: str = typer.Option("parallel", "--tag", help="Tag prefix for per-device checkpoints"),
    use_cpu: bool = typer.Option(True, "--cpu/--no-cpu", help="Include CPU device"),
    use_cuda: bool = typer.Option(True, "--cuda/--no-cuda", help="Include NVIDIA/ROCm CUDA device(s)"),
    use_xpu: bool = typer.Option(True, "--xpu/--no-xpu", help="Include Intel XPU device if available"),
    use_dml: bool = typer.Option(True, "--dml/--no-dml", help="Include DirectML device if available (Windows Intel/AMD)"),
    use_mps: bool = typer.Option(True, "--mps/--no-mps", help="Include Apple MPS device if available"),
    gpu_mem_frac: float = typer.Option(0.9, "--gpu-mem-frac", help="Default cap per-process GPU memory fraction for CUDA/ROCm"),
    average: bool = typer.Option(True, "--average/--no-average", help="Average per-device checkpoints into a merged checkpoint"),
    dml_python: Optional[str] = typer.Option(None, "--dml-python", help="Path to Python interpreter with torch-directml installed (overrides default config)"),
    domains: Optional[str] = typer.Option(None, "--domains", help="Comma-separated domains/languages to bias training, e.g., english,python,bash"),
    dataset_file: Optional[str] = typer.Option(None, "--dataset-file", help="Optional path to a text/CSV file to sample lines from and seed training"),
    hybrid: bool = typer.Option(False, "--hybrid/--no-hybrid", help="Combine dataset samples with synthetic domain-biased samples"),
    cuda_ids: Optional[str] = typer.Option(None, "--cuda-ids", help="Comma-separated CUDA device IDs to use (e.g., 0,2). If omitted, use all available."),
    cuda_mem_map: Optional[str] = typer.Option(None, "--cuda-mem-map", help="JSON or 'id:frac;id2:frac2' mapping for per-GPU memory fraction overrides"),
    num_threads: int = typer.Option(0, "--num-threads", help="Override CPU threads per process (0 = auto: half cores)"),
    train_flags: Optional[str] = typer.Option(None, "--train-flags", help="Additional flags to pass to each per-device 'aios train' command (e.g., '--dynamic-width --checkpoint-every 50 --emit-metrics')"),
):
    devs: list[str] = []
    selected_cuda_ids: list[int] = []
    try:
        import torch  # type: ignore
        if use_cuda and torch.cuda.is_available():
            n = int(torch.cuda.device_count())
            all_ids = list(range(n)) if n > 0 else []
            if cuda_ids:
                try:
                    selected_cuda_ids = [int(x.strip()) for x in str(cuda_ids).split(",") if x.strip() != ""]
                    selected_cuda_ids = [i for i in selected_cuda_ids if i in all_ids]
                except Exception:
                    selected_cuda_ids = all_ids
            else:
                selected_cuda_ids = all_ids
            if selected_cuda_ids:
                devs.extend(["cuda" for _ in selected_cuda_ids])
        if use_xpu and getattr(torch, "xpu", None) and torch.xpu.is_available():  # type: ignore[attr-defined]
            devs.append("xpu")
        if use_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            devs.append("mps")
    except Exception:
        pass

    if not dml_python:
        try:
            cfg_path = Path.home() / ".config/aios/dml_python.txt"
            if cfg_path.exists():
                txt = cfg_path.read_text(encoding="utf-8").strip()
                if txt:
                    dml_python = txt
        except Exception:
            pass

    if use_dml:
        try:
            import torch_directml as _dml  # type: ignore
            _ = _dml.device()
            devs.append("dml")
        except Exception:
            if dml_python:
                devs.append("dml")
    if use_cpu:
        devs.append("cpu")

    if not devs:
        print({"started": False, "error": "no devices available"})
        raise typer.Exit(code=1)

    per_gpu_frac: dict[int, float] = {}
    if cuda_mem_map:
        try:
            m = json.loads(cuda_mem_map)
            if isinstance(m, dict):
                for k, v in m.items():
                    try:
                        kid = int(k)
                        per_gpu_frac[kid] = float(v)
                    except Exception:
                        continue
        except Exception:
            try:
                pairs = [p for p in re.split(r"[;,]", str(cuda_mem_map)) if p.strip()]
                for pr in pairs:
                    if ":" in pr or "=" in pr:
                        kv = re.split(r"[:=]", pr)
                        if len(kv) >= 2:
                            try:
                                kid = int(kv[0].strip())
                                per_gpu_frac[kid] = float(kv[1].strip())
                            except Exception:
                                pass
            except Exception:
                pass

    tmp = Path(_tmp.gettempdir())
    procs: list[_sp.Popen] = []
    ckpts: list[str] = []
    started: list[dict] = []
    cores = max(1, os.cpu_count() or 1)
    half = max(1, int(cores // 2))
    threads_per_proc = half if num_threads <= 0 else max(1, int(num_threads))
    for idx, d in enumerate(devs):
        ck = tmp / f"{tag}-{d}-{int(time.time())}.npz"
        ckpts.append(str(ck))
        exe = sys.executable
        if d == "dml" and dml_python:
            exe = dml_python
        args = [
            exe,
            "-m",
            "aios.cli.aios",
            "train",
            "--torch",
            "--device",
            d,
            "--steps",
            str(int(steps)),
            "--batch-size",
            str(int(batch_size)),
            "--num-threads",
            str(int(threads_per_proc)),
            "--tag",
            tag,
            "--save-ckpt",
            str(ck),
        ]
        if train_flags:
            try:
                from shlex import split as _shsplit
                args += _shsplit(train_flags)
            except Exception:
                pass
        if domains:
            args.extend(["--domains", domains])
        if dataset_file:
            args.extend(["--dataset-file", dataset_file])
            if hybrid:
                args.append("--hybrid")
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(half)
        env["OPENBLAS_NUM_THREADS"] = str(half)
        env["MKL_NUM_THREADS"] = str(half)
        if d == "cuda" and selected_cuda_ids:
            cuda_index = sum(1 for t in devs[:idx+1] if t == "cuda") - 1
            gpu_id = selected_cuda_ids[min(cuda_index, len(selected_cuda_ids)-1)]
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            frac = float(max(0.1, min(0.99, per_gpu_frac.get(gpu_id, gpu_mem_frac))))
            args.extend(["--gpu-mem-frac", str(frac)])
        else:
            args.extend(["--gpu-mem-frac", str(float(max(0.1, min(0.99, gpu_mem_frac))))])
        p = _sp.Popen(args, env=env)
        procs.append(p)
        started.append({"device": d, "pid": p.pid})

    rcodes = []
    for p in procs:
        try:
            rc = p.wait()
        except Exception:
            rc = -1
        rcodes.append(rc)

    merged: Optional[Path] = None
    if average:
        paths = [p for p in ckpts if Path(p).exists()]
        if len(paths) >= 2:
            out_dir = Path.home() / ".local/share/aios/checkpoints"
            out_dir.mkdir(parents=True, exist_ok=True)
            merged = out_dir / f"{tag}-parallel-merged-{int(time.time())}.npz"
            ok = average_checkpoints_npz(paths, str(merged))
            if ok:
                try:
                    conn = get_db()
                    init_db(conn)
                    from aios.memory.store import save_artifact
                    save_artifact(conn, kind="training_checkpoint", label=tag, data={"path": str(merged)})
                finally:
                    try:
                        conn.close()  # type: ignore[name-defined]
                    except Exception:
                        pass

    print({"started": started, "returncodes": rcodes, "merged": (str(merged) if merged is not None else None)})
