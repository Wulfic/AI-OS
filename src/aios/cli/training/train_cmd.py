from __future__ import annotations

import json
import os
import re
import shutil
import subprocess as _sp
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import typer

from aios.cli.utils import load_config, setup_logging
from aios.core.replay import ReplayBuffer
from aios.core.train import Trainer, TrainConfig
from aios.memory.store import get_db, init_db
from aios.ml.english_metrics import summarize_corpus

from .english_eval_cmd import english_eval


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
    cost_budget: float = typer.Option(None, "--budget", help="Optional total cost budget; if set, enforce budget"),
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
    cfg = load_config(None)
    setup_logging(cfg)

    rb = ReplayBuffer(capacity=max(256, batch_size * 8))
    doms: List[str] = []
    if domains:
        doms = [d.strip() for d in str(domains).split(",") if d.strip()]
    if not doms:
        doms = ["generic"]

    used_dataset = False
    english_corpus_metrics: Optional[dict] = None
    adherence_metrics: Optional[dict] = None
    _instr_spec: Optional[dict] = None
    _ad_pass: int = 0
    _ad_tot: int = 0
    _ad_passive: int = 0
    if (not dataset_file) and text_feats:
        dataset_file = text_feats

    def _parse_instr(spec: str) -> dict:
        d: dict = {}
        try:
            parts = [p.strip() for p in spec.split(";") if p.strip()]
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    d[k.strip().lower()] = v.strip()
                else:
                    d[p.strip().lower()] = True
        except Exception:
            pass
        for list_key in ("require", "require_any", "forbid"):
            if list_key in d and isinstance(d[list_key], str):
                d[list_key] = [w.strip() for w in d[list_key].split(",") if w.strip()]
        try:
            if "max_words" in d:
                d["max_words"] = int(d["max_words"])  # type: ignore[assignment]
            if "min_words" in d:
                d["min_words"] = int(d["min_words"])  # type: ignore[assignment]
            if "max_chars" in d:
                d["max_chars"] = int(d["max_chars"])  # type: ignore[assignment]
            if "min_chars" in d:
                d["min_chars"] = int(d["min_chars"])  # type: ignore[assignment]
            if "bullets_min" in d:
                d["bullets_min"] = int(d["bullets_min"])  # type: ignore[assignment]
            if "sentences_max" in d:
                d["sentences_max"] = int(d["sentences_max"])  # type: ignore[assignment]
        except Exception:
            d.pop("max_words", None)
            d.pop("min_words", None)
            d.pop("max_chars", None)
            d.pop("min_chars", None)
            d.pop("bullets_min", None)
            d.pop("sentences_max", None)
        d["no_passive"] = str(d.get("no_passive", "false")).lower() in ("1", "true", "yes")
        return d

    def _pass_instr(line: str, spec: dict) -> tuple[bool, dict]:
        lw = (line or "").lower()
        words = [w for w in lw.split() if w]
        wcount = len(words)
        chars = len(lw)
        required = list(spec.get("require", []) or [])
        require_any = list(spec.get("require_any", []) or [])
        forbid = list(spec.get("forbid", []) or [])
        req_ok = all((r.lower() in lw) for r in required)
        any_ok = (True if not require_any else any((r.lower() in lw) for r in require_any))
        forbid_ok = all((f.lower() not in lw) for f in forbid)
        maxw = spec.get("max_words")
        minw = spec.get("min_words")
        len_ok = True
        if isinstance(maxw, int) and maxw > 0:
            len_ok = len_ok and (wcount <= int(maxw))
        if isinstance(minw, int) and minw > 0:
            len_ok = len_ok and (wcount >= int(minw))
        maxc = spec.get("max_chars")
        minc = spec.get("min_chars")
        char_ok = True
        if isinstance(maxc, int) and maxc > 0:
            char_ok = char_ok and (chars <= int(maxc))
        if isinstance(minc, int) and minc > 0:
            char_ok = char_ok and (chars >= int(minc))
        bullets_min = spec.get("bullets_min")
        sentences_max = spec.get("sentences_max")
        bullets_ok = True
        sentences_ok = True
        try:
            import re as _re
            bullet_count = 0
            for ln in (line or "").splitlines():
                if _re.match(r"^\s*(?:[-*•]|\d+[\.)])\s+", ln):
                    bullet_count += 1
            if isinstance(bullets_min, int) and bullets_min > 0:
                bullets_ok = (bullet_count >= int(bullets_min))
            sent_count = len([s for s in _re.split(r"[.!?]+", lw) if s.strip()])
            if isinstance(sentences_max, int) and sentences_max > 0:
                sentences_ok = (sent_count <= int(sentences_max))
        except Exception:
            pass
        passive = False
        if bool(spec.get("no_passive", False)):
            try:
                import re as _re
                passive = bool(_re.search(r"\b(?:was|were|be|been|being|is|are|am)\b\s+\w+ed\b", lw))
            except Exception:
                passive = False
        ok = (req_ok and any_ok and forbid_ok and len_ok and char_ok and bullets_ok and sentences_ok and (not passive))
        return ok, {"words": wcount, "chars": chars, "req_ok": req_ok, "any_ok": any_ok, "forbid_ok": forbid_ok, "len_ok": len_ok, "char_ok": char_ok, "bullets_ok": bullets_ok, "sentences_ok": sentences_ok, "passive": passive}

    if dataset_file:
        try:
            from aios.data.datasets import read_text_lines_sample_any
            from aios.ml.text_features import featurize_bow_hashing, featurize_hashing
            lines = read_text_lines_sample_any(dataset_file, max_lines=max(2000, batch_size * 20))
            texts: list[str] = []
            for ln in lines:
                t = ln
                if ln and ln.strip().startswith("{"):
                    try:
                        rec = json.loads(ln)
                        if isinstance(rec, dict) and rec.get("text"):
                            t = str(rec.get("text"))
                    except Exception:
                        pass
                if t:
                    texts.append(t)
            if texts:
                lines = texts
            try:
                if lines:
                    english_corpus_metrics = summarize_corpus(lines, max_samples=5000)
            except Exception:
                english_corpus_metrics = None
            dim = int(feature_dim) if feature_dim and feature_dim > 0 else None
            if instr and _instr_spec is None:
                try:
                    _instr_spec = _parse_instr(instr)
                except Exception:
                    _instr_spec = {}
            task_match_count = 0
            task_total = 0
            for i, ln in enumerate(lines):
                a = (i % 5)
                r = float(1.0 + (len(ln) % 3) * 0.1)
                task_metrics: dict | None = None
                if task:
                    try:
                        spec = str(task).strip()
                        if spec.lower().startswith("extract:"):
                            rhs = spec.split(":", 1)[1].strip()
                            if "=" in rhs:
                                _, rgx = rhs.split("=", 1)
                            else:
                                rgx = rhs
                            import re as _re
                            m = _re.search(rgx, ln or "")
                            r = 1.0 if m else 0.0
                            task_total += 1
                            if m:
                                task_match_count += 1
                    except Exception:
                        pass
                if instr and _instr_spec is not None:
                    try:
                        ok, meta = _pass_instr(ln or "", _instr_spec)
                        r = 1.0 if ok else 0.0
                        _ad_pass += 1 if ok else 0
                        _ad_tot += 1
                        if meta.get("passive"):
                            _ad_passive += 1
                    except Exception:
                        pass
                if dim:
                    try:
                        x = featurize_bow_hashing(ln, dim=dim).astype("float32")
                    except Exception:
                        x = featurize_hashing(ln, dim=dim).astype("float32")
                    rb.push(x, a, r, x, False)
                else:
                    rb.push([0], a, r, [0], False)
            used_dataset = len(lines) > 0
            english_task_metrics: Optional[dict] = None
            if task and task_total > 0:
                try:
                    english_task_metrics = {
                        "matches": int(task_match_count),
                        "total": int(task_total),
                        "match_rate": float(task_match_count / max(1, task_total)),
                    }
                except Exception:
                    english_task_metrics = None
            if instr and _ad_tot > 0:
                adherence_metrics = {
                    "spec": instr,
                    "count": int(_ad_tot),
                    "pass": int(_ad_pass),
                    "rate": float(_ad_pass / _ad_tot),
                    "passive": int(_ad_passive),
                }
        except Exception:
            used_dataset = False

    if supervised_csv and feature_dim and (csv_text_col is not None) and (csv_label_col is not None):
        try:
            from aios.data.datasets import read_csv_text_label_samples
            from aios.ml.text_features import featurize_bow_hashing, featurize_hashing
            pairs = read_csv_text_label_samples(supervised_csv, text_col=(csv_text_col if csv_text_col.isdigit() else csv_text_col), label_col=(csv_label_col if csv_label_col.isdigit() else csv_label_col), max_rows=max(5000, batch_size * 50))
            try:
                if pairs:
                    english_corpus_metrics = summarize_corpus([t for (t, _) in pairs], max_samples=5000)
            except Exception:
                pass
            dim = int(feature_dim)
            english_task_metrics: Optional[dict] = None
            tp = fp = tn = fn = 0
            thr = 0.5
            rx: Optional[str] = None
            if task:
                try:
                    spec = str(task).strip()
                    if spec.lower().startswith("extract:"):
                        rhs = spec.split(":", 1)[1].strip()
                        rx = (rhs.split("=", 1)[1] if "=" in rhs else rhs)
                except Exception:
                    rx = None
            if instr and _instr_spec is None:
                try:
                    _instr_spec = _parse_instr(instr)
                except Exception:
                    _instr_spec = {}
            for i, (txt, y) in enumerate(pairs):
                try:
                    x = featurize_bow_hashing(txt, dim=dim).astype("float32")
                except Exception:
                    x = featurize_hashing(txt, dim=dim).astype("float32")
                a = (i % 5)
                r = float(y)
                if rx is not None:
                    try:
                        import re as _re
                        pred_pos = bool(_re.search(rx, txt or ""))
                        try:
                            yv = float(y)
                        except Exception:
                            yv = 1.0 if str(y).strip() not in ("0", "", "false", "False") else 0.0
                        actual_pos = (yv >= thr)
                        if pred_pos and actual_pos:
                            tp += 1
                        elif pred_pos and (not actual_pos):
                            fp += 1
                        elif (not pred_pos) and (not actual_pos):
                            tn += 1
                        else:
                            fn += 1
                    except Exception:
                        pass
                if instr and _instr_spec is not None:
                    try:
                        ok, meta = _pass_instr(txt or "", _instr_spec)
                        r = 1.0 if ok else 0.0
                        _ad_pass += 1 if ok else 0
                        _ad_tot += 1
                        if meta.get("passive"):
                            _ad_passive += 1
                    except Exception:
                        pass
                rb.push(x, a, float(r), x, False)
            used_dataset = used_dataset or (len(pairs) > 0)
            if rx is not None:
                try:
                    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                    rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                    acc = ((tp + tn) / max(1, (tp + tn + fp + fn)))
                    english_task_metrics = {
                        "precision": float(prec),
                        "recall": float(rec),
                        "f1": float(f1),
                        "accuracy": float(acc),
                        "tp": int(tp),
                        "fp": int(fp),
                        "tn": int(tn),
                        "fn": int(fn),
                        "threshold": float(thr),
                    }
                except Exception:
                    english_task_metrics = None
            if instr and _ad_tot > 0:
                adherence_metrics = {
                    "spec": instr,
                    "count": int(_ad_tot),
                    "pass": int(_ad_pass),
                    "rate": float(_ad_pass / _ad_tot),
                    "passive": int(_ad_passive),
                }
        except Exception:
            pass

    if hybrid or not used_dataset:
        domain_actions = {name: (idx % 5) for idx, name in enumerate(doms)}
        for i in range(128):
            dom = doms[i % len(doms)]
            a = domain_actions[dom]
            reward = float((i % 3) + (0.5 if dom != "generic" else 0.0))
            rb.push([0], a, reward, [0], False)

    tcfg = TrainConfig(
        use_torch=use_torch,
        max_steps=steps,
        batch_size=batch_size,
        cost_coef=cost_coef,
        device=device,
        amp=amp,
        num_threads=num_threads,
        data_parallel=data_parallel,
        ddp=ddp,
        dynamic_width=dynamic_width,
        width_min=width_min,
        width_max=width_max,
        grow_patience=grow_patience,
        shrink_patience=shrink_patience,
        grow_factor=grow_factor,
        shrink_factor=shrink_factor,
        grow_threshold=grow_threshold,
        sleep_downscale=sleep_downscale,
        sleep_consolidation_steps=sleep_steps,
    )
    if feature_dim and feature_dim > 0:
        tcfg.input_dim = int(feature_dim)

    if use_torch:
        try:
            import torch  # type: ignore
            frac = float(max(0.1, min(0.99, float(gpu_mem_frac))))
            if torch.cuda.is_available() and frac < 0.995:
                try:
                    ndev = int(torch.cuda.device_count())
                    for d in range(ndev):
                        try:
                            torch.cuda.set_per_process_memory_fraction(frac, device=d)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass
    if cuda_devices:
        try:
            tcfg.cuda_devices = [int(x.strip()) for x in str(cuda_devices).split(",") if x.strip() != ""]
        except Exception:
            tcfg.cuda_devices = None
    if cost_budget is not None:
        tcfg.cost_budget = float(cost_budget)
    tr = Trainer(tcfg)
    if load_ckpt:
        try:
            ok = tr.load_checkpoint(load_ckpt)
            if not ok:
                import logging
                logging.getLogger(__name__).warning("failed to load checkpoint: %s", load_ckpt)
        except Exception:
            import logging
            logging.getLogger(__name__).exception("error loading checkpoint: %s", load_ckpt)

    if checkpoint_every and checkpoint_every > 0:
        losses: list[float] = []
        costs: list[float] = []
        ck_dir = Path(checkpoint_dir or (Path.home() / ".local/share/aios/checkpoints"))
        ck_dir.mkdir(parents=True, exist_ok=True)
        prefix = tag or "ckpt"
        steps_run = 0
        over = False
        last_ck: Optional[str] = None
        for i in range(steps):
            if cost_budget is None:
                loss = tr.train_step(rb)
                cost = 0.0
            else:
                loss, cost = tr.train_step_with_cost(rb)
                if tr.total_cost > tr.cfg.cost_budget:
                    over = True
            steps_run = i + 1
            losses.append(float(loss))
            costs.append(float(cost))
            if progress:
                try:
                    every = max(1, int(steps // 100) or 1)
                    if (i < 5) or (steps_run % every == 0) or (steps - steps_run) < 5:
                        print(f"step {steps_run} / {steps}", flush=True)
                except Exception:
                    pass
            if sleep_every and sleep_every > 0 and ((i + 1) % sleep_every == 0):
                try:
                    tr.sleep_cycle(rb)
                except Exception:
                    pass
            if steps_run % checkpoint_every == 0 or (i + 1) == steps:
                ts = int(time.time())
                ck_path = ck_dir / f"{prefix}-step{steps_run}-{ts}.npz"
                try:
                    tr.save_checkpoint(str(ck_path), {"steps": steps_run})
                    last_ck = str(ck_path)
                    try:
                        conn = get_db()
                        init_db(conn)
                        from aios.memory.store import save_artifact
                        save_artifact(conn, kind="training_checkpoint", label=prefix, data={"path": str(ck_path), "steps": steps_run})
                    finally:
                        try:
                            conn.close()  # type: ignore[name-defined]
                        except Exception:
                            pass
                except Exception:
                    pass
            if over:
                break
        if emit_metrics:
            try:
                conn = get_db()
                init_db(conn)
                from aios.memory.store import save_artifact
                data_dict = {
                    "steps": steps_run,
                    "losses": losses[-min(len(losses), 200):],
                    "costs": costs[-min(len(costs), 200):],
                    "over_budget": over,
                }
                if english_corpus_metrics:
                    data_dict["english_corpus"] = english_corpus_metrics
                if task:
                    et: dict = {"spec": task}
                    try:
                        if 'english_task_metrics' in locals() and locals().get('english_task_metrics'):
                            et.update(locals().get('english_task_metrics'))  # type: ignore[arg-type]
                    except Exception:
                        pass
                    data_dict["english_task"] = et
                if adherence_metrics:
                    data_dict["english_adherence"] = adherence_metrics
                save_artifact(conn, kind="training_metrics", label=prefix, data=data_dict)
            finally:
                try:
                    conn.close()  # type: ignore[name-defined]
                except Exception:
                    pass
        print({
            "torch": tr.torch_available,
            "steps_run": steps_run,
            "last_loss": float(losses[-1]) if losses else None,
            "over_budget": over,
            "checkpoint_dir": str(ck_dir),
            "tag": prefix,
        })
        try:
            if dataset_file:
                english_eval(
                    dataset_file=dataset_file,
                    checkpoint=last_ck,
                    feature_dim=feature_dim,
                    task=task,
                    instr=instr,
                    max_lines=max(2000, batch_size * 20),
                    label=prefix,
                )
        except Exception:
            pass
        return

    if cost_budget is None:
        if progress:
            loss = 0.0
            steps_run = int(steps)
            for i in range(steps_run):
                loss = tr.train_step(rb)
                if sleep_every and sleep_every > 0 and ((i + 1) % sleep_every == 0):
                    try:
                        tr.sleep_cycle(rb)
                    except Exception:
                        pass
                try:
                    every = max(1, int(steps_run // 100) or 1)
                    if (i < 5) or ((i + 1) % every == 0) or (steps_run - (i + 1)) < 5:
                        print(f"step {i+1} / {steps_run}", flush=True)
                except Exception:
                    pass
        else:
            loss = tr.train(rb, steps=steps)
        out: dict = {"loss": loss, "torch": tr.torch_available}
        if save_ckpt:
            try:
                tr.save_checkpoint(save_ckpt, {"steps": steps})
                out["checkpoint_saved"] = save_ckpt
                try:
                    conn = get_db()
                    init_db(conn)
                    from aios.memory.store import save_artifact
                    save_artifact(conn, kind="training_checkpoint", label="train", data={"path": save_ckpt, "steps": steps})
                except Exception:
                    pass
                finally:
                    try:
                        conn.close()  # type: ignore[name-defined]
                    except Exception:
                        pass
            except Exception:
                out["checkpoint_saved"] = False
        print(out)
        try:
            if dataset_file:
                english_eval(
                    dataset_file=dataset_file,
                    checkpoint=(save_ckpt if save_ckpt else None),
                    feature_dim=feature_dim,
                    task=task,
                    instr=instr,
                    max_lines=max(2000, batch_size * 20),
                    label=(tag or "train"),
                )
        except Exception:
            pass
        if emit_metrics:
            try:
                conn = get_db()
                init_db(conn)
                from aios.memory.store import save_artifact
                data_dict = {
                    "steps": int(steps),
                    "last_loss": float(loss) if loss is not None else None,
                }
                if english_corpus_metrics:
                    data_dict["english_corpus"] = english_corpus_metrics
                if task:
                    et: dict = {"spec": task}
                    try:
                        if 'english_task_metrics' in locals() and locals().get('english_task_metrics'):
                            et.update(locals().get('english_task_metrics'))  # type: ignore[arg-type]
                    except Exception:
                        pass
                    data_dict["english_task"] = et
                if adherence_metrics:
                    data_dict["english_adherence"] = adherence_metrics
                save_artifact(conn, kind="training_metrics", label=(tag or "train"), data=data_dict)
            finally:
                try:
                    conn.close()  # type: ignore[name-defined]
                except Exception:
                    pass
    else:
        if progress:
            n = int(steps)
            tr.total_cost = 0.0
            last_loss = 0.0
            over = False
            ran = 0
            for i in range(n):
                last_loss, _ = tr.train_step_with_cost(rb)
                ran = i + 1
                try:
                    every = max(1, int(n // 100) or 1)
                    if (i < 5) or (ran % every == 0) or (n - ran) < 5:
                        print(f"step {ran} / {n}", flush=True)
                except Exception:
                    pass
                if tr.total_cost > tr.cfg.cost_budget:
                    over = True
                    break
            summ = {
                "loss": float(last_loss),
                "total_cost": float(tr.total_cost),
                "over_budget": over,
                "steps_run": ran,
            }
        else:
            summ = tr.train_with_budgets(rb, steps=steps)
        summ["torch"] = tr.torch_available
        if save_ckpt:
            try:
                tr.save_checkpoint(save_ckpt, {"steps": summ.get("steps_run", steps)})
                summ["checkpoint_saved"] = save_ckpt
                try:
                    conn = get_db()
                    init_db(conn)
                    from aios.memory.store import save_artifact
                    save_artifact(conn, kind="training_checkpoint", label="train", data={"path": save_ckpt, "steps": int(summ.get("steps_run", steps))})
                except Exception:
                    pass
                finally:
                    try:
                        conn.close()  # type: ignore[name-defined]
                    except Exception:
                        pass
            except Exception:
                summ["checkpoint_saved"] = False
        print(summ)
        try:
            if dataset_file:
                english_eval(
                    dataset_file=dataset_file,
                    checkpoint=(save_ckpt if (save_ckpt and isinstance(save_ckpt, str)) else None),
                    feature_dim=feature_dim,
                    task=task,
                    instr=instr,
                    max_lines=max(2000, batch_size * 20),
                    label=(tag or "train"),
                )
        except Exception:
            pass
        if emit_metrics:
            try:
                conn = get_db()
                init_db(conn)
                from aios.memory.store import save_artifact
                data_dict = {
                    "steps": int(summ.get("steps_run", steps) or steps),
                    "over_budget": bool(summ.get("over_budget", False)),
                }
                if "last_loss" in summ:
                    try:
                        data_dict["last_loss"] = float(summ["last_loss"])  # type: ignore[arg-type]
                    except Exception:
                        pass
                if english_corpus_metrics:
                    data_dict["english_corpus"] = english_corpus_metrics
                if task:
                    et: dict = {"spec": task}
                    try:
                        if 'english_task_metrics' in locals() and locals().get('english_task_metrics'):
                            et.update(locals().get('english_task_metrics'))  # type: ignore[arg-type]
                    except Exception:
                        pass
                    data_dict["english_task"] = et
                if adherence_metrics:
                    data_dict["english_adherence"] = adherence_metrics
                save_artifact(conn, kind="training_metrics", label=(tag or "train"), data=data_dict)
            finally:
                try:
                    conn.close()  # type: ignore[name-defined]
                except Exception:
                    pass
