from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print

from aios.data.datasets import read_text_lines_sample_any
from aios.core.hrm_models.hf_adapter import build_hf_adapter


def implant_brain_impl(
    *,
    model: str,
    dataset_file: str,
    max_seq_len: int,
    batch_size: int,
    steps: int,
    lr: float,
    device: str,
    save_dir: Optional[str],
    train_lm: bool,
    halt_max_steps: int,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target: str,
):
    try:
        import torch
        from transformers import AutoTokenizer
        from aios.core.hrm_models.train_utils import segment_rollout  # type: ignore
    except Exception as e:
        print({"started": False, "error": f"Missing deps: {e}", "hint": "pip install -e .[hf]"})
        raise typer.Exit(code=1)

    dev = device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    lines = read_text_lines_sample_any(dataset_file, max_lines=max(2000, batch_size * 50))
    lines = [ln for ln in lines if ln and ln.strip()]
    if not lines:
        print({"started": False, "error": "no lines"})
        raise typer.Exit(code=1)

    tokenizer = AutoTokenizer.from_pretrained(model)
    try:
        tokenizer.padding_side = "left"  # type: ignore[attr-defined]
    except Exception:
        pass
    if tokenizer.pad_token_id is None and getattr(tokenizer, "eos_token_id", None) is not None:
        try:
            tokenizer.pad_token = tokenizer.eos_token  # type: ignore[assignment]
        except Exception:
            pass

    enc = tokenizer(
        lines,
        padding="max_length",
        truncation=True,
        max_length=int(max_seq_len),
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    labels = input_ids.clone()
    if tokenizer.pad_token_id is not None:
        labels[enc["attention_mask"] == 0] = -100

    adapter = build_hf_adapter(
        model_name_or_path=model,
        max_seq_len=int(max_seq_len),
        halt_max_steps=int(halt_max_steps),
        halt_exploration_prob=0.0,
        device=dev,
        forward_dtype="bfloat16" if torch.cuda.is_available() else "float32",
        use_lora=bool(use_lora),
        lora_r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        lora_target_modules=tuple([t for t in lora_target.split(",") if t]),
    )
    adapter.train(True)

    params = list(adapter.q_head.parameters())
    if train_lm:
        params += [p for p in adapter.model.parameters() if p.requires_grad]
    OptClass = getattr(torch.optim, "AdamW", None) or getattr(torch.optim, "Adam")
    opt = OptClass(params, lr=float(lr))

    N = input_ids.shape[0]
    step = 0
    last_loss = 0.0
    adapter.model.train(mode=bool(train_lm))
    for _ in range(max(1, int(steps))):
        idx = torch.randint(0, N, (batch_size,))
        inp = input_ids.index_select(0, idx).to(adapter.device)
        tgt = labels.index_select(0, idx).to(adapter.device)
        batch = {
            "inputs": inp,
            "targets": tgt,
            "puzzle_identifiers": torch.zeros((inp.shape[0],), dtype=torch.int64, device=adapter.device),
        }
        opt.zero_grad(set_to_none=True)
        loss, metrics = segment_rollout(
            model=adapter,
            batch=batch,
            max_segments=int(halt_max_steps),
            epsilon=0.0,
            ignore_index=-100,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        last_loss = float(loss.detach().cpu().item())
        step += 1
        if step % 10 == 0:
            try:
                print({"step": step, "loss": last_loss, "ce": float(metrics["ce"].cpu()), "bce_h": float(metrics["bce_halt"].cpu()), "bce_c": float(metrics["bce_continue"].cpu())})
            except Exception:
                print({"step": step, "loss": last_loss})

    out_dir = Path(save_dir) if save_dir else None
    saved = {}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            import torch as _t
            _t.save(adapter.q_head.state_dict(), str(out_dir / "q_head.pt"))
            saved["q_head"] = str(out_dir / "q_head.pt")
        except Exception:
            pass
        if train_lm:
            try:
                adapter.model.save_pretrained(str(out_dir))
                saved["model"] = str(out_dir)
            except Exception:
                pass

    print({"implanted": True, "model": model, "train_lm": bool(train_lm), "steps": step, "loss": last_loss, "saved": saved})
