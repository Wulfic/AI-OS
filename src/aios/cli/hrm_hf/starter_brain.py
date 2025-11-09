from __future__ import annotations

from typing import Optional
from pathlib import Path

import typer
from rich import print


def save_starter_brain_impl(
    *,
    model: str,
    out_dir: str,
    max_seq_len: int,
    halt_max_steps: int,
    save_model: bool,
    device: str,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target: str,
):
    try:
        from aios.core.hrm_models.hf_adapter import build_hf_adapter
        import json
        import torch
    except Exception as e:
        print({"error": f"Missing deps: {e}"})
        raise typer.Exit(code=1)

    dev = "cuda" if device == "auto" and torch.cuda.is_available() else (device if device != "auto" else "cpu")
    adapter = build_hf_adapter(
        model_name_or_path=model,
        max_seq_len=int(max_seq_len),
        halt_max_steps=int(halt_max_steps),
        device=dev,
        forward_dtype="bfloat16" if torch.cuda.is_available() else "float32",
        use_lora=bool(use_lora),
        lora_r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        lora_target_modules=tuple([t for t in lora_target.split(",") if t]),
    )
    out = adapter.save_brain(out_dir, save_model=bool(save_model))
    cfg = {
        "model": model if save_model else out.get("model_dir", model),
        "q_head": out.get("q_head"),
        "max_seq_len": int(max_seq_len),
        "halt_max_steps": int(halt_max_steps),
        "device": dev,
        "forward_dtype": "bfloat16" if torch.cuda.is_available() else "float32",
        "peft_dir": out.get("peft_dir"),
    }
    cfg_path = Path(out_dir) / "starter_brain.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print({"saved": True, "config": str(cfg_path), "artifacts": out})


def load_starter_brain_impl(*, config: str) -> None:
    try:
        from aios.core.hrm_models.hf_adapter import build_hf_starter_from_config
    except Exception as e:
        print({"error": f"Missing deps: {e}"})
        raise typer.Exit(code=1)
    adapter = build_hf_starter_from_config(config)
    print({
        "loaded": True,
        "device": str(adapter.device),
        "hidden_size": int(adapter.hidden_size),
        "pad_token_id": int(adapter.tokenizer.pad_token_id) if adapter.tokenizer.pad_token_id is not None else None,
    })
