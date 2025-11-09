from __future__ import annotations

import typer
from rich import print


def arch_summary_impl(
    *,
    model: str,
    max_seq_len: int,
    halt_max_steps: int,
    h_layers: int,
    l_layers: int,
    hidden_size: int,
    expansion: float,
    num_heads: int,
    h_cycles: int,
    l_cycles: int,
    pos_encodings: str,
):
    try:
        from transformers import AutoTokenizer
        from aios.core.hrm_models import build_act_v1
        import torch
    except Exception as e:
        print({"error": f"Missing deps: {e}", "hint": "pip install -e .[hf]"})
        raise typer.Exit(code=1)

    tok = AutoTokenizer.from_pretrained(model)
    vocab_size = int(getattr(tok, "vocab_size", 50257) or 50257)
    cfg = dict(
        batch_size=1,
        seq_len=int(max_seq_len),
        num_puzzle_identifiers=4,
        puzzle_emb_ndim=0,
        vocab_size=vocab_size,
        H_cycles=int(h_cycles),
        L_cycles=int(l_cycles),
        H_layers=int(h_layers),
        L_layers=int(l_layers),
        hidden_size=int(hidden_size),
        expansion=float(expansion),
        num_heads=int(num_heads),
        pos_encodings=str(pos_encodings),
        halt_max_steps=int(halt_max_steps),
        halt_exploration_prob=0.0,
        forward_dtype="float32",
    )
    m = build_act_v1(cfg)
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    embed_params = 0
    head_params = 0
    for n, p in m.named_parameters():
        if ".embed_tokens" in n:
            embed_params += p.numel()
        elif ".lm_head" in n:
            head_params += p.numel()
    print({
        "arch": {
            "H_layers": int(h_layers), "L_layers": int(l_layers), "hidden_size": int(hidden_size),
            "num_heads": int(num_heads), "expansion": float(expansion), "H_cycles": int(h_cycles), "L_cycles": int(l_cycles),
            "pos_encodings": str(pos_encodings), "seq_len": int(max_seq_len), "halt_max_steps": int(halt_max_steps),
            "vocab_size": vocab_size,
        },
        "params": {
            "total": int(total),
            "trainable": int(trainable),
            "embeddings": int(embed_params),
            "lm_head": int(head_params),
        }
    })
