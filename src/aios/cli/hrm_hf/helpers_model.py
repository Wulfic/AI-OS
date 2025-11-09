from __future__ import annotations

from typing import Dict, Any, Optional, Callable


def build_student(
    cfg: Dict[str, Any],
    *,
    student_init: Optional[str] = None,
    build_act_v1=None,
    print_fn: Optional[Callable[[dict], None]] = None,
):
    """Build ACT V1 student model, optionally load weights, and print param stats.

    - cfg: dict passed to build_act_v1
    - student_init: optional .pt path to load with strict=False
    - build_act_v1: injectable constructor to avoid import-time cycles
    - print_fn: optional printer (defaults to rich.print)
    """
    if build_act_v1 is None:
        from aios.core.hrm_models import build_act_v1 as _build
        build_act_v1 = _build
    model_student = build_act_v1(cfg)
    if student_init:
        try:
            import torch as _t
            # Use weights_only=False for model checkpoints (we control the source)
            # Suppress the FutureWarning since we trust our own checkpoint files
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
                sd = _t.load(str(student_init), map_location="cpu", weights_only=False)
            missing, unexpected = model_student.load_state_dict(sd, strict=False)
            if print_fn:
                try:
                    print_fn({
                        "loaded_student": True,
                        "path": str(student_init),
                        "missing": list(missing),
                        "unexpected": list(unexpected),
                    })
                except Exception:
                    pass
        except Exception as e:
            if print_fn:
                try:
                    print_fn({"loaded_student": False, "error": str(e), "path": str(student_init)})
                except Exception:
                    pass
    try:
        total_params = sum(p.numel() for p in model_student.parameters())
        trainable_params = sum(p.numel() for p in model_student.parameters() if p.requires_grad)
        if print_fn:
            try:
                print_fn({
                    "params": {"total": int(total_params), "trainable": int(trainable_params)}
                })
            except Exception:
                pass
    except Exception:
        pass
    return model_student


def build_actv1_config(
    *,
    batch_size: int,
    max_seq_len: int,
    vocab_size: int,
    h_cycles: int,
    l_cycles: int,
    h_layers: int,
    l_layers: int,
    hidden_size: int,
    expansion: float,
    num_heads: int,
    pos_encodings: str,
    halt_max_steps: int,
) -> Dict[str, Any]:
    """Return the config dict expected by build_act_v1."""
    return dict(
        batch_size=int(batch_size),
        seq_len=int(max_seq_len),
        num_puzzle_identifiers=4,
        puzzle_emb_ndim=0,
        vocab_size=int(vocab_size),
        H_cycles=int(h_cycles),
        L_cycles=int(l_cycles),
        H_layers=int(h_layers),
        L_layers=int(l_layers),
        hidden_size=int(hidden_size),
        expansion=float(expansion),
        num_heads=int(num_heads),
        pos_encodings=str(pos_encodings),
        halt_max_steps=int(halt_max_steps),
        halt_exploration_prob=0.1,
        forward_dtype="float32",
    )
