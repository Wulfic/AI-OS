from __future__ import annotations

from typing import List, Tuple, Any, Optional


def encode_lines(tok, lines: List[str], max_seq_len: int) -> Tuple[Any, Any]:
    """Tokenize lines to (input_ids, labels) applying ignore_index for padding."""
    enc = tok(
        lines,
        padding="max_length",
        truncation=True,
        max_length=int(max_seq_len),
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    labels = input_ids.clone()
    try:
        if tok.pad_token_id is not None:
            labels[enc["attention_mask"] == 0] = -100
    except Exception:
        pass
    return input_ids, labels

def adjust_tokenizer_padding(tok):
    """Ensure tokenizer has left padding and a pad token; prefer eos as pad if missing."""
    try:
        tok.padding_side = "left"  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
            tok.pad_token = tok.eos_token  # type: ignore[assignment]
    except Exception:
        pass
    return tok

def encode_eval_lines(
    tok,
    eval_file: Optional[str],
    ascii_only: bool,
    max_seq_len: int,
    read_text_lines_sample_any,
):
    """Read and encode eval lines if provided; returns (ids, labels) or (None, None)."""
    if not eval_file:
        return None, None
    try:
        eval_lines = read_text_lines_sample_any(eval_file, max_lines=4000)
        eval_lines = [ln for ln in eval_lines if ln and str(ln).strip()]
        if ascii_only:
            eval_lines = [ln for ln in eval_lines if ln.encode("ascii", errors="ignore").decode("ascii") == ln]
        if not eval_lines:
            return None, None
        enc = tok(
            eval_lines,
            padding="max_length",
            truncation=True,
            max_length=int(max_seq_len),
            return_tensors="pt",
        )
        ids = enc["input_ids"]
        labels = ids.clone()
        if getattr(tok, "pad_token_id", None) is not None:
            labels[enc["attention_mask"] == 0] = -100
        return ids, labels
    except Exception:
        return None, None


def load_teacher_kl_model(
    *,
    teacher_name: str | None,
    teacher_device: str,
    strict: bool,
    gradient_checkpointing: bool,
    AutoModelForCausalLM,
    torch,
):
    """Load teacher model for KL regularization; returns model or None.

    Handles CUDA/DML/CPU placement with strict behavior.
    Supports gradient checkpointing to reduce VRAM usage.
    """
    if not teacher_name:
        return None
    try:
        try:
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher_name, use_safetensors=True)
        except Exception:
            try:
                teacher_model = AutoModelForCausalLM.from_pretrained(teacher_name)
            except Exception:
                teacher_model = AutoModelForCausalLM.from_pretrained(teacher_name, from_tf=True)
        if str(teacher_device).lower() == "auto":
            t_dev_obj = torch.device("cpu")
        elif str(teacher_device).lower() == "cuda" and torch.cuda.is_available():
            t_dev_obj = torch.device("cuda")
        elif str(teacher_device).lower() == "dml":
            try:
                import torch_directml as _dml  # type: ignore
                t_dev_obj = _dml.device()
            except Exception:
                t_dev_obj = torch.device("cpu")
        else:
            t_dev_obj = torch.device("cpu")
        try:
            teacher_model.to(t_dev_obj)  # type: ignore[misc]
        except Exception as _t_move_e:
            if strict and str(teacher_device).lower() == "cuda":
                from rich import print
                import typer
                print({"error": f"Failed moving teacher KL model to CUDA in strict mode: {_t_move_e}"})
                raise typer.Exit(code=7)
            else:
                raise
        
        # Enable gradient checkpointing to reduce VRAM usage
        if gradient_checkpointing:
            try:
                if hasattr(teacher_model, 'gradient_checkpointing_enable'):
                    teacher_model.gradient_checkpointing_enable()
                    from rich import print as _p
                    try:
                        _p({"teacher_gradient_checkpointing": True, "vram_savings": "~30-50%"})
                    except Exception:
                        pass
            except Exception:
                pass
        
        teacher_model.eval()
        return teacher_model
    except Exception:
        return None
