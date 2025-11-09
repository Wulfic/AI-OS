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
    dataset_chunk_size: int = 4000,
):
    """Read and encode eval lines if provided; returns (ids, labels) or (None, None).
    
    Args:
        tok: Tokenizer
        eval_file: Path to eval file
        ascii_only: Filter to ASCII-only lines
        max_seq_len: Maximum sequence length
        read_text_lines_sample_any: Function to read lines
        dataset_chunk_size: Number of samples to load
    """
    if not eval_file:
        return None, None
    try:
        eval_lines = read_text_lines_sample_any(eval_file, max_lines=dataset_chunk_size)
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
