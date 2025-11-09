from __future__ import annotations

from pathlib import Path


def load_tokenizer(model_or_path: str):
    """Load a tokenizer robustly.

    Preference order:
    1) If a local directory is provided/available (e.g., artifacts/hf_implant/<model>), load fast tokenizer locally
    2) Try to load remote fast tokenizer (use_fast=True)
    3) Fallback to any available local slow tokenizer only if necessary

    This avoids accidental fallback to slow tokenizers which may require protobuf and fail when files are not resolved.
    """
    try:
        from transformers import AutoTokenizer  # lazy import
    except Exception as e:  # pragma: no cover - heavy dep
        raise RuntimeError(f"transformers not available: {e}")

    candidates: list[Path] = []
    try:
        p = Path(model_or_path)
        if p.is_dir():
            candidates.append(p)
    except Exception:
        pass

    # Known local bundled tokenizer location (when running from repo)
    bundled = Path("artifacts") / "hf_implant" / model_or_path
    if bundled.is_dir():
        candidates.append(bundled)

    def _post_config(tok):
        """Standardize tokenizer settings for decoder-only models.

        - Prefer left padding to avoid right-padding warnings in decoder-only generation
        - If pad_token is missing, map it to eos_token (common decoder-only model convention)
        """
        try:
            # Set left padding for decoder-only use
            if getattr(tok, "padding_side", None) != "left":
                tok.padding_side = "left"
        except Exception:
            pass
        try:
            if getattr(tok, "pad_token_id", None) is None:
                eos_tok = getattr(tok, "eos_token", None)
                if eos_tok is not None:
                    tok.pad_token = eos_tok
        except Exception:
            pass
        return tok

    for c in candidates:
        if (c / "tokenizer.json").exists() or (
            (c / "vocab.json").exists() and (c / "merges.txt").exists()
        ):
            try:
                return _post_config(AutoTokenizer.from_pretrained(str(c), use_fast=True, local_files_only=True))
            except Exception:
                pass

    try:
        return _post_config(AutoTokenizer.from_pretrained(model_or_path, use_fast=True))
    except Exception:
        for c in candidates:
            try:
                return _post_config(AutoTokenizer.from_pretrained(str(c), use_fast=False, local_files_only=True))
            except Exception:
                pass
        raise
