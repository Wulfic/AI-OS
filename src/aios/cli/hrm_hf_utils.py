from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _find_project_root() -> Path | None:
    """Find project root by searching for pyproject.toml.
    
    Searches from module location and current working directory.
    Returns None if not found.
    """
    # First, try from this module's location
    try:
        source_dir = Path(__file__).resolve().parent
        cur = source_dir
        for _ in range(10):  # src/aios/cli -> ~3-4 levels up + buffer
            if (cur / "pyproject.toml").exists():
                return cur
            parent = cur.parent
            if parent == cur:
                break
            cur = parent
    except Exception:
        pass
    
    # Fallback: try from CWD
    try:
        cur = Path.cwd().resolve()
        for _ in range(8):
            if (cur / "pyproject.toml").exists():
                return cur
            parent = cur.parent
            if parent == cur:
                break
            cur = parent
    except Exception:
        pass
    
    return None


def _resolve_tokenizer_path(model_or_path: str) -> str:
    """Resolve a tokenizer path, handling relative paths.
    
    If the path looks like a local relative path (contains path separators
    or starts with 'artifacts'), try to resolve it relative to the project root.
    """
    # Check if it's already an absolute path that exists
    try:
        p = Path(model_or_path)
        if p.is_absolute() and p.is_dir():
            return str(p)
    except Exception:
        pass
    
    # Check if it looks like a local relative path (not a HuggingFace repo ID)
    # HuggingFace repo IDs are like "username/model" or just "model"
    # Local paths typically have backslashes on Windows or start with "artifacts/"
    is_likely_local = (
        os.sep in model_or_path or 
        "/" in model_or_path and model_or_path.count("/") > 1 or  # More than one slash suggests local path
        model_or_path.startswith("artifacts") or
        model_or_path.startswith(".")
    )
    
    if is_likely_local:
        # Try to resolve relative to project root
        project_root = _find_project_root()
        if project_root:
            resolved = project_root / model_or_path
            if resolved.is_dir():
                logger.info(f"Resolved relative tokenizer path: {model_or_path} -> {resolved}")
                return str(resolved)
        
        # Try relative to CWD as well
        try:
            cwd_resolved = Path.cwd() / model_or_path
            if cwd_resolved.is_dir():
                logger.info(f"Resolved tokenizer path from CWD: {model_or_path} -> {cwd_resolved}")
                return str(cwd_resolved)
        except Exception:
            pass
    
    # Return as-is if we couldn't resolve it
    return model_or_path


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

    # First, try to resolve relative paths to absolute paths
    resolved_path = _resolve_tokenizer_path(model_or_path)
    if resolved_path != model_or_path:
        logger.info(f"Resolved tokenizer path: {model_or_path} -> {resolved_path}")
        model_or_path = resolved_path

    candidates: list[Path] = []
    try:
        p = Path(model_or_path)
        if p.is_dir():
            candidates.append(p)
    except Exception:
        pass

    # Known local bundled tokenizer location (when running from repo)
    # Only try this if model_or_path looks like a simple name (not a path)
    if "/" not in model_or_path and "\\" not in model_or_path:
        bundled = Path("artifacts") / "hf_implant" / model_or_path
        if bundled.is_dir():
            candidates.append(bundled)
        
        # Also try with project root for bundled tokenizers
        project_root = _find_project_root()
        if project_root:
            bundled_abs = project_root / "artifacts" / "hf_implant" / model_or_path
            if bundled_abs.is_dir() and bundled_abs not in candidates:
                candidates.append(bundled_abs)

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
                logger.info(f"Loading tokenizer from local cache: {c}")
                return _post_config(AutoTokenizer.from_pretrained(str(c), use_fast=True, local_files_only=True))
            except Exception as e:
                logger.debug(f"Failed to load local tokenizer from {c}: {e}")
                pass

    logger.info(f"Downloading tokenizer from HuggingFace: {model_or_path}")
    try:
        tok = _post_config(AutoTokenizer.from_pretrained(model_or_path, use_fast=True))
        logger.info(f"Successfully loaded tokenizer for {model_or_path}")
        return tok
    except Exception as e:
        logger.warning(f"Failed to download fast tokenizer for {model_or_path}: {e}, trying slow tokenizer")
        for c in candidates:
            try:
                logger.debug(f"Trying slow tokenizer from {c}")
                return _post_config(AutoTokenizer.from_pretrained(str(c), use_fast=False, local_files_only=True))
            except Exception:
                pass
        logger.error(f"All tokenizer loading attempts failed for {model_or_path}")
        raise
