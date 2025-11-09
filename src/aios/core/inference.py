from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

from .train import Trainer, TrainConfig
from .watch import (
    default_ckpt_dir,
    latest_checkpoint_from_db,
    latest_checkpoint_from_fs,
    detect_repo_root,
)


@dataclass
class InferenceResult:
    ok: bool
    text: str
    score: Optional[float] = None
    checkpoint: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


def _active_checkpoint_path() -> Optional[Path]:
    d = default_ckpt_dir()
    p = d / "active.npz"
    return p if p.exists() else None


def pick_checkpoint(prefer_active: bool = True) -> Optional[Path]:
    if prefer_active:
        p = _active_checkpoint_path()
        if p is not None:
            return p
    # Prefer DB, then filesystem
    ck = latest_checkpoint_from_db() or latest_checkpoint_from_fs()
    if ck is not None:
        return ck
    # Fallback: repo artifacts/checkpoints if running from source
    try:
        root = detect_repo_root()
        if root is not None:
            repo_ckpts = sorted((root / "artifacts" / "checkpoints").glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
            return repo_ckpts[0] if repo_ckpts else None
    except Exception:
        pass
    return None


def _featurize(text: str, dim: int) -> np.ndarray:
    """Cheap hash-based featurizer to fixed dim.

    Maps UTF-8 bytes to buckets in [0, dim), accumulates normalized values.
    Returns a (dim,) float32 vector.
    """
    v = np.zeros((dim,), dtype=np.float32)
    if not text:
        return v
    try:
        bs = text.encode("utf-8", errors="ignore")
    except Exception:
        bs = bytes()
    if len(bs) == 0:
        return v
    for b in bs:
        i = int(b) % dim
        v[i] += float(b) / 255.0
    # simple l2 normalize to keep scale stable across message lengths
    n = float(np.linalg.norm(v))
    if n > 0:
        v /= n
    return v


def run_inference(message: str, ckpt_path: Optional[str] = None) -> InferenceResult:
    """Run a lightweight inference using the latest numpy MLP checkpoint.

    This is not a generative LM; it produces a scalar score that we summarize as text.
    Uses numpy-only path for portability; if torch weights exist, they are ignored here.
    """
    # Resolve checkpoint
    ckpt = Path(ckpt_path) if ckpt_path else pick_checkpoint(prefer_active=True)
    if ckpt is None or not ckpt.exists():
        return InferenceResult(ok=False, text="No checkpoint found. Train the model first.")

    # Build a minimal trainer and load weights (numpy path)
    tr = Trainer(TrainConfig(use_torch=False))
    if not tr.load_checkpoint(str(ckpt)):
        return InferenceResult(ok=False, text=f"Failed to load checkpoint: {ckpt}")

    # Infer input dim from loaded weights to keep featurizer in sync
    try:
        input_dim = int(tr.model_np.W1.shape[0])
    except Exception:
        input_dim = int(tr.cfg.input_dim)
    x = _featurize(message, input_dim).reshape(1, input_dim)

    # Forward pass (numpy)
    try:
        y, _ = tr.model_np.forward(x)
        score = float(np.asarray(y).reshape(-1)[0])
    except Exception:
        return InferenceResult(ok=False, text="Inference error during forward pass.")

    # Map score to a concise, more contextual reply (still non-generative).
    def _friendly_reply(msg: str, s: float) -> str:
        try:
            if np.isnan(s) or np.isinf(s):
                return "I’m ready. Ask a question or give me a task."
        except Exception:
            return "I’m ready. Ask a question or give me a task."
        m = (msg or "").strip()
        low = m.lower()
        is_question = low.endswith("?") or low.startswith((
            "do ", "did ", "are ", "is ", "can ", "could ", "would ", "will ", "should ", "what ", "how ", "why ", "where ", "when ",
        ))
        # lightweight intent routing for common asks
        if any(k in low for k in ("train", "checkpoint", "learn")):
            return "To train, open Control → Train or run /train. You can also provide a dataset in Resources → Datasets."
        if any(k in low for k in ("status", "state", "health")):
            return "System status: use /status for a concise summary."
        if "crawl" in low or "website" in low or "url" in low:
            return "To crawl a site, run: /crawl https://example.com — it will respect robots.txt unless you disable it in Resources."
        if any(k in low for k in ("goal", "directive")):
            return "You can manage goals with /goals-list and add new ones with /goals-add Your objective here."
        if any(k in low for k in ("hello", "hi ", "hey", "greetings")):
            return "Hi! I’m here and listening. What would you like to do?"

        # tone based on score
        if s > 0.25:
            tone = "Sure"
        elif s < -0.25:
            tone = "Got it"
        else:
            tone = "Okay"

        if is_question:
            # echo the question topic briefly
            topic = (m.rstrip("?"))
            if len(topic) > 80:
                topic = topic[:80] + "…"
            return f"{tone}. For your question: ‘{topic}’ — I can check system status (/status), train (/train), crawl (/crawl URL), or manage goals (/goals-list)."

        # statement: reflect a compact acknowledgement
        preview = (m[:80] + ("…" if len(m) > 80 else "")) if m else ""
        if preview:
            return f"{tone}. Noted: ‘{preview}’. You can say /status, /train, or /crawl URL to proceed."
        return "I’m listening. Say /status, /train, or /crawl URL to begin."

    txt = _friendly_reply(message, score)

    return InferenceResult(ok=True, text=txt, score=score, checkpoint=str(ckpt))
