"""Checkpoint saving and loading methods for Trainer."""

from __future__ import annotations

from typing import Optional
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CheckpointMixin:
    """Mixin providing checkpoint save/load functionality for Trainer."""

    def save_checkpoint(self, path: str, extra: Optional[dict] = None) -> None:
        """Save a lightweight checkpoint.

        Primary artifact is a .npz containing numpy MLP weights and metadata.
        If torch backend is active, also attempts to save a sibling .pt file with state_dict (best-effort).
        """
        logger.info(f"Starting checkpoint save to {path}")
        meta = {
            "input_dim": int(self.cfg.input_dim),
            "hidden": int(self.cfg.hidden),
            "output_dim": int(self.cfg.output_dim),
            "use_torch": bool(self.cfg.use_torch),
        }
        if extra:
            try:
                # ensure JSON-safe by stringifying unknown types later if needed
                meta.update({str(k): v for k, v in extra.items()})
            except Exception as e:
                logger.warning(f"Failed to add extra metadata to checkpoint: {e}")
        
        try:
            np.savez_compressed(
                path,
                W1=self.model_np.W1,
                b1=self.model_np.b1,
                W2=self.model_np.W2,
                b2=self.model_np.b2,
                meta=str(meta),
            )
            file_size = os.path.getsize(path) if os.path.exists(path) else 0
            logger.info(f"NPZ checkpoint saved successfully ({file_size} bytes)")
        except Exception as e:
            logger.error(f"Failed to save NPZ checkpoint to {path}: {e}")
            raise
        
        # Optional torch state (best-effort)
        if self.torch_available:  # pragma: no cover - optional
            try:
                t = self.torch  # type: ignore[attr-defined]
                torch_path = path + ".pt"
                base = self._torch_base()
                state = base.state_dict() if base is not None else self.mlp_t.state_dict()
                t.save(state, torch_path)  # type: ignore[attr-defined]
                logger.info(f"Optional .pt file saved: {torch_path}")
            except Exception as e:
                logger.debug(f"Could not save optional .pt file: {e}")

    def load_checkpoint(self, path: str) -> bool:
        """Load a checkpoint saved by save_checkpoint. Returns True on success."""
        logger.info(f"Starting checkpoint load from {path}")
        
        if not os.path.exists(path):
            logger.warning(f"Checkpoint file not found: {path}")
            return False
        
        try:
            with np.load(path, allow_pickle=True) as data:  # type: ignore[no-untyped-call]
                self.model_np.W1 = data["W1"].astype(np.float32)
                self.model_np.b1 = data["b1"].astype(np.float32)
                self.model_np.W2 = data["W2"].astype(np.float32)
                self.model_np.b2 = data["b2"].astype(np.float32)
            logger.info(f"Checkpoint loaded successfully from {path}")
            
            # Optional torch restore if available and sibling .pt exists; do not fail if missing
            if self.torch_available:  # pragma: no cover - optional
                try:
                    t = self.torch  # type: ignore[attr-defined]
                    torch_path = path + ".pt"
                    if os.path.exists(torch_path):
                        state = t.load(torch_path, map_location=self.device)  # type: ignore[attr-defined]
                        base = self._torch_base() or self.mlp_t
                        base.load_state_dict(state)  # type: ignore[attr-defined]
                        logger.info(f"Optional .pt file restored: {torch_path}")
                except Exception as e:
                    logger.debug(f"Could not restore optional .pt file: {e}")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {path}: {e}")
            return False


def average_checkpoints_npz(paths: list[str], out_path: str) -> bool:
    """Average numpy MLP weights across checkpoints and save to out_path (.npz).

    Only averages numpy weights (W1, b1, W2, b2) for identical shapes. Metadata is copied from the first file.
    Returns True on success.
    """
    if not paths:
        logger.warning("No checkpoint paths provided for averaging")
        return False
    
    logger.info(f"Starting checkpoint averaging: {len(paths)} checkpoints -> {out_path}")
    
    try:
        w1s: list[np.ndarray] = []
        b1s: list[np.ndarray] = []
        w2s: list[np.ndarray] = []
        b2s: list[np.ndarray] = []
        meta = {}
        
        for i, p in enumerate(paths):
            logger.debug(f"Processing checkpoint {i+1}/{len(paths)}: {p}")
            with np.load(p, allow_pickle=True) as d:  # type: ignore[no-untyped-call]
                if i == 0:
                    meta = {"meta": d.get("meta", "{}")}
                w1s.append(d["W1"])  # type: ignore[index]
                b1s.append(d["b1"])  # type: ignore[index]
                w2s.append(d["W2"])  # type: ignore[index]
                b2s.append(d["b2"])  # type: ignore[index]
        
        # ensure identical shapes
        if len({tuple(a.shape) for a in w1s}) != 1 or len({tuple(a.shape) for a in b1s}) != 1 or len({tuple(a.shape) for a in w2s}) != 1 or len({tuple(a.shape) for a in b2s}) != 1:
            logger.warning("Shape mismatch between checkpoints - cannot average")
            return False
        
        W1 = sum(w1s) / float(len(w1s))
        b1 = sum(b1s) / float(len(b1s))
        W2 = sum(w2s) / float(len(w2s))
        b2 = sum(b2s) / float(len(b2s))
        np.savez_compressed(out_path, W1=W1, b1=b1, W2=W2, b2=b2, **meta)
        logger.info(f"Checkpoint averaging complete: {out_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to average checkpoints: {e}")
        return False
