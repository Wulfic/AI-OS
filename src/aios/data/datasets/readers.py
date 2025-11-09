"""Basic text file reading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List


def read_text_lines_sample(path: str | Path, max_lines: int = 1000) -> List[str]:
    """Read up to max_lines of UTF-8 text lines from a dataset file.

    Designed for quick sampling to seed small replay buffers.
    """
    p = Path(path)
    out: List[str] = []
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for i, ln in enumerate(f):
                out.append(ln.strip())
                if (i + 1) >= max_lines:
                    break
    except Exception:
        return []
    return out
