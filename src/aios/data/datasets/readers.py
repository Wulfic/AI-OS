"""Basic text file reading utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def read_text_lines_sample(path: str | Path, max_lines: int = 1000) -> List[str]:
    """Read up to max_lines of UTF-8 text lines from a dataset file.

    Designed for quick sampling to seed small replay buffers.
    """
    p = Path(path)
    out: List[str] = []
    
    try:
        # Log file information
        if p.exists():
            file_size_mb = p.stat().st_size / (1024 * 1024)
            logger.info(f"Reading dataset file: {p.name} ({file_size_mb:.2f} MB)")
        else:
            logger.warning(f"Dataset file not found: {p}")
            return []
            
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for i, ln in enumerate(f):
                out.append(ln.strip())
                if (i + 1) >= max_lines:
                    break
        
        # Log read statistics
        avg_line_length = sum(len(line) for line in out) / len(out) if out else 0
        logger.info(
            f"Dataset sample read: {len(out)} lines, "
            f"avg length: {avg_line_length:.1f} chars"
        )
        
    except Exception as e:
        logger.error(f"Failed to read dataset file {p}: {e}")
        return []
    
    return out
