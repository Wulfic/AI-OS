"""CSV file reading with text and label extraction."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import csv


def read_csv_text_label_samples(
    path: str | Path,
    *,
    text_col: str | int,
    label_col: str | int,
    max_rows: int = 5000,
    label_map: Optional[dict[str, float]] = None,
) -> List[tuple[str, float]]:
    """Read text and label columns from a CSV file.

    - text_col/label_col can be header names or 0-based indices.
    - label_map maps raw label strings to numeric values (floats).
    - If label_map not provided and labels are numeric, cast to float; otherwise hash to a small integer bucket.
    """
    p = Path(path)
    out: List[tuple[str, float]] = []
    if not p.exists():
        return out
    try:
        with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            sniffer = csv.Sniffer()
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = sniffer.sniff(sample)
            except Exception:
                dialect = csv.excel
            reader = csv.reader(f, dialect)
            # Peek header
            header: Optional[list[str]] = None
            try:
                pos = f.tell()
                f.seek(0)
                header = next(reader)
            except Exception:
                header = None
            finally:
                try:
                    f.seek(0)
                    reader = csv.reader(f, dialect)
                    if header:
                        next(reader, None)
                except Exception:
                    pass

            # Resolve column indices
            def _resolve(col: str | int) -> int:
                if isinstance(col, int):
                    return int(col)
                if header is None:
                    return int(col) if str(col).isdigit() else 0
                try:
                    return int(header.index(str(col)))
                except Exception:
                    # fallback to first/second
                    return 0

            ti = _resolve(text_col)
            li = _resolve(label_col)
            for i, row in enumerate(reader):
                if not row or max(ti, li) >= len(row):
                    continue
                txt = str(row[ti]).strip()
                raw_lbl = str(row[li]).strip()
                if not txt:
                    continue
                y: float
                if label_map and raw_lbl in label_map:
                    y = float(label_map[raw_lbl])
                else:
                    try:
                        y = float(raw_lbl)
                    except Exception:
                        # Map to a small numeric via hash and center roughly around 0..N-1
                        y = float((hash(raw_lbl) % 3) - 1)  # -1,0,1 buckets
                out.append((txt, y))
                if len(out) >= max_rows:
                    break
    except Exception:
        return []
    return out
