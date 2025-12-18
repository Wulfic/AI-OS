"""Dataset format constants and defaults."""

from __future__ import annotations

# Supported text file extensions
TEXT_EXTS = {".txt", ".csv", ".tsv", ".jsonl", ".json"}

# Supported archive formats
ARCHIVE_EXTS = {
    ".zip", ".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tbz2",
    ".tar.xz", ".txz", ".gz", ".bz2", ".xz", ".rar", ".7z"
}

# Default storage cap in gigabytes
_DATASETS_CAP_GB = 300.0
