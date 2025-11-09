"""Archive format reading utilities for datasets."""

from __future__ import annotations

from pathlib import Path
from typing import List
import io
import zipfile
import tarfile
import gzip

from .constants import TEXT_EXTS, ARCHIVE_EXTS


def _is_archive(path: Path) -> bool:
    """Check if a path points to an archive file."""
    name = path.name.lower()
    for ext in ARCHIVE_EXTS:
        if name.endswith(ext):
            return True
    return False


def _iter_text_from_zip(zp: zipfile.ZipFile, max_lines: int) -> List[str]:
    """Extract text lines from files inside a ZIP archive."""
    out: List[str] = []
    for info in zp.infolist():
        if info.is_dir():
            continue
        suf = "." + info.filename.split(".")[-1].lower() if "." in info.filename else ""
        if suf not in TEXT_EXTS:
            continue
        try:
            with zp.open(info, mode="r") as f:
                for b in io.TextIOWrapper(f, encoding="utf-8", errors="ignore"):
                    out.append(b.strip())
                    if len(out) >= max_lines:
                        return out
        except Exception:
            continue
    return out


def _iter_text_from_tar(tf: tarfile.TarFile, max_lines: int) -> List[str]:
    """Extract text lines from files inside a TAR archive."""
    out: List[str] = []
    for m in tf.getmembers():
        if not m.isfile():
            continue
        name = m.name.lower()
        suf = "." + name.split(".")[-1] if "." in name else ""
        if suf not in TEXT_EXTS:
            continue
        try:
            f = tf.extractfile(m)
            if not f:
                continue
            with io.TextIOWrapper(f, encoding="utf-8", errors="ignore") as r:
                for line in r:
                    out.append(line.strip())
                    if len(out) >= max_lines:
                        return out
        except Exception:
            continue
    return out


def read_archive_text_lines(path: Path, max_lines: int = 1000) -> List[str]:
    """Read text lines from various archive formats.
    
    Supports: ZIP, TAR (including .tar.gz, .tar.bz2, .tar.xz), GZIP, RAR, 7Z
    """
    if not path.exists():
        return []
    
    name = path.name.lower()
    
    # ZIP files
    if name.endswith(".zip"):
        try:
            with zipfile.ZipFile(path, mode="r") as zf:
                return _iter_text_from_zip(zf, max_lines)
        except Exception:
            return []
    
    # TAR variants
    if name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")):
        try:
            mode = "r"
            if name.endswith((".tar.gz", ".tgz")):
                mode = "r:gz"
            elif name.endswith((".tar.bz2", ".tbz2")):
                mode = "r:bz2"
            elif name.endswith((".tar.xz", ".txz")):
                mode = "r:xz"
            with tarfile.open(path, mode) as tf:
                return _iter_text_from_tar(tf, max_lines)
        except Exception:
            return []
    
    # Single-file GZIP
    if name.endswith(".gz") and not name.endswith((".tar.gz", ".tgz")):
        try:
            with gzip.open(path, mode="rt", encoding="utf-8", errors="ignore") as f:
                out: List[str] = []
                for line in f:
                    out.append(line.strip())
                    if len(out) >= max_lines:
                        break
                return out
        except Exception:
            return []
    
    # RAR files (requires rarfile library)
    if name.endswith(".rar"):
        try:
            import rarfile  # type: ignore
            with rarfile.RarFile(path) as rf:  # type: ignore
                out: List[str] = []
                for info in rf.infolist():  # type: ignore[attr-defined]
                    if getattr(info, "is_dir", lambda: False)():
                        continue
                    fn = str(getattr(info, "filename", "")).lower()
                    suf = "." + fn.split(".")[-1] if "." in fn else ""
                    if suf not in TEXT_EXTS:
                        continue
                    with rf.open(info) as f:  # type: ignore[attr-defined]
                        for line in io.TextIOWrapper(f, encoding="utf-8", errors="ignore"):
                            out.append(line.strip())
                            if len(out) >= max_lines:
                                return out
                return out
        except Exception:
            return []
    
    # 7Z files (requires py7zr library)
    if name.endswith(".7z"):
        try:
            import py7zr  # type: ignore
            out: List[str] = []
            with py7zr.SevenZipFile(path, mode="r") as z:  # type: ignore
                for n, bio in z.readall().items():  # type: ignore[attr-defined]
                    fn = str(n).lower()
                    suf = "." + fn.split(".")[-1] if "." in fn else ""
                    if suf not in TEXT_EXTS:
                        continue
                    with io.TextIOWrapper(bio, encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            out.append(line.strip())
                            if len(out) >= max_lines:
                                return out
            return out
        except Exception:
            return []
    
    return []
