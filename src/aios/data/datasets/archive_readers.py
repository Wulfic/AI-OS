"""Archive format reading utilities for datasets."""

from __future__ import annotations

from pathlib import Path
from typing import List
import io
import zipfile
import tarfile
import gzip

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .constants import TEXT_EXTS, ARCHIVE_EXTS


def _is_archive(path: Path) -> bool:
    """Check if a path points to an archive file using extension and content sniffing."""
    name = path.name.lower()
    
    # First check: file extension
    for ext in ARCHIVE_EXTS:
        if name.endswith(ext):
            return True
    
    # Second check: content-based detection (magic bytes)
    if not path.exists() or not path.is_file():
        return False
    
    try:
        with open(path, 'rb') as f:
            header = f.read(16)  # Read first 16 bytes
            
            # ZIP: PK\x03\x04 or PK\x05\x06 (empty archive)
            if header[:4] in (b'PK\x03\x04', b'PK\x05\x06'):
                return True
            
            # GZIP: \x1f\x8b
            if header[:2] == b'\x1f\x8b':
                return True
            
            # TAR: ustar at offset 257 (we can't check this with just 16 bytes)
            # But tar files often have specific patterns in first 512 bytes
            # Skip for now to avoid reading too much
            
            # RAR: Rar! or Rar!\x1a\x07
            if header[:4] == b'Rar!':
                return True
            
            # 7Z: 7z\xbc\xaf\x27\x1c
            if header[:6] == b'7z\xbc\xaf\x27\x1c':
                return True
            
            # BZ2: BZ
            if header[:2] == b'BZ':
                return True
            
            # XZ: \xfd7zXZ\x00
            if header[:6] == b'\xfd7zXZ\x00':
                return True
            
    except Exception:
        pass  # If we can't read, fall back to extension only
    
    return False


def _iter_text_from_zip(zp: zipfile.ZipFile, max_lines: int) -> List[str]:
    """Extract text lines from files inside a ZIP archive."""
    out: List[str] = []
    # Filter text files for progress bar
    text_files = [info for info in zp.infolist() if not info.is_dir() and 
                  ("." + info.filename.split(".")[-1].lower() if "." in info.filename else "") in TEXT_EXTS]
    
    # Show progress if there are many files
    disable_pbar = not TQDM_AVAILABLE or len(text_files) < 5
    
    for info in (tqdm(text_files, desc="Extracting from ZIP", unit="files", disable=disable_pbar) if TQDM_AVAILABLE else text_files):
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
    # Filter text files for progress bar
    text_members = [m for m in tf.getmembers() if m.isfile() and 
                    ("." + m.name.lower().split(".")[-1] if "." in m.name else "") in TEXT_EXTS]
    
    # Show progress if there are many files
    disable_pbar = not TQDM_AVAILABLE or len(text_members) < 5
    
    for m in (tqdm(text_members, desc="Extracting from TAR", unit="files", disable=disable_pbar) if TQDM_AVAILABLE else text_members):
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
