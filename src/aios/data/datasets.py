from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Iterator
import os
import io
import zipfile
import tarfile
import gzip
import csv


TEXT_EXTS = {".txt", ".csv", ".tsv", ".jsonl"}
ARCHIVE_EXTS = {".zip", ".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".gz", ".bz2", ".xz", ".rar", ".7z"}

_DATASETS_CAP_GB = 300.0

def _cap_config_path() -> Path:
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    base = Path(home) if home else Path.home()
    p = base / ".config" / "aios" / "datasets.json"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p


def datasets_base_dir() -> Path:
    """Return the base directory for storing datasets.

    Resolution order (first match wins):
    1) Environment override via `AIOS_DATASETS_DIR`
    2) Project root detected from CWD (pyproject.toml/.git) → `training_data/curated_datasets`
    3) Fallback: `~/.local/share/aios/datasets`
    """
    # 1) Explicit override
    override = os.environ.get("AIOS_DATASETS_DIR")
    if override:
        p = Path(override).expanduser().resolve()
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return p

    # 2) Try to detect project root from CWD and place under training_data/curated_datasets
    def _find_project_root(start: Path) -> Path | None:
        cur = start
        # search up to filesystem root
        while True:
            if (cur / "pyproject.toml").exists() or (cur / ".git").exists():
                return cur
            parent = cur.parent
            if parent == cur:
                return None
            cur = parent

    cwd = Path.cwd()
    root = _find_project_root(cwd)
    if root is not None:
        base = root / "training_data" / "curated_datasets"
        try:
            base.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return base

    # 3) Fallback to user-local data dir
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    if home:
        base = Path(home) / ".local" / "share" / "aios" / "datasets"
    else:
        base = Path.home() / ".local" / "share" / "aios" / "datasets"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return base


def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except Exception:
                continue
    except Exception:
        pass
    return int(total)


def datasets_storage_usage_gb() -> float:
    """Compute total size of dataset storage directory in GB."""
    base = datasets_base_dir()
    return _dir_size_bytes(base) / (1024 ** 3)


def datasets_storage_cap_gb() -> float:
    """Return the configured cap (GB) for dataset storage.

    Reads ~/.config/aios/datasets.json {"cap_gb": float} when present; falls back to default.
    """
    try:
        p = _cap_config_path()
        if p.exists():
            import json as _json
            with p.open("r", encoding="utf-8") as f:
                data = _json.load(f) or {}
            cap = float(data.get("cap_gb", _DATASETS_CAP_GB))
            if cap > 0:
                return cap
    except Exception:
        pass
    return _DATASETS_CAP_GB


def set_datasets_storage_cap_gb(cap_gb: float) -> bool:
    """Persistently set the dataset storage cap in GB.

    Stores value under ~/.config/aios/datasets.json.
    """
    try:
        cap = float(cap_gb)
        if not (cap > 0):
            return False
        import json as _json
        p = _cap_config_path()
        with p.open("w", encoding="utf-8") as f:
            _json.dump({"cap_gb": cap}, f)
        return True
    except Exception:
        return False


def can_store_additional_gb(required_gb: float) -> bool:
    try:
        used = datasets_storage_usage_gb()
        return (used + float(required_gb)) <= datasets_storage_cap_gb()
    except Exception:
        return False


@dataclass
class KnownDataset:
    name: str
    url: str
    approx_size_gb: float
    notes: str = ""


def known_datasets(max_size_gb: float = 15.0) -> List[KnownDataset]:
    """A curated list of popular NLP datasets under a size threshold.

    The canonical list: https://github.com/niderhoff/nlp-datasets
    Here we provide a small, practical subset with approximate sizes.
    """
    all_ds: List[KnownDataset] = [
        KnownDataset("ag_news", "https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv", 0.1, "AG News classification"),
        KnownDataset("dbpedia", "https://github.com/le-scientifique/torchDatasets/tree/master/dbpedia_csv", 0.3, "DBPedia ontology classification"),
        KnownDataset("yelp_review_polarity", "https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz", 3.6, "Yelp polarity reviews"),
        KnownDataset("amazon_review_polarity", "https://s3.amazonaws.com/fast-ai-nlp/amazon_review_polarity_csv.tgz", 4.0, "Amazon polarity reviews"),
        KnownDataset("wikitext-103", "https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/", 0.6, "Language modeling"),
        KnownDataset("snli", "https://nlp.stanford.edu/projects/snli/", 0.3, "Stanford NLI"),
        KnownDataset("multi_nli", "https://cims.nyu.edu/~sbowman/multinli/", 0.4, "MultiNLI"),
        KnownDataset("squad_v1", "https://rajpurkar.github.io/SQuAD-explorer/", 0.2, "QA dataset"),
        KnownDataset("imdb_reviews", "https://ai.stanford.edu/~amaas/data/sentiment/", 0.25, "IMDB reviews"),
    ]
    return [d for d in all_ds if d.approx_size_gb <= float(max_size_gb)]


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


def _is_archive(path: Path) -> bool:
    name = path.name.lower()
    for ext in ARCHIVE_EXTS:
        if name.endswith(ext):
            return True
    return False


def _iter_text_from_zip(zp: zipfile.ZipFile, max_lines: int) -> List[str]:
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


def read_text_lines_sample_any(path: str | Path, max_lines: int = 1000, cycle: int = 0) -> List[str]:
    """Read up to max_lines lines from a dataset path that can be a file, directory, or archive.

    Supported archives (best-effort): zip, tar(.gz/.bz2/.xz), gz (single file), and optionally rar/7z if libraries are installed.
    Supports HuggingFace datasets via .arrow files or dataset_info.json.
    
    Special format: hf://<dataset_path>[:config][:split] for streaming from HuggingFace Hub.
    Example: hf://wikitext:wikitext-2-raw-v1:train
    """
    # Check for HuggingFace Hub streaming format
    if isinstance(path, str) and path.startswith("hf://"):
        # Coordinate with stream manager to avoid conflicts
        stream_mgr = None
        dataset_id = None
        try:
            from .stream_manager import get_stream_manager
            stream_mgr = get_stream_manager()
        except ImportError:
            pass  # Stream manager not available
        
        # Get streaming cache
        chunk_cache = None
        try:
            from .streaming_cache import get_cache
            chunk_cache = get_cache()
        except ImportError:
            pass  # Cache not available
        
        try:
            from datasets import load_dataset
            
            # Parse hf://dataset_path[:config][:split]
            hf_path = path[5:]  # Remove 'hf://' prefix
            parts = hf_path.split(":")
            
            dataset_path = parts[0]
            config = parts[1] if len(parts) > 1 else None
            split = parts[2] if len(parts) > 2 else "train"
            dataset_id = dataset_path
            
            # Calculate chunk index based on cycle to rotate through different data portions
            # Rotate through 5 chunks to provide variety across training cycles
            # cycle=0 uses chunk 0, cycle=1 uses chunk 1, etc.
            chunk_index = cycle % 5
            
            # Try to get from cache first
            if chunk_cache:
                cached_lines = chunk_cache.get_cached_chunk(
                    dataset_path=dataset_path,
                    config=config,
                    split=split,
                    chunk_index=chunk_index,
                    max_age_hours=72.0  # Cache valid for 3 days
                )
                if cached_lines:
                    try:
                        print(f"[Cache] Using cached chunk for {dataset_path} (chunk {chunk_index}, {len(cached_lines)} lines)")
                    except Exception:
                        pass  # Ignore encoding errors in print
                    # Return the requested number of lines from cache
                    return cached_lines[:max_lines]
            
            # Register training with stream manager
            if stream_mgr:
                can_proceed, reason = stream_mgr.can_train(dataset_path)
                if can_proceed:
                    success, msg = stream_mgr.register_training(dataset_path)
                    if success and "pause" in reason.lower():
                        try:
                            print(f"[Info] Training registered: {reason}")
                        except Exception:
                            pass  # Ignore encoding errors
            
            # Load with streaming to avoid downloading entire dataset
            dataset = load_dataset(
                dataset_path,
                name=config,
                split=split,
                streaming=True
            )
            
            # Try common text column names
            text_columns = ['text', 'content', 'sentence', 'document', 'article', 'body', 'input', 'output']
            found_column = None
            
            # Peek at first item to find text column
            try:
                first_item = next(iter(dataset))
                if isinstance(first_item, dict):
                    # Check for standard text columns
                    for col in text_columns:
                        if col in first_item and first_item[col]:
                            found_column = col
                            break
                    # If no standard column, use first string-like column
                    if found_column is None:
                        for key, value in first_item.items():
                            if isinstance(value, str) and value.strip():
                                found_column = key
                                break
            except Exception:
                found_column = "text"  # Default fallback
            
            # Extract lines with limit
            out: List[str] = []
            if found_column:
                for item in dataset:
                    if len(out) >= max_lines:
                        break
                    try:
                        if isinstance(item, dict):
                            text = item.get(found_column, "")  # type: ignore
                        else:
                            text = item[found_column]  # type: ignore
                        if text and str(text).strip():
                            out.append(str(text).strip())
                    except Exception:
                        continue
            
            # Cache the downloaded chunk for future use
            if chunk_cache and out:
                try:
                    chunk_cache.save_chunk(
                        dataset_path=dataset_path,
                        config=config,
                        split=split,
                        chunk_index=chunk_index,
                        lines=out
                    )
                    try:
                        print(f"[Cache] Saved chunk for {dataset_path} (chunk {chunk_index}, {len(out)} lines)")
                    except Exception:
                        pass  # Ignore encoding errors in print
                except Exception as e:
                    # Non-critical failure
                    pass
            
            # Unregister training
            if stream_mgr and dataset_id:
                stream_mgr.unregister_training(dataset_id)
            
            return out
            
        except ImportError:
            print("Warning: datasets library not installed. Cannot stream from HuggingFace. Install with: pip install datasets")
            # Unregister training on error
            if stream_mgr and dataset_id:
                stream_mgr.unregister_training(dataset_id)
            return []
        except Exception as e:
            print(f"Warning: Failed to load HuggingFace dataset {path}: {e}")
            # Unregister training on error
            if stream_mgr and dataset_id:
                stream_mgr.unregister_training(dataset_id)
            return []
    
    p = Path(path)
    try:
        if p.is_dir():
            # Check if this is a HuggingFace dataset directory
            # HF datasets have dataset_info.json, data/ directory, or .arrow files
            is_hf_dataset = False
            try:
                if (p / "dataset_info.json").exists() or (p / "data").is_dir() or any(p.glob("*.arrow")):
                    is_hf_dataset = True
            except Exception:
                pass
            
            # Try loading as HuggingFace dataset first
            if is_hf_dataset:
                try:
                    from datasets import load_from_disk
                    dataset = load_from_disk(str(p))
                    
                    # Extract text from the dataset
                    out: List[str] = []
                    
                    # Try common text column names
                    text_columns = ['text', 'content', 'sentence', 'document', 'article', 'body']
                    found_column = None
                    
                    # Check which columns exist
                    if hasattr(dataset, 'column_names'):
                        available_columns = dataset.column_names
                        for col in text_columns:
                            if col in available_columns:
                                found_column = col
                                break
                        
                        # If no standard text column, use the first string column
                        if found_column is None and len(available_columns) > 0:
                            try:
                                found_column = str(list(available_columns)[0])
                            except Exception:
                                pass
                    
                    # Extract lines from dataset
                    if found_column:
                        count = 0
                        for item in dataset:
                            if count >= max_lines:
                                break
                            try:
                                # Handle dict-like items
                                if hasattr(item, 'get'):
                                    text = item.get(found_column, '')  # type: ignore
                                elif hasattr(item, '__getitem__'):
                                    text = item[found_column]  # type: ignore
                                else:
                                    continue
                                    
                                if text and str(text).strip():
                                    out.append(str(text).strip())
                                    count += 1
                            except Exception:
                                continue
                        
                        if out:
                            return out
                except ImportError:
                    # datasets library not available, fall back to file scanning
                    pass
                except Exception:
                    # HF dataset loading failed, fall back to file scanning
                    pass
            
            # Fallback: Scan text-like files in directory
            out: List[str] = []
            for fp in p.rglob("*"):
                if not fp.is_file():
                    continue
                if fp.suffix.lower() in TEXT_EXTS:
                    try:
                        with fp.open("r", encoding="utf-8", errors="ignore") as f:
                            for line in f:
                                out.append(line.strip())
                                if len(out) >= max_lines:
                                    return out
                    except Exception:
                        continue
            return out
        if not p.exists():
            return []
        if not _is_archive(p):
            # If CSV, prefer treating each row as a JSON-ish line only when asked; otherwise fallback to raw lines
            if p.suffix.lower() == ".csv":
                # Return raw lines so CLI can re-parse with proper column selection
                return read_text_lines_sample(p, max_lines=max_lines)
            return read_text_lines_sample(p, max_lines=max_lines)

        # Archive handling
        name = p.name.lower()
        # zip
        if name.endswith(".zip"):
            try:
                with zipfile.ZipFile(p, mode="r") as zf:
                    return _iter_text_from_zip(zf, max_lines)
            except Exception:
                return []
        # tar variants
        if name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")):
            try:
                mode = "r"
                if name.endswith((".tar.gz", ".tgz")):
                    mode = "r:gz"
                elif name.endswith((".tar.bz2", ".tbz2")):
                    mode = "r:bz2"
                elif name.endswith((".tar.xz", ".txz")):
                    mode = "r:xz"
                with tarfile.open(p, mode) as tf:
                    return _iter_text_from_tar(tf, max_lines)
            except Exception:
                return []
        # single-file gzip/bz2/xz
        if name.endswith(".gz") and not name.endswith((".tar.gz", ".tgz")):
            try:
                with gzip.open(p, mode="rt", encoding="utf-8", errors="ignore") as f:
                    out: List[str] = []
                    for line in f:
                        out.append(line.strip())
                        if len(out) >= max_lines:
                            break
                    return out
            except Exception:
                return []

        # rar and 7z via optional libs
        if name.endswith(".rar"):
            try:
                import rarfile  # type: ignore
                with rarfile.RarFile(p) as rf:  # type: ignore
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
        if name.endswith(".7z"):
            try:
                import py7zr  # type: ignore
                out: List[str] = []
                with py7zr.SevenZipFile(p, mode="r") as z:  # type: ignore
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
    except Exception:
        return []


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
