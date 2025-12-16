"""
HuggingFace Dataset Size Detection

Utilities to fetch dataset sizes and row counts from HuggingFace Hub
without downloading the actual dataset. Supports text, image, audio, and video datasets.

Uses a 3-tier detection approach:
1. Dataset Viewer /size API (fastest, exact counts)
2. Dataset Viewer /info API (exact counts + feature types)
3. Hub API with type-aware estimation (optional fallback, disabled by default)
"""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Optional, Dict, Any, Tuple, List

import httpx

logger = logging.getLogger(__name__)

# Create a reusable HTTP client for synchronous requests
_http_client = httpx.Client(timeout=10.0, follow_redirects=True)

# Lazy import - will be None if library not installed
try:
    from huggingface_hub import HfApi
    HF_API_AVAILABLE = True
except ImportError:
    HfApi = None  # type: ignore
    HF_API_AVAILABLE = False

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}

_MANUAL_HUB_ENV = os.getenv("AIOS_ENABLE_HF_HUB_FALLBACK")
if _MANUAL_HUB_ENV is not None:
    _MANUAL_HUB_ENV = _MANUAL_HUB_ENV.strip().lower()
    MANUAL_HUB_FALLBACK = _MANUAL_HUB_ENV in _TRUE_ENV_VALUES
    AUTO_HUB_FALLBACK = False
else:
    MANUAL_HUB_FALLBACK = False
    _AUTO_HUB_ENV = os.getenv("AIOS_AUTO_ENABLE_HF_HUB_FALLBACK", "1").strip().lower()
    AUTO_HUB_FALLBACK = _AUTO_HUB_ENV not in _FALSE_ENV_VALUES

ENABLE_HUB_FALLBACK = MANUAL_HUB_FALLBACK or AUTO_HUB_FALLBACK
if MANUAL_HUB_FALLBACK:
    HUB_FALLBACK_MODE = "manual"
elif AUTO_HUB_FALLBACK:
    HUB_FALLBACK_MODE = "auto"
else:
    HUB_FALLBACK_MODE = "disabled"

_FAILURE_LOCK = threading.Lock()
_FAILURE_CACHE: Dict[str, Tuple[int, float]] = {}
_FAILURE_BACKOFF_BASE = 30.0
_FAILURE_BACKOFF_MAX = 600.0

# Constants
SAMPLES_PER_BLOCK = 100000  # Standard block size for training
API_TIMEOUT_DEFAULT = 3  # seconds - reduced to prevent hanging

# Bytes per row estimates for different data types
BYTES_PER_ROW_ESTIMATES = {
    "text": 500,            # Plain text, JSON, CSV
    "image_small": 10_000,   # Thumbnails, CIFAR (32x32)
    "image_medium": 100_000, # Photos, ImageNet (224x224)
    "image_large": 500_000,  # High-resolution images
    "audio": 500_000,        # Audio clips (varies by duration)
    "video": 10_000_000,     # Video clips (very rough estimate)
    "embedding": 3_000,      # Fixed-size vectors (e.g., 768 dims)
}
DEFAULT_BYTES_PER_ROW = BYTES_PER_ROW_ESTIMATES["text"]


def _should_skip_dataset(dataset_name: str) -> bool:
    now = time.monotonic()
    with _FAILURE_LOCK:
        entry = _FAILURE_CACHE.get(dataset_name)
        if not entry:
            return False
        _, retry_at = entry
        return retry_at > now


def _record_failure(dataset_name: str) -> None:
    now = time.monotonic()
    with _FAILURE_LOCK:
        count, _ = _FAILURE_CACHE.get(dataset_name, (0, 0.0))
        count += 1
        delay = min(_FAILURE_BACKOFF_BASE * (2 ** (count - 1)), _FAILURE_BACKOFF_MAX)
        _FAILURE_CACHE[dataset_name] = (count, now + delay)


def _record_success(dataset_name: str) -> None:
    with _FAILURE_LOCK:
        if dataset_name in _FAILURE_CACHE:
            del _FAILURE_CACHE[dataset_name]


def get_hf_dataset_size_api(
    dataset_name: str,
    config: Optional[str] = None,
    split: str = "train",
    *,
    request_timeout: float | None = None,
    max_attempts: int | None = None,
) -> Optional[Dict[str, Any]]:
    """
    Get dataset size information from HuggingFace Dataset Viewer /size API.
    
    This is the preferred method as it provides exact row counts and sizes
    without downloading the dataset.
    
    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "ibm/duorc")
        config: Optional config/subset name
        split: Dataset split (default: "train")
    
    Returns:
        Dictionary with keys:
            - num_rows: Number of rows in the dataset
            - num_bytes: Size in bytes
            - size_mb: Size in megabytes
            - size_gb: Size in gigabytes
            - total_blocks: Number of 100k-sample blocks
            - samples_per_block: Block size constant
            - is_partial: Whether data is incomplete
            - source: "dataset_viewer_api"
        Returns None if request fails
    
    Example:
        >>> info = get_hf_dataset_size_api("stanfordnlp/imdb")
        >>> print(f"{info['num_rows']:,} rows, {info['total_blocks']} blocks")
        25,000 rows, 1 blocks
    """
    try:
        url = f"https://datasets-server.huggingface.co/size?dataset={dataset_name}"
        timeout = request_timeout or API_TIMEOUT_DEFAULT
        logger.debug(f"🔍 [SIZE API] Requesting: {dataset_name} (config={config}, split={split}, timeout={timeout}s)")
        
        # Use synchronous HTTP client (works in threads without async loop)
        response = _http_client.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        logger.debug(f"📦 [SIZE API] Response keys for {dataset_name}: {list(data.keys())}")
        
        # Check if partial (incomplete data)
        is_partial = data.get("partial", False)
        
        # Extract size info based on config and split
        size_info = data.get("size", {})
        
        # Try to find specific split first
        target_split = None
        for split_data in size_info.get("splits", []):
            if split_data.get("split") == split:
                if config is None or split_data.get("config") == config:
                    target_split = split_data
                    break
        
        # Fallback to config level if no split found
        if target_split is None and config:
            for config_data in size_info.get("configs", []):
                if config_data.get("config") == config:
                    target_split = config_data
                    break
        
        # Fallback to dataset level if still not found
        if target_split is None:
            target_split = size_info.get("dataset", {})
        
        if not target_split or "num_rows" not in target_split:
            logger.debug(f"❌ [SIZE API] No row count for {dataset_name} - target_split: {target_split}")
            logger.debug(f"   Available splits: {[s.get('split') for s in size_info.get('splits', [])]}")
            logger.debug(f"   Available configs: {[c.get('config') for c in size_info.get('configs', [])]}")
            return None
        
        num_rows = target_split["num_rows"]
        num_bytes = target_split.get(
            "num_bytes_original_files",
            target_split.get("num_bytes_parquet_files", 0)
        )
        
        # Calculate blocks (100k samples per block)
        total_blocks = calculate_blocks(num_rows)
        
        result = {
            "num_rows": num_rows,
            "num_bytes": num_bytes,
            "size_mb": num_bytes / (1024 * 1024),
            "size_gb": num_bytes / (1024 * 1024 * 1024),
            "total_blocks": total_blocks,
            "samples_per_block": SAMPLES_PER_BLOCK,
            "is_partial": is_partial,
            "source": "dataset_viewer_api"
        }
        logger.debug(f"✅ [SIZE API] Success for {dataset_name}: {num_rows:,} rows, {result['size_gb']:.3f} GB")
        return result
        
    except httpx.HTTPStatusError as e:
        logger.debug(f"❌ [SIZE API] HTTP {e.response.status_code} for {dataset_name}: {e}")
        return None
    except httpx.HTTPError as e:
        logger.debug(f"❌ [SIZE API] Network error for {dataset_name}: {e}")
        return None
    except Exception as e:
        logger.debug(f"❌ [SIZE API] Unexpected error for {dataset_name}: {type(e).__name__}: {e}")
        return None


def get_hf_dataset_info_api(
    dataset_name: str,
    config: Optional[str] = None,
    split: str = "train",
    *,
    request_timeout: float | None = None,
    max_attempts: int | None = None,
) -> Optional[Dict[str, Any]]:
    """
    Get dataset info from HuggingFace Dataset Viewer /info API.
    
    This provides exact row counts AND feature types (Image, Audio, Video, etc.)
    which helps with accurate size estimation for non-text datasets.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        config: Optional config/subset name
        split: Dataset split (default: "train")
    
    Returns:
        Dictionary with keys:
            - num_rows: Number of rows (exact)
            - num_bytes: Size in bytes (estimated from row count if not available)
            - size_mb: Size in megabytes
            - size_gb: Size in gigabytes
            - total_blocks: Number of 100k-sample blocks
            - samples_per_block: Block size constant
            - feature_types: List of feature types (_type values)
            - has_images: Boolean indicating Image features
            - has_audio: Boolean indicating Audio features
            - has_video: Boolean indicating Video features
            - source: "info_api"
        Returns None if request fails
    
    Example:
        >>> info = get_hf_dataset_info_api("uoft-cs/cifar10")
        >>> print(f"Has images: {info['has_images']}, Rows: {info['num_rows']:,}")
        Has images: True, Rows: 50,000
    """
    try:
        url = f"https://datasets-server.huggingface.co/info?dataset={dataset_name}"
        timeout = request_timeout or API_TIMEOUT_DEFAULT
        logger.debug(f"🔍 [INFO API] Requesting: {dataset_name} (config={config}, split={split}, timeout={timeout}s)")
        
        # Use synchronous HTTP client (works in threads without async loop)
        response = _http_client.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        logger.debug(f"📦 [INFO API] Response keys for {dataset_name}: {list(data.keys())}")
        dataset_info = data.get("dataset_info", {})
        logger.debug(f"   Configs available: {list(dataset_info.keys())}")
        
        # Find the right config
        if config and config in dataset_info:
            config_info = dataset_info[config]
        else:
            # Use first available config
            if not dataset_info:
                return None
            config_info = next(iter(dataset_info.values()), {})
        
        # Get features to detect data types
        features = config_info.get("features", {})
        feature_types = [f.get("_type") for f in features.values() if "_type" in f]
        
        # Get split info
        splits = config_info.get("splits", {})
        if split not in splits:
            # Try to use first available split
            if not splits:
                return None
            split = next(iter(splits.keys()))
        
        split_info = splits[split]
        num_examples = split_info.get("num_examples", 0)
        num_bytes = split_info.get("num_bytes", 0)
        
        if num_examples == 0:
            logger.debug(f"❌ [INFO API] Zero examples for {dataset_name}")
            return None
        
        # Calculate blocks
        total_blocks = calculate_blocks(num_examples)
        
        result = {
            "num_rows": num_examples,
            "num_bytes": num_bytes,
            "size_mb": num_bytes / (1024 * 1024),
            "size_gb": num_bytes / (1024 * 1024 * 1024),
            "total_blocks": total_blocks,
            "samples_per_block": SAMPLES_PER_BLOCK,
            "feature_types": feature_types,
            "has_images": "Image" in feature_types,
            "has_audio": "Audio" in feature_types,
            "has_video": "Video" in feature_types,
            "num_rows_estimated": False,
            "source": "info_api"
        }
        logger.debug(f"✅ [INFO API] Success for {dataset_name}: {num_examples:,} rows, {result['size_gb']:.3f} GB, features={feature_types}")
        return result
        
    except httpx.HTTPStatusError as e:
        logger.debug(f"❌ [INFO API] HTTP {e.response.status_code} for {dataset_name}: {e}")
        return None
    except httpx.HTTPError as e:
        logger.debug(f"❌ [INFO API] Network error for {dataset_name}: {e}")
        return None
    except Exception as e:
        logger.debug(f"❌ [INFO API] Unexpected error for {dataset_name}: {type(e).__name__}: {e}")
        return None


def get_hf_dataset_size_hub_api(
    dataset_name: str,
    *,
    request_timeout: float | None = None,
    max_siblings: int = 2000,
) -> Optional[Dict[str, Any]]:
    """
    Get dataset size using HuggingFace Hub API (file-based).
    
    This method returns the total size of all files in the dataset repository.
    It doesn't provide row counts, so those must be estimated.
    
    Args:
        dataset_name: HuggingFace dataset identifier
    
    Returns:
        Dictionary with keys:
            - num_files: Number of files in repository
            - total_bytes: Total size in bytes
            - size_mb: Size in megabytes
            - size_gb: Size in gigabytes
            - source: "hub_api"
        Returns None if request fails
    
    Example:
        >>> info = get_hf_dataset_size_hub_api("stanfordnlp/imdb")
        >>> print(f"{info['size_gb']:.2f} GB across {info['num_files']} files")
        0.06 GB across 8 files
    """
    if not HF_API_AVAILABLE:
        logger.debug("huggingface_hub library not available")
        return None
    
    try:
        api = HfApi()
        timeout = max(request_timeout or API_TIMEOUT_DEFAULT, 0.1)

        def _load_dataset_info():
            return api.dataset_info(
                repo_id=dataset_name,
                files_metadata=True,
                timeout=timeout,
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_load_dataset_info)
            try:
                dataset_info = future.result(timeout=timeout + 0.5)
            except TimeoutError:
                future.cancel()
                logger.debug(
                    "Hub API size lookup timed out after %.2fs for %s",
                    timeout,
                    dataset_name,
                )
                return None

        total_bytes = 0
        file_count = 0

        for sibling in dataset_info.siblings[:max_siblings]:
            size_in_bytes = sibling.size or 0
            total_bytes += size_in_bytes
            file_count += 1

        if len(dataset_info.siblings) > max_siblings:
            logger.debug(
                "Hub API truncated %d of %d file entries for %s",
                len(dataset_info.siblings) - max_siblings,
                len(dataset_info.siblings),
                dataset_name,
            )

        return {
            "num_files": file_count,
            "total_bytes": total_bytes,
            "size_mb": total_bytes / (1024 * 1024),
            "size_gb": total_bytes / (1024 * 1024 * 1024),
            "source": "hub_api"
        }
        
    except TimeoutError:
        logger.debug("Hub API size lookup thread hung for %s", dataset_name)
        return None
    except Exception as e:
        logger.debug(f"Error fetching dataset size via Hub API for {dataset_name}: {e}")
        return None


def estimate_bytes_per_row_from_features(feature_types: List[str]) -> int:
    """
    Estimate bytes per row based on detected feature types.
    
    Different data types have vastly different sizes:
    - Text: ~500 bytes
    - Images: 10 KB - 500 KB depending on resolution
    - Audio: ~500 KB (varies by duration)
    - Video: ~10 MB (very rough estimate)
    
    Args:
        feature_types: List of feature type strings from dataset info
    
    Returns:
        Estimated bytes per row for the dataset
    
    Example:
        >>> estimate_bytes_per_row_from_features(["Image", "ClassLabel"])
        100000  # Image dataset (medium-sized images)
    """
    if "Video" in feature_types:
        return BYTES_PER_ROW_ESTIMATES["video"]
    elif "Audio" in feature_types:
        return BYTES_PER_ROW_ESTIMATES["audio"]
    elif "Image" in feature_types:
        # Default to medium-sized images (most common)
        return BYTES_PER_ROW_ESTIMATES["image_medium"]
    else:
        # Text or structured data
        return BYTES_PER_ROW_ESTIMATES["text"]


def estimate_rows_from_size(size_bytes: int, avg_bytes_per_row: int = DEFAULT_BYTES_PER_ROW) -> int:
    """
    Estimate number of rows based on file size.
    
    This is used when we only have file sizes (from Hub API) and not actual row counts.
    The default of 500 bytes per row is reasonable for most text datasets.
    
    Args:
        size_bytes: Total dataset size in bytes
        avg_bytes_per_row: Average bytes per row (default: 500 for text)
    
    Returns:
        Estimated number of rows
    
    Example:
        >>> estimate_rows_from_size(50_000_000)  # 50 MB text
        100000
        >>> estimate_rows_from_size(50_000_000, BYTES_PER_ROW_ESTIMATES["image_medium"])  # 50 MB images
        500
    """
    if size_bytes <= 0:
        return 0
    return max(1, size_bytes // avg_bytes_per_row)


def calculate_blocks(num_rows: int) -> int:
    """
    Calculate number of training blocks for a dataset.
    
    A "block" represents 100k samples - the standard chunk size for
    streaming and processing datasets in AI-OS training.
    
    Args:
        num_rows: Total number of rows in dataset
    
    Returns:
        Number of blocks (minimum 1 if num_rows > 0)
    
    Example:
        >>> calculate_blocks(250000)
        3
        >>> calculate_blocks(50000)
        1
    """
    if num_rows <= 0:
        return 0
    return (num_rows + SAMPLES_PER_BLOCK - 1) // SAMPLES_PER_BLOCK


def estimate_download_size_gb(num_rows: int, modality: str = "text", feature_types: Optional[List[str]] = None) -> float:
    """
    Estimate download size based on row count and data type.
    
    Args:
        num_rows: Number of rows/samples
        modality: Dataset modality from tags (text, image, audio, video, etc.)
        feature_types: List of feature types from dataset_info (if available)
        
    Returns:
        Estimated size in GB
    """
    # Detect data type from feature_types if available
    if feature_types:
        if "Image" in feature_types:
            bytes_per_row = BYTES_PER_ROW_ESTIMATES["image_medium"]
        elif "Audio" in feature_types:
            bytes_per_row = BYTES_PER_ROW_ESTIMATES["audio"]
        elif "Video" in feature_types:
            bytes_per_row = BYTES_PER_ROW_ESTIMATES["video"]
        else:
            bytes_per_row = BYTES_PER_ROW_ESTIMATES.get(modality.lower(), DEFAULT_BYTES_PER_ROW)
    else:
        bytes_per_row = BYTES_PER_ROW_ESTIMATES.get(modality.lower(), DEFAULT_BYTES_PER_ROW)
    
    total_bytes = num_rows * bytes_per_row
    return total_bytes / (1024 ** 3)


def get_hf_dataset_metadata(
    dataset_name: str,
    config: Optional[str] = None,
    split: str = "train",
    *,
    request_timeout: float | None = None,
    max_attempts: int | None = None,
) -> Optional[Dict[str, Any]]:
    """
    Get HuggingFace dataset metadata using best available method.
    
    This is the main entry point for getting dataset size information.
    It tries multiple methods in order of preference:
    1. Dataset Viewer /size API (fastest, exact rows)
    2. Dataset Viewer /info API (exact rows + feature types)
    3. Hub API with type-aware estimation (slowest, estimated rows)
    
    Args:
        dataset_name: HuggingFace dataset identifier
        config: Optional config/subset name
        split: Dataset split (default: "train")
    
    Returns:
        Dictionary with comprehensive size information including:
            - num_rows: Number of rows (exact or estimated)
            - num_rows_estimated: Boolean indicating if rows are estimated
            - num_bytes: Size in bytes
            - size_mb: Size in megabytes
            - size_gb: Size in gigabytes
            - total_blocks: Number of 100k-sample blocks
            - samples_per_block: Block size constant
            - feature_types: List of feature types (if available)
            - has_images/has_audio/has_video: Booleans (if detected)
            - estimation_quality: "exact", "good", or "low"
            - source: Which API provided the data
        Returns None if all methods fail
    
    Example:
        >>> info = get_hf_dataset_metadata("uoft-cs/cifar10")
        >>> print(f"CIFAR-10: {info['num_rows']:,} rows, has_images={info.get('has_images', False)}")
        CIFAR-10: 50,000 rows, has_images=True
    """
    if _should_skip_dataset(dataset_name):
        logger.debug("Skipping dataset %s due to recent failures", dataset_name)
        return None

    # Try Dataset Viewer /size API first (fastest, exact row counts)
    timeout = request_timeout or API_TIMEOUT_DEFAULT
    attempts = max_attempts or 2
    
    logger.debug(f"📋 [METADATA] Starting lookup for {dataset_name}")

    info = get_hf_dataset_size_api(
        dataset_name,
        config,
        split,
        request_timeout=timeout,
        max_attempts=attempts,
    )
    if info and not info.get("is_partial", False):
        info["num_rows_estimated"] = False
        info["estimation_quality"] = "exact"
        _record_success(dataset_name)
        logger.debug(f"✅ [METADATA] Got exact data from SIZE API for {dataset_name}")
        return info
    elif info:
        logger.debug(f"⚠️ [METADATA] SIZE API returned partial data for {dataset_name}, trying INFO API")
    
    # Try Dataset Viewer /info API (exact rows + feature types)
    info = get_hf_dataset_info_api(
        dataset_name,
        config,
        split,
        request_timeout=timeout,
        max_attempts=attempts,
    )
    if info:
        info["estimation_quality"] = "exact"
        _record_success(dataset_name)
        logger.debug(f"✅ [METADATA] Got exact data from INFO API for {dataset_name}")
        return info
    
    allow_hub_fallback = ENABLE_HUB_FALLBACK and HF_API_AVAILABLE
    if allow_hub_fallback:
        logger.debug(
            "Using HuggingFace Hub fallback (%s mode) for %s",
            HUB_FALLBACK_MODE,
            dataset_name,
        )
        hub_info = get_hf_dataset_size_hub_api(dataset_name, request_timeout=timeout)
        if hub_info:
            bytes_per_row = DEFAULT_BYTES_PER_ROW
            estimated_rows = estimate_rows_from_size(hub_info["total_bytes"], bytes_per_row)

            total_blocks = calculate_blocks(estimated_rows)

            _record_success(dataset_name)
            metadata = {
                "num_rows": estimated_rows,
                "num_rows_estimated": True,
                "estimation_quality": "low",
                "num_bytes": hub_info["total_bytes"],
                "size_mb": hub_info["size_mb"],
                "size_gb": hub_info["size_gb"],
                "total_blocks": total_blocks,
                "samples_per_block": SAMPLES_PER_BLOCK,
                "source": "hub_api_estimated",
            }
            if HUB_FALLBACK_MODE != "disabled":
                metadata["fallback_mode"] = HUB_FALLBACK_MODE
            return metadata
    else:
        if ENABLE_HUB_FALLBACK and not HF_API_AVAILABLE:
            logger.debug(
                "Hub API fallback (%s mode) requested but huggingface_hub is not installed",
                HUB_FALLBACK_MODE,
            )
        else:
            logger.debug("Hub API fallback disabled; skipping dataset %s", dataset_name)

    logger.warning(f"❌ [METADATA] All methods failed for {dataset_name} - dataset will show as 'Unknown'")
    _record_failure(dataset_name)
    return None


def enrich_dataset_with_size(
    dataset: Dict[str, Any],
    timeout: float | None = None,
    *,
    max_attempts: int | None = None,
) -> Dict[str, Any]:
    """
    Enrich a dataset dictionary with size and block information.
    
    This modifies the dataset dictionary in-place, adding size-related fields.
    If size detection fails, the original dictionary is returned unchanged.
    
    Args:
        dataset: Dataset dictionary from search results with at least:
            - path or id: Dataset identifier
            - config (optional): Config/subset name
            - split (optional): Split name (default: "train")
        timeout: Maximum time to wait for API response (default: 3.0 seconds)
    
    Returns:
        Updated dataset dictionary with additional fields:
            - size_gb: Size in gigabytes
            - size_mb: Size in megabytes
            - num_rows: Number of rows
            - total_blocks: Number of training blocks
            - samples_per_block: Block size (100000)
            - size_estimated: Whether row count is estimated
    
    Example:
        >>> dataset = {"path": "stanfordnlp/imdb", "name": "IMDB"}
        >>> enriched = enrich_dataset_with_size(dataset)
        >>> print(f"{enriched['num_rows']} rows, {enriched['size_gb']:.2f} GB")
        25000 rows, 0.06 GB
    """
    dataset_name = dataset.get("path", dataset.get("id", ""))
    config = dataset.get("config")
    split = dataset.get("split", "train")
    
    if not dataset_name:
        logger.debug("Cannot enrich dataset: no path or id found")
        return dataset
    
    # Get metadata with timeout protection
    try:
        import signal
        
        # Use timeout for the metadata retrieval
        metadata = get_hf_dataset_metadata(
            dataset_name,
            config,
            split,
            request_timeout=timeout,
            max_attempts=max_attempts,
        )
        
        if metadata:
            dataset["size_gb"] = metadata.get("size_gb", 0.0)
            dataset["size_mb"] = metadata.get("size_mb", 0.0)
            dataset["num_rows"] = metadata.get("num_rows", 0)
            dataset["total_blocks"] = metadata.get("total_blocks", 0)
            dataset["samples_per_block"] = metadata.get("samples_per_block", SAMPLES_PER_BLOCK)
            dataset["size_estimated"] = metadata.get("num_rows_estimated", False)
            dataset["size_source"] = metadata.get("source", "unknown")
            dataset["estimation_quality"] = metadata.get("estimation_quality", "unknown")
            
            # Add feature type info if available
            if "feature_types" in metadata:
                dataset["feature_types"] = metadata["feature_types"]
                dataset["has_images"] = metadata.get("has_images", False)
                dataset["has_audio"] = metadata.get("has_audio", False)
                dataset["has_video"] = metadata.get("has_video", False)
            
            logger.info(
                f"✅ [ENRICH] {dataset_name}: {dataset['num_rows']:,} rows, "
                f"{dataset['size_gb']:.2f} GB, {dataset['total_blocks']} blocks "
                f"(quality: {dataset['estimation_quality']}, source: {dataset['size_source']})"
            )
        else:
            logger.warning(f"❌ [ENRICH] Failed to enrich {dataset_name} - will show as 'Unknown' in UI")
            logger.debug(f"Could not fetch size metadata for {dataset_name}")
            
    except Exception as e:
        logger.debug(f"Error enriching dataset {dataset_name}: {e}")
    
    return dataset


def format_size_display(dataset: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Format dataset size information for display in UI.
    
    Args:
        dataset: Enriched dataset dictionary
    
    Returns:
        Tuple of (size_str, rows_str, blocks_str) formatted for display
    
    Example:
        >>> dataset = {"size_gb": 1.5, "num_rows": 150000, "total_blocks": 2, "size_estimated": False}
        >>> size, rows, blocks = format_size_display(dataset)
        >>> print(f"{size} | {rows} | {blocks}")
        1.50 GB | 150,000 rows | 2 blocks
    """
    # Format size
    size_gb = dataset.get("size_gb", 0.0)
    if size_gb >= 1.0:
        size_str = f"{size_gb:.2f} GB"
    elif size_gb > 0:
        size_mb = dataset.get("size_mb", 0.0)
        size_str = f"{size_mb:.1f} MB"
    else:
        size_str = "Unknown"
    
    # Format rows
    num_rows = dataset.get("num_rows", 0)
    is_estimated = dataset.get("size_estimated", False)
    if num_rows > 0:
        rows_str = f"{num_rows:,} rows"
        if is_estimated:
            rows_str += " (est.)"
    else:
        rows_str = "Unknown"
    
    # Format blocks
    total_blocks = dataset.get("total_blocks", 0)
    if total_blocks > 0:
        blocks_str = f"{total_blocks} block{'s' if total_blocks != 1 else ''}"
    else:
        blocks_str = "Unknown"
    
    return size_str, rows_str, blocks_str


if __name__ == "__main__":
    # Test the module
    import sys
    
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        config = sys.argv[2] if len(sys.argv) > 2 else None
        split = sys.argv[3] if len(sys.argv) > 3 else "train"
        
        print(f"\nFetching size info for: {dataset_name}")
        if config:
            print(f"Config: {config}")
        print(f"Split: {split}\n")
        
        info = get_hf_dataset_metadata(dataset_name, config, split)
        
        if info:
            print("✅ Success!")
            print(f"Rows: {info['num_rows']:,}")
            if info.get('num_rows_estimated'):
                print("  (estimated from file size)")
            print(f"Size: {info['size_gb']:.2f} GB ({info['size_mb']:.1f} MB)")
            print(f"Blocks: {info['total_blocks']}")
            print(f"Samples per block: {info['samples_per_block']:,}")
            
            # Show feature types if available
            if 'feature_types' in info:
                print(f"Feature types: {', '.join(info['feature_types'])}")
                if info.get('has_images'):
                    print("  📷 Contains images")
                if info.get('has_audio'):
                    print("  🔊 Contains audio")
                if info.get('has_video'):
                    print("  🎥 Contains video")
            
            print(f"Estimation quality: {info.get('estimation_quality', 'unknown')}")
            print(f"Source: {info['source']}")
        else:
            print("❌ Failed to get size information")
    else:
        # Run basic tests
        print("Testing dataset size detection...\n")
        
        test_datasets = [
            ("stanfordnlp/imdb", None, "train"),
            ("uoft-cs/cifar10", "plain_text", "train"),
            ("ibm/duorc", "ParaphraseRC", "train"),
        ]
        
        for dataset_name, config, split in test_datasets:
            print(f"Testing: {dataset_name}")
            if config:
                print(f"  Config: {config}, Split: {split}")
            
            info = get_hf_dataset_metadata(dataset_name, config, split)
            
            if info:
                type_info = ""
                if info.get('has_images'):
                    type_info = " 📷"
                elif info.get('has_audio'):
                    type_info = " 🔊"
                elif info.get('has_video'):
                    type_info = " 🎥"
                
                print(f"  ✅ {info['num_rows']:,} rows | {info['size_gb']:.2f} GB | {info['total_blocks']} blocks{type_info}")
                print(f"     Quality: {info.get('estimation_quality', 'unknown')}, Source: {info['source']}")
            else:
                print(f"  ❌ Failed")
            print()
