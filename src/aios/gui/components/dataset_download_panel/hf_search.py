"""
HuggingFace Dataset Search

Functions for searching HuggingFace Hub for datasets.
Lazy imports HuggingFace libraries for graceful degradation.
"""

from typing import Dict, List, Any, Optional

# Lazy import - will be None if library not installed
try:
    from huggingface_hub import login, whoami, HfFolder, list_datasets
except ImportError:
    login = None  # type: ignore
    whoami = None  # type: ignore
    HfFolder = None  # type: ignore
    list_datasets = None  # type: ignore


def search_huggingface_datasets(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Search HuggingFace Hub for datasets.
    
    Args:
        query: Search query string
        limit: Maximum number of results to return
        
    Returns:
        List of dataset info dictionaries with keys: id, name, description, 
        downloads, likes, tags, path, config, split, category, size_gb, etc.
        
    Raises:
        ImportError: If huggingface_hub library is not installed
        Exception: If search fails for other reasons
    """
    if list_datasets is None:
        raise ImportError("huggingface_hub library is not installed")
    
    try:
        # Search datasets on HuggingFace Hub
        datasets = list_datasets(
            search=query if query.strip() else None,
            sort="downloads",
            direction=-1,
            limit=limit
        )
        
        results = []
        for ds in datasets:
            # Skip private or gated datasets that users cannot access
            is_private = getattr(ds, 'private', False)
            is_gated = getattr(ds, 'gated', False)
            
            # Filter out inaccessible datasets
            if is_private or is_gated:
                continue
            
            # Extract relevant info
            dataset_info = {
                "id": ds.id,
                "name": ds.id.split("/")[-1] if "/" in ds.id else ds.id,
                "full_name": ds.id,
                "path": ds.id,
                "author": ds.author if hasattr(ds, 'author') else ds.id.split("/")[0] if "/" in ds.id else "unknown",
                "description": getattr(ds, 'description', 'No description available')[:200] if getattr(ds, 'description', None) else 'No description available',
                "downloads": getattr(ds, 'downloads', 0),
                "likes": getattr(ds, 'likes', 0),
                "tags": getattr(ds, 'tags', []),
                "private": False,  # Already filtered
                "gated": False,  # Already filtered
                # Set defaults for download
                "config": None,
                "split": "train",
                "category": "Custom",
                "size_gb": 0.0,  # Unknown until downloaded
                "verified": False,
                "streaming": True,  # Default to streaming for search results
                "max_samples": 0,  # 0 = unlimited (download entire dataset)
            }
            results.append(dataset_info)
        
        return results
        
    except Exception as e:
        raise Exception(f"Failed to search datasets: {str(e)}")
