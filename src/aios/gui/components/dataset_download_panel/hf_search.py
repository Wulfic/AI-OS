"""
HuggingFace Dataset Search

Functions for searching HuggingFace Hub for datasets.
Lazy imports HuggingFace libraries for graceful degradation.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Lazy import - will be None if library not installed
try:
    from huggingface_hub import login, whoami, HfFolder, list_datasets
except ImportError:
    login = None  # type: ignore
    whoami = None  # type: ignore
    HfFolder = None  # type: ignore
    list_datasets = None  # type: ignore

# Mapping from UI modality names to HuggingFace Hub filter tags
# Based on HuggingFace Hub dataset modality tags
MODALITY_TO_HF_FILTER: Dict[str, Optional[str]] = {
    "All": None,              # No filter
    "Text": "modality:text",  # Text datasets
    "Audio": "modality:audio",
    "Document": "modality:document", 
    "Geospatial": "modality:geospatial",
    "Image": "modality:image",
    "Tabular": "modality:tabular",
    "Time-series": "modality:timeseries",
    "Video": "modality:video",
    "3D": "modality:3d",
}


def search_huggingface_datasets(
    query: str, 
    limit: int = 50, 
    modality: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search HuggingFace Hub for datasets.
    
    Args:
        query: Search query string
        limit: Maximum number of results to return
        modality: Modality filter ("Text", "Audio", "Image", etc.) or None/All for no filter
        
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
        # Build filter list for modality
        filter_list: List[str] = []
        
        if modality and modality != "All":
            hf_filter = MODALITY_TO_HF_FILTER.get(modality)
            if hf_filter:
                filter_list.append(hf_filter)
                logger.debug(f"Applying modality filter: {hf_filter}")
        
        # Search datasets on HuggingFace Hub
        # Note: The filter parameter accepts a list of strings for tag-based filtering
        list_kwargs = {
            "search": query if query.strip() else None,
            "sort": "downloads",
            "direction": -1,
            "limit": limit,
        }
        
        # Add filter if we have modality criteria
        if filter_list:
            list_kwargs["filter"] = filter_list
        
        logger.debug(f"Searching HuggingFace datasets with kwargs: {list_kwargs}")
        datasets = list_datasets(**list_kwargs)
        
        results = []
        for ds in datasets:
            # Skip private or gated datasets that users cannot access
            is_private = getattr(ds, 'private', False)
            is_gated = getattr(ds, 'gated', False)
            
            # Filter out inaccessible datasets
            if is_private or is_gated:
                continue
            
            # Extract modality from tags if available
            tags = getattr(ds, 'tags', [])
            detected_modality = "Unknown"
            for tag in tags:
                if tag.startswith("modality:"):
                    detected_modality = tag.replace("modality:", "").capitalize()
                    break
            
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
                "tags": tags,
                "modality": detected_modality,
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
        
        logger.info(f"HuggingFace search returned {len(results)} results for query='{query}', modality='{modality}'")
        return results
        
    except Exception as e:
        logger.error(f"Failed to search datasets: {e}")
        raise Exception(f"Failed to search datasets: {str(e)}")
