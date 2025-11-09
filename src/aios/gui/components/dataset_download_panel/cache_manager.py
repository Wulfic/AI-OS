"""
Search Cache Manager

Functions for caching search results to disk for faster startup.
Cache is stored in ~/.config/aios/dataset_search_cache.json
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional


def get_cache_file_path() -> Path:
    """Get the path to the search cache file."""
    cache_file = Path.home() / ".config" / "aios" / "dataset_search_cache.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    return cache_file


def load_search_cache() -> Optional[Dict[str, Any]]:
    """
    Load cached search results from disk.
    
    Returns:
        Cache data dict with 'timestamp', 'query', and 'results' keys,
        or None if cache doesn't exist or is invalid
    """
    cache_file = get_cache_file_path()
    
    try:
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Validate cache age (only use if less than 24 hours old)
            cache_time = cache_data.get('timestamp', 0)
            current_time = time.time()
            if current_time - cache_time < 86400:  # 24 hours
                return cache_data
    except Exception:
        pass  # Cache read failed, return None
    
    return None


def save_search_cache(query: str, results: List[Dict[str, Any]]) -> bool:
    """
    Save search results to cache file.
    
    Args:
        query: Search query that was used
        results: List of dataset info dictionaries
        
    Returns:
        True if successful, False otherwise
    """
    if not results:
        return False
    
    try:
        cache_data = {
            'timestamp': time.time(),
            'query': query,
            'results': results
        }
        
        cache_file = get_cache_file_path()
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
        
        return True
    except Exception:
        # Silently fail cache save - not critical
        return False
