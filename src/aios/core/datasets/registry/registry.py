"""
Dataset registry for managing dataset metadata.

Provides CRUD operations, search, filtering, and persistence.
"""

from typing import List, Optional, Dict
from pathlib import Path
import json
import logging

from .metadata import DatasetMetadata

logger = logging.getLogger(__name__)


class DatasetRegistry:
    """
    Registry for managing dataset metadata.
    
    Provides CRUD operations, search, filtering, and persistence.
    """
    
    def __init__(self):
        """Initialize empty dataset registry."""
        self.datasets: Dict[str, DatasetMetadata] = {}
        
    def add_dataset(self, metadata: DatasetMetadata) -> None:
        """
        Add a dataset to the registry.
        
        Args:
            metadata: DatasetMetadata object
        """
        self.datasets[metadata.dataset_id] = metadata
        logger.info(f"[DatasetRegistry] Added dataset: {metadata.dataset_id} ({metadata.name})")
    
    def remove_dataset(self, dataset_id: str) -> bool:
        """
        Remove a dataset from the registry.
        
        Args:
            dataset_id: Dataset identifier
        
        Returns:
            True if removed, False if not found
        """
        if dataset_id in self.datasets:
            dataset_name = self.datasets[dataset_id].name
            del self.datasets[dataset_id]
            logger.info(f"[DatasetRegistry] Removed dataset: {dataset_id} ({dataset_name})")
            return True
        logger.warning(f"[DatasetRegistry] Dataset not found: {dataset_id}")
        return False
    
    def get_dataset(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """
        Get a dataset by ID.
        
        Args:
            dataset_id: Dataset identifier
        
        Returns:
            DatasetMetadata if found, None otherwise
        """
        return self.datasets.get(dataset_id)
    
    def get_all_datasets(self) -> List[DatasetMetadata]:
        """Get all datasets in the registry."""
        return list(self.datasets.values())
    
    def get_available_datasets(self) -> List[DatasetMetadata]:
        """Get only available datasets."""
        return [ds for ds in self.datasets.values() if ds.is_available]
    
    def search_by_domain(self, domain: str) -> List[DatasetMetadata]:
        """
        Search datasets by domain.
        
        Args:
            domain: Domain to search for (e.g., "coding", "math")
        
        Returns:
            List of matching datasets
        """
        return [
            ds for ds in self.datasets.values()
            if ds.matches_domain(domain)
        ]
    
    def search_by_tags(self, tags: List[str]) -> List[DatasetMetadata]:
        """
        Search datasets by tags.
        
        Args:
            tags: List of tags to search for
        
        Returns:
            List of datasets matching any of the tags
        """
        return [
            ds for ds in self.datasets.values()
            if ds.matches_tags(tags)
        ]
    
    def search_by_category(self, category: str) -> List[DatasetMetadata]:
        """
        Search datasets by category.
        
        Args:
            category: Category to search for
        
        Returns:
            List of matching datasets
        """
        category_lower = category.lower()
        return [
            ds for ds in self.datasets.values()
            if category_lower in [cat.lower() for cat in ds.categories]
        ]
    
    def search_by_expert(self, expert_id: str) -> List[DatasetMetadata]:
        """
        Get datasets used by a specific expert.
        
        Args:
            expert_id: Expert identifier
        
        Returns:
            List of datasets used by this expert
        """
        return [
            ds for ds in self.datasets.values()
            if expert_id in ds.used_by_experts
        ]
    
    def filter_by_size(
        self,
        min_bytes: Optional[int] = None,
        max_bytes: Optional[int] = None
    ) -> List[DatasetMetadata]:
        """
        Filter datasets by size range.
        
        Args:
            min_bytes: Minimum size in bytes (inclusive)
            max_bytes: Maximum size in bytes (inclusive)
        
        Returns:
            List of datasets within size range
        """
        results = []
        for ds in self.datasets.values():
            if ds.size_bytes is None:
                continue
            
            if min_bytes is not None and ds.size_bytes < min_bytes:
                continue
            
            if max_bytes is not None and ds.size_bytes > max_bytes:
                continue
            
            results.append(ds)
        
        return results
    
    def recommend_for_expert(
        self,
        domain: str,
        categories: Optional[List[str]] = None,
        max_results: int = 5
    ) -> List[DatasetMetadata]:
        """
        Recommend datasets for a new expert based on domain/categories.
        
        Args:
            domain: Expert's primary domain
            categories: Expert's specific categories
            max_results: Maximum number of recommendations
        
        Returns:
            List of recommended datasets, sorted by relevance
        """
        # Start with domain matches
        candidates = self.search_by_domain(domain)
        
        # Add category matches if provided
        if categories:
            for category in categories:
                category_matches = self.search_by_category(category)
                candidates.extend(category_matches)
        
        # Remove duplicates and filter unavailable
        seen = set()
        unique_candidates = []
        for ds in candidates:
            if ds.dataset_id not in seen and ds.is_available:
                seen.add(ds.dataset_id)
                unique_candidates.append(ds)
        
        # Sort by quality score (if available), then by size
        unique_candidates.sort(
            key=lambda ds: (
                ds.quality_score if ds.quality_score is not None else 0.5,
                ds.size_bytes if ds.size_bytes is not None else 0
            ),
            reverse=True
        )
        
        return unique_candidates[:max_results]
    
    def save(self, path: str) -> None:
        """
        Save registry to JSON file.
        
        Args:
            path: Path to JSON file
        """
        import time
        start_time = time.time()
        
        logger.debug(f"[DatasetRegistry] Serializing {len(self.datasets)} datasets")
        data = {
            "datasets": {
                dataset_id: metadata.to_dict()
                for dataset_id, metadata in self.datasets.items()
            }
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        duration = (time.time() - start_time) * 1000  # Convert to ms
        file_size = Path(path).stat().st_size / 1024  # Convert to KB
        logger.info(f"[DatasetRegistry] Saved {len(self.datasets)} datasets to {path}")
        logger.debug(f"[DatasetRegistry] Registry saved in {duration:.1f}ms, file size: {file_size:.1f} KB")
    
    @classmethod
    def load(cls, path: str) -> "DatasetRegistry":
        """
        Load registry from JSON file.
        
        Args:
            path: Path to JSON file
        
        Returns:
            DatasetRegistry instance
        """
        import time
        start_time = time.time()
        
        registry = cls()
        
        if not Path(path).exists():
            logger.warning(f"[DatasetRegistry] File not found: {path}")
            return registry
        
        logger.debug(f"[DatasetRegistry] Loading registry from {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.debug(f"[DatasetRegistry] Validating {len(data.get('datasets', {}))} dataset entries")
        for dataset_id, dataset_data in data.get("datasets", {}).items():
            metadata = DatasetMetadata.from_dict(dataset_data)
            registry.datasets[dataset_id] = metadata
        
        duration = (time.time() - start_time) * 1000  # Convert to ms
        logger.info(f"[DatasetRegistry] Loaded {len(registry.datasets)} datasets from {path}")
        logger.debug(f"[DatasetRegistry] Registry loaded in {duration:.1f}ms")
        return registry
