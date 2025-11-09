"""
Dataset Registry for Dynamic Subbrains Auto-Training.

This module provides dataset discovery, cataloging, and management capabilities
to support the auto-training workflow. When a user requests "learn X", the system
can search for relevant datasets and automatically start training an expert.

Key Features:
- Dataset metadata tracking (name, path, domain, size, quality)
- Local dataset scanning (automatic discovery)
- HuggingFace Hub search integration
- Dataset-expert category mapping
- JSON persistence for registry state

Author: AI-OS Team
Date: January 2025
"""

from typing import List, Optional, Dict, Set, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import json
import logging
import os
import glob

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """
    Metadata for a dataset used in expert training.
    
    Attributes:
        dataset_id: Unique identifier (e.g., "python_code_dataset_001")
        name: Human-readable name (e.g., "Python Code Examples")
        description: What the dataset contains
        source_type: "local", "huggingface", "url", "custom"
        source_path: Local path or HuggingFace identifier
        domain: Primary domain (e.g., "coding", "math", "writing")
        categories: Specific categories (e.g., ["python", "programming"])
        file_format: "txt", "jsonl", "parquet", "csv", etc.
        num_examples: Number of examples in dataset (if known)
        size_bytes: Total size in bytes
        estimated_tokens: Estimated total tokens (if computed)
        quality_score: Optional quality score 0.0-1.0
        created_at: When this metadata was created
        last_used: When this dataset was last used for training
        used_by_experts: List of expert IDs that were trained on this dataset
        tags: Additional searchable tags
        is_available: Whether dataset is currently accessible
        custom_metadata: Any additional custom fields
    """
    
    dataset_id: str
    name: str
    description: str
    source_type: str  # "local", "huggingface", "url", "custom"
    source_path: str
    domain: str = "general"
    categories: List[str] = field(default_factory=list)
    file_format: str = "txt"
    num_examples: Optional[int] = None
    size_bytes: Optional[int] = None
    estimated_tokens: Optional[int] = None
    quality_score: Optional[float] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: Optional[str] = None
    used_by_experts: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    is_available: bool = True
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetMetadata":
        """Create DatasetMetadata from dictionary."""
        return cls(**data)
    
    def mark_used(self, expert_id: str) -> None:
        """Mark this dataset as used by an expert."""
        if expert_id not in self.used_by_experts:
            self.used_by_experts.append(expert_id)
        self.last_used = datetime.now().isoformat()
    
    def matches_domain(self, domain: str) -> bool:
        """Check if dataset matches a domain."""
        return (
            self.domain.lower() == domain.lower() or
            domain.lower() in [cat.lower() for cat in self.categories]
        )
    
    def matches_tags(self, tags: List[str]) -> bool:
        """Check if dataset matches any of the given tags."""
        dataset_tags_lower = set(tag.lower() for tag in self.tags)
        search_tags_lower = set(tag.lower() for tag in tags)
        return bool(dataset_tags_lower & search_tags_lower)


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
            del self.datasets[dataset_id]
            logger.info(f"[DatasetRegistry] Removed dataset: {dataset_id}")
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
        data = {
            "datasets": {
                dataset_id: metadata.to_dict()
                for dataset_id, metadata in self.datasets.items()
            }
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[DatasetRegistry] Saved {len(self.datasets)} datasets to {path}")
    
    @classmethod
    def load(cls, path: str) -> "DatasetRegistry":
        """
        Load registry from JSON file.
        
        Args:
            path: Path to JSON file
        
        Returns:
            DatasetRegistry instance
        """
        registry = cls()
        
        if not Path(path).exists():
            logger.warning(f"[DatasetRegistry] File not found: {path}")
            return registry
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for dataset_id, dataset_data in data.get("datasets", {}).items():
            metadata = DatasetMetadata.from_dict(dataset_data)
            registry.datasets[dataset_id] = metadata
        
        logger.info(f"[DatasetRegistry] Loaded {len(registry.datasets)} datasets from {path}")
        return registry


class LocalDatasetScanner:
    """
    Scans local directories for datasets and creates metadata.
    """
    
    SUPPORTED_FORMATS = {
        '.txt': 'txt',
        '.jsonl': 'jsonl',
        '.json': 'json',
        '.parquet': 'parquet',
        '.csv': 'csv',
        '.tsv': 'tsv',
    }
    
    @staticmethod
    def scan_directory(
        directory: str,
        recursive: bool = True,
        auto_categorize: bool = True
    ) -> List[DatasetMetadata]:
        """
        Scan a directory for dataset files.
        
        Args:
            directory: Directory path to scan
            recursive: Whether to scan subdirectories
            auto_categorize: Attempt to infer domain from filename/path
        
        Returns:
            List of DatasetMetadata objects
        """
        datasets = []
        
        if not Path(directory).exists():
            logger.warning(f"[LocalScanner] Directory not found: {directory}")
            return datasets
        
        # Build search pattern
        if recursive:
            patterns = [
                os.path.join(directory, '**', f'*{ext}')
                for ext in LocalDatasetScanner.SUPPORTED_FORMATS.keys()
            ]
        else:
            patterns = [
                os.path.join(directory, f'*{ext}')
                for ext in LocalDatasetScanner.SUPPORTED_FORMATS.keys()
            ]
        
        # Find all matching files
        found_files = []
        for pattern in patterns:
            found_files.extend(glob.glob(pattern, recursive=recursive))
        
        logger.info(f"[LocalScanner] Found {len(found_files)} dataset files in {directory}")
        
        # Create metadata for each file
        for filepath in found_files:
            try:
                metadata = LocalDatasetScanner._create_metadata_from_file(
                    filepath,
                    auto_categorize=auto_categorize
                )
                datasets.append(metadata)
            except Exception as e:
                logger.error(f"[LocalScanner] Error processing {filepath}: {e}")
        
        return datasets
    
    @staticmethod
    def _create_metadata_from_file(
        filepath: str,
        auto_categorize: bool = True
    ) -> DatasetMetadata:
        """Create DatasetMetadata from a local file."""
        path = Path(filepath)
        
        # Basic metadata
        dataset_id = f"local_{path.stem}_{hash(filepath) % 10000:04d}"
        name = path.stem.replace('_', ' ').replace('-', ' ').title()
        file_format = LocalDatasetScanner.SUPPORTED_FORMATS.get(path.suffix.lower(), 'unknown')
        size_bytes = path.stat().st_size
        
        # Auto-categorize from filename/path
        domain = "general"
        categories = []
        tags = []
        
        if auto_categorize:
            filepath_lower = filepath.lower()
            
            # Domain detection
            if any(kw in filepath_lower for kw in ['python', 'code', 'programming', 'script']):
                domain = "coding"
                categories.extend(["python", "programming"])
                tags.extend(["code", "python"])
            elif any(kw in filepath_lower for kw in ['math', 'equation', 'calcul', 'algebra']):
                domain = "math"
                categories.append("mathematics")
                tags.extend(["math", "equations"])
            elif any(kw in filepath_lower for kw in ['write', 'story', 'creative', 'literature']):
                domain = "writing"
                categories.append("creative_writing")
                tags.extend(["writing", "stories"])
            elif any(kw in filepath_lower for kw in ['science', 'physics', 'chemistry', 'biology']):
                domain = "science"
                categories.append("science")
                tags.append("science")
            elif any(kw in filepath_lower for kw in ['wiki', 'encyclopedia', 'knowledge']):
                domain = "general"
                categories.append("knowledge")
                tags.append("general_knowledge")
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        estimated_tokens = size_bytes // 4 if size_bytes else None
        
        # Create metadata
        return DatasetMetadata(
            dataset_id=dataset_id,
            name=name,
            description=f"Local dataset from {path.name}",
            source_type="local",
            source_path=str(path.absolute()),
            domain=domain,
            categories=categories,
            file_format=file_format,
            size_bytes=size_bytes,
            estimated_tokens=estimated_tokens,
            tags=tags,
            is_available=path.exists()
        )


def create_dataset_metadata(
    dataset_id: str,
    name: str,
    source_path: str,
    description: str = "",
    source_type: str = "local",
    domain: str = "general",
    **kwargs
) -> DatasetMetadata:
    """
    Helper function to create DatasetMetadata with defaults.
    
    Args:
        dataset_id: Unique dataset identifier
        name: Human-readable name
        source_path: Path or identifier
        description: Dataset description (defaults to empty string)
        source_type: "local", "huggingface", "url", "custom"
        domain: Primary domain
        **kwargs: Additional metadata fields
    
    Returns:
        DatasetMetadata instance
    """
    if not description:
        description = f"{name} dataset"
    
    return DatasetMetadata(
        dataset_id=dataset_id,
        name=name,
        description=description,
        source_path=source_path,
        source_type=source_type,
        domain=domain,
        **kwargs
    )


# ============================================================================
# EMBEDDED TESTS
# ============================================================================

if __name__ == "__main__":
    import tempfile
    import shutil
    
    print("\nTesting DatasetRegistry...")
    print("="*70)
    
    # Create temp directory for testing
    temp_dir = tempfile.mkdtemp(prefix="dataset_test_")
    print(f"\n[Setup] Created temp directory: {temp_dir}")
    
    try:
        # ===== Test 1: Create DatasetMetadata =====
        print("\n[Test 1] Creating DatasetMetadata...")
        
        metadata1 = create_dataset_metadata(
            dataset_id="test_python_001",
            name="Python Code Examples",
            source_path="/data/python_code.txt",
            source_type="local",
            domain="coding",
            categories=["python", "programming"],
            tags=["code", "python", "examples"],
            size_bytes=1024000,
            estimated_tokens=256000
        )
        
        print(f"[OK] Created dataset: {metadata1.name}")
        print(f"     ID: {metadata1.dataset_id}")
        print(f"     Domain: {metadata1.domain}")
        print(f"     Categories: {metadata1.categories}")
        print(f"     Tags: {metadata1.tags}")
        
        # ===== Test 2: DatasetRegistry CRUD =====
        print("\n[Test 2] Testing DatasetRegistry CRUD...")
        
        registry = DatasetRegistry()
        
        # Add datasets
        registry.add_dataset(metadata1)
        
        metadata2 = create_dataset_metadata(
            dataset_id="test_math_001",
            name="Math Problems",
            source_path="/data/math.jsonl",
            source_type="local",
            domain="math",
            categories=["algebra", "calculus"],
            tags=["math", "equations"]
        )
        registry.add_dataset(metadata2)
        
        metadata3 = create_dataset_metadata(
            dataset_id="test_writing_001",
            name="Creative Stories",
            source_path="hf://stories/dataset",
            source_type="huggingface",
            domain="writing",
            categories=["creative_writing"],
            tags=["stories", "fiction"]
        )
        registry.add_dataset(metadata3)
        
        print(f"[OK] Added 3 datasets to registry")
        print(f"     Total datasets: {len(registry.get_all_datasets())}")
        
        # Get dataset
        retrieved = registry.get_dataset("test_python_001")
        print(f"[OK] Retrieved dataset: {retrieved.name if retrieved else 'None'}")
        
        # ===== Test 3: Search by Domain =====
        print("\n[Test 3] Searching by domain...")
        
        coding_datasets = registry.search_by_domain("coding")
        print(f"[OK] Found {len(coding_datasets)} coding datasets")
        for ds in coding_datasets:
            print(f"     - {ds.name} ({ds.domain})")
        
        # ===== Test 4: Search by Tags =====
        print("\n[Test 4] Searching by tags...")
        
        code_datasets = registry.search_by_tags(["code", "python"])
        print(f"[OK] Found {len(code_datasets)} datasets with code/python tags")
        for ds in code_datasets:
            print(f"     - {ds.name} (tags: {ds.tags})")
        
        # ===== Test 5: Recommend for Expert =====
        print("\n[Test 5] Recommending datasets for expert...")
        
        recommendations = registry.recommend_for_expert(
            domain="coding",
            categories=["python"],
            max_results=3
        )
        print(f"[OK] Got {len(recommendations)} recommendations for Python expert")
        for i, ds in enumerate(recommendations, 1):
            print(f"     {i}. {ds.name} ({ds.domain})")
        
        # ===== Test 6: Mark Dataset as Used =====
        print("\n[Test 6] Marking dataset as used...")
        
        metadata1.mark_used("expert_python_001")
        metadata1.mark_used("expert_general_001")
        print(f"[OK] Dataset used by {len(metadata1.used_by_experts)} experts")
        print(f"     Experts: {metadata1.used_by_experts}")
        print(f"     Last used: {metadata1.last_used}")
        
        # ===== Test 7: Save and Load Registry =====
        print("\n[Test 7] Testing save/load...")
        
        registry_path = os.path.join(temp_dir, "dataset_registry.json")
        registry.save(registry_path)
        print(f"[OK] Saved registry to {registry_path}")
        
        loaded_registry = DatasetRegistry.load(registry_path)
        print(f"[OK] Loaded registry with {len(loaded_registry.get_all_datasets())} datasets")
        
        # Verify loaded data
        loaded_metadata = loaded_registry.get_dataset("test_python_001")
        if loaded_metadata:
            print(f"[OK] Verified loaded dataset: {loaded_metadata.name}")
            print(f"     Used by experts: {loaded_metadata.used_by_experts}")
        
        # ===== Test 8: Local Dataset Scanner =====
        print("\n[Test 8] Testing LocalDatasetScanner...")
        
        # Create test dataset files
        test_data_dir = os.path.join(temp_dir, "datasets")
        os.makedirs(test_data_dir, exist_ok=True)
        
        test_files = [
            "python_code_examples.txt",
            "math_problems.jsonl",
            "creative_writing.txt",
            "general_knowledge.csv"
        ]
        
        for filename in test_files:
            filepath = os.path.join(test_data_dir, filename)
            with open(filepath, 'w') as f:
                f.write("Sample data for testing\n" * 100)
        
        print(f"[OK] Created {len(test_files)} test dataset files")
        
        # Scan directory
        scanner = LocalDatasetScanner()
        discovered = scanner.scan_directory(test_data_dir, recursive=True)
        
        print(f"[OK] Discovered {len(discovered)} datasets")
        for ds in discovered:
            print(f"     - {ds.name}")
            print(f"       Domain: {ds.domain}, Categories: {ds.categories}")
            print(f"       Size: {ds.size_bytes} bytes, Est. tokens: {ds.estimated_tokens}")
        
        # ===== Test 9: Filter by Size =====
        print("\n[Test 9] Filtering by size...")
        
        # Add discovered datasets to registry
        for ds in discovered:
            loaded_registry.add_dataset(ds)
        
        large_datasets = loaded_registry.filter_by_size(min_bytes=1000)
        print(f"[OK] Found {len(large_datasets)} datasets >= 1000 bytes")
        
        # ===== Test 10: Search by Expert =====
        print("\n[Test 10] Searching by expert...")
        
        expert_datasets = loaded_registry.search_by_expert("expert_python_001")
        print(f"[OK] Found {len(expert_datasets)} datasets used by expert_python_001")
        for ds in expert_datasets:
            print(f"     - {ds.name}")
        
        print("\n" + "="*70)
        print("[SUCCESS] All DatasetRegistry tests passed!")
        print("="*70)
        
    finally:
        # Cleanup
        print(f"\n[Cleanup] Removing temp directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
