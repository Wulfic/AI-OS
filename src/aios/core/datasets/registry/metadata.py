"""
Dataset metadata dataclass for the registry system.

Defines the DatasetMetadata structure used to track dataset information.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime


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
