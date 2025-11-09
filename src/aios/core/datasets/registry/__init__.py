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

from .metadata import DatasetMetadata
from .registry import DatasetRegistry
from .scanner import LocalDatasetScanner
from .helpers import create_dataset_metadata

__all__ = [
    "DatasetMetadata",
    "DatasetRegistry",
    "LocalDatasetScanner",
    "create_dataset_metadata",
]
