"""Dataset utilities for AI-OS.

This module provides comprehensive dataset management including:
- Storage configuration and capacity management
- Known dataset catalog
- Text file reading from various formats
- Archive extraction (ZIP, TAR, GZIP, etc.)
- HuggingFace dataset integration with streaming
- CSV reading with text/label extraction
"""

from __future__ import annotations

# Re-export all public APIs for backward compatibility
from .storage import (
    datasets_base_dir,
    datasets_storage_usage_gb,
    datasets_storage_cap_gb,
    set_datasets_storage_cap_gb,
    can_store_additional_gb,
)
from .catalog import KnownDataset, known_datasets
from .readers import read_text_lines_sample
from .advanced_readers import read_text_lines_sample_any
from .csv_readers import read_csv_text_label_samples

__all__ = [
    # Storage management
    "datasets_base_dir",
    "datasets_storage_usage_gb",
    "datasets_storage_cap_gb",
    "set_datasets_storage_cap_gb",
    "can_store_additional_gb",
    # Catalog
    "KnownDataset",
    "known_datasets",
    # Readers
    "read_text_lines_sample",
    "read_text_lines_sample_any",
    "read_csv_text_label_samples",
]
