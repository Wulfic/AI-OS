"""
Local dataset scanner for automatic discovery.

Scans directories for dataset files and creates metadata entries.
"""

from typing import List
from pathlib import Path
import os
import glob
import logging

from .metadata import DatasetMetadata

logger = logging.getLogger(__name__)


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
