"""
Helper functions for creating dataset metadata.
"""

from .metadata import DatasetMetadata


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
