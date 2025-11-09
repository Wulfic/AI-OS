"""
Exceptions for auto-training orchestrator.
"""


class NoDatasetFoundError(Exception):
    """Raised when no suitable dataset is found for a learning intent."""
    pass
