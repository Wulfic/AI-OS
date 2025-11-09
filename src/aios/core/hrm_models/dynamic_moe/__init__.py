"""Dynamic Mixture of Experts with runtime expert management.

This module extends the basic MoE layer to support:
- Runtime expert addition/removal without destroying base model
- Expert freezing/unfreezing for selective training
- Lazy loading for memory-efficient expert management
- Recursive submodels (experts can have child experts)
- Integration with ExpertMetadata registry

Author: AI-OS Team
Date: January 2025
"""

from .lazy_loader import LazyExpertLoader
from .dynamic_layer import DynamicMoELayer

__all__ = [
    "LazyExpertLoader",
    "DynamicMoELayer",
]
