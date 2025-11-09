"""Expert metadata system for dynamic subbrains.

This module provides data structures and utilities for managing expert metadata
in the dynamic subbrains architecture.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import os
from pathlib import Path


@dataclass
class ExpertMetadata:
    """Metadata for a single expert/subbrain.
    
    This stores all information about an expert including its configuration,
    performance metrics, usage statistics, and relationships to other experts.
    """
    
    expert_id: str
    """Unique identifier (UUID)."""
    
    name: str
    """Human-readable name (e.g., 'Python Programming')."""
    
    description: str
    """Detailed description of expert's specialization."""
    
    category: str
    """Category for organization (e.g., 'Programming', 'Mathematics')."""
    
    goals: List[str] = field(default_factory=list)
    """List of goal IDs from goals system that should activate this expert."""
    
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    """When this expert was created (ISO format)."""
    
    last_trained: Optional[str] = None
    """Last training session timestamp (ISO format)."""
    
    training_datasets: List[str] = field(default_factory=list)
    """Paths/URLs of datasets used for training."""
    
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    """Metrics like validation loss, accuracy, perplexity."""
    
    total_activations: int = 0
    """How many times this expert has been routed to."""
    
    avg_routing_weight: float = 0.0
    """Average routing probability when active."""
    
    is_active: bool = False
    """Whether this expert is currently available for routing."""
    
    is_frozen: bool = False
    """Whether this expert's weights are frozen."""
    
    parent_expert_id: Optional[str] = None
    """ID of parent expert (for recursive submodels)."""
    
    child_expert_ids: List[str] = field(default_factory=list)
    """IDs of child sub-experts."""
    
    model_architecture: str = "feedforward"
    """Type: 'feedforward', 'moe', 'lora_adapted'."""
    
    checkpoint_path: str = ""
    """Relative path to expert weights file."""
    
    training_config: Dict[str, Any] = field(default_factory=dict)
    """Config used during training (learning rate, epochs, etc.)."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpertMetadata":
        """Create from dictionary (JSON deserialization)."""
        return cls(**data)
    
    def update_activation_stats(self, routing_weight: float) -> None:
        """Update usage statistics after expert activation.
        
        Args:
            routing_weight: The routing probability for this activation
        """
        self.total_activations += 1
        
        # Update running average of routing weight
        # New average = (old_avg * (n-1) + new_value) / n
        n = self.total_activations
        self.avg_routing_weight = (
            (self.avg_routing_weight * (n - 1) + routing_weight) / n
        )


@dataclass
class ExpertRegistry:
    """Registry managing all experts in a model.
    
    This provides CRUD operations and persistence for expert metadata.
    """
    
    version: str = "1.0"
    """Schema version for compatibility."""
    
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    """When this registry was created."""
    
    base_model: str = "HRM-ACT-v1-87M"
    """Base model identifier."""
    
    experts: List[ExpertMetadata] = field(default_factory=list)
    """List of all experts."""
    
    def __post_init__(self):
        """Initialize expert lookup dict."""
        self._expert_dict: Dict[str, ExpertMetadata] = {
            expert.expert_id: expert for expert in self.experts
        }
    
    def add_expert(self, expert: ExpertMetadata) -> None:
        """Add a new expert to the registry.
        
        Args:
            expert: Expert metadata to add
            
        Raises:
            ValueError: If expert_id already exists
        """
        if expert.expert_id in self._expert_dict:
            raise ValueError(f"Expert {expert.expert_id} already exists")
        
        self.experts.append(expert)
        self._expert_dict[expert.expert_id] = expert
    
    def remove_expert(self, expert_id: str) -> ExpertMetadata:
        """Remove an expert from the registry.
        
        Args:
            expert_id: ID of expert to remove
            
        Returns:
            The removed expert metadata
            
        Raises:
            KeyError: If expert_id not found
        """
        if expert_id not in self._expert_dict:
            raise KeyError(f"Expert {expert_id} not found")
        
        expert = self._expert_dict.pop(expert_id)
        self.experts.remove(expert)
        
        return expert
    
    def get_expert(self, expert_id: str) -> Optional[ExpertMetadata]:
        """Get expert metadata by ID.
        
        Args:
            expert_id: Expert ID to lookup
            
        Returns:
            Expert metadata or None if not found
        """
        return self._expert_dict.get(expert_id)
    
    def get_active_experts(self) -> List[ExpertMetadata]:
        """Get all active experts.
        
        Returns:
            List of active expert metadata
        """
        return [e for e in self.experts if e.is_active]
    
    def get_experts_by_goal(self, goal_id: str) -> List[ExpertMetadata]:
        """Get all experts linked to a specific goal.
        
        Args:
            goal_id: Goal ID to search for
            
        Returns:
            List of experts linked to this goal
        """
        return [e for e in self.experts if goal_id in e.goals]
    
    def get_experts_by_category(self, category: str) -> List[ExpertMetadata]:
        """Get all experts in a category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of experts in this category
        """
        return [e for e in self.experts if e.category == category]
    
    def get_child_experts(self, parent_id: str) -> List[ExpertMetadata]:
        """Get all child experts of a parent.
        
        Args:
            parent_id: Parent expert ID
            
        Returns:
            List of child experts
        """
        return [e for e in self.experts if e.parent_expert_id == parent_id]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "base_model": self.base_model,
            "experts": [expert.to_dict() for expert in self.experts],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpertRegistry":
        """Create from dictionary (JSON deserialization).
        
        Args:
            data: Dictionary from JSON
            
        Returns:
            ExpertRegistry instance
        """
        experts = [
            ExpertMetadata.from_dict(expert_data)
            for expert_data in data.get("experts", [])
        ]
        
        return cls(
            version=data.get("version", "1.0"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            base_model=data.get("base_model", "HRM-ACT-v1-87M"),
            experts=experts,
        )
    
    def save(self, registry_path: str) -> None:
        """Save registry to JSON file.
        
        Args:
            registry_path: Path to save JSON file
        """
        # Create directory if needed
        dir_path = os.path.dirname(registry_path)
        if dir_path:  # Only create if there's a directory component
            os.makedirs(dir_path, exist_ok=True)
        
        # Write JSON with pretty formatting
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, registry_path: str) -> "ExpertRegistry":
        """Load registry from JSON file.
        
        Args:
            registry_path: Path to JSON file
            
        Returns:
            ExpertRegistry instance
            
        Raises:
            FileNotFoundError: If registry file doesn't exist
        """
        if not os.path.exists(registry_path):
            raise FileNotFoundError(f"Registry not found: {registry_path}")
        
        with open(registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def create_empty(cls, base_model: str = "HRM-ACT-v1-87M") -> "ExpertRegistry":
        """Create a new empty registry.
        
        Args:
            base_model: Base model identifier
            
        Returns:
            Empty ExpertRegistry
        """
        return cls(base_model=base_model, experts=[])


def create_expert_metadata(
    expert_id: str,
    name: str,
    description: str,
    category: str,
    goals: Optional[List[str]] = None,
    parent_expert_id: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    is_active: bool = True,
) -> ExpertMetadata:
    """Helper function to create expert metadata with defaults.
    
    Args:
        expert_id: Unique expert ID (typically UUID)
        name: Human-readable name
        description: Detailed description
        category: Category for organization
        goals: List of goal IDs (optional)
        parent_expert_id: Parent expert ID for recursive experts (optional)
        checkpoint_path: Path to checkpoint file (optional)
        is_active: Whether expert is active (default: True)
        
    Returns:
        ExpertMetadata instance
    """
    if checkpoint_path is None:
        checkpoint_path = f"experts/expert_{expert_id}.pt"
    
    return ExpertMetadata(
        expert_id=expert_id,
        name=name,
        description=description,
        category=category,
        goals=goals or [],
        parent_expert_id=parent_expert_id,
        checkpoint_path=checkpoint_path,
        is_active=is_active,
    )


# Example usage and testing
if __name__ == "__main__":
    import uuid
    
    # Create registry
    registry = ExpertRegistry.create_empty()
    
    # Create some experts
    expert1 = create_expert_metadata(
        expert_id=str(uuid.uuid4()),
        name="Python Programming",
        description="Expert in Python syntax, libraries, and best practices",
        category="Programming",
        goals=["learn_python", "write_code"],
    )
    
    expert2 = create_expert_metadata(
        expert_id=str(uuid.uuid4()),
        name="Django Framework",
        description="Specialized in Django web framework",
        category="Programming",
        parent_expert_id=expert1.expert_id,
        goals=["build_web_app"],
    )
    
    # Add to registry
    registry.add_expert(expert1)
    registry.add_expert(expert2)
    
    # Update expert 1 child list
    expert1.child_expert_ids.append(expert2.expert_id)
    
    # Save to file
    test_path = "test_expert_registry.json"
    registry.save(test_path)
    print(f"✅ Saved registry to {test_path}")
    
    # Load back
    loaded_registry = ExpertRegistry.load(test_path)
    print(f"✅ Loaded registry with {len(loaded_registry.experts)} experts")
    
    # Verify
    assert len(loaded_registry.experts) == 2
    loaded_expert = loaded_registry.get_expert(expert1.expert_id)
    assert loaded_expert is not None
    assert loaded_expert.name == "Python Programming"
    
    # Clean up
    os.remove(test_path)
    print(f"✅ Test passed!")
