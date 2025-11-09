"""
Data models for auto-training orchestrator.

Defines LearningIntent, TrainingTask, and related structures.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any


@dataclass
class LearningIntent:
    """Represents a detected learning intent from user input.
    
    Attributes:
        domain: Primary domain (coding, math, writing, science, general)
        categories: Specific categories within domain (e.g., ["python", "programming"])
        description: Human-readable description of what to learn
        confidence: Confidence score 0.0-1.0
        raw_message: Original user message
        extracted_topic: Topic extracted from message
    """
    domain: str
    categories: List[str]
    description: str
    confidence: float
    raw_message: str
    extracted_topic: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningIntent":
        """Create from dictionary."""
        return cls(**data)


class TrainingTaskStatus(Enum):
    """Status of an auto-training task."""
    PENDING = "pending"
    DATASET_SEARCH = "searching_datasets"
    EXPERT_CREATION = "creating_expert"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingTask:
    """Represents an auto-training task.
    
    Tracks the complete lifecycle of automatic expert training from
    intent detection through completion.
    
    Attributes:
        task_id: Unique task identifier
        expert_id: ID of the expert being trained
        dataset_id: ID of the dataset being used
        intent: Original learning intent
        status: Current task status
        progress: Training progress 0.0-1.0
        created_at: When task was created
        started_at: When training started
        completed_at: When training completed
        error_message: Error details if failed
        metrics: Training metrics (loss, accuracy, etc.)
        config: Training configuration used
    """
    task_id: str
    expert_id: str
    dataset_id: str
    intent: LearningIntent
    status: TrainingTaskStatus
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            "task_id": self.task_id,
            "expert_id": self.expert_id,
            "dataset_id": self.dataset_id,
            "intent": self.intent.to_dict(),
            "status": self.status.value,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "metrics": self.metrics,
            "config": self.config,
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingTask":
        """Create from dictionary."""
        # Parse datetime fields
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("started_at"):
            data["started_at"] = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            data["completed_at"] = datetime.fromisoformat(data["completed_at"])
        
        # Parse enum
        data["status"] = TrainingTaskStatus(data["status"])
        
        # Parse intent
        data["intent"] = LearningIntent.from_dict(data["intent"])
        
        return cls(**data)
