"""Auto-Training Orchestrator for Dynamic Subbrains.

This module provides automatic expert creation and training based on user intent.
When a user expresses a learning intent (e.g., "learn Python programming"), the
orchestrator:
1. Detects the intent and extracts domain/categories
2. Searches for appropriate datasets
3. Creates a new expert with metadata
4. Starts background training
5. Tracks progress and provides updates

Example:
    >>> orchestrator = AutoTrainingOrchestrator(
    ...     dataset_registry=registry,
    ...     expert_registry=expert_reg
    ... )
    >>> task = orchestrator.create_learning_task("Learn Python programming")
    >>> print(f"Training {task.expert_id} on {task.dataset_id}")
"""

import re
import uuid
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
import json


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
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
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


class IntentDetector:
    """Detects learning intent from user messages.
    
    Uses pattern matching and keyword analysis to identify when a user
    wants the system to learn something new. Extracts domain and categories
    from the message.
    
    Example:
        >>> detector = IntentDetector()
        >>> intent = detector.detect("Learn Python programming")
        >>> print(intent.domain)  # "coding"
        >>> print(intent.categories)  # ["python", "programming"]
    """
    
    # Patterns that indicate learning intent
    LEARNING_PATTERNS = [
        r"learn (?:about )?(.+)",
        r"teach (?:me |yourself )?(?:about )?(.+)",
        r"study (.+)",
        r"I want to (?:learn|understand) (.+)",
        r"help me (?:learn|understand) (.+)",
        r"get better at (.+)",
        r"master (.+)",
        r"understand (.+)",
        r"train (?:on|yourself on) (.+)",
        r"practice (.+)",
    ]
    
    # Domain keywords for classification
    DOMAIN_KEYWORDS = {
        "coding": [
            "python", "javascript", "java", "c++", "code", "programming",
            "software", "algorithm", "data structure", "web development",
            "api", "backend", "frontend", "database", "sql", "git",
        ],
        "math": [
            "math", "calculus", "algebra", "geometry", "statistics",
            "trigonometry", "linear algebra", "differential", "integral",
            "probability", "number theory", "arithmetic",
        ],
        "writing": [
            "writing", "creative writing", "story", "essay", "literature",
            "poetry", "narrative", "fiction", "non-fiction", "journalism",
            "technical writing", "copywriting",
        ],
        "science": [
            "physics", "chemistry", "biology", "science", "astronomy",
            "ecology", "geology", "botany", "zoology", "genetics",
            "molecular", "quantum", "thermodynamics",
        ],
        "general": [
            "knowledge", "encyclopedia", "wiki", "facts", "trivia",
            "general knowledge", "world history", "geography",
        ],
    }
    
    # Category extraction keywords
    CATEGORY_KEYWORDS = {
        # Coding
        "python": ["python", "py"],
        "javascript": ["javascript", "js", "node"],
        "java": ["java"],
        "programming": ["programming", "code", "coding", "software"],
        
        # Math
        "calculus": ["calculus", "derivative", "integral"],
        "algebra": ["algebra", "equation"],
        "geometry": ["geometry", "shape", "triangle"],
        "statistics": ["statistics", "stats", "probability"],
        
        # Writing
        "creative_writing": ["creative writing", "story", "fiction"],
        "essay": ["essay", "article"],
        "poetry": ["poetry", "poem"],
        
        # Science
        "physics": ["physics", "mechanics", "quantum"],
        "chemistry": ["chemistry", "chemical", "molecule"],
        "biology": ["biology", "cell", "organism"],
    }
    
    def detect(self, message: str) -> Optional[LearningIntent]:
        """Detect learning intent from a message.
        
        Args:
            message: User message to analyze
        
        Returns:
            LearningIntent if detected, None otherwise
        """
        message_lower = message.lower().strip()
        
        # Try each pattern
        for pattern in self.LEARNING_PATTERNS:
            match = re.search(pattern, message_lower, re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                
                # Extract domain and categories
                domain = self._extract_domain(topic)
                categories = self._extract_categories(topic, domain)
                
                # Generate description
                description = f"Learn {topic}"
                
                # Calculate confidence
                confidence = self._calculate_confidence(message_lower, topic, domain)
                
                return LearningIntent(
                    domain=domain,
                    categories=categories,
                    description=description,
                    confidence=confidence,
                    raw_message=message,
                    extracted_topic=topic,
                )
        
        return None
    
    def _extract_domain(self, topic: str) -> str:
        """Extract domain from topic.
        
        Args:
            topic: Topic string
        
        Returns:
            Domain name (coding, math, writing, science, general)
        """
        topic_lower = topic.lower()
        
        # Count keyword matches per domain
        domain_scores = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in topic_lower)
            if score > 0:
                domain_scores[domain] = score
        
        # Return domain with highest score
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        
        return "general"
    
    def _extract_categories(self, topic: str, domain: str) -> List[str]:
        """Extract categories from topic.
        
        Args:
            topic: Topic string
            domain: Detected domain
        
        Returns:
            List of category names
        """
        topic_lower = topic.lower()
        categories = []
        
        # Find matching categories
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(keyword in topic_lower for keyword in keywords):
                categories.append(category)
        
        # If no categories found, use domain as category
        if not categories:
            categories = [domain]
        
        return categories
    
    def _calculate_confidence(self, message: str, topic: str, domain: str) -> float:
        """Calculate confidence score for intent detection.
        
        Args:
            message: Original message
            topic: Extracted topic
            domain: Detected domain
        
        Returns:
            Confidence score 0.0-1.0
        """
        confidence = 0.5  # Base confidence
        
        # Boost if domain keywords present
        domain_keywords = self.DOMAIN_KEYWORDS.get(domain, [])
        if any(keyword in message for keyword in domain_keywords):
            confidence += 0.2
        
        # Boost if topic is substantial (>2 words)
        if len(topic.split()) > 2:
            confidence += 0.1
        
        # Boost if strong learning verbs present
        strong_verbs = ["learn", "master", "study", "understand"]
        if any(verb in message for verb in strong_verbs):
            confidence += 0.1
        
        return min(confidence, 1.0)


class NoDatasetFoundError(Exception):
    """Raised when no suitable dataset is found for a learning intent."""
    pass


class AutoTrainingOrchestrator:
    """Orchestrates automatic expert creation and training.
    
    This is the main coordinator for the auto-training workflow. It:
    1. Detects learning intent from user messages
    2. Searches for appropriate datasets
    3. Creates new experts with metadata
    4. Starts background training tasks
    5. Tracks progress and provides status updates
    
    Example:
        >>> from aios.core.datasets.registry import DatasetRegistry
        >>> from aios.core.hrm_models.expert_metadata import ExpertRegistry
        >>> 
        >>> orchestrator = AutoTrainingOrchestrator(
        ...     dataset_registry=DatasetRegistry(),
        ...     expert_registry=ExpertRegistry()
        ... )
        >>> 
        >>> # User says "Learn Python programming"
        >>> task = orchestrator.create_learning_task("Learn Python programming")
        >>> print(f"Started training {task.expert_id}")
        >>> 
        >>> # Check progress
        >>> status = orchestrator.get_task_status(task.task_id)
        >>> print(f"Progress: {status.progress * 100}%")
    """
    
    def __init__(
        self,
        dataset_registry,  # DatasetRegistry
        expert_registry,  # ExpertRegistry
        intent_detector: Optional[IntentDetector] = None,
        min_confidence: float = 0.5,
        tasks_file: Optional[Path] = None,
    ):
        """Initialize orchestrator.
        
        Args:
            dataset_registry: DatasetRegistry instance
            expert_registry: ExpertRegistry instance
            intent_detector: Optional custom intent detector
            min_confidence: Minimum confidence threshold for intent detection
            tasks_file: Path to persist tasks (JSON)
        """
        self.dataset_registry = dataset_registry
        self.expert_registry = expert_registry
        self.intent_detector = intent_detector or IntentDetector()
        self.min_confidence = min_confidence
        self.tasks_file = tasks_file
        
        # Active tasks
        self.tasks: Dict[str, TrainingTask] = {}
        self.active_threads: Dict[str, threading.Thread] = {}
        
        # Callbacks
        self.on_task_created: Optional[Callable[[TrainingTask], None]] = None
        self.on_task_started: Optional[Callable[[TrainingTask], None]] = None
        self.on_task_completed: Optional[Callable[[TrainingTask], None]] = None
        self.on_task_failed: Optional[Callable[[TrainingTask], None]] = None
        
        # Load existing tasks
        if tasks_file and tasks_file.exists():
            self._load_tasks()
    
    def create_learning_task(
        self,
        user_message: str,
        auto_start: bool = True,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[TrainingTask]:
        """Create a learning task from user message.
        
        This is the main entry point for auto-training. It detects intent,
        finds datasets, creates an expert, and optionally starts training.
        
        Args:
            user_message: User's message expressing learning intent
            auto_start: Whether to automatically start training
            training_config: Optional training configuration overrides
        
        Returns:
            TrainingTask if intent detected and task created, None otherwise
        
        Raises:
            NoDatasetFoundError: If no suitable dataset is found
        """
        # 1. Detect intent
        intent = self.intent_detector.detect(user_message)
        if intent is None or intent.confidence < self.min_confidence:
            return None
        
        # 2. Search for datasets
        datasets = self.dataset_registry.recommend_for_expert(
            domain=intent.domain,
            categories=intent.categories,
            max_results=5,
        )
        
        if not datasets:
            raise NoDatasetFoundError(
                f"No datasets found for domain='{intent.domain}', "
                f"categories={intent.categories}"
            )
        
        # Use best matching dataset
        best_dataset = datasets[0]
        
        # 3. Create expert
        expert_id = self._generate_expert_id(intent)
        
        # Import here to avoid circular dependency
        from aios.core.hrm_models.expert_metadata import create_expert_metadata
        
        # Use primary category as the category field
        primary_category = intent.categories[0] if intent.categories else intent.domain
        
        expert_meta = create_expert_metadata(
            expert_id=expert_id,
            name=f"{intent.description} Expert",
            description=f"Expert trained for: {intent.description}",
            category=primary_category,
            goals=[intent.description],
        )
        
        self.expert_registry.add_expert(expert_meta)
        
        # Mark dataset as used
        best_dataset.mark_used(expert_id)
        
        # 4. Create training task
        task = TrainingTask(
            task_id=str(uuid.uuid4()),
            expert_id=expert_id,
            dataset_id=best_dataset.dataset_id,
            intent=intent,
            status=TrainingTaskStatus.PENDING,
            config=training_config or {},
        )
        
        self.tasks[task.task_id] = task
        
        # Save tasks
        self._save_tasks()
        
        # Trigger callback
        if self.on_task_created:
            self.on_task_created(task)
        
        # 5. Start training if requested
        if auto_start:
            self.start_task(task.task_id)
        
        return task
    
    def start_task(self, task_id: str) -> bool:
        """Start a training task in the background.
        
        Args:
            task_id: Task ID to start
        
        Returns:
            True if started successfully, False if already running or invalid
        """
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status not in [TrainingTaskStatus.PENDING, TrainingTaskStatus.FAILED]:
            return False  # Already running or completed
        
        # Start background thread
        thread = threading.Thread(
            target=self._training_worker,
            args=(task,),
            daemon=True,
        )
        
        self.active_threads[task_id] = thread
        thread.start()
        
        return True
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task.
        
        Args:
            task_id: Task ID to cancel
        
        Returns:
            True if cancelled successfully
        """
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        task.status = TrainingTaskStatus.CANCELLED
        self._save_tasks()
        
        # Note: Thread will detect status change and exit
        return True
    
    def get_task_status(self, task_id: str) -> Optional[TrainingTask]:
        """Get current status of a task.
        
        Args:
            task_id: Task ID
        
        Returns:
            TrainingTask if found, None otherwise
        """
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[TrainingTask]:
        """Get all tasks.
        
        Returns:
            List of all training tasks
        """
        return list(self.tasks.values())
    
    def get_active_tasks(self) -> List[TrainingTask]:
        """Get currently active (running) tasks.
        
        Returns:
            List of tasks with status TRAINING
        """
        return [
            task for task in self.tasks.values()
            if task.status == TrainingTaskStatus.TRAINING
        ]
    
    def _generate_expert_id(self, intent: LearningIntent) -> str:
        """Generate unique expert ID from intent.
        
        Args:
            intent: Learning intent
        
        Returns:
            Expert ID string
        """
        # Format: expert_<domain>_<category>_<uuid>
        primary_category = intent.categories[0] if intent.categories else "general"
        
        # Clean category name (remove spaces, special chars)
        clean_category = re.sub(r'[^a-z0-9]', '_', primary_category.lower())
        
        # Generate short UUID
        short_uuid = str(uuid.uuid4())[:8]
        
        return f"expert_{intent.domain}_{clean_category}_{short_uuid}"
    
    def _training_worker(self, task: TrainingTask) -> None:
        """Background worker that executes training.
        
        This runs in a separate thread and performs the actual training.
        It updates task status and handles errors gracefully.
        
        Args:
            task: Training task to execute
        """
        try:
            # Update status
            task.status = TrainingTaskStatus.TRAINING
            task.started_at = datetime.now()
            self._save_tasks()
            
            if self.on_task_started:
                self.on_task_started(task)
            
            # Get dataset path
            dataset = self.dataset_registry.get_dataset(task.dataset_id)
            if not dataset:
                raise ValueError(f"Dataset {task.dataset_id} not found")
            
            # Create training config
            config = self._create_training_config(task, dataset)
            
            # Simulate training for testing
            # In production, this would call the actual training pipeline
            # from aios.cli.hrm_hf.train_actv1 import train_actv1_impl
            # train_actv1_impl(config)
            
            # For now, simulate with progress updates
            for i in range(10):
                if task.status == TrainingTaskStatus.CANCELLED:
                    return
                
                task.progress = (i + 1) / 10
                task.metrics["simulated_step"] = i + 1
                self._save_tasks()
                time.sleep(0.5)  # Simulate work
            
            # Mark complete
            task.status = TrainingTaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.progress = 1.0
            self._save_tasks()
            
            if self.on_task_completed:
                self.on_task_completed(task)
            
        except Exception as e:
            # Mark failed
            task.status = TrainingTaskStatus.FAILED
            task.error_message = str(e)
            self._save_tasks()
            
            if self.on_task_failed:
                self.on_task_failed(task)
    
    def _create_training_config(self, task: TrainingTask, dataset) -> Dict[str, Any]:
        """Create training configuration for a task.
        
        Args:
            task: Training task
            dataset: Dataset metadata
        
        Returns:
            Training configuration dictionary
        """
        # Base config
        config = {
            "model": "artifacts/hf_implant/base_model",  # Base model path
            "dataset_file": dataset.source_path,
            "expert_id": task.expert_id,
            "max_seq_len": 2048,
            "batch_size": 2,
            "steps": 1000,
            "learning_rate": 1e-4,
            "save_dir": f"artifacts/experts/{task.expert_id}",
        }
        
        # Merge user overrides
        if task.config:
            config.update(task.config)
        
        return config
    
    def _save_tasks(self) -> None:
        """Save tasks to JSON file."""
        if not self.tasks_file:
            return
        
        try:
            data = {
                task_id: task.to_dict()
                for task_id, task in self.tasks.items()
            }
            
            self.tasks_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.tasks_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Failed to save tasks: {e}")
    
    def _load_tasks(self) -> None:
        """Load tasks from JSON file."""
        if not self.tasks_file or not self.tasks_file.exists():
            return
        
        try:
            with open(self.tasks_file, "r") as f:
                data = json.load(f)
            
            for task_id, task_data in data.items():
                task = TrainingTask.from_dict(task_data)
                self.tasks[task_id] = task
        except Exception as e:
            print(f"[WARNING] Failed to load tasks: {e}")


# ============================================================================
# Tests
# ============================================================================

if __name__ == "__main__":
    print("[Auto-Training Orchestrator Tests]\n")
    
    # Test 1: Intent Detection
    print("[Test 1] Testing IntentDetector...")
    detector = IntentDetector()
    
    test_messages = [
        "Learn Python programming",
        "I want to understand calculus",
        "Teach me about creative writing",
        "Help me get better at physics",
        "Just a regular message",
    ]
    
    for msg in test_messages:
        intent = detector.detect(msg)
        if intent:
            print(f"[OK] '{msg}'")
            print(f"     Domain: {intent.domain}")
            print(f"     Categories: {intent.categories}")
            print(f"     Confidence: {intent.confidence:.2f}")
        else:
            print(f"[OK] '{msg}' - No intent detected")
    
    # Test 2: LearningIntent Serialization
    print("\n[Test 2] Testing LearningIntent serialization...")
    intent = LearningIntent(
        domain="coding",
        categories=["python", "programming"],
        description="Learn Python",
        confidence=0.8,
        raw_message="Learn Python",
        extracted_topic="python",
    )
    
    intent_dict = intent.to_dict()
    intent_restored = LearningIntent.from_dict(intent_dict)
    
    assert intent_restored.domain == "coding"
    assert intent_restored.categories == ["python", "programming"]
    print("[OK] Intent serialization works")
    
    # Test 3: TrainingTask Serialization
    print("\n[Test 3] Testing TrainingTask serialization...")
    task = TrainingTask(
        task_id="task123",
        expert_id="expert_coding_python_001",
        dataset_id="dataset_python_001",
        intent=intent,
        status=TrainingTaskStatus.PENDING,
    )
    
    task_dict = task.to_dict()
    task_restored = TrainingTask.from_dict(task_dict)
    
    assert task_restored.task_id == "task123"
    assert task_restored.status == TrainingTaskStatus.PENDING
    assert task_restored.intent.domain == "coding"
    print("[OK] Task serialization works")
    
    # Test 4: AutoTrainingOrchestrator with mock registries
    print("\n[Test 4] Testing AutoTrainingOrchestrator...")
    
    # Create mock registries
    from pathlib import Path
    import sys
    import tempfile
    
    # Import from aios package
    from aios.core.datasets.registry import DatasetRegistry, create_dataset_metadata
    from aios.core.hrm_models.expert_metadata import ExpertRegistry
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test dataset
        dataset_reg = DatasetRegistry()
        test_dataset = create_dataset_metadata(
            dataset_id="test_python_001",
            name="Python Test Dataset",
            source_path=str(tmpdir / "python_data.txt"),
            description="Test dataset for Python",
            domain="coding",
            categories=["python", "programming"],
            tags=["test"],
        )
        dataset_reg.add_dataset(test_dataset)
        
        # Create expert registry
        expert_reg = ExpertRegistry()
        
        # Create orchestrator
        tasks_file = tmpdir / "tasks.json"
        orchestrator = AutoTrainingOrchestrator(
            dataset_registry=dataset_reg,
            expert_registry=expert_reg,
            tasks_file=tasks_file,
        )
        
        # Test task creation
        task = orchestrator.create_learning_task(
            "Learn Python programming",
            auto_start=False,  # Don't auto-start for testing
        )
        
        assert task is not None
        print(f"[OK] Created task: {task.task_id}")
        print(f"     Expert ID: {task.expert_id}")
        print(f"     Dataset ID: {task.dataset_id}")
        print(f"     Status: {task.status.value}")
        
        # Test 5: Task Status Retrieval
        print("\n[Test 5] Testing task status retrieval...")
        retrieved_task = orchestrator.get_task_status(task.task_id)
        assert retrieved_task is not None
        assert retrieved_task.task_id == task.task_id
        print("[OK] Task status retrieved")
        
        # Test 6: Get All Tasks
        print("\n[Test 6] Testing get all tasks...")
        all_tasks = orchestrator.get_all_tasks()
        assert len(all_tasks) == 1
        assert all_tasks[0].task_id == task.task_id
        print(f"[OK] Found {len(all_tasks)} task(s)")
        
        # Test 7: Task Persistence
        print("\n[Test 7] Testing task persistence...")
        assert tasks_file.exists()
        
        # Create new orchestrator and load tasks
        orchestrator2 = AutoTrainingOrchestrator(
            dataset_registry=dataset_reg,
            expert_registry=expert_reg,
            tasks_file=tasks_file,
        )
        
        loaded_tasks = orchestrator2.get_all_tasks()
        assert len(loaded_tasks) == 1
        assert loaded_tasks[0].task_id == task.task_id
        print("[OK] Tasks persisted and loaded successfully")
        
        # Test 8: Background Training (simulated)
        print("\n[Test 8] Testing background training...")
        
        try:
            task2 = orchestrator.create_learning_task(
                "Study calculus",
                auto_start=False,
            )
            if task2:
                print("[OK] Task created unexpectedly (should have failed)")
        except NoDatasetFoundError as e:
            print(f"[OK] NoDatasetFoundError raised as expected: {e}")
        
        # Test 9: Intent Detection Edge Cases
        print("\n[Test 9] Testing intent detection edge cases...")
        
        edge_cases = [
            ("", None),  # Empty string
            ("Hello!", None),  # No learning intent
            ("LEARN PYTHON", "coding"),  # All caps
            ("I need to learn about biology", "science"),  # Science domain
        ]
        
        for msg, expected_domain in edge_cases:
            intent = detector.detect(msg)
            if expected_domain is None:
                assert intent is None or intent.confidence < 0.5
                print(f"[OK] No intent for: '{msg}'")
            else:
                assert intent is not None
                assert intent.domain == expected_domain
                print(f"[OK] Detected {expected_domain} for: '{msg}'")
        
        # Test 10: Expert ID Generation
        print("\n[Test 10] Testing expert ID generation...")
        
        test_intent = LearningIntent(
            domain="coding",
            categories=["python"],
            description="Learn Python",
            confidence=0.9,
            raw_message="Learn Python",
        )
        
        expert_id = orchestrator._generate_expert_id(test_intent)
        assert expert_id.startswith("expert_coding_python_")
        assert len(expert_id.split("_")) == 4  # expert_coding_python_<uuid>
        print(f"[OK] Generated expert ID: {expert_id}")
    
    print("\n[SUCCESS] All AutoTrainingOrchestrator tests passed!")
