"""
Auto-training orchestrator for coordinating expert creation and training.

Main coordinator that detects intent, finds datasets, creates experts, and manages training.
"""

import re
import uuid
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
import json

from .models import LearningIntent, TrainingTask, TrainingTaskStatus
from .intent_detector import IntentDetector
from .exceptions import NoDatasetFoundError


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
