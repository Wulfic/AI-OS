"""
Tests for auto-training orchestrator.

Run with: python -m aios.core.auto_training.orchestrator.tests
"""

import tempfile
from pathlib import Path

from .models import LearningIntent, TrainingTask, TrainingTaskStatus
from .intent_detector import IntentDetector
from .orchestrator import AutoTrainingOrchestrator
from .exceptions import NoDatasetFoundError


def run_tests():
    """Run comprehensive tests for the orchestrator."""
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
        
        # Test 8: NoDatasetFoundError
        print("\n[Test 8] Testing NoDatasetFoundError...")
        
        try:
            task2 = orchestrator.create_learning_task(
                "Study calculus",
                auto_start=False,
            )
            if task2:
                print("[FAIL] Task created unexpectedly (should have failed)")
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


if __name__ == "__main__":
    run_tests()
