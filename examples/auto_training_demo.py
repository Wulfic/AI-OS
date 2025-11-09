"""Integration example for Auto-Training Orchestrator.

This example demonstrates the complete auto-training workflow:
1. User says "Learn Python programming"
2. System detects intent
3. Finds appropriate dataset
4. Creates expert automatically
5. Starts background training
6. Tracks progress

This is a working example that can be run directly.
"""

from pathlib import Path
import tempfile
import time

# Import our components
from aios.core.auto_training import (
    AutoTrainingOrchestrator,
    IntentDetector,
    TrainingTaskStatus,
)
from aios.core.datasets.registry import DatasetRegistry, create_dataset_metadata
from aios.core.hrm_models.expert_metadata import ExpertRegistry


def demo_auto_training():
    """Demonstrate the complete auto-training workflow."""
    
    print("=" * 80)
    print("Auto-Training Orchestrator - Integration Demo")
    print("=" * 80)
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # ========================================================================
        # Step 1: Setup - Create registries and sample data
        # ========================================================================
        print("\n[Step 1] Setting up registries and sample data...")
        
        # Create dataset registry with sample datasets
        dataset_registry = DatasetRegistry()
        
        # Add Python dataset
        python_dataset = create_dataset_metadata(
            dataset_id="python_code_001",
            name="Python Code Examples",
            source_path=str(tmpdir / "python_examples.txt"),
            description="Collection of Python code examples and tutorials",
            domain="coding",
            categories=["python", "programming"],
            tags=["python", "code", "tutorial"],
            size_bytes=1_000_000,
            num_examples=5000,
        )
        dataset_registry.add_dataset(python_dataset)
        
        # Add Math dataset
        math_dataset = create_dataset_metadata(
            dataset_id="calculus_problems_001",
            name="Calculus Problem Set",
            source_path=str(tmpdir / "calculus.txt"),
            description="Calculus problems and solutions",
            domain="math",
            categories=["calculus"],
            tags=["math", "calculus"],
            size_bytes=500_000,
            num_examples=2000,
        )
        dataset_registry.add_dataset(math_dataset)
        
        # Create expert registry
        expert_registry = ExpertRegistry()
        
        print(f"   ✓ Created dataset registry with {len(dataset_registry.get_all_datasets())} datasets")
        print(f"   ✓ Created expert registry")
        
        # ========================================================================
        # Step 2: Create orchestrator
        # ========================================================================
        print("\n[Step 2] Creating Auto-Training Orchestrator...")
        
        orchestrator = AutoTrainingOrchestrator(
            dataset_registry=dataset_registry,
            expert_registry=expert_registry,
            min_confidence=0.5,
            tasks_file=tmpdir / "training_tasks.json",
        )
        
        # Add callbacks to show what's happening
        def on_task_created(task):
            print(f"\n   🎯 Task Created!")
            print(f"      Task ID: {task.task_id}")
            print(f"      Expert: {task.expert_id}")
            print(f"      Dataset: {task.dataset_id}")
        
        def on_task_started(task):
            print(f"\n   🚀 Training Started!")
            print(f"      Expert: {task.expert_id}")
        
        def on_task_completed(task):
            print(f"\n   ✅ Training Completed!")
            print(f"      Expert: {task.expert_id}")
            print(f"      Duration: {(task.completed_at - task.started_at).total_seconds():.1f}s")
        
        orchestrator.on_task_created = on_task_created
        orchestrator.on_task_started = on_task_started
        orchestrator.on_task_completed = on_task_completed
        
        print(f"   ✓ Orchestrator initialized")
        
        # ========================================================================
        # Step 3: User says "Learn Python programming"
        # ========================================================================
        print("\n[Step 3] User input: 'Learn Python programming'")
        
        user_message = "Learn Python programming"
        
        # Detect intent first (for demonstration)
        detector = IntentDetector()
        intent = detector.detect(user_message)
        
        if intent:
            print(f"\n   Intent Detected:")
            print(f"      Domain: {intent.domain}")
            print(f"      Categories: {intent.categories}")
            print(f"      Description: {intent.description}")
            print(f"      Confidence: {intent.confidence:.2%}")
        
        # ========================================================================
        # Step 4: Create learning task (auto-starts training)
        # ========================================================================
        print("\n[Step 4] Creating learning task...")
        
        task = orchestrator.create_learning_task(
            user_message,
            auto_start=True,  # Start training automatically
        )
        
        if not task:
            print("   ⚠️  No learning intent detected")
            return
        
        # ========================================================================
        # Step 5: Monitor progress
        # ========================================================================
        print("\n[Step 5] Monitoring training progress...")
        print("   (Note: This is simulated training for demo purposes)")
        
        # Poll for updates
        last_progress = 0.0
        while True:
            status = orchestrator.get_task_status(task.task_id)
            
            if status is None:
                break
            
            if status.progress != last_progress:
                print(f"   Progress: {status.progress * 100:.0f}% | Status: {status.status.value}")
                last_progress = status.progress
            
            if status.status in [TrainingTaskStatus.COMPLETED, TrainingTaskStatus.FAILED]:
                break
            
            time.sleep(0.2)
        
        # ========================================================================
        # Step 6: Check results
        # ========================================================================
        print("\n[Step 6] Checking results...")
        
        final_status = orchestrator.get_task_status(task.task_id)
        
        if final_status and final_status.status == TrainingTaskStatus.COMPLETED:
            print(f"\n   ✅ Training Successful!")
            print(f"      Expert ID: {final_status.expert_id}")
            print(f"      Dataset: {final_status.dataset_id}")
            print(f"      Final Progress: {final_status.progress * 100:.0f}%")
            
            # Check expert registry
            expert = expert_registry.get_expert(final_status.expert_id)
            if expert:
                print(f"\n   Expert Details:")
                print(f"      Name: {expert.name}")
                print(f"      Category: {expert.category}")
                print(f"      Goals: {expert.goals}")
                print(f"      Active: {expert.is_active}")
        elif final_status:
            print(f"\n   ❌ Training Failed")
            print(f"      Error: {final_status.error_message}")
        
        # ========================================================================
        # Step 7: Demonstrate multiple tasks
        # ========================================================================
        print("\n[Step 7] Creating another task (Math)...")
        
        task2 = orchestrator.create_learning_task(
            "I want to understand calculus",
            auto_start=True,
        )
        
        if task2:
            print(f"   ✓ Created task for: {task2.intent.description}")
            
            # Wait for completion
            while True:
                status = orchestrator.get_task_status(task2.task_id)
                if status and status.status in [TrainingTaskStatus.COMPLETED, TrainingTaskStatus.FAILED]:
                    break
                time.sleep(0.2)
        
        # ========================================================================
        # Step 8: Show all tasks
        # ========================================================================
        print("\n[Step 8] All training tasks:")
        
        all_tasks = orchestrator.get_all_tasks()
        for i, t in enumerate(all_tasks, 1):
            print(f"\n   Task {i}:")
            print(f"      ID: {t.task_id}")
            print(f"      Expert: {t.expert_id}")
            print(f"      Intent: {t.intent.description}")
            print(f"      Dataset: {t.dataset_id}")
            print(f"      Status: {t.status.value}")
            print(f"      Progress: {t.progress * 100:.0f}%")
        
        # ========================================================================
        # Step 9: Show expert registry
        # ========================================================================
        print("\n[Step 9] Expert Registry:")
        
        all_experts = expert_registry.get_active_experts()  # Get all active experts
        print(f"\n   Total Experts: {len(all_experts)}")
        for expert in all_experts:
            print(f"\n   - {expert.name}")
            print(f"     ID: {expert.expert_id}")
            print(f"     Category: {expert.category}")
            print(f"     Goals: {', '.join(expert.goals)}")
            print(f"     Active: {expert.is_active}")
        
        # ========================================================================
        # Step 10: Verify persistence
        # ========================================================================
        print("\n[Step 10] Testing persistence...")
        
        tasks_file = tmpdir / "training_tasks.json"
        print(f"   Tasks saved to: {tasks_file}")
        print(f"   File exists: {tasks_file.exists()}")
        
        if tasks_file.exists():
            # Load in new orchestrator
            orchestrator2 = AutoTrainingOrchestrator(
                dataset_registry=dataset_registry,
                expert_registry=expert_registry,
                tasks_file=tasks_file,
            )
            
            loaded_tasks = orchestrator2.get_all_tasks()
            print(f"   ✓ Loaded {len(loaded_tasks)} tasks from disk")
        
        print("\n" + "=" * 80)
        print("Demo Complete!")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("  • User says 'Learn Python' → system automatically trains expert")
        print("  • Intent detection works with natural language")
        print("  • Dataset discovery finds best match automatically")
        print("  • Expert creation and registration fully automated")
        print("  • Background training with progress tracking")
        print("  • Full persistence - survives restarts")
        print("  • Ready for chat integration!")
        print("=" * 80)


if __name__ == "__main__":
    demo_auto_training()
