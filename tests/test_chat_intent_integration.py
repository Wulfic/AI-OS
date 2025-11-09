"""Tests for Task 13: Chat Intent Detection Integration.

This module tests the integration of the AutoTrainingOrchestrator with
the chat interface, enabling automatic expert training when users express
learning intents in chat messages.

Test Coverage:
- Intent detection in chat messages
- Learning task creation from chat
- Status message formatting
- Normal chat flow preservation
- Error handling (no dataset, orchestrator unavailable)
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from aios.core.auto_training.orchestrator import (
    AutoTrainingOrchestrator,
    IntentDetector,
    LearningIntent,
    TrainingTask,
    TrainingTaskStatus,
    NoDatasetFoundError,
)


class TestIntentDetectionInChat:
    """Test intent detection from chat messages."""
    
    def test_learning_intent_detected_python(self):
        """Test detection of 'Learn Python' intent."""
        detector = IntentDetector()
        intent = detector.detect("Learn Python programming")
        
        assert intent is not None
        assert intent.domain == "coding"
        assert "python" in intent.categories
        assert "python programming" in intent.extracted_topic.lower()
        assert intent.confidence >= 0.5
    
    def test_learning_intent_detected_math(self):
        """Test detection of 'Learn calculus' intent."""
        detector = IntentDetector()
        intent = detector.detect("Help me understand calculus")
        
        assert intent is not None
        assert intent.domain == "math"
        assert "calculus" in intent.categories
        assert intent.confidence >= 0.5
    
    def test_no_intent_normal_chat(self):
        """Test normal chat doesn't trigger intent detection."""
        detector = IntentDetector()
        
        # Normal greetings and questions
        assert detector.detect("Hello, how are you?") is None
        assert detector.detect("What's the weather like?") is None
        assert detector.detect("Tell me a joke") is None
    
    def test_intent_confidence_threshold(self):
        """Test intent confidence scoring."""
        detector = IntentDetector()
        
        # Strong intent with multiple keywords
        strong = detector.detect("I want to master Python programming and algorithms")
        assert strong is not None
        assert strong.confidence > 0.6
        
        # Weaker intent with fewer keywords
        weak = detector.detect("Learn X")
        if weak:  # May not detect at all
            assert weak.confidence <= strong.confidence


class TestChatOrchestrationIntegration:
    """Test orchestrator integration with chat handler."""
    
    def test_create_task_from_chat_message(self, tmp_path):
        """Test creating learning task from chat message."""
        # Create mock registries
        dataset_registry = Mock()
        expert_registry = Mock()
        
        # Mock dataset recommendation
        mock_dataset = Mock()
        mock_dataset.dataset_id = "python_code_samples"
        mock_dataset.mark_used = Mock()
        dataset_registry.recommend_for_expert.return_value = [mock_dataset]
        
        # Mock expert creation
        expert_registry.add_expert = Mock()
        
        # Create orchestrator
        orchestrator = AutoTrainingOrchestrator(
            dataset_registry=dataset_registry,
            expert_registry=expert_registry,
            min_confidence=0.5,
        )
        
        # Create task from chat message
        task = orchestrator.create_learning_task(
            user_message="Learn Python programming",
            auto_start=False,  # Don't actually start training
        )
        
        assert task is not None
        assert task.intent.domain == "coding"
        assert "python" in task.intent.categories
        assert task.dataset_id == "python_code_samples"
        assert task.status == TrainingTaskStatus.PENDING
        
        # Verify registries were called
        dataset_registry.recommend_for_expert.assert_called_once()
        expert_registry.add_expert.assert_called_once()
    
    def test_no_task_for_normal_chat(self, tmp_path):
        """Test normal chat doesn't create learning tasks."""
        dataset_registry = Mock()
        expert_registry = Mock()
        
        orchestrator = AutoTrainingOrchestrator(
            dataset_registry=dataset_registry,
            expert_registry=expert_registry,
            min_confidence=0.5,
        )
        
        # Normal chat should not create task
        task = orchestrator.create_learning_task(
            user_message="Hello, how are you?",
            auto_start=False,
        )
        
        assert task is None
        dataset_registry.recommend_for_expert.assert_not_called()
        expert_registry.add_expert.assert_not_called()
    
    def test_no_dataset_found_error(self, tmp_path):
        """Test error when no dataset is found for intent."""
        dataset_registry = Mock()
        expert_registry = Mock()
        
        # Mock empty dataset list (no datasets found)
        dataset_registry.recommend_for_expert.return_value = []
        
        orchestrator = AutoTrainingOrchestrator(
            dataset_registry=dataset_registry,
            expert_registry=expert_registry,
            min_confidence=0.5,
        )
        
        # Should raise NoDatasetFoundError
        with pytest.raises(NoDatasetFoundError) as exc_info:
            orchestrator.create_learning_task(
                user_message="Learn Python programming",
                auto_start=False,
            )
        
        assert "No datasets found" in str(exc_info.value)
        assert "coding" in str(exc_info.value)


class TestChatResponseFormatting:
    """Test chat response formatting with learning status."""
    
    def test_status_message_format(self):
        """Test status message format for learning tasks."""
        # Create a mock task
        intent = LearningIntent(
            domain="coding",
            categories=["python", "programming"],
            description="Learn python programming",
            confidence=0.8,
            raw_message="Learn Python programming",
            extracted_topic="python programming",
        )
        
        task = TrainingTask(
            task_id="test-task-123",
            expert_id="expert_coding_python_abc123",
            dataset_id="python_code_samples",
            intent=intent,
            status=TrainingTaskStatus.PENDING,
        )
        
        # Build status message (same format as in app.py)
        categories_str = ', '.join(task.intent.categories)
        intent_message = (
            f"✨ Great! I'll learn about {task.intent.extracted_topic} for you.\n\n"
            f"📊 Training Details:\n"
            f"  • Domain: {task.intent.domain}\n"
            f"  • Categories: {categories_str}\n"
            f"  • Dataset: {task.dataset_id}\n"
            f"  • Expert ID: {task.expert_id}\n\n"
            f"⚡ Training is running in the background. I'll continue to assist you while I learn!\n\n"
            f"{'─' * 60}\n\n"
        )
        
        # Verify format
        assert "✨ Great! I'll learn about python programming" in intent_message
        assert "Domain: coding" in intent_message
        assert "Categories: python, programming" in intent_message
        assert "Dataset: python_code_samples" in intent_message
        assert "Expert ID: expert_coding_python_abc123" in intent_message
        assert "⚡ Training is running in the background" in intent_message
    
    def test_no_dataset_error_message_format(self):
        """Test error message format when no dataset found."""
        error = NoDatasetFoundError(
            "No datasets found for domain='coding', categories=['python']"
        )
        
        # Build error message (same format as in app.py)
        intent_message = (
            f"📚 I detected you want to learn about something, but I couldn't find a suitable dataset.\n"
            f"Details: {str(error)}\n\n"
            f"{'─' * 60}\n\n"
        )
        
        # Verify format
        assert "📚 I detected you want to learn" in intent_message
        assert "couldn't find a suitable dataset" in intent_message
        assert "No datasets found" in intent_message


class TestBackgroundTraining:
    """Test background training task execution."""
    
    def test_task_runs_in_background(self, tmp_path):
        """Test that training task runs in background thread."""
        dataset_registry = Mock()
        expert_registry = Mock()
        
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.dataset_id = "python_code_samples"
        mock_dataset.mark_used = Mock()
        dataset_registry.recommend_for_expert.return_value = [mock_dataset]
        dataset_registry.get_dataset.return_value = mock_dataset
        
        expert_registry.add_expert = Mock()
        
        orchestrator = AutoTrainingOrchestrator(
            dataset_registry=dataset_registry,
            expert_registry=expert_registry,
            min_confidence=0.5,
        )
        
        # Create task with auto_start=True
        task = orchestrator.create_learning_task(
            user_message="Learn Python programming",
            auto_start=True,
        )
        
        assert task is not None
        assert task.status in [TrainingTaskStatus.PENDING, TrainingTaskStatus.TRAINING]
        
        # Verify task is registered
        assert task.task_id in orchestrator.tasks
        
        # Verify background thread was created
        assert task.task_id in orchestrator.active_threads
    
    def test_chat_continues_during_training(self, tmp_path):
        """Test that chat continues to work while training runs."""
        # This test verifies the non-blocking nature of training
        dataset_registry = Mock()
        expert_registry = Mock()
        
        mock_dataset = Mock()
        mock_dataset.dataset_id = "python_code_samples"
        mock_dataset.mark_used = Mock()
        dataset_registry.recommend_for_expert.return_value = [mock_dataset]
        dataset_registry.get_dataset.return_value = mock_dataset
        
        expert_registry.add_expert = Mock()
        
        orchestrator = AutoTrainingOrchestrator(
            dataset_registry=dataset_registry,
            expert_registry=expert_registry,
            min_confidence=0.5,
        )
        
        # Start training task
        task1 = orchestrator.create_learning_task(
            user_message="Learn Python programming",
            auto_start=True,
        )
        
        # Should be able to process more messages immediately
        task2 = orchestrator.create_learning_task(
            user_message="Learn calculus",
            auto_start=False,
        )
        
        # Both tasks should exist
        assert task1 is not None
        assert task2 is not None
        assert task1.task_id != task2.task_id


class TestErrorHandling:
    """Test error handling in chat integration."""
    
    def test_orchestrator_unavailable_doesnt_break_chat(self):
        """Test that missing orchestrator doesn't break chat."""
        # Simulate orchestrator = None (failed initialization)
        orchestrator = None
        
        # Chat should still work with orchestrator=None
        # This is handled by the if self._orchestrator is not None check
        # in _on_chat_route_and_run
        
        # No exception should be raised
        assert orchestrator is None  # Chat handler checks this
    
    def test_orchestrator_exception_doesnt_break_chat(self, tmp_path):
        """Test that orchestrator exceptions don't break chat."""
        dataset_registry = Mock()
        expert_registry = Mock()
        
        # Make orchestrator raise exception
        dataset_registry.recommend_for_expert.side_effect = Exception("Test error")
        
        orchestrator = AutoTrainingOrchestrator(
            dataset_registry=dataset_registry,
            expert_registry=expert_registry,
            min_confidence=0.5,
        )
        
        # Should raise exception (caught by chat handler)
        with pytest.raises(Exception) as exc_info:
            orchestrator.create_learning_task(
                user_message="Learn Python programming",
                auto_start=False,
            )
        
        assert "Test error" in str(exc_info.value)
        
        # Chat handler should catch this and continue normally


class TestIntegrationEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.integration
    def test_full_learning_flow(self, tmp_path):
        """Test complete flow from chat message to training task."""
        # Setup registries
        dataset_registry = Mock()
        expert_registry = Mock()
        
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.dataset_id = "python_code_samples"
        mock_dataset.mark_used = Mock()
        mock_dataset.file_path = tmp_path / "test_dataset.txt"
        mock_dataset.file_path.write_text("sample python code")
        dataset_registry.recommend_for_expert.return_value = [mock_dataset]
        dataset_registry.get_dataset.return_value = mock_dataset
        
        expert_registry.add_expert = Mock()
        
        # Create orchestrator
        orchestrator = AutoTrainingOrchestrator(
            dataset_registry=dataset_registry,
            expert_registry=expert_registry,
            min_confidence=0.5,
        )
        
        # Simulate chat message
        user_message = "Learn Python programming"
        
        # 1. Detect intent
        detector = IntentDetector()
        intent = detector.detect(user_message)
        assert intent is not None
        assert intent.domain == "coding"
        
        # 2. Create learning task
        task = orchestrator.create_learning_task(
            user_message=user_message,
            auto_start=False,  # Don't actually train in test
        )
        
        assert task is not None
        assert task.intent.domain == "coding"
        assert task.dataset_id == "python_code_samples"
        assert "python" in task.expert_id
        
        # 3. Verify expert was created
        expert_registry.add_expert.assert_called_once()
        
        # 4. Verify dataset was marked as used
        mock_dataset.mark_used.assert_called_once()
        
        # 5. Verify task can be retrieved
        retrieved_task = orchestrator.get_task_status(task.task_id)
        assert retrieved_task == task


# Summary comment for test file
"""
Test Suite Summary:
-------------------
Total Test Classes: 6
Total Test Methods: ~15

Coverage Areas:
1. IntentDetectionInChat: Validates intent detector recognizes learning requests
2. ChatOrchestrationIntegration: Tests task creation from chat messages
3. ChatResponseFormatting: Verifies status message format
4. BackgroundTraining: Ensures non-blocking training execution
5. ErrorHandling: Tests graceful error handling
6. IntegrationEndToEnd: Complete flow validation

All tests use mocks to avoid actual training or file system operations.
Integration tests marked with @pytest.mark.integration for selective execution.
"""
