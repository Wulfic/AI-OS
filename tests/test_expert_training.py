"""Test expert-specific training mode (Task 12).

This test validates:
1. Expert training CLI works
2. Expert checkpoint saved correctly
3. Expert registry updated
4. Goal linking works
"""

import os
import json
import shutil
from pathlib import Path
import torch
import pytest

from aios.core.hrm_training.training_config import TrainingConfig
from aios.cli.hrm_hf.train_actv1 import train_expert_only
from aios.core.hrm_models.expert_metadata import ExpertRegistry


def test_expert_training_basic():
    """Test basic expert training workflow."""
    
    # Setup test environment
    test_expert_id = "test-expert-001"
    test_dataset = "training_data/curated_datasets/test_sample.txt"
    
    # Clean up any existing test expert
    expert_dir = Path("artifacts") / "experts" / test_expert_id
    if expert_dir.exists():
        shutil.rmtree(expert_dir)
    
    # Create config for expert training
    config = TrainingConfig(
        expert_id=test_expert_id,
        model="gpt2",  # Use small model for faster test
        dataset_file=test_dataset,
        max_seq_len=64,
        batch_size=2,
        steps=5,  # Just a few steps for testing
        lr=1e-4,
        device="cpu",  # Force CPU for test reliability
        hidden_size=256,
        expansion=2.0,
        use_amp=False,  # Disable AMP for CPU
        gradient_checkpointing=False,  # Not needed for small test
    )
    
    # Run expert training
    try:
        train_expert_only(config)
    except Exception as e:
        pytest.fail(f"Expert training failed: {e}")
    
    # Verify expert checkpoint exists
    expert_checkpoint = expert_dir / "expert.pt"
    assert expert_checkpoint.exists(), f"Expert checkpoint not found: {expert_checkpoint}"
    
    # Load and verify checkpoint structure
    checkpoint = torch.load(expert_checkpoint, map_location="cpu")
    assert isinstance(checkpoint, dict), "Checkpoint should be a state_dict"
    
    # Verify registry updated
    registry_path = Path("artifacts") / "experts" / "registry.json"
    assert registry_path.exists(), "Expert registry not found"
    
    registry = ExpertRegistry.load(str(registry_path))
    expert_metadata = registry.get_expert(test_expert_id)
    
    assert expert_metadata is not None, f"Expert {test_expert_id} not in registry"
    assert expert_metadata.expert_id == test_expert_id
    assert expert_metadata.checkpoint_path == str(expert_checkpoint)
    assert expert_metadata.is_active == True
    
    # Verify training config stored
    assert "hidden_size" in expert_metadata.training_config
    assert expert_metadata.training_config["hidden_size"] == 256
    assert expert_metadata.training_config["intermediate_size"] == 512  # 256 * 2.0
    
    print(f"✅ Expert training test passed!")
    print(f"   Expert ID: {test_expert_id}")
    print(f"   Checkpoint: {expert_checkpoint}")
    print(f"   Params: {expert_metadata.training_config}")
    
    # Cleanup
    shutil.rmtree(expert_dir)


def test_expert_training_with_goal():
    """Test expert training with goal linking."""
    
    test_expert_id = "test-expert-002"
    test_goal = "test-goal-123"
    test_dataset = "training_data/curated_datasets/test_sample.txt"
    
    # Clean up
    expert_dir = Path("artifacts") / "experts" / test_expert_id
    if expert_dir.exists():
        shutil.rmtree(expert_dir)
    
    config = TrainingConfig(
        expert_id=test_expert_id,
        model="gpt2",
        dataset_file=test_dataset,
        max_seq_len=64,
        batch_size=2,
        steps=3,
        device="cpu",
        hidden_size=128,
        default_goal=test_goal,  # Link to goal
        use_amp=False,
        gradient_checkpointing=False,
    )
    
    # Run training
    try:
        train_expert_only(config)
    except Exception as e:
        pytest.fail(f"Expert training with goal failed: {e}")
    
    # Verify goal linked
    registry_path = Path("artifacts") / "experts" / "registry.json"
    registry = ExpertRegistry.load(str(registry_path))
    expert_metadata = registry.get_expert(test_expert_id)
    
    assert expert_metadata is not None
    assert test_goal in expert_metadata.goals, f"Goal {test_goal} not linked to expert"
    
    print(f"✅ Expert-goal linking test passed!")
    print(f"   Expert: {test_expert_id}")
    print(f"   Goal: {test_goal}")
    print(f"   Goals list: {expert_metadata.goals}")
    
    # Cleanup
    shutil.rmtree(expert_dir)


def test_expert_can_be_loaded():
    """Test that trained expert can be loaded via LazyExpertLoader."""
    
    from aios.core.hrm_models.dynamic_moe import LazyExpertLoader
    
    test_expert_id = "test-expert-003"
    test_dataset = "training_data/curated_datasets/test_sample.txt"
    
    # Clean up
    expert_dir = Path("artifacts") / "experts" / test_expert_id
    if expert_dir.exists():
        shutil.rmtree(expert_dir)
    
    # Train expert
    config = TrainingConfig(
        expert_id=test_expert_id,
        model="gpt2",
        dataset_file=test_dataset,
        max_seq_len=64,
        batch_size=2,
        steps=3,
        device="cpu",
        hidden_size=256,
        expansion=2.0,
        use_amp=False,
        gradient_checkpointing=False,
    )
    
    try:
        train_expert_only(config)
    except Exception as e:
        pytest.fail(f"Expert training failed: {e}")
    
    # Try to load via LazyExpertLoader
    loader = LazyExpertLoader(max_gpu_experts=2, max_cpu_experts=4, device="cpu")
    
    expert_checkpoint = expert_dir / "expert.pt"
    expert_module = loader.load_expert(
        expert_id=test_expert_id,
        checkpoint_path=str(expert_checkpoint),
        hidden_size=256,
        intermediate_size=512,
    )
    
    assert expert_module is not None, "Failed to load expert"
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    hidden_size = 256
    
    dummy_input = torch.randn(batch_size, seq_len, hidden_size)
    output = expert_module(dummy_input)
    
    assert output.shape == dummy_input.shape, f"Output shape mismatch: {output.shape} vs {dummy_input.shape}"
    
    # Check cache stats
    stats = loader.get_stats()
    assert stats["total_cached"] == 1, "Expert should be cached"
    assert stats["cpu_experts"] == 1, "Expert should be on CPU"
    
    print(f"✅ Expert loading test passed!")
    print(f"   Expert loaded: {test_expert_id}")
    print(f"   Cache stats: {stats}")
    
    # Cleanup
    shutil.rmtree(expert_dir)


if __name__ == "__main__":
    # Run tests directly
    print("=" * 60)
    print("Testing Expert Training (Task 12)")
    print("=" * 60)
    
    test_expert_training_basic()
    print()
    test_expert_training_with_goal()
    print()
    test_expert_can_be_loaded()
    
    print()
    print("=" * 60)
    print("✅ All Expert Training Tests Passed!")
    print("=" * 60)
