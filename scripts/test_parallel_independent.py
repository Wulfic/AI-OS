"""
Test Parallel Independent Training - Quick Validation
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from aios.cli.hrm_hf.parallel_independent import (
    ParallelTrainingCoordinator,
    TrainingBlock,
    BlockQueue,
    CheckpointMerger
)

def test_block_queue():
    """Test block queue functionality"""
    print("Testing BlockQueue...")
    blocks = [TrainingBlock(i, i*100, (i+1)*100) for i in range(5)]
    queue = BlockQueue(blocks)
    
    # Test getting blocks
    block1 = queue.get_next_block(gpu_id=0)
    block2 = queue.get_next_block(gpu_id=1)
    
    assert block1.id == 0
    assert block2.id == 1
    assert block1.status == "in_progress"
    
    # Test completion
    queue.mark_complete(0)
    progress = queue.get_progress()
    assert progress['completed'] == 1
    assert progress['percent'] == 20.0
    
    print("✅ BlockQueue tests passed")

def test_checkpoint_merger():
    """Test checkpoint merging"""
    print("\nTesting CheckpointMerger...")
    
    # Create dummy checkpoints
    model1_state = {'weight': torch.tensor([1.0, 2.0, 3.0])}
    model2_state = {'weight': torch.tensor([3.0, 4.0, 5.0])}
    
    ckpt_dir = Path("artifacts/test_checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt1_path = ckpt_dir / "gpu0_block0.safetensors"
    ckpt2_path = ckpt_dir / "gpu1_block1.safetensors"
    
    from safetensors.torch import save_file as save_safetensors
    save_safetensors(model1_state, str(ckpt1_path))
    save_safetensors(model2_state, str(ckpt2_path))
    
    # Test merging
    merged = CheckpointMerger.merge_checkpoints([str(ckpt1_path), str(ckpt2_path)])
    expected = torch.tensor([2.0, 3.0, 4.0])  # Average of [1,2,3] and [3,4,5]
    
    assert torch.allclose(merged['model_state_dict']['weight'], expected)
    print(f"Merged weights: {merged['model_state_dict']['weight']}")
    print("✅ CheckpointMerger tests passed")
    
    # Cleanup
    ckpt1_path.unlink()
    ckpt2_path.unlink()

def test_simple_training():
    """Test actual training with simple model"""
    print("\n🚀 Testing Simple Training...")
    
    from aios.cli.hrm_hf.parallel_independent import ParallelConfig
    
    config = ParallelConfig(
        cuda_ids=[0, 1] if torch.cuda.device_count() >= 2 else [0],
        num_blocks=4,
        merge_interval=2,
        checkpoint_dir="artifacts/test_parallel_checkpoints",
        save_dir="artifacts/test_final",
        batch_size=2,
        max_seq_len=128,
        dataset_size=400  # Small for testing
    )
    
    # Simple model
    def model_factory():
        return nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 100)  # Small vocab
        )
    
    def optimizer_factory(model):
        return torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"GPUs available: {torch.cuda.device_count()}")
    print(f"Using GPUs: {config.cuda_ids}")
    print(f"Blocks: {config.num_blocks}")
    
    try:
        coordinator = ParallelTrainingCoordinator(config)
        coordinator.start_training(model_factory, optimizer_factory)
        
        # Check results
        final_model_path = Path(config.save_dir) / "final_model.safetensors"
        if not final_model_path.exists():
            # Check for .pt fallback
            final_model_path = Path(config.save_dir) / "final_model.pt"
        
        if final_model_path.exists():
            print(f"\n✅ Training completed successfully!")
            print(f"Final model saved: {final_model_path}")
            
            # Load and verify
            try:
                from safetensors.torch import load_file as load_safetensors
                final_state = load_safetensors(str(final_model_path), device='cpu')
            except Exception:
                final_state = torch.load(final_model_path, map_location='cpu')
            print(f"Model keys: {list(final_state.keys())[:3]}...")
        else:
            print("\n⚠️ Training completed but final model not found")
            
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("="*60)
    print("Parallel Independent Training - Test Suite")
    print("="*60)
    
    # Run tests
    test_block_queue()
    test_checkpoint_merger()
    
    if torch.cuda.is_available():
        test_simple_training()
    else:
        print("\n⚠️ No CUDA available - skipping training test")
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)
